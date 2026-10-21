"""Tests de la recherche hybride (develop.search) : analyse de la question et moteur multi-canaux."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import httpx
import numpy as np
import pytest

from develop.llm.base import LLMClient, LLMUnavailable
from develop.llm.ollama import OllamaClient
from develop.search import CHANNELS, SearchEngine, SearchHit, SearchResult, analyze_query
from develop.search.engine import (
    FUZZY_MIN_LENGTH,
    MAX_PER_SOURCE,
    W_BARE,
    W_FUZZY,
    edit_distance,
    fuzzy_key,
    lexical_terms_found,
    make_snippet,
    max_edits,
)
from develop.storage import (
    ChunkRow,
    Database,
    DocumentRow,
    EndpointRow,
    FileRow,
    ModelRow,
    NoteRow,
    RelationRow,
    RepoRow,
    SymbolRow,
    node_key,
)

TODAY = "2026-10-17"
EMBED_MODEL = "fake-embed"


# --------------------------------------------------------------------------- analyse de la question

@pytest.mark.parametrize("text, intent", [
    ("Quels champs a la table orders ?", "fields"),
    ("quelles colonnes contient customers ?", "fields"),
    ("Donne-moi la structure de Order", "fields"),
    ("what are the fields of the customers table?", "fields"),
    ("Qui appelle OrderService::confirm ?", "who_calls"),
    ("Où est utilisé confirmOrder ?", "who_calls"),
    ("liste des appelants de totalFor", "who_calls"),
    ("who calls getOrder?", "who_calls"),
    ("Qu'est-ce que je casse si je renomme total_amount ?", "impact"),
    ("quel est l'impact d'une suppression de la colonne status", "impact"),
    ("si je modifie OrderService, quoi d'autre change ?", "impact"),
    ("Pourquoi a-t-on décidé de ne pas ajouter le statut paid ?", "decision"),
    ("compte-rendu de la réunion sur les statuts", "decision"),
    ("quand a-t-on choisi axios ?", "decision"),
    ("historique des changements du notifier", "decision"),
    ("Fais un audit de l'architecture globale", "broad"),
    ("vue d'ensemble du projet", "broad"),
    ("explique le projet", "broad"),
    ("Comment ajouter un endpoint ?", "howto"),
    ("comment faire pour lancer les tests", "howto"),
    ("how to add a migration", "howto"),
    ("montant total d'une commande", "factual"),
    ("OrderService", "factual"),
])
def test_analyze_query_intents(text, intent):
    assert analyze_query(text, today=TODAY).intent == intent


@pytest.mark.parametrize("text, date_from, date_to", [
    ("notes d'hier", "2026-10-16", "2026-10-16"),
    ("ce qu'on a décidé la semaine dernière", "2026-10-05", "2026-10-11"),
    ("les réunions en septembre", "2026-09-01", "2026-09-30"),
    ("depuis le 1er mars", "2026-03-01", None),
    ("réunion du 2026-09-01", "2026-09-01", "2026-09-01"),
    ("entre le 1er et le 15 septembre 2026", "2026-09-01", "2026-09-15"),
    ("between 2026-09-01 and 2026-09-30", "2026-09-01", "2026-09-30"),
    ("le mois dernier", "2026-09-01", "2026-09-30"),
    ("il y a 3 jours", "2026-10-14", "2026-10-14"),
    ("jusqu'au 10 octobre 2026", None, "2026-10-10"),
    ("OrderService confirm", None, None),
])
def test_analyze_query_dates(text, date_from, date_to):
    a = analyze_query(text, today=TODAY)
    assert (a.date_from, a.date_to) == (date_from, date_to)


def test_analyze_query_identifiers_paths_routes_tables():
    a = analyze_query("Où est utilisé `confirmOrder` dans frontend/src/api/client.ts et OrderService::confirm ?", today=TODAY)
    assert "confirmOrder" in a.identifiers
    assert "OrderService::confirm" in a.identifiers and "OrderService" in a.identifiers and "confirm" in a.identifiers
    assert a.paths == ["frontend/src/api/client.ts"]
    b = analyze_query("Comment ajouter un endpoint POST /orders/{id}/confirm et GET /api/orders ?", today=TODAY)
    assert b.routes == ["POST /orders/{id}/confirm", "GET /api/orders"]
    c = analyze_query("le modèle Order et la table order_lines ; snake_case total_amount, kebab v2-api-2024", today=TODAY)
    assert "Order" in c.tables and "order_lines" in c.tables
    assert "total_amount" in c.identifiers and "order_lines" in c.identifiers and "v2-api-2024" in c.identifiers
    d = analyze_query("ouvre OrderService.php", today=TODAY)
    assert "OrderService.php" in d.identifiers and "OrderService" in d.identifiers and d.paths == ["OrderService.php"]
    e = analyze_query("the customers table", today=TODAY)
    assert e.tables == ["customers"]


def test_analyze_query_keywords_accents_stopwords_and_dirty_input():
    a = analyze_query("Quel est le montant commandé par le client hier ?", today=TODAY)
    assert a.keywords == ["montant", "commandé", "client"]  # accents conservés, mots vides et date retirés
    assert a.lexical_query == "montant commandé client"
    assert analyze_query("", today=TODAY).keywords == [] and analyze_query("   ", today=TODAY).lexical_query == ""
    for dirty in ['"); DROP TABLE chunks; --', "OR AND NOT", "(((", "🙂 ééé", "x" * 3000, None]:
        res = analyze_query(dirty, today=TODAY)  # type: ignore[arg-type]
        assert res.intent in ("factual", "fields", "who_calls", "impact", "decision", "broad", "howto")
    d = analyze_query("OrderService", today=TODAY).to_dict()
    assert d["identifiers"] == ["OrderService"] and d["intent"] == "factual" and "lexical_query" in d


# --------------------------------------------------------------------------- index de test

def _read(root: Path, repo: str, rel: str) -> str:
    return (root / repo / rel).read_text(encoding="utf-8")


def _code_chunk(repo: str, rel: str, text: str, line_start: int, line_end: int, hint: str = "") -> ChunkRow:
    """Même forme que l'indexeur : en-tête « # repo/chemin Lx-y hint » + corps, ``body_offset`` en meta."""
    header = "# %s/%s L%d-%d%s\n" % (repo, rel, line_start, line_end, (" " + hint) if hint else "")
    lines = text.splitlines()
    body = "\n".join(lines[line_start - 1:line_end])
    meta: Dict[str, Any] = {"body_offset": len(header)}
    if hint:
        meta["symbol_hint"] = hint
    return ChunkRow(text=header + body, line_start=line_start, line_end=line_end, meta=meta)


def _vec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


@pytest.fixture
def index(sample_project: Path) -> SimpleNamespace:
    """Base ':memory:' peuplée à la main à partir de la fixture (texte réel, lignes réelles)."""
    root = sample_project
    db = Database(":memory:")
    db.upsert_repo(RepoRow(id="backend", name="backend", path=str(root / "backend"), kind="backend"))
    db.upsert_repo(RepoRow(id="frontend", name="frontend", path=str(root / "frontend"), kind="frontend"))
    ns = SimpleNamespace(db=db, root=root, chunks={})

    def add_file(repo: str, rel: str, language: str, symbols=(), endpoints=(), models=(), chunk_ranges=None,
                 hint: str = ""):
        text = _read(root, repo, rel)
        total = len(text.splitlines())
        f = FileRow(repo_id=repo, path=rel, language=language, kind="code", size=len(text))
        db.upsert_file(f)
        ranges = chunk_ranges or [(1, total)]
        chunks = [_code_chunk(repo, rel, text, a, b, hint) for a, b in ranges]
        ids = db.replace_file_artifacts(f.id, symbols=list(symbols), endpoints=list(endpoints), models=list(models),
                                        chunks=chunks)
        ns.chunks[rel] = ids["chunks"]
        return f, ids, text

    # --- backend
    ns.svc, svc_ids, _ = add_file(
        "backend", "app/Services/OrderService.php", "php",
        symbols=[SymbolRow(name="OrderService", kind="class", line_start=8, line_end=29),
                 SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=10, line_end=23,
                           signature="confirm(Order $order): Order"),
                 SymbolRow(name="totalFor", kind="method", parent="OrderService", line_start=25, line_end=28,
                           signature="totalFor(Order $order): int")],
        hint="OrderService")
    ns.sym_service, ns.sym_service_confirm, ns.sym_total_for = svc_ids["symbols"]
    ns.ctl, ctl_ids, _ = add_file(
        "backend", "app/Http/Controllers/OrderController.php", "php",
        symbols=[SymbolRow(name="OrderController", kind="class", line_start=9, line_end=29),
                 SymbolRow(name="index", kind="method", parent="OrderController", line_start=15, line_end=18),
                 SymbolRow(name="show", kind="method", parent="OrderController", line_start=20, line_end=23),
                 SymbolRow(name="confirm", kind="method", parent="OrderController", line_start=25, line_end=28,
                           signature="confirm(Request $request, Order $order)")],
        hint="OrderController")
    ns.sym_controller, _, _, ns.sym_controller_confirm = ctl_ids["symbols"]
    ns.order_model_file, model_ids, _ = add_file(
        "backend", "app/Models/Order.php", "php",
        symbols=[SymbolRow(name="Order", kind="class", line_start=9, line_end=26)],
        models=[ModelRow(name="Order", table_name="orders", kind="eloquent", line=9,
                         fields=[{"name": "reference"}, {"name": "customer_id"}, {"name": "status"}, {"name": "total_amount"}],
                         relations=[{"kind": "belongsTo", "model": "Customer"}, {"kind": "hasMany", "model": "OrderLine"}])],
        hint="Order")
    ns.model_order = model_ids["models"][0]
    ns.routes, route_ids, _ = add_file(
        "backend", "routes/api.php", "php",
        endpoints=[EndpointRow(side="server", method="GET", path="/orders", normalized_path="/orders",
                               handler="OrderController@index", framework="laravel", line=11),
                   EndpointRow(side="server", method="GET", path="/orders/{order}", normalized_path="/orders/:param",
                               handler="OrderController@show", framework="laravel", line=12),
                   EndpointRow(side="server", method="POST", path="/orders/{order}/confirm",
                               normalized_path="/orders/:param/confirm", handler="OrderController@confirm",
                               framework="laravel", line=13)])
    ns.ep_list, ns.ep_show, ns.ep_confirm = route_ids["endpoints"]
    ns.migration, mig_ids, _ = add_file(
        "backend", "database/migrations/2024_01_01_000000_create_orders_table.php", "php",
        models=[ModelRow(name="orders", table_name="orders", kind="migration", line=11,
                         fields=[{"name": "id"}, {"name": "reference"}, {"name": "customer_id"}, {"name": "status"},
                                 {"name": "total_amount"}, {"name": "confirmed_at"}])])
    ns.model_orders_migration = mig_ids["models"][0]
    ns.customer_ctl, cust_ids, _ = add_file(
        "backend", "app/Http/Controllers/CustomerController.php", "php",
        symbols=[SymbolRow(name="CustomerController", kind="class", line_start=8, line_end=20)])
    ns.sym_customer_controller = cust_ids["symbols"][0]

    # --- frontend
    ns.client, client_ids, _ = add_file(
        "frontend", "src/api/client.ts", "typescript",
        symbols=[SymbolRow(name="Order", kind="interface", line_start=7, line_end=14),
                 SymbolRow(name="listOrders", kind="function", line_start=22, line_end=25),
                 SymbolRow(name="getOrder", kind="function", line_start=27, line_end=30),
                 SymbolRow(name="confirmOrder", kind="function", line_start=32, line_end=35,
                           signature="confirmOrder(id: number): Promise<Order>")],
        endpoints=[EndpointRow(side="client", method="GET", path="/orders", normalized_path="/orders",
                               framework="axios", line=23),
                   EndpointRow(side="client", method="GET", path="/orders/${id}", normalized_path="/orders/:param",
                               framework="axios", line=28),
                   EndpointRow(side="client", method="POST", path="/orders/${id}/confirm",
                               normalized_path="/orders/:param/confirm", framework="axios", line=33)],
        models=[ModelRow(name="Order", kind="typescript_interface", line=7,
                         fields=[{"name": "id"}, {"name": "reference"}, {"name": "status"}, {"name": "total_amount"}])],
        chunk_ranges=[(1, 20), (21, 40)])
    ns.sym_confirm_order = client_ids["symbols"][3]
    ns.ep_client_confirm = client_ids["endpoints"][2]
    ns.model_order_ts = client_ids["models"][0]
    ns.customer_search, cs_ids, _ = add_file(
        "frontend", "src/components/CustomerSearch.tsx", "tsx",
        symbols=[SymbolRow(name="CustomerSearch", kind="component", line_start=4, line_end=12)])
    ns.sym_customer_search = cs_ids["symbols"][0]
    add_file("frontend", "src/components/OrderList.tsx", "tsx")

    # --- documents (texte réel des docs de la fixture)
    dico_text = (root / "docs/dictionnaire-donnees.md").read_text(encoding="utf-8")
    ns.doc_dico = db.add_document(DocumentRow(title="Dictionnaire de données", kind="repo_file", format="md",
                                              source_path=str(root / "docs/dictionnaire-donnees.md")))
    sections = dico_text.split("\n## ")
    doc_chunks = []
    line = 1
    for i, section in enumerate(sections):
        body = section if i == 0 else "## " + section
        n = len(body.splitlines())
        doc_chunks.append(ChunkRow(text=body, line_start=line, line_end=line + n - 1,
                                   meta={"section": body.splitlines()[0].lstrip("# ").strip()}))
        line += n
    ns.chunks["dico"] = db.replace_chunks("document", ns.doc_dico, doc_chunks)
    spec_text = (root / "docs/specification-commandes.md").read_text(encoding="utf-8")
    ns.doc_spec = db.add_document(DocumentRow(title="Spécification commandes", kind="repo_file", format="md",
                                              source_path=str(root / "docs/specification-commandes.md")))
    spec_chunks = [ChunkRow(text=l, line_start=i + 1, line_end=i + 1, meta={"page": 1})
                   for i, l in enumerate(spec_text.splitlines()) if l.strip()]
    ns.chunks["spec"] = db.replace_chunks("document", ns.doc_spec, spec_chunks)

    # --- notes datées
    reunion = (root / "docs/decisions/2026-09-01-reunion-statuts.md").read_text(encoding="utf-8")
    ns.note_reunion = db.add_note(NoteRow(title="Réunion statuts de commande", body=reunion, note_date="2026-09-01",
                                          tags=["réunion", "statuts"]))
    header = "Réunion statuts de commande"
    ns.chunks["note_reunion"] = db.replace_chunks("note", ns.note_reunion, [
        ChunkRow(text=header + "\n\n" + reunion, line_start=1, line_end=len(reunion.splitlines()),
                 meta={"kind": "note", "title": header, "note_date": "2026-09-01", "body_offset": len(header) + 2})])
    ns.note_paiement = db.add_note(NoteRow(title="Idée paiement", body="Ajouter paid_at sur orders après la migration.",
                                           note_date="2026-10-10", tags=["paiement"]))
    ns.chunks["note_paiement"] = db.replace_chunks("note", ns.note_paiement, [
        ChunkRow(text="Idée paiement\n\nAjouter paid_at sur orders après la migration.", line_start=1, line_end=1,
                 meta={"kind": "note", "body_offset": 15})])
    ns.note_vieille = db.add_note(NoteRow(title="Première version du notifier", body="Le notifier tourne sur le port 3001.",
                                          note_date="2025-03-15"))
    ns.chunks["note_vieille"] = db.replace_chunks("note", ns.note_vieille, [
        ChunkRow(text="Première version du notifier\n\nLe notifier tourne sur le port 3001.", line_start=1, line_end=1)])
    ns.note_sans_chunk = db.add_note(NoteRow(title="Note non indexée", body="Point rapide : revoir le cache.",
                                             note_date="2026-10-16"))

    # --- relations (1 saut)
    db.add_relations([
        RelationRow(src=node_key("file", ns.ctl.id), dst=node_key("file", ns.svc.id), kind="imports", confidence=0.9),
        RelationRow(src=node_key("file", ns.ctl.id), dst=node_key("file", ns.order_model_file.id), kind="imports"),
        RelationRow(src=node_key("file", ns.svc.id), dst=node_key("file", ns.order_model_file.id), kind="imports"),
        RelationRow(src=node_key("symbol", ns.sym_controller_confirm), dst=node_key("symbol", ns.sym_service_confirm),
                    kind="calls", confidence=0.9),
        RelationRow(src=node_key("endpoint", ns.ep_confirm), dst=node_key("symbol", ns.sym_controller_confirm), kind="handles"),
        RelationRow(src=node_key("endpoint", ns.ep_client_confirm), dst=node_key("endpoint", ns.ep_confirm),
                    kind="calls_endpoint", confidence=0.8),
        RelationRow(src=node_key("symbol", ns.sym_service), dst=node_key("model", ns.model_order), kind="uses_model"),
        RelationRow(src=node_key("model", ns.model_order), dst="table:orders", kind="defines_table", confidence=0.8),
        RelationRow(src=node_key("model", ns.model_orders_migration), dst="table:orders", kind="defines_table"),
        RelationRow(src=node_key("symbol", ns.sym_service), dst=node_key("document", ns.doc_dico), kind="documented_by"),
        RelationRow(src=node_key("document", ns.doc_spec), dst=node_key("endpoint", ns.ep_confirm), kind="mentions", confidence=0.7),
        RelationRow(src=node_key("note", ns.note_reunion), dst="table:orders", kind="mentions", confidence=0.7),
    ])

    # --- embeddings artificiels (dimension 4)
    db.upsert_embeddings([
        (ns.chunks["app/Services/OrderService.php"][0], EMBED_MODEL, _vec(1, 0, 0, 0)),
        (ns.chunks["app/Http/Controllers/OrderController.php"][0], EMBED_MODEL, _vec(0.9, 0.1, 0, 0)),
        (ns.chunks["app/Models/Order.php"][0], EMBED_MODEL, _vec(0, 1, 0, 0)),
        (ns.chunks["spec"][0], EMBED_MODEL, _vec(0, 0, 1, 0)),
        (ns.chunks["note_reunion"][0], EMBED_MODEL, _vec(0, 0, 0, 1)),
    ])
    yield ns
    db.close()


@pytest.fixture
def engine(index: SimpleNamespace) -> SearchEngine:
    return SearchEngine(index.db, embedding_model=EMBED_MODEL)


def _first(result: SearchResult, source_type: str) -> SearchHit:
    for h in result.hits:
        if h.source_type == source_type:
            return h
    raise AssertionError("aucun résultat de type %s dans %s" % (source_type, [(h.source_type, h.title) for h in result.hits]))


# --------------------------------------------------------------------------- canal identifiants

def test_exact_identifier_puts_symbol_first(engine, index):
    r = engine.search("OrderService")
    assert r.hits, "index peuplé : des résultats sont attendus"
    top = r.hits[0]
    assert top.source_type == "symbol" and top.title == "OrderService" and top.extra["symbol_id"] == index.sym_service
    assert "identifier" in top.channels and "nom de symbole exact « OrderService »" in top.reason
    assert top.path == "app/Services/OrderService.php" and top.repo_id == "backend" and top.file_id == index.svc.id
    assert (top.line_start, top.line_end) == (8, 29)
    assert top.text.startswith("class OrderService") and "public function confirm" in top.text  # lu sur disque
    assert top.chunk_id is None and len(top.snippet) <= 300 and "\n" not in top.snippet
    assert top.extra["kind"] == "class" and top.extra["language"] == "php"
    assert r.channel_counts["identifier"] >= 1 and r.semantic_available is False
    # le morceau du même fichier (lignes 1-29) a été fusionné dans le symbole : pas de doublon
    assert sum(1 for h in r.hits if h.file_id == index.svc.id and h.source_type == "file") == 0
    assert r.hits == sorted(r.hits, key=lambda h: -h.score)


def test_qualified_member_and_case_insensitive_lookup(engine, index):
    r = engine.search("qui appelle OrderService::confirm ?")
    assert r.query.intent == "who_calls"
    top = r.hits[0]
    assert top.source_type == "symbol" and top.extra["symbol_id"] == index.sym_service_confirm
    assert top.title == "OrderService::confirm" and "membre exact" in top.reason
    assert top.extra["signature"] == "confirm(Order $order): Order"
    # insensible à la casse
    r2 = engine.search("orderservice")
    assert r2.hits[0].source_type == "symbol" and r2.hits[0].extra["symbol_id"] == index.sym_service


def test_glued_words_match_symbol_name(engine, index):
    r = engine.search("customer search")
    hit = _first(r, "symbol")
    assert hit.extra["symbol_id"] == index.sym_customer_search and "mots collés" in hit.reason
    r2 = engine.search("customer controller")
    assert _first(r2, "symbol").extra["symbol_id"] == index.sym_customer_controller


def test_path_and_filename_identifier_channel(engine, index):
    r = engine.search("ouvre frontend/src/api/client.ts")
    hit = r.hits[0]
    assert hit.source_type == "file" and hit.path == "src/api/client.ts" and hit.repo_id == "frontend"
    assert "chemin de fichier" in hit.reason or "nom de fichier" in hit.reason
    assert hit.chunk_id == index.chunks["src/api/client.ts"][0] and (hit.line_start, hit.line_end) == (1, 20)
    r2 = engine.search("OrderService.php")
    assert any(h.path == "app/Services/OrderService.php" for h in r2.hits)


def test_routes_channel_finds_both_sides(engine, index):
    r = engine.search("/orders/{id}/confirm")
    endpoints = [h for h in r.hits if h.source_type == "endpoint"]
    assert {h.extra["endpoint_id"] for h in endpoints} >= {index.ep_confirm, index.ep_client_confirm}
    server = next(h for h in endpoints if h.extra["endpoint_id"] == index.ep_confirm)
    assert server.title == "POST /orders/{order}/confirm" and server.path == "routes/api.php"
    assert server.line_start == 13 and "Route::post('/orders/{order}/confirm'" in server.text
    assert "route exacte" in server.reason and server.extra["handler"] == "OrderController@confirm"
    r2 = engine.search("POST /orders/{id}/confirm")
    ids = [h.extra["endpoint_id"] for h in r2.hits if h.source_type == "endpoint"]
    assert index.ep_confirm in ids and index.ep_show not in ids  # la méthode filtre
    # préfixe de base absent côté serveur : /api/orders → /orders
    r3 = engine.search("GET /api/orders")
    assert index.ep_list in [h.extra["endpoint_id"] for h in r3.hits if h.source_type == "endpoint"]


def test_tables_channel_finds_models_and_intent_fields(engine, index):
    r = engine.search("quels champs a la table orders ?")
    assert r.query.intent == "fields" and r.query.tables == ["orders"]
    models = [h for h in r.hits if h.source_type == "model"]
    assert {h.extra["model_id"] for h in models} >= {index.model_order, index.model_orders_migration}
    eloquent = next(h for h in models if h.extra["model_id"] == index.model_order)
    assert eloquent.title == "Order (table orders)" and eloquent.extra["fields"] == ["reference", "customer_id", "status", "total_amount"]
    assert eloquent.text.startswith("class Order extends Model") and "table exacte « orders »" in eloquent.reason
    assert r.hits[0].source_type == "model"  # exact + bonus « fields »


def test_bare_table_trigger_word_is_not_tried_as_symbol_name(engine, index):
    """Non-régression : le mot générique « table » (ou « modèle ») d'une question sur une table n'est pas
    essayé comme nom de symbole nu (palier 3), sinon la propriété Eloquent ``protected $table``
    (symbole ``Order::table``) devient une graine du graphe et devance les vraies sources."""
    text = _read(index.root, "backend", "app/Models/Order.php")
    ids = index.db.replace_file_artifacts(
        index.order_model_file.id,
        symbols=[SymbolRow(name="Order", kind="class", line_start=9, line_end=26),
                 SymbolRow(name="table", kind="property", parent="Order", line_start=11, line_end=11)],
        models=[ModelRow(name="Order", table_name="orders", kind="eloquent", line=9,
                         fields=[{"name": "reference"}, {"name": "customer_id"}, {"name": "status"}])],
        chunks=[_code_chunk("backend", "app/Models/Order.php", text, 1, len(text.splitlines()), "Order")])
    sym_table = ids["symbols"][1]
    # la mention explicite reste servie par le palier 1 (membre exact)
    explicit = engine.search("Order::table")
    assert explicit.hits[0].source_type == "symbol" and explicit.hits[0].extra["symbol_id"] == sym_table
    assert "membre exact « Order::table »" in explicit.hits[0].reason
    # mais le mot nu « table » / « modèle » d'une question sur une table n'est jamais un nom de symbole
    for q in ("quels champs a la table orders ?", "what are the columns of the orders table", "table orders",
              "structure du modèle Order"):
        r = engine.search(q)
        assert {"table", "modèle"} & set(r.query.keywords), q  # le mot déclencheur est bien un mot-clé nu
        reasons = " ".join(h.reason for h in r.hits)
        assert "nom de symbole exact « table »" not in reasons, (q, reasons)
        assert "nom de symbole exact « modèle »" not in reasons, (q, reasons)
        assert not any(h.source_type == "symbol" and h.extra["symbol_id"] == sym_table for h in r.hits), q
        assert r.hits[0].source_type == "model", (q, [(h.source_type, h.title) for h in r.hits[:3]])


# --------------------------------------------------------------------------- canal lexical

def test_lexical_channel_matches_accents(engine, index):
    r = engine.search("commandé")
    assert r.hits and r.channel_counts["lexical"] >= 1
    assert all("lexical" in h.channels for h in r.hits)
    assert any("correspondance lexicale sur « commande" in h.reason for h in r.hits)
    assert any("commande" in h.snippet.lower() for h in r.hits)
    assert all("[" not in h.snippet and len(h.snippet) <= 300 for h in r.hits)
    doc_hit = _first(r, "document")
    assert doc_hit.title in ("Spécification commandes", "Dictionnaire de données") and doc_hit.extra["document_id"] in (index.doc_spec, index.doc_dico)
    assert doc_hit.text  # texte complet du morceau pour l'assembleur de contexte


def test_lexical_snippet_skips_code_header(engine, index):
    r = engine.search("abort 422 brouillon", channels=["lexical"])
    hit = r.hits[0]
    assert hit.source_type == "file" and hit.path == "app/Services/OrderService.php"
    assert hit.text.startswith("# backend/app/Services/OrderService.php L1-29")  # texte complet conservé
    assert hit.snippet.startswith("<?php")  # mais l'extrait saute l'en-tête technique
    assert hit.extra["symbol_hint"] == "OrderService" and hit.extra["language"] == "php" and hit.title == "OrderService.php"


def test_lexical_reason_ignores_literal_brackets_of_indexed_text(engine, index):
    """Non-régression : les crochets du texte indexé (tableaux PHP ``[Ctl::class, 'index']``, tags d'une note
    ``[décision, commandes]``) ne sont pas des marques de surlignage FTS5 ; la raison ne cite que des termes
    de la question."""
    # note indexée comme le fait l'indexeur : en-tête « Note du … — titre [tag, tag] »
    note_id = index.db.add_note(NoteRow(title="Décision paiement", note_date="2026-10-01",
                                        tags=["décision", "commandes"],
                                        body="Le paiement sera porté par `paid_at` sur la table `orders`."))
    header = "Note du 2026-10-01 — Décision paiement [décision, commandes]"
    index.db.replace_chunks("note", note_id, [
        ChunkRow(text=header + "\n\nLe paiement sera porté par `paid_at` sur la table `orders`.", line_start=1,
                 line_end=1, meta={"kind": "note", "title": "Décision paiement", "note_date": "2026-10-01",
                                   "tags": ["décision", "commandes"], "body_offset": len(header) + 2})])

    r = engine.search("quels champs a la table orders", channels=["lexical"], limit=50)
    routes = next(h for h in r.hits if h.path == "routes/api.php")  # Route::get('/orders', [OrderController::class, 'index'])
    assert routes.extra["fts_terms"] == ["orders"]
    assert routes.reason == "correspondance lexicale sur « orders »"
    note = next(h for h in r.hits if h.source_type == "note" and h.extra["note_id"] == note_id)
    assert note.extra["fts_terms"] == ["table", "orders"]
    assert note.reason == "correspondance lexicale sur « table », « orders »"
    for h in r.hits:  # jamais un mot qui n'est pas un terme cherché (« orders » est cherché en préfixe)
        assert h.extra["fts_terms"], (h.title, h.reason)
        assert all(t in ("champs", "table") or t.startswith("orders") for t in h.extra["fts_terms"]), (h.title, h.reason)
        assert "::class" not in h.reason and "décision" not in h.reason


def test_lexical_reason_keeps_prefix_and_accent_matches(engine, index):
    """Les marques légitimes restent citées avec le mot du texte : accents repliés (``commandé`` → « commande »,
    « commandes ») et préfixe du dernier terme (``confirm*`` → « confirmed », « confirmation »)."""
    r = engine.search("commandé", channels=["lexical"])
    assert r.hits and all(set(h.extra["fts_terms"]) <= {"commande", "commandes"} and h.extra["fts_terms"] for h in r.hits)
    r = engine.search("confirm", channels=["lexical"], limit=50)
    dico = next(h for h in r.hits if h.extra.get("document_id") == index.doc_dico and "confirmed_at" in h.text)
    assert dico.extra["fts_terms"] and all(t.startswith("confirm") for t in dico.extra["fts_terms"])
    assert dico.reason.startswith("correspondance lexicale sur « confirm")
    # règle de validation des marques : égalité repliée, préfixe pour le dernier terme seulement
    assert lexical_terms_found("$[table]->foreignId('order_id'); $items[0]; [TODO] [x]", ["table", "orders"]) == ["table"]
    assert lexical_terms_found("[OrderController::class, 'index'] [décision, commandes]", ["orders", "table"]) == []
    assert lexical_terms_found("Pas de [statut] `paid` [OrderService]::[confirm]", ["orderservice", "confirm"]) == ["orderservice", "confirm"]
    assert lexical_terms_found("[Commandes] et [commande] et [Commandes]", ["commandé"]) == ["commandes", "commande"]
    assert lexical_terms_found("[confirmed] puis [orders]", ["confirm", "orders"]) == ["orders"]
    assert lexical_terms_found("", ["orders"]) == [] and lexical_terms_found("[orders]", []) == []


# --------------------------------------------------------------------------- fusion / dédup / diversité

def test_rrf_fusion_boosts_chunk_present_in_two_channels(engine, index):
    q = _vec(1, 0, 0, 0)
    lexical_only = engine.search("abort 422 brouillon", channels=["lexical"])
    both = engine.search("abort 422 brouillon", channels=["lexical", "semantic"], query_vector=q)
    svc_chunk = index.chunks["app/Services/OrderService.php"][0]
    a = next(h for h in lexical_only.hits if h.chunk_id == svc_chunk)
    b = next(h for h in both.hits if h.chunk_id == svc_chunk)
    assert b.score > a.score and b.channels == ["lexical", "semantic"]
    assert "correspondance lexicale" in b.reason and "proche sémantiquement" in b.reason
    assert both.hits[0].chunk_id == svc_chunk
    assert both.channel_counts["lexical"] >= 1 and both.channel_counts["semantic"] >= 1
    assert 0 < a.score < 1 and b.score == pytest.approx(round(1 / 61 + 1 / 61, 6), abs=1e-6)


def test_dedup_overlapping_lines_and_diversity_cap(engine, index):
    # symbole exact + morceau du même fichier qui se chevauchent → un seul résultat, preuves cumulées
    r = engine.search("OrderService")
    svc_hits = [h for h in r.hits if h.file_id == index.svc.id]
    assert len(svc_hits) == 1 and svc_hits[0].source_type == "symbol"
    assert {"identifier", "lexical"} <= set(svc_hits[0].channels)
    # diversité : la spécification compte 5 lignes-morceaux contenant « commande » → au plus 3
    r2 = engine.search("commande", limit=50, channels=["lexical"])
    spec_hits = [h for h in r2.hits if h.source_type == "document" and h.extra["document_id"] == index.doc_spec]
    assert len(index.chunks["spec"]) > MAX_PER_SOURCE and 1 <= len(spec_hits) <= MAX_PER_SOURCE
    # deux éléments structurés distincts du même fichier restent tous deux visibles (classe + méthode)
    r3 = engine.search("OrderService confirm", channels=["identifier"])
    structured = {(h.source_type, h.source_id) for h in r3.hits if h.file_id == index.svc.id}
    assert ("symbol", index.sym_service) in structured and ("symbol", index.sym_service_confirm) in structured


def test_limit_source_types_and_channels_filters(engine, index):
    r = engine.search("commande", limit=2)
    assert len(r.hits) == 2
    assert engine.search("commande", limit=0).hits == []
    docs = engine.search("commande orders", source_types=["document"], limit=20)
    assert docs.hits and all(h.source_type == "document" for h in docs.hits)
    notes = engine.search("commande", source_types=("note",))
    assert notes.hits and all(h.source_type == "note" for h in notes.hits)
    files = engine.search("OrderService", source_types=["file"])  # 'file' englobe le code : symboles compris
    assert files.hits and all(h.source_type in ("file", "symbol", "endpoint", "model") for h in files.hits)
    only_ident = engine.search("OrderService commande", channels=["identifier"])
    assert only_ident.channel_counts["lexical"] == 0 and only_ident.channel_counts["identifier"] >= 1
    assert all(h.channels == ["identifier"] for h in only_ident.hits)
    bogus = engine.search("OrderService", channels=["inconnu"])  # canal inconnu → tous les canaux
    assert bogus.hits and bogus.channel_counts["identifier"] >= 1


# --------------------------------------------------------------------------- canal sémantique

def test_semantic_channel_with_artificial_vectors(engine, index):
    r = engine.search("texte sans rapport lexical zzzz", query_vector=_vec(0, 1, 0, 0), channels=["semantic"])
    assert r.semantic_available is True and r.channel_counts["semantic"] >= 1 and not r.warnings
    top = r.hits[0]
    assert top.chunk_id == index.chunks["app/Models/Order.php"][0] and top.channels == ["semantic"]
    assert top.reason.startswith("proche sémantiquement (similarité 1,00)") and top.extra["similarity"] == pytest.approx(1.0)
    # vecteur non normalisé accepté ; le filtre de type s'applique aussi au canal sémantique
    r2 = engine.search("zzzz", query_vector=_vec(5, 0, 0, 0), channels=["semantic"], source_types=["file"])
    assert [h.chunk_id for h in r2.hits[:2]] == [index.chunks["app/Services/OrderService.php"][0],
                                                 index.chunks["app/Http/Controllers/OrderController.php"][0]]
    r3 = engine.search("zzzz", query_vector=_vec(0, 0, 0, 1), channels=["semantic"], source_types=["note"])
    assert r3.hits and r3.hits[0].source_type == "note" and r3.hits[0].extra["note_id"] == index.note_reunion


def test_semantic_unavailable_without_vector_or_embeddings(engine, index):
    r = engine.search("OrderService")  # pas de vecteur de question : canal ignoré sans erreur
    assert r.semantic_available is False and r.channel_counts["semantic"] == 0 and r.warnings == []
    empty = SearchEngine(Database(":memory:"), embedding_model="autre")
    r2 = empty.search("OrderService", query_vector=_vec(1, 0, 0, 0))
    assert r2.semantic_available is False and any("canal sémantique ignoré" in w for w in r2.warnings)
    r3 = engine.search("OrderService", query_vector=_vec(1, 0, 0))  # mauvaise dimension
    assert r3.semantic_available is False and any("Dimension" in w for w in r3.warnings) and r3.hits
    r4 = engine.search("OrderService", query_vector=_vec(0, 0, 0, 0))  # vecteur nul
    assert r4.semantic_available is False and r4.hits


def test_semantic_matrix_cache_refreshes_when_embeddings_change(engine, index):
    q = _vec(0, 0, 1, 0)
    r = engine.search("zzzz", query_vector=q, channels=["semantic"])
    assert r.hits[0].chunk_id == index.chunks["spec"][0]
    cached = engine._matrix_cache
    assert cached is not None and cached["matrix"].shape == (5, 4)
    engine.search("zzzz", query_vector=q, channels=["semantic"])
    assert engine._matrix_cache is cached  # même contenu en base : pas de rechargement
    index.db.upsert_embeddings([(index.chunks["note_paiement"][0], EMBED_MODEL, _vec(0, 0, 1, 0))])
    r2 = engine.search("zzzz", query_vector=q, channels=["semantic"])
    assert engine._matrix_cache is not cached and engine._matrix_cache["matrix"].shape == (6, 4)
    assert {h.chunk_id for h in r2.hits[:2]} == {index.chunks["spec"][0], index.chunks["note_paiement"][0]}
    # modèle non précisé : le moteur prend le modèle présent en base
    auto = SearchEngine(index.db)
    assert auto.search("zzzz", query_vector=q, channels=["semantic"]).semantic_available is True


# --------------------------------------------------------------------------- canal graphe

def test_graph_channel_expands_one_hop(engine, index):
    r = engine.search("OrderService", limit=20)
    assert r.channel_counts["graph"] >= 1
    graph_hits = [h for h in r.hits if "graph" in h.channels]
    assert graph_hits
    importer = next(h for h in r.hits if h.file_id == index.ctl.id)
    assert "graph" in importer.channels and "relié à OrderService via imports" in importer.reason
    model_hit = next(h for h in r.hits if h.source_type == "model" and h.extra["model_id"] == index.model_order)
    assert "relié à OrderService via utilisation du modèle" in model_hit.reason
    doc_hit = next(h for h in r.hits if h.source_type == "document" and h.extra["document_id"] == index.doc_dico)
    assert "relié à OrderService via documentation" in doc_hit.reason and doc_hit.chunk_id == index.chunks["dico"][0]
    # un saut seulement : le document « spec » n'est relié qu'à l'endpoint, pas à OrderService
    assert not any(h.source_type == "document" and h.extra["document_id"] == index.doc_spec for h in graph_hits)


def test_graph_channel_from_endpoint_and_table(engine, index):
    r = engine.search("POST /orders/{id}/confirm", limit=20)
    handler = next(h for h in r.hits if h.source_type == "symbol" and h.extra["symbol_id"] == index.sym_controller_confirm)
    assert "graph" in handler.channels and "via gestionnaire de route" in handler.reason
    assert any(h.source_type == "document" and h.extra["document_id"] == index.doc_spec for h in r.hits)
    r2 = engine.search("qui appelle OrderService::confirm ?", limit=20)
    caller = next(h for h in r2.hits if h.source_type == "symbol" and h.extra["symbol_id"] == index.sym_controller_confirm)
    assert "graph" in caller.channels and "relié à OrderService::confirm via appels" in caller.reason
    r3 = engine.search("table orders", limit=20)
    note = next(h for h in r3.hits if h.source_type == "note")
    assert note.extra["note_id"] == index.note_reunion and "relié à orders via mention" in note.reason


# --------------------------------------------------------------------------- canal temporel

def test_temporal_channel_by_date_range(engine, index):
    r = engine.search("réunion du 2026-09-01")
    assert (r.query.date_from, r.query.date_to) == ("2026-09-01", "2026-09-01")
    note = _first(r, "note")
    assert note.extra["note_id"] == index.note_reunion and "temporal" in note.channels
    assert "note du 2026-09-01" in note.reason and note.extra["note_date"] == "2026-09-01" and note.extra["tags"] == ["réunion", "statuts"]
    assert note.snippet.startswith("# Réunion du 2026-09-01")  # extrait sans l'en-tête de la note
    r2 = engine.search("notes de septembre 2026", limit=20)
    ids = [h.extra["note_id"] for h in r2.hits if h.source_type == "note"]
    assert ids == [index.note_reunion]
    r3 = engine.search("notes", date_from="2026-10-01", date_to="2026-10-31", limit=20)
    ids3 = [h.extra["note_id"] for h in r3.hits if h.source_type == "note"]
    assert set(ids3) == {index.note_paiement, index.note_sans_chunk}
    sans_chunk = next(h for h in r3.hits if h.source_type == "note" and h.extra["note_id"] == index.note_sans_chunk)
    assert sans_chunk.chunk_id is None and "revoir le cache" in sans_chunk.text and sans_chunk.title == "Note non indexée"


def test_temporal_channel_decision_intent_prefers_recent_matching_notes(engine, index):
    r = engine.search("pourquoi pas de statut paid pour les commandes ?", limit=20)
    assert r.query.intent == "decision" and r.channel_counts["temporal"] >= 3
    notes = [h for h in r.hits if h.source_type == "note"]
    assert notes and notes[0].extra["note_id"] == index.note_reunion  # cite « statut » et « paid »
    assert "mots-clés présents" in notes[0].reason and {"lexical", "temporal"} <= set(notes[0].channels)
    assert notes[0].extra["channel_scores"].keys() >= {"lexical", "temporal"}  # preuves conservées par canal
    r2 = engine.search("historique", limit=20, channels=["temporal"])
    dates = [h.extra["note_date"] for h in r2.hits]
    assert dates == sorted(dates, reverse=True) and len(dates) == 4  # récentes d'abord, anciennes visibles
    assert engine.search("OrderService", channels=["temporal"]).hits == []  # ni date ni décision : canal muet


# --------------------------------------------------------------------------- repli « faute de frappe »

def test_edit_distance_tolerance_and_fuzzy_key():
    assert edit_distance("confrim", "confirm") == 1  # transposition = une faute
    assert edit_distance("orderservce", "orderservice") == 1 and edit_distance("confirmordr", "confirmorder") == 1
    assert edit_distance("kitten", "sitting") == 3 and edit_distance("", "abc") == 3 and edit_distance("abc", "abc") == 0
    assert edit_distance("kitten", "sitting", cap=2) == 3 and edit_distance("ca", "abc") == 3
    assert max_edits(FUZZY_MIN_LENGTH - 1) == 0 and max_edits(9) == 1 and max_edits(10) == 2 and max_edits(15) == 3
    assert fuzzy_key("confirm_order") == fuzzy_key("confirmOrder") == "confirmorder" and fuzzy_key("Été-1") == "ete1"
    assert W_FUZZY < W_BARE  # une approximation pèse moins qu'un mot nu exact


def test_typo_in_identifier_falls_back_to_close_symbol_names(engine, index):
    # Non-régression : « OrderServce::confrim », « confirmOrdr » et « qui apelle confrimOrder ? » renvoyaient 0 résultat.
    r = engine.search("OrderServce::confrim")
    top = r.hits[0]
    assert top.source_type == "symbol" and top.extra["symbol_id"] == index.sym_service_confirm
    assert top.title == "OrderService::confirm" and "approché" in top.reason and "OrderServce::confrim" in top.reason
    assert top.extra["approximate"] is True and top.extra["edit_distance"] == 2
    assert r.warnings == ["Aucune correspondance exacte pour « OrderServce::confrim » ; noms proches : OrderService::confirm."]
    assert r.channel_counts["graph"] >= 1  # le graphe repart du membre approché (appelants, fichier, modèle…)
    assert index.sym_controller_confirm not in [h.extra.get("symbol_id") for h in r.hits if "approché" in h.reason]

    r2 = engine.search("confirmOrdr")
    assert r2.hits[0].source_type == "symbol" and r2.hits[0].extra["symbol_id"] == index.sym_confirm_order
    assert r2.hits[0].extra["edit_distance"] == 1 and any("confirmOrder" in w for w in r2.warnings)

    r3 = engine.search("qui apelle confrimOrder ?")
    assert r3.hits[0].extra.get("symbol_id") == index.sym_confirm_order and "identifier" in r3.hits[0].channels
    assert all("apelle" not in w for w in r3.warnings)  # un mot de la langue n'est pas un identifiant approché
    assert r3.channel_counts["identifier"] == 1 and r3.hits == sorted(r3.hits, key=lambda h: -h.score)

    # Une correspondance exacte ne déclenche pas le repli (ni avertissement, ni candidat approché).
    exact = engine.search("OrderService::confirm")
    assert exact.hits[0].extra["symbol_id"] == index.sym_service_confirm and exact.warnings == []
    assert not any(h.extra.get("approximate") for h in exact.hits)
    # Trop court ou trop éloigné : toujours rien, sans bruit.
    assert engine.search("zzz").hits == [] and engine.search("nope_zzz").hits == []
    assert engine.search("qui apelle zzz_introuvable ?").hits == []


def test_typo_in_bare_keywords_is_last_resort(engine, index):
    # Aucun canal ne trouve rien : les mots nus sont essayés comme noms de symboles approchés, puis le graphe.
    r = engine.search("orderservce confrim")
    ids = [h.extra.get("symbol_id") for h in r.hits]
    assert index.sym_service in ids and index.sym_service_confirm in ids and index.sym_controller_confirm in ids
    assert r.channel_counts["identifier"] == 3 and r.channel_counts["lexical"] == 0 and r.channel_counts["graph"] >= 1
    assert sorted(r.warnings) == sorted([
        "Aucune correspondance exacte pour « orderservce » ; noms proches : OrderService.",
        "Aucune correspondance exacte pour « confrim » ; noms proches : OrderService::confirm, OrderController::confirm.",
    ])
    # Dès qu'un canal trouve quelque chose, les mots nus ne sont pas approximés (pas de bruit sur « commande »).
    r2 = engine.search("montant commande")
    assert r2.hits and not any(h.extra.get("approximate") for h in r2.hits) and r2.warnings == []
    # Restreint aux types sans symbole : pas de repli, pas d'avertissement.
    r3 = engine.search("orderservce confrim", source_types=["note", "document"])
    assert r3.hits == [] and r3.warnings == []


def test_lookup_symbol_falls_back_to_approximate_names(engine, index):
    close = engine.lookup_symbol("confrim")
    assert {h.extra["symbol_id"] for h in close} == {index.sym_service_confirm, index.sym_controller_confirm}
    assert all("approché" in h.reason and h.extra["approximate"] for h in close)
    assert [h.extra["symbol_id"] for h in engine.lookup_symbol("OrderServce::confrim")] == [index.sym_service_confirm]
    assert [h.title for h in engine.lookup_symbol("listOrd")] == ["listOrders"]  # la sous-chaîne garde la priorité
    assert engine.lookup_symbol("zzz_introuvable") == [] and engine.lookup_symbol("confrim", limit=1)[0].title.endswith("confirm")


# --------------------------------------------------------------------------- robustesse

def test_empty_index_returns_consistent_result():
    engine = SearchEngine(Database(":memory:"))
    r = engine.search("OrderService commande POST /orders/{id} table orders 2026-09-01")
    assert isinstance(r, SearchResult) and r.hits == [] and r.semantic_available is False
    assert set(r.channel_counts) == set(CHANNELS) and all(v == 0 for v in r.channel_counts.values())
    d = r.to_dict()
    assert d["hits"] == [] and d["query"]["identifiers"] == ["OrderService"] and d["warnings"] == []
    assert engine.lookup_symbol("OrderService") == [] and engine.lookup_endpoint("/orders") == []
    assert engine.lookup_model("orders") == [] and engine.file_snippet(1, 1, 10) is None


@pytest.mark.parametrize("dirty", ['"); DROP TABLE chunks; --', "status OR", "a", "", "   ", "OR", "AND NOT", "(((",
                                   "NEAR(x y)", "text:status", '"', "*", "ééé", "🙂", "x" * 5000, "/", "//", "POST",
                                   "`", "``", "a/b/c/d/e.php", "http://localhost:8000/api/orders?x=1", None, 42],
                         ids=lambda v: repr(v)[:24])
def test_dirty_queries_never_raise(engine, dirty):
    r = engine.search(dirty)  # type: ignore[arg-type]
    assert isinstance(r, SearchResult) and isinstance(r.hits, list)
    for h in r.hits:
        assert h.source_type in ("file", "document", "note", "symbol", "endpoint", "model")
        assert len(h.snippet) <= 300 and h.channels and h.reason


def test_search_hit_and_result_to_dict_match_api_contract(engine, index):
    r = engine.search("OrderService", limit=3)
    d = r.to_dict()
    assert set(d) == {"query", "hits", "channel_counts", "semantic_available", "warnings"}
    hit = d["hits"][0]
    for key in ("chunk_id", "source_type", "source_id", "title", "path", "repo_id", "file_id", "line_start", "line_end",
                "page", "snippet", "text", "score", "channels", "reason", "extra"):
        assert key in hit, key
    json.dumps(d)  # sérialisable tel quel
    assert d["query"]["intent"] == "factual" and isinstance(d["channel_counts"]["identifier"], int)


def test_channel_failure_is_logged_not_raised(engine, index, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("base corrompue")
    monkeypatch.setattr(index.db, "fts_search", boom)
    r = engine.search("OrderService")
    assert r.hits and r.hits[0].source_type == "symbol"
    assert r.channel_counts["lexical"] == 0 and any("lexical" in w and "base corrompue" in w for w in r.warnings)


def test_unreadable_file_falls_back_to_signature(index):
    index.db.upsert_repo(RepoRow(id="backend", name="backend", path="/chemin/inexistant", kind="backend"))
    engine = SearchEngine(index.db)
    r = engine.search("OrderService::confirm", channels=["identifier"])
    hit = r.hits[0]
    assert hit.source_type == "symbol" and hit.text == "confirm(Order $order): Order" and hit.snippet == "confirm(Order $order): Order"
    assert (hit.line_start, hit.line_end) == (10, 23)
    assert engine.file_snippet(index.svc.id, 1, 5) is None


# --------------------------------------------------------------------------- aides pour les outils du chat

def test_lookup_helpers(engine, index):
    sym = engine.lookup_symbol("confirm")
    assert [h.extra["symbol_id"] for h in sym] == [index.sym_service_confirm, index.sym_controller_confirm]
    assert sym[0].channels == ["identifier"] and sym[0].score > sym[1].score > 0
    assert engine.lookup_symbol("OrderController::confirm")[0].extra["symbol_id"] == index.sym_controller_confirm
    partial = engine.lookup_symbol("Order")
    assert partial and all(h.title == "Order" for h in partial)  # exacts seulement quand il y en a
    assert [h.title for h in engine.lookup_symbol("listOrd")] == ["listOrders"]  # sinon correspondance partielle
    assert engine.lookup_symbol("") == [] and engine.lookup_symbol("zzz_introuvable") == []
    eps = engine.lookup_endpoint("/orders/{id}/confirm", "POST")
    assert {h.extra["endpoint_id"] for h in eps} == {index.ep_confirm, index.ep_client_confirm}
    assert {h.extra["side"] for h in eps} == {"server", "client"}
    by_suffix = [h.extra["endpoint_id"] for h in engine.lookup_endpoint("/api/orders/42", "get")]
    assert index.ep_show in by_suffix and index.ep_confirm not in by_suffix  # préfixe /api + segment numérique
    assert index.ep_show in [h.extra["endpoint_id"] for h in engine.lookup_endpoint("http://localhost:8000/api/orders/42")]
    assert engine.lookup_endpoint("/") == [] and engine.lookup_endpoint("") == []
    models = engine.lookup_model("orders")
    assert {h.extra["model_id"] for h in models} == {index.model_order, index.model_orders_migration}
    assert engine.lookup_model("order")[0].extra["model_id"] in (index.model_order, index.model_order_ts)
    assert engine.lookup_model("nope") == []


def test_file_snippet_reads_from_disk_with_caps(engine, index):
    win = engine.file_snippet(index.svc.id, 10, 12)
    assert win == {"path": "app/Services/OrderService.php", "start": 10, "end": 12, "total_lines": 29,
                   "text": "    public function confirm(Order $order): Order\n    {\n        if ($order->status !== 'draft') {"}
    capped = engine.file_snippet(index.svc.id, 1, 500, max_lines=5)
    assert (capped["start"], capped["end"]) == (1, 5) and capped["text"].count("\n") == 4
    default = engine.file_snippet(index.client.id, 0, 0)
    assert (default["start"], default["end"], default["total_lines"]) == (1, 40, 40)
    beyond = engine.file_snippet(index.svc.id, 100, 200)
    assert (beyond["start"], beyond["end"]) == (29, 29)
    assert engine.file_snippet(99999, 1, 10) is None
    # jamais hors du dépôt
    index.db.upsert_file(FileRow(repo_id="backend", path="../../../../etc/passwd", language="txt"))
    evil = index.db.get_file("backend", "../../../../etc/passwd")
    assert engine.file_snippet(evil.id, 1, 5) is None


# --------------------------------------------------------------------------- embeddings de la question (asynchrone)

class FakeLLM(LLMClient):
    """Moteur en mémoire : renvoie un vecteur fixe, ou lève si ``fail``."""

    engine = "fake"

    def __init__(self, vector=(0, 1, 0, 0), fail: bool = False):
        self.vector = list(vector)
        self.fail = fail
        self.calls: List[Any] = []

    async def embed(self, texts, model):  # type: ignore[override]
        self.calls.append((list(texts), model))
        if self.fail:
            raise LLMUnavailable("Ollama injoignable")
        return [self.vector for _ in texts]


async def test_embed_query_and_search_async(index):
    llm = FakeLLM(vector=(0, 3, 0, 0))
    engine = SearchEngine(index.db, llm=llm, embedding_model=EMBED_MODEL)
    vec = await engine.embed_query("structure de la commande")
    assert vec is not None and vec.dtype == np.float32 and np.linalg.norm(vec) == pytest.approx(1.0)
    assert llm.calls == [(["structure de la commande"], EMBED_MODEL)]
    r = await engine.search_async("texte sans rapport zzzz", channels=["semantic"])
    assert r.semantic_available is True and r.hits[0].chunk_id == index.chunks["app/Models/Order.php"][0]
    assert await engine.embed_query("   ") is None
    # vecteur fourni explicitement : pas d'appel au moteur
    before = len(llm.calls)
    r2 = await engine.search_async("zzzz", query_vector=_vec(1, 0, 0, 0), channels=["semantic"])
    assert len(llm.calls) == before and r2.hits[0].chunk_id == index.chunks["app/Services/OrderService.php"][0]


async def test_embed_query_failure_is_a_warning_not_an_error(index):
    engine = SearchEngine(index.db, llm=FakeLLM(fail=True), embedding_model=EMBED_MODEL)
    assert await engine.embed_query("OrderService") is None
    assert engine.warnings and "Ollama injoignable" in engine.warnings[-1]
    r = await engine.search_async("OrderService")
    assert r.hits[0].source_type == "symbol" and r.semantic_available is False
    assert any("Embedding de la question indisponible" in w for w in r.warnings)
    none = SearchEngine(index.db)  # sans moteur ni modèle : None, silencieusement
    assert await none.embed_query("OrderService") is None and none.warnings == []
    assert (await SearchEngine(index.db, llm=FakeLLM()).embed_query("x")) is None  # modèle absent


async def test_embed_query_with_mock_ollama_transport(index):
    def fake_ollama(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            body = json.loads(request.content)
            assert body["model"] == "nomic-embed-text"
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in body["input"]]})
        return httpx.Response(404)
    client = OllamaClient(transport=httpx.MockTransport(fake_ollama))
    engine = SearchEngine(index.db, llm=client, embedding_model="nomic-embed-text")
    vec = await engine.embed_query("commande")
    assert vec is not None and vec.shape == (3,)
    # la base contient des vecteurs de dimension 4 sous un autre modèle : le canal est ignoré proprement
    r = await engine.search_async("OrderService")
    assert r.hits[0].source_type == "symbol" and r.semantic_available is False
    assert any("canal sémantique ignoré" in w for w in r.warnings)
    await client.aclose()


# --------------------------------------------------------------------------- performance

def test_search_stays_fast_on_a_larger_index(tmp_path):
    db = Database(":memory:")
    db.upsert_repo(RepoRow(id="r", name="r", path=str(tmp_path), kind="backend"))
    words = ["commande", "client", "facture", "paiement", "livraison", "statut", "montant", "notifier"]
    with db.transaction():
        for i in range(300):
            f = FileRow(repo_id="r", path="src/mod%d/File%d.php" % (i % 20, i), language="php", kind="code")
            db.upsert_file(f)
            syms = [SymbolRow(name="Service%d" % i, kind="class", line_start=1, line_end=40),
                    SymbolRow(name="handle%d" % i, kind="method", parent="Service%d" % i, line_start=5, line_end=20)]
            chunks = [ChunkRow(text="class Service%d { %s %s }" % (i, words[i % 8], words[(i * 3) % 8]),
                               line_start=1 + 10 * k, line_end=10 + 10 * k) for k in range(5)]
            ids = db.replace_file_artifacts(f.id, symbols=syms, chunks=chunks)
            db.add_relations([RelationRow(src=node_key("file", f.id), dst=node_key("file", max(1, f.id - 1)), kind="imports")])
            db.upsert_embeddings([(cid, "m", np.random.default_rng(cid).random(16).astype(np.float32)) for cid in ids["chunks"]])
    engine = SearchEngine(db, embedding_model="m")
    q = np.random.default_rng(1).random(16).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(3):
        r = engine.search("qui appelle Service42::handle42 commande client ?", query_vector=q, limit=20)
    elapsed = time.perf_counter() - t0
    assert r.hits and r.hits[0].source_type == "symbol" and r.semantic_available
    assert elapsed < 2.0, "3 recherches sur 1500 morceaux en %.2fs" % elapsed


def test_make_snippet_strips_tags_and_caps_length():
    assert make_snippet("  <b>Bonjour</b>\n\n  le   monde ") == "Bonjour le monde"
    long = make_snippet("mot " * 200)
    assert len(long) <= 300 and long.endswith("…")
    assert make_snippet("") == "" and make_snippet(None) == ""  # type: ignore[arg-type]
