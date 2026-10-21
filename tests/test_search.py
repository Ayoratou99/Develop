"""Tests de la recherche hybride (develop.search) : analyse de la question et moteur multi-canaux."""
from __future__ import annotations

import json
import sys
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
    GENERATED_PENALTY,
    GRAPH_IMPORTS_PER_SEED,
    MAX_PER_SOURCE,
    ROUTE_LIMIT,
    RRF_K,
    W_BARE,
    W_EXPANSION,
    W_FUZZY,
    _Loader,
    edit_distance,
    fuzzy_key,
    handler_words,
    is_dynamic_client_call,
    is_generated_file,
    lexical_terms_found,
    make_snippet,
    max_edits,
    question_language,
    request_class_names,
    route_contains,
    word_names_symbol,
)
from develop.search.query import COMMON_WORDS, expand_terms, has_code_shape, is_common_word, is_function_word, is_stopword
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


def test_question_words_are_stopwords_but_explicit_mentions_survive():
    """Mots de question (« exemple », « explique », « détails », « modèle », « document », « api »…) : jamais
    des mots-clés, donc jamais des noms nus ; mais une mention explicite reste un nom."""
    a = analyze_query("explique moi la définition des champs avec exemple", today=TODAY)
    assert a.keywords == ["définition", "champs"] and a.lexical_query == "définition champs"
    b = analyze_query("peux-tu me décrire le modèle User et me donner un exemple d'appel de l'api", today=TODAY)
    assert b.keywords == ["user"] and b.tables == ["User"] and b.identifiers == ["User"] and b.intent == "fields"
    c = analyze_query("que dit le document paramétrage des formulaires", today=TODAY)
    assert c.keywords == ["paramétrage", "formulaires"]
    assert analyze_query("comment fonctionne le paramétrage", today=TODAY).keywords == ["paramétrage"]
    assert analyze_query("donne-moi un exemple d'appel", today=TODAY).keywords == []
    # cités explicitement : « le modèle Document », ``document.pdf``, ``api.get()`` restent des noms
    d = analyze_query("le modèle Document est lu par api.get() dans document.pdf", today=TODAY)
    assert d.tables == ["Document"] and {"Document", "api.get", "api", "document.pdf", "document"} <= set(d.identifiers)
    assert is_stopword("exemple") and is_stopword("Modèle") and is_stopword("api") and is_stopword("avec")
    assert not is_function_word("api") and not is_function_word("document") and is_function_word("avec")
    assert analyze_query("Quel est le montant commandé par le client hier ?", today=TODAY).keywords == ["montant", "commandé", "client"]


def test_expansions_translate_question_vocabulary_to_code_terms():
    a = analyze_query("comment mettre à jour un utilisateur et lister ses droits ?", today=TODAY)
    assert a.expansions[:4] == ["update", "edit", "put", "patch"]  # l'expression « mettre à jour » d'abord
    assert {"user", "users", "index", "list", "all", "permission", "role", "policy", "gate"} <= set(a.expansions)
    b = analyze_query("quel est l'api d'authentification et un exemple d'appel", today=TODAY)
    assert b.keywords == ["authentification"] and b.expansions == ["auth", "login", "logout", "token", "jwt", "session"]
    # une expression de deux ou trois mots est « collée » par le moteur, pas développée par une table métier ; un mot
    # de question (« api », « modèle ») n'est pas développé
    c = analyze_query("comment on crée un ordre de mission et les détails du modèle", today=TODAY)
    assert c.expansions == ["create", "store", "post", "new"] and c.keywords == ["crée", "ordre", "mission"]
    assert "controller" not in c.expansions and "model" not in c.expansions
    assert SearchEngine._glued_names(["ordre", "mission", "valider"]) == ["ordremissionvalider", "ordremission", "missionvalider"]
    # un terme déjà mot-clé n'est pas répété (« configuration » → parametrage, déjà cherché via « paramétrage ») ;
    # un texte sans vocabulaire connu : rien
    assert expand_terms("configuration et paramétrage", exclude=["paramétrage"]) == ["config", "settings", "parameter"]
    assert expand_terms("montant commandé par le client") == [] and analyze_query("", today=TODAY).expansions == []
    assert "expansions" in analyze_query("créer", today=TODAY).to_dict()


def test_parametrage_is_a_domain_noun_not_config_vocabulary():
    """« Paramétrage » nomme un module métier (« paramétrage des formulaires de commande »), pas la configuration du
    cadriciel : développé en ``config`` / ``settings``, il ramenait tous les ``config/*.php`` de Laravel et les
    « Settings » de n'importe quel document (8 sources hors sujet, 48 % du contexte à 16k sur un projet réel), au
    détriment des sections du document demandé. « configuration », « réglages » et « paramètres » restent développés."""
    from develop.search import FOLLOWUP_MARKER
    assert expand_terms("paramétrage des formulaires") == [] and expand_terms("les paramétrages") == []
    assert analyze_query("que dit le document paramétrage des formulaires", today=TODAY).expansions == []
    assert analyze_query("comment fonctionne le paramétrage", today=TODAY).expansions == []
    # la question de suivi qui hérite de « paramétrage » du tour précédent n'en hérite pas une expansion non plus
    f = analyze_query("et les champs ? " + FOLLOWUP_MARKER % "paramétrage formulaires", today=TODAY)
    assert "field" in f.expansions and not {"config", "settings", "parameter"} & set(f.expansions)
    # « configuration » désigne bien les fichiers de configuration (→ config/database.php) ; « réglages » aussi
    assert analyze_query("où est la configuration de la base de données", today=TODAY).expansions[:2] == ["config", "settings"]
    assert expand_terms("les réglages") == ["config", "settings", "parametrage", "parameter"]


def test_identifier_is_auth_vocabulary_only_in_its_reflexive_form():
    """« identifier » nu est ambigu : nom anglais (« the identifier of the User model »), verbe transitif français
    (« identifier une commande en double » = reconnaître) ; seul le pronominal « s'identifier » veut dire se connecter.
    Développé en auth/login/… dans tous les cas, il ramenait AuthController et POST /login (rangs 4-5) sur une
    question anglaise à propos du modèle User d'un projet réel. Les formes élidées (« s'identifier », « m'identifier »,
    « t'identifier », « s'identifie(nt) ») sont des expressions cherchées dans le texte entier, apostrophe typographique
    comprise ; « identification » reste développé et « identifier » reste un mot courant (jamais tenté comme nom de
    méthode nu)."""
    auth = ["auth", "login", "logout", "token", "jwt", "session"]
    assert analyze_query("what is the identifier of the order", today=TODAY).expansions == []
    assert analyze_query("what is the identifier of the User model and how is it validated", today=TODAY).expansions == []
    assert analyze_query("comment identifier une commande en double", today=TODAY).expansions == []
    assert analyze_query("comment s’identifier sur l’api", today=TODAY).expansions == auth  # apostrophe typographique
    s = analyze_query("l'utilisateur doit s'identifier avant d'appeler l'api", today=TODAY)
    assert s.expansions == auth + ["user", "users"] and s.keywords == ["utilisateur", "identifier", "avant"]
    assert expand_terms("je veux m'identifier") == auth and expand_terms("tu dois t'identifier") == auth
    assert expand_terms("il s'identifie puis") == auth and expand_terms("ils s'identifient") == auth
    assert expand_terms("comment fonctionne l'identification") == auth
    assert is_common_word("identifier") and is_common_word("Identifier")


def test_expansions_carry_only_generic_code_vocabulary():
    """:data:`EXPANSIONS` traduit des mots de la langue vers le vocabulaire du code et des cadriciels (auth, store,
    fillable, rules…), jamais vers le nom d'une entité d'un projet particulier : une telle entrée serait le
    vocabulaire du seul projet qui l'a inspirée, pas celui du moteur. Un nom composé du domaine (« ordre de
    mission ») reste un mot-clé ordinaire, que le moteur retrouve par ses « mots collés » (``ordremission``)."""
    from develop.search import query as q
    generic = set(q._AUTH + q._CREATE + q._UPDATE + q._DELETE + q._LIST + q._FIELD + q._CONFIG + q._VALIDATION
                  + q._ROUTE + q._MODEL + q._PERMISSION + q._FILE + q._SEARCH) | {"user", "users", "status", "state"}
    assert {k: sorted(set(v) - generic) for k, v in q.EXPANSIONS.items() if set(v) - generic} == {}
    assert all(t.isascii() and t.isalpha() and t.islower() for v in q.EXPANSIONS.values() for t in v)
    assert expand_terms("un ordre de mission") == [] and analyze_query("un ordre de mission", today=TODAY).keywords == ["ordre", "mission"]
    assert SearchEngine._glued_names(["ordre", "mission"]) == ["ordremission"]


def test_common_words_and_code_shapes():
    assert all(is_common_word(w) for w in ("definition", "put", "get", "user", "User", "champs", "garde", "exemple"))
    assert not any(is_common_word(w) for w in ("OrderService", "ordremission", "login", "confrimOrder"))
    assert len(COMMON_WORDS) >= 400 and set(("definition", "put", "store", "index")) <= COMMON_WORDS
    assert all(has_code_shape(t) for t in ("OrderService", "confirmOrder", "total_amount", "Order.php", "v2-api", "a::b", "ABC", "x1"))
    assert not any(has_code_shape(t) for t in ("User", "definition", "Garde", "put", ""))


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


def test_same_text_indexed_as_file_and_as_document_is_one_result(engine, index):
    """Même PDF indexé comme fichier du dépôt et téléversé comme document : morceaux de même ``chunks.sha1``
    → un seul résultat, sous la source la mieux classée, qui cumule les preuves sans double compte."""
    db = index.db
    before = engine.search("confirmed_at", channels=["lexical"], limit=50)
    dico = next(h for h in before.hits if h.extra.get("document_id") == index.doc_dico and "confirmed_at" in h.text)
    assert dico.extra["sha1"] == db.get_chunk(dico.chunk_id).sha1  # empreinte transmise à l'assembleur de contexte
    # le même texte, morceau par morceau, indexé une seconde fois comme fichier du dépôt
    twin_file = FileRow(repo_id="backend", path="docs/dictionnaire-donnees.md", language="markdown", kind="doc", size=1)
    db.upsert_file(twin_file)
    originals = db.chunks_for("document", index.doc_dico)
    twins = [ChunkRow(text=c.text, line_start=c.line_start, line_end=c.line_end) for c in originals]
    db.replace_chunks("file", twin_file.id, twins)
    assert [c.sha1 for c in twins] == [c.sha1 for c in originals]
    after = engine.search("confirmed_at", channels=["lexical"], limit=50)
    same = [h for h in after.hits if h.extra.get("sha1") == dico.extra["sha1"]]
    assert len(same) == 1 and same[0].source_type in ("file", "document")
    kept = same[0]
    assert set(kept.extra["channel_scores"]) == {"lexical"}  # un seul apport lexical, pas deux
    assert kept.score == round(kept.extra["channel_scores"]["lexical"], 6) == dico.score
    shas = [h.extra["sha1"] for h in after.hits if h.extra.get("sha1")]
    assert len(shas) == len(set(shas))  # aucune empreinte en double dans tout le résultat
    # l'étape elle-même : le mieux classé gagne, le jumeau lui apporte ses canaux (meilleur apport par canal),
    # un morceau au texte différent de la même source reste ; un élément structuré n'a pas d'empreinte
    a = SearchHit(chunk_id=1, source_type="file", source_id=10, file_id=10, path="doc.pdf", line_start=1, line_end=5,
                  text="même texte", score=0.5, channels=["lexical"], reason="lexical",
                  extra={"sha1": "s", "channel_scores": {"lexical": 0.5}})
    s = SearchHit(source_type="symbol", source_id=7, file_id=10, path="doc.pdf", line_start=1, line_end=5, text="même texte",
                  score=0.45, channels=["identifier"], extra={"symbol_id": 7, "channel_scores": {"identifier": 0.45}})
    b = SearchHit(chunk_id=2, source_type="document", source_id=3, title="Doc", line_start=1, line_end=5,
                  text="même texte", score=0.42, channels=["graph"], reason="relié",
                  extra={"sha1": "s", "channel_scores": {"graph": 0.4, "lexical": 0.02}})
    c = SearchHit(chunk_id=3, source_type="document", source_id=3, title="Doc", line_start=6, line_end=9,
                  text="autre page", score=0.3, channels=["lexical"], extra={"sha1": "t", "channel_scores": {"lexical": 0.3}})
    out = SearchEngine._dedupe_and_diversify([a, s, b, c])
    # a a fusionné dans le symbole (mêmes lignes du même fichier) ; le jumeau b, arrivé après, cumule sur ce gagnant
    assert [(h.source_type, h.chunk_id) for h in out] == [("symbol", None), ("document", 3)]
    assert out[0].extra["merged_chunk_id"] == 1
    assert out[0].extra["channel_scores"] == {"identifier": 0.45, "lexical": 0.5, "graph": 0.4} and out[0].score == 1.35
    assert out[0].channels == ["identifier", "lexical", "graph"] and "relié" in out[0].reason
    assert SearchEngine._content_key(s) is None and SearchEngine._content_key(a) == "s"


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
        # « table » reste un mot-clé nu (générique, jamais un nom de symbole) ; « modèle » est un mot de
        # question, retiré d'office des mots-clés : l'un comme l'autre n'atteignent jamais le palier 3.
        if "modèle" in q:
            assert "modèle" not in r.query.keywords, q
        else:
            assert "table" in r.query.keywords, q
        reasons = " ".join(h.reason for h in r.hits)
        assert "nom de symbole exact « table »" not in reasons, (q, reasons)
        assert "nom de symbole exact « modèle »" not in reasons, (q, reasons)
        assert not any(h.source_type == "symbol" and h.extra["symbol_id"] == sym_table for h in r.hits), q
        assert r.hits[0].source_type == "model", (q, [(h.source_type, h.title) for h in r.hits[:3]])


def test_common_words_only_match_classes_models_or_members_of_cited_classes(engine, index):
    """Un mot courant sans forme de code (« definition », « put », « user ») ne désigne qu'une classe / un modèle
    ou un membre d'une classe citée dans la même question : les méthodes homonymes (factories Laravel
    ``definition``, ``Cache::put``, relation ``user()``) ne sont jamais remontées sans autre preuve."""
    f = FileRow(repo_id="backend", path="database/factories/OrderFactory.php", language="php", kind="code")
    index.db.upsert_file(f)
    ids = index.db.replace_file_artifacts(f.id, symbols=[
        SymbolRow(name="OrderFactory", kind="class", line_start=1, line_end=30),
        SymbolRow(name="definition", kind="method", parent="OrderFactory", line_start=10, line_end=20),
        SymbolRow(name="put", kind="method", parent="OrderFactory", line_start=22, line_end=25),
        SymbolRow(name="user", kind="method", parent="OrderFactory", line_start=26, line_end=28),
        SymbolRow(name="User", kind="class", line_start=29, line_end=30)])
    sym_factory, sym_definition, sym_put, sym_user_method, sym_user_class = ids["symbols"]
    # « definition » (sans accent, comme l'utilisateur l'écrit) : aucune factory, aucun nom de symbole exact
    r = engine.search("explique moi la definition des champs avec exemple")
    assert r.query.keywords == ["definition", "champs"] and r.warnings == []
    assert sym_definition not in [h.extra.get("symbol_id") for h in r.hits]
    assert "nom de symbole exact" not in " ".join(h.reason for h in r.hits)
    # « User » : la classe, pas la relation user() ; intention « fields » pour « décrire le modèle »
    r2 = engine.search("peux-tu me décrire le modèle User")
    assert r2.query.intent == "fields" and r2.query.keywords == ["user"] and r2.query.identifiers == ["User"]
    syms = [h for h in r2.hits if h.source_type == "symbol"]
    assert [h.extra["symbol_id"] for h in syms] == [sym_user_class] and "nom de classe exact « User »" in syms[0].reason
    # membre d'une classe citée dans la même question : « OrderFactory definition » désigne bien la méthode
    r3 = engine.search("OrderFactory definition", channels=["identifier"])
    assert {h.extra["symbol_id"] for h in r3.hits if h.source_type == "symbol"} == {sym_factory, sym_definition}
    assert any("membre « definition » de la classe citée OrderFactory" in h.reason for h in r3.hits)
    # cité explicitement entre accents graves : exact, comme avant
    r4 = engine.search("que fait `put` ?", channels=["identifier"])
    assert [h.extra["symbol_id"] for h in r4.hits] == [sym_put]
    # un mot courant n'est pas non plus une faute de frappe d'identifiant : pas de repli approché ni d'avertissement
    r5 = engine.search("definitions")
    assert r5.warnings == [] and not any(h.extra.get("approximate") for h in r5.hits)
    assert sym_user_method not in [h.extra.get("symbol_id") for h in engine.search("user").hits]


def test_common_word_used_as_code_name_matches_its_methods(engine, index):
    """Régression : les actions des contrôleurs (confirm, store, index, show, login…) sont presque toutes des mots
    courants ; quand la question **emploie** le mot comme un nom de code (:func:`word_names_symbol` : « la méthode
    confirm », « explique confirm », « où est défini confirm ? », « confirm() », « what does confirm do », « le code
    de confirm »), ses méthodes homonymes sont bien remontées (avant la règle des mots courants, « que fait la méthode
    confirm ? » donnait OrderService::confirm et OrderController::confirm aux rangs 1-2 ; après, aucun symbole). Le
    membre d'une classe citée est seul retenu (« la méthode confirm de OrderController »). La règle ne s'étend pas aux
    mots de la phrase : « la definition des champs » (factories), « tu put m'expliquer » (faute de frappe pour
    « peux »), « le modèle User » et « les champs de user » (relation ``user()``), « les logs »."""
    f = FileRow(repo_id="backend", path="database/factories/OrderFactory.php", language="php", kind="code")
    index.db.upsert_file(f)
    ids = index.db.replace_file_artifacts(f.id, symbols=[
        SymbolRow(name="OrderFactory", kind="class", line_start=1, line_end=30),
        SymbolRow(name="definition", kind="method", parent="OrderFactory", line_start=10, line_end=20),
        SymbolRow(name="put", kind="method", parent="OrderFactory", line_start=22, line_end=25),
        SymbolRow(name="user", kind="method", parent="OrderFactory", line_start=26, line_end=28),
        SymbolRow(name="logs", kind="method", parent="OrderFactory", line_start=29, line_end=30)])
    sym_factory, sym_definition, sym_put, sym_user_method, sym_logs = ids["symbols"]
    confirms = {index.sym_service_confirm, index.sym_controller_confirm}

    def symbols(q: str) -> List[int]:
        return [h.extra["symbol_id"] for h in engine.search(q, channels=["identifier"]).hits if h.source_type == "symbol"]

    # le mot employé comme nom de code : ses méthodes homonymes, avec la raison d'un nom exact
    for q in ("que fait la méthode confirm ?", "explique confirm", "explique-moi confirm", "où est défini confirm ?",
              "que fait confirm() ?", "what does confirm do", "le code de confirm", "l'action confirm du contrôleur"):
        assert set(symbols(q)) == confirms, q
        assert all(word_names_symbol("confirm", q) for _ in [0]), q
    r = engine.search("que fait la méthode confirm ?")
    top = [h for h in r.hits[:5] if h.source_type == "symbol" and h.extra["symbol_id"] in confirms]
    assert len(top) == 2 and all("identifier" in h.channels and "nom de symbole exact « confirm »" in h.reason for h in top)
    # classe citée : son membre seulement, pas l'homonyme de l'autre classe
    r2 = engine.search("que fait la méthode confirm de OrderController ?", channels=["identifier"])
    assert {h.extra["symbol_id"] for h in r2.hits if h.source_type == "symbol"} == {index.sym_controller, index.sym_controller_confirm}
    assert any("membre « confirm » de la classe citée OrderController" in h.reason for h in r2.hits)
    assert set(symbols("que fait la méthode definition de OrderFactory ?")) == {sym_factory, sym_definition}
    # mots de la phrase : rien ne change (classes et modèles seulement)
    assert sym_definition not in symbols("explique moi la definition des champs avec exemple")
    assert not {sym_put, sym_definition} & set(symbols("tu put m'expliquer la definition des champs"))
    assert sym_user_method not in symbols("peux-tu me décrire le modèle User")
    assert sym_user_method not in symbols("quels sont les champs de user ?")
    assert sym_logs not in symbols("où sont les logs ?") and sym_logs in symbols("explique logs")
    # la règle de la langue : un mot anglais seul dans une question française (et l'inverse) nomme du code
    assert question_language("que fait la méthode confirm ?") == "fr" and question_language("what does confirm do") == "en"
    assert question_language("OrderService confirm") is None and question_language("") is None
    assert word_names_symbol("confirm", "confirm c'est quoi") and word_names_symbol("calculer", "what does calculer do")
    assert not word_names_symbol("confirm", "confirm") and not word_names_symbol("template", "comment marche le template")
    assert not word_names_symbol("user", "la table de user") and not word_names_symbol("config", "le fichier de config")
    assert not word_names_symbol("user", "les champs de user", intent="fields")
    assert word_names_symbol("show", "quels champs renvoie la méthode show ?", intent="fields")
    assert not word_names_symbol("", "explique confirm") and not word_names_symbol("confirm", "")


# --------------------------------------------------------------------------- canal route

def test_route_channel_matches_keywords_in_paths_and_handlers(engine, index):
    f = FileRow(repo_id="backend", path="app/Http/Requests/ConfirmOrderRequest.php", language="php", kind="code")
    index.db.upsert_file(f)
    req_id = index.db.replace_file_artifacts(
        f.id, symbols=[SymbolRow(name="ConfirmOrderRequest", kind="class", line_start=5, line_end=20)])["symbols"][0]
    r = engine.search("comment faire pour confirmer une commande ?", limit=20)  # « confirmer » → radical « confirm »
    assert r.query.intent == "howto" and r.channel_counts["route"] == 2
    eps = {h.extra["endpoint_id"]: h for h in r.hits if h.source_type == "endpoint"}
    assert set(eps) == {index.ep_confirm, index.ep_client_confirm}
    server = eps[index.ep_confirm]
    assert "route" in server.channels and "route contenant « confirm »" in server.reason
    assert server.extra["handler"] == "OrderController@confirm" and server.extra["framework"] == "laravel"
    assert server.extra["request_class"] == "ConfirmOrderRequest" and server.extra["request_class_id"] == req_id
    assert server.extra["request_file_id"] == f.id and server.extra["route_terms"] == ["confirm"]
    assert server.extra["route_strong"] == 1  # « confirm » égale un segment (et l'action) : une preuve forte
    # les deux routes (serveur et appel client, qui se renvoient du crédit par le graphe) devant tous les morceaux
    assert {h.source_type for h in r.hits[:2]} == {"endpoint"} and server in r.hits[:2]
    assert eps[index.ep_client_confirm].extra["request_class"] == ""  # côté client : pas de classe de validation
    assert eps[index.ep_client_confirm].extra["request_class_id"] is None
    # le gestionnaire de la route trouvée arrive par le graphe (graine « route »)
    handler = next(h for h in r.hits if h.source_type == "symbol" and h.extra["symbol_id"] == index.sym_controller_confirm)
    assert "graph" in handler.channels and "via gestionnaire de route" in handler.reason
    # une route citée explicitement reste servie à l'exact par le canal identifier : ses mots n'alimentent pas le canal
    r2 = engine.search("POST /orders/{id}/confirm")
    assert r2.channel_counts["route"] == 0 and r2.hits[0].source_type == "endpoint"
    assert index.ep_show not in [h.extra["endpoint_id"] for h in r2.hits if h.source_type == "endpoint"]
    # le gestionnaire ne « contient » un terme que mot entier : « orders » ne désigne pas OrderService::confirm
    notif = index.db.replace_file_artifacts(index.svc.id, endpoints=[EndpointRow(
        side="client", method="POST", path="http://localhost:3001/notifications", normalized_path="/notifications",
        handler="OrderService::confirm", framework="guzzle", line=15)])["endpoints"][0]
    r3 = engine.search("les commandes orders", channels=["route"])
    ids = [h.extra["endpoint_id"] for h in r3.hits]
    assert index.ep_list in ids and notif not in ids
    assert handler_words("ShipmentController@storeFromOrder") == {"shipment", "controller", "store", "from", "order"}
    assert route_contains(index.db.get_endpoint(notif), "confirm") and not route_contains(index.db.get_endpoint(notif), "orders")
    # le radical « order » (de « orders ») est bien un mot du gestionnaire, mais un radical ne vaut que dans le chemin
    assert route_contains(index.db.get_endpoint(notif), "order") and not route_contains(index.db.get_endpoint(notif), "order", path_only=True)
    # une question de structure (« quels champs ») ne vise pas une route : canal muet
    r4 = engine.search("quels champs a la table orders ?")
    assert r4.query.intent == "fields" and r4.channel_counts["route"] == 0 and r4.hits[0].source_type == "model"
    # noms de classes de validation essayés, par priorité
    assert request_class_names("ShipmentController@storeFromOrder")[:3] == [
        "StoreFromOrderShipmentRequest", "StoreShipmentFromOrderRequest", "ShipmentStoreFromOrderRequest"]
    assert "StoreShipmentRequest" in request_class_names("App\\Http\\Controllers\\ShipmentController@store")
    assert request_class_names("closure") == [] and request_class_names("") == []


def test_route_channel_generic_terms_only_qualify(engine, index):
    """Un terme présent dans plus de ROUTE_LIMIT routes (« api », « store ») ne produit pas de candidat : il ne
    fait que départager les routes trouvées par un terme précis."""
    f = FileRow(repo_id="backend", path="routes/widgets.php", language="php", kind="code")
    index.db.upsert_file(f)
    eps = [EndpointRow(side="server", method="GET", path="/widgets/%d" % i, normalized_path="/widgets/%d" % i,
                       handler="WidgetController@show%d" % i, framework="laravel", line=i + 1)
           for i in range(ROUTE_LIMIT + 1)]
    eps.append(EndpointRow(side="server", method="POST", path="/widgets/special", normalized_path="/widgets/special",
                           handler="WidgetController@special", framework="laravel", line=40))
    ids = index.db.replace_file_artifacts(f.id, endpoints=eps)
    r0 = engine.search("widgets", channels=["route"])
    assert r0.query.keywords == ["widgets"] and r0.hits == []  # trop générique : rien
    r = engine.search("special widgets", channels=["route"])
    assert [h.extra["endpoint_id"] for h in r.hits] == [ids["endpoints"][-1]]
    assert r.hits[0].reason == "route contenant « special », « widgets »"  # le terme générique départage
    assert r.hits[0].extra["route_terms"] == ["special", "widgets"]  # le radical « widget » n'est pas compté deux fois
    assert r.hits[0].extra["route_strong"] == 2 and r.hits[0].extra["request_class"] == ""


def test_expansion_terms_feed_secondary_lexical_query_and_route_channel(engine, index):
    r = engine.search("comment lister les commandes ?", limit=50)
    assert r.query.expansions == ["index", "list", "all"] and r.query.keywords == ["lister", "commandes"]
    listing = next(h for h in r.hits if h.source_type == "endpoint" and h.extra["endpoint_id"] == index.ep_list)
    assert "route" in listing.channels and "route contenant « index »" in listing.reason  # « lister » → index
    assert listing.extra["route_strong"] == 1 and listing.extra["handler"] == "OrderController@index"
    lex = engine.search("comment lister les commandes ?", channels=["lexical"], limit=50)
    expanded = [h for h in lex.hits if h.extra.get("expanded")]
    direct = [h for h in lex.hits if not h.extra.get("expanded")]
    assert direct and all("correspondance lexicale sur « commande" in h.reason for h in direct)
    assert expanded and all(h.reason.startswith("correspondance lexicale élargie sur « ") for h in expanded)
    assert all(set(h.extra["fts_terms"]) <= {"index", "list", "all"} for h in expanded)
    assert all(h.score <= W_EXPANSION / (RRF_K + 1) + 1e-6 for h in expanded)  # poids plus faible (score arrondi)
    assert lex.hits.index(expanded[0]) > lex.hits.index(direct[-1])  # jamais devant un morceau des mots-clés
    assert engine.search("montant commande", channels=["lexical"]).hits and \
        not any(h.extra.get("expanded") for h in engine.search("montant commande", channels=["lexical"]).hits)


def test_generated_or_huge_files_are_penalised_in_lexical_channel(engine, index):
    big = FileRow(repo_id="backend", path="storage/api-docs/api-docs.json", language="json", kind="data", size=1_200_000)
    index.db.upsert_file(big)
    big_chunks = index.db.replace_file_artifacts(big.id, chunks=[
        ChunkRow(text="commande commande commande commande client %d" % i, line_start=i * 10 + 1, line_end=i * 10 + 9)
        for i in range(3)])["chunks"]
    dump = FileRow(repo_id="backend", path="docker/dump.sql", language="sql", kind="data", size=400_000)
    index.db.upsert_file(dump)
    dump_chunk = index.db.replace_file_artifacts(
        dump.id, chunks=[ChunkRow(text="INSERT INTO commande (client) VALUES (1)", line_start=1, line_end=1)])["chunks"][0]
    # dans le canal : les 3 morceaux d'api-docs seraient en tête (BM25) ; un seul y reste, les 2 autres vont en queue
    cands = engine._channel_lexical(analyze_query("commande", today=TODAY), ["file", "document", "note"], _Loader(index.db))
    positions = [i for i, c in enumerate(cands) if c.chunk_id in big_chunks]
    assert len(cands) > 6 and positions[0] < 3 and positions[1:] == [len(cands) - 2, len(cands) - 1]
    penalised = set(big_chunks) | {dump_chunk}
    assert all(bool(c.extra.get("generated")) == (c.chunk_id in penalised) for c in cands)
    assert all(c.weight == GENERATED_PENALTY and c.reason.endswith("(fichier généré volumineux)")
               for c in cands if c.chunk_id in penalised)
    assert all(c.weight == 1.0 and "généré" not in c.reason for c in cands if c.chunk_id not in penalised)
    # dans les résultats : score divisé par deux, derrière tous les morceaux normaux
    r = engine.search("commande", channels=["lexical"], limit=50)
    gen_hits = [h for h in r.hits if h.extra.get("generated")]
    normal = [h for h in r.hits if not h.extra.get("generated")]
    assert gen_hits and normal and all("fichier généré volumineux" in h.reason for h in gen_hits)
    assert all(h.score <= GENERATED_PENALTY / (RRF_K + 1) + 1e-6 for h in gen_hits)  # score arrondi à 6 décimales
    assert r.hits.index(gen_hits[0]) >= len(normal)
    assert {h.path for h in gen_hits} == {"storage/api-docs/api-docs.json", "docker/dump.sql"}
    # règle de détection
    for path, size in (("storage/api-docs/api-docs.json", 10), ("public/build/app.js", 10), ("composer.lock", 1),
                       ("package-lock.json", 1), ("assets/app.min.js", 1), ("docs/swagger.json", 1), ("openapi.yaml", 1),
                       ("docker/dump.sql", 400_000), ("x/storage/logs/a.log", 1)):
        assert is_generated_file(path, size), path
    for path, size in (("docker/init.sql", 900), ("Documentation/Swagger/AuthControllerSwagger.php", 2000),
                       ("app/Models/Order.php", 500), ("storage.php", 1), (None, None)):
        assert not is_generated_file(path, size), path


def test_source_modules_named_storage_are_not_generated_files(engine, index):
    """Seul le dossier d'exécution Laravel (``storage/app|framework|logs/``…, en minuscules, à la racine ou dans un
    sous-projet) est un fichier généré : un module source nommé ``storage`` (ce dépôt même, Go, TS, PHP, C#)
    garde son apport lexical entier et aucune raison « fichier généré volumineux »."""
    for path in ("develop/storage/db.py", "src/storage/s3.ts", "internal/storage/db.go", "app/Storage/FileStore.php",
                 "storage/db.go", "Services/Storage/Logs/Writer.cs", "storage/README.md"):
        assert not is_generated_file(path, 1000), path
    for path in ("storage/app/public/scan.pdf", "storage/framework/views/9f3a.php", "storage/logs/laravel.log",
                 "backend/storage/app/export.csv", "storage/debugbar/req.json", "storage/clockwork/a.json"):
        assert is_generated_file(path, 10), path
    assert not is_generated_file("Storage/App/x.php", 10)  # la casse compte : ce n'est pas le dossier Laravel
    # dans le canal lexical : le module source garde le poids 1.0, le dump sous storage/app/ est pénalisé
    module = FileRow(repo_id="backend", path="develop/storage/db.py", language="python", kind="code", size=20_000)
    index.db.upsert_file(module)
    module_chunk = index.db.replace_file_artifacts(module.id, chunks=[
        ChunkRow(text="def tarification_commande(db): return db.query('commande tarification')", line_start=1,
                 line_end=1)])["chunks"][0]
    dump = FileRow(repo_id="backend", path="storage/app/exports/commandes.csv", language="csv", kind="data", size=20_000)
    index.db.upsert_file(dump)
    dump_chunk = index.db.replace_file_artifacts(dump.id, chunks=[
        ChunkRow(text="commande;tarification;client", line_start=1, line_end=1)])["chunks"][0]
    cands = {c.chunk_id: c for c in engine._channel_lexical(analyze_query("tarification commande", today=TODAY),
                                                            ["file", "document", "note"], _Loader(index.db))}
    assert cands[module_chunk].weight == 1.0 and "généré" not in cands[module_chunk].reason
    assert not cands[module_chunk].extra.get("generated")
    assert cands[dump_chunk].weight == GENERATED_PENALTY and cands[dump_chunk].extra.get("generated")


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
        if h.extra.get("expanded"):  # requête secondaire (« champs » → field, fillable… ; « table » → model…) : autres termes
            assert h.reason.startswith("correspondance lexicale élargie sur « "), h.reason
            continue
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


def test_graph_channel_does_not_fan_out_through_route_and_model_files():
    """Éventail « via imports » mesuré sur un projet réel (jusqu'à 70 % du contexte à 16k) : le fichier de routes
    d'une route trouvée importe les douze contrôleurs du module, le fichier d'un modèle est importé par ses
    requêtes, politiques, fournisseurs et tests — autant d'en-têtes sans rapport avec la question. Ni l'un ni
    l'autre n'est une graine du graphe (le gestionnaire vient de ``handles`` ; le fichier du modèle reste écarté
    quand un mot nu de la question est son radical, « Shipment » → Shipment.php) ; seul le fichier d'une classe
    citée l'est, avec au plus GRAPH_IMPORTS_PER_SEED fichiers via imports, après ses autres relations ; une
    question d'impact garde tous les importateurs d'un modèle."""
    db = Database(":memory:")
    db.upsert_repo(RepoRow(id="r", name="r", path="/nulle-part", kind="backend"))

    def add(path, symbols=(), endpoints=(), models=()):
        f = FileRow(repo_id="r", path=path, language="php", kind="code")
        db.upsert_file(f)
        name = path.rsplit("/", 1)[-1][:-4]
        text = "<?php\nnamespace App;\nuse App\\A;\nuse App\\B;\nclass %s\n{\n    public function run() {}\n}" % name
        ids = db.replace_file_artifacts(f.id, symbols=list(symbols), endpoints=list(endpoints), models=list(models),
                                        chunks=[ChunkRow(text=text, line_start=1, line_end=8)])
        return f, ids

    routes, rt = add("routes/api.php", endpoints=[EndpointRow(side="server", method="POST", path="/shipments",
                                                             handler="ShipmentController@store", framework="laravel", line=3)])
    ctl, ct = add("app/Http/Controllers/ShipmentController.php",
                  symbols=[SymbolRow(name="ShipmentController", kind="class", line_start=5, line_end=8),
                           SymbolRow(name="store", kind="method", parent="ShipmentController", line_start=7, line_end=7)])
    other_controllers = [add("app/Http/Controllers/%sController.php" % n)[0] for n in ("FormTemplate", "Survey", "Order", "Review")]
    model_file, md = add("app/Models/Shipment.php", models=[ModelRow(name="Shipment", table_name="shipments", kind="eloquent",
                                                                   line=5, fields=[{"name": "carrier"}])])
    model_importers = [add(p)[0] for p in ("app/Http/Requests/StoreShipmentRequest.php", "app/Policies/ShipmentPolicy.php",
                                           "app/Providers/ShopServiceProvider.php", "tests/Feature/ShipmentControllerTest.php")]
    svc, _ = add("app/Services/ShipmentService.php", symbols=[SymbolRow(name="ShipmentService", kind="class", line_start=5, line_end=8)])
    svc_importers = [add("app/Jobs/Job%d.php" % i)[0] for i in range(5)]
    doc = db.add_document(DocumentRow(title="Guide des expéditions", kind="upload", format="md", sha1="guide"))
    db.replace_chunks("document", doc, [ChunkRow(text="Le service logistique clôture l'expédition après 48 heures.")])
    db.add_relations(
        [RelationRow(src=node_key("endpoint", rt["endpoints"][0]), dst=node_key("symbol", ct["symbols"][1]), kind="handles"),
         RelationRow(src=node_key("model", md["models"][0]), dst="table:shipments", kind="defines_table"),
         RelationRow(src=node_key("file", svc.id), dst=node_key("document", doc), kind="documented_by", confidence=0.6)]
        + [RelationRow(src=node_key("file", routes.id), dst=node_key("file", f.id), kind="imports") for f in [ctl] + other_controllers]
        + [RelationRow(src=node_key("file", f.id), dst=node_key("file", model_file.id), kind="imports") for f in model_importers]
        + [RelationRow(src=node_key("file", f.id), dst=node_key("file", svc.id), kind="imports") for f in svc_importers])
    engine = SearchEngine(db)

    r = engine.search("comment créer une Shipment avec POST /shipments", limit=40)
    assert r.query.intent == "howto" and r.channel_counts["graph"] >= 1
    handler = next(h for h in r.hits if h.source_type == "symbol" and h.extra["symbol_id"] == ct["symbols"][1])
    assert "relié à POST /shipments via gestionnaire de route" in handler.reason
    assert any(h.path == model_file.path and "nom de fichier « Shipment »" in h.reason for h in r.hits)
    by_graph = {h.path for h in r.hits if "graph" in h.channels}
    assert not any(f.path in by_graph for f in other_controllers), "le fichier de routes n'est pas une graine"
    assert not any(f.path in by_graph for f in model_importers), "le fichier du modèle n'est pas une graine"
    assert not any("via imports" in h.reason for h in r.hits)

    r2 = engine.search("ShipmentService", limit=40)
    via_imports = [h for h in r2.hits if "relié à ShipmentService via imports" in h.reason]
    assert len(via_imports) == GRAPH_IMPORTS_PER_SEED and all(h.path.startswith("app/Jobs/") for h in via_imports)
    guide = next(h for h in r2.hits if h.source_type == "document")
    assert "relié à ShipmentService via documentation" in guide.reason
    assert guide.score > max(h.score for h in via_imports)  # les imports passent après les autres relations

    r3 = engine.search("qui utilise le modèle Shipment ?", limit=40)
    assert r3.query.intent == "who_calls"
    importers = {h.path for h in r3.hits if "relié à Shipment via imports" in h.reason}
    assert importers == {f.path for f in model_importers}


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


def test_route_channel_ignores_variable_only_client_calls_and_ranks_server_routes_first(engine, index):
    """Sur un projet réel, « POST $url » (PartnerBPushService@login, chemin normalisé « / ») était 5e pour « quel est
    l'api d'authentification » : un appel client dont le chemin n'est qu'une variable ou une expression ne
    désigne aucune route et n'est jamais candidat, même si son gestionnaire contient le terme. À preuves fortes
    égales, la route serveur passe devant l'appel client qui la vise, quel que soit son nombre de termes."""
    f = FileRow(repo_id="backend", path="app/Services/PartnerBPushService.php", language="php", kind="code")
    index.db.upsert_file(f)
    dyn = index.db.replace_file_artifacts(f.id, endpoints=[
        EndpointRow(side="client", method="POST", path="$url", normalized_path="/", handler="PartnerBPushService@confirm",
                    framework="laravel-http", line=15),
        EndpointRow(side="client", method="POST", path="{$this->baseUrl}{$endpoint}", normalized_path="/",
                    handler="PartnerBPushService@confirmOrder", framework="laravel-http", line=30),
        EndpointRow(side="client", method="GET", path="{self::CONFIRM_URL}", normalized_path="/",
                    handler="PartnerBPushService@confirm", framework="laravel-http", line=40)])["endpoints"]
    assert all(is_dynamic_client_call(index.db.get_endpoint(i)) for i in dyn)
    assert is_dynamic_client_call(EndpointRow(side="client", method="GET", path="${API_BASE}", normalized_path="/"))
    for e in (index.db.get_endpoint(index.ep_confirm), index.db.get_endpoint(index.ep_client_confirm),
              EndpointRow(side="client", method="GET", path="documents/{id}/", normalized_path="/documents/:param"),
              EndpointRow(side="client", method="GET", path="https://example.com", normalized_path="/"),
              EndpointRow(side="client", method="GET", path="/", normalized_path="/"),
              EndpointRow(side="server", method="POST", path="$url", normalized_path="/")):  # côté serveur : hors sujet
        assert not is_dynamic_client_call(e)
    # le fetch() d'une vue : mêmes preuves fortes que la route (« confirm » segment du chemin) mais un terme de plus
    # (« commande » sous-chaîne de « commandes ») : il passait devant la route serveur
    view = FileRow(repo_id="backend", path="resources/views/commandes/show.blade.php", language="blade", kind="code")
    index.db.upsert_file(view)
    blade = index.db.replace_file_artifacts(view.id, endpoints=[EndpointRow(
        side="client", method="POST", path="/commandes/{id}/confirm", normalized_path="/commandes/:param/confirm",
        handler="submit", framework="fetch", line=40)])["endpoints"][0]
    r = engine.search("comment faire pour confirmer une commande ?", channels=["route"], limit=20)
    found = [h.extra["endpoint_id"] for h in r.hits]
    assert not set(found) & set(dyn), "« POST $url » n'est pas une route"
    assert found == [index.ep_confirm, blade, index.ep_client_confirm]
    assert [h.extra["route_strong"] for h in r.hits] == [1, 1, 1]
    assert r.hits[1].extra["route_terms"] == ["confirm", "commande"] and r.hits[2].extra["route_terms"] == ["confirm"]


def test_client_call_never_precedes_the_server_route_it_targets(engine, index):
    """Sur un projet réel, le fetch() de login.blade.php (« POST /partner_a_stats/login → submit ») était classé 1er pour
    « quel est l'api d'authentification », devant la vraie route POST login : la route est une graine du graphe
    (qui ne crédite que ses voisins), l'appel cumulait canal route + graphe (« appel d'API ») + morceau de la
    vue. Après la fusion, un appel client qui précède la route serveur qu'il vise (même chemin normalisé ou fin
    de chemin, méthode compatible) est replacé juste derrière elle, à son score ; un appel vers un service tiers
    ne bouge pas ; une question d'impact (« qui appelle ») garde l'ordre des preuves."""

    def ep(side: str, method: str, norm: str, score: float, eid: int) -> SearchHit:
        return SearchHit(source_type="endpoint", source_id=eid, title="%s %s" % (method, norm), score=score,
                         extra={"endpoint_id": eid, "side": side, "method": method, "normalized_path": norm})

    any_client = ep("client", "ANY", "/api/orders/:param/confirm", 0.041, 6)  # route('orders.confirm') d'une vue
    get_client = ep("client", "GET", "/orders/:param/confirm", 0.040, 5)
    client = ep("client", "POST", "/orders/:param/confirm", 0.036, 2)
    third_party = ep("client", "GET", "/documents/:param", 0.030, 3)  # service tiers : aucune route serveur
    sym = SearchHit(source_type="symbol", source_id=9, title="OrderController::confirm", score=0.028)
    server = ep("server", "POST", "/api/orders/:param/confirm", 0.025, 1)  # préfixe de base côté serveur
    get_server = ep("server", "GET", "/orders/:param/confirm", 0.020, 4)
    out = SearchEngine._server_routes_first([any_client, get_client, client, third_party, sym, server, get_server])
    assert [h.source_id for h in out] == [3, 9, 1, 2, 6, 4, 5]
    assert client.score == any_client.score == server.score == 0.025
    assert client.extra["after_server_route"] == any_client.extra["after_server_route"] == 1
    assert get_client.score == get_server.score and get_client.extra["after_server_route"] == 4  # la méthode compte
    assert "after_server_route" not in third_party.extra and third_party.score == 0.030
    # de bout en bout : la vue qui appelle la route cumule route + graphe + morceau, la route serveur reste devant
    view = FileRow(repo_id="backend", path="resources/views/orders/show.blade.php", language="blade", kind="code")
    index.db.upsert_file(view)
    text = "\n".join(["<script>"] + ["// confirmer la commande : bouton confirmer commande"] * 6
                     + ["function submit() { fetch('/orders/' + id + '/confirm', { method: 'POST' }); }", "</script>"])
    ids = index.db.replace_file_artifacts(view.id, endpoints=[EndpointRow(
        side="client", method="POST", path="/orders/{id}/confirm", normalized_path="/orders/:param/confirm",
        handler="submit", framework="fetch", line=8)], chunks=[ChunkRow(text=text, line_start=1, line_end=9)])
    blade = ids["endpoints"][0]
    index.db.add_relations([RelationRow(src=node_key("endpoint", blade), dst=node_key("endpoint", index.ep_confirm),
                                        kind="calls_endpoint", confidence=1.0)])
    r = engine.search("comment faire pour confirmer une commande ?", limit=20)
    order = [h.extra.get("endpoint_id") for h in r.hits]
    assert order.index(index.ep_confirm) < order.index(blade)
    moved = next(h for h in r.hits if h.extra.get("endpoint_id") == blade)
    assert moved.extra["after_server_route"] == index.ep_confirm and "graph" in moved.channels
    assert moved.score == next(h for h in r.hits if h.extra.get("endpoint_id") == index.ep_confirm).score
    # question d'impact : les appelants sont la réponse, l'ordre des preuves est conservé
    r2 = engine.search("qui appelle POST /orders/{order}/confirm ?", limit=20)
    assert r2.query.intent == "who_calls" and not any("after_server_route" in h.extra for h in r2.hits)


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
    limit = 6.0 if sys.platform == "win32" else 2.0  # exécuteurs Windows de la CI nettement plus lents
    assert elapsed < limit, "3 recherches sur 1500 morceaux en %.2fs" % elapsed


def test_make_snippet_strips_tags_and_caps_length():
    assert make_snippet("  <b>Bonjour</b>\n\n  le   monde ") == "Bonjour le monde"
    long = make_snippet("mot " * 200)
    assert len(long) <= 300 and long.endswith("…")
    assert make_snippet("") == "" and make_snippet(None) == ""  # type: ignore[arg-type]


# --------------------------------------------------------------------------- suivi de conversation (marqueur du chat)

def test_analyze_query_understands_the_followup_marker():
    """Le chat enrichit une question de suivi courte du marqueur ``[contexte précédent : termes]`` (identifiants,
    titres des sources citées, mots-clés du tour précédent) : l'analyse l'isole au lieu d'y lire « contexte » et
    « précédent » comme mots-clés."""
    from develop.search import FOLLOWUP_MARKER, split_followup
    q = "et les champs ? " + FOLLOWUP_MARKER % "Shipment StoreShipmentRequest paramétrage formulaires"
    assert split_followup(q) == ("et les champs ?", "Shipment StoreShipmentRequest paramétrage formulaires")
    assert split_followup("sans marqueur [x]") == ("sans marqueur [x]", "")
    a = analyze_query(q, today=TODAY)
    assert a.text == q and a.intent == "factual"
    # un terme capitalisé en tête de la liste est un nom (« Shipment »), pas un début de phrase
    assert a.identifiers == ["StoreShipmentRequest", "Shipment"]
    # mots-clés de la question d'abord, puis ceux du marqueur, jamais ses propres mots
    assert a.keywords == ["champs", "shipment", "storeshipmentrequest", "paramétrage", "formulaires"]
    assert a.previous_terms == ["shipment", "storeshipmentrequest", "paramétrage", "formulaires"]
    assert "contexte" not in a.keywords and "précédent" not in a.keywords and "previous_terms" in a.to_dict()
    assert a.lexical_query == "champs shipment storeshipmentrequest paramétrage formulaires"
    assert {"field", "column"} <= set(a.expansions) and "config" not in a.expansions  # « champs » développé, pas « paramétrage »
    # l'intention et les dates viennent de la question seule, les tables et identifiants des deux
    b = analyze_query("quels champs a la table orders ? " + FOLLOWUP_MARKER % "pourquoi décision statut hier", today=TODAY)
    assert b.intent == "fields" and b.tables == ["orders"] and (b.date_from, b.date_to) == (None, None)
    assert b.keywords == ["champs", "table", "orders", "décision", "statut"] and b.previous_terms == ["décision", "statut"]
    d = analyze_query("ouvre OrderService.php " + FOLLOWUP_MARKER % "contenu_configuration specification-formulaires-commandes", today=TODAY)
    assert d.paths == ["OrderService.php"]
    # un radical de fichier à tirets du marqueur est un nom entier, pas trois mots (voir le test dédié)
    assert d.identifiers == ["OrderService.php", "OrderService", "contenu_configuration", "specification-formulaires-commandes"]
    assert d.previous_terms == ["contenu_configuration", "specification-formulaires-commandes"]
    # marqueur seul : ses termes sont la question
    c = analyze_query(FOLLOWUP_MARKER % "Shipment formulaires", today=TODAY)
    assert c.identifiers == ["Shipment"] and c.keywords == ["shipment", "formulaires"] and c.previous_terms == []
    assert analyze_query("", today=TODAY).previous_terms == []


def test_route_channel_ignores_routes_found_by_action_words_alone(engine, index):
    """« créer » développe create/store/post/new : seuls, ces mots d'action trouvent la route de création de TOUTES
    les entités (GET /persons/create, GET /documents/create…). Dès qu'un terme précis a trouvé des routes, celles
    qui ne tiennent qu'à un mot d'action sont écartées ; sans terme précis, elles restent (rien de mieux)."""
    f = FileRow(repo_id="backend", path="routes/web.php", language="php", kind="code")
    index.db.upsert_file(f)
    ids = index.db.replace_file_artifacts(f.id, endpoints=[
        EndpointRow(side="server", method="GET", path="/widgets/create", normalized_path="/widgets/create",
                    handler="WidgetController@create", framework="laravel", line=3),
        EndpointRow(side="server", method="GET", path="/persons/create", normalized_path="/persons/create",
                    handler="PersonController@create", framework="laravel", line=4)])["endpoints"]
    r = engine.search("comment créer un order ?", channels=["route"])
    assert {"create", "store", "post", "new"} <= set(r.query.expansions)
    found = [h.extra["endpoint_id"] for h in r.hits]
    assert index.ep_list in found and index.ep_show in found and index.ep_confirm in found and not set(ids) & set(found)
    assert all("order" in h.extra["route_terms"] for h in r.hits)  # chaque route retenue tient à un terme précis
    alone = engine.search("comment créer ?", channels=["route"])
    assert set(ids) <= {h.extra["endpoint_id"] for h in alone.hits}
    assert alone.hits[0].extra["endpoint_id"] == ids[0] and alone.hits[0].reason == "route contenant « create »"
    # les hits structurés portent le nom du dépôt (en-tête « dépôt/chemin » des passages du chat), comme les morceaux
    assert r.hits[0].extra["repo_name"] == "backend"
    assert engine.search("OrderService").hits[0].extra["repo_name"] == "backend"
    assert engine.search("décris le modèle Order").hits[0].source_type == "model"
    assert all(h.extra.get("repo_name") == "backend" for h in engine.search("décris le modèle Order").hits if h.repo_id == "backend")


# --------------------------------------------------------------------------- suivi : termes hérités du tour précédent

def test_followup_marker_keeps_hyphenated_source_labels_whole():
    """Le marqueur de suivi porte les radicaux de fichiers des sources citées (« specification-formulaires-commandes ») : un
    tel libellé est un nom — identifiant et **un** mot-clé — et non trois mots ordinaires (« commandes ») que le moteur
    prenait pour des mots de la question (routes /commandes, modèle Commande devant le document cité). Dans la question, un
    tiret lie des mots de la langue (« peux-tu ») qui restent découpés ; ``previous_terms`` ne garde du marqueur que
    les mots que la question ne contient pas déjà (``keywords`` = mots de la question + ``previous_terms``)."""
    from develop.search import FOLLOWUP_MARKER
    q = "explique moi la définition des champs avec exemple"
    a = analyze_query(q + " " + FOLLOWUP_MARKER % "contenu_configuration specification-formulaires-commandes "
                      "Paramétrage des formulaires de commande paramétrage formulaires", today=TODAY)
    assert a.identifiers == ["contenu_configuration", "specification-formulaires-commandes"]
    assert a.keywords == ["définition", "champs", "contenu_configuration", "specification-formulaires-commandes", "paramétrage",
                          "formulaires", "commande"]
    assert a.previous_terms == ["contenu_configuration", "specification-formulaires-commandes", "paramétrage", "formulaires", "commande"]
    assert a.lexical_query == " ".join(a.keywords) and a.intent == "factual"
    b = analyze_query("et les formulaires ? " + FOLLOWUP_MARKER % "Paramétrage des formulaires de commande", today=TODAY)
    assert b.keywords == ["formulaires", "paramétrage", "commande"] and b.previous_terms == ["paramétrage", "commande"]
    assert b.identifiers == []  # un titre n'est pas un identifiant, « Paramétrage » (accent) non plus
    # marqueur seul : ses libellés restent entiers aussi ; un libellé avec chiffre était déjà un identifiant
    c = analyze_query(FOLLOWUP_MARKER % "guide-frontend-commandes v2-api", today=TODAY)
    assert c.identifiers == ["v2-api", "guide-frontend-commandes"] and c.keywords == ["guide-frontend-commandes", "v2-api"]
    assert c.previous_terms == []
    # dans une phrase, rien ne change : « peux-tu » est fait de mots vides, « point-de-vente » de mots-clés
    assert analyze_query("peux-tu me décrire le modèle User ?", today=TODAY).keywords == ["user"]
    assert analyze_query("comment ouvrir un point-de-vente ?", today=TODAY).keywords == ["ouvrir", "point", "vente"]
    assert "point-de-vente" not in analyze_query("comment ouvrir un point-de-vente ?", today=TODAY).identifiers


def test_route_channel_inherited_words_only_qualify(engine, index):
    """Un mot hérité du tour précédent qui n'est pas un identifiant (mot d'un titre cité, mot-clé de la question
    précédente) ne désigne aucune route : il ne fait que départager les routes trouvées par la question elle-même.
    Sinon « et les champs ? » après le document « Paramétrage des formulaires de commande » ramenait GET /commandes,
    POST /commandes… devant le document. Un identifiant hérité (« Order ») reste un terme à part entière."""
    from develop.search import FOLLOWUP_MARKER
    from develop.search.engine import inherited_words
    marker = FOLLOWUP_MARKER % "Dictionnaire de données orders"
    r = engine.search("et les champs ? " + marker, channels=["route"], limit=50)
    assert r.query.identifiers == ["Dictionnaire"] and r.query.previous_terms == ["dictionnaire", "données", "orders"]
    assert inherited_words(r.query) == {"donnees", "orders"}
    terms, qualifiers = engine._route_terms(r.query)
    assert "orders" not in [t for t, _ in terms] and ("orders", False) in qualifiers and ("order", True) in qualifiers
    assert r.hits == []  # « orders » hérité ne trouve pas /orders…
    assert engine.search("et les champs ? orders", channels=["route"]).hits  # … alors que le même mot dans la question, si
    # il départage : « confirm » (question) trouve POST /orders/{order}/confirm, « orders » hérité compte comme preuve
    c = engine.search("et confirm ? " + marker, channels=["route"])
    assert c.hits and c.hits[0].extra["endpoint_id"] == index.ep_confirm
    assert c.hits[0].reason == "route contenant « confirm », « orders »" and c.hits[0].extra["route_strong"] == 2
    assert c.hits[0].extra["route_terms"] == ["confirm", "orders"]  # le radical « order » n'est pas compté deux fois
    # un identifiant hérité désigne : « Order » (modèle cité au tour précédent) trouve les routes /orders
    o = engine.search("et les champs ? " + FOLLOWUP_MARKER % "Order", channels=["route"], limit=50)
    assert inherited_words(o.query) == set() and engine._route_terms(o.query)[1] == []
    assert {index.ep_list, index.ep_show, index.ep_confirm} <= {h.extra["endpoint_id"] for h in o.hits}


def test_identifier_channel_does_not_try_inherited_words_as_bare_names(engine, index):
    """« commande » d'un titre cité au tour précédent n'est pas un nom que l'utilisateur a tapé : le palier des mots nus
    l'ignore (il ramenait le modèle Commande et sa migration devant le document cité). Un modèle réellement cité au tour
    précédent figure dans le marqueur par son libellé, un identifiant, qui garde son poids."""
    from develop.search import FOLLOWUP_MARKER
    r = engine.search("et les champs ? " + FOLLOWUP_MARKER % "Dictionnaire de données orders", channels=["identifier"], limit=50)
    assert not any(h.source_type == "model" for h in r.hits)  # ni Order (table orders) ni la migration orders
    plain = engine.search("et les champs ? orders", channels=["identifier"], limit=50)
    assert {h.extra["model_id"] for h in plain.hits if h.source_type == "model"} == {index.model_order, index.model_orders_migration}
    named = engine.search("et les champs ? " + FOLLOWUP_MARKER % "Order", channels=["identifier"], limit=50)
    model = next(h for h in named.hits if h.source_type == "model" and h.extra["model_id"] == index.model_order)
    assert model.reason == "modèle exact « Order »" and named.hits[0].title in ("Order", "Order.php", "app/Models/Order.php")


def test_lexical_channel_does_not_require_the_previous_turn_terms(engine, index):
    """Pour une question de suivi, la requête lexicale principale ne porte que les mots de la question : exigés en ET,
    les termes hérités filtraient (« montant total » + « OrderService » hérité n'acceptait que la ligne du dictionnaire
    qui cite OrderService::confirm, et perdait la règle de calcul de la spécification). Les termes hérités forment une
    requête secondaire en OU à :data:`W_PREVIOUS`, après les morceaux de la question ; un morceau trouvé par les deux
    cumule leurs apports."""
    from develop.search import FOLLOWUP_MARKER
    from develop.search.engine import RRF_K, W_PREVIOUS
    q = "quel est le montant total ?"
    alone = engine.search(q, channels=["lexical"], limit=50)
    follow = engine.search(q + " " + FOLLOWUP_MARKER % "OrderService", channels=["lexical"], limit=50)
    assert follow.query.keywords == ["montant", "total", "orderservice"] and follow.query.previous_terms == ["orderservice"]
    spec_rule = next(h for h in alone.hits if h.source_type == "document" and "somme des lignes" in h.text)
    dico_orders = next(h for h in alone.hits if h.source_type == "document" and "OrderService::confirm" in h.text)
    assert spec_rule.chunk_id in {h.chunk_id for h in follow.hits}  # la règle de calcul n'est plus filtrée
    assert follow.hits[0].chunk_id == dico_orders.chunk_id  # trouvé par les deux requêtes : en tête (poids 1 + W_PREVIOUS)
    assert follow.hits[0].extra["previous"] is True and follow.hits[0].reason.startswith("correspondance lexicale sur « ")
    assert follow.hits[0].reason.endswith(" ; avec le tour précédent sur « orderservice »")
    assert abs(follow.hits[0].score - (1.0 + W_PREVIOUS) / (RRF_K + 2)) < 1e-6  # rang 2 de la requête principale, deux poids
    previous = [h for h in follow.hits if h.reason.startswith("correspondance lexicale avec le tour précédent")]
    assert previous and all(h.extra["fts_terms"] == ["orderservice"] for h in previous)
    assert {h.file_id for h in previous} >= {index.svc.id, index.ctl.id}  # OrderService.php et son importateur
    assert all(h.score <= W_PREVIOUS / (RRF_K + 1) + 1e-6 for h in previous)
    direct = [h for h in follow.hits if h.reason.startswith("correspondance lexicale sur")]
    assert follow.hits.index(previous[0]) > follow.hits.index(direct[-1])  # jamais devant un morceau de la question
    # sans mot propre à la question (« et alors ? » : mots vides), les termes hérités restent la requête principale
    only = engine.search("et alors ? " + FOLLOWUP_MARKER % "OrderService", channels=["lexical"], limit=50)
    assert only.query.keywords == ["orderservice"] and only.hits and not any(h.extra.get("previous") for h in only.hits)
    assert {h.file_id for h in only.hits} >= {index.svc.id, index.ctl.id}
