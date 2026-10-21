"""Banc d'essai de la chaîne de récupération du chat (recherche → budget → contexte) sur ``sample_project``.

Équivalent générique, sur la fixture publique, du banc d'essai mené sur un vrai projet : huit questions en
français telles qu'un développeur les pose, avec les sources attendues (``expected_any`` dans le top 5 de la
recherche, ``must_not_top3``, ``expected_all_in_top10``), ce que le contexte envoyé au modèle doit contenir
(fiches structurées, champs exacts, route) et le budget de contexte à ``num_ctx`` 8192 et 16384 — le tout
sans LLM ni modèle d'embeddings (identifiants + routes + lexical + graphe), comme sur la machine d'un
utilisateur sans embeddings.

Le banc reproduit l'orchestrateur : une question de suivi courte est enrichie des termes du tour précédent
(:data:`develop.search.FOLLOWUP_MARKER`), le budget vient de :meth:`ChatOrchestrator._plan` (consignes
système réellement mesurées, définitions d'outils natifs comprises) et le contexte de :func:`build_context`
avec les fiches. ``pytest -s tests/test_retrieval_bench.py`` imprime le tableau récapitulatif.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest

from develop.chat.context_builder import build_context
from develop.chat.orchestrator import ChatOrchestrator, followup_terms, is_smalltalk
from develop.config import RepoConfig, WorkspaceConfig, load_workspace, save_workspace
from develop.indexing import Indexer
from develop.llm.base import ModelInfo, NullClient
from develop.search import FOLLOWUP_MARKER, SearchEngine, SearchHit
from develop.storage import Database

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"
WINDOWS = (8192, 16384)
MIN_CONTEXT_TOKENS = {8192: 3000, 16384: 7500}  # objectifs du budget (mode natif, question factuelle)

# Huit questions réalistes + une petite conversation. Les attentes portent sur des chemins (suffixes), des titres
# ou des noms de gestionnaires / classes de requête ; ``context`` liste des extraits qui doivent figurer dans le
# texte du contexte (champs exacts d'une fiche, règle d'un document, route).
QUESTIONS: List[Dict[str, Any]] = [
    {"id": "fields_orders", "text": "quels champs a la table orders ?", "intent": "fields",
     "expected_any": ["database/migrations/2024_01_01_000000_create_orders_table.php", "app/Models/Order.php",
                      "dictionnaire-donnees.md"],
     "expected_cards": ["Modèle Order"],
     "context": ["reference (string, requis)", "customer_id (foreignId, requis)",
                 "status (enum draft|confirmed|shipped|cancelled, requis, défaut draft)", "total_amount", "confirmed_at"]},
    {"id": "confirm_api", "text": "comment confirmer une commande via l'api ?",
     "expected_any": ["routes/api.php", "OrderController", "src/api/client.ts"],
     "expected_all_in_top10": ["OrderController@confirm", "specification-commandes.md"],
     "expected_endpoint": "POST /orders/{order}/confirm",
     "context": ["Seule une commande `draft` peut être confirmée"]},
    {"id": "who_calls", "text": "qui appelle OrderService::confirm ?", "intent": "who_calls",
     "expected_any": ["app/Http/Controllers/OrderController.php"], "expected_top3": True,
     "expected_all_in_top10": ["OrderController", "routes/api.php", "src/api/client.ts"],
     "expected_cards": ["Classe OrderService"],
     "context": ["symbole OrderController::confirm", "confirm(Order $order): Order"]},
    {"id": "decision_paid", "text": "pourquoi n'a-t-on pas ajouté le statut paid ?", "intent": "decision",
     "expected_any": ["docs/decisions/2026-09-01-reunion-statuts.md", "2026-09-01-reunion-statuts.md"], "expected_top3": True,
     "context": ["on n'ajoute PAS de statut `paid`", "paid_at"]},
    {"id": "customer_model", "text": "décris-moi le modèle Customer", "intent": "fields",
     "expected_any": ["app/Models/Customer.php"], "expected_top3": True,
     "expected_cards": ["Modèle Customer"],
     # sans migration, les types viennent de la table SQL de référence (schema.sql)
     "context": ["email (VARCHAR(255), requis)", "full_name (VARCHAR(255), requis)", "phone (VARCHAR(30), nullable)",
                 "has_many Order", "GET /customers"]},
    {"id": "customer_search_api", "text": "quelle est l'api pour rechercher un client ?",
     "expected_any": ["src/api/client.ts", "routes/api.php", "CustomerController"],
     "expected_all_in_top10": ["CustomerController@index", "src/components/CustomerSearch.tsx"],
     "expected_endpoint": "GET /customers", "expected_cards": ["Route GET /customers"],
     "context": ["CustomerController@index"]},
    {"id": "followup_orders", "text": "et ses commandes ?", "previous_question": "décris-moi le modèle Customer",
     "expected_any": ["app/Models/Customer.php"], "expected_top3": True,
     "expected_all_in_top10": ["dictionnaire-donnees.md"], "expected_cards": ["Modèle Customer"],
     "context": ["has_many Order"]},
    {"id": "total_amount", "text": "comment est calculé le montant total d'une commande ?",
     "expected_any": ["specification-commandes.md", "dictionnaire-donnees.md"], "expected_top3": True,
     "expected_all_in_top10": ["app/Services/OrderService.php"],
     "must_not_top3": ["node_modules", "package.json"],
     "context": ["quantité × prix unitaire", "Montant total en centimes"]},
    {"id": "smalltalk", "text": "bonjour", "smalltalk": True},
]
IDS = [q["id"] for q in QUESTIONS]


# --------------------------------------------------------------------------- aides

def _label(h: SearchHit) -> str:
    return "%s%s" % (h.path or h.title or "", (" → %s" % h.extra["handler"]) if h.extra.get("handler") else "")


def _matches(h: SearchHit, needle: str) -> bool:
    """Un hit correspond à une attente par son chemin, son titre, son gestionnaire ou sa classe de requête."""
    n = needle.lower()
    stem = n.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    hay = [h.path or "", h.title or "", str(h.extra.get("handler") or ""), str(h.extra.get("request_class") or ""),
           str(h.extra.get("source_path") or "")]
    return any(n in x.lower() or (bool(stem) and stem in x.lower()) for x in hay)


def _endpoint_in(text: str, expected: str) -> bool:
    method, path = expected.split(" ", 1)
    return re.search(r"\b%s\s+/?%s\b" % (re.escape(method), re.escape(path.lstrip("/"))), text) is not None


def _search_text(q: Dict[str, Any]) -> str:
    """Question telle que l'orchestrateur la cherche : enrichie du tour précédent quand c'est un suivi."""
    text = q["text"]
    if q.get("previous_question"):
        terms = followup_terms(text, q["previous_question"], [])
        if terms:
            return "%s %s" % (text, FOLLOWUP_MARKER % " ".join(terms))
    return text


# --------------------------------------------------------------------------- fixture : index de sample_project

@pytest.fixture(scope="module")
def bench(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    base = tmp_path_factory.mktemp("bench")
    home = base / "home"
    home.mkdir()
    project = base / "sample_project"
    shutil.copytree(FIXTURE, project)  # la fixture n'est jamais modifiée
    previous = os.environ.get("DEVELOP_HOME")
    os.environ["DEVELOP_HOME"] = str(home)
    ws = save_workspace(WorkspaceConfig(id="boutique", name="Boutique", repos=[
        RepoConfig(id="frontend", name="frontend", path=str(project / "frontend"), kind="frontend"),
        RepoConfig(id="backend", name="backend", path=str(project / "backend"), kind="backend"),
        RepoConfig(id="notifier", name="notifier", path=str(project / "services" / "notifier"), kind="backend"),
        RepoConfig(id="docs", name="docs", path=str(project / "docs"), kind="docs"),
        RepoConfig(id="database", name="database", path=str(project / "database"), kind="database"),
    ]))
    assert load_workspace("boutique") is not None
    db = Database(home / "workspaces" / "boutique" / "memory.sqlite3")
    try:
        run = Indexer(db, ws).run()  # sans moteur : aucun embedding, comme chez un utilisateur sans modèle d'embeddings
        assert run.status == "ok", run.error
        engine = SearchEngine(db)
        # modèle avec outils natifs et grande fenêtre : num_ctx effectif = valeur configurée (pas de profil machine)
        info = ModelInfo(name="fake:7b", context_length=131072, capabilities=["completion", "tools"])
        orch = ChatOrchestrator(db, ws, NullClient(), search_engine=engine, profile=None, model_info=info, today="2026-10-21")
        yield SimpleNamespace(db=db, ws=ws, engine=engine, orch=orch, runs={})
    finally:
        db.close()
        if previous is None:
            os.environ.pop("DEVELOP_HOME", None)
        else:
            os.environ["DEVELOP_HOME"] = previous


def run_question(bench: SimpleNamespace, q: Dict[str, Any], num_ctx: int) -> SimpleNamespace:
    """Recherche (limit 30), budget (consignes mesurées, outils natifs) et contexte d'une question, en cache."""
    key = (q["id"], num_ctx)
    if key not in bench.runs:
        search_text = _search_text(q)
        result = bench.engine.search(search_text, limit=30)
        bench.ws.llm.num_ctx = num_ctx
        budget, _system = bench.orch._plan("native", result.query.intent, None, q["text"])
        pack = build_context(result.hits, budget.context_tokens_max, result.query.intent, db=bench.db)
        bench.runs[key] = SimpleNamespace(search_text=search_text, result=result, hits=result.hits, budget=budget, pack=pack)
    return bench.runs[key]


def question(qid: str) -> Dict[str, Any]:
    return next(q for q in QUESTIONS if q["id"] == qid)


# --------------------------------------------------------------------------- tests

def _count(db: Database, table: str) -> int:
    return int(db.query("SELECT count(*) AS n FROM %s" % table)[0]["n"])


def test_bench_index_is_complete_and_has_no_embeddings(bench: SimpleNamespace):
    assert _count(bench.db, "files") >= 20 and _count(bench.db, "symbols") >= 40 and _count(bench.db, "endpoints") >= 10
    assert _count(bench.db, "data_models") >= 8 and _count(bench.db, "relations") > 50 and _count(bench.db, "embeddings") == 0
    r = bench.engine.search("OrderService")
    assert r.semantic_available is False and r.hits and r.hits[0].title == "OrderService"


@pytest.mark.parametrize("qid", [q["id"] for q in QUESTIONS if not q.get("smalltalk")])
def test_bench_search_finds_the_expected_sources(bench: SimpleNamespace, qid: str):
    q = question(qid)
    run = run_question(bench, q, 8192)
    hits, exp = run.hits, q["expected_any"]
    top = [(h.source_type, _label(h), h.reason[:60]) for h in hits[:10]]
    if q.get("intent"):
        assert run.result.query.intent == q["intent"], run.result.query
    assert any(_matches(h, e) for h in hits[:5] for e in exp), (qid, top)
    if q.get("expected_top3"):
        assert any(_matches(h, e) for h in hits[:3] for e in exp), (qid, top)
    for bad in q.get("must_not_top3", []):
        assert not any(_matches(h, bad) for h in hits[:3]), (qid, bad, top)
    for needed in q.get("expected_all_in_top10", []):
        assert any(_matches(h, needed) for h in hits[:10]), (qid, needed, top)
    assert hits == sorted(hits, key=lambda h: -h.score) and all(h.channels and h.reason for h in hits)
    assert run.result.semantic_available is False  # banc sans embeddings : lexical + identifiants + routes + graphe


@pytest.mark.parametrize("qid", [q["id"] for q in QUESTIONS if not q.get("smalltalk")])
def test_bench_context_at_8192_holds_cards_fields_and_routes(bench: SimpleNamespace, qid: str):
    q = question(qid)
    run = run_question(bench, q, 8192)
    budget, pack = run.budget, run.pack
    # régression de départ : 1 081 tokens et 2 à 5 sources à 8192 ; objectif ≥ 3 000 tokens disponibles
    assert budget.num_ctx == 8192 and budget.tools_mode == "native" and budget.context_tokens_max >= MIN_CONTEXT_TOKENS[8192]
    assert budget.total <= 8192 and pack.tokens_used <= budget.context_tokens_max, budget.source
    assert len(pack.sources) >= 4, [(s.source_type, s.path or s.title) for s in pack.sources]
    assert [s.ref for s in pack.sources] == ["S%d" % (i + 1) for i in range(len(pack.sources))]
    titles = [c.title for c in pack.cards]
    for card in q.get("expected_cards", []):
        assert card in titles, (qid, titles)
    if pack.cards:
        assert pack.text.startswith("Fiches (") and pack.sources[0].ref == pack.cards[0].ref == "S1"
    for snippet in q.get("context", []):
        assert snippet in pack.text, (qid, snippet, pack.text[:1500])
    if q.get("expected_endpoint"):
        assert _endpoint_in(pack.text, q["expected_endpoint"]), (qid, pack.text[:1500])
    # chaque source du contexte est citable : référence, type, chemin ou titre, canaux et raison
    for s in pack.sources:
        assert s.ref and s.source_type and (s.path or s.title) and s.channels and s.reason


@pytest.mark.parametrize("qid", [q["id"] for q in QUESTIONS if not q.get("smalltalk")])
def test_bench_context_grows_with_the_window(bench: SimpleNamespace, qid: str):
    q = question(qid)
    small, large = run_question(bench, q, 8192), run_question(bench, q, 16384)
    assert large.budget.context_tokens_max >= MIN_CONTEXT_TOKENS[16384] > small.budget.context_tokens_max
    assert large.budget.total <= 16384 and large.pack.tokens_used <= large.budget.context_tokens_max
    # à fenêtre double : jamais moins de sources, jamais plus d'écartées, mêmes fiches en tête
    assert len(large.pack.sources) >= len(small.pack.sources) and len(large.pack.truncated) <= len(small.pack.truncated)
    assert [c.title for c in large.pack.cards] == [c.title for c in small.pack.cards]
    assert large.pack.tokens_used >= small.pack.tokens_used


def test_bench_followup_keeps_the_previous_subject(bench: SimpleNamespace):
    q = question("followup_orders")
    run = run_question(bench, q, 8192)
    assert run.search_text == "et ses commandes ? " + FOLLOWUP_MARKER % "Customer"
    assert run.result.query.identifiers == ["Customer"] and run.result.query.previous_terms == ["customer"]
    assert run.result.query.keywords == ["commandes", "customer"] and run.result.query.intent == "factual"
    assert _matches(run.hits[0], "app/Models/Customer.php")
    # sans le marqueur, la question seule (« et ses commandes ? ») perd le sujet : ni modèle Customer ni fiche
    alone = bench.engine.search(q["text"], limit=30)
    assert not any(_matches(h, "app/Models/Customer.php") for h in alone.hits)
    pack = build_context(alone.hits, run.budget.context_tokens_max, "factual", db=bench.db)
    assert pack.cards == [] and len(pack.sources) < len(run.pack.sources)


async def test_bench_smalltalk_is_answered_without_search(bench: SimpleNamespace):
    q = question("smalltalk")
    assert is_smalltalk(q["text"]) and not is_smalltalk("quels champs a la table orders ?")
    conv = bench.db.create_conversation("banc")
    events = [ev async for ev in bench.orch.respond(conv, q["text"])]  # NullClient : aucun appel de modèle possible
    assert [e.type for e in events] == ["status", "sources", "delta", "done"] and events[1].data["items"] == []
    assert "décris-moi le modèle " in events[2].data["text"] and "comment appeler " in events[2].data["text"]  # exemples tirés de l'index


def test_bench_summary_table(bench: SimpleNamespace) -> None:
    """Tableau récapitulatif (visible avec ``pytest -s``) : top-5, attentes, sources / fiches / tokens par fenêtre."""
    rows = []
    for q in QUESTIONS:
        if q.get("smalltalk"):
            rows.append("%-20s petite conversation : %s" % (q["id"], is_smalltalk(q["text"])))
            continue
        cells = []
        run8 = run_question(bench, q, 8192)
        exp = q["expected_any"]
        top3 = any(_matches(h, e) for h in run8.hits[:3] for e in exp)
        top5 = any(_matches(h, e) for h in run8.hits[:5] for e in exp)
        for ctx in WINDOWS:
            run = run_question(bench, q, ctx)
            cells.append("%2d src / %d fiches / %4d tok ≤ %4d" % (len(run.pack.sources), len(run.pack.cards),
                                                                run.pack.tokens_used, run.budget.context_tokens_max))
        rows.append("%-20s top3=%-5s top5=%-5s | %s | %s" % (q["id"], top3, top5, cells[0], cells[1]))
        for i, h in enumerate(run8.hits[:5], 1):
            rows.append("    %d. [%s] %s" % (i, h.source_type, _label(h)))
        assert top5
    print("\n" + "\n".join(rows))  # capturé par pytest, affiché avec -s
