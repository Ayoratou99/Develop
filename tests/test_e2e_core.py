"""Test de bout en bout du cœur de Develop (sans HTTP) sur une copie Git de ``sample_project``.

Scénario, dans l'ordre du fichier (fixture de module partagée, comme le ferait un utilisateur) :

1. cinq dépôts (frontend, backend, services/notifier, docs, database) indexés par :class:`Indexer` ;
2. graphe : appels ↔ routes entre dépôts, même entité entre modèles, routes → gestionnaires → appels ;
3. recherche hybride (:class:`SearchEngine`) sans modèle d'embeddings ;
4. analyse d'impact depuis un symbole puis depuis Git (dépôt = sous-dossier d'un seul dépôt Git) ;
5. chat en mode ``search_only`` puis ``auto`` avec un faux LLM scripté (outil ``search`` natif) ;
6. second run incrémental, note ajoutée puis retrouvée, document déposé puis indexé (mentions, dérive).

``DEVELOP_HOME`` pointe vers un dossier temporaire ; la fixture n'est jamais modifiée (copie).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Sequence, Set, Tuple

import pytest

from develop.chat import TOOL_SPECS, ChatEvent, ChatOrchestrator
from develop.config import RepoConfig, WorkspaceConfig, documents_dir, load_workspace, save_workspace
from develop.git_tools import git_available
from develop.impact_analysis import ImpactAnalyzer
from develop.indexing import Indexer
from develop.knowledge_graph import document_mentions, is_stale
from develop.llm.base import ChatChunk, LLMClient, ModelInfo, NullClient, ToolCall
from develop.search import SearchEngine
from develop.storage import Database, DocumentRow, EndpointRow, NoteRow, RelationRow, SymbolRow, node_key
from develop.system.profile import MachineProfile

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"
_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
            "GIT_CONFIG_NOSYSTEM": "1", "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")}

pytestmark = pytest.mark.skipif(not git_available(), reason="git absent")


# --------------------------------------------------------------------------- aides

def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=dict(_GIT_ENV, HOME=str(root)))


def _relations(db: Database, kind: str) -> Dict[Tuple[str, str], RelationRow]:
    rows = db.query("SELECT * FROM relations WHERE kind = ? ORDER BY src, dst", (kind,))
    return {(r["src"], r["dst"]): RelationRow.from_row(r) for r in rows}


def _snapshot(db: Database) -> Set[Tuple[str, str, str, float]]:
    rows = db.query("SELECT src, dst, kind, confidence FROM relations")
    return {(r["src"], r["dst"], r["kind"], round(float(r["confidence"]), 4)) for r in rows}


def _undirected(rels: Dict[Tuple[str, str], RelationRow], a: str, b: str) -> RelationRow:
    rel = rels.get((a, b)) or rels.get((b, a))
    assert rel is not None, "relation absente entre %s et %s" % (a, b)
    return rel


def _endpoint(db: Database, side: str, method: str, path: str, repo_id: str) -> EndpointRow:
    for ep in db.list_endpoints(side=side):
        f = db.get_file_by_id(int(ep.file_id)) if ep.file_id is not None else None
        if f is not None and f.repo_id == repo_id and ep.method == method and ep.path == path:
            return ep
    raise AssertionError("endpoint introuvable : %s %s %s (%s)" % (side, method, path, repo_id))


def _symbol(db: Database, name: str, parent: str, path_suffix: str) -> SymbolRow:
    for s in db.search_symbols(name, limit=50):
        f = db.get_file_by_id(int(s.file_id)) if s.file_id is not None else None
        if s.name == name and (s.parent or "") == parent and f is not None and f.path.endswith(path_suffix):
            return s
    raise AssertionError("symbole introuvable : %s.%s dans %s" % (parent, name, path_suffix))


def _model(db: Database, kind: str, name: str, repo_id: str) -> int:
    for m in db.find_models(name=name):
        f = db.get_file_by_id(int(m.file_id))
        if m.kind == kind and f is not None and f.repo_id == repo_id:
            return int(m.id)
    raise AssertionError("modèle introuvable : %s %s (%s)" % (kind, name, repo_id))


def _profile() -> MachineProfile:
    return MachineProfile(os="macos", os_version="15", arch="arm64", cpu_model="test", cpu_cores_physical=8,
                          cpu_cores_logical=8, ram_total_gb=32.0, ram_available_gb=16.0, apple_silicon=True,
                          unified_memory=True)


class ScriptedLLM(LLMClient):
    """Faux moteur : ``rounds[i]`` est la liste de ``ChatChunk`` renvoyée au i-ième appel ``chat_stream``."""

    engine = "fake"

    def __init__(self, rounds: Sequence[Sequence[ChatChunk]]):
        self.rounds = [list(r) for r in rounds]
        self.calls: List[Dict[str, Any]] = []

    async def chat_stream(self, messages, model, *, temperature=0.2, num_ctx=None, tools=None, max_tokens=None):  # type: ignore[override]
        self.calls.append({"messages": list(messages), "model": model, "num_ctx": num_ctx,
                           "tools": list(tools) if tools else None})
        for chunk in self.rounds[min(len(self.calls) - 1, len(self.rounds) - 1)]:
            await asyncio.sleep(0)
            yield chunk

    async def embed(self, texts, model):  # type: ignore[override]
        return [[0.0] for _ in texts]


async def _collect(orch: ChatOrchestrator, conversation_id: int, text: str, **kwargs: Any) -> List[ChatEvent]:
    return [ev async for ev in orch.respond(conversation_id, text, **kwargs)]


# --------------------------------------------------------------------------- fixture partagée

@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    base = tmp_path_factory.mktemp("e2e")
    home = base / "home"
    home.mkdir()
    project = base / "sample_project"
    shutil.copytree(FIXTURE, project)
    _git(project, "init", "-q", "-b", "main")
    _git(project, "add", "-A")
    _git(project, "commit", "-q", "-m", "Import initial")
    previous = os.environ.get("DEVELOP_HOME")
    os.environ["DEVELOP_HOME"] = str(home)
    ws = save_workspace(WorkspaceConfig(id="boutique", name="Boutique", repos=[
        RepoConfig(id="frontend", name="frontend", path=str(project / "frontend"), kind="frontend"),
        RepoConfig(id="backend", name="backend", path=str(project / "backend"), kind="backend"),
        RepoConfig(id="notifier", name="notifier", path=str(project / "services" / "notifier"), kind="backend"),
        RepoConfig(id="docs", name="docs", path=str(project / "docs"), kind="docs"),
        RepoConfig(id="database", name="database", path=str(project / "database"), kind="database"),
    ]))
    assert load_workspace("boutique") is not None and (home / "workspaces" / "boutique" / "workspace.json").is_file()
    db = Database(home / "workspaces" / "boutique" / "memory.sqlite3")
    try:
        indexer = Indexer(db, ws)
        run = indexer.run()
        engine = SearchEngine(db)
        analyzer = ImpactAnalyzer(db, ws, search_engine=engine, stale_checker=indexer.stale_files)
        yield SimpleNamespace(home=home, project=project, ws=ws, db=db, indexer=indexer, run=run, engine=engine,
                              analyzer=analyzer)
    finally:
        db.close()
        if previous is None:
            os.environ.pop("DEVELOP_HOME", None)
        else:
            os.environ["DEVELOP_HOME"] = previous


# --------------------------------------------------------------------------- 1. indexation

def test_first_run_indexes_five_repos(env: SimpleNamespace):
    run, db = env.run, env.db
    assert run.status == "ok" and run.error is None, run.stats.get("errors")
    stats = run.stats
    assert stats["errors"] == [] and stats["files_skipped"].get("binary", 0) >= 1  # logo.png détecté, jamais lu
    assert stats["files_indexed"] >= 25 and stats["files_indexed"] == stats["files_scanned"]
    assert stats["files_unchanged"] == 0 and stats["files_deleted"] == 0
    assert stats["symbols"] > 0 and stats["chunks"] > 0 and stats["models"] >= 6
    servers, clients = db.list_endpoints(side="server"), db.list_endpoints(side="client")
    assert len(servers) >= 8 and len(clients) >= 5 and stats["endpoints"] == len(servers) + len(clients)
    assert {r.id for r in db.list_repos()} == {"frontend", "backend", "notifier", "docs", "database"}
    assert all(stats["repos"][rid]["status"] == "ok" for rid in ("frontend", "backend", "notifier", "docs", "database"))
    # relations brutes + dérivées comptées dans le run, graphe construit sans erreur
    linker = stats["linker"]
    assert isinstance(linker, dict) and linker["errors"] == [] and linker["total"] > 0
    assert linker["base_prefixes"]["frontend"] == ["/api"] and linker["base_prefixes"]["backend"] == ["/api"]
    assert stats["relations"] > linker["total"] > 0
    totals = db.stats()
    assert totals["files"] == stats["files_scanned"] and totals["chunks"] == stats["chunks"]
    assert totals["relations"] >= stats["relations"]
    # état Git mémorisé pour chaque dépôt (un seul dépôt Git englobant)
    for rid in ("frontend", "backend", "docs"):
        repo = db.get_repo(rid)
        assert repo is not None and repo.head_sha and repo.last_indexed_at is not None
    # le gabarit .env.example est indexé, le dossier node_modules ignoré
    assert db.get_file("frontend", ".env.example") is not None
    assert db.get_file("frontend", "node_modules/leftpad/index.js") is None


# --------------------------------------------------------------------------- 2. graphe inter-dépôts

def test_calls_endpoint_links_frontend_and_services(env: SimpleNamespace):
    db = env.db
    calls = _relations(db, "calls_endpoint")
    # confirmOrder (client.ts, axios baseURL /api) → POST /orders/{order}/confirm (routes/api.php)
    client = _endpoint(db, "client", "POST", "/orders/${id}/confirm", "frontend")
    server = _endpoint(db, "server", "POST", "/orders/{order}/confirm", "backend")
    assert client.handler == "confirmOrder" and server.handler == "OrderController@confirm"
    rel = calls[(node_key("endpoint", client.id), node_key("endpoint", server.id))]
    assert rel.confidence == 1.0 and rel.meta["method"] == "POST"
    # OrderService (Http::post) → Express POST /notifications (dépôt notifier)
    client = _endpoint(db, "client", "POST", "http://localhost:3001/notifications", "backend")
    server = _endpoint(db, "server", "POST", "/notifications", "notifier")
    assert "OrderService" in (client.handler or "")
    assert calls[(node_key("endpoint", client.id), node_key("endpoint", server.id))].confidence == 1.0
    assert env.run.stats["linker"]["endpoints_unmatched_client"] == 0


def test_same_entity_links_order_across_repos(env: SimpleNamespace):
    db = env.db
    nodes = [node_key("model", _model(db, "eloquent", "Order", "backend")),
             node_key("model", _model(db, "migration", "orders", "backend")),
             node_key("model", _model(db, "sql_table", "orders", "database")),
             node_key("model", _model(db, "ts_interface", "Order", "frontend"))]
    same = _relations(db, "same_entity")
    for i in range(4):
        for j in range(i + 1, 4):
            assert _undirected(same, nodes[i], nodes[j]).confidence >= 0.7
    assert _undirected(same, nodes[0], nodes[1]).confidence == 0.9  # même table déclarée
    # les trois modèles « table » définissent la table orders
    defines = _relations(db, "defines_table")
    assert all((n, "table:orders") in defines for n in nodes[:3])


def test_handles_and_calls_cross_files(env: SimpleNamespace):
    """Route Laravel → méthode du contrôleur (autre fichier) → méthode du service ; composant → fonction d'API."""
    db = env.db
    route = _endpoint(db, "server", "POST", "/orders/{order}/confirm", "backend")
    ctl_confirm = _symbol(db, "confirm", "OrderController", "OrderController.php")
    svc_confirm = _symbol(db, "confirm", "OrderService", "OrderService.php")
    handles = _relations(db, "handles")
    rel = handles[(node_key("endpoint", route.id), node_key("symbol", ctl_confirm.id))]
    assert rel.confidence >= 0.9 and rel.meta["handler"] == "OrderController@confirm"
    calls = _relations(db, "calls")
    rel = calls[(node_key("symbol", ctl_confirm.id), node_key("symbol", svc_confirm.id))]
    assert rel.confidence >= 0.8 and rel.meta["name"] == "confirm"
    order_list = _symbol(db, "OrderList", "", "OrderList.tsx")
    confirm_order = _symbol(db, "confirmOrder", "", "client.ts")
    assert calls[(node_key("symbol", order_list.id), node_key("symbol", confirm_order.id))].confidence >= 0.8
    # aucun appel fantaisiste : chaque cible est un symbole appelable existant
    for (_src, dst), r in calls.items():
        assert db.get_symbol(int(dst.split(":")[1])) is not None and 0.5 <= r.confidence <= 0.8


# --------------------------------------------------------------------------- 3. recherche hybride

def test_search_fields_of_table_orders(env: SimpleNamespace):
    r = env.engine.search("quels champs a la table orders")
    assert r.query.intent == "fields" and r.query.tables == ["orders"]
    assert r.semantic_available is False and r.warnings == []  # pas de modèle d'embeddings : lexical + graphe
    top3 = r.hits[:3]
    assert any((h.source_type == "model" and h.extra.get("kind") == "migration")
               or (h.path or "").endswith("dictionnaire-donnees.md") for h in top3), [(h.source_type, h.title) for h in top3]
    assert any((h.path or "").endswith("dictionnaire-donnees.md") for h in r.hits)
    assert r.hits == sorted(r.hits, key=lambda h: -h.score)
    # forme citable (docs/API.md § 4)
    d = r.hits[0].to_dict()
    assert {"chunk_id", "source_type", "source_id", "title", "path", "line_start", "line_end", "snippet", "score",
            "channels", "reason"} <= set(d)


def test_search_exact_symbol_first(env: SimpleNamespace):
    r = env.engine.search("OrderService")
    top = r.hits[0]
    assert top.source_type == "symbol" and top.title == "OrderService" and "identifier" in top.channels
    assert top.path == "app/Services/OrderService.php" and top.repo_id == "backend"
    assert top.text.startswith("class OrderService")  # lu sur disque, lecture seule


def test_search_who_calls_confirm(env: SimpleNamespace):
    r = env.engine.search("qui appelle confirm")
    assert r.query.intent == "who_calls"
    titles = [h.title for h in r.hits[:3]]
    assert "OrderService::confirm" in titles and "OrderController::confirm" in titles
    assert any("graph" in h.channels for h in r.hits[:3])  # le graphe relie les deux confirm (appels)
    paths = {h.path for h in r.hits}
    assert "routes/api.php" in paths or "src/api/client.ts" in paths


# --------------------------------------------------------------------------- 4. analyse d'impact

def test_impact_from_symbol_order_service_confirm(env: SimpleNamespace):
    db, analyzer = env.db, env.analyzer
    confirm = _symbol(db, "confirm", "OrderService", "OrderService.php")
    report = analyzer.from_symbol(confirm.id, max_hops=3)
    assert report.source["name"] == "OrderService.confirm" and report.warnings == []
    assert report.stale_index is False and report.stats["seeds"] == 1
    assert [e.name for e in report.endpoints] == ["POST /api/orders/:param/confirm"]
    assert report.endpoints[0].reason == "route gérée par OrderController.confirm"
    assert [c.extra.get("handler") for c in report.frontend_calls] == ["confirmOrder"]
    assert report.frontend_calls[0].path == "src/api/client.ts" and report.frontend_calls[0].repo_id == "frontend"
    docs = {d.name for d in report.documents}
    assert "dictionnaire-donnees.md" in docs or "specification-commandes.md" in docs
    assert all(0 < i.confidence <= 1 and i.hops >= 1 for i in report.affected)
    assert len({i.node for i in report.affected}) == len(report.affected)
    # profondeur par défaut : la route est atteinte (appel → gestionnaire), pas encore l'appel frontend
    short = analyzer.from_symbol(confirm.id)
    assert [e.name for e in short.endpoints] == ["POST /api/orders/:param/confirm"] and short.frontend_calls == []


def test_impact_from_git_after_editing_order_service(env: SimpleNamespace):
    """Le dépôt « backend » est un sous-dossier du dépôt Git : les chemins sont ramenés au dépôt configuré."""
    analyzer = env.analyzer
    assert analyzer.from_git("backend").changed == []  # arbre propre
    service = env.project / "backend" / "app" / "Services" / "OrderService.php"
    original = service.read_text(encoding="utf-8")
    service.write_text(original.replace("$order->status = 'confirmed';", "$order->status = 'validated';"),
                       encoding="utf-8")
    try:
        report = analyzer.from_git("backend")
        (entry,) = report.changed
        assert entry["path"] == "app/Services/OrderService.php" and entry["status"] == "modified"
        assert entry["indexed"] is True and entry["whole_file"] is False and entry["ranges"] == [[15, 15]]
        assert "OrderService.confirm" in entry["symbols_changed"]
        assert "OrderService.totalFor" not in entry["symbols_changed"]  # seule confirm() est touchée
        assert entry["seeds"] == len(entry["symbols_changed"]) == report.stats["seeds"]
        assert [e.name for e in report.endpoints] == ["POST /api/orders/:param/confirm"]
        assert report.stale_index is True and report.stale_files == ["app/Services/OrderService.php"]
        assert any("Réindexez" in w for w in report.warnings)
        # les autres dépôts du même dépôt Git ne voient pas ce changement
        assert analyzer.from_git("frontend").changed == []
    finally:
        service.write_text(original, encoding="utf-8")
    assert analyzer.from_git("backend").changed == []


def test_impact_from_git_ignores_touch_without_content_change(env: SimpleNamespace):
    """Un ``touch`` (mtime seul) sur un arbre propre ne rend pas l'index obsolète : pas d'avertissement."""
    service = env.project / "backend" / "app" / "Services" / "OrderService.php"
    st = service.stat()
    os.utime(service, (st.st_atime + 60, st.st_mtime + 60))
    assert env.indexer.stale_files("backend") == []
    report = env.analyzer.from_git("backend")
    assert report.changed == [] and report.stale_index is False and report.stale_files == []
    assert not any("Index obsolète" in w for w in report.warnings)


# --------------------------------------------------------------------------- 5. chat

async def test_chat_search_only_without_engine(env: SimpleNamespace):
    db = env.db
    orch = ChatOrchestrator(db, env.ws, NullClient(), search_engine=env.engine, impact_analyzer=env.analyzer,
                            profile=_profile(), model_info=None, today="2026-10-20")
    conv = db.create_conversation("e2e")
    events = await _collect(orch, conv, "Quels champs a la table orders ?", mode="search_only")
    assert [e.type for e in events] == ["status", "sources", "done"]
    sources = events[1].data["items"]
    assert sources and [s["ref"] for s in sources] == ["S%d" % (i + 1) for i in range(len(sources))]
    assert any((s.get("path") or "").endswith("dictionnaire-donnees.md") or s["source_type"] == "model" for s in sources)
    assert all({"ref", "source_type", "score", "channels", "reason"} <= set(s) for s in sources)
    done = events[-1].data
    assert done["message_id"] and done["usage"] == {} and done["budget"]["tool_rounds"] == 0
    assert done["budget"]["num_ctx"] > 0 and done["budget"]["context_tokens_used"] > 0
    msgs = db.get_conversation(conv)["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[-1]["content"].startswith("Sources trouvées")
    assert [s["ref"] for s in msgs[-1]["sources"]] == [s["ref"] for s in sources]
    # chaque événement est sérialisable en SSE
    for ev in events:
        assert ev.to_sse().startswith("event: %s\ndata: " % ev.type) and ev.to_sse().endswith("\n\n")


async def test_chat_auto_with_scripted_llm_and_native_search_tool(env: SimpleNamespace):
    db = env.db
    llm = ScriptedLLM([
        [ChatChunk(tool_calls=[ToolCall(name="search", arguments={"query": "statut confirmed commande"})]),
         ChatChunk(done=True, usage={"prompt_tokens": 50, "completion_tokens": 5})],
        [ChatChunk(text="Seule une commande brouillon peut être confirmée "), ChatChunk(text="[S1]."),
         ChatChunk(done=True, usage={"prompt_tokens": 400, "completion_tokens": 30})],
    ])
    info = ModelInfo(name="fake:7b", context_length=32768, capabilities=["completion", "tools"])
    orch = ChatOrchestrator(db, env.ws, llm, search_engine=env.engine, impact_analyzer=env.analyzer,
                            profile=_profile(), model_info=info, today="2026-10-20")
    conv = db.create_conversation("e2e auto")
    events = await _collect(orch, conv, "Quand une commande peut-elle être confirmée ?")
    assert [e.type for e in events] == ["status", "sources", "tool_call", "tool_result", "delta", "delta", "done"]
    assert events[1].data["items"] and events[1].data["items"][0]["ref"] == "S1"
    assert events[2].data["name"] == "search" and events[2].data["arguments"] == {"query": "statut confirmed commande"}
    result = events[3].data
    assert result["name"] == "search" and "error" not in result and result["summary"]
    assert all(s["ref"].startswith("S") for s in result["added_sources"])
    assert "".join(e.data["text"] for e in events if e.type == "delta") == "Seule une commande brouillon peut être confirmée [S1]."
    done = events[-1].data
    assert done["invalid_citations"] == [] and done["cancelled"] is False
    assert done["usage"] == {"prompt_tokens": 450, "completion_tokens": 35} and done["budget"]["tool_rounds"] == 1
    assert len(llm.calls) == 2 and llm.calls[0]["tools"] == TOOL_SPECS and llm.calls[0]["model"] == ""
    roles = [m.role for m in llm.calls[1]["messages"]]
    assert roles[-3:] == ["user", "assistant", "tool"] and llm.calls[1]["messages"][-1].name == "search"
    msgs = db.get_conversation(conv)["messages"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"].endswith("[S1].") and msgs[-1]["sources"]


# --------------------------------------------------------------------------- 6. incrémental, notes, documents

def test_second_run_is_incremental_and_graph_is_stable(env: SimpleNamespace):
    db = env.db
    before = _snapshot(db)
    second = Indexer(db, env.ws).run()
    assert second.status == "ok" and second.stats["errors"] == []
    assert second.stats["files_scanned"] == env.run.stats["files_scanned"]
    assert second.stats["files_unchanged"] == second.stats["files_scanned"]
    assert second.stats["files_indexed"] == 0 and second.stats["chunks"] == 0 and second.stats["symbols"] == 0
    assert second.stats["linker"]["total"] == env.run.stats["linker"]["total"]
    assert _snapshot(db) == before
    assert db.stats()["chunks"] == env.run.stats["chunks"]


def test_note_is_indexed_searchable_and_linked(env: SimpleNamespace):
    db, engine = env.db, env.engine
    note_id = db.add_note(NoteRow(title="Décision paiement", note_date="2026-10-01", tags=["décision", "commandes"],
                                  body="On garde `OrderService::confirm` idempotent. Pas de statut paid avant 2027 ; "
                                       "le paiement sera porté par `paid_at` sur la table `orders`."))
    run = Indexer(db, env.ws).run()
    assert run.status == "ok" and run.stats["notes"] == 1 and run.stats["files_indexed"] == 0
    assert len(db.chunks_for("note", note_id)) == 1
    r = engine.search("décision paiement idempotent")
    assert r.hits and r.hits[0].source_type == "note" and r.hits[0].title == "Décision paiement"
    assert r.hits[0].extra.get("note_id") == note_id or r.hits[0].source_id == note_id
    r = engine.search("notes du 2026-10-01")
    assert r.hits and r.hits[0].source_type == "note" and "temporal" in r.hits[0].channels
    # le graphe relie la note au symbole et à la table qu'elle cite
    confirm = _symbol(db, "confirm", "OrderService", "OrderService.php")
    documented = _relations(db, "documented_by")
    assert (node_key("symbol", confirm.id), node_key("note", note_id)) in documented
    assert ("table:orders", node_key("note", note_id)) in documented
    report = env.analyzer.from_symbol(confirm.id)
    assert [n.name for n in report.notes] == ["Décision paiement"]


def test_uploaded_document_is_indexed_with_mentions_and_drift(env: SimpleNamespace):
    db, engine = env.db, env.engine
    dest = documents_dir(env.ws.id)
    dest.mkdir(parents=True, exist_ok=True)
    stored = dest / "abcd1234-dictionnaire-complement.md"
    stored.write_text(
        "# Dictionnaire complémentaire\n\n"
        "## Table `orders`\n\n"
        "| Colonne | Type | Description |\n|---|---|---|\n"
        "| status | enum | draft, confirmed, shipped, cancelled |\n"
        "| confirmed_at | timestamp | renseignée par OrderService::confirm |\n\n"
        "La route POST /api/orders/{id}/confirm est appelée par `confirmOrder`. "
        "L'ancienne classe `OrderArchiver` n'existe plus.\n",
        encoding="utf-8")
    doc_id = db.add_document(DocumentRow(title="", source_path="dictionnaire-complement.md", stored_path=str(stored),
                                         kind="upload", format="md", size=stored.stat().st_size, sha1="abcd1234"))
    run = Indexer(db, env.ws).run()
    assert run.status == "ok" and run.stats["documents"] == 1 and run.stats["files_indexed"] == 0
    doc = db.get_document(doc_id)
    assert doc is not None and doc.title == "Dictionnaire complémentaire" and doc.meta["chunk_count"] >= 1
    assert doc.meta["looks_like_data_dictionary"] is True and doc.meta["tables"]
    assert len(db.chunks_for("document", doc_id)) == doc.meta["chunk_count"]
    # mentions : la table orders existe, OrderArchiver est une dérive documentation / code
    mentions = {m["name"]: m for m in document_mentions(db, "document", doc_id)}
    assert mentions["orders"]["exists"] is True and "table:orders" in mentions["orders"]["nodes"]
    assert mentions["orders"]["kind"] in ("table", "model")
    assert mentions["confirmOrder"]["exists"] is True and mentions["confirmOrder"]["kind"] == "symbol"
    assert mentions["OrderArchiver"]["exists"] is False and mentions["OrderArchiver"]["code_like"] is True
    assert is_stale(db, "document", doc_id) is True
    # relations dérivées vers le document et recherche
    documented = _relations(db, "documented_by")
    assert ("table:orders", node_key("document", doc_id)) in documented
    confirm_order = _symbol(db, "confirmOrder", "", "client.ts")
    assert (node_key("symbol", confirm_order.id), node_key("document", doc_id)) in documented
    r = engine.search("quels champs a la table orders")
    assert any(h.source_type == "document" and h.source_id == doc_id for h in r.hits[:5])
    # le dictionnaire déposé fait partie des documents affectés par OrderService::confirm
    confirm = _symbol(db, "confirm", "OrderService", "OrderService.php")
    report = env.analyzer.from_symbol(confirm.id, max_hops=3)
    assert "Dictionnaire complémentaire" in {d.name for d in report.documents}
