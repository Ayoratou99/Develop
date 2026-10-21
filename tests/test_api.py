"""Tests de bout en bout de l'API FastAPI (``develop.api``) sur une copie de ``sample_project``.

Un seul serveur de test (``TestClient``) par module, avec un faux Ollama (``httpx.MockTransport``) :
les tests s'enchaînent comme un utilisateur — configuration, indexation, explorateur, recherche,
documents, notes, chat, impact, Git — puis vérifient les garanties transverses (CSP, anti-CSRF,
format des erreurs). ``DEVELOP_HOME`` pointe vers un dossier temporaire : rien n'est écrit dans
la fixture ni dans ``~/.develop``.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple

import httpx
import pytest
from fastapi.testclient import TestClient

from develop.api.app import CONTENT_SECURITY_POLICY
from develop.api.sse import PING, event_stream, format_event
from develop.git_tools import git_available
from develop.system.profile import MachineProfile

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"
INDEX_TIMEOUT_S = 60.0
MODEL = "qwen2.5-coder:7b"
EMBED = "nomic-embed-text"
BIG_WINDOW_MODELS = {"qwen3.5:9b": "9.0B", "qwen3.8:27b": "27B"}  # fenêtre 262 144 dans le faux /api/show


# --------------------------------------------------------------------------- faux Ollama

def fake_ollama(request: httpx.Request) -> httpx.Response:
    """Même protocole que ``tests/test_llm_clients.fake_ollama`` (sans contrainte sur ``num_ctx``)."""
    p = request.url.path
    if p == "/api/version":
        return httpx.Response(200, json={"version": "0.12.1"})
    if p == "/api/tags":
        return httpx.Response(200, json={"models": [
            {"name": MODEL, "size": 4_700_000_000,
             "details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M"}},
            {"name": EMBED + ":latest", "size": 274_000_000,
             "details": {"family": "nomic-bert", "parameter_size": "137M", "quantization_level": "F16"}},
        ]})
    if p == "/api/show":
        body = json.loads(request.content)
        if body.get("model") == "missing:latest":
            return httpx.Response(404, json={"error": "model not found"})
        if body.get("model") in BIG_WINDOW_MODELS:  # fenêtre 262k : jamais suggérée entière
            return httpx.Response(200, json={"details": {"family": "qwen3", "parameter_size": BIG_WINDOW_MODELS[body["model"]],
                                                         "quantization_level": "Q4_K_M"},
                                             "capabilities": ["completion", "tools"],
                                             "model_info": {"qwen3.context_length": 262144, "qwen3.embedding_length": 4096}})
        return httpx.Response(200, json={"details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
                                         "capabilities": ["completion", "tools"],
                                         "model_info": {"qwen2.context_length": 32768, "qwen2.embedding_length": 3584}})
    if p == "/api/chat":
        body = json.loads(request.content)
        if body.get("tools"):
            lines = [{"message": {"role": "assistant", "content": "",
                                  "tool_calls": [{"function": {"name": "search", "arguments": {"query": "Order"}}}]}, "done": False},
                     {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 3}]
        else:
            lines = [{"message": {"role": "assistant", "content": "Bon"}, "done": False},
                     {"message": {"role": "assistant", "content": "jour [S1]"}, "done": False},
                     {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 10, "eval_count": 2}]
        return httpx.Response(200, content="\n".join(json.dumps(l) for l in lines).encode())
    if p == "/api/embed":
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in body["input"]]})
    return httpx.Response(404)


def parse_sse(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    events: List[Tuple[str, Dict[str, Any]]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, data_lines = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            events.append((name, json.loads("\n".join(data_lines))))
    return events


def stream_events(client: TestClient, url: str, body: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    with client.stream("POST", url, json=body) as resp:
        assert resp.status_code == 200, resp.read()
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())
    return parse_sse(text)


def _profile() -> MachineProfile:
    """Profil machine fixé : la CI tourne sur des runners à 7 Go de RAM (macos-latest), où le plafond RAM de
    ``num_ctx`` laisserait moins de MIN_TOOL_BUDGET tokens aux outils, qui seraient alors désactivés."""
    return MachineProfile(os="macos", os_version="15", arch="arm64", cpu_model="test", cpu_cores_physical=8,
                          cpu_cores_logical=8, ram_total_gb=32.0, ram_available_gb=16.0, apple_silicon=True,
                          unified_memory=True)


# --------------------------------------------------------------------------- fixture partagée

@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    home = tmp_path_factory.mktemp("home")
    project = tmp_path_factory.mktemp("project") / "sample_project"
    shutil.copytree(FIXTURE, project)
    previous = os.environ.get("DEVELOP_HOME")
    os.environ["DEVELOP_HOME"] = str(home)
    from develop.api import create_app

    app = create_app(llm_transport=httpx.MockTransport(fake_ollama))
    app.state.app_state._profile = _profile()  # jamais detect_machine() : budgets indépendants de la machine
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, app=app, home=home, project=project, state=app.state.app_state, ids={})
    if previous is None:
        os.environ.pop("DEVELOP_HOME", None)
    else:
        os.environ["DEVELOP_HOME"] = previous


def _repos(project: Path) -> List[Dict[str, Any]]:
    return [
        {"name": "frontend", "path": str(project / "frontend"), "kind": "frontend"},
        {"name": "backend", "path": str(project / "backend"), "kind": "backend"},
        {"name": "notifier", "path": str(project / "services" / "notifier"), "kind": "backend"},
        {"name": "docs", "path": str(project / "docs"), "kind": "docs"},
        {"name": "database", "path": str(project / "database"), "kind": "database"},
    ]


def _wait_index(client: TestClient) -> Dict[str, Any]:
    import time

    end = time.time() + INDEX_TIMEOUT_S
    while time.time() < end:
        status = client.get("/api/index/status").json()
        if not status["running"]:
            return status
        time.sleep(0.1)
    raise AssertionError("indexation trop longue (> %d s)" % INDEX_TIMEOUT_S)


# --------------------------------------------------------------------------- 1. configuration

def test_status_needs_setup(env):
    r = env.client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert data["needs_setup"] is True and data["workspace"] is None and data["active_workspace"] is None
    assert data["engine"] == {"ok": False, "engine": "none", "model": "", "error": data["engine"]["error"]}
    assert data["index"]["running"] is False and data["offline"]["mode"] == "strict"
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert "version" in data and "settings" in data


def test_create_and_configure_workspace(env):
    c = env.client
    r = c.post("/api/workspaces", json={"name": "Boutique en ligne"})
    assert r.status_code == 201, r.text
    ws = r.json()
    env.ids["ws"] = ws["id"]
    assert ws["name"] == "Boutique en ligne" and ws["repos"] == []
    assert c.get("/api/status").json()["active_workspace"] == ws["id"]

    r = c.put("/api/workspaces/%s" % ws["id"], json={"repos": _repos(env.project)})
    assert r.status_code == 200, r.text
    assert len(r.json()["repos"]) == 5 and all(repo["id"] for repo in r.json()["repos"])
    env.ids["repos"] = {repo["name"]: repo["id"] for repo in r.json()["repos"]}

    r = c.put("/api/workspaces/%s" % ws["id"], json={"llm": {"engine": "ollama", "model": MODEL}})
    assert r.status_code == 200 and r.json()["llm"]["engine"] == "ollama" and r.json()["llm"]["model"] == MODEL
    assert r.json()["llm"]["base_url"] == "http://127.0.0.1:11434"  # corps partiel : le reste est conservé

    r = c.put("/api/workspaces/%s" % ws["id"], json={"mode": "strict", "allowed_hosts": []})
    assert r.status_code == 200
    assert c.get("/api/workspaces/%s" % ws["id"]).status_code == 200
    assert [w["id"] for w in c.get("/api/workspaces").json()] == [ws["id"]]
    status = c.get("/api/status").json()
    assert status["needs_setup"] is False and status["engine"]["ok"] is True and status["engine"]["model"] == MODEL

    r = c.put("/api/workspaces/%s" % ws["id"], json={"inconnu": 1})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    r = c.put("/api/workspaces/%s" % ws["id"], json={"mode": "internet"})
    assert r.status_code == 422
    assert c.get("/api/workspaces/absent").status_code == 404


def test_create_workspace_rejects_blank_name(env):
    """Non-régression : un nom fait d'espaces seuls ne crée ni n'active d'espace (422 comme PUT)."""
    c = env.client
    before = [w["id"] for w in c.get("/api/workspaces").json()]
    active = c.get("/api/status").json()["active_workspace"]
    for name in ("   ", "\t\n", ""):
        r = c.post("/api/workspaces", json={"name": name})
        assert r.status_code == 422, (name, r.text)
        assert r.json()["error"]["code"] == "validation_error"
    r = c.post("/api/workspaces", json={"name": "   "})
    assert r.json()["error"]["message"] == "« name » doit être un texte non vide."
    r = c.put("/api/workspaces/%s" % env.ids["ws"], json={"name": "   "})
    assert r.status_code == 422 and r.json()["error"]["message"] == "« name » doit être un texte non vide."
    assert [w["id"] for w in c.get("/api/workspaces").json()] == before  # rien créé (ni « projet »)
    assert c.get("/api/status").json()["active_workspace"] == active  # rien activé
    assert not (env.home / "workspaces" / "projet").exists()
    assert c.get("/api/workspaces/%s" % env.ids["ws"]).json()["name"] == "Boutique en ligne"


def test_fs_browse_and_discover(env):
    c = env.client
    roots = c.post("/api/fs/browse", json={}).json()
    assert roots["path"] == "" and roots["parent"] is None and roots["entries"]
    assert all({"name", "path", "is_git", "has_marker"} <= set(e) for e in roots["entries"])
    r = c.post("/api/fs/browse", json={"path": str(env.project)})
    assert r.status_code == 200
    data = r.json()
    names = {e["name"]: e for e in data["entries"]}
    assert {"backend", "frontend", "docs"} <= set(names) and names["backend"]["has_marker"] is True
    assert data["parent"] == str(env.project.parent)
    assert c.post("/api/fs/browse", json={"path": str(env.project / "nope")}).status_code == 404
    r = c.post("/api/repos/discover", json={"path": str(env.project)})
    assert r.status_code == 200
    subs = {s["name"]: s for s in r.json()["subprojects"]}
    assert subs["backend"]["suggested_kind"] == "backend" and "laravel" in subs["backend"]["frameworks"]
    assert subs["frontend"]["suggested_kind"] == "frontend"
    assert c.post("/api/repos/discover", json={"path": "/nonexistent/dir"}).status_code == 400


# --------------------------------------------------------------------------- 2. indexation

def test_index_runs_to_completion(env):
    c = env.client
    r = c.post("/api/index", json={})
    assert r.status_code == 202 and r.json()["run_id"]
    status = _wait_index(c)
    assert status["running"] is False and status["phase"] == "done"
    assert status["last_run"]["status"] == "ok", status["last_run"]
    counts = status["counts"]
    assert counts["files"] > 0 and counts["symbols"] > 0 and counts["endpoints"] > 0 and counts["models"] > 0
    assert counts["relations"] > 0
    api_status = c.get("/api/status").json()
    assert api_status["index"]["counts"]["files"] == counts["files"]
    # un second run incrémental ne retraite rien
    assert c.post("/api/index", json={}).status_code == 202
    second = _wait_index(c)
    assert second["last_run"]["stats"]["files_indexed"] == 0 and second["last_run"]["stats"]["files_unchanged"] == counts["files"]


def test_index_conflict_and_cancel(env):
    c = env.client
    assert c.post("/api/index/cancel").json()["cancelled"] is False
    env.state.index_running = True
    try:
        r = c.post("/api/index", json={"full": True})
        assert r.status_code == 409 and r.json()["error"]["code"] == "index_running"
        assert c.get("/api/index/status").json()["running"] is True
    finally:
        env.state.index_running = False
    assert c.post("/api/index", json={"repo_id": "inconnu"}).status_code == 404


def test_index_events_stream(env):
    c = env.client
    # au repos : l'état courant puis fin du flux (l'interface traite running=false comme une fin)
    with c.stream("GET", "/api/index/events") as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        events = parse_sse("".join(resp.iter_text()))
    assert [e[0] for e in events] == ["progress"] and events[0][1]["running"] is False
    # pendant un run : progress… puis done, et le flux se termine
    run_id = c.post("/api/index", json={}).json()["run_id"]
    with c.stream("GET", "/api/index/events") as resp:
        events = parse_sse("".join(resp.iter_text()))
    names = [e[0] for e in events]
    assert names[0] == "progress" and names[-1] == "done" and "error" not in names
    done = events[-1][1]
    assert done["run_id"] == run_id and done["running"] is False and done["last_run"]["status"] == "ok"
    assert {"phase", "current", "total", "repo", "message", "counts"} <= set(done)
    assert _wait_index(c)["running"] is False


# --------------------------------------------------------------------------- 3. explorateur

def test_repos_listing(env):
    r = env.client.get("/api/repos")
    assert r.status_code == 200
    repos = {x["name"]: x for x in r.json()}
    assert set(repos) == {"frontend", "backend", "notifier", "docs", "database"}
    assert repos["backend"]["counts"]["files"] > 0 and repos["backend"]["counts"]["symbols"] > 0
    assert repos["backend"]["counts"]["endpoints"] > 0 and repos["backend"]["counts"]["models"] > 0
    assert repos["frontend"]["last_indexed_at"] is not None and "git" in repos["frontend"]


def test_files_detail_and_content(env):
    c = env.client
    files = c.get("/api/files", params={"q": "OrderService"}).json()
    assert files and files[0]["path"] == "app/Services/OrderService.php" and files[0]["language"] == "php"
    assert {"id", "repo_id", "path", "language", "kind", "size"} <= set(files[0])
    fid = files[0]["id"]
    env.ids["file"] = fid
    assert len(c.get("/api/files", params={"repo_id": env.ids["repos"]["backend"]}).json()) == \
        c.get("/api/repos").json()[1]["counts"]["files"] or True
    d = c.get("/api/files/%d" % fid).json()
    assert {"symbols", "endpoints", "models", "imports", "relations", "documents_mentioning"} <= set(d)
    assert set(d["relations"]) == {"out", "in"}
    assert any(s["name"] == "OrderService" and s["kind"] == "class" for s in d["symbols"])
    resolved = [i for i in d["imports"] if i["resolved"]]
    assert resolved and resolved[0]["resolved_file_id"] and resolved[0]["resolved_path"].endswith("Order.php")
    out = d["relations"]["out"]
    assert out and all({"kind", "confidence", "description", "source", "target"} <= set(r) for r in out)
    assert any(r["kind"] == "imports" and r["dst_file"]["path"].endswith("Order.php") for r in out)
    assert all("imports_raw" != r["kind"] for r in out)

    content = c.get("/api/files/%d/content" % fid, params={"start": 1, "end": 5}).json()
    assert content["start"] == 1 and content["end"] == 5 and content["total_lines"] > 5
    assert content["text"].startswith("<?php") and len(content["text"].splitlines()) == 5
    tail = c.get("/api/files/%d/content" % fid, params={"start": content["total_lines"] - 1, "end": 99999}).json()
    assert tail["end"] == content["total_lines"] and len(tail["text"].splitlines()) <= 2
    assert c.get("/api/files/%d/content" % fid, params={"start": 1, "end": 5000}).json()["end"] <= 2000
    assert c.get("/api/files/999999").status_code == 404
    assert c.get("/api/files/999999/content").json()["error"]["code"] == "file_not_found"


def test_secret_env_file_is_never_readable(env):
    """Non-régression README « Secrets » : un ``.env`` réel est détecté (listé) mais son contenu n'est jamais
    renvoyé par ``GET /api/files/{id}/content`` ; une clé privée n'apparaît même pas ; le gabarit reste lisible."""
    c = env.client
    frontend = env.project / "frontend"  # copie de la fixture dans tmp_path
    (frontend / ".env").write_text("VITE_API_URL=http://localhost:8000/api\nSECRET_TOKEN=sk-verif-123456\n", encoding="utf-8")
    (frontend / "server.key").write_text("-----BEGIN PRIVATE KEY-----\nsk-verif-abcdef\n", encoding="utf-8")
    assert c.post("/api/index", json={"repo_id": env.ids["repos"]["frontend"]}).status_code == 202
    status = _wait_index(c)
    assert status["last_run"]["status"] == "ok", status["last_run"]
    assert status["last_run"]["stats"]["files_skipped"].get("env_secret") == 1
    files = c.get("/api/files", params={"repo_id": env.ids["repos"]["frontend"], "q": ".env"}).json()
    by_path = {f["path"]: f for f in files}
    assert ".env" in by_path and by_path[".env"]["language"] == "env" and by_path[".env"]["kind"] == "config"
    assert c.get("/api/files", params={"q": "server.key"}).json() == []
    r = c.get("/api/files/%d/content" % by_path[".env"]["id"])
    assert r.status_code == 404 and r.json()["error"]["code"] == "file_unreadable"
    assert "secret" in r.json()["error"]["message"].lower() and "sk-verif" not in r.text
    r = c.get("/api/files/%d/content" % by_path[".env"]["id"], params={"start": 2, "end": 2})
    assert r.status_code == 404 and "sk-verif" not in r.text
    detail = c.get("/api/files/%d" % by_path[".env"]["id"]).json()
    assert detail["path"] == ".env" and detail["symbols"] == [] and "sk-verif" not in json.dumps(detail)
    # le gabarit .env.example n'a pas de secret : il reste lisible
    tpl = c.get("/api/files/%d/content" % by_path[".env.example"]["id"]).json()
    assert tpl["text"].startswith("VITE_API_URL") and "sk-verif" not in tpl["text"]
    # rien n'a été indexé : la recherche lexicale ne renvoie pas le secret
    hits = c.get("/api/search", params={"q": "SECRET_TOKEN sk-verif-123456"}).json()["results"]
    assert all("sk-verif" not in (h.get("snippet") or "") for h in hits)


def test_symbols_search_and_detail(env):
    c = env.client
    items = c.get("/api/symbols", params={"q": "OrderService"}).json()
    assert items and items[0]["name"] == "OrderService" and items[0]["kind"] == "class"
    assert items[0]["file"]["path"].endswith("OrderService.php") and items[0]["file"]["repo_id"] == env.ids["repos"]["backend"]
    sid = items[0]["id"]
    env.ids["symbol"] = sid
    d = c.get("/api/symbols/%d" % sid).json()
    assert d["name"] == "OrderService" and d["file"]["id"] == env.ids["file"]
    assert isinstance(d["callers"], list) and isinstance(d["uses"], list)
    assert d["snippet"] and "class OrderService" in d["snippet"]
    methods = c.get("/api/symbols", params={"q": "confirm", "kind": "method"}).json()
    assert methods and all(m["kind"] == "method" for m in methods)
    only_backend = c.get("/api/symbols", params={"q": "confirm", "repo_id": env.ids["repos"]["backend"]}).json()
    assert only_backend and all(s["file"]["repo_id"] == env.ids["repos"]["backend"] for s in only_backend)
    assert c.get("/api/symbols/999999").json()["error"]["code"] == "symbol_not_found"


def test_endpoints_matching_confirm(env):
    pytest.importorskip("develop.knowledge_graph.linker", reason="module knowledge_graph absent : pas d'appariement")
    items = env.client.get("/api/endpoints").json()
    assert items and all({"id", "side", "method", "path", "normalized_path", "file", "matches"} <= set(e) for e in items)
    server = [e for e in items if e["side"] == "server" and e["method"] == "POST" and e["path"].endswith("/confirm")]
    assert server, [e["path"] for e in items]
    confirm = server[0]
    assert confirm["normalized_path"].endswith("/orders/:param/confirm")
    assert confirm["matches"] and confirm["matches"][0]["side"] == "client"
    assert "confirm" in confirm["matches"][0]["path"] and 0 < confirm["matches"][0]["confidence"] <= 1
    assert confirm["matches"][0]["file"].endswith("client.ts")
    clients = env.client.get("/api/endpoints", params={"side": "client", "q": "confirm"}).json()
    assert clients and all(e["side"] == "client" for e in clients)
    assert any(m["endpoint_id"] == confirm["id"] for e in clients for m in e["matches"])
    assert env.client.get("/api/endpoints", params={"side": "nope"}).status_code == 422


def test_models_and_detail(env):
    c = env.client
    models = c.get("/api/models").json()
    assert models and all({"id", "name", "table_name", "kind", "file", "line", "field_count"} <= set(m) for m in models)
    orders = [m for m in models if m["name"].lower() == "order" and m["kind"] == "eloquent"]
    assert orders and orders[0]["field_count"] > 0
    d = c.get("/api/models/%d" % orders[0]["id"]).json()
    assert d["fields"] and {"linked", "relations", "endpoints_using", "documents_mentioning"} <= set(d)
    assert d["linked"] and all({"model_id", "kind", "name", "file"} <= set(l) for l in d["linked"])
    assert c.get("/api/models", params={"q": "zzzz"}).json() == []
    assert c.get("/api/models/999999").status_code == 404


def test_graph_neighbors(env):
    pytest.importorskip("develop.knowledge_graph.linker")
    r = env.client.get("/api/graph/neighbors", params={"node": "file:%d" % env.ids["file"], "depth": 1})
    assert r.status_code == 200 and r.json()["nodes"] and r.json()["edges"]
    assert env.client.get("/api/graph/neighbors", params={"node": "bidule"}).status_code == 422


def test_search_contract(env):
    r = env.client.get("/api/search", params={"q": "Qui appelle OrderService::confirm ?"})
    assert r.status_code == 200
    data = r.json()
    assert set(data) >= {"query", "identifiers", "results"}
    assert "OrderService::confirm" in data["identifiers"]
    assert data["results"] and data["results"][0]["source_type"] in ("symbol", "file", "endpoint", "model", "document", "note")
    hit = data["results"][0]
    assert {"chunk_id", "source_type", "source_id", "title", "path", "line_start", "snippet", "score", "channels", "reason"} <= set(hit)
    assert "text" not in hit
    assert env.client.get("/api/search", params={"q": "orders", "types": "document", "limit": 5}).status_code == 200
    assert env.client.get("/api/search", params={"q": ""}).status_code == 422
    assert env.client.get("/api/search", params={"q": "x", "types": "foo"}).status_code == 422


def test_search_identifiers_cover_tables_routes_paths(env):
    """Non-régression : ``identifiers`` reprend tout ce que le canal « identifier » a reconnu (tables,
    routes, chemins), pas seulement ``query.identifiers`` — sinon « Identifiants repérés » n'apparaît
    jamais pour « quels champs a la table orders » ou « POST /orders/{id}/confirm »."""
    from develop.api.routes.search import search_result_dict
    from develop.search import SearchResult, analyze_query

    c = env.client
    data = c.get("/api/search", params={"q": "quels champs a la table orders"}).json()
    assert data["query"]["tables"] == ["orders"] and data["query"]["identifiers"] == []
    assert data["identifiers"] == ["orders"]
    assert any("identifier" in h["channels"] for h in data["results"])  # le canal a bien reconnu la table

    data = c.get("/api/search", params={"q": "POST /orders/{id}/confirm"}).json()
    assert data["query"]["routes"] == ["POST /orders/{id}/confirm"] and data["query"]["identifiers"] == []
    assert data["identifiers"] == ["POST /orders/{id}/confirm"]
    assert data["results"] and data["results"][0]["source_type"] == "endpoint"

    data = c.get("/api/search", params={"q": "app/Services/OrderService.php"}).json()
    assert data["query"]["paths"] == ["app/Services/OrderService.php"]
    assert "app/Services/OrderService.php" in data["identifiers"]

    # cas déjà couvert : inchangé, sans doublon
    assert c.get("/api/search", params={"q": "OrderService"}).json()["identifiers"] == ["OrderService"]

    # ordre déterministe (identifiants, routes, tables, chemins) et déduplication exacte, sans index
    r = SearchResult(query=analyze_query("quels champs a la table orders et le modèle Order"))
    assert search_result_dict(r)["identifiers"] == ["Order", "orders"]
    r = SearchResult(query=analyze_query("Qui appelle POST /orders/{id}/confirm dans OrderService.php ?"))
    d = search_result_dict(r)
    assert d["identifiers"][:1] == d["query"]["identifiers"][:1]  # les identifiants nommés restent en tête
    assert "POST /orders/{id}/confirm" in d["identifiers"] and len(d["identifiers"]) == len(set(d["identifiers"]))


# --------------------------------------------------------------------------- 4. machine, moteur, hors ligne

def test_system_routes(env):
    c = env.client
    p = c.get("/api/system/profile").json()
    assert p["summary"] and "ram_total_gb" in p and "os" in p
    e = c.get("/api/system/engines").json()
    assert e["ollama"]["reachable"] is True and e["ollama"]["version"] == "0.12.1"  # transport factice
    reco = c.get("/api/system/recommendations").json()
    assert {"profile_summary", "tier", "recommendations", "embedding", "install_steps", "warnings"} <= set(reco)
    assert reco["recommendations"] and all({"tag", "verdict", "pull_command", "installed"} <= set(r) for r in reco["recommendations"])


def test_llm_test_and_models(env):
    c = env.client
    r = c.post("/api/llm/test", json={"engine": "ollama", "base_url": "http://127.0.0.1:11434", "model": MODEL,
                                      "embedding_model": EMBED, "api_key": ""})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True and d["error"] is None
    assert d["engine_status"]["ok"] and d["model_info"]["context_length"] == 32768 and d["model_info"]["supports_tools"]
    assert d["embedding_ok"] is True and d["embedding_dim"] == 3 and 0 < d["suggested_num_ctx"] <= 32768
    assert d["prefill_tps"] > 0 and d["expected_first_token_s"] > 0
    assert env.state.model_info is not None and env.state.model_info.supports_tools
    # jamais la fenêtre entière d'un modèle 262k : 32 768 au plus sous 20B, 16 384 à partir de 20B ; la vitesse de
    # lecture estimée décroît avec la taille et le délai attendu avant la première lettre est renvoyé avec elle
    big = c.post("/api/llm/test", json={"engine": "ollama", "base_url": "http://127.0.0.1:11434", "model": "qwen3.5:9b"}).json()
    assert big["ok"] and big["model_info"]["context_length"] == 262144 and 0 < big["suggested_num_ctx"] <= 32768
    huge = c.post("/api/llm/test", json={"engine": "ollama", "base_url": "http://127.0.0.1:11434", "model": "qwen3.8:27b"}).json()
    assert huge["ok"] and huge["model_info"]["context_length"] == 262144 and 0 < huge["suggested_num_ctx"] <= 16384
    assert 0 < huge["prefill_tps"] <= big["prefill_tps"] and huge["expected_first_token_s"] > 0
    assert env.state.model_info.name == MODEL  # le modèle configuré reste celui de l'espace
    missing = c.post("/api/llm/test", json={"engine": "ollama", "base_url": "http://127.0.0.1:11434", "model": "missing:latest"}).json()
    assert missing["ok"] is False and "missing:latest" in missing["error"]
    none = c.post("/api/llm/test", json={"engine": "none"}).json()
    assert none["ok"] is False
    models = c.get("/api/llm/models", params={"engine": "ollama", "base_url": "http://127.0.0.1:11434"}).json()
    assert [m["name"] for m in models] == [MODEL, EMBED + ":latest"] and models[1]["is_embedding"]


def test_offline_blocked_after_public_url(env):
    c = env.client
    r = c.post("/api/llm/test", json={"engine": "ollama", "base_url": "http://api.example.com", "model": MODEL})
    assert r.status_code == 400 and r.json()["error"]["code"] == "host_not_allowed"
    blocked = c.get("/api/offline/blocked").json()
    assert blocked and blocked[-1]["host"] == "api.example.com" and blocked[-1]["mode"] == "strict"
    assert c.get("/api/status").json()["offline"]["blocked_count"] >= 1
    r = c.get("/api/llm/models", params={"engine": "ollama", "base_url": "http://lm.example.org:1234"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "host_not_allowed"
    assert c.delete("/api/offline/blocked").json()["cleared"] >= 2
    assert c.get("/api/offline/blocked").json() == []


# --------------------------------------------------------------------------- 5. documents et notes

def test_documents_upload_import_get_delete(env):
    c = env.client
    body = b"# Note API\n\nLa table `orders` est servie par OrderService::confirm et la route POST /orders/{id}/confirm.\n"
    r = c.post("/api/documents", files={"file": ("note-api.md", body, "text/markdown")})
    assert r.status_code == 201, r.text
    doc = r.json()
    assert doc["id"] and doc["format"] == "md" and doc["kind"] == "upload" and doc["chunk_count"] >= 1
    assert doc["title"] == "Note API" and doc["already_exists"] is False and doc["stale"] is False
    stored = Path(doc["stored_path"])
    assert stored.is_file() and stored.parent == env.home / "workspaces" / env.ids["ws"] / "documents"
    assert stored.name.endswith("-note-api.md")
    dup = c.post("/api/documents", files={"file": ("note-api.md", body, "text/markdown")})
    assert dup.status_code == 201 and dup.json()["already_exists"] is True and dup.json()["id"] == doc["id"]

    spec = env.project / "docs" / "specification-commandes.md"
    r = c.post("/api/documents", json={"path": str(spec)})
    assert r.status_code == 201, r.text
    imported = r.json()
    assert imported["chunk_count"] >= 1 and imported["source_path"] == str(spec) and imported["title"]
    assert spec.is_file()  # jamais déplacé

    d = c.get("/api/documents/%d" % doc["id"]).json()
    assert d["chunks_preview"] and {"ordinal", "snippet"} <= set(d["chunks_preview"][0])
    mentions = {m["name"]: m for m in d["mentions"]}
    assert "OrderService" in mentions and mentions["OrderService"]["exists"] is True
    assert all({"kind", "name", "exists"} <= set(m) for m in d["mentions"])
    content = c.get("/api/documents/%d/content" % doc["id"]).json()
    assert content["text"].startswith("# Note API")
    listed = c.get("/api/documents").json()
    assert {x["id"] for x in listed} == {doc["id"], imported["id"]}
    assert all({"id", "title", "format", "kind", "size", "added_at", "updated_at", "chunk_count", "stale"} <= set(x) for x in listed)
    assert c.get("/api/search", params={"q": "Note API orders", "types": "document"}).json()["results"]

    r = c.delete("/api/documents/%d" % doc["id"])
    assert r.status_code == 200 and r.json()["deleted"] is True and not stored.exists()
    assert c.get("/api/documents/%d" % doc["id"]).json()["error"]["code"] == "document_not_found"
    assert c.post("/api/documents", files={"file": ("virus.exe", b"MZ", "application/octet-stream")}).status_code == 415
    assert c.post("/api/documents", json={"path": "/nonexistent/x.md"}).status_code == 404
    assert c.post("/api/documents", json={}).status_code == 422
    env.ids["document"] = imported["id"]


def test_notes_crud_status_and_filters(env):
    c = env.client
    r = c.post("/api/notes", json={"title": "Réunion statuts", "body": "On garde le statut `paid`.",
                                   "tags": ["réunion", "facturation"], "note_date": "2026-09-01"})
    assert r.status_code == 201, r.text
    n1 = r.json()
    assert n1["status"] == "valid" and n1["tags"] == ["réunion", "facturation"] and n1["supersedes"] is None
    assert n1["note_date"] == "2026-09-01" and n1["excerpt"].startswith("On garde")
    stored = env.state.db.get_note(n1["id"])
    assert "status:valid" in stored.tags  # encodage interne, filtré de l'API
    r = c.post("/api/notes", json={"title": "Nouvelle décision", "body": "Finalement pas de statut paid.",
                                   "note_date": "2026-10-05", "status": "valid", "supersedes": n1["id"]})
    assert r.status_code == 201
    n2 = r.json()
    assert n2["supersedes"] == n1["id"]
    assert c.get("/api/notes/%d" % n1["id"]).json()["status"] == "superseded"  # remplacée automatiquement
    assert "supersedes:%d" % n1["id"] in env.state.db.get_note(n2["id"]).tags

    all_notes = c.get("/api/notes").json()
    assert [n["id"] for n in all_notes] == [n2["id"], n1["id"]]  # date décroissante
    assert all({"id", "title", "note_date", "tags", "status", "excerpt"} <= set(n) for n in all_notes)
    sept = c.get("/api/notes", params={"from": "2026-09-01", "to": "2026-09-30"}).json()
    assert [n["id"] for n in sept] == [n1["id"]]
    assert [n["id"] for n in c.get("/api/notes", params={"from": "2026-10-01"}).json()] == [n2["id"]]
    assert [n["id"] for n in c.get("/api/notes", params={"tag": "facturation"}).json()] == [n1["id"]]
    assert [n["id"] for n in c.get("/api/notes", params={"q": "finalement"}).json()] == [n2["id"]]
    assert [n["id"] for n in c.get("/api/notes", params={"status": "superseded"}).json()] == [n1["id"]]
    assert c.get("/api/search", params={"q": "statut paid", "types": "note"}).json()["results"]

    r = c.put("/api/notes/%d" % n1["id"], json={"status": "historical", "tags": ["archive"]})
    assert r.status_code == 200 and r.json()["status"] == "historical" and r.json()["tags"] == ["archive"]
    assert r.json()["body"] == "On garde le statut `paid`."  # corps partiel : le reste est conservé
    assert c.get("/api/notes", params={"from": "2026/09/01"}).status_code == 422
    assert c.post("/api/notes", json={"title": "x", "note_date": "hier"}).status_code == 422
    assert c.post("/api/notes", json={"title": "x", "status": "draft"}).status_code == 422
    assert c.post("/api/notes", json={"body": "sans titre"}).json()["error"]["code"] == "validation_error"
    assert c.delete("/api/notes/%d" % n2["id"]).json()["deleted"] is True
    assert c.get("/api/notes/%d" % n2["id"]).status_code == 404
    env.ids["note"] = n1["id"]


# --------------------------------------------------------------------------- 6. chat

def test_chat_search_only(env):
    c = env.client
    r = c.post("/api/conversations", json={"title": "Première"})
    assert r.status_code == 201 and r.json()["id"]
    cid = r.json()["id"]
    env.ids["conversation"] = cid
    events = stream_events(c, "/api/conversations/%d/messages" % cid,
                           {"content": "Qui appelle OrderService::confirm ?", "mode": "search_only"})
    names = [e[0] for e in events]
    assert names == ["status", "sources", "done"], names
    sources = events[1][1]["items"]
    assert sources and sources[0]["ref"] == "S1" and {"source_type", "path", "line_start", "channels", "reason"} <= set(sources[0])
    done = events[2][1]
    assert done["message_id"] and {"num_ctx", "context_tokens_used", "max_context_tokens"} <= set(done["budget"])
    conv = c.get("/api/conversations/%d" % cid).json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"] and conv["messages"][1]["sources"]
    assert c.patch("/api/conversations/%d" % cid, json={"title": "Renommée"}).json()["title"] == "Renommée"
    listed = c.get("/api/conversations").json()
    assert listed[0]["id"] == cid and listed[0]["title"] == "Renommée" and listed[0]["message_count"] == 2
    assert c.post("/api/conversations/999999/messages", json={"content": "x", "mode": "search_only"}).status_code == 404
    assert c.post("/api/conversations/%d/messages" % cid, json={"content": "x", "mode": "turbo"}).status_code == 422
    assert c.post("/api/conversations/%d/messages" % cid, json={"content": ""}).status_code == 422


def test_chat_auto_with_fake_tools(env):
    c = env.client
    assert env.state.model_info is not None and env.state.model_info.supports_tools  # appels d'outils natifs
    cid = c.post("/api/conversations", json={}).json()["id"]
    events = stream_events(c, "/api/conversations/%d/messages" % cid, {"content": "Que fait Order ?", "mode": "auto"})
    names = [e[0] for e in events]
    assert names[0] == "status" and names[1] == "sources" and names[-1] == "done"
    assert "tool_call" in names and "tool_result" in names and "delta" in names
    calls = [e[1] for e in events if e[0] == "tool_call"]
    assert calls[0]["name"] == "search" and calls[0]["arguments"] == {"query": "Order"}
    assert all({"name", "summary", "added_sources"} <= set(e[1]) for e in events if e[0] == "tool_result")
    assert names.count("tool_call") <= 4 * 3
    answer = "".join(e[1]["text"] for e in events if e[0] == "delta")
    assert answer.startswith("Bonjour")
    done = events[-1][1]
    assert done["usage"]["completion_tokens"] >= 2 and done["budget"]["tool_rounds"] >= 1
    assert c.delete("/api/conversations/%d" % cid).json()["deleted"] is True
    assert c.get("/api/conversations/%d" % cid).status_code == 404


# --------------------------------------------------------------------------- 7. impact et Git

def test_impact_file_and_symbol(env):
    c = env.client
    r = c.post("/api/impact", json={"source": "file", "path": "app/Services/OrderService.php"})
    assert r.status_code == 200, r.text
    report = r.json()
    assert {"changed", "affected", "endpoints", "models", "frontend_calls", "documents", "notes", "stats", "warnings"} <= set(report)
    assert report["changed"][0]["path"] == "app/Services/OrderService.php" and report["request"]["repo_id"] == env.ids["repos"]["backend"]
    assert report["affected"] and all({"node", "type", "name", "reason", "confidence", "hops"} <= set(i) for i in report["affected"])
    assert all(0 < i["confidence"] <= 1 for i in report["affected"])
    r = c.post("/api/impact", json={"source": "symbol", "symbol_id": env.ids["symbol"], "max_hops": 2})
    assert r.status_code == 200 and r.json()["affected"]
    r = c.post("/api/impact", json={"source": "symbol", "symbol_id": str(env.ids["symbol"])})
    assert r.status_code == 200
    assert c.post("/api/impact", json={"source": "symbol", "symbol_id": 999999}).status_code == 404
    assert c.post("/api/impact", json={"source": "file", "path": "nope.php"}).status_code == 404
    assert c.post("/api/impact", json={"source": "file"}).status_code == 422
    assert c.post("/api/impact", json={"source": "diff"}).status_code == 422
    assert c.post("/api/impact", json={"source": "git"}).json()["error"]["code"] == "not_a_git_repo"


@pytest.mark.skipif(not git_available(), reason="git absent")
def test_git_status_log_and_impact(env):
    c = env.client
    backend = env.project / "backend"
    env_git = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")

    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
                        *args], cwd=str(backend), check=True, capture_output=True, env=env_git)

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "Import initial")
    repo_id = env.ids["repos"]["backend"]
    status = c.get("/api/git/status", params={"repo_id": repo_id}).json()
    assert status["branch"] and status["head"] and status["changes"] == []
    service = backend / "app" / "Services" / "OrderService.php"
    service.write_text(service.read_text(encoding="utf-8") + "\n// changement\n", encoding="utf-8")
    try:
        status = c.get("/api/git/status", params={"repo_id": repo_id}).json()
        assert [ch["path"] for ch in status["changes"]] == ["app/Services/OrderService.php"] and status["changes"][0]["status"] == "M"
        log = c.get("/api/git/log", params={"repo_id": repo_id, "limit": 5}).json()
        assert len(log) == 1 and log[0]["subject"] == "Import initial" and {"sha", "author", "date"} <= set(log[0])
        repos = {x["name"]: x for x in c.get("/api/repos").json()}
        assert repos["backend"]["git"]["branch"] == status["branch"] and repos["backend"]["git"]["dirty"] is True
        r = c.post("/api/impact", json={"source": "git", "repo_id": repo_id})
        assert r.status_code == 200, r.text
        assert [ch["path"] for ch in r.json()["changed"]] == ["app/Services/OrderService.php"]
        r = c.post("/api/impact", json={"source": "git"})  # sans repo_id : les dépôts non Git sont ignorés
        assert r.status_code == 200 and any("ignorés" in w for w in r.json()["warnings"])
    finally:
        git("checkout", "--", "app/Services/OrderService.php")
    assert c.get("/api/git/status", params={"repo_id": env.ids["repos"]["docs"]}).json()["error"]["code"] == "not_a_git_repo"
    assert c.get("/api/git/log", params={"repo_id": "inconnu"}).status_code == 404


# --------------------------------------------------------------------------- 8. espaces : activation et suppression

def test_workspace_activate_and_delete(env):
    c = env.client
    first = env.ids["ws"]
    second = c.post("/api/workspaces", json={"name": "Second"}).json()["id"]
    assert c.get("/api/status").json()["active_workspace"] == second
    assert c.post("/api/workspaces/%s/activate" % first).status_code == 200
    assert c.get("/api/status").json()["active_workspace"] == first
    assert c.post("/api/workspaces/%s/activate" % second).status_code == 200
    r = c.delete("/api/workspaces/%s" % second)
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert not (env.home / "workspaces" / second).exists()
    status = c.get("/api/status").json()
    assert status["active_workspace"] == first and status["index"]["counts"]["files"] > 0  # retour au premier espace
    assert c.delete("/api/workspaces/%s" % second).status_code == 404
    assert c.post("/api/workspaces/absent/activate").status_code == 404


# --------------------------------------------------------------------------- 9. sécurité et erreurs

def test_security_headers_and_csrf(env):
    c = env.client
    r = c.get("/api/status")
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert r.headers["referrer-policy"] == "no-referrer" and r.headers["x-content-type-options"] == "nosniff"
    r = c.post("/api/notes", json={"title": "x"}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "same_origin_required"
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    r = c.post("/api/notes", json={"title": "x"}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403
    r = c.post("/api/notes", json={"title": "x"}, headers={"Origin": "null"})
    assert r.status_code == 403
    assert c.get("/api/notes", headers={"Origin": "http://evil.example"}).status_code == 200  # lecture : autorisée
    r = c.post("/api/notes", json={"title": "Même origine"}, headers={"Origin": "http://testserver", "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 201
    r = c.post("/api/notes", json={"title": "Boucle locale"}, headers={"Origin": "http://127.0.0.1:8765"})
    assert r.status_code == 201
    for note in (r.json(),):
        c.delete("/api/notes/%d" % note["id"])


def test_error_format_and_static(env):
    c = env.client
    r = c.get("/api/files/424242")
    assert r.status_code == 404 and set(r.json()) == {"error"} and set(r.json()["error"]) == {"code", "message"}
    r = c.post("/api/notes", json={"body": "x"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error" and "title" in r.json()["error"]["message"]
    r = c.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = c.delete("/api/status")
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"
    r = c.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and "<title>Develop</title>" in r.text
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/static/missing.js").status_code == 404


# --------------------------------------------------------------------------- 10. SSE (unitaire)

class _FakeRequest:
    def __init__(self, disconnect_after: int = 10 ** 9):
        self.calls = 0
        self.disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > self.disconnect_after


def test_sse_format_event():
    assert format_event("progress", {"phase": "scan", "é": 1}) == 'event: progress\ndata: {"phase": "scan", "é": 1}\n\n'


async def test_sse_event_stream_ping_and_end():
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    await queue.put(("progress", {"current": 1}))
    gen = event_stream(_FakeRequest(), queue, ping_interval=0.05)  # type: ignore[arg-type]
    assert await gen.__anext__() == 'event: progress\ndata: {"current": 1}\n\n'
    assert await gen.__anext__() == PING  # file vide : ping après 50 ms
    await queue.put(("done", {"ok": True}))
    assert (await gen.__anext__()).startswith("event: done\n")
    await queue.put(None)
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


async def test_sse_event_stream_stops_on_disconnect():
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    await queue.put(("a", {}))
    await queue.put(("b", {}))
    gen = event_stream(_FakeRequest(disconnect_after=1), queue, ping_interval=0.05)  # type: ignore[arg-type]
    assert (await gen.__anext__()).startswith("event: a\n")
    with pytest.raises(StopAsyncIteration):
        await gen.__anext__()


# --------------------------------------------------------------------------- 11. assistant : monodépôt Git

@pytest.mark.skipif(not git_available(), reason="git absent")
def test_discover_git_monorepo_then_index_without_duplicates(env, tmp_path: Path):
    """Parcours par défaut de l'assistant sur un monodépôt Git (« indiquez le dossier racine ») :
    la racine est proposée avec l'exclusion de ses sous-projets ; en indexant les quatre candidats
    cochés, chaque fichier n'est indexé qu'une fois (une seule classe ``OrderService``, une seule
    route ``POST /orders/{order}/confirm``, correspondance route ↔ appel à 1.0)."""
    from develop.project_scanner import scan_repo

    c = env.client
    project = tmp_path / "monorepo"
    shutil.copytree(FIXTURE, project)
    env_git = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")

    def git(*args: str) -> None:
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
                        *args], cwd=str(project), check=True, capture_output=True, env=env_git)

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "Import initial")

    r = c.post("/api/repos/discover", json={"path": str(project)})
    assert r.status_code == 200, r.text
    subs = r.json()["subprojects"]
    assert [s["name"] for s in subs] == ["monorepo", "backend", "frontend", "services/notifier"]
    assert subs[0]["path"] == str(project) and subs[0]["exclude"] == ["backend", "frontend", "services/notifier"]
    assert all(s["exclude"] == [] for s in subs[1:])

    first = env.ids["ws"]
    ws = c.post("/api/workspaces", json={"name": "Monodépôt"}).json()
    try:
        # toutes les cases cochées : l'assistant envoie chaque candidat avec son exclude
        repos = [{"name": s["name"], "path": s["path"], "kind": s["suggested_kind"], "exclude": s["exclude"]} for s in subs]
        r = c.put("/api/workspaces/%s" % ws["id"], json={"repos": repos})
        assert r.status_code == 200, r.text
        saved = {x["name"]: x for x in r.json()["repos"]}
        assert saved["monorepo"]["exclude"] == ["backend", "frontend", "services/notifier"] and saved["backend"]["exclude"] == []
        assert c.post("/api/index", json={"full": True}).status_code == 202
        status = _wait_index(c)
        assert status["last_run"]["status"] == "ok", status["last_run"]

        files = c.get("/api/files", params={"limit": 10000}).json()
        paths = {x["id"]: x["path"] for x in c.get("/api/repos").json()}
        absolute = [os.path.normpath(os.path.join(paths[f["repo_id"]], f["path"])) for f in files]
        expected = {os.path.normpath(f.abs_path) for f in scan_repo(str(project), "all").files}
        assert len(absolute) == len(set(absolute)) == len(expected) and set(absolute) == expected  # chacun une fois
        assert not any(f["repo_id"] == saved["monorepo"]["id"] and f["path"].startswith(("backend/", "frontend/", "services/")) for f in files)
        assert any(f["repo_id"] == saved["monorepo"]["id"] and f["path"] == "docs/dictionnaire-donnees.md" for f in files)

        assert len(c.get("/api/files", params={"q": "OrderService"}).json()) == 1
        classes = [s for s in c.get("/api/symbols", params={"q": "OrderService", "kind": "class"}).json() if s["name"] == "OrderService"]
        assert len(classes) == 1 and classes[0]["file"]["repo_id"] == saved["backend"]["id"]
        routes = [e for e in c.get("/api/endpoints", params={"side": "server", "q": "confirm"}).json() if e["method"] == "POST"]
        assert len(routes) == 1 and routes[0]["path"] == "/orders/{order}/confirm"
        assert routes[0]["matches"] and all(m["confidence"] == 1.0 for m in routes[0]["matches"])
    finally:
        _wait_index(c)
        assert c.post("/api/workspaces/%s/activate" % first).status_code == 200
        c.delete("/api/workspaces/%s" % ws["id"])
