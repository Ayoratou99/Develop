"""Chargement du modèle façon LM Studio : ``GET /api/llm/state``, ``POST /api/llm/load``, ``POST /api/llm/unload``,
``GET /api/llm/events`` (``develop.api.routes.llm``, ``develop.api.state``).

Faux Ollama **à mémoire** (``httpx.MockTransport`` asynchrone) : ``/api/ps`` renvoie les modèles « en mémoire »
avec la fenêtre (``context_length``) de leur chargement, ``POST /api/generate`` sans prompt y charge un modèle
(``keep_alive`` 30m, ``options.num_ctx`` sinon fenêtre « auto » 32 768 comme Ollama ≥ 0.40, délai réglable) ou le
décharge (``keep_alive: 0``), ``/api/embed`` charge le modèle d'embeddings. ``DEVELOP_HOME`` pointe vers un
dossier temporaire : rien n'est écrit dans ``~/.develop``.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Set, Tuple

import httpx
import pytest
from fastapi.testclient import TestClient

from develop.api.state import model_is_loaded

MODEL = "qwen3:14b"
EMBED = "qwen3-embedding:0.6b"
WAIT_S = 5.0
AUTO_NUM_CTX = 32768  # fenêtre choisie par Ollama quand la requête ne donne pas options.num_ctx (« 4k/32k/256k selon la VRAM »)


class FakeOllamaMemory:
    """Ollama simulé avec une mémoire de modèles ; chaque requête est journalisée dans ``calls``."""

    def __init__(self) -> None:
        self.loaded: Set[str] = set()
        self.contexts: Dict[str, Any] = {}  # modèle → fenêtre chargée (context_length de /api/ps ; None : non dite)
        self.generate_delay = 0.0  # secondes pour « charger » un modèle (POST /api/generate)
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def generate_calls(self, model: str, keep_alive: Any) -> int:
        return sum(1 for p, b in self.calls if p == "/api/generate" and b.get("model") == model and b.get("keep_alive") == keep_alive)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        p = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.calls.append((p, body))
        if p == "/api/version":
            return httpx.Response(200, json={"version": "0.40.0"})
        if p == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": MODEL, "size": 9_000_000_000, "details": {"family": "qwen3", "parameter_size": "14.8B", "quantization_level": "Q4_K_M"}},
                {"name": EMBED, "size": 600_000_000, "details": {"family": "qwen3", "parameter_size": "0.6B", "quantization_level": "F16"}},
            ]})
        if p == "/api/show":
            return httpx.Response(200, json={"details": {"family": "qwen3", "parameter_size": "14.8B"}, "capabilities": ["completion", "tools"],
                                             "model_info": {"qwen3.context_length": 40960}})
        if p == "/api/ps":
            models = []
            for n in sorted(self.loaded):
                entry: Dict[str, Any] = {"name": n, "model": n, "size": 1}
                if self.contexts.get(n):
                    entry["context_length"] = self.contexts[n]
                models.append(entry)
            return httpx.Response(200, json={"models": models})
        if p == "/api/generate":
            model = body.get("model", "")
            if model not in (MODEL, EMBED):
                return httpx.Response(404, json={"error": "model '%s' not found" % model})
            if body.get("keep_alive") == 0:
                self.loaded.discard(model)
                self.contexts.pop(model, None)
                return httpx.Response(200, json={"model": model, "done": True, "done_reason": "unload"})
            if self.generate_delay:
                await asyncio.sleep(self.generate_delay)
            self.loaded.add(model)
            self.contexts[model] = int((body.get("options") or {}).get("num_ctx") or AUTO_NUM_CTX)
            return httpx.Response(200, json={"model": model, "done": True, "done_reason": "load"})
        if p == "/api/embed":
            self.loaded.add(body.get("model", ""))
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in body.get("input", [])]})
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


def wait_not_loading(client: TestClient, timeout: float = WAIT_S) -> Dict[str, Any]:
    end = time.time() + timeout
    while True:
        state = client.get("/api/llm/state").json()
        if not state["loading"] or time.time() > end:
            assert not state["loading"], "chargement trop long : %r" % state
            return state
        time.sleep(0.05)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    home = tmp_path_factory.mktemp("home")
    previous = os.environ.get("DEVELOP_HOME")
    os.environ["DEVELOP_HOME"] = str(home)
    from develop.api import create_app

    fake = FakeOllamaMemory()
    app = create_app(llm_transport=httpx.MockTransport(fake))
    with TestClient(app) as client:
        ws = client.post("/api/workspaces", json={"name": "Chargement"}).json()
        r = client.put("/api/workspaces/%s" % ws["id"], json={"llm": {"engine": "ollama", "model": MODEL, "embedding_model": EMBED}})
        assert r.status_code == 200, r.text
        yield SimpleNamespace(client=client, fake=fake, state=app.state.app_state, ws_id=ws["id"])
    if previous is None:
        os.environ.pop("DEVELOP_HOME", None)
    else:
        os.environ["DEVELOP_HOME"] = previous


def test_model_is_loaded_matches_tags_exactly():
    assert model_is_loaded("qwen3", ["qwen3:latest"]) and model_is_loaded("qwen3:latest", ["qwen3"])
    assert model_is_loaded(MODEL, [EMBED, MODEL]) and not model_is_loaded("qwen3:14b", ["qwen3:8b"])
    assert not model_is_loaded("", [MODEL]) and not model_is_loaded(MODEL, [])


def test_state_reports_what_the_engine_has_in_memory(env):
    """Après la configuration du moteur, le préchauffage automatique a chargé les deux modèles ; l'état suit /api/ps."""
    c, fake = env.client, env.fake
    state = wait_not_loading(c)
    assert state["engine"] == "ollama" and state["model"] == MODEL and state["embedding_model"] == EMBED
    assert state["known"] is True and state["loaded"] is True and state["embedding_loaded"] is True
    assert state["loading"] is False and state["loading_since"] is None and state["error"] is None
    assert isinstance(state["last_load_s"], (int, float)) and state["last_load_s"] >= 0
    assert fake.generate_calls(MODEL, "30m") == 1 and any(p == "/api/embed" and b["model"] == EMBED for p, b in fake.calls)
    # Ollama a déchargé les modèles (délai keep_alive, redémarrage) : l'état le dit, sans erreur (moteur joignable)
    fake.loaded.clear()
    state = c.get("/api/llm/state").json()
    assert state["loaded"] is False and state["embedding_loaded"] is False and state["error"] is None
    fake.loaded.add(EMBED)
    state = c.get("/api/llm/state").json()
    assert state["loaded"] is False and state["embedding_loaded"] is True
    fake.loaded.add(MODEL)
    assert c.get("/api/llm/state").json()["loaded"] is True


def test_load_runs_in_background_and_events_follow_until_loaded(env):
    """POST /api/llm/load → 202 et chargement en tâche de fond (un second appel le rejoint) ; GET /api/llm/events
    envoie un ``state`` par seconde puis le dernier (``loading: false``, ``loaded: true``) et se termine."""
    c, fake = env.client, env.fake
    # au repos : un seul événement, flux terminé aussitôt
    with c.stream("GET", "/api/llm/events") as resp:
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
        idle = parse_sse("".join(resp.iter_text()))
    assert [e[0] for e in idle] == ["state"] and idle[0][1]["loading"] is False

    fake.loaded.clear()
    fake.generate_delay = 0.8
    before = fake.generate_calls(MODEL, "30m")
    r = c.post("/api/llm/load")
    assert r.status_code == 202, r.text
    started = r.json()
    assert started["loading"] is True and started["loaded"] is False
    assert abs(started["loading_since"] - time.time()) < 5
    assert c.post("/api/llm/load").status_code == 202  # rejoint le chargement en cours
    assert c.get("/api/llm/state").json()["loading"] is True

    with c.stream("GET", "/api/llm/events") as resp:
        events = parse_sse("".join(resp.iter_text()))
    assert events and all(name == "state" for name, _ in events)
    assert events[0][1]["loading"] is True
    last = events[-1][1]
    assert last["loading"] is False and last["loaded"] is True and last["embedding_loaded"] is True and last["error"] is None
    assert last["last_load_s"] >= 0.8
    assert fake.generate_calls(MODEL, "30m") == before + 1  # un seul chargement malgré deux POST /api/llm/load
    assert c.get("/api/llm/state").json()["loading"] is False
    fake.generate_delay = 0.0


def test_unload_frees_both_models_and_cancels_a_running_load(env):
    c, fake = env.client, env.fake
    fake.loaded.update({MODEL, EMBED})
    r = c.post("/api/llm/unload")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["loaded"] is False and data["embedding_loaded"] is False and data["loading"] is False
    assert data["unloaded"] == [MODEL, EMBED]
    assert fake.generate_calls(MODEL, 0) == 1 and fake.generate_calls(EMBED, 0) == 1 and fake.loaded == set()
    # décharger pendant un chargement l'interrompt : plus de chargement, modèle hors mémoire
    fake.generate_delay = 0.8
    assert c.post("/api/llm/load").json()["loading"] is True
    data = c.post("/api/llm/unload").json()
    assert data["loading"] is False and data["loaded"] is False
    time.sleep(1.0)  # la tâche annulée ne doit pas réapparaître
    state = c.get("/api/llm/state").json()
    assert state["loading"] is False and state["loaded"] is False and MODEL not in fake.loaded
    fake.generate_delay = 0.0


def test_warm_loads_with_the_chat_window_and_state_flags_a_pending_reload(env):
    """Double chargement : Ollama recharge un modèle dès que ``options.num_ctx`` diffère de celui du runner. Le
    préchauffage charge donc avec la fenêtre que le chat demandera (``num_ctx`` effectif du budget : ici la
    configuration, 4 096, sous la fenêtre du modèle 40 960 lue sur ``/api/show`` avant le chargement), et
    ``GET /api/llm/state`` expose ``num_ctx`` / ``loaded_num_ctx`` / ``reload_pending`` (modèle chargé par un autre
    client ou avec la fenêtre « auto » d'Ollama : rechargement au prochain message, que « Recharger » fait aussitôt)."""
    c, fake, state = env.client, env.fake, env.state
    fake.loaded.clear()
    fake.contexts.clear()
    fake.calls.clear()
    r = c.put("/api/workspaces/%s" % env.ws_id, json={"llm": {"engine": "ollama", "model": MODEL, "embedding_model": EMBED, "num_ctx": 4096}})
    assert r.status_code == 200, r.text
    st = wait_not_loading(c)
    loads = [b for p, b in fake.calls if p == "/api/generate" and b.get("model") == MODEL]
    assert loads and loads[-1]["options"] == {"num_ctx": 4096} and loads[-1]["keep_alive"] == "30m"
    # la fenêtre du modèle a été demandée (/api/show) avant le chargement : même num_ctx que le chat (budget)
    paths = [p for p, _ in fake.calls]
    assert "/api/show" in paths and paths.index("/api/show") < paths.index("/api/generate")
    assert state.model_info is not None and state.model_info.context_length == 40960
    assert state.effective_num_ctx() == 4096
    assert st["loaded"] is True and st["num_ctx"] == 4096 and st["loaded_num_ctx"] == 4096 and st["reload_pending"] is False
    assert st["error"] is None and fake.contexts[MODEL] == 4096

    # Ollama a (re)chargé le modèle avec sa fenêtre « auto » (autre client, « ollama run ») : en mémoire, mais le
    # prochain message du chat (num_ctx 4096) le rechargera → reload_pending
    fake.contexts[MODEL] = AUTO_NUM_CTX
    st = c.get("/api/llm/state").json()
    assert st["loaded"] is True and st["loaded_num_ctx"] == AUTO_NUM_CTX and st["num_ctx"] == 4096 and st["reload_pending"] is True
    # un Ollama ancien qui ne dit pas la fenêtre : rien à comparer, pas de fausse alerte
    fake.contexts[MODEL] = None
    st = c.get("/api/llm/state").json()
    assert st["loaded"] is True and st["loaded_num_ctx"] is None and st["reload_pending"] is False
    # « Recharger » = POST /api/llm/load : préchauffage avec la fenêtre du chat, plus de rechargement en attente
    fake.contexts[MODEL] = AUTO_NUM_CTX
    assert c.get("/api/llm/state").json()["reload_pending"] is True
    before = len(fake.calls)
    assert c.post("/api/llm/load").status_code == 202
    st = wait_not_loading(c)
    reloads = [b for p, b in fake.calls[before:] if p == "/api/generate" and b.get("model") == MODEL]
    assert reloads == [{"model": MODEL, "keep_alive": "30m", "options": {"num_ctx": 4096}}]
    assert st["loaded"] is True and st["loaded_num_ctx"] == 4096 and st["reload_pending"] is False
    # modèle hors mémoire : pas de fenêtre chargée, pas de rechargement en attente (c'est un chargement)
    fake.loaded.discard(MODEL)
    st = c.get("/api/llm/state").json()
    assert st["loaded"] is False and st["loaded_num_ctx"] is None and st["reload_pending"] is False and st["num_ctx"] == 4096
    fake.loaded.add(MODEL)
    # retour à la configuration du module (num_ctx par défaut) pour les tests suivants
    assert c.put("/api/workspaces/%s" % env.ws_id, json={"llm": {"engine": "ollama", "model": MODEL, "embedding_model": EMBED}}).status_code == 200
    wait_not_loading(c)


def test_engine_without_memory_info_blocks_nothing_and_reconfigure_restarts_warm(env):
    """Serveur compatible OpenAI : ``known: false``, ``loaded: true`` (rien n'est bloqué), pas de déchargement ;
    moteur « aucun » : 400 ; retour à Ollama : le préchauffage repart tout seul."""
    c, fake = env.client, env.fake
    r = c.put("/api/workspaces/%s" % env.ws_id, json={"llm": {"engine": "openai_compat", "base_url": "http://127.0.0.1:1234/v1",
                                                               "model": "local-model", "embedding_model": ""}})
    assert r.status_code == 200, r.text
    state = wait_not_loading(c)
    assert state["engine"] == "openai_compat" and state["known"] is False
    assert state["loaded"] is True and state["embedding_loaded"] is True and state["error"] is None
    assert state["loaded_num_ctx"] is None and state["reload_pending"] is False and state["num_ctx"] >= 2048
    r = c.post("/api/llm/unload")
    assert r.status_code == 400 and r.json()["error"]["code"] == "unload_unsupported"
    assert c.post("/api/llm/load").status_code == 202  # accepté, sans effet bloquant

    assert c.put("/api/workspaces/%s" % env.ws_id, json={"llm": {"engine": "none"}}).status_code == 200
    state = c.get("/api/llm/state").json()
    assert state["engine"] == "none" and state["known"] is False and state["loaded"] is True and state["loading"] is False
    assert state["num_ctx"] is None and state["loaded_num_ctx"] is None and state["reload_pending"] is False
    assert c.post("/api/llm/load").json()["error"]["code"] == "no_engine"
    assert c.post("/api/llm/unload").json()["error"]["code"] == "no_engine"

    fake.loaded.clear()
    before = fake.generate_calls(MODEL, "30m")
    r = c.put("/api/workspaces/%s" % env.ws_id, json={"llm": {"engine": "ollama", "base_url": "http://127.0.0.1:11434",
                                                               "model": MODEL, "embedding_model": EMBED}})
    assert r.status_code == 200, r.text
    state = wait_not_loading(c)
    assert state["known"] is True and state["loaded"] is True and state["embedding_loaded"] is True and state["error"] is None
    assert fake.generate_calls(MODEL, "30m") == before + 1
