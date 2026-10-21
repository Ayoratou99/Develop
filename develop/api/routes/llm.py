"""Test d'un moteur LLM, liste des modèles, chargement du modèle en mémoire et journal des requêtes bloquées.

``POST /api/llm/test`` construit un client **temporaire** d'après le corps (rien n'est
sauvegardé) : la garde hors ligne est consultée avant tout contact (400 ``host_not_allowed`` si
l'hôte est interdit, tentative journalisée), puis ``health``, ``model_info``, un embedding d'essai
et un ``num_ctx`` suggéré par :func:`develop.chat.budget.compute_budget` : min(fenêtre réelle du
modèle, :data:`SUGGESTED_NUM_CTX_LARGE` dès 20 milliards de paramètres sinon
:data:`SUGGESTED_NUM_CTX_DEFAULT`, plafond RAM de la machine) — jamais la fenêtre entière d'un
modèle 128k, qu'un portable met des minutes à lire — avec le délai attendu avant la première lettre
(``expected_first_token_s`` = prompt maximal du budget ÷ ``prefill_tps``, vitesse de lecture mesurée
par le chat sinon estimée).

Chargement façon LM Studio, sur le moteur **configuré** : ``GET /api/llm/state`` dit si le modèle de
chat est en mémoire (Ollama : ``/api/ps``) et si un préchauffage est en cours ; ``POST /api/llm/load``
lance (ou rejoint) le préchauffage en tâche de fond ; ``POST /api/llm/unload`` libère la mémoire ;
``GET /api/llm/events`` suit le chargement en SSE (un événement ``state`` par seconde, puis fin).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from develop.api.routes import api_error, get_state
from develop.api.sse import format_event, sse_response
from develop.chat.budget import compute_budget
from develop.config import LLMConfig
from develop.llm import factory
from develop.llm.advisor import model_params_b
from develop.llm.base import LLMClient, LLMError, LLMUnavailable, ModelNotFound
from develop.llm.offline_guard import OfflineViolation

router = APIRouter(prefix="/api", tags=["llm"])

TEST_TIMEOUT_S = 15.0
SUGGESTED_NUM_CTX_DEFAULT = 32768  # modèles < 20B
SUGGESTED_NUM_CTX_LARGE = 16384  # ≥ 20B : la lecture du prompt est déjà lente, la fenêtre n'a pas à l'allonger
LARGE_MODEL_PARAMS_B = 20.0


class LLMTestRequest(BaseModel):
    engine: str = Field(default="ollama")
    base_url: str = Field(default="")
    model: str = Field(default="")
    embedding_model: str = Field(default="")
    api_key: str = Field(default="")
    num_ctx: Optional[int] = None


def _temp_client(state: Any, engine: str, base_url: str, api_key: str = "") -> LLMClient:
    if engine not in ("ollama", "openai_compat"):
        raise api_error(400, "invalid_engine", "Moteur inconnu « %s » : attendu ollama ou openai_compat." % engine)
    base_url = (base_url or "").strip() or ("http://127.0.0.1:11434" if engine == "ollama" else "http://127.0.0.1:1234/v1")
    try:
        state.guard.check(base_url)
    except OfflineViolation as exc:
        raise api_error(400, "host_not_allowed",
                        "Adresse refusée par le mode hors ligne (%s) : %s. Utilisez une adresse locale, ou passez en "
                        "mode « réseau local » et autorisez cet hôte." % (exc.mode, exc.host))
    cfg = LLMConfig(engine=engine, base_url=base_url, api_key=api_key or "")  # type: ignore[arg-type]
    return factory.create_client(cfg, state.guard, transport=state.llm_transport)


@router.post("/llm/test")
async def llm_test(body: LLMTestRequest, request: Request) -> Dict[str, Any]:
    """``{ok, engine_status, model_info, embedding_ok, embedding_dim, suggested_num_ctx, expected_first_token_s,
    prefill_tps, error}``."""
    state = get_state(request)
    result: Dict[str, Any] = {"ok": False, "engine_status": None, "model_info": None, "embedding_ok": False,
                              "embedding_dim": None, "suggested_num_ctx": None, "expected_first_token_s": None,
                              "prefill_tps": None, "error": None}
    if body.engine == "none":
        result["error"] = "Aucun moteur sélectionné : la recherche fonctionne, mais pas le chat."
        return result
    client = _temp_client(state, body.engine, body.base_url, body.api_key)
    try:
        try:
            status = await asyncio.wait_for(client.health(), TEST_TIMEOUT_S)
        except asyncio.TimeoutError:
            result["error"] = "Le moteur ne répond pas (délai de %d s dépassé)." % int(TEST_TIMEOUT_S)
            return result
        result["engine_status"] = status.to_dict()
        if not status.ok:
            result["error"] = status.error or "Moteur injoignable."
            return result
        info = None
        if body.model.strip():
            try:
                info = await asyncio.wait_for(client.model_info(body.model.strip()), TEST_TIMEOUT_S)
                result["model_info"] = info.to_dict()
            except ModelNotFound:
                result["error"] = ("Le modèle « %s » n'est pas installé sur ce moteur. Installez-le (ex. « ollama pull %s ») "
                                   "ou choisissez-en un autre." % (body.model, body.model))
                return result
            except (LLMError, OfflineViolation, asyncio.TimeoutError) as exc:
                result["error"] = "Informations du modèle indisponibles : %s" % exc
                return result
        else:
            result["error"] = "Indiquez le modèle de chat à tester."
        if body.embedding_model.strip():
            try:
                vectors = await asyncio.wait_for(client.embed(["test"], body.embedding_model.strip()), TEST_TIMEOUT_S)
                dim = len(vectors[0]) if vectors and vectors[0] else 0
                result["embedding_ok"] = dim > 0
                result["embedding_dim"] = dim or None
                if dim == 0:
                    result["error"] = "Le modèle d'embeddings « %s » a renvoyé un vecteur vide." % body.embedding_model
            except (LLMError, OfflineViolation, asyncio.TimeoutError) as exc:
                result["embedding_ok"] = False
                result["error"] = "Embeddings indisponibles avec « %s » : %s" % (body.embedding_model, exc)
        model = body.model.strip()
        params_b = model_params_b(info) or model_params_b(model)
        size_cap = SUGGESTED_NUM_CTX_LARGE if (params_b or 0) >= LARGE_MODEL_PARAMS_B else SUGGESTED_NUM_CTX_DEFAULT
        cfg = LLMConfig(engine=body.engine, base_url=body.base_url, model=model, num_ctx=size_cap)  # type: ignore[arg-type]
        tps, _ = state.prefill_tps(body.engine, model, info)
        tools_mode = "native" if (info is not None and info.supports_tools) else "text"
        budget = compute_budget(cfg, info, state.profile(), tools_mode=tools_mode, prefill_tps=tps)
        result["suggested_num_ctx"] = budget.num_ctx
        result["prefill_tps"] = round(tps, 1)
        expected = budget.expected_first_token_s
        result["expected_first_token_s"] = round(expected, 1) if expected is not None else None
        result["ok"] = result["error"] is None
        ws = state.workspace
        if info is not None and ws is not None and ws.llm.engine == body.engine and ws.llm.model == body.model.strip():
            state.set_model_info(info)
        return result
    finally:
        await client.aclose()


@router.get("/llm/models")
async def llm_models(request: Request, engine: str = Query(default="ollama"),
                     base_url: str = Query(default=""), api_key: str = Query(default="")) -> List[Dict[str, Any]]:
    """``[ModelInfo]`` du moteur indiqué, sans rien sauvegarder."""
    state = get_state(request)
    if engine == "none":
        return []
    client = _temp_client(state, engine, base_url, api_key)
    try:
        models = await asyncio.wait_for(client.list_models(), TEST_TIMEOUT_S)
    except OfflineViolation as exc:
        raise api_error(400, "host_not_allowed", "Adresse refusée par le mode hors ligne : %s." % exc.host)
    except (LLMUnavailable, asyncio.TimeoutError) as exc:
        raise api_error(503, "llm_unavailable", str(exc) or "Moteur injoignable.")
    except LLMError as exc:
        raise api_error(502, "llm_error", str(exc))
    finally:
        await client.aclose()
    return [m.to_dict() for m in models]


# --------------------------------------------------------------------------- chargement du modèle

EVENTS_TICK_S = 1.0


@router.get("/llm/state")
async def llm_state(request: Request) -> Dict[str, Any]:
    """``{engine, model, embedding_model, known, loaded, embedding_loaded, loading, loading_since, last_load_s, error}``."""
    return await get_state(request).llm_state()


@router.post("/llm/load", status_code=202)
async def llm_load(request: Request) -> Dict[str, Any]:
    """Lance le préchauffage du modèle configuré en tâche de fond (ou rejoint celui en cours) ; 202 ``{loading: true, …}``."""
    state = get_state(request)
    if state.start_warm() is None:
        raise api_error(400, "no_engine", "Aucun moteur ou modèle de chat configuré : choisissez-en un dans Réglages › Moteur d'IA.")
    return await state.llm_state()


@router.post("/llm/unload")
async def llm_unload(request: Request) -> Dict[str, Any]:
    """Décharge le modèle de chat et le modèle d'embeddings de la mémoire du moteur ; ``{loaded: false, unloaded: [...], …}``."""
    state = get_state(request)
    ws = state.workspace
    if ws is None or ws.llm.engine == "none" or not ws.llm.model:
        raise api_error(400, "no_engine", "Aucun moteur ou modèle de chat configuré : rien à décharger.")
    if not state.llm.reports_loaded_models:
        raise api_error(400, "unload_unsupported", "Ce moteur ne permet pas de décharger un modèle depuis Develop "
                        "(utilisez son interface ou arrêtez le serveur).")
    state.cancel_warm()
    unloaded: List[str] = []
    for name in (ws.llm.model, ws.llm.embedding_model):
        if not name or name in unloaded:
            continue
        try:
            if await asyncio.wait_for(state.llm.unload(name), 60.0):
                unloaded.append(name)
        except OfflineViolation as exc:
            raise api_error(400, "host_not_allowed", "Adresse refusée par le mode hors ligne : %s." % exc.host)
        except (LLMUnavailable, asyncio.TimeoutError) as exc:
            raise api_error(503, "llm_unavailable", str(exc) or "Moteur injoignable.")
        except LLMError as exc:
            raise api_error(502, "llm_error", str(exc))
    result = await state.llm_state()
    result["unloaded"] = unloaded
    return result


@router.get("/llm/events")
async def llm_events(request: Request):  # type: ignore[no-untyped-def]
    """SSE : un événement ``state`` (forme de ``GET /api/llm/state``) par seconde tant qu'un chargement est en
    cours, puis un dernier ``state`` et fin du flux. Sans chargement, un seul ``state`` puis fin."""
    state = get_state(request)

    async def generate() -> AsyncIterator[str]:
        while True:
            current = await state.llm_state()
            yield format_event("state", current)
            if not current["loading"] or await request.is_disconnected():
                return
            await state.wait_warm(EVENTS_TICK_S)  # réveil dès la fin du chargement, sinon toutes les secondes

    return sse_response(generate())


@router.get("/offline/blocked")
def offline_blocked(request: Request) -> List[Dict[str, Any]]:
    """Journal des tentatives réseau bloquées par la garde hors ligne."""
    return get_state(request).blocked_attempts()


@router.delete("/offline/blocked")
def offline_blocked_clear(request: Request) -> Dict[str, Any]:
    state = get_state(request)
    n = len(state.guard.blocked_attempts())
    state.guard.clear_log()
    return {"cleared": n}
