"""Indexation en arrière-plan : lancement, état, flux SSE de progression, annulation."""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, Body, Request
from pydantic import BaseModel

from develop.api.routes import api_error, get_state, require_workspace
from develop.api.sse import event_stream, format_event, sse_response
from develop.api.state import IndexRunning

router = APIRouter(prefix="/api/index", tags=["index"])


class IndexRequest(BaseModel):
    full: bool = False
    repo_id: Optional[str] = None
    with_embeddings: bool = True


@router.post("", status_code=202)
def start_index(request: Request, body: IndexRequest = Body(default_factory=IndexRequest)) -> Dict[str, Any]:
    """Démarre un run (incrémental par défaut) ; ``{run_id}`` ; 409 ``index_running`` si déjà en cours."""
    state = get_state(request)
    require_workspace(state)
    try:
        run_id = state.start_index(full=body.full, repo_id=body.repo_id, with_embeddings=body.with_embeddings)
    except IndexRunning as exc:
        raise api_error(409, "index_running", str(exc))
    except KeyError:
        raise api_error(404, "repo_not_found", "Dépôt « %s » inconnu dans l'espace actif." % body.repo_id)
    return {"run_id": run_id, "running": True}


@router.get("/status")
def index_status(request: Request) -> Dict[str, Any]:
    return get_state(request).index_status()


@router.get("/events")
async def index_events(request: Request):  # type: ignore[no-untyped-def]
    """SSE : ``progress`` (même forme que ``/status``), ``done``, ``error`` ; ``: ping`` toutes les 15 s.

    Le flux commence par l'état courant (``progress``) ; sans run en cours il se termine aussitôt
    (l'interface traite un ``progress`` avec ``running=false`` comme une fin), sinon il se termine
    après ``done`` ou ``error``. L'abonnement précède la lecture de l'état : aucun événement perdu.
    """
    state = get_state(request)
    queue = state.subscribe()

    async def generate() -> AsyncIterator[str]:
        try:
            status = state.index_status()
            yield format_event("progress", status)
            if not status["running"]:
                return
            async for block in event_stream(request, queue):
                yield block
                if block.startswith(("event: done\n", "event: error\n")):
                    return
        finally:
            state.unsubscribe(queue)

    return sse_response(generate())


@router.post("/cancel")
def cancel_index(request: Request) -> Dict[str, Any]:
    state = get_state(request)
    running = state.cancel_index()
    return {"cancelled": running, "run_id": state.index_run_id,
            "message": "Arrêt demandé." if running else "Aucune indexation en cours."}
