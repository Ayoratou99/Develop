"""Conversations et envoi de messages en flux SSE (``ChatOrchestrator.respond``)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, Field

from develop.api.routes import api_error, get_state, require_db
from develop.api.sse import sse_response
from develop.storage import Database

logger = logging.getLogger("develop.api.chat")
router = APIRouter(prefix="/api/conversations", tags=["chat"])

MODES = ("auto", "search_only", "no_tools")


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=300)


class ConversationPatch(BaseModel):
    title: str = Field(min_length=1, max_length=300)


class MessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    mode: str = "auto"
    max_context_tokens: Optional[int] = Field(default=None, ge=1)


def _conv_or_404(db: Database, conversation_id: int) -> Dict[str, Any]:
    conv = db.get_conversation(conversation_id)
    if conv is None:
        raise api_error(404, "conversation_not_found", "Conversation %s introuvable." % conversation_id)
    return conv


@router.get("")
def list_conversations(request: Request) -> List[Dict[str, Any]]:
    db = require_db(get_state(request))
    return [c.to_dict() for c in db.list_conversations(limit=500)]


@router.post("", status_code=201)
def create_conversation(request: Request, body: ConversationCreate = Body(default_factory=ConversationCreate)) -> Dict[str, Any]:
    db = require_db(get_state(request))
    title = (body.title or "").strip()
    conv_id = db.create_conversation(title)
    return {"id": conv_id, "title": title}


@router.get("/{conversation_id}")
def get_conversation(conversation_id: int, request: Request) -> Dict[str, Any]:
    return _conv_or_404(require_db(get_state(request)), conversation_id)


@router.patch("/{conversation_id}")
def rename_conversation(conversation_id: int, body: ConversationPatch, request: Request) -> Dict[str, Any]:
    db = require_db(get_state(request))
    _conv_or_404(db, conversation_id)
    db.rename_conversation(conversation_id, body.title.strip())
    return {"id": conversation_id, "title": body.title.strip()}


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: int, request: Request) -> Dict[str, Any]:
    db = require_db(get_state(request))
    _conv_or_404(db, conversation_id)
    db.delete_conversation(conversation_id)
    return {"deleted": True, "id": conversation_id}


@router.post("/{conversation_id}/messages")
async def send_message(conversation_id: int, body: MessageIn, request: Request):  # type: ignore[no-untyped-def]
    """Flux SSE : ``status``, ``sources``, ``tool_call``/``tool_result``, ``delta``, ``done`` (ou ``error``)."""
    state = get_state(request)
    db = require_db(state)
    _conv_or_404(db, conversation_id)
    if body.mode not in MODES:
        raise api_error(422, "validation_error", "« mode » doit valoir %s." % ", ".join(MODES))
    orchestrator = state.orchestrator
    assert orchestrator is not None
    cancel = asyncio.Event()

    async def generate() -> AsyncIterator[str]:
        stream = orchestrator.respond(conversation_id, body.content, body.mode, body.max_context_tokens, cancel=cancel)
        try:
            async for event in stream:
                yield event.to_sse()
                if await request.is_disconnected():
                    cancel.set()
        except asyncio.CancelledError:
            cancel.set()
            raise
        finally:
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass

    return sse_response(generate())
