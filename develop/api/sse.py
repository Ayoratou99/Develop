"""Flux Server-Sent Events : format ``event: <type>\\ndata: <json>\\n\\n``, ping, déconnexion.

Le générateur :func:`event_stream` lit une file asyncio alimentée par :meth:`AppState.publish`
(depuis n'importe quel thread), envoie un commentaire ``: ping`` toutes les
:data:`PING_INTERVAL_S` secondes pour garder la connexion ouverte et s'arrête dès que le client
se déconnecte ou qu'une sentinelle ``None`` est lue.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Optional

from fastapi import Request
from fastapi.responses import StreamingResponse

PING_INTERVAL_S = 15.0
PING = ": ping\n\n"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def format_event(event: str, data: Any) -> str:
    """Un bloc SSE : ``event: <type>`` puis ``data: <json>`` et une ligne vide."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return "event: %s\ndata: %s\n\n" % (event, payload)


async def event_stream(request: Optional[Request], queue: "asyncio.Queue[Any]",
                       ping_interval: float = PING_INTERVAL_S) -> AsyncIterator[str]:
    """Produit les blocs SSE d'une file ``(type, data)`` ; ``None`` termine le flux."""
    while True:
        if request is not None and await request.is_disconnected():
            return
        try:
            item = await asyncio.wait_for(queue.get(), timeout=ping_interval)
        except asyncio.TimeoutError:
            yield PING
            continue
        if item is None:
            return
        event, data = item
        yield format_event(event, data)


def sse_response(generator: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(generator, media_type="text/event-stream", headers=dict(SSE_HEADERS))
