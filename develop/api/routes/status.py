"""``GET /api/status`` : état complet du serveur (espace actif, moteur, index, mode hors ligne)."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

from develop.api.routes import get_state

router = APIRouter(prefix="/api", tags=["status"])


@router.get("/status")
async def get_status(request: Request) -> Dict[str, Any]:
    """``{version, settings, active_workspace, workspace, engine, index, offline, needs_setup}``."""
    return await get_state(request).status_dict()
