"""Machine, moteurs détectés et conseiller de modèles (``/api/system/*``)."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Query, Request

from develop.api.routes import get_state
from develop.llm import advisor
from develop.system.profile import describe

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/profile")
def system_profile(request: Request) -> Dict[str, Any]:
    """``MachineProfile.to_dict()`` + ``summary`` (une ligne)."""
    profile = get_state(request).profile()
    data = profile.to_dict()
    data["summary"] = describe(profile)
    return data


@router.get("/engines")
async def system_engines(request: Request, refresh: bool = Query(default=False)) -> Dict[str, Any]:
    """``EnginesReport.to_dict()`` (cache 30 s ; ``?refresh=true`` force une nouvelle sonde)."""
    report = await get_state(request).engines_report(force=refresh)
    return report.to_dict()


@router.get("/recommendations")
async def system_recommendations(request: Request, num_ctx: int = Query(default=16384, ge=512)) -> Dict[str, Any]:
    """Conseils de modèles (``advisor.advise``) : recommandations, embeddings, étapes d'installation."""
    state = get_state(request)
    profile = state.profile()
    engines = await state.engines_report()
    installed: List[Any] = list(engines.ollama.models)
    for probe in engines.openai_servers:
        installed.extend(probe.models)
    ws = state.workspace
    if ws is not None and ws.llm.num_ctx:
        num_ctx = ws.llm.num_ctx if num_ctx == 16384 else num_ctx
    advice = advisor.advise(profile, engines, installed_models=installed, num_ctx=num_ctx)
    data = advice.to_dict()
    data["profile_summary"] = data.get("profile_summary") or describe(profile)
    return data
