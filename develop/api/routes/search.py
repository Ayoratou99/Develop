"""``GET /api/search`` : recherche hybride multi-canaux → ``{query, identifiers, results}``.

``SearchResult.to_dict()`` renvoie ``{query, hits, …}`` ; cette route l'adapte au contrat
(``results`` plutôt que ``hits``, ``identifiers`` remontés au premier niveau) et retire le texte
complet des morceaux (``text``) qui ne sert qu'au chat.

``identifiers`` est la liste de **tout** ce que l'analyse de la question a reconnu et que le canal
« identifier » cherche à l'exact : identifiants nommés (``query.identifiers``), routes
(``query.routes``), tables / modèles (``query.tables``) et chemins (``query.paths``), dans cet ordre,
sans doublon — c'est ce que l'explorateur et la CLI affichent sous « Identifiants repérés ».
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Request

from develop.api.routes import api_error, get_state, require_workspace
from develop.search import HIT_SOURCE_TYPES, SearchResult

router = APIRouter(prefix="/api", tags=["search"])

# Listes de ``QueryAnalysis`` reprises dans ``identifiers``, dans l'ordre d'affichage.
_RECOGNIZED_LISTS = ("identifiers", "routes", "tables", "paths")


def recognized_identifiers(query: Any) -> List[str]:
    """Identifiants, routes, tables et chemins repérés dans la question, dans cet ordre et sans doublon
    (comparaison exacte : « orders » (table) et « Order » (modèle) sont deux mentions distinctes)."""
    out: List[str] = []
    seen = set()
    for attr in _RECOGNIZED_LISTS:
        for value in getattr(query, attr, None) or []:
            text = str(value).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
    return out


def search_result_dict(result: SearchResult, limit: Optional[int] = None) -> Dict[str, Any]:
    """Forme de ``docs/API.md`` : ``{query, identifiers, results:[SearchHit], channel_counts, semantic_available, warnings}``."""
    hits = result.hits if limit is None else result.hits[:limit]
    results: List[Dict[str, Any]] = []
    for h in hits:
        d = h.to_dict()
        d.pop("text", None)
        results.append(d)
    query = result.query.to_dict() if hasattr(result.query, "to_dict") else {"text": str(result.query)}
    return {
        "query": query,
        "identifiers": recognized_identifiers(result.query),
        "results": results,
        "channel_counts": dict(result.channel_counts),
        "semantic_available": bool(result.semantic_available),
        "warnings": list(result.warnings),
    }


@router.get("/search")
async def search(request: Request, q: str = Query(default=""), types: Optional[str] = Query(default=None),
                 limit: int = Query(default=20, ge=1, le=200), date_from: Optional[str] = Query(default=None, alias="from"),
                 date_to: Optional[str] = Query(default=None, alias="to")) -> Dict[str, Any]:
    state = get_state(request)
    require_workspace(state)
    engine = state.search_engine
    assert engine is not None
    text = (q or "").strip()
    if not text:
        raise api_error(422, "validation_error", "Le paramètre « q » (question ou identifiant) est obligatoire.")
    source_types: Optional[List[str]] = None
    if types:
        source_types = [t.strip() for t in types.split(",") if t.strip()]
        unknown = [t for t in source_types if t not in HIT_SOURCE_TYPES]
        if unknown:
            raise api_error(422, "validation_error", "Types inconnus : %s (attendus : %s)."
                            % (", ".join(unknown), ", ".join(HIT_SOURCE_TYPES)))
    result = await engine.search_async(text, limit=limit, source_types=source_types, date_from=date_from, date_to=date_to)
    return search_result_dict(result)
