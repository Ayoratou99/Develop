"""Routes de l'API locale (une par groupe de ``docs/API.md``) et aides communes.

Toutes les erreurs passent par :func:`api_error`, qui produit une ``HTTPException`` dont le
``detail`` est ``{"code", "message"}`` ; les gestionnaires de ``develop.api.app`` le rendent en
``{"error": {"code", "message"}}``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request

from develop.api.state import AppState, NoWorkspace
from develop.config import WorkspaceConfig
from develop.storage import Database, FileRow


def api_error(status: int, code: str, message: str) -> HTTPException:
    """Erreur HTTP au format du contrat (message en français, actionnable)."""
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def get_state(request: Request) -> AppState:
    return request.app.state.app_state


def require_workspace(state: AppState) -> WorkspaceConfig:
    try:
        return state.require_workspace()
    except NoWorkspace as exc:
        raise api_error(400, "no_workspace", str(exc))


def require_db(state: AppState) -> Database:
    require_workspace(state)
    assert state.db is not None
    return state.db


def parse_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise api_error(422, "validation_error", "Le paramètre « %s » doit être un entier." % name)


def file_ref(f: Optional[FileRow]) -> Optional[Dict[str, Any]]:
    """``{id, path, repo_id, language}`` tel qu'attendu par l'interface (``fileRef``)."""
    if f is None:
        return None
    return {"id": f.id, "path": f.path, "repo_id": f.repo_id, "language": f.language}


def repo_root(state: AppState, repo_id: Optional[str]) -> Optional[str]:
    """Racine absolue d'un dépôt (configuration, sinon table ``repos``)."""
    if not repo_id:
        return None
    ws = state.workspace
    repo = ws.repo_by_id(repo_id) if ws is not None else None
    path = repo.path if repo is not None else None
    if not path and state.db is not None:
        row = state.db.get_repo(repo_id)
        path = row.path if row is not None else None
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(path))


def all_routers() -> List[APIRouter]:
    """Routeurs dans l'ordre d'enregistrement (import tardif : évite les cycles)."""
    from develop.api.routes import (chat, documents, explorer, git, impact, index, llm, notes, search, status,
                                    system, workspaces)

    return [status.router, workspaces.router, system.router, llm.router, index.router, explorer.router,
            search.router, documents.router, notes.router, chat.router, impact.router, git.router]
