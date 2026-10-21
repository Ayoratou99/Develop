"""Espaces de travail, découverte de dépôts et sélecteur de dossiers côté serveur.

* ``/api/workspaces`` : liste, création (+ activation), lecture, mise à jour partielle
  (``name, repos[], llm{}, mode, allowed_hosts[]``), suppression de la mémoire Develop (jamais des
  dépôts), activation. Le ``name`` est nettoyé (espaces de bord retirés) et doit rester non vide,
  à la création comme à la mise à jour (422 ``validation_error`` sinon) ;
* ``POST /api/repos/discover`` ``{path}`` : pile de la racine + sous-projets détectés ;
* ``POST /api/fs/browse`` ``{path?}`` : dossiers seulement, cachés filtrés, liens symboliques non
  suivis ; sans ``path``, les racines (dossier personnel, volumes macOS, lecteurs Windows). Outil
  local : tout le disque lisible est navigable.
"""
from __future__ import annotations

import asyncio
import dataclasses
import os
import stat as stat_module
import string
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Request
from pydantic import BaseModel, Field, ValidationError

from develop import config
from develop.api.routes import api_error, get_state
from develop.api.state import MODEL_INFO_TIMEOUT_S, AppState
from develop.config import LLMConfig, RepoConfig, WorkspaceConfig
from develop.project_scanner import PROJECT_MARKER_NAMES, detect_stack, discover_subprojects

router = APIRouter(prefix="/api", tags=["workspaces"])

UPDATABLE_FIELDS = ("name", "repos", "llm", "mode", "allowed_hosts", "language")
MAX_BROWSE_ENTRIES = 500


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class PathBody(BaseModel):
    path: str = Field(default="")


def _ws_dict(ws: WorkspaceConfig) -> Dict[str, Any]:
    return ws.model_dump()


def _load_or_404(workspace_id: str) -> WorkspaceConfig:
    ws = config.load_workspace(workspace_id)
    if ws is None:
        raise api_error(404, "workspace_not_found", "Espace de travail « %s » introuvable." % workspace_id)
    return ws


def _clean_name(value: Any) -> str:
    """Nom d'espace nettoyé ; 422 s'il n'est pas un texte non vide (espaces seuls compris)."""
    if not isinstance(value, str) or not value.strip():
        raise api_error(422, "validation_error", "« name » doit être un texte non vide.")
    return value.strip()


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        parts.append("%s : %s" % (loc or "corps", err.get("msg", "invalide")))
    return "Configuration invalide — " + " ; ".join(parts)


def _schedule_model_info(state: AppState) -> None:
    """Rafraîchit ``model_info`` en arrière-plan (jamais bloquant pour la requête)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(state.refresh_model_info(MODEL_INFO_TIMEOUT_S))


# --------------------------------------------------------------------------- espaces

@router.get("/workspaces")
def list_workspaces() -> List[Dict[str, Any]]:
    return [_ws_dict(ws) for ws in config.list_workspaces()]


@router.post("/workspaces", status_code=201)
async def create_workspace(body: WorkspaceCreate, request: Request) -> Dict[str, Any]:
    """Crée l'espace et le rend actif. Un nom fait d'espaces seuls est refusé (422) avant toute création."""
    state = get_state(request)
    ws = config.create_workspace(_clean_name(body.name))
    state.activate_workspace(ws.id)
    return _ws_dict(ws)


@router.get("/workspaces/{workspace_id}")
def get_workspace(workspace_id: str) -> Dict[str, Any]:
    return _ws_dict(_load_or_404(workspace_id))


@router.put("/workspaces/{workspace_id}")
async def update_workspace(workspace_id: str, request: Request,
                           body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Mise à jour partielle ; si l'espace est actif, la garde hors ligne et le client LLM sont reconfigurés."""
    state = get_state(request)
    ws = _load_or_404(workspace_id)
    if not isinstance(body, dict):
        raise api_error(422, "validation_error", "Le corps doit être un objet JSON.")
    unknown = sorted(k for k in body if k not in UPDATABLE_FIELDS)
    if unknown:
        raise api_error(422, "validation_error", "Champs inconnus : %s (acceptés : %s)."
                        % (", ".join(unknown), ", ".join(UPDATABLE_FIELDS)))
    data = ws.model_dump()
    for key in UPDATABLE_FIELDS:
        if key not in body:
            continue
        value = body[key]
        if key == "llm":
            if not isinstance(value, dict):
                raise api_error(422, "validation_error", "« llm » doit être un objet.")
            merged = dict(data["llm"])
            merged.update(value)
            data["llm"] = merged
        elif key == "repos":
            if not isinstance(value, list):
                raise api_error(422, "validation_error", "« repos » doit être une liste de dépôts.")
            data["repos"] = value
        elif key == "name":
            data["name"] = _clean_name(value)
        else:
            data[key] = value
    try:
        updated = WorkspaceConfig.model_validate(data)
    except ValidationError as exc:
        raise api_error(422, "validation_error", _validation_message(exc))
    seen: set = set()
    for repo in updated.repos:
        if repo.id in seen:
            raise api_error(422, "validation_error", "Identifiant de dépôt en double : %s." % repo.id)
        seen.add(repo.id)
    updated.id = ws.id
    updated.created_at = ws.created_at
    config.save_workspace(updated)
    if state.workspace is not None and state.workspace.id == updated.id:
        state.reconfigure(updated)
        _schedule_model_info(state)
    return _ws_dict(updated)


@router.delete("/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str, request: Request) -> Dict[str, Any]:
    """Supprime la mémoire Develop de l'espace (index, notes, documents, conversations). Jamais les dépôts."""
    state = get_state(request)
    _load_or_404(workspace_id)
    was_active = state.workspace is not None and state.workspace.id == workspace_id
    if was_active:
        if state.index_running:
            raise api_error(409, "index_running", "Une indexation est en cours : attendez sa fin ou annulez-la.")
        state.detach()  # ferme memory.sqlite3 (et son -wal) : indispensable sous Windows avant de supprimer
    try:
        config.delete_workspace(workspace_id)
    except OSError as exc:
        raise api_error(409, "workspace_locked",
                        "Impossible de supprimer la mémoire de l'espace « %s » : un fichier est encore utilisé "
                        "(%s). Fermez les autres programmes qui l'ouvrent, puis réessayez." % (workspace_id, exc))
    state.settings = config.load_settings()
    if was_active:
        state.load_workspace(None)
    return {"deleted": True, "id": workspace_id, "active_workspace": state.settings.active_workspace}


@router.post("/workspaces/{workspace_id}/activate")
async def activate_workspace(workspace_id: str, request: Request) -> Dict[str, Any]:
    state = get_state(request)
    _load_or_404(workspace_id)
    if state.index_running:
        raise api_error(409, "index_running", "Une indexation est en cours : attendez sa fin avant de changer d'espace.")
    ws = state.activate_workspace(workspace_id)
    _schedule_model_info(state)
    return _ws_dict(ws)


# --------------------------------------------------------------------------- dépôts

@router.post("/repos/discover")
def discover_repos(body: PathBody) -> Dict[str, Any]:
    """``{root: {...stack}, subprojects: [{path, name, suggested_kind, frameworks}]}``."""
    path = os.path.abspath(os.path.expanduser(body.path.strip()))
    if not body.path.strip() or not os.path.isdir(path):
        raise api_error(400, "not_a_directory", "« %s » n'est pas un dossier accessible." % (body.path or "?"))
    root = detect_stack(path, [])
    subprojects = discover_subprojects(path)
    return {
        "path": path,
        "root": dict(dataclasses.asdict(root), path=path, kind=root.suggested_kind),
        "subprojects": [dataclasses.asdict(sp) for sp in subprojects],
    }


# --------------------------------------------------------------------------- dossiers

def _is_git_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, ".git"))


def _has_marker(path: str, names: Optional[List[str]] = None) -> bool:
    try:
        names = names if names is not None else os.listdir(path)
    except OSError:
        return False
    return any(n in PROJECT_MARKER_NAMES for n in names)


def _entry(path: str, name: Optional[str] = None) -> Dict[str, Any]:
    return {"name": name or os.path.basename(path.rstrip("/\\")) or path, "path": path,
            "is_git": _is_git_dir(path), "has_marker": _has_marker(path)}


def _windows_drives() -> List[str]:
    """Lecteurs présents (``C:\\``, ``D:\\``…) : ``os.listdrives`` (Python ≥ 3.12, instantané) ou, à défaut,
    les lettres dont la racine existe. Un lecteur réseau déconnecté peut faire attendre ``exists`` : tolérable."""
    listdrives = getattr(os, "listdrives", None)
    if listdrives is not None:
        try:
            return sorted(str(d) for d in listdrives())
        except OSError:
            pass
    drives: List[str] = []
    for letter in string.ascii_uppercase:
        drive = "%s:\\" % letter
        try:
            if os.path.exists(drive):
                drives.append(drive)
        except OSError:
            continue
    return drives


def _windows_hidden(path: str) -> bool:
    """Dossier caché ou système sous Windows (équivalent du « . » initial ailleurs) ; faux en cas de doute."""
    try:
        attrs = getattr(os.stat(path, follow_symlinks=False), "st_file_attributes", 0)
    except OSError:
        return False
    hidden = getattr(stat_module, "FILE_ATTRIBUTE_HIDDEN", 0x2) | getattr(stat_module, "FILE_ATTRIBUTE_SYSTEM", 0x4)
    return bool(attrs & hidden)


def browse_roots() -> List[Dict[str, Any]]:
    """Racines proposées : dossier personnel, volumes (macOS), lecteurs (Windows), ``/`` (Linux)."""
    roots: List[Dict[str, Any]] = []
    home = str(Path.home())
    roots.append(_entry(home, "Dossier personnel (%s)" % os.path.basename(home)))
    if sys.platform == "darwin":
        volumes = "/Volumes"
        try:
            for name in sorted(os.listdir(volumes), key=str.lower):
                full = os.path.join(volumes, name)
                if name.startswith(".") or not os.path.isdir(full):
                    continue
                roots.append(_entry(full, name))
        except OSError:
            pass
    elif sys.platform.startswith("win"):
        for drive in _windows_drives():
            roots.append(_entry(drive, drive))
    else:
        roots.append(_entry("/", "/"))
    return roots


@router.post("/fs/browse")
def browse(body: PathBody = Body(default_factory=PathBody)) -> Dict[str, Any]:
    """``{path, parent, entries:[{name, path, is_git, has_marker}]}`` ; dossiers uniquement."""
    raw = (body.path or "").strip()
    if not raw:
        return {"path": "", "parent": None, "entries": browse_roots(), "roots": True}
    path = os.path.abspath(os.path.expanduser(raw))
    if not os.path.isdir(path):
        raise api_error(404, "not_a_directory", "« %s » n'est pas un dossier accessible." % raw)
    try:
        names = sorted(os.listdir(path), key=str.lower)
    except PermissionError:
        raise api_error(403, "permission_denied", "Lecture refusée pour « %s »." % path)
    except OSError as exc:
        raise api_error(400, "unreadable_directory", "Impossible de lire « %s » : %s." % (path, exc))
    entries: List[Dict[str, Any]] = []
    truncated = False
    on_windows = sys.platform.startswith("win")
    for name in names:
        if name.startswith("."):
            continue
        full = os.path.join(path, name)
        try:
            if os.path.islink(full) or not os.path.isdir(full):
                continue
            if on_windows and _windows_hidden(full):
                continue  # $RECYCLE.BIN, System Volume Information, AppData… : cachés comme les « .x » ailleurs
        except OSError:
            continue
        if len(entries) >= MAX_BROWSE_ENTRIES:
            truncated = True
            break
        entries.append(_entry(full, name))
    parent = os.path.dirname(path)
    if parent == path:
        parent = None  # racine (« / », « C:\ », partage UNC)
    return {"path": path, "parent": parent, "entries": entries, "truncated": truncated,
            "is_git": _is_git_dir(path), "has_marker": _has_marker(path, names)}
