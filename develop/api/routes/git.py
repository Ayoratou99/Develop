"""Lecture seule de Git : ``/api/git/status`` et ``/api/git/log`` d'un dépôt de l'espace.

Un dépôt configuré peut être un sous-dossier d'un dépôt Git (monorepo : ``backend`` =
``<racine Git>/backend``). Les deux routes se limitent alors à ce sous-dossier et parlent en
chemins relatifs au dépôt configuré (docs/API.md §8), comme ``/api/files``, ``/api/repos[].git``
et l'analyse d'impact : ``changes[].path`` ne porte jamais le préfixe du sous-dossier, et le
paramètre ``path`` de ``/api/git/log`` est celui renvoyé par ``/api/files``.
"""
from __future__ import annotations

import dataclasses
import os
import posixpath
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Query, Request

from develop.api.routes import api_error, get_state, require_workspace
from develop.api.state import AppState
from develop.git_tools import ChangedFile, GitError, GitRepo, find_git_root, git_available
from develop.project_scanner import nfc

router = APIRouter(prefix="/api/git", tags=["git"])


def _git_repo(state: AppState, repo_id: Optional[str]) -> Tuple[GitRepo, str]:
    """Dépôt Git contenant le dépôt configuré et préfixe POSIX de celui-ci dans la racine Git.

    Le préfixe vaut ``"backend/"`` quand le dépôt configuré est un sous-dossier du dépôt Git,
    ``""`` quand il en est la racine.
    """
    ws = require_workspace(state)
    if not git_available():
        raise api_error(503, "git_unavailable", "Git n'est pas installé : les fonctions Git sont indisponibles.")
    if not repo_id:
        candidates = [r for r in ws.repos if r.enabled and find_git_root(r.path) is not None]
        if not candidates:
            raise api_error(400, "not_a_git_repo", "Aucun dépôt Git dans l'espace : indiquez repo_id.")
        repo = candidates[0]
    else:
        repo = ws.repo_by_id(repo_id)
        if repo is None:
            raise api_error(404, "repo_not_found", "Dépôt « %s » inconnu dans l'espace actif." % repo_id)
    if not os.path.isdir(repo.path):
        raise api_error(404, "repo_missing", "Le dossier du dépôt « %s » est introuvable : %s" % (repo.name, repo.path))
    root = find_git_root(repo.path)
    if root is None:
        raise api_error(400, "not_a_git_repo", "« %s » n'est pas dans un dépôt Git." % repo.path)
    try:
        git = GitRepo(str(root))
    except GitError as exc:
        raise api_error(400, "git_error", str(exc))
    return git, _subdir_prefix(git, repo.path)


def _subdir_prefix(git: GitRepo, repo_path: str) -> str:
    """``"backend/"`` si ``repo_path`` est un sous-dossier de ``git.root``, ``""`` s'il en est la racine.

    Forme brute (non normalisée) : c'est un pathspec Git et le préfixe des chemins que Git renvoie.
    """
    try:
        rel = Path(repo_path).resolve().relative_to(git.root).as_posix()
    except ValueError:
        return ""
    return "" if rel in ("", ".") else rel.rstrip("/") + "/"


def _within_repo(ch: ChangedFile, prefix: str) -> Optional[ChangedFile]:
    """Chemin Git (relatif à la racine Git) → relatif au dépôt configuré, en NFC ; ``None`` hors du dépôt.

    Même règle que l'analyse d'impact : Git et le chemin configuré peuvent écrire un nom accentué sous
    deux formes (NFD sur un disque macOS, NFC recomposé par git) ; tout est comparé et renvoyé en NFC,
    la forme sous laquelle le scanner indexe les fichiers. L'ancien chemin d'un renommage venu d'un
    autre sous-dossier est omis (le fichier est nouveau pour ce dépôt).
    """
    path = nfc(ch.path)
    old_path = nfc(ch.old_path) if ch.old_path else None
    prefix = nfc(prefix)
    if not prefix:
        return ChangedFile(path=path, status=ch.status, old_path=old_path)
    if not path.startswith(prefix):
        return None
    old = old_path[len(prefix):] if old_path and old_path.startswith(prefix) else None
    return ChangedFile(path=path[len(prefix):], status=ch.status, old_path=old)


def _log_pathspec(prefix: str, path: Optional[str]) -> Optional[str]:
    """Pathspec Git (relatif à la racine Git) pour ``path`` relatif au dépôt configuré.

    Sans ``path`` (ou ``"."``) : le sous-dossier du dépôt configuré, ou aucun pathspec à la racine
    Git. Un chemin absolu ou qui remonte hors du dépôt (``../frontend/…``) sortirait du dépôt
    configuré : 422 ``validation_error``.
    """
    rel = posixpath.normpath(path.strip()) if path and path.strip() else "."
    if rel == ".":
        return prefix.rstrip("/") or None
    if posixpath.isabs(rel) or rel == ".." or rel.startswith("../"):
        raise api_error(422, "validation_error",
                        "Le paramètre « path » doit être relatif à la racine du dépôt (tel que renvoyé par /api/files), "
                        "sans « .. » ni chemin absolu.")
    return prefix + rel


@router.get("/status")
def git_status(request: Request, repo_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    """``{repo_id, root, subdir, branch, head, changes:[{path, status, old_path}]}``.

    ``root`` est la racine Git, ``subdir`` le sous-dossier du dépôt configuré dans celle-ci (``""`` à la
    racine) ; ``changes`` ne contient que les fichiers de ce dépôt, en chemins relatifs à celui-ci.
    """
    git, prefix = _git_repo(get_state(request), repo_id)
    try:
        changes = [c for c in (_within_repo(ch, prefix) for ch in git.status()) if c is not None]
        return {"repo_id": repo_id, "root": str(git.root), "subdir": prefix.rstrip("/"), "branch": git.current_branch(),
                "head": git.head(), "changes": [dataclasses.asdict(c) for c in changes]}
    except GitError as exc:
        raise api_error(400, "git_error", str(exc))


@router.get("/log")
def git_log(request: Request, repo_id: Optional[str] = Query(default=None),
            limit: int = Query(default=50, ge=1, le=1000), path: Optional[str] = Query(default=None)) -> List[Dict[str, Any]]:
    """``[{sha, author, date, subject}]`` (les plus récents d'abord), limités au dépôt configuré.

    ``path`` est relatif au dépôt configuré (tel que renvoyé par ``/api/files``) ; sans ``path``, l'historique
    des commits qui touchent ce dépôt (tout le dépôt Git s'il en est la racine).
    """
    git, prefix = _git_repo(get_state(request), repo_id)
    pathspec = _log_pathspec(prefix, path)
    try:
        return [dataclasses.asdict(c) for c in git.log(limit=limit, path=pathspec)]
    except GitError as exc:
        raise api_error(400, "git_error", str(exc))
