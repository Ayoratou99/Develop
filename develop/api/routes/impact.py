"""``POST /api/impact`` : analyse d'impact déterministe depuis Git, un fichier ou un symbole.

:func:`compute_impact` est aussi utilisée par la CLI (``develop impact``). Sans ``repo_id`` :

* ``git`` analyse chaque dépôt activé qui est un dépôt Git et fusionne les rapports ;
* ``file`` retrouve le fichier par son chemin (exact, puis suffixe) dans tous les dépôts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from develop.api.routes import api_error, get_state, require_workspace
from develop.api.state import AppState
from develop.git_tools import GitError, find_git_root
from develop.impact_analysis import MAX_HOPS, ImpactReport

router = APIRouter(prefix="/api", tags=["impact"])

SOURCES = ("git", "file", "symbol")


class ImpactRequest(BaseModel):
    source: str = "git"
    repo_id: Optional[str] = None
    rev_range: Optional[str] = None
    path: Optional[str] = None
    symbol_id: Optional[Union[int, str]] = None
    max_hops: int = Field(default=3, ge=1, le=MAX_HOPS)


def _merge(reports: List[ImpactReport]) -> ImpactReport:
    if len(reports) == 1:
        return reports[0]
    merged = ImpactReport(source={"source": "git", "repos": [r.source.get("repo_id") for r in reports]})
    seen: set = set()
    for r in reports:
        merged.changed.extend(r.changed)
        for attr in ("affected", "endpoints", "models", "frontend_calls", "documents", "notes"):
            for item in getattr(r, attr):
                if attr == "affected":
                    if item.node in seen:
                        continue
                    seen.add(item.node)
                getattr(merged, attr).append(item)
        merged.warnings.extend(r.warnings)
        merged.stale_index = merged.stale_index or r.stale_index
        merged.stale_files.extend(r.stale_files)
    merged.affected.sort(key=lambda i: (-i.confidence, i.hops, i.name))
    merged.stats = {"affected_total": len(merged.affected), "changed_files": len(merged.changed),
                    "repos": len(reports)}
    return merged


def _find_file(state: AppState, path: str, repo_id: Optional[str]) -> Optional[Dict[str, Any]]:
    db = state.db
    assert db is not None
    rel = path.replace("\\", "/").strip().strip("/")
    params: List[Any] = [rel]
    where = "path = ?"
    if repo_id:
        where += " AND repo_id = ?"
        params.append(repo_id)
    rows = db.query("SELECT id, repo_id, path FROM files WHERE %s ORDER BY id LIMIT 2" % where, params)
    if not rows:
        params = ["%/" + rel.replace("%", "\\%").replace("_", "\\_")]
        where = "path LIKE ? ESCAPE '\\'"
        if repo_id:
            where += " AND repo_id = ?"
            params.append(repo_id)
        rows = db.query("SELECT id, repo_id, path FROM files WHERE %s ORDER BY length(path), id LIMIT 2" % where, params)
    if not rows:
        return None
    return dict(rows[0])


def compute_impact(state: AppState, source: str, repo_id: Optional[str] = None, rev_range: Optional[str] = None,
                   path: Optional[str] = None, symbol_id: Optional[Union[int, str]] = None,
                   max_hops: int = 3) -> Dict[str, Any]:
    """Rapport d'impact (``ImpactReport.to_dict()`` + ``request``) ; lève ``HTTPException`` sur entrée invalide."""
    ws = require_workspace(state)
    analyzer = state.impact
    assert analyzer is not None
    if source not in SOURCES:
        raise api_error(422, "validation_error", "« source » doit valoir %s." % ", ".join(SOURCES))
    if repo_id and ws.repo_by_id(repo_id) is None:
        raise api_error(404, "repo_not_found", "Dépôt « %s » inconnu dans l'espace actif." % repo_id)
    hops = max(1, min(int(max_hops or 3), MAX_HOPS))
    request_echo = {"source": source, "repo_id": repo_id, "rev_range": rev_range, "path": path,
                    "symbol_id": symbol_id, "max_hops": hops}

    if source == "git":
        targets = [ws.repo_by_id(repo_id)] if repo_id else [r for r in ws.repos if r.enabled]
        reports: List[ImpactReport] = []
        skipped: List[str] = []
        for repo in targets:
            if repo is None:
                continue
            if not repo_id and find_git_root(repo.path) is None:
                skipped.append(repo.name or repo.id)
                continue
            try:
                reports.append(analyzer.from_git(repo.id, rev_range=rev_range or None, max_hops=hops))
            except GitError as exc:
                if repo_id:
                    raise api_error(400, "git_error", "Git a échoué sur « %s » : %s" % (repo.name or repo.id, exc))
                skipped.append("%s (%s)" % (repo.name or repo.id, exc))
        if not reports:
            raise api_error(400, "not_a_git_repo",
                            "Aucun dépôt Git analysable%s. Indiquez un dépôt Git (repo_id) ou utilisez source=file / symbol."
                            % (" : " + ", ".join(skipped) if skipped else ""))
        report = _merge(reports)
        if skipped:
            report.warnings.append("Dépôts ignorés (pas un dépôt Git) : %s." % ", ".join(skipped))
    elif source == "file":
        if not path or not str(path).strip():
            raise api_error(422, "validation_error", "« path » (chemin relatif au dépôt) est obligatoire pour source=file.")
        found = _find_file(state, str(path), repo_id)
        if found is None:
            raise api_error(404, "file_not_found", "Fichier « %s » introuvable dans l'index%s."
                            % (path, " du dépôt " + repo_id if repo_id else ""))
        report = analyzer.from_file(found["repo_id"], found["path"], max_hops=hops)
        request_echo["repo_id"] = found["repo_id"]
        request_echo["path"] = found["path"]
    else:
        try:
            sid = int(symbol_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise api_error(422, "validation_error", "« symbol_id » (entier) est obligatoire pour source=symbol.")
        assert state.db is not None
        if state.db.get_symbol(sid) is None:
            raise api_error(404, "symbol_not_found", "Symbole %s introuvable dans l'index." % sid)
        report = analyzer.from_symbol(sid, max_hops=hops)
    data = report.to_dict()
    data["request"] = request_echo
    return data


@router.post("/impact")
def impact(body: ImpactRequest, request: Request) -> Dict[str, Any]:
    """``{source:'git'|'file'|'symbol', repo_id?, rev_range?, path?, symbol_id?, max_hops?}`` → ``ImpactReport``."""
    return compute_impact(get_state(request), body.source, body.repo_id, body.rev_range, body.path, body.symbol_id,
                          body.max_hops)
