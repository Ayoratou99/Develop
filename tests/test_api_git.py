"""Tests de ``/api/git/*`` quand les dépôts configurés sont des sous-dossiers d'un même dépôt Git (monorepo).

Non-régression du module ``develop.api.routes.git`` : ``status`` et ``log`` parlent en chemins relatifs
au dépôt configuré (docs/API.md §8) et ne débordent jamais sur les dépôts voisins, comme
``/api/repos[].git.dirty`` et ``ImpactAnalyzer.from_git``. Copie de ``sample_project`` dans un dossier
temporaire (``git init`` à la racine du monorepo) ; rien n'est écrit dans la fixture ni dans ``~/.develop``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from develop.git_tools import git_available

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"
SERVICE = "app/Services/OrderService.php"  # chemin tel que renvoyé par /api/files pour le dépôt backend
MONEY = "src/utils/money.ts"  # idem pour le dépôt frontend

pytestmark = pytest.mark.skipif(not git_available(), reason="git absent")


def _git(root: Path, *args: str) -> None:
    env = dict(os.environ, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, env=env)


def _append(path: Path, text: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    base = tmp_path_factory.mktemp("git-api")
    home = base / "home"
    home.mkdir()
    project = base / "sample_project"
    shutil.copytree(FIXTURE, project)
    _git(project, "init", "-q", "-b", "main")
    _git(project, "add", "-A")
    _git(project, "commit", "-q", "-m", "Import initial")
    # un second commit qui ne touche que le frontend : absent de l'historique du dépôt backend
    _append(project / "frontend" / MONEY, "\n// arrondi\n")
    _git(project, "add", "-A")
    _git(project, "commit", "-q", "-m", "Arrondi des montants")
    previous = os.environ.get("DEVELOP_HOME")
    os.environ["DEVELOP_HOME"] = str(home)
    from develop.api import create_app

    app = create_app(llm_transport=httpx.MockTransport(lambda request: httpx.Response(404)))
    with TestClient(app) as client:
        ws = client.post("/api/workspaces", json={"name": "Monorepo"}).json()
        r = client.put("/api/workspaces/%s" % ws["id"], json={"repos": [
            {"name": "backend", "path": str(project / "backend"), "kind": "backend"},
            {"name": "frontend", "path": str(project / "frontend"), "kind": "frontend"},
        ]})
        assert r.status_code == 200, r.text
        ids = {repo["name"]: repo["id"] for repo in r.json()["repos"]}
        yield SimpleNamespace(client=client, project=project, ids=ids)
    if previous is None:
        os.environ.pop("DEVELOP_HOME", None)
    else:
        os.environ["DEVELOP_HOME"] = previous


def test_status_is_scoped_to_configured_repo(env: SimpleNamespace):
    """Les changements d'un dépôt voisin n'apparaissent pas ; les chemins n'ont pas le préfixe du sous-dossier."""
    c, project = env.client, env.project
    backend, frontend = env.ids["backend"], env.ids["frontend"]
    for repo_id in (backend, frontend):
        status = c.get("/api/git/status", params={"repo_id": repo_id}).json()
        assert status["changes"] == [] and status["branch"] == "main" and status["head"]
    try:
        _append(project / "backend" / SERVICE, "\n// changement\n")
        _append(project / "frontend" / MONEY, "\n// changement\n")
        (project / "frontend" / "src" / "nouveau.ts").write_text("export const x = 1;\n", encoding="utf-8")

        status = c.get("/api/git/status", params={"repo_id": backend}).json()
        assert status["changes"] == [{"path": SERVICE, "status": "M", "old_path": None}]
        assert status["root"] == str(project.resolve()) and status["subdir"] == "backend"
        status = c.get("/api/git/status", params={"repo_id": frontend}).json()
        assert {ch["path"]: ch["status"] for ch in status["changes"]} == {MONEY: "M", "src/nouveau.ts": "?"}
        assert status["subdir"] == "frontend"

        # renommage indexé : les deux chemins (``path`` et ``old_path``) sont relatifs au dépôt configuré
        # (l'ordre ancien/nouveau est celui de ``GitRepo.status``, hors du périmètre de cette route)
        _git(project, "mv", "backend/" + SERVICE, "backend/app/Services/OrderSvc.php")
        status = c.get("/api/git/status", params={"repo_id": backend}).json()
        assert len(status["changes"]) == 1 and status["changes"][0]["status"] == "R"
        assert {status["changes"][0]["path"], status["changes"][0]["old_path"]} == {"app/Services/OrderSvc.php", SERVICE}

        # cohérence avec /api/repos[].git.dirty, déjà limité au dépôt configuré
        _git(project, "checkout", "--", "frontend")
        (project / "frontend" / "src" / "nouveau.ts").unlink()
        assert c.get("/api/git/status", params={"repo_id": frontend}).json()["changes"] == []
        repos = {x["name"]: x for x in c.get("/api/repos").json()}
        assert repos["backend"]["git"]["dirty"] is True and repos["frontend"]["git"]["dirty"] is False
    finally:
        _git(project, "reset", "-q", "--hard")
        _git(project, "clean", "-qfd")
    assert c.get("/api/git/status", params={"repo_id": backend}).json()["changes"] == []


def test_log_takes_paths_relative_to_configured_repo(env: SimpleNamespace):
    """``path`` est celui de ``/api/files`` ; sans ``path``, seuls les commits touchant le dépôt configuré."""
    c = env.client
    backend, frontend = env.ids["backend"], env.ids["frontend"]
    log = c.get("/api/git/log", params={"repo_id": backend, "path": SERVICE}).json()
    assert [x["subject"] for x in log] == ["Import initial"] and {"sha", "author", "date"} <= set(log[0])
    log = c.get("/api/git/log", params={"repo_id": frontend, "path": MONEY}).json()
    assert [x["subject"] for x in log] == ["Arrondi des montants", "Import initial"]
    assert [x["subject"] for x in c.get("/api/git/log", params={"repo_id": backend}).json()] == ["Import initial"]
    assert len(c.get("/api/git/log", params={"repo_id": frontend}).json()) == 2
    assert len(c.get("/api/git/log", params={"repo_id": frontend, "limit": 1}).json()) == 1
    assert c.get("/api/git/log", params={"repo_id": backend, "path": "app/Services/Absent.php"}).json() == []
    # un chemin qui sort du dépôt configuré est refusé (il atteindrait le dépôt voisin via la racine Git)
    for bad in ("../frontend/" + MONEY, "/etc/passwd", ".."):
        r = c.get("/api/git/log", params={"repo_id": backend, "path": bad})
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error", bad
