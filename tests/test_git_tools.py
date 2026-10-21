from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from develop.git_tools import GitError, GitRepo, find_git_root, git_available

pytestmark = pytest.mark.skipif(not git_available(), reason="git absent")

# Environnement du processus (PATH réel, SYSTEMROOT indispensable sous Windows) ; la configuration git de
# l'utilisateur est neutralisée (fichiers global et système ignorés) pour des dépôts de test reproductibles.
_GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
                GIT_COMMITTER_EMAIL="t@x", GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=dict(_GIT_ENV, HOME=str(repo)))


def _write(path: Path, content: str) -> None:
    """Écrit ``content`` tel quel : ``write_text`` traduirait les fins de ligne LF en CRLF sous Windows, et le
    blob (donc ``show_file``) ne serait plus celui attendu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path / ".gitignore", "ignored/\n*.log\n")
    (tmp_path / "src").mkdir()
    _write(tmp_path / "src" / "a.py", "print('a')\n")
    _write(tmp_path / "src" / "b.py", "print('b')\n")
    (tmp_path / "ignored").mkdir()
    _write(tmp_path / "ignored" / "junk.txt", "x")
    _write(tmp_path / "debug.log", "x")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_find_git_root(repo: Path):
    assert find_git_root(str(repo / "src" / "a.py")) == repo.resolve()
    assert find_git_root("/") is None


def test_not_a_repo(tmp_path: Path):
    with pytest.raises(GitError):
        GitRepo(str(tmp_path))


def test_ls_files_respects_gitignore_and_includes_untracked(repo: Path):
    _write(repo / "src" / "new.py", "print('new')\n")
    files = GitRepo(str(repo)).ls_files()
    assert "src/a.py" in files and "src/new.py" in files
    assert "ignored/junk.txt" not in files and "debug.log" not in files


def test_diff_names_working_tree(repo: Path):
    _write(repo / "src" / "a.py", "print('A')\n")
    (repo / "src" / "b.py").unlink()
    _write(repo / "src" / "c.py", "print('c')\n")
    changes = {c.path: c.status for c in GitRepo(str(repo)).diff_names()}
    assert changes == {"src/a.py": "M", "src/b.py": "D", "src/c.py": "?"}


def test_diff_names_range_and_rename(repo: Path):
    g = GitRepo(str(repo))
    _git(repo, "mv", "src/a.py", "src/renamed.py")
    _write(repo / "src" / "b.py", "print('bb')\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")
    changes = g.diff_names("HEAD~1..HEAD")
    by_path = {c.path: c for c in changes}
    assert by_path["src/renamed.py"].status == "R" and by_path["src/renamed.py"].old_path == "src/a.py"
    assert by_path["src/b.py"].status == "M"
    assert "print('bb')" in g.diff_text("HEAD~1..HEAD", path="src/b.py")


def test_status_reports_staged_rename_with_new_path_first(repo: Path):
    _git(repo, "mv", "src/a.py", "src/renamed.py")
    changes = [c for c in GitRepo(str(repo)).status() if c.status == "R"]
    assert changes == [__import__("develop.git_tools", fromlist=["ChangedFile"]).ChangedFile(path="src/renamed.py", status="R", old_path="src/a.py")]


def test_log_and_show(repo: Path):
    g = GitRepo(str(repo))
    commits = g.log()
    assert len(commits) == 1 and commits[0].subject == "init" and commits[0].author == "t"
    assert g.show_file("src/a.py") == "print('a')\n"
    assert g.show_file("src/missing.py") is None
    assert g.current_branch() == "main"


def test_empty_repo_without_commits(tmp_path: Path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path / "x.txt", "x")
    g = GitRepo(str(tmp_path))
    assert g.head() is None and g.log() == []
    assert [c.status for c in g.diff_names()] == ["?"]
    assert g.ls_files() == ["x.txt"]
