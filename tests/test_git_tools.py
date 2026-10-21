from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from develop.git_tools import GitError, GitRepo, find_git_root, git_available

pytestmark = pytest.mark.skipif(not git_available(), reason="git absent")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@x", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(repo)})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / ".gitignore").write_text("ignored/\n*.log\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('a')\n")
    (tmp_path / "src" / "b.py").write_text("print('b')\n")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "junk.txt").write_text("x")
    (tmp_path / "debug.log").write_text("x")
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
    (repo / "src" / "new.py").write_text("print('new')\n")
    files = GitRepo(str(repo)).ls_files()
    assert "src/a.py" in files and "src/new.py" in files
    assert "ignored/junk.txt" not in files and "debug.log" not in files


def test_diff_names_working_tree(repo: Path):
    (repo / "src" / "a.py").write_text("print('A')\n")
    (repo / "src" / "b.py").unlink()
    (repo / "src" / "c.py").write_text("print('c')\n")
    changes = {c.path: c.status for c in GitRepo(str(repo)).diff_names()}
    assert changes == {"src/a.py": "M", "src/b.py": "D", "src/c.py": "?"}


def test_diff_names_range_and_rename(repo: Path):
    g = GitRepo(str(repo))
    _git(repo, "mv", "src/a.py", "src/renamed.py")
    (repo / "src" / "b.py").write_text("print('bb')\n")
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
    (tmp_path / "x.txt").write_text("x")
    g = GitRepo(str(tmp_path))
    assert g.head() is None and g.log() == []
    assert [c.status for c in g.diff_names()] == ["?"]
    assert g.ls_files() == ["x.txt"]
