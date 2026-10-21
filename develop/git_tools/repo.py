"""Enveloppe minimale et sûre autour de ``git`` (lecture seule).

Aucune commande n'écrit dans le dépôt : uniquement ``rev-parse``, ``ls-files``,
``status``, ``diff``, ``log``, ``show``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


class GitError(RuntimeError):
    pass


# Sous Windows, git ne doit jamais ouvrir de console (fenêtre native « develop app ») ; sans effet ailleurs.
_NO_WINDOW = ({"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
              if hasattr(subprocess, "CREATE_NO_WINDOW") else {})


@dataclass(frozen=True)
class ChangedFile:
    path: str  # relatif à la racine du dépôt, séparateur "/"
    status: str  # "A" ajouté, "M" modifié, "D" supprimé, "R" renommé, "?" non suivi
    old_path: Optional[str] = None  # pour les renommages


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    author: str
    date: str  # ISO 8601
    subject: str


def git_available() -> bool:
    return shutil.which("git") is not None


def find_git_root(path: str) -> Optional[Path]:
    """Racine du dépôt contenant ``path``, ou None si ce n'est pas un dépôt Git."""
    if not git_available():
        return None
    p = Path(path).resolve()
    start = p if p.is_dir() else p.parent
    try:
        # Décodage UTF-8 explicite : sous Windows, text=True seul lirait le chemin en cp1252 (accents altérés).
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=10, check=False,
                             env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0"), **_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    top = out.stdout.strip()
    return Path(top) if top else None


class GitRepo:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        if not git_available():
            raise GitError("git n'est pas installé")
        if not (self.root / ".git").exists() and find_git_root(str(self.root)) != self.root:
            raise GitError("%s n'est pas la racine d'un dépôt Git" % self.root)

    # ----- bas niveau -----
    def _run(self, args: Sequence[str], timeout: float = 60.0) -> str:
        cmd = ["git", "-C", str(self.root), "-c", "core.quotepath=off", *args]
        # GIT_OPTIONAL_LOCKS=0 : ne jamais rafraîchir .git/index (lecture seule stricte, pas de verrou disputé avec l'IDE)
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C.UTF-8")
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False, env=env, **_NO_WINDOW)
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitError("git %s: %s" % (" ".join(args[:2]), exc)) from exc
        if out.returncode != 0:
            raise GitError("git %s a échoué: %s" % (" ".join(args[:2]), out.stderr.decode("utf-8", "replace").strip()))
        return out.stdout.decode("utf-8", "replace")

    # ----- requêtes -----
    def head(self) -> Optional[str]:
        try:
            return self._run(["rev-parse", "HEAD"]).strip() or None
        except GitError:
            return None  # dépôt sans commit

    def current_branch(self) -> Optional[str]:
        try:
            b = self._run(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
            return b if b and b != "HEAD" else None
        except GitError:
            return None

    def ls_files(self, include_untracked: bool = True) -> List[str]:
        """Fichiers suivis (+ non suivis non ignorés), chemins relatifs POSIX, triés et dédupliqués."""
        args = ["ls-files", "-z", "--cached"]
        if include_untracked:
            args += ["--others", "--exclude-standard"]
        raw = self._run(args)
        files = {p for p in raw.split("\0") if p}
        # ls-files liste aussi les fichiers supprimés mais encore indexés : on ne garde que ceux présents
        return sorted(f for f in files if (self.root / f).is_file())

    def status(self) -> List[ChangedFile]:
        """Changements du working tree + index par rapport à HEAD (porcelain v1, -z)."""
        raw = self._run(["status", "--porcelain=v1", "-z", "--untracked-files=all"])
        entries = raw.split("\0")
        out: List[ChangedFile] = []
        i = 0
        while i < len(entries):
            e = entries[i]
            i += 1
            if len(e) < 4:
                continue
            x, y, path = e[0], e[1], e[3:]
            if x == "R" or y == "R" or x == "C" or y == "C":
                # En mode -z, git imprime d'abord le NOUVEAU chemin puis l'ancien (« R  new\0old\0 »)
                new = path
                old = entries[i] if i < len(entries) else path
                i += 1
                out.append(ChangedFile(path=new, status="R", old_path=old))
                continue
            if x == "?" and y == "?":
                out.append(ChangedFile(path=path, status="?"))
                continue
            if "D" in (x, y):
                out.append(ChangedFile(path=path, status="D"))
            elif "A" in (x, y):
                out.append(ChangedFile(path=path, status="A"))
            else:
                out.append(ChangedFile(path=path, status="M"))
        return out

    def diff_names(self, rev_range: Optional[str] = None) -> List[ChangedFile]:
        """Fichiers changés : ``rev_range`` (ex: "HEAD~3..HEAD", "main...feature") ou working tree vs HEAD."""
        if rev_range:
            raw = self._run(["diff", "--name-status", "-z", "-M", rev_range])
        else:
            if self.head() is None:
                return self.status()
            raw = self._run(["diff", "--name-status", "-z", "-M", "HEAD"])
        parts = raw.split("\0")
        out: List[ChangedFile] = []
        i = 0
        while i < len(parts):
            st = parts[i]
            i += 1
            if not st:
                continue
            code = st[0]
            if code in ("R", "C"):
                old, new = parts[i], parts[i + 1]
                i += 2
                out.append(ChangedFile(path=new, status="R", old_path=old))
            else:
                path = parts[i]
                i += 1
                out.append(ChangedFile(path=path, status=code if code in ("A", "M", "D") else "M"))
        if not rev_range:
            # ajoute les fichiers non suivis (nouveaux, pas encore indexés)
            seen = {c.path for c in out}
            out += [c for c in self.status() if c.status == "?" and c.path not in seen]
        return out

    def diff_text(self, rev_range: Optional[str] = None, path: Optional[str] = None, max_chars: int = 200_000) -> str:
        args = ["diff", "--no-color", "-M"]
        if rev_range:
            args.append(rev_range)
        elif self.head() is not None:
            args.append("HEAD")
        if path:
            args += ["--", path]
        text = self._run(args)
        return text[:max_chars]

    def log(self, limit: int = 50, path: Optional[str] = None) -> List[CommitInfo]:
        if self.head() is None:
            return []
        args = ["log", "-n", str(limit), "--date=iso-strict", "--pretty=format:%H%x1f%an%x1f%ad%x1f%s%x1e"]
        if path:
            args += ["--", path]
        raw = self._run(args)
        commits: List[CommitInfo] = []
        for rec in raw.split("\x1e"):
            rec = rec.strip("\n")
            if not rec:
                continue
            fields = rec.split("\x1f")
            if len(fields) == 4:
                commits.append(CommitInfo(sha=fields[0], author=fields[1], date=fields[2], subject=fields[3]))
        return commits

    def show_file(self, path: str, rev: str = "HEAD", max_bytes: int = 2_000_000) -> Optional[str]:
        try:
            return self._run(["show", "%s:%s" % (rev, path)])[:max_bytes]
        except GitError:
            return None
