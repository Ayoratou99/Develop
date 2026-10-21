"""Parcours en lecture seule d'un dépôt : liste des fichiers indexables.

Deux stratégies, toutes deux déterministes (résultat trié par chemin) :

* ``git ls-files`` quand ``root`` est la racine d'un dépôt Git (respecte ``.gitignore``,
  inclut les fichiers non suivis non ignorés) ;
* sinon ``os.scandir`` récursif, sans suivre les liens symboliques, avec un parseur
  ``.gitignore`` *simple* appliqué niveau par niveau (motifs de dossier ``x/``, globs
  ``*.ext``, motifs ancrés ``a/b`` ; les négations ``!x`` sont ignorées).

Dans les deux cas les exclusions par défaut (:data:`DEFAULT_EXCLUDED_DIRS`,
:data:`DEFAULT_EXCLUDED_FILES`) et les motifs ``exclude`` de l'utilisateur s'appliquent.
Rien n'est jamais écrit dans le dépôt. Aucune erreur d'E/S ne remonte : elle devient un
compteur ``skipped`` ou un avertissement ; les chemins concernés sont de plus listés dans
:attr:`ScanResult.unreadable` (fichiers) et :attr:`ScanResult.unreadable_dirs` (dossiers
impossibles à lister, ``""`` pour la racine) afin que l'indexeur puisse distinguer un fichier
disparu d'un fichier simplement devenu inaccessible.

Les chemins relatifs (:attr:`FileEntry.path`) sont POSIX et en Unicode **NFC**, quelle que
soit la forme renvoyée par le système de fichiers ou par git : sur macOS, le Finder et HFS+
écrivent les noms accentués décomposés (NFD) alors que git (``core.precomposeunicode``)
renvoie la forme composée ; sans normalisation, un même fichier aurait deux clés. Les
motifs d'exclusion et de ``.gitignore`` sont normalisés de la même façon.
:attr:`FileEntry.abs_path` garde en revanche le nom tel que le système le fournit, pour
que le fichier reste lisible sur un système de fichiers sensible à la forme (ext4).
"""
from __future__ import annotations

import codecs
import fnmatch
import hashlib
import os
import stat as stat_module
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from develop.git_tools import GitError, GitRepo, find_git_root
from develop.project_scanner.languages import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    TEXTLESS_LANGUAGES,
    classify_kind,
    detect_language,
    has_binary_extension,
    sniff_binary,
)

#: Raisons possibles d'écarter un fichier (clés de :attr:`ScanResult.skipped`).
SKIP_REASONS: Tuple[str, ...] = (
    "excluded_dir", "excluded_file", "binary", "too_large", "unreadable", "symlink", "no_language",
)
_SNIFF_BYTES = 8192
_HASH_BLOCK = 64 * 1024
_MAX_WARNINGS = 50
_GITIGNORE_MAX_BYTES = 256_000

_EXCLUDED_DIR_NAMES = frozenset(d for d in DEFAULT_EXCLUDED_DIRS if "/" not in d)
_EXCLUDED_DIR_PATHS = tuple(sorted(d.strip("/") for d in DEFAULT_EXCLUDED_DIRS if "/" in d))


@dataclass
class FileEntry:
    repo_id: str
    path: str  # relatif à la racine, séparateur "/", Unicode NFC
    abs_path: str  # tel que fourni par le système de fichiers (forme Unicode non modifiée)
    size: int
    mtime: float
    language: Optional[str]
    kind: str  # "code" | "doc" | "config" | "data" | "other"
    is_binary: bool  # vrai seulement pour les formats extraits par un parseur dédié (pdf, docx)


@dataclass
class ScanResult:
    files: List[FileEntry] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=lambda: {r: 0 for r in SKIP_REASONS})
    root: str = ""
    used_git: bool = False
    warnings: List[str] = field(default_factory=list)
    #: Chemins relatifs des fichiers comptés ``skipped["unreadable"]`` (stat/ouverture impossible,
    #: fichier non régulier), triés.
    unreadable: List[str] = field(default_factory=list)
    #: Dossiers (relatifs, ``""`` = racine) dont le contenu n'a pas pu être listé ; leur contenu
    #: n'apparaît ni dans ``files`` ni dans ``unreadable``. Parcours fichier seulement.
    unreadable_dirs: List[str] = field(default_factory=list)

    @property
    def total_skipped(self) -> int:
        return sum(self.skipped.values())


# --------------------------------------------------------------------------- exclusions

def nfc(text: str) -> str:
    """Forme Unicode composée (NFC) d'un chemin ou d'un motif ; ``text`` inchangé s'il l'est déjà."""
    return unicodedata.normalize("NFC", text) if text else text


def path_is_within(root: str, path: str) -> bool:
    """Vrai si ``path`` est ``root`` ou un chemin en dessous (chemins absolus, déjà résolus par l'appelant).

    La comparaison respecte la casse du système de fichiers (``os.path.normcase`` : insensible sous Windows)
    et exige une frontière de dossier : ``/a/b`` ne contient pas ``/a/bc``.
    """
    r = os.path.normcase(os.path.normpath(root))
    p = os.path.normcase(os.path.normpath(path))
    if p == r:
        return True
    return p.startswith(r if r.endswith(os.sep) else r + os.sep)


def _is_junction(entry: "os.DirEntry[str]") -> bool:
    """Vrai pour une jonction ou un point de montage NTFS (Windows), que ``is_symlink`` ne signale pas ;
    toujours faux ailleurs. Les fichiers « dans le nuage » (OneDrive) portent aussi un point d'analyse mais
    ne sont pas des jonctions : seule l'étiquette MOUNT_POINT compte. Une jonction suivie pourrait boucler."""
    if os.name != "nt":
        return False
    is_junction = getattr(entry, "is_junction", None)  # Python ≥ 3.12
    if is_junction is not None:
        try:
            return bool(is_junction())
        except OSError:
            return False
    try:
        st = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    attrs = getattr(st, "st_file_attributes", 0)
    tag = getattr(st, "st_reparse_tag", 0)
    return bool(attrs & stat_module.FILE_ATTRIBUTE_REPARSE_POINT) and tag == _IO_REPARSE_TAG_MOUNT_POINT


_IO_REPARSE_TAG_MOUNT_POINT = getattr(stat_module, "IO_REPARSE_TAG_MOUNT_POINT", 0xA0000003)


def _normalize_patterns(patterns: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for raw in patterns or ():
        p = nfc((raw or "").strip().replace("\\", "/"))
        while p.startswith("./"):
            p = p[2:]
        p = p.strip("/")
        if p:
            out.append(p)
    return tuple(out)


def is_excluded_dir_path(rel_dir: str) -> bool:
    """Vrai si un segment du chemin (ou une séquence ``a/b``) figure dans les dossiers exclus par défaut."""
    if not rel_dir:
        return False
    segments = rel_dir.split("/")
    if any(seg in _EXCLUDED_DIR_NAMES for seg in segments):
        return True
    wrapped = "/" + rel_dir + "/"
    return any(("/" + p + "/") in wrapped for p in _EXCLUDED_DIR_PATHS)


def is_excluded_file_name(name: str) -> bool:
    """Vrai si le nom de fichier correspond à un motif de :data:`DEFAULT_EXCLUDED_FILES`."""
    return any(fnmatch.fnmatchcase(name, pat) for pat in DEFAULT_EXCLUDED_FILES)


def _match_user_exclude(rel_path: str, patterns: Sequence[str]) -> Optional[str]:
    """Raison d'exclusion (``excluded_dir`` / ``excluded_file``) si un motif utilisateur
    correspond au chemin relatif, à un dossier ancêtre (nom ou chemin) ou au nom du fichier."""
    if not patterns:
        return None
    segments = rel_path.split("/")
    for pat in patterns:
        for i in range(1, len(segments)):
            prefix = "/".join(segments[:i])
            if fnmatch.fnmatchcase(segments[i - 1], pat) or fnmatch.fnmatchcase(prefix, pat):
                return "excluded_dir"
        if fnmatch.fnmatchcase(segments[-1], pat) or fnmatch.fnmatchcase(rel_path, pat):
            return "excluded_file"
    return None


def _dir_matches_user_exclude(rel_dir: str, patterns: Sequence[str]) -> bool:
    """Vrai si un motif utilisateur correspond au nom du dossier ou à son chemin relatif."""
    name = rel_dir.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(rel_dir, pat) for pat in patterns)


# --------------------------------------------------------------------------- .gitignore simple

@dataclass(frozen=True)
class _IgnoreRule:
    pattern: str
    base: str  # dossier (relatif, "" = racine) du .gitignore qui porte la règle
    dir_only: bool
    anchored: bool


def parse_gitignore(text: str, base: str = "") -> List[_IgnoreRule]:
    """Parseur ``.gitignore`` simplifié : commentaires et négations ignorés, ``x/`` = dossier,
    motif contenant ``/`` = ancré au dossier du fichier, sinon glob sur le nom à toute profondeur."""
    rules: List[_IgnoreRule] = []
    for raw in text.splitlines():
        line = nfc(raw.strip())
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        if line.startswith("\\"):
            line = line[1:]
        dir_only = line.endswith("/")
        line = line.rstrip("/")
        if line.startswith("**/"):
            line = line[3:]
        anchored = "/" in line
        line = line.lstrip("/")
        if line:
            rules.append(_IgnoreRule(line, base, dir_only, anchored))
    return rules


def _is_ignored(rules: Sequence[_IgnoreRule], rel: str, is_dir: bool) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for rule in rules:
        if rule.dir_only and not is_dir:
            continue
        if rule.base:
            if not rel.startswith(rule.base + "/"):
                continue
            sub = rel[len(rule.base) + 1:]
        else:
            sub = rel
        if rule.anchored:
            if fnmatch.fnmatchcase(sub, rule.pattern):
                return True
        elif fnmatch.fnmatchcase(name, rule.pattern):
            return True
    return False


def _load_gitignore(abs_dir: str, rel_dir: str) -> List[_IgnoreRule]:
    path = os.path.join(abs_dir, ".gitignore")
    try:
        if not os.path.isfile(path) or os.path.islink(path):
            return []
    except OSError:
        return []
    text = read_text(path, max_bytes=_GITIGNORE_MAX_BYTES)
    return parse_gitignore(text, rel_dir) if text else []


# --------------------------------------------------------------------------- parcours

def _warn(result: ScanResult, message: str) -> None:
    if len(result.warnings) < _MAX_WARNINGS:
        result.warnings.append(message)
    elif len(result.warnings) == _MAX_WARNINGS:
        result.warnings.append("… d'autres avertissements ont été tronqués.")


def _unreadable(result: ScanResult, rel: str) -> None:
    """Compte un fichier illisible et retient son chemin (voir :attr:`ScanResult.unreadable`)."""
    result.skipped["unreadable"] += 1
    result.unreadable.append(rel)


def _walk_candidates(root_abs: str, patterns: Sequence[str], result: ScanResult) -> Iterator[Tuple[str, str]]:
    """Génère ``(chemin relatif POSIX NFC, chemin absolu brut)`` des fichiers candidats, en
    élaguant les dossiers exclus (défaut, motifs utilisateur, .gitignore) et sans suivre les liens."""
    stack: List[Tuple[str, str, Tuple[_IgnoreRule, ...]]] = [("", root_abs, ())]
    while stack:
        rel_dir, abs_dir, inherited = stack.pop()
        rules = inherited + tuple(_load_gitignore(abs_dir, rel_dir))
        try:
            with os.scandir(abs_dir) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            _warn(result, "Dossier illisible, ignoré : %s (%s)" % (rel_dir or ".", exc.strerror or exc))
            result.unreadable_dirs.append(rel_dir)
            continue
        for entry in entries:
            name = nfc(entry.name)  # clé NFC ; entry.path garde le nom brut pour les accès disque
            rel = name if not rel_dir else rel_dir + "/" + name
            try:
                if entry.is_symlink() or _is_junction(entry):
                    result.skipped["symlink"] += 1
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                _unreadable(result, rel)
                continue
            if is_dir:
                if is_excluded_dir_path(rel) or _dir_matches_user_exclude(rel, patterns) \
                        or _is_ignored(rules, rel, True):
                    result.skipped["excluded_dir"] += 1
                    continue
                stack.append((rel, entry.path, rules))
            else:
                if _is_ignored(rules, rel, False):
                    result.skipped["excluded_file"] += 1
                    continue
                yield rel, entry.path


def _git_candidates(root_abs: str, rel_paths: Sequence[str], result: ScanResult) -> Iterator[Tuple[str, str]]:
    """Génère ``(chemin relatif POSIX NFC, chemin absolu)`` depuis la liste de git ; le chemin
    absolu reprend la forme écrite par git, qui est celle du disque quand git ne recompose pas."""
    for raw in rel_paths:
        raw = raw.replace("\\", "/")
        rel = nfc(raw)
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        if is_excluded_dir_path(parent):
            result.skipped["excluded_dir"] += 1
            continue
        abs_path = os.path.join(root_abs, *raw.split("/"))
        try:
            if os.path.islink(abs_path):
                result.skipped["symlink"] += 1
                continue
        except OSError:
            _unreadable(result, rel)
            continue
        yield rel, abs_path


def _add_file(rel: str, abs_path: str, repo_id: str, patterns: Sequence[str], max_file_size: int,
              include_unknown: bool, result: ScanResult) -> None:
    name = rel.rsplit("/", 1)[-1]
    if is_excluded_file_name(name):
        result.skipped["excluded_file"] += 1
        return
    reason = _match_user_exclude(rel, patterns)
    if reason:
        result.skipped[reason] += 1
        return
    if has_binary_extension(name):
        result.skipped["binary"] += 1
        return
    language = detect_language(name)
    if language is None and not include_unknown:
        result.skipped["no_language"] += 1
        return
    try:
        st = os.stat(abs_path)
    except OSError:
        _unreadable(result, rel)
        return
    if not stat_module.S_ISREG(st.st_mode):
        _unreadable(result, rel)
        return
    if st.st_size > max_file_size:
        result.skipped["too_large"] += 1
        return
    is_binary = False
    if language in TEXTLESS_LANGUAGES:
        is_binary = True
    else:
        try:
            with open(abs_path, "rb") as fh:
                head = fh.read(_SNIFF_BYTES)
        except OSError:
            _unreadable(result, rel)
            return
        if sniff_binary(head):
            result.skipped["binary"] += 1
            return
    result.files.append(FileEntry(
        repo_id=repo_id, path=rel, abs_path=abs_path, size=int(st.st_size), mtime=float(st.st_mtime),
        language=language, kind=classify_kind(rel, language), is_binary=is_binary,
    ))


def scan_repo(root: str, repo_id: str, exclude: Sequence[str] = (), max_file_size: int = 2_000_000,
              use_git: bool = True, include_unknown: bool = False) -> ScanResult:
    """Liste les fichiers indexables de ``root`` (jamais modifié).

    * ``exclude`` : motifs glob supplémentaires, comparés au chemin relatif, au nom du
      fichier et à chaque dossier ancêtre (nom ou chemin) ; ``docs``, ``*.sql``,
      ``backend/database`` fonctionnent tous.
    * ``use_git`` : si ``root`` est la racine d'un dépôt Git, la liste vient de
      ``git ls-files`` (fichiers ignorés par Git absents sans être comptés) ; sinon, ou en
      cas d'échec de git, parcours du système de fichiers.
    * ``include_unknown`` : garder les fichiers sans langage connu (``kind`` = ``other``).

    Les chemins relatifs renvoyés sont en Unicode NFC (voir le module) : c'est la clé stable
    d'un fichier dans la mémoire, partagée avec les chemins lus dans git.

    Compteurs ``skipped`` : en parcours fichier, ``excluded_dir`` compte les *dossiers*
    élagués (leur contenu n'est pas visité) ; en mode git, il compte les fichiers listés
    situés sous un dossier exclu. ``binary`` ne concerne jamais les PDF/DOCX, conservés
    avec ``is_binary=True`` pour les extracteurs dédiés. Les chemins comptés ``unreadable``
    sont listés dans ``unreadable`` ; un dossier impossible à lister (droits) est signalé par
    un avertissement et listé dans ``unreadable_dirs`` (``""`` si c'est la racine elle-même).
    """
    root_abs = os.path.abspath(os.path.expanduser(root or "."))
    result = ScanResult(root=root_abs)
    if not os.path.isdir(root_abs):
        _warn(result, "Dossier introuvable ou inaccessible : %s" % root)
        return result
    patterns = _normalize_patterns(exclude)

    candidates: Optional[Iterator[Tuple[str, str]]] = None
    if use_git:
        try:
            git_root = find_git_root(root_abs)
            if git_root is not None and git_root == Path(root_abs).resolve():
                rel_paths = GitRepo(root_abs).ls_files()
                candidates = _git_candidates(root_abs, rel_paths, result)
                result.used_git = True
        except GitError as exc:
            _warn(result, "git indisponible (%s) ; parcours du système de fichiers à la place." % exc)
        except OSError as exc:
            _warn(result, "Erreur en consultant git (%s) ; parcours du système de fichiers à la place." % exc)
    if candidates is None:
        candidates = _walk_candidates(root_abs, patterns, result)

    for rel, abs_path in candidates:
        _add_file(rel, abs_path, repo_id, patterns, max_file_size, include_unknown, result)
    result.files.sort(key=lambda f: f.path)
    result.unreadable.sort()
    result.unreadable_dirs.sort()
    return result


# --------------------------------------------------------------------------- lecture

def compute_sha1(abs_path: str) -> str:
    """SHA-1 hexadécimal du contenu, calculé par blocs ; chaîne vide si le fichier est illisible."""
    h = hashlib.sha1()
    try:
        with open(abs_path, "rb") as fh:
            while True:
                block = fh.read(_HASH_BLOCK)
                if not block:
                    break
                h.update(block)
    except OSError:
        return ""
    return h.hexdigest()


def decode_bytes(data: bytes, truncated: bool = False) -> Optional[str]:
    """Décode un contenu : UTF-16 si BOM, sinon UTF-8 (BOM toléré), sinon latin-1 ; ``None`` si binaire.

    ``truncated`` indique que ``data`` a été coupé : une séquence UTF-8 incomplète en fin
    de tampon est alors simplement abandonnée au lieu de faire basculer en latin-1.
    """
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        body = data if len(data) % 2 == 0 else data[:-1]
        return body.decode("utf-16", errors="replace")
    if b"\0" in data:
        return None
    try:
        decoder = codecs.getincrementaldecoder("utf-8-sig")()
        return decoder.decode(data, final=not truncated)
    except UnicodeDecodeError:
        return data.decode("latin-1")


def read_text(abs_path: str, max_bytes: int = 2_000_000) -> Optional[str]:
    """Contenu texte du fichier (au plus ``max_bytes`` octets), ``None`` si binaire ou illisible."""
    try:
        with open(abs_path, "rb") as fh:
            data = fh.read(max(0, max_bytes))
    except OSError:
        return None
    return decode_bytes(data, truncated=len(data) >= max_bytes)
