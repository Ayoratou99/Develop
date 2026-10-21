"""Analyse d'impact déterministe (sans LLM) à partir du graphe de relations de la mémoire Develop.

Principe : on part de « graines » (symboles, endpoints, modèles ou fichiers changés, confiance
1.0) et on suit les arêtes du graphe **dans le sens des dépendants** : qui importe ce
fichier, quelle route est gérée par ce symbole, quel appel frontend vise cette route, quel
modèle est la même entité, quelle table en découle, quel document ou quelle note les
décrit. Chaque saut multiplie la confiance par le poids de l'arête puis par une décroissance
(:data:`DECAY`) ; un noeud atteint par plusieurs chemins garde le meilleur (chemin lisible
dans ``via``). Les noeuds « centraux » (plus de :data:`HUB_THRESHOLD` dépendants, ex. un
module utilitaire importé partout) sont signalés mais pas développés, et l'exploration est
plafonnée à :data:`NODE_CAP` noeuds pour rester instantanée sur de gros projets.

Analyse Git et index obsolète : les plages d'un diff sont des numéros de ligne **côté nouveau**
(arbre de travail, ou commit ``B`` d'une plage ``A..B``). Elles ne peuvent être croisées avec les
positions des symboles indexés que si l'index reflète ce contenu ; c'est vérifié fichier par
fichier (taille et mtime, puis sha1, comme l'indexation incrémentale). Sinon, les symboles,
routes et modèles du fichier sont relus depuis le contenu côté nouveau (même extracteur que
l'indexation), croisés avec les plages, puis ramenés aux noeuds indexés par nom qualifié ; le
rapport est marqué ``stale_index``. Si cette relecture est impossible, le fichier est pris en
entier plutôt que de produire une attribution fausse.

Rien ici n'écrit dans la base ni dans les dépôts.
"""
from __future__ import annotations

import hashlib
import heapq
import os
import subprocess
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from develop.code_analysis import ExtractedEndpoint, ExtractedModel, ExtractedSymbol, analyze_file, normalize_route_path
from develop.code_analysis.model import route_paths_match
from develop.config import WorkspaceConfig
from develop.git_tools import ChangedFile, GitError, GitRepo, find_git_root
from develop.impact_analysis.diff import changed_line_ranges, parse_unified_diff, ranges_intersect, symbols_touched
from develop.project_scanner import (
    compute_sha1,
    decode_bytes,
    detect_language,
    has_binary_extension,
    is_excluded_dir_path,
    is_excluded_file_name,
    nfc,
    read_text,
)
from develop.storage import Database, EndpointRow, FileRow, ModelRow, RelationRow, SymbolRow, node_key

__all__ = ["ImpactItem", "ImpactReport", "ImpactAnalyzer", "DECAY", "HUB_THRESHOLD", "NODE_CAP", "MAX_HOPS"]

# --------------------------------------------------------------------------- réglages

DECAY = 0.85  # décroissance de la confiance à chaque saut
HUB_THRESHOLD = 200  # au-delà de ce nombre de dépendants, un noeud est signalé mais pas développé
NODE_CAP = 500  # nombre maximal de noeuds explorés par analyse
MAX_HOPS = 3  # profondeur maximale (bornée quelle que soit la demande)
MIN_CONFIDENCE = 0.02  # en dessous, on n'explore plus
LEXICAL_LIMIT = 10  # documents/notes ajoutés par la recherche lexicale
LEXICAL_CONFIDENCE = 0.5

# poids des arêtes (× DECAY à chaque saut)
W_CONTAINS = 1.0  # symbole / endpoint / modèle → son fichier
W_IMPORTS = 0.9  # fichier → fichiers qui l'importent
W_HANDLES = 0.95  # symbole → endpoint serveur qu'il gère
W_CLIENT_IMPORTER = 0.7  # appel client → fichiers qui importent le fichier de l'appel
W_SAME_ENTITY = 0.8  # modèle ↔ modèle (même entité), table → modèles qui la définissent
W_DEFINES_TABLE = 0.9  # modèle → table
W_USES_MODEL = 0.6  # modèle → fichiers / symboles qui l'utilisent
W_DOCUMENTED_BY = 0.9  # noeud → document / note qui le décrit
W_MENTIONS = 0.7  # document / note qui mentionne le noeud (relation inverse, moins sûre)
W_CALLS = 0.9  # symbole → symboles qui l'appellent
W_HANDLER = 0.6  # endpoint → symbole qui le gère (route modifiée : à vérifier)
W_MODEL_CLASS = 1.0  # symbole de classe → modèle de données du même nom

_DEPENDENT_KINDS = ("imports", "calls_endpoint", "uses_model", "calls")
_TYPE_ORDER = {"endpoint": 0, "model": 1, "table": 2, "symbol": 3, "file": 4, "document": 5, "note": 6}
_STATUS_LABELS = {"M": "modified", "A": "added", "D": "deleted", "R": "renamed", "?": "untracked"}
_MODEL_SYMBOL_KINDS = ("class", "interface", "type", "enum", "dataclass", "model")
_MAX_REREAD_BYTES = 2_000_000  # même plafond que le scanner : au-delà, le fichier est pris en entier


# --------------------------------------------------------------------------- structures de résultat

@dataclass
class ImpactItem:
    """Un élément potentiellement affecté par le changement analysé."""

    node: str  # clé du noeud ("file:12", "endpoint:3", "table:orders"…)
    type: str  # 'file' | 'symbol' | 'endpoint' | 'model' | 'document' | 'note' | 'table'
    name: str
    path: Optional[str] = None
    line: Optional[int] = None
    repo_id: Optional[str] = None
    reason: str = ""  # explication lisible, en français
    confidence: float = 0.0  # 0..1
    hops: int = 0
    via: List[str] = field(default_factory=list)  # chaîne lisible depuis la graine
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ident(self) -> Optional[int]:
        """Identifiant numérique du noeud (None pour une table)."""
        return self.extra.get("id") if isinstance(self.extra.get("id"), int) else None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(float(self.confidence), 4)
        d["id"] = self.ident
        d["file_id"] = self.extra.get("file_id")
        return d


@dataclass
class ImpactReport:
    """Rapport complet, conforme à ``ImpactReport`` dans docs/API.md."""

    source: Dict[str, Any] = field(default_factory=dict)
    changed: List[Dict[str, Any]] = field(default_factory=list)
    affected: List[ImpactItem] = field(default_factory=list)
    endpoints: List[ImpactItem] = field(default_factory=list)
    models: List[ImpactItem] = field(default_factory=list)
    frontend_calls: List[ImpactItem] = field(default_factory=list)
    documents: List[ImpactItem] = field(default_factory=list)
    notes: List[ImpactItem] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    stale_index: bool = False
    stale_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": dict(self.source),
            "changed": [dict(c) for c in self.changed],
            "affected": [i.to_dict() for i in self.affected],
            "endpoints": [i.to_dict() for i in self.endpoints],
            "models": [i.to_dict() for i in self.models],
            "frontend_calls": [i.to_dict() for i in self.frontend_calls],
            "documents": [i.to_dict() for i in self.documents],
            "notes": [i.to_dict() for i in self.notes],
            "stats": dict(self.stats),
            "warnings": list(self.warnings),
            "stale_index": bool(self.stale_index),
            "stale_files": list(self.stale_files),
        }


# --------------------------------------------------------------------------- structures internes

@dataclass
class _NodeInfo:
    """Description d'un noeud du graphe, résolue une fois par analyse."""

    key: str
    type: str
    name: str
    label: str  # texte court pour la chaîne ``via``
    path: Optional[str] = None
    line: Optional[int] = None
    repo_id: Optional[str] = None
    file_id: Optional[int] = None
    side: Optional[str] = None  # endpoints : "server" | "client"
    extra: Dict[str, Any] = field(default_factory=dict)
    row: Any = None


@dataclass
class _Edge:
    dst: str
    factor: float
    reason: str


@dataclass
class _Reach:
    confidence: float
    hops: int
    reason: str
    via: List[str]


class _Run:
    """État d'une analyse : caches, avertissements, graines, bornes."""

    def __init__(self, max_hops: int) -> None:
        self.max_hops = max_hops
        self.warnings: List[str] = []
        self.infos: Dict[str, Optional[_NodeInfo]] = {}
        self.files: Dict[int, Optional[FileRow]] = {}
        self.hubs: List[str] = []
        self.exclude: Set[str] = set()
        self.seed_count = 0
        self.truncated = False
        self.nodes_visited = 0
        self.stale_index = False
        self.stale_files: List[str] = []
        self.reanalyzed: List[str] = []  # fichiers dont les symboles ont été relus (index obsolète)

    def warn(self, message: str) -> None:
        if message and message not in self.warnings:
            self.warnings.append(message)

    def mark_stale(self, path: str) -> None:
        self.stale_index = True
        if path not in self.stale_files:
            self.stale_files.append(path)


@dataclass
class _GitContext:
    """Ce qu'il faut pour retrouver le contenu « côté nouveau » d'un fichier changé.

    ``new_rev`` : commit du côté nouveau d'une plage ``A..B`` ; ``None`` quand le côté nouveau est
    l'arbre de travail (diff sans plage, ou contre une seule révision). ``prefix`` : sous-dossier du
    dépôt configuré dans le dépôt Git (monorepo), pour reconstituer les chemins Git.
    """

    repo_path: str
    git: Optional[GitRepo] = None
    prefix: str = ""
    new_rev: Optional[str] = None


# --------------------------------------------------------------------------- analyseur

class ImpactAnalyzer:
    """Analyse d'impact sur la mémoire Develop d'un espace de travail.

    :param db: base de l'espace de travail (lecture seule ici).
    :param workspace: configuration (chemins des dépôts, pour Git).
    :param search_engine: optionnel ; objet exposant ``search(query, limit=, types=)`` ou
        ``fts_search(query, limit=, source_types=)`` (une :class:`Database` convient), ou un
        simple appelable ``f(query)`` ; sert à compléter documents et notes par recherche
        lexicale sur les noms des symboles changés.
    :param stale_checker: optionnel ; ``f(repo_id) -> [chemins]`` des fichiers modifiés depuis
        la dernière indexation (ex. ``Indexer.stale_files``) ; s'il en renvoie, le rapport est
        marqué ``stale_index``.
    """

    def __init__(self, db: Database, workspace: WorkspaceConfig, search_engine: Any = None,
                 stale_checker: Optional[Callable[[str], List[str]]] = None):
        self.db = db
        self.workspace = workspace
        self.search_engine = search_engine
        self.stale_checker = stale_checker

    # ------------------------------------------------------------------ points d'entrée

    def from_git(self, repo_id: str, rev_range: Optional[str] = None, max_hops: int = 2) -> ImpactReport:
        """Impact des changements Git d'un dépôt : working tree vs HEAD, ou une plage ``A..B``.

        Pour chaque fichier changé présent dans l'index, seules les lignes modifiées
        (diff en contexte zéro) servent à choisir les symboles graines ; un fichier ajouté,
        supprimé ou non suivi est pris en entier. Un fichier changé absent de l'index
        déclenche un avertissement « non indexé » et ``stale_index``.

        Les chemins lus dans git sont ramenés en Unicode NFC (clé des fichiers dans l'index,
        voir :mod:`develop.project_scanner`) : git peut renvoyer la forme décomposée (NFD)
        d'un nom accentué selon ``core.precomposeunicode`` et le système de fichiers.

        Quand l'index ne reflète plus le contenu côté nouveau d'un fichier (modifié depuis la
        dernière indexation), ses symboles sont relus depuis ce contenu avant le croisement
        (entrée ``reanalyzed: True``) ; le rapport est marqué ``stale_index`` et invite à
        réindexer. Si la relecture est impossible, le fichier est pris en entier.
        """
        run = self._new_run(max_hops)
        rev_range = (rev_range or "").strip() or None
        source: Dict[str, Any] = {"kind": "git", "repo_id": repo_id, "rev_range": rev_range, "max_hops": run.max_hops}
        changed: List[Dict[str, Any]] = []
        seeds: List[Tuple[str, float, str]] = []
        names: List[str] = []
        try:
            repo_path = self._repo_path(repo_id)
            if repo_path is None:
                run.warn("Dépôt inconnu : %s" % repo_id)
                return self._finish(run, source, changed, [])
            if rev_range and rev_range.startswith("-"):
                run.warn("Plage de commits invalide : %s" % rev_range)
                return self._finish(run, source, changed, [])
            try:
                git, prefix = self._open_git(repo_path)
            except GitError as exc:
                run.warn("%s n'est pas un dépôt Git : analyse impossible (%s)" % (repo_path, exc))
                return self._finish(run, source, changed, [])
            try:
                changes = git.diff_names(rev_range)
            except GitError as exc:
                run.warn("Lecture des changements Git impossible : %s" % exc)
                return self._finish(run, source, changed, [])
            changes = [c for c in ((ch, _within_repo(ch, prefix)) for ch in changes) if c[1] is not None]
            hunks_by_path = self._zero_context_hunks(run, git, rev_range, prefix) if changes else {}
            ctx = _GitContext(repo_path=repo_path, git=git, prefix=prefix, new_rev=_new_side_rev(rev_range))
            for git_ch, ch in changes:
                try:
                    entry = self._changed_entry(run, repo_id, ch, hunks_by_path.get(nfc(git_ch.path)), seeds, ctx)
                except Exception as exc:  # noqa: BLE001 - un fichier ne bloque jamais l'analyse
                    run.warn("Fichier %s ignoré : %s" % (ch.path, exc))
                    entry = {"path": ch.path, "status": _STATUS_LABELS.get(ch.status, ch.status), "repo_id": repo_id,
                             "file_id": None, "symbols_changed": [], "ranges": [], "indexed": False}
                changed.append(entry)
                names.extend(entry.get("symbols_changed", []))
            if run.reanalyzed:
                run.warn(_reanalyzed_warning(run.reanalyzed))
            self._check_stale(run, repo_id)
            items = self._propagate(run, seeds)
            items = self._lexical_complement(run, names, items)
            return self._finish(run, source, changed, items)
        except Exception as exc:  # noqa: BLE001 - jamais de 500 pour une analyse
            run.warn("Analyse interrompue : %s" % exc)
            return self._finish(run, source, changed, [])

    def from_file(self, repo_id: str, path: str, max_hops: int = 2) -> ImpactReport:
        """Impact d'un fichier entier : tous ses symboles, endpoints et modèles sont des graines."""
        run = self._new_run(max_hops)
        rel = self._relative_path(repo_id, path)
        source: Dict[str, Any] = {"kind": "file", "repo_id": repo_id, "path": rel, "max_hops": run.max_hops}
        try:
            file_row = self.db.get_file(repo_id, rel) if rel else None
            if file_row is None:
                run.warn("Fichier introuvable dans l'index : %s (dépôt %s). Indexez le projet puis relancez." % (rel or path, repo_id))
                return self._finish(run, source, [], [])
            seeds: List[Tuple[str, float, str]] = []
            entry = self._whole_file_entry(run, file_row, "selected", seeds)
            self._check_stale(run, repo_id)
            items = self._propagate(run, seeds)
            items = self._lexical_complement(run, entry["symbols_changed"], items)
            return self._finish(run, source, [entry], items)
        except Exception as exc:  # noqa: BLE001
            run.warn("Analyse interrompue : %s" % exc)
            return self._finish(run, source, [], [])

    def from_symbol(self, symbol_id: int, max_hops: int = 2) -> ImpactReport:
        """Impact d'un symbole (fonction, méthode, classe…) identifié par son id."""
        run = self._new_run(max_hops)
        source: Dict[str, Any] = {"kind": "symbol", "symbol_id": symbol_id, "max_hops": run.max_hops}
        try:
            try:
                sym = self.db.get_symbol(int(symbol_id))
            except (TypeError, ValueError):
                sym = None
            if sym is None or sym.id is None:
                run.warn("Symbole introuvable : id=%s" % symbol_id)
                return self._finish(run, source, [], [])
            file_row = self._file(run, sym.file_id)
            qname = _qualified(sym)
            source["name"] = qname
            source["repo_id"] = file_row.repo_id if file_row else None
            source["path"] = file_row.path if file_row else None
            seeds = [(node_key("symbol", sym.id), 1.0, qname)]
            changed = [{
                "path": file_row.path if file_row else None,
                "status": "selected",
                "repo_id": file_row.repo_id if file_row else None,
                "file_id": sym.file_id,
                "symbols_changed": [qname],
                "ranges": [[sym.line_start, sym.line_end]] if sym.line_start is not None else [],
                "indexed": True,
            }]
            if file_row is not None:
                self._check_stale(run, file_row.repo_id)
            items = self._propagate(run, seeds)
            items = self._lexical_complement(run, [qname], items)
            return self._finish(run, source, changed, items)
        except Exception as exc:  # noqa: BLE001
            run.warn("Analyse interrompue : %s" % exc)
            return self._finish(run, source, [], [])

    def propagate(self, seeds: List[Tuple[str, float, str]], max_hops: int) -> List[ImpactItem]:
        """Propagation seule : ``seeds`` = ``[(clé de noeud, confiance, libellé)]``.

        Renvoie les éléments affectés (graines exclues), dédupliqués par noeud, triés par
        confiance décroissante puis nombre de sauts.
        """
        run = self._new_run(max_hops)
        return self._sorted(self._propagate(run, seeds))

    # ------------------------------------------------------------------ préparation

    @staticmethod
    def _new_run(max_hops: int) -> _Run:
        try:
            requested = int(max_hops)
        except (TypeError, ValueError):
            requested = 2
        effective = max(1, min(MAX_HOPS, requested))
        run = _Run(effective)
        if requested > MAX_HOPS:
            run.warn("Profondeur limitée à %d sauts (demandé : %d)." % (MAX_HOPS, requested))
        return run

    def _repo_path(self, repo_id: str) -> Optional[str]:
        cfg = self.workspace.repo_by_id(repo_id) if self.workspace is not None else None
        if cfg is not None and cfg.path:
            return cfg.path
        row = self.db.get_repo(repo_id)
        if row is not None and row.path:
            return row.path
        return None

    def _repo_label(self, repo_id: Optional[str]) -> str:
        if not repo_id:
            return "client"
        cfg = self.workspace.repo_by_id(repo_id) if self.workspace is not None else None
        if cfg is not None and cfg.name:
            return cfg.name
        row = self.db.get_repo(repo_id)
        if row is not None and row.name:
            return row.name
        return repo_id

    def _relative_path(self, repo_id: str, path: str) -> str:
        """Chemin relatif au dépôt, séparateur ``/``, Unicode NFC (forme stockée dans l'index) ;
        accepte un chemin absolu sous la racine du dépôt."""
        p = (path or "").strip().replace("\\", "/")
        if not p:
            return ""
        root = self._repo_path(repo_id)
        if root and os.path.isabs(p):
            try:
                rel = Path(p).resolve().relative_to(Path(root).resolve())
                p = rel.as_posix()
            except (ValueError, OSError):
                pass
        while p.startswith("./"):
            p = p[2:]
        return nfc(p.strip("/"))

    def _check_stale(self, run: _Run, repo_id: Optional[str]) -> None:
        if self.stale_checker is None or not repo_id:
            return
        try:
            paths = [str(p) for p in (self.stale_checker(repo_id) or [])]
        except Exception as exc:  # noqa: BLE001
            run.warn("Vérification de fraîcheur de l'index impossible : %s" % exc)
            return
        if paths:
            run.stale_index = True
            for p in paths:
                if p not in run.stale_files:
                    run.stale_files.append(p)
            run.warn("Index obsolète : %d fichier(s) modifié(s) depuis la dernière indexation. "
                     "Réindexez puis relancez l'analyse." % len(paths))

    # ------------------------------------------------------------------ Git

    @staticmethod
    def _open_git(repo_path: str) -> Tuple[GitRepo, str]:
        """Dépôt Git contenant ``repo_path`` et préfixe POSIX (``"backend/"``) du dépôt configuré dans celui-ci.

        Un dépôt configuré peut être un sous-dossier d'un dépôt Git (monorepo) : Git renvoie alors des
        chemins relatifs à sa racine, que l'analyse ramène au dépôt configuré (préfixe vide si le
        dépôt configuré est la racine Git). Lève :class:`GitError` hors d'un dépôt Git.
        """
        root = find_git_root(repo_path)
        git = GitRepo(str(root) if root is not None else repo_path)
        try:
            rel = Path(repo_path).resolve().relative_to(git.root).as_posix()
        except ValueError:
            rel = ""
        return git, "" if rel in ("", ".") else rel.rstrip("/") + "/"

    def _zero_context_hunks(self, run: _Run, git: GitRepo, rev_range: Optional[str],
                            prefix: str = "") -> Dict[str, List[Any]]:
        """Diff en contexte zéro (``git diff --unified=0``), parsé ; ``{}`` + avertissement si indisponible.

        Les clés sont les chemins tels que Git les écrit (relatifs à la racine Git), ramenés en
        NFC ; ``prefix`` (forme brute, c'est un pathspec) restreint le diff au sous-dossier du
        dépôt configuré.
        """
        if not rev_range and git.head() is None:
            return {}  # dépôt sans commit : tout est non suivi
        args = ["git", "-C", str(git.root), "-c", "core.quotepath=off", "diff", "--no-color", "--no-ext-diff",
                "--unified=0", "-M", rev_range if rev_range else "HEAD"]
        if prefix:
            args += ["--", prefix.rstrip("/")]
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C.UTF-8")
        try:
            out = subprocess.run(args, capture_output=True, timeout=60, check=False, env=env)
        except (OSError, subprocess.SubprocessError) as exc:
            run.warn("Diff détaillé indisponible (%s) : les fichiers modifiés sont analysés en entier." % exc)
            return {}
        if out.returncode != 0:
            err = out.stderr.decode("utf-8", "replace").strip()
            run.warn("Diff détaillé indisponible (%s) : les fichiers modifiés sont analysés en entier." % err)
            return {}
        try:
            hunks = parse_unified_diff(out.stdout.decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            run.warn("Diff illisible (%s) : les fichiers modifiés sont analysés en entier." % exc)
            return {}
        return {nfc(path): file_hunks for path, file_hunks in hunks.items()}

    def _changed_entry(self, run: _Run, repo_id: str, ch: ChangedFile, hunks: Optional[List[Any]],
                       seeds: List[Tuple[str, float, str]], ctx: Optional[_GitContext] = None) -> Dict[str, Any]:
        """Entrée ``changed`` et graines d'un fichier changé.

        Un fichier ajouté, supprimé, non suivi ou sans bloc de diff est pris en entier. Sinon les
        plages modifiées (lignes côté nouveau) sont croisées avec les symboles, routes et modèles
        indexés, à condition que l'index reflète le contenu côté nouveau (vérifié grâce à ``ctx``) ;
        s'il est obsolète, ils sont relus depuis ce contenu (:meth:`_reanalyzed_entry`), et le
        fichier est pris en entier si cette relecture est impossible. Sans ``ctx``, aucune
        vérification n'est possible : croisement direct.
        """
        status = _STATUS_LABELS.get(ch.status, ch.status or "modified")
        file_row = self.db.get_file(repo_id, ch.path)
        if file_row is None and ch.old_path:
            file_row = self.db.get_file(repo_id, ch.old_path)
            if file_row is not None:
                run.stale_index = True
                if ch.path not in run.stale_files:
                    run.stale_files.append(ch.path)
                run.warn("Fichier renommé non réindexé : %s → %s" % (ch.old_path, ch.path))
        if file_row is None or file_row.id is None:
            if ch.status != "D" and self._looks_indexable(ch.path):
                run.stale_index = True
                if ch.path not in run.stale_files:
                    run.stale_files.append(ch.path)
                run.warn("Fichier modifié non indexé : %s. Réindexez pour une analyse complète." % ch.path)
            return {"path": ch.path, "status": status, "repo_id": repo_id, "file_id": None, "old_path": ch.old_path,
                    "symbols_changed": [], "ranges": [], "indexed": False}
        whole = ch.status in ("A", "D", "?") or hunks is None or len(hunks) == 0
        if whole:
            entry = self._whole_file_entry(run, file_row, status, seeds)
            entry["path"] = ch.path  # chemin côté nouveau, même si l'index ne connaît encore que l'ancien
            entry["old_path"] = ch.old_path
            return entry
        ranges = changed_line_ranges(hunks)
        if not ranges:
            entry = self._whole_file_entry(run, file_row, status, seeds)
            entry["path"] = ch.path
            entry["old_path"] = ch.old_path
            return entry
        index_symbols = self.db.list_symbols(file_id=file_row.id, limit=100_000)
        index_endpoints = self._file_endpoints(file_row.id)
        index_models = self._file_models(file_row.id)
        in_sync, text = True, None
        if ctx is not None and (index_symbols or index_endpoints or index_models):
            # sans artefact indexé, le croisement est vide quoi qu'il arrive : le fichier sera la graine
            in_sync, text = self._index_matches_new_side(file_row, ch, ctx)
        if not in_sync:
            run.mark_stale(ch.path)
            entry: Optional[Dict[str, Any]] = None
            if text is not None:
                entry = self._reanalyzed_entry(run, repo_id, ch, file_row, status, ranges, text, seeds,
                                               index_symbols, index_endpoints, index_models)
            if entry is None:
                run.warn("Index obsolète pour %s et relecture impossible : fichier analysé en entier. "
                         "Réindexez pour une analyse exacte." % ch.path)
                entry = self._whole_file_entry(run, file_row, status, seeds)
                entry["path"] = ch.path
                entry["old_path"] = ch.old_path
                return entry
            run.reanalyzed.append(ch.path)
            return entry
        symbols = [s for s in index_symbols if ranges_intersect(s.line_start, s.line_end, ranges)]
        touched_names = {s.name for s in symbols}
        endpoints = [e for e in index_endpoints if ranges_intersect(e.line, e.line, ranges)]
        models = [m for m in index_models if m.name in touched_names or ranges_intersect(m.line, m.line, ranges)]
        names: List[str] = []
        before = len(seeds)
        for s in symbols:
            if s.id is not None:
                q = _qualified(s)
                names.append(q)
                seeds.append((node_key("symbol", s.id), 1.0, q))
        for e in endpoints:
            if e.id is not None:
                seeds.append((node_key("endpoint", e.id), 1.0, _endpoint_label(e, None)))
        for m in models:
            if m.id is not None:
                seeds.append((node_key("model", m.id), 1.0, m.name))
        file_key = node_key("file", file_row.id)
        if len(seeds) == before:
            # changement hors de tout symbole (imports, code de haut niveau) : le fichier est la graine
            seeds.append((file_key, 1.0, file_row.path))
        else:
            run.exclude.add(file_key)  # le fichier changé figure dans `changed`, pas dans `affected`
        return {"path": ch.path, "status": status, "repo_id": repo_id, "file_id": file_row.id, "old_path": ch.old_path,
                "symbols_changed": names, "ranges": [[a, b] for a, b in ranges], "indexed": True,
                "whole_file": False, "seeds": len(seeds) - before, "reanalyzed": False}

    def _index_matches_new_side(self, file_row: FileRow, ch: ChangedFile,
                                ctx: _GitContext) -> Tuple[bool, Optional[str]]:
        """L'index reflète-t-il le contenu côté nouveau du diff ? Renvoie ``(à jour, texte)``.

        - côté nouveau = arbre de travail (pas de plage, ou une seule révision) : signature disque
          (taille, mtime) puis sha1, comme l'indexation incrémentale ;
        - côté nouveau = commit ``B`` d'une plage ``A..B`` : sha1 du contenu de ``B:chemin``.
        ``texte`` est le contenu côté nouveau quand l'index est obsolète, ``None`` s'il est
        illisible (binaire, trop gros, ``git show`` en échec). Une ligne d'index sans aucune
        signature (l'indexeur en écrit toujours une) est tenue pour à jour, faute de pouvoir
        vérifier.
        """
        if not file_row.sha1 and file_row.size is None and file_row.mtime is None:
            return True, None
        if ctx.new_rev is None:
            abs_path = _disk_path(ctx.repo_path, ch.path)
            try:
                st = os.stat(abs_path)
            except OSError:
                return False, None
            if (file_row.size is not None and file_row.mtime is not None
                    and int(file_row.size) == int(st.st_size) and float(file_row.mtime) == float(st.st_mtime)):
                return True, None
            sha1 = compute_sha1(abs_path)
            if sha1 and file_row.sha1 and sha1 == file_row.sha1:
                return True, None
            if st.st_size > _MAX_REREAD_BYTES:
                return False, None
            return False, read_text(abs_path, _MAX_REREAD_BYTES)
        data = _git_show_bytes(ctx.git, ctx.prefix + ch.path, ctx.new_rev) if ctx.git is not None else None
        if data is None:
            return False, None
        if file_row.sha1 and hashlib.sha1(data).hexdigest() == file_row.sha1:
            return True, None
        if len(data) > _MAX_REREAD_BYTES:
            return False, None
        return False, decode_bytes(data)

    def _reanalyzed_entry(self, run: _Run, repo_id: str, ch: ChangedFile, file_row: FileRow, status: str,
                          ranges: Sequence[Tuple[int, int]], text: str, seeds: List[Tuple[str, float, str]],
                          index_symbols: Sequence[SymbolRow], index_endpoints: Sequence[EndpointRow],
                          index_models: Sequence[ModelRow]) -> Optional[Dict[str, Any]]:
        """Graines d'un fichier modifié dont l'index est obsolète (lignes décalées).

        Les symboles, routes et modèles sont relus depuis ``text`` (contenu côté nouveau, même
        extracteur que l'indexation), croisés avec ``ranges`` (lignes côté nouveau, donc cohérentes),
        puis ramenés aux noeuds indexés : par nom qualifié pour un symbole, côté + méthode + chemin
        normalisé pour une route, nom pour un modèle. Un élément touché mais inconnu de l'index
        (ex. une méthode ajoutée) figure dans ``symbols_changed`` sans devenir une graine : il n'a
        pas encore de dépendants dans le graphe. ``None`` si aucun extracteur ne s'applique.
        """
        assert file_row.id is not None
        language = detect_language(ch.path) or file_row.language
        analysis = analyze_file(text, language, ch.path)
        if analysis.warnings and not (analysis.symbols or analysis.endpoints or analysis.models):
            return None
        fresh_symbols = sorted(analysis.symbols or [], key=lambda s: (int(s.line_start or 0), int(s.line_end or 0)))
        touched = [s for s in fresh_symbols
                   if ranges_intersect(_line_or_none(s.line_start), _line_or_none(s.line_end), ranges)]
        touched_names = {s.name for s in touched}
        touched_endpoints = [e for e in (analysis.endpoints or [])
                             if ranges_intersect(_line_or_none(e.line), _line_or_none(e.line), ranges)]
        touched_models = [m for m in (analysis.models or [])
                          if m.name in touched_names or ranges_intersect(_line_or_none(m.line), _line_or_none(m.line), ranges)]
        by_qname: Dict[str, List[SymbolRow]] = {}
        for s in index_symbols:
            by_qname.setdefault(_qualified(s), []).append(s)
        names: List[str] = []
        before = len(seeds)
        seen: Set[str] = set()
        for s in touched:
            q = _qualified_extracted(s)
            if q not in names:
                names.append(q)
            match = _closest_symbol(by_qname.get(q, []), s)
            if match is not None and match.id is not None:
                key = node_key("symbol", match.id)
                if key not in seen:
                    seen.add(key)
                    seeds.append((key, 1.0, q))
        for e in touched_endpoints:
            match_e = _matching_endpoint(index_endpoints, e)
            if match_e is not None and match_e.id is not None:
                key = node_key("endpoint", match_e.id)
                if key not in seen:
                    seen.add(key)
                    seeds.append((key, 1.0, _endpoint_label(match_e, None)))
        for m in touched_models:
            match_m = _matching_model(index_models, m)
            if match_m is not None and match_m.id is not None:
                key = node_key("model", match_m.id)
                if key not in seen:
                    seen.add(key)
                    seeds.append((key, 1.0, match_m.name))
        file_key = node_key("file", file_row.id)
        if len(seeds) == before:
            # rien de connu du graphe n'est touché (hors symbole, ou symboles tout neufs) : le fichier est la graine
            seeds.append((file_key, 1.0, file_row.path))
        else:
            run.exclude.add(file_key)
        return {"path": ch.path, "status": status, "repo_id": repo_id, "file_id": file_row.id, "old_path": ch.old_path,
                "symbols_changed": names, "ranges": [[a, b] for a, b in ranges], "indexed": True,
                "whole_file": False, "seeds": len(seeds) - before, "reanalyzed": True}

    def _whole_file_entry(self, run: _Run, file_row: FileRow, status: str,
                          seeds: List[Tuple[str, float, str]]) -> Dict[str, Any]:
        """Graines d'un fichier entier : le fichier lui-même, ses symboles, endpoints et modèles."""
        assert file_row.id is not None
        before = len(seeds)
        names: List[str] = []
        seeds.append((node_key("file", file_row.id), 1.0, file_row.path))
        for s in symbols_touched(self.db, file_row.id, []):
            if s.id is not None:
                q = _qualified(s)
                names.append(q)
                seeds.append((node_key("symbol", s.id), 1.0, q))
        for e in self._file_endpoints(file_row.id):
            if e.id is not None:
                seeds.append((node_key("endpoint", e.id), 1.0, _endpoint_label(e, None)))
        for m in self._file_models(file_row.id):
            if m.id is not None:
                seeds.append((node_key("model", m.id), 1.0, m.name))
        return {"path": file_row.path, "status": status, "repo_id": file_row.repo_id, "file_id": file_row.id,
                "old_path": None, "symbols_changed": names, "ranges": [], "indexed": True, "whole_file": True,
                "seeds": len(seeds) - before}

    @staticmethod
    def _looks_indexable(path: str) -> bool:
        """Vrai si le scanner aurait indexé ce chemin (sinon, son absence de l'index est normale)."""
        name = path.rsplit("/", 1)[-1]
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if is_excluded_dir_path(parent) or is_excluded_file_name(name) or has_binary_extension(path):
            return False
        return detect_language(path) is not None

    def _file_endpoints(self, file_id: int) -> List[EndpointRow]:
        rows = self.db.query("SELECT * FROM endpoints WHERE file_id = ? ORDER BY line, id", (file_id,))
        return [EndpointRow.from_row(r) for r in rows]

    def _file_models(self, file_id: int) -> List[ModelRow]:
        rows = self.db.query("SELECT * FROM data_models WHERE file_id = ? ORDER BY line, id", (file_id,))
        return [ModelRow.from_row(r) for r in rows]

    # ------------------------------------------------------------------ propagation

    def _propagate(self, run: _Run, seeds: Sequence[Tuple[str, float, str]]) -> List[ImpactItem]:
        """Parcours en largeur pondéré (meilleur chemin d'abord) depuis les graines."""
        heap: List[Tuple[float, int, int, str, str, List[str]]] = []
        best: Dict[str, _Reach] = {}
        expanded_at: Dict[str, int] = {}
        seed_keys: Set[str] = set()
        seq = 0
        for node, conf, label in seeds:
            info = self._info(run, node)
            if info is None:
                run.warn("Graine ignorée (noeud introuvable) : %s" % node)
                continue
            if node in seed_keys:
                continue
            seed_keys.add(node)
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = 1.0
            c = max(0.0, min(1.0, c))
            heapq.heappush(heap, (-c, 0, seq, node, "", [label or info.label]))
            seq += 1
        run.seed_count = len(seed_keys)
        reached = 0  # noeuds atteints hors graines (plafond NODE_CAP)

        while heap:
            neg, hops, _, node, reason, via = heapq.heappop(heap)
            conf = -neg
            known = best.get(node)
            if known is None:
                if node not in seed_keys:
                    if reached >= NODE_CAP:
                        run.truncated = True
                        continue
                    reached += 1
                best[node] = _Reach(conf, hops, reason, list(via))
            elif conf > known.confidence + 1e-9 or (abs(conf - known.confidence) <= 1e-9 and hops < known.hops):
                best[node] = _Reach(conf, hops, reason, list(via))
            if hops >= run.max_hops:
                continue
            if node in expanded_at and expanded_at[node] <= hops:
                continue
            expanded_at[node] = hops
            run.nodes_visited += 1
            try:
                edges = self._edges(run, node)
            except Exception as exc:  # noqa: BLE001 - un noeud défaillant ne stoppe pas l'analyse
                run.warn("Expansion de %s impossible : %s" % (node, exc))
                continue
            for edge in edges:
                child_conf = conf * edge.factor * DECAY
                if child_conf < MIN_CONFIDENCE or edge.dst == node:
                    continue
                child = self._info(run, edge.dst)
                if child is None:
                    continue
                heapq.heappush(heap, (-child_conf, hops + 1, seq, edge.dst, edge.reason, via + [child.label]))
                seq += 1

        if run.truncated:
            run.warn("Analyse tronquée à %d noeuds : affinez la source ou réduisez la profondeur." % NODE_CAP)
        items: List[ImpactItem] = []
        for node, reach in best.items():
            if node in seed_keys or node in run.exclude:
                continue
            info = self._info(run, node)
            if info is None:
                continue
            items.append(ImpactItem(node=node, type=info.type, name=info.name, path=info.path, line=info.line,
                                    repo_id=info.repo_id, reason=reach.reason, confidence=reach.confidence,
                                    hops=reach.hops, via=reach.via, extra=dict(info.extra)))
        return items

    def _edges(self, run: _Run, node: str) -> List[_Edge]:
        """Arêtes sortantes « vers les dépendants » d'un noeud, avec poids et raison."""
        info = self._info(run, node)
        if info is None or info.type in ("document", "note"):
            return []
        incoming = self.db.relations_to(node)
        outgoing = self.db.relations_from(node)
        edges: List[_Edge] = []
        dependents = [r for r in incoming if r.kind in _DEPENDENT_KINDS]
        client_importers: List[RelationRow] = []
        if info.type == "endpoint" and info.side == "client" and info.file_id is not None:
            client_importers = self.db.relations_to(node_key("file", info.file_id), "imports")
        n_dependents = len(dependents) + len(client_importers)
        is_hub = n_dependents > HUB_THRESHOLD
        if is_hub:
            if node not in run.hubs:
                run.hubs.append(node)
            run.warn("Noeud central non développé : %s a %d dépendants (ses dépendants ne sont pas listés)."
                     % (info.label, n_dependents))

        # --- contenant : symbole / endpoint / modèle → son fichier
        if info.type in ("symbol", "endpoint", "model") and info.file_id is not None:
            if info.type == "symbol":
                reason = "contient le symbole %s" % info.name
            elif info.type == "endpoint":
                reason = ("contient l'appel %s" if info.side == "client" else "contient la route %s") % info.name
            else:
                reason = "contient le modèle %s" % info.name
            edges.append(_Edge(node_key("file", info.file_id), W_CONTAINS, reason))

        # --- documentation (toujours développée, même pour un hub)
        for r in outgoing:
            if r.kind == "documented_by":
                edges.append(_Edge(r.dst, W_DOCUMENTED_BY * _conf(r), "documente %s" % _what(info)))
        for r in incoming:
            if r.kind == "mentions" and (r.src.startswith("document:") or r.src.startswith("note:")):
                edges.append(_Edge(r.src, W_MENTIONS * _conf(r), "mentionne %s" % _what(info)))

        if info.type == "symbol":
            for r in incoming:
                if r.kind == "handles":
                    edges.append(_Edge(r.src, W_HANDLES * _conf(r), "route gérée par %s" % info.name))
            if not is_hub:
                for r in dependents:
                    if r.kind == "calls":
                        edges.append(_Edge(r.src, W_CALLS * _conf(r), "appelle %s" % info.name))
            # classe de modèle : le modèle de données du même nom dans le même fichier
            sym = info.row
            if isinstance(sym, SymbolRow) and (sym.kind or "").lower() in _MODEL_SYMBOL_KINDS:
                for m in self.db.find_models(name=sym.name):
                    if m.id is not None and (m.file_id == sym.file_id or sym.file_id is None):
                        edges.append(_Edge(node_key("model", m.id), W_MODEL_CLASS,
                                           "modèle défini par la classe %s" % info.name))
        elif info.type == "file":
            if not is_hub:
                for r in dependents:
                    if r.kind == "imports":
                        edges.append(_Edge(r.src, W_IMPORTS * _conf(r), "importe %s" % info.path))
        elif info.type == "endpoint":
            if info.side == "client":
                if not is_hub:
                    for r in client_importers:
                        edges.append(_Edge(r.src, W_CLIENT_IMPORTER * _conf(r),
                                           "importe %s qui appelle %s" % (info.path, info.name)))
            else:
                if not is_hub:
                    for r in dependents:
                        if r.kind == "calls_endpoint":
                            caller = self._info(run, r.src)
                            where = caller.path if caller is not None and caller.path else r.src
                            edges.append(_Edge(r.src, _conf(r), "appelle %s depuis %s" % (info.name, where)))
                for r in outgoing:
                    if r.kind == "handles":
                        edges.append(_Edge(r.dst, W_HANDLER * _conf(r), "gère la route %s" % info.name))
        elif info.type == "model":
            for r in outgoing:
                if r.kind == "same_entity":
                    edges.append(_Edge(r.dst, W_SAME_ENTITY * _conf(r), "même entité que %s" % info.name))
                elif r.kind == "defines_table":
                    edges.append(_Edge(r.dst, W_DEFINES_TABLE * _conf(r), "table définie par le modèle %s" % info.name))
            for r in incoming:
                if r.kind == "same_entity":
                    edges.append(_Edge(r.src, W_SAME_ENTITY * _conf(r), "même entité que %s" % info.name))
            if not is_hub:
                for r in dependents:
                    if r.kind == "uses_model":
                        edges.append(_Edge(r.src, W_USES_MODEL * _conf(r), "utilise le modèle %s" % info.name))
        elif info.type == "table":
            for r in incoming:
                if r.kind == "defines_table":
                    edges.append(_Edge(r.src, W_SAME_ENTITY * _conf(r), "définit la table %s" % info.name))
        return edges

    # ------------------------------------------------------------------ résolution des noeuds

    def _file(self, run: _Run, file_id: Optional[int]) -> Optional[FileRow]:
        if file_id is None:
            return None
        if file_id not in run.files:
            run.files[file_id] = self.db.get_file_by_id(int(file_id))
        return run.files[file_id]

    def _info(self, run: _Run, node: str) -> Optional[_NodeInfo]:
        if node in run.infos:
            return run.infos[node]
        info: Optional[_NodeInfo] = None
        try:
            info = self._resolve(run, node)
        except Exception as exc:  # noqa: BLE001
            run.warn("Noeud illisible %s : %s" % (node, exc))
            info = None
        run.infos[node] = info
        return info

    def _resolve(self, run: _Run, node: str) -> Optional[_NodeInfo]:
        if not node or ":" not in node:
            return None
        prefix, ident = node.split(":", 1)
        if prefix == "table":
            name = ident.strip()
            return _NodeInfo(key=node, type="table", name=name, label="table %s" % name,
                             extra={"table_name": name}) if name else None
        try:
            num = int(ident)
        except ValueError:
            return None
        if prefix == "file":
            f = self._file(run, num)
            if f is None:
                return None
            return _NodeInfo(key=node, type="file", name=f.path, label=f.path, path=f.path, repo_id=f.repo_id,
                             file_id=f.id, extra={"id": f.id, "file_id": f.id, "language": f.language, "kind": f.kind},
                             row=f)
        if prefix == "symbol":
            s = self.db.get_symbol(num)
            if s is None:
                return None
            f = self._file(run, s.file_id)
            q = _qualified(s)
            return _NodeInfo(key=node, type="symbol", name=q, label=q, path=f.path if f else None,
                             line=s.line_start, repo_id=f.repo_id if f else None, file_id=s.file_id,
                             extra={"id": s.id, "file_id": s.file_id, "kind": s.kind, "parent": s.parent,
                                    "signature": s.signature, "line_end": s.line_end}, row=s)
        if prefix == "endpoint":
            e = self.db.get_endpoint(num)
            if e is None:
                return None
            f = self._file(run, e.file_id)
            repo_id = f.repo_id if f else None
            name = _endpoint_name(e)
            label = _endpoint_label(e, self._repo_label(repo_id) if e.side == "client" else None)
            return _NodeInfo(key=node, type="endpoint", name=name, label=label, path=f.path if f else None,
                             line=e.line, repo_id=repo_id, file_id=e.file_id, side=e.side,
                             extra={"id": e.id, "file_id": e.file_id, "side": e.side, "method": e.method,
                                    "route": e.path, "normalized_path": e.normalized_path, "handler": e.handler,
                                    "framework": e.framework}, row=e)
        if prefix == "model":
            m = self.db.get_model(num)
            if m is None:
                return None
            f = self._file(run, m.file_id)
            return _NodeInfo(key=node, type="model", name=m.name, label=m.name, path=f.path if f else None,
                             line=m.line, repo_id=f.repo_id if f else None, file_id=m.file_id,
                             extra={"id": m.id, "file_id": m.file_id, "kind": m.kind, "table_name": m.table_name},
                             row=m)
        if prefix == "document":
            d = self.db.get_document(num)
            if d is None:
                return None
            title = d.title or d.source_path or ("document %s" % d.id)
            return _NodeInfo(key=node, type="document", name=title, label=title,
                             path=d.source_path or d.stored_path,
                             extra={"id": d.id, "format": d.format, "kind": d.kind}, row=d)
        if prefix == "note":
            n = self.db.get_note(num)
            if n is None:
                return None
            title = n.title or ("note %s" % n.id)
            return _NodeInfo(key=node, type="note", name=title, label="note « %s »" % title,
                             extra={"id": n.id, "note_date": n.note_date, "tags": list(n.tags or [])}, row=n)
        return None

    # ------------------------------------------------------------------ complément lexical

    def _lexical_complement(self, run: _Run, names: Sequence[str], items: List[ImpactItem]) -> List[ImpactItem]:
        """Ajoute les documents/notes dont le texte mentionne un symbole changé (confiance 0.5)."""
        if self.search_engine is None or not names:
            return items
        known: Set[str] = {i.node for i in items}
        added = 0
        seen_names: Set[str] = set()
        for name in names:
            if added >= LEXICAL_LIMIT:
                break
            query = (name or "").strip()
            if not query or query in seen_names:
                continue
            seen_names.add(query)
            for hit in self._search(run, query):
                if added >= LEXICAL_LIMIT:
                    break
                st = _hit_field(hit, "source_type")
                sid = _hit_field(hit, "source_id")
                if st not in ("document", "note") or sid is None:
                    continue
                try:
                    node = node_key(st, int(sid))
                except (TypeError, ValueError):
                    continue
                if node in known or node in run.exclude:
                    continue
                info = self._info(run, node)
                if info is None:
                    continue
                known.add(node)
                added += 1
                extra = dict(info.extra)
                extra["channel"] = "lexical"
                snippet = _hit_field(hit, "snippet")
                if snippet:
                    extra["snippet"] = str(snippet)
                items.append(ImpactItem(node=node, type=info.type, name=info.name, path=info.path, line=info.line,
                                        repo_id=info.repo_id, reason="mentionne %s" % query,
                                        confidence=LEXICAL_CONFIDENCE, hops=1, via=[query, info.label], extra=extra))
        return items

    def _search(self, run: _Run, query: str) -> List[Any]:
        """Résultats (documents et notes) du moteur de recherche, quelle que soit sa forme.

        Accepte ``search(query, limit=, source_types=)`` (:class:`develop.search.SearchEngine`, qui
        renvoie un ``SearchResult`` à attribut ``hits``), ``search(query, limit=, types=)``,
        ``search(query)``, ``fts_search(query, limit=, source_types=)`` (une :class:`Database`) ou un
        simple appelable ; une liste ou un objet à attribut ``hits`` sont tous deux acceptés.
        """
        se = self.search_engine
        if se is None:
            return []
        try:
            search = getattr(se, "search", None)
            if callable(search):
                result: Any = None
                for kwargs in ({"limit": LEXICAL_LIMIT, "source_types": ["document", "note"]},
                               {"limit": LEXICAL_LIMIT, "types": ["document", "note"]}, {}):
                    try:
                        result = search(query, **kwargs)
                        break
                    except TypeError:
                        continue
                return _as_hits(result)
            fts = getattr(se, "fts_search", None)
            if callable(fts):
                return _as_hits(fts(query, limit=LEXICAL_LIMIT, source_types=["document", "note"]))
            if callable(se):
                return _as_hits(se(query))
        except Exception as exc:  # noqa: BLE001
            run.warn("Recherche lexicale indisponible : %s" % exc)
        return []

    # ------------------------------------------------------------------ rapport

    @staticmethod
    def _sorted(items: Iterable[ImpactItem]) -> List[ImpactItem]:
        return sorted(items, key=lambda i: (-round(i.confidence, 6), i.hops, _TYPE_ORDER.get(i.type, 9),
                                            (i.name or "").lower(), i.node))

    def _finish(self, run: _Run, source: Dict[str, Any], changed: List[Dict[str, Any]],
                items: List[ImpactItem]) -> ImpactReport:
        affected = self._sorted(items)
        by_type: Dict[str, int] = {}
        for i in affected:
            by_type[i.type] = by_type.get(i.type, 0) + 1
        stats: Dict[str, Any] = {
            "changed_files": len(changed),
            "seeds": run.seed_count,
            "affected_total": len(affected),
            "by_type": by_type,
            "hubs_skipped": len(run.hubs),
            "max_hops": run.max_hops,
            "nodes_visited": run.nodes_visited,
            "truncated": run.truncated,
        }
        return ImpactReport(
            source=source,
            changed=changed,
            affected=affected,
            endpoints=[i for i in affected if i.type == "endpoint" and i.extra.get("side") != "client"],
            models=[i for i in affected if i.type in ("model", "table")],
            frontend_calls=[i for i in affected if i.type == "endpoint" and i.extra.get("side") == "client"],
            # les fichiers de documentation des dépôts (README, docs/*.md : ``files.kind == 'doc'``) sont
            # des documents pour l'utilisateur ; ils gardent ``type='file'`` (lien vers le fichier)
            documents=[i for i in affected if i.type == "document"
                       or (i.type == "file" and i.extra.get("kind") == "doc")],
            notes=[i for i in affected if i.type == "note"],
            stats=stats,
            warnings=list(run.warnings),
            stale_index=run.stale_index,
            stale_files=list(run.stale_files),
        )


# --------------------------------------------------------------------------- petites aides

def _conf(r: RelationRow) -> float:
    try:
        c = float(r.confidence if r.confidence is not None else 1.0)
    except (TypeError, ValueError):
        c = 1.0
    return max(0.0, min(1.0, c))


def _qualified(s: SymbolRow) -> str:
    return "%s.%s" % (s.parent, s.name) if s.parent else s.name


def _endpoint_name(e: EndpointRow) -> str:
    method = (e.method or "ANY").upper()
    return "%s %s" % (method, e.normalized_path or e.path or "/")


def _endpoint_label(e: EndpointRow, repo_label: Optional[str]) -> str:
    """Libellé court pour la chaîne ``via`` : ``POST /orders/:param/confirm`` ou ``confirmOrder (frontend)``."""
    if e.side == "client":
        base = e.handler or _endpoint_name(e)
        return "%s (%s)" % (base, repo_label or "client")
    return _endpoint_name(e)


def _what(info: _NodeInfo) -> str:
    """Complément d'objet pour « documente … » / « mentionne … »."""
    if info.type == "table":
        return "la table %s" % info.name
    if info.type == "model":
        return "le modèle %s" % info.name
    if info.type == "endpoint":
        return "la route %s" % info.name if info.side != "client" else "l'appel %s" % info.name
    if info.type == "file":
        return "le fichier %s" % info.name
    return info.name


def _within_repo(ch: ChangedFile, prefix: str) -> Optional[ChangedFile]:
    """Chemin Git (relatif à la racine Git) → chemin relatif au dépôt configuré, en NFC ; ``None`` hors du dépôt.

    Git et le chemin configuré peuvent écrire un même nom accentué sous deux formes (NFD sur
    un disque macOS, NFC recomposé par git) : tout est comparé et renvoyé en NFC, la forme
    sous laquelle le scanner indexe les fichiers.
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


def _new_side_rev(rev_range: Optional[str]) -> Optional[str]:
    """Révision du côté nouveau d'une plage ``A..B`` / ``A...B`` (``HEAD`` si ``B`` est vide).

    ``None`` quand le côté nouveau est l'arbre de travail : pas de plage, ou une seule révision
    (``git diff A`` compare ``A`` à l'arbre de travail).
    """
    if not rev_range or ".." not in rev_range:
        return None
    sep = "..." if "..." in rev_range else ".."
    right = rev_range.split(sep, 1)[1].strip()
    return right or "HEAD"


def _disk_path(repo_path: str, rel: str) -> str:
    """Chemin disque d'un fichier du dépôt : la clé NFC de l'index ou, si c'est elle qui existe, sa forme NFD
    (nom brut d'un fichier accentué sur un système qui ne normalise pas)."""
    candidate = os.path.join(repo_path, *rel.split("/"))
    if os.path.exists(candidate):
        return candidate
    nfd = unicodedata.normalize("NFD", rel)
    alt = os.path.join(repo_path, *nfd.split("/"))
    return alt if nfd != rel and os.path.exists(alt) else candidate


def _git_show_bytes(git: GitRepo, path: str, rev: str) -> Optional[bytes]:
    """Contenu brut de ``rev:path`` (``git show``, lecture seule) ; ``None`` en cas d'échec."""
    args = ["git", "-C", str(git.root), "show", "%s:%s" % (rev, path)]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C.UTF-8")
    try:
        out = subprocess.run(args, capture_output=True, timeout=60, check=False, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _reanalyzed_warning(paths: Sequence[str]) -> str:
    if len(paths) == 1:
        return ("Index obsolète : les symboles de %s ont été relus depuis son contenu actuel (lignes décalées). "
                "Réindexez pour une analyse exacte." % paths[0])
    shown = ", ".join(paths[:3]) + ("…" if len(paths) > 3 else "")
    return ("Index obsolète : les symboles de %d fichiers (%s) ont été relus depuis leur contenu actuel. "
            "Réindexez pour une analyse exacte." % (len(paths), shown))


def _qualified_extracted(s: ExtractedSymbol) -> str:
    return "%s.%s" % (s.parent, s.name) if s.parent else s.name


def _line_or_none(value: Any) -> Optional[int]:
    """Numéro de ligne exploitable, ou ``None`` (inconnu : l'élément est alors considéré comme touché)."""
    try:
        n = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def _closest_symbol(candidates: Sequence[SymbolRow], fresh: ExtractedSymbol) -> Optional[SymbolRow]:
    """Parmi des symboles indexés de même nom qualifié, celui de même genre le plus proche en position."""
    if not candidates:
        return None
    same_kind = [c for c in candidates if (c.kind or "").lower() == (fresh.kind or "").lower()]
    pool = same_kind or list(candidates)
    return min(pool, key=lambda c: abs(int(c.line_start or 0) - int(fresh.line_start or 0)))


def _matching_endpoint(candidates: Sequence[EndpointRow], fresh: ExtractedEndpoint) -> Optional[EndpointRow]:
    """Route indexée de même côté et méthode dont le chemin normalisé concorde (à défaut, même gestionnaire)."""
    side = str(getattr(fresh, "side", "") or "")
    method = str(fresh.method or "ANY").upper()
    meta = fresh.meta or {}
    full_path = meta.get("full_path")
    source = full_path if isinstance(full_path, str) and full_path.strip() else (fresh.path or "")
    norm = normalize_route_path(source)
    same = [c for c in candidates if (c.side or "") == side and (c.method or "ANY").upper() == method]
    for c in same:
        if (c.normalized_path or normalize_route_path(c.path or "")) == norm:
            return c
    for c in same:
        cp = c.normalized_path or normalize_route_path(c.path or "")
        if route_paths_match(cp, norm) or route_paths_match(norm, cp):
            return c
    handler = (fresh.handler or "").strip()
    if handler:
        for c in same:
            if (c.handler or "").strip() == handler:
                return c
    return None


def _matching_model(candidates: Sequence[ModelRow], fresh: ExtractedModel) -> Optional[ModelRow]:
    same_name = [c for c in candidates if c.name == fresh.name]
    if not same_name:
        return None
    same_kind = [c for c in same_name if (c.kind or "") == (fresh.kind or "")]
    return (same_kind or same_name)[0]


def _as_hits(result: Any) -> List[Any]:
    """Liste de résultats depuis une liste, un ``SearchResult`` (attribut ``hits``) ou ``None``."""
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        return list(result)
    hits = getattr(result, "hits", None)
    if hits is not None:
        return list(hits)
    return list(result)


def _hit_field(hit: Any, name: str) -> Any:
    if isinstance(hit, dict):
        return hit.get(name)
    return getattr(hit, name, None)
