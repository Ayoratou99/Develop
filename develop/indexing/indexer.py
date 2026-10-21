"""Indexeur incrémental de la mémoire Develop.

Un *run* (:meth:`Indexer.run`) enchaîne, dans l'ordre :

1. synchronisation de la table ``repos`` avec la configuration de l'espace de travail
   (dépôts retirés de la configuration → supprimés de la base avec tout ce qui en dépend) ;
2. pour chaque dépôt activé : parcours en lecture seule (:func:`scan_repo`), comparaison avec
   les signatures en base (``(taille, mtime)`` identiques → inchangé sans lecture ; sinon
   sha1 ; sha1 identique → simple mise à jour du mtime ; sinon réindexation), suppression
   des fichiers disparus — mais **jamais** d'un fichier déjà indexé devenu simplement
   inaccessible (droits retirés, dossier parent impossible à lister) : il est conservé tel
   quel et signalé par une erreur, comme un dépôt dont le dossier est introuvable ;
3. documents déposés dont les morceaux manquent (ou tous si ``full``) ;
4. notes sans morceaux (ou toutes si ``full``) ;
5. relations : appel du *linker* (``develop.knowledge_graph.linker.link_all`` par défaut) ;
6. embeddings manquants (si un modèle et un moteur sont configurés) ;
7. ``head_sha`` / ``last_indexed_at`` des dépôts (via Git si le dossier en est un) ;
8. métadonnées ``extractor_version`` et ``last_index_at``.

Robustesse : une erreur sur un fichier, un document ou une note est consignée
(:attr:`Indexer.errors`) et n'arrête jamais l'ensemble ; le run se termine en ``partial``.
Les dépôts de l'utilisateur ne sont **jamais** modifiés. Les fichiers ``.env`` réels sont
enregistrés sans leur contenu (secrets).
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from develop.code_analysis import FileAnalysis, analyze_file, normalize_route_path
from develop.config import RepoConfig, WorkspaceConfig
from develop.document_indexer import (
    SUPPORTED_FORMATS,
    Chunk,
    LoadedDocument,
    chunk_for_file,
    chunk_text,
    extract_markdown_tables,
    format_for_path,
    load_document,
    page_for_offset,
)
from develop.git_tools import GitError, GitRepo, find_git_root
from develop.indexing.embedder import Embedder
from develop.llm.base import LLMClient
from develop.project_scanner import FileEntry, ScanResult, compute_sha1, read_text, scan_repo
from develop.storage import (
    ChunkRow,
    Database,
    EndpointRow,
    FileRow,
    IndexRunRow,
    ModelRow,
    RelationRow,
    RepoRow,
    SymbolRow,
    node_key,
)

#: Version des extracteurs : stockée dans meta ``extractor_version`` ; si elle diffère au
#: démarrage d'un run, tous les fichiers sont réextraits (pas les documents ni les notes).
EXTRACTOR_VERSION = "1"

#: Phases d'un run, dans l'ordre d'apparition dans les rappels de progression.
PHASES: Tuple[str, ...] = ("scan", "analyze", "documents", "notes", "relations", "embeddings", "done")

#: Fréquence (en fichiers) des rappels de progression et des contrôles d'annulation.
PROGRESS_EVERY = 25
#: Taille des transactions groupées (fichiers par ``BEGIN``/``COMMIT``).
BATCH_FILES = PROGRESS_EVERY
#: Plafond des erreurs conservées dans les statistiques d'un run.
MAX_ERRORS = 200
MAX_WARNINGS = 200
#: Une note est un seul morceau, sauf au-delà de cette taille (caractères).
NOTE_SPLIT_CHARS = 4000
#: Un fichier ``.env`` dont le nom contient l'un de ces mots est un gabarit sans secret.
ENV_TEMPLATE_MARKERS = ("example", "sample", "template", "dist")
#: Genres de modèles qui définissent (ou matérialisent) une table.
TABLE_MODEL_KINDS = ("migration", "sql_table", "eloquent")
_LARGE_LIMIT = 10_000_000


@dataclass
class IndexProgress:
    """État d'avancement transmis au rappel ``progress`` (voir :data:`PHASES`)."""

    phase: str
    current: int
    total: int
    repo: str = ""
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndexCancelled(Exception):
    """Annulation demandée via ``cancel_event`` ; convertie en statut ``cancelled`` par :meth:`Indexer.run`."""


# --------------------------------------------------------------------------- utilitaires

def _batches(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _line_offsets(text: str) -> List[int]:
    """Position (caractères) du début de chaque ligne, cohérente avec ``text.splitlines()``."""
    offsets: List[int] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        offsets.append(pos)
        pos += len(line)
    return offsets


def _is_secret_env(entry: FileEntry) -> bool:
    """Vrai pour un ``.env`` réel (``.env``, ``.env.local``, ``prod.env``…), faux pour un gabarit (``.env.example``)."""
    if entry.language != "env":
        return False
    name = entry.path.rsplit("/", 1)[-1].lower()
    return not any(marker in name for marker in ENV_TEMPLATE_MARKERS)


def _chunk_mode_for_format(fmt: Optional[str]) -> str:
    if fmt == "markdown":
        return "markdown"
    if fmt in ("csv", "json", "yaml", "xml", "sql"):
        return "data"
    return "prose"


def _as_dict(value: Any) -> Dict[str, Any]:
    """Champ ou relation de modèle → dict JSON (tolère dataclasses, dicts et valeurs brutes)."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return {"name": str(value)}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    return str(value)


def _split_handler(handler: str) -> Tuple[Optional[str], str]:
    """``"OrderController@index"`` → ``("OrderController", "index")`` ; ``"index"`` → ``(None, "index")``."""
    h = (handler or "").strip().strip("'\"")
    for sep in ("@", "::", "#"):
        if sep in h:
            cls, meth = h.rsplit(sep, 1)
            return cls.split("\\")[-1].split(".")[-1] or None, meth
    if "." in h and not h.endswith("."):
        cls, meth = h.rsplit(".", 1)
        return cls.split("\\")[-1].split(".")[-1] or None, meth
    return None, h.split("\\")[-1]


def _find_handler_symbol(handler: str, symbols: Sequence[SymbolRow]) -> Optional[SymbolRow]:
    """Symbole du même fichier qui implémente ``handler`` (``Classe@méthode`` ou nom de fonction)."""
    cls, meth = _split_handler(handler)
    if not meth:
        return None
    if cls:
        for s in symbols:
            parent = (s.parent or "").split("\\")[-1]
            if s.name == meth and parent == cls:
                return s
    # fonction de module (ou méthode unique portant ce nom)
    candidates = [s for s in symbols if s.name == meth and s.kind in ("function", "method", "component", "view", "class")]
    if not candidates:
        return None
    top = [s for s in candidates if not s.parent]
    if top:
        return top[0]
    return candidates[0] if len(candidates) == 1 or cls is None else None


# --------------------------------------------------------------------------- indexeur

class Indexer:
    """Indexation incrémentale d'un espace de travail (synchrone : l'API l'exécute dans un thread).

    * ``llm`` + ``embedding_model`` : nécessaires seulement pour les embeddings ;
    * ``progress`` : rappel appelé à chaque phase et tous les :data:`PROGRESS_EVERY` fichiers ;
    * ``cancel_event`` : vérifié aux mêmes points ; un run annulé se termine en statut ``cancelled`` ;
    * ``linker`` : ``callable(db, workspace)`` qui construit le graphe de relations ; par défaut
      ``develop.knowledge_graph.linker.link_all`` importé paresseusement (absent → avertissement).
    """

    def __init__(self, db: Database, workspace: WorkspaceConfig, llm: Optional[LLMClient] = None,
                 embedding_model: str = "", progress: Optional[Callable[[IndexProgress], None]] = None,
                 cancel_event: Optional[threading.Event] = None,
                 linker: Optional[Callable[[Database, WorkspaceConfig], Any]] = None):
        self.db = db
        self.workspace = workspace
        self.llm = llm
        self.embedding_model = (embedding_model or "").strip()
        self.progress_callback = progress
        self.cancel_event = cancel_event
        self.linker = linker
        self.errors: List[Dict[str, str]] = []
        self.warnings: List[str] = []
        self.errors_total = 0
        self._stats: Dict[str, Any] = self._new_stats(False)
        self._repo_names: Dict[str, str] = {r.id: (r.name or r.id) for r in workspace.repos}

    # ------------------------------------------------------------------ journalisation
    def _error(self, path: str, stage: str, message: str) -> None:
        self.errors_total += 1
        if len(self.errors) < MAX_ERRORS:
            self.errors.append({"path": path or "", "stage": stage, "message": str(message)})
        elif len(self.errors) == MAX_ERRORS:
            self.errors.append({"path": "", "stage": stage, "message": "… d'autres erreurs ont été tronquées."})

    def _warn(self, message: str) -> None:
        if len(self.warnings) < MAX_WARNINGS:
            self.warnings.append(str(message))
        elif len(self.warnings) == MAX_WARNINGS:
            self.warnings.append("… d'autres avertissements ont été tronqués.")

    def _skip(self, stats: Dict[str, Any], reason: str, repo_id: Optional[str] = None) -> None:
        skipped = stats.setdefault("files_skipped", {})
        skipped[reason] = skipped.get(reason, 0) + 1
        if repo_id is not None:
            rs = stats.setdefault("repos", {}).get(repo_id)
            if rs is not None:
                rskip = rs.setdefault("files_skipped", {})
                rskip[reason] = rskip.get(reason, 0) + 1

    def _progress(self, phase: str, current: int, total: int, repo: str = "", message: str = "") -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(IndexProgress(phase=phase, current=int(current), total=int(total), repo=repo,
                                                 message=message))
        except Exception:  # noqa: BLE001 - un rappel d'interface ne doit jamais casser l'indexation
            pass

    def _check_cancel(self) -> None:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise IndexCancelled("Indexation annulée par l'utilisateur.")

    @staticmethod
    def _new_stats(full: bool) -> Dict[str, Any]:
        return {
            "full": bool(full), "extractor_version": EXTRACTOR_VERSION,
            "files_scanned": 0, "files_indexed": 0, "files_unchanged": 0, "files_deleted": 0, "files_skipped": {},
            "files_kept_unreadable": 0,
            "symbols": 0, "endpoints": 0, "models": 0, "chunks": 0, "documents": 0, "notes": 0,
            "relations": 0, "embeddings": 0, "embeddings_pending": 0,
            "errors": [], "warnings": [], "duration_s": 0.0, "repos": {}, "repos_removed": [],
        }

    # ------------------------------------------------------------------ run
    def run(self, full: bool = False, repo_id: Optional[str] = None, with_embeddings: bool = True) -> IndexRunRow:
        """Exécute un run complet (voir le module) et renvoie la ligne ``index_runs`` finale.

        Ne lève jamais : statut ``ok`` | ``partial`` (erreurs consignées) | ``cancelled`` | ``failed``.
        """
        started = time.time()
        self.errors = []
        self.warnings = []
        self.errors_total = 0
        stats = self._new_stats(full)
        self._stats = stats
        self._repo_names = {r.id: (r.name or r.id) for r in self.workspace.repos}
        run_id = self.db.start_index_run()
        status = "ok"
        error_text: Optional[str] = None
        try:
            self._run_phases(stats, full, repo_id, with_embeddings)
            status = "partial" if self.errors else "ok"
        except IndexCancelled:
            status = "cancelled"
            self._warn("Indexation annulée par l'utilisateur.")
        except Exception as exc:  # noqa: BLE001 - jamais d'exception hors de run()
            status = "failed"
            error_text = "%s : %s" % (exc.__class__.__name__, exc)
            self._error("", "run", error_text)
        stats["errors"] = list(self.errors)
        stats["errors_total"] = self.errors_total
        stats["warnings"] = list(self.warnings)
        stats["duration_s"] = round(time.time() - started, 3)
        try:
            stats["totals"] = self.db.stats()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.db.finish_index_run(run_id, status, stats, error_text)
        except Exception as exc:  # noqa: BLE001
            error_text = (error_text or "") + " | enregistrement du run impossible : %s" % exc
        self._progress("done", stats["files_indexed"], stats["files_scanned"], "",
                       {"ok": "Indexation terminée.", "partial": "Indexation terminée avec des erreurs.",
                        "cancelled": "Indexation annulée.", "failed": "Indexation échouée."}[status])
        return self._run_row(run_id, status, stats, error_text, started)

    def _run_row(self, run_id: int, status: str, stats: Dict[str, Any], error: Optional[str],
                 started: float) -> IndexRunRow:
        try:
            rows = self.db.query("SELECT * FROM index_runs WHERE id = ?", (run_id,))
            if rows:
                return IndexRunRow.from_row(rows[0])
        except Exception:  # noqa: BLE001
            pass
        return IndexRunRow(id=run_id, started_at=started, finished_at=time.time(), status=status, stats=stats,
                           error=error)

    def _run_phases(self, stats: Dict[str, Any], full: bool, repo_id: Optional[str], with_embeddings: bool) -> None:
        stored_version = self.db.get_meta("extractor_version")
        force_files = bool(full)
        if stored_version != EXTRACTOR_VERSION:
            if stored_version is not None:
                self._warn("Version des extracteurs passée de %s à %s : réextraction complète des fichiers."
                           % (stored_version, EXTRACTOR_VERSION))
            force_files = True
        stats["reextract"] = force_files

        # (1) dépôts
        self._sync_repos(stats)
        # (2) fichiers
        targets = self._target_repos(repo_id, stats)
        for repo in targets:
            self._check_cancel()
            self._index_repo(repo, force_files, stats)
        # (3) documents
        self._check_cancel()
        self._index_documents(full, stats)
        # (4) notes
        self._check_cancel()
        self._index_notes(full, stats)
        # (5) relations
        self._check_cancel()
        self._link(stats)
        # (6) embeddings
        self._check_cancel()
        self._embed(with_embeddings, stats)
        # (7) état Git des dépôts traités
        self._finalize_repos(targets, stats)
        # (8) métadonnées
        self.db.set_meta("extractor_version", EXTRACTOR_VERSION)
        self.db.set_meta("last_index_at", "%.3f" % time.time())

    # ------------------------------------------------------------------ (1) dépôts
    def _sync_repos(self, stats: Dict[str, Any]) -> None:
        existing = {r.id: r for r in self.db.list_repos()}
        wanted = {r.id for r in self.workspace.repos}
        with self.db.transaction():
            for repo in self.workspace.repos:
                old = existing.get(repo.id)
                self.db.upsert_repo(RepoRow(id=repo.id, name=repo.name or repo.id, path=repo.path, kind=repo.kind,
                                            head_sha=old.head_sha if old else None,
                                            last_indexed_at=old.last_indexed_at if old else None))
            for rid in sorted(existing):
                if rid not in wanted:
                    self.db.delete_repo(rid)
                    stats["repos_removed"].append(rid)
                    self._warn("Dépôt retiré de la configuration, supprimé de la mémoire : %s" % existing[rid].name)

    def _target_repos(self, repo_id: Optional[str], stats: Dict[str, Any]) -> List[RepoConfig]:
        if repo_id is not None:
            repo = self.workspace.repo_by_id(repo_id)
            if repo is None:
                self._error("", "scan", "Dépôt inconnu dans la configuration de l'espace : %s" % repo_id)
                return []
            return [repo]
        targets: List[RepoConfig] = []
        for repo in self.workspace.repos:
            if repo.enabled:
                targets.append(repo)
            else:
                stats["repos"][repo.id] = {"name": repo.name, "path": repo.path, "status": "disabled"}
        return targets

    # ------------------------------------------------------------------ (2) fichiers
    def _index_repo(self, repo: RepoConfig, force: bool, stats: Dict[str, Any]) -> None:
        rstats: Dict[str, Any] = {"name": repo.name, "path": repo.path, "status": "ok", "files_scanned": 0,
                                  "files_indexed": 0, "files_unchanged": 0, "files_deleted": 0, "files_skipped": {},
                                  "files_kept_unreadable": 0}
        stats["repos"][repo.id] = rstats
        self._progress("scan", 0, 0, repo.id, "Parcours de %s…" % (repo.name or repo.path))
        if not os.path.isdir(repo.path):
            msg = "Dossier introuvable ou inaccessible : %s" % repo.path
            self._error(repo.path, "scan", msg)
            rstats["status"] = "error"
            rstats["error"] = msg
            return
        try:
            scan = scan_repo(repo.path, repo.id, exclude=list(repo.exclude or []))
        except Exception as exc:  # noqa: BLE001 - le scanner ne lève pas, par sécurité
            msg = "Parcours impossible : %s" % exc
            self._error(repo.path, "scan", msg)
            rstats["status"] = "error"
            rstats["error"] = msg
            return
        for w in scan.warnings:
            self._warn("[%s] %s" % (repo.name or repo.id, w))
        rstats["used_git"] = scan.used_git
        if "" in scan.unreadable_dirs:
            # la racine existe mais ne peut pas être listée : même protection qu'un dossier introuvable
            msg = ("Dossier inaccessible (impossible à lister, droits insuffisants ?) : %s ; "
                   "index du dépôt conservé tel quel." % repo.path)
            self._error(repo.path, "scan", msg)
            rstats["status"] = "error"
            rstats["error"] = msg
            return
        for reason, count in scan.skipped.items():
            if count:
                rstats["files_skipped"][reason] = rstats["files_skipped"].get(reason, 0) + count
                stats["files_skipped"][reason] = stats["files_skipped"].get(reason, 0) + count
        total = len(scan.files)
        rstats["files_scanned"] = total
        stats["files_scanned"] += total

        # fichiers déjà indexés devenus inaccessibles : conservés tels quels, une erreur chacun
        signatures = self.db.file_signatures(repo.id)
        scanned_paths = {e.path for e in scan.files}
        kept = self._unreadable_known_paths(repo, scan, signatures, scanned_paths)
        for path in sorted(kept):
            self._error(path, "read", "%s : conservé dans l'index tel quel, non réindexé." % kept[path])
        for path in scan.unreadable:
            if path not in signatures:
                self._warn("[%s] Fichier illisible, ignoré : %s" % (repo.name or repo.id, path))

        # décision par fichier, sans lecture quand (taille, mtime) n'ont pas bougé
        to_index: List[Tuple[FileEntry, Optional[str]]] = []
        to_touch: List[Tuple[FileEntry, str]] = []
        unchanged = 0
        for entry in scan.files:
            sig = signatures.get(entry.path)
            if sig is None or force:
                to_index.append((entry, None))
                continue
            sha1, mtime, size = sig
            if sha1 and size == entry.size and mtime is not None and float(mtime) == float(entry.mtime):
                unchanged += 1
                continue
            current = compute_sha1(entry.abs_path)
            if current and current == sha1:
                to_touch.append((entry, current))
                unchanged += 1
            else:
                to_index.append((entry, current or None))

        if to_touch:
            with self.db.transaction():
                for entry, sha1 in to_touch:
                    self.db.upsert_file(FileRow(repo_id=repo.id, path=entry.path, language=entry.language,
                                                kind=entry.kind, size=entry.size, mtime=entry.mtime, sha1=sha1))

        self._progress("analyze", 0, len(to_index), repo.id,
                       "%d fichier(s) à analyser, %d inchangé(s)" % (len(to_index), unchanged))
        done = 0
        indexed = 0
        for batch in _batches(to_index, BATCH_FILES):
            self._check_cancel()
            with self.db.transaction():
                for entry, sha1 in batch:
                    if self._index_file(repo.id, entry, None, sha1, stats) is not None:
                        indexed += 1
                    done += 1
            self._progress("analyze", done, len(to_index), repo.id, batch[-1][0].path)

        deleted = self.db.delete_files_not_in(repo.id, scanned_paths | set(kept))
        rstats["files_indexed"] = indexed
        rstats["files_unchanged"] = unchanged
        rstats["files_deleted"] = deleted
        rstats["files_kept_unreadable"] = len(kept)
        stats["files_indexed"] += indexed
        stats["files_unchanged"] += unchanged
        stats["files_deleted"] += deleted
        stats["files_kept_unreadable"] += len(kept)

    @staticmethod
    def _unreadable_known_paths(repo: RepoConfig, scan: ScanResult, signatures: Dict[str, Any],
                                scanned_paths: Set[str]) -> Dict[str, str]:
        """Fichiers en base absents du parcours parce qu'ils sont devenus inaccessibles → ``{chemin: cause}``.

        Trois indices, du moins coûteux au plus coûteux : le scanner l'a compté ``unreadable`` ;
        il est sous un dossier que le scanner n'a pas pu lister ; ``os.lstat`` échoue par manque
        de droits (dossier parent fermé en mode git, où le scanner ne parcourt pas l'arborescence).
        Tout autre fichier absent du parcours a réellement disparu, ou n'est plus indexable
        (exclu, binaire, trop gros, lien), et sera supprimé de l'index.
        """
        unreadable = set(scan.unreadable)
        closed_dirs = [d + "/" for d in scan.unreadable_dirs if d]
        kept: Dict[str, str] = {}
        for path in signatures:
            if path in scanned_paths:
                continue
            if path in unreadable:
                kept[path] = "Fichier devenu illisible (droits insuffisants ?)"
            elif any(path.startswith(d) for d in closed_dirs):
                kept[path] = "Dossier parent impossible à lister (droits insuffisants ?)"
            else:
                try:
                    os.lstat(os.path.join(repo.path, *path.split("/")))
                except PermissionError:
                    kept[path] = "Fichier inaccessible (droits insuffisants ?)"
                except OSError:
                    pass  # réellement disparu
        return kept

    def index_file(self, repo_id: str, entry: FileEntry, text: Optional[str] = None) -> Optional[int]:
        """Indexe un fichier (lecture, analyse, découpage, stockage) ; renvoie son id ou ``None``.

        Ne lève jamais : les erreurs sont consignées dans :attr:`errors`. ``text`` permet de
        fournir un contenu déjà lu.
        """
        try:
            return self._index_file(repo_id, entry, text, None, self._stats)
        except IndexCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            self._error(entry.path, "index_file", "%s : %s" % (exc.__class__.__name__, exc))
            return None

    def _index_file(self, repo_id: str, entry: FileEntry, text: Optional[str], sha1: Optional[str],
                    stats: Dict[str, Any]) -> Optional[int]:
        path = entry.path
        repo_name = self._repo_names.get(repo_id) or repo_id
        loaded: Optional[LoadedDocument] = None

        # --- lecture
        if text is None:
            if entry.is_binary or entry.language in ("pdf", "docx"):
                try:
                    loaded = load_document(entry.abs_path)
                except Exception as exc:  # noqa: BLE001
                    self._error(path, "read", "Extraction impossible : %s" % exc)
                    self._skip(stats, "unreadable", repo_id)
                    return None
                for w in loaded.warnings:
                    self._warn("%s : %s" % (path, w))
                text = loaded.text or ""
            else:
                text = read_text(entry.abs_path)
                if text is None:
                    self._error(path, "read", "Fichier illisible ou binaire : non indexé.")
                    self._skip(stats, "unreadable", repo_id)
                    return None
        if sha1 is None:
            sha1 = compute_sha1(entry.abs_path) or hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        row = FileRow(repo_id=repo_id, path=path, language=entry.language, kind=entry.kind, size=entry.size,
                      mtime=entry.mtime, sha1=sha1)

        # --- .env réel : le fichier existe dans l'index, jamais son contenu
        if _is_secret_env(entry):
            row.kind = "config"
            self._skip(stats, "env_secret", repo_id)
            return self._store(row, [], [], [], [], FileAnalysis(language="env"), stats)

        # --- analyse statique (code, et SQL pour les tables)
        analysis = FileAnalysis(language=entry.language or "")
        if text.strip() and (entry.kind == "code" or entry.language == "sql"):
            try:
                analysis = analyze_file(text, entry.language, path)
            except Exception as exc:  # noqa: BLE001 - analyze_file ne lève pas, par sécurité
                self._error(path, "analyze", "Échec de l'analyse : %s" % exc)
                analysis = FileAnalysis(language=entry.language or "")

        # --- découpage
        chunk_rows: List[ChunkRow] = []
        if text.strip():
            try:
                chunks = chunk_for_file(text, entry.language, entry.kind)
            except Exception as exc:  # noqa: BLE001
                self._error(path, "chunk", "Échec du découpage : %s" % exc)
                chunks = []
            page_offsets = loaded.meta.get("page_offsets") if loaded is not None else None
            chunk_rows = self._file_chunk_rows(chunks, text, repo_name, path,
                                               with_header=(entry.kind == "code" or entry.language == "sql"),
                                               page_offsets=page_offsets if isinstance(page_offsets, list) else None)
        else:
            self._skip(stats, "empty", repo_id)

        symbols, endpoints, models = self._map_analysis(analysis, path)
        return self._store(row, symbols, endpoints, models, chunk_rows, analysis, stats)

    def _file_chunk_rows(self, chunks: Sequence[Chunk], text: str, repo_name: str, path: str, with_header: bool,
                         page_offsets: Optional[List[int]] = None) -> List[ChunkRow]:
        offsets = _line_offsets(text) if page_offsets else []
        rows: List[ChunkRow] = []
        for c in chunks:
            meta = dict(c.meta or {})
            body = c.text
            if with_header:
                hint = meta.get("symbol_hint")
                header = "# %s/%s L%d-%d%s\n" % (repo_name, path, c.line_start, c.line_end,
                                                   (" " + str(hint)) if hint else "")
                meta["body_offset"] = len(header)
                body = header + body
            if page_offsets and offsets:
                idx = min(max(c.line_start - 1, 0), len(offsets) - 1)
                meta["page"] = page_for_offset(page_offsets, offsets[idx])
            rows.append(ChunkRow(text=body, ordinal=c.ordinal, line_start=c.line_start, line_end=c.line_end, meta=meta))
        return rows

    def _map_analysis(self, analysis: FileAnalysis, path: str) -> Tuple[List[SymbolRow], List[EndpointRow], List[ModelRow]]:
        symbols: List[SymbolRow] = []
        for s in analysis.symbols or []:
            if not getattr(s, "name", None):
                continue
            symbols.append(SymbolRow(name=str(s.name), kind=str(s.kind or ""), line_start=s.line_start,
                                     line_end=s.line_end, signature=(s.signature or None), parent=(s.parent or None),
                                     meta=dict(s.meta or {})))
        endpoints: List[EndpointRow] = []
        for e in analysis.endpoints or []:
            side = str(getattr(e, "side", "") or "")
            if side not in ("server", "client"):
                self._warn("%s : endpoint ignoré, côté inconnu « %s » (%s)" % (path, side, getattr(e, "path", "")))
                continue
            meta = dict(e.meta or {})
            full_path = meta.get("full_path")
            source = full_path if isinstance(full_path, str) and full_path.strip() else (e.path or "")
            endpoints.append(EndpointRow(side=side, method=str(e.method or "ANY").upper(), path=str(e.path or ""),
                                         normalized_path=normalize_route_path(source), handler=(e.handler or None),
                                         framework=(e.framework or None), line=e.line, meta=meta))
        models: List[ModelRow] = []
        for m in analysis.models or []:
            if not getattr(m, "name", None):
                continue
            models.append(ModelRow(name=str(m.name), table_name=(m.table_name or None), kind=(m.kind or None),
                                   fields=[_as_dict(f) for f in (m.fields or [])],
                                   relations=[_as_dict(r) for r in (m.relations or [])], line=m.line))
        return symbols, endpoints, models

    def _raw_relations(self, file_key: str, symbols: Sequence[SymbolRow], endpoints: Sequence[EndpointRow],
                       models: Sequence[ModelRow], analysis: FileAnalysis) -> List[RelationRow]:
        relations: List[RelationRow] = []
        # imports bruts : un seul arc par module, noms fusionnés
        by_module: Dict[str, Dict[str, Any]] = {}
        for imp in analysis.imports or []:
            module = str(getattr(imp, "module", "") or "").strip()
            if not module:
                continue
            entry = by_module.get(module)
            if entry is None:
                entry = {"names": [], "line": imp.line, "is_relative": bool(getattr(imp, "is_relative", False))}
                by_module[module] = entry
            for name in getattr(imp, "names", None) or []:
                if name and name not in entry["names"]:
                    entry["names"].append(str(name))
        for module, meta in by_module.items():
            relations.append(RelationRow(src=file_key, dst=node_key("module", module), kind="imports_raw", meta=meta))
        # modèle → table
        for m in models:
            if m.id is None or not m.table_name or m.kind not in TABLE_MODEL_KINDS:
                continue
            table = str(m.table_name).strip().strip("`\"'[]")
            if not table:
                continue
            relations.append(RelationRow(src=node_key("model", m.id), dst=node_key("table", table), kind="defines_table",
                                         confidence=1.0 if m.kind != "eloquent" else 0.8, meta={"model_kind": m.kind}))
        # endpoint serveur → symbole gestionnaire du même fichier
        for e in endpoints:
            if e.id is None or e.side != "server" or not e.handler:
                continue
            target = _find_handler_symbol(e.handler, symbols)
            if target is not None and target.id is not None:
                relations.append(RelationRow(src=node_key("endpoint", e.id), dst=node_key("symbol", target.id),
                                             kind="handles", meta={"handler": e.handler}))
        return relations

    def _store(self, row: FileRow, symbols: List[SymbolRow], endpoints: List[EndpointRow], models: List[ModelRow],
               chunks: List[ChunkRow], analysis: FileAnalysis, stats: Dict[str, Any]) -> Optional[int]:
        """Écrit fichier + artefacts + relations brutes dans UNE transaction (tout ou rien)."""
        try:
            with self.db.transaction():
                file_id = self.db.upsert_file(row)
                file_key = node_key("file", file_id)
                self.db.delete_relations_from(file_key)
                self.db.replace_file_artifacts(file_id, symbols=symbols, endpoints=endpoints, models=models,
                                               chunks=chunks)
                relations = self._raw_relations(file_key, symbols, endpoints, models, analysis)
                n_rel = self.db.add_relations(relations) if relations else 0
        except Exception as exc:  # noqa: BLE001
            self._error(row.path, "store", "Écriture en base impossible : %s" % exc)
            return None
        stats["symbols"] += len(symbols)
        stats["endpoints"] += len(endpoints)
        stats["models"] += len(models)
        stats["chunks"] += len(chunks)
        stats["relations"] += n_rel
        return file_id

    # ------------------------------------------------------------------ (3) documents
    def _sources_with_chunks(self, source_type: str) -> Set[int]:
        rows = self.db.query("SELECT DISTINCT source_id FROM chunks WHERE source_type = ?", (source_type,))
        return {int(r["source_id"]) for r in rows}

    def _index_documents(self, full: bool, stats: Dict[str, Any]) -> None:
        self._progress("documents", 0, 0, "", "Vérification des documents…")
        docs = self.db.list_documents(limit=_LARGE_LIMIT)
        with_chunks = self._sources_with_chunks("document")
        todo = []
        for doc in docs:
            if doc.id is None or not doc.stored_path:
                continue
            if not os.path.isfile(doc.stored_path):
                self._warn("Document « %s » : fichier introuvable (%s), non réindexé." % (doc.title or doc.id, doc.stored_path))
                continue
            already_empty = (doc.meta or {}).get("chunk_count") == 0
            if full or (doc.id not in with_chunks and not already_empty):
                todo.append(doc)
        total = len(todo)
        for i, doc in enumerate(todo):
            self._check_cancel()
            self._progress("documents", i, total, "", doc.title or (doc.stored_path or ""))
            self.index_document(doc.id)  # type: ignore[arg-type]
            stats["documents"] += 1
        if total:
            self._progress("documents", total, total, "", "%d document(s) indexé(s)" % total)

    def index_document(self, document_id: int) -> int:
        """(Ré)indexe un document déposé : texte, morceaux (titres, pages), métadonnées. Renvoie le nombre de morceaux."""
        doc = self.db.get_document(document_id)
        label = "document:%s" % document_id
        if doc is None:
            self._error(label, "document", "Document introuvable en base.")
            return 0
        label = doc.stored_path or doc.title or label
        if not doc.stored_path:
            self._error(label, "document", "Document sans fichier stocké : impossible à indexer.")
            return 0
        hint: Optional[str] = None
        if format_for_path(doc.stored_path) is None and doc.format:
            fmt = doc.format.lower().lstrip(".")
            hint = SUPPORTED_FORMATS.get(fmt) or (fmt if fmt in set(SUPPORTED_FORMATS.values()) else None)
        try:
            loaded = load_document(doc.stored_path, format_hint=hint)
        except Exception as exc:  # noqa: BLE001 - load_document ne lève pas, par sécurité
            self._error(label, "document", "Chargement impossible : %s" % exc)
            return 0
        for w in loaded.warnings:
            self._warn("%s : %s" % (doc.title or label, w))
        text = loaded.text or ""
        chunks: List[Chunk] = []
        if text.strip():
            try:
                chunks = chunk_text(text, _chunk_mode_for_format(loaded.format), language=loaded.format)
            except Exception as exc:  # noqa: BLE001
                self._error(label, "chunk", "Échec du découpage : %s" % exc)
        page_offsets = loaded.meta.get("page_offsets")
        rows = self._document_chunk_rows(chunks, text, page_offsets if isinstance(page_offsets, list) else None)

        tables: List[Dict[str, Any]] = []
        if loaded.format in ("markdown", "text") and text:
            try:
                tables = [{"headers": list(t.headers), "rows": len(t.rows), "heading_path": list(t.heading_path),
                           "line_start": t.line_start, "line_end": t.line_end} for t in extract_markdown_tables(text)]
            except Exception:  # noqa: BLE001
                tables = []
        meta = dict(doc.meta or {})
        meta.update({
            "title": loaded.title or doc.title or "",
            "pages": loaded.pages,
            "warnings": list(loaded.warnings),
            "looks_like_data_dictionary": bool(loaded.meta.get("looks_like_data_dictionary", False)),
            "tables": tables,
            "format": loaded.format,
            "chunk_count": len(rows),
            "indexed_at": time.time(),
            "indexed_sha1": doc.sha1,
        })
        if loaded.meta.get("headings"):
            meta["headings"] = list(loaded.meta["headings"])[:200]
        if loaded.format == "sql" and loaded.meta.get("tables"):
            meta["sql_tables"] = list(loaded.meta["tables"])
        doc.meta = meta
        if not doc.title and loaded.title:
            doc.title = loaded.title
        if not doc.format and loaded.format and loaded.format != "unknown":
            doc.format = loaded.format
        try:
            with self.db.transaction():
                self.db.update_document(doc)
                self.db.replace_chunks("document", doc.id, rows)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            self._error(label, "store", "Écriture en base impossible : %s" % exc)
            return 0
        return len(rows)

    @staticmethod
    def _document_chunk_rows(chunks: Sequence[Chunk], text: str, page_offsets: Optional[List[int]]) -> List[ChunkRow]:
        offsets = _line_offsets(text) if page_offsets else []
        rows: List[ChunkRow] = []
        for c in chunks:
            meta = dict(c.meta or {})
            meta.setdefault("heading_path", [])
            if page_offsets and offsets:
                idx = min(max(c.line_start - 1, 0), len(offsets) - 1)
                meta["page"] = page_for_offset(page_offsets, offsets[idx])
            rows.append(ChunkRow(text=c.text, ordinal=c.ordinal, line_start=c.line_start, line_end=c.line_end, meta=meta))
        return rows

    # ------------------------------------------------------------------ (4) notes
    def _index_notes(self, full: bool, stats: Dict[str, Any]) -> None:
        self._progress("notes", 0, 0, "", "Vérification des notes…")
        notes = self.db.list_notes(limit=_LARGE_LIMIT)
        with_chunks = self._sources_with_chunks("note")
        todo = [n for n in notes if n.id is not None and (full or n.id not in with_chunks)]
        total = len(todo)
        for i, note in enumerate(todo):
            if i % PROGRESS_EVERY == 0:
                self._check_cancel()
                self._progress("notes", i, total, "", note.title or "")
            self.index_note(note.id)  # type: ignore[arg-type]
            stats["notes"] += 1
        if total:
            self._progress("notes", total, total, "", "%d note(s) indexée(s)" % total)

    def index_note(self, note_id: int) -> int:
        """Une note = un morceau (plusieurs seulement au-delà de :data:`NOTE_SPLIT_CHARS`), avec en-tête daté."""
        note = self.db.get_note(note_id)
        label = "note:%s" % note_id
        if note is None:
            self._error(label, "note", "Note introuvable en base.")
            return 0
        tags = [str(t).strip() for t in (note.tags or []) if str(t).strip()]
        header = "Note du %s — %s" % (note.note_date or "?", note.title or "(sans titre)")
        if tags:
            header += " [%s]" % ", ".join(tags)
        body = (note.body or "").replace("\r\n", "\n").replace("\r", "\n")
        base_meta: Dict[str, Any] = {"kind": "note", "title": note.title or "", "note_date": note.note_date, "tags": tags}
        rows: List[ChunkRow] = []
        if len(body) <= NOTE_SPLIT_CHARS:
            meta = dict(base_meta)
            if body.strip():
                meta["body_offset"] = len(header) + 2
                text = header + "\n\n" + body
            else:
                meta["body_offset"] = len(header)
                text = header
            rows.append(ChunkRow(text=text, ordinal=0, line_start=1, line_end=max(1, len(body.splitlines())), meta=meta))
        else:
            try:
                parts = chunk_text(body, "markdown", max_chars=max(500, NOTE_SPLIT_CHARS - len(header) - 2),
                                   overlap_chars=200)
            except Exception as exc:  # noqa: BLE001
                self._error(label, "chunk", "Échec du découpage : %s" % exc)
                parts = []
            for c in parts:
                meta = dict(c.meta or {})
                meta.update(base_meta)
                meta["body_offset"] = len(header) + 2
                meta["part"] = c.ordinal
                rows.append(ChunkRow(text=header + "\n\n" + c.text, ordinal=c.ordinal, line_start=c.line_start,
                                     line_end=c.line_end, meta=meta))
        try:
            self.db.replace_chunks("note", note.id, rows)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            self._error(label, "store", "Écriture en base impossible : %s" % exc)
            return 0
        return len(rows)

    # ------------------------------------------------------------------ (5) relations
    def _link(self, stats: Dict[str, Any]) -> None:
        self._progress("relations", 0, 0, "", "Construction du graphe de relations…")
        linker = self.linker
        if linker is None:
            try:
                from develop.knowledge_graph.linker import link_all  # import paresseux : module optionnel
                linker = link_all
            except ImportError as exc:
                self._warn("Graphe de relations non construit (module develop.knowledge_graph indisponible : %s)." % exc)
                stats["linker"] = None
                return
        try:
            result = linker(self.db, self.workspace)
        except IndexCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            self._error("", "relations", "Échec de la construction des relations : %s" % exc)
            return
        stats["linker"] = _jsonable(result)
        if isinstance(result, bool):
            return
        if isinstance(result, int):
            stats["relations"] += result
            return
        # dict, ou objet à champs (``LinkStats`` : dataclass avec ``total`` / ``relations``) → même lecture
        data = result if isinstance(result, dict) else stats["linker"]
        if isinstance(data, dict):
            for key in ("relations", "added", "count", "total"):
                value = data.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    stats["relations"] += value
                    break

    # ------------------------------------------------------------------ (6) embeddings
    def _embed(self, with_embeddings: bool, stats: Dict[str, Any]) -> None:
        if not with_embeddings:
            self._progress("embeddings", 0, 0, "", "Embeddings ignorés pour ce run.")
            return
        if not self.embedding_model or self.llm is None:
            self._progress("embeddings", 0, 0, "", "Aucun modèle d'embeddings configuré : recherche lexicale seule.")
            return
        self._progress("embeddings", 0, 0, "", "Calcul des embeddings (%s)…" % self.embedding_model)
        stats["embeddings"] += self.embed_missing()
        try:
            stats["embeddings_pending"] = len(self.db.chunk_ids_missing_embeddings(self.embedding_model))
        except Exception:  # noqa: BLE001
            pass
        self._check_cancel()

    def embed_missing(self, batch_size: int = 32, limit: Optional[int] = None) -> int:
        """Calcule les vecteurs manquants via :class:`Embedder` ; renvoie le nombre écrit (jamais d'exception)."""
        if self.llm is None or not self.embedding_model:
            return 0
        embedder = Embedder(self.db, self.llm, self.embedding_model)

        def on_progress(done: int, total: int) -> None:
            self._progress("embeddings", done, total, "", "Vecteurs calculés : %d/%d" % (done, total))

        n = embedder.embed_missing(batch_size=batch_size, limit=limit, progress=on_progress,
                                   cancel_event=self.cancel_event)
        for w in embedder.warnings:
            self._warn(w)
        for e in embedder.errors:
            self._error(e.get("path", ""), e.get("stage", "embeddings"), e.get("message", ""))
        return n

    # ------------------------------------------------------------------ (7) état Git
    @staticmethod
    def _git_head(path: str) -> Optional[str]:
        try:
            root = find_git_root(path)
            if root is None:
                return None
            return GitRepo(str(root)).head()
        except (GitError, OSError, ValueError):
            return None

    def _finalize_repos(self, targets: Sequence[RepoConfig], stats: Dict[str, Any]) -> None:
        now = time.time()
        for repo in targets:
            rstats = stats["repos"].get(repo.id) or {}
            if rstats.get("status") == "error":
                continue
            head = self._git_head(repo.path)
            rstats["head_sha"] = head
            try:
                self.db.upsert_repo(RepoRow(id=repo.id, name=repo.name or repo.id, path=repo.path, kind=repo.kind,
                                            head_sha=head, last_indexed_at=now))
            except Exception as exc:  # noqa: BLE001
                self._error(repo.path, "repo", "Mise à jour du dépôt impossible : %s" % exc)

    # ------------------------------------------------------------------ dérive de l'index
    def stale_files(self, repo_id: str) -> List[str]:
        """Chemins dont le contenu sur disque diffère de la base, absents de la base ou disparus du disque.

        Même décision que :meth:`run` : ``(taille, mtime)`` identiques → inchangé sans lecture ;
        taille différente → obsolète sans lecture ; taille identique mais mtime différent (simple
        ``touch``, ``git checkout``…) → le sha1 du contenu est recalculé et seul un sha1 différent
        (ou un fichier illisible) rend le fichier obsolète. Lecture seule : la base n'est pas
        modifiée (c'est le run suivant qui rafraîchit le mtime). Utilisé par l'analyse d'impact
        pour signaler ``stale_index``.
        """
        repo = self.workspace.repo_by_id(repo_id)
        path: Optional[str] = None
        exclude: List[str] = []
        if repo is not None:
            path, exclude = repo.path, list(repo.exclude or [])
        else:
            stored = self.db.get_repo(repo_id)
            if stored is not None:
                path = stored.path
        signatures = self.db.file_signatures(repo_id)
        if not path or not os.path.isdir(path):
            return sorted(signatures)
        try:
            scan = scan_repo(path, repo_id, exclude=exclude)
        except Exception:  # noqa: BLE001
            return sorted(signatures)
        stale: Set[str] = set()
        seen: Set[str] = set()
        for entry in scan.files:
            seen.add(entry.path)
            sig = signatures.get(entry.path)
            if sig is None:
                stale.add(entry.path)
                continue
            sha1, mtime, size = sig
            if size != entry.size:
                stale.add(entry.path)
            elif mtime is None or float(mtime) != float(entry.mtime):
                # même taille, mtime différent : un touch ne change pas le contenu, le sha1 tranche
                current = compute_sha1(entry.abs_path)
                if not sha1 or not current or current != sha1:
                    stale.add(entry.path)
        stale.update(p for p in signatures if p not in seen)
        return sorted(stale)
