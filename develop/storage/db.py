"""Mémoire Develop : persistance SQLite (index, relations, documents, notes, conversations).

Une seule connexion par :class:`Database`, protégée par un verrou réentrant : toutes les
méthodes publiques peuvent être appelées depuis plusieurs threads (serveur HTTP,
indexation en arrière-plan). Le schéma vit dans ``schema.sql`` et est appliqué de façon
idempotente par :meth:`Database.migrate`.

Conventions :

- les colonnes « JSON » (``meta``, ``fields``, ``relations``, ``tags``, ``sources``,
  ``stats``) sont stockées en texte et exposées en objets Python dans les dataclasses ;
- les noeuds du graphe de relations sont des clés textuelles ``file:<id>``,
  ``symbol:<id>``, ``endpoint:<id>``, ``model:<id>``, ``document:<id>``, ``note:<id>``,
  ``table:<nom>`` (voir :func:`node_key`) ;
- les vecteurs d'embedding sont des BLOB float32 little-endian ;
- la base ne lève jamais sur une entrée utilisateur « sale » (recherche plein texte) ;
  seules les erreurs de programmation (fichier inexistant, type de source inconnu…)
  lèvent :class:`StorageError`.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Type, TypeVar, Union

import numpy as np

SCHEMA_VERSION = 1
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Migrations incrémentales futures : version cible -> script SQL (appliqué après schema.sql).
_MIGRATIONS: Dict[int, str] = {}

SOURCE_TYPES = ("file", "document", "note")
NODE_PREFIXES = ("file", "symbol", "endpoint", "model", "document", "note", "table")

_IN_BATCH = 400  # taille des lots pour les clauses IN (limite de variables SQLite)
_VECTOR_DTYPE = np.dtype("<f4")  # float32 little-endian, quelle que soit la machine


class StorageError(RuntimeError):
    """Erreur d'utilisation de la mémoire Develop (ex: référence à un fichier inexistant)."""


def node_key(prefix: str, ident: Union[int, str]) -> str:
    """Clé textuelle d'un noeud du graphe, ex. ``node_key("file", 3) == "file:3"``."""
    return "%s:%s" % (prefix, ident)


# --------------------------------------------------------------------------- JSON

def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(text: Any, default: Any) -> Any:
    """Décodage tolérant : renvoie ``default`` si le texte est vide, invalide ou du mauvais type."""
    if text is None or text == "":
        return default
    if not isinstance(text, (str, bytes)):
        return default
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return default
    if isinstance(default, dict) and not isinstance(value, dict):
        return default
    if isinstance(default, list) and not isinstance(value, list):
        return default
    return value


# --------------------------------------------------------------------------- dataclasses

T = TypeVar("T", bound="_Row")

# Colonnes JSON et le type Python attendu (utilisé par ``from_row``).
_JSON_DEFAULTS: Dict[str, type] = {
    "meta": dict,
    "fields": list,
    "relations": list,
    "tags": list,
    "sources": list,
    "stats": dict,
}


@dataclass
class _Row:
    """Base commune : conversion dict <-> ligne SQLite."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_row(cls: Type[T], row: sqlite3.Row) -> T:
        keys = row.keys()
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(cls):
            if f.name not in keys:
                continue
            value = row[f.name]
            if f.name in _JSON_DEFAULTS:
                value = _loads(value, _JSON_DEFAULTS[f.name]())
            kwargs[f.name] = value
        return cls(**kwargs)


@dataclass
class RepoRow(_Row):
    id: str
    name: str = ""
    path: str = ""
    kind: str = "other"
    head_sha: Optional[str] = None
    last_indexed_at: Optional[float] = None


@dataclass
class FileRow(_Row):
    repo_id: str
    path: str  # relatif à la racine du dépôt, séparateur "/"
    id: Optional[int] = None
    language: Optional[str] = None
    kind: Optional[str] = None  # "code" | "doc" | "config" | "data" | ...
    size: Optional[int] = None
    mtime: Optional[float] = None
    sha1: Optional[str] = None
    indexed_at: Optional[float] = None


@dataclass
class SymbolRow(_Row):
    name: str
    kind: str = ""  # "function" | "class" | "method" | "component" | ...
    id: Optional[int] = None
    file_id: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    signature: Optional[str] = None
    parent: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointRow(_Row):
    side: str  # "server" | "client"
    method: str = ""
    path: str = ""
    id: Optional[int] = None
    file_id: Optional[int] = None
    normalized_path: Optional[str] = None
    handler: Optional[str] = None
    framework: Optional[str] = None
    line: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelRow(_Row):
    name: str
    id: Optional[int] = None
    file_id: Optional[int] = None
    table_name: Optional[str] = None
    kind: Optional[str] = None  # "eloquent" | "sql_table" | "typescript_interface" | ...
    fields: List[Any] = field(default_factory=list)
    relations: List[Any] = field(default_factory=list)
    line: Optional[int] = None


@dataclass
class RelationRow(_Row):
    src: str
    dst: str
    kind: str  # "imports" | "calls" | "handles" | "uses_model" | "mentions" | ...
    confidence: float = 1.0
    id: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentRow(_Row):
    title: str = ""
    id: Optional[int] = None
    source_path: Optional[str] = None
    stored_path: Optional[str] = None
    kind: str = "upload"  # "upload" | "repo_file"
    format: Optional[str] = None  # "pdf" | "md" | "txt" | "docx" | ...
    size: Optional[int] = None
    sha1: Optional[str] = None
    added_at: Optional[float] = None
    updated_at: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NoteRow(_Row):
    title: str = ""
    body: str = ""
    id: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    note_date: Optional[str] = None  # "YYYY-MM-DD"
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


@dataclass
class ChunkRow(_Row):
    text: str
    ordinal: int = 0
    id: Optional[int] = None
    source_type: Optional[str] = None  # "file" | "document" | "note"
    source_id: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    sha1: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationRow(_Row):
    id: int
    title: str = ""
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    message_count: int = 0


@dataclass
class MessageRow(_Row):
    role: str  # "user" | "assistant" | "system"
    content: str = ""
    id: Optional[int] = None
    conversation_id: Optional[int] = None
    sources: List[Any] = field(default_factory=list)
    created_at: Optional[float] = None


@dataclass
class IndexRunRow(_Row):
    id: int
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "running"  # "running" | "ok" | "error" | "cancelled"
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(frozen=True)
class FtsHit:
    """Résultat de recherche lexicale ; ``score`` = -bm25, donc plus grand = meilleur."""

    chunk_id: int
    score: float
    snippet: str
    source_type: Optional[str] = None
    source_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- requêtes FTS5 sûres

_TERM_RE = re.compile(r"[^\W_]+", re.UNICODE)


def fts_terms(text: str, min_length: int = 2) -> List[str]:
    """Termes alphanumériques d'un texte utilisateur, en minuscules, sans ponctuation.

    Les termes trop courts (``min_length``) sont ignorés : ils n'apportent rien au
    classement et un préfixe d'une lettre renverrait toute la base.
    """
    if not isinstance(text, str):
        return []
    out: List[str] = []
    for term in _TERM_RE.findall(text):
        term = term.lower()
        if len(term) >= min_length:
            out.append(term)
    return out


def build_fts_query(text: str, operator: str = "AND", prefix_last: bool = True) -> str:
    """Construit une requête FTS5 **sûre** à partir d'un texte arbitraire.

    Chaque terme est entouré de guillemets doubles (donc jamais interprété comme
    opérateur ``OR``/``NOT``/``NEAR`` ni comme filtre de colonne), les termes sont
    joints par espace (ET implicite) ou par ``OR``, et le dernier terme reçoit un
    préfixe ``*``. Renvoie ``""`` si aucun terme exploitable ne subsiste.
    """
    terms = fts_terms(text)
    if not terms:
        return ""
    quoted = ['"%s"' % t.replace('"', '""') for t in terms]
    if prefix_last:
        quoted[-1] = quoted[-1] + "*"
    joiner = " OR " if operator.upper() == "OR" else " "
    return joiner.join(quoted)


def _batches(items: Sequence[Any], size: int = _IN_BATCH) -> Iterator[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


# --------------------------------------------------------------------------- base de données

class Database:
    """Accès thread-safe à la mémoire Develop d'un espace de travail.

    ``path`` peut valoir ``":memory:"`` (tests). Le dossier parent est créé si besoin.
    Utilisable comme context manager (``with Database(p) as db:``).
    """

    def __init__(self, path: Union[str, Path] = ":memory:", migrate: bool = True):
        self.path = str(path)
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._closed = False
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None : pas de BEGIN implicite, on gère les transactions nous-mêmes.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        if migrate:
            self.migrate()

    def __repr__(self) -> str:
        return "Database(%r)" % self.path

    # ----- cycle de vie -----
    def _configure(self) -> None:
        c = self._conn
        c.execute("PRAGMA journal_mode = WAL")  # renvoie 'memory' pour :memory:, sans erreur
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("PRAGMA synchronous = NORMAL")
        c.execute("PRAGMA temp_store = MEMORY")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._tx_depth == 0 and self.path != ":memory:":
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def migrate(self) -> int:
        """Applique le schéma (idempotent) et renvoie la version de schéma courante."""
        with self._lock:
            if self._tx_depth:
                raise StorageError("migrate() ne peut pas être appelé dans une transaction")
            self._conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
            current = self.schema_version()
            if current < SCHEMA_VERSION:
                for version in range(current + 1, SCHEMA_VERSION + 1):
                    script = _MIGRATIONS.get(version)
                    if script:
                        self._conn.executescript(script)
                self.set_meta("schema_version", str(SCHEMA_VERSION))
                current = SCHEMA_VERSION
            return current

    def schema_version(self) -> int:
        raw = self.get_meta("schema_version")
        try:
            return int(raw) if raw else 0
        except ValueError:
            return 0

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Transaction (``BEGIN IMMEDIATE``) ; les appels imbriqués deviennent des SAVEPOINT.

        Le verrou est tenu pendant toute la transaction : aucun autre thread ne peut
        intercaler d'écriture.
        """
        with self._lock:
            depth = self._tx_depth
            if depth == 0:
                self._conn.execute("BEGIN IMMEDIATE")
            else:
                self._conn.execute("SAVEPOINT sp%d" % depth)
            self._tx_depth = depth + 1
            try:
                yield self._conn
            except BaseException:
                self._tx_depth = depth
                if depth == 0:
                    self._conn.execute("ROLLBACK")
                else:
                    self._conn.execute("ROLLBACK TO sp%d" % depth)
                    self._conn.execute("RELEASE sp%d" % depth)
                raise
            else:
                self._tx_depth = depth
                if depth == 0:
                    self._conn.execute("COMMIT")
                else:
                    self._conn.execute("RELEASE sp%d" % depth)

    # ----- bas niveau -----
    def _exec(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def _all(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _count(self, table: str) -> int:
        row = self._one("SELECT count(*) AS n FROM %s" % table)
        return int(row["n"]) if row else 0

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        """Échappatoire **en lecture** pour des statistiques ad hoc (SELECT uniquement)."""
        if not sql.lstrip().lower().startswith(("select", "with", "pragma", "explain")):
            raise StorageError("query() n'accepte que des requêtes de lecture")
        return self._all(sql, params)

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self._one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else default

    def set_meta(self, key: str, value: Optional[str]) -> None:
        with self._lock:
            if value is None:
                self._exec("DELETE FROM meta WHERE key = ?", (key,))
            else:
                self._exec("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                           (key, str(value)))

    # ------------------------------------------------------------------ repos
    def upsert_repo(self, repo: RepoRow) -> RepoRow:
        self._exec(
            "INSERT INTO repos(id, name, path, kind, head_sha, last_indexed_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name = excluded.name, path = excluded.path, kind = excluded.kind, "
            "head_sha = excluded.head_sha, last_indexed_at = excluded.last_indexed_at",
            (repo.id, repo.name or "", repo.path or "", repo.kind or "other", repo.head_sha, repo.last_indexed_at),
        )
        return repo

    def get_repo(self, repo_id: str) -> Optional[RepoRow]:
        row = self._one("SELECT * FROM repos WHERE id = ?", (repo_id,))
        return RepoRow.from_row(row) if row else None

    def list_repos(self) -> List[RepoRow]:
        return [RepoRow.from_row(r) for r in self._all("SELECT * FROM repos ORDER BY name, id")]

    def delete_repo(self, repo_id: str) -> bool:
        """Supprime le dépôt et tout ce qui en dépend (fichiers, symboles, chunks, embeddings, relations)."""
        with self.transaction():
            if self._one("SELECT 1 FROM repos WHERE id = ?", (repo_id,)) is None:
                return False
            for row in self._all("SELECT id FROM files WHERE repo_id = ?", (repo_id,)):
                self._purge_file(int(row["id"]))
            self._exec("DELETE FROM repos WHERE id = ?", (repo_id,))
            return True

    # ------------------------------------------------------------------ files
    def upsert_file(self, file: FileRow) -> int:
        """Insère ou met à jour un fichier ; l'id est conservé si (repo_id, path) existe déjà."""
        with self.transaction():
            indexed_at = file.indexed_at if file.indexed_at is not None else time.time()
            self._exec(
                "INSERT INTO files(repo_id, path, language, kind, size, mtime, sha1, indexed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(repo_id, path) DO UPDATE SET language = excluded.language, kind = excluded.kind, "
                "size = excluded.size, mtime = excluded.mtime, sha1 = excluded.sha1, indexed_at = excluded.indexed_at",
                (file.repo_id, file.path, file.language, file.kind, file.size, file.mtime, file.sha1, indexed_at),
            )
            row = self._one("SELECT id FROM files WHERE repo_id = ? AND path = ?", (file.repo_id, file.path))
            file.id = int(row["id"])
            file.indexed_at = indexed_at
            return file.id

    def get_file(self, repo_id: str, path: str) -> Optional[FileRow]:
        row = self._one("SELECT * FROM files WHERE repo_id = ? AND path = ?", (repo_id, path))
        return FileRow.from_row(row) if row else None

    def get_file_by_id(self, file_id: int) -> Optional[FileRow]:
        row = self._one("SELECT * FROM files WHERE id = ?", (file_id,))
        return FileRow.from_row(row) if row else None

    def list_files(self, repo_id: Optional[str] = None, kind: Optional[str] = None,
                   language: Optional[str] = None, limit: Optional[int] = None) -> List[FileRow]:
        where: List[str] = []
        params: List[Any] = []
        if repo_id is not None:
            where.append("repo_id = ?")
            params.append(repo_id)
        if kind is not None:
            where.append("kind = ?")
            params.append(kind)
        if language is not None:
            where.append("language = ?")
            params.append(language)
        sql = "SELECT * FROM files"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY repo_id, path"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [FileRow.from_row(r) for r in self._all(sql, params)]

    def delete_file(self, file_id: int) -> bool:
        with self.transaction():
            if self._one("SELECT 1 FROM files WHERE id = ?", (file_id,)) is None:
                return False
            self._purge_file(file_id)
            self._exec("DELETE FROM files WHERE id = ?", (file_id,))
            return True

    def delete_files_not_in(self, repo_id: str, keep_paths: Iterable[str]) -> int:
        """Supprime les fichiers du dépôt absents de ``keep_paths`` ; renvoie le nombre supprimé."""
        keep = set(keep_paths)
        with self.transaction():
            doomed = [int(r["id"]) for r in self._all("SELECT id, path FROM files WHERE repo_id = ?", (repo_id,))
                      if r["path"] not in keep]
            for file_id in doomed:
                self.delete_file(file_id)
            return len(doomed)

    def file_signatures(self, repo_id: str) -> Dict[str, Tuple[Optional[str], Optional[float], Optional[int]]]:
        """``{path: (sha1, mtime, size)}`` pour décider quoi réindexer (indexation incrémentale)."""
        rows = self._all("SELECT path, sha1, mtime, size FROM files WHERE repo_id = ?", (repo_id,))
        return {r["path"]: (r["sha1"], r["mtime"], r["size"]) for r in rows}

    # ----- helpers internes fichiers -----
    def _artifact_node_keys(self, file_id: int) -> List[str]:
        keys: List[str] = []
        keys += [node_key("symbol", r["id"]) for r in self._all("SELECT id FROM symbols WHERE file_id = ?", (file_id,))]
        keys += [node_key("endpoint", r["id"]) for r in self._all("SELECT id FROM endpoints WHERE file_id = ?", (file_id,))]
        keys += [node_key("model", r["id"]) for r in self._all("SELECT id FROM data_models WHERE file_id = ?", (file_id,))]
        return keys

    def _purge_file(self, file_id: int) -> None:
        """Relations + chunks (+ embeddings par cascade) d'un fichier ; les artefacts partent en cascade FK."""
        keys = [node_key("file", file_id)] + self._artifact_node_keys(file_id)
        self.delete_relations_touching(keys)
        self.replace_chunks("file", file_id, [])

    # ------------------------------------------------------------------ artefacts par fichier
    def replace_file_artifacts(self, file_id: int, symbols: Sequence[SymbolRow] = (),
                               endpoints: Sequence[EndpointRow] = (), models: Sequence[ModelRow] = (),
                               chunks: Sequence[ChunkRow] = ()) -> Dict[str, List[int]]:
        """Remplace en UNE transaction tous les artefacts d'un fichier.

        Supprime les anciens symboles/endpoints/modèles/chunks (et les relations qui
        référençaient leurs ids), insère les nouveaux et renvoie
        ``{"symbols": [...], "endpoints": [...], "models": [...], "chunks": [...]}``.
        Les ``file_id``/``id`` des objets passés sont renseignés en place.
        """
        with self.transaction():
            if self._one("SELECT 1 FROM files WHERE id = ?", (file_id,)) is None:
                raise StorageError("fichier introuvable: id=%s" % file_id)
            old_keys = self._artifact_node_keys(file_id)
            if old_keys:
                self.delete_relations_touching(old_keys)
            self._exec("DELETE FROM symbols WHERE file_id = ?", (file_id,))
            self._exec("DELETE FROM endpoints WHERE file_id = ?", (file_id,))
            self._exec("DELETE FROM data_models WHERE file_id = ?", (file_id,))
            ids: Dict[str, List[int]] = {"symbols": [], "endpoints": [], "models": [], "chunks": []}
            for s in symbols:
                s.file_id = file_id
                cur = self._exec(
                    "INSERT INTO symbols(file_id, name, kind, line_start, line_end, signature, parent, meta) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (file_id, s.name, s.kind, s.line_start, s.line_end, s.signature, s.parent, _dumps(s.meta or {})),
                )
                s.id = int(cur.lastrowid)
                ids["symbols"].append(s.id)
            for e in endpoints:
                e.file_id = file_id
                cur = self._exec(
                    "INSERT INTO endpoints(file_id, side, method, path, normalized_path, handler, framework, line, meta) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (file_id, e.side, e.method, e.path, e.normalized_path, e.handler, e.framework, e.line,
                     _dumps(e.meta or {})),
                )
                e.id = int(cur.lastrowid)
                ids["endpoints"].append(e.id)
            for m in models:
                m.file_id = file_id
                cur = self._exec(
                    "INSERT INTO data_models(file_id, name, table_name, kind, fields, relations, line) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (file_id, m.name, m.table_name, m.kind, _dumps(m.fields or []), _dumps(m.relations or []), m.line),
                )
                m.id = int(cur.lastrowid)
                ids["models"].append(m.id)
            ids["chunks"] = self.replace_chunks("file", file_id, chunks)
            return ids

    # ------------------------------------------------------------------ symbols
    def list_symbols(self, file_id: Optional[int] = None, kind: Optional[str] = None, limit: int = 500) -> List[SymbolRow]:
        where: List[str] = []
        params: List[Any] = []
        if file_id is not None:
            where.append("file_id = ?")
            params.append(file_id)
        if kind is not None:
            where.append("kind = ?")
            params.append(kind)
        sql = "SELECT * FROM symbols"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY file_id, line_start, id LIMIT ?"
        params.append(int(limit))
        return [SymbolRow.from_row(r) for r in self._all(sql, params)]

    def search_symbols(self, name_query: str, limit: int = 50) -> List[SymbolRow]:
        """Recherche par nom, insensible à la casse ; correspondances exactes puis préfixes puis le reste."""
        q = (name_query or "").strip()
        if not q:
            return []
        esc = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._all(
            "SELECT * FROM symbols WHERE name LIKE ? ESCAPE '\\' "
            "ORDER BY (lower(name) = lower(?)) DESC, (name LIKE ? ESCAPE '\\') DESC, length(name), name, id LIMIT ?",
            ("%" + esc + "%", q, esc + "%", int(limit)),
        )
        return [SymbolRow.from_row(r) for r in rows]

    def get_symbol(self, symbol_id: int) -> Optional[SymbolRow]:
        row = self._one("SELECT * FROM symbols WHERE id = ?", (symbol_id,))
        return SymbolRow.from_row(row) if row else None

    # ------------------------------------------------------------------ endpoints
    def list_endpoints(self, side: Optional[str] = None, framework: Optional[str] = None,
                       limit: Optional[int] = None) -> List[EndpointRow]:
        where: List[str] = []
        params: List[Any] = []
        if side is not None:
            where.append("side = ?")
            params.append(side)
        if framework is not None:
            where.append("framework = ?")
            params.append(framework)
        sql = "SELECT * FROM endpoints"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY normalized_path, method, id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [EndpointRow.from_row(r) for r in self._all(sql, params)]

    def find_endpoints(self, normalized_path: Optional[str] = None, method: Optional[str] = None) -> List[EndpointRow]:
        where: List[str] = []
        params: List[Any] = []
        if normalized_path is not None:
            where.append("normalized_path = ?")
            params.append(normalized_path)
        if method:
            where.append("upper(method) = ?")
            params.append(method.upper())
        sql = "SELECT * FROM endpoints"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY side, normalized_path, method, id"
        return [EndpointRow.from_row(r) for r in self._all(sql, params)]

    def get_endpoint(self, endpoint_id: int) -> Optional[EndpointRow]:
        row = self._one("SELECT * FROM endpoints WHERE id = ?", (endpoint_id,))
        return EndpointRow.from_row(row) if row else None

    # ------------------------------------------------------------------ data models
    def list_models(self, limit: Optional[int] = None) -> List[ModelRow]:
        sql = "SELECT * FROM data_models ORDER BY name, id"
        params: List[Any] = []
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [ModelRow.from_row(r) for r in self._all(sql, params)]

    def find_models(self, name: Optional[str] = None, table_name: Optional[str] = None) -> List[ModelRow]:
        """Correspondance exacte insensible à la casse sur le nom et/ou la table."""
        where: List[str] = []
        params: List[Any] = []
        if name is not None:
            where.append("lower(name) = lower(?)")
            params.append(name)
        if table_name is not None:
            where.append("lower(table_name) = lower(?)")
            params.append(table_name)
        sql = "SELECT * FROM data_models"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY name, id"
        return [ModelRow.from_row(r) for r in self._all(sql, params)]

    def get_model(self, model_id: int) -> Optional[ModelRow]:
        row = self._one("SELECT * FROM data_models WHERE id = ?", (model_id,))
        return ModelRow.from_row(row) if row else None

    # ------------------------------------------------------------------ relations
    def add_relations(self, relations: Iterable[RelationRow]) -> int:
        """``INSERT OR REPLACE`` sur (src, dst, kind) ; renvoie le nombre de lignes écrites."""
        n = 0
        with self.transaction():
            for r in relations:
                if not r.src or not r.dst or not r.kind:
                    continue
                cur = self._exec(
                    "INSERT OR REPLACE INTO relations(src, dst, kind, confidence, meta) VALUES (?, ?, ?, ?, ?)",
                    (r.src, r.dst, r.kind, float(r.confidence if r.confidence is not None else 1.0), _dumps(r.meta or {})),
                )
                r.id = int(cur.lastrowid)
                n += 1
        return n

    def relations_from(self, src: str, kind: Optional[str] = None) -> List[RelationRow]:
        if kind is None:
            rows = self._all("SELECT * FROM relations WHERE src = ? ORDER BY kind, dst", (src,))
        else:
            rows = self._all("SELECT * FROM relations WHERE src = ? AND kind = ? ORDER BY dst", (src, kind))
        return [RelationRow.from_row(r) for r in rows]

    def relations_to(self, dst: str, kind: Optional[str] = None) -> List[RelationRow]:
        if kind is None:
            rows = self._all("SELECT * FROM relations WHERE dst = ? ORDER BY kind, src", (dst,))
        else:
            rows = self._all("SELECT * FROM relations WHERE dst = ? AND kind = ? ORDER BY src", (dst, kind))
        return [RelationRow.from_row(r) for r in rows]

    def delete_relations_from(self, src: str, kind: Optional[str] = None) -> int:
        """Supprime les relations sortantes d'un noeud (utile pour réindexer un fichier)."""
        if kind is None:
            cur = self._exec("DELETE FROM relations WHERE src = ?", (src,))
        else:
            cur = self._exec("DELETE FROM relations WHERE src = ? AND kind = ?", (src, kind))
        return int(cur.rowcount or 0)

    def delete_relations_touching(self, node_keys: Iterable[str]) -> int:
        """Supprime toute relation dont ``src`` ou ``dst`` figure dans ``node_keys``."""
        keys = sorted({k for k in node_keys if k})
        if not keys:
            return 0
        total = 0
        with self.transaction():
            for batch in _batches(keys):
                ph = _placeholders(len(batch))
                cur = self._exec("DELETE FROM relations WHERE src IN (%s) OR dst IN (%s)" % (ph, ph),
                                 list(batch) + list(batch))
                total += int(cur.rowcount or 0)
        return total

    def neighbors(self, node: str, depth: int = 1) -> Set[str]:
        """Noeuds atteignables depuis ``node`` en au plus ``depth`` sauts, dans les deux sens (sans ``node``)."""
        seen: Set[str] = {node}
        frontier: List[str] = [node]
        with self._lock:
            for _ in range(max(0, int(depth))):
                if not frontier:
                    break
                found: Set[str] = set()
                for batch in _batches(frontier):
                    ph = _placeholders(len(batch))
                    rows = self._all("SELECT src, dst FROM relations WHERE src IN (%s) OR dst IN (%s)" % (ph, ph),
                                     list(batch) + list(batch))
                    for r in rows:
                        found.add(r["src"])
                        found.add(r["dst"])
                frontier = sorted(found - seen)
                seen |= found
        seen.discard(node)
        return seen

    # ------------------------------------------------------------------ documents
    def add_document(self, doc: DocumentRow) -> int:
        now = time.time()
        doc.added_at = doc.added_at if doc.added_at is not None else now
        doc.updated_at = doc.updated_at if doc.updated_at is not None else doc.added_at
        cur = self._exec(
            "INSERT INTO documents(title, source_path, stored_path, kind, format, size, sha1, added_at, updated_at, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc.title or "", doc.source_path, doc.stored_path, doc.kind or "upload", doc.format, doc.size, doc.sha1,
             doc.added_at, doc.updated_at, _dumps(doc.meta or {})),
        )
        doc.id = int(cur.lastrowid)
        return doc.id

    def update_document(self, doc: DocumentRow) -> bool:
        if doc.id is None:
            return False
        doc.updated_at = time.time()
        cur = self._exec(
            "UPDATE documents SET title = ?, source_path = ?, stored_path = ?, kind = ?, format = ?, size = ?, "
            "sha1 = ?, updated_at = ?, meta = ? WHERE id = ?",
            (doc.title or "", doc.source_path, doc.stored_path, doc.kind or "upload", doc.format, doc.size, doc.sha1,
             doc.updated_at, _dumps(doc.meta or {}), doc.id),
        )
        return bool(cur.rowcount)

    def get_document(self, doc_id: int) -> Optional[DocumentRow]:
        row = self._one("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return DocumentRow.from_row(row) if row else None

    def list_documents(self, kind: Optional[str] = None, limit: int = 500) -> List[DocumentRow]:
        if kind is None:
            rows = self._all("SELECT * FROM documents ORDER BY updated_at DESC, id DESC LIMIT ?", (int(limit),))
        else:
            rows = self._all("SELECT * FROM documents WHERE kind = ? ORDER BY updated_at DESC, id DESC LIMIT ?",
                             (kind, int(limit)))
        return [DocumentRow.from_row(r) for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        """Supprime le document, ses chunks (+ embeddings) et les relations qui le mentionnent."""
        with self.transaction():
            if self._one("SELECT 1 FROM documents WHERE id = ?", (doc_id,)) is None:
                return False
            self.replace_chunks("document", doc_id, [])
            self.delete_relations_touching([node_key("document", doc_id)])
            self._exec("DELETE FROM documents WHERE id = ?", (doc_id,))
            return True

    def find_document_by_sha(self, sha1: str) -> Optional[DocumentRow]:
        if not sha1:
            return None
        row = self._one("SELECT * FROM documents WHERE sha1 = ? ORDER BY id LIMIT 1", (sha1,))
        return DocumentRow.from_row(row) if row else None

    # ------------------------------------------------------------------ notes
    def add_note(self, note: NoteRow) -> int:
        now = time.time()
        note.created_at = note.created_at if note.created_at is not None else now
        note.updated_at = note.updated_at if note.updated_at is not None else note.created_at
        note.note_date = note.note_date or time.strftime("%Y-%m-%d", time.localtime(note.created_at))
        cur = self._exec(
            "INSERT INTO notes(title, body, tags, note_date, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (note.title or "", note.body or "", _dumps(list(note.tags or [])), note.note_date, note.created_at,
             note.updated_at),
        )
        note.id = int(cur.lastrowid)
        return note.id

    def update_note(self, note: NoteRow) -> bool:
        if note.id is None:
            return False
        note.updated_at = time.time()
        cur = self._exec(
            "UPDATE notes SET title = ?, body = ?, tags = ?, note_date = COALESCE(?, note_date), updated_at = ? WHERE id = ?",
            (note.title or "", note.body or "", _dumps(list(note.tags or [])), note.note_date, note.updated_at, note.id),
        )
        return bool(cur.rowcount)

    def get_note(self, note_id: int) -> Optional[NoteRow]:
        row = self._one("SELECT * FROM notes WHERE id = ?", (note_id,))
        return NoteRow.from_row(row) if row else None

    def list_notes(self, date_from: Optional[str] = None, date_to: Optional[str] = None,
                   tag: Optional[str] = None, limit: int = 200) -> List[NoteRow]:
        """Notes les plus récentes d'abord ; filtres inclusifs sur ``note_date`` et sur une étiquette."""
        where: List[str] = []
        params: List[Any] = []
        if date_from:
            where.append("note_date >= ?")
            params.append(date_from)
        if date_to:
            where.append("note_date <= ?")
            params.append(date_to)
        sql = "SELECT * FROM notes"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY note_date DESC, id DESC"
        if tag is None:
            sql += " LIMIT ?"
            params.append(int(limit))
        notes = [NoteRow.from_row(r) for r in self._all(sql, params)]
        if tag is not None:
            wanted = tag.strip().lower()
            notes = [n for n in notes if any(str(t).strip().lower() == wanted for t in n.tags)][:int(limit)]
        return notes

    def delete_note(self, note_id: int) -> bool:
        with self.transaction():
            if self._one("SELECT 1 FROM notes WHERE id = ?", (note_id,)) is None:
                return False
            self.replace_chunks("note", note_id, [])
            self.delete_relations_touching([node_key("note", note_id)])
            self._exec("DELETE FROM notes WHERE id = ?", (note_id,))
            return True

    # ------------------------------------------------------------------ chunks
    def replace_chunks(self, source_type: str, source_id: int, chunks: Sequence[ChunkRow]) -> List[int]:
        """Remplace les chunks d'une source (transactionnel) ; les embeddings orphelins partent en cascade.

        L'``ordinal`` est la position dans la liste. Les ``id``/``sha1`` des objets passés
        sont renseignés en place.
        """
        if source_type not in SOURCE_TYPES:
            raise StorageError("type de source inconnu: %r (attendu: %s)" % (source_type, ", ".join(SOURCE_TYPES)))
        ids: List[int] = []
        with self.transaction():
            self._exec("DELETE FROM chunks WHERE source_type = ? AND source_id = ?", (source_type, source_id))
            for i, c in enumerate(chunks):
                text = c.text if isinstance(c.text, str) else str(c.text or "")
                sha = c.sha1 or hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
                c.source_type, c.source_id, c.ordinal, c.sha1 = source_type, source_id, i, sha
                cur = self._exec(
                    "INSERT INTO chunks(source_type, source_id, ordinal, text, line_start, line_end, sha1, meta) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (source_type, source_id, i, text, c.line_start, c.line_end, sha, _dumps(c.meta or {})),
                )
                c.id = int(cur.lastrowid)
                ids.append(c.id)
        return ids

    def get_chunk(self, chunk_id: int) -> Optional[ChunkRow]:
        row = self._one("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        return ChunkRow.from_row(row) if row else None

    def get_chunks(self, ids: Iterable[int]) -> List[ChunkRow]:
        """Chunks dans l'ordre des ids demandés (les ids inconnus sont ignorés)."""
        wanted = [int(i) for i in ids]
        if not wanted:
            return []
        found: Dict[int, ChunkRow] = {}
        with self._lock:
            for batch in _batches(sorted(set(wanted))):
                ph = _placeholders(len(batch))
                for r in self._all("SELECT * FROM chunks WHERE id IN (%s)" % ph, list(batch)):
                    found[int(r["id"])] = ChunkRow.from_row(r)
        out: List[ChunkRow] = []
        seen: Set[int] = set()
        for i in wanted:
            if i in found and i not in seen:
                out.append(found[i])
                seen.add(i)
        return out

    def chunks_for(self, source_type: str, source_id: int) -> List[ChunkRow]:
        rows = self._all("SELECT * FROM chunks WHERE source_type = ? AND source_id = ? ORDER BY ordinal",
                         (source_type, source_id))
        return [ChunkRow.from_row(r) for r in rows]

    def count_chunks(self, source_type: Optional[str] = None) -> int:
        if source_type is None:
            return self._count("chunks")
        row = self._one("SELECT count(*) AS n FROM chunks WHERE source_type = ?", (source_type,))
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------ recherche lexicale
    def fts_search(self, query: str, limit: int = 20,
                   source_types: Optional[Iterable[str]] = None) -> List[FtsHit]:
        """Recherche plein texte (FTS5, bm25) tolérante à n'importe quel texte utilisateur.

        Tous les termes doivent être présents (ET) ; si rien ne correspond, on retente
        en OU. Renvoie ``[]`` si la requête ne contient aucun terme exploitable.
        """
        terms = fts_terms(query)
        if not terms:
            return []
        types: Optional[List[str]] = None
        if source_types is not None:
            types = [t for t in source_types if t in SOURCE_TYPES]
            if not types:
                return []
        hits = self._fts_run(build_fts_query(query, "AND"), int(limit), types)
        if not hits and len(terms) > 1:
            hits = self._fts_run(build_fts_query(query, "OR"), int(limit), types)
        return hits

    def _fts_run(self, match: str, limit: int, types: Optional[List[str]]) -> List[FtsHit]:
        if not match or limit <= 0:
            return []
        sql = ("SELECT chunks_fts.rowid AS chunk_id, bm25(chunks_fts) AS score, "
               "snippet(chunks_fts, 0, '[', ']', '…', 12) AS snippet, c.source_type AS source_type, "
               "c.source_id AS source_id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
               "WHERE chunks_fts MATCH ?")
        params: List[Any] = [match]
        if types:
            sql += " AND c.source_type IN (%s)" % _placeholders(len(types))
            params += types
        sql += " ORDER BY bm25(chunks_fts), chunks_fts.rowid LIMIT ?"
        params.append(limit)
        try:
            rows = self._all(sql, params)
        except sqlite3.OperationalError:
            return []  # requête FTS invalide malgré l'assainissement : on préfère « rien » à une erreur
        return [FtsHit(chunk_id=int(r["chunk_id"]), score=-float(r["score"]), snippet=r["snippet"] or "",
                       source_type=r["source_type"], source_id=r["source_id"]) for r in rows]

    # ------------------------------------------------------------------ embeddings
    def upsert_embeddings(self, items: Iterable[Tuple[int, str, Any]]) -> int:
        """Enregistre des vecteurs ``(chunk_id, model, vecteur)`` ; renvoie le nombre écrit.

        Les vecteurs sont convertis en float32 little-endian. Un ``chunk_id`` qui n'existe
        plus (chunks remplacés entre-temps) est ignoré silencieusement.
        """
        n = 0
        with self.transaction():
            for chunk_id, model, vec in items:
                try:
                    arr = np.asarray(vec, dtype=np.float32).astype(_VECTOR_DTYPE, copy=False).ravel()
                except (TypeError, ValueError):
                    continue
                if arr.size == 0:
                    continue
                try:
                    self._exec(
                        "INSERT OR REPLACE INTO embeddings(chunk_id, model, dim, vector) VALUES (?, ?, ?, ?)",
                        (int(chunk_id), model, int(arr.size), sqlite3.Binary(arr.tobytes())),
                    )
                except sqlite3.IntegrityError:
                    continue  # chunk disparu : l'instruction seule est annulée, la transaction continue
                n += 1
        return n

    def load_embeddings(self, model: str) -> Tuple[np.ndarray, np.ndarray]:
        """``(ids int64 (n,), matrice float32 (n, dim))`` pour un modèle, ou tableaux vides."""
        rows = self._all("SELECT chunk_id, dim, vector FROM embeddings WHERE model = ? ORDER BY chunk_id", (model,))
        empty = (np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32))
        if not rows:
            return empty
        dim = int(rows[0]["dim"])
        if dim <= 0:
            return empty
        ids: List[int] = []
        blobs: List[bytes] = []
        for r in rows:
            blob = r["vector"]
            if int(r["dim"]) != dim or blob is None or len(blob) != dim * 4:
                continue  # vecteur incohérent (changement de modèle à mi-parcours) : ignoré
            ids.append(int(r["chunk_id"]))
            blobs.append(bytes(blob))
        if not ids:
            return empty
        matrix = np.frombuffer(b"".join(blobs), dtype=_VECTOR_DTYPE).reshape(len(ids), dim)
        return np.asarray(ids, dtype=np.int64), np.array(matrix, dtype=np.float32)

    def chunk_ids_missing_embeddings(self, model: str, limit: Optional[int] = None) -> List[int]:
        sql = ("SELECT c.id AS id FROM chunks c LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = ? "
               "WHERE e.chunk_id IS NULL ORDER BY c.id")
        params: List[Any] = [model]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [int(r["id"]) for r in self._all(sql, params)]

    def delete_embeddings(self, model: Optional[str] = None) -> int:
        if model is None:
            cur = self._exec("DELETE FROM embeddings")
        else:
            cur = self._exec("DELETE FROM embeddings WHERE model = ?", (model,))
        return int(cur.rowcount or 0)

    def embedding_models(self) -> List[Dict[str, Any]]:
        """Modèles présents : ``[{"model": ..., "count": ..., "dim": ...}]`` (plus nombreux d'abord)."""
        rows = self._all("SELECT model, count(*) AS count, min(dim) AS dim FROM embeddings GROUP BY model "
                         "ORDER BY count DESC, model")
        return [{"model": r["model"], "count": int(r["count"]), "dim": int(r["dim"])} for r in rows]

    # ------------------------------------------------------------------ conversations
    def create_conversation(self, title: str = "") -> int:
        now = time.time()
        cur = self._exec("INSERT INTO conversations(title, created_at, updated_at) VALUES (?, ?, ?)",
                         (title or "", now, now))
        return int(cur.lastrowid)

    def list_conversations(self, limit: int = 100) -> List[ConversationRow]:
        rows = self._all(
            "SELECT c.*, (SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
            "FROM conversations c ORDER BY c.updated_at DESC, c.id DESC LIMIT ?",
            (int(limit),),
        )
        return [ConversationRow.from_row(r) for r in rows]

    def get_conversation(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Conversation + ``messages`` (liste de dicts, ordre chronologique), ou None."""
        with self._lock:
            row = self._one("SELECT * FROM conversations WHERE id = ?", (conversation_id,))
            if row is None:
                return None
            messages = self._all("SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,))
        data = ConversationRow.from_row(row).to_dict()
        data["messages"] = [MessageRow.from_row(m).to_dict() for m in messages]
        data["message_count"] = len(data["messages"])
        return data

    def rename_conversation(self, conversation_id: int, title: str) -> bool:
        cur = self._exec("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                         (title or "", time.time(), conversation_id))
        return bool(cur.rowcount)

    def delete_conversation(self, conversation_id: int) -> bool:
        cur = self._exec("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return bool(cur.rowcount)

    def add_message(self, conversation_id: int, role: str, content: str,
                    sources: Optional[List[Any]] = None) -> int:
        """Ajoute un message et rafraîchit ``updated_at`` de la conversation."""
        now = time.time()
        with self.transaction():
            if self._one("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)) is None:
                raise StorageError("conversation introuvable: id=%s" % conversation_id)
            cur = self._exec(
                "INSERT INTO messages(conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content or "", _dumps(list(sources or [])), now),
            )
            self._exec("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            return int(cur.lastrowid)

    # ------------------------------------------------------------------ index runs
    def start_index_run(self) -> int:
        cur = self._exec("INSERT INTO index_runs(started_at, status) VALUES (?, 'running')", (time.time(),))
        return int(cur.lastrowid)

    def finish_index_run(self, run_id: int, status: str, stats: Optional[Dict[str, Any]] = None,
                         error: Optional[str] = None) -> bool:
        cur = self._exec("UPDATE index_runs SET finished_at = ?, status = ?, stats = ?, error = ? WHERE id = ?",
                         (time.time(), status, _dumps(stats or {}), error, run_id))
        return bool(cur.rowcount)

    def last_index_run(self) -> Optional[IndexRunRow]:
        row = self._one("SELECT * FROM index_runs ORDER BY id DESC LIMIT 1")
        return IndexRunRow.from_row(row) if row else None

    def list_index_runs(self, limit: int = 20) -> List[IndexRunRow]:
        rows = self._all("SELECT * FROM index_runs ORDER BY id DESC LIMIT ?", (int(limit),))
        return [IndexRunRow.from_row(r) for r in rows]

    # ------------------------------------------------------------------ statistiques
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "repos": self._count("repos"),
                "files": self._count("files"),
                "symbols": self._count("symbols"),
                "endpoints": self._count("endpoints"),
                "models": self._count("data_models"),
                "relations": self._count("relations"),
                "documents": self._count("documents"),
                "notes": self._count("notes"),
                "chunks": self._count("chunks"),
                "embeddings": self._count("embeddings"),
                "conversations": self._count("conversations"),
                "messages": self._count("messages"),
            }
