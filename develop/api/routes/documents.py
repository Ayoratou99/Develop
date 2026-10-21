"""Documents déposés : dépôt (multipart ou chemin local), liste, détail, contenu, suppression.

Le fichier est **copié** dans ``config.documents_dir(ws)/<sha1>-<nom>`` (jamais lu en place
ensuite), enregistré en base puis indexé immédiatement (``Indexer.index_document``). La détection
de dérive (``stale``, ``mentions``) repose sur ``develop.knowledge_graph.mentions`` importé
paresseusement : sans ce module, ``stale`` vaut ``False`` et ``mentions`` est vide.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool

from develop import config
from develop.api.routes import api_error, get_state, require_db
from develop.api.state import AppState
from develop.document_indexer import SUPPORTED_FORMATS, format_for_path, load_document
from develop.indexing import Indexer
from develop.storage import Database, DocumentRow

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PREVIEW_CHUNKS = 5
PREVIEW_CHARS = 300
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _mentions_module() -> Any:
    try:
        from develop.knowledge_graph import mentions
    except Exception:  # noqa: BLE001 - module optionnel
        return None
    return mentions


def _chunk_counts(db: Database) -> Dict[int, int]:
    rows = db.query("SELECT source_id, count(*) AS n FROM chunks WHERE source_type = 'document' GROUP BY source_id")
    return {int(r["source_id"]): int(r["n"]) for r in rows}


def document_dict(doc: DocumentRow, chunk_count: Optional[int] = None, stale: bool = False) -> Dict[str, Any]:
    meta = doc.meta or {}
    return {
        "id": doc.id, "title": doc.title, "format": doc.format, "kind": doc.kind, "size": doc.size,
        "source_path": doc.source_path, "stored_path": doc.stored_path, "sha1": doc.sha1,
        "added_at": doc.added_at, "updated_at": doc.updated_at,
        "chunk_count": chunk_count if chunk_count is not None else int(meta.get("chunk_count") or 0),
        "stale": bool(stale), "pages": meta.get("pages"),
        "looks_like_data_dictionary": bool(meta.get("looks_like_data_dictionary", False)),
        "warnings": list(meta.get("warnings") or []),
    }


def _stale_flags(db: Database, doc_ids: List[int]) -> Dict[int, bool]:
    mentions = _mentions_module()
    if mentions is None or not doc_ids:
        return {i: False for i in doc_ids}
    try:
        index = mentions.NameIndex.build(db)
    except Exception:  # noqa: BLE001
        return {i: False for i in doc_ids}
    out: Dict[int, bool] = {}
    for i in doc_ids:
        try:
            out[i] = bool(mentions.is_stale(db, "document", i, index))
        except Exception:  # noqa: BLE001
            out[i] = False
    return out


def _safe_name(name: str) -> str:
    base = os.path.basename(name or "").strip() or "document"
    base = _SAFE_NAME_RE.sub("_", base).strip("._") or "document"
    return base[:80]


def store_document(state: AppState, data: bytes, filename: str, source_path: Optional[str] = None) -> Dict[str, Any]:
    """Copie, enregistre et indexe un document ; renvoie sa forme API (``already_exists`` si doublon)."""
    ws = state.require_workspace()
    db = state.db
    assert db is not None
    fmt = format_for_path(filename)
    if fmt is None:
        raise api_error(415, "unsupported_format", "Format non pris en charge : « %s ». Formats acceptés : %s."
                        % (filename, ", ".join("." + e for e in sorted(SUPPORTED_FORMATS))))
    if len(data) > MAX_UPLOAD_BYTES:
        raise api_error(413, "payload_too_large", "Document trop volumineux (max %d Mo)." % (MAX_UPLOAD_BYTES // (1024 * 1024)))
    sha1 = hashlib.sha1(data).hexdigest()
    existing = db.find_document_by_sha(sha1)
    if existing is not None and existing.id is not None:
        d = document_dict(existing, _chunk_counts(db).get(existing.id, 0))
        d["already_exists"] = True
        return d
    safe = _safe_name(filename)
    dest_dir = config.documents_dir(ws.id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stored = dest_dir / ("%s-%s" % (sha1, safe))
    stored.write_bytes(data)
    ext = os.path.splitext(safe)[1].lstrip(".").lower()
    doc = DocumentRow(title="", source_path=source_path or filename, stored_path=str(stored), kind="upload",
                      format=ext or fmt, size=len(data), sha1=sha1)
    doc_id = db.add_document(doc)
    indexer = Indexer(db, ws, state.llm, ws.llm.embedding_model if ws.llm.engine != "none" else "")
    n = indexer.index_document(doc_id)
    fresh = db.get_document(doc_id) or doc
    if not fresh.title:
        fresh.title = os.path.splitext(os.path.basename(filename))[0] or safe
        db.update_document(fresh)
    if state.search_engine is not None:
        state.search_engine._matrix_cache = None
    d = document_dict(fresh, n, _stale_flags(db, [doc_id]).get(doc_id, False))
    d["already_exists"] = False
    d["index_errors"] = list(indexer.errors)
    return d


# --------------------------------------------------------------------------- routes

@router.get("")
def list_documents(request: Request) -> List[Dict[str, Any]]:
    """``[{id, title, format, kind, size, added_at, updated_at, chunk_count, stale}]``."""
    db = require_db(get_state(request))
    docs = db.list_documents(limit=10000)
    counts = _chunk_counts(db)
    stale = _stale_flags(db, [d.id for d in docs if d.id is not None])
    return [document_dict(d, counts.get(int(d.id), 0), stale.get(int(d.id), False)) for d in docs if d.id is not None]  # type: ignore[arg-type]


@router.post("", status_code=201)
async def add_document(request: Request) -> Dict[str, Any]:
    """Multipart ``file`` **ou** JSON ``{path}`` (fichier local copié dans l'espace) → document indexé."""
    state = get_state(request)
    require_db(state)
    ctype = (request.headers.get("content-type") or "").lower()
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "filename"):
            raise api_error(400, "missing_file", "Le champ multipart « file » est obligatoire.")
        data = await upload.read()  # type: ignore[union-attr]
        filename = os.path.basename(getattr(upload, "filename", "") or "document")
        source_path = None
    else:
        try:
            body = await request.json()
        except ValueError:
            raise api_error(400, "invalid_json", "Corps attendu : multipart « file » ou JSON {\"path\": \"…\"}.")
        path = (body or {}).get("path") if isinstance(body, dict) else None
        if not path or not isinstance(path, str):
            raise api_error(422, "validation_error", "Indiquez « path » (chemin d'un fichier local) ou envoyez un fichier.")
        abs_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(abs_path):
            raise api_error(404, "file_not_found", "Fichier introuvable : %s" % path)
        try:
            with open(abs_path, "rb") as fh:
                data = fh.read(MAX_UPLOAD_BYTES + 1)
        except OSError as exc:
            raise api_error(400, "unreadable_file", "Lecture impossible : %s (%s)." % (path, exc))
        filename = os.path.basename(abs_path)
        source_path = abs_path
    return await run_in_threadpool(store_document, state, data, filename, source_path)


def _doc_or_404(db: Database, doc_id: int) -> DocumentRow:
    doc = db.get_document(doc_id)
    if doc is None:
        raise api_error(404, "document_not_found", "Document %s introuvable." % doc_id)
    return doc


@router.get("/{doc_id}")
def get_document(doc_id: int, request: Request) -> Dict[str, Any]:
    """Métadonnées + ``chunks_preview[]`` + ``mentions:[{kind, name, exists}]`` (dérive doc / code)."""
    db = require_db(get_state(request))
    doc = _doc_or_404(db, doc_id)
    chunks = db.chunks_for("document", doc_id)
    data = document_dict(doc, len(chunks), _stale_flags(db, [doc_id]).get(doc_id, False))
    data["chunks_preview"] = [
        {"id": c.id, "ordinal": c.ordinal, "line_start": c.line_start, "line_end": c.line_end,
         "page": (c.meta or {}).get("page"), "heading_path": (c.meta or {}).get("heading_path"),
         "snippet": (c.text or "")[:PREVIEW_CHARS]}
        for c in chunks[:PREVIEW_CHUNKS]]
    mentions_mod = _mentions_module()
    mentions: List[Dict[str, Any]] = []
    if mentions_mod is not None:
        try:
            for m in mentions_mod.document_mentions(db, "document", doc_id):
                mentions.append({"kind": m.get("kind"), "name": m.get("name"), "exists": bool(m.get("exists")),
                                 "count": m.get("count"), "code_like": m.get("code_like"), "nodes": m.get("nodes")})
        except Exception:  # noqa: BLE001
            mentions = []
    data["mentions"] = mentions
    data["meta"] = dict(doc.meta or {})
    return data


@router.get("/{doc_id}/content")
def get_document_content(doc_id: int, request: Request) -> Dict[str, Any]:
    """Texte extrait du document (fichier stocké ; à défaut, concaténation des morceaux indexés)."""
    db = require_db(get_state(request))
    doc = _doc_or_404(db, doc_id)
    text = ""
    source = "chunks"
    if doc.stored_path and os.path.isfile(doc.stored_path):
        loaded = load_document(doc.stored_path)
        text = loaded.text or ""
        source = "file"
    if not text:
        text = "\n\n".join(c.text for c in db.chunks_for("document", doc_id))
        source = "chunks"
    return {"id": doc.id, "title": doc.title, "format": doc.format, "text": text, "source": source}


@router.delete("/{doc_id}")
def delete_document(doc_id: int, request: Request) -> Dict[str, Any]:
    """Supprime la copie locale (si elle est dans ``documents/``), les morceaux et les relations."""
    state = get_state(request)
    db = require_db(state)
    ws = state.workspace
    doc = _doc_or_404(db, doc_id)
    removed_file = False
    if doc.stored_path and ws is not None:
        docs_dir = os.path.realpath(str(config.documents_dir(ws.id)))
        try:
            real = os.path.realpath(doc.stored_path)
            if real.startswith(docs_dir + os.sep) and os.path.isfile(real):
                os.remove(real)
                removed_file = True
        except OSError:
            removed_file = False
    db.delete_document(doc_id)
    if state.search_engine is not None:
        state.search_engine._matrix_cache = None
    return {"deleted": True, "id": doc_id, "file_removed": removed_file}
