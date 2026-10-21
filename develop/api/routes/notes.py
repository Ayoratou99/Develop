"""Journal : notes internes datées, taguées, avec statut.

Choix d'implémentation : ``NoteRow`` n'a pas de colonnes ``status`` ni ``supersedes``. Ils sont
encodés dans la liste ``tags`` sous la forme ``status:<valid|superseded|historical>`` et
``supersedes:<id>``, et **filtrés** des ``tags`` renvoyés par l'API (:func:`split_tags` /
:func:`encode_tags`). L'indexeur voit donc ces marqueurs dans l'en-tête du morceau, ce qui
rend une note « remplacée » reconnaissable par la recherche. Quand une note en remplace une
autre (``supersedes``), l'ancienne note encore ``valid`` passe automatiquement en ``superseded``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field, field_validator

from develop.api.routes import api_error, get_state, require_db
from develop.indexing import Indexer
from develop.storage import Database, NoteRow

router = APIRouter(prefix="/api/notes", tags=["notes"])

STATUSES = ("valid", "superseded", "historical")
STATUS_PREFIX = "status:"
SUPERSEDES_PREFIX = "supersedes:"
EXCERPT_CHARS = 160
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WS_RE = re.compile(r"\s+")


def split_tags(tags: List[Any]) -> Tuple[List[str], str, Optional[int]]:
    """``(tags visibles, status, supersedes)`` à partir des tags stockés."""
    clean: List[str] = []
    status = "valid"
    supersedes: Optional[int] = None
    for raw in tags or []:
        t = str(raw).strip()
        if not t:
            continue
        low = t.lower()
        if low.startswith(STATUS_PREFIX):
            value = low[len(STATUS_PREFIX):]
            if value in STATUSES:
                status = value
            continue
        if low.startswith(SUPERSEDES_PREFIX):
            try:
                supersedes = int(low[len(SUPERSEDES_PREFIX):])
            except ValueError:
                pass
            continue
        if t not in clean:
            clean.append(t)
    return clean, status, supersedes


def encode_tags(tags: List[str], status: str, supersedes: Optional[int]) -> List[str]:
    out = [t.strip() for t in tags if t and t.strip() and not t.strip().lower().startswith((STATUS_PREFIX, SUPERSEDES_PREFIX))]
    out.append(STATUS_PREFIX + (status if status in STATUSES else "valid"))
    if supersedes is not None:
        out.append(SUPERSEDES_PREFIX + str(int(supersedes)))
    return out


def note_dict(n: NoteRow) -> Dict[str, Any]:
    tags, status, supersedes = split_tags(n.tags)
    body = n.body or ""
    excerpt = _WS_RE.sub(" ", body).strip()
    if len(excerpt) > EXCERPT_CHARS:
        excerpt = excerpt[:EXCERPT_CHARS - 1].rstrip() + "…"
    return {"id": n.id, "title": n.title, "body": body, "tags": tags, "note_date": n.note_date, "status": status,
            "supersedes": supersedes, "excerpt": excerpt, "created_at": n.created_at, "updated_at": n.updated_at}


def _check_date(value: Optional[str]) -> Optional[str]:
    if value is None or value == "":
        return None
    if not _DATE_RE.match(value):
        raise ValueError("date attendue au format AAAA-MM-JJ")
    return value


class NoteCreate(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    body: str = ""
    tags: List[str] = Field(default_factory=list)
    note_date: Optional[str] = None
    status: str = "valid"
    supersedes: Optional[Union[int, str]] = None

    @field_validator("note_date")
    @classmethod
    def _date(cls, v: Optional[str]) -> Optional[str]:
        return _check_date(v)

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        if v not in STATUSES:
            raise ValueError("statut attendu : %s" % ", ".join(STATUSES))
        return v


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=300)
    body: Optional[str] = None
    tags: Optional[List[str]] = None
    note_date: Optional[str] = None
    status: Optional[str] = None
    supersedes: Optional[Union[int, str]] = None

    @field_validator("note_date")
    @classmethod
    def _date(cls, v: Optional[str]) -> Optional[str]:
        return _check_date(v)

    @field_validator("status")
    @classmethod
    def _status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in STATUSES:
            raise ValueError("statut attendu : %s" % ", ".join(STATUSES))
        return v


def _supersedes_id(value: Optional[Union[int, str]]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise api_error(422, "validation_error", "« supersedes » doit être l'identifiant numérique d'une note.")


def _index_note(state: Any, db: Database, note_id: int) -> None:
    ws = state.workspace
    Indexer(db, ws, state.llm, ws.llm.embedding_model if ws.llm.engine != "none" else "").index_note(note_id)
    if state.search_engine is not None:
        state.search_engine._matrix_cache = None


def _mark_superseded(state: Any, db: Database, old_id: Optional[int], new_id: int) -> None:
    if old_id is None or old_id == new_id:
        return
    old = db.get_note(old_id)
    if old is None:
        return
    tags, status, sup = split_tags(old.tags)
    if status == "valid":
        old.tags = encode_tags(tags, "superseded", sup)
        db.update_note(old)
        _index_note(state, db, old_id)


def _note_or_404(db: Database, note_id: int) -> NoteRow:
    n = db.get_note(note_id)
    if n is None:
        raise api_error(404, "note_not_found", "Note %s introuvable." % note_id)
    return n


# --------------------------------------------------------------------------- routes

@router.get("")
def list_notes(request: Request, date_from: Optional[str] = Query(default=None, alias="from"),
               date_to: Optional[str] = Query(default=None, alias="to"), tag: Optional[str] = None,
               q: Optional[str] = None, status: Optional[str] = None,
               limit: int = Query(default=200, ge=1, le=5000)) -> List[Dict[str, Any]]:
    """Notes triées par date décroissante ; filtres inclusifs ``from``/``to``, ``tag``, ``q`` (titre/corps), ``status``."""
    db = require_db(get_state(request))
    for name, value in (("from", date_from), ("to", date_to)):
        if value and not _DATE_RE.match(value):
            raise api_error(422, "validation_error", "« %s » doit être une date AAAA-MM-JJ." % name)
    if status and status not in STATUSES:
        raise api_error(422, "validation_error", "« status » doit valoir %s." % ", ".join(STATUSES))
    needs_filter = bool((q and q.strip()) or status)
    rows = db.list_notes(date_from=date_from or None, date_to=date_to or None, tag=(tag or "").strip() or None,
                         limit=limit if not needs_filter else 100000)
    needle = (q or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for n in rows:
        d = note_dict(n)
        if status and d["status"] != status:
            continue
        if needle and needle not in ("%s %s %s" % (n.title or "", n.body or "", " ".join(d["tags"]))).lower():
            continue
        out.append(d)
        if len(out) >= limit:
            break
    return out


@router.post("", status_code=201)
def create_note(body: NoteCreate, request: Request) -> Dict[str, Any]:
    state = get_state(request)
    db = require_db(state)
    supersedes = _supersedes_id(body.supersedes)
    note = NoteRow(title=body.title.strip(), body=body.body or "", tags=encode_tags(body.tags, body.status, supersedes),
                   note_date=body.note_date or None)
    note_id = db.add_note(note)
    _index_note(state, db, note_id)
    _mark_superseded(state, db, supersedes, note_id)
    return note_dict(_note_or_404(db, note_id))


@router.get("/{note_id}")
def get_note(note_id: int, request: Request) -> Dict[str, Any]:
    return note_dict(_note_or_404(require_db(get_state(request)), note_id))


@router.put("/{note_id}")
def update_note(note_id: int, body: NoteUpdate, request: Request) -> Dict[str, Any]:
    """Mise à jour (corps partiel accepté) puis réindexation de la note."""
    state = get_state(request)
    db = require_db(state)
    n = _note_or_404(db, note_id)
    tags, status, supersedes = split_tags(n.tags)
    if body.title is not None:
        if not body.title.strip():
            raise api_error(422, "validation_error", "Le titre ne peut pas être vide.")
        n.title = body.title.strip()
    if body.body is not None:
        n.body = body.body
    if body.tags is not None:
        tags = list(body.tags)
    if body.status is not None:
        status = body.status
    if "supersedes" in body.model_fields_set:
        supersedes = _supersedes_id(body.supersedes)
    if body.note_date is not None:
        n.note_date = body.note_date
    n.tags = encode_tags(tags, status, supersedes)
    db.update_note(n)
    _index_note(state, db, note_id)
    _mark_superseded(state, db, supersedes, note_id)
    return note_dict(_note_or_404(db, note_id))


@router.delete("/{note_id}")
def delete_note(note_id: int, request: Request) -> Dict[str, Any]:
    state = get_state(request)
    db = require_db(state)
    _note_or_404(db, note_id)
    db.delete_note(note_id)
    if state.search_engine is not None:
        state.search_engine._matrix_cache = None
    return {"deleted": True, "id": note_id}
