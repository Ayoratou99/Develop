"""Tests de l'indexeur incrémental (``develop.indexing``).

La fixture ``sample_project`` est copiée dans ``tmp_path`` pour pouvoir la modifier ; la mémoire
Develop vit dans ``DEVELOP_HOME = tmp_path``. Le moteur d'embeddings est un faux client en
mémoire qui renvoie des vecteurs déterministes.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import threading
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from develop.code_analysis import get_extractor
from develop.config import RepoConfig, WorkspaceConfig, documents_dir
from develop.git_tools import git_available
from develop.indexing import (
    EMBEDDING_MODEL_META_KEY,
    EXTRACTOR_VERSION,
    PHASES,
    PROGRESS_EVERY,
    Embedder,
    IndexCancelled,
    Indexer,
    IndexProgress,
    normalize_vectors,
    run_coroutine_sync,
)
from develop.indexing.indexer import NOTE_SPLIT_CHARS
from develop.llm.base import LLMClient, LLMError, LLMUnavailable
from develop.project_scanner import FileEntry, scan_repo
from develop.storage import Database, DocumentRow, NoteRow, RepoRow, node_key

REPO_ID = "proj"
REPO_NAME = "sample"


# --------------------------------------------------------------------------- faux moteur d'embeddings

class FakeEmbedClient(LLMClient):
    """Client LLM en mémoire : ``embed`` renvoie des vecteurs déterministes (dérivés d'un sha1)."""

    engine = "fake"

    def __init__(self, dim: int = 8, fail_with: Optional[Exception] = None, fail_after_calls: Optional[int] = None):
        self.dim = dim
        self.fail_with = fail_with
        self.fail_after_calls = fail_after_calls
        self.calls: List[Tuple[str, int]] = []  # (modèle, nb de textes)

    async def embed(self, texts: Sequence[str], model: str) -> List[List[float]]:
        if self.fail_with is not None and (self.fail_after_calls is None or len(self.calls) >= self.fail_after_calls):
            raise self.fail_with
        self.calls.append((model, len(texts)))
        out: List[List[float]] = []
        for text in texts:
            digest = hashlib.sha1(("%s|%s" % (model, text)).encode("utf-8")).digest()
            out.append([float(digest[i % len(digest)]) + 1.0 for i in range(self.dim)])
        return out


# --------------------------------------------------------------------------- aides

def _make_pdf(path: Path, texts: List[str]) -> None:
    """Écrit un PDF minimal valide avec une page de texte par élément de ``texts`` (via pypdf)."""
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, StreamObject

    writer = PdfWriter()
    add_object = getattr(writer, "add_object", None) or getattr(writer, "_add_object")
    for txt in texts:
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): add_object(font)}),
        })
        stream = StreamObject()
        stream.set_data(("BT /F1 12 Tf 20 150 Td (%s) Tj ET" % txt).encode("latin-1"))
        page[NameObject("/Contents")] = add_object(stream)
    with open(path, "wb") as fh:
        writer.write(fh)


def _chunk_ids(db: Database) -> Dict[Tuple[str, int, int], int]:
    """``{(source_type, source_id, ordinal): chunk_id}`` pour vérifier la stabilité des morceaux."""
    rows = db.query("SELECT id, source_type, source_id, ordinal FROM chunks")
    return {(r["source_type"], int(r["source_id"]), int(r["ordinal"])): int(r["id"]) for r in rows}


def _file_chunk_ids(db: Database, path: str) -> List[int]:
    f = db.get_file(REPO_ID, path)
    assert f is not None, path
    return [c.id for c in db.chunks_for("file", f.id)]  # type: ignore[misc]


def _bump_mtime(path: Path, seconds: float = 30.0) -> None:
    st = path.stat()
    os.utime(path, (st.st_atime + seconds, st.st_mtime + seconds))


def _relation_kinds(db: Database) -> Dict[str, int]:
    return {r["kind"]: int(r["n"]) for r in db.query("SELECT kind, count(*) AS n FROM relations GROUP BY kind")}


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def develop_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("DEVELOP_HOME", str(home))
    return home


@pytest.fixture
def project(tmp_path: Path, sample_project: Path) -> Path:
    """Copie modifiable de la fixture (jamais la fixture elle-même)."""
    dst = tmp_path / "sample_project"
    shutil.copytree(sample_project, dst)
    return dst


@pytest.fixture
def workspace(project: Path) -> WorkspaceConfig:
    return WorkspaceConfig(id="ws", name="Espace de test",
                           repos=[RepoConfig(id=REPO_ID, name=REPO_NAME, path=str(project), kind="other")])


@pytest.fixture
def db(tmp_path: Path):
    with Database(tmp_path / "memory.sqlite3") as database:
        yield database


# --------------------------------------------------------------------------- premier run

def test_first_run_indexes_everything(db: Database, workspace: WorkspaceConfig, project: Path):
    indexer = Indexer(db, workspace)
    started = time.time()
    run = indexer.run()
    assert run.status == "ok" and run.error is None
    assert time.time() - started < (10.0 if sys.platform == "win32" else 3.0)  # E/S fichier plus lentes sous Windows

    stats = run.stats
    for key in ("files_scanned", "files_indexed", "files_unchanged", "files_deleted", "files_skipped", "symbols",
                "endpoints", "models", "chunks", "documents", "notes", "relations", "embeddings", "errors",
                "warnings", "duration_s", "repos"):
        assert key in stats, key
    assert isinstance(stats["files_skipped"], dict) and stats["files_skipped"].get("binary", 0) >= 1
    assert stats["files_scanned"] > 20
    assert stats["files_indexed"] == stats["files_scanned"]
    assert stats["files_unchanged"] == 0 and stats["files_deleted"] == 0
    assert stats["chunks"] > 0 and stats["errors"] == []
    assert stats["repos"][REPO_ID]["status"] == "ok"
    assert stats["repos"][REPO_ID]["files_indexed"] == stats["files_scanned"]
    assert isinstance(stats["duration_s"], float)
    # avec des extracteurs présents, le code produit symboles, endpoints et modèles
    if get_extractor("php") is not None:
        assert stats["symbols"] > 0 and stats["endpoints"] > 0 and stats["models"] > 0
        assert _relation_kinds(db).get("imports_raw", 0) > 0

    # la base reflète les stats
    totals = db.stats()
    assert totals["files"] == stats["files_scanned"]
    assert totals["chunks"] == stats["chunks"]
    assert totals["symbols"] == stats["symbols"]
    # métadonnées de fin de run
    assert db.get_meta("extractor_version") == EXTRACTOR_VERSION
    assert db.get_meta("last_index_at") is not None
    repo = db.get_repo(REPO_ID)
    assert repo is not None and repo.last_indexed_at is not None and repo.name == REPO_NAME
    last = db.last_index_run()
    assert last is not None and last.id == run.id and last.status == "ok" and last.finished_at is not None
    assert last.stats["files_scanned"] == stats["files_scanned"]


def test_code_chunks_have_header_and_body_offset(db: Database, workspace: WorkspaceConfig, project: Path):
    Indexer(db, workspace).run()
    path = "backend/routes/api.php"
    original = (project / path).read_text(encoding="utf-8")
    chunks = db.chunks_for("file", db.get_file(REPO_ID, path).id)  # type: ignore[union-attr]
    assert chunks
    first = chunks[0]
    header_line = first.text.split("\n", 1)[0]
    assert header_line.startswith("# %s/%s L%d-%d" % (REPO_NAME, path, first.line_start, first.line_end))
    offset = first.meta["body_offset"]
    assert offset == len(header_line) + 1
    body = first.text[offset:]
    assert body.startswith("<?php")
    # le corps est bien le texte brut des lignes annoncées
    lines = original.splitlines()
    assert body == "\n".join(lines[first.line_start - 1:first.line_end])
    # la recherche lexicale retrouve le fichier par son chemin et le symbole par son nom
    hits = db.fts_search("api.php", source_types=["file"])
    assert any(h.chunk_id == first.id for h in hits)


def test_markdown_repo_files_are_doc_chunks_with_heading_path(db: Database, workspace: WorkspaceConfig):
    Indexer(db, workspace).run()
    f = db.get_file(REPO_ID, "docs/dictionnaire-donnees.md")
    assert f is not None and f.kind == "doc" and f.language == "markdown"
    chunks = db.chunks_for("file", f.id)  # type: ignore[arg-type]
    assert len(chunks) >= 2
    assert all("heading_path" in c.meta for c in chunks)
    assert any(any("order_lines" in h for h in c.meta["heading_path"]) for c in chunks)
    # pas d'en-tête de code sur un document
    assert not chunks[0].text.startswith("# %s/" % REPO_NAME)
    assert "body_offset" not in chunks[0].meta


def test_sql_file_is_analyzed_and_defines_tables(db: Database, workspace: WorkspaceConfig):
    if get_extractor("sql") is None:
        pytest.skip("extracteur SQL absent")
    Indexer(db, workspace).run()
    f = db.get_file(REPO_ID, "database/schema.sql")
    assert f is not None and f.kind == "data"
    models = [m for m in db.list_models() if m.file_id == f.id]
    assert {m.table_name for m in models} >= {"orders", "order_lines"}
    for m in models:
        assert db.relations_from(node_key("model", m.id), "defines_table")  # type: ignore[arg-type]
    assert db.relations_to(node_key("table", "orders"), "defines_table")


# --------------------------------------------------------------------------- incrémental

def test_second_run_without_change_is_noop(db: Database, workspace: WorkspaceConfig):
    indexer = Indexer(db, workspace)
    first = indexer.run()
    before = _chunk_ids(db)
    second = indexer.run()
    assert second.status == "ok"
    assert second.stats["files_scanned"] == first.stats["files_scanned"]
    assert second.stats["files_unchanged"] == second.stats["files_scanned"]
    assert second.stats["files_indexed"] == 0 and second.stats["chunks"] == 0 and second.stats["symbols"] == 0
    assert _chunk_ids(db) == before
    assert db.stats()["chunks"] == first.stats["chunks"]


def test_modified_file_is_the_only_one_reprocessed(db: Database, workspace: WorkspaceConfig, project: Path):
    indexer = Indexer(db, workspace)
    indexer.run()
    before = _chunk_ids(db)
    target = "frontend/src/utils/money.ts"
    old_ids = set(_file_chunk_ids(db, target))
    file_id = db.get_file(REPO_ID, target).id  # type: ignore[union-attr]
    p = project / target
    p.write_text(p.read_text(encoding="utf-8") + "\nexport function formatCents(cents: number): number {\n  return cents / 100;\n}\n",
                 encoding="utf-8")
    _bump_mtime(p)

    run = indexer.run()
    assert run.status == "ok"
    assert run.stats["files_indexed"] == 1
    assert run.stats["files_unchanged"] == run.stats["files_scanned"] - 1
    # le fichier garde son id, ses morceaux sont neufs, ceux des autres fichiers sont intacts
    assert db.get_file(REPO_ID, target).id == file_id  # type: ignore[union-attr]
    new_ids = set(_file_chunk_ids(db, target))
    assert new_ids and not (new_ids & old_ids)
    after = _chunk_ids(db)
    for key, cid in before.items():
        if key[0] == "file" and key[1] == file_id:
            continue
        assert after.get(key) == cid, key
    if get_extractor("typescript") is not None:
        assert any(s.name == "formatCents" for s in db.list_symbols(file_id=file_id))


def test_touched_file_with_same_content_is_not_reprocessed(db: Database, workspace: WorkspaceConfig, project: Path):
    indexer = Indexer(db, workspace)
    indexer.run()
    target = "tools/report.py"
    before = _file_chunk_ids(db, target)
    _bump_mtime(project / target, 120.0)
    run = indexer.run()
    assert run.stats["files_indexed"] == 0
    assert run.stats["files_unchanged"] == run.stats["files_scanned"]
    assert _file_chunk_ids(db, target) == before
    # le mtime en base est rafraîchi : le prochain run ne relira même plus le fichier
    f = db.get_file(REPO_ID, target)
    assert f is not None and abs(float(f.mtime or 0) - (project / target).stat().st_mtime) < 1e-6
    assert indexer.stale_files(REPO_ID) == []


def test_deleted_file_cascades(db: Database, workspace: WorkspaceConfig, project: Path):
    indexer = Indexer(db, workspace)
    indexer.run()
    target = "backend/app/Models/OrderLine.php"
    f = db.get_file(REPO_ID, target)
    assert f is not None
    file_id = int(f.id)  # type: ignore[arg-type]
    symbol_ids = [s.id for s in db.list_symbols(file_id=file_id)]
    assert db.chunks_for("file", file_id)
    (project / target).unlink()

    run = indexer.run()
    assert run.status == "ok" and run.stats["files_deleted"] == 1
    assert db.get_file(REPO_ID, target) is None
    assert db.chunks_for("file", file_id) == []
    assert db.query("SELECT count(*) AS n FROM symbols WHERE file_id = ?", (file_id,))[0]["n"] == 0
    assert db.relations_from(node_key("file", file_id)) == []
    for sid in symbol_ids:
        assert db.get_symbol(sid) is None  # type: ignore[arg-type]
        assert db.relations_to(node_key("symbol", sid)) == []  # type: ignore[arg-type]


def test_added_file_is_indexed(db: Database, workspace: WorkspaceConfig, project: Path):
    indexer = Indexer(db, workspace)
    first = indexer.run()
    new = project / "tools" / "extra.py"
    new.write_text("def extra_helper(x):\n    \"\"\"Aide.\"\"\"\n    return x * 2\n", encoding="utf-8")
    run = indexer.run()
    assert run.stats["files_scanned"] == first.stats["files_scanned"] + 1
    assert run.stats["files_indexed"] == 1 and run.stats["files_unchanged"] == first.stats["files_scanned"]
    f = db.get_file(REPO_ID, "tools/extra.py")
    assert f is not None and f.language == "python" and f.kind == "code"
    chunks = db.chunks_for("file", f.id)  # type: ignore[arg-type]
    assert len(chunks) == 1 and "extra_helper" in chunks[0].text
    if get_extractor("python") is not None:
        assert [s.name for s in db.list_symbols(file_id=f.id)] == ["extra_helper"]


def test_full_run_reprocesses_all_files(db: Database, workspace: WorkspaceConfig):
    indexer = Indexer(db, workspace)
    first = indexer.run()
    before = _chunk_ids(db)
    run = indexer.run(full=True)
    assert run.status == "ok"
    assert run.stats["full"] is True
    assert run.stats["files_indexed"] == first.stats["files_scanned"] and run.stats["files_unchanged"] == 0
    after = _chunk_ids(db)
    assert set(after) == set(before) and all(after[k] != before[k] for k in before)
    assert db.stats()["chunks"] == first.stats["chunks"]


# --------------------------------------------------------------------------- dépôts

def test_repo_removed_from_config_is_deleted_from_db(db: Database, workspace: WorkspaceConfig, tmp_path: Path):
    other = tmp_path / "other_repo"
    other.mkdir()
    (other / "util.py").write_text("def util():\n    return 1\n", encoding="utf-8")
    workspace.repos.append(RepoConfig(id="other", name="Autre", path=str(other), kind="backend"))
    indexer = Indexer(db, workspace)
    run = indexer.run()
    assert run.status == "ok"
    assert sorted(r.id for r in db.list_repos()) == ["other", REPO_ID]
    other_file = db.get_file("other", "util.py")
    assert other_file is not None and db.chunks_for("file", other_file.id)  # type: ignore[arg-type]

    workspace.repos = [r for r in workspace.repos if r.id != "other"]
    run = Indexer(db, workspace).run()
    assert run.status == "ok"
    assert [r.id for r in db.list_repos()] == [REPO_ID]
    assert db.list_files("other") == []
    assert db.chunks_for("file", other_file.id) == []  # type: ignore[arg-type]
    assert run.stats["repos_removed"] == ["other"]
    assert any("Autre" in w for w in run.stats["warnings"])


def test_missing_directory_is_partial_and_logged(db: Database, workspace: WorkspaceConfig, tmp_path: Path):
    missing = str(tmp_path / "nowhere")
    workspace.repos.append(RepoConfig(id="ghost", name="Fantôme", path=missing, kind="docs"))
    run = Indexer(db, workspace).run()
    assert run.status == "partial"
    errors = run.stats["errors"]
    assert any(e["stage"] == "scan" and e["path"] == missing for e in errors)
    assert run.stats["repos"]["ghost"]["status"] == "error"
    assert "introuvable" in run.stats["repos"]["ghost"]["error"]
    # l'autre dépôt a été indexé normalement
    assert run.stats["repos"][REPO_ID]["status"] == "ok" and run.stats["files_indexed"] > 0
    assert db.get_repo("ghost") is not None  # il reste configuré, simplement vide
    assert db.last_index_run().status == "partial"  # type: ignore[union-attr]


def test_disabled_repo_and_single_repo_run(db: Database, workspace: WorkspaceConfig, tmp_path: Path):
    other = tmp_path / "other_repo"
    other.mkdir()
    (other / "a.py").write_text("x = 1\n", encoding="utf-8")
    workspace.repos.append(RepoConfig(id="other", name="Autre", path=str(other), enabled=False))
    run = Indexer(db, workspace).run()
    assert run.stats["repos"]["other"]["status"] == "disabled"
    assert db.list_files("other") == []
    # run limité à un seul dépôt (même désactivé : la demande est explicite)
    run = Indexer(db, workspace).run(repo_id="other")
    assert run.status == "ok" and run.stats["files_indexed"] == 1 and REPO_ID not in run.stats["repos"]
    assert db.get_file("other", "a.py") is not None
    # dépôt inconnu : erreur consignée, pas d'exception
    run = Indexer(db, workspace).run(repo_id="nope")
    assert run.status == "partial" and any("nope" in e["message"] for e in run.stats["errors"])


@pytest.mark.skipif(not git_available(), reason="git absent")
def test_git_head_sha_recorded(db: Database, workspace: WorkspaceConfig, project: Path):
    def git(*args: str) -> str:
        out = subprocess.run(["git", "-C", str(project), "-c", "user.email=t@example.com", "-c", "user.name=t",
                              *args], check=True, capture_output=True, text=True)
        return out.stdout.strip()

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    head = git("rev-parse", "HEAD")
    run = Indexer(db, workspace).run()
    assert run.status == "ok"
    assert run.stats["repos"][REPO_ID]["used_git"] is True
    assert run.stats["repos"][REPO_ID]["head_sha"] == head
    assert db.get_repo(REPO_ID).head_sha == head  # type: ignore[union-attr]


# --------------------------------------------------------------------------- annulation, versions, progression

def test_cancel_event_gives_cancelled_status(db: Database, workspace: WorkspaceConfig):
    cancel = threading.Event()
    seen: List[str] = []
    armed = [True]  # n'annule que le premier run

    def on_progress(p: IndexProgress) -> None:
        seen.append(p.phase)
        if p.phase == "analyze" and armed[0]:
            armed[0] = False
            cancel.set()

    indexer = Indexer(db, workspace, progress=on_progress, cancel_event=cancel)
    run = indexer.run()
    assert run.status == "cancelled"
    assert run.stats["files_indexed"] == 0
    assert "documents" not in seen and seen[-1] == "done"
    assert db.last_index_run().status == "cancelled"  # type: ignore[union-attr]
    assert db.get_meta("last_index_at") is None
    # une fois l'événement levé, le même indexeur termine le travail
    cancel.clear()
    run = indexer.run()
    assert run.status == "ok" and run.stats["files_indexed"] == run.stats["files_scanned"]


def test_cancel_before_start(db: Database, workspace: WorkspaceConfig):
    cancel = threading.Event()
    cancel.set()
    run = Indexer(db, workspace, cancel_event=cancel).run()
    assert run.status == "cancelled" and db.stats()["files"] == 0


def test_extractor_version_change_forces_reextraction(db: Database, workspace: WorkspaceConfig):
    note_id = db.add_note(NoteRow(title="Décision", body="On garde SQLite.", tags=["archi"], note_date="2026-10-01"))
    indexer = Indexer(db, workspace)
    first = indexer.run()
    note_chunks = [c.id for c in db.chunks_for("note", note_id)]
    assert note_chunks
    before = _chunk_ids(db)

    db.set_meta("extractor_version", "0")
    run = indexer.run()
    assert run.status == "ok"
    assert run.stats["reextract"] is True and run.stats["full"] is False
    assert run.stats["files_indexed"] == first.stats["files_scanned"] and run.stats["files_unchanged"] == 0
    assert any("extracteurs" in w for w in run.stats["warnings"])
    assert db.get_meta("extractor_version") == EXTRACTOR_VERSION
    # les fichiers ont de nouveaux morceaux, la note n'a pas été retouchée
    after = _chunk_ids(db)
    assert all(after[k] != before[k] for k in before if k[0] == "file")
    assert [c.id for c in db.chunks_for("note", note_id)] == note_chunks
    assert run.stats["notes"] == 0

    # troisième run : plus rien à refaire
    run = indexer.run()
    assert run.stats["reextract"] is False and run.stats["files_indexed"] == 0


def test_progress_phases_in_order_and_every_n_files(db: Database, workspace: WorkspaceConfig):
    events: List[IndexProgress] = []
    run = Indexer(db, workspace, progress=events.append).run()
    assert run.status == "ok"
    phases = [e.phase for e in events]
    assert phases[0] == "scan" and phases[-1] == "done"
    assert set(phases) == set(PHASES)
    order = [PHASES.index(p) for p in phases]
    assert order == sorted(order), phases
    assert all(isinstance(e.current, int) and isinstance(e.total, int) for e in events)
    analyze = [e for e in events if e.phase == "analyze"]
    assert analyze and all(e.repo == REPO_ID for e in analyze)
    total = analyze[0].total
    assert total == run.stats["files_scanned"]
    assert analyze[-1].current == total
    if total > PROGRESS_EVERY:
        assert any(e.current == PROGRESS_EVERY for e in analyze)
    assert events[-1].to_dict()["message"] == "Indexation terminée."


def test_progress_callback_errors_do_not_break_run(db: Database, workspace: WorkspaceConfig):
    def broken(_: IndexProgress) -> None:
        raise RuntimeError("UI cassée")

    run = Indexer(db, workspace, progress=broken).run()
    assert run.status == "ok" and run.stats["files_indexed"] > 0


# --------------------------------------------------------------------------- notes et documents

def test_note_is_one_chunk_with_dated_header(db: Database, workspace: WorkspaceConfig):
    note_id = db.add_note(NoteRow(title="Réunion statuts", body="Ligne 1\nLigne 2\nLigne 3",
                                  tags=["réunion", "API"], note_date="2026-09-01"))
    empty_id = db.add_note(NoteRow(title="Vide", body="", note_date="2026-09-02"))
    indexer = Indexer(db, workspace)
    run = indexer.run()
    assert run.stats["notes"] == 2
    chunks = db.chunks_for("note", note_id)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.text.startswith("Note du 2026-09-01 — Réunion statuts [réunion, API]\n\n")
    assert c.text[c.meta["body_offset"]:] == "Ligne 1\nLigne 2\nLigne 3"
    assert c.line_start == 1 and c.line_end == 3
    assert c.meta["note_date"] == "2026-09-01" and c.meta["tags"] == ["réunion", "API"]
    assert len(db.chunks_for("note", empty_id)) == 1
    # second run : notes déjà indexées, rien à refaire ; full → réindexées
    assert indexer.run().stats["notes"] == 0
    assert indexer.run(full=True).stats["notes"] == 2
    # réindexation directe après modification
    note = db.get_note(note_id)
    assert note is not None
    note.body = "Nouveau corps"
    db.update_note(note)
    assert indexer.index_note(note_id) == 1
    assert db.chunks_for("note", note_id)[0].text.endswith("Nouveau corps")
    assert indexer.index_note(999_999) == 0 and indexer.errors[-1]["stage"] == "note"


def test_long_note_is_split_with_header_on_each_part(db: Database, workspace: WorkspaceConfig):
    paragraphs = ["Paragraphe %d. " % i + ("Texte de remplissage. " * 20) for i in range(60)]
    body = "\n\n".join(paragraphs)
    assert len(body) > NOTE_SPLIT_CHARS
    note_id = db.add_note(NoteRow(title="Longue", body=body, note_date="2026-10-05"))
    n = Indexer(db, workspace).index_note(note_id)
    chunks = db.chunks_for("note", note_id)
    assert n == len(chunks) >= 2
    for c in chunks:
        assert c.text.startswith("Note du 2026-10-05 — Longue\n\n")
        assert len(c.text) <= NOTE_SPLIT_CHARS + 200
        assert c.meta["part"] == c.ordinal
    assert chunks[0].line_start == 1 and chunks[-1].line_end == len(body.splitlines())


def test_uploaded_markdown_document_is_indexed(db: Database, workspace: WorkspaceConfig, sample_project: Path):
    docs = documents_dir(workspace.id)
    docs.mkdir(parents=True, exist_ok=True)
    stored = docs / "dictionnaire.md"
    shutil.copy(sample_project / "docs" / "dictionnaire-donnees.md", stored)
    doc_id = db.add_document(DocumentRow(title="", stored_path=str(stored), kind="upload", format="md",
                                         sha1=hashlib.sha1(stored.read_bytes()).hexdigest()))
    indexer = Indexer(db, workspace)
    run = indexer.run()
    assert run.status == "ok" and run.stats["documents"] == 1
    chunks = db.chunks_for("document", doc_id)
    assert len(chunks) >= 2
    assert all(isinstance(c.meta.get("heading_path"), list) for c in chunks)
    assert any(any("order_lines" in h for h in c.meta["heading_path"]) for c in chunks)
    doc = db.get_document(doc_id)
    assert doc is not None
    assert doc.title == "Dictionnaire de données" and doc.meta["title"] == "Dictionnaire de données"
    assert doc.meta["looks_like_data_dictionary"] is True
    assert doc.meta["tables"] and all({"headers", "rows"} <= set(t) for t in doc.meta["tables"])
    assert doc.meta["chunk_count"] == len(chunks) and doc.meta["warnings"] == []
    assert doc.meta["pages"] is None
    # déjà indexé : pas refait ; full : refait ; appel direct : renvoie le nombre de morceaux
    assert indexer.run().stats["documents"] == 0
    assert indexer.run(full=True).stats["documents"] == 1
    assert indexer.index_document(doc_id) == len(chunks)
    assert indexer.index_document(123_456) == 0 and indexer.errors[-1]["stage"] == "document"


def test_uploaded_pdf_document_chunks_carry_pages(db: Database, workspace: WorkspaceConfig):
    docs = documents_dir(workspace.id)
    docs.mkdir(parents=True, exist_ok=True)
    pdf = docs / "spec.pdf"
    _make_pdf(pdf, ["Premiere page du document", "Deuxieme page du document", "Troisieme page"])
    doc_id = db.add_document(DocumentRow(title="Spec", stored_path=str(pdf), format="pdf"))
    indexer = Indexer(db, workspace)
    n = indexer.index_document(doc_id)
    chunks = db.chunks_for("document", doc_id)
    assert n == len(chunks) >= 1
    assert all(isinstance(c.meta.get("page"), int) and 1 <= c.meta["page"] <= 3 for c in chunks)
    assert chunks[0].meta["page"] == 1
    doc = db.get_document(doc_id)
    assert doc is not None and doc.meta["pages"] == 3 and doc.meta["format"] == "pdf"


def test_document_with_missing_file_is_skipped_with_warning(db: Database, workspace: WorkspaceConfig, tmp_path: Path):
    db.add_document(DocumentRow(title="Perdu", stored_path=str(tmp_path / "absent.md"), format="md"))
    db.add_document(DocumentRow(title="Sans fichier"))
    run = Indexer(db, workspace).run()
    assert run.status == "ok" and run.stats["documents"] == 0
    assert any("Perdu" in w and "introuvable" in w for w in run.stats["warnings"])


# --------------------------------------------------------------------------- fichiers particuliers

def test_real_env_file_is_registered_without_content(db: Database, workspace: WorkspaceConfig, project: Path):
    secret = "supersecret123xyz"
    (project / "frontend" / ".env").write_text("APP_ENV=local\nDB_PASSWORD=%s\n" % secret, encoding="utf-8")
    (project / "backend" / ".env.local").write_text("TOKEN=%s\n" % secret, encoding="utf-8")
    run = Indexer(db, workspace).run()
    assert run.status == "ok"
    assert run.stats["files_skipped"].get("env_secret") == 2
    for path in ("frontend/.env", "backend/.env.local"):
        f = db.get_file(REPO_ID, path)
        assert f is not None and f.kind == "config" and f.language == "env"
        assert db.chunks_for("file", f.id) == []  # type: ignore[arg-type]
    assert db.fts_search(secret) == []
    # le gabarit .env.example, lui, est indexé
    example = db.get_file(REPO_ID, "frontend/.env.example")
    assert example is not None and db.chunks_for("file", example.id)  # type: ignore[arg-type]


def test_index_file_never_raises(db: Database, workspace: WorkspaceConfig, project: Path):
    db.upsert_repo(RepoRow(id=REPO_ID, name=REPO_NAME, path=str(project)))
    indexer = Indexer(db, workspace)
    ghost = FileEntry(repo_id=REPO_ID, path="ghost.py", abs_path=str(project / "ghost.py"), size=10, mtime=1.0,
                      language="python", kind="code", is_binary=False)
    assert indexer.index_file(REPO_ID, ghost) is None
    assert indexer.errors and indexer.errors[-1]["path"] == "ghost.py" and indexer.errors[-1]["stage"] == "read"
    # texte fourni : indexé sans lecture disque
    fid = indexer.index_file(REPO_ID, ghost, text="def ghost():\n    return 1\n")
    assert fid is not None and db.chunks_for("file", fid)
    # dépôt inconnu en base : erreur de stockage consignée, pas d'exception
    assert indexer.index_file("unknown_repo", ghost, text="x = 1\n") is None
    assert indexer.errors[-1]["stage"] == "store"


def test_handles_defines_table_and_imports_raw_relations(db: Database, workspace: WorkspaceConfig, project: Path):
    if get_extractor("python") is None or get_extractor("php") is None:
        pytest.skip("extracteurs absents")
    api = project / "tools" / "api.py"
    api.write_text(
        "from fastapi import FastAPI\nfrom . import report\n\napp = FastAPI()\n\n"
        "@app.get(\"/items/{item_id}\")\ndef read_item(item_id: int):\n    return {\"id\": item_id}\n",
        encoding="utf-8")
    run = Indexer(db, workspace).run()
    assert run.status == "ok"
    kinds = _relation_kinds(db)
    assert kinds.get("handles", 0) >= 1 and kinds.get("defines_table", 0) >= 1 and kinds.get("imports_raw", 0) >= 1

    f = db.get_file(REPO_ID, "tools/api.py")
    assert f is not None
    endpoints = [e for e in db.list_endpoints(side="server") if e.file_id == f.id]
    assert len(endpoints) == 1 and endpoints[0].normalized_path == "/items/:param"
    handles = db.relations_from(node_key("endpoint", endpoints[0].id), "handles")  # type: ignore[arg-type]
    assert len(handles) == 1
    target = db.get_symbol(int(handles[0].dst.split(":", 1)[1]))
    assert target is not None and target.name == "read_item" and target.file_id == f.id
    imports = db.relations_from(node_key("file", f.id), "imports_raw")  # type: ignore[arg-type]
    by_module = {r.dst: r for r in imports}
    assert node_key("module", "fastapi") in by_module
    fastapi = by_module[node_key("module", "fastapi")]
    assert fastapi.meta["names"] == ["FastAPI"] and fastapi.meta["line"] == 1 and fastapi.meta["is_relative"] is False
    assert any(r.meta.get("is_relative") for r in imports)

    # migration Laravel → table ; route Laravel → normalized_path avec le préfixe (meta.full_path)
    migration = db.get_file(REPO_ID, "backend/database/migrations/2024_01_01_000000_create_orders_table.php")
    assert migration is not None
    models = [m for m in db.list_models() if m.file_id == migration.id and m.kind == "migration"]
    assert models and models[0].table_name == "orders"
    rel = db.relations_from(node_key("model", models[0].id), "defines_table")  # type: ignore[arg-type]
    assert rel and rel[0].dst == node_key("table", "orders") and rel[0].meta["model_kind"] == "migration"
    routes = db.get_file(REPO_ID, "backend/routes/api.php")
    assert routes is not None
    laravel = [e for e in db.list_endpoints(side="server") if e.file_id == routes.id]
    assert any(e.path == "/orders" and e.normalized_path == "/api/orders" for e in laravel)

    # réindexation : les relations brutes sont régénérées, pas dupliquées
    before = _relation_kinds(db)
    Indexer(db, workspace).run(full=True)
    assert _relation_kinds(db) == before


def test_linker_callable_is_used(db: Database, workspace: WorkspaceConfig):
    calls: List[Tuple[Database, WorkspaceConfig]] = []

    def linker(database: Database, ws: WorkspaceConfig) -> Dict[str, Any]:
        calls.append((database, ws))
        return {"relations": 3, "detail": "ok"}

    run = Indexer(db, workspace, linker=linker).run()
    assert run.status == "ok"
    assert calls == [(db, workspace)]
    assert run.stats["linker"] == {"relations": 3, "detail": "ok"}
    assert run.stats["relations"] >= 3

    def broken(database: Database, ws: WorkspaceConfig) -> None:
        raise RuntimeError("boum")

    run = Indexer(db, workspace, linker=broken).run()
    assert run.status == "partial"
    assert any(e["stage"] == "relations" and "boum" in e["message"] for e in run.stats["errors"])


def test_default_linker_missing_module_is_a_warning(db: Database, workspace: WorkspaceConfig, monkeypatch: pytest.MonkeyPatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("develop.knowledge_graph"):
            raise ImportError("No module named 'develop.knowledge_graph'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    run = Indexer(db, workspace).run()
    assert run.status == "ok"
    assert any("knowledge_graph" in w for w in run.stats["warnings"])
    assert run.stats["linker"] is None


# --------------------------------------------------------------------------- dérive de l'index

def test_stale_files_detects_changes_without_reading(db: Database, workspace: WorkspaceConfig, project: Path):
    indexer = Indexer(db, workspace)
    assert indexer.stale_files(REPO_ID)  # rien en base : tout est « à indexer »
    indexer.run()
    assert indexer.stale_files(REPO_ID) == []
    (project / "tools" / "report.py").write_text("print('changed')\n", encoding="utf-8")
    _bump_mtime(project / "tools" / "report.py")
    (project / "tools" / "new.py").write_text("x = 1\n", encoding="utf-8")
    (project / "frontend" / "src" / "utils" / "money.ts").unlink()
    stale = indexer.stale_files(REPO_ID)
    assert stale == sorted(["tools/report.py", "tools/new.py", "frontend/src/utils/money.ts"])
    # dépôt inconnu : liste vide, jamais d'exception
    assert indexer.stale_files("nope") == []
    # après réindexation, plus rien n'est obsolète
    indexer.run()
    assert indexer.stale_files(REPO_ID) == []


def test_stale_files_ignores_touch_with_same_content(db: Database, workspace: WorkspaceConfig, project: Path):
    """Un simple ``touch`` (mtime modifié, contenu identique) n'est pas une dérive : le sha1 tranche."""
    indexer = Indexer(db, workspace)
    indexer.run()
    target = "tools/report.py"
    path = project / target
    stored = db.get_file(REPO_ID, target)
    assert stored is not None and stored.sha1
    _bump_mtime(path, 120.0)
    assert float(stored.mtime or 0) != path.stat().st_mtime
    assert indexer.stale_files(REPO_ID) == []  # faux positif corrigé
    # vérification en lecture seule : la base n'est pas rafraîchie par stale_files (c'est le rôle du run)
    after = db.get_file(REPO_ID, target)
    assert after is not None and after.mtime == stored.mtime and after.sha1 == stored.sha1
    # même taille mais contenu différent : détecté par le sha1
    data = path.read_bytes()
    altered = data[:-1] + (b"#" if data[-1:] != b"#" else b"!")
    assert len(altered) == len(data)
    path.write_bytes(altered)
    _bump_mtime(path, 240.0)
    assert indexer.stale_files(REPO_ID) == [target]
    path.write_bytes(data)
    _bump_mtime(path, 360.0)
    assert indexer.stale_files(REPO_ID) == []
    # fichier devenu illisible : impossible de confirmer, donc signalé (sauf en root, qui lit tout)
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        os.chmod(path, 0)
        try:
            _bump_mtime(path, 480.0)
            assert indexer.stale_files(REPO_ID) == [target]
        finally:
            os.chmod(path, 0o644)
    # le run suivant ne relit pas le contenu : simple rafraîchissement du mtime
    run = indexer.run()
    assert run.stats["files_indexed"] == 0
    refreshed = db.get_file(REPO_ID, target)
    assert refreshed is not None and abs(float(refreshed.mtime or 0) - path.stat().st_mtime) < 1e-6
    assert indexer.stale_files(REPO_ID) == []


# --------------------------------------------------------------------------- embeddings

def test_embeddings_with_fake_llm_then_model_change(db: Database, workspace: WorkspaceConfig):
    llm = FakeEmbedClient(dim=6)
    indexer = Indexer(db, workspace, llm=llm, embedding_model="fake-embed-v1")
    run = indexer.run()
    assert run.status == "ok"
    n_chunks = db.stats()["chunks"]
    assert run.stats["embeddings"] == n_chunks > 0 and run.stats["embeddings_pending"] == 0
    assert all(model == "fake-embed-v1" for model, _ in llm.calls)
    assert sum(n for _, n in llm.calls) == n_chunks
    assert db.get_meta(EMBEDDING_MODEL_META_KEY) == "fake-embed-v1"
    ids, matrix = db.load_embeddings("fake-embed-v1")
    assert len(ids) == n_chunks and matrix.shape == (n_chunks, 6) and matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    # second run : rien à recalculer
    run = indexer.run()
    assert run.stats["embeddings"] == 0 and len(llm.calls) == n_chunks // 32 + (1 if n_chunks % 32 else 0)

    # changement de modèle : tout est supprimé puis recalculé, jamais de mélange
    llm2 = FakeEmbedClient(dim=4)
    run = Indexer(db, workspace, llm=llm2, embedding_model="fake-embed-v2").run()
    assert run.status == "ok" and run.stats["embeddings"] == n_chunks
    assert any("fake-embed-v1" in w and "fake-embed-v2" in w for w in run.stats["warnings"])
    assert db.get_meta(EMBEDDING_MODEL_META_KEY) == "fake-embed-v2"
    assert db.embedding_models() == [{"model": "fake-embed-v2", "count": n_chunks, "dim": 4}]
    assert db.load_embeddings("fake-embed-v1")[0].size == 0


def test_embeddings_skipped_without_model_or_when_disabled(db: Database, workspace: WorkspaceConfig):
    llm = FakeEmbedClient()
    run = Indexer(db, workspace, llm=llm).run()  # pas de modèle configuré
    assert run.status == "ok" and run.stats["embeddings"] == 0 and llm.calls == []
    assert db.stats()["embeddings"] == 0
    run = Indexer(db, workspace, llm=llm, embedding_model="m").run(with_embeddings=False)
    assert run.stats["embeddings"] == 0 and llm.calls == []
    assert Indexer(db, workspace).embed_missing() == 0


def test_embeddings_engine_unavailable_is_not_a_failure(db: Database, workspace: WorkspaceConfig):
    llm = FakeEmbedClient(fail_with=LLMUnavailable("Ollama injoignable"))
    run = Indexer(db, workspace, llm=llm, embedding_model="m").run()
    assert run.status == "ok"
    assert run.stats["embeddings"] == 0 and run.stats["embeddings_pending"] == db.stats()["chunks"]
    assert any("injoignable" in w for w in run.stats["warnings"])
    # index intact : les fichiers et morceaux sont là, la recherche lexicale fonctionne
    assert db.stats()["chunks"] > 0 and db.fts_search("OrderController")


def test_embeddings_incremental_after_file_change(db: Database, workspace: WorkspaceConfig, project: Path):
    llm = FakeEmbedClient()
    indexer = Indexer(db, workspace, llm=llm, embedding_model="m")
    indexer.run()
    total = db.stats()["embeddings"]
    p = project / "tools" / "report.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n\ndef added():\n    return 42\n", encoding="utf-8")
    _bump_mtime(p)
    run = indexer.run()
    new_chunks = len(db.chunks_for("file", db.get_file(REPO_ID, "tools/report.py").id))  # type: ignore[union-attr]
    assert run.stats["embeddings"] == new_chunks
    assert db.stats()["embeddings"] == db.stats()["chunks"]
    assert db.stats()["embeddings"] >= total  # les vecteurs des autres fichiers sont conservés


def test_embedder_batches_limit_errors_and_cancel(db: Database, workspace: WorkspaceConfig):
    Indexer(db, workspace).run()
    n_chunks = db.stats()["chunks"]
    llm = FakeEmbedClient(dim=3)
    embedder = Embedder(db, llm, "m")
    progress: List[Tuple[int, int]] = []
    assert embedder.embed_missing(batch_size=4, limit=5, progress=lambda d, t: progress.append((d, t))) == 5
    assert [n for _, n in llm.calls] == [4, 1] and progress == [(4, 5), (5, 5)]
    assert db.stats()["embeddings"] == 5
    # annulation entre deux lots
    cancel = threading.Event()
    cancel.set()
    assert embedder.embed_missing(batch_size=4, cancel_event=cancel) == 0
    # erreur autre qu'« injoignable » : consignée, puis abandon après quelques lots
    bad = Embedder(db, FakeEmbedClient(fail_with=LLMError("réponse invalide")), "m")
    assert bad.embed_missing(batch_size=2) == 0
    assert bad.errors and not bad.unavailable and any("abandonnés" in w for w in bad.warnings)
    # moteur injoignable : arrêt propre
    down = Embedder(db, FakeEmbedClient(fail_with=LLMUnavailable("down")), "m")
    assert down.embed_missing() == 0 and down.unavailable and down.errors == []
    # reprise : seuls les manquants sont calculés
    assert embedder.embed_missing(batch_size=50) == n_chunks - 5
    assert db.chunk_ids_missing_embeddings("m") == []
    # sans moteur ou sans modèle : 0
    assert Embedder(db, None, "m").embed_missing() == 0
    assert Embedder(db, llm, "").embed_missing() == 0


def test_embedder_rejects_inconsistent_vectors(db: Database, workspace: WorkspaceConfig):
    Indexer(db, workspace).run()

    class Ragged(LLMClient):
        async def embed(self, texts, model):  # type: ignore[override]
            return [[1.0, 2.0]] * (len(texts) - 1)  # un vecteur de moins que de textes

    e = Embedder(db, Ragged(), "m")
    assert e.embed_missing(batch_size=8) == 0
    assert e.errors and "vecteur" in e.errors[0]["message"]


async def test_embed_texts_and_sync_bridge_inside_running_loop(db: Database, workspace: WorkspaceConfig):
    llm = FakeEmbedClient(dim=5)
    embedder = Embedder(db, llm, "m")
    matrix = await embedder.embed_texts(["commande confirmée", "client"])
    assert matrix.shape == (2, 5) and matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    assert (await embedder.embed_texts([])).shape == (0, 0)
    with pytest.raises(LLMUnavailable):
        await Embedder(db, None, "m").embed_texts(["x"])
    with pytest.raises(LLMUnavailable):
        await Embedder(db, llm, "").embed_texts(["x"])
    # l'indexation synchrone (thread de l'API) fonctionne même si un event loop tourne ici
    assert asyncio.get_running_loop() is not None
    run = Indexer(db, workspace, llm=llm, embedding_model="m").run()
    assert run.status == "ok" and run.stats["embeddings"] == db.stats()["chunks"] > 0

    async def coro() -> int:
        await asyncio.sleep(0)
        return 7

    assert run_coroutine_sync(coro()) == 7

    async def failing() -> None:
        raise ValueError("dans le thread")

    with pytest.raises(ValueError):
        run_coroutine_sync(failing())


def test_normalize_vectors():
    m = normalize_vectors([[3.0, 4.0], [0.0, 0.0], [float("nan"), 1.0]])
    assert m.dtype == np.float32 and m.shape == (3, 2)
    assert np.allclose(m[0], [0.6, 0.8]) and np.allclose(m[1], [0.0, 0.0]) and np.allclose(m[2], [0.0, 1.0])
    assert normalize_vectors([]).shape == (0, 0)
    with pytest.raises(ValueError):
        normalize_vectors([[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError):
        normalize_vectors([[]])


# --------------------------------------------------------------------------- divers

def test_index_progress_and_exceptions_api():
    p = IndexProgress(phase="scan", current=1, total=10)
    assert p.repo == "" and p.message == ""
    assert p.to_dict() == {"phase": "scan", "current": 1, "total": 10, "repo": "", "message": ""}
    assert issubclass(IndexCancelled, Exception)
    assert PHASES == ("scan", "analyze", "documents", "notes", "relations", "embeddings", "done")
    assert isinstance(EXTRACTOR_VERSION, str) and EXTRACTOR_VERSION


def test_run_never_raises_even_if_storage_breaks(db: Database, workspace: WorkspaceConfig, monkeypatch: pytest.MonkeyPatch):
    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("disque plein")

    monkeypatch.setattr(db, "file_signatures", explode)
    run = Indexer(db, workspace).run()
    assert run.status == "failed"
    assert run.error and "disque plein" in run.error
    assert any(e["stage"] == "run" for e in run.stats["errors"])
    assert db.last_index_run().status == "failed"  # type: ignore[union-attr]


def test_errors_are_capped(db: Database, workspace: WorkspaceConfig):
    indexer = Indexer(db, workspace)
    for i in range(250):
        indexer._error("f%d" % i, "test", "erreur %d" % i)
    assert len(indexer.errors) == 201 and indexer.errors_total == 250
    assert "tronquées" in indexer.errors[-1]["message"]


def test_scan_matches_indexed_paths(db: Database, workspace: WorkspaceConfig, project: Path):
    Indexer(db, workspace).run()
    scanned = {e.path for e in scan_repo(str(project), REPO_ID).files}
    assert {f.path for f in db.list_files(REPO_ID)} == scanned
    assert "frontend/node_modules/leftpad/index.js" not in scanned and "frontend/logo.png" not in scanned


def test_linker_dataclass_result_is_counted_like_a_dict(db: Database, workspace: WorkspaceConfig):
    """``link_all`` renvoie un ``LinkStats`` (dataclass à champ ``total``) : il est stocké en JSON dans
    ``stats['linker']`` et compté dans ``stats['relations']`` comme le serait un dict."""
    from dataclasses import dataclass, field

    @dataclass
    class Stats:
        total: int = 0
        errors: List[str] = field(default_factory=list)

    def linker(database: Database, ws: WorkspaceConfig) -> Stats:
        return Stats(total=7)

    run = Indexer(db, workspace, linker=linker).run()
    assert run.status == "ok"
    assert run.stats["linker"] == {"total": 7, "errors": []}
    assert run.stats["relations"] >= 7


# --------------------------------------------------------------------------- fichiers devenus inaccessibles

needs_posix_perms = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                       reason="droits POSIX requis")


class _Locked:
    """``chmod 000`` sur des chemins le temps d'un bloc ``with`` ; droits restaurés ensuite (nettoyage de tmp_path)."""

    def __init__(self, *paths: Path):
        self.paths = list(paths)
        self.modes: List[int] = []

    def __enter__(self) -> "_Locked":
        self.modes = [p.stat().st_mode & 0o7777 for p in self.paths]
        for p in self.paths:
            p.chmod(0)
        return self

    def __exit__(self, *exc: Any) -> None:
        for p, mode in zip(self.paths, self.modes):
            p.chmod(mode)


def _git_init_and_commit(project: Path) -> None:
    for args in (("init", "-q"), ("add", "-A"), ("commit", "-q", "-m", "init")):
        subprocess.run(["git", "-C", str(project), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
                       check=True, capture_output=True)


@needs_posix_perms
def test_indexed_file_becoming_unreadable_is_kept_with_error(db: Database, workspace: WorkspaceConfig, project: Path):
    """Un fichier déjà indexé devenu illisible (droits retirés) n'est pas supprimé de l'index : il est conservé
    tel quel (morceaux, symboles, relations), signalé par une erreur, et le run se termine en ``partial``."""
    indexer = Indexer(db, workspace)
    indexer.run()
    target = "backend/app/Models/OrderLine.php"
    f = db.get_file(REPO_ID, target)
    assert f is not None
    file_id = int(f.id)  # type: ignore[arg-type]
    chunk_ids = _file_chunk_ids(db, target)
    symbol_ids = [s.id for s in db.list_symbols(file_id=file_id)]
    n_relations = db.query("SELECT count(*) AS n FROM relations")[0]["n"]
    assert chunk_ids and symbol_ids

    with _Locked(project / target):
        run = indexer.run()
        stale = indexer.stale_files(REPO_ID)
    assert run.status == "partial"
    assert run.stats["files_deleted"] == 0 and run.stats["files_kept_unreadable"] == 1
    assert run.stats["repos"][REPO_ID]["files_kept_unreadable"] == 1
    assert run.stats["files_skipped"].get("unreadable") == 1
    assert len(run.stats["errors"]) == 1
    err = run.stats["errors"][0]
    assert err["path"] == target and err["stage"] == "read" and "conservé" in err["message"]
    assert db.last_index_run().status == "partial"  # type: ignore[union-attr]
    # tout est conservé : fichier (même id), morceaux, symboles, relations
    kept = db.get_file(REPO_ID, target)
    assert kept is not None and kept.id == file_id
    assert _file_chunk_ids(db, target) == chunk_ids
    assert [s.id for s in db.list_symbols(file_id=file_id)] == symbol_ids
    assert db.query("SELECT count(*) AS n FROM relations")[0]["n"] == n_relations
    assert stale == [target]  # signalé obsolète (analyse d'impact) tant qu'il est illisible

    # droits restaurés, contenu inchangé : simplement « inchangé », plus d'erreur
    run = indexer.run()
    assert run.status == "ok" and run.stats["errors"] == []
    assert run.stats["files_kept_unreadable"] == 0 and run.stats["files_deleted"] == 0
    assert run.stats["files_indexed"] == 0
    assert db.get_file(REPO_ID, target).id == file_id  # type: ignore[union-attr]
    assert indexer.stale_files(REPO_ID) == []


@needs_posix_perms
@pytest.mark.parametrize("with_git", [False, True])
def test_unreadable_directory_keeps_its_indexed_files(db: Database, workspace: WorkspaceConfig, project: Path,
                                                      with_git: bool):
    """Un dossier devenu impossible à lister ne fait pas disparaître son contenu de l'index : une erreur par
    fichier conservé (parcours fichier, et mode git où le scanner retombe sur le parcours fichier)."""
    folder = "backend/app/Models"
    if with_git:
        if not git_available():
            pytest.skip("git absent")
        _git_init_and_commit(project)
        (project / folder / "Draft.php").write_text("<?php\nclass Draft {}\n", encoding="utf-8")  # non suivi
    indexer = Indexer(db, workspace)
    first = indexer.run()
    assert first.status == "ok" and first.stats["repos"][REPO_ID]["used_git"] is with_git
    inside = sorted(f.path for f in db.list_files(REPO_ID) if f.path.startswith(folder + "/"))
    assert len(inside) >= 2 and (not with_git or folder + "/Draft.php" in inside)
    ids = {p: db.get_file(REPO_ID, p).id for p in inside}  # type: ignore[union-attr]
    n_files = len(db.list_files(REPO_ID))

    with _Locked(project / folder):
        run = indexer.run()
    assert run.status == "partial"
    assert run.stats["files_deleted"] == 0 and run.stats["files_kept_unreadable"] == len(inside)
    assert sorted(e["path"] for e in run.stats["errors"]) == inside
    assert all(e["stage"] == "read" and "conservé" in e["message"] for e in run.stats["errors"])
    assert any("illisible" in w and folder in w for w in run.stats["warnings"])
    assert len(db.list_files(REPO_ID)) == n_files
    for p in inside:
        f = db.get_file(REPO_ID, p)
        assert f is not None and f.id == ids[p] and db.chunks_for("file", f.id)  # type: ignore[arg-type]

    run = indexer.run()
    assert run.status == "ok" and run.stats["files_deleted"] == 0 and run.stats["files_indexed"] == 0
    assert len(db.list_files(REPO_ID)) == n_files


@needs_posix_perms
def test_unreadable_repo_root_is_an_error_and_keeps_index(db: Database, workspace: WorkspaceConfig, project: Path):
    """La racine d'un dépôt impossible à lister est traitée comme un dossier introuvable : erreur de parcours,
    dépôt en ``error``, rien n'est supprimé."""
    indexer = Indexer(db, workspace)
    first = indexer.run()
    n_files = len(db.list_files(REPO_ID))
    assert first.status == "ok" and n_files > 20

    with _Locked(project):
        run = indexer.run()
    assert run.status == "partial"
    rstats = run.stats["repos"][REPO_ID]
    assert rstats["status"] == "error" and "inaccessible" in rstats["error"]
    assert any(e["stage"] == "scan" and e["path"] == str(project) for e in run.stats["errors"])
    assert run.stats["files_deleted"] == 0 and run.stats["files_scanned"] == 0
    assert len(db.list_files(REPO_ID)) == n_files

    run = indexer.run()
    assert run.status == "ok" and run.stats["files_deleted"] == 0 and run.stats["files_unchanged"] == n_files


@needs_posix_perms
def test_never_indexed_unreadable_file_is_a_warning(db: Database, workspace: WorkspaceConfig, project: Path):
    """Un fichier illisible qui n'a jamais été indexé n'est pas une perte : avertissement nommé, run ``ok``."""
    locked = project / "backend" / "app" / "Secret.php"
    locked.write_text("<?php\nclass Secret {}\n", encoding="utf-8")
    with _Locked(locked):
        run = Indexer(db, workspace).run()
    assert run.status == "ok" and run.stats["errors"] == []
    assert run.stats["files_skipped"]["unreadable"] == 1 and run.stats["files_kept_unreadable"] == 0
    assert any("illisible" in w and "backend/app/Secret.php" in w for w in run.stats["warnings"])
    assert db.get_file(REPO_ID, "backend/app/Secret.php") is None


@needs_posix_perms
def test_unreadable_known_paths_falls_back_to_lstat(tmp_path: Path):
    """Troisième indice : le scanner n'a rien signalé (mode git) mais ``lstat`` échoue par manque de droits.
    Un fichier présent sur disque mais plus indexable (exclu, binaire…) et un fichier disparu ne sont pas conservés."""
    from develop.project_scanner import ScanResult

    repo_dir = tmp_path / "repo"
    (repo_dir / "closed").mkdir(parents=True)
    (repo_dir / "closed" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (repo_dir / "excluded.py").write_text("y = 1\n", encoding="utf-8")
    repo = RepoConfig(id="r", name="r", path=str(repo_dir))
    signatures = {"closed/a.py": ("sha", 1.0, 6), "gone.py": ("sha", 1.0, 6), "excluded.py": ("sha", 1.0, 6)}
    scan = ScanResult(root=str(repo_dir))  # aucun fichier, rien de signalé
    with _Locked(repo_dir / "closed"):
        kept = Indexer._unreadable_known_paths(repo, scan, signatures, set())
    assert set(kept) == {"closed/a.py"} and "inaccessible" in kept["closed/a.py"]
    # signalé par le scanner : causes dédiées, sans lstat
    scan.unreadable = ["gone.py"]
    scan.unreadable_dirs = ["closed"]
    kept = Indexer._unreadable_known_paths(repo, scan, signatures, {"excluded.py"})
    assert set(kept) == {"closed/a.py", "gone.py"}
    assert "illisible" in kept["gone.py"] and "Dossier parent" in kept["closed/a.py"]
    # la racine fermée ("") n'est pas un préfixe : elle est traitée à part par ``_index_repo``
    assert Indexer._unreadable_known_paths(repo, ScanResult(unreadable_dirs=[""]), signatures, set()) == {}
