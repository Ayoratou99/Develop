"""Tests de la mémoire Develop (develop.storage)."""
from __future__ import annotations

import sqlite3
import threading
import time

import numpy as np
import pytest

from develop.storage import (
    SCHEMA_VERSION,
    ChunkRow,
    Database,
    DocumentRow,
    EndpointRow,
    FileRow,
    FtsHit,
    ModelRow,
    NoteRow,
    RelationRow,
    RepoRow,
    StorageError,
    SymbolRow,
    build_fts_query,
    fts_terms,
    node_key,
)


@pytest.fixture
def db():
    with Database(":memory:") as d:
        yield d


def _repo(db: Database, repo_id: str = "r1") -> RepoRow:
    return db.upsert_repo(RepoRow(id=repo_id, name="backend", path="/tmp/backend", kind="backend"))


def _file(db: Database, path: str = "app/Models/Order.php", repo_id: str = "r1", sha1: str = "abc",
          language: str = "php") -> FileRow:
    f = FileRow(repo_id=repo_id, path=path, language=language, kind="code", size=120, mtime=1.0, sha1=sha1)
    db.upsert_file(f)
    return f


# --------------------------------------------------------------------------- cycle de vie / schéma

def test_migrate_is_idempotent_and_sets_schema_version(db):
    assert db.schema_version() == SCHEMA_VERSION
    assert db.get_meta("schema_version") == str(SCHEMA_VERSION)
    assert db.migrate() == SCHEMA_VERSION  # rejouable sans erreur
    assert db.migrate() == SCHEMA_VERSION
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for t in ("meta", "repos", "files", "symbols", "endpoints", "data_models", "relations", "documents", "notes",
              "chunks", "chunks_fts", "embeddings", "conversations", "messages", "index_runs"):
        assert t in tables, t


def test_on_disk_database_uses_wal_and_persists(tmp_path):
    path = tmp_path / "sub" / "memory.sqlite3"
    with Database(path) as d:
        assert d.query("PRAGMA journal_mode")[0][0] == "wal"
        assert d.query("PRAGMA foreign_keys")[0][0] == 1
        d.set_meta("hello", "monde")
        _repo(d)
    assert path.exists()
    with Database(path) as d2:  # réouverture : migrate() idempotent, données conservées
        assert d2.get_meta("hello") == "monde"
        assert d2.get_repo("r1") is not None
        assert d2.schema_version() == SCHEMA_VERSION


def test_close_is_safe_twice_and_query_is_read_only(db):
    with pytest.raises(StorageError):
        db.query("DELETE FROM meta")
    db.close()
    db.close()
    assert db.closed


def test_transaction_rollback_and_nested_savepoint(db):
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.set_meta("k", "1")
            raise RuntimeError("boum")
    assert db.get_meta("k") is None
    with db.transaction():
        db.set_meta("outer", "1")
        try:
            with db.transaction():  # SAVEPOINT imbriqué
                db.set_meta("inner", "1")
                raise ValueError("annule seulement l'intérieur")
        except ValueError:
            pass
        db.set_meta("after", "1")
    assert db.get_meta("outer") == "1" and db.get_meta("after") == "1" and db.get_meta("inner") is None


def test_meta_get_set_delete(db):
    assert db.get_meta("absent") is None
    assert db.get_meta("absent", "dflt") == "dflt"
    db.set_meta("a", "1")
    db.set_meta("a", "2")
    assert db.get_meta("a") == "2"
    db.set_meta("a", None)
    assert db.get_meta("a") is None


# --------------------------------------------------------------------------- repos / fichiers

def test_repos_crud(db):
    _repo(db, "r1")
    db.upsert_repo(RepoRow(id="r2", name="alpha", path="/tmp/a", kind="frontend", head_sha="deadbeef"))
    assert [r.id for r in db.list_repos()] == ["r2", "r1"]  # trié par nom
    db.upsert_repo(RepoRow(id="r1", name="backend2", path="/tmp/b", kind="backend", last_indexed_at=5.0))
    r = db.get_repo("r1")
    assert r is not None and r.name == "backend2" and r.last_indexed_at == 5.0
    assert r.to_dict()["kind"] == "backend"
    assert db.delete_repo("r1") is True
    assert db.delete_repo("r1") is False
    assert db.get_repo("r1") is None


def test_upsert_file_keeps_id_and_signatures_for_incremental_indexing(db):
    _repo(db)
    f = _file(db, sha1="aaa")
    first_id = f.id
    assert first_id is not None and f.indexed_at is not None
    f2 = FileRow(repo_id="r1", path="app/Models/Order.php", language="php", kind="code", size=130, mtime=2.0, sha1="bbb")
    assert db.upsert_file(f2) == first_id  # même (repo_id, path) => même id
    got = db.get_file("r1", "app/Models/Order.php")
    assert got is not None and got.id == first_id and got.sha1 == "bbb" and got.size == 130
    assert db.get_file_by_id(first_id).path == "app/Models/Order.php"
    assert db.get_file("r1", "nope") is None and db.get_file_by_id(999) is None
    _file(db, "src/App.tsx", sha1="ccc", language="typescript")
    db.upsert_file(FileRow(repo_id="r1", path="README.md", language="markdown", kind="doc"))
    assert db.file_signatures("r1") == {
        "app/Models/Order.php": ("bbb", 2.0, 130),
        "src/App.tsx": ("ccc", 1.0, 120),
        "README.md": (None, None, None),
    }
    assert len(db.list_files()) == 3
    assert [x.path for x in db.list_files(repo_id="r1", kind="doc")] == ["README.md"]
    assert [x.path for x in db.list_files(language="php")] == ["app/Models/Order.php"]
    assert db.list_files(repo_id="autre") == []


def test_delete_file_and_delete_files_not_in(db):
    _repo(db)
    a = _file(db, "a.php")
    _file(db, "b.php")
    _file(db, "c.php")
    db.replace_file_artifacts(a.id, symbols=[SymbolRow(name="A", kind="class")], chunks=[ChunkRow(text="class A")])
    assert db.delete_file(a.id) is True
    assert db.delete_file(a.id) is False
    assert db.list_symbols(file_id=a.id) == [] and db.chunks_for("file", a.id) == []
    assert db.delete_files_not_in("r1", ["b.php"]) == 1
    assert [f.path for f in db.list_files("r1")] == ["b.php"]
    assert db.delete_files_not_in("r1", ["b.php", "zzz"]) == 0


# --------------------------------------------------------------------------- artefacts

def test_replace_file_artifacts_is_transactional_and_cleans_relations(db):
    _repo(db)
    f = _file(db)
    ids = db.replace_file_artifacts(
        f.id,
        symbols=[SymbolRow(name="Order", kind="class", line_start=1, line_end=40, meta={"abstract": False}),
                 SymbolRow(name="total", kind="method", parent="Order", signature="total(): float")],
        endpoints=[EndpointRow(side="server", method="get", path="/orders/{id}", normalized_path="/orders/:id",
                               handler="OrderController@show", framework="laravel", line=12)],
        models=[ModelRow(name="Order", table_name="orders", kind="eloquent", fields=[{"name": "id"}],
                         relations=[{"kind": "belongsTo", "model": "User"}])],
        chunks=[ChunkRow(text="class Order extends Model", line_start=1, line_end=10)],
    )
    assert len(ids["symbols"]) == 2 and len(ids["endpoints"]) == 1 and len(ids["models"]) == 1 and len(ids["chunks"]) == 1
    sym_key = node_key("symbol", ids["symbols"][0])
    db.add_relations([RelationRow(src=node_key("file", f.id), dst=sym_key, kind="defines"),
                      RelationRow(src=sym_key, dst=node_key("model", ids["models"][0]), kind="uses_model"),
                      RelationRow(src=node_key("file", f.id), dst="table:orders", kind="mentions")])
    assert len(db.relations_from(node_key("file", f.id))) == 2
    # réindexation : anciens artefacts et relations vers leurs ids disparaissent
    ids2 = db.replace_file_artifacts(f.id, symbols=[SymbolRow(name="Order", kind="class")], chunks=[])
    assert ids2["symbols"] != ids["symbols"]
    assert db.get_symbol(ids["symbols"][0]) is None and db.get_symbol(ids2["symbols"][0]).file_id == f.id
    assert db.list_endpoints() == [] and db.list_models() == [] and db.chunks_for("file", f.id) == []
    assert db.relations_from(sym_key) == [] and db.relations_to(sym_key) == []
    assert [r.dst for r in db.relations_from(node_key("file", f.id))] == ["table:orders"]  # relation de fichier conservée
    with pytest.raises(StorageError):
        db.replace_file_artifacts(9999, symbols=[SymbolRow(name="x")])
    # une erreur au milieu annule tout (transaction) : l'ancien état reste
    with pytest.raises(sqlite3.IntegrityError):
        db.replace_file_artifacts(f.id, symbols=[SymbolRow(name="Nouveau")],
                                  endpoints=[EndpointRow(side="invalide", method="GET", path="/x")])
    assert [s.name for s in db.list_symbols(file_id=f.id)] == ["Order"]


def test_symbols_list_search_get(db):
    _repo(db)
    f = _file(db)
    db.replace_file_artifacts(f.id, symbols=[
        SymbolRow(name="OrderService", kind="class", line_start=1),
        SymbolRow(name="order", kind="function", line_start=5),
        SymbolRow(name="getOrderTotal", kind="function", line_start=9),
        SymbolRow(name="user_id", kind="property", line_start=20),
        SymbolRow(name="Unrelated", kind="class", line_start=30),
    ])
    assert [s.name for s in db.list_symbols(file_id=f.id)] == ["OrderService", "order", "getOrderTotal", "user_id", "Unrelated"]
    assert [s.name for s in db.list_symbols(kind="class")] == ["OrderService", "Unrelated"]
    assert len(db.list_symbols(limit=2)) == 2
    names = [s.name for s in db.search_symbols("ORDER")]
    assert names[0] == "order"  # exact (insensible à la casse) d'abord
    assert names[1] == "OrderService"  # puis préfixe
    assert "getOrderTotal" in names and "Unrelated" not in names
    assert [s.name for s in db.search_symbols("user_id")] == ["user_id"]  # "_" n'est pas un joker
    assert db.search_symbols("%") == [] and db.search_symbols("   ") == []
    s = db.get_symbol(db.search_symbols("user_id")[0].id)
    assert s is not None and s.kind == "property" and s.to_dict()["meta"] == {}
    assert db.get_symbol(12345) is None


def test_endpoints_list_find_get(db):
    _repo(db)
    f = _file(db, "routes/api.php")
    g = _file(db, "src/api/client.ts")
    db.replace_file_artifacts(f.id, endpoints=[
        EndpointRow(side="server", method="GET", path="/api/orders/{id}", normalized_path="/api/orders/:id", framework="laravel"),
        EndpointRow(side="server", method="POST", path="/api/orders", normalized_path="/api/orders", framework="laravel"),
    ])
    db.replace_file_artifacts(g.id, endpoints=[
        EndpointRow(side="client", method="get", path="/api/orders/${id}", normalized_path="/api/orders/:id",
                    framework="axios", meta={"call": "api.get"}),
    ])
    assert len(db.list_endpoints()) == 3
    assert [e.side for e in db.list_endpoints(side="client")] == ["client"]
    assert len(db.list_endpoints(framework="laravel")) == 2
    assert len(db.list_endpoints(limit=1)) == 1
    found = db.find_endpoints(normalized_path="/api/orders/:id", method="get")
    assert {e.side for e in found} == {"client", "server"}
    assert [e.method for e in db.find_endpoints(method="post")] == ["POST"]
    assert db.find_endpoints(normalized_path="/nope") == []
    e = db.get_endpoint(found[0].id)
    assert e is not None and e.normalized_path == "/api/orders/:id"
    client = [x for x in found if x.side == "client"][0]
    assert client.meta == {"call": "api.get"}
    assert db.get_endpoint(999) is None


def test_models_list_find_get(db):
    _repo(db)
    f = _file(db)
    db.replace_file_artifacts(f.id, models=[
        ModelRow(name="Order", table_name="orders", kind="eloquent", fields=[{"name": "id", "type": "int"}]),
        ModelRow(name="User", table_name="users", kind="eloquent"),
        ModelRow(name="orders", table_name="orders", kind="sql_table", fields=[{"name": "id"}, {"name": "user_id"}]),
    ])
    assert [m.name for m in db.list_models()] == ["Order", "User", "orders"]
    assert len(db.list_models(limit=1)) == 1
    assert {m.kind for m in db.find_models(table_name="ORDERS")} == {"eloquent", "sql_table"}
    assert [m.name for m in db.find_models(name="user")] == ["User"]
    assert db.find_models(name="Order", table_name="users") == []
    m = db.get_model(db.find_models(name="Order")[0].id)
    assert m is not None and m.fields == [{"name": "id", "type": "int"}] and m.relations == []
    assert db.get_model(999) is None


# --------------------------------------------------------------------------- relations

def test_relations_add_query_delete_and_neighbors_depth2(db):
    n = db.add_relations([
        RelationRow(src="file:1", dst="file:2", kind="imports", confidence=0.9, meta={"line": 3}),
        RelationRow(src="file:2", dst="symbol:7", kind="defines"),
        RelationRow(src="symbol:7", dst="model:3", kind="uses_model"),
        RelationRow(src="model:3", dst="table:orders", kind="maps_to"),
        RelationRow(src="note:1", dst="file:1", kind="mentions"),
        RelationRow(src="", dst="x", kind="ignored"),  # invalide : ignorée
    ])
    assert n == 5
    assert db.add_relations([RelationRow(src="file:1", dst="file:2", kind="imports", confidence=0.5)]) == 1
    rels = db.relations_from("file:1")
    assert len(rels) == 1 and rels[0].confidence == 0.5  # INSERT OR REPLACE
    assert [r.src for r in db.relations_to("file:1")] == ["note:1"]
    assert db.relations_from("file:2", kind="defines")[0].dst == "symbol:7"
    assert db.relations_from("file:2", kind="nope") == [] and db.relations_to("file:1", kind="nope") == []
    assert db.neighbors("file:1") == {"file:2", "note:1"}
    assert db.neighbors("file:1", depth=2) == {"file:2", "note:1", "symbol:7"}
    assert db.neighbors("file:1", depth=10) == {"file:2", "note:1", "symbol:7", "model:3", "table:orders"}
    assert db.neighbors("file:1", depth=0) == set() and db.neighbors("inconnu:9") == set()
    assert db.delete_relations_from("file:2") == 1
    assert db.delete_relations_touching(["model:3", "absent:1"]) == 2
    assert db.delete_relations_touching([]) == 0
    assert db.stats()["relations"] == 2


def test_delete_relations_touching_handles_large_batches(db):
    db.add_relations([RelationRow(src="file:%d" % i, dst="symbol:%d" % i, kind="defines") for i in range(1000)])
    keys = ["symbol:%d" % i for i in range(0, 1000, 2)] + ["file:%d" % i for i in range(1, 1000, 2)]
    assert db.delete_relations_touching(keys) == 1000
    assert db.stats()["relations"] == 0


# --------------------------------------------------------------------------- cascades

def test_delete_repo_cascades_everything(db):
    _repo(db)
    f = _file(db)
    ids = db.replace_file_artifacts(
        f.id, symbols=[SymbolRow(name="A")], endpoints=[EndpointRow(side="server", method="GET", path="/a")],
        models=[ModelRow(name="A")], chunks=[ChunkRow(text="du texte indexé")],
    )
    db.upsert_embeddings([(ids["chunks"][0], "m", np.ones(4, dtype=np.float32))])
    db.add_relations([RelationRow(src=node_key("symbol", ids["symbols"][0]), dst="table:a", kind="maps_to"),
                      RelationRow(src="note:1", dst=node_key("file", f.id), kind="mentions"),
                      RelationRow(src="note:1", dst="note:2", kind="links")])
    assert db.fts_search("texte")
    assert db.delete_repo("r1") is True
    s = db.stats()
    assert s["files"] == 0 and s["symbols"] == 0 and s["endpoints"] == 0 and s["models"] == 0
    assert s["chunks"] == 0 and s["embeddings"] == 0 and s["relations"] == 1
    assert db.fts_search("texte") == []  # l'index FTS est synchronisé par les triggers


def test_chunks_follow_their_source_even_on_raw_delete(db):
    """Filet de sécurité : les triggers suppriment les chunks d'une source supprimée hors API."""
    _repo(db)
    f = _file(db)
    db.replace_chunks("file", f.id, [ChunkRow(text="orphelin potentiel")])
    doc_id = db.add_document(DocumentRow(title="d"))
    db.replace_chunks("document", doc_id, [ChunkRow(text="document chunk")])
    note_id = db.add_note(NoteRow(title="n"))
    db.replace_chunks("note", note_id, [ChunkRow(text="note chunk")])
    assert db.count_chunks() == 3
    db._conn.execute("DELETE FROM files WHERE id = ?", (f.id,))
    db._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    db._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    assert db.count_chunks() == 0 and db.fts_search("chunk") == []


# --------------------------------------------------------------------------- documents / notes

def test_documents_crud_and_sha_lookup(db):
    d = DocumentRow(title="Cahier des charges", source_path="/home/me/cdc.pdf", stored_path="documents/cdc.pdf",
                    kind="upload", format="pdf", size=1234, sha1="s1", meta={"pages": 3})
    doc_id = db.add_document(d)
    assert d.id == doc_id and d.added_at is not None
    assert db.get_document(doc_id).meta == {"pages": 3}
    assert db.find_document_by_sha("s1").id == doc_id
    assert db.find_document_by_sha("zzz") is None and db.find_document_by_sha("") is None
    db.add_document(DocumentRow(title="README", kind="repo_file", format="md", sha1="s2"))
    assert [x.title for x in db.list_documents(kind="repo_file")] == ["README"]
    assert len(db.list_documents()) == 2 and len(db.list_documents(limit=1)) == 1
    d.title = "CDC v2"
    d.meta = {"pages": 4}
    before = d.updated_at
    time.sleep(0.01)
    assert db.update_document(d) is True
    g = db.get_document(doc_id)
    assert g.title == "CDC v2" and g.meta == {"pages": 4} and g.updated_at > before
    assert db.update_document(DocumentRow(title="sans id")) is False
    assert db.update_document(DocumentRow(id=999, title="inconnu")) is False
    chunk_ids = db.replace_chunks("document", doc_id, [ChunkRow(text="contenu du cahier des charges")])
    db.upsert_embeddings([(chunk_ids[0], "m", np.zeros(3, dtype=np.float32))])
    db.add_relations([RelationRow(src=node_key("document", doc_id), dst="file:1", kind="mentions")])
    assert db.delete_document(doc_id) is True
    assert db.delete_document(doc_id) is False
    assert db.get_document(doc_id) is None and db.chunks_for("document", doc_id) == []
    assert db.stats()["embeddings"] == 0 and db.relations_from(node_key("document", doc_id)) == []


def test_notes_crud_and_filters(db):
    n1 = db.add_note(NoteRow(title="Réunion", body="Décisions", tags=["réunion", "API"], note_date="2026-10-01"))
    n2 = db.add_note(NoteRow(title="Idée", body="…", tags=["api"], note_date="2026-10-10"))
    n3 = db.add_note(NoteRow(title="Sans date", body=""))
    note = db.get_note(n3)
    assert note is not None and note.note_date == time.strftime("%Y-%m-%d") and note.tags == []
    assert [n.id for n in db.list_notes()] == [n3, n2, n1]  # plus récentes d'abord
    assert [n.id for n in db.list_notes(date_from="2026-10-02", date_to="2026-10-12")] == [n2]
    assert [n.id for n in db.list_notes(date_to="2026-10-05")] == [n1]
    assert [n.id for n in db.list_notes(tag="API")] == [n2, n1]  # insensible à la casse
    assert [n.id for n in db.list_notes(tag="api", limit=1)] == [n2]
    assert db.list_notes(tag="inconnu") == []
    assert len(db.list_notes(limit=2)) == 2
    first = db.get_note(n1)
    first.body = "Décisions prises"
    first.tags = ["réunion"]
    assert db.update_note(first) is True
    assert db.get_note(n1).body == "Décisions prises" and db.get_note(n1).tags == ["réunion"]
    assert db.update_note(NoteRow(title="x")) is False
    db.replace_chunks("note", n1, [ChunkRow(text="Décisions prises en réunion")])
    assert db.fts_search("decisions")[0].source_id == n1
    assert db.delete_note(n1) is True and db.delete_note(n1) is False
    assert db.get_note(n1) is None and db.chunks_for("note", n1) == [] and db.fts_search("decisions") == []


# --------------------------------------------------------------------------- chunks

def test_replace_chunks_get_and_count(db):
    _repo(db)
    f = _file(db)
    c1, c2 = ChunkRow(text="premier", line_start=1, line_end=5, meta={"k": 1}), ChunkRow(text="second", line_start=6)
    ids = db.replace_chunks("file", f.id, [c1, c2])
    assert ids == [c1.id, c2.id] and (c1.ordinal, c2.ordinal) == (0, 1)
    assert c1.source_type == "file" and c1.source_id == f.id and len(c1.sha1) == 40
    got = db.get_chunk(ids[1])
    assert got is not None and got.text == "second" and got.ordinal == 1 and got.meta == {}
    assert [c.text for c in db.get_chunks([ids[1], ids[0], ids[1], 999])] == ["second", "premier"]
    assert db.get_chunks([]) == []
    assert [c.text for c in db.chunks_for("file", f.id)] == ["premier", "second"]
    assert db.count_chunks() == 2 and db.count_chunks("file") == 2 and db.count_chunks("note") == 0
    db.upsert_embeddings([(ids[0], "m", np.ones(2, dtype=np.float32))])
    new_ids = db.replace_chunks("file", f.id, [ChunkRow(text="remplacé")])
    assert new_ids != ids and db.get_chunk(ids[0]) is None and db.stats()["embeddings"] == 0
    assert db.replace_chunks("file", f.id, []) == [] and db.count_chunks() == 0
    with pytest.raises(StorageError):
        db.replace_chunks("inconnu", 1, [])


# --------------------------------------------------------------------------- recherche lexicale

def test_build_fts_query_is_safe():
    assert fts_terms("L'utilisateur a Commandé 2 fois, OR not!") == ["utilisateur", "commandé", "fois", "or", "not"]
    assert build_fts_query("commande client") == '"commande" "client"*'
    assert build_fts_query("commande client", "OR") == '"commande" OR "client"*'
    assert build_fts_query("commande", prefix_last=False) == '"commande"'
    assert build_fts_query('"); DROP TABLE chunks; --') == '"drop" "table" "chunks"*'
    assert build_fts_query("status OR") == '"status" "or"*'
    assert build_fts_query("a") == "" and build_fts_query("") == "" and build_fts_query("  ...  ") == ""
    assert build_fts_query("col:value NEAR(x y)") == '"col" "value" "near"*'
    assert build_fts_query("user_id") == '"user" "id"*'  # unicode61 coupe aussi sur "_"
    assert fts_terms(None) == []  # type: ignore[arg-type]


def test_fts_search_matches_accents_and_ranks(db):
    doc = db.add_document(DocumentRow(title="d"))
    db.replace_chunks("document", doc, [
        ChunkRow(text="La commande a été Commandée par le client hier."),
        ChunkRow(text="Rien à voir : facture et paiement."),
        ChunkRow(text="commande commande commande : page dédiée aux commandes."),
    ])
    hits = db.fts_search("commande")
    assert [h.chunk_id for h in hits] == [3, 1]  # la page dédiée est mieux classée
    assert isinstance(hits[0], FtsHit) and hits[0].score >= hits[1].score
    assert "[commande]" in hits[0].snippet.lower() and hits[0].source_type == "document" and hits[0].source_id == doc
    assert hits[0].to_dict()["chunk_id"] == 3
    assert [h.chunk_id for h in db.fts_search("Commandé")] == [3, 1]  # accents ignorés des deux côtés
    assert [h.chunk_id for h in db.fts_search("CLIENT commande")] == [1]  # ET implicite
    assert [h.chunk_id for h in db.fts_search("comm")] == [3, 1]  # préfixe sur le dernier terme
    assert len(db.fts_search("commande", limit=1)) == 1


def test_fts_search_dirty_queries_never_raise(db):
    db.replace_chunks("note", 1, [ChunkRow(text="status: active — la commande est en cours")])
    for q in ['"); DROP TABLE chunks; --', "status OR", "a", "", "   ", "OR", "AND NOT", "(((", "NEAR(x y)",
              "text:status", '"', "*", "status*", "ééé", "🙂", "x" * 5000]:
        hits = db.fts_search(q)
        assert isinstance(hits, list), q
    assert db.fts_search("a") == [] and db.fts_search("OR") == [] and db.fts_search("") == []
    assert db.fts_search("status OR")[0].chunk_id == 1  # repli en OU : "or" absent mais "status" présent
    assert db.fts_search("text:status")[0].chunk_id == 1
    assert db.count_chunks() == 1  # la table est toujours là


def test_fts_search_or_fallback_and_source_filter(db):
    db.replace_chunks("note", 1, [ChunkRow(text="livraison express")])
    db.replace_chunks("file", 1, [ChunkRow(text="facture pdf")])
    assert db.fts_search("livraison facture") and {h.chunk_id for h in db.fts_search("livraison facture")} == {1, 2}
    assert [h.source_type for h in db.fts_search("livraison facture", source_types=["note"])] == ["note"]
    assert db.fts_search("livraison facture", source_types=("file",))[0].chunk_id == 2
    assert db.fts_search("livraison", source_types=["inconnu"]) == []
    assert db.fts_search("livraison", limit=0) == []
    assert db.fts_search("inexistant introuvable") == []


# --------------------------------------------------------------------------- embeddings

def test_embeddings_numpy_roundtrip(db):
    db.replace_chunks("note", 1, [ChunkRow(text="un"), ChunkRow(text="deux"), ChunkRow(text="trois")])
    ids, matrix = db.load_embeddings("m")
    assert ids.shape == (0,) and matrix.shape == (0, 0) and matrix.dtype == np.float32
    assert db.chunk_ids_missing_embeddings("m") == [1, 2, 3]
    assert db.chunk_ids_missing_embeddings("m", limit=2) == [1, 2]
    v1 = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    v2 = np.array([1.5, 2.5, 3.5], dtype=np.float64)  # converti en float32
    v3 = np.array([7, 8, 9], dtype=">f4")  # big-endian : converti en little-endian
    assert db.upsert_embeddings([(1, "m", v1), (2, "m", v2), (3, "m", v3), (999, "m", v1), (1, "m", [])]) == 3
    ids, matrix = db.load_embeddings("m")
    assert ids.dtype == np.int64 and ids.tolist() == [1, 2, 3]
    assert matrix.dtype == np.float32 and matrix.shape == (3, 3) and matrix.flags.writeable
    np.testing.assert_allclose(matrix[0], v1)
    np.testing.assert_allclose(matrix[1], v2.astype(np.float32))
    np.testing.assert_allclose(matrix[2], [7, 8, 9])
    raw = db.query("SELECT vector FROM embeddings WHERE chunk_id = 3")[0][0]
    assert bytes(raw) == np.array([7, 8, 9], dtype="<f4").tobytes()
    assert db.chunk_ids_missing_embeddings("m") == [] and db.chunk_ids_missing_embeddings("autre") == [1, 2, 3]
    assert db.embedding_models() == [{"model": "m", "count": 3, "dim": 3}]
    # changement de modèle : remplacement (une seule ligne par chunk)
    assert db.upsert_embeddings([(1, "n", np.ones(5, dtype=np.float32))]) == 1
    assert db.chunk_ids_missing_embeddings("m") == [1]
    assert db.load_embeddings("n")[1].shape == (1, 5)
    assert db.delete_embeddings("n") == 1 and db.embedding_models() == [{"model": "m", "count": 2, "dim": 3}]
    assert db.delete_embeddings() == 2 and db.load_embeddings("m")[0].size == 0


def test_load_embeddings_skips_inconsistent_dims(db):
    db.replace_chunks("note", 1, [ChunkRow(text="un"), ChunkRow(text="deux")])
    db.upsert_embeddings([(1, "m", np.ones(4, dtype=np.float32)), (2, "m", np.ones(8, dtype=np.float32))])
    ids, matrix = db.load_embeddings("m")
    assert ids.tolist() == [1] and matrix.shape == (1, 4)


# --------------------------------------------------------------------------- conversations

def test_conversations_and_messages(db):
    cid = db.create_conversation("Première question")
    other = db.create_conversation("")
    assert db.list_conversations()[0].id == other  # la plus récemment mise à jour d'abord
    m1 = db.add_message(cid, "user", "Où est la commande ?")
    time.sleep(0.01)
    m2 = db.add_message(cid, "assistant", "Dans Order.php", sources=[{"chunk_id": 1, "path": "app/Models/Order.php"}])
    convs = db.list_conversations()
    assert [c.id for c in convs] == [cid, other] and convs[0].message_count == 2 and convs[1].message_count == 0
    assert len(db.list_conversations(limit=1)) == 1
    conv = db.get_conversation(cid)
    assert conv is not None and conv["title"] == "Première question" and conv["message_count"] == 2
    assert [m["id"] for m in conv["messages"]] == [m1, m2]
    assert conv["messages"][1]["sources"] == [{"chunk_id": 1, "path": "app/Models/Order.php"}]
    assert conv["messages"][0]["sources"] == [] and conv["updated_at"] >= conv["created_at"]
    assert db.rename_conversation(cid, "Renommée") is True and db.get_conversation(cid)["title"] == "Renommée"
    assert db.rename_conversation(999, "x") is False
    with pytest.raises(StorageError):
        db.add_message(999, "user", "perdu")
    assert db.delete_conversation(cid) is True and db.delete_conversation(cid) is False
    assert db.get_conversation(cid) is None and db.stats()["messages"] == 0


# --------------------------------------------------------------------------- index runs / stats

def test_index_runs(db):
    assert db.last_index_run() is None
    run = db.start_index_run()
    r = db.last_index_run()
    assert r is not None and r.id == run and r.status == "running" and r.finished_at is None and r.stats == {}
    assert db.finish_index_run(run, "ok", stats={"files": 12}) is True
    r = db.last_index_run()
    assert r.status == "ok" and r.stats == {"files": 12} and r.finished_at >= r.started_at and r.error is None
    run2 = db.start_index_run()
    assert db.finish_index_run(run2, "error", error="disque plein") is True
    assert db.last_index_run().id == run2 and db.last_index_run().error == "disque plein"
    assert [x.id for x in db.list_index_runs()] == [run2, run]
    assert db.finish_index_run(999, "ok") is False


def test_stats_counts_everything(db):
    _repo(db)
    f = _file(db)
    ids = db.replace_file_artifacts(f.id, symbols=[SymbolRow(name="A")], endpoints=[EndpointRow(side="server")],
                                    models=[ModelRow(name="A")], chunks=[ChunkRow(text="x")])
    db.add_relations([RelationRow(src="a", dst="b", kind="k")])
    db.add_document(DocumentRow(title="d"))
    db.add_note(NoteRow(title="n"))
    db.upsert_embeddings([(ids["chunks"][0], "m", np.ones(2, dtype=np.float32))])
    cid = db.create_conversation("c")
    db.add_message(cid, "user", "hello")
    assert db.stats() == {"repos": 1, "files": 1, "symbols": 1, "endpoints": 1, "models": 1, "relations": 1,
                          "documents": 1, "notes": 1, "chunks": 1, "embeddings": 1, "conversations": 1, "messages": 1}


# --------------------------------------------------------------------------- robustesse

def test_tolerant_json_decoding(db):
    db.add_note(NoteRow(title="n", tags=["ok"]))
    db._conn.execute("UPDATE notes SET tags = 'pas du json'")
    db._conn.execute("INSERT INTO index_runs(started_at, status, stats) VALUES (1, 'ok', '[1, 2]')")
    assert db.get_note(1).tags == []  # texte invalide -> valeur par défaut
    assert db.last_index_run().stats == {}  # mauvais type -> valeur par défaut


def test_thread_safety_concurrent_writes_and_reads(db):
    errors = []

    def worker(k: int) -> None:
        try:
            for i in range(25):
                nid = db.add_note(NoteRow(title="t%d-%d" % (k, i), tags=["t"]))
                db.replace_chunks("note", nid, [ChunkRow(text="note %d %d thread" % (k, i))])
                with db.transaction():
                    db.set_meta("last", str(nid))
                    db.fts_search("thread")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert db.stats()["notes"] == 200 and db.count_chunks("note") == 200
    assert len(db.fts_search("thread", limit=500)) == 200
