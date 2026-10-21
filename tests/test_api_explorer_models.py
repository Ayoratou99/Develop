"""Tests de l'explorateur des modèles (``develop.api.routes.explorer``) : « Documents qui en parlent ».

Non-régression : les documents et notes qui documentent la table ``orders`` (dictionnaire de données
déposé, note datée) apparaissent sur **tous** les modèles de la même entité — migration ``orders``,
modèle Eloquent ``Order``, table SQL ``orders`` et interface TypeScript ``Order`` — et non plus
seulement sur les modèles cités littéralement dans le texte. Un serveur de test (``TestClient``)
indexe une copie de ``sample_project`` (``DEVELOP_HOME`` temporaire : rien n'est écrit dans la
fixture ni dans ``~/.develop``) ; un cas unitaire sur une base en mémoire vérifie les règles de
priorité (relation directe > détour) et de confiance (``min(relation, chemin)``).
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from develop.api.routes.explorer import _documents_mentioning
from develop.storage import Database, DocumentRow, FileRow, ModelRow, NoteRow, RelationRow, RepoRow, node_key

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"
INDEX_TIMEOUT_S = 60.0


def _fake_ollama(request: httpx.Request) -> httpx.Response:
    """Moteur factice minimal : le chat n'est pas exercé ici."""
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.12.1"})
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": []})
    return httpx.Response(404)


def _wait_index(client: TestClient) -> Dict[str, Any]:
    end = time.time() + INDEX_TIMEOUT_S
    while time.time() < end:
        status = client.get("/api/index/status").json()
        if not status["running"]:
            return status
        time.sleep(0.1)
    raise AssertionError("indexation trop longue (> %d s)" % INDEX_TIMEOUT_S)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[SimpleNamespace]:
    home = tmp_path_factory.mktemp("home")
    project = tmp_path_factory.mktemp("project") / "sample_project"
    shutil.copytree(FIXTURE, project)
    previous = os.environ.get("DEVELOP_HOME")
    os.environ["DEVELOP_HOME"] = str(home)
    from develop.api import create_app

    app = create_app(llm_transport=httpx.MockTransport(_fake_ollama))
    with TestClient(app) as client:
        ws = client.post("/api/workspaces", json={"name": "Boutique"}).json()
        repos = [
            {"name": "frontend", "path": str(project / "frontend"), "kind": "frontend"},
            {"name": "backend", "path": str(project / "backend"), "kind": "backend"},
            {"name": "notifier", "path": str(project / "services" / "notifier"), "kind": "backend"},
            {"name": "docs", "path": str(project / "docs"), "kind": "docs"},
            {"name": "database", "path": str(project / "database"), "kind": "database"},
        ]
        assert client.put("/api/workspaces/%s" % ws["id"], json={"repos": repos}).status_code == 200
        assert client.post("/api/index", json={"full": True}).status_code == 202
        assert _wait_index(client)["last_run"]["status"] == "ok"
        # dépôt du dictionnaire (import d'un fichier local, jamais déplacé) et d'une note datée
        r = client.post("/api/documents", json={"path": str(project / "docs" / "dictionnaire-donnees.md")})
        assert r.status_code == 201, r.text
        doc_id = int(r.json()["id"])
        r = client.post("/api/notes", json={"title": "Décision paiement", "note_date": "2026-10-01",
                                            "tags": ["décision"],
                                            "body": "Le paiement sera porté par `paid_at` sur la table `orders`."})
        assert r.status_code == 201, r.text
        note_id = int(r.json()["id"])
        # les relations documentaires (documented_by / mentions) sont recalculées par le linker au run suivant
        assert client.post("/api/index", json={}).status_code == 202
        assert _wait_index(client)["last_run"]["status"] == "ok"
        yield SimpleNamespace(client=client, project=project, doc_id=doc_id, note_id=note_id)
    if previous is None:
        os.environ.pop("DEVELOP_HOME", None)
    else:
        os.environ["DEVELOP_HOME"] = previous


def _models_named(client: TestClient, *names: str) -> Dict[str, Dict[str, Any]]:
    """Modèles de l'index dont le nom est dans ``names`` (insensible à la casse), par genre."""
    wanted = {n.lower() for n in names}
    return {m["kind"]: m for m in client.get("/api/models").json() if m["name"].lower() in wanted}


def _by_node(detail: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {item["node"]: item for item in detail["documents_mentioning"]}


def test_model_detail_lists_documents_of_the_whole_entity(env: SimpleNamespace):
    """Non-régression : « Documents qui en parlent » suit defines_table / same_entity (fiche du frontendeur)."""
    c = env.client
    by_kind = _models_named(c, "Order", "orders")
    assert {"eloquent", "migration", "sql_table", "ts_interface"} <= set(by_kind), sorted(by_kind)
    doc_node, note_node = node_key("document", env.doc_id), node_key("note", env.note_id)
    details = {kind: c.get("/api/models/%d" % m["id"]).json() for kind, m in by_kind.items()}
    for kind, d in details.items():
        items = _by_node(d)
        assert doc_node in items and note_node in items, (kind, d["documents_mentioning"])
        assert all({"id", "kind", "title", "node", "relation", "confidence", "via"} <= set(x)
                   for x in d["documents_mentioning"])
        doc = items[doc_node]
        assert doc["kind"] == "document" and doc["document_id"] == env.doc_id and doc["id"] == env.doc_id
        assert doc["title"] == "Dictionnaire de données" and doc["relation"] == "documented_by"
        assert 0 < doc["confidence"] <= 0.9
        note = items[note_node]
        assert note["kind"] == "note" and note["note_id"] == env.note_id
        assert note["title"].startswith("Décision paiement") and note["relation"] == "documented_by"
        assert [x["kind"] for x in d["documents_mentioning"]] == sorted(x["kind"] for x in d["documents_mentioning"])
    # la migration et la table SQL sont citées littéralement (« Table `orders` ») : relation directe
    for kind in ("migration", "sql_table"):
        assert _by_node(details[kind])[doc_node]["via"] is None, kind
        assert _by_node(details[kind])[doc_node]["confidence"] == 0.9
    # le modèle Eloquent et l'interface TS ne sont pas cités : le document arrive par l'entité (table ou
    # modèle same_entity), avec une confiance bornée par le chemin
    entity_nodes = {"table:orders"} | {node_key("model", m["id"]) for m in by_kind.values()}
    for kind in ("eloquent", "ts_interface"):
        doc = _by_node(details[kind])[doc_node]
        assert doc["via"] is not None and doc["via"]["node"] in entity_nodes, (kind, doc)
        assert doc["via"]["type"] in ("table", "model") and doc["via"]["label"]
        if doc["via"]["type"] == "table":
            assert doc["via"]["label"] == "table orders"
    assert _by_node(details["ts_interface"])[doc_node]["confidence"] < 0.9  # same_entity ≤ 0.8 vers l'interface
    # un modèle d'une autre entité n'hérite pas de la note (qui ne cite que la table orders)
    customers = _models_named(c, "Customer", "customers")
    assert customers, "modèle Customer attendu dans la fixture"
    for kind, m in customers.items():
        items = _by_node(c.get("/api/models/%d" % m["id"]).json())
        assert note_node not in items, (kind, items)


def test_documents_mentioning_prefers_direct_relation_and_bounds_confidence():
    """Règles du détour : relation directe > détour ; confiance = min(relation, chemin) ; un fichier n'a pas de détour."""
    with Database(":memory:") as db:
        db.upsert_repo(RepoRow(id="backend", name="backend", path="/nulle-part/backend", kind="backend"))
        fid = db.upsert_file(FileRow(repo_id="backend", path="app/Models/Order.php", language="php", kind="code",
                                     size=1, mtime=1.0, sha1="a"))
        ids = db.replace_file_artifacts(fid, models=[
            ModelRow(name="Order", kind="eloquent", table_name="orders", fields=[{"name": "id"}]),
            ModelRow(name="orders", kind="migration", table_name="orders"),
            ModelRow(name="Order", kind="ts_interface"),
        ])
        eloquent, migration, ts = (int(i) for i in ids["models"])
        doc = db.add_document(DocumentRow(title="Dictionnaire", kind="upload", format="md", stored_path="/x/d.md"))
        other = db.add_document(DocumentRow(title="Spécification", kind="upload", format="md", stored_path="/x/s.md"))
        note = db.add_note(NoteRow(title="Décision", body="…", note_date="2026-10-01"))
        m = lambda i: node_key("model", i)  # noqa: E731
        db.add_relations([
            RelationRow(src=m(migration), dst="table:orders", kind="defines_table"),
            RelationRow(src=m(eloquent), dst=m(migration), kind="same_entity", confidence=0.9),
            RelationRow(src=m(ts), dst=m(eloquent), kind="same_entity", confidence=0.7),
            RelationRow(src=m(ts), dst=m(migration), kind="same_entity", confidence=0.7),
            RelationRow(src="table:orders", dst=node_key("document", doc), kind="documented_by", confidence=0.9),
            RelationRow(src=m(ts), dst=node_key("document", doc), kind="documented_by", confidence=0.5),
            RelationRow(src=m(migration), dst=node_key("note", note), kind="documented_by", confidence=0.9),
            RelationRow(src=node_key("document", other), dst=node_key("file", fid), kind="mentions", confidence=0.7),
        ])
        doc_node, note_node = node_key("document", doc), node_key("note", note)
        # migration : dictionnaire via la table qu'elle définit ; note directe
        items = {x["node"]: x for x in _documents_mentioning(db, m(migration))}
        assert items[doc_node]["via"] == {"node": "table:orders", "type": "table", "label": "table orders"}
        assert items[doc_node]["confidence"] == 0.9 and items[doc_node]["relation"] == "documented_by"
        assert items[note_node]["via"] is None and items[note_node]["confidence"] == 0.9
        # eloquent : deux sauts (same_entity 0,9 → migration → table orders) : min(0,9, 0,9) ; note via la migration
        items = {x["node"]: x for x in _documents_mentioning(db, m(eloquent))}
        assert items[doc_node]["confidence"] == 0.9 and items[doc_node]["via"]["node"] in ("table:orders", m(migration))
        assert items[note_node]["via"]["node"] == m(migration) and items[note_node]["confidence"] == 0.9
        # interface TS : sa relation directe (0,5) l'emporte sur un détour plus confiant (0,7) ; note bornée par le chemin
        listed = _documents_mentioning(db, m(ts))
        items = {x["node"]: x for x in listed}
        assert items[doc_node]["via"] is None and items[doc_node]["confidence"] == 0.5
        assert items[note_node]["via"]["node"] == m(migration) and items[note_node]["confidence"] == 0.7
        assert [x["kind"] for x in listed] == ["document", "note"]  # tri (genre, titre) ; « Spécification » absente
        assert node_key("document", other) not in items
        # un fichier : relations directes seulement, jamais de détour par une entité
        file_items = _documents_mentioning(db, node_key("file", fid))
        assert [x["node"] for x in file_items] == [node_key("document", other)]
        assert file_items[0]["via"] is None and file_items[0]["relation"] == "mentions"
