"""Non-régression du module « web/documents + knowledge_graph/mentions ».

Deux corrections vérifiées ici :

* le **type résumé** d'une mention (``kind``) : une table citée par son nom (``orders``) est
  ``table`` même si des modèles homonymes la définissent (DDL ``sql_table``, ``migration``) ;
  une classe métier homonyme de sa table (entité JPA ``Customer`` ↔ table ``customer``) reste
  ``model`` ; un identifiant absent est ``unknown`` ; tous les noeuds restent dans ``nodes`` ;
* l'écran Documents (``develop/web/static/views/documents.js``) possède un libellé français pour
  **chaque** type que le module peut produire (plus de ``model`` / ``unknown`` bruts dans la
  colonne « Type »).

Tout se passe dans ``Database(":memory:")`` ; rien n'est écrit sur disque.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator

import pytest

from develop.code_analysis import normalize_route_path
from develop.knowledge_graph.mentions import MENTION_KINDS, TABLE_DEFINITION_KINDS, NameIndex, document_mentions
from develop.storage import ChunkRow, Database, DocumentRow, EndpointRow, FileRow, ModelRow, RepoRow, SymbolRow

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_JS = ROOT / "develop" / "web" / "static" / "views" / "documents.js"


@pytest.fixture
def db() -> Iterator[Database]:
    """Mini-index : table ``orders`` définie par un DDL et une migration, entité ``Customer`` sur la
    table ``customer``, fonction ``confirmOrder`` et route ``POST /orders/{id}/confirm``."""
    with Database(":memory:") as d:
        d.upsert_repo(RepoRow(id="backend", name="backend", path="/nonexistent/backend", kind="backend"))
        d.upsert_repo(RepoRow(id="database", name="database", path="/nonexistent/database", kind="database"))
        schema = d.upsert_file(FileRow(repo_id="database", path="schema.sql", language="sql", kind="data"))
        d.replace_file_artifacts(schema, models=[
            ModelRow(name="orders", kind="sql_table", table_name="orders", line=9,
                     fields=[{"name": "id"}, {"name": "status"}, {"name": "confirmed_at"}]),
            ModelRow(name="customer", kind="sql_table", table_name="customer", line=2, fields=[{"name": "email"}]),
        ])
        migration = d.upsert_file(FileRow(repo_id="backend", path="database/migrations/2026_01_01_000000_create_orders_table.php",
                                          language="php", kind="code"))
        d.replace_file_artifacts(migration, models=[
            ModelRow(name="orders", kind="migration", table_name="orders", line=12, fields=[{"name": "id"}])])
        entity = d.upsert_file(FileRow(repo_id="backend", path="src/main/java/com/acme/Customer.java", language="java",
                                       kind="code"))
        d.replace_file_artifacts(entity,
                                 symbols=[SymbolRow(name="Customer", kind="class", line_start=3, line_end=30)],
                                 models=[ModelRow(name="Customer", kind="jpa", table_name="customer", line=3,
                                                  fields=[{"name": "email"}])])
        client = d.upsert_file(FileRow(repo_id="backend", path="src/api/client.ts", language="typescript", kind="code"))
        d.replace_file_artifacts(client,
                                 symbols=[SymbolRow(name="confirmOrder", kind="function", line_start=10, line_end=14)],
                                 endpoints=[EndpointRow(side="server", method="POST", path="/orders/{id}/confirm",
                                                        normalized_path=normalize_route_path("/orders/{id}/confirm"),
                                                        line=11)])
        yield d


def _mentions(db: Database, text: str) -> Dict[str, Dict[str, object]]:
    doc_id = db.add_document(DocumentRow(title="Spécification", stored_path="/nonexistent/spec.md", format="md"))
    db.replace_chunks("document", doc_id, [ChunkRow(text=text, ordinal=0, line_start=1, line_end=1)])
    return {m["name"]: m for m in document_mentions(db, "document", doc_id)}


# --------------------------------------------------------------------------- type résumé des mentions

def test_cited_table_name_is_reported_as_table_not_model(db: Database):
    # reproduction du défaut : « La table `orders` … `OrderArchiver` … `confirmOrder` … `POST /orders/{id}/confirm` »
    m = _mentions(db, "La table `orders` porte les commandes. L'ancienne classe `OrderArchiver` est retirée. "
                      "La fonction `confirmOrder` appelle `POST /orders/{id}/confirm`.")
    assert m["orders"]["kind"] == "table" and m["orders"]["exists"] is True
    # les modèles qui définissent la table restent dans nodes (relations documented_by inchangées)
    assert "table:orders" in m["orders"]["nodes"] and sum(n.startswith("model:") for n in m["orders"]["nodes"]) == 2
    assert m["confirmOrder"]["kind"] == "symbol" and m["confirmOrder"]["exists"] is True
    assert m["/orders/{id}/confirm"]["kind"] == "endpoint" and m["/orders/{id}/confirm"]["exists"] is True
    assert m["OrderArchiver"]["kind"] == "unknown" and m["OrderArchiver"]["exists"] is False
    assert {x["kind"] for x in m.values()} == {"table", "symbol", "endpoint", "unknown"}
    assert all(x["kind"] in MENTION_KINDS for x in m.values())
    # une table citée par son nom en Capitale (mot ordinaire) : toujours une table
    assert _mentions(db, "Les Orders sont confirmées.")["Orders"]["kind"] == "table"


def test_entity_class_named_like_its_table_stays_a_model(db: Database):
    idx = NameIndex.build(db)
    assert "customer" in idx.entity_models and "orders" not in idx.entity_models
    assert {"sql_table", "migration"} == set(TABLE_DEFINITION_KINDS)
    m = _mentions(db, "L'entité `Customer` porte l'adresse e-mail du client ; voir `confirmed_at`.")
    assert m["Customer"]["kind"] == "model" and m["Customer"]["exists"] is True
    assert "table:customer" in m["Customer"]["nodes"]  # la table reste une correspondance
    assert m["confirmed_at"]["kind"] == "field"
    # les mentions d'une table disparue restent signalées comme inconnues
    gone = _mentions(db, "La table `invoices` et la classe `InvoiceService` n'existent plus.")
    assert gone["invoices"]["kind"] == "unknown" and gone["InvoiceService"]["kind"] == "unknown"


# --------------------------------------------------------------------------- écran Documents

def test_documents_view_labels_every_mention_kind_in_french():
    js = DOCUMENTS_JS.read_text(encoding="utf-8")
    block = re.search(r"const MENTION_LABELS = \{([^}]*)\}", js)
    assert block, "MENTION_LABELS introuvable dans documents.js"
    labels = dict(re.findall(r"(\w+):\s*'([^']*)'", block.group(1)))
    assert set(labels) == set(MENTION_KINDS), "chaque type de mention doit avoir un libellé"
    assert labels == {"symbol": "symbole", "model": "modèle", "table": "table", "endpoint": "endpoint",
                      "field": "champ", "file": "fichier", "unknown": "inconnu"}
    # la colonne Type passe toujours par le dictionnaire de libellés
    assert "MENTION_LABELS[m.kind]" in js
