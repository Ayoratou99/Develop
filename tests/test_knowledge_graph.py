"""Tests du graphe de connaissances (``develop.knowledge_graph``).

Deux niveaux :

* un **mini-graphe inséré à la main** dans ``Database(":memory:")`` qui reproduit la fixture
  ``sample_project`` (dépôts, fichiers, symboles, endpoints, modèles, relations brutes, morceaux
  de documents) et permet de vérifier chaque règle de liaison avec des confiances exactes ;
* un **test d'intégration** : copie de la fixture dans ``tmp_path``, indexation réelle de cinq
  dépôts (frontend, backend, services/notifier, docs, database) puis vérification du graphe.

Les dépôts de l'utilisateur et la fixture ne sont jamais modifiés : tout fichier lu par le
linker (``.env.example``, ``composer.json``, ``vite.config.ts``) est écrit dans ``tmp_path``.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pytest

from develop.code_analysis import normalize_route_path
from develop.config import RepoConfig, WorkspaceConfig
from develop.indexing import Indexer
from develop.knowledge_graph import (
    DERIVED_KINDS,
    LinkStats,
    NameIndex,
    describe_relation,
    document_mentions,
    extract_identifiers,
    is_stale,
    learn_base_prefixes,
    link_all,
    link_calls,
    link_documents,
    link_handlers,
    link_models,
    match_endpoints,
    neighbors_graph,
    resolve_imports,
)
from develop.storage import (
    ChunkRow,
    Database,
    DocumentRow,
    EndpointRow,
    FileRow,
    ModelRow,
    NoteRow,
    RelationRow,
    RepoRow,
    SymbolRow,
    node_key,
)

CLIENT_TS = '''import axios from "axios";

export const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api";

export const http = axios.create({ baseURL: API_BASE });

export interface Order {
  id: number;
  reference: string;
  customer_id: number;
  status: "draft" | "confirmed" | "shipped" | "cancelled";
  total_amount: number;
}

export async function confirmOrder(id: number): Promise<Order> {
  const res = await http.post<Order>(`/orders/${id}/confirm`);
  return res.data;
}

export async function searchCustomers(q: string): Promise<Customer[]> {
  const res = await fetch(`${API_BASE}/customers?q=${encodeURIComponent(q)}`);
  return res.json();
}
'''

DICTIONARY_MD = """# Dictionnaire de données

## Table `orders` (Commandes)

| Colonne      | Type            | Description                                               |
|--------------|-----------------|-----------------------------------------------------------|
| id           | bigint          | Identifiant technique                                     |
| customer_id  | bigint (FK)     | Client propriétaire (customers.id)                        |
| status       | enum            | draft, confirmed, shipped, cancelled                      |
| total_amount | unsigned bigint | Montant total en centimes d'euro                          |
| confirmed_at | timestamp       | Date de confirmation, renseignée par OrderService::confirm |

## Table `order_lines` (Lignes de commande)

| Colonne     | Type        | Description        |
|-------------|-------------|--------------------|
| order_id    | bigint (FK) | Commande parente   |
| product_sku | varchar(40) | Référence produit  |
"""

SPEC_MD = """# Spécification des commandes

Le bouton « Confirmer » de la liste appelle `confirmOrder`, qui envoie POST /api/orders/{id}/confirm.
Le serveur refuse (422) toute commande dont le statut n'est pas `draft`.
"""

NOTE_BODY = """Décision : `OrderService::confirm` passe le statut à `confirmed` et renseigne `confirmed_at`.
La classe `OrderArchiver` sera retirée du backend. Voir POST /orders/{id}/confirm."""


# --------------------------------------------------------------------------- construction du mini-graphe

class Graph:
    """Mini-graphe en mémoire reproduisant la fixture ; conserve les identifiants utiles aux tests."""

    def __init__(self, db: Database, root: Path, sample_project: Path) -> None:
        self.db = db
        self.root = root
        self.files: Dict[str, int] = {}
        self.symbols: Dict[str, int] = {}
        self.endpoints: Dict[str, int] = {}
        self.models: Dict[str, int] = {}
        self.documents: Dict[str, int] = {}
        self.notes: Dict[str, int] = {}
        self.workspace = self._workspace(sample_project)
        for repo in self.workspace.repos:
            db.upsert_repo(RepoRow(id=repo.id, name=repo.name, path=repo.path, kind=repo.kind))
        self._frontend()
        self._backend()
        self._notifier()
        self._database()
        self._tools()
        self._mobile_and_java()
        self._documents()

    # ----- espace de travail (dossiers dans tmp_path, jamais la fixture) -----
    def _workspace(self, sample_project: Path) -> WorkspaceConfig:
        for name in ("frontend", "backend", "notifier", "database", "tools", "mobile", "java", "spa"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        shutil.copy(sample_project / "frontend" / ".env.example", self.root / "frontend" / ".env.example")
        (self.root / "backend" / "composer.json").write_text(json.dumps({
            "name": "monprojet/backend",
            "autoload": {"psr-4": {"App\\": "app/", "Acme\\Billing\\": "packages/billing/src/"}},
        }), encoding="utf-8")
        (self.root / "spa" / "vite.config.ts").write_text(
            "export default { server: { proxy: { '/backend': { target: 'http://localhost:9000' } } } };\n",
            encoding="utf-8")
        return WorkspaceConfig(id="ws", name="Espace de test", repos=[
            RepoConfig(id="frontend", name="frontend", path=str(self.root / "frontend"), kind="frontend"),
            RepoConfig(id="backend", name="backend", path=str(self.root / "backend"), kind="backend"),
            RepoConfig(id="notifier", name="notifier", path=str(self.root / "notifier"), kind="backend"),
            RepoConfig(id="database", name="database", path=str(self.root / "database"), kind="database"),
            RepoConfig(id="tools", name="tools", path=str(self.root / "tools"), kind="other"),
            RepoConfig(id="mobile", name="mobile", path=str(self.root / "mobile"), kind="mobile"),
            RepoConfig(id="java", name="java", path=str(self.root / "java"), kind="backend"),
            RepoConfig(id="spa", name="spa", path=str(self.root / "spa"), kind="frontend"),
        ])

    # ----- aides -----
    def add_file(self, repo_id: str, path: str, language: str, kind: str = "code",
                 symbols: Sequence[SymbolRow] = (), endpoints: Sequence[EndpointRow] = (),
                 models: Sequence[ModelRow] = (), text: Optional[str] = None,
                 imports: Sequence[Tuple[str, Sequence[str], bool]] = ()) -> int:
        """Insère un fichier et ses artefacts comme le ferait l'indexeur (relations brutes comprises)."""
        fid = self.db.upsert_file(FileRow(repo_id=repo_id, path=path, language=language, kind=kind, size=100,
                                          mtime=1.0, sha1="sha-" + path))
        chunks = [ChunkRow(text=text, ordinal=0, line_start=1, line_end=max(1, text.count("\n") + 1))] if text else []
        self.db.replace_file_artifacts(fid, symbols=list(symbols), endpoints=list(endpoints), models=list(models),
                                       chunks=chunks)
        self.files["%s/%s" % (repo_id, path)] = fid
        for s in symbols:
            label = "%s.%s" % (s.parent, s.name) if s.parent else s.name
            self.symbols[label] = int(s.id)  # type: ignore[arg-type]
        for e in endpoints:
            self.endpoints["%s %s %s" % (e.side, e.method, e.path)] = int(e.id)  # type: ignore[arg-type]
        for m in models:
            self.models["%s:%s" % (m.kind, m.name)] = int(m.id)  # type: ignore[arg-type]
        rels: List[RelationRow] = []
        file_key = node_key("file", fid)
        for module, names, is_relative in imports:
            rels.append(RelationRow(src=file_key, dst=node_key("module", module), kind="imports_raw",
                                    meta={"names": list(names), "line": 1, "is_relative": is_relative}))
        for m in models:
            if m.table_name and m.kind in ("migration", "sql_table", "eloquent"):
                rels.append(RelationRow(src=node_key("model", m.id), dst=node_key("table", m.table_name),  # type: ignore[arg-type]
                                        kind="defines_table", confidence=1.0 if m.kind != "eloquent" else 0.8,
                                        meta={"model_kind": m.kind}))
        if rels:
            self.db.add_relations(rels)
        return fid

    @staticmethod
    def server(method: str, path: str, handler: Optional[str] = None, framework: str = "laravel",
               full_path: Optional[str] = None, api_prefix: Optional[str] = None, line: int = 1) -> EndpointRow:
        meta: Dict[str, Any] = {}
        if full_path:
            meta["full_path"] = full_path
        if api_prefix:
            meta["api_prefix"] = api_prefix
        return EndpointRow(side="server", method=method, path=path, normalized_path=normalize_route_path(full_path or path),
                           handler=handler, framework=framework, line=line, meta=meta)

    @staticmethod
    def client(method: str, path: str, handler: str, framework: str = "axios", line: int = 1,
               **meta: Any) -> EndpointRow:
        return EndpointRow(side="client", method=method, path=path, normalized_path=normalize_route_path(path),
                           handler=handler, framework=framework, line=line, meta=dict(meta))

    # ----- dépôts -----
    def _frontend(self) -> None:
        self.add_file(
            "frontend", "src/api/client.ts", "typescript",
            symbols=[SymbolRow(name="API_BASE", kind="constant", line_start=3, line_end=3),
                     SymbolRow(name="http", kind="constant", line_start=5, line_end=5),
                     SymbolRow(name="Order", kind="interface", line_start=7, line_end=13),
                     SymbolRow(name="confirmOrder", kind="function", line_start=15, line_end=18),
                     SymbolRow(name="searchCustomers", kind="function", line_start=20, line_end=23)],
            endpoints=[self.client("GET", "/orders", "listOrders", receiver="http"),
                       self.client("GET", "/orders/${id}", "getOrder", receiver="http"),
                       self.client("POST", "/orders/${id}/confirm", "confirmOrder", receiver="http", line=16),
                       self.client("GET", "${API_BASE}/customers?q=${encodeURIComponent(q)}", "searchCustomers",
                                   framework="fetch", receiver="fetch", call="fetch", base="API_BASE", line=21)],
            models=[ModelRow(name="Order", kind="ts_interface", line=7,
                             fields=[{"name": "id"}, {"name": "customer_id"}, {"name": "status"}, {"name": "total_amount"}]),
                    ModelRow(name="Customer", kind="ts_interface", line=25, fields=[{"name": "id"}, {"name": "email"}]),
                    ModelRow(name="OrderLine", kind="ts_interface", line=30, fields=[{"name": "product_sku"}]),
                    ModelRow(name="Invoice", kind="ts_interface", line=35, fields=[{"name": "number"}])],
            text=CLIENT_TS,
            imports=[("axios", ["axios"], False)],
        )
        self.add_file(
            "frontend", "src/components/OrderList.tsx", "tsx",
            symbols=[SymbolRow(name="OrderList", kind="component", line_start=5, line_end=20)],
            imports=[("react", ["React", "useEffect", "useState"], False),
                     ("../api/client", ["listOrders", "confirmOrder", "Order"], True),
                     ("../utils/money", ["formatAmount"], True),
                     ("../lib", ["debounce"], True),
                     ("@/config/env", ["ENV"], False)],
        )
        self.add_file(
            "frontend", "src/components/CustomerSearch.tsx", "tsx",
            symbols=[SymbolRow(name="CustomerSearch", kind="component", line_start=4, line_end=20)],
            imports=[("react", ["React", "useState"], False), ("../api/client", ["searchCustomers", "Customer"], True)],
        )
        self.add_file(
            "frontend", "src/components/Notifications.jsx", "jsx",
            symbols=[SymbolRow(name="Notifications", kind="component", line_start=3, line_end=10)],
            endpoints=[self.client("GET", "http://localhost:3001/notifications/${userId}", "Notifications",
                                   framework="fetch", receiver="fetch", call="fetch", base="http://localhost:3001", line=6)],
            imports=[("react", ["useEffect", "useState"], False)],
        )
        self.add_file(
            "frontend", "src/components/OrderActions.tsx", "tsx",
            symbols=[SymbolRow(name="OrderActions", kind="component", line_start=3, line_end=12)],
            endpoints=[self.client("DELETE", "/orders/${id}", "OrderActions", receiver="http", line=5),
                       self.client("GET", "/status", "OrderActions", framework="fetch", receiver="fetch", line=8)],
            imports=[("../api/client", ["http"], True)],
        )
        self.add_file("frontend", "src/utils/money.ts", "typescript",
                      symbols=[SymbolRow(name="formatAmount", kind="function", line_start=1, line_end=3)])
        self.add_file("frontend", "src/lib/index.ts", "typescript",
                      symbols=[SymbolRow(name="debounce", kind="function", line_start=1, line_end=5)])
        self.add_file("frontend", "src/config/env.ts", "typescript",
                      symbols=[SymbolRow(name="ENV", kind="constant", line_start=1, line_end=1)])

    def _backend(self) -> None:
        self.add_file(
            "backend", "routes/api.php", "php",
            endpoints=[self.server("GET", "/orders", "OrderController@index", full_path="/api/orders", api_prefix="/api", line=10),
                       self.server("GET", "/orders/{order}", "OrderController@show", full_path="/api/orders/{order}",
                                   api_prefix="/api", line=11),
                       self.server("POST", "/orders/{order}/confirm", "OrderController@confirm",
                                   full_path="/api/orders/{order}/confirm", api_prefix="/api", line=12),
                       self.server("GET", "/customers", "CustomerController@index", line=13),
                       self.server("GET", "/status", "closure", line=14)],
            imports=[("App\\Http\\Controllers\\OrderController", ["OrderController"], False)],
        )
        self.add_file(
            "backend", "app/Http/Controllers/OrderController.php", "php",
            symbols=[SymbolRow(name="OrderController", kind="class", line_start=9, line_end=30),
                     SymbolRow(name="confirm", kind="method", parent="OrderController", line_start=24, line_end=27,
                               signature="confirm(Request $request, Order $order)")],
            imports=[("App\\Models\\Order", ["Order"], False), ("App\\Services\\OrderService", ["OrderService"], False),
                     ("Acme\\Billing\\Invoice", ["Invoice"], False), ("Illuminate\\Http\\Request", ["Request"], False)],
        )
        self.add_file(
            "backend", "app/Services/OrderService.php", "php",
            symbols=[SymbolRow(name="OrderService", kind="class", line_start=8, line_end=30),
                     SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=10, line_end=24,
                               signature="confirm(Order $order): Order")],
            endpoints=[self.client("POST", "http://localhost:3001/notifications", "OrderService@confirm",
                                   framework="laravel-http", line=18, base="http://localhost:3001")],
            imports=[("App\\Models\\Order", ["Order"], False), ("Illuminate\\Support\\Facades\\Http", ["Http"], False)],
        )
        self.add_file(
            "backend", "app/Models/Order.php", "php",
            symbols=[SymbolRow(name="Order", kind="class", line_start=9, line_end=28)],
            models=[ModelRow(name="Order", kind="eloquent", table_name="orders", line=9,
                             fields=[{"name": "reference"}, {"name": "customer_id"}, {"name": "status"}, {"name": "total_amount"}])],
            imports=[("Illuminate\\Database\\Eloquent\\Model", ["Model"], False)],
        )
        self.add_file("backend", "app/Models/OrderLine.php", "php",
                      symbols=[SymbolRow(name="OrderLine", kind="class", line_start=7, line_end=20)],
                      models=[ModelRow(name="OrderLine", kind="eloquent", table_name="order_lines", line=7,
                                       fields=[{"name": "order_id"}, {"name": "product_sku"}])])
        self.add_file("backend", "app/Models/Customer.php", "php",
                      symbols=[SymbolRow(name="Customer", kind="class", line_start=7, line_end=20)],
                      models=[ModelRow(name="Customer", kind="eloquent", table_name="customers", line=7,
                                       fields=[{"name": "email"}, {"name": "full_name"}])])
        self.add_file("backend", "app/Models/Invoice.php", "php",
                      symbols=[SymbolRow(name="Invoice", kind="class", line_start=7, line_end=20)],
                      models=[ModelRow(name="Invoice", kind="eloquent", table_name=None, line=7, fields=[{"name": "number"}])])
        self.add_file("backend", "packages/billing/src/Invoice.php", "php",
                      symbols=[SymbolRow(name="Invoice", kind="class", line_start=5, line_end=30)])
        self.add_file("backend", "database/migrations/2024_01_01_000000_create_orders_table.php", "php",
                      models=[ModelRow(name="orders", kind="migration", table_name="orders", line=11,
                                       fields=[{"name": "id"}, {"name": "reference"}, {"name": "customer_id"},
                                               {"name": "status"}, {"name": "total_amount"}, {"name": "confirmed_at"}])])
        self.add_file("backend", "database/migrations/2024_02_01_000000_create_order_lines_table.php", "php",
                      models=[ModelRow(name="order_lines", kind="migration", table_name="order_lines", line=11,
                                       fields=[{"name": "order_id"}, {"name": "product_sku"}, {"name": "quantity"}])])

    def _notifier(self) -> None:
        self.add_file(
            "notifier", "src/server.js", "javascript",
            symbols=[SymbolRow(name="app", kind="constant", line_start=4, line_end=4)],
            endpoints=[self.server("GET", "/notifications/:userId", None, framework="express", line=7),
                       self.server("POST", "/notifications", None, framework="express", line=11),
                       self.server("DELETE", "/notifications/:id", None, framework="express", line=16),
                       self.server("GET", "/status", None, framework="express", line=20)],
            imports=[("express", ["express"], False), ("./store", ["store"], True)],
        )
        self.add_file("notifier", "src/store.js", "javascript",
                      symbols=[SymbolRow(name="add", kind="function", line_start=4, line_end=8)])

    def _database(self) -> None:
        self.add_file("database", "schema.sql", "sql", kind="data",
                      models=[ModelRow(name="orders", kind="sql_table", table_name="orders", line=9,
                                       fields=[{"name": "id"}, {"name": "reference"}, {"name": "status"}]),
                              ModelRow(name="order_lines", kind="sql_table", table_name="order_lines", line=18,
                                       fields=[{"name": "order_id"}, {"name": "quantity"}]),
                              ModelRow(name="customers", kind="sql_table", table_name="customers", line=2,
                                       fields=[{"name": "email"}])])

    def _tools(self) -> None:
        self.add_file("tools", "report.py", "python",
                      symbols=[SymbolRow(name="main", kind="function", line_start=5, line_end=20)],
                      imports=[("helpers", ["fmt"], False), ("requests", [], False)])
        self.add_file("tools", "helpers/__init__.py", "python")
        self.add_file("tools", "helpers/fmt.py", "python",
                      symbols=[SymbolRow(name="money", kind="function", line_start=3, line_end=5)],
                      imports=[(".util", ["money"], True)])
        self.add_file("tools", "helpers/util.py", "python",
                      symbols=[SymbolRow(name="cents", kind="function", line_start=1, line_end=2)])

    def _mobile_and_java(self) -> None:
        self.add_file("mobile", "lib/api.dart", "dart",
                      symbols=[SymbolRow(name="ApiClient", kind="class", line_start=3, line_end=20)],
                      text='final dio = Dio(BaseOptions(baseUrl: "http://10.0.2.2:8000/api/v1"));\n')
        self.add_file("mobile", "lib/models/order.dart", "dart",
                      symbols=[SymbolRow(name="Order", kind="class", line_start=1, line_end=10)])
        self.add_file("mobile", "lib/screens/orders.dart", "dart",
                      symbols=[SymbolRow(name="OrdersScreen", kind="class", line_start=4, line_end=40)],
                      imports=[("package:app/api.dart", [], False), ("../models/order.dart", [], True),
                               ("package:flutter/material.dart", [], False)])
        self.add_file("java", "src/main/java/com/acme/OrderService.java", "java",
                      symbols=[SymbolRow(name="OrderService", kind="class", line_start=5, line_end=40)],
                      imports=[("com.acme.model.Order", ["Order"], False), ("java.util.List", ["List"], False)])
        self.add_file("java", "src/main/java/com/acme/model/Order.java", "java",
                      symbols=[SymbolRow(name="Order", kind="class", line_start=3, line_end=30)])

    # ----- documents et notes -----
    def _documents(self) -> None:
        self.documents["dictionnaire"] = self._document("Dictionnaire de données", DICTIONARY_MD)
        self.documents["spec"] = self._document("Spécification des commandes", SPEC_MD)
        self.notes["reunion"] = self.db.add_note(NoteRow(title="Réunion statuts", body=NOTE_BODY, note_date="2026-09-01",
                                                         tags=["statuts", "commandes"]))

    def _document(self, title: str, text: str) -> int:
        doc_id = self.db.add_document(DocumentRow(title=title, stored_path="/nonexistent/%s.md" % title, format="md"))
        parts = text.split("\n## ")
        rows = [ChunkRow(text=(p if i == 0 else "## " + p), ordinal=i, line_start=1, line_end=1) for i, p in enumerate(parts)]
        self.db.replace_chunks("document", doc_id, rows)
        return doc_id

    # ----- accès -----
    def file(self, repo_id: str, path: str) -> int:
        return self.files["%s/%s" % (repo_id, path)]

    def fkey(self, repo_id: str, path: str) -> str:
        return node_key("file", self.file(repo_id, path))

    def ekey(self, side: str, method: str, path: str) -> str:
        return node_key("endpoint", self.endpoints["%s %s %s" % (side, method, path)])

    def mkey(self, kind: str, name: str) -> str:
        return node_key("model", self.models["%s:%s" % (kind, name)])

    def skey(self, label: str) -> str:
        return node_key("symbol", self.symbols[label])


def _relations(db: Database, kind: str) -> Dict[Tuple[str, str], RelationRow]:
    rows = db.query("SELECT * FROM relations WHERE kind = ? ORDER BY src, dst", (kind,))
    return {(r["src"], r["dst"]): RelationRow.from_row(r) for r in rows}


def _snapshot(db: Database) -> Set[Tuple[str, str, str, float]]:
    rows = db.query("SELECT src, dst, kind, confidence FROM relations")
    return {(r["src"], r["dst"], r["kind"], round(float(r["confidence"]), 4)) for r in rows}


def _undirected(rels: Dict[Tuple[str, str], RelationRow], a: str, b: str) -> Optional[RelationRow]:
    return rels.get((a, b)) or rels.get((b, a))


# --------------------------------------------------------------------------- fixtures

@pytest.fixture
def db():
    with Database(":memory:") as database:
        yield database


@pytest.fixture
def graph(db: Database, tmp_path: Path, sample_project: Path) -> Graph:
    return Graph(db, tmp_path, sample_project)


@pytest.fixture
def linked(graph: Graph) -> Graph:
    """Mini-graphe après une passe complète de liaison."""
    link_all(graph.db, graph.workspace)
    return graph


# --------------------------------------------------------------------------- préfixes de base

def test_learn_base_prefixes_from_env_axios_vite_and_laravel(graph: Graph):
    db, ws = graph.db, graph.workspace
    # frontend : .env.example (VITE_API_URL=http://localhost:8000/api) + axios.create({ baseURL: API_BASE })
    assert learn_base_prefixes(db, ws.repo_by_id("frontend")) == ["/api"]
    assert json.loads(db.get_meta("repo:frontend:base_prefixes")) == ["/api"]
    variables = json.loads(db.get_meta("repo:frontend:base_vars"))
    assert variables["VITE_API_URL"] == "/api" and variables["API_BASE"] == "/api" and variables["http"] == "/api"
    assert not any("://" in k for k in variables), "jamais d'hôte dans les variables apprises"
    # backend : routes/api.php (convention Laravel) et meta.api_prefix des routes
    assert learn_base_prefixes(db, ws.repo_by_id("backend")) == ["/api"]
    # notifier : rien à apprendre
    assert learn_base_prefixes(db, ws.repo_by_id("notifier")) == []
    assert db.get_meta("repo:notifier:base_prefixes") == "[]"
    # vite.config.ts : clé de proxy
    assert learn_base_prefixes(db, ws.repo_by_id("spa")) == ["/backend"]
    # Dio(BaseOptions(baseUrl: ...)) : chemin après l'hôte
    assert learn_base_prefixes(db, ws.repo_by_id("mobile")) == ["/api/v1"]


def test_learn_base_prefixes_without_env_file_falls_back_to_literal(graph: Graph):
    (graph.root / "frontend" / ".env.example").unlink()
    assert learn_base_prefixes(graph.db, graph.workspace.repo_by_id("frontend")) == ["/api"]


def test_learn_base_prefixes_ignores_database_and_secret_variables(db: Database, tmp_path: Path):
    repo_dir = tmp_path / "svc"
    repo_dir.mkdir()
    (repo_dir / ".env").write_text("DATABASE_URL=postgres://user:secret@db:5432/app\nAPI_URL=https://example.test/v2/\n"
                                   "REDIS_HOST=redis\nMAIL_HOST=smtp.example.test\n", encoding="utf-8")
    repo = RepoConfig(id="svc", name="svc", path=str(repo_dir), kind="backend")
    db.upsert_repo(RepoRow(id="svc", name="svc", path=str(repo_dir)))
    assert learn_base_prefixes(db, repo) == ["/v2"]
    stored = db.get_meta("repo:svc:base_vars") or ""
    assert "secret" not in stored and "postgres" not in stored


# --------------------------------------------------------------------------- résolution des imports

def test_resolve_imports_all_languages(graph: Graph):
    db = graph.db
    raw_before = len(_relations(db, "imports_raw"))
    resolved, unresolved = resolve_imports(db, graph.workspace)
    assert (resolved, unresolved) == (17, 11)
    imports = _relations(db, "imports")
    assert len(imports) == 18
    assert len(_relations(db, "imports_raw")) == raw_before, "les imports bruts sont conservés"
    f = graph.fkey
    # JS/TS : relatifs avec extension implicite, index.ts, alias @/ → src/
    assert (f("frontend", "src/components/OrderList.tsx"), f("frontend", "src/api/client.ts")) in imports
    assert (f("frontend", "src/components/OrderList.tsx"), f("frontend", "src/utils/money.ts")) in imports
    assert (f("frontend", "src/components/OrderList.tsx"), f("frontend", "src/lib/index.ts")) in imports
    assert (f("frontend", "src/components/OrderList.tsx"), f("frontend", "src/config/env.ts")) in imports
    assert (f("notifier", "src/server.js"), f("notifier", "src/store.js")) in imports
    rel = imports[(f("frontend", "src/components/OrderList.tsx"), f("frontend", "src/api/client.ts"))]
    assert rel.confidence == 1.0 and rel.meta["names"] == ["listOrders", "confirmOrder", "Order"]
    # modules nus (react, axios, express, requests…) : jamais résolus
    assert not any(dst.startswith("module:") for _src, dst in imports)
    assert not any(src == f("frontend", "src/api/client.ts") for src, _dst in imports)
    # Python : paquet (__init__.py) + sous-module importé, relatif .util
    assert (f("tools", "report.py"), f("tools", "helpers/__init__.py")) in imports
    assert (f("tools", "report.py"), f("tools", "helpers/fmt.py")) in imports
    assert (f("tools", "helpers/fmt.py"), f("tools", "helpers/util.py")) in imports
    # PHP : PSR-4 depuis composer.json (préfixe personnalisé) et App\ → app/
    assert (f("backend", "app/Http/Controllers/OrderController.php"), f("backend", "packages/billing/src/Invoice.php")) in imports
    assert (f("backend", "app/Http/Controllers/OrderController.php"), f("backend", "app/Models/Order.php")) in imports
    assert (f("backend", "app/Services/OrderService.php"), f("backend", "app/Models/Order.php")) in imports
    assert (f("backend", "routes/api.php"), f("backend", "app/Http/Controllers/OrderController.php")) in imports
    # Java et Dart
    assert (f("java", "src/main/java/com/acme/OrderService.java"), f("java", "src/main/java/com/acme/model/Order.java")) in imports
    assert (f("mobile", "lib/screens/orders.dart"), f("mobile", "lib/api.dart")) in imports
    assert (f("mobile", "lib/screens/orders.dart"), f("mobile", "lib/models/order.dart")) in imports
    # jamais d'import entre dépôts différents
    for src, dst in imports:
        assert db.get_file_by_id(int(src[5:])).repo_id == db.get_file_by_id(int(dst[5:])).repo_id


def test_resolve_imports_without_composer_uses_app_convention(graph: Graph):
    (graph.root / "backend" / "composer.json").unlink()
    resolve_imports(graph.db, graph.workspace)
    imports = _relations(graph.db, "imports")
    f = graph.fkey
    assert (f("backend", "app/Services/OrderService.php"), f("backend", "app/Models/Order.php")) in imports
    assert (f("backend", "app/Http/Controllers/OrderController.php"), f("backend", "packages/billing/src/Invoice.php")) not in imports


def test_resolve_imports_php_fallback_ignores_vendor_namespaces(graph: Graph):
    """Non-régression : sans PSR-4 (``composer.json`` sans ``autoload``), ``use Illuminate\\Http\\Request`` ne doit
    pas tomber sur ``app/Http/Request.php`` : le repli ne vaut que pour un dossier portant le nom de la racine de
    l'espace de noms (``App`` → ``app/`` ou ``src/``), jamais pour un espace de noms vendor."""
    db = graph.db
    f = graph.fkey
    (graph.root / "backend" / "composer.json").write_text('{"name": "monprojet/backend"}', encoding="utf-8")
    graph.add_file("backend", "app/Http/Request.php", "php",
                   symbols=[SymbolRow(name="Request", kind="class", line_start=3, line_end=10)])
    graph.add_file("backend", "src/Entity/User.php", "php",
                   symbols=[SymbolRow(name="User", kind="class", line_start=3, line_end=10)])
    ping = graph.add_file("backend", "app/Http/Controllers/PingController.php", "php",
                          symbols=[SymbolRow(name="PingController", kind="class", line_start=5, line_end=12)],
                          imports=[("App\\Http\\Request", ["Request"], False), ("App\\Entity\\User", ["User"], False),
                                   ("Vendor\\Entity\\User", ["User"], False)])
    resolve_imports(db, graph.workspace)
    imports = _relations(db, "imports")
    request = f("backend", "app/Http/Request.php")
    # OrderController.php importe Illuminate\Http\Request : aucune relation (racine Illuminate ≠ App)
    assert (f("backend", "app/Http/Controllers/OrderController.php"), request) not in imports
    assert [src for src, dst in imports if dst == request] == [node_key("file", ping)]
    # l'espace de noms du projet garde ses conventions : App\ → app/ (Laravel) ou src/ (Symfony), Vendor\ → rien
    user = f("backend", "src/Entity/User.php")
    assert (node_key("file", ping), user) in imports and imports[(node_key("file", ping), user)].meta["modules"] == ["App\\Entity\\User"]
    # avec composer.json absent (fixture réelle) le résultat est le même
    (graph.root / "backend" / "composer.json").unlink()
    resolve_imports(db, graph.workspace)
    assert [src for src, dst in _relations(db, "imports") if dst == request] == [node_key("file", ping)]


def test_resolve_imports_js_bare_module_requires_a_folder_not_a_file(graph: Graph):
    """Non-régression : un module nu n'est résolu que vers un dossier du dépôt (``shared`` → ``src/shared/index.ts``,
    ``shared/format`` → ``src/shared/format.ts``) ; un fichier homonyme (``src/axios.ts`` pour ``axios``) ou un paquet
    déclaré dans ``package.json`` ne sont jamais retenus, donc aucun appel ``axios.create()`` vers le fichier local."""
    db = graph.db
    f = graph.fkey
    graph.add_file("frontend", "src/axios.ts", "typescript",
                   symbols=[SymbolRow(name="create", kind="function", line_start=2, line_end=2)],
                   text="export default function axios() { return 1; }\nexport function create() { return 2; }\n")
    graph.add_file("frontend", "src/shared/index.ts", "typescript",
                   symbols=[SymbolRow(name="throttle", kind="function", line_start=1, line_end=3)])
    graph.add_file("frontend", "src/shared/format.ts", "typescript")
    graph.add_file("frontend", "packages/ui/src/index.ts", "typescript")
    home = node_key("file", graph.add_file(
        "frontend", "src/pages/Home.tsx", "tsx",
        imports=[("shared", ["throttle"], False), ("shared/format", ["fmt"], False), ("@acme/ui", ["Button"], False),
                 ("dayjs", ["dayjs"], False)]))
    resolve_imports(db, graph.workspace)
    imports = _relations(db, "imports")
    # client.ts : ``import axios from "axios"`` → rien, même avec src/axios.ts dans le dépôt
    assert not any(src == f("frontend", "src/api/client.ts") for src, _dst in imports)
    assert not any(dst == f("frontend", "src/axios.ts") for _src, dst in imports)
    link_calls(db, graph.workspace)
    assert not any(dst == node_key("symbol", graph.symbols["create"]) for _src, dst in _relations(db, "calls"))
    # dossiers du dépôt : résolus (index.ts, fichier sous le dossier, paquet d'espace de noms d'un monorepo)
    assert (home, f("frontend", "src/shared/index.ts")) in imports
    assert (home, f("frontend", "src/shared/format.ts")) in imports
    assert (home, f("frontend", "packages/ui/src/index.ts")) in imports
    # paquet déclaré dans package.json : jamais local même si un dossier porte son nom ; ``workspace:`` reste local
    (graph.root / "frontend" / "package.json").write_text(json.dumps({
        "dependencies": {"shared": "^1.0.0", "@acme/ui": "workspace:*", "dayjs": "^1.11.0"}}), encoding="utf-8")
    resolve_imports(db, graph.workspace)
    imports = _relations(db, "imports")
    assert (home, f("frontend", "src/shared/index.ts")) not in imports
    assert (home, f("frontend", "src/shared/format.ts")) not in imports
    assert (home, f("frontend", "packages/ui/src/index.ts")) in imports


def test_resolve_imports_is_idempotent(graph: Graph):
    first = resolve_imports(graph.db, graph.workspace)
    snap = _snapshot(graph.db)
    assert resolve_imports(graph.db, graph.workspace) == first
    assert _snapshot(graph.db) == snap


# --------------------------------------------------------------------------- appariement des endpoints

def test_match_endpoints_rules(graph: Graph):
    db, ws = graph.db, graph.workspace
    for repo in ws.repos:
        learn_base_prefixes(db, repo)
    matched, unmatched_client, unmatched_server = match_endpoints(db, ws)
    assert (matched, unmatched_client, unmatched_server) == (7, 1, 1)
    calls = _relations(db, "calls_endpoint")
    e = graph.ekey
    # http.post(`/orders/${id}/confirm`) ↔ Laravel POST /orders/{order}/confirm : préfixe /api appris → 1.0
    rel = calls[(e("client", "POST", "/orders/${id}/confirm"), e("server", "POST", "/orders/{order}/confirm"))]
    assert rel.confidence == 1.0 and rel.meta["method"] == "POST" and rel.meta["rule"] == "exact"
    assert rel.meta["client_path"] == "/api/orders/:param/confirm"
    # fetch(`${API_BASE}/customers`) ↔ GET /customers : chemin client = préfixe appris + chemin serveur → 0.85
    rel = calls[(e("client", "GET", "${API_BASE}/customers?q=${encodeURIComponent(q)}"), e("server", "GET", "/customers"))]
    assert rel.confidence == 0.85 and rel.meta["rule"] == "learned_prefix" and rel.meta["prefix"] == "/api"
    # Http::post('http://localhost:3001/notifications') ↔ Express POST /notifications → 1.0, entre dépôts
    rel = calls[(e("client", "POST", "http://localhost:3001/notifications"), e("server", "POST", "/notifications"))]
    assert rel.confidence == 1.0
    # fetch('http://localhost:3001/notifications/' + userId) ↔ GET /notifications/:userId (pas DELETE)
    assert (e("client", "GET", "http://localhost:3001/notifications/${userId}"),
            e("server", "GET", "/notifications/:userId")) in calls
    assert (e("client", "GET", "http://localhost:3001/notifications/${userId}"),
            e("server", "DELETE", "/notifications/:id")) not in calls
    # méthodes contradictoires explicites (DELETE client vs GET serveur) → aucune arête
    assert not any(src == e("client", "DELETE", "/orders/${id}") for src, _dst in calls)
    # ambiguïté : deux routes GET /status ex æquo → 1 ÷ √2 chacune
    tied = [r for (src, _dst), r in calls.items() if src == e("client", "GET", "/status")]
    assert len(tied) == 2
    for r in tied:
        assert r.confidence == round(1 / math.sqrt(2), 4) and r.meta["tied"] == 2
    # jamais plus de 3 candidats par appel, jamais sous 0.4
    per_client: Dict[str, int] = {}
    for (src, _dst), r in calls.items():
        per_client[src] = per_client.get(src, 0) + 1
        assert r.confidence >= 0.4
    assert max(per_client.values()) <= 3


def test_match_endpoints_generic_prefix_suffix_and_param_rules(db: Database, tmp_path: Path):
    ws = WorkspaceConfig(id="ws", name="t", repos=[RepoConfig(id="r", name="r", path=str(tmp_path), kind="other")])
    db.upsert_repo(RepoRow(id="r", name="r", path=str(tmp_path)))
    fid = db.upsert_file(FileRow(repo_id="r", path="a.ts", language="typescript", kind="code"))
    clients = [Graph.client("GET", "/api/v1/items", "a"),            # préfixe générique → 0.75
               Graph.client("GET", "/x/y/reports/daily", "b"),       # se termine par le chemin serveur → 0.65
               Graph.client("ANY", "/widgets/special", "c"),          # :param absorbe « special » (×0.9 méthode inconnue)
               Graph.client("GET", "/nowhere", "d")]
    servers = [Graph.server("GET", "/items", framework="express"),
               Graph.server("GET", "/reports/daily", framework="express"),
               Graph.server("POST", "/widgets/:id", framework="express")]
    ids = db.replace_file_artifacts(fid, endpoints=clients + servers)
    cid, sid = ids["endpoints"][:4], ids["endpoints"][4:]
    assert match_endpoints(db, ws) == (3, 1, 0)
    calls = _relations(db, "calls_endpoint")
    assert calls[(node_key("endpoint", cid[0]), node_key("endpoint", sid[0]))].confidence == 0.75
    assert calls[(node_key("endpoint", cid[0]), node_key("endpoint", sid[0]))].meta["rule"] == "generic_prefix"
    assert calls[(node_key("endpoint", cid[1]), node_key("endpoint", sid[1]))].confidence == 0.65
    assert calls[(node_key("endpoint", cid[2]), node_key("endpoint", sid[2]))].confidence == 0.45
    assert calls[(node_key("endpoint", cid[2]), node_key("endpoint", sid[2]))].meta["rule"] == "param"
    assert not any(src == node_key("endpoint", cid[3]) for src, _dst in calls)


# --------------------------------------------------------------------------- modèles

def test_link_models_same_entity_and_uses_model(graph: Graph):
    db, ws = graph.db, graph.workspace
    resolve_imports(db, ws)
    same_count, uses_count = link_models(db, ws)
    same = _relations(db, "same_entity")
    m = graph.mkey
    order = [m("eloquent", "Order"), m("migration", "orders"), m("sql_table", "orders"), m("ts_interface", "Order")]
    # Order ×4 : les 6 paires sont reliées, confiance graduée
    for i in range(4):
        for j in range(i + 1, 4):
            assert _undirected(same, order[i], order[j]) is not None, (order[i], order[j])
    assert _undirected(same, m("eloquent", "Order"), m("migration", "orders")).confidence == 0.9
    assert _undirected(same, m("migration", "orders"), m("sql_table", "orders")).confidence == 0.9
    assert _undirected(same, m("eloquent", "Order"), m("sql_table", "orders")).confidence == 0.9
    assert _undirected(same, m("ts_interface", "Order"), m("eloquent", "Order")).confidence >= 0.7
    # nom de classe ↔ table par snake_case pluriel : OrderLine ↔ order_lines → 0.8
    assert _undirected(same, m("ts_interface", "OrderLine"), m("migration", "order_lines")).confidence == 0.8
    assert _undirected(same, m("ts_interface", "OrderLine"), m("sql_table", "order_lines")).confidence == 0.8
    # interface TS de même nom qu'un modèle serveur sans table → 0.7
    assert _undirected(same, m("ts_interface", "Invoice"), m("eloquent", "Invoice")).confidence == 0.7
    # jamais de lien entre entités différentes, jamais de boucle
    assert _undirected(same, m("eloquent", "Order"), m("eloquent", "Customer")) is None
    assert all(src != dst for src, dst in same)
    assert same_count == len(same) and uses_count == len(_relations(db, "uses_model"))

    uses = _relations(db, "uses_model")
    f = graph.fkey
    # import du fichier définissant le modèle (en respectant les noms importés)
    assert uses[(f("frontend", "src/components/OrderList.tsx"), m("ts_interface", "Order"))].confidence == 0.6
    assert (f("frontend", "src/components/OrderList.tsx"), m("ts_interface", "Customer")) not in uses
    assert (f("backend", "app/Services/OrderService.php"), m("eloquent", "Order")) in uses
    # gestionnaire d'endpoint citant le nom du modèle (OrderController@confirm → Order)
    assert (f("backend", "routes/api.php"), m("eloquent", "Order")) in uses
    # un fichier n'utilise pas le modèle qu'il définit
    assert (f("backend", "app/Models/Order.php"), m("eloquent", "Order")) not in uses


def test_link_models_name_rule_never_crosses_repositories(graph: Graph):
    """Non-régression : un fichier n'« utilise » par son nom qu'un modèle de son propre dépôt.

    Avant la correction, ``app/Models/Order.php`` (qui définit l'entité côté serveur) était dit
    « utiliser » l'interface TS ``Order`` du frontend parce que son propre nom de classe
    correspondait, et ``routes/api.php`` héritait de ces liens par la règle ``handler``.
    """
    db, ws = graph.db, graph.workspace
    # classe homonyme d'un autre dépôt (OrderController en Java) : la règle « handler » ne doit pas en hériter
    graph.add_file("java", "src/main/java/com/acme/model/Shipment.java", "java",
                   symbols=[SymbolRow(name="Shipment", kind="class", line_start=3, line_end=20)],
                   models=[ModelRow(name="Shipment", kind="jpa_entity", table_name="shipments", line=3,
                                    fields=[{"name": "tracking"}])])
    graph.add_file("java", "src/main/java/com/acme/web/OrderController.java", "java",
                   symbols=[SymbolRow(name="OrderController", kind="class", line_start=5, line_end=40),
                            SymbolRow(name="index", kind="method", parent="OrderController", line_start=10, line_end=20,
                                      signature="index(Shipment shipment)")])
    resolve_imports(db, ws)
    link_models(db, ws)
    uses = _relations(db, "uses_model")
    f, m = graph.fkey, graph.mkey
    # le fichier qui définit l'entité serveur n'utilise pas l'interface TS homonyme du frontend (et réciproquement)
    assert (f("backend", "app/Models/Order.php"), m("ts_interface", "Order")) not in uses
    assert (f("backend", "app/Models/Customer.php"), m("ts_interface", "Customer")) not in uses
    assert (f("frontend", "src/api/client.ts"), m("eloquent", "Order")) not in uses
    assert (f("frontend", "src/api/client.ts"), m("eloquent", "Customer")) not in uses
    # les fichiers du backend qui citent « Order » utilisent le modèle eloquent, jamais l'interface TS
    assert uses[(f("backend", "app/Services/OrderService.php"), m("eloquent", "Order"))].meta["rule"] == "import"
    assert (f("backend", "app/Services/OrderService.php"), m("ts_interface", "Order")) not in uses
    assert (f("backend", "app/Http/Controllers/OrderController.php"), m("ts_interface", "Order")) not in uses
    # « OrderList » cite « Order » : seule l'interface TS du même dépôt est retenue
    assert (f("frontend", "src/components/OrderList.tsx"), m("ts_interface", "Order")) in uses
    assert (f("frontend", "src/components/OrderList.tsx"), m("eloquent", "Order")) not in uses
    # règle « handler » : routes/api.php hérite des usages d'OrderController.php (backend), pas de la classe Java
    assert uses[(f("backend", "routes/api.php"), m("eloquent", "Order"))].meta["rule"] == "handler"
    assert (f("backend", "routes/api.php"), m("ts_interface", "Order")) not in uses
    assert (f("backend", "routes/api.php"), m("jpa_entity", "Shipment")) not in uses
    assert uses[(f("java", "src/main/java/com/acme/web/OrderController.java"), m("jpa_entity", "Shipment"))].meta["rule"] == "name"
    # invariant : aucune relation uses_model ne traverse les dépôts
    models_file = {int(row.id): int(row.file_id) for row in db.list_models() if row.id is not None}
    for src, dst in uses:
        a = db.get_file_by_id(int(src[len("file:"):]))
        b = db.get_file_by_id(models_file[int(dst[len("model:"):])])
        assert a is not None and b is not None and a.repo_id == b.repo_id, (src, dst)


def test_link_models_monorepo_prefers_own_language_and_ignores_own_names(db: Database, tmp_path: Path):
    """Monodépôt (frontend et backend dans un seul dépôt) : parmi les modèles homonymes, un fichier
    « utilise » ceux de sa famille de langage ; il n'utilise jamais un homonyme d'un modèle qu'il définit."""
    root = tmp_path / "mono"
    root.mkdir()
    db.upsert_repo(RepoRow(id="mono", name="mono", path=str(root), kind="other"))
    ws = WorkspaceConfig(id="ws", name="mono", repos=[RepoConfig(id="mono", name="mono", path=str(root), kind="other")])

    def add(path: str, language: str, symbols: Sequence[SymbolRow] = (), models: Sequence[ModelRow] = ()) -> str:
        fid = db.upsert_file(FileRow(repo_id="mono", path=path, language=language, kind="code", size=10, mtime=1.0,
                                     sha1="sha-" + path))
        db.replace_file_artifacts(fid, symbols=list(symbols), endpoints=[], models=list(models), chunks=[])
        return node_key("file", fid)

    eloquent = ModelRow(name="Order", kind="eloquent", table_name="orders", line=9, fields=[{"name": "status"}])
    ts = ModelRow(name="Order", kind="ts_interface", line=7, fields=[{"name": "status"}])
    order_php = add("backend/app/Models/Order.php", "php",
                    symbols=[SymbolRow(name="Order", kind="class", line_start=9, line_end=28)], models=[eloquent])
    service = add("backend/app/Services/OrderService.php", "php",
                  symbols=[SymbolRow(name="OrderService", kind="class", line_start=8, line_end=30),
                           SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=10, line_end=24,
                                     signature="confirm(Order $order): Order")])
    client = add("frontend/src/api/client.ts", "typescript",
                 symbols=[SymbolRow(name="Order", kind="interface", line_start=7, line_end=13)], models=[ts])
    view = add("frontend/src/components/OrderList.vue", "vue",
               symbols=[SymbolRow(name="OrderList", kind="component", line_start=1, line_end=20,
                                  signature="OrderList(props: { orders: Order[] })")])
    script = add("tools/report.py", "python",
                 symbols=[SymbolRow(name="export_orders", kind="function", line_start=1, line_end=9,
                                    signature="export_orders(order: Order) -> str")])
    link_models(db, ws)
    uses = _relations(db, "uses_model")
    m_eloquent, m_ts = node_key("model", int(eloquent.id)), node_key("model", int(ts.id))
    # PHP → modèle eloquent ; Vue → interface TS ; jamais l'homonyme de l'autre famille
    assert uses[(service, m_eloquent)].meta["rule"] == "name" and (service, m_ts) not in uses
    assert uses[(view, m_ts)].meta["rule"] == "name" and (view, m_eloquent) not in uses
    # un fichier n'utilise pas l'homonyme d'un modèle qu'il définit
    assert (order_php, m_ts) not in uses and (order_php, m_eloquent) not in uses
    assert (client, m_eloquent) not in uses and (client, m_ts) not in uses
    # sans modèle de sa famille de langage, un fichier reste relié aux homonymes du dépôt (ambiguïté assumée)
    assert (script, m_eloquent) in uses and (script, m_ts) in uses


# --------------------------------------------------------------------------- documents et notes

def test_link_documents_documented_by_and_mentions(graph: Graph):
    db = graph.db
    linked = link_documents(db, graph.workspace)
    assert linked == 3
    doc_dict = node_key("document", graph.documents["dictionnaire"])
    doc_spec = node_key("document", graph.documents["spec"])
    note = node_key("note", graph.notes["reunion"])
    documented = _relations(db, "documented_by")
    # dictionnaire ↔ table orders, OrderService (classe et méthode qualifiée)
    rel = documented[("table:orders", doc_dict)]
    assert rel.confidence == 0.9 and rel.meta["mentions"] == 1
    assert ("table:order_lines", doc_dict) in documented
    assert (graph.skey("OrderService"), doc_dict) in documented
    assert (graph.skey("OrderService.confirm"), doc_dict) in documented
    assert (graph.mkey("migration", "orders"), doc_dict) in documented
    # spécification ↔ confirmOrder et POST /api/orders/{id}/confirm (route serveur)
    assert (graph.skey("confirmOrder"), doc_spec) in documented
    assert (graph.ekey("server", "POST", "/orders/{order}/confirm"), doc_spec) in documented
    # note ↔ OrderService::confirm ; l'identifiant disparu ne crée rien
    assert (graph.skey("OrderService.confirm"), note) in documented
    assert not any(dst == note and "archiver" in src.lower() for src, dst in documented)
    # mentions : document → fichiers dont des éléments sont cités (0.7), jamais vers un module
    mentions = _relations(db, "mentions")
    assert mentions[(doc_dict, graph.fkey("backend", "app/Services/OrderService.php"))].confidence == 0.7
    assert (note, graph.fkey("backend", "app/Services/OrderService.php")) in mentions
    assert all(src in (doc_dict, doc_spec, note) for src, _dst in mentions)
    # pas de doublon (src, dst, kind) et idempotence
    snap = _snapshot(db)
    assert link_documents(db, graph.workspace) == linked
    assert _snapshot(db) == snap


def test_link_documents_handles_repo_doc_files(graph: Graph):
    db = graph.db
    doc_fid = graph.add_file("backend", "README.md", "markdown", kind="doc",
                             text="# Backend\n\nLa confirmation passe par `OrderService::confirm` sur la table `orders`.\n")
    link_documents(db, graph.workspace)
    documented = _relations(db, "documented_by")
    assert ("table:orders", node_key("file", doc_fid)) in documented
    assert (graph.skey("OrderService.confirm"), node_key("file", doc_fid)) in documented
    mentions = _relations(db, "mentions")
    assert (node_key("file", doc_fid), graph.fkey("backend", "app/Services/OrderService.php")) in mentions
    assert (node_key("file", doc_fid), node_key("file", doc_fid)) not in mentions


# --------------------------------------------------------------------------- mentions et dérive

def test_extract_identifiers():
    found = extract_identifiers("La classe `OrderService` et la table orders ; appelle POST /api/orders/{id}/confirm ; "
                                "champ customer_id ; voir src/api/client.ts et OrderLine. Fin.")
    assert found == ["OrderService", "/api/orders/{id}/confirm", "customer_id", "src/api/client.ts", "OrderLine"]
    assert extract_identifiers("") == [] and extract_identifiers("Bonjour tout le monde.") == []
    assert "Date" not in extract_identifiers("La Date de la commande")  # CamelCase < 5 ou simple majuscule : non


def test_document_mentions_and_is_stale(graph: Graph):
    db = graph.db
    note_id = graph.notes["reunion"]
    mentions = document_mentions(db, "note", note_id)
    by_name = {m["name"]: m for m in mentions}
    assert by_name["OrderService.confirm"]["kind"] == "symbol" and by_name["OrderService.confirm"]["exists"] is True
    assert by_name["OrderArchiver"]["exists"] is False and by_name["OrderArchiver"]["kind"] == "unknown"
    assert by_name["OrderArchiver"]["count"] == 1
    assert all(set(m) >= {"kind", "name", "exists", "count"} for m in mentions)
    assert all(m["kind"] in ("symbol", "model", "table", "endpoint", "field", "file", "unknown") for m in mentions)
    # une valeur en accents graves (`confirmed`) n'est pas « du code » disparu
    assert by_name["confirmed"]["exists"] is False and by_name["confirmed"]["code_like"] is False
    # la route citée avec un point final est reconnue
    assert any(m["kind"] == "endpoint" and m["exists"] for m in mentions)
    assert is_stale(db, "note", note_id) is True
    # le dictionnaire ne cite que des éléments présents : pas de dérive
    assert is_stale(db, "document", graph.documents["dictionnaire"]) is False
    assert all(m["exists"] for m in document_mentions(db, "document", graph.documents["dictionnaire"]))
    # un texte ordinaire sans aucune mention existante n'est jamais signalé
    plain = db.add_note(NoteRow(title="Pause", body="On a parlé de `MonProjetFabuleux` et de café.", note_date="2026-09-02"))
    assert is_stale(db, "note", plain) is False
    # un index déjà construit peut être réutilisé
    idx = NameIndex.build(db)
    assert document_mentions(db, "note", note_id, index=idx) == mentions
    assert document_mentions(db, "note", 999999) == [] and is_stale(db, "document", 999999) is False


def test_is_stale_after_symbol_disappears(linked: Graph):
    db = linked.db
    doc_id = linked.documents["spec"]
    assert is_stale(db, "document", doc_id) is False
    # confirmOrder disparaît du frontend (réindexation du fichier sans ce symbole)
    fid = linked.file("frontend", "src/api/client.ts")
    db.replace_file_artifacts(fid, symbols=[SymbolRow(name="API_BASE", kind="constant", line_start=3, line_end=3)],
                              endpoints=[], models=[], chunks=[])
    mentions = {m["name"]: m for m in document_mentions(db, "document", doc_id)}
    assert mentions["confirmOrder"]["exists"] is False
    assert is_stale(db, "document", doc_id) is True


# --------------------------------------------------------------------------- explorateur

def test_neighbors_graph_depth_and_labels(linked: Graph):
    db = linked.db
    g1 = neighbors_graph(db, "table:orders", depth=1)
    keys1 = {n["key"] for n in g1["nodes"]}
    assert "table:orders" in keys1
    assert {linked.mkey("eloquent", "Order"), linked.mkey("migration", "orders"), linked.mkey("sql_table", "orders")} <= keys1
    assert node_key("document", linked.documents["dictionnaire"]) in keys1
    assert linked.mkey("ts_interface", "Order") not in keys1  # à deux sauts seulement (same_entity)
    g2 = neighbors_graph(db, "table:orders", depth=2, limit=200)
    keys2 = {n["key"] for n in g2["nodes"]}
    assert keys1 < keys2 and linked.mkey("ts_interface", "Order") in keys2
    assert linked.fkey("backend", "app/Models/Order.php") in keys2
    labels = {n["key"]: n["label"] for n in g2["nodes"]}
    assert labels["table:orders"] == "table orders"
    assert labels[linked.mkey("eloquent", "Order")] == "Order"
    assert labels[node_key("document", linked.documents["dictionnaire"])] == "Dictionnaire de données"
    for n in g2["nodes"]:
        assert set(n) >= {"key", "type", "label", "path", "line", "repo_id"}
    model_node = next(n for n in g2["nodes"] if n["key"] == linked.mkey("eloquent", "Order"))
    assert model_node["path"] == "app/Models/Order.php" and model_node["line"] == 9 and model_node["repo_id"] == "backend"
    # arêtes : uniquement entre noeuds conservés, avec genre et confiance ; jamais imports_raw
    for e in g2["edges"]:
        assert e["src"] in keys2 and e["dst"] in keys2 and e["kind"] != "imports_raw" and 0 < e["confidence"] <= 1
    kinds = {e["kind"] for e in g2["edges"]}
    assert {"defines_table", "same_entity", "documented_by"} <= kinds
    # borne : limit et profondeur 0
    g_small = neighbors_graph(db, "table:orders", depth=2, limit=3)
    assert len(g_small["nodes"]) <= 3 and g_small["truncated"] is True
    g0 = neighbors_graph(db, "table:orders", depth=0)
    assert [n["key"] for n in g0["nodes"]] == ["table:orders"] and g0["edges"] == []
    # un appel client remonte jusqu'à la route serveur puis à sa documentation
    g = neighbors_graph(db, linked.ekey("client", "POST", "/orders/${id}/confirm"), depth=2)
    keys = {n["key"] for n in g["nodes"]}
    assert linked.ekey("server", "POST", "/orders/{order}/confirm") in keys
    assert node_key("document", linked.documents["spec"]) in keys
    labels = {n["key"]: n["label"] for n in g["nodes"]}
    assert labels[linked.ekey("server", "POST", "/orders/{order}/confirm")] == "POST /api/orders/:param/confirm"
    # noeud inconnu ou mal formé : graphe vide, jamais d'exception
    assert neighbors_graph(db, "symbol:999999")["nodes"] == [] and neighbors_graph(db, "n'importe quoi")["nodes"] == []


def test_describe_relation_french_sentences(linked: Graph):
    db = linked.db
    imports = _relations(db, "imports")
    rel = imports[(linked.fkey("frontend", "src/components/OrderList.tsx"), linked.fkey("frontend", "src/api/client.ts"))]
    assert describe_relation(rel, db) == "importe src/api/client.ts"
    calls = _relations(db, "calls_endpoint")
    rel = calls[(linked.ekey("client", "POST", "/orders/${id}/confirm"), linked.ekey("server", "POST", "/orders/{order}/confirm"))]
    assert describe_relation(rel, db) == "appelle POST /api/orders/:param/confirm"
    rel = calls[(linked.ekey("client", "GET", "${API_BASE}/customers?q=${encodeURIComponent(q)}"),
                 linked.ekey("server", "GET", "/customers"))]
    assert describe_relation(rel, db) == "appelle GET /customers (0,85)"
    documented = _relations(db, "documented_by")
    rel = documented[("table:orders", node_key("document", linked.documents["dictionnaire"]))]
    assert describe_relation(rel, db) == "documente la table orders"
    rel = documented[(linked.skey("OrderService.confirm"), node_key("note", linked.notes["reunion"]))]
    assert describe_relation(rel, db) == "documente le symbole OrderService.confirm"
    same = _relations(db, "same_entity")
    rel = _undirected(same, linked.mkey("eloquent", "Order"), linked.mkey("migration", "orders"))
    assert describe_relation(rel, db).startswith("même entité que ") and describe_relation(rel, db).endswith(" (0,9)")
    uses = _relations(db, "uses_model")
    rel = uses[(linked.fkey("backend", "app/Services/OrderService.php"), linked.mkey("eloquent", "Order"))]
    assert describe_relation(rel, db) == "utilise le modèle Order (0,6)"
    defines = _relations(db, "defines_table")
    assert describe_relation(defines[(linked.mkey("sql_table", "orders"), "table:orders")], db) == "définit la table orders"
    # relation vers un noeud disparu : phrase de repli, jamais d'exception
    ghost = RelationRow(src="file:999999", dst="file:999998", kind="imports", confidence=0.5)
    assert "file:999998" in describe_relation(ghost, db)


# --------------------------------------------------------------------------- link_all

def test_link_all_is_idempotent_and_reports_stats(graph: Graph):
    db, ws = graph.db, graph.workspace
    messages: List[str] = []
    stats = link_all(db, ws, progress=messages.append)
    assert isinstance(stats, LinkStats)
    assert stats.imports_resolved == 17 and stats.imports_unresolved == 11
    assert (stats.endpoints_matched, stats.endpoints_unmatched_client, stats.endpoints_unmatched_server) == (7, 1, 1)
    assert stats.models_linked > 0 and stats.model_uses > 0 and stats.documents_linked == 3
    assert stats.base_prefixes["frontend"] == ["/api"] and stats.base_prefixes["backend"] == ["/api"]
    assert stats.errors == [] and stats.duration_s >= 0
    derived = {r["kind"]: int(r["n"]) for r in db.query("SELECT kind, count(*) AS n FROM relations GROUP BY kind")}
    assert stats.total == sum(n for k, n in derived.items() if k in DERIVED_KINDS) > 0
    assert {"imports", "calls_endpoint", "same_entity", "uses_model", "documented_by"} <= set(derived)
    data = stats.to_dict()
    for key in ("imports_resolved", "imports_unresolved", "endpoints_matched", "endpoints_unmatched_client",
                "endpoints_unmatched_server", "models_linked", "model_uses", "documents_linked", "base_prefixes",
                "duration_s", "total"):
        assert key in data, key
    assert data["relations"] == data["total"] == stats.relations
    assert messages and all(isinstance(m, str) and m for m in messages)

    # seconde passe : même graphe, aucun doublon, relations brutes intactes
    before = _snapshot(db)
    raw = {k: n for k, n in derived.items() if k in ("imports_raw", "defines_table", "handles")}
    again = link_all(db, ws)
    assert _snapshot(db) == before
    assert again.total == stats.total
    after = {r["kind"]: int(r["n"]) for r in db.query("SELECT kind, count(*) AS n FROM relations GROUP BY kind")}
    assert {k: after.get(k) for k in raw} == raw
    dupes = db.query("SELECT src, dst, kind, count(*) AS n FROM relations GROUP BY src, dst, kind HAVING n > 1")
    assert dupes == []


def test_link_all_recomputes_after_changes(linked: Graph):
    db, ws = linked.db, linked.workspace
    # le document disparaît : ses relations dérivées aussi, les autres restent
    doc = linked.documents["spec"]
    assert db.delete_document(doc)
    stats = link_all(db, ws)
    assert stats.documents_linked == 2
    assert not any(dst == node_key("document", doc) for src, dst in _relations(db, "documented_by"))
    assert ("table:orders", node_key("document", linked.documents["dictionnaire"])) in _relations(db, "documented_by")
    # un dépôt désactivé n'est plus analysé pour ses préfixes mais le graphe reste cohérent
    ws.repo_by_id("spa").enabled = False
    stats = link_all(db, ws)
    assert "spa" not in stats.base_prefixes and stats.errors == []


def test_link_all_on_empty_database(db: Database, tmp_path: Path):
    ws = WorkspaceConfig(id="ws", name="vide", repos=[])
    stats = link_all(db, ws)
    assert stats.total == 0 and stats.errors == [] and stats.to_dict()["base_prefixes"] == {}


# --------------------------------------------------------------------------- gestionnaires et appels

def test_link_handlers_completes_cross_file_handles_and_keeps_raw_ones(graph: Graph):
    db = graph.db
    route = graph.ekey("server", "POST", "/orders/{order}/confirm")
    customers = graph.ekey("server", "GET", "/customers")
    # relation brute posée par l'indexeur (gestionnaire du même fichier) : conservée telle quelle
    raw = RelationRow(src=customers, dst=graph.skey("OrderController"), kind="handles", confidence=1.0,
                      meta={"handler": "CustomerController@index"})
    db.add_relations([raw])
    n = link_handlers(db, graph.workspace)
    handles = _relations(db, "handles")
    rel = handles[(route, graph.skey("OrderController.confirm"))]
    assert rel.confidence == 0.9 and rel.meta == {"handler": "OrderController@confirm", "rule": "linker", "ambiguous": False}
    assert handles[(customers, graph.skey("OrderController"))].confidence == 1.0  # brute intacte, pas de doublon
    # closures, routes Express sans gestionnaire nommé, contrôleur absent de l'index : rien
    assert not any(src == graph.ekey("server", "GET", "/status") for src, _dst in handles)
    assert not any(src == graph.ekey("server", "POST", "/notifications") for src, _dst in handles)
    # idempotent : recalculée à l'identique (les relations du linker sont reconstruites à chaque passe)
    assert n == 1 and link_handlers(db, graph.workspace) == n and _relations(db, "handles").keys() == handles.keys()
    assert describe_relation(rel, db) == "route gérée par OrderController.confirm"


def test_link_handlers_recomputes_when_an_ambiguity_disappears(graph: Graph):
    """Non-régression : les relations ``handles`` du linker sont recalculées à chaque passe. Quand l'homonyme qui
    rendait un gestionnaire ambigu disparaît, la relation restante repasse de 0.6 (``ambiguous``) à 0.9 sans que la
    route ni le contrôleur restant n'aient changé ; les relations brutes de l'indexeur restent intactes."""
    db = graph.db
    main_confirm, main_class = graph.skey("OrderController.confirm"), graph.skey("OrderController")
    admin = graph.add_file("backend", "app/Http/Controllers/Admin/OrderController.php", "php",
                           symbols=[SymbolRow(name="OrderController", kind="class", line_start=5, line_end=12),
                                    SymbolRow(name="confirm", kind="method", parent="OrderController", line_start=7,
                                              line_end=10)])
    admin_confirm = node_key("symbol", next(s.id for s in db.list_symbols(file_id=admin) if s.name == "confirm"))
    graph.add_file("backend", "routes/admin.php", "php",
                   endpoints=[graph.server("POST", "/admin/orders/{order}/confirm", "OrderController@confirm", line=5)])
    route = graph.ekey("server", "POST", "/admin/orders/{order}/confirm")
    customers = graph.ekey("server", "GET", "/customers")
    raw = RelationRow(src=customers, dst=main_class, kind="handles", confidence=1.0,
                      meta={"handler": "CustomerController@index"})
    db.add_relations([raw])
    resolve_imports(db, graph.workspace)
    n = link_handlers(db, graph.workspace)
    handles = _relations(db, "handles")
    assert handles[(route, main_confirm)].confidence == 0.6 and handles[(route, main_confirm)].meta["ambiguous"] is True
    assert handles[(route, admin_confirm)].confidence == 0.6
    assert link_handlers(db, graph.workspace) == n and _relations(db, "handles").keys() == handles.keys()
    # l'homonyme disparaît : ni la route ni OrderController.confirm ne changent, la relation est pourtant recalculée
    assert db.delete_file(admin)
    link_handlers(db, graph.workspace)
    handles = _relations(db, "handles")
    rel = handles[(route, main_confirm)]
    assert rel.confidence == 0.9 and rel.meta == {"handler": "OrderController@confirm", "rule": "linker", "ambiguous": False}
    assert (route, admin_confirm) not in handles
    kept = handles[(customers, main_class)]
    assert kept.confidence == 1.0 and kept.meta == {"handler": "CustomerController@index"}


def test_link_calls_derives_calls_from_indexed_text(graph: Graph):
    db = graph.db
    controller = "<?php\nnamespace App\\Http\\Controllers;\n\nuse App\\Models\\Order;\nuse App\\Services\\OrderService;\n\n" \
                 "class OrderController\n{\n    public function __construct(private OrderService $orders) {}\n\n" \
                 "    public function confirm(Request $request, Order $order)\n    {\n" \
                 "        if ($order->status !== 'draft') { abort(422); }\n" \
                 "        return $this->orders->confirm($order);\n    }\n}\n"
    fid = graph.file("backend", "app/Http/Controllers/OrderController.php")
    header = "# backend/app/Http/Controllers/OrderController.php L1-16\n"  # en-tête posé par l'indexeur
    db.replace_chunks("file", fid, [ChunkRow(text=header + controller, line_start=1, line_end=16,
                                             meta={"body_offset": len(header)})])
    db.replace_file_artifacts(fid, symbols=[
        SymbolRow(name="OrderController", kind="class", line_start=7, line_end=16),
        SymbolRow(name="__construct", kind="method", parent="OrderController", line_start=9, line_end=9),
        SymbolRow(name="confirm", kind="method", parent="OrderController", line_start=11, line_end=15),
    ], chunks=db.chunks_for("file", fid))
    graph.add_file("notifier", "src/handlers.js", "javascript",
                   symbols=[SymbolRow(name="notify", kind="function", line_start=3, line_end=5)],
                   text='const { add } = require("./store");\n\nfunction notify(userId, message) {\n'
                        '  return add(userId, message);\n}\n\nmodule.exports = { notify };\n',
                   imports=[("./store", ["add"], True)])
    resolve_imports(db, graph.workspace)
    n = link_calls(db, graph.workspace)
    calls = _relations(db, "calls")
    ctl_confirm = node_key("symbol", next(s.id for s in db.list_symbols(file_id=fid) if s.name == "confirm"))
    rel = calls[(ctl_confirm, graph.skey("OrderService.confirm"))]
    assert rel.confidence == 0.8 and rel.meta["rule"] == "import" and rel.meta["name"] == "confirm" and rel.meta["line"] == 14
    rel = calls[(node_key("symbol", graph.symbols["notify"]), graph.skey("add"))]
    assert rel.confidence == 0.8 and rel.meta["rule"] == "import"
    # ni ``abort(`` (inconnu), ni ``if (`` (mot-clé), ni la ligne de définition ``confirm(`` elle-même
    assert all(r.meta["name"] not in ("abort", "if", "__construct") for r in calls.values())
    assert not any(dst == ctl_confirm for _src, dst in calls)
    assert n == len(calls) >= 2 and describe_relation(rel, db) == "appelle add (0,8)"
    # dérivé : recalculé à l'identique, puis supprimé si le texte ne cite plus l'appel
    assert link_calls(db, graph.workspace) == n and _relations(db, "calls").keys() == calls.keys()
    db.replace_chunks("file", fid, [])
    link_calls(db, graph.workspace)
    assert (ctl_confirm, graph.skey("OrderService.confirm")) not in _relations(db, "calls")


def test_link_calls_prefers_the_current_class_for_this_receiver(graph: Graph):
    """Non-régression : ``$this->index()`` dans la classe A vise ``A.index`` (0.7, ``same_class``) même si plus de
    trois symboles ``index`` sont visibles (classe B du même fichier, contrôleurs importés) ; ``this.index()`` en
    TypeScript et ``self.index()`` en Python suivent la même règle."""
    db = graph.db
    for name in ("CustomerController", "ProductController"):
        graph.add_file("backend", "app/Http/Controllers/%s.php" % name, "php",
                       symbols=[SymbolRow(name=name, kind="class", line_start=5, line_end=12),
                                SymbolRow(name="index", kind="method", parent=name, line_start=7, line_end=10)])
    mixed = "<?php\nnamespace App\\Services;\n\nuse App\\Http\\Controllers\\CustomerController;\n" \
            "use App\\Http\\Controllers\\ProductController;\n\nclass A\n{\n    public function index()\n    {\n" \
            "        return 'a';\n    }\n\n    public function run()\n    {\n        return $this->index();\n    }\n}\n\n" \
            "class B\n{\n    public function index()\n    {\n        return 'b';\n    }\n}\n"
    graph.add_file("backend", "app/Services/Mixed.php", "php",
                   symbols=[SymbolRow(name="A", kind="class", line_start=7, line_end=18),
                            SymbolRow(name="index", kind="method", parent="A", line_start=9, line_end=12),
                            SymbolRow(name="run", kind="method", parent="A", line_start=14, line_end=17),
                            SymbolRow(name="B", kind="class", line_start=20, line_end=26),
                            SymbolRow(name="index", kind="method", parent="B", line_start=22, line_end=25)],
                   text=mixed,
                   imports=[("App\\Http\\Controllers\\CustomerController", ["CustomerController"], False),
                            ("App\\Http\\Controllers\\ProductController", ["ProductController"], False)])
    graph.add_file("frontend", "src/services/svc.ts", "typescript",
                   symbols=[SymbolRow(name="Svc", kind="class", line_start=1, line_end=4),
                            SymbolRow(name="index", kind="method", parent="Svc", line_start=2, line_end=2),
                            SymbolRow(name="run", kind="method", parent="Svc", line_start=3, line_end=3),
                            SymbolRow(name="Other", kind="class", line_start=5, line_end=7),
                            SymbolRow(name="index", kind="method", parent="Other", line_start=6, line_end=6)],
                   text="export class Svc {\n  index() { return 1; }\n  run() { return this.index(); }\n}\n"
                        "export class Other {\n  index() { return 2; }\n}\n")
    graph.add_file("tools", "jobs.py", "python",
                   symbols=[SymbolRow(name="Job", kind="class", line_start=1, line_end=6),
                            SymbolRow(name="index", kind="method", parent="Job", line_start=2, line_end=3),
                            SymbolRow(name="run", kind="method", parent="Job", line_start=5, line_end=6),
                            SymbolRow(name="index", kind="function", line_start=9, line_end=10)],
                   text="class Job:\n    def index(self):\n        return 1\n\n    def run(self):\n        return self.index()\n\n\n"
                        "def index():\n    return 2\n")
    resolve_imports(db, graph.workspace)
    link_calls(db, graph.workspace)
    calls = _relations(db, "calls")
    run = node_key("symbol", graph.symbols["A.run"])
    rel = calls[(run, node_key("symbol", graph.symbols["A.index"]))]
    assert rel.confidence == 0.7 and rel.meta == {"rule": "same_class", "name": "index", "line": 16}
    assert [dst for src, dst in calls if src == run] == [node_key("symbol", graph.symbols["A.index"])]
    for caller, target in (("Svc.run", "Svc.index"), ("Job.run", "Job.index")):
        src = node_key("symbol", graph.symbols[caller])
        assert calls[(src, node_key("symbol", graph.symbols[target]))].meta["rule"] == "same_class"
        assert [dst for s, dst in calls if s == src] == [node_key("symbol", graph.symbols[target])]
    assert describe_relation(rel, db) == "appelle A.index (0,7)"


# --------------------------------------------------------------------------- intégration : fixture réelle

@pytest.fixture
def develop_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("DEVELOP_HOME", str(home))
    return home


@pytest.fixture
def project(tmp_path: Path, sample_project: Path) -> Path:
    dst = tmp_path / "sample_project"
    shutil.copytree(sample_project, dst)
    return dst


def _five_repos(project: Path) -> WorkspaceConfig:
    return WorkspaceConfig(id="ws", name="Projet", repos=[
        RepoConfig(id="frontend", name="frontend", path=str(project / "frontend"), kind="frontend"),
        RepoConfig(id="backend", name="backend", path=str(project / "backend"), kind="backend"),
        RepoConfig(id="notifier", name="notifier", path=str(project / "services" / "notifier"), kind="backend"),
        RepoConfig(id="docs", name="docs", path=str(project / "docs"), kind="docs"),
        RepoConfig(id="database", name="database", path=str(project / "database"), kind="database"),
    ])


def _endpoint(db: Database, side: str, method: str, path: str, repo_id: str) -> EndpointRow:
    for ep in db.list_endpoints(side=side):
        f = db.get_file_by_id(int(ep.file_id)) if ep.file_id is not None else None
        if f is not None and f.repo_id == repo_id and ep.method == method and ep.path == path:
            return ep
    raise AssertionError("endpoint introuvable : %s %s %s (%s)" % (side, method, path, repo_id))


def test_integration_real_index_of_sample_project(develop_home: Path, project: Path, tmp_path: Path):
    ws = _five_repos(project)
    with Database(tmp_path / "memory.sqlite3") as db:
        run = Indexer(db, ws).run()
        assert run.status == "ok", run.stats.get("errors")
        linker = run.stats["linker"]
        assert isinstance(linker, dict) and linker["total"] > 0 and linker["errors"] == []
        assert linker["base_prefixes"]["frontend"] == ["/api"] and linker["base_prefixes"]["backend"] == ["/api"]

        # confirmOrder (client.ts) → POST /orders/{order}/confirm (routes/api.php), 1.0 grâce au préfixe /api
        client = _endpoint(db, "client", "POST", "/orders/${id}/confirm", "frontend")
        server = _endpoint(db, "server", "POST", "/orders/{order}/confirm", "backend")
        assert client.handler == "confirmOrder"
        calls = _relations(db, "calls_endpoint")
        rel = calls[(node_key("endpoint", client.id), node_key("endpoint", server.id))]
        assert rel.confidence == 1.0 and rel.meta["method"] == "POST"
        # OrderService (Http::post) → Express POST /notifications
        client = _endpoint(db, "client", "POST", "http://localhost:3001/notifications", "backend")
        server = _endpoint(db, "server", "POST", "/notifications", "notifier")
        assert "OrderService" in (client.handler or "")
        assert calls[(node_key("endpoint", client.id), node_key("endpoint", server.id))].confidence == 1.0
        # fetch(`${API_BASE}/customers`) → GET /customers (full_path /api/customers) : exact
        client = _endpoint(db, "client", "GET", "${API_BASE}/customers?q=${encodeURIComponent(q)}", "frontend")
        server = _endpoint(db, "server", "GET", "/customers", "backend")
        assert (node_key("endpoint", client.id), node_key("endpoint", server.id)) in calls
        assert linker["endpoints_unmatched_client"] == 0

        # same_entity : Order eloquent ↔ migration orders ↔ sql_table orders ↔ interface TS Order
        def model(kind: str, name: str, repo_id: str) -> int:
            for m in db.find_models(name=name):
                f = db.get_file_by_id(int(m.file_id))
                if m.kind == kind and f is not None and f.repo_id == repo_id:
                    return int(m.id)
            raise AssertionError("modèle introuvable : %s %s (%s)" % (kind, name, repo_id))

        order = [node_key("model", model("eloquent", "Order", "backend")),
                 node_key("model", model("migration", "orders", "backend")),
                 node_key("model", model("sql_table", "orders", "database")),
                 node_key("model", model("ts_interface", "Order", "frontend"))]
        same = _relations(db, "same_entity")
        for i in range(4):
            for j in range(i + 1, 4):
                rel = _undirected(same, order[i], order[j])
                assert rel is not None and rel.confidence >= 0.7, (order[i], order[j])
        assert _undirected(same, order[0], order[1]).confidence == 0.9

        # imports : OrderList.tsx → client.ts ; OrderService.php → Order.php (App\ → app/ sans PSR-4 déclaré)
        imports = _relations(db, "imports")
        f = lambda repo, path: node_key("file", db.get_file(repo, path).id)  # noqa: E731
        assert (f("frontend", "src/components/OrderList.tsx"), f("frontend", "src/api/client.ts")) in imports
        assert (f("backend", "app/Services/OrderService.php"), f("backend", "app/Models/Order.php")) in imports
        assert (f("notifier", "src/server.js"), f("notifier", "src/store.js")) in imports
        # uses_model : OrderService.php utilise le modèle Order
        uses = _relations(db, "uses_model")
        assert (f("backend", "app/Services/OrderService.php"), order[0]) in uses
        # la règle du nom ne traverse pas les dépôts : Order.php (backend) n'« utilise » pas l'interface TS
        # homonyme du frontend, routes/api.php n'en hérite pas, client.ts n'utilise pas le modèle eloquent
        assert (f("backend", "app/Models/Order.php"), order[3]) not in uses
        assert (f("backend", "routes/api.php"), order[3]) not in uses
        assert (f("frontend", "src/api/client.ts"), order[0]) not in uses
        models_file = {int(row.id): int(row.file_id) for row in db.list_models() if row.id is not None}
        for src, dst in uses:
            assert db.get_file_by_id(int(src[len("file:"):])).repo_id == \
                db.get_file_by_id(models_file[int(dst[len("model:"):])]).repo_id, (src, dst)

        # le dictionnaire de données (dépôt docs) est documented_by de la table orders et d'OrderService
        dictionary = f("docs", "dictionnaire-donnees.md")
        documented = _relations(db, "documented_by")
        assert ("table:orders", dictionary) in documented
        assert any(src.startswith("symbol:") and dst == dictionary and r.meta.get("name") == "OrderService"
                   for (src, dst), r in documented.items())
        assert not is_stale(db, "file", db.get_file("docs", "dictionnaire-donnees.md").id)

        # explorateur et descriptions sur le graphe réel
        g = neighbors_graph(db, "table:orders", depth=2)
        assert dictionary in {n["key"] for n in g["nodes"]} and order[3] in {n["key"] for n in g["nodes"]}
        assert describe_relation(documented[("table:orders", dictionary)], db) == "documente la table orders"

        # une seconde passe (ou un second run) ne change rien
        before = _snapshot(db)
        again = link_all(db, ws)
        assert _snapshot(db) == before and again.total == linker["total"]
        second = Indexer(db, ws).run()
        assert second.status == "ok" and _snapshot(db) == before
