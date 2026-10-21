"""Tests de l'analyse d'impact déterministe (develop.impact_analysis)."""
from __future__ import annotations

import json
import os
import subprocess
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import pytest

from develop.config import RepoConfig, WorkspaceConfig
from develop.git_tools import ChangedFile, git_available
from develop.impact_analysis import (
    HUB_THRESHOLD,
    MAX_HOPS,
    NODE_CAP,
    Hunk,
    ImpactAnalyzer,
    ImpactItem,
    ImpactReport,
    changed_line_ranges,
    merge_ranges,
    parse_unified_diff,
    ranges_intersect,
    symbols_touched,
)
from develop.project_scanner import compute_sha1, scan_repo
from develop.storage import (
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
    SymbolRow,
    node_key,
)

# --------------------------------------------------------------------------- aides


def _disk_signature(db: Database, repo_id: str, path: str):
    """``(taille, mtime, sha1)`` du fichier sur le disque du dépôt enregistré (clé NFC ou nom NFD), sinon None."""
    repo = db.get_repo(repo_id)
    if repo is None or not repo.path:
        return None
    for rel in (path, unicodedata.normalize("NFD", path)):
        abs_path = os.path.join(repo.path, *rel.split("/"))
        if os.path.isfile(abs_path):
            st = os.stat(abs_path)
            return int(st.st_size), float(st.st_mtime), compute_sha1(abs_path)
    return None


def _file(db: Database, repo_id: str, path: str, language: str = "php") -> FileRow:
    """Ligne de fichier posée à la main. Sa signature est celle du disque quand le fichier existe sous le dépôt
    enregistré (comme l'indexeur l'écrirait), fictive sinon."""
    size, mtime, sha1 = _disk_signature(db, repo_id, path) or (100, 1.0, path)
    f = FileRow(repo_id=repo_id, path=path, language=language, kind="code", size=size, mtime=mtime, sha1=sha1)
    db.upsert_file(f)
    return f


def _mark_reindexed(db: Database, f: FileRow) -> None:
    """Simule une réindexation du fichier après une modification sur le disque : la signature de l'index redevient
    celle du disque (les symboles posés à la main sont supposés déjà corrects)."""
    sig = _disk_signature(db, f.repo_id, f.path)
    assert sig is not None, "fichier absent du disque : %s" % f.path
    f.size, f.mtime, f.sha1 = sig
    db.upsert_file(f)


def _workspace(back_path: str = "/nonexistent/backend", front_path: str = "/nonexistent/frontend") -> WorkspaceConfig:
    return WorkspaceConfig(id="ws", name="Démo", repos=[
        RepoConfig(id="back", name="backend", path=back_path, kind="backend"),
        RepoConfig(id="front", name="frontend", path=front_path, kind="frontend"),
    ])


def _by_node(report: ImpactReport) -> Dict[str, ImpactItem]:
    return {i.node: i for i in report.affected}


def _by_path(report: ImpactReport) -> Dict[str, ImpactItem]:
    return {i.path: i for i in report.affected if i.path}


def build_graph(db: Database) -> SimpleNamespace:
    """Graphe posé à la main qui modèle la fixture ``sample_project``.

    OrderService.confirm ← handles ← POST /orders/{order}/confirm ← calls_endpoint ← confirmOrder
    (client.ts) ← imports ← OrderList.tsx ; modèle Order (eloquent) same_entity migration orders ;
    les deux définissent la table ``orders``, documentée par le dictionnaire de données ; la route est
    documentée par la spécification ; une note de réunion documente OrderService.confirm.
    """
    db.upsert_repo(RepoRow(id="back", name="backend", path="/nonexistent/backend", kind="backend"))
    db.upsert_repo(RepoRow(id="front", name="frontend", path="/nonexistent/frontend", kind="frontend"))

    f_service = _file(db, "back", "app/Services/OrderService.php")
    f_controller = _file(db, "back", "app/Http/Controllers/OrderController.php")
    f_routes = _file(db, "back", "routes/api.php")
    f_order = _file(db, "back", "app/Models/Order.php")
    f_migration = _file(db, "back", "database/migrations/2024_01_01_000000_create_orders_table.php")
    f_client = _file(db, "front", "src/api/client.ts", "typescript")
    f_list = _file(db, "front", "src/components/OrderList.tsx", "tsx")
    f_search = _file(db, "front", "src/components/CustomerSearch.tsx", "tsx")

    s_service = SymbolRow(name="OrderService", kind="class", line_start=6, line_end=60)
    s_confirm = SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=20, line_end=35)
    s_cancel = SymbolRow(name="cancel", kind="method", parent="OrderService", line_start=40, line_end=55)
    db.replace_file_artifacts(f_service.id, symbols=[s_service, s_confirm, s_cancel])
    s_ctrl_confirm = SymbolRow(name="confirm", kind="method", parent="OrderController", line_start=15, line_end=25)
    db.replace_file_artifacts(f_controller.id, symbols=[s_ctrl_confirm])
    s_order_class = SymbolRow(name="Order", kind="class", line_start=8, line_end=40)
    m_order = ModelRow(name="Order", kind="eloquent", table_name="orders", line=8)
    db.replace_file_artifacts(f_order.id, symbols=[s_order_class], models=[m_order])
    m_migration = ModelRow(name="orders", kind="migration", table_name="orders", line=12)
    db.replace_file_artifacts(f_migration.id, models=[m_migration])
    e_server = EndpointRow(side="server", method="POST", path="/orders/{order}/confirm",
                           normalized_path="/orders/:param/confirm", handler="OrderController@confirm",
                           framework="laravel", line=12)
    db.replace_file_artifacts(f_routes.id, endpoints=[e_server])
    e_client = EndpointRow(side="client", method="POST", path="/orders/${id}/confirm",
                           normalized_path="/orders/:param/confirm", handler="confirmOrder", framework="fetch", line=30)
    s_confirm_order = SymbolRow(name="confirmOrder", kind="function", line_start=28, line_end=34)
    db.replace_file_artifacts(f_client.id, symbols=[s_confirm_order], endpoints=[e_client])
    s_order_list = SymbolRow(name="OrderList", kind="component", line_start=5, line_end=60)
    db.replace_file_artifacts(f_list.id, symbols=[s_order_list])

    d_dict = DocumentRow(title="Dictionnaire de données", source_path="docs/dictionnaire-donnees.md",
                         kind="repo_file", format="md")
    db.add_document(d_dict)
    d_spec = DocumentRow(title="Spécification commandes", source_path="docs/specification-commandes.md",
                         kind="repo_file", format="md")
    db.add_document(d_spec)
    n_meeting = NoteRow(title="Réunion statuts", body="Décision : la confirmation passe par OrderService.confirm.",
                        tags=["décision"], note_date="2026-09-01")
    db.add_note(n_meeting)

    db.add_relations([
        RelationRow(src=node_key("file", f_controller.id), dst=node_key("file", f_service.id), kind="imports"),
        RelationRow(src=node_key("file", f_routes.id), dst=node_key("file", f_controller.id), kind="imports"),
        RelationRow(src=node_key("endpoint", e_server.id), dst=node_key("symbol", s_confirm.id), kind="handles"),
        RelationRow(src=node_key("endpoint", e_client.id), dst=node_key("endpoint", e_server.id),
                    kind="calls_endpoint", confidence=1.0),
        RelationRow(src=node_key("file", f_list.id), dst=node_key("file", f_client.id), kind="imports"),
        RelationRow(src=node_key("file", f_search.id), dst=node_key("file", f_client.id), kind="imports"),
        RelationRow(src=node_key("model", m_order.id), dst=node_key("table", "orders"), kind="defines_table"),
        RelationRow(src=node_key("model", m_migration.id), dst=node_key("table", "orders"), kind="defines_table"),
        RelationRow(src=node_key("model", m_order.id), dst=node_key("model", m_migration.id), kind="same_entity",
                    confidence=0.9),
        RelationRow(src=node_key("file", f_service.id), dst=node_key("model", m_order.id), kind="uses_model"),
        RelationRow(src=node_key("table", "orders"), dst=node_key("document", d_dict.id), kind="documented_by"),
        RelationRow(src=node_key("endpoint", e_server.id), dst=node_key("document", d_spec.id), kind="documented_by"),
        RelationRow(src=node_key("symbol", s_confirm.id), dst=node_key("note", n_meeting.id), kind="documented_by"),
    ])
    return SimpleNamespace(**{k: v for k, v in locals().items() if k != "db"})


@pytest.fixture
def db():
    with Database(":memory:") as d:
        yield d


@pytest.fixture
def graph(db: Database) -> SimpleNamespace:
    return build_graph(db)


@pytest.fixture
def analyzer(db: Database, graph: SimpleNamespace) -> ImpactAnalyzer:
    return ImpactAnalyzer(db, _workspace())


# --------------------------------------------------------------------------- diff.py

U0_DIFF = """diff --git a/app/Services/OrderService.php b/app/Services/OrderService.php
index 1111111..2222222 100644
--- a/app/Services/OrderService.php
+++ b/app/Services/OrderService.php
@@ -12 +12 @@ class OrderService
-        $old = 1;
+        $new = 2;
@@ -30,0 +31,2 @@ class OrderService
+        // ajout
+        // ajout 2
@@ -45,2 +47,0 @@ class OrderService
-        // supprimé
-        // supprimé 2
"""

STANDARD_DIFF = """diff --git a/a.py b/a.py
index 1..2 100644
--- a/a.py
+++ b/a.py
@@ -3,7 +3,8 @@ def f():
     x = 1
     y = 2
-    z = 3
+    z = 4
+    w = 5
     return x
-    gone = 1

 def g():
"""

RENAME_DELETE_ADD_DIFF = """diff --git a/old/name.ts b/new/name.ts
similarity index 90%
rename from old/name.ts
rename to new/name.ts
index 1..2 100644
--- a/old/name.ts
+++ b/new/name.ts
@@ -1 +1 @@
-export const a = 1;
+export const a = 2;
diff --git a/pure.ts b/pure2.ts
similarity index 100%
rename from pure.ts
rename to pure2.ts
diff --git a/gone.py b/gone.py
deleted file mode 100644
index 1..0
--- a/gone.py
+++ /dev/null
@@ -1,3 +0,0 @@
-a
-b
-c
diff --git a/new.py b/new.py
new file mode 100644
index 0..1
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+a
+b
\\ No newline at end of file
Binary files a/logo.png and b/logo.png differ
"""


def test_parse_unified_diff_zero_context():
    hunks = parse_unified_diff(U0_DIFF)
    assert list(hunks) == ["app/Services/OrderService.php"]
    hs = hunks["app/Services/OrderService.php"]
    assert [(h.old_start, h.old_count, h.new_start, h.new_count) for h in hs] == [(12, 1, 12, 1), (30, 0, 31, 2), (45, 2, 47, 0)]
    assert hs[0].added_lines == [12] and hs[0].removed_count == 1
    assert hs[1].added_lines == [31, 32] and hs[1].removed_count == 0
    assert hs[2].added_lines == [] and hs[2].removed_count == 2 and hs[2].is_pure_removal
    # ligne remplacée → elle seule ; ajout → ses lignes ; suppression pure → les deux lignes qui encadrent le trou
    assert changed_line_ranges(hs) == [(12, 12), (31, 32), (47, 48)]


def test_parse_unified_diff_standard_context():
    hunks = parse_unified_diff(STANDARD_DIFF)
    (h,) = hunks["a.py"]
    assert (h.old_start, h.old_count, h.new_start, h.new_count) == (3, 7, 3, 8)
    assert h.added_lines == [5, 6]  # z = 4 ; w = 5 (numéros côté nouveau)
    assert h.removed_count == 2
    # "gone = 1" supprimée juste après "return x" (ligne 7) : trou encadré par 7 et 8, fusionné avec 5-6
    assert changed_line_ranges(hunks["a.py"]) == [(5, 8)]


def test_parse_unified_diff_rename_delete_add_and_binary():
    hunks = parse_unified_diff(RENAME_DELETE_ADD_DIFF)
    assert set(hunks) == {"new/name.ts", "pure2.ts", "gone.py", "new.py", }
    assert "old/name.ts" not in hunks and "pure.ts" not in hunks  # rapporté sous le nouveau chemin
    assert changed_line_ranges(hunks["new/name.ts"]) == [(1, 1)]
    assert hunks["pure2.ts"] == []  # renommage pur : présent, sans bloc
    (gone,) = hunks["gone.py"]
    assert gone.is_pure_removal and gone.removed_count == 3 and gone.new_count == 0
    assert changed_line_ranges(hunks["new.py"]) == [(1, 2)]
    assert all(h.path == path for path, hs in hunks.items() for h in hs)


def test_parse_unified_diff_is_tolerant():
    assert parse_unified_diff("") == {}
    assert parse_unified_diff("n'importe quoi\nsans en-tête\n@@ -1 +1 @@\n+x\n") == {}
    # un bloc tronqué ne lève pas
    truncated = "--- a/x.py\n+++ b/x.py\n@@ -1,5 +1,6 @@\n+une seule ligne\n"
    assert changed_line_ranges(parse_unified_diff(truncated)["x.py"]) == [(1, 1)]


def test_merge_ranges_and_intersection():
    assert merge_ranges([(5, 7), (1, 2), (3, 4), (10, 12), (11, 15)]) == [(1, 7), (10, 15)]
    assert merge_ranges([]) == []
    assert ranges_intersect(10, 20, [(1, 5), (20, 25)]) is True
    assert ranges_intersect(10, 20, [(1, 9), (21, 25)]) is False
    assert ranges_intersect(10, None, [(10, 10)]) is True
    assert ranges_intersect(None, None, [(1, 1)]) is True  # position inconnue : on signale


def test_changed_line_ranges_from_hand_made_hunks():
    hunks = [Hunk(path="x", old_start=1, old_count=1, new_start=1, new_count=1, added_lines=[1, 2, 3, 7, 8]),
             Hunk(path="x", old_start=20, old_count=1, new_start=19, new_count=0, removed_count=1)]
    assert changed_line_ranges(hunks) == [(1, 3), (7, 8), (19, 20)]


def test_symbols_touched(db: Database, graph: SimpleNamespace):
    fid = graph.f_service.id
    assert [s.name for s in symbols_touched(db, fid, [(22, 22)])] == ["OrderService", "confirm"]
    assert [s.name for s in symbols_touched(db, fid, [(41, 50)])] == ["OrderService", "cancel"]
    assert [s.name for s in symbols_touched(db, fid, [(1, 3)])] == []
    assert [s.name for s in symbols_touched(db, fid, [])] == ["OrderService", "confirm", "cancel"]  # fichier entier
    assert symbols_touched(db, 999_999, []) == []


# --------------------------------------------------------------------------- from_symbol

def test_from_symbol_reaches_route_client_importer_documents_and_note(analyzer: ImpactAnalyzer, graph: SimpleNamespace):
    report = analyzer.from_symbol(graph.s_confirm.id, max_hops=3)
    assert report.source["kind"] == "symbol" and report.source["name"] == "OrderService.confirm"
    assert report.changed == [{
        "path": "app/Services/OrderService.php", "status": "selected", "repo_id": "back",
        "file_id": graph.f_service.id, "symbols_changed": ["OrderService.confirm"], "ranges": [[20, 35]], "indexed": True,
    }]
    nodes = _by_node(report)
    assert node_key("symbol", graph.s_confirm.id) not in nodes  # la graine n'est pas dans affected

    # route serveur gérée par le symbole
    assert [e.name for e in report.endpoints] == ["POST /orders/:param/confirm"]
    route = report.endpoints[0]
    assert route.hops == 1 and route.reason == "route gérée par OrderService.confirm"
    assert route.confidence == pytest.approx(0.95 * 0.85)
    assert route.path == "routes/api.php" and route.line == 12 and route.repo_id == "back"

    # appel client (frontend)
    assert len(report.frontend_calls) == 1
    call = report.frontend_calls[0]
    assert call.hops == 2 and call.repo_id == "front" and call.extra["handler"] == "confirmOrder"
    assert call.reason == "appelle POST /orders/:param/confirm depuis src/api/client.ts"
    assert call.confidence == pytest.approx(0.95 * 0.85 * 1.0 * 0.85)

    # composant qui importe le client : 3 sauts, chaîne lisible
    paths = _by_path(report)
    order_list = paths["src/components/OrderList.tsx"]
    assert order_list.hops == 3 and order_list.type == "file"
    assert order_list.via == ["OrderService.confirm", "POST /orders/:param/confirm", "confirmOrder (frontend)",
                              "src/components/OrderList.tsx"]
    assert order_list.reason.startswith("importe src/api/client.ts")
    assert order_list.confidence == pytest.approx(0.95 * 0.85 * 0.85 * 0.7 * 0.85)
    assert "src/components/CustomerSearch.tsx" in paths

    # fichier contenant, importateur, documents, note
    assert paths["app/Services/OrderService.php"].reason == "contient le symbole OrderService.confirm"
    assert paths["app/Services/OrderService.php"].confidence == pytest.approx(0.85)
    assert paths["app/Http/Controllers/OrderController.php"].reason == "importe app/Services/OrderService.php"
    assert paths["app/Http/Controllers/OrderController.php"].hops == 2
    assert [d.name for d in report.documents] == ["Spécification commandes"]
    assert report.documents[0].reason == "documente la route POST /orders/:param/confirm"
    assert [n.name for n in report.notes] == ["Réunion statuts"]
    assert report.notes[0].reason == "documente OrderService.confirm" and report.notes[0].hops == 1
    # le symbole voisin non modifié n'est pas touché
    assert not any(i.name == "OrderService.cancel" for i in report.affected)

    # tri : confiance décroissante puis sauts ; dédoublonnage par noeud
    confs = [i.confidence for i in report.affected]
    assert confs == sorted(confs, reverse=True)
    assert len({i.node for i in report.affected}) == len(report.affected)
    assert report.stats["seeds"] == 1 and report.stats["max_hops"] == 3 and report.stats["hubs_skipped"] == 0
    assert report.stats["affected_total"] == len(report.affected)
    assert report.stats["by_type"]["endpoint"] == 2 and report.stats["by_type"]["note"] == 1
    assert report.warnings == [] and report.stale_index is False


def test_from_symbol_default_depth_stops_before_the_importer(analyzer: ImpactAnalyzer, graph: SimpleNamespace):
    report = analyzer.from_symbol(graph.s_confirm.id)
    assert report.stats["max_hops"] == 2
    paths = _by_path(report)
    assert "src/components/OrderList.tsx" not in paths
    assert len(report.frontend_calls) == 1 and len(report.endpoints) == 1


def test_max_hops_is_bounded(analyzer: ImpactAnalyzer, graph: SimpleNamespace):
    report = analyzer.from_symbol(graph.s_confirm.id, max_hops=10)
    assert report.stats["max_hops"] == MAX_HOPS == 3
    assert any("Profondeur limitée" in w for w in report.warnings)
    assert analyzer.from_symbol(graph.s_confirm.id, max_hops=0).stats["max_hops"] == 1
    assert analyzer.from_symbol(graph.s_confirm.id, max_hops="abc").stats["max_hops"] == 2  # type: ignore[arg-type]


def test_from_symbol_unknown(analyzer: ImpactAnalyzer):
    report = analyzer.from_symbol(987_654)
    assert report.affected == [] and report.changed == []
    assert any("introuvable" in w for w in report.warnings)


def test_from_symbol_model_class_reaches_models_tables_and_dictionary(analyzer: ImpactAnalyzer, graph: SimpleNamespace):
    report = analyzer.from_symbol(graph.s_order_class.id, max_hops=3)
    nodes = _by_node(report)
    model = nodes[node_key("model", graph.m_order.id)]
    assert model.hops == 1 and model.reason == "modèle défini par la classe Order"
    assert {(i.type, i.name) for i in report.models} == {("model", "Order"), ("model", "orders"), ("table", "orders")}
    migration = nodes[node_key("model", graph.m_migration.id)]
    assert migration.reason == "même entité que Order" and migration.confidence == pytest.approx(0.85 * 0.8 * 0.9 * 0.85)
    assert nodes["table:orders"].reason == "table définie par le modèle Order"
    assert [d.name for d in report.documents] == ["Dictionnaire de données"]
    assert report.documents[0].reason == "documente la table orders"
    assert _by_path(report)["app/Services/OrderService.php"].reason == "utilise le modèle Order"


# --------------------------------------------------------------------------- from_file

def test_from_file_seeds_whole_file(analyzer: ImpactAnalyzer, graph: SimpleNamespace):
    report = analyzer.from_file("back", "app/Models/Order.php")
    assert report.source == {"kind": "file", "repo_id": "back", "path": "app/Models/Order.php", "max_hops": 2}
    (entry,) = report.changed
    assert entry["status"] == "selected" and entry["symbols_changed"] == ["Order"] and entry["whole_file"] is True
    assert entry["seeds"] == 3  # fichier + classe + modèle
    assert report.stats["seeds"] == 3
    nodes = _by_node(report)
    for seed in (node_key("file", graph.f_order.id), node_key("symbol", graph.s_order_class.id),
                 node_key("model", graph.m_order.id)):
        assert seed not in nodes
    assert {(i.type, i.name) for i in report.models} == {("model", "orders"), ("table", "orders")}
    assert nodes[node_key("model", graph.m_migration.id)].hops == 1
    assert nodes[node_key("model", graph.m_migration.id)].confidence == pytest.approx(0.8 * 0.9 * 0.85)
    assert [d.name for d in report.documents] == ["Dictionnaire de données"]
    paths = _by_path(report)
    assert paths["app/Services/OrderService.php"].reason == "utilise le modèle Order"
    assert paths["app/Services/OrderService.php"].confidence == pytest.approx(0.6 * 0.85)
    assert paths["app/Http/Controllers/OrderController.php"].hops == 2  # importe OrderService.php


def test_from_file_accepts_absolute_and_dotted_paths(db: Database, graph: SimpleNamespace, tmp_path: Path):
    ws = _workspace(back_path=str(tmp_path))
    analyzer = ImpactAnalyzer(db, ws)
    absolute = str(tmp_path / "app" / "Models" / "Order.php")
    assert analyzer.from_file("back", absolute).source["path"] == "app/Models/Order.php"
    assert analyzer.from_file("back", "./app/Models/Order.php").changed[0]["path"] == "app/Models/Order.php"


def test_from_file_unknown(analyzer: ImpactAnalyzer):
    report = analyzer.from_file("back", "nope/Missing.php")
    assert report.affected == [] and report.changed == []
    assert any("introuvable" in w for w in report.warnings)


# --------------------------------------------------------------------------- propagation : décroissance, meilleur chemin, hubs, plafond

def test_decay_and_best_path_kept(db: Database):
    db.upsert_repo(RepoRow(id="r", name="r", path="/nonexistent"))
    f = _file(db, "r", "src/f.py", "python")
    g = _file(db, "r", "src/g.py", "python")
    s = SymbolRow(name="S", kind="function", line_start=1, line_end=5)
    db.replace_file_artifacts(f.id, symbols=[s])
    d = DocumentRow(title="Doc S")
    db.add_document(d)
    db.add_relations([
        RelationRow(src=node_key("file", g.id), dst=node_key("file", f.id), kind="imports"),
        RelationRow(src=node_key("symbol", s.id), dst=node_key("document", d.id), kind="documented_by"),
        RelationRow(src=node_key("file", f.id), dst=node_key("document", d.id), kind="documented_by"),
    ])
    analyzer = ImpactAnalyzer(db, WorkspaceConfig(id="ws", name="ws", repos=[RepoConfig(id="r", name="r", path="/nonexistent")]))
    items = analyzer.propagate([(node_key("symbol", s.id), 1.0, "S")], max_hops=2)
    by_node = {i.node: i for i in items}
    assert by_node[node_key("file", f.id)].confidence == pytest.approx(0.85)
    assert by_node[node_key("file", g.id)].confidence == pytest.approx(0.85 * 0.9 * 0.85)
    assert by_node[node_key("file", g.id)].via == ["S", "src/f.py", "src/g.py"]
    # le document est atteignable en 1 saut (0.765) et en 2 sauts (0.65) : le meilleur chemin est gardé
    doc = by_node[node_key("document", d.id)]
    assert doc.confidence == pytest.approx(0.9 * 0.85) and doc.hops == 1 and doc.via == ["S", "Doc S"]
    assert doc.reason == "documente S"
    assert [i.confidence for i in items] == sorted((i.confidence for i in items), reverse=True)


def test_propagate_ignores_unknown_seeds_and_seed_confidence_scales(db: Database, graph: SimpleNamespace):
    analyzer = ImpactAnalyzer(db, _workspace())
    assert analyzer.propagate([("symbol:999999", 1.0, "?"), ("bidon", 1.0, "?")], max_hops=2) == []
    half = analyzer.propagate([(node_key("symbol", graph.s_confirm.id), 0.5, "OrderService.confirm")], max_hops=1)
    assert {i.node for i in half} >= {node_key("file", graph.f_service.id), node_key("endpoint", graph.e_server.id)}
    assert all(i.hops == 1 for i in half)
    assert max(i.confidence for i in half) == pytest.approx(0.5 * 0.85)


def _file_with_importers(db: Database, repo_id: str, path: str, count: int) -> FileRow:
    hub = _file(db, repo_id, path, "python")
    relations = []
    for i in range(count):
        imp = _file(db, repo_id, "%s.importer%03d.py" % (path, i), "python")
        relations.append(RelationRow(src=node_key("file", imp.id), dst=node_key("file", hub.id), kind="imports"))
    db.add_relations(relations)
    return hub


def test_hub_is_reported_but_not_expanded(db: Database):
    db.upsert_repo(RepoRow(id="r", name="r", path="/nonexistent"))
    hub = _file_with_importers(db, "r", "src/utils.py", HUB_THRESHOLD + 1)
    busy = _file_with_importers(db, "r", "src/helpers.py", HUB_THRESHOLD)
    analyzer = ImpactAnalyzer(db, WorkspaceConfig(id="ws", name="ws", repos=[RepoConfig(id="r", name="r", path="/nonexistent")]))

    report = analyzer.from_file("r", hub.path)
    assert report.affected == []
    assert report.stats["hubs_skipped"] == 1 and report.stats["affected_total"] == 0
    assert any("central" in w and "src/utils.py" in w and str(HUB_THRESHOLD + 1) in w for w in report.warnings)

    report = analyzer.from_file("r", busy.path)  # exactement le seuil : développé normalement
    assert report.stats["hubs_skipped"] == 0 and report.stats["affected_total"] == HUB_THRESHOLD
    assert all(i.reason == "importe src/helpers.py" for i in report.affected)


def test_node_cap_truncates_with_warning(db: Database):
    db.upsert_repo(RepoRow(id="r", name="r", path="/nonexistent"))
    root = _file(db, "r", "src/core.py", "python")
    relations = []
    n_importers = (NODE_CAP // 3) + 24  # 190 importateurs × (1 + 2 documents) = 570 noeuds > 500
    for i in range(n_importers):
        imp = _file(db, "r", "src/mod%03d.py" % i, "python")
        relations.append(RelationRow(src=node_key("file", imp.id), dst=node_key("file", root.id), kind="imports"))
        for j in range(2):
            doc = DocumentRow(title="doc %d-%d" % (i, j))
            db.add_document(doc)
            relations.append(RelationRow(src=node_key("file", imp.id), dst=node_key("document", doc.id), kind="documented_by"))
    db.add_relations(relations)
    analyzer = ImpactAnalyzer(db, WorkspaceConfig(id="ws", name="ws", repos=[RepoConfig(id="r", name="r", path="/nonexistent")]))
    report = analyzer.from_file("r", "src/core.py", max_hops=2)
    assert report.stats["affected_total"] == NODE_CAP and report.stats["truncated"] is True
    assert any("tronquée" in w for w in report.warnings)
    assert report.stats["by_type"]["file"] == n_importers  # les plus sûrs d'abord, puis les documents jusqu'au plafond


# --------------------------------------------------------------------------- index obsolète, recherche lexicale

def test_stale_checker_marks_report(db: Database, graph: SimpleNamespace):
    calls: List[str] = []

    def stale(repo_id: str) -> List[str]:
        calls.append(repo_id)
        return ["app/Services/OrderService.php", "routes/api.php"] if repo_id == "back" else []

    analyzer = ImpactAnalyzer(db, _workspace(), stale_checker=stale)
    report = analyzer.from_symbol(graph.s_confirm.id)
    assert calls == ["back"]
    assert report.stale_index is True
    assert report.stale_files == ["app/Services/OrderService.php", "routes/api.php"]
    assert any(w.startswith("Index obsolète") and "Réindexez puis relancez" in w for w in report.warnings)
    assert report.affected  # l'analyse est tout de même rendue

    fresh = ImpactAnalyzer(db, _workspace(), stale_checker=lambda repo_id: []).from_file("front", "src/api/client.ts")
    assert fresh.stale_index is False and fresh.stale_files == [] and fresh.warnings == []


def test_stale_checker_failure_is_only_a_warning(db: Database, graph: SimpleNamespace):
    def broken(repo_id: str) -> List[str]:
        raise RuntimeError("disque en feu")

    report = ImpactAnalyzer(db, _workspace(), stale_checker=broken).from_symbol(graph.s_confirm.id)
    assert report.stale_index is False and report.affected
    assert any("disque en feu" in w for w in report.warnings)


def test_lexical_complement_with_database_fts(db: Database, graph: SimpleNamespace):
    guide = DocumentRow(title="Guide d'exploitation", source_path="docs/guide.md", kind="upload", format="md")
    db.add_document(guide)
    db.replace_chunks("document", guide.id, [ChunkRow(text="En production, OrderService confirm est appelé par le batch nocturne.")])
    db.replace_chunks("document", graph.d_spec.id, [ChunkRow(text="La route de confirmation appelle OrderService confirm.")])
    other = DocumentRow(title="Sans rapport")
    db.add_document(other)
    db.replace_chunks("document", other.id, [ChunkRow(text="Rien à voir avec les commandes.")])

    report = ImpactAnalyzer(db, _workspace(), search_engine=db).from_symbol(graph.s_confirm.id)
    docs = {d.name: d for d in report.documents}
    assert set(docs) == {"Spécification commandes", "Guide d'exploitation"}
    assert docs["Spécification commandes"].reason.startswith("documente")  # le graphe prime, pas de doublon
    lexical = docs["Guide d'exploitation"]
    assert lexical.confidence == 0.5 and lexical.reason == "mentionne OrderService.confirm"
    assert lexical.extra["channel"] == "lexical" and "confirm" in lexical.extra["snippet"].lower()
    assert lexical.via == ["OrderService.confirm", "Guide d'exploitation"]
    assert len({i.node for i in report.affected}) == len(report.affected)
    # sans moteur : rien d'ajouté
    assert [d.name for d in ImpactAnalyzer(db, _workspace()).from_symbol(graph.s_confirm.id).documents] == ["Spécification commandes"]


def test_lexical_complement_is_limited_and_tolerant(db: Database, graph: SimpleNamespace):
    notes = []
    for i in range(15):
        n = NoteRow(title="note %02d" % i, body="confirm")
        db.add_note(n)
        notes.append(n)
    queries: List[str] = []

    def engine(query: str):
        queries.append(query)
        return [FtsHit(chunk_id=i, score=1.0, snippet="…confirm…", source_type="note", source_id=n.id)
                for i, n in enumerate(notes)] + [FtsHit(chunk_id=99, score=1.0, snippet="", source_type="file", source_id=1)]

    report = ImpactAnalyzer(db, _workspace(), search_engine=engine).from_symbol(graph.s_confirm.id)
    assert queries == ["OrderService.confirm"]
    lexical = [n for n in report.notes if n.extra.get("channel") == "lexical"]
    assert len(lexical) == 10 and all(n.confidence == 0.5 for n in lexical)
    assert len(report.notes) == 11  # + la note du graphe

    def exploding(query: str):
        raise RuntimeError("moteur HS")

    report = ImpactAnalyzer(db, _workspace(), search_engine=exploding).from_symbol(graph.s_confirm.id)
    assert [n.name for n in report.notes] == ["Réunion statuts"]
    assert any("moteur HS" in w for w in report.warnings)


# --------------------------------------------------------------------------- sérialisation (docs/API.md)

def test_to_dict_conforms_to_api(analyzer: ImpactAnalyzer, graph: SimpleNamespace):
    report = analyzer.from_symbol(graph.s_confirm.id, max_hops=3)
    data = report.to_dict()
    assert set(data) == {"source", "changed", "affected", "endpoints", "models", "frontend_calls", "documents", "notes",
                         "stats", "warnings", "stale_index", "stale_files"}
    for key in ("affected", "endpoints", "models", "frontend_calls", "documents", "notes"):
        assert isinstance(data[key], list)
        for item in data[key]:
            assert {"node", "type", "name", "path", "line", "repo_id", "reason", "confidence", "hops", "via", "extra",
                    "id", "file_id"} <= set(item)
            assert 0.0 <= item["confidence"] <= 1.0 and item["hops"] >= 1
            assert item["type"] in ("file", "symbol", "endpoint", "model", "document", "note", "table")
            assert item["reason"]
    for entry in data["changed"]:
        assert {"path", "status", "repo_id", "symbols_changed"} <= set(entry)
    assert {"changed_files", "seeds", "affected_total", "by_type", "hubs_skipped", "max_hops"} <= set(data["stats"])
    assert data["stale_index"] is False and data["stale_files"] == [] and data["warnings"] == []
    assert len(data["affected"]) == sum(len(data[k]) for k in ("endpoints", "models", "frontend_calls", "documents", "notes")) \
        + sum(1 for i in data["affected"] if i["type"] in ("file", "symbol"))
    json.dumps(data, ensure_ascii=False)  # sérialisable tel quel
    endpoint = data["endpoints"][0]
    assert endpoint["id"] == graph.e_server.id and endpoint["file_id"] == graph.f_routes.id
    assert endpoint["extra"]["side"] == "server" and data["frontend_calls"][0]["extra"]["side"] == "client"
    assert ImpactReport().to_dict()["affected"] == [] and ImpactItem(node="table:x", type="table", name="x").to_dict()["id"] is None


# --------------------------------------------------------------------------- from_git (vrai dépôt temporaire)

SERVICE_SOURCE = """<?php
namespace App\\Services;

use App\\Models\\Order;

class OrderService
{
    public function confirm(Order $order): Order
    {
        $order->status = 'confirmed';
        $order->save();
        return $order;
    }

    public function cancel(Order $order): Order
    {
        $order->status = 'cancelled';
        $order->save();
        return $order;
    }
}
"""

ORDER_SOURCE = """<?php
namespace App\\Models;

class Order
{
    protected $table = 'orders';
}
"""

_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin")}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, env=dict(_GIT_ENV, HOME=str(repo)))


@pytest.fixture
def git_project(tmp_path: Path, db: Database):
    """Dépôt Git réel + index posé à la main sur ses fichiers ; renvoie (analyzer, dossier, objets)."""
    if not git_available():
        pytest.skip("git absent")
    root = tmp_path / "backend"
    (root / "app" / "Services").mkdir(parents=True)
    (root / "app" / "Models").mkdir(parents=True)
    (root / "app" / "Services" / "OrderService.php").write_text(SERVICE_SOURCE, encoding="utf-8")
    (root / "app" / "Models" / "Order.php").write_text(ORDER_SOURCE, encoding="utf-8")
    (root / "routes.php").write_text("<?php\n// routes\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")

    db.upsert_repo(RepoRow(id="back", name="backend", path=str(root), kind="backend"))
    f_service = _file(db, "back", "app/Services/OrderService.php")
    f_order = _file(db, "back", "app/Models/Order.php")
    f_routes = _file(db, "back", "routes.php")
    s_class = SymbolRow(name="OrderService", kind="class", line_start=6, line_end=21)
    s_confirm = SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=8, line_end=13)
    s_cancel = SymbolRow(name="cancel", kind="method", parent="OrderService", line_start=15, line_end=20)
    db.replace_file_artifacts(f_service.id, symbols=[s_class, s_confirm, s_cancel])
    s_order = SymbolRow(name="Order", kind="class", line_start=4, line_end=7)
    m_order = ModelRow(name="Order", kind="eloquent", table_name="orders", line=4)
    db.replace_file_artifacts(f_order.id, symbols=[s_order], models=[m_order])
    e_confirm = EndpointRow(side="server", method="POST", path="/orders/{order}/confirm",
                            normalized_path="/orders/:param/confirm", line=2)
    e_cancel = EndpointRow(side="server", method="POST", path="/orders/{order}/cancel",
                           normalized_path="/orders/:param/cancel", line=3)
    db.replace_file_artifacts(f_routes.id, endpoints=[e_confirm, e_cancel])
    db.add_relations([
        RelationRow(src=node_key("endpoint", e_confirm.id), dst=node_key("symbol", s_confirm.id), kind="handles"),
        RelationRow(src=node_key("endpoint", e_cancel.id), dst=node_key("symbol", s_cancel.id), kind="handles"),
        RelationRow(src=node_key("file", f_routes.id), dst=node_key("file", f_service.id), kind="imports"),
        RelationRow(src=node_key("file", f_service.id), dst=node_key("model", m_order.id), kind="uses_model"),
    ])
    ws = WorkspaceConfig(id="ws", name="ws", repos=[RepoConfig(id="back", name="backend", path=str(root), kind="backend"),
                                                   RepoConfig(id="plain", name="plain", path=str(tmp_path / "plain"))])
    (tmp_path / "plain").mkdir()
    objs = SimpleNamespace(f_service=f_service, f_order=f_order, f_routes=f_routes, s_class=s_class, s_confirm=s_confirm,
                           s_cancel=s_cancel, s_order=s_order, m_order=m_order, e_confirm=e_confirm, e_cancel=e_cancel)
    return ImpactAnalyzer(db, ws), root, objs


def test_from_git_clean_tree(git_project):
    analyzer, root, _ = git_project
    report = analyzer.from_git("back")
    assert report.changed == [] and report.affected == [] and report.warnings == []
    assert report.source == {"kind": "git", "repo_id": "back", "rev_range": None, "max_hops": 2}
    assert report.stats["changed_files"] == 0 and report.stats["seeds"] == 0


def test_from_git_only_intersected_symbols_are_seeds(git_project):
    analyzer, root, o = git_project
    path = root / "app" / "Services" / "OrderService.php"
    lines = SERVICE_SOURCE.splitlines(keepends=True)
    lines[9] = "        $order->status = 'validated';\n"  # ligne 10, dans confirm()
    path.write_text("".join(lines), encoding="utf-8")
    _mark_reindexed(analyzer.db, o.f_service)  # index à jour : les plages sont croisées avec les positions indexées

    report = analyzer.from_git("back")
    (entry,) = report.changed
    assert entry["path"] == "app/Services/OrderService.php" and entry["status"] == "modified" and entry["indexed"] is True
    assert entry["ranges"] == [[10, 10]] and entry["whole_file"] is False and entry["reanalyzed"] is False
    assert entry["symbols_changed"] == ["OrderService", "OrderService.confirm"]  # la classe englobe la ligne, pas cancel()
    assert entry["seeds"] == 2 and report.stats["seeds"] == 2
    assert report.warnings == [] and report.stale_index is False

    assert [e.name for e in report.endpoints] == ["POST /orders/:param/confirm"]  # pas la route de cancel()
    nodes = _by_node(report)
    assert node_key("file", o.f_service.id) not in nodes  # le fichier changé est dans `changed`, pas dans `affected`
    assert node_key("symbol", o.s_cancel.id) not in nodes
    # routes.php est atteignable par « importe OrderService.php » (0.65) et par « contient la route » (0.69) :
    # le meilleur chemin est conservé
    routes = _by_path(report)["routes.php"]
    assert routes.hops == 2 and routes.reason == "contient la route POST /orders/:param/confirm"
    assert routes.confidence == pytest.approx(0.95 * 0.85 * 1.0 * 0.85)
    assert report.to_dict()["changed"][0]["symbols_changed"] == ["OrderService", "OrderService.confirm"]


def test_from_git_change_outside_symbols_seeds_the_file(git_project):
    analyzer, root, o = git_project
    path = root / "app" / "Services" / "OrderService.php"
    path.write_text(SERVICE_SOURCE.replace("use App\\Models\\Order;", "use App\\Models\\Order;\nuse App\\Models\\Customer;"),
                    encoding="utf-8")
    report = analyzer.from_git("back")
    (entry,) = report.changed
    assert entry["ranges"] == [[5, 5]] and entry["symbols_changed"] == [] and entry["seeds"] == 1
    assert _by_path(report)["routes.php"].hops == 1  # le fichier est la graine : ses importateurs sont à 1 saut
    assert report.endpoints == []
    # la ligne insérée décale tous les symboles : index obsolète, symboles relus (aucun n'est touché pour autant)
    assert entry["reanalyzed"] is True and report.stale_index is True
    assert report.stale_files == ["app/Services/OrderService.php"] and any("relus" in w for w in report.warnings)


def test_from_git_untracked_file_not_indexed_warns(git_project):
    analyzer, root, _ = git_project
    (root / "app" / "Services" / "NewService.php").write_text("<?php\nclass NewService {}\n", encoding="utf-8")
    (root / "notes.bin").write_bytes(b"\x00\x01")  # binaire : son absence de l'index est normale
    report = analyzer.from_git("back")
    by_path = {c["path"]: c for c in report.changed}
    assert by_path["app/Services/NewService.php"]["status"] == "untracked"
    assert by_path["app/Services/NewService.php"]["indexed"] is False
    assert by_path["notes.bin"]["indexed"] is False
    assert report.stale_index is True and report.stale_files == ["app/Services/NewService.php"]
    assert any("non indexé" in w and "NewService.php" in w for w in report.warnings)
    assert not any("notes.bin" in w for w in report.warnings)
    assert report.affected == []


def test_from_git_deleted_file_is_seeded_whole(git_project):
    analyzer, root, o = git_project
    (root / "app" / "Models" / "Order.php").unlink()
    report = analyzer.from_git("back")
    (entry,) = report.changed
    assert entry["status"] == "deleted" and entry["whole_file"] is True
    assert entry["symbols_changed"] == ["Order"] and entry["seeds"] == 3
    assert _by_path(report)["app/Services/OrderService.php"].reason == "utilise le modèle Order"
    assert report.warnings == []


def test_from_git_rev_range_and_rename(git_project):
    analyzer, root, o = git_project
    path = root / "app" / "Services" / "OrderService.php"
    lines = SERVICE_SOURCE.splitlines(keepends=True)
    lines[16] = "        $order->status = 'void';\n"  # ligne 17, dans cancel()
    path.write_text("".join(lines), encoding="utf-8")
    _git(root, "mv", "routes.php", "routes_api.php")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "second")

    assert analyzer.from_git("back").changed == []  # working tree propre
    report = analyzer.from_git("back", "HEAD~1..HEAD")
    assert report.source["rev_range"] == "HEAD~1..HEAD"
    by_path = {c["path"]: c for c in report.changed}
    assert by_path["app/Services/OrderService.php"]["symbols_changed"] == ["OrderService", "OrderService.cancel"]
    renamed = by_path["routes_api.php"]  # chemin côté nouveau, même si l'index ne connaît que routes.php
    assert renamed["status"] == "renamed" and renamed["old_path"] == "routes.php" and renamed["indexed"] is True
    assert renamed["whole_file"] is True and renamed["seeds"] == 3  # fichier + ses deux routes
    assert report.stale_index is True and "routes_api.php" in report.stale_files
    assert any("renommé" in w for w in report.warnings)
    # les deux routes sont des graines (fichier renommé pris en entier) : elles sont dans `changed`, pas dans `affected`
    assert report.endpoints == [] and report.stats["seeds"] == 5
    nodes = _by_node(report)
    assert nodes[node_key("symbol", o.s_confirm.id)].reason == "gère la route POST /orders/:param/confirm"
    assert node_key("symbol", o.s_cancel.id) not in nodes  # graine


def test_from_git_invalid_range_and_not_a_repo(git_project):
    analyzer, root, _ = git_project
    report = analyzer.from_git("back", "nope..nothing")
    assert report.changed == [] and any("Git" in w for w in report.warnings)
    report = analyzer.from_git("back", "--output=/tmp/x")
    assert report.changed == [] and any("invalide" in w for w in report.warnings)
    report = analyzer.from_git("plain")
    assert report.changed == [] and report.affected == []
    assert any("dépôt Git" in w for w in report.warnings)
    report = analyzer.from_git("inconnu")
    assert any("Dépôt inconnu" in w for w in report.warnings)


def test_from_git_with_stale_checker_and_search_engine(git_project, db: Database):
    analyzer, root, o = git_project
    note = NoteRow(title="Décision confirm", body="OrderService confirm doit rester idempotent.")
    db.add_note(note)
    db.replace_chunks("note", note.id, [ChunkRow(text=note.body)])
    analyzer.search_engine = db
    analyzer.stale_checker = lambda repo_id: ["app/Services/OrderService.php"]
    path = root / "app" / "Services" / "OrderService.php"
    path.write_text(SERVICE_SOURCE.replace("'confirmed'", "'ok'"), encoding="utf-8")
    report = analyzer.from_git("back")
    assert report.stale_index is True and report.stale_files == ["app/Services/OrderService.php"]
    assert [n.name for n in report.notes] == ["Décision confirm"]
    # la classe OrderService (qui englobe la ligne) et confirm() sont changées : la première requête qui touche gagne
    assert report.notes[0].reason in ("mentionne OrderService", "mentionne OrderService.confirm")
    assert report.notes[0].confidence == 0.5 and report.notes[0].extra["channel"] == "lexical"


# --------------------------------------------------------------------------- from_git : index obsolète (lignes décalées)

def _shifted_service_source() -> str:
    """SERVICE_SOURCE avec une méthode ``ping()`` insérée AVANT ``confirm()`` (tout ce qui suit descend de 5 lignes)
    et le corps de ``cancel()`` modifié : dans l'index obsolète, les lignes 8-13 sont encore celles de confirm()."""
    text = SERVICE_SOURCE.replace("    public function confirm(Order $order): Order",
                                  "    public function ping(): string\n    {\n        return 'pong';\n    }\n\n"
                                  "    public function confirm(Order $order): Order")
    return text.replace("'cancelled'", "'void'")


_SHIFTED_RANGES = [[8, 12], [22, 22]]  # ping() + ligne vide ; ligne modifiée de cancel() (17 + 5)


def test_from_git_stale_index_with_shifted_lines_rereads_current_symbols(git_project):
    """Non-régression : index obsolète et lignes insérées avant le symbole modifié. Les plages du diff (côté nouveau)
    ne sont plus croisées avec les positions périmées de l'index (qui désignaient confirm() « en toute confiance »)
    mais avec les symboles relus depuis le disque, puis ramenées aux noeuds indexés par nom qualifié."""
    analyzer, root, o = git_project
    path = root / "app" / "Services" / "OrderService.php"
    path.write_text(_shifted_service_source(), encoding="utf-8")

    report = analyzer.from_git("back")
    (entry,) = report.changed
    assert entry["status"] == "modified" and entry["indexed"] is True and entry["whole_file"] is False
    assert entry["ranges"] == _SHIFTED_RANGES and entry["reanalyzed"] is True
    assert entry["symbols_changed"] == ["OrderService", "OrderService.ping", "OrderService.cancel"]
    assert "OrderService.confirm" not in entry["symbols_changed"]  # avant : confirm signalé, cancel absent
    assert entry["seeds"] == 2 and report.stats["seeds"] == 2  # ping() n'est pas encore indexé : listé, pas graine
    assert [e.name for e in report.endpoints] == ["POST /orders/:param/cancel"]  # pas la route de confirm()
    nodes = _by_node(report)
    assert node_key("symbol", o.s_confirm.id) not in nodes and node_key("file", o.f_service.id) not in nodes
    assert report.stale_index is True and report.stale_files == ["app/Services/OrderService.php"]
    assert [w for w in report.warnings if "relus" in w and "Réindexez" in w and "OrderService.php" in w]
    assert report.to_dict()["changed"][0]["reanalyzed"] is True


def test_from_git_reindexed_after_shift_is_crossed_with_the_index(git_project):
    """Après réindexation (positions à jour, signature du disque) : croisement direct, sans relecture ni avertissement."""
    analyzer, root, o = git_project
    path = root / "app" / "Services" / "OrderService.php"
    path.write_text(_shifted_service_source(), encoding="utf-8")
    fresh = [SymbolRow(name="OrderService", kind="class", line_start=6, line_end=26),
             SymbolRow(name="ping", kind="method", parent="OrderService", line_start=8, line_end=11),
             SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=13, line_end=18),
             SymbolRow(name="cancel", kind="method", parent="OrderService", line_start=20, line_end=25)]
    analyzer.db.replace_file_artifacts(o.f_service.id, symbols=fresh)
    _mark_reindexed(analyzer.db, o.f_service)

    report = analyzer.from_git("back")
    (entry,) = report.changed
    assert entry["reanalyzed"] is False and entry["ranges"] == _SHIFTED_RANGES
    assert entry["symbols_changed"] == ["OrderService", "OrderService.ping", "OrderService.cancel"]
    assert entry["seeds"] == 3 and report.stale_index is False and report.warnings == []


def test_from_git_rev_range_checks_the_index_against_the_new_side_commit(git_project):
    """Plage ``A..B`` : le côté nouveau est le commit B, pas le disque. Index bâti avant le commit → relecture du
    contenu de B (``git show``) ; index remis au niveau de B → croisement direct, même si le disque a changé depuis."""
    analyzer, root, o = git_project
    path = root / "app" / "Services" / "OrderService.php"
    path.write_text(_shifted_service_source(), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "ping")

    report = analyzer.from_git("back", "HEAD~1..HEAD")
    (entry,) = report.changed
    assert entry["reanalyzed"] is True and entry["ranges"] == _SHIFTED_RANGES
    assert entry["symbols_changed"] == ["OrderService", "OrderService.ping", "OrderService.cancel"]
    assert [e.name for e in report.endpoints] == ["POST /orders/:param/cancel"]
    assert report.stale_index is True and report.stale_files == ["app/Services/OrderService.php"]

    # réindexation au niveau de HEAD, puis le disque repart sur l'ancien contenu : B reste la référence
    fresh = [SymbolRow(name="OrderService", kind="class", line_start=6, line_end=26),
             SymbolRow(name="ping", kind="method", parent="OrderService", line_start=8, line_end=11),
             SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=13, line_end=18),
             SymbolRow(name="cancel", kind="method", parent="OrderService", line_start=20, line_end=25)]
    analyzer.db.replace_file_artifacts(o.f_service.id, symbols=fresh)
    _mark_reindexed(analyzer.db, o.f_service)
    path.write_text(SERVICE_SOURCE, encoding="utf-8")
    report = analyzer.from_git("back", "HEAD~1..HEAD")
    (entry,) = report.changed
    assert entry["reanalyzed"] is False and entry["symbols_changed"] == ["OrderService", "OrderService.ping", "OrderService.cancel"]
    assert report.stale_index is False and report.warnings == []


@pytest.mark.parametrize("broken", ["no_extractor", "unreadable"])
def test_from_git_stale_index_falls_back_to_whole_file_when_rereading_fails(git_project, monkeypatch, broken):
    """Index obsolète et relecture impossible (aucun extracteur, fichier illisible) : fichier pris en entier, avec
    avertissement, plutôt qu'un croisement incohérent."""
    import develop.impact_analysis.analyzer as mod
    from develop.code_analysis import FileAnalysis

    if broken == "no_extractor":
        monkeypatch.setattr(mod, "analyze_file",
                            lambda text, language, path="": FileAnalysis(language="php", warnings=["aucun extracteur"]))
    else:
        monkeypatch.setattr(mod, "read_text", lambda abs_path, max_bytes=0: None)
    analyzer, root, o = git_project
    (root / "app" / "Services" / "OrderService.php").write_text(_shifted_service_source(), encoding="utf-8")

    report = analyzer.from_git("back")
    (entry,) = report.changed
    assert entry["whole_file"] is True and entry["ranges"] == [] and "reanalyzed" not in entry
    assert entry["symbols_changed"] == ["OrderService", "OrderService.confirm", "OrderService.cancel"]
    assert entry["seeds"] == 4 and entry["path"] == "app/Services/OrderService.php"
    assert sorted(e.name for e in report.endpoints) == ["POST /orders/:param/cancel", "POST /orders/:param/confirm"]
    assert report.stale_index is True and report.stale_files == ["app/Services/OrderService.php"]
    assert any("en entier" in w and "Réindexez" in w for w in report.warnings)


def test_from_git_index_row_without_signature_is_trusted(git_project):
    """Une ligne d'index sans signature (jamais produite par l'indexeur) ne peut pas être vérifiée : croisement direct."""
    analyzer, root, o = git_project
    o.f_service.size, o.f_service.mtime, o.f_service.sha1 = None, None, None
    analyzer.db.upsert_file(o.f_service)
    (root / "app" / "Services" / "OrderService.php").write_text(_shifted_service_source(), encoding="utf-8")
    report = analyzer.from_git("back")
    (entry,) = report.changed
    assert entry["reanalyzed"] is False and report.stale_index is False
    assert entry["symbols_changed"] == ["OrderService", "OrderService.confirm"]  # positions de l'index, telles quelles


def test_new_side_rev_of_a_range():
    from develop.impact_analysis.analyzer import _new_side_rev

    assert _new_side_rev(None) is None and _new_side_rev("") is None and _new_side_rev("HEAD~2") is None
    assert _new_side_rev("A..B") == "B" and _new_side_rev("main...feature") == "feature"
    assert _new_side_rev("HEAD~3..") == "HEAD" and _new_side_rev("..HEAD~1") == "HEAD~1"


# --------------------------------------------------------------------------- non-régression : interfaces entre modules

def test_search_engine_returning_search_result_object_is_accepted(db: Database, graph: SimpleNamespace):
    """``SearchEngine.search`` renvoie un ``SearchResult`` (attribut ``hits``) et accepte ``source_types=`` :
    l'analyse ne doit ni l'ignorer ni avertir « Recherche lexicale indisponible »."""
    guide = DocumentRow(title="Guide d'exploitation", source_path="docs/guide.md", kind="upload", format="md")
    db.add_document(guide)
    db.replace_chunks("document", guide.id, [ChunkRow(text="En production, OrderService confirm tourne la nuit.")])
    calls: List[Dict[str, Any]] = []

    class Engine:
        def search(self, query: str, limit: int = 20, source_types=None, query_vector=None):
            calls.append({"query": query, "limit": limit, "source_types": source_types})
            hits = [FtsHit(chunk_id=1, score=1.0, snippet="…confirm…", source_type="document", source_id=guide.id),
                    FtsHit(chunk_id=2, score=0.5, snippet="", source_type="file", source_id=1)]
            return SimpleNamespace(hits=hits, query=None, warnings=[])

    report = ImpactAnalyzer(db, _workspace(), search_engine=Engine()).from_symbol(graph.s_confirm.id)
    assert report.warnings == []
    assert calls == [{"query": "OrderService.confirm", "limit": 10, "source_types": ["document", "note"]}]
    lexical = [d for d in report.documents if d.extra.get("channel") == "lexical"]
    assert [d.name for d in lexical] == ["Guide d'exploitation"] and lexical[0].confidence == 0.5


def test_repo_documentation_files_are_reported_as_documents(db: Database, graph: SimpleNamespace):
    """Un fichier de documentation d'un dépôt (``kind='doc'``, cible ``documented_by`` du linker) apparaît dans
    ``documents`` tout en restant un fichier (``type='file'``, lien vers le fichier)."""
    readme = FileRow(repo_id="back", path="docs/confirmation.md", language="markdown", kind="doc", size=10, mtime=1.0, sha1="d")
    db.upsert_file(readme)
    db.add_relations([RelationRow(src=node_key("symbol", graph.s_confirm.id), dst=node_key("file", readme.id),
                                  kind="documented_by", confidence=0.9)])
    report = ImpactAnalyzer(db, _workspace()).from_symbol(graph.s_confirm.id)
    docs = {d.name: d for d in report.documents}
    assert set(docs) == {"Spécification commandes", "docs/confirmation.md"}
    assert docs["docs/confirmation.md"].type == "file" and docs["docs/confirmation.md"].reason == "documente OrderService.confirm"
    data = report.to_dict()
    assert any(d["type"] == "file" and d["file_id"] == readme.id for d in data["documents"])
    assert len(data["affected"]) == len({i["node"] for i in data["affected"]})  # un seul exemplaire dans affected


@pytest.mark.skipif(not git_available(), reason="git absent")
def test_from_git_repo_inside_a_monorepo_checkout(tmp_path: Path, db: Database):
    """Dépôt configuré = sous-dossier d'un dépôt Git (monorepo) : chemins ramenés au dépôt, autres dossiers ignorés."""
    root = tmp_path / "mono"
    backend = root / "backend"
    (backend / "app" / "Services").mkdir(parents=True)
    (backend / "app" / "Services" / "OrderService.php").write_text(SERVICE_SOURCE, encoding="utf-8")
    (root / "frontend").mkdir()
    (root / "frontend" / "app.ts").write_text("export const a = 1;\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    db.upsert_repo(RepoRow(id="back", name="backend", path=str(backend), kind="backend"))
    db.upsert_repo(RepoRow(id="front", name="frontend", path=str(root / "frontend"), kind="frontend"))
    f_service = _file(db, "back", "app/Services/OrderService.php")
    s_confirm = SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=8, line_end=13)
    s_cancel = SymbolRow(name="cancel", kind="method", parent="OrderService", line_start=15, line_end=20)
    db.replace_file_artifacts(f_service.id, symbols=[s_confirm, s_cancel])
    _file(db, "front", "app.ts", "typescript")
    ws = WorkspaceConfig(id="ws", name="ws", repos=[RepoConfig(id="back", name="backend", path=str(backend), kind="backend"),
                                                   RepoConfig(id="front", name="frontend", path=str(root / "frontend"), kind="frontend")])
    analyzer = ImpactAnalyzer(db, ws)
    assert analyzer.from_git("back").changed == [] and analyzer.from_git("back").warnings == []

    lines = SERVICE_SOURCE.splitlines(keepends=True)
    lines[9] = "        $order->status = 'validated';\n"  # ligne 10, dans confirm()
    (backend / "app" / "Services" / "OrderService.php").write_text("".join(lines), encoding="utf-8")
    _mark_reindexed(db, f_service)  # index à jour pour ce fichier
    (root / "frontend" / "app.ts").write_text("export const a = 2;\n", encoding="utf-8")
    report = analyzer.from_git("back")
    (entry,) = report.changed  # le changement du frontend n'apparaît pas dans l'analyse du backend
    assert entry["path"] == "app/Services/OrderService.php" and entry["status"] == "modified" and entry["indexed"] is True
    assert entry["ranges"] == [[10, 10]] and entry["symbols_changed"] == ["OrderService.confirm"]
    assert report.warnings == []
    front = analyzer.from_git("front")
    assert [c["path"] for c in front.changed] == ["app.ts"] and front.changed[0]["indexed"] is True
    # plage de commits : mêmes chemins relatifs, renommage ramené au dépôt
    _git(root, "mv", "backend/app/Services/OrderService.php", "backend/app/Services/OrdersService.php")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "second")
    report = analyzer.from_git("back", "HEAD~1..HEAD")
    (entry,) = report.changed
    assert entry["status"] == "renamed" and entry["path"] == "app/Services/OrdersService.php"
    assert entry["old_path"] == "app/Services/OrderService.php" and entry["indexed"] is True


# --------------------------------------------------------------------------- chemins accentués (NFC / NFD)

_NFD_DIR = unicodedata.normalize("NFD", "Décomposé")
_NFD_FILE = unicodedata.normalize("NFD", "Résumé.php")
_NFC_REL = unicodedata.normalize("NFC", "app/Services/Décomposé/Résumé.php")
_NFC_NEW = unicodedata.normalize("NFC", "app/Services/Décomposé/Nouveauté.php")
RESUME_SOURCE = """<?php
namespace App\\Services;

class Resume
{
    public function go()
    {
        return 2;
    }
}
"""


def _index_from_scan(db: Database, repo_id: str, root: Path) -> Dict[str, FileRow]:
    """Pose dans l'index les fichiers tels que le scanner les voit (clé = ``FileEntry.path``)."""
    rows: Dict[str, FileRow] = {}
    for entry in scan_repo(str(root), repo_id).files:
        row = FileRow(repo_id=repo_id, path=entry.path, language=entry.language, kind=entry.kind,
                      size=entry.size, mtime=entry.mtime, sha1=entry.path)
        db.upsert_file(row)
        rows[entry.path] = row
    return rows


def test_within_repo_compares_prefix_and_paths_in_nfc():
    """Préfixe configuré en NFD (chemin collé depuis le Finder) et chemins git en NFC, ou l'inverse : même clé."""
    from develop.impact_analysis.analyzer import _within_repo

    nfd_prefix = unicodedata.normalize("NFD", "Équipe/backend/")
    ch = ChangedFile(path=unicodedata.normalize("NFC", "Équipe/backend/app/Résumé.php"), status="R",
                     old_path=unicodedata.normalize("NFD", "Équipe/backend/app/Ancien.php"))
    out = _within_repo(ch, nfd_prefix)
    assert out == ChangedFile(path=unicodedata.normalize("NFC", "app/Résumé.php"), status="R", old_path="app/Ancien.php")
    assert _within_repo(ChangedFile(path="autre/x.php", status="M"), nfd_prefix) is None
    plain = _within_repo(ChangedFile(path=unicodedata.normalize("NFD", "app/Résumé.php"), status="M"), "")
    assert plain.path == unicodedata.normalize("NFC", "app/Résumé.php") and plain.old_path is None


@pytest.mark.skipif(not git_available(), reason="git absent")
def test_from_git_finds_nfd_named_file_indexed_by_the_scanner(tmp_path: Path, db: Database):
    """Défaut d'origine : dépôt configuré = sous-dossier d'un dépôt Git (parcours fichier), nom accentué écrit en NFD
    sur le disque (Finder/HFS+), git renvoie la forme NFC : le scanner et l'analyse partagent la clé NFC."""
    assert _NFD_FILE != unicodedata.normalize("NFC", _NFD_FILE)
    root = tmp_path / "mono"
    backend = root / "backend"
    target = backend / "app" / "Services" / _NFD_DIR / _NFD_FILE
    target.parent.mkdir(parents=True)
    target.write_text(RESUME_SOURCE, encoding="utf-8")
    (root / "frontend").mkdir()
    (root / "frontend" / "app.ts").write_text("export const a = 1;\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    db.upsert_repo(RepoRow(id="back", name="backend", path=str(backend), kind="backend"))
    rows = _index_from_scan(db, "back", backend)
    assert list(rows) == [_NFC_REL]  # le scanner indexe la clé NFC, pas le nom NFD du disque
    f_resume = rows[_NFC_REL]
    s_class = SymbolRow(name="Resume", kind="class", line_start=4, line_end=10)
    s_go = SymbolRow(name="go", kind="method", parent="Resume", line_start=6, line_end=9)
    db.replace_file_artifacts(f_resume.id, symbols=[s_class, s_go])
    ws = WorkspaceConfig(id="ws", name="ws", repos=[RepoConfig(id="back", name="backend", path=str(backend), kind="backend")])
    analyzer = ImpactAnalyzer(db, ws)
    clean = analyzer.from_git("back")
    assert clean.changed == [] and clean.warnings == []

    target.write_text(RESUME_SOURCE.replace("return 2;", "return 3;"), encoding="utf-8")
    _mark_reindexed(db, f_resume)  # index à jour (sans décalage de lignes) : croisement direct avec l'index
    report = analyzer.from_git("back")
    (entry,) = report.changed
    assert entry["path"] == _NFC_REL and entry["indexed"] is True and entry["file_id"] == f_resume.id
    assert entry["status"] == "modified" and entry["whole_file"] is False
    assert entry["ranges"] == [[8, 8]] and entry["symbols_changed"] == ["Resume", "Resume.go"]
    assert report.stale_index is False and report.stale_files == [] and report.warnings == []
    # le détecteur de fraîcheur fondé sur le scanner (Indexer.stale_files) parle la même langue : clé NFC
    assert {e.path for e in scan_repo(str(backend), "back").files} == set(db.file_signatures("back"))
    # from_file accepte le chemin tel que l'utilisateur le colle : relatif NFD ou absolu NFD
    for given in (unicodedata.normalize("NFD", _NFC_REL), str(target)):
        by_file = analyzer.from_file("back", given)
        assert by_file.source["path"] == _NFC_REL and [c["file_id"] for c in by_file.changed] == [f_resume.id]
        assert not any("introuvable" in w for w in by_file.warnings)


@pytest.mark.skipif(not git_available(), reason="git absent")
def test_from_git_normalizes_nfd_paths_written_by_git(tmp_path: Path, db: Database):
    """Git qui ne recompose pas (``core.precomposeunicode=false``, cas Linux) renvoie les octets NFD du disque :
    chemins changés, renommages et non suivis sont ramenés en NFC, la clé de l'index."""
    root = tmp_path / "backend"
    target = root / "app" / "Services" / _NFD_DIR / _NFD_FILE
    target.parent.mkdir(parents=True)
    target.write_text(RESUME_SOURCE, encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "core.precomposeunicode", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    raw = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True,
                         env=dict(_GIT_ENV, HOME=str(root))).stdout.decode("utf-8")
    assert raw.rstrip("\0") == unicodedata.normalize("NFD", _NFC_REL)  # git écrit bien du NFD ici
    db.upsert_repo(RepoRow(id="back", name="backend", path=str(root), kind="backend"))
    f_resume = _file(db, "back", _NFC_REL)  # clé NFC, comme le scanner
    db.replace_file_artifacts(f_resume.id, symbols=[SymbolRow(name="go", kind="method", parent="Resume",
                                                              line_start=6, line_end=9)])
    ws = WorkspaceConfig(id="ws", name="ws", repos=[RepoConfig(id="back", name="backend", path=str(root), kind="backend")])
    analyzer = ImpactAnalyzer(db, ws)

    target.write_text(RESUME_SOURCE.replace("return 2;", "return 3;"), encoding="utf-8")
    _mark_reindexed(db, f_resume)  # index à jour (sans décalage de lignes) : croisement direct avec l'index
    (target.parent / unicodedata.normalize("NFD", "Nouveauté.php")).write_text("<?php\nclass Nouveaute {}\n", encoding="utf-8")
    report = analyzer.from_git("back")
    by_path = {c["path"]: c for c in report.changed}
    assert set(by_path) == {_NFC_REL, _NFC_NEW}
    changed = by_path[_NFC_REL]
    assert changed["indexed"] is True and changed["file_id"] == f_resume.id
    assert changed["ranges"] == [[8, 8]] and changed["symbols_changed"] == ["Resume.go"]
    new = by_path[_NFC_NEW]
    assert new["status"] == "untracked" and new["indexed"] is False
    assert report.stale_index is True and report.stale_files == [_NFC_NEW]
    assert all(unicodedata.normalize("NFC", w) == w for w in report.warnings)

    # renommage dans une plage de commits : ancien et nouveau chemins en NFC
    _git(root, "mv", unicodedata.normalize("NFD", _NFC_REL), "app/Services/Resume.php")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "second")
    report = analyzer.from_git("back", "HEAD~1..HEAD")
    by_path = {c["path"]: c for c in report.changed}
    renamed = by_path["app/Services/Resume.php"]
    assert renamed["status"] == "renamed" and renamed["old_path"] == _NFC_REL and renamed["indexed"] is True
    assert by_path[_NFC_NEW]["status"] == "added"
