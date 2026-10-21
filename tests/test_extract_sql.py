"""Tests de l'extracteur SQL : fixture réelle, extraits par dialecte, cas limites et performance."""
from __future__ import annotations

import json
import time
from pathlib import Path

from develop.code_analysis import analyze_file, supported_languages
from develop.code_analysis.extractors.sql import EXTRACTOR
from develop.code_analysis.registry import get_extractor


def _analyze(text: str, path: str = "x.sql"):
    return EXTRACTOR.analyze(text, path)


def _field(model, name):
    for f in model.fields:
        if f.name == name:
            return f
    raise AssertionError("champ %s absent de %s (%s)" % (name, model.name, [f.name for f in model.fields]))


# --------------------------------------------------------------------------- fixture réelle


def test_fixture_schema_tables_and_lines(sample_project: Path):
    text = (sample_project / "database" / "schema.sql").read_text(encoding="utf-8")
    fa = _analyze(text, "database/schema.sql")
    assert fa.language == "sql"
    assert fa.warnings == [] and fa.symbols == [] and fa.endpoints == [] and fa.imports == []
    assert [m.name for m in fa.models] == ["customers", "orders", "order_lines"]
    assert [m.kind for m in fa.models] == ["sql_table"] * 3
    assert [m.table_name for m in fa.models] == ["customers", "orders", "order_lines"]
    assert [m.line for m in fa.models] == [2, 9, 18]
    assert [m.meta["line_end"] for m in fa.models] == [7, 16, 24]
    assert all(m.meta["indexes"] == 0 for m in fa.models)
    assert all("schema" not in m.meta for m in fa.models)
    json.dumps(fa.to_dict())  # sérialisable


def test_fixture_customers_fields(sample_project: Path):
    fa = _analyze((sample_project / "database" / "schema.sql").read_text(encoding="utf-8"))
    customers = fa.models[0]
    assert [f.name for f in customers.fields] == ["id", "email", "full_name", "phone"]
    cid = _field(customers, "id")
    assert cid.type == "BIGINT" and cid.nullable is False
    assert cid.meta.get("primary") is True and cid.meta.get("auto_increment") is True
    assert cid.meta["line"] == 3
    email = _field(customers, "email")
    assert email.type == "VARCHAR(255)" and email.nullable is False and email.meta.get("unique") is True
    assert email.meta["line"] == 4
    phone = _field(customers, "phone")
    assert phone.type == "VARCHAR(30)" and phone.nullable is True and phone.meta["line"] == 6
    assert customers.relations == []


def test_fixture_orders_fields_enum_and_fk(sample_project: Path):
    fa = _analyze((sample_project / "database" / "schema.sql").read_text(encoding="utf-8"))
    orders = fa.models[1]
    assert [f.name for f in orders.fields] == ["id", "reference", "customer_id", "status", "total_amount",
                                               "confirmed_at"]
    status = _field(orders, "status")
    assert status.type == "ENUM('draft','confirmed','shipped','cancelled')"
    assert status.meta["enum_values"] == ["draft", "confirmed", "shipped", "cancelled"]
    assert status.default == "'draft'" and status.nullable is False and status.meta["line"] == 13
    reference = _field(orders, "reference")
    assert reference.nullable is False and reference.meta.get("unique") is True and reference.type == "VARCHAR(20)"
    confirmed = _field(orders, "confirmed_at")
    assert confirmed.nullable is True and confirmed.type == "TIMESTAMP" and confirmed.default is None
    total = _field(orders, "total_amount")
    assert total.type == "BIGINT UNSIGNED" and total.meta.get("unsigned") is True and total.default == "0"
    customer_id = _field(orders, "customer_id")
    assert customer_id.nullable is False
    assert customer_id.meta["references"] == {"table": "customers", "columns": ["id"]}
    assert "on_delete" not in customer_id.meta
    assert [(r.kind, r.target, r.field) for r in orders.relations] == [("foreign_key", "customers", "customer_id")]
    assert orders.meta["foreign_keys"] == [{"columns": ["customer_id"], "table": "customers", "ref_columns": ["id"]}]


def test_fixture_order_lines_on_delete_cascade(sample_project: Path):
    fa = _analyze((sample_project / "database" / "schema.sql").read_text(encoding="utf-8"))
    lines = fa.models[2]
    assert [f.name for f in lines.fields] == ["id", "order_id", "product_sku", "quantity", "unit_price"]
    assert [(r.kind, r.target, r.field) for r in lines.relations] == [("foreign_key", "orders", "order_id")]
    order_id = _field(lines, "order_id")
    assert order_id.meta["on_delete"] == "CASCADE" and order_id.meta["references"]["table"] == "orders"
    assert lines.meta["foreign_keys"][0]["on_delete"] == "CASCADE"
    assert _field(lines, "quantity").type == "INT UNSIGNED"
    assert _field(lines, "unit_price").meta["line"] == 23


def test_registry_wiring(sample_project: Path):
    assert get_extractor("sql") is EXTRACTOR
    assert "sql" in supported_languages()
    assert EXTRACTOR.languages == ["sql"]
    fa = analyze_file((sample_project / "database" / "schema.sql").read_text(encoding="utf-8"), "sql", "s.sql")
    assert [m.name for m in fa.models] == ["customers", "orders", "order_lines"]


# --------------------------------------------------------------------------- PostgreSQL

PG = '''-- PostgreSQL
CREATE TYPE order_status AS ENUM ('draft', 'confirmed');
CREATE TABLE public."Customers" (
    id SERIAL PRIMARY KEY,
    "Email" character varying(255) NOT NULL,
    tags text[] DEFAULT '{}'::text[],
    balance NUMERIC(10, 2) DEFAULT 0.00,
    created_at TIMESTAMP(6) WITH TIME ZONE DEFAULT now() NOT NULL,
    status order_status DEFAULT 'draft'
);

CREATE TABLE IF NOT EXISTS public.orders (
    id bigint GENERATED ALWAYS AS IDENTITY,
    customer_id integer,
    seq integer DEFAULT nextval('orders_seq'::regclass) NOT NULL,
    price double precision,
    CONSTRAINT orders_pkey PRIMARY KEY (id),
    CONSTRAINT orders_customer_fk FOREIGN KEY (customer_id) REFERENCES public."Customers" (id) ON DELETE SET NULL ON UPDATE CASCADE,
    UNIQUE (seq),
    CHECK (price > 0)
);

ALTER TABLE ONLY public.orders
    ADD CONSTRAINT fk_other FOREIGN KEY (customer_id) REFERENCES customers(id) ON DELETE CASCADE;
ALTER TABLE orders ADD COLUMN note text NULL, ADD COLUMN IF NOT EXISTS qty int NOT NULL DEFAULT 1;
ALTER TABLE ONLY orders ADD CONSTRAINT pk2 PRIMARY KEY (id);

CREATE OR REPLACE VIEW public.order_summary AS
  SELECT o.id, c."Email" FROM public.orders o JOIN public."Customers" c ON c.id = o.customer_id;

CREATE UNIQUE INDEX customers_email_idx ON public."Customers" (lower("Email"));
CREATE INDEX ON orders USING btree (customer_id, created_at DESC);

CREATE OR REPLACE FUNCTION public.touch(p_id integer, p_note text DEFAULT 'x;y)')
RETURNS integer LANGUAGE plpgsql AS $$
BEGIN
  UPDATE orders SET note = p_note; -- ; in body )
  RETURN 1;
END;
$$;

CREATE TRIGGER orders_touch
  BEFORE INSERT OR UPDATE OF status, note ON public.orders
  FOR EACH ROW EXECUTE FUNCTION public.touch();
'''


def test_postgres_tables():
    fa = _analyze(PG)
    assert fa.warnings == []
    assert [(m.name, m.line, m.meta.get("alter", False)) for m in fa.models] == [
        ("Customers", 3, False), ("orders", 12, False), ("orders", 23, True), ("orders", 25, True), ("orders", 26, True)]
    customers = fa.models[0]
    assert customers.table_name == "Customers" and customers.meta["schema"] == "public"
    assert customers.meta["line_end"] == 10
    cid = _field(customers, "id")
    assert cid.type == "SERIAL" and cid.meta.get("auto_increment") is True and cid.meta.get("primary") is True
    assert _field(customers, "Email").type == "character varying(255)"
    assert _field(customers, "Email").nullable is False
    tags = _field(customers, "tags")
    assert tags.type == "text[]" and tags.default == "'{}'::text[]"
    assert _field(customers, "balance").type == "NUMERIC(10, 2)" and _field(customers, "balance").default == "0.00"
    created = _field(customers, "created_at")
    assert created.type == "TIMESTAMP(6) WITH TIME ZONE" and created.default == "now()" and created.nullable is False
    assert _field(customers, "status").type == "order_status"

    orders = fa.models[1]
    assert orders.meta["schema"] == "public" and orders.meta["line_end"] == 21
    oid = _field(orders, "id")
    assert oid.meta.get("identity") is True and oid.meta.get("auto_increment") is True
    assert oid.meta.get("primary") is True and oid.nullable is False
    customer_id = _field(orders, "customer_id")
    assert customer_id.meta["references"] == {"table": "Customers", "columns": ["id"], "schema": "public"}
    assert customer_id.meta["on_delete"] == "SET NULL" and customer_id.meta["on_update"] == "CASCADE"
    assert [(r.kind, r.target, r.field) for r in orders.relations] == [("foreign_key", "Customers", "customer_id")]
    seq = _field(orders, "seq")
    assert seq.meta.get("unique") is True and seq.default == "nextval('orders_seq'::regclass)" and seq.nullable is False
    assert _field(orders, "price").type == "double precision"
    assert orders.meta["primary_key"] == ["id"] and orders.meta["unique_keys"] == [["seq"]]
    assert orders.meta["checks"] == 1 and orders.meta["indexes"] == 0
    assert orders.meta["foreign_keys"][0]["constraint"] == "orders_customer_fk"


def test_postgres_alter_table():
    fa = _analyze(PG)
    fk_alter, add_cols, add_pk = fa.models[2], fa.models[3], fa.models[4]
    assert fk_alter.fields == [] and fk_alter.meta["alter"] is True and fk_alter.meta["schema"] == "public"
    assert [(r.kind, r.target, r.field) for r in fk_alter.relations] == [("foreign_key", "customers", "customer_id")]
    assert fk_alter.meta["foreign_keys"][0]["on_delete"] == "CASCADE"
    assert fk_alter.meta["foreign_keys"][0]["constraint"] == "fk_other"

    assert [f.name for f in add_cols.fields] == ["note", "qty"] and add_cols.relations == []
    note = _field(add_cols, "note")
    assert note.type == "text" and note.nullable is True and note.meta["alter"] is True and note.meta["line"] == 25
    qty = _field(add_cols, "qty")
    assert qty.type == "int" and qty.nullable is False and qty.default == "1"

    pk = _field(add_pk, "id")
    assert pk.meta.get("primary") is True and pk.meta["alter"] is True and add_pk.meta["primary_key"] == ["id"]


def test_postgres_symbols():
    fa = _analyze(PG)
    by_name = {s.name: s for s in fa.symbols}
    assert [s.name for s in fa.symbols] == ["order_status", "order_summary", "customers_email_idx", "orders_idx",
                                            "touch", "orders_touch"]
    enum = by_name["order_status"]
    assert enum.kind == "enum" and enum.line_start == 2 and enum.meta["values"] == ["draft", "confirmed"]

    view = by_name["order_summary"]
    assert view.kind == "view" and (view.line_start, view.line_end) == (28, 29)
    assert view.meta["schema"] == "public" and view.meta["tables"] == ["public.orders", "public.Customers"]
    assert view.signature.startswith("CREATE OR REPLACE VIEW public.order_summary AS SELECT")

    idx = by_name["customers_email_idx"]
    assert idx.kind == "variable" and idx.meta["index"] is True and idx.meta["unique"] is True
    assert idx.meta["table"] == "Customers" and idx.meta["schema"] == "public"
    assert idx.meta["columns"] == ['lower("Email")'] and idx.line_start == 31
    anon = by_name["orders_idx"]
    assert anon.meta["anonymous"] is True and anon.meta["columns"] == ["customer_id", "created_at DESC"]
    assert anon.meta["unique"] is False and anon.line_start == 32

    fn = by_name["touch"]
    assert fn.kind == "function" and (fn.line_start, fn.line_end) == (34, 40)
    assert fn.signature == "touch(p_id integer, p_note text DEFAULT 'x;y)') RETURNS integer"
    assert fn.meta == {"routine": "function", "schema": "public", "returns": "integer", "language": "plpgsql"}

    trg = by_name["orders_touch"]
    assert trg.kind == "function" and (trg.line_start, trg.line_end) == (42, 44)
    assert trg.meta["routine"] == "trigger" and trg.meta["timing"] == "BEFORE"
    assert trg.meta["events"] == ["INSERT", "UPDATE"] and trg.meta["table"] == "orders"
    assert trg.meta["function"] == "public.touch"


# --------------------------------------------------------------------------- MySQL / MariaDB

MYSQL = '''# MySQL dump
/*!40101 SET NAMES utf8 */;
CREATE TABLE `shop`.`products` (
  `id` int(11) unsigned NOT NULL AUTO_INCREMENT,
  `sku` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL COMMENT 'ref; (x)',
  `price` decimal(10,2) NOT NULL DEFAULT '0.00',
  `kind` enum('a','b','it''s') DEFAULT NULL,
  `key` int(11) DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `category_id` int unsigned,
  PRIMARY KEY (`id`),
  UNIQUE KEY `sku_unique` (`sku`),
  KEY `idx_price` (`price`),
  FULLTEXT KEY ft (sku),
  CONSTRAINT `fk_cat` FOREIGN KEY (`category_id`) REFERENCES `categories` (`id`) ON DELETE CASCADE ON UPDATE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COMMENT='Produits';
INSERT INTO `products` VALUES (1, 'a;b)', 2.5, 'a', NULL, NULL, 1);
DELIMITER $$
CREATE DEFINER=`root`@`localhost` PROCEDURE `restock`(IN p_id INT)
BEGIN
  UPDATE products SET price = price; -- x
  SELECT 1;
END$$
DELIMITER ;
CREATE TABLE after_proc (id INT);
ALTER TABLE products ADD INDEX idx_kind (kind), ADD FOREIGN KEY (`key`) REFERENCES keys(id);
'''


def test_mysql_table():
    fa = _analyze(MYSQL)
    assert fa.warnings == []
    assert [(m.name, m.line) for m in fa.models] == [("products", 3), ("after_proc", 25), ("products", 26)]
    p = fa.models[0]
    assert p.meta["schema"] == "shop" and p.meta["engine"] == "InnoDB" and p.meta["comment"] == "Produits"
    assert p.meta["indexes"] == 3 and p.meta["line_end"] == 16
    assert p.meta["primary_key"] == ["id"] and p.meta["unique_keys"] == [["sku"]]
    assert [f.name for f in p.fields] == ["id", "sku", "price", "kind", "key", "updated_at", "category_id"]
    pid = _field(p, "id")
    assert pid.type == "int(11) unsigned" and pid.meta.get("unsigned") and pid.meta.get("auto_increment")
    assert pid.meta.get("primary") is True and pid.nullable is False
    sku = _field(p, "sku")
    assert sku.type == "varchar(40)" and sku.nullable is False and sku.meta.get("unique") is True
    assert sku.meta["comment"] == "ref; (x)" and sku.meta["collation"] == "utf8mb4_unicode_ci"
    price = _field(p, "price")
    assert price.type == "decimal(10,2)" and price.default == "'0.00'"
    kind = _field(p, "kind")
    assert kind.meta["enum_values"] == ["a", "b", "it's"] and kind.default == "NULL" and kind.nullable is None
    key = _field(p, "key")  # colonne nommée key, pas un index
    assert key.type == "int(11)" and "primary" not in key.meta
    upd = _field(p, "updated_at")
    assert upd.nullable is True and upd.default == "CURRENT_TIMESTAMP" and upd.meta["on_update_expr"] == "CURRENT_TIMESTAMP"
    cat = _field(p, "category_id")
    assert cat.type == "int unsigned" and cat.meta["on_delete"] == "CASCADE" and cat.meta["on_update"] == "RESTRICT"
    assert [(r.kind, r.target, r.field) for r in p.relations] == [("foreign_key", "categories", "category_id")]
    assert p.meta["foreign_keys"][0]["constraint"] == "fk_cat"


def test_mysql_delimiter_procedure_and_alter():
    fa = _analyze(MYSQL)
    assert [(s.name, s.kind, s.line_start, s.line_end) for s in fa.symbols] == [("restock", "function", 19, 23)]
    proc = fa.symbols[0]
    assert proc.meta == {"routine": "procedure"} and proc.signature == "restock(IN p_id INT)"
    alter = fa.models[2]
    assert alter.meta["alter"] is True and alter.meta["indexes"] == 1 and alter.fields == []
    assert [(r.kind, r.target, r.field) for r in alter.relations] == [("foreign_key", "keys", "key")]


# --------------------------------------------------------------------------- SQLite / SQL Server

SQLITE = '''CREATE TABLE IF NOT EXISTS "users" (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  age,
  parent_id INTEGER REFERENCES users(id) ON DELETE SET NULL
) WITHOUT ROWID;
.read ./other.sql
CREATE TABLE t2 (a);
'''


def test_sqlite():
    fa = _analyze(SQLITE)
    assert fa.warnings == []
    assert [(m.name, m.line, m.meta["line_end"]) for m in fa.models] == [("users", 1, 6), ("t2", 8, 8)]
    users = fa.models[0]
    uid = _field(users, "id")
    assert uid.type == "INTEGER" and uid.meta.get("primary") and uid.meta.get("auto_increment")
    assert _field(users, "age").type == "" and _field(users, "age").nullable is None
    parent = _field(users, "parent_id")
    assert parent.meta["on_delete"] == "SET NULL" and parent.meta["references"]["table"] == "users"
    assert [(r.target, r.field) for r in users.relations] == [("users", "parent_id")]
    assert [(i.module, i.line, i.is_relative) for i in fa.imports] == [("./other.sql", 7, True)]


MSSQL = '''CREATE TABLE [dbo].[Orders] (
  [Id] INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
  [Note] NVARCHAR(MAX) NULL,
  [CustomerId] INT NOT NULL CONSTRAINT FK_Orders_Customers FOREIGN KEY REFERENCES [dbo].[Customers]([Id])
);
GO
CREATE VIEW dbo.v AS SELECT * FROM dbo.Orders;
GO
'''


def test_sql_server():
    fa = _analyze(MSSQL)
    assert fa.warnings == []
    orders = fa.models[0]
    assert orders.name == "Orders" and orders.meta["schema"] == "dbo" and orders.meta["line_end"] == 5
    oid = _field(orders, "Id")
    assert oid.meta.get("identity") and oid.meta.get("auto_increment") and oid.meta.get("primary")
    assert _field(orders, "Note").type == "NVARCHAR(MAX)" and _field(orders, "Note").nullable is True
    cust = _field(orders, "CustomerId")
    assert "primary" not in cust.meta and cust.meta["references"] == {"table": "Customers", "columns": ["Id"], "schema": "dbo"}
    assert orders.meta["foreign_keys"][0]["constraint"] == "FK_Orders_Customers"
    assert [(s.kind, s.name, s.line_start, s.meta["tables"]) for s in fa.symbols] == [("view", "v", 7, ["dbo.Orders"])]


# --------------------------------------------------------------------------- heuristiques diverses

MISC = '''CREATE TEMPORARY TABLE tmp_x (id INT);
CREATE TABLE copy_x AS SELECT * FROM tmp_x;
CREATE TABLE like_x LIKE tmp_x;
CREATE MATERIALIZED VIEW mv AS SELECT id FROM tmp_x;
CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER VIEW `v2` AS select 1;
CREATE FUNCTION f2(a int) RETURNS TABLE(id int, n text) AS 'select 1' LANGUAGE sql;
CREATE TYPE pt AS (x int, y int);
CREATE TABLE idx_vs_col (
  key INT NOT NULL,
  index VARCHAR(255),
  KEY idx_a (key),
  INDEX (index)
);
INSERT INTO idx_vs_col VALUES (1, 'x');
UPDATE idx_vs_col SET key = 2;
DROP TABLE tmp_x;
'''


def test_misc_create_forms():
    fa = _analyze(MISC)
    assert fa.warnings == []
    models = {m.name: m for m in fa.models}
    assert models["tmp_x"].meta["temporary"] is True
    assert models["copy_x"].meta["as_select"] is True and models["copy_x"].fields == []
    assert models["like_x"].meta["like"] == "tmp_x"
    ivc = models["idx_vs_col"]
    assert [f.name for f in ivc.fields] == ["key", "index"] and ivc.meta["indexes"] == 2
    assert _field(ivc, "key").type == "INT" and _field(ivc, "index").type == "VARCHAR(255)"
    syms = {s.name: s for s in fa.symbols}
    assert syms["mv"].kind == "view" and syms["mv"].meta["materialized"] is True and syms["mv"].meta["tables"] == ["tmp_x"]
    assert syms["v2"].kind == "view" and syms["v2"].line_start == 5
    assert syms["f2"].kind == "function" and syms["f2"].meta["returns"] == "TABLE (id int, n text)"
    assert syms["f2"].meta["language"] == "sql"
    assert syms["pt"].kind == "type"
    assert len(fa.models) == 4  # INSERT / UPDATE / DROP ignorés


# --------------------------------------------------------------------------- commentaires et chaînes

DUMP = """CREATE TABLE a (id INT, s VARCHAR(10) DEFAULT 'x;y)' -- comment ; )
, /* ; ) */ t INT);
INSERT INTO a VALUES (1, ');', 2);
INSERT INTO a VALUES (2, '-- not a comment /* nor this */', 3);
CREATE TABLE b (id INT);
-- CREATE TABLE commented_out (id INT);
/* CREATE TABLE
   also_commented (id INT); */
CREATE TABLE c (id INT,
  name TEXT"""


def test_comments_strings_and_truncated_table():
    fa = _analyze(DUMP)
    assert [(m.name, m.line) for m in fa.models] == [("a", 1), ("b", 5), ("c", 9)]
    a = fa.models[0]
    assert [f.name for f in a.fields] == ["id", "s", "t"]
    assert _field(a, "s").default == "'x;y)'" and _field(a, "t").meta["line"] == 2
    assert a.meta["line_end"] == 2
    c = fa.models[2]
    assert [f.name for f in c.fields] == ["id", "name"] and _field(c, "name").meta["line"] == 10
    assert fa.warnings == ["CREATE TABLE c (ligne 9) : corps tronqué"]


# --------------------------------------------------------------------------- cas limites


def test_empty_and_whitespace():
    for text in ("", "   \n\n\t", ";;;", "-- rien", "/* rien */"):
        fa = _analyze(text)
        assert fa.models == [] and fa.symbols == [] and fa.warnings == []


def test_invalid_syntax_never_raises():
    samples = [
        "CREATE TABLE (((;;; SELECT ;;; ALTER",
        "CREATE TABLE",
        "CREATE TABLE x (",
        "CREATE TABLE x (id INT) ) ) ;",
        "ALTER TABLE ADD ADD ADD;",
        "CREATE VIEW ;;; CREATE INDEX ON ; CREATE FUNCTION ( ) RETURNS",
        "CREATE TRIGGER t BEFORE ON; CREATE TYPE AS ENUM (",
        "\x00\x01\x02 CREATE TABLE \xff (\xfe INT);",
        "DELIMITER\nDELIMITER $$\nCREATE TABLE y (id INT)$$\nDELIMITER ;\n",
    ]
    for text in samples:
        fa = _analyze(text)
        assert isinstance(fa.warnings, list)
        json.dumps(fa.to_dict())
    assert _analyze(None, "none.sql").models == []  # type: ignore[arg-type]


def test_bom_nul_and_unicode():
    fa = _analyze("﻿CREATE TABLE été (prénom VARCHAR(10));")
    assert [(m.name, m.line) for m in fa.models] == [("été", 1)]
    assert [f.name for f in fa.models[0].fields] == ["prénom"]
    fa = _analyze("CREATE TABLE x (a INT);\nINSERT INTO x VALUES ('oops")
    assert [m.name for m in fa.models] == ["x"] and fa.warnings == ["chaîne non terminée à la ligne 2"]
    fa = _analyze("CREATE TABLE x (a INT);\n/* jamais fermé\nCREATE TABLE y (b INT);")
    assert [m.name for m in fa.models] == ["x"] and fa.warnings == ["commentaire non terminé à la ligne 2"]


def test_unbalanced_parentheses_forced_split():
    fa = _analyze("CREATE TABLE x (a INT, b VARCHAR(10;\nCREATE TABLE y (b INT);")
    assert [m.name for m in fa.models] == ["x", "y"]
    assert any("parenthèses non équilibrées" in w for w in fa.warnings)


def test_huge_line_and_many_lines():
    huge = "INSERT INTO x VALUES " + ",".join("(%d,'v;)')" % i for i in range(120000)) + ";\nCREATE TABLE after (id INT);"
    assert len(huge) > 1_000_000
    t0 = time.perf_counter()
    fa = _analyze(huge)
    assert time.perf_counter() - t0 < 2.5
    assert [(m.name, m.line) for m in fa.models] == [("after", 2)] and fa.warnings == []

    many_lines = "\n".join("-- ligne %d" % i for i in range(50000)) + "\nCREATE TABLE z (id INT);"
    fa = _analyze(many_lines)
    assert fa.models[0].line == 50001 and fa.models[0].meta["line_end"] == 50001


def test_pathological_inputs_stay_linear():
    for text in ("'" * 200000, "$$" * 100000, "(" * 200000, ")" * 200000, "--" * 100000, "/*" * 50000, ";" * 200000):
        t0 = time.perf_counter()
        fa = _analyze(text + "\nCREATE TABLE p (id INT);")
        assert time.perf_counter() - t0 < 2.5
        assert isinstance(fa.models, list)


def test_one_megabyte_dump_under_one_second():
    tables = "\n".join(
        "CREATE TABLE t%d (id INT PRIMARY KEY, ref INT REFERENCES t%d(id), name VARCHAR(100) NOT NULL DEFAULT 'x');"
        % (i, i) for i in range(2000))
    inserts = "\n".join("INSERT INTO t0 VALUES (%d, %d, 'nom ; ) -- %d');" % (i, i, i) for i in range(15000))
    text = tables + "\n" + inserts + "\nCREATE TABLE t_last (id INT);\n"
    assert len(text) > 1_000_000
    t0 = time.perf_counter()
    fa = _analyze(text)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, elapsed
    assert len(fa.models) == 2001 and fa.models[-1].name == "t_last" and fa.warnings == []
    assert fa.models[-1].line == 2000 + 15000 + 1
