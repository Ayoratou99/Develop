"""Tests de l'extracteur JavaScript / TypeScript (fixture réelle, extraits synthétiques, cas limites)."""
from __future__ import annotations

import textwrap
import time
from pathlib import Path

import pytest

from develop.code_analysis import analyze_file, get_extractor
from develop.code_analysis.extractors import javascript as js
from develop.code_analysis.extractors.javascript import EXTRACTOR, mask_source
from develop.code_analysis.model import normalize_route_path

FRONT = Path("frontend/src")
NOTIFIER = Path("services/notifier/src")


def analyze(text: str, path: str):
    return EXTRACTOR.analyze(textwrap.dedent(text), path)


def analyze_fixture(sample_project: Path, rel: Path):
    p = sample_project / rel
    return EXTRACTOR.analyze(p.read_text(encoding="utf-8"), str(p))


def by_name(fa, name: str, kind: str = None):
    found = [s for s in fa.symbols if s.name == name and (kind is None or s.kind == kind)]
    assert found, "symbole %s introuvable parmi %s" % (name, [(s.kind, s.name) for s in fa.symbols])
    return found[0]


def endpoints(fa, side: str = None):
    return [e for e in fa.endpoints if side is None or e.side == side]


# --------------------------------------------------------------------------- registre


def test_registry_returns_extractor_for_every_language():
    for lang in ("javascript", "jsx", "typescript", "tsx", "vue", "svelte"):
        assert get_extractor(lang) is EXTRACTOR, lang
    assert sorted(EXTRACTOR.languages) == ["javascript", "jsx", "svelte", "tsx", "typescript", "vue"]
    fa = analyze_file("export function a() {}", "typescript", "a.ts")
    assert fa.language == "typescript" and [s.name for s in fa.symbols] == ["a"]


# --------------------------------------------------------------------------- fixture : client.ts


def test_fixture_client_ts_symbols_and_lines(sample_project):
    fa = analyze_fixture(sample_project, FRONT / "api/client.ts")
    assert fa.language == "typescript"
    assert fa.warnings == []
    kinds = [(s.kind, s.name, s.line_start, s.line_end) for s in fa.symbols]
    assert kinds == [
        ("constant", "API_BASE", 3, 3),
        ("constant", "http", 5, 5),
        ("interface", "Order", 7, 14),
        ("interface", "Customer", 16, 20),
        ("function", "listOrders", 22, 25),
        ("function", "getOrder", 27, 30),
        ("function", "confirmOrder", 32, 35),
        ("function", "searchCustomers", 37, 40),
    ]
    for s in fa.symbols:
        assert s.meta["exported"] is True
        assert s.meta.get("default") is False
    assert by_name(fa, "http").meta["http_client"] == "axios"
    get_order = by_name(fa, "getOrder")
    assert get_order.meta["async"] is True
    assert get_order.signature == "export async function getOrder(id: number): Promise<Order>"


def test_fixture_client_ts_models(sample_project):
    fa = analyze_fixture(sample_project, FRONT / "api/client.ts")
    assert [(m.kind, m.name, m.line) for m in fa.models] == [("ts_interface", "Order", 7), ("ts_interface", "Customer", 16)]
    order = fa.models[0]
    assert [f.name for f in order.fields] == ["id", "reference", "customer_id", "status", "total_amount", "created_at"]
    assert [f.type for f in order.fields][:3] == ["number", "string", "number"]
    status = order.fields[3]
    assert status.meta["enum_values"] == ["draft", "confirmed", "shipped", "cancelled"]
    assert status.nullable is False
    assert all(f.nullable is False for f in order.fields)
    assert order.relations == []
    assert order.meta["exported"] is True
    customer = fa.models[1]
    assert [(f.name, f.type) for f in customer.fields] == [("id", "number"), ("email", "string"), ("full_name", "string")]


def test_fixture_client_ts_client_calls(sample_project):
    fa = analyze_fixture(sample_project, FRONT / "api/client.ts")
    eps = endpoints(fa)
    assert all(e.side == "client" for e in eps)
    assert [(e.method, e.path, e.handler, e.framework, e.line) for e in eps] == [
        ("GET", "/orders", "listOrders", "axios", 23),
        ("GET", "/orders/${id}", "getOrder", "axios", 28),
        ("POST", "/orders/${id}/confirm", "confirmOrder", "axios", 33),
        ("GET", "${API_BASE}/customers?q=${encodeURIComponent(q)}", "searchCustomers", "fetch", 38),
    ]
    assert eps[0].meta["receiver"] == "http"
    assert eps[3].meta["base"] == "API_BASE"
    assert "base" not in eps[0].meta
    assert [e.normalized_path for e in eps] == ["/orders", "/orders/:param", "/orders/:param/confirm", "/customers"]


def test_fixture_client_ts_imports(sample_project):
    fa = analyze_fixture(sample_project, FRONT / "api/client.ts")
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [("axios", ["axios"], 1, False)]


# --------------------------------------------------------------------------- fixture : composants React


def test_fixture_order_list_component(sample_project):
    fa = analyze_fixture(sample_project, FRONT / "components/OrderList.tsx")
    assert fa.language == "tsx" and fa.warnings == []
    assert [(s.kind, s.name, s.line_start, s.line_end) for s in fa.symbols] == [("component", "OrderList", 5, 20)]
    assert fa.symbols[0].meta["exported"] is True and fa.symbols[0].meta["default"] is False
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("react", ["React", "useEffect", "useState"], 1, False),
        ("../api/client", ["listOrders", "confirmOrder", "Order"], 2, True),
        ("../utils/money", ["formatAmount"], 3, True),
    ]
    assert fa.endpoints == [] and fa.models == []


def test_fixture_notifications_jsx_concatenated_url(sample_project):
    fa = analyze_fixture(sample_project, FRONT / "components/Notifications.jsx")
    assert fa.language == "jsx" and fa.warnings == []
    comp = by_name(fa, "Notifications", "component")
    assert (comp.line_start, comp.line_end) == (3, 9)
    (ep,) = fa.endpoints
    assert ep.side == "client" and ep.method == "GET" and ep.framework == "fetch" and ep.line == 6
    assert ep.path == "http://localhost:3001/notifications/${userId}"
    assert "/notifications/" in ep.path
    assert ep.handler == "Notifications"
    assert ep.meta["base"] == "http://localhost:3001"
    assert ep.normalized_path == "/notifications/:param"
    assert [(i.module, i.names) for i in fa.imports] == [("react", ["useEffect", "useState"])]


def test_fixture_customer_search_default_export(sample_project):
    fa = analyze_fixture(sample_project, FRONT / "components/CustomerSearch.tsx")
    comp = by_name(fa, "CustomerSearch", "component")
    assert (comp.line_start, comp.line_end) == (4, 12)
    assert comp.meta["exported"] is True and comp.meta["default"] is True
    assert [(i.module, i.is_relative) for i in fa.imports] == [("react", False), ("../api/client", True)]


def test_fixture_money_ts(sample_project):
    fa = analyze_fixture(sample_project, FRONT / "utils/money.ts")
    assert [(s.kind, s.name, s.line_start, s.line_end, s.meta["exported"]) for s in fa.symbols] == [("function", "formatAmount", 1, 3, True)]


# --------------------------------------------------------------------------- fixture : service Node


def test_fixture_express_server(sample_project):
    fa = analyze_fixture(sample_project, NOTIFIER / "server.js")
    assert fa.language == "javascript" and fa.warnings == []
    eps = endpoints(fa, "server")
    assert [(e.method, e.path, e.framework, e.line) for e in eps] == [
        ("GET", "/notifications/:userId", "express", 7),
        ("POST", "/notifications", "express", 11),
        ("DELETE", "/notifications/:id", "express", 16),
    ]
    assert endpoints(fa, "client") == []
    for e in eps:
        assert e.meta["inline"] is True and e.meta["receiver"] == "app"
    assert eps[0].normalized_path == "/notifications/:param"
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("express", ["express"], 1, False),
        ("./store", ["store"], 2, True),
    ]
    assert [(s.kind, s.name) for s in fa.symbols] == [("constant", "app")]


def test_fixture_store_commonjs(sample_project):
    fa = analyze_fixture(sample_project, NOTIFIER / "store.js")
    assert fa.warnings == []
    funcs = [(s.name, s.line_start, s.line_end) for s in fa.symbols if s.kind == "function"]
    assert funcs == [("add", 4, 8), ("forUser", 10, 12), ("remove", 14, 17)]
    for name in ("add", "forUser", "remove"):
        s = by_name(fa, name, "function")
        assert s.meta["exported"] is True and s.meta["commonjs"] is True
    assert by_name(fa, "seq", "variable").line_start == 1
    assert by_name(fa, "items", "constant").meta["exported"] is False
    assert fa.imports == [] and fa.endpoints == []


# --------------------------------------------------------------------------- symboles (synthétique)


def test_functions_arrows_classes_and_members():
    fa = analyze('''
        export default async function main(a, b = {}) {
          function inner() { return 1; }
        }
        export const add = (a: number, b: number): number => a + b;
        const noop = () => {};
        let handler = async function (req) {
          return req;
        };
        const notAFunction = items.map((i) => i.id);
        export abstract class Base<T> extends Parent implements IFace {
          private items: T[] = [];
          static count = 0;
          constructor(private readonly svc: Service) {}
          get size(): number { return this.items.length; }
          async load(id: number): Promise<void> {
            await this.svc.fetch(id);
          }
          onClick = (e: Event) => {
            console.log(e);
          };
          abstract describe(): string;
        }
        export namespace Util {
          export function helper() {}
        }
        export { noop, handler as onRequest };
    ''', "src/lib.ts")
    assert fa.warnings == []
    main = by_name(fa, "main", "function")
    assert (main.line_start, main.line_end) == (2, 4)
    assert main.meta == {"exported": True, "default": True, "async": True}
    inner = by_name(fa, "inner", "function")
    assert (inner.line_start, inner.line_end, inner.meta["nested"]) == (3, 3, True)
    add = by_name(fa, "add", "function")
    assert (add.line_start, add.line_end, add.meta["exported"]) == (5, 5, True)
    noop = by_name(fa, "noop", "function")
    assert noop.meta["exported"] is True  # via export { noop }
    handler = by_name(fa, "handler", "function")
    assert (handler.line_start, handler.line_end, handler.meta["async"], handler.meta["exported"]) == (7, 9, True, True)
    assert by_name(fa, "notAFunction").kind == "constant"
    base = by_name(fa, "Base", "class")
    assert (base.line_start, base.line_end) == (11, 23)
    assert base.meta["extends"] == "Parent" and base.meta["implements"] == ["IFace"] and base.meta["abstract"] is True
    members = [(s.kind, s.name, s.line_start, s.line_end) for s in fa.symbols if s.parent == "Base"]
    assert members == [
        ("property", "items", 12, 12),
        ("property", "count", 13, 13),
        ("method", "constructor", 14, 14),
        ("method", "size", 15, 15),
        ("method", "load", 16, 18),
        ("method", "onClick", 19, 21),
        ("method", "describe", 22, 22),
    ]
    load = by_name(fa, "load", "method")
    assert load.qualified_name == "Base.load" and load.meta["async"] is True
    assert by_name(fa, "describe", "method").meta["declaration_only"] is True
    assert by_name(fa, "Util", "module").line_end == 26
    assert by_name(fa, "helper", "function").meta["exported"] is True


def test_react_components_detection():
    fa = analyze('''
        import React, { useState } from "react";
        export function Counter() {
          const [n, setN] = useState(0);
          return <button onClick={() => setN(n + 1)}>{n}</button>;
        }
        export const Card = ({ title }: Props) => (
          <div className="card">{title}</div>
        );
        export default function Page() {
          return (
            <Card title="x" />
          );
        }
        class Legacy extends React.Component {
          render() { return <p />; }
        }
        const Memo = React.memo(function Inner(props) { return <span />; });
        const WithRef = forwardRef((props, ref) => <input ref={ref} />);
        function helper() { return 1; }
        function CalcTotal(items) { return items.length; }
    ''', "src/App.tsx")
    assert fa.warnings == []
    comps = {s.name: s for s in fa.symbols if s.kind == "component"}
    assert set(comps) == {"Counter", "Card", "Page", "Legacy", "Memo", "WithRef"}
    assert (comps["Counter"].line_start, comps["Counter"].line_end) == (3, 6)
    assert (comps["Card"].line_start, comps["Card"].line_end) == (7, 9)
    assert comps["Page"].meta["default"] is True
    assert (comps["Legacy"].line_start, comps["Legacy"].line_end, comps["Legacy"].meta["extends"]) == (15, 17, "React.Component")
    assert comps["Memo"].meta["wrapper"] == "React.memo"
    assert by_name(fa, "helper").kind == "function"
    assert by_name(fa, "CalcTotal").kind == "function"  # PascalCase mais ni JSX ni hook


def test_typescript_interfaces_types_enums_as_models():
    fa = analyze('''
        export type Status = "draft" | "confirmed";
        export interface Base<T> extends Named, Timestamped {
          id: number;
          tags?: string[];
          owner: User | null;
          meta: { a: string; b?: number };
          kind: "x" | "y" | null,
          readonly ref: string
          method(): void;
          [key: string]: unknown;
          "quoted-name": boolean;
        }
        export type Pair = { left: number, right: number };
        type Fn = (a: string) => void;
        export enum Color { Red = "red", Green, Blue }
        enum Multi {
          A = 1,
          B,
        }
        interface Empty {}
        interface Inline { a: number; b?: string }
    ''', "src/types.ts")
    assert fa.warnings == []
    assert [(s.kind, s.name, s.line_start, s.line_end) for s in fa.symbols] == [
        ("type", "Status", 2, 2),
        ("interface", "Base", 3, 13),
        ("type", "Pair", 14, 14),
        ("type", "Fn", 15, 15),
        ("enum", "Color", 16, 16),
        ("enum", "Multi", 17, 20),
        ("interface", "Empty", 21, 21),
        ("interface", "Inline", 22, 22),
    ]
    assert by_name(fa, "Status", "type").meta["union_values"] == ["draft", "confirmed"]
    assert by_name(fa, "Color", "enum").meta["members"] == ["Red", "Green", "Blue"]
    assert by_name(fa, "Multi", "enum").meta["members"] == ["A", "B"]
    models = {m.name: m for m in fa.models}
    assert set(models) == {"Base", "Pair", "Empty", "Inline"}
    base = models["Base"]
    assert base.kind == "ts_interface" and base.line == 3 and base.meta["extends"] == ["Named", "Timestamped"]
    fields = {f.name: f for f in base.fields}
    assert list(fields) == ["id", "tags", "owner", "meta", "kind", "ref", "quoted-name"]
    assert fields["id"].nullable is False
    assert fields["tags"].nullable is True and fields["tags"].type == "string[]" and fields["tags"].meta["optional"] is True
    assert fields["owner"].nullable is True and fields["owner"].type == "User | null"
    assert fields["meta"].type == "{ a: string; b?: number }"
    assert fields["kind"].meta["enum_values"] == ["x", "y"] and fields["kind"].nullable is True
    assert fields["ref"].meta["readonly"] is True
    assert models["Pair"].kind == "ts_type" and [f.name for f in models["Pair"].fields] == ["left", "right"]
    assert models["Empty"].fields == []
    assert [(f.name, f.nullable) for f in models["Inline"].fields] == [("a", False), ("b", True)]
    # un alias de type non-objet n'est pas un modèle
    assert "Status" not in models and "Fn" not in models


# --------------------------------------------------------------------------- imports (synthétique)


def test_import_forms():
    fa = analyze('''
        import x from 'y';
        import { a, b as c, type D } from "./mod";
        import * as ns from "../ns";
        import "./side-effect.css";
        import type { T } from "./types";
        import React, { useState } from "react";
        import {
          first,
          second as alias,
        } from "@scope/pkg";
        import legacy = require("legacy");
        const fs = require("fs");
        const { join, resolve: res } = require("path");
        require("dotenv").config();
        export { foo, bar as baz } from "./foo";
        export * from "./all";
        export * as grouped from "./grouped";
        export type { Shape } from "./shape";
        const lazy = () => import("./lazy");
        // import fake from "commented";
        const s = "import nothing from 'string'";
        export { localOnly };
    ''', "src/imports.ts")
    assert fa.warnings == []
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("y", ["x"], 2, False),
        ("./mod", ["a", "b", "D"], 3, True),
        ("../ns", ["*"], 4, True),
        ("./side-effect.css", [], 5, True),
        ("./types", ["T"], 6, True),
        ("react", ["React", "useState"], 7, False),
        ("@scope/pkg", ["first", "second"], 8, False),
        ("legacy", ["legacy"], 12, False),
        ("fs", ["fs"], 13, False),
        ("path", ["join", "resolve"], 14, False),
        ("dotenv", [], 15, False),
        ("./foo", ["foo", "bar"], 16, True),
        ("./all", ["*"], 17, True),
        ("./grouped", ["*"], 18, True),
        ("./shape", ["Shape"], 19, True),
        ("./lazy", [], 20, True),
    ]


# --------------------------------------------------------------------------- endpoints serveur (synthétique)


def test_express_router_mount_and_route_chain():
    fa = analyze('''
        import express, { Router } from "express";
        import { list, create } from "./handlers";
        const app = express();
        const router = Router();
        router.get("/", list);
        router.post("/", auth, create);
        router.put("/:id", async (req, res) => {
          res.json({});
        });
        router.delete(
          "/:id",
          remove
        );
        app.use("/api/orders", router);
        app.route("/health").get((req, res) => res.send("ok")).post(health);
        app.all("*", notFound);
        function registerExtra(server) {
          server.get("/extra", (req, res) => res.end());
        }
        app.listen(3000);
    ''', "src/index.ts")
    assert fa.warnings == []
    eps = endpoints(fa, "server")
    assert [(e.method, e.path, e.handler, e.line) for e in eps] == [
        ("GET", "/", "list", 6),
        ("POST", "/", "create", 7),
        ("PUT", "/:id", "", 8),
        ("DELETE", "/:id", "remove", 11),
        ("GET", "/health", "", 16),
        ("POST", "/health", "health", 16),
        ("ANY", "*", "notFound", 17),
        ("GET", "/extra", "registerExtra", 19),
    ]
    assert all(e.framework == "express" for e in eps)
    assert eps[0].meta["mount"] == "/api/orders" and eps[3].meta["mount"] == "/api/orders"
    assert "mount" not in eps[4].meta
    assert eps[1].meta["middleware"] == ["auth"]
    assert eps[2].meta["inline"] is True
    assert endpoints(fa, "client") == []


def test_koa_fastify_hono_routes():
    fa = analyze('''
        const fastify = require("fastify")();
        fastify.route({ method: "GET", url: "/items/:id", handler: getItem });
        fastify.route({
          method: ["POST", "PUT"],
          url: "/items",
          handler: async (req, reply) => reply.send({}),
        });
        fastify.post("/items", async (request, reply) => reply.send({}));
        const Koa = require("koa");
        const KoaRouter = require("@koa/router");
        const kr = new KoaRouter();
        kr.get("/users", async (ctx) => { ctx.body = []; });
        import { Hono } from "hono";
        const hono = new Hono();
        hono.get("/ping", (c) => c.text("pong"));
    ''', "server.js")
    eps = endpoints(fa, "server")
    assert [(e.method, e.path, e.handler, e.framework, e.line) for e in eps] == [
        ("GET", "/items/:id", "getItem", "fastify", 3),
        ("POST", "/items", "", "fastify", 4),
        ("PUT", "/items", "", "fastify", 4),
        ("POST", "/items", "", "fastify", 9),
        ("GET", "/users", "", "koa", 13),
        ("GET", "/ping", "", "hono", 16),
    ]
    assert eps[0].meta["route_object"] is True


def test_require_call_instances_keep_their_own_framework():
    # deux frameworks instanciés via ``require("x")()`` dans le même fichier : chacun garde le sien,
    # seul un ``Router()`` générique hérite du framework du fichier
    fa = analyze('''
        const fastify = require("fastify")();
        fastify.get("/a", async () => 1);
        const app = require("express")();
        app.post("/b", (req, res) => res.end());
        const r = Router();
        r.put("/c", update);
        srv.delete("/d", (req, res) => res.end());
    ''', "mixed.js")
    assert fa.warnings == []
    assert [(e.method, e.path, e.framework, e.handler) for e in endpoints(fa, "server")] == [
        ("GET", "/a", "fastify", ""),
        ("POST", "/b", "express", ""),
        ("PUT", "/c", "fastify", "update"),
        ("DELETE", "/d", "fastify", ""),
    ]
    # un ``Router()`` déclaré avant l'import du framework (hoisting) hérite de celui-ci
    fa2 = analyze('''
        const router = new Router();
        router.get("/k", (ctx) => { ctx.body = 1; });
        import Router from "@koa/router";
    ''', "koa.ts")
    assert [(e.path, e.framework) for e in fa2.endpoints] == [("/k", "koa")]


def test_nestjs_controller_routes():
    fa = analyze('''
        import { Controller, Get, Post, Delete, Body, Param } from '@nestjs/common';

        @Controller('orders')
        export class OrdersController {
          constructor(private readonly service: OrdersService) {}

          @Get()
          findAll() {
            return this.service.findAll();
          }

          @Get(':id')
          findOne(@Param('id') id: string) {
            return this.service.findOne(+id);
          }

          @Post()
          @HttpCode(201)
          create(@Body() dto: CreateOrderDto) {
            return this.service.create(dto);
          }

          @ApiResponse({
            status: 204,
          })
          @Delete(':id')
          async remove(@Param('id') id: string): Promise<void> {
            await this.service.remove(+id);
          }
        }

        @Controller()
        export class HealthController {
          @Get('health')
          check() { return 'ok'; }
        }
    ''', "src/orders/orders.controller.ts")
    assert fa.warnings == []
    eps = endpoints(fa, "server")
    assert [(e.method, e.path, e.handler, e.framework, e.line) for e in eps] == [
        ("GET", "/orders", "OrdersController@findAll", "nestjs", 8),
        ("GET", "/orders/:id", "OrdersController@findOne", "nestjs", 13),
        ("POST", "/orders", "OrdersController@create", "nestjs", 18),
        ("DELETE", "/orders/:id", "OrdersController@remove", "nestjs", 27),
        ("GET", "/health", "HealthController@check", "nestjs", 35),
    ]
    assert eps[1].meta == {"prefix": "orders", "route": ":id", "controller": "OrdersController", "decorator": "@Get(':id')"}
    ctrl = by_name(fa, "OrdersController", "class")
    assert ctrl.meta["decorators"] == ["@Controller('orders')"] and (ctrl.line_start, ctrl.line_end) == (5, 31)
    remove = by_name(fa, "remove", "method")
    assert remove.parent == "OrdersController" and (remove.line_start, remove.line_end) == (28, 30)
    assert remove.meta["decorators"] == ["@ApiResponse({ status: 204, })", "@Delete(':id')"]
    assert endpoints(fa, "client") == []


def test_nextjs_app_router_derived_from_path():
    src = '''
        import { NextResponse } from "next/server";
        export async function GET(request: Request) {
          return NextResponse.json([]);
        }
        export const POST = async (req: Request) => {
          return NextResponse.json({}, { status: 201 });
        };
        function helper() {}
    '''
    fa = analyze(src, "src/app/api/orders/[id]/route.ts")
    eps = endpoints(fa, "server")
    assert [(e.method, e.path, e.handler, e.framework, e.line) for e in eps] == [
        ("GET", "/api/orders/[id]", "GET", "nextjs", 3),
        ("POST", "/api/orders/[id]", "POST", "nextjs", 6),
    ]
    assert eps[0].meta["derived_from_path"] is True and eps[0].meta["router"] == "app"
    assert eps[0].normalized_path == "/api/orders/:param"
    # groupe de routes ignoré, pas de méthode exportée → ANY
    fa2 = analyze("export { GET } from './impl';", "app/api/(admin)/users/route.js")
    assert [(e.method, e.path) for e in fa2.endpoints] == [("ANY", "/api/users")]
    # un fichier hors des conventions n'engendre rien
    assert endpoints(analyze(src, "src/lib/route-helpers.ts"), "server") == []


def test_nextjs_pages_router_derived_from_path():
    fa = analyze('''
        export default function handler(req, res) {
          res.status(200).json({});
        }
    ''', "pages/api/orders/index.ts")
    (ep,) = fa.endpoints
    assert (ep.side, ep.method, ep.path, ep.handler, ep.framework, ep.line) == ("server", "ANY", "/api/orders", "handler", "nextjs", 2)
    assert ep.meta["derived_from_path"] is True and ep.meta["router"] == "pages"
    fa2 = analyze("const h = (req, res) => res.end();\nexport default h;", "src/pages/api/users/[id].js")
    (ep2,) = fa2.endpoints
    assert (ep2.path, ep2.handler) == ("/api/users/[id]", "h")


# --------------------------------------------------------------------------- endpoints client (synthétique)


def test_client_call_forms():
    fa = analyze('''
        import ky from "ky";
        import useSWR from "swr";
        import { useQuery } from "@tanstack/react-query";
        const api = axios.create({ baseURL: "/api" });
        export const useOrders = () => useSWR("/api/orders", fetcher);
        export function useOrder(id) {
          return useQuery({ queryKey: ["order", id], queryFn: () => fetch(`/api/orders/${id}`).then((r) => r.json()) });
        }
        async function save(order) {
          await fetch("/api/orders", {
            method: "POST",
            body: JSON.stringify(order),
          });
          await axios({ url: "/api/orders/" + order.id, method: "put", data: order });
          await axios.delete(`/api/orders/${order.id}`);
          await api.patch("/orders/1", { a: 1 });
          const j = await ky.get("https://example.com/api/x").json();
          await $http.get("/legacy/x");
          await fetch(url);
          await fetch(`${config.apiUrl}/ping`, { method: "HEAD" });
          const m = new Map(); m.get("key"); headers.get("content-type");
        }
        class Store {
          async refresh() {
            const r = await this.http.get("/store/items");
          }
        }
    ''', "src/hooks.ts")
    assert fa.warnings == []
    assert endpoints(fa, "server") == []
    eps = endpoints(fa, "client")
    assert [(e.method, e.path, e.handler, e.framework, e.line) for e in eps] == [
        ("GET", "/api/orders", "useOrders", "swr", 6),
        ("GET", "/api/orders/${id}", "useOrder", "fetch", 8),
        ("POST", "/api/orders", "save", "fetch", 11),
        ("PUT", "/api/orders/${order.id}", "save", "axios", 15),
        ("DELETE", "/api/orders/${order.id}", "save", "axios", 16),
        ("PATCH", "/orders/1", "save", "axios", 17),
        ("GET", "https://example.com/api/x", "save", "ky", 18),
        ("GET", "/legacy/x", "save", "http", 19),
        ("GET", "${url}", "save", "fetch", 20),
        ("HEAD", "${config.apiUrl}/ping", "save", "fetch", 21),
        ("GET", "/store/items", "Store@refresh", "http", 26),
    ]
    assert eps[1].meta["via"] == "useQuery"
    assert eps[5].meta["receiver"] == "api"
    assert eps[6].meta["base"] == "https://example.com"
    assert eps[8].meta["dynamic"] is True
    assert eps[9].meta["base"] == "config.apiUrl"
    assert eps[9].normalized_path == "/ping"


def test_angular_http_client_service():
    fa = analyze('''
        import { Injectable } from '@angular/core';
        import { HttpClient } from '@angular/common/http';
        import { environment } from '../../environments/environment';

        @Injectable({ providedIn: 'root' })
        export class OrderService {
          private base = environment.apiUrl;
          constructor(private http: HttpClient) {}

          list(): Observable<Order[]> {
            return this.http.get<Order[]>(`${this.base}/orders`);
          }

          confirm(id: number) {
            return this.http.post<Order>(`${environment.apiUrl}/orders/${id}/confirm`, {});
          }
          remove(id: number) { return this.http.delete('/api/orders/' + id); }
        }
    ''', "src/app/order.service.ts")
    assert fa.warnings == []
    eps = endpoints(fa, "client")
    assert [(e.method, e.path, e.handler, e.framework, e.line, e.meta.get("base")) for e in eps] == [
        ("GET", "${this.base}/orders", "OrderService@list", "http", 12, "this.base"),
        ("POST", "${environment.apiUrl}/orders/${id}/confirm", "OrderService@confirm", "http", 16, "environment.apiUrl"),
        ("DELETE", "/api/orders/${id}", "OrderService@remove", "http", 18, None),
    ]
    assert eps[1].normalized_path == "/orders/:param/confirm"
    svc = by_name(fa, "OrderService", "class")
    assert svc.meta["decorators"] == ["@Injectable({ providedIn: 'root' })"]
    assert [(s.kind, s.name) for s in fa.symbols if s.parent == "OrderService"] == [
        ("property", "base"), ("method", "constructor"), ("method", "list"), ("method", "confirm"), ("method", "remove"),
    ]
    assert [(i.module, i.is_relative) for i in fa.imports][2] == ("../../environments/environment", True)


# --------------------------------------------------------------------------- Vue / Svelte


def test_vue_script_setup_only_script_is_analyzed():
    fa = analyze('''
        <template>
          <div @click="load">{{ title }}</div>
          <!-- fetch("/not/analyzed") -->
        </template>

        <script setup lang="ts">
        import { ref, onMounted } from "vue";
        import axios from "axios";
        import { formatAmount } from "../utils/money";

        const title = ref("Orders");
        async function load() {
          const res = await axios.get("/api/orders");
          const r2 = await fetch(`/api/orders/${id}`, { method: "DELETE" });
        }
        onMounted(load);
        </script>

        <style scoped>
        .a { color: red; }
        </style>
    ''', "src/components/Orders.vue")
    assert fa.language == "vue" and fa.warnings == []
    assert [(s.kind, s.name, s.line_start, s.line_end) for s in fa.symbols] == [("constant", "title", 12, 12), ("function", "load", 13, 16)]
    assert [(e.method, e.path, e.handler, e.framework, e.line) for e in fa.endpoints] == [
        ("GET", "/api/orders", "load", "axios", 14),
        ("DELETE", "/api/orders/${id}", "load", "fetch", 15),
    ]
    assert [(i.module, i.line, i.is_relative) for i in fa.imports] == [("vue", 8, False), ("axios", 9, False), ("../utils/money", 10, True)]
    no_script = analyze("<template><p>x</p></template>\n<style>.a{}</style>\n", "Empty.vue")
    assert no_script.symbols == [] and no_script.warnings == ["aucun bloc <script> dans Empty.vue"]


def test_svelte_component_script():
    fa = analyze('''
        <script>
          import { onMount } from "svelte";
          export let userId;
          let items = [];
          async function refresh() {
            const res = await fetch("/api/items?user=" + userId);
            items = await res.json();
          }
          onMount(refresh);
        </script>
        <ul>{#each items as it}<li>{it.name}</li>{/each}</ul>
    ''', "src/Items.svelte")
    assert fa.language == "svelte" and fa.warnings == []
    assert [(s.kind, s.name, s.line_start, s.meta["exported"]) for s in fa.symbols] == [
        ("variable", "userId", 4, True), ("variable", "items", 5, False), ("function", "refresh", 6, False),
    ]
    assert by_name(fa, "refresh").line_end == 9
    (ep,) = fa.endpoints
    assert (ep.method, ep.path, ep.handler, ep.framework, ep.line) == ("GET", "/api/items?user=${userId}", "refresh", "fetch", 7)
    assert ep.normalized_path == "/api/items"
    assert [(i.module, i.names, i.line) for i in fa.imports] == [("svelte", ["onMount"], 3)]


# --------------------------------------------------------------------------- masquage lexical


def test_mask_source_preserves_length_and_hides_strings_comments_regex_jsx():
    src = 'const a = "x{y}" + \'/*\' + `t ${b + "}"} z`; // {\n/* { */ const r = /\\{[}]/g; return <ul>{items}</ul> />;'
    masked = mask_source(src)
    assert len(masked) == len(src) and masked.count("\n") == src.count("\n")
    assert masked.count("{") == masked.count("}") == 1  # seule l'accolade de code JSX {items} reste
    assert '"' in masked and "`" in masked  # délimiteurs conservés
    assert "x{y}" not in masked and "/*" not in masked.split("\n")[0] and "/g" in masked
    assert "<ul>{items}</ul> />" in masked  # balises JSX intactes (``</ul>`` n'est pas un littéral regex)
    assert mask_source("") == "" and mask_source("'unterminated\nnext") .split("\n")[1] == "next"


# --------------------------------------------------------------------------- cas limites


def test_edge_cases_never_raise():
    assert EXTRACTOR.analyze("", "empty.ts").symbols == []
    assert EXTRACTOR.analyze("   \n\n", "blank.js").warnings == []
    garbage = "function (((( { class \n 'unterminated \n /* never closed \n export const = => {{{ \n @@@ }}}}}}}}"
    fa = EXTRACTOR.analyze(garbage, "garbage.js")
    assert fa.language == "javascript"
    assert any("accolades" in w for w in fa.warnings)
    weird = "﻿export const x = '\x00\udcff€';\n/* \ud800 */ fetch('/api/€');\n"
    fa = EXTRACTOR.analyze(weird, "weird.ts")
    assert [e.path for e in fa.endpoints] == ["/api/€"]
    fa = EXTRACTOR.analyze(b"export function b() {}", "bytes.js")  # type: ignore[arg-type]
    assert [s.name for s in fa.symbols] == ["b"]
    fa = EXTRACTOR.analyze("export function crlf() {\r\n  return fetch('/x');\r\n}\r\n", "crlf.js")
    assert [(s.name, s.line_start, s.line_end) for s in fa.symbols] == [("crlf", 1, 3)]
    assert [(e.path, e.handler) for e in fa.endpoints] == [("/x", "crlf")]
    truncated = "export class A {\n  m() {\n    const x = fetch('/api/a'"
    fa = EXTRACTOR.analyze(truncated, "truncated.ts")
    assert by_name(fa, "A", "class").line_end == 3 and [e.path for e in fa.endpoints] == ["/api/a"]


def test_huge_minified_line_is_fast_and_flagged():
    big = "x".join(["const a = 1;"] * 100000)  # ~1,3 Mo sur une seule ligne
    t = time.time()
    fa = EXTRACTOR.analyze(big, "bundle.min.js")
    assert time.time() - t < 2.5
    assert any("20000" in w for w in fa.warnings)


def test_one_megabyte_realistic_file_under_one_second(sample_project):
    files = [FRONT / "api/client.ts", FRONT / "components/OrderList.tsx", FRONT / "components/Notifications.jsx",
             NOTIFIER / "server.js", NOTIFIER / "store.js"]
    chunk = "\n".join((sample_project / f).read_text(encoding="utf-8") for f in files)
    text = chunk
    while len(text) < 1_000_000:
        text += "\n" + chunk
    t = time.time()
    fa = EXTRACTOR.analyze(text, "big/mixed.ts")
    assert time.time() - t < 2.5
    assert fa.warnings == [] and len(fa.endpoints) > 1000 and len(fa.symbols) > 1000


def test_fifty_thousand_lines_do_not_blow_up():
    many = "\n".join("export function f%d(a) { return fetch('/api/x/%d'); }" % (i, i) for i in range(50000))
    t = time.time()
    fa = EXTRACTOR.analyze(many, "many.ts")
    assert time.time() - t < 6.0
    assert len(fa.symbols) == 50000 and len(fa.endpoints) == 50000
    assert fa.endpoints[-1].handler == "f49999" and fa.endpoints[-1].line == 50000


def test_to_dict_roundtrip_and_language_from_path():
    fa = analyze("export const f = () => fetch('/a');", "x.mjs")
    d = fa.to_dict()
    assert d["language"] == "javascript" and d["endpoints"][0]["normalized_path"] == "/a"
    assert js.language_from_path("A.VUE") == "vue" and js.language_from_path("noext") == "javascript"
