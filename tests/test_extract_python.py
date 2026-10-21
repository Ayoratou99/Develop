"""Tests de l'extracteur Python (``develop.code_analysis.extractors.python``)."""
from __future__ import annotations

import textwrap
import time
from pathlib import Path

from develop.code_analysis.extractors.python import EXTRACTOR, PythonExtractor
from develop.code_analysis.model import FileAnalysis
from develop.code_analysis.registry import get_extractor


def analyze(src: str, path: str = "x.py") -> FileAnalysis:
    """Analyse un extrait dédenté ; la première ligne de code est la ligne 1."""
    return EXTRACTOR.analyze(textwrap.dedent(src).lstrip("\n"), path)


def by_name(fa: FileAnalysis, name: str, kind: str = None, parent: str = None):
    for s in fa.symbols:
        if s.name == name and (kind is None or s.kind == kind) and (parent is None or s.parent == parent):
            return s
    raise AssertionError("symbole %s introuvable dans %s" % (name, [(s.kind, s.name) for s in fa.symbols]))


def eps(fa: FileAnalysis):
    return [(e.side, e.method, e.path, e.handler, e.framework, e.line) for e in fa.endpoints]


# --------------------------------------------------------------------------- fixture réelle


def test_registry_returns_python_extractor():
    assert get_extractor("python") is EXTRACTOR
    assert isinstance(EXTRACTOR, PythonExtractor)
    assert EXTRACTOR.languages == ["python"]


def test_fixture_report_imports(sample_project: Path):
    src = (sample_project / "tools" / "report.py").read_text(encoding="utf-8")
    fa = EXTRACTOR.analyze(src, "tools/report.py")
    assert fa.language == "python"
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("csv", [], 2, False),
        ("json", [], 3, False),
        ("dataclasses", ["dataclass"], 4, False),
        ("typing", ["List"], 5, False),
    ]


def test_fixture_report_symbols(sample_project: Path):
    src = (sample_project / "tools" / "report.py").read_text(encoding="utf-8")
    fa = EXTRACTOR.analyze(src, "tools/report.py")
    assert [(s.kind, s.name, s.line_start, s.line_end, s.parent) for s in fa.symbols] == [
        ("class", "OrderRow", 9, 12, None),
        ("function", "load_orders", 15, 17, None),
        ("function", "export_csv", 20, 25, None),
        ("class", "ReportBuilder", 28, 33, None),
        ("method", "__init__", 29, 30, "ReportBuilder"),
        ("method", "total", 32, 33, "ReportBuilder"),
    ]
    assert by_name(fa, "load_orders").signature == "def load_orders(path: str) -> List[OrderRow]"
    assert by_name(fa, "export_csv").signature == "def export_csv(rows: List[OrderRow], out: str) -> None"
    assert by_name(fa, "__init__").signature == "def __init__(self, rows: List[OrderRow])"
    assert by_name(fa, "total").signature == "def total(self) -> int"
    assert by_name(fa, "total").qualified_name == "ReportBuilder.total"
    assert by_name(fa, "OrderRow").meta["decorators"] == ["dataclass"]
    assert by_name(fa, "OrderRow").signature == "class OrderRow"
    assert by_name(fa, "ReportBuilder").meta == {"bases": [], "decorators": []}
    assert by_name(fa, "load_orders").meta == {"async": False, "decorators": []}
    m = by_name(fa, "total")
    assert m.meta["staticmethod"] is False and m.meta["classmethod"] is False and m.meta["property"] is False


def test_fixture_report_dataclass_model(sample_project: Path):
    src = (sample_project / "tools" / "report.py").read_text(encoding="utf-8")
    fa = EXTRACTOR.analyze(src, "tools/report.py")
    assert len(fa.models) == 1
    m = fa.models[0]
    assert (m.name, m.kind, m.table_name, m.line) == ("OrderRow", "dataclass", None, 9)
    assert [(f.name, f.type, f.nullable, f.default) for f in m.fields] == [
        ("reference", "str", False, None),
        ("status", "str", False, None),
        ("total_amount", "int", False, None),
    ]
    assert m.relations == []
    assert fa.endpoints == []
    assert fa.warnings == []
    # to_dict reste sérialisable
    d = fa.to_dict()
    assert d["models"][0]["fields"][0]["name"] == "reference"


# --------------------------------------------------------------------------- FastAPI


FASTAPI_SRC = '''
from fastapi import FastAPI, APIRouter
from .schemas import OrderIn

app = FastAPI()
router = APIRouter(prefix="/orders", tags=["orders"])
api_router = APIRouter()

@app.get("/health")
def health():
    return {"ok": True}

@router.post("/", status_code=201, response_model=OrderIn)
async def create_order(payload: OrderIn):
    return payload

@router.get("/{order_id}")
async def get_order(order_id: int):
    ...

@api_router.delete("/customers/{customer_id}")
def delete_customer(customer_id: int):
    ...

@app.api_route("/multi", methods=["GET", "POST"])
def multi(): ...

@app.websocket("/ws")
async def ws(socket): ...
'''


def test_fastapi_routes():
    fa = analyze(FASTAPI_SRC)
    assert eps(fa) == [
        ("server", "GET", "/health", "health", "fastapi", 8),
        ("server", "POST", "/", "create_order", "fastapi", 12),
        ("server", "GET", "/{order_id}", "get_order", "fastapi", 16),
        ("server", "DELETE", "/customers/{customer_id}", "delete_customer", "fastapi", 20),
        ("server", "GET", "/multi", "multi", "fastapi", 24),
        ("server", "POST", "/multi", "multi", "fastapi", 24),
        ("server", "ANY", "/ws", "ws", "fastapi", 27),
    ]
    create = fa.endpoints[1]
    assert create.meta == {"receiver": "router", "prefix": "/orders", "status_code": 201, "response_model": "OrderIn"}
    assert fa.endpoints[2].meta == {"receiver": "router", "prefix": "/orders"}
    assert fa.endpoints[2].normalized_path == "/:param"
    assert fa.endpoints[3].meta == {"receiver": "api_router"}
    assert fa.endpoints[6].meta["websocket"] is True
    sym = by_name(fa, "create_order")
    assert sym.meta["async"] is True
    assert sym.meta["decorators"] == ["router.post('/', status_code=201, response_model=OrderIn)"]
    assert sym.signature == "async def create_order(payload: OrderIn)"
    assert [(i.module, i.names, i.is_relative) for i in fa.imports] == [
        ("fastapi", ["FastAPI", "APIRouter"], False),
        (".schemas", ["OrderIn"], True),
    ]
    assert by_name(fa, "app", "variable").signature == "app = FastAPI()"
    assert fa.warnings == []


# --------------------------------------------------------------------------- Flask


FLASK_SRC = '''
from flask import Flask, Blueprint
app = Flask(__name__)
bp = Blueprint("orders", __name__, url_prefix="/orders")

@app.route("/orders", methods=["GET", "POST"])
def orders():
    pass

@bp.get("/<int:order_id>")
def show_order(order_id):
    pass

@app.route("/ping")
def ping(): pass

class OrderAPI(MethodView):
    def get(self, id): pass

app.add_url_rule("/api/orders/<id>", view_func=OrderAPI.as_view("orders_api"), methods=["GET", "PUT"])

def create_app():
    app2 = Flask(__name__)
    @app2.route("/factory")
    def inner(): pass
    return app2
'''


def test_flask_routes():
    fa = analyze(FLASK_SRC)
    assert eps(fa) == [
        ("server", "GET", "/orders", "orders", "flask", 5),
        ("server", "POST", "/orders", "orders", "flask", 5),
        ("server", "GET", "/<int:order_id>", "show_order", "flask", 9),
        ("server", "GET", "/ping", "ping", "flask", 13),
        ("server", "GET", "/api/orders/<id>", "OrderAPI", "flask", 19),
        ("server", "PUT", "/api/orders/<id>", "OrderAPI", "flask", 19),
        ("server", "GET", "/factory", "inner", "flask", 23),
    ]
    assert fa.endpoints[0].meta == {"methods": ["GET", "POST"], "receiver": "app"}
    assert fa.endpoints[2].meta == {"receiver": "bp", "prefix": "/orders"}
    assert fa.endpoints[2].normalized_path == "/:param"
    assert fa.endpoints[4].meta["view"] == "OrderAPI"
    # la méthode get de la MethodView n'est pas confondue avec un appel client
    assert all(e.side == "server" for e in fa.endpoints)
    assert by_name(fa, "get", "method", "OrderAPI").line_start == 17


# --------------------------------------------------------------------------- Django


DJANGO_URLS_SRC = '''
from django.urls import path, re_path, include
from . import views
from .views import OrderListView
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register("orders", views.OrderViewSet, basename="orders")

urlpatterns = [
    path("orders/", views.order_list, name="order-list"),
    path("orders/<int:pk>/", OrderListView.as_view(), name="order-detail"),
    re_path(r"^legacy/(?P<pk>[0-9]+)/$", views.legacy),
    path("api/", include("shop.api_urls")),
]
urlpatterns += [path("extra/", views.extra)]
'''


def test_django_urlpatterns():
    fa = analyze(DJANGO_URLS_SRC, "shop/urls.py")
    assert eps(fa) == [
        ("server", "ANY", "orders", "views.OrderViewSet", "django", 7),
        ("server", "ANY", "orders/", "order_list", "django", 10),
        ("server", "ANY", "orders/<int:pk>/", "OrderListView", "django", 11),
        ("server", "ANY", "^legacy/(?P<pk>[0-9]+)/$", "legacy", "django", 12),
        ("server", "ANY", "api/", "", "django", 13),
        ("server", "ANY", "extra/", "extra", "django", 15),
    ]
    assert fa.endpoints[0].meta == {"receiver": "router", "viewset": True, "basename": "orders"}
    assert fa.endpoints[1].meta == {"view": "views.order_list", "name": "order-list"}
    assert fa.endpoints[2].normalized_path == "/orders/:param"
    assert fa.endpoints[3].meta["regex"] is True
    assert fa.endpoints[4].meta == {"include": "shop.api_urls"}
    assert [(i.module, i.names, i.is_relative) for i in fa.imports][:3] == [
        ("django.urls", ["path", "re_path", "include"], False),
        (".", ["views"], True),
        (".views", ["OrderListView"], True),
    ]


def test_django_path_without_django_context_is_ignored():
    fa = analyze('''
    from pathlib import Path
    def path(a, b):
        return a
    x = path("orders/", "y")
    ''')
    assert fa.endpoints == []


DJANGO_MODELS_SRC = '''
from django.db import models

class Customer(models.Model):
    name = models.CharField(max_length=80, unique=True)

class Order(models.Model):
    reference = models.CharField(max_length=20, unique=True)
    note = models.TextField(null=True, blank=True)
    total_amount = models.IntegerField(default=0)
    customer = models.ForeignKey("Customer", on_delete=models.CASCADE, related_name="orders")
    tags = models.ManyToManyField("shop.Tag")
    parent = models.OneToOneField("self", null=True, on_delete=models.SET_NULL)

    class Meta:
        db_table = "commandes"

    def __str__(self):
        return self.reference

class Base(models.Model):
    class Meta:
        abstract = True
'''


def test_django_models():
    fa = analyze(DJANGO_MODELS_SRC, "apps/shop/models.py")
    assert [(m.name, m.kind, m.table_name, m.line) for m in fa.models] == [
        ("Customer", "django", "shop_customer", 3),
        ("Order", "django", "commandes", 6),
        ("Base", "django", None, 20),
    ]
    order = fa.models[1]
    assert [(f.name, f.type, f.nullable, f.default) for f in order.fields] == [
        ("reference", "CharField", False, None),
        ("note", "TextField", True, None),
        ("total_amount", "IntegerField", False, "0"),
        ("customer", "ForeignKey", False, None),
        ("tags", "ManyToManyField", False, None),
        ("parent", "OneToOneField", True, None),
    ]
    assert order.fields[0].meta == {"length": 20, "unique": True}
    assert order.fields[1].meta == {"blank": True}
    assert [(r.kind, r.target, r.field) for r in order.relations] == [
        ("belongs_to", "Customer", "customer"),
        ("many_to_many", "Tag", "tags"),
        ("has_one", "Order", "parent"),
    ]
    assert order.fields[3].meta["related_name"] == "orders"
    assert fa.models[2].meta == {"abstract": True}
    # la classe Meta imbriquée et __str__ restent des symboles rattachés à Order
    assert by_name(fa, "Meta", "class", "Order").line_start == 14
    assert by_name(fa, "__str__", "method", "Order").line_start == 17
    # sans chemin exploitable, pas de nom de table deviné
    fa2 = analyze(DJANGO_MODELS_SRC, "")
    assert fa2.models[0].table_name is None and fa2.models[1].table_name == "commandes"


# --------------------------------------------------------------------------- SQLAlchemy


SQLA_SRC = '''
from typing import List, Optional
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column, DeclarativeBase

class Base(DeclarativeBase):
    pass

class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    name = Column("full_name", String(80), nullable=False, unique=True)
    orders = relationship("Order", back_populates="customer")

class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    reference: Mapped[str] = mapped_column(String(20), unique=True, default="X")
    note: Mapped[Optional[str]]
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"))
    customer: Mapped["Customer"] = relationship(back_populates="orders")
    lines: Mapped[List["OrderLine"]] = relationship(back_populates="order")
    tags = relationship("Tag", secondary="order_tags")
    profile = relationship("Profile", uselist=False)

class Legacy(Base):
    __tablename__ = "legacy"
    id = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=True)
    customer = relationship("Customer")

orders_table = Table("orders_core", metadata, Column("id", Integer, primary_key=True), Column("ref", String(10)))
'''


def test_sqlalchemy_models():
    fa = analyze(SQLA_SRC)
    assert [(m.name, m.kind, m.table_name, m.line) for m in fa.models] == [
        ("Base", "sqlalchemy", None, 5),
        ("Customer", "sqlalchemy", "customers", 8),
        ("Order", "sqlalchemy", "orders", 14),
        ("Legacy", "sqlalchemy", "legacy", 25),
        ("orders_core", "sqlalchemy", "orders_core", 31),
    ]
    customer = fa.models[1]
    assert [(f.name, f.type, f.nullable, f.meta) for f in customer.fields] == [
        ("id", "Integer", False, {"primary": True}),
        ("name", "String(80)", False, {"column": "full_name", "unique": True}),
    ]
    assert [(r.kind, r.target, r.field) for r in customer.relations] == [("has_many", "Order", "orders")]
    order = fa.models[2]
    assert [(f.name, f.type, f.nullable, f.default) for f in order.fields] == [
        ("id", "int", False, None),
        ("reference", "str", False, "'X'"),
        ("note", "str", True, None),
        ("customer_id", "int", False, None),
    ]
    assert order.fields[3].meta == {"foreign_key": "customers.id"}
    assert [(r.kind, r.target, r.field) for r in order.relations] == [
        ("foreign_key", "customers", "customer_id"),
        ("belongs_to", "Customer", "customer"),
        ("has_many", "OrderLine", "lines"),
        ("many_to_many", "Tag", "tags"),
        ("has_one", "Profile", "profile"),
    ]
    legacy = fa.models[3]
    assert legacy.fields[1].nullable is True
    assert [(r.kind, r.target, r.field) for r in legacy.relations] == [
        ("foreign_key", "customers", "customer_id"),
        ("belongs_to", "Customer", "customer"),
    ]
    core = fa.models[4]
    assert core.meta == {"style": "core"}
    assert [(f.name, f.type) for f in core.fields] == [("id", "Integer"), ("ref", "String(10)")]


# --------------------------------------------------------------------------- pydantic / dataclass / schema


PYDANTIC_SRC = '''
import pydantic
from pydantic import BaseModel, Field
from typing import Optional, List, ClassVar

class OrderIn(BaseModel):
    reference: str
    qty: int = 1
    note: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    customer_id: int = Field(..., alias="customerId")
    price: float = Field(0.0, ge=0)
    model_config = {"extra": "forbid"}
    kind: ClassVar[str] = "x"

    @pydantic.validator("reference")
    def check_ref(cls, v):
        return v

class Settings(pydantic.BaseSettings):
    url: str = "http://localhost"

class Point(TypedDict):
    x: int
'''


def test_pydantic_models():
    fa = analyze(PYDANTIC_SRC)
    assert [(m.name, m.kind, m.line, m.meta) for m in fa.models] == [
        ("OrderIn", "pydantic", 5, {"base": "BaseModel"}),
        ("Settings", "pydantic", 19, {"base": "BaseSettings"}),
        ("Point", "schema", 22, {"base": "TypedDict"}),
    ]
    order = fa.models[0]
    assert [(f.name, f.type, f.nullable, f.default) for f in order.fields] == [
        ("reference", "str", False, None),
        ("qty", "int", False, "1"),
        ("note", "Optional[str]", True, "None"),
        ("tags", "List[str]", False, "list()"),
        ("customer_id", "int", False, None),
        ("price", "float", False, "0.0"),
    ]
    assert order.fields[4].meta == {"alias": "customerId"}
    assert fa.models[1].fields[0].default == "'http://localhost'"
    validator = by_name(fa, "check_ref", "method", "OrderIn")
    assert validator.meta["decorators"] == ["pydantic.validator('reference')"]
    # aucun appel client ne doit être inventé depuis le validator ou les Field()
    assert fa.endpoints == []


DATACLASS_SRC = '''
from dataclasses import dataclass, field
import dataclasses
from typing import ClassVar, Optional

@dataclass(frozen=True)
class Money:
    amount: int = 0
    currency: str = field(default="EUR")
    tags: list = field(default_factory=list)
    note: Optional[str] = None
    registry: ClassVar[dict] = {}

@dataclasses.dataclass
class Other:
    x: int
'''


def test_dataclass_models():
    fa = analyze(DATACLASS_SRC)
    assert [(m.name, m.kind, m.line, m.meta) for m in fa.models] == [
        ("Money", "dataclass", 6, {"frozen": True}),
        ("Other", "dataclass", 14, {}),
    ]
    assert [(f.name, f.type, f.nullable, f.default) for f in fa.models[0].fields] == [
        ("amount", "int", False, "0"),
        ("currency", "str", False, "'EUR'"),
        ("tags", "list", False, "list()"),
        ("note", "Optional[str]", True, "None"),
    ]
    assert by_name(fa, "Money").meta["decorators"] == ["dataclass(frozen=True)"]
    assert by_name(fa, "Other").meta["decorators"] == ["dataclasses.dataclass"]


# --------------------------------------------------------------------------- appels clients


CLIENT_SRC = '''
import requests
import httpx
from urllib.request import urlopen, Request

BASE = "http://localhost:8000/api"

def fetch_orders(oid):
    r = requests.get(f"{BASE}/orders/{oid}", timeout=3)
    requests.post("http://localhost:8000/api/orders", json={})
    requests.request("DELETE", url=BASE + "/orders/" + str(oid))
    return r

class ApiClient:
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "http://api.example.com"

    def list(self):
        return self.session.get(self.base_url + "/orders")

    def put_order(self, order):
        return self.session.put("%s/orders/%s" % (self.base_url, order.id))

async def fetch_httpx():
    async with httpx.AsyncClient() as client:
        resp = await client.get("/orders")
        await client.post("{}/customers".format(BASE))

def raw():
    urlopen("https://example.com/api/items")
    urlopen(Request(BASE + "/x", data=b"", method="POST"))

def untyped(session):
    session.patch("/orders/1")

d = {}
d.get("a/b")
cache.get("key")
'''


def test_client_calls_requests_httpx_urllib():
    fa = analyze(CLIENT_SRC)
    assert eps(fa) == [
        ("client", "GET", "{BASE}/orders/{oid}", "fetch_orders", "requests", 8),
        ("client", "POST", "http://localhost:8000/api/orders", "fetch_orders", "requests", 9),
        ("client", "DELETE", "{BASE}/orders/{str(oid)}", "fetch_orders", "requests", 10),
        ("client", "GET", "{self.base_url}/orders", "ApiClient@list", "requests", 19),
        ("client", "PUT", "%s/orders/%s", "ApiClient@put_order", "requests", 22),
        ("client", "GET", "/orders", "fetch_httpx", "httpx", 26),
        ("client", "POST", "{}/customers", "fetch_httpx", "httpx", 27),
        ("client", "GET", "https://example.com/api/items", "raw", "urllib", 30),
        ("client", "POST", "{BASE}/x", "raw", "urllib", 31),
        ("client", "PATCH", "/orders/1", "untyped", "requests", 34),
    ]
    e0 = fa.endpoints[0]
    assert e0.meta == {"receiver": "requests", "base": "BASE", "base_value": "http://localhost:8000/api"}
    assert e0.normalized_path == "/orders/:param"
    assert fa.endpoints[1].meta == {"receiver": "requests", "base": "http://localhost:8000"}
    assert fa.endpoints[1].normalized_path == "/api/orders"
    assert fa.endpoints[3].meta == {"receiver": "self.session", "base": "self.base_url"}
    assert fa.endpoints[5].meta == {"receiver": "client"}
    assert fa.endpoints[7].meta == {"base": "https://example.com"}
    assert by_name(fa, "BASE", "constant").signature == 'BASE = "http://localhost:8000/api"'


def test_client_call_at_module_level_and_aliased_import():
    fa = analyze('''
    import requests as rq
    resp = rq.get("https://api.example.com/v1/items?page=2")
    ''')
    assert eps(fa) == [("client", "GET", "https://api.example.com/v1/items?page=2", "<module>", "requests", 2)]
    assert fa.endpoints[0].normalized_path == "/v1/items"


# --------------------------------------------------------------------------- aiohttp (serveur et client)


AIOHTTP_SRC = '''
import aiohttp
from aiohttp import web

routes = web.RouteTableDef()

@routes.get("/items")
async def items(request):
    pass

async def handler(request): pass

app = web.Application()
app.add_routes([web.post("/create", handler), web.route("PUT", "/put", handler)])
app.router.add_get("/get", handler)
app.router.add_route("DELETE", "/del", handler)

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

async def typed(session: aiohttp.ClientSession):
    await session.post("/typed")
'''


def test_aiohttp_server_and_client():
    fa = analyze(AIOHTTP_SRC)
    assert eps(fa) == [
        ("server", "GET", "/items", "items", "aiohttp", 6),
        ("server", "PUT", "/put", "handler", "aiohttp", 13),
        ("server", "POST", "/create", "handler", "aiohttp", 13),
        ("server", "GET", "/get", "handler", "aiohttp", 14),
        ("server", "DELETE", "/del", "handler", "aiohttp", 15),
        ("client", "GET", "{url}", "fetch", "aiohttp", 19),
        ("client", "POST", "/typed", "typed", "aiohttp", 23),
    ]
    assert fa.endpoints[5].meta == {"receiver": "session", "base": "url"}
    assert by_name(fa, "items").meta["async"] is True


# --------------------------------------------------------------------------- imports, constantes, symboles


SYMBOLS_SRC = '''
from __future__ import annotations
import os.path
import a, b as bb
from . import sibling
from ..pkg.mod import alpha as al, beta
from .local import *
from enum import Enum

MAX = 10
__version__ = "1.0"
logger = None
x, y = 1, 2
MAX = 11

class Color(Enum):
    RED = 1
    GREEN = 2

class Svc:
    """Service.

    Détails.
    """
    @staticmethod
    def s(): pass
    @classmethod
    def c(cls): pass
    @property
    def p(self): return 1
    @p.setter
    def p(self, v): pass
    class Inner:
        def m(self): pass

try:
    import ujson as json
except ImportError:
    import json

if True:
    def conditional(a, *args, k: int = 1, **kw) -> None:
        """Doc de conditional."""
'''


def test_imports_relative_and_aliases():
    fa = analyze(SYMBOLS_SRC)
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [
        ("__future__", ["annotations"], 1, False),
        ("os.path", [], 2, False),
        ("a", [], 3, False),
        ("b", [], 3, False),
        (".", ["sibling"], 4, True),
        ("..pkg.mod", ["alpha", "beta"], 5, True),
        (".local", ["*"], 6, True),
        ("enum", ["Enum"], 7, False),
        ("ujson", [], 36, False),
        ("json", [], 38, False),
    ]


def test_symbols_constants_variables_enum_methods():
    fa = analyze(SYMBOLS_SRC)
    assert [(s.kind, s.name, s.line_start) for s in fa.symbols if s.kind in ("constant", "variable")] == [
        ("constant", "MAX", 9),
        ("variable", "__version__", 10),
        ("variable", "logger", 11),
        ("variable", "x", 12),
        ("variable", "y", 12),
    ]
    color = by_name(fa, "Color")
    assert color.kind == "enum" and color.meta["members"] == ["RED", "GREEN"] and color.signature == "class Color(Enum)"
    assert (color.line_start, color.line_end) == (15, 17)
    svc = by_name(fa, "Svc")
    assert (svc.line_start, svc.line_end) == (19, 33)
    assert svc.meta["docstring"] == "Service."
    assert by_name(fa, "s", "method", "Svc").meta["staticmethod"] is True
    assert by_name(fa, "c", "method", "Svc").meta["classmethod"] is True
    props = [s for s in fa.symbols if s.name == "p" and s.parent == "Svc"]
    assert [p.line_start for p in props] == [29, 31] and all(p.meta["property"] for p in props)
    assert props[1].meta["decorators"] == ["p.setter"]
    inner = by_name(fa, "Inner", "class")
    assert inner.parent == "Svc" and by_name(fa, "m", "method", "Inner").qualified_name == "Inner.m"
    cond = by_name(fa, "conditional", "function")
    assert (cond.line_start, cond.line_end) == (41, 42)
    assert cond.signature == "def conditional(a, *args, k: int = 1, **kw) -> None"
    assert cond.meta["docstring"] == "Doc de conditional."
    # les symboles sont triés par ligne
    assert [s.line_start for s in fa.symbols] == sorted(s.line_start for s in fa.symbols)


def test_module_variables_limited_to_fifty():
    src = "\n".join("var_%d = %d" % (i, i) for i in range(80)) + "\nCONST_A = 1\nCONST_B = 2\n"
    fa = analyze(src)
    variables = [s for s in fa.symbols if s.kind == "variable"]
    assert len(variables) == 50 and variables[-1].name == "var_49"
    assert [s.name for s in fa.symbols if s.kind == "constant"] == ["CONST_A", "CONST_B"]
    assert any("50" in w for w in fa.warnings)


def test_signature_formatting_variants():
    fa = analyze('''
    def f(a: int = 1, /, b=2, *c, d: str = "x", e, **kw) -> Optional[List[int]]:
        pass
    async def g(self, y=-1, z=[1, 2], *, k: Dict[str, int] = {}) -> "Foo":
        pass
    ''')
    assert by_name(fa, "f").signature == "def f(a: int = 1, /, b=2, *c, d: str = 'x', e, **kw) -> Optional[List[int]]"
    assert by_name(fa, "g").signature == "async def g(self, y=-1, z=[1, 2], *, k: Dict[str, int] = {}) -> 'Foo'"


# --------------------------------------------------------------------------- cas limites et repli regex


INVALID_SRC = '''
import os
from .x import y

def ok():
    return 1

def broken(:
    pass

class Foo(Base, Mixin):
    """doc"""
    def method(self, a):
        return a

    async def amethod(self):
        pass

@app.route("/legacy", methods=["GET", "POST"])
def legacy():
    pass

CONST = 3
'''


def test_invalid_syntax_falls_back_to_regex_with_warning():
    fa = analyze(INVALID_SRC)
    assert len(fa.warnings) == 1 and "ligne 7" in fa.warnings[0] and "expressions régulières" in fa.warnings[0]
    # le préfixe valide (lignes 1-6) est analysé par ast, le reste par regex
    assert [(i.module, i.names, i.line, i.is_relative) for i in fa.imports] == [("os", [], 1, False), (".x", ["y"], 2, True)]
    assert [(s.kind, s.name, s.line_start, s.line_end, s.parent) for s in fa.symbols] == [
        ("function", "ok", 4, 5, None),
        ("function", "broken", 7, 8, None),
        ("class", "Foo", 10, 16, None),
        ("method", "method", 12, 13, "Foo"),
        ("method", "amethod", 15, 16, "Foo"),
        ("function", "legacy", 19, 20, None),
        ("constant", "CONST", 22, 22, None),
    ]
    assert by_name(fa, "ok").meta.get("fallback") is None
    assert by_name(fa, "amethod").meta["async"] is True and by_name(fa, "amethod").meta["fallback"] is True
    assert by_name(fa, "Foo").meta["bases"] == ["Base", "Mixin"]
    assert eps(fa) == [
        ("server", "GET", "/legacy", "legacy", "flask", 18),
        ("server", "POST", "/legacy", "legacy", "flask", 18),
    ]


def test_python2_print_statement():
    fa = analyze('''
    import sys
    print "hello"
    def f():
        pass
    ''')
    assert [(s.kind, s.name, s.line_start) for s in fa.symbols] == [("function", "f", 3)]
    assert [i.module for i in fa.imports] == ["sys"]
    assert len(fa.warnings) == 1


def test_empty_whitespace_and_odd_inputs_never_raise():
    assert EXTRACTOR.analyze("", "e.py") == FileAnalysis(language="python")
    assert EXTRACTOR.analyze("   \n\n\t\n", "e.py").symbols == []
    fa = EXTRACTOR.analyze("import os\x00\ndef f(): pass\n", "nul.py")
    assert [s.name for s in fa.symbols] == ["f"] and fa.warnings == ["octets nuls supprimés avant analyse"]
    fa = EXTRACTOR.analyze("﻿import os\r\ndef g(): pass\r\n", "bom.py")
    assert [(s.name, s.line_start) for s in fa.symbols] == [("g", 2)] and fa.warnings == []
    fa = EXTRACTOR.analyze(None, "none.py")  # type: ignore[arg-type]
    assert fa.symbols == [] and fa.endpoints == []
    fa = EXTRACTOR.analyze(b"def h(): pass\n", "bytes.py")  # type: ignore[arg-type]
    assert [s.name for s in fa.symbols] == ["h"]
    for weird in ["\x01\x02\x03", "(" * 100000, "def " * 20000, "x = " + "(" * 500 + "1" + ")" * 500,
                  "\t\tdef f():\n\t\t\tpass", " def z(): pass", "if True:\n" + "    " * 300 + "x = 1\n"]:
        fa = EXTRACTOR.analyze(weird, "weird.py")
        assert isinstance(fa, FileAnalysis)


def test_truncated_file_keeps_valid_prefix():
    fa = analyze('''
    class Repo:
        def save(self, obj):
            return obj

    def later():
        x = compute(
    ''')
    assert [(s.kind, s.name, s.line_start, s.line_end) for s in fa.symbols] == [
        ("class", "Repo", 1, 3),
        ("method", "save", 2, 3),
        ("function", "later", 5, 6),
    ]
    assert len(fa.warnings) == 1


def test_huge_single_line_is_fast_and_truncated():
    src = "DATA = " + repr("a" * 1000000) + "\ndef after(): pass\n"
    t = time.perf_counter()
    fa = EXTRACTOR.analyze(src, "big_line.py")
    assert time.perf_counter() - t < 2.5
    assert [(s.kind, s.name, s.line_start) for s in fa.symbols] == [("constant", "DATA", 1), ("function", "after", 2)]
    assert len(by_name(fa, "DATA").signature) <= 120
    # même ligne énorme mais syntaxe invalide : repli regex borné
    fa = EXTRACTOR.analyze("X = [" + "1, " * 300000 + "\ndef after(): pass\n", "bad_line.py")
    assert [s.name for s in fa.symbols if s.kind == "function"] == ["after"] and fa.warnings


def test_one_megabyte_file_under_one_second():
    parts = ["import requests\nfrom dataclasses import dataclass\nBASE = 'http://localhost:8000'\n"]
    i = 0
    while sum(len(p) for p in parts) < 1_000_000:
        parts.append('''
@dataclass
class Model%d:
    """Modèle %d."""
    id: int
    name: str = "x"

class Service%d:
    def compute(self, a: int, b: int = 2) -> int:
        total = 0
        for k in range(a):
            total += k * b
        return total

    def fetch(self):
        %s
''' % (i, i, i, "return requests.get(f'{BASE}/items/%d')" % i if i % 10 == 0 else "return None"))
        i += 1
    src = "".join(parts)
    assert len(src) >= 1_000_000
    best = min(_timed(src) for _ in range(2))
    assert best < 2.5, "analyse trop lente : %.2fs" % best
    fa = EXTRACTOR.analyze(src, "big.py")
    assert len(fa.models) == i and len(fa.endpoints) == (i + 9) // 10 and fa.warnings == []
    assert fa.endpoints[0].handler == "Service0@fetch"


def _timed(src: str) -> float:
    t = time.perf_counter()
    EXTRACTOR.analyze(src, "big.py")
    return time.perf_counter() - t
