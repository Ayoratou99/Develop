"""Structures de résultat des extracteurs (indépendantes du stockage)."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SYMBOL_KINDS = ("function", "method", "class", "interface", "type", "enum", "trait", "component",
                "variable", "constant", "property", "module", "view")
ENDPOINT_SIDES = ("server", "client")
MODEL_KINDS = ("eloquent", "migration", "sql_table", "ts_interface", "ts_type", "dataclass", "pydantic",
               "sqlalchemy", "django", "jpa", "dart_class", "prisma", "schema")


@dataclass
class ExtractedSymbol:
    name: str
    kind: str  # voir SYMBOL_KINDS
    line_start: int
    line_end: int
    signature: str = ""
    parent: Optional[str] = None  # classe/namespace parent
    meta: Dict[str, Any] = field(default_factory=dict)  # exported, async, decorators, docstring…

    @property
    def qualified_name(self) -> str:
        return "%s.%s" % (self.parent, self.name) if self.parent else self.name


@dataclass
class ExtractedEndpoint:
    side: str  # "server" (route définie) | "client" (appel HTTP émis)
    method: str  # GET, POST, PUT, PATCH, DELETE, ANY
    path: str  # chemin brut tel qu'écrit dans le code (peut contenir ${id}, {order}, :id, une base URL)
    handler: str = ""  # "OrderController@index", nom de fonction, ou nom de la fonction appelante côté client
    framework: str = ""  # laravel, express, nestjs, fastapi, flask, django, spring, axios, fetch, dio, http, guzzle…
    line: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)  # prefix, middleware, base_url_var, status_codes…

    @property
    def normalized_path(self) -> str:
        return normalize_route_path(self.path)


@dataclass
class ModelField:
    name: str
    type: str = ""
    nullable: Optional[bool] = None
    default: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # unique, primary, length, enum_values, cast…


@dataclass
class ModelRelation:
    kind: str  # belongs_to, has_many, has_one, many_to_many, foreign_key, references
    target: str  # modèle ou table cible
    field: str = ""  # champ/méthode porteur de la relation


@dataclass
class ExtractedModel:
    name: str  # nom de classe ou de table
    kind: str  # voir MODEL_KINDS
    table_name: Optional[str] = None
    fields: List[ModelField] = field(default_factory=list)
    relations: List[ModelRelation] = field(default_factory=list)
    line: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedImport:
    module: str  # tel qu'écrit : "../api/client", "App\\Models\\Order", "os.path", "package:http/http.dart"
    names: List[str] = field(default_factory=list)  # symboles importés (vide = module entier / défaut)
    line: int = 0
    is_relative: bool = False
    resolved_path: Optional[str] = None  # rempli plus tard par le résolveur (chemin relatif au dépôt)


@dataclass
class FileAnalysis:
    language: str
    symbols: List[ExtractedSymbol] = field(default_factory=list)
    endpoints: List[ExtractedEndpoint] = field(default_factory=list)
    models: List[ExtractedModel] = field(default_factory=list)
    imports: List[ExtractedImport] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for ep, raw in zip(self.endpoints, d["endpoints"]):
            raw["normalized_path"] = ep.normalized_path
        return d


# --------------------------------------------------------------------------- normalisation des routes

_PARAM_PATTERNS = [
    re.compile(r"\$\{[^}]*\}"),  # ${id}, ${order.id}
    re.compile(r"\{[^}/]*\}"),  # {order}, {id?}, {id:[0-9]+}
    re.compile(r"<[^>/]*>"),  # <int:pk>, <slug>
    re.compile(r":[A-Za-z_][A-Za-z0-9_]*\??"),  # :id, :userId?
    re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*"),  # $id (PHP/Dart interpolation)
    re.compile(r"\(\?P<[^>]+>[^)]*\)"),  # (?P<pk>[0-9]+) Django
    re.compile(r"\[[^\]/]*\]"),  # [id] Next.js
]
_SCHEME_HOST = re.compile(r"^[a-z][a-z0-9+.-]*://[^/]*", re.IGNORECASE)
_LEADING_BASE = re.compile(r"^(?:\$\{[^}]*\}|\{\{[^}]*\}\}|\$[A-Za-z_][A-Za-z0-9_.]*|\{[^}]*\})+(?=/|$)")
_NUMERIC_SEG = re.compile(r"^\d+$")


def normalize_route_path(path: str) -> str:
    """Rend comparables les routes serveur et les appels client.

    - supprime le schéma et l'hôte (``http://localhost:8000/api/orders`` → ``/api/orders``) ;
    - supprime la chaîne de requête et le fragment ;
    - remplace tout paramètre (``{order}``, ``:id``, ``${id}``, ``<int:pk>``, ``[id]``, ``$id``,
      segment purement numérique) par ``:param`` ;
    - compacte les ``/`` doubles, retire le ``/`` final, impose un ``/`` initial, minuscules.
    Exemple : ``/orders/${id}/confirm`` et ``/orders/{order}/confirm`` → ``/orders/:param/confirm``.
    """
    p = (path or "").strip().strip("'\"`")
    p = _SCHEME_HOST.sub("", p)
    # Une interpolation en tête (${API_BASE}/orders, $baseUrl/orders, {{base}}/orders) est une base URL : on la retire.
    p = _LEADING_BASE.sub("", p)
    p = p.split("?", 1)[0].split("#", 1)[0]
    for pat in _PARAM_PATTERNS:
        p = pat.sub(":param", p)
    p = re.sub(r"/+", "/", p)
    segs = []
    for seg in p.split("/"):
        if seg == "":
            continue
        if _NUMERIC_SEG.match(seg):
            seg = ":param"
        segs.append(seg.lower())
    return "/" + "/".join(segs)


def route_paths_match(server_path: str, client_path: str) -> bool:
    """Deux chemins normalisés concordent si égaux, ou si le chemin client se termine par le chemin serveur
    (préfixe de base URL absent côté serveur, ex: client ``/api/orders`` vs serveur ``/orders``)."""
    s, c = normalize_route_path(server_path), normalize_route_path(client_path)
    if s == c:
        return True
    return s != "/" and c.endswith(s)  # s commence par "/", donc la frontière de segment est garantie
