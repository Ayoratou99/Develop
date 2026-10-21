"""Extracteur Python : symboles, modèles de données, endpoints et imports via ``ast``.

Stratégie :

1. ``ast.parse`` sur le texte complet ;
2. en cas d'erreur de syntaxe (fichier tronqué, Python 2, encodage étrange…), tentative sur le
   préfixe valide puis repli sur une analyse ligne à ligne par expressions régulières pour le
   reste ; un avertissement est toujours ajouté dans ``FileAnalysis.warnings`` ;
3. aucune exception ne sort de ``analyze`` : tout problème devient un avertissement.

Aucune dépendance hors bibliothèque standard. Compatible Python 3.9 (``ast.unparse`` est utilisé
quand il existe, avec un repli minimal sinon).
"""
from __future__ import annotations

import ast
import re
import warnings
from bisect import bisect_left
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from develop.code_analysis.model import (
    ExtractedEndpoint,
    ExtractedImport,
    ExtractedModel,
    ExtractedSymbol,
    FileAnalysis,
    ModelField,
    ModelRelation,
)

LANGUAGE = "python"
HTTP_VERBS = ("get", "post", "put", "patch", "delete", "head", "options")
MAX_VARIABLES = 50  # variables de module simples conservées
MAX_CONSTANTS = 1000  # garde-fou pour les fichiers générés
SNIPPET_LEN = 120  # longueur max d'un extrait de source reconstruit
BIG_NODE_LINES = 40  # au-delà, on ne reconstruit pas l'expression (coût)
BIG_NODE_COLS = 1500
MAX_LINE_SCAN = 10000  # longueur de ligne examinée par le repli regex

_CONST_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SCHEME_HOST = re.compile(r"^[a-z][a-z0-9+.-]*://[^/?#]*", re.IGNORECASE)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Fabriques d'objets « serveur » : nom de classe -> framework par défaut
_SERVER_FACTORIES: Dict[str, str] = {
    "FastAPI": "fastapi",
    "APIRouter": "fastapi",
    "Flask": "flask",
    "Blueprint": "flask",
    "RouteTableDef": "aiohttp",
    "Application": "aiohttp",
    "Starlette": "starlette",
    "Sanic": "sanic",
    "Quart": "quart",
    "Bottle": "bottle",
}
_SERVER_ROOTS = ("fastapi", "flask", "aiohttp", "starlette", "sanic", "quart", "bottle", "django")
# Fabriques de clients HTTP : nom de classe -> framework
_CLIENT_FACTORIES: Dict[str, str] = {
    "Session": "requests",
    "session": "requests",
    "Client": "httpx",
    "AsyncClient": "httpx",
    "ClientSession": "aiohttp",
    "HTTPConnection": "http.client",
    "HTTPSConnection": "http.client",
}
_UNAMBIGUOUS_CLIENTS = ("ClientSession", "HTTPConnection", "HTTPSConnection")
_CLIENT_LIBS = ("requests", "httpx", "aiohttp")

_DJANGO_BASES = ("Model", "AbstractUser", "AbstractBaseUser")
_DJANGO_RELATIONS: Dict[str, str] = {
    "ForeignKey": "belongs_to",
    "OneToOneField": "has_one",
    "ManyToManyField": "many_to_many",
}
_PYDANTIC_BASES = ("BaseModel", "BaseSettings", "SQLModel")
_SQLA_BASES = ("DeclarativeBase", "DeclarativeMeta")
_SCHEMA_BASES = ("TypedDict", "NamedTuple")
_ENUM_BASES = ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag")

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
# nœuds « instruction » (toujours parcourus) : les expressions ne le sont que sur les lignes candidates
_STMT_LIKE: Tuple[type, ...] = (ast.stmt, ast.ExceptHandler) + (
    (getattr(ast, "match_case"),) if hasattr(ast, "match_case") else ()
)
# feuilles sans enfant utile : inutile de les empiler pendant le parcours des expressions
_LEAF_TYPES = (ast.Constant, ast.Name, ast.Load, ast.Store, ast.Del)
# une ligne doit contenir l'un de ces appels pour que ses expressions soient parcourues
_CANDIDATE_LINE = re.compile(
    r"\b(?:get|post|put|patch|delete|head|options|request|urlopen|route|view|add_url_rule|add_resource|register"
    r"|add_[a-z]{3,10}|path|re_path|url|Route|WebSocketRoute)\s*\("
)


# --------------------------------------------------------------------------- utilitaires AST


def _has_candidate(node: Any, cands: List[int]) -> bool:
    """Vrai si l'intervalle de lignes du nœud contient une ligne candidate (recherche dichotomique)."""
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return True
    end = getattr(node, "end_lineno", None) or lineno
    i = bisect_left(cands, lineno)
    return i < len(cands) and cands[i] <= end


def _node_is_big(node: Any) -> bool:
    """Vrai si l'expression est trop volumineuse pour être reconstruite à coût raisonnable."""
    lineno = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    if lineno is None or end is None:
        return False
    if end - lineno > BIG_NODE_LINES:
        return True
    if end == lineno:
        c0 = getattr(node, "col_offset", 0) or 0
        c1 = getattr(node, "end_col_offset", 0) or 0
        return (c1 - c0) > BIG_NODE_COLS
    return False


def _unparse_fallback(node: Any) -> str:
    """Reconstruction minimale quand ``ast.unparse`` est absent ou échoue."""
    try:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return "%s.%s" % (_unparse_fallback(node.value), node.attr)
        if isinstance(node, ast.Constant):
            rep = repr(node.value)
            return rep if len(rep) <= SNIPPET_LEN else rep[:SNIPPET_LEN] + "…"
        if isinstance(node, ast.Subscript):
            sl = node.slice
            index_cls = getattr(ast, "Index", None)
            if index_cls is not None and isinstance(sl, index_cls):  # Python 3.8
                sl = sl.value  # type: ignore[attr-defined]
            return "%s[%s]" % (_unparse_fallback(node.value), _unparse_fallback(sl))
        if isinstance(node, ast.Tuple):
            return ", ".join(_unparse_fallback(e) for e in node.elts)
        if isinstance(node, ast.Call):
            return "%s(…)" % _unparse_fallback(node.func)
        if isinstance(node, ast.List):
            return "[…]"
        if isinstance(node, ast.Dict):
            return "{…}"
        if isinstance(node, ast.arguments):
            names = [a.arg for a in getattr(node, "posonlyargs", []) + node.args + node.kwonlyargs]
            if node.vararg:
                names.append("*" + node.vararg.arg)
            if node.kwarg:
                names.append("**" + node.kwarg.arg)
            return ", ".join(names)
    except Exception:  # noqa: BLE001 - reconstruction « au mieux »
        pass
    return "…"


def _unparse(node: Any, limit: int = SNIPPET_LEN) -> str:
    """Texte source reconstruit d'un nœud, tronqué à ``limit`` caractères ; jamais d'exception."""
    if node is None:
        return ""
    text = ""
    if not _node_is_big(node):
        try:
            text = ast.unparse(node)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - AttributeError (3.8), RecursionError…
            text = ""
    if not text:
        text = _unparse_fallback(node)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _dotted(node: Any) -> Optional[str]:
    """``a.b.c`` pour une chaîne Name/Attribute, sinon None."""
    parts: List[str] = []
    depth = 0
    while isinstance(node, ast.Attribute) and depth < 64:
        parts.append(node.attr)
        node = node.value
        depth += 1
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _last(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _const_str(node: Any) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _kw(call: ast.Call, name: str) -> Optional[ast.expr]:
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _kw_bool(call: ast.Call, name: str, default: Optional[bool] = None) -> Optional[bool]:
    v = _kw(call, name)
    if isinstance(v, ast.Constant) and isinstance(v.value, bool):
        return v.value
    return default


def _str_list(node: Any) -> List[str]:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    s = _const_str(node)
    return [s] if s is not None else []


def _expr_text(node: Any, limit: int = 60) -> str:
    return _dotted(node) or _unparse(node, limit)


def _is_optional(ann: str) -> bool:
    t = ann.replace(" ", "")
    if t.startswith("Optional[") or t.startswith("typing.Optional[") or t.startswith("Mapped[Optional["):
        return True
    if "|None" in t or "None|" in t:
        return True
    return t.startswith(("Union[", "typing.Union[")) and "None" in t


def _strip_wrapper(ann: str, names: Tuple[str, ...]) -> str:
    t = ann.strip()
    for n in names:
        if t.startswith(n + "[") and t.endswith("]"):
            return t[len(n) + 1 : -1].strip()
    return t


def _strip_optional(ann: str) -> str:
    t = _strip_wrapper(ann, ("Optional", "typing.Optional"))
    t = t.replace(" | None", "").replace("None | ", "")
    return t.strip()


def _strip_quotes(s: str) -> str:
    return s.strip().strip("'\"")


def _docstring(node: Any) -> Optional[str]:
    """Première ligne non vide de la docstring (sans passer par ``inspect.cleandoc``, trop coûteux)."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        for line in first.value.value.splitlines():
            line = line.strip()
            if line:
                return line[:200]
    return None


def _iter_block(stmts: List[ast.stmt]) -> Iterator[ast.stmt]:
    """Parcourt un bloc en traversant les conteneurs transparents (if / try / with)."""
    stack: List[ast.stmt] = list(reversed(stmts))
    guard = 0
    while stack and guard < 2000000:
        guard += 1
        stmt = stack.pop()
        if isinstance(stmt, ast.If):
            stack.extend(reversed(stmt.orelse))
            stack.extend(reversed(stmt.body))
        elif isinstance(stmt, ast.Try):
            stack.extend(reversed(stmt.finalbody))
            stack.extend(reversed(stmt.orelse))
            for h in reversed(stmt.handlers):
                stack.extend(reversed(h.body))
            stack.extend(reversed(stmt.body))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            stack.extend(reversed(stmt.body))
        else:
            yield stmt


def _all_args(args: ast.arguments) -> List[ast.arg]:
    out = list(getattr(args, "posonlyargs", [])) + list(args.args) + list(args.kwonlyargs)
    if args.vararg:
        out.append(args.vararg)
    if args.kwarg:
        out.append(args.kwarg)
    return out


def _django_app_from_path(path: str) -> Optional[str]:
    """Devine le nom de l'application Django depuis ``<app>/models.py`` ou ``<app>/models/x.py``."""
    parts = [p for p in re.split(r"[\\/]+", path or "") if p]
    for i, part in enumerate(parts):
        if (part == "models.py" or part == "models") and i >= 1:
            app = parts[i - 1]
            if _IDENT.match(app):
                return app.lower()
            return None
    return None


# --------------------------------------------------------------------------- contexte d'analyse


class _Context:
    """État partagé pendant l'analyse d'un arbre."""

    def __init__(self, lines: List[str], path: str, fa: FileAnalysis) -> None:
        self.lines = lines
        self.path = path
        self.fa = fa
        self.modules: Set[str] = set()  # racines des modules importés ("fastapi", "requests"…)
        self.imported_names: Dict[str, str] = {}  # nom local -> module d'origine
        self.client_aliases: Dict[str, str] = {}  # variable -> framework client HTTP
        self.receivers: Dict[str, Tuple[str, Optional[str]]] = {}  # variable -> (framework serveur, prefix)
        self.constants: Dict[str, str] = {}  # constantes str de module
        self.has_urlpatterns = False
        self.cand_lines: List[int] = []  # lignes (1-based, triées) pouvant contenir une route ou un appel HTTP
        self.skip_calls: Set[int] = set()  # id() des appels déjà traités comme décorateurs
        self.pending_calls: List[Tuple[ast.Call, Optional[str], Optional[str]]] = []
        self.pending_decorators: List[Tuple[ast.Call, ast.stmt, Optional[str]]] = []

    def root_module(self, text: str) -> Optional[str]:
        """Racine du module dont provient le premier identifiant de ``text`` (``rq.get`` -> requests)."""
        first = text.split(".", 1)[0]
        src = self.imported_names.get(first)
        if src:
            return src.split(".", 1)[0]
        if "." in text:
            return first
        return None

    def resolve_str(self, node: Any) -> Optional[str]:
        """Chaîne constante, directement ou via une constante de module."""
        s = _const_str(node)
        if s is not None:
            return s
        if isinstance(node, ast.Name):
            return self.constants.get(node.id)
        return None

    def server_framework_hint(self) -> Optional[str]:
        for fw in _SERVER_ROOTS:
            if fw in self.modules:
                return fw
        return None


# --------------------------------------------------------------------------- extracteur


class PythonExtractor:
    """Extracteur pour les fichiers ``.py`` (enregistré sous le langage ``python``)."""

    languages: List[str] = [LANGUAGE]

    # ----- point d'entrée

    def analyze(self, text: str, path: str) -> FileAnalysis:
        fa = FileAnalysis(language=LANGUAGE)
        try:
            self._analyze(text, path or "", fa)
        except Exception as exc:  # noqa: BLE001 - garde-fou ultime, jamais d'exception
            fa.warnings.append("analyse interrompue (%s : %s)" % (type(exc).__name__, exc))
        return fa

    def _analyze(self, text: Any, path: str, fa: FileAnalysis) -> None:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        elif not isinstance(text, str):
            text = str(text or "")
        if text.startswith("﻿"):
            text = text[1:]
        if "\x00" in text:
            text = text.replace("\x00", "")
            fa.warnings.append("octets nuls supprimés avant analyse")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            return
        lines = text.split("\n")

        tree, err = _safe_parse(text)
        if tree is not None:
            self._from_tree(tree, lines, path, fa)
            return

        err_line, err_msg = err or (0, "?")
        cut = err_line - 1 if 1 <= err_line <= len(lines) else 0
        fallback_from = 0
        if cut > 0:
            prefix_tree, _ = _safe_parse("\n".join(lines[:cut]))
            if prefix_tree is not None:
                self._from_tree(prefix_tree, lines, path, fa)
                fallback_from = cut
        fa.warnings.append(
            "erreur de syntaxe ligne %s : %s ; repli sur une analyse par expressions régulières à partir de la ligne %d"
            % (err_line or "?", err_msg, fallback_from + 1)
        )
        _regex_fallback(lines, fallback_from, fa)
        _finalize(fa)

    def _from_tree(self, tree: ast.Module, lines: List[str], path: str, fa: FileAnalysis) -> None:
        ctx = _Context(lines, path, fa)
        ctx.cand_lines = [i + 1 for i, line in enumerate(lines) if _CANDIDATE_LINE.search(line, 0, MAX_LINE_SCAN)]
        self._scan_tree(tree, ctx)
        self._resolve_pending(ctx)
        self._collect_definitions(tree, ctx)
        _finalize(fa)

    # ----- passe générique : imports, alias, appels, décorateurs

    def _scan_tree(self, tree: ast.AST, ctx: _Context) -> None:
        """Parcours itératif. Les instructions sont toujours visitées ; les sous-arbres d'expressions ne le
        sont que si une ligne de l'instruction ressemble à une route ou un appel HTTP (filtre regex), ce qui
        évite de visiter les centaines de milliers de nœuds d'un gros fichier."""
        cands = ctx.cand_lines
        stack: List[Tuple[ast.AST, Optional[str], Optional[str], bool]] = [(tree, None, None, False)]
        while stack:
            node, cls, fn, in_expr = stack.pop()
            t = type(node)
            if in_expr:
                if t is ast.Call and id(node) not in ctx.skip_calls and _interesting_call(node):  # type: ignore[arg-type]
                    ctx.pending_calls.append((node, cls, fn))  # type: ignore[arg-type]
                for child in ast.iter_child_nodes(node):
                    if type(child) not in _LEAF_TYPES:
                        stack.append((child, cls, fn, True))
                continue
            if t is ast.Import:
                self._note_import(node, ctx)  # type: ignore[arg-type]
            elif t is ast.ImportFrom:
                self._note_import_from(node, ctx)  # type: ignore[arg-type]
            elif t in _FUNC_TYPES:
                for dec in node.decorator_list:  # type: ignore[attr-defined]
                    if isinstance(dec, ast.Call):
                        ctx.skip_calls.add(id(dec))
                        ctx.pending_decorators.append((dec, node, cls))  # type: ignore[arg-type]
                for a in _all_args(node.args):  # type: ignore[attr-defined]
                    if a.annotation is not None:
                        self._note_annotation_alias(a.arg, a.annotation, ctx)
            elif t is ast.Assign or t is ast.AnnAssign:
                self._note_assignment(node, cls, fn, ctx)  # type: ignore[arg-type]
            elif t is ast.AugAssign:
                tgt = node.target  # type: ignore[attr-defined]
                if isinstance(tgt, ast.Name) and tgt.id == "urlpatterns":
                    ctx.has_urlpatterns = True
            elif t is ast.With or t is ast.AsyncWith:
                for item in node.items:  # type: ignore[attr-defined]
                    if isinstance(item.optional_vars, ast.Name) and isinstance(item.context_expr, ast.Call):
                        self._note_factory(item.optional_vars.id, item.context_expr, ctx)

            ccls, cfn = cls, fn
            if t is ast.ClassDef:
                ccls, cfn = node.name, None  # type: ignore[attr-defined]
            elif t in _FUNC_TYPES:
                cfn = node.name  # type: ignore[attr-defined]
            cand = _has_candidate(node, cands)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, _STMT_LIKE):
                    stack.append((child, ccls, cfn, False))
                elif cand:
                    stack.append((child, ccls, cfn, True))

    def _note_import(self, node: ast.Import, ctx: _Context) -> None:
        for alias in node.names:
            mod = alias.name or ""
            ctx.fa.imports.append(ExtractedImport(module=mod, names=[], line=node.lineno, is_relative=False))
            root = mod.split(".", 1)[0]
            if root:
                ctx.modules.add(root)
                ctx.imported_names[alias.asname or root] = mod

    def _note_import_from(self, node: ast.ImportFrom, ctx: _Context) -> None:
        level = node.level or 0
        mod = "." * level + (node.module or "")
        names = [a.name for a in node.names]
        ctx.fa.imports.append(ExtractedImport(module=mod, names=names, line=node.lineno, is_relative=level > 0))
        if level == 0 and node.module:
            ctx.modules.add(node.module.split(".", 1)[0])
            for a in node.names:
                if a.name != "*":
                    ctx.imported_names[a.asname or a.name] = node.module

    def _note_assignment(self, node: ast.stmt, cls: Optional[str], fn: Optional[str], ctx: _Context) -> None:
        value = node.value  # type: ignore[attr-defined]
        if value is None:
            return
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]  # type: ignore[attr-defined]
        if cls is None and fn is None:
            for tgt in targets:
                if isinstance(tgt, ast.Name):
                    if tgt.id == "urlpatterns":
                        ctx.has_urlpatterns = True
                    s = _const_str(value)
                    if s is not None:
                        ctx.constants.setdefault(tgt.id, s)
        if not isinstance(value, ast.Call):
            return
        for tgt in targets:
            name = tgt.id if isinstance(tgt, ast.Name) else (tgt.attr if isinstance(tgt, ast.Attribute) else None)
            if name:
                self._note_factory(name, value, ctx)
        self._note_core_table(value, ctx)

    def _note_factory(self, name: str, call: ast.Call, ctx: _Context) -> None:
        """Mémorise ``app = FastAPI()``, ``router = APIRouter(prefix=…)``, ``session = requests.Session()``…"""
        func_text = _dotted(call.func)
        if not func_text:
            return
        last = _last(func_text)
        root = ctx.root_module(func_text)
        if last in _SERVER_FACTORIES:
            if last == "Application" and root != "aiohttp":
                return
            fw = root if root in _SERVER_ROOTS else _SERVER_FACTORIES[last]
            prefix = ctx.resolve_str(_kw(call, "prefix"))
            if prefix is None:
                prefix = ctx.resolve_str(_kw(call, "url_prefix"))
            ctx.receivers[name] = (fw, prefix)
            return
        fw2 = self._client_framework_for(func_text, ctx)
        if fw2:
            ctx.client_aliases[name] = fw2

    def _client_framework_for(self, func_text: str, ctx: _Context) -> Optional[str]:
        """Framework client correspondant à une fabrique (``httpx.AsyncClient``) ou annotation."""
        base = _strip_optional(func_text)
        last = _last(base)
        if last not in _CLIENT_FACTORIES:
            return None
        fw = _CLIENT_FACTORIES[last]
        root = ctx.root_module(base)
        if root is None:
            return fw if last in _UNAMBIGUOUS_CLIENTS else None
        return fw if root == fw.split(".", 1)[0] else None

    def _note_annotation_alias(self, arg_name: str, annotation: ast.expr, ctx: _Context) -> None:
        """``client: httpx.AsyncClient`` ou ``session: Optional[aiohttp.ClientSession]`` -> alias client."""
        text = _dotted(annotation)
        if text is None and isinstance(annotation, ast.Subscript):
            sl: Any = annotation.slice
            index_cls = getattr(ast, "Index", None)
            if index_cls is not None and isinstance(sl, index_cls):  # Python 3.8
                sl = sl.value
            text = _dotted(sl)
        if text is None:
            return
        fw = self._client_framework_for(text, ctx)
        if fw:
            ctx.client_aliases[arg_name] = fw

    def _note_core_table(self, call: ast.Call, ctx: _Context) -> None:
        """``orders = Table("orders", metadata, Column(...), ...)`` (SQLAlchemy Core)."""
        if _last(_dotted(call.func) or "") != "Table" or not call.args:
            return
        table = _const_str(call.args[0])
        if table is None:
            return
        fields: List[ModelField] = []
        relations: List[ModelRelation] = []
        for a in call.args[1:]:
            if isinstance(a, ast.Call) and _last(_dotted(a.func) or "") == "Column" and a.args:
                cname = _const_str(a.args[0])
                if cname is None:
                    continue
                f, rels = _sqla_column_field(cname, a, "", ann_known=False)
                fields.append(f)
                relations.extend(rels)
        if not fields:
            return
        ctx.fa.models.append(
            ExtractedModel(name=table, kind="sqlalchemy", table_name=table, fields=fields, relations=relations,
                           line=call.lineno, meta={"style": "core"})
        )

    # ----- résolution des appels et décorateurs en attente

    def _resolve_pending(self, ctx: _Context) -> None:
        for dec, func, cls in ctx.pending_decorators:
            try:
                ctx.fa.endpoints.extend(self._route_from_decorator(dec, func, cls, ctx))
            except Exception as exc:  # noqa: BLE001
                ctx.fa.warnings.append("décorateur ignoré ligne %d (%s)" % (dec.lineno, exc))
        for call, cls, fn in ctx.pending_calls:
            try:
                eps = self._server_call(call, ctx)
                if not eps:
                    ep = self._client_call(call, cls, fn, ctx)
                    eps = [ep] if ep else []
                ctx.fa.endpoints.extend(eps)
            except Exception as exc:  # noqa: BLE001
                ctx.fa.warnings.append("appel ignoré ligne %d (%s)" % (call.lineno, exc))

    def _route_from_decorator(self, dec: ast.Call, func: ast.stmt, cls: Optional[str], ctx: _Context) -> List[ExtractedEndpoint]:
        f = dec.func
        framework: Optional[str] = None
        if isinstance(f, ast.Attribute):
            attr = f.attr
            recv_text = _dotted(f.value) or _unparse(f.value, 40)
        elif isinstance(f, ast.Name):
            attr = f.id
            recv_text = ""
            root = (ctx.imported_names.get(attr) or "").split(".", 1)[0]
            if root not in _SERVER_ROOTS:
                return []
            framework = root
        else:
            return []

        meta: Dict[str, Any] = {}
        if attr in HTTP_VERBS:
            methods = [attr.upper()]
        elif attr in ("route", "api_route"):
            methods = [m.upper() for m in _str_list(_kw(dec, "methods"))]
            single = _const_str(_kw(dec, "method"))  # bottle
            if single:
                methods.append(single.upper())
            if methods:
                meta["methods"] = list(methods)
            else:
                methods = ["GET"]
        elif attr == "websocket":
            methods = ["ANY"]
            meta["websocket"] = True
        elif attr == "view":  # aiohttp routes.view
            methods = ["ANY"]
        else:
            return []

        path_node: Any = dec.args[0] if dec.args else (_kw(dec, "path") or _kw(dec, "rule"))
        if path_node is None:
            return []
        path = ctx.resolve_str(path_node)
        if path is None:
            path, _base = _url_parts(path_node, ctx)
        if path is None:
            return []
        last = _last(recv_text) if recv_text else ""
        known = last in ctx.receivers
        if not (path == "" or path.startswith("/") or path.startswith("{")) and not known:
            return []

        prefix: Optional[str] = None
        if known:
            framework, prefix = ctx.receivers[last]
        if framework is None:
            framework = ctx.server_framework_hint() or ("flask" if attr == "route" else "fastapi")
        if recv_text:
            meta["receiver"] = recv_text
        if prefix:
            meta["prefix"] = prefix
        sc = _kw(dec, "status_code")
        if isinstance(sc, ast.Constant) and isinstance(sc.value, int):
            meta["status_code"] = sc.value
        rm = _kw(dec, "response_model")
        if rm is not None:
            meta["response_model"] = _expr_text(rm)
        name = func.name  # type: ignore[attr-defined]
        handler = "%s@%s" % (cls, name) if cls else name
        return [
            ExtractedEndpoint(side="server", method=m, path=path, handler=handler, framework=framework,
                              line=dec.lineno, meta=dict(meta))
            for m in methods
        ]

    def _server_call(self, call: ast.Call, ctx: _Context) -> List[ExtractedEndpoint]:
        """Routes déclarées par appel : Django ``path()``, aiohttp ``web.get()``, Flask ``add_url_rule``…"""
        f = call.func
        line = call.lineno
        if isinstance(f, ast.Name):
            fname = f.id
            if fname in ("path", "re_path", "url"):
                return self._django_path(call, fname, ctx)
            if fname in ("Route", "WebSocketRoute") and "starlette" in ctx.modules and len(call.args) >= 2:
                path = _const_str(call.args[0])
                if path is None:
                    return []
                methods = [m.upper() for m in _str_list(_kw(call, "methods"))] or ["GET"]
                handler = _expr_text(call.args[1])
                return [ExtractedEndpoint("server", m, path, handler, "starlette", line, {"receiver": fname}) for m in methods]
            return []
        if not isinstance(f, ast.Attribute):
            return []
        attr = f.attr
        recv_text = _dotted(f.value) or ""

        # aiohttp : web.get("/x", handler) / app.router.add_get("/x", handler) / add_route("GET", "/x", handler)
        if (attr in HTTP_VERBS or attr in ("route", "view")) and _last(recv_text) == "web" and len(call.args) >= 2:
            if attr == "route" and len(call.args) >= 3:
                method = (_const_str(call.args[0]) or "ANY").upper()
                path, hnode = _const_str(call.args[1]), call.args[2]
            else:
                method = "ANY" if attr == "view" else attr.upper()
                path, hnode = _const_str(call.args[0]), call.args[1]
            if path is None:
                return []
            return [ExtractedEndpoint("server", method, path, _expr_text(hnode), "aiohttp", line, {"receiver": recv_text})]
        if attr.startswith("add_") and (attr[4:] in HTTP_VERBS or attr[4:] in ("route", "view")) and len(call.args) >= 2:
            verb = attr[4:]
            if verb == "route" and len(call.args) >= 3:
                method = (_const_str(call.args[0]) or "ANY").upper()
                path, hnode = _const_str(call.args[1]), call.args[2]
            else:
                method = "ANY" if verb == "view" else verb.upper()
                path, hnode = _const_str(call.args[0]), call.args[1]
            if path is None or not path.startswith("/"):
                return []
            return [ExtractedEndpoint("server", method, path, _expr_text(hnode), "aiohttp", line, {"receiver": recv_text})]
        # Flask : app.add_url_rule("/x", "endpoint", view_func=..., methods=[...])
        if attr == "add_url_rule" and call.args:
            path = _const_str(call.args[0])
            if path is None:
                return []
            view = _kw(call, "view_func") if _kw(call, "view_func") is not None else (call.args[2] if len(call.args) >= 3 else None)
            handler, vmeta = _view_name(view)
            methods = [m.upper() for m in _str_list(_kw(call, "methods"))] or ["GET"]
            vmeta["receiver"] = recv_text
            fw = ctx.receivers.get(_last(recv_text), ("flask", None))[0]
            return [ExtractedEndpoint("server", m, path, handler, fw, line, dict(vmeta)) for m in methods]
        # Flask-RESTful : api.add_resource(OrderResource, "/orders", "/orders/<int:id>")
        if attr == "add_resource" and len(call.args) >= 2:
            handler = _expr_text(call.args[0])
            out = []
            for a in call.args[1:]:
                p = _const_str(a)
                if p is not None:
                    out.append(ExtractedEndpoint("server", "ANY", p, handler, "flask", line, {"receiver": recv_text, "resource": True}))
            return out
        # Django REST framework : router.register("orders", OrderViewSet)
        if attr == "register" and len(call.args) >= 2 and ("rest_framework" in ctx.modules or "django" in ctx.modules):
            p = _const_str(call.args[0])
            if p is None or not isinstance(call.args[1], (ast.Name, ast.Attribute)):
                return []
            meta: Dict[str, Any] = {"receiver": recv_text, "viewset": True}
            bn = _const_str(_kw(call, "basename"))
            if bn:
                meta["basename"] = bn
            return [ExtractedEndpoint("server", "ANY", p, _expr_text(call.args[1]), "django", line, meta)]
        return []

    def _django_path(self, call: ast.Call, fname: str, ctx: _Context) -> List[ExtractedEndpoint]:
        src = ctx.imported_names.get(fname, "")
        if not (ctx.has_urlpatterns or "django" in ctx.modules or src.startswith("django")):
            return []
        if not call.args:
            return []
        route = _const_str(call.args[0])
        if route is None:
            return []
        view = call.args[1] if len(call.args) >= 2 else _kw(call, "view")
        if view is None:
            return []
        handler, meta = _view_name(view)
        name = _const_str(_kw(call, "name"))
        if name:
            meta["name"] = name
        if fname == "re_path" or fname == "url":
            meta["regex"] = True
        return [ExtractedEndpoint("server", "ANY", route, handler, "django", call.lineno, meta)]

    def _client_call(self, call: ast.Call, cls: Optional[str], fn: Optional[str], ctx: _Context) -> Optional[ExtractedEndpoint]:
        f = call.func
        framework: Optional[str] = None
        method: Optional[str] = None
        url_node: Any = None
        recv_text = ""

        if isinstance(f, ast.Attribute):
            attr = f.attr
            if isinstance(f.value, ast.Call):  # requests.Session().get(...)
                recv_text = _dotted(f.value.func) or ""
                framework = self._client_framework_for(recv_text, ctx) if recv_text else None
            else:
                recv_text = _dotted(f.value) or ""
                if not recv_text:
                    return None
                framework = self._client_framework_for_receiver(recv_text, ctx)
            if attr in HTTP_VERBS:
                method = attr.upper()
                url_node = call.args[0] if call.args else _kw(call, "url")
            elif attr == "request":
                m = call.args[0] if call.args else _kw(call, "method")
                method = (_const_str(m) or "ANY").upper() if m is not None else "ANY"
                url_node = call.args[1] if len(call.args) >= 2 else _kw(call, "url")
            elif attr == "urlopen":
                if framework is None and _last(recv_text) in ("request", "urllib2"):
                    framework = "urllib"
                if framework != "urllib":
                    return None
                method = "GET"
                url_node = call.args[0] if call.args else _kw(call, "url")
            else:
                return None
        elif isinstance(f, ast.Name):
            src = (ctx.imported_names.get(f.id) or "").split(".", 1)[0]
            if f.id == "urlopen" and src == "urllib":
                framework, method = "urllib", "GET"
                url_node = call.args[0] if call.args else _kw(call, "url")
            elif f.id in HTTP_VERBS and src in ("requests", "httpx"):
                framework, method = src, f.id.upper()
                url_node = call.args[0] if call.args else _kw(call, "url")
            elif f.id == "request" and src in ("requests", "httpx"):
                framework = src
                m = call.args[0] if call.args else _kw(call, "method")
                method = (_const_str(m) or "ANY").upper() if m is not None else "ANY"
                url_node = call.args[1] if len(call.args) >= 2 else _kw(call, "url")
            else:
                return None
        else:
            return None

        if url_node is None:
            return None
        # urllib : urlopen(Request(url, method="POST"))
        if framework == "urllib" and isinstance(url_node, ast.Call) and _last(_dotted(url_node.func) or "") == "Request":
            req = url_node
            url_node = req.args[0] if req.args else _kw(req, "url")
            rm = _const_str(_kw(req, "method"))
            if rm:
                method = rm.upper()
            if url_node is None:
                return None
        if framework is None:
            # receveur inconnu (session, client, s…) : on accepte si l'URL a une forme d'URL et qu'un client
            # HTTP connu est importé dans le fichier
            libs = [lib for lib in _CLIENT_LIBS if lib in ctx.modules]
            if not libs or not _looks_like_url(url_node):
                return None
            framework = libs[0] if len(libs) == 1 else "http"
        path, base = _url_parts(url_node, ctx)
        if path is None:
            return None
        meta: Dict[str, Any] = {}
        if recv_text:
            meta["receiver"] = recv_text
        if base:
            meta["base"] = base
            if base in ctx.constants:
                meta["base_value"] = ctx.constants[base]
        handler = "%s@%s" % (cls, fn) if (cls and fn) else (fn or "<module>")
        return ExtractedEndpoint(side="client", method=method or "ANY", path=path, handler=handler,
                                 framework=framework, line=call.lineno, meta=meta)

    def _client_framework_for_receiver(self, recv_text: str, ctx: _Context) -> Optional[str]:
        """``requests`` / ``httpx`` / ``urllib.request`` / alias de variable (``session``, ``self.client``)."""
        root = ctx.root_module(recv_text)
        if root in ("requests", "httpx") and "." not in recv_text:
            return root
        if root is None and recv_text in ("requests", "httpx"):
            return recv_text
        if root == "urllib" or recv_text in ("urllib.request", "urllib2"):
            return "urllib"
        return ctx.client_aliases.get(_last(recv_text))

    # ----- définitions : symboles et modèles

    def _collect_definitions(self, tree: ast.Module, ctx: _Context) -> None:
        fa = ctx.fa
        n_vars = 0
        n_consts = 0
        seen: Set[str] = set()
        truncated = False
        for stmt in _iter_block(tree.body):
            if isinstance(stmt, _FUNC_TYPES):
                fa.symbols.append(self._symbol_for_function(stmt, None, ctx))
            elif isinstance(stmt, ast.ClassDef):
                self._collect_class(stmt, None, ctx)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                for name in _assigned_names(stmt):
                    if name in seen:
                        continue
                    seen.add(name)
                    snippet = _line_snippet(ctx.lines, stmt.lineno)
                    end = getattr(stmt, "end_lineno", None) or stmt.lineno
                    if _CONST_NAME.match(name):
                        if n_consts >= MAX_CONSTANTS:
                            continue
                        n_consts += 1
                        fa.symbols.append(ExtractedSymbol(name, "constant", stmt.lineno, end, snippet))
                    else:
                        if n_vars >= MAX_VARIABLES:
                            truncated = True
                            continue
                        n_vars += 1
                        fa.symbols.append(ExtractedSymbol(name, "variable", stmt.lineno, end, snippet))
        if truncated:
            fa.warnings.append("variables de module limitées à %d" % MAX_VARIABLES)

    def _symbol_for_function(self, node: ast.stmt, cls_name: Optional[str], ctx: _Context) -> ExtractedSymbol:
        decorators = [_unparse(d, 80) for d in node.decorator_list]  # type: ignore[attr-defined]
        meta: Dict[str, Any] = {"async": isinstance(node, ast.AsyncFunctionDef), "decorators": decorators}
        doc = _docstring(node)
        if doc:
            meta["docstring"] = doc
        kind = "function"
        if cls_name:
            kind = "method"
            bases = [d.split("(", 1)[0] for d in decorators]
            lasts = [_last(b) for b in bases]
            meta["staticmethod"] = "staticmethod" in lasts
            meta["classmethod"] = "classmethod" in lasts
            meta["property"] = (
                "property" in lasts
                or "cached_property" in lasts
                or any(b.endswith((".setter", ".getter", ".deleter")) for b in bases)
            )
        end = getattr(node, "end_lineno", None) or node.lineno
        return ExtractedSymbol(node.name, kind, node.lineno, end, _signature(node), parent=cls_name, meta=meta)  # type: ignore[attr-defined]

    def _collect_class(self, node: ast.ClassDef, parent: Optional[str], ctx: _Context) -> None:
        fa = ctx.fa
        stack: List[Tuple[ast.ClassDef, Optional[str]]] = [(node, parent)]
        while stack:
            cls, par = stack.pop()
            bases = [_unparse(b, 80) for b in cls.bases]
            decorators = [_unparse(d, 80) for d in cls.decorator_list]
            base_lasts = [_last(b.split("[", 1)[0]) for b in bases]
            kind = "enum" if any(b in _ENUM_BASES for b in base_lasts) else "class"
            meta: Dict[str, Any] = {"bases": bases, "decorators": decorators}
            doc = _docstring(cls)
            if doc:
                meta["docstring"] = doc
            if kind == "enum":
                members: List[str] = []
                for stmt in _iter_block(cls.body):
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and len(members) < 100:
                        members.extend(_assigned_names(stmt))
                meta["members"] = members
            sig = "class %s(%s)" % (cls.name, ", ".join(bases)) if bases else "class %s" % cls.name
            end = getattr(cls, "end_lineno", None) or cls.lineno
            fa.symbols.append(ExtractedSymbol(cls.name, kind, cls.lineno, end, sig, parent=par, meta=meta))
            for stmt in _iter_block(cls.body):
                if isinstance(stmt, _FUNC_TYPES):
                    fa.symbols.append(self._symbol_for_function(stmt, cls.name, ctx))
                elif isinstance(stmt, ast.ClassDef):
                    stack.append((stmt, cls.name))
            try:
                model = self._model_from_class(cls, bases, base_lasts, decorators, ctx)
            except Exception as exc:  # noqa: BLE001
                model = None
                fa.warnings.append("modèle %s ignoré (%s)" % (cls.name, exc))
            if model is not None:
                fa.models.append(model)

    # ----- modèles

    def _model_from_class(self, cls: ast.ClassDef, bases: List[str], base_lasts: List[str], decorators: List[str],
                          ctx: _Context) -> Optional[ExtractedModel]:
        dec_lasts = [_last(d.split("(", 1)[0]) for d in decorators]
        if "dataclass" in dec_lasts:
            meta: Dict[str, Any] = {}
            for d in decorators:
                if _last(d.split("(", 1)[0]) == "dataclass" and "frozen=True" in d.replace(" ", ""):
                    meta["frozen"] = True
            return self._annotated_model(cls, "dataclass", meta)
        if any(b in _PYDANTIC_BASES for b in base_lasts):
            base = next(b for b in base_lasts if b in _PYDANTIC_BASES)
            return self._annotated_model(cls, "pydantic", {"base": base})
        if any(b in _SCHEMA_BASES for b in base_lasts):
            base = next(b for b in base_lasts if b in _SCHEMA_BASES)
            return self._annotated_model(cls, "schema", {"base": base})
        if self._is_django_model(cls, bases, base_lasts, ctx):
            return self._django_model(cls, ctx)
        if self._is_sqlalchemy_model(cls, bases, base_lasts):
            return self._sqlalchemy_model(cls)
        return None

    def _annotated_model(self, cls: ast.ClassDef, kind: str, meta: Dict[str, Any]) -> ExtractedModel:
        fields: List[ModelField] = []
        for stmt in _iter_block(cls.body):
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
                continue
            ann = _unparse(stmt.annotation, 80)
            if ann.startswith(("ClassVar", "typing.ClassVar")):
                continue
            name = stmt.target.id
            if kind == "pydantic" and name.startswith("model_"):
                continue
            default, fmeta = _field_default(stmt.value, kind)
            fields.append(ModelField(name=name, type=ann, nullable=_is_optional(ann), default=default, meta=fmeta))
        return ExtractedModel(name=cls.name, kind=kind, table_name=None, fields=fields, relations=[], line=cls.lineno, meta=meta)

    def _is_django_model(self, cls: ast.ClassDef, bases: List[str], base_lasts: List[str], ctx: _Context) -> bool:
        if any(b.endswith("models.Model") for b in bases):
            return True
        for b in base_lasts:
            if b in _DJANGO_BASES and (ctx.imported_names.get(b, "").startswith("django") or "django" in ctx.modules):
                return True
        for stmt in _iter_block(cls.body):
            value = getattr(stmt, "value", None)
            if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and isinstance(value, ast.Call):
                ft = _dotted(value.func) or ""
                if ft.startswith("models.") and (ft.endswith("Field") or _last(ft) in _DJANGO_RELATIONS):
                    return True
        return False

    def _is_sqlalchemy_model(self, cls: ast.ClassDef, bases: List[str], base_lasts: List[str]) -> bool:
        if any(b in _SQLA_BASES for b in base_lasts) or any(b.endswith("db.Model") for b in bases):
            return True
        for stmt in _iter_block(cls.body):
            if isinstance(stmt, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "__tablename__" for t in stmt.targets):
                    return True
            if isinstance(stmt, ast.AnnAssign) and _unparse(stmt.annotation, 80).startswith("Mapped["):
                return True
            value = getattr(stmt, "value", None)
            if isinstance(value, ast.Call) and _last(_dotted(value.func) or "") in ("Column", "mapped_column", "relationship"):
                return True
        return False

    def _django_model(self, cls: ast.ClassDef, ctx: _Context) -> ExtractedModel:
        table: Optional[str] = None
        meta: Dict[str, Any] = {}
        fields: List[ModelField] = []
        relations: List[ModelRelation] = []
        for stmt in _iter_block(cls.body):
            if isinstance(stmt, ast.ClassDef) and stmt.name == "Meta":
                for s in _iter_block(stmt.body):
                    if isinstance(s, (ast.Assign, ast.AnnAssign)) and s.value is not None:  # type: ignore[attr-defined]
                        for n in _assigned_names(s):
                            if n == "db_table":
                                table = _const_str(s.value)  # type: ignore[attr-defined]
                            elif n == "abstract" and isinstance(s.value, ast.Constant) and s.value.value is True:  # type: ignore[attr-defined]
                                meta["abstract"] = True
                continue
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            value = stmt.value  # type: ignore[attr-defined]
            names = _assigned_names(stmt)
            if len(names) != 1 or not isinstance(value, ast.Call):
                continue
            name = names[0]
            ft = _dotted(value.func) or ""
            fl = _last(ft)
            if not (ft.startswith("models.") or fl.endswith("Field") or fl in _DJANGO_RELATIONS):
                continue
            nullable = _kw_bool(value, "null", False)
            dflt = _kw(value, "default")
            default = _unparse(dflt, 80) if dflt is not None else None
            fmeta: Dict[str, Any] = {}
            ml = _kw(value, "max_length")
            if isinstance(ml, ast.Constant) and isinstance(ml.value, int):
                fmeta["length"] = ml.value
            for flag in ("unique", "primary_key", "blank"):
                if _kw_bool(value, flag, False):
                    fmeta[flag if flag != "primary_key" else "primary"] = True
            if fl in _DJANGO_RELATIONS:
                tnode = value.args[0] if value.args else _kw(value, "to")
                target = _const_str(tnode) if tnode is not None else None
                if target is None and tnode is not None:
                    target = _dotted(tnode) or _unparse(tnode, 40)
                target = target or "?"
                if target == "self":
                    target = cls.name
                target = _last(target)
                rn = _const_str(_kw(value, "related_name"))
                if rn:
                    fmeta["related_name"] = rn
                fmeta["target"] = target
                relations.append(ModelRelation(kind=_DJANGO_RELATIONS[fl], target=target, field=name))
            fields.append(ModelField(name=name, type=fl, nullable=nullable, default=default, meta=fmeta))
        if table is None and not meta.get("abstract"):
            app = _django_app_from_path(ctx.path)
            if app:
                table = "%s_%s" % (app, cls.name.lower())
        return ExtractedModel(name=cls.name, kind="django", table_name=table, fields=fields, relations=relations,
                              line=cls.lineno, meta=meta)

    def _sqlalchemy_model(self, cls: ast.ClassDef) -> ExtractedModel:
        table: Optional[str] = None
        meta: Dict[str, Any] = {}
        fields: List[ModelField] = []
        relations: List[ModelRelation] = []
        rel_calls: List[Tuple[str, ast.Call, str]] = []
        for stmt in _iter_block(cls.body):
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                continue
            names = _assigned_names(stmt)
            if len(names) != 1:
                continue
            name = names[0]
            value = stmt.value  # type: ignore[attr-defined]
            ann = _unparse(stmt.annotation, 80) if isinstance(stmt, ast.AnnAssign) else ""
            if name == "__tablename__":
                table = _const_str(value)
                continue
            if name == "__abstract__":
                meta["abstract"] = True
                continue
            if name.startswith("__"):
                continue
            call = value if isinstance(value, ast.Call) else None
            fl = _last(_dotted(call.func) or "") if call is not None else ""
            if fl == "relationship":
                rel_calls.append((name, call, ann))  # type: ignore[arg-type]
                continue
            if fl in ("Column", "mapped_column") or ann.startswith("Mapped["):
                f, rels = _sqla_column_field(name, call, ann, ann_known=ann.startswith("Mapped["))
                fields.append(f)
                relations.extend(rels)
        fk_fields = {r.field for r in relations if r.kind == "foreign_key"}
        for name, call, ann in rel_calls:
            relations.append(_sqla_relationship(name, call, ann, fk_fields))
        return ExtractedModel(name=cls.name, kind="sqlalchemy", table_name=table, fields=fields, relations=relations,
                              line=cls.lineno, meta=meta)


# --------------------------------------------------------------------------- aides modèles / endpoints


def _assigned_names(stmt: ast.stmt) -> List[str]:
    """Noms simples affectés par ``a = …``, ``a = b = …``, ``a, b = …`` ou ``a: int = …``."""
    targets = stmt.targets if isinstance(stmt, ast.Assign) else [getattr(stmt, "target", None)]
    out: List[str] = []
    for t in targets:
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            out.extend(e.id for e in t.elts if isinstance(e, ast.Name))
    return out


def _line_snippet(lines: List[str], lineno: int) -> str:
    if 1 <= lineno <= len(lines):
        s = lines[lineno - 1].strip()
        return s if len(s) <= SNIPPET_LEN else s[: SNIPPET_LEN - 1] + "…"
    return ""


def _ann_text(node: Any) -> str:
    """Texte d'une annotation ou d'une valeur par défaut, par le chemin rapide quand c'est possible."""
    d = _dotted(node)
    if d is not None:
        return d
    if isinstance(node, ast.Constant):
        rep = repr(node.value)
        return rep if len(rep) <= 80 else rep[:79] + "…"
    return _unparse(node, 80)


def _format_args(args: ast.arguments) -> str:
    """Reconstruit la liste de paramètres au style PEP 8 (``a: int = 1, /, *rest, key=2, **kw``)."""
    parts: List[str] = []
    posonly = list(getattr(args, "posonlyargs", []))
    positional = posonly + list(args.args)
    defaults = list(args.defaults)
    n_nodef = len(positional) - len(defaults)
    for i, a in enumerate(positional):
        s = a.arg
        if a.annotation is not None:
            s += ": " + _ann_text(a.annotation)
        if i >= n_nodef:
            s += (" = " if a.annotation is not None else "=") + _ann_text(defaults[i - n_nodef])
        parts.append(s)
        if posonly and i == len(posonly) - 1:
            parts.append("/")
    if args.vararg is not None:
        s = "*" + args.vararg.arg
        if args.vararg.annotation is not None:
            s += ": " + _ann_text(args.vararg.annotation)
        parts.append(s)
    elif args.kwonlyargs:
        parts.append("*")
    for a, d in zip(args.kwonlyargs, args.kw_defaults):
        s = a.arg
        if a.annotation is not None:
            s += ": " + _ann_text(a.annotation)
        if d is not None:
            s += (" = " if a.annotation is not None else "=") + _ann_text(d)
        parts.append(s)
    if args.kwarg is not None:
        s = "**" + args.kwarg.arg
        if args.kwarg.annotation is not None:
            s += ": " + _ann_text(args.kwarg.annotation)
        parts.append(s)
    return ", ".join(parts)


def _signature(node: ast.stmt) -> str:
    """``def nom(args) -> ret`` reconstruit depuis l'AST (``async def`` le cas échéant)."""
    try:
        args = _format_args(node.args)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - repli sur ast.unparse
        args = _unparse(node.args, 300)  # type: ignore[attr-defined]
    if len(args) > 300:
        args = args[:299] + "…"
    ret = ""
    returns = getattr(node, "returns", None)
    if returns is not None:
        ret = " -> " + _ann_text(returns)
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return "%s %s(%s)%s" % (prefix, node.name, args, ret)  # type: ignore[attr-defined]


def _field_default(value: Any, kind: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Valeur par défaut d'un champ annoté (``= 1``, ``field(default_factory=list)``, ``Field(None, alias=…)``)."""
    if value is None:
        return None, {}
    if isinstance(value, ast.Constant) and value.value is Ellipsis:
        return None, {}
    if isinstance(value, ast.Call) and _last(_dotted(value.func) or "") in ("field", "Field"):
        meta: Dict[str, Any] = {}
        default: Optional[str] = None
        d = _kw(value, "default")
        if d is None and kind == "pydantic" and value.args:
            d = value.args[0]
        if d is not None and not (isinstance(d, ast.Constant) and d.value is Ellipsis):
            default = _unparse(d, 80)
        df = _kw(value, "default_factory")
        if df is not None:
            default = _unparse(df, 60) + "()"
        alias = _const_str(_kw(value, "alias"))
        if alias:
            meta["alias"] = alias
        return default, meta
    return _unparse(value, 80), {}


def _sqla_column_field(name: str, call: Optional[ast.Call], ann: str, ann_known: bool) -> Tuple[ModelField, List[ModelRelation]]:
    """Champ issu de ``Column(...)`` / ``mapped_column(...)`` / ``Mapped[T]`` et relations ForeignKey."""
    ftype = ""
    nullable: Optional[bool] = None
    default: Optional[str] = None
    meta: Dict[str, Any] = {}
    relations: List[ModelRelation] = []
    if ann_known:
        inner = _strip_wrapper(ann, ("Mapped",))
        nullable = _is_optional(inner)
        ftype = _strip_optional(inner)
    if call is not None:
        for a in call.args:
            if isinstance(a, ast.Call) and _last(_dotted(a.func) or "") == "ForeignKey":
                target = _const_str(a.args[0]) if a.args else None
                if target:
                    meta["foreign_key"] = target
                    relations.append(ModelRelation(kind="foreign_key", target=target.split(".", 1)[0], field=name))
            elif isinstance(a, ast.Constant) and isinstance(a.value, str):
                meta["column"] = a.value
            elif not ftype:
                ftype = _unparse(a, 60)
        nk = _kw_bool(call, "nullable")
        if nk is not None:
            nullable = nk
        if _kw_bool(call, "primary_key", False):
            meta["primary"] = True
            if nullable is None:
                nullable = False
        for flag in ("unique", "index"):
            if _kw_bool(call, flag, False):
                meta[flag] = True
        d = _kw(call, "default")
        if d is None:
            d = _kw(call, "server_default")
        if d is not None:
            default = _unparse(d, 80)
    return ModelField(name=name, type=ftype, nullable=nullable, default=default, meta=meta), relations


def _sqla_relationship(name: str, call: ast.Call, ann: str, fk_fields: Set[str]) -> ModelRelation:
    target: Optional[str] = None
    if call.args:
        a0 = call.args[0]
        target = _const_str(a0) or _dotted(a0)
    if target is None:
        arg = _kw(call, "argument")
        if arg is not None:
            target = _const_str(arg) or _dotted(arg)
    inner = _strip_optional(_strip_wrapper(ann, ("Mapped",))) if ann else ""
    is_list = inner.startswith(("List[", "list[", "Set[", "set[", "typing.List["))
    if target is None and inner:
        target = _strip_quotes(_strip_wrapper(inner, ("List", "list", "Set", "set", "typing.List")))
    target = _last(_strip_quotes(target or "?"))
    if _kw(call, "secondary") is not None:
        kind = "many_to_many"
    elif _kw_bool(call, "uselist") is False:
        kind = "has_one"
    elif is_list:
        kind = "has_many"
    elif name + "_id" in fk_fields:
        kind = "belongs_to"
    elif name.endswith("s"):
        kind = "has_many"
    else:
        kind = "references"
    return ModelRelation(kind=kind, target=target, field=name)


def _view_name(view: Any) -> Tuple[str, Dict[str, Any]]:
    """Nom de vue Django/Flask : ``views.order_list`` -> ``order_list`` ; ``X.as_view()`` -> ``X`` ; ``include(…)``."""
    meta: Dict[str, Any] = {}
    if view is None:
        return "", meta
    if isinstance(view, ast.Call):
        fname = _dotted(view.func) or ""
        if _last(fname) == "as_view":
            full = fname[: -len(".as_view")] if fname.endswith(".as_view") else fname
            meta["view"] = full
            return _last(full), meta
        if _last(fname) == "include":
            inc = view.args[0] if view.args else None
            meta["include"] = _const_str(inc) if _const_str(inc) is not None else (_expr_text(inc) if inc is not None else "")
            return "", meta
        return _expr_text(view), meta
    dotted = _dotted(view)
    if dotted:
        meta["view"] = dotted
        return _last(dotted), meta
    return _unparse(view, 60), meta


def _interesting_call(call: ast.Call) -> bool:
    """Filtre bon marché : seuls les appels pouvant être une route ou un appel HTTP sont conservés."""
    f = call.func
    if isinstance(f, ast.Attribute):
        a = f.attr
        return (
            a in HTTP_VERBS
            or a in ("request", "urlopen", "route", "view", "add_url_rule", "add_resource", "register")
            or (a.startswith("add_") and (a[4:] in HTTP_VERBS or a[4:] in ("route", "view")))
        )
    if isinstance(f, ast.Name):
        return f.id in ("path", "re_path", "url", "Route", "WebSocketRoute", "urlopen", "request") or f.id in HTTP_VERBS
    return False


def _flatten_add(node: ast.expr) -> List[ast.expr]:
    parts: List[ast.expr] = []
    stack: List[ast.expr] = [node]
    while stack and len(parts) < 50:
        n = stack.pop()
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            stack.append(n.right)
            stack.append(n.left)
        else:
            parts.append(n)
    return parts


def _url_parts(node: Any, ctx: Optional[_Context] = None) -> Tuple[Optional[str], Optional[str]]:
    """(chemin reconstitué, base) depuis une chaîne, f-string, concaténation, ``.format``, ``urljoin`` ou variable."""
    if node is None:
        return None, None
    s = _const_str(node)
    if s is not None:
        m = _SCHEME_HOST.match(s)
        return s, (m.group(0) if m else None)
    if isinstance(node, ast.JoinedStr):
        pieces: List[str] = []
        first_expr: Optional[str] = None
        for v in node.values:
            if isinstance(v, ast.Constant):
                pieces.append(str(v.value))
            elif isinstance(v, ast.FormattedValue):
                e = _expr_text(v.value)
                if not pieces:
                    first_expr = e
                pieces.append("{%s}" % e)
        p = "".join(pieces)
        return p, _base_of(p, first_expr)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        pieces = []
        first_expr = None
        for part in _flatten_add(node):
            cs = _const_str(part)
            if cs is not None:
                pieces.append(cs)
            else:
                e = _expr_text(part)
                if not pieces:
                    first_expr = e
                pieces.append("{%s}" % e)
        p = "".join(pieces)
        return p, _base_of(p, first_expr)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        left = _const_str(node.left)
        if left is not None:
            base = None
            if left.startswith("%"):
                r = node.right.elts[0] if isinstance(node.right, ast.Tuple) and node.right.elts else node.right
                base = _expr_text(r)
            return left, _base_of(left, base)
        return None, None
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr == "format":
            templ = _const_str(f.value)
            if templ is not None:
                base = _expr_text(node.args[0]) if templ.startswith("{") and node.args else None
                return templ, _base_of(templ, base)
            return None, None
        if _last(_dotted(f) or "") == "urljoin" and len(node.args) >= 2:
            p, _ = _url_parts(node.args[1], ctx)
            return p, _expr_text(node.args[0])
        return None, None
    if isinstance(node, (ast.Name, ast.Attribute)):
        e = _expr_text(node)
        return "{%s}" % e, e
    return None, None


def _base_of(path: str, first_expr: Optional[str]) -> Optional[str]:
    if path.startswith("{") or path.startswith("%"):
        return first_expr
    m = _SCHEME_HOST.match(path)
    return m.group(0) if m else None


def _looks_like_url(node: Any) -> bool:
    s = _const_str(node)
    if s is not None:
        return s.startswith("/") or s.startswith("http")
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(v, ast.Constant) and "/" in str(v.value) for v in node.values)
    if isinstance(node, ast.BinOp):
        return any("/" in (_const_str(p) or "") for p in _flatten_add(node))
    return False


def _safe_parse(text: str) -> Tuple[Optional[ast.Module], Optional[Tuple[int, str]]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(text), None
    except SyntaxError as exc:
        return None, (exc.lineno or 0, exc.msg or str(exc))
    except Exception as exc:  # noqa: BLE001 - ValueError (octets nuls), RecursionError, MemoryError…
        return None, (0, "%s : %s" % (type(exc).__name__, exc))


def _finalize(fa: FileAnalysis) -> None:
    fa.symbols.sort(key=lambda s: s.line_start)
    fa.endpoints.sort(key=lambda e: e.line)
    fa.models.sort(key=lambda m: m.line)
    fa.imports.sort(key=lambda i: i.line)


# --------------------------------------------------------------------------- repli regex (syntaxe invalide)

_RE_DEF = re.compile(r"^(?P<indent>[ \t]*)(?P<async>async[ \t]+)?def[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*\(")
_RE_CLASS = re.compile(r"^(?P<indent>[ \t]*)class[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*(?:\((?P<bases>[^)]{0,300})\))?[ \t]*:")
_RE_IMPORT = re.compile(r"^[ \t]*import[ \t]+(?P<mods>[\w.]+(?:[ \t]+as[ \t]+\w+)?(?:[ \t]*,[ \t]*[\w.]+(?:[ \t]+as[ \t]+\w+)?){0,50})")
_RE_FROM = re.compile(r"^[ \t]*from[ \t]+(?P<mod>\.*[\w.]*)[ \t]+import[ \t]+(?P<names>[^#]{1,500})")
_RE_CONST = re.compile(r"^(?P<name>[A-Z][A-Z0-9_]*)[ \t]*(?::[^=]{0,100})?=[^=]")
_RE_ROUTE_DEC = re.compile(
    r"^[ \t]*@(?P<recv>[A-Za-z_][\w.]{0,60})\.(?P<verb>get|post|put|patch|delete|head|options|route|api_route)\("
    r"[ \t]*(?P<q>['\"])(?P<path>[^'\"]{0,500})(?P=q)"
)
_RE_METHODS = re.compile(r"methods[ \t]*=[ \t]*[\[(]([^\])]{0,200})[\])]")
_RE_QUOTED = re.compile(r"['\"]([A-Za-z]{3,7})['\"]")


def _indent_of(line: str) -> int:
    """Indentation (tabulation = 8) ; -1 pour une ligne vide ou un commentaire."""
    stripped = line.lstrip(" \t")
    if not stripped or stripped.startswith("#"):
        return -1
    return len(line[: len(line) - len(stripped)].expandtabs(8))


def _block_end(indents: List[int], i: int) -> int:
    base = indents[i]
    last = i
    j = i + 1
    n = len(indents)
    while j < n:
        ind = indents[j]
        if ind >= 0:
            if ind <= base:
                break
            last = j
        j += 1
    return last + 1


def _regex_fallback(lines: List[str], start: int, fa: FileAnalysis) -> None:
    """Analyse ligne à ligne (def/class/import/constantes/décorateurs de route) sans AST."""
    n = len(lines)
    indents = [_indent_of(l[:MAX_LINE_SCAN]) for l in lines]
    class_stack: List[Tuple[int, str]] = []
    pending_routes: List[ExtractedEndpoint] = []
    fallback_routes: List[ExtractedEndpoint] = []
    seen_consts: Set[str] = {s.name for s in fa.symbols if s.kind == "constant"}
    for i in range(start, n):
        ind = indents[i]
        if ind < 0:
            continue
        line = lines[i][:MAX_LINE_SCAN]
        while class_stack and ind <= class_stack[-1][0]:
            class_stack.pop()
        m = _RE_DEF.match(line)
        if m:
            name = m.group("name")
            parent = class_stack[-1][1] if class_stack else None
            sig = line.strip().rstrip(":")
            sig = sig if len(sig) <= SNIPPET_LEN else sig[: SNIPPET_LEN - 1] + "…"
            meta: Dict[str, Any] = {"async": bool(m.group("async")), "decorators": [], "fallback": True}
            fa.symbols.append(ExtractedSymbol(name, "method" if parent else "function", i + 1, _block_end(indents, i), sig, parent, meta))
            handler = "%s@%s" % (parent, name) if parent else name
            for ep in pending_routes:
                ep.handler = handler
                fallback_routes.append(ep)
            pending_routes = []
            continue
        m = _RE_CLASS.match(line)
        if m:
            name = m.group("name")
            bases = [b.strip() for b in (m.group("bases") or "").split(",") if b.strip()]
            parent = class_stack[-1][1] if class_stack else None
            sig = line.strip().rstrip(":")
            fa.symbols.append(ExtractedSymbol(name, "class", i + 1, _block_end(indents, i), sig[:SNIPPET_LEN], parent,
                                              {"bases": bases, "decorators": [], "fallback": True}))
            class_stack.append((ind, name))
            pending_routes = []
            continue
        m = _RE_ROUTE_DEC.match(line)
        if m:
            verb = m.group("verb")
            methods = [verb.upper()] if verb in HTTP_VERBS else []
            if not methods:
                mm = _RE_METHODS.search(line)
                methods = [q.upper() for q in _RE_QUOTED.findall(mm.group(1))] if mm else ["GET"]
            for method in methods:
                pending_routes.append(ExtractedEndpoint("server", method, m.group("path"), "", "", i + 1,
                                                        {"receiver": m.group("recv"), "decorator": verb, "fallback": True}))
            continue
        m = _RE_FROM.match(line)
        if m:
            mod = m.group("mod")
            names_raw = m.group("names").strip().strip("()").strip()
            names = [x.strip().split(" as ")[0].strip() for x in names_raw.split(",") if x.strip()]
            names = [x for x in names if x == "*" or _IDENT.match(x)]
            fa.imports.append(ExtractedImport(module=mod, names=names, line=i + 1, is_relative=mod.startswith(".")))
            continue
        m = _RE_IMPORT.match(line)
        if m:
            for chunk in m.group("mods").split(","):
                mod = chunk.strip().split(" as ")[0].split("\tas\t")[0].strip()
                if mod:
                    fa.imports.append(ExtractedImport(module=mod, names=[], line=i + 1, is_relative=False))
            continue
        if ind == 0:
            m = _RE_CONST.match(line)
            if m and m.group("name") not in seen_consts:
                seen_consts.add(m.group("name"))
                snippet = line.strip()
                fa.symbols.append(ExtractedSymbol(m.group("name"), "constant", i + 1, i + 1,
                                                  snippet if len(snippet) <= SNIPPET_LEN else snippet[: SNIPPET_LEN - 1] + "…"))
    roots = {imp.module.lstrip(".").split(".", 1)[0] for imp in fa.imports}
    hint = next((fw for fw in _SERVER_ROOTS if fw in roots), None)
    for ep in fallback_routes:
        ep.framework = hint or ("flask" if ep.meta.get("decorator") == "route" else "fastapi")
    fa.endpoints.extend(fallback_routes)


EXTRACTOR = PythonExtractor()
