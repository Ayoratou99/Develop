"""Mentions d'identifiants dans les documents et les notes ; dérive documentation / code.

Un document (dictionnaire de données, spécification, note de réunion) cite des éléments du
code : noms de classes (``OrderService``), de tables (``orders``), de colonnes
(``confirmed_at``), de routes (``POST /api/orders/{id}/confirm``) ou de fichiers
(``src/api/client.ts``). Ce module :

* extrait ces **candidats** d'un texte en un seul passage par famille d'expressions
  régulières (:func:`extract_identifiers`, :func:`scan_text`) ;
* les fait correspondre, exactement et sans tenir compte de la casse, aux symboles, modèles,
  tables, champs, endpoints et fichiers de l'index (:class:`NameIndex`, construit une seule
  fois par passe de liaison, jamais une requête par identifiant) ;
* expose la **dérive** doc / code : :func:`document_mentions` dit, pour chaque identifiant
  « qui ressemble à du code », s'il existe encore dans l'index ; :func:`is_stale` résume
  cela en un booléen prudent (un identifiant disparu **et** au moins un autre encore présent,
  pour éviter les faux positifs sur du texte ordinaire) ;
* résume chaque mention par un **type** (:data:`MENTION_KINDS`) : une table citée par son nom
  est une ``table`` même si des modèles homonymes la définissent (DDL, migration) ; une classe
  métier homonyme de sa table reste un ``model`` ; un identifiant absent est ``unknown``.

Rien ici n'écrit dans la base ni dans les dépôts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from develop.code_analysis.model import normalize_route_path
from develop.storage import Database, node_key

__all__ = [
    "MAX_MENTIONS",
    "MENTION_KINDS",
    "TABLE_DEFINITION_KINDS",
    "Candidate",
    "Mention",
    "NameIndex",
    "document_mentions",
    "extract_identifiers",
    "is_stale",
    "match_candidates",
    "scan_text",
    "source_text",
]

#: Nombre maximal de mentions distinctes conservées par document.
MAX_MENTIONS = 200
#: Types résumés possibles d'une mention (``Mention.kind``) ; ``unknown`` = identifiant absent de l'index.
MENTION_KINDS = ("symbol", "model", "table", "endpoint", "field", "file", "unknown")
#: Modèles qui ne font que *définir* une table (DDL, migration) : nommés d'après elle, ce ne sont pas des
#: classes métier ; une mention de ce nom désigne la table.
TABLE_DEFINITION_KINDS = frozenset({"sql_table", "migration"})
#: Taille maximale du texte analysé (caractères) ; au-delà, le texte est tronqué.
MAX_TEXT_CHARS = 2_000_000
#: Préfixes d'URL courants, retirés ou ajoutés pour retrouver une route citée sans son préfixe.
GENERIC_ROUTE_PREFIXES = ("/api/v1", "/api/v2", "/api", "/v1", "/v2", "/backend")
#: Formes de candidats considérées comme « du code » (par opposition à un mot ordinaire).
STRONG_SHAPES = frozenset({"backtick", "camel", "snake", "upper", "qualified", "route", "path"})
#: Extensions de fichiers de code reconnues dans une mention nue (``OrderController.php``).
_CODE_EXTENSIONS = ("php", "ts", "tsx", "js", "jsx", "mjs", "cjs", "py", "java", "kt", "dart", "sql", "vue",
                    "svelte", "rb", "go", "rs", "cs", "swift", "blade.php")
_HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS")
_MIN_CAMEL = 5
_TYPE_SYMBOL_KINDS = frozenset({"class", "interface", "type", "enum", "trait", "component", "module"})

# --------------------------------------------------------------------------- expressions régulières

_BACKTICK_RE = re.compile(r"`([^`\n]{1,200})`")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUALIFIED_RE = re.compile(r"(?<![\w.])([A-Z][A-Za-z0-9]*)(?:::|->|\.|@|#)([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?(?![\w.])")
_ROUTE_RE = re.compile(
    r"(?:\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+)?"
    r"(?<![\w.:/\\])((?:https?://[^\s'\"`<>)]+)|(?:/[A-Za-z0-9_\-{}:$.]+(?:/[A-Za-z0-9_\-{}:$.]+)*))",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"(?<![\w/.\-\\])((?:[A-Za-z0-9_.\-]+/)+[A-Za-z0-9_.\-]+\.[A-Za-z0-9]{1,6})(?![\w/])")
_BARE_FILE_RE = re.compile(r"(?<![\w/.\-])([A-Za-z0-9_\-]+\.(?:%s))(?![\w/])" % "|".join(_CODE_EXTENSIONS))
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BACKTICK_ROUTE_RE = re.compile(r"^(?:(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+)?((?:https?://\S+)|(?:/\S+))$",
                                re.IGNORECASE)
_BACKTICK_QUALIFIED_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:::|->|\.|@|#)([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?$")
_CAMEL_RE = re.compile(r"^(?:[a-z]+(?:[A-Z][a-z0-9]*)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+)$")
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
_UPPER_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_CAPITALIZED_RE = re.compile(r"^[A-Z][a-z0-9]{3,}$")
_WORD_PARTS_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def split_words(name: str) -> List[str]:
    """Mots (en minuscules) d'un identifiant : ``OrderService`` → ``["order", "service"]``,
    ``customer_id`` → ``["customer", "id"]``, ``API_BASE`` → ``["api", "base"]``."""
    return [w.lower() for w in _WORD_PARTS_RE.findall(name or "")]


def _shape_of(word: str) -> Optional[str]:
    """Forme d'un mot hors accents graves : ``camel`` / ``snake`` / ``upper`` / ``capitalized`` / ``None``."""
    if _SNAKE_RE.match(word):
        return "snake"
    if _UPPER_RE.match(word):
        return "upper"
    if len(word) >= _MIN_CAMEL and _CAMEL_RE.match(word):
        return "camel"
    if _CAPITALIZED_RE.match(word):
        return "capitalized"
    return None


def _segments(path: str) -> int:
    return len([s for s in (path or "").split("/") if s])


# --------------------------------------------------------------------------- candidats

@dataclass
class Candidate:
    """Un identifiant candidat trouvé dans un texte.

    ``shape`` : ``backtick`` | ``camel`` | ``snake`` | ``upper`` | ``qualified`` | ``route`` |
    ``path`` | ``word`` (mot ordinaire, retenu seulement s'il nomme une table ou un modèle
    connu). ``key`` identifie le candidat (nom en minuscules, ou chemin normalisé pour une
    route). ``method`` ne sert qu'aux routes.
    """

    name: str
    shape: str
    key: str
    count: int = 1
    position: int = 0
    method: Optional[str] = None
    parent: Optional[str] = None  # pour ``qualified`` : la partie gauche (classe)

    @property
    def strong(self) -> bool:
        return self.shape in STRONG_SHAPES


_SHAPE_RANK = {"word": 0, "capitalized": 0, "camel": 1, "snake": 1, "upper": 1, "path": 2, "route": 2,
               "qualified": 3, "backtick": 3}


class _Collector:
    """Agrège les candidats par clé en conservant la première position et la forme la plus sûre."""

    def __init__(self) -> None:
        self.items: Dict[str, Candidate] = {}

    def add(self, name: str, shape: str, key: str, position: int, method: Optional[str] = None,
            parent: Optional[str] = None) -> None:
        cur = self.items.get(key)
        if cur is None:
            self.items[key] = Candidate(name=name, shape=shape, key=key, count=1, position=position, method=method,
                                        parent=parent)
            return
        cur.count += 1
        if _SHAPE_RANK.get(shape, 0) > _SHAPE_RANK.get(cur.shape, 0):
            cur.shape, cur.name = shape, name
        if method and not cur.method:
            cur.method = method
        if parent and not cur.parent:
            cur.parent = parent
        if position < cur.position:
            cur.position = position

    def sorted(self) -> List[Candidate]:
        return sorted(self.items.values(), key=lambda c: (c.position, c.key))


def _route_key(path: str) -> Optional[str]:
    norm = normalize_route_path(path)
    if not norm or norm == "/":
        return None
    return norm


def _add_route(col: _Collector, method: Optional[str], path: str, position: int, require_method_single: bool) -> None:
    path = path.rstrip(".,;:)")  # ponctuation de fin de phrase : « voir POST /orders/{id}/confirm. »
    key = _route_key(path)
    if key is None:
        return
    if require_method_single and not method and _segments(key) < 2:
        return  # « /health » seul est trop ambigu (chemin de fichier, fraction…) sans méthode
    col.add(path.strip(), "route", "route:" + key, position, method=(method or "").upper() or None)


def _add_identifier(col: _Collector, word: str, shape: str, position: int, parent: Optional[str] = None) -> None:
    if len(word) < 2:
        return
    col.add(word, shape, word.lower(), position, parent=parent)


def _add_backtick(col: _Collector, raw: str, position: int) -> None:
    content = raw.strip().strip("'\"")
    if not content:
        return
    m = _BACKTICK_ROUTE_RE.match(content)
    if m:
        _add_route(col, m.group(1), m.group(2), position, require_method_single=False)
        return
    if _PATH_RE.fullmatch(content) or _BARE_FILE_RE.fullmatch(content):
        col.add(content, "path", "path:" + content.lower(), position)
        return
    m = _BACKTICK_QUALIFIED_RE.match(content)
    if m:
        col.add(m.group(1) + "." + m.group(2), "qualified", "qualified:%s.%s" % (m.group(1).lower(), m.group(2).lower()),
                position, parent=m.group(1))
        _add_identifier(col, m.group(1), "backtick", position)
        return
    token = content[:-2] if content.endswith("()") else content
    if _IDENT_RE.match(token):
        _add_identifier(col, token, "backtick", position)
        return
    # expression composite (``quantité × prix``, ``SELECT * FROM orders``) : seuls les mots « code » comptent
    for wm in _WORD_RE.finditer(content):
        word = wm.group(0)
        shape = _shape_of(word)
        if shape in ("camel", "snake", "upper"):
            _add_identifier(col, word, shape, position + wm.start())
        elif shape is None or shape == "capitalized":
            _add_identifier(col, word, "word", position + wm.start())


def scan_text(text: str) -> List[Candidate]:
    """Candidats d'un texte, triés par première apparition, agrégés par identifiant.

    Un seul passage par famille : accents graves (puis masqués pour ne pas être recomptés),
    tokens qualifiés (``OrderService::confirm``), routes (``POST /api/orders/{id}``), chemins de
    fichiers, puis tous les mots (classés CamelCase ≥ 5, snake_case, UPPER_SNAKE, ou mot
    ordinaire). Les mots ordinaires sont renvoyés avec la forme ``word`` : à l'appelant de ne
    les retenir que s'ils nomment une table ou un modèle de l'index.
    """
    if not isinstance(text, str) or not text:
        return []
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
    col = _Collector()
    pieces: List[str] = []
    last = 0
    for m in _BACKTICK_RE.finditer(text):
        _add_backtick(col, m.group(1), m.start())
        pieces.append(text[last:m.start()])
        pieces.append(" " * (m.end() - m.start()))
        last = m.end()
    pieces.append(text[last:])
    plain = "".join(pieces)

    for m in _QUALIFIED_RE.finditer(plain):
        col.add(m.group(1) + "." + m.group(2), "qualified", "qualified:%s.%s" % (m.group(1).lower(), m.group(2).lower()),
                m.start(), parent=m.group(1))
    for m in _ROUTE_RE.finditer(plain):
        _add_route(col, m.group(1), m.group(2), m.start(), require_method_single=True)
    for m in _PATH_RE.finditer(plain):
        col.add(m.group(1), "path", "path:" + m.group(1).lower(), m.start())
    for m in _BARE_FILE_RE.finditer(plain):
        col.add(m.group(1), "path", "path:" + m.group(1).lower(), m.start())
    for m in _WORD_RE.finditer(plain):
        word = m.group(0)
        shape = _shape_of(word)
        if shape in ("camel", "snake", "upper"):
            _add_identifier(col, word, shape, m.start())
        elif len(word) >= 3:
            _add_identifier(col, word, "word", m.start())
    return col.sorted()


def extract_identifiers(text: str) -> List[str]:
    """Identifiants « qui ressemblent à du code » d'un texte, uniques, dans l'ordre d'apparition.

    Mots en CamelCase (≥ 5 caractères), snake_case, UPPER_SNAKE, tokens entre accents graves,
    tokens qualifiés (``OrderService::confirm``), chemins de route (``/api/orders/{id}``) et
    chemins de fichiers. Les mots ordinaires n'y figurent pas.
    """
    return [c.name for c in scan_text(text) if c.strong]


# --------------------------------------------------------------------------- index des noms

@dataclass
class NameIndex:
    """Dictionnaire en mémoire des noms de l'index, construit en quelques requêtes en lot.

    Toutes les clés sont en minuscules. Les tuples ``(kind, node_key, file_id)`` renvoyés par
    :meth:`lookup` servent à créer les relations ``documented_by`` et à expliquer la dérive.
    """

    symbols: Dict[str, List[Tuple[int, Optional[int], str, Optional[str], str]]] = field(default_factory=dict)
    qualified: Dict[str, List[Tuple[int, Optional[int], str]]] = field(default_factory=dict)
    models: Dict[str, List[Tuple[int, Optional[int], str]]] = field(default_factory=dict)
    tables: Dict[str, str] = field(default_factory=dict)
    fields: Dict[str, List[Tuple[int, Optional[int], str]]] = field(default_factory=dict)
    endpoints: Dict[str, List[Tuple[int, Optional[int], str, str]]] = field(default_factory=dict)
    endpoint_suffixes: Dict[str, List[Tuple[int, Optional[int], str, str]]] = field(default_factory=dict)
    files_by_path: Dict[str, Tuple[int, str]] = field(default_factory=dict)
    files_by_name: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
    words: Set[str] = field(default_factory=set)
    #: Noms (minuscules) des modèles qui sont des classes métier (Eloquent, JPA, interface TS…), par
    #: opposition aux définitions de table (:data:`TABLE_DEFINITION_KINDS`).
    entity_models: Set[str] = field(default_factory=set)

    @classmethod
    def build(cls, db: Database) -> "NameIndex":
        idx = cls()
        for r in db.query("SELECT id, repo_id, path FROM files"):
            path = r["path"] or ""
            if not path:
                continue
            fid = int(r["id"])
            idx.files_by_path[path.lower()] = (fid, path)
            name = path.rsplit("/", 1)[-1].lower()
            idx.files_by_name.setdefault(name, []).append((fid, path))
            stem = name.rsplit(".", 1)[0]
            idx._learn_words(stem)
        for r in db.query("SELECT id, file_id, name, parent, kind FROM symbols"):
            name = r["name"] or ""
            if len(name) < 2:
                continue
            sid, fid, parent = int(r["id"]), r["file_id"], r["parent"]
            idx.symbols.setdefault(name.lower(), []).append((sid, fid, name, parent, r["kind"] or ""))
            if parent:
                short = str(parent).split("\\")[-1].split(".")[-1]
                idx.qualified.setdefault("%s.%s" % (short.lower(), name.lower()), []).append((sid, fid, name))
            idx._learn_words(name)
        for r in db.query("SELECT id, file_id, name, table_name, fields, kind FROM data_models"):
            name = r["name"] or ""
            if not name:
                continue
            mid, fid = int(r["id"]), r["file_id"]
            idx.models.setdefault(name.lower(), []).append((mid, fid, name))
            if (r["kind"] or "") not in TABLE_DEFINITION_KINDS:
                idx.entity_models.add(name.lower())
            idx._learn_words(name)
            table = (r["table_name"] or "").strip().strip("`\"'[]")
            if table:
                idx.tables.setdefault(table.lower(), table)
                idx._learn_words(table)
            for fname in _field_names(r["fields"]):
                idx.fields.setdefault(fname.lower(), []).append((mid, fid, name))
                idx._learn_words(fname)
        for r in db.query("SELECT dst FROM relations WHERE kind = 'defines_table'"):
            dst = r["dst"] or ""
            if dst.startswith("table:"):
                table = dst[len("table:"):].strip()
                if table:
                    idx.tables.setdefault(table.lower(), table)
                    idx._learn_words(table)
        for r in db.query("SELECT id, file_id, side, method, normalized_path FROM endpoints"):
            norm = (r["normalized_path"] or "").strip()
            if not norm or norm == "/":
                continue
            entry = (int(r["id"]), r["file_id"], (r["method"] or "").upper(), r["side"] or "")
            idx.endpoints.setdefault(norm, []).append(entry)
            segs = [s for s in norm.split("/") if s]
            for i in range(1, len(segs) - 1):
                suffix = "/" + "/".join(segs[i:])
                idx.endpoint_suffixes.setdefault(suffix, []).append(entry)
        return idx

    def _learn_words(self, name: str) -> None:
        for w in split_words(name):
            if len(w) >= 3:
                self.words.add(w)

    # ----- correspondances -----
    def lookup(self, cand: Candidate) -> List[Tuple[str, str, Optional[int]]]:
        """Noeuds de l'index correspondant à un candidat : ``[(kind, node_key, file_id)]``."""
        if cand.shape == "route":
            return self._lookup_route(cand)
        if cand.shape == "path":
            return self._lookup_path(cand.name)
        if cand.shape == "qualified":
            return self._lookup_qualified(cand)
        return self._lookup_identifier(cand.name, cand.shape)

    def _lookup_identifier(self, name: str, shape: str) -> List[Tuple[str, str, Optional[int]]]:
        low = name.lower()
        out: List[Tuple[str, str, Optional[int]]] = []
        if shape in ("word", "capitalized"):
            table = self.tables.get(low)
            if table:
                out.append(("table", node_key("table", table), None))
            for mid, fid, mname in self.models.get(low, []):
                if mname == name:  # mot ordinaire : seule une correspondance exacte (casse comprise) compte
                    out.append(("model", node_key("model", mid), fid))
            return out
        model_files: Set[Tuple[str, Optional[int]]] = set()
        for mid, fid, mname in self.models.get(low, []):
            out.append(("model", node_key("model", mid), fid))
            model_files.add((mname.lower(), fid))
        table = self.tables.get(low)
        if table:
            out.append(("table", node_key("table", table), None))
        entity_matched = bool(out)
        for sid, fid, sname, _parent, skind in self.symbols.get(low, []):
            if (sname.lower(), fid) in model_files:
                continue  # la classe du modèle : le modèle suffit
            if entity_matched and skind not in _TYPE_SYMBOL_KINDS:
                continue  # « orders » désigne la table, pas la méthode de relation ``orders()``
            out.append(("symbol", node_key("symbol", sid), fid))
        for mid, fid, _mname in self.fields.get(low, []):
            out.append(("field", node_key("model", mid), fid))
        return out

    def _lookup_qualified(self, cand: Candidate) -> List[Tuple[str, str, Optional[int]]]:
        out: List[Tuple[str, str, Optional[int]]] = []
        for sid, fid, _name in self.qualified.get(cand.key[len("qualified:"):], []):
            out.append(("symbol", node_key("symbol", sid), fid))
        if out:
            return out
        if cand.parent:
            return self._lookup_identifier(cand.parent, "backtick")
        return out

    def _lookup_path(self, path: str) -> List[Tuple[str, str, Optional[int]]]:
        low = path.lower().lstrip("./")
        hit = self.files_by_path.get(low)
        if hit:
            return [("file", node_key("file", hit[0]), hit[0])]
        name = low.rsplit("/", 1)[-1]
        matches = [(fid, p) for fid, p in self.files_by_name.get(name, [])
                   if "/" not in low or p.lower().endswith("/" + low) or p.lower() == low]
        if 0 < len(matches) <= 5:
            return [("file", node_key("file", fid), fid) for fid, _p in matches]
        return []

    def _lookup_route(self, cand: Candidate) -> List[Tuple[str, str, Optional[int]]]:
        norm = cand.key[len("route:"):]
        found = list(self.endpoints.get(norm, []))
        if not found:
            for g in GENERIC_ROUTE_PREFIXES:
                if norm.startswith(g + "/"):
                    found += self.endpoints.get(norm[len(g):], [])
                found += self.endpoints.get(g + norm, [])
        if not found and _segments(norm) >= 2:
            found += self.endpoint_suffixes.get(norm, [])
            segs = [s for s in norm.split("/") if s]
            for i in range(1, len(segs) - 1):
                found += self.endpoints.get("/" + "/".join(segs[i:]), [])
        if cand.method:
            found = [e for e in found if not e[2] or e[2] in ("ANY", "*") or e[2] == cand.method
                     or {e[2], cand.method} == {"GET", "HEAD"}]
        servers = [e for e in found if e[3] == "server"]
        chosen = servers or found
        seen: Set[int] = set()
        out: List[Tuple[str, str, Optional[int]]] = []
        for eid, fid, _m, _side in chosen:
            if eid in seen:
                continue
            seen.add(eid)
            out.append(("endpoint", node_key("endpoint", eid), fid))
        return out

    def missing_is_code_like(self, cand: Candidate) -> bool:
        """Vrai si un candidat absent de l'index ressemble vraiment à un identifiant du projet.

        Évite les faux positifs : un mot en accents graves tout en minuscules (``draft``) est
        une valeur, pas un symbole ; un nom de produit en PascalCase (``MonProjet``) ne partage
        aucun mot avec le vocabulaire de l'index.
        """
        shape = cand.shape
        name = cand.name
        if shape == "qualified":
            return True
        if shape == "backtick":
            token = name[:-2] if name.endswith("()") else name
            if "_" in token or _CAMEL_RE.match(token) or _UPPER_RE.match(token):
                return True
            return bool(_CAPITALIZED_RE.match(token))
        if shape in ("camel", "snake", "upper"):
            words = [w for w in split_words(name) if len(w) >= 3]
            return any(w in self.words for w in words)
        return False


def _field_names(raw: Any) -> List[str]:
    """Noms de champs d'un modèle depuis la colonne JSON ``fields`` (tolérant)."""
    import json

    if raw is None or raw == "":
        return []
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except (TypeError, ValueError):
        return []
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for f in value:
        if isinstance(f, dict):
            name = f.get("name")
        else:
            name = f
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


# --------------------------------------------------------------------------- mentions

#: Ordre de préférence du type résumé quand plusieurs noeuds correspondent (voir :func:`_summary_kind`).
_KIND_PRIORITY = {"model": 0, "table": 1, "symbol": 2, "field": 3, "endpoint": 4, "file": 5}


def _summary_kind(index: NameIndex, cand: Candidate, nodes: Sequence[Tuple[str, str, Optional[int]]]) -> str:
    """Type résumé d'une mention (un élément de :data:`MENTION_KINDS`) d'après ses noeuds.

    ``unknown`` sans correspondance. ``table`` dès qu'une table porte le nom cité, sauf si une
    classe métier homonyme existe (entité JPA ``Customer`` ↔ table ``customer``) : les modèles
    ``sql_table`` / ``migration`` ne font que définir la table et ne la supplantent pas. Sinon
    le type le plus prioritaire : modèle, symbole, champ, endpoint, fichier.
    """
    kinds = {n[0] for n in nodes}
    if not kinds:
        return "unknown"
    if "table" in kinds and cand.name.lower() not in index.entity_models:
        return "table"
    return min(kinds, key=lambda k: _KIND_PRIORITY.get(k, 9))


@dataclass
class Mention:
    """Une mention d'identifiant dans un document, confrontée à l'index."""

    name: str
    kind: str  # un élément de MENTION_KINDS (voir _summary_kind)
    exists: bool
    count: int
    shape: str
    strong: bool
    code_like: bool  # pour une mention absente : ressemble-t-elle vraiment à du code ?
    nodes: List[Tuple[str, str, Optional[int]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "exists": self.exists,
            "count": self.count,
            "shape": self.shape,
            "strong": self.strong,
            "code_like": self.code_like,
            "nodes": [n[1] for n in self.nodes],
        }


def match_candidates(index: NameIndex, candidates: Sequence[Candidate], limit: int = MAX_MENTIONS) -> List[Mention]:
    """Confronte des candidats à l'index ; au plus ``limit`` mentions, les existantes d'abord."""
    mentions: List[Mention] = []
    for cand in candidates:
        nodes = index.lookup(cand)
        if cand.shape in ("word", "capitalized"):
            if not nodes:
                continue  # un mot ordinaire absent de l'index n'est pas une mention
            shape = "word"
        else:
            shape = cand.shape
        kind = _summary_kind(index, cand, nodes)
        code_like = bool(nodes) or index.missing_is_code_like(cand)
        mentions.append(Mention(name=cand.name, kind=kind, exists=bool(nodes), count=cand.count, shape=shape,
                                strong=shape in STRONG_SHAPES, code_like=code_like, nodes=nodes))
    mentions.sort(key=lambda m: (not m.exists, not m.code_like, -m.count, m.name.lower()))
    return mentions[:max(0, int(limit))]


def source_text(db: Database, source_type: str, source_id: int) -> str:
    """Texte d'un document, d'une note ou d'un fichier indexé (depuis ses morceaux).

    Pour un fichier, l'en-tête ajouté par l'indexeur (``# dépôt/chemin L1-20``) est retiré
    grâce à ``meta.body_offset`` afin de ne pas compter le fichier comme se citant lui-même.
    """
    if source_type == "note":
        note = db.get_note(int(source_id))
        if note is not None:
            return "%s\n%s" % (note.title or "", note.body or "")
        return ""
    if source_type not in ("document", "file"):
        return ""
    parts: List[str] = []
    total = 0
    for chunk in db.chunks_for(source_type, int(source_id)):
        text = chunk.text or ""
        if source_type == "file":
            offset = (chunk.meta or {}).get("body_offset")
            if isinstance(offset, int) and 0 < offset <= len(text):
                text = text[offset:]
        parts.append(text)
        total += len(text)
        if total >= MAX_TEXT_CHARS:
            break
    if source_type == "document":
        doc = db.get_document(int(source_id))
        if doc is not None and doc.title:
            parts.insert(0, doc.title)
    return "\n".join(parts)


def document_mentions(db: Database, source_type: str, source_id: int,
                      index: Optional[NameIndex] = None) -> List[Dict[str, Any]]:
    """Mentions d'un document (``document``), d'une note (``note``) ou d'un fichier (``file``).

    Chaque élément : ``{kind, name, exists, count, shape, strong, code_like, nodes}`` avec
    ``kind`` dans :data:`MENTION_KINDS` (``'symbol'|'model'|'table'|'endpoint'|'field'|'file'|
    'unknown'``). Une table citée par son nom (``orders``) est de type ``table`` même si des
    modèles la définissent (DDL, migration) ; ``nodes`` garde toutes les correspondances.
    ``exists=False`` signale un identifiant « qui ressemble à du code » (CamelCase, snake_case,
    accents graves, route, chemin) absent de l'index : c'est la matière de la détection de
    dérive doc / code. ``index`` permet de réutiliser un :class:`NameIndex` déjà construit
    (passe de liaison).
    """
    text = source_text(db, source_type, source_id)
    if not text.strip():
        return []
    idx = index if index is not None else NameIndex.build(db)
    return [m.to_dict() for m in match_candidates(idx, scan_text(text))]


def is_stale(db: Database, source_type: str, source_id: int, index: Optional[NameIndex] = None) -> bool:
    """Vrai si le document semble ne plus correspondre au code.

    Condition prudente : au moins une mention de type modèle / table / symbole / champ (en
    accents graves, CamelCase ou snake_case) **absente** de l'index alors qu'au moins une autre
    mention « code » existe encore. Un texte ordinaire (aucune mention existante) ou qui ne
    cite que des valeurs (``draft``) n'est jamais signalé.
    """
    text = source_text(db, source_type, source_id)
    if not text.strip():
        return False
    idx = index if index is not None else NameIndex.build(db)
    mentions = match_candidates(idx, scan_text(text))
    has_existing = any(m.exists and m.strong for m in mentions)
    if not has_existing:
        return False
    return any((not m.exists) and m.code_like for m in mentions)
