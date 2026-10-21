"""Assembleur de contexte : transforme des résultats de recherche en passages cités [S1], [S2]…

Chaque passage injecté dans le prompt porte un identifiant de source, son chemin et ses lignes
(ou sa page), son type et la raison de sa sélection. Le remplissage est glouton par score, garde
au moins un passage par canal de recherche ayant répondu, respecte une diversité par fichier et
tronque proprement (jamais au milieu d'une ligne) quand le budget est atteint.

Les « hits » acceptés sont des objets au format ``develop.search.engine.SearchHit`` (ou tout objet
ou dict exposant les mêmes attributs) ; :class:`LocalHit` en est une version locale utilisée par
les outils du chat.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from develop.chat.budget import estimate_tokens
from develop.search.engine import is_generated_file

logger = logging.getLogger("develop.chat")

# Même ordre que ``develop.search.engine._CHANNEL_PRIORITY`` : un représentant par canal ayant répondu, le canal
# « route » (endpoints contenant un terme de la question) juste après les correspondances exactes.
CHANNEL_PRIORITY = ("identifier", "route", "graph", "lexical", "semantic", "temporal")
CODE_SOURCE_TYPES = ("file", "symbol", "endpoint", "model")
MIN_PARTIAL_LINES = 3
MIN_PARTIAL_TOKENS = 40
TRUNCATED_MARKER = "… (tronqué : %d lignes non affichées)"

# Fiches structurées (modèle, route, classe) construites depuis l'index, placées en tête du contexte.
# Une fiche qui déborde de CARD_MAX_TOKENS raccourcit d'abord ses sections les moins prioritaires (variantes
# plus courtes ci-dessous) avant d'en retirer une : les règles de validation d'un modèle survivent à sa liste
# de routes, jamais l'inverse (voir ``_fit_sections``).
CARD_MAX_TOKENS = 350
CARD_HITS_WINDOW = 8  # seuls les premiers hits donnent lieu à une fiche
MAX_CARDS = 6
CARDS_BUDGET_SHARE = 2 / 3  # part maximale du budget de contexte prise par les fiches (la première passe toujours)
CARD_MAX_ROUTES = 8
CARD_ROUTES_SHORT = (4, 2)  # variantes plus courtes de la liste des routes (le reste est compté « … (+N) »)
CARD_MAX_METHODS = 15
CARD_MAX_FIELDS = 40
CARD_MAX_RELATIONS = 10
CARD_MAX_MIGRATIONS = 6
CARD_MIGRATIONS_SHORT = (1,)  # variante plus courte de la liste des migrations (la première, celle de la table)
CARD_RULES_MAX_LINES = 40  # lignes du corps de rules() lues au plus
CARD_RULES_MAX_CHARS = 520  # caractères de règles « champ : règles » gardés (entrées entières, le reste compté)
CARD_RULES_SHORT_CHARS = (320, 200)  # variantes plus courtes des règles
CARD_LINE_MAX_CHARS = 700
CARD_MAX_FILE_BYTES = 2_000_000
CARD_CHANNEL = "card"
CARD_REASON = "fiche structurée (synthèse de l'index)"
# Genres de modèles qui définissent une table (mêmes valeurs que ``develop.knowledge_graph.linker.TABLE_LIKE_KINDS``) :
# ils enrichissent la fiche du modèle (types, nullable, défauts, énumérations) et sont couverts par elle.
TABLE_LIKE_KINDS = frozenset({"migration", "sql_table"})
CARDS_HEADER = "Fiches (synthèse exacte de l'index : champs, règles de validation, routes) :"
PASSAGES_HEADER = "Passages :"
# Fichier atteint par le canal graphe seul (importateur d'une classe, fichier utilisant un modèle, document qui
# le mentionne) : la relation est la preuve, pas le contenu de son premier morceau (namespace, use, déclaration
# de classe), qui coûtait 300 à 400 tokens par fichier relié sur un projet réel. Le passage devient un aperçu :
# lignes d'en-tête omises (comptées), puis GRAPH_FILE_PREVIEW_LINES lignes non vides au plus, le reste compté
# comme tronqué. Un fichier trouvé aussi par un autre canal (identifiant, route, mot de la question, outil…) est
# montré entier : son contenu a répondu.
GRAPH_FILE_PREVIEW_LINES = 8
IMPORTS_OMITTED_MARKER = "… (%d lignes d'en-tête omises : namespace, use, import)"
_HEADER_LINE_RE = re.compile(
    r"^(?:# \S+ L\d+(?:-\d+)?\b|<\?php\b|declare\s*\(|namespace\s|package\s|use\s|import\s|from\s+\S+\s+import\b"
    r"|export\s+(?:\*|\{[^}]*\})\s+from\b|(?:const|let|var)\s+[\w{},\s*]+=\s*require\s*\("
    r"|(?:require|include)(?:_once)?\s*[\s(]|using\s|#include\b)")

_LANGUAGE_BY_EXT = {
    ".py": "python", ".php": "php", ".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".jsx": "jsx",
    ".mjs": "javascript", ".cjs": "javascript", ".java": "java", ".kt": "kotlin", ".dart": "dart", ".rb": "ruby",
    ".go": "go", ".rs": "rust", ".cs": "csharp", ".sql": "sql", ".sh": "bash", ".yml": "yaml", ".yaml": "yaml",
    ".json": "json", ".md": "markdown", ".html": "html", ".css": "css", ".scss": "scss", ".vue": "vue",
    ".xml": "xml", ".toml": "toml", ".ini": "ini", ".env": "bash", ".swift": "swift", ".c": "c", ".h": "c",
    ".cpp": "cpp", ".hpp": "cpp", ".txt": "text", ".csv": "csv",
}


@dataclass
class LocalHit:
    """Résultat local au format ``SearchHit`` (même attributs), produit par les outils du chat."""

    source_type: str = "file"
    source_id: Optional[int] = None
    chunk_id: Optional[int] = None
    title: str = ""
    path: Optional[str] = None
    repo_id: Optional[str] = None
    file_id: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    page: Optional[int] = None
    snippet: str = ""
    text: str = ""
    score: float = 0.0
    channels: List[str] = field(default_factory=list)
    reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    """Métadonnées d'une source citée (format de ``docs/API.md``, sans le texte)."""

    ref: str
    source_type: str
    path: Optional[str] = None
    title: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    page: Optional[int] = None
    repo_id: Optional[str] = None
    file_id: Optional[int] = None
    document_id: Optional[int] = None
    note_id: Optional[int] = None
    symbol_id: Optional[int] = None
    endpoint_id: Optional[int] = None
    model_id: Optional[int] = None
    score: float = 0.0
    channels: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref, "source_type": self.source_type, "path": self.path, "title": self.title,
            "line_start": self.line_start, "line_end": self.line_end, "page": self.page,
            "repo_id": self.repo_id, "file_id": self.file_id, "document_id": self.document_id,
            "note_id": self.note_id, "symbol_id": self.symbol_id, "endpoint_id": self.endpoint_id,
            "model_id": self.model_id, "score": self.score, "channels": list(self.channels), "reason": self.reason,
        }

    @classmethod
    def from_hit(cls, ref: str, hit: Any) -> "Source":
        source_type = str(_g(hit, "source_type", "file") or "file")
        source_id = _as_int(_g(hit, "source_id"))
        extra = _g(hit, "extra", {}) or {}
        if not isinstance(extra, dict):
            extra = {}
        file_id = _as_int(_g(hit, "file_id"))
        if file_id is None and source_type == "file":
            file_id = source_id
        document_id = _as_int(extra.get("document_id")) if extra.get("document_id") is not None else None
        note_id = _as_int(extra.get("note_id")) if extra.get("note_id") is not None else None
        symbol_id = _as_int(extra.get("symbol_id")) if extra.get("symbol_id") is not None else None
        endpoint_id = _as_int(extra.get("endpoint_id")) if extra.get("endpoint_id") is not None else None
        model_id = _as_int(extra.get("model_id")) if extra.get("model_id") is not None else None
        if source_type == "document" and document_id is None:
            document_id = source_id
        elif source_type == "note" and note_id is None:
            note_id = source_id
        elif source_type == "symbol" and symbol_id is None:
            symbol_id = source_id
        elif source_type == "endpoint" and endpoint_id is None:
            endpoint_id = source_id
        elif source_type == "model" and model_id is None:
            model_id = source_id
        try:
            score = float(_g(hit, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        channels = [str(c) for c in (_g(hit, "channels", []) or [])]
        return cls(
            ref=ref, source_type=source_type, path=_g(hit, "path") or None, title=_g(hit, "title") or None,
            line_start=_as_int(_g(hit, "line_start")), line_end=_as_int(_g(hit, "line_end")),
            page=_as_int(_g(hit, "page")), repo_id=_g(hit, "repo_id") or None, file_id=file_id,
            document_id=document_id, note_id=note_id, symbol_id=symbol_id, endpoint_id=endpoint_id,
            model_id=model_id, score=score, channels=channels, reason=str(_g(hit, "reason", "") or ""),
        )


@dataclass
class Card:
    """Fiche structurée (modèle, route ou classe) rendue depuis l'index, citée comme une source [Sn].

    ``covers`` liste les hits ``(source_type, source_id)`` que la fiche résume (ils ne sont plus répétés
    comme passages bruts) et ``file_ids`` les fichiers dont les morceaux bruts sont couverts de même.
    """

    ref: str = ""
    kind: str = "model"  # model | endpoint | symbol
    title: str = ""
    text: str = ""
    tokens: int = 0
    source: Optional[Source] = None
    hit: Any = None  # LocalHit équivalent (hits_by_ref, identité pour les outils)
    covers: Set[Tuple[str, Any]] = field(default_factory=set)
    file_ids: Set[int] = field(default_factory=set)


@dataclass
class ContextPack:
    """Contexte assemblé pour une question : sources citables, texte formaté et budget consommé."""

    sources: List[Source] = field(default_factory=list)
    text: str = ""
    tokens_used: int = 0
    truncated: List[Source] = field(default_factory=list)  # sources écartées faute de budget
    hits_by_ref: Dict[str, Any] = field(default_factory=dict)
    blocks: Dict[str, str] = field(default_factory=dict)  # ref -> bloc formaté (pour les outils)
    cards: List[Card] = field(default_factory=list)  # fiches retenues (en tête du contexte)


# --------------------------------------------------------------------------- accès tolérant aux hits

def _g(hit: Any, name: str, default: Any = None) -> Any:
    """Attribut ou clé d'un hit (objet ou dict), avec valeur par défaut."""
    if hit is None:
        return default
    if isinstance(hit, dict):
        value = hit.get(name, default)
    else:
        value = getattr(hit, name, default)
    return default if value is None else value


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def hit_channels(hit: Any) -> List[str]:
    return [str(c) for c in (_g(hit, "channels", []) or [])]


def hit_text(hit: Any) -> str:
    text = _g(hit, "text", "") or ""
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        text = _g(hit, "snippet", "") or ""
        if not isinstance(text, str):
            text = str(text)
    return text.rstrip("\n")


def hit_is_code(hit: Any) -> bool:
    extra = _g(hit, "extra", {}) or {}
    if isinstance(extra, dict) and extra.get("card"):
        return False  # une fiche est du texte, pas du code
    return str(_g(hit, "source_type", "file") or "file") in CODE_SOURCE_TYPES


def hit_identity(hit: Any) -> Tuple[str, Any, Optional[int], Optional[int]]:
    return (str(_g(hit, "source_type", "file")), _g(hit, "source_id"), _as_int(_g(hit, "line_start")),
            _as_int(_g(hit, "line_end")))


def hit_content_key(hit: Any) -> Optional[str]:
    """Empreinte ``chunks.sha1`` du morceau, transmise par le moteur dans ``extra.sha1`` (``None`` sinon).

    Un même PDF indexé comme fichier du dépôt et comme document téléversé donne deux morceaux de même
    empreinte : un seul entre dans le contexte. Sans empreinte (élément structuré, hit d'outil construit
    sans elle), rien n'est rapproché par le texte : deux fichiers distincts au contenu identique restent
    deux sources.
    """
    extra = _g(hit, "extra", {}) or {}
    sha = extra.get("sha1") if isinstance(extra, dict) else None
    return str(sha) if sha else None


def _language(hit: Any) -> str:
    extra = _g(hit, "extra", {}) or {}
    if isinstance(extra, dict) and extra.get("language"):
        return str(extra["language"])
    path = _g(hit, "path", "") or ""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in _LANGUAGE_BY_EXT:
        return _LANGUAGE_BY_EXT[ext]
    return "text" if not hit_is_code(hit) else ""


def _lines_label(hit: Any) -> str:
    start, end = _as_int(_g(hit, "line_start")), _as_int(_g(hit, "line_end"))
    if start is None and end is None:
        return ""
    if end is None or end == start:
        return ":L%d" % (start if start is not None else end)
    if start is None:
        return ":L%d" % end
    return ":L%d-%d" % (start, end)


def _repo_prefix(hit: Any) -> str:
    extra = _g(hit, "extra", {}) or {}
    if isinstance(extra, dict) and extra.get("repo_name"):
        return str(extra["repo_name"])
    return str(_g(hit, "repo_id", "") or "")


def _prefixed_path(prefix: str, path: str) -> str:
    """« dépôt/chemin » : le chemin d'un fichier précédé du nom de son dépôt, tel quel s'il le porte déjà.

    Un sous-dépôt « Modules/Shop » donne ainsi « Modules/Shop/app/Models/X.php », résoluble depuis la
    racine du projet ; c'est le libellé commun des passages (:func:`format_header`) et des fiches.
    """
    return "%s/%s" % (prefix, path) if prefix and not path.startswith(prefix + "/") else path


def _code_label(hit: Any) -> str:
    """Partie « (symbole X) » de l'en-tête d'un bloc de code."""
    source_type = str(_g(hit, "source_type", "file") or "file")
    title = str(_g(hit, "title", "") or "").strip()
    extra = _g(hit, "extra", {}) or {}
    if isinstance(extra, dict) and extra.get("symbol"):
        return "(symbole %s)" % extra["symbol"]
    if source_type == "symbol" and title:
        return "(symbole %s)" % title
    if source_type == "endpoint" and title:
        return "(endpoint %s)" % title
    if source_type == "model" and title:
        return "(modèle %s)" % title
    path = str(_g(hit, "path", "") or "")
    if title and title != path and title != os.path.basename(path):
        return "(%s)" % title
    return ""


def format_header(ref: str, hit: Any) -> str:
    """En-tête d'un bloc : ``[S1] repo/chemin:L10-40 (symbole X) — raison`` selon le type."""
    source_type = str(_g(hit, "source_type", "file") or "file")
    reason = str(_g(hit, "reason", "") or "").strip()
    extra = _g(hit, "extra", {}) or {}
    if not isinstance(extra, dict):
        extra = {}
    tag = "[%s]" % ref if ref else "[source non retenue]"
    if source_type == "document":
        title = str(_g(hit, "title", "") or _g(hit, "path", "") or "document")
        head = "%s Document « %s »" % (tag, title)
        section = extra.get("section") or extra.get("heading")
        if section:
            head += " §%s" % section
        page = _as_int(_g(hit, "page"))
        if page is not None:
            head += ", page %d" % page
        if reason:
            head += " — " + reason
        return head
    if source_type == "note":
        title = str(_g(hit, "title", "") or "note")
        date = extra.get("note_date") or extra.get("date")
        head = "%s Note du %s — %s" % (tag, date, title) if date else "%s Note — %s" % (tag, title)
        tags = extra.get("tags") or []
        if isinstance(tags, (list, tuple)) and tags:
            head += " [%s]" % ", ".join(str(t) for t in tags)
        if reason:
            head += " — " + reason
        return head
    path = str(_g(hit, "path", "") or _g(hit, "title", "") or "?")
    where = _prefixed_path(_repo_prefix(hit), path)
    head = "%s %s%s" % (tag, where, _lines_label(hit))
    label = _code_label(hit)
    if label:
        head += " " + label
    if reason:
        head += " — " + reason
    return head


def _fence_for(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return fence


def format_block(ref: str, hit: Any, body: Optional[str] = None) -> str:
    """Bloc complet : en-tête puis texte dans un bloc de code délimité (langage détecté)."""
    text = hit_text(hit) if body is None else body
    fence = _fence_for(text)
    return "%s\n%s%s\n%s\n%s" % (format_header(ref, hit), fence, _language(hit), text, fence)


def _block_cost(block: str, hit: Any) -> int:
    return estimate_tokens(block, is_code=hit_is_code(hit))


def truncate_block(ref: str, hit: Any, max_tokens: int) -> Optional[str]:
    """Version tronquée d'un bloc tenant dans ``max_tokens`` (coupe entre deux lignes), ou None."""
    if max_tokens < MIN_PARTIAL_TOKENS:
        return None
    lines = hit_text(hit).splitlines()
    if len(lines) <= MIN_PARTIAL_LINES:
        return None
    is_code = hit_is_code(hit)
    per_token = 3.5 if is_code else 4.0
    header = format_header(ref, hit)
    overhead = len(header) + 24 + len(TRUNCATED_MARKER) + 6
    allowed_chars = int(max_tokens * per_token) - overhead
    if allowed_chars <= 0:
        return None
    kept: List[str] = []
    used = 0
    for line in lines:
        cost = len(line) + 1
        if used + cost > allowed_chars:
            break
        kept.append(line)
        used += cost
    # vérification avec l'estimateur réel, en retirant des lignes si besoin
    while len(kept) >= MIN_PARTIAL_LINES:
        body = "\n".join(kept) + "\n" + TRUNCATED_MARKER % (len(lines) - len(kept))
        block = format_block(ref, hit, body)
        if _block_cost(block, hit) <= max_tokens:
            return block
        kept.pop()
    return None


# --------------------------------------------------------------------------- fiches structurées

_NODE_ID_RE = re.compile(r"^(\w+):(\d+)$")
_RULE_LINE_RE = re.compile(r"""^\s*['"]([\w.*\-]+)['"]\s*=>\s*(.+?)\s*,?\s*$""")
_QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")
_RULE_CALL_RE = re.compile(r"Rule::\w+\([^)]*\)")
_CLOSURE_START_RE = re.compile(r"\bfunction\b|\bfn\s*\(")
_RULE_ARRAY_MAX_LINES = 6
_REQUEST_CLASS_RE = re.compile(r"\b([A-Z]\w*Request)\b")
_EXTENDS_RE = re.compile(r"\bextends\s+([\w\\]+)")
_SIGNATURE_PREFIX_RE = re.compile(r"^\s*(?:(?:public|protected|private|static|abstract|final|async)\s+)*(?:function\s+|def\s+)?")
_WS_RE = re.compile(r"\s+")
_METHOD_ORDER = ("GET", "POST", "PUT", "PATCH", "DELETE", "ANY")


def _node_id(key: Any, prefix: str) -> Optional[int]:
    """``model:184`` → 184 si le préfixe correspond, sinon None."""
    m = _NODE_ID_RE.match(str(key or ""))
    if m and m.group(1) == prefix:
        return int(m.group(2))
    return None


def _short_class(name: Any) -> str:
    """Nom de classe sans espace de noms (``Modules\\X\\ShipmentController`` → ``ShipmentController``)."""
    s = str(name or "").strip()
    return s.rsplit("\\", 1)[-1].rsplit(".", 1)[-1] if s else ""


def _clip(text: str, limit: int) -> str:
    text = _WS_RE.sub(" ", str(text or "")).strip()
    return text if len(text) <= limit else text[: max(0, limit - 1)].rstrip() + "…"


def _lines_from_chunks(chunks: Sequence[Any]) -> Optional[List[str]]:
    """Reconstitue les lignes d'un fichier depuis ses morceaux indexés (repli quand le disque est illisible)."""
    total = 0
    pieces: List[Tuple[int, List[str]]] = []
    for c in chunks or []:
        start = _as_int(getattr(c, "line_start", None))
        text = getattr(c, "text", "") or ""
        if start is None or not text:
            continue
        meta = getattr(c, "meta", None) or {}
        offset = _as_int(meta.get("body_offset")) if isinstance(meta, dict) else None
        body = text[offset:] if offset and 0 < offset < len(text) else text
        body_lines = body.splitlines()
        pieces.append((start, body_lines))
        total = max(total, start - 1 + len(body_lines))
    if not pieces:
        return None
    lines = [""] * total
    for start, body_lines in pieces:
        for i, line in enumerate(body_lines):
            idx = start - 1 + i
            if 0 <= idx < total and not lines[idx]:
                lines[idx] = line
    return lines


class _CardLoader:
    """Accès en lot à la base (fichiers, dépôts, symboles, endpoints, lignes sur disque) le temps d'un build."""

    def __init__(self, db: Any):
        self.db = db
        self._files: Dict[int, Any] = {}
        self._repos: Optional[Dict[str, Any]] = None
        self._endpoints: Optional[List[Any]] = None
        self._symbols_by_file: Dict[int, List[Any]] = {}
        self._class_by_name: Dict[str, Any] = {}
        self._models_by_name: Dict[str, List[Any]] = {}
        self._models: Dict[int, Any] = {}
        self._lines: Dict[int, Optional[List[str]]] = {}

    def file(self, file_id: Any) -> Any:
        fid = _as_int(file_id)
        if fid is None:
            return None
        if fid not in self._files:
            try:
                self._files[fid] = self.db.get_file_by_id(fid)
            except Exception:  # noqa: BLE001
                self._files[fid] = None
        return self._files[fid]

    def path(self, file_id: Any) -> str:
        """Chemin d'un fichier relatif à son dépôt (clé de tri, nom de base) ; voir :meth:`where` pour l'affichage."""
        f = self.file(file_id)
        return str(getattr(f, "path", "") or "") if f is not None else ""

    def where(self, file_id: Any) -> str:
        """Chemin affiché d'un fichier : « dépôt/chemin », comme l'en-tête des passages (:func:`format_header`).

        Une fiche cite ainsi des fichiers résolubles depuis la racine du projet, même quand les dépôts sont des
        sous-dossiers (modules) dont deux ont chacun leur ``routes/api.php``. Vide si le fichier est inconnu.
        """
        f = self.file(file_id)
        path = str(getattr(f, "path", "") or "") if f is not None else ""
        return _prefixed_path(self.repo_name(getattr(f, "repo_id", None)), path) if path else ""

    def repo(self, repo_id: Any) -> Any:
        if self._repos is None:
            try:
                self._repos = {r.id: r for r in self.db.list_repos()}
            except Exception:  # noqa: BLE001
                self._repos = {}
        return self._repos.get(str(repo_id or ""))

    def repo_name(self, repo_id: Any) -> str:
        """Nom du dépôt (préfixe des chemins affichés), sinon son identifiant comme :func:`_repo_prefix` ; vide sans dépôt."""
        repo = self.repo(repo_id)
        return str((getattr(repo, "name", "") if repo is not None else "") or repo_id or "")

    def repo_root(self, repo_id: Any) -> Optional[str]:
        repo = self.repo(repo_id)
        path = getattr(repo, "path", None) if repo is not None else None
        return str(path) if path else None

    def model(self, model_id: Any) -> Any:
        mid = _as_int(model_id)
        if mid is None:
            return None
        if mid not in self._models:
            try:
                self._models[mid] = self.db.get_model(mid)
            except Exception:  # noqa: BLE001
                self._models[mid] = None
        return self._models[mid]

    def models_named(self, name: str) -> List[Any]:
        key = (name or "").lower()
        if key not in self._models_by_name:
            rows: List[Any] = []
            if key:
                try:
                    rows = list(self.db.find_models(name=name) or [])
                except Exception:  # noqa: BLE001
                    rows = []
            self._models_by_name[key] = rows
            for m in rows:
                if getattr(m, "id", None) is not None:
                    self._models.setdefault(int(m.id), m)
        return self._models_by_name[key]

    def models_of_table(self, table: Optional[str]) -> List[Any]:
        if not table:
            return []
        try:
            return list(self.db.find_models(table_name=table) or [])
        except Exception:  # noqa: BLE001
            return []

    def endpoints(self) -> List[Any]:
        if self._endpoints is None:
            try:
                self._endpoints = list(self.db.list_endpoints(side="server") or [])
            except Exception:  # noqa: BLE001
                self._endpoints = []
        return self._endpoints

    def file_symbols(self, file_id: Any) -> List[Any]:
        fid = _as_int(file_id)
        if fid is None:
            return []
        if fid not in self._symbols_by_file:
            try:
                self._symbols_by_file[fid] = list(self.db.list_symbols(file_id=fid) or [])
            except Exception:  # noqa: BLE001
                self._symbols_by_file[fid] = []
        return self._symbols_by_file[fid]

    def symbol(self, symbol_id: Any) -> Any:
        sid = _as_int(symbol_id)
        if sid is None:
            return None
        try:
            return self.db.get_symbol(sid)
        except Exception:  # noqa: BLE001
            return None

    def class_symbol(self, name: str) -> Any:
        """Symbole de classe portant exactement ce nom (insensible à la casse), ou None."""
        key = _short_class(name).lower()
        if not key:
            return None
        if key not in self._class_by_name:
            found = None
            try:
                rows = self.db.search_symbols(_short_class(name), limit=20) or []
            except Exception:  # noqa: BLE001
                rows = []
            for s in rows:
                if str(s.name or "").lower() == key and str(s.kind or "") in ("class", "interface", "trait", ""):
                    found = s
                    break
            self._class_by_name[key] = found
        return self._class_by_name[key]

    def method(self, class_sym: Any, name: str) -> Any:
        if class_sym is None:
            return None
        for s in self.file_symbols(class_sym.file_id):
            if str(s.name or "") == name and (s.parent or "") in ("", class_sym.name):
                return s
        return None

    def lines(self, file_row: Any) -> Optional[List[str]]:
        """Lignes d'un fichier : disque (lecture bornée, jamais hors du dépôt), sinon morceaux indexés."""
        if file_row is None or getattr(file_row, "id", None) is None:
            return None
        fid = int(file_row.id)
        if fid in self._lines:
            return self._lines[fid]
        self._lines[fid] = self._read_disk(file_row)
        if self._lines[fid] is None:
            try:
                self._lines[fid] = _lines_from_chunks(self.db.chunks_for("file", fid))
            except Exception:  # noqa: BLE001
                self._lines[fid] = None
        return self._lines[fid]

    def _read_disk(self, file_row: Any) -> Optional[List[str]]:
        root = self.repo_root(getattr(file_row, "repo_id", None))
        rel = str(getattr(file_row, "path", "") or "").replace("\\", "/").lstrip("/")
        if not root or not rel:
            return None
        try:
            root_real = os.path.realpath(os.path.expanduser(root))
            full = os.path.realpath(os.path.join(root_real, rel))
        except (OSError, ValueError):
            return None
        if full != root_real and not full.startswith(root_real.rstrip(os.sep) + os.sep):
            return None
        try:
            if not os.path.isfile(full) or os.path.getsize(full) > CARD_MAX_FILE_BYTES:
                return None
            with open(full, "rb") as fh:
                data = fh.read()
        except OSError:
            return None
        return data.decode("utf-8", "replace").splitlines()


def _field_dict(fd: Any) -> Dict[str, Any]:
    if isinstance(fd, dict):
        return fd
    return {"name": str(fd)}


def _field_enum(fd: Dict[str, Any]) -> List[str]:
    meta = fd.get("meta") if isinstance(fd.get("meta"), dict) else {}
    values = fd.get("enum_values") or meta.get("enum_values") or meta.get("values") or []
    return [str(v) for v in values] if isinstance(values, (list, tuple)) else []


def _merge_entity_fields(main: Any, migrations: Sequence[Any], loader: _CardLoader) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Champs du modèle enrichis par ses migrations et tables SQL (type, nullable, défaut, énumération).

    Les tables SQL (``sql_table``, schéma de référence ou dump) sont lues d'abord, puis les migrations par
    ordre de chemin (préfixe daté) : une migration, source de vérité du code, l'emporte sur le schéma SQL ;
    dans un même fichier, la première définition d'une colonne l'emporte (``up()`` précède ``down()``).
    Renvoie ``(champs, colonnes supplémentaires)`` : les colonnes connues des tables seulement sont listées à part.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for fd in (getattr(main, "fields", None) or []):
        d = _field_dict(fd)
        name = str(d.get("name") or "").strip()
        if not name or name in merged:
            continue
        order.append(name)
        merged[name] = {"name": name, "type": str(d.get("type") or ""), "nullable": d.get("nullable"),
                        "default": d.get("default"), "enum": _field_enum(d)}
    by_file: Dict[int, List[Any]] = {}
    for m in migrations:
        by_file.setdefault(int(m.file_id or 0), []).append(m)
    extra: List[str] = []

    def file_order(fid: int) -> Tuple[int, str, int]:
        sql_only = all(str(getattr(r, "kind", "") or "") == "sql_table" for r in by_file[fid])
        return (0 if sql_only else 1, loader.path(fid), fid)

    for fid in sorted(by_file, key=file_order):
        seen: Set[str] = set()
        for m in sorted(by_file[fid], key=lambda r: (int(r.line or 0), int(r.id or 0))):
            for fd in (getattr(m, "fields", None) or []):
                d = _field_dict(fd)
                name = str(d.get("name") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                cur = merged.get(name)
                if cur is None:
                    cur = {"name": name, "type": "", "nullable": None, "default": None, "enum": []}
                    merged[name] = cur
                    extra.append(name)
                if d.get("type"):
                    cur["type"] = str(d.get("type"))
                if d.get("nullable") is not None:
                    cur["nullable"] = bool(d.get("nullable"))
                if d.get("default") is not None:
                    cur["default"] = d.get("default")
                enum = _field_enum(d)
                if enum:
                    cur["enum"] = enum
    return [merged[n] for n in order], extra


def _render_field(fd: Dict[str, Any], with_type: bool = True) -> str:
    parts: List[str] = []
    ftype = str(fd.get("type") or "")
    enum = fd.get("enum") or []
    if with_type:
        if enum:
            parts.append("enum %s" % "|".join(enum[:8]) + ("|…" if len(enum) > 8 else ""))
        elif ftype:
            parts.append(ftype)
        if fd.get("nullable") is True:
            parts.append("nullable")
        elif fd.get("nullable") is False:
            parts.append("requis")
        if fd.get("default") not in (None, ""):
            parts.append("défaut %s" % fd.get("default"))
    return "%s (%s)" % (fd["name"], ", ".join(parts)) if parts else str(fd["name"])


def _render_fields(fields: Sequence[Dict[str, Any]], limit: int = CARD_MAX_FIELDS) -> str:
    if not fields:
        return "aucun champ connu"
    out = ", ".join(_render_field(f) for f in fields[:limit])
    if len(fields) > limit:
        out += ", … (+%d)" % (len(fields) - limit)
    return out


def _entity_models(loader: _CardLoader, model: Any) -> Tuple[Any, List[Any]]:
    """Modèle principal (Eloquent de préférence) et migrations de la même entité (``same_entity`` / même table)."""
    ids: Set[int] = {int(model.id)}
    key = "model:%d" % int(model.id)
    try:
        for r in loader.db.relations_from(key, "same_entity"):
            mid = _node_id(r.dst, "model")
            if mid is not None:
                ids.add(mid)
        for r in loader.db.relations_to(key, "same_entity"):
            mid = _node_id(r.src, "model")
            if mid is not None:
                ids.add(mid)
    except Exception:  # noqa: BLE001
        pass
    rows: List[Any] = [model]
    for mid in sorted(ids):
        if mid == int(model.id):
            continue
        row = loader.model(mid)
        if row is not None:
            rows.append(row)
    for row in loader.models_of_table(getattr(model, "table_name", None)):
        if getattr(row, "id", None) is not None and int(row.id) not in ids:
            ids.add(int(row.id))
            rows.append(row)
    migrations = [r for r in rows if str(getattr(r, "kind", "") or "") in TABLE_LIKE_KINDS]
    others = [r for r in rows if str(getattr(r, "kind", "") or "") not in TABLE_LIKE_KINDS]
    if str(getattr(model, "kind", "") or "") not in TABLE_LIKE_KINDS:
        main = model
    elif others:
        main = sorted(others, key=lambda r: (0 if str(r.kind or "") == "eloquent" else 1, int(r.id or 0)))[0]
    else:
        main = model
    return main, migrations


def _extract_rule_entries(body: Sequence[str]) -> List[str]:
    """``'champ' => règles`` compactés : ``date_debut : required|date`` (closures signalées, ``Rule::x(...)`` gardés).

    Un tableau ouvert sur sa ligne (``'unite_id' => [``) est complété par les lignes suivantes jusqu'au ``]``.
    """
    out: List[str] = []
    lines = list(body)
    for i, raw in enumerate(lines):
        m = _RULE_LINE_RE.match(raw)
        if not m:
            continue
        name, value = m.group(1), m.group(2).strip()
        if value.rstrip(" ,") == "[":  # tableau ouvert seul sur sa ligne (pas une closure entamée)
            for extra in lines[i + 1:i + 1 + _RULE_ARRAY_MAX_LINES]:
                value += " " + extra.strip()
                if "]" in extra:
                    break
        if value[:1] in ("'", '"'):
            rules = value.strip("'\"")
            rules = rules.split("'")[0].split('"')[0]
        elif value.startswith("["):
            head = _CLOSURE_START_RE.split(value, 1)[0]  # les chaînes d'une closure (messages d'erreur) ne sont pas des règles
            parts = [q for q in _QUOTED_RE.findall(head) if q and not q.startswith("@")]
            parts += _RULE_CALL_RE.findall(head)
            rules = "|".join(parts) or _clip(head, 60)
            if head != value:
                rules += "|(closure)"
        else:
            rules = _clip(value, 60)
        if rules:
            out.append("%s : %s" % (name, rules))
    return out


def _rules_summary(loader: _CardLoader, class_sym: Any) -> Optional[Tuple[str, List[str]]]:
    """``(où, entrées)`` de la méthode ``rules()`` d'une classe de requête ; None sans méthode ``rules``.

    Les entrées sont les ``champ : règles`` compactés (ou le corps brut compacté, seul, quand rien n'est
    reconnu) ; liste vide quand le fichier n'a pu être lu.
    """
    rules = loader.method(class_sym, "rules")
    if rules is None:
        return None
    f = loader.file(class_sym.file_id)
    where = "%s:%s" % (loader.where(class_sym.file_id) or "?", rules.line_start or class_sym.line_start or "?")
    lines = loader.lines(f)
    start = _as_int(rules.line_start)
    if lines is None or start is None or start < 1 or start > len(lines):
        return where, []
    end = _as_int(rules.line_end) or (start + CARD_RULES_MAX_LINES - 1)
    end = min(end, start + CARD_RULES_MAX_LINES - 1, len(lines))
    body = lines[start - 1:end]
    entries = _extract_rule_entries(body)
    if not entries:
        raw = _clip(" ".join(l.strip() for l in body if l.strip()), CARD_RULES_MAX_CHARS)
        entries = [raw] if raw else []
    return where, entries


def _join_rule_entries(entries: Sequence[str], max_chars: int) -> str:
    """Entrées jointes par « ; » dans ``max_chars`` caractères, entières ; les suivantes sont comptées (« … (+N) »)."""
    out: List[str] = []
    used = 0
    for i, entry in enumerate(entries):
        cost = len(entry) + (3 if out else 0)
        if used + cost > max_chars and out:
            return " ; ".join(out) + " ; … (+%d)" % (len(entries) - i)
        out.append(entry if used + cost <= max_chars else _clip(entry, max_chars))
        used += cost
    return " ; ".join(out)


def _rules_variants(entries: Sequence[str]) -> List[str]:
    """Texte des règles, de la version la plus longue (``CARD_RULES_MAX_CHARS``) aux plus courtes, sans doublon."""
    out: List[str] = []
    for max_chars in (CARD_RULES_MAX_CHARS,) + tuple(CARD_RULES_SHORT_CHARS):
        text = _join_rule_entries(entries, max_chars)
        if text and (not out or len(text) < len(out[-1])):
            out.append(text)
    return out


def _request_section(loader: _CardLoader, class_name: str, label: str) -> Optional[List[str]]:
    """Variantes (de la plus longue à la plus courte) de la ligne d'une classe de requête, None si inconnue."""
    sym = loader.class_symbol(class_name)
    if sym is None:
        return None
    summary = _rules_summary(loader, sym)
    if summary is None:
        return ["%s : %s (%s:%s)" % (label, sym.name, loader.where(sym.file_id) or "?", sym.line_start or "?")]
    where, entries = summary
    if not entries:
        return ["%s : %s (%s) — rules() non lues (ouvrir le fichier)" % (label, sym.name, where)]
    return ["%s : %s (%s) rules() : %s" % (label, sym.name, where, rules) for rules in _rules_variants(entries)]


def _route_label(e: Any) -> str:
    path = str(getattr(e, "path", "") or getattr(e, "normalized_path", "") or "")
    if path and not path.startswith("/"):
        path = "/" + path
    return ("%s %s" % (str(getattr(e, "method", "") or "").upper(), path)).strip()


def _route_prefix(e: Any) -> str:
    meta = getattr(e, "meta", None) or {}
    full = str(meta.get("full_path") or "") if isinstance(meta, dict) else ""
    path = str(getattr(e, "path", "") or "")
    if full and path:
        p = path if path.startswith("/") else "/" + path
        if full.endswith(p):
            return full[: len(full) - len(p)]
    return ""


def _routes_for_model(loader: _CardLoader, main: Any) -> List[Any]:
    """Routes serveur dont le handler est ``<Modèle>Controller@…``, un contrôleur qui utilise le modèle, ou
    qui sont reliées au modèle par le graphe (``uses_model`` porté par la route elle-même, ``endpoint:E``)."""
    name = str(getattr(main, "name", "") or "")
    if not name:
        return []
    primary = (name + "Controller").lower()
    controllers: Set[str] = {primary}
    direct: Set[int] = set()  # routes reliées au modèle par le graphe (gestionnaire nommé d'après lui ou l'important)
    try:
        for r in loader.db.relations_to("model:%d" % int(main.id), "uses_model"):
            eid = _node_id(r.src, "endpoint")
            if eid is not None:
                direct.add(eid)
                continue
            fid = _node_id(r.src, "file")
            if fid is None:
                continue
            for s in loader.file_symbols(fid):
                if str(s.kind or "") == "class" and str(s.name or "").endswith("Controller"):
                    controllers.add(str(s.name).lower())
    except Exception:  # noqa: BLE001
        pass
    found: List[Tuple[int, str, int, int, Any]] = []
    for e in loader.endpoints():
        handler = str(getattr(e, "handler", "") or "")
        cls = _short_class(handler.split("@", 1)[0].split("::", 1)[0]).lower()
        if cls in controllers or (e.id is not None and int(e.id) in direct):
            rank = 0 if cls == primary else 1
            method = str(e.method or "").upper()
            method_rank = _METHOD_ORDER.index(method) if method in _METHOD_ORDER else len(_METHOD_ORDER)
            found.append((rank, str(e.normalized_path or e.path or ""), method_rank, int(e.id or 0), e))
    found.sort(key=lambda t: t[:4])
    return [t[4] for t in found]


def _render_routes(routes: Sequence[Any], limit: Optional[int] = None) -> str:
    limit = CARD_MAX_ROUTES if limit is None else limit
    prefixes = {_route_prefix(e) for e in routes}
    prefix = prefixes.pop() if len(prefixes) == 1 else ""
    parts: List[str] = []
    last_cls = ""
    for e in routes[:limit]:
        handler = str(getattr(e, "handler", "") or "")
        cls = _short_class(handler.split("@", 1)[0]) if "@" in handler else ""
        if cls and cls == last_cls:
            shown = "@" + handler.split("@", 1)[1]
        else:
            shown = _short_class(handler.split("@", 1)[0]) + ("@" + handler.split("@", 1)[1] if "@" in handler else "")
        last_cls = cls or last_cls
        parts.append("%s → %s" % (_route_label(e), shown or "?"))
    text = ", ".join(parts)
    if len(routes) > limit:
        text += ", … (+%d)" % (len(routes) - limit)
    if prefix:
        text = "(préfixe %s) " % prefix + text
    return text


def _laravel_sections(loader: _CardLoader, model_name: str, seen_requests: Set[str]) -> List[Tuple[int, Any]]:
    """Conventions Laravel (tolérantes) : Store/Update<Modèle>Request, <Modèle>Policy, <Modèle>Swagger.

    Les règles de création (priorité 2) passent avant la liste des routes (3) : à la question « quels sont les
    champs / leur définition », ce sont elles qui répondent, et les routes se raccourcissent mieux.
    """
    out: List[Tuple[int, Any]] = []
    for prefix, label in (("Store", "validation (création)"), ("Update", "validation (mise à jour)")):
        cls = "%s%sRequest" % (prefix, model_name)
        if cls.lower() in seen_requests:
            continue
        section = _request_section(loader, cls, label)
        if section:
            seen_requests.add(cls.lower())
            out.append((2 if prefix == "Store" else 5, section))
    policy = loader.class_symbol(model_name + "Policy")
    if policy is not None:
        out.append((8, "policy : %s (%s:%s)" % (policy.name, loader.where(policy.file_id) or "?", policy.line_start or "?")))
    swagger = loader.class_symbol(model_name + "Swagger")
    if swagger is not None:
        out.append((7, "documentation Swagger : %s" % (loader.where(swagger.file_id) or swagger.name)))
    return out


def _fit_sections(header: str, sections: Sequence[Tuple[int, Any]], max_tokens: int) -> str:
    """Texte de la fiche tenant dans ``max_tokens``.

    Une section est ``(priorité, texte)`` ou ``(priorité, [texte, variante plus courte, …])`` (1 = la plus
    importante). Lignes bornées, puis, tant que la fiche déborde, la section la moins prioritaire est
    raccourcie (variante suivante) avant d'être retirée ; la place restante est enfin rendue, par priorité,
    aux sections raccourcies ou retirées (leur variante la plus longue qui tient encore). Une section n'est
    donc jamais retirée tant qu'une moins prioritaire pouvait encore être raccourcie.
    """
    header = _clip(header, CARD_LINE_MAX_CHARS)
    entries: List[Tuple[int, int, List[str]]] = []  # (priorité, rang d'affichage, variantes de la plus longue à la plus courte)
    for rank, (prio, variants) in enumerate(sections):
        texts: List[str] = []
        for t in ([variants] if isinstance(variants, str) else list(variants or [])):
            t = _clip(t, CARD_LINE_MAX_CHARS)
            if t and (not texts or len(t) < len(texts[-1])):  # une variante qui ne raccourcit pas ne sert à rien
                texts.append(t)
        if texts:
            entries.append((int(prio), rank, texts))
    chosen: Dict[int, int] = {rank: 0 for _, rank, _ in entries}  # rang -> variante affichée (absent = retirée)

    def render() -> str:
        lines = [header]
        for _, rank, texts in sorted(entries, key=lambda e: e[1]):
            if rank in chosen:
                lines.append("· " + texts[chosen[rank]])
        return "\n".join(lines)

    text = render()
    while estimate_tokens(text) > max_tokens and chosen:
        _, rank, texts = max((e for e in entries if e[1] in chosen), key=lambda e: (e[0], e[1]))
        if chosen[rank] + 1 < len(texts):
            chosen[rank] += 1  # la section la moins prioritaire se raccourcit…
        else:
            del chosen[rank]  # … puis part
        text = render()
    for _, rank, texts in sorted(entries, key=lambda e: (e[0], e[1])):
        current = chosen.get(rank, len(texts))
        for variant in range(current):  # variantes plus longues que l'actuelle, la plus longue d'abord
            chosen[rank] = variant
            candidate = render()
            if estimate_tokens(candidate) <= max_tokens:
                text = candidate
                break
        else:
            if current < len(texts):
                chosen[rank] = current
            else:
                chosen.pop(rank, None)
    if estimate_tokens(text) > max_tokens:
        text = text[: max(0, int(max_tokens * 4) - 1)].rstrip() + "…"
    return text


def _section_shown(text: str, section: str) -> bool:
    """Vrai si la fiche ``text`` affiche la ligne ``section`` telle quelle (bornée comme dans ``_fit_sections``)."""
    return "· " + _clip(section, CARD_LINE_MAX_CHARS) in text.splitlines()


def _route_limits() -> List[int]:
    """Nombres de routes affichées, du plus grand au plus petit (``CARD_MAX_ROUTES`` puis ``CARD_ROUTES_SHORT``)."""
    return [CARD_MAX_ROUTES] + sorted((n for n in CARD_ROUTES_SHORT if 0 < n < CARD_MAX_ROUTES), reverse=True)


def _make_card(kind: str, title: str, text: str, source_type: str, source_id: Any, file_row: Any,
               line_start: Any, line_end: Any, extra: Dict[str, Any]) -> Card:
    hit = LocalHit(source_type=source_type, source_id=_as_int(source_id), title=title,
                   path=getattr(file_row, "path", None) if file_row is not None else None,
                   repo_id=getattr(file_row, "repo_id", None) if file_row is not None else None,
                   file_id=_as_int(getattr(file_row, "id", None)) if file_row is not None else None,
                   line_start=_as_int(line_start), line_end=_as_int(line_end), text=text, snippet=_clip(text, 300),
                   score=0.0, channels=[CARD_CHANNEL], reason=CARD_REASON, extra=dict(extra))
    return Card(kind=kind, title=title, text=text, tokens=estimate_tokens(text), hit=hit)


def _model_card(loader: _CardLoader, model: Any, seen_requests: Set[str]) -> Optional[Card]:
    main, migrations = _entity_models(loader, model)
    f = loader.file(main.file_id)
    path = loader.where(main.file_id)
    name = str(main.name or "")
    is_migration = str(main.kind or "") in TABLE_LIKE_KINDS  # table définie sans modèle de classe (migration ou SQL)
    table = str(main.table_name or "")
    where = "%s:%s" % (path or "?", main.line or "?")
    if is_migration:
        header = "Table %s (%s, %s)" % (table or name, "migration" if str(main.kind or "") == "migration" else "table SQL", where)
    else:
        header = "Modèle %s (%s%s)" % (name, ("table %s, " % table) if table and table != name else "", where)
    fields, extra_cols = _merge_entity_fields(main, migrations, loader)
    sections: List[Tuple[int, Any]] = [(1, "champs : " + _render_fields(fields))]
    if extra_cols:
        sections.append((6, "autres colonnes (migrations) : " + ", ".join(extra_cols[:CARD_MAX_FIELDS])))
    rels: List[str] = []
    for r in (getattr(main, "relations", None) or [])[:CARD_MAX_RELATIONS]:
        if isinstance(r, dict):
            label = "%s %s" % (r.get("kind") or "relation", r.get("target") or "?")
            if r.get("field"):
                label += " (%s)" % r.get("field")
            rels.append(label)
        else:
            rels.append(str(r))
    if rels:
        sections.append((4, "relations : " + ", ".join(rels)))
    mig_files: List[str] = []
    for m in sorted(migrations, key=lambda r: (loader.path(r.file_id), int(r.line or 0))):
        base = os.path.basename(loader.path(m.file_id))
        if base and base not in mig_files:
            mig_files.append(base)
    if mig_files:
        label = "migrations" if all(str(m.kind or "") == "migration" for m in migrations) else "migrations / tables SQL"
        limits = [CARD_MAX_MIGRATIONS] + sorted((n for n in CARD_MIGRATIONS_SHORT if 0 < n < CARD_MAX_MIGRATIONS), reverse=True)
        sections.append((6, ["%s : %s%s" % (label, ", ".join(mig_files[:n]), ", … (+%d)" % (len(mig_files) - n) if len(mig_files) > n else "")
                             for n in limits]))  # la liste se raccourcit (première migration + compte) avant de partir
    routes = _routes_for_model(loader, main) if not is_migration else []
    route_sections = ["routes : " + _render_routes(routes, n) for n in _route_limits()] if routes else []
    if routes:
        sections.append((3, route_sections))  # la liste se raccourcit (4, puis 2 routes) avant de partir
    if not is_migration and name:
        sections.extend(_laravel_sections(loader, name, seen_requests))
    text = _fit_sections(header, sections, CARD_MAX_TOKENS)
    title = ("Table %s" % (table or name)) if is_migration else ("Modèle %s" % name)
    card = _make_card("model", title, text, "model", main.id, f, main.line, main.line,
                      {"model_id": main.id, "kind": main.kind or "", "table_name": table, "card": "model",
                       "repo_name": loader.repo_name(getattr(f, "repo_id", None)),
                       "fields": [fd["name"] for fd in fields][:50], "language": getattr(f, "language", "") or ""})
    card.covers.add(("model", int(main.id)))
    for limit, section in zip(_route_limits(), route_sections):
        if _section_shown(text, section):  # routes affichées : leurs hits bruts sont couverts par la fiche
            for e in routes[:limit]:
                if e.id is not None:
                    card.covers.add(("endpoint", int(e.id)))
            break
    for m in migrations:
        card.covers.add(("model", int(m.id)))
        if m.file_id is not None:
            card.file_ids.add(int(m.file_id))
    if main.file_id is not None:
        card.file_ids.add(int(main.file_id))
        for s in loader.file_symbols(main.file_id):
            if str(s.kind or "") == "class" and str(s.name or "") == name and s.id is not None:
                card.covers.add(("symbol", int(s.id)))
    return card


def _handler_symbol(loader: _CardLoader, e: Any) -> Any:
    try:
        for r in loader.db.relations_from("endpoint:%d" % int(e.id), "handles"):
            sid = _node_id(r.dst, "symbol")
            if sid is not None:
                s = loader.symbol(sid)
                if s is not None:
                    return s
    except Exception:  # noqa: BLE001
        pass
    handler = str(getattr(e, "handler", "") or "")
    if "@" in handler:
        cls, method = handler.split("@", 1)
        return loader.method(loader.class_symbol(cls), method)
    return None


def _model_for_controller(loader: _CardLoader, controller: str, controller_file_id: Any) -> Any:
    base = _short_class(controller)
    name = base[: -len("Controller")] if base.endswith("Controller") else ""
    candidates = [m for m in loader.models_named(name) if str(m.kind or "") not in TABLE_LIKE_KINDS] if name else []
    if candidates:
        return candidates[0]
    if controller_file_id is not None:
        try:
            for r in loader.db.relations_from("file:%d" % int(controller_file_id), "uses_model"):
                mid = _node_id(r.dst, "model")
                m = loader.model(mid) if mid is not None else None
                if m is not None and str(m.kind or "") not in TABLE_LIKE_KINDS:
                    if name and str(m.name or "").lower() == name.lower():
                        return m
                    candidates.append(m)
        except Exception:  # noqa: BLE001
            pass
    return candidates[0] if candidates else None


def _endpoint_card(loader: _CardLoader, e: Any, seen_requests: Set[str], seen_models: Set[int],
                   request_class: str = "") -> Optional[Card]:
    """Fiche d'une route serveur. ``request_class`` est la classe de validation déjà résolue par le moteur de
    recherche (``extra.request_class`` d'un hit du canal « route »), utilisée quand la signature du gestionnaire
    n'en nomme pas ; les conventions ``Store|Update<Modèle>Request`` ne servent qu'en dernier recours."""
    if str(getattr(e, "side", "server") or "server") != "server":
        return None
    f = loader.file(e.file_id)
    path = loader.where(e.file_id)
    handler = str(getattr(e, "handler", "") or "")
    header = "Route %s (%s:%s)" % (_route_label(e), path or "?", e.line or "?")
    if handler:
        header += " → %s" % handler
    meta = getattr(e, "meta", None) or {}
    if isinstance(meta, dict):
        if meta.get("full_path"):
            header += " · chemin complet : %s" % meta["full_path"]
        if meta.get("name"):
            header += " · nom : %s" % meta["name"]
        mw = meta.get("middleware")
        if isinstance(mw, (list, tuple)) and mw:
            header += " · middleware : %s" % ", ".join(str(m) for m in mw[:6])
    sections: List[Tuple[int, Any]] = []
    sym = _handler_symbol(loader, e)
    controller = handler.split("@", 1)[0] if "@" in handler else ""
    controller_file_id = sym.file_id if sym is not None else None
    if sym is not None:
        sig = _SIGNATURE_PREFIX_RE.sub("", str(sym.signature or sym.name or ""))
        sections.append((2, "contrôleur : %s (%s:%s-%s)" % (sig or sym.name, loader.where(sym.file_id) or "?",
                                                           sym.line_start or "?", sym.line_end or sym.line_start or "?")))
    request_cls = ""
    if sym is not None:
        m = _REQUEST_CLASS_RE.search(str(sym.signature or ""))
        if m:
            request_cls = m.group(1)
    if not request_cls and request_class:
        request_cls = _short_class(request_class)
    model = _model_for_controller(loader, controller, controller_file_id) if controller else None
    if not request_cls and model is not None and "@" in handler:
        method = handler.split("@", 1)[1]
        if method in ("store", "create"):
            request_cls = "Store%sRequest" % model.name
        elif method in ("update", "edit"):
            request_cls = "Update%sRequest" % model.name
    if request_cls and request_cls.lower() not in seen_requests:
        section = _request_section(loader, request_cls, "classe de requête")
        if section:
            seen_requests.add(request_cls.lower())
            sections.append((1, section))
    elif request_cls:
        sections.append((1, "classe de requête : %s (règles : voir la fiche du modèle)" % request_cls))
    if model is not None:
        if int(model.id) in seen_models:
            sections.append((4, "modèle principal : %s (champs : voir sa fiche)" % model.name))
        else:
            main, migrations = _entity_models(loader, model)
            fields, _ = _merge_entity_fields(main, migrations, loader)
            table = str(main.table_name or "")
            sections.append((3, "modèle principal : %s (%s%s:%s) — champs : %s" % (
                main.name, ("table %s, " % table) if table and table != main.name else "", loader.where(main.file_id) or "?",
                main.line or "?", _render_fields(fields, 25))))
            seen_models.add(int(model.id))  # une autre route ne répète pas les champs
        swagger = loader.class_symbol(str(model.name) + "Swagger")
        if swagger is not None:
            sections.append((6, "documentation Swagger : %s" % (loader.where(swagger.file_id) or swagger.name)))
    text = _fit_sections(header, sections, CARD_MAX_TOKENS)
    card = _make_card("endpoint", "Route %s" % _route_label(e), text, "endpoint", e.id, f, e.line, e.line,
                      {"endpoint_id": e.id, "side": e.side, "method": str(e.method or "").upper(), "route": e.path,
                       "normalized_path": e.normalized_path, "handler": handler, "framework": e.framework or "",
                       "card": "endpoint", "repo_name": loader.repo_name(getattr(f, "repo_id", None))})
    card.covers.add(("endpoint", int(e.id)))
    return card


def _symbol_card(loader: _CardLoader, sym: Any, seen_requests: Set[str]) -> Optional[Card]:
    f = loader.file(sym.file_id)
    path = loader.where(sym.file_id)
    sig = _clip(str(sym.signature or ""), 160)
    lines_label = "%s" % (sym.line_start or "?")
    if sym.line_end and sym.line_end != sym.line_start:
        lines_label += "-%s" % sym.line_end
    header = "Classe %s (%s:%s)" % (sym.name, path or "?", lines_label)
    if sig and sig != sym.name:
        header += " — " + sig
    sections: List[Tuple[int, Any]] = []
    meta = getattr(sym, "meta", None) or {}
    parent = ""
    if isinstance(meta, dict):
        parent = str(meta.get("extends") or meta.get("parent") or meta.get("base") or "")
        if not parent and isinstance(meta.get("bases"), (list, tuple)) and meta["bases"]:
            parent = ", ".join(str(b) for b in meta["bases"][:3])
    if not parent:
        m = _EXTENDS_RE.search(str(sym.signature or ""))
        parent = m.group(1) if m else ""
    if parent:
        sections.append((3, "parent : %s" % _short_class(parent)))
    if isinstance(meta, dict) and isinstance(meta.get("traits"), (list, tuple)) and meta["traits"]:
        sections.append((6, "traits : %s" % ", ".join(_short_class(t) for t in meta["traits"][:8])))
    methods: List[str] = []
    total = 0
    for s in loader.file_symbols(sym.file_id):
        if (s.parent or "") != str(sym.name or "") or str(s.kind or "") not in ("method", "function", ""):
            continue
        smeta = getattr(s, "meta", None) or {}
        visibility = str(smeta.get("visibility") or "public") if isinstance(smeta, dict) else "public"
        if visibility != "public" or str(s.name or "").startswith("_") and str(s.name) != "__construct":
            continue
        total += 1
        if len(methods) < CARD_MAX_METHODS:
            shown = _SIGNATURE_PREFIX_RE.sub("", str(s.signature or s.name or ""))
            methods.append("%s L%s" % (_clip(shown or s.name, 90), s.line_start or "?"))
    if methods:
        label = "méthodes publiques (%d) : " % total + " ; ".join(methods)
        if total > len(methods):
            label += " ; … (+%d)" % (total - len(methods))
        sections.append((2, label))
    rules = _rules_summary(loader, sym) if loader.method(sym, "rules") is not None else None
    if rules is not None:
        where, entries = rules
        seen_requests.add(str(sym.name or "").lower())
        sections.append((1, ["règles de validation (rules(), %s) : %s" % (where, r)
                             for r in (_rules_variants(entries) or ["non lues (ouvrir le fichier)"])]))
    text = _fit_sections(header, sections, CARD_MAX_TOKENS)
    card = _make_card("symbol", "Classe %s" % sym.name, text, "symbol", sym.id, f, sym.line_start, sym.line_end,
                      {"symbol_id": sym.id, "kind": sym.kind, "signature": sym.signature or "", "parent": sym.parent or "",
                       "card": "symbol", "language": getattr(f, "language", "") or "",
                       "repo_name": loader.repo_name(getattr(f, "repo_id", None))})
    card.covers.add(("symbol", int(sym.id)))
    return card


def build_cards(db: Any, hits: Sequence[Any], max_cards: int = MAX_CARDS) -> List[Card]:
    """Fiches structurées pour les hits modèle / route / classe parmi les ``CARD_HITS_WINDOW`` premiers.

    Une fiche par entité : un modèle et ses migrations (``same_entity``) ne donnent qu'une fiche, le
    symbole de classe d'un modèle devient la fiche du modèle, une route dont le modèle a déjà sa fiche
    n'en répète ni les champs ni les règles. Ne lève jamais : une fiche impossible est simplement omise.
    """
    if db is None or not hits or max_cards <= 0:
        return []
    loader = _CardLoader(db)
    window = [h for h in list(hits)[:CARD_HITS_WINDOW] if h is not None]

    def hit_id(h: Any, key: str) -> Optional[int]:
        extra = _g(h, "extra", {}) or {}
        if isinstance(extra, dict) and extra.get(key) is not None:
            return _as_int(extra.get(key))
        return _as_int(_g(h, "source_id"))

    def model_of(h: Any) -> Any:
        """Modèle principal visé par un hit modèle ou par le symbole de classe d'un modèle, sinon None."""
        source_type = str(_g(h, "source_type", "file") or "file")
        try:
            if source_type == "model":
                model = loader.model(hit_id(h, "model_id"))
                return _entity_models(loader, model)[0] if model is not None and model.id is not None else None
            if source_type == "symbol":
                sym = loader.symbol(hit_id(h, "symbol_id"))
                if sym is not None and str(sym.kind or "") == "class":
                    for m in loader.models_named(str(sym.name or "")):
                        if m.file_id == sym.file_id and str(m.kind or "") not in TABLE_LIKE_KINDS:
                            return m
        except Exception as exc:  # noqa: BLE001
            logger.debug("modèle du hit introuvable: %s", exc)
        return None

    # Pré-passe : les modèles qui auront leur fiche et les routes que ces fiches afficheront ; une route de
    # cette liste n'a pas de fiche à elle, quel que soit l'ordre des hits, et sa fiche éventuelle ne répète
    # pas les champs d'un modèle fiché.
    planned_models: Dict[int, Any] = {}
    planned_endpoints: Set[int] = set()
    for h in window:
        main = model_of(h)
        if main is not None and int(main.id) not in planned_models and len(planned_models) < max_cards:
            planned_models[int(main.id)] = main
    for main in planned_models.values():
        try:
            for e in _routes_for_model(loader, main)[:CARD_MAX_ROUTES]:
                if e.id is not None:
                    planned_endpoints.add(int(e.id))
        except Exception:  # noqa: BLE001
            continue

    cards: List[Card] = []
    covered: Set[Tuple[str, Any]] = set()
    seen_requests: Set[str] = set()
    seen_models: Set[int] = set(planned_models)
    seen_controllers: Set[str] = set()  # une fiche de route par contrôleur : ses autres routes restent des passages
    # Deux passes : les fiches de modèle d'abord (elles portent champs, règles et routes de l'entité), puis les
    # routes et classes, qui renvoient à ces fiches au lieu de répéter leur contenu.
    ordered = [h for h in window if model_of(h) is not None] + [h for h in window if model_of(h) is None]
    for h in ordered:
        if len(cards) >= max_cards:
            break
        source_type = str(_g(h, "source_type", "file") or "file")
        source_id = _as_int(_g(h, "source_id"))
        if source_type not in ("model", "endpoint", "symbol") or (source_type, source_id) in covered:
            continue
        card: Optional[Card] = None
        key: Tuple[str, Any] = (source_type, source_id)
        try:
            main = model_of(h) if source_type in ("model", "symbol") else None
            if main is not None:
                key = ("model", int(main.id))
                if key not in covered:
                    card = _model_card(loader, main, seen_requests)
            elif source_type == "endpoint":
                ep = db.get_endpoint(hit_id(h, "endpoint_id"))
                if ep is not None and ep.id is not None:
                    key = ("endpoint", int(ep.id))
                    handler = str(ep.handler or "")
                    controller = _short_class(handler.split("@", 1)[0]).lower() if "@" in handler else ""
                    if key not in covered and int(ep.id) not in planned_endpoints and (not controller or controller not in seen_controllers):
                        extra = _g(h, "extra", {}) or {}
                        hint = str(extra.get("request_class") or "") if isinstance(extra, dict) else ""
                        card = _endpoint_card(loader, ep, seen_requests, seen_models, request_class=hint)
                        if card is not None and controller:
                            seen_controllers.add(controller)
            elif source_type == "symbol":
                sym = loader.symbol(hit_id(h, "symbol_id"))
                if sym is not None and sym.id is not None and str(sym.kind or "") == "class":
                    key = ("symbol", int(sym.id))
                    if key not in covered:
                        card = _symbol_card(loader, sym, seen_requests)
        except Exception as exc:  # noqa: BLE001 - une fiche impossible n'empêche pas les autres
            logger.debug("fiche impossible pour %s %s: %s", source_type, source_id, exc)
            card = None
        if card is None or not card.text.strip() or key in covered:
            continue
        covered |= card.covers | {key}
        cards.append(card)
    return cards


def format_card_block(ref: str, card: Card) -> str:
    """Bloc d'une fiche : ``[S1] Modèle Shipment (…)`` puis ses lignes « · … » (texte, pas de code)."""
    tag = "[%s]" % ref if ref else "[fiche]"
    return "%s %s" % (tag, card.text)


def _card_covers_hit(hit: Any, keys: Set[Tuple[str, Any]], file_ids: Set[int]) -> bool:
    source_type = str(_g(hit, "source_type", "file") or "file")
    source_id = _as_int(_g(hit, "source_id"))
    if (source_type, source_id) in keys:
        return True
    if source_type == "file":
        fid = _as_int(_g(hit, "file_id"))
        if fid is None:
            fid = source_id
        return fid is not None and fid in file_ids
    return False


# --------------------------------------------------------------------------- sélection

def _is_generated_hit(hit: Any) -> bool:
    """Passage d'un fichier généré volumineux : étiqueté ``extra.generated`` par le canal lexical du moteur,
    ou morceau de fichier au chemin typique (:func:`develop.search.engine.is_generated_file` : ``api-docs``,
    ``swagger.json``, ``storage/``, ``*.lock``, ``*.min.*``) remonté par un autre canal sans l'étiquette."""
    extra = _g(hit, "extra", {}) or {}
    if isinstance(extra, dict) and extra.get("generated"):
        return True
    if str(_g(hit, "source_type", "file") or "file") != "file":
        return False
    return is_generated_file(str(_g(hit, "path", "") or ""), None)


def _without_generated(hits: Sequence[Any]) -> Tuple[List[Any], List[Any]]:
    """Écarte les passages des fichiers générés volumineux (``storage/api-docs/api-docs.json``, dumps, ``*.lock``…).

    Le moteur les pénalise et les classe en queue, mais un grand budget (16k) les laissait entrer : sur un
    projet réel, trois schémas OpenAPI tronqués (1 358 tokens) qui n'apportaient rien à la réponse. Un tel
    passage reste candidat si la question cite le fichier (canal « identifier » : chemin ou nom de fichier) ;
    les autres n'entrent qu'à défaut de tout autre passage. Renvoie ``(gardés, écartés)`` dans l'ordre reçu.
    """
    kept: List[Any] = []
    dropped: List[Any] = []
    for h in hits:
        if _is_generated_hit(h) and "identifier" not in hit_channels(h):
            dropped.append(h)
        else:
            kept.append(h)
    if dropped and not any(hit_text(h).strip() for h in kept):
        return list(hits), []  # rien d'autre à montrer : mieux vaut un passage faible qu'un contexte vide
    return kept, dropped


def _dedupe_and_cap(hits: Sequence[Any], max_per_file: int) -> List[Any]:
    """Trie par score décroissant (stable), supprime les doublons et limite le nombre par fichier.

    Doublons : même ``chunk_id``, même (source, lignes), même passage déclaré deux fois sur les mêmes
    lignes, et même texte indexé sous deux sources (:func:`hit_content_key`, PDF du dépôt = document
    téléversé) : la copie la mieux classée reste, l'autre ne consomme ni budget ni place par fichier.
    """
    indexed = [(i, h) for i, h in enumerate(hits) if h is not None]

    def key(item: Tuple[int, Any]) -> Tuple[float, int]:
        try:
            score = float(_g(item[1], "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (-score, item[0])

    indexed.sort(key=key)
    seen: Set[Tuple[str, Any, Optional[int], Optional[int]]] = set()
    seen_spans: Set[Tuple[str, Optional[int], Optional[int], str]] = set()  # même fichier, mêmes lignes, même texte
    seen_chunks: Set[int] = set()
    seen_content: Set[str] = set()  # même texte sous deux sources (fichier du dépôt et document téléversé)
    per_source: Dict[Tuple[str, Any], int] = {}
    out: List[Any] = []
    for _, h in indexed:
        text = hit_text(h)
        if not text.strip():
            continue
        chunk_id = _as_int(_g(h, "chunk_id"))
        if chunk_id is not None:
            if chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
        ident = hit_identity(h)
        if ident in seen:
            continue
        seen.add(ident)
        content_key = hit_content_key(h)
        if content_key is not None:
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
        where = str(_g(h, "path", "") or _g(h, "title", "") or "")
        span = (where, ident[2], ident[3], text)
        if where and ident[2] is not None and span in seen_spans:
            continue  # plusieurs routes déclarées sur la même ligne : un seul passage
        seen_spans.add(span)
        source_key = (ident[0], ident[1])
        if ident[1] is not None and max_per_file > 0:
            count = per_source.get(source_key, 0)
            if count >= max_per_file:
                continue
            per_source[source_key] = count + 1
        out.append(h)
    return out


def graph_file_preview(hit: Any) -> Optional[str]:
    """Corps réduit d'un passage de fichier atteint par le canal graphe seul : lignes d'en-tête (chunk, ``<?php``,
    namespace, use, import, require…) remplacées par :data:`IMPORTS_OMITTED_MARKER`, puis au plus
    :data:`GRAPH_FILE_PREVIEW_LINES` lignes non vides, le reste compté par :data:`TRUNCATED_MARKER`. ``None`` quand
    le passage est montré entier : autre type de source, autre canal (son contenu a répondu), ou rien à réduire."""
    if str(_g(hit, "source_type", "file") or "file") != "file" or set(hit_channels(hit)) != {"graph"}:
        return None
    omitted = 0
    kept: List[str] = []
    for line in hit_text(hit).splitlines():
        if _HEADER_LINE_RE.match(line):
            omitted += 1
        elif line.strip():
            kept.append(line)
        elif kept and kept[-1].strip():
            kept.append(line)  # une ligne vide d'affilée au plus, jamais en tête
    while kept and not kept[-1].strip():
        kept.pop()
    shown: List[str] = []
    non_blank = 0
    for line in kept:
        if line.strip():
            if non_blank >= GRAPH_FILE_PREVIEW_LINES:
                break
            non_blank += 1
        shown.append(line)
    while shown and not shown[-1].strip():
        shown.pop()
    hidden = len(kept) - len(shown)
    if not omitted and not hidden:
        return None
    parts = [IMPORTS_OMITTED_MARKER % omitted] if omitted else []
    parts.extend(shown)
    if hidden:
        parts.append(TRUNCATED_MARKER % hidden)
    return "\n".join(parts)


def _representatives(candidates: Sequence[Any]) -> List[int]:
    """Indice du meilleur hit de chaque canal ayant répondu, par ordre de priorité des canaux."""
    picked: List[int] = []
    for channel in CHANNEL_PRIORITY:
        for i, h in enumerate(candidates):
            if channel in hit_channels(h):
                if i not in picked:
                    picked.append(i)
                break
    return picked


def build_context(hits: Sequence[Any], budget_tokens: int, intent: str = "factual",
                  max_per_file: int = 3, db: Any = None, cards: Optional[Sequence[Card]] = None,
                  max_cards: int = MAX_CARDS) -> ContextPack:
    """Assemble le contexte dans ``budget_tokens`` tokens.

    - fiches structurées en tête (section « Fiches ») : construites par :func:`build_cards` quand ``db``
      est fourni (ou données via ``cards``), entières ou pas du tout, numérotées [S1]… en premier ; les
      passages bruts qu'une fiche couvre (même modèle / migrations / symbole de classe, fichier du modèle)
      ne sont plus répétés ;
    - ordre de score conservé pour la numérotation des passages qui suivent ;
    - au moins un passage par canal ayant répondu (identifier > graph > lexical > semantic > temporal) ;
    - un fichier atteint par le graphe seul est réduit à un aperçu (:func:`graph_file_preview`) ;
    - les passages des fichiers générés volumineux (``api-docs.json``, dumps, ``*.lock``…) sont écartés, sauf
      fichier cité par la question ou absence de tout autre passage (:func:`_without_generated`) ;
    - remplissage glouton, le dernier passage retenu est tronqué entre deux lignes si nécessaire ;
    - les passages écartés sont listés dans ``truncated`` (sans référence).
    """
    budget = max(0, int(budget_tokens or 0))
    if cards is None and db is not None:
        try:
            cards = build_cards(db, hits or [], max_cards=max_cards)
        except Exception as exc:  # noqa: BLE001 - jamais de contexte perdu à cause d'une fiche
            logger.warning("fiches structurées indisponibles: %s", exc)
            cards = []
    kept_cards: List[Card] = []
    covered_keys: Set[Tuple[str, Any]] = set()
    covered_files: Set[int] = set()
    used = 0
    if cards:
        used = estimate_tokens(CARDS_HEADER) + estimate_tokens(PASSAGES_HEADER)
        cards_cap = int(budget * CARDS_BUDGET_SHARE)  # les passages gardent toujours une part du budget
        for card in cards:
            cost = _block_cost(format_card_block("S%d" % (len(kept_cards) + 1), card), card.hit)
            if used + cost > budget or (kept_cards and used + cost > cards_cap):
                continue
            kept_cards.append(card)
            used += cost
            covered_keys |= set(card.covers)
            covered_files |= set(card.file_ids)
        if not kept_cards:
            used = 0
    remaining = [h for h in (hits or []) if h is not None and not _card_covers_hit(h, covered_keys, covered_files)]
    remaining, generated = _without_generated(remaining)
    candidates = _dedupe_and_cap(remaining, max_per_file)
    previews: Dict[int, str] = {}  # indice -> aperçu d'un fichier atteint par le graphe seul
    for i, h in enumerate(candidates):
        preview = graph_file_preview(h)
        if preview is not None:
            previews[i] = preview
    bodies: Dict[int, Optional[str]] = {}  # indice -> bloc tronqué ou aperçu (None = complet)
    costs: Dict[int, int] = {}
    excluded: List[int] = []

    def try_include(i: int) -> str:
        """Tente d'inclure le candidat ``i`` ; renvoie 'full', 'partial' ou 'excluded'."""
        nonlocal used
        h = candidates[i]
        preview = previews.get(i)
        block = format_block("S%d" % (i + 1), h, preview)
        cost = _block_cost(block, h)
        if used + cost <= budget:
            bodies[i] = block if preview is not None else None
            costs[i] = cost
            used += cost
            return "full"
        partial = truncate_block("S%d" % (i + 1), h, budget - used) if preview is None else None
        if partial is not None:
            bodies[i] = partial
            cost = _block_cost(partial, h)
            costs[i] = cost
            used += cost
            return "partial"
        excluded.append(i)
        return "excluded"

    # Phase A : un représentant par canal (priorité aux canaux les plus précis).
    for i in _representatives(candidates):
        try_include(i)

    # Phase B : remplissage glouton par score ; le budget est atteint dès qu'un passage ne tient
    # pas en entier (il est alors tronqué s'il reste de la place, puis tout le reste est écarté).
    stopped = False
    for i in range(len(candidates)):
        if i in bodies or i in excluded:
            continue
        if stopped:
            excluded.append(i)
            continue
        if try_include(i) != "full":
            stopped = True

    pack = ContextPack()
    card_blocks: List[str] = []
    for n, card in enumerate(kept_cards):
        ref = "S%d" % (n + 1)
        block = format_card_block(ref, card)
        card.ref = ref
        card.source = Source.from_hit(ref, card.hit)
        pack.sources.append(card.source)
        pack.hits_by_ref[ref] = card.hit
        pack.blocks[ref] = block
        pack.cards.append(card)
        card_blocks.append(block)
    offset = len(kept_cards)
    blocks: List[str] = []
    for n, i in enumerate(sorted(bodies.keys())):
        ref = "S%d" % (offset + n + 1)
        h = candidates[i]
        body = bodies[i]
        if body is None:
            block = format_block(ref, h)
        else:
            # le bloc tronqué ou l'aperçu a été formaté avec un identifiant provisoire : on le refait avec le bon
            block = _reformat_truncated(ref, h, body)
        src = Source.from_hit(ref, h)
        pack.sources.append(src)
        pack.hits_by_ref[ref] = h
        pack.blocks[ref] = block
        blocks.append(block)
    if card_blocks:
        parts = [CARDS_HEADER, "\n\n".join(card_blocks)]
        if blocks:
            parts += [PASSAGES_HEADER, "\n\n".join(blocks)]
        pack.text = "\n\n".join(parts)
    else:
        pack.text = "\n\n".join(blocks)
    pack.tokens_used = sum(_block_cost(pack.blocks[s.ref], pack.hits_by_ref[s.ref]) for s in pack.sources)
    if card_blocks:
        pack.tokens_used += estimate_tokens(CARDS_HEADER) + estimate_tokens(PASSAGES_HEADER)
    pack.truncated = [Source.from_hit("", candidates[i]) for i in sorted(set(excluded))]
    pack.truncated += [Source.from_hit("", h) for h in _dedupe_and_cap(generated, max_per_file)]
    return pack


def _reformat_truncated(ref: str, hit: Any, block: str) -> str:
    """Remplace l'en-tête provisoire d'un bloc tronqué par l'en-tête définitif."""
    first_newline = block.find("\n")
    if first_newline < 0:
        return block
    return format_header(ref, hit) + block[first_newline:]


def add_tool_sources(pack: ContextPack, hits: Sequence[Any], budget_tokens: int) -> List[Source]:
    """Ajoute au pack les sources issues d'un outil, dans la limite de ``budget_tokens`` tokens.

    Les hits déjà présents (même source et lignes identiques ou incluses, ou même texte déjà cité sous
    une autre source : :func:`hit_content_key`) sont ignorés ; les nouveaux reçoivent les références
    suivantes (S<n+1>…). Renvoie les sources ajoutées.
    """
    budget = max(0, int(budget_tokens or 0))
    added: List[Source] = []
    if not hits or budget <= 0:
        return added
    existing: List[Tuple[str, Any, Optional[int], Optional[int]]] = [hit_identity(h) for h in pack.hits_by_ref.values()]
    existing_content: Set[str] = set()
    for h in pack.hits_by_ref.values():
        key = hit_content_key(h)
        if key is not None:
            existing_content.add(key)
    card_keys: Set[Tuple[str, Any]] = set()
    for card in pack.cards:
        card_keys |= set(card.covers)  # un modèle / symbole déjà résumé par une fiche n'est pas réajouté (ses fichiers, si)
    used = 0
    for h in hits:
        if h is None or not hit_text(h).strip():
            continue
        ident = hit_identity(h)
        if _covered(ident, existing) or _card_covers_hit(h, card_keys, set()):
            continue
        content_key = hit_content_key(h)
        if content_key is not None and content_key in existing_content:
            continue
        ref = "S%d" % (len(pack.sources) + 1)
        block = format_block(ref, h)
        cost = _block_cost(block, h)
        if used + cost > budget:
            block_partial = truncate_block(ref, h, budget - used)
            if block_partial is None:
                continue
            block = block_partial
            cost = _block_cost(block, h)
        src = Source.from_hit(ref, h)
        pack.sources.append(src)
        pack.hits_by_ref[ref] = h
        pack.blocks[ref] = block
        pack.text = (pack.text + "\n\n" + block) if pack.text else block
        pack.tokens_used += cost
        used += cost
        existing.append(ident)
        if content_key is not None:
            existing_content.add(content_key)
        added.append(src)
    return added


def _covered(ident: Tuple[str, Any, Optional[int], Optional[int]],
             existing: Sequence[Tuple[str, Any, Optional[int], Optional[int]]]) -> bool:
    source_type, source_id, start, end = ident
    for e_type, e_id, e_start, e_end in existing:
        if e_type != source_type or e_id != source_id or source_id is None:
            continue
        if e_start == start and e_end == end:
            return True
        if start is None or end is None or e_start is None or e_end is None:
            continue
        if e_start <= start and end <= e_end:
            return True  # la plage existante contient déjà la nouvelle
    return False


def render_sources_for_model(pack: ContextPack, refs: Optional[Sequence[str]] = None) -> str:
    """Texte des sources tel qu'il est donné au modèle (toutes, ou seulement ``refs``)."""
    if refs is None:
        if not pack.sources:
            return "Aucune source n'a été trouvée dans le projet pour cette question."
        return "Sources disponibles (cite-les avec [Sn]) :\n\n" + pack.text
    blocks = [pack.blocks[r] for r in refs if r in pack.blocks]
    if not blocks:
        return ""
    return "Nouvelles sources citables :\n\n" + "\n\n".join(blocks)


def render_sources_as_text(pack: ContextPack) -> str:
    """Liste lisible des sources (réponse du mode « recherche seule »)."""
    if not pack.sources:
        return "Aucune source trouvée dans le projet pour cette question."
    lines = ["Sources trouvées (%d) :" % len(pack.sources)]
    for s in pack.sources:
        where = s.path or s.title or "?"
        if s.source_type in ("document",):
            where = "Document « %s »" % (s.title or s.path or "?")
            if s.page is not None:
                where += " (page %d)" % s.page
        elif s.source_type == "note":
            where = "Note « %s »" % (s.title or "?")
        elif s.line_start is not None:
            where += ":L%d" % s.line_start
            if s.line_end is not None and s.line_end != s.line_start:
                where += "-%d" % s.line_end
        line = "- [%s] %s" % (s.ref, where)
        if s.reason:
            line += " — " + s.reason
        lines.append(line)
    if pack.truncated:
        lines.append("Non retenues faute de budget : %d." % len(pack.truncated))
    return "\n".join(lines)
