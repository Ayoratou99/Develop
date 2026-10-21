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

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from develop.chat.budget import estimate_tokens

CHANNEL_PRIORITY = ("identifier", "graph", "lexical", "semantic", "temporal")
CODE_SOURCE_TYPES = ("file", "symbol", "endpoint", "model")
MIN_PARTIAL_LINES = 3
MIN_PARTIAL_TOKENS = 40
TRUNCATED_MARKER = "… (tronqué : %d lignes non affichées)"

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
class ContextPack:
    """Contexte assemblé pour une question : sources citables, texte formaté et budget consommé."""

    sources: List[Source] = field(default_factory=list)
    text: str = ""
    tokens_used: int = 0
    truncated: List[Source] = field(default_factory=list)  # sources écartées faute de budget
    hits_by_ref: Dict[str, Any] = field(default_factory=dict)
    blocks: Dict[str, str] = field(default_factory=dict)  # ref -> bloc formaté (pour les outils)


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
    return str(_g(hit, "source_type", "file") or "file") in CODE_SOURCE_TYPES


def hit_identity(hit: Any) -> Tuple[str, Any, Optional[int], Optional[int]]:
    return (str(_g(hit, "source_type", "file")), _g(hit, "source_id"), _as_int(_g(hit, "line_start")),
            _as_int(_g(hit, "line_end")))


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
    prefix = _repo_prefix(hit)
    where = "%s/%s" % (prefix, path) if prefix and not path.startswith(prefix + "/") else path
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


# --------------------------------------------------------------------------- sélection

def _dedupe_and_cap(hits: Sequence[Any], max_per_file: int) -> List[Any]:
    """Trie par score décroissant (stable), supprime les doublons et limite le nombre par fichier."""
    indexed = [(i, h) for i, h in enumerate(hits) if h is not None]

    def key(item: Tuple[int, Any]) -> Tuple[float, int]:
        try:
            score = float(_g(item[1], "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        return (-score, item[0])

    indexed.sort(key=key)
    seen: Set[Tuple[str, Any, Optional[int], Optional[int]]] = set()
    seen_chunks: Set[int] = set()
    per_source: Dict[Tuple[str, Any], int] = {}
    out: List[Any] = []
    for _, h in indexed:
        if not hit_text(h).strip():
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
        source_key = (ident[0], ident[1])
        if ident[1] is not None and max_per_file > 0:
            count = per_source.get(source_key, 0)
            if count >= max_per_file:
                continue
            per_source[source_key] = count + 1
        out.append(h)
    return out


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
                  max_per_file: int = 3) -> ContextPack:
    """Assemble le contexte dans ``budget_tokens`` tokens.

    - ordre de score conservé pour la numérotation [S1], [S2]… ;
    - au moins un passage par canal ayant répondu (identifier > graph > lexical > semantic > temporal) ;
    - remplissage glouton, le dernier passage retenu est tronqué entre deux lignes si nécessaire ;
    - les passages écartés sont listés dans ``truncated`` (sans référence).
    """
    budget = max(0, int(budget_tokens or 0))
    candidates = _dedupe_and_cap(hits or [], max_per_file)
    bodies: Dict[int, Optional[str]] = {}  # indice -> corps tronqué (None = complet)
    costs: Dict[int, int] = {}
    excluded: List[int] = []
    used = 0

    def try_include(i: int) -> str:
        """Tente d'inclure le candidat ``i`` ; renvoie 'full', 'partial' ou 'excluded'."""
        nonlocal used
        h = candidates[i]
        block = format_block("S%d" % (i + 1), h)
        cost = _block_cost(block, h)
        if used + cost <= budget:
            bodies[i] = None
            costs[i] = cost
            used += cost
            return "full"
        partial = truncate_block("S%d" % (i + 1), h, budget - used)
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
    blocks: List[str] = []
    for n, i in enumerate(sorted(bodies.keys())):
        ref = "S%d" % (n + 1)
        h = candidates[i]
        body = bodies[i]
        if body is None:
            block = format_block(ref, h)
        else:
            # le bloc tronqué a été formaté avec un identifiant provisoire : on le refait avec le bon
            block = _reformat_truncated(ref, h, body)
        src = Source.from_hit(ref, h)
        pack.sources.append(src)
        pack.hits_by_ref[ref] = h
        pack.blocks[ref] = block
        blocks.append(block)
    pack.text = "\n\n".join(blocks)
    pack.tokens_used = sum(_block_cost(pack.blocks[s.ref], pack.hits_by_ref[s.ref]) for s in pack.sources)
    pack.truncated = [Source.from_hit("", candidates[i]) for i in sorted(set(excluded))]
    return pack


def _reformat_truncated(ref: str, hit: Any, block: str) -> str:
    """Remplace l'en-tête provisoire d'un bloc tronqué par l'en-tête définitif."""
    first_newline = block.find("\n")
    if first_newline < 0:
        return block
    return format_header(ref, hit) + block[first_newline:]


def add_tool_sources(pack: ContextPack, hits: Sequence[Any], budget_tokens: int) -> List[Source]:
    """Ajoute au pack les sources issues d'un outil, dans la limite de ``budget_tokens`` tokens.

    Les hits déjà présents (même source et lignes identiques ou incluses) sont ignorés ; les
    nouveaux reçoivent les références suivantes (S<n+1>…). Renvoie les sources ajoutées.
    """
    budget = max(0, int(budget_tokens or 0))
    added: List[Source] = []
    if not hits or budget <= 0:
        return added
    existing: List[Tuple[str, Any, Optional[int], Optional[int]]] = [hit_identity(h) for h in pack.hits_by_ref.values()]
    used = 0
    for h in hits:
        if h is None or not hit_text(h).strip():
            continue
        ident = hit_identity(h)
        if _covered(ident, existing):
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
