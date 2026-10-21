"""Découpage de textes en morceaux (*chunks*) pour l'indexation et la recherche.

Quatre modes, choisis par :func:`chunk_for_file` ou passés à :func:`chunk_text` :

* ``code`` : frontières de définitions (fonctions, classes, routes…) puis lignes vides ;
* ``markdown`` : sections par titres (``#``…``######``) puis paragraphes, tableaux jamais coupés ;
* ``prose`` : paragraphes regroupés avec chevauchement de phrases ;
* ``data`` : fenêtres de lignes avec chevauchement et rappel de l'en-tête.

Invariants : aucun morceau vide, ordinaux séquentiels, ``line_start``/``line_end`` 1-based
inclusifs exacts par rapport à ``text.splitlines()``, ``max_chars`` respecté sauf ligne unique
plus longue (et tableaux Markdown de moins de ``2 * max_chars``).
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

CHUNK_KINDS = ("code", "markdown", "prose", "data")
DATA_LANGUAGES = {"sql", "csv", "tsv", "json", "yaml", "yml", "xml", "toml", "ini", "env", "dotenv", "properties"}
DATA_WINDOW_LINES = 40  # taille de fenêtre (en lignes) pour le mode ``data``


@dataclass
class Chunk:
    """Morceau de texte avec sa position (lignes 1-based inclusives) et ses métadonnées."""

    text: str
    ordinal: int
    line_start: int
    line_end: int
    meta: Dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Estimation grossière du nombre de jetons (≈ 4 caractères par jeton)."""
    if not text:
        return 0
    return (len(text) + 3) // 4


# --------------------------------------------------------------------------- représentation interne

@dataclass
class _Span:
    """Bloc atomique : lignes ``[a, b)`` (0-based) et métadonnées."""

    a: int
    b: int
    meta: Dict[str, Any]
    keep_whole_below: int = 0  # taille maximale (chars) en deçà de laquelle le bloc n'est jamais coupé
    split: str = "lines"  # "lines" ou "sentences" pour un bloc trop long
    mergeable: bool = True
    head: int = 0  # lignes de tête (titres Markdown) collées au premier morceau si le bloc est coupé
    sub_split: bool = True  # False : ne pas re-découper sur les lignes vides (bloc de code clôturé)


@dataclass
class _Piece:
    """Portion de texte : de ``(sl, sc)`` à ``(el, ec)`` exclusif, en indices de lignes 0-based."""

    sl: int
    sc: int
    el: int
    ec: int
    meta: Dict[str, Any]


def _span_len(lines: List[str], a: int, b: int) -> int:
    if b <= a:
        return 0
    return sum(len(l) for l in lines[a:b]) + (b - a - 1)


def _piece_text(lines: List[str], p: _Piece) -> str:
    if p.sl == p.el:
        return lines[p.sl][p.sc:p.ec]
    parts = [lines[p.sl][p.sc:]]
    parts.extend(lines[p.sl + 1:p.el])
    parts.append(lines[p.el][:p.ec])
    return "\n".join(parts)


def _whole(a: int, b: int, lines: List[str], meta: Dict[str, Any]) -> _Piece:
    return _Piece(a, 0, b - 1, len(lines[b - 1]), dict(meta))


# --------------------------------------------------------------------------- fenêtres de lignes

def _line_windows(lines: List[str], a: int, b: int, max_chars: int, overlap_chars: int,
                  max_lines: Optional[int] = None) -> List[Tuple[int, int]]:
    """Fenêtres ``[i, j)`` couvrant ``[a, b)`` : ≤ ``max_chars`` (sauf ligne unique), chevauchement."""
    def window_end(i: int) -> int:
        j = i
        total = 0
        while j < b:
            add = len(lines[j]) + (1 if j > i else 0)
            if j > i and (total + add > max_chars or (max_lines and j - i >= max_lines)):
                break
            total += add
            j += 1
        return j

    out: List[Tuple[int, int]] = []
    i = a
    while i < b:
        j = window_end(i)
        out.append((i, j))
        if j >= b:
            break
        k = j
        back = 0
        while k - 1 > i and back + len(lines[k - 1]) + 1 <= overlap_chars:
            k -= 1
            back += len(lines[k]) + 1
        if k < j and window_end(k) <= j:
            k = j  # le chevauchement ne ferait pas progresser (ligne suivante trop longue)
        i = max(k, i + 1)
    return out


# --------------------------------------------------------------------------- phrases

# Fin de phrase : ponctuation forte (+ guillemet/parenthèse fermante) suivie d'un blanc ;
# « 1. » ou « a. » en tête de liste ne compte pas.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])(?<!\b\d\.)(?<!\b[a-zA-Z]\.)[\"'»)\]]*\s+")


def _sentence_bounds(text: str) -> List[Tuple[int, int]]:
    """Intervalles ``[début, fin)`` des phrases (fin incluant l'espace qui suit)."""
    bounds: List[Tuple[int, int]] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(text):
        end = m.end()
        if end > start and text[start:end].strip():
            bounds.append((start, end))
            start = end
    if start < len(text) and text[start:].strip():
        bounds.append((start, len(text)))
    return bounds


_WS_RE = re.compile(r"\s")


def _hard_split(text: str, start: int, end: int, max_chars: int) -> List[Tuple[int, int]]:
    """Coupe ``text[start:end]`` sur des blancs en morceaux ≤ max_chars (un mot plus long reste entier)."""
    out: List[Tuple[int, int]] = []
    i = start
    while i < end:
        if end - i <= max_chars:
            out.append((i, end))
            break
        window = text[i + 1:i + max_chars + 1]
        positions = [m.start() for m in _WS_RE.finditer(window)]
        if positions:
            cut = i + 1 + positions[-1]
        else:
            m = _WS_RE.search(text, i + 1, end)
            cut = m.start() if m else end
        out.append((i, cut))
        i = cut
        while i < end and text[i].isspace():
            i += 1
    return out


def _sentence_windows(text: str, max_chars: int, overlap_chars: int) -> List[Tuple[int, int]]:
    """Groupes de phrases ≤ max_chars avec chevauchement de phrases (positions dans ``text``)."""
    sentences = _sentence_bounds(text)
    units: List[Tuple[int, int]] = []
    for s, e in sentences:
        if e - s > max_chars:
            units.extend(_hard_split(text, s, e, max_chars))
        else:
            units.append((s, e))
    if not units:
        return []
    budget = max(1, max_chars - overlap_chars) if overlap_chars < max_chars // 2 else max_chars
    out: List[Tuple[int, int]] = []
    i = 0
    n = len(units)
    while i < n:
        j = i
        while j < n and (j == i or units[j][1] - units[i][0] <= budget):
            j += 1
        win = (units[i][0], units[j - 1][1])
        if not out or not (win[0] >= out[-1][0] and win[1] <= out[-1][1]):
            out.append(win)  # (une fenêtre entièrement incluse dans la précédente est inutile)
        if j >= n:
            break
        k = j
        while k - 1 > i and units[j - 1][1] - units[k - 1][0] <= overlap_chars:
            k -= 1
        i = max(k, i + 1)
    return out


def _line_starts(lines: List[str], a: int, b: int) -> List[int]:
    """Positions de début de chaque ligne de ``lines[a:b]`` dans ``"\\n".join(lines[a:b])``."""
    starts = [0]
    for line in lines[a:b - 1]:
        starts.append(starts[-1] + len(line) + 1)
    return starts


def _locate(starts: List[int], a: int, offset: int) -> Tuple[int, int]:
    """Convertit une position dans ``"\\n".join(lines[a:b])`` en ``(index de ligne, colonne)``."""
    k = bisect_right(starts, offset) - 1
    return a + k, offset - starts[k]


def _split_sentences_piece(lines: List[str], a: int, b: int, meta: Dict[str, Any],
                           max_chars: int, overlap_chars: int) -> List[_Piece]:
    text = "\n".join(lines[a:b])
    starts = _line_starts(lines, a, b)
    pieces: List[_Piece] = []
    for s, e in _sentence_windows(text, max_chars, overlap_chars):
        while e > s and text[e - 1].isspace():
            e -= 1
        while s < e and text[s].isspace():
            s += 1
        if e <= s:
            continue
        sl, sc = _locate(starts, a, s)
        el, ec = _locate(starts, a, e)
        if ec == 0 and el > sl:
            el -= 1
            ec = len(lines[el])
        pieces.append(_Piece(sl, sc, el, ec, dict(meta)))
    return pieces


# --------------------------------------------------------------------------- blocs et empaquetage

def _paragraph_blocks(lines: List[str], a: int, b: int) -> List[Tuple[int, int]]:
    """Blocs de lignes non vides consécutives dans ``[a, b)``."""
    out: List[Tuple[int, int]] = []
    i = a
    while i < b:
        if not lines[i].strip():
            i += 1
            continue
        j = i
        while j < b and lines[j].strip():
            j += 1
        out.append((i, j))
        i = j
    return out


def _split_span(lines: List[str], sp: _Span, max_chars: int, overlap_chars: int) -> List[_Piece]:
    """Découpe un bloc trop long : sous-blocs (lignes vides) puis fenêtres de lignes ou phrases."""
    sub = _paragraph_blocks(lines, sp.a + sp.head, sp.b) if sp.sub_split else []
    if len(sub) > 1:
        subspans = [_Span(x, y, sp.meta, split=sp.split) for x, y in sub]
        subspans[0].a = sp.a  # les lignes de tête restent collées au premier sous-bloc
        subspans[0].head = sp.head
        return _pack(lines, subspans, max_chars, overlap_chars)
    pieces: List[_Piece] = []
    if sp.split == "sentences":
        pieces = _split_sentences_piece(lines, sp.a, sp.b, sp.meta, max_chars, overlap_chars)
        if pieces:
            return pieces
    for i, j in _line_windows(lines, sp.a, sp.b, max_chars, overlap_chars):
        pieces.append(_whole(i, j, lines, sp.meta))
    return pieces


def _merge_meta(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    for k, v in src.items():
        if k not in dst or dst[k] in (None, "", [], False):
            dst[k] = v


def _pack(lines: List[str], spans: List[_Span], max_chars: int, overlap_chars: int) -> List[_Piece]:
    """Regroupe des blocs adjacents jusqu'à ``max_chars`` ; coupe ceux qui dépassent."""
    out: List[_Piece] = []
    cur: Optional[_Span] = None

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            out.append(_whole(cur.a, cur.b, lines, cur.meta))
            cur = None

    for sp in spans:
        if sp.b <= sp.a:
            continue
        length = _span_len(lines, sp.a, sp.b)
        if length > max_chars:
            flush()
            single_long_line = sp.b - sp.a == 1 and sp.split != "sentences"
            if length < sp.keep_whole_below or single_long_line:
                out.append(_whole(sp.a, sp.b, lines, sp.meta))
            else:
                out.extend(_split_span(lines, sp, max_chars, overlap_chars))
            continue
        if cur is None:
            cur = _Span(sp.a, sp.b, dict(sp.meta), mergeable=sp.mergeable)
            continue
        if cur.mergeable and sp.mergeable and _span_len(lines, cur.a, sp.b) <= max_chars:
            cur.b = sp.b
            _merge_meta(cur.meta, sp.meta)
        else:
            flush()
            cur = _Span(sp.a, sp.b, dict(sp.meta), mergeable=sp.mergeable)
    flush()
    return out


# --------------------------------------------------------------------------- mode code

_IDENT = r"[A-Za-z_$][\w$]*"
_CODE_DEF_PATTERNS = [re.compile(p, re.X) for p in (
    # Python / Ruby / Nim : def, async def, class
    r"^\s*(?:async\s+)?(?:def|class)\s+(?P<name>" + _IDENT + ")",
    # JS/TS : export (default)? (async)? function|class|interface|enum|type|abstract class
    r"^\s*(?:export\s+(?:default\s+)?)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?"
    r"(?:function\s*\*?|class|interface|enum|type|namespace|module)\s+(?P<name>" + _IDENT + ")",
    # JS/TS : (export)? const|let|var X = (async)? (...) => | function
    r"^\s*(?:export\s+(?:default\s+)?)?(?:const|let|var)\s+(?P<name>" + _IDENT + r")\s*(?::[^=]+)?=\s*"
    r"(?:async\s*)?(?:\([^)]*\)\s*(?::\s*[^=]+?)?\s*=>|" + _IDENT + r"\s*=>|function\b|React\.memo|forwardRef|memo\()",
    # PHP / Java / C# / Kotlin : (modifiers) function|fun name  -- et classes/traits
    r"^\s*(?:(?:public|private|protected|static|abstract|final|async|override|open|suspend)\s+)*"
    r"(?:function|fun)\s+&?(?P<name>" + _IDENT + ")",
    r"^\s*(?:(?:abstract|final|readonly|public|private|protected|internal|sealed|static|data|open|partial)\s+)*"
    r"(?:class|interface|trait|enum|struct|record|object|protocol|extension)\s+(?P<name>" + _IDENT + ")",
    # Rust / Go / Swift / Zig
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:fn|func)\s+(?:\([^)]*\)\s*)?(?P<name>" + _IDENT + ")",
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|impl(?:<[^>]*>)?|mod|type)\s+(?P<name>" + _IDENT + ")",
    # Java / C# / C++ : public static Type name(
    r"^\s*(?:public|private|protected)\s+(?:(?:static|final|abstract|virtual|override|async|synchronized)\s+)*"
    r"[\w<>\[\],.?]+\s+(?P<name>" + _IDENT + r")\s*\(",
    # Laravel : Route::get(...), $router->get(...)
    r"^\s*(?P<name>Route::\w+)\s*\(",
    r"^\s*\$?(?:router|app)->(?P<name>get|post|put|patch|delete|group|resource|apiResource)\s*\(",
    # Express / Flask / FastAPI style : app.get('/x', …), router.post(…)
    r"^\s*(?P<name>(?:app|router|server|api|fastify)\.(?:get|post|put|patch|delete|all|use|route|ws))\s*\(",
    # SQL : CREATE TABLE/VIEW/FUNCTION/PROCEDURE name
    r"^\s*create\s+(?:or\s+replace\s+)?(?:temp(?:orary)?\s+)?(?:table|view|function|procedure|trigger|index)\s+"
    r"(?:if\s+not\s+exists\s+)?[`\"\[]?(?P<name>[\w.]+)",
    # Shell : function name  |  name() {
    r"^\s*function\s+(?P<name>[A-Za-z_][\w-]*)\s*(?:\(\s*\))?\s*\{?\s*$",
    r"^\s*(?P<name>[A-Za-z_][\w-]*)\s*\(\s*\)\s*\{\s*$",
    # Méthodes JS/TS/Java indentées : name(params) [: type] {   (mots-clés de contrôle exclus)
    r"^\s+(?:async\s+|static\s+|private\s+|public\s+|protected\s+)*"
    r"(?!(?:if|for|while|switch|catch|return|else|function|new|do|try|with|constructor)\b)"
    r"(?P<name>" + _IDENT + r")\s*\([^()]*\)\s*(?::\s*[^{;]+)?\s*\{\s*$",
)]
_CODE_DECORATOR_RE = re.compile(r"^\s*(?:@(?P<name>[\w.]+)|#\[(?P<attr>\w+)|\[(?P<csattr>[A-Z]\w*)[(\]])")
_CODE_COMMENT_RE = re.compile(r"^\s*(?://|#|\*|/\*|\*/|'''|\"\"\"|--|<!--|///|\{-|rem\b)")
_ATTACH_BACK_MAX = 40


def _match_definition(line: str) -> Optional[str]:
    for pat in _CODE_DEF_PATTERNS:
        m = pat.match(line)
        if m:
            return m.group("name")
    return None


def _code_spans(lines: List[str], language: Optional[str]) -> List[_Span]:
    n = len(lines)
    defs: Dict[int, str] = {}
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        name = _match_definition(line)
        if name:
            defs[i] = name
    base_meta: Dict[str, Any] = {"kind": "code"}
    if language:
        base_meta["language"] = language
    if not defs:
        return [_Span(a, b, dict(base_meta)) for a, b in _paragraph_blocks(lines, 0, n)]

    # Chaque définition absorbe les décorateurs / commentaires qui la précèdent immédiatement.
    starts: Dict[int, str] = {}
    absorbed = set()
    for i in sorted(defs):
        s = i
        j = i - 1
        while j >= 0 and i - j <= _ATTACH_BACK_MAX and j not in defs and j not in absorbed:
            ln = lines[j]
            if not ln.strip():
                break
            if _CODE_DECORATOR_RE.match(ln) or _CODE_COMMENT_RE.match(ln):
                s = j
                j -= 1
                continue
            break
        for k in range(s, i + 1):
            absorbed.add(k)
        starts[s] = defs[i]
    # Décorateurs orphelins (ex: annotations de routes sans définition reconnue juste après)
    for i, line in enumerate(lines):
        if i in absorbed or i in starts:
            continue
        m = _CODE_DECORATOR_RE.match(line)
        if m and (m.group("name") or m.group("attr") or m.group("csattr")):
            starts[i] = m.group("name") or m.group("attr") or m.group("csattr")

    boundaries = sorted(starts)
    spans: List[_Span] = []
    first = boundaries[0]
    for a, b in _paragraph_blocks(lines, 0, first):
        spans.append(_Span(a, b, dict(base_meta)))
    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else n
        while end > start and not lines[end - 1].strip():
            end -= 1
        if end <= start:
            continue
        meta = dict(base_meta)
        meta["symbol_hint"] = starts[start]
        spans.append(_Span(start, end, meta))
    return spans


# --------------------------------------------------------------------------- mode markdown

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?\s*$")


def _markdown_structure(lines: List[str]) -> Tuple[List[bool], List[Tuple[int, int, str]]]:
    """Renvoie ``(ligne_dans_code[], [(index, niveau, titre)])`` en suivant les clôtures ``` / ~~~."""
    in_code = [False] * len(lines)
    headings: List[Tuple[int, int, str]] = []
    fence: Optional[str] = None
    for i, line in enumerate(lines):
        m = _FENCE_RE.match(line)
        if m:
            marker = m.group(1)
            if fence is None:
                fence = marker
                in_code[i] = True
                continue
            if marker == fence:
                fence = None
                in_code[i] = True
                continue
        in_code[i] = fence is not None
        if fence is None:
            h = _HEADING_RE.match(line)
            if h:
                headings.append((i, len(h.group(1)), h.group(2).strip()))
    return in_code, headings


def _is_table_block(lines: List[str], a: int, b: int) -> bool:
    return b - a >= 2 and "|" in lines[a] and "|" in lines[a + 1] and bool(_TABLE_SEP_RE.match(lines[a + 1]))


def _markdown_blocks(lines: List[str], a: int, b: int, in_code: List[bool], meta: Dict[str, Any],
                     max_chars: int) -> List[_Span]:
    """Paragraphes d'une section : blocs de code clôturés et tableaux restent atomiques."""
    spans: List[_Span] = []
    i = a
    while i < b:
        if not lines[i].strip():
            i += 1
            continue
        if in_code[i]:
            j = i
            while j < b and in_code[j]:
                j += 1
            m = dict(meta)
            m["has_code_block"] = True
            spans.append(_Span(i, j, m, sub_split=False))
            i = j
            continue
        j = i
        while j < b and lines[j].strip() and not in_code[j]:
            j += 1
        m = dict(meta)
        if _is_table_block(lines, i, j):
            m["has_table"] = True
            m["table_header"] = lines[i].strip()
            spans.append(_Span(i, j, m, keep_whole_below=max_chars * 2))
        else:
            spans.append(_Span(i, j, m, split="sentences"))
        i = j
    return spans


def _markdown_pieces(lines: List[str], max_chars: int, overlap_chars: int, language: Optional[str]) -> List[_Piece]:
    n = len(lines)
    in_code, headings = _markdown_structure(lines)
    base: Dict[str, Any] = {"kind": "markdown"}
    if language:
        base["language"] = language

    # Sections : (début, fin, heading_path) ; une section dont le corps est vide est fusionnée avec la suivante.
    sections: List[Tuple[int, int, List[str]]] = []
    stack: List[Tuple[int, str]] = []
    first_heading = headings[0][0] if headings else n
    if first_heading > 0:
        sections.append((0, first_heading, []))
    for k, (idx, level, title) in enumerate(headings):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[k + 1][0] if k + 1 < len(headings) else n
        sections.append((idx, end, [t for _, t in stack]))

    pieces: List[_Piece] = []
    carry_start: Optional[int] = None  # titres sans corps reportés sur la section suivante
    for si, (start, end, path) in enumerate(sections):
        is_heading_section = start >= first_heading
        body_blank = is_heading_section and all(not lines[i].strip() for i in range(start + 1, end))
        if body_blank and si + 1 < len(sections):
            if carry_start is None:
                carry_start = start
            continue
        real_start = carry_start if carry_start is not None else start
        carry_start = None
        meta = dict(base)
        meta["heading_path"] = list(path)
        body_start = start + 1 if is_heading_section else start
        spans = _markdown_blocks(lines, body_start, end, in_code, meta, max_chars)
        if spans:
            # La ou les lignes de titre sont rattachées au premier bloc (jamais un morceau « titre seul »).
            spans[0].head = spans[0].a - real_start
            spans[0].a = real_start
        elif is_heading_section:
            spans = [_Span(real_start, start + 1, dict(meta))]
        pieces.extend(_pack(lines, spans, max_chars, overlap_chars))
    return pieces


# --------------------------------------------------------------------------- mode prose

def _prose_pieces(lines: List[str], max_chars: int, overlap_chars: int, language: Optional[str]) -> List[_Piece]:
    meta: Dict[str, Any] = {"kind": "prose"}
    if language:
        meta["language"] = language
    budget = max(1, max_chars - overlap_chars) if overlap_chars < max_chars // 2 else max_chars
    spans = [_Span(a, b, dict(meta), split="sentences") for a, b in _paragraph_blocks(lines, 0, len(lines))]
    pieces = _pack(lines, spans, budget, overlap_chars)
    if overlap_chars <= 0 or len(pieces) < 2:
        return pieces
    # Chevauchement : chaque morceau reprend les dernières phrases du précédent (si contiguës).
    out: List[_Piece] = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        if cur.sl <= prev.el and not (cur.sl == prev.el and cur.sc >= prev.ec):
            out.append(cur)  # déjà en chevauchement (découpe par phrases)
            continue
        last_line = lines[prev.el][:prev.ec]
        bounds = _sentence_bounds(last_line)
        tail_start: Optional[int] = None
        for s, _ in reversed(bounds):
            if prev.ec - s <= overlap_chars:
                tail_start = s
            else:
                break
        if tail_start is None or (tail_start == 0 and prev.sl == prev.el and prev.sc == 0):
            out.append(cur)
            continue
        while tail_start < prev.ec and last_line[tail_start].isspace():
            tail_start += 1
        extended = _Piece(prev.el, tail_start, cur.el, cur.ec, cur.meta)
        if len(_piece_text(lines, extended)) > max_chars:
            out.append(cur)  # le morceau courant est déjà une ligne trop longue : pas de rallonge
            continue
        out.append(extended)
    return out


# --------------------------------------------------------------------------- mode data

_NUMERIC_RE = re.compile(r"^[-+]?\d[\d\s.,]*%?$")


def _looks_like_header(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 2000:
        return False
    delim = max(",;\t|", key=lambda d: s.count(d))
    if s.count(delim) == 0:
        return False
    cells = [c.strip().strip("\"'") for c in s.strip("|").split(delim)]
    if len(cells) < 2:
        return False
    for c in cells:
        if not c or len(c) > 80 or _NUMERIC_RE.match(c):
            return False
    return True


def _data_pieces(lines: List[str], max_chars: int, overlap_chars: int, language: Optional[str]) -> List[_Piece]:
    meta: Dict[str, Any] = {"kind": "data"}
    if language:
        meta["language"] = language
    first = next((l for l in lines if l.strip()), "")
    if _looks_like_header(first):
        meta["header"] = first.strip()
    return [_whole(i, j, lines, meta) for i, j in
            _line_windows(lines, 0, len(lines), max_chars, overlap_chars, max_lines=DATA_WINDOW_LINES)]


# --------------------------------------------------------------------------- finalisation

def _finalize(lines: List[str], pieces: Iterable[_Piece]) -> List[Chunk]:
    chunks: List[Chunk] = []
    last: Optional[Tuple[int, int, int, int]] = None
    for p in pieces:
        sl, sc, el, ec = p.sl, p.sc, p.el, p.ec
        if ec == 0 and el > sl:
            el -= 1
            ec = len(lines[el])
        # rogne les lignes vides en tête et en queue
        while sl < el and not lines[sl][sc:].strip():
            sl += 1
            sc = 0
        while el > sl and not lines[el][:ec].strip():
            el -= 1
            ec = len(lines[el])
        if sl > el or (sl == el and sc >= ec):
            continue
        key = (sl, sc, el, ec)
        if last is not None and key[:2] >= last[:2] and key[2:] <= last[2:]:
            continue  # déjà couvert par le morceau précédent
        text = _piece_text(lines, _Piece(sl, sc, el, ec, p.meta))
        if not text.strip():
            continue
        chunks.append(Chunk(text=text, ordinal=len(chunks), line_start=sl + 1, line_end=el + 1, meta=dict(p.meta)))
        last = key
    return chunks


# --------------------------------------------------------------------------- API publique

def chunk_text(text: str, kind: str, *, max_chars: int = 1500, overlap_chars: int = 150,
               language: Optional[str] = None) -> List[Chunk]:
    """Découpe ``text`` selon ``kind`` (``code``, ``markdown``, ``prose`` ou ``data``).

    Voir le module pour les invariants. ``max_chars`` < 1 ou ``kind`` inconnu sont des erreurs
    de programmation (``ValueError``).
    """
    if kind not in CHUNK_KINDS:
        raise ValueError("kind inconnu: %r (attendu: %s)" % (kind, ", ".join(CHUNK_KINDS)))
    if max_chars < 1:
        raise ValueError("max_chars doit être >= 1")
    if not text or not text.strip():
        return []
    overlap = max(0, min(overlap_chars, max_chars // 3))
    lines = text.splitlines()
    if kind == "code":
        pieces = _pack(lines, _code_spans(lines, language), max_chars, overlap)
    elif kind == "markdown":
        pieces = _markdown_pieces(lines, max_chars, overlap, language)
    elif kind == "prose":
        pieces = _prose_pieces(lines, max_chars, overlap, language)
    else:
        pieces = _data_pieces(lines, max_chars, overlap, language)
    return _finalize(lines, pieces)


def chunk_mode_for(language: Optional[str], kind: Optional[str]) -> str:
    """Mode de découpe d'après la nature du fichier et son langage."""
    lang = (language or "").strip().lower()
    k = (kind or "").strip().lower()
    if k == "code":
        return "code"
    if k == "markdown" or lang in ("markdown", "md"):
        return "markdown"
    if k == "data" or lang in DATA_LANGUAGES:
        return "data"
    if lang and k in ("", "source", "script") and lang not in ("text", "txt", "prose", "pdf", "docx", "html", "rst"):
        return "code"
    return "prose"


def chunk_for_file(text: str, language: Optional[str], kind: Optional[str], *, max_chars: int = 1500,
                   overlap_chars: int = 150) -> List[Chunk]:
    """Découpe un fichier en choisissant le mode : ``code`` pour le code, ``markdown`` pour le
    Markdown, ``data`` pour SQL/CSV/JSON/YAML, ``prose`` sinon."""
    mode = chunk_mode_for(language, kind)
    return chunk_text(text, mode, max_chars=max_chars, overlap_chars=overlap_chars, language=language)
