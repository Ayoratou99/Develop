"""Extracteur JavaScript / TypeScript (JS, JSX, TS, TSX, Vue, Svelte).

Analyse purement lexicale : aucun parseur externe. Le texte est d'abord « masqué »
(chaînes, commentaires, gabarits et littéraux regex remplacés par des espaces, longueur
et retours à la ligne conservés) ce qui permet de compter les accolades de façon fiable
et d'appliquer des expressions régulières ligne par ligne sans risque de rétro-suivi
catastrophique. Les lignes brutes servent à relire les chemins des appels HTTP et les
modules importés.

Pour les fichiers ``.vue`` / ``.svelte`` seul le contenu des blocs ``<script>`` est
analysé ; les autres lignes sont vidées pour conserver les numéros de ligne.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from develop.code_analysis.model import (
    ExtractedEndpoint,
    ExtractedImport,
    ExtractedModel,
    ExtractedSymbol,
    FileAnalysis,
    ModelField,
)

LANGUAGES = ["javascript", "jsx", "typescript", "tsx", "vue", "svelte"]

# Au-delà de cette longueur une ligne est considérée comme minifiée : elle est tronquée pour les
# expressions régulières ligne par ligne (le masquage, linéaire, reste complet).
_MAX_LINE_FOR_REGEX = 20000
# Fenêtre de lignes utilisée pour lire les arguments d'un appel (options fetch, chemin sur la ligne suivante…).
_LOOKAHEAD_LINES = 3
_MAX_SIGNATURE = 200

_IDENT = r"[A-Za-z_$][\w$]*"
_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "del", "head", "options", "all", "request")
_METHOD_ALIASES = {"del": "DELETE", "all": "ANY", "request": "ANY"}
_JS_KEYWORDS = frozenset(
    "if for while switch catch return new typeof await yield else do try function class const let var "
    "import export default delete void in of instanceof throw with super this null true false undefined "
    "async static get set extends implements interface type enum declare abstract readonly namespace module "
    "break continue case finally".split()
)
_PASCAL_CASE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_HTTP_EXPORT_NAMES = frozenset(["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])

# --------------------------------------------------------------------------- masquage lexical

_TOKEN_RE = re.compile(r"//|/\*|['\"`/]")
_SQ_STRING_RE = re.compile(r"(?:[^'\\\n]|\\.)*'")
_DQ_STRING_RE = re.compile(r"(?:[^\"\\\n]|\\.)*\"")
_REGEX_BODY_RE = re.compile(r"((?:[^/\\\n\[]|\\.|\[(?:[^\]\\\n]|\\.)*\])+)/([a-z]*)")
_TEMPLATE_TOKEN_RE = re.compile(r"\\.|`|\$\{")
_REGEX_PREFIX_RE = re.compile(r"(?:^|[(,=:\[!&|?{};+\-*%<>~^]|\breturn|\btypeof|\bcase|\bdo|\belse|\bin|\bof)\s*$")


def _blank(segment: str) -> str:
    """Remplace un segment par des espaces en conservant les retours à la ligne."""
    if "\n" not in segment:
        return " " * len(segment)
    return "".join("\n" if ch == "\n" else " " for ch in segment)


def mask_source(text: str) -> str:
    """Renvoie le texte où chaînes, gabarits, commentaires et regex sont remplacés par des espaces.

    Les délimiteurs de chaînes (``'``, ``"``, `````) sont conservés pour que les motifs
    ``(['"])`` restent détectables ; longueur et positions sont strictement préservées.
    """
    out: List[str] = []
    pos = 0
    n = len(text)
    # pile des gabarits ouverts : chaque entrée = profondeur d'accolades de l'expression ${ } courante (-1 = hors expression)
    template_stack: List[int] = []
    while pos < n:
        if template_stack and template_stack[-1] < 0:
            # à l'intérieur d'un gabarit (hors ${ }) : chercher ` ou ${
            m = _TEMPLATE_TOKEN_RE.search(text, pos)
            if m is None:
                out.append(_blank(text[pos:]))
                pos = n
                break
            out.append(_blank(text[pos:m.start()]))
            tok = m.group(0)
            if tok == "`":
                out.append("`")
                template_stack.pop()
            elif tok == "${":
                out.append("  ")
                template_stack[-1] = 0  # entrée dans une expression interpolée
            else:  # séquence d'échappement
                out.append("  ")
            pos = m.end()
            continue
        m = _TOKEN_RE.search(text, pos)
        if m is None:
            out.append(text[pos:])
            pos = n
            break
        # code ordinaire entre deux jetons : conservé, mais on suit les accolades d'une expression ${ }
        chunk = text[pos:m.start()]
        if template_stack:
            depth = template_stack[-1]
            closed_at = -1
            for i, ch in enumerate(chunk):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        closed_at = i
                        break
                    depth -= 1
            if closed_at >= 0:
                out.append(chunk[:closed_at] + " ")
                template_stack[-1] = -1
                pos = pos + closed_at + 1
                continue
            template_stack[-1] = depth
        out.append(chunk)
        pos = m.start()
        tok = m.group(0)
        if tok == "//":
            end = text.find("\n", pos)
            if end < 0:
                end = n
            out.append(_blank(text[pos:end]))
            pos = end
        elif tok == "/*":
            end = text.find("*/", pos + 2)
            end = n if end < 0 else end + 2
            out.append(_blank(text[pos:end]))
            pos = end
        elif tok == "'" or tok == '"':
            body_re = _SQ_STRING_RE if tok == "'" else _DQ_STRING_RE
            bm = body_re.match(text, pos + 1)
            if bm is None:  # chaîne non terminée sur la ligne : on masque jusqu'à la fin de ligne
                end = text.find("\n", pos)
                end = n if end < 0 else end
                out.append(tok + _blank(text[pos + 1:end]))
                pos = end
            else:
                out.append(tok + _blank(text[pos + 1:bm.end() - 1]) + tok)
                pos = bm.end()
        elif tok == "`":
            out.append("`")
            template_stack.append(-1)
            pos += 1
        else:  # '/' : littéral regex ou division
            prefix = "".join(piece[-40:] for piece in out[-3:])[-40:] if out else ""
            nxt = text[pos + 1:pos + 2]
            # ``</li>`` et ``/>`` (JSX) ne sont jamais des littéraux regex
            is_regex = _REGEX_PREFIX_RE.search(prefix) is not None and not prefix.endswith("<") and nxt not in (">", "")
            rm = _REGEX_BODY_RE.match(text, pos + 1) if is_regex else None
            if rm is None:
                out.append("/")
                pos += 1
            else:  # corps masqué, délimiteurs et drapeaux conservés (``/   /g``)
                out.append("/" + _blank(rm.group(1)) + "/" + rm.group(2))
                pos = rm.end()
    return "".join(out)


# --------------------------------------------------------------------------- utilitaires de lignes


def _split_lines(text: str) -> List[str]:
    return text.split("\n")


def _strip_cr(line: str) -> str:
    return line[:-1] if line.endswith("\r") else line


def _find_body_open(masked: List[str], line_idx: int, col: int, max_lines: int = 20) -> Optional[Tuple[int, int]]:
    """Trouve la première accolade ouvrante « de corps » à partir de (ligne, colonne) dans le texte masqué.

    Les parenthèses/crochets ouverts sont ignorés (paramètres par défaut ``{}``) ; un ``;`` ou un
    ``=>`` à profondeur zéro arrête la recherche. Si la ligne de départ se termine par ``{``, cette
    accolade finale est retenue (évite de confondre un type de retour objet avec le corps).
    """
    first = masked[line_idx][col:].rstrip()
    if first.endswith("{") and "=>" not in first:
        return line_idx, col + len(first) - 1
    depth = 0
    for li in range(line_idx, min(len(masked), line_idx + max_lines)):
        seg = masked[li]
        start = col if li == line_idx else 0
        i = start
        while i < len(seg):
            ch = seg[i]
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "{" and depth <= 0:
                return li, i
            elif ch == "=" and depth <= 0 and seg[i + 1:i + 2] == ">":
                return None
            elif ch == ";" and depth <= 0:
                return None
            i += 1
    return None


def _find_block_end(masked: List[str], line_idx: int, col: int) -> Optional[int]:
    """Renvoie l'index (0-based) de la ligne contenant l'accolade fermante associée à celle en (ligne, col)."""
    depth = 0
    for li in range(line_idx, len(masked)):
        seg = masked[li]
        start = col if li == line_idx else 0
        for ch in seg[start:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return li
    return None


def _find_expression_end(masked: List[str], line_idx: int, col: int, max_lines: int = 400) -> int:
    """Fin (index 0-based) d'une expression commençant en (ligne, col) : premier ``;`` à profondeur zéro,
    ou fin de la première ligne où la profondeur de parenthèses/accolades est revenue à zéro."""
    depth = 0
    seen = False
    for li in range(line_idx, min(len(masked), line_idx + max_lines)):
        seg = masked[li]
        start = col if li == line_idx else 0
        for ch in seg[start:]:
            if ch in "([{":
                depth += 1
                seen = True
            elif ch in ")]}":
                depth -= 1
            elif ch == ";" and depth <= 0:
                return li
            elif not ch.isspace():
                seen = True
        if depth <= 0 and seen:
            return li
    return min(len(masked) - 1, line_idx + max_lines - 1)


def _collapse(text: str, limit: int = _MAX_SIGNATURE) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _signature(raw: List[str], line_idx: int, open_pos: Optional[Tuple[int, int]], fallback_end: int) -> str:
    """Texte de la signature : du début de la définition jusqu'à l'accolade du corps (exclue)."""
    if open_pos is not None:
        oli, oci = open_pos
        if oli == line_idx:
            seg = raw[line_idx][:oci]
        else:
            seg = "\n".join([raw[line_idx]] + raw[line_idx + 1:oli] + [raw[oli][:oci]])
    else:
        end = min(fallback_end, line_idx + 2)
        seg = "\n".join(raw[line_idx:end + 1])
    return _collapse(seg.rstrip().rstrip("{").rstrip())


# --------------------------------------------------------------------------- motifs de symboles

_FUNC_DECL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?function\s*\*?\s*(" + _IDENT + r")?\s*(?:<[^>]{0,200}>)?\s*\("
)
_VAR_DECL_RE = re.compile(
    r"^\s*(?:export\s+)?(?:declare\s+)?(const|let|var)\s+(" + _IDENT + r")\s*(?::\s*[^=]{0,300}?)?=(?!=)\s*(.*)$"
)
_VAR_NOINIT_RE = re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(const|let|var)\s+(" + _IDENT + r")\s*(?::\s*[^=;]{0,200})?;?\s*$")
_ARROW_INIT_RE = re.compile(
    r"^(?:async\s+)?(?:<[^>]{0,100}>)?\s*(?:\([^)]{0,500}\)|" + _IDENT + r")\s*(?::\s*[^=]{0,200})?=>"
)
_FUNC_INIT_RE = re.compile(r"^(?:async\s+)?function\b")
_WRAPPED_INIT_RE = re.compile(r"^(?:React\.)?(?:memo|forwardRef|observer|defineComponent|styled)\s*[(.<]")
_CLASS_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\b\s*(" + _IDENT + r")?(?:\s*<[^>]{0,200}>)?"
    r"(?:\s+extends\s+([\w$.]+(?:<[^>]{0,200}>)?))?"
)
_INTERFACE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?interface\s+(" + _IDENT + r")(?:\s*<[^>]{0,200}>)?(?:\s+extends\s+([^{]{0,300}))?"
)
_TYPE_RE = re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?type\s+(" + _IDENT + r")(?:\s*<[^>]{0,200}>)?\s*=\s*(.*)$")
_ENUM_RE = re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:const\s+)?enum\s+(" + _IDENT + r")")
_NAMESPACE_RE = re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?(?:namespace|module)\s+(" + _IDENT + r"(?:\.[\w$]+)*)\s*\{")
_MODIFIERS = r"(?:(?:public|private|protected|static|readonly|async|override|abstract|declare|get|set)\s+)*"
_METHOD_RE = re.compile(r"^\s*" + _MODIFIERS + r"\*?\s*#?(" + _IDENT + r"|\[[^\]]{0,100}\])\s*(?:<[^>]{0,200}>)?\s*\(")
_ARROW_PROP_RE = re.compile(r"^\s*" + _MODIFIERS + r"#?(" + _IDENT + r")\s*[?!]?\s*(?::\s*[^=]{0,300}?)?=(?!=)\s*(.*)$")
_PROPERTY_RE = re.compile(r"^\s*" + _MODIFIERS + r"#?(" + _IDENT + r")\s*[?!]?\s*(?::|=(?!=))")
_OBJ_METHOD_RE = re.compile(
    r"^\s*(?:async\s+)?(?:get\s+|set\s+)?\*?\s*(" + _IDENT + r")\s*\([^)]{0,500}\)\s*(?::\s*[^{;]{0,200})?\{\s*$"
)
_DECORATOR_LINE_RE = re.compile(r"^\s*@(" + _IDENT + r"(?:\.[\w$]+)*)")
_EXPORT_LIST_RE = re.compile(r"^\s*export\s*\{([^}]{0,2000})\}\s*;?\s*$")
_EXPORT_DEFAULT_NAME_RE = re.compile(r"^\s*export\s+default\s+(" + _IDENT + r")\s*;?\s*$")
_MODULE_EXPORTS_RE = re.compile(r"^\s*module\.exports\s*=\s*(.*)$")
_EXPORTS_PROP_RE = re.compile(r"^\s*(?:module\.)?exports\.(" + _IDENT + r")\s*=")
_JSX_TAG_RE = re.compile(r"<[A-Za-z][\w.]*(?:\s|/?>)")
_HOOK_CALL_RE = re.compile(r"\buse[A-Z]\w*\s*\(")
_RETURN_PAREN_RE = re.compile(r"\breturn\s*\(")
_REACT_BASE_RE = re.compile(r"^(?:React\.)?(?:Pure)?Component\b")


class _Analysis:
    """Contexte d'analyse d'un fichier : lignes brutes, lignes masquées, profondeur d'accolades."""

    def __init__(self, text: str, path: str, language: str) -> None:
        self.path = path.replace("\\", "/")
        self.language = language
        self.fa = FileAnalysis(language=language)
        self.raw = _split_lines(text)
        masked = mask_source(text)
        self.masked = _split_lines(masked)
        if len(self.masked) != len(self.raw):  # ne devrait jamais arriver : le masquage préserve les \n
            self.masked = (self.masked + [""] * len(self.raw))[:len(self.raw)]
            self.fa.warnings.append("masquage lexical incohérent : numéros de ligne approximatifs")
        self.lines_for_regex: List[str] = []
        truncated = 0
        for ml in self.masked:
            ml = _strip_cr(ml)
            if len(ml) > _MAX_LINE_FOR_REGEX:
                truncated += 1
                ml = ml[:_MAX_LINE_FOR_REGEX]
            self.lines_for_regex.append(ml)
        if truncated:
            self.fa.warnings.append(
                "%d ligne(s) de plus de %d caractères (fichier minifié ?) : analyse tronquée" % (truncated, _MAX_LINE_FOR_REGEX)
            )
        # profondeur d'accolades au début de chaque ligne
        self.depth_start: List[int] = []
        depth = 0
        for ml in self.masked:
            self.depth_start.append(depth)
            depth += ml.count("{") - ml.count("}")
        if depth != 0:
            self.fa.warnings.append("accolades non équilibrées (profondeur %d en fin de fichier)" % depth)
        # plages de classes : (début, fin, nom, profondeur de la ligne class)
        self.class_ranges: List[Tuple[int, int, str, int]] = []
        # index de la classe englobante la plus interne par ligne (-1 = aucune)
        self.class_owner: List[int] = [-1] * len(self.raw)
        # plages de fonctions nommées pour retrouver la fonction englobante : (début, fin, nom qualifié)
        self.func_ranges: List[Tuple[int, int, str]] = []
        self._func_owner: Optional[List[str]] = None
        self.symbols_by_name: Dict[str, List[ExtractedSymbol]] = {}
        # récepteurs HTTP : nom -> ("server"|"client", framework)
        self.receivers: Dict[str, Tuple[str, str]] = {}
        self.mounts: Dict[str, str] = {}
        self.server_framework: Optional[str] = None

    # ----------------------------------------------------------------- helpers

    def add_symbol(self, name: str, kind: str, start: int, end: int, signature: str,
                   parent: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> ExtractedSymbol:
        if end < start:
            end = start
        sym = ExtractedSymbol(name=name, kind=kind, line_start=start + 1, line_end=end + 1,
                              signature=signature, parent=parent, meta=meta or {})
        self.fa.symbols.append(sym)
        self.symbols_by_name.setdefault(name, []).append(sym)
        if kind in ("function", "method", "component"):
            qualified = "%s@%s" % (parent, name) if parent else name
            self.func_ranges.append((start, end, qualified))
        return sym

    def enclosing_function(self, idx: int) -> str:
        """Nom (qualifié ``Classe@méthode``) de la plus petite fonction nommée contenant la ligne.

        L'index par ligne est construit une fois (les fonctions internes, découvertes après leurs
        parentes, écrasent celles-ci) : coût linéaire en la somme des longueurs des fonctions.
        """
        if self._func_owner is None:
            owner = [""] * len(self.raw)
            for start, end, name in self.func_ranges:
                for i in range(start, min(end, len(owner) - 1) + 1):
                    owner[i] = name
            self._func_owner = owner
        return self._func_owner[idx] if 0 <= idx < len(self._func_owner) else ""

    def enclosing_class(self, idx: int) -> Optional[Tuple[int, int, str, int]]:
        k = self.class_owner[idx] if 0 <= idx < len(self.class_owner) else -1
        return self.class_ranges[k] if k >= 0 else None

    def decorators_before(self, idx: int) -> List[Tuple[int, str]]:
        """Décorateurs ``@X(...)`` situés juste au-dessus de la ligne ``idx`` : liste de (ligne, texte)."""
        found: List[Tuple[int, str]] = []
        buf: List[str] = []
        raw_buf: List[str] = []
        j = idx - 1
        steps = 0
        while j >= 0 and steps < 40:
            steps += 1
            ml = self.lines_for_regex[j]
            buf.insert(0, ml)
            raw_buf.insert(0, _strip_cr(self.raw[j]))
            joined = "".join(buf)
            if joined.count("(") == joined.count(")"):
                if _DECORATOR_LINE_RE.match(ml):
                    found.insert(0, (j, _collapse(" ".join(raw_buf), 160)))
                    buf, raw_buf = [], []
                elif ml.strip() == "" and not buf[1:]:
                    buf, raw_buf = [], []
                    if found:  # une ligne vide sépare les décorateurs de la définition : on s'arrête
                        break
                else:
                    break
            j -= 1
        return found

    def is_code_at(self, idx: int, col: int) -> bool:
        """Vrai si la position n'est pas dans un commentaire / une chaîne (compare brut et masqué)."""
        ml = self.masked[idx]
        raw = self.raw[idx]
        return col < len(ml) and col < len(raw) and ml[col] == raw[col]

    def window(self, idx: int, extra: int = _LOOKAHEAD_LINES, limit: int = 2000) -> str:
        """Ligne brute + quelques lignes suivantes (bornées) pour lire les arguments d'un appel."""
        parts = [_strip_cr(self.raw[i]) for i in range(idx, min(len(self.raw), idx + 1 + extra))]
        return "\n".join(parts)[:limit]

    # ----------------------------------------------------------------- symboles

    def body_text(self, start: int, end: int, cap: int = 400) -> str:
        return "\n".join(self.lines_for_regex[start:min(end, start + cap) + 1])

    def looks_like_component(self, name: str, start: int, end: int) -> bool:
        if not _PASCAL_CASE.match(name) or self.language in ("vue", "svelte"):
            return False
        body = self.body_text(start, end)
        return bool(_JSX_TAG_RE.search(body) or _RETURN_PAREN_RE.search(body) or _HOOK_CALL_RE.search(body))

    def function_end(self, idx: int, col: int) -> Tuple[Optional[Tuple[int, int]], int]:
        """(position de l'accolade de corps, index de la dernière ligne) pour une définition classique."""
        open_pos = _find_body_open(self.masked, idx, col)
        if open_pos is None:
            return None, idx
        end = _find_block_end(self.masked, open_pos[0], open_pos[1])
        return open_pos, (end if end is not None else len(self.masked) - 1)

    def arrow_end(self, idx: int, col: int) -> Tuple[Optional[Tuple[int, int]], int]:
        """Fin d'une fonction fléchée dont l'initialisation commence en (idx, col) dans le texte masqué."""
        ml = self.masked[idx]
        arrow = ml.find("=>", col)
        if arrow < 0:
            return None, _find_expression_end(self.masked, idx, col)
        after = arrow + 2
        open_pos = _find_body_open(self.masked, idx, after, max_lines=2)
        if open_pos is not None:
            end = _find_block_end(self.masked, open_pos[0], open_pos[1])
            return open_pos, (end if end is not None else len(self.masked) - 1)
        return None, _find_expression_end(self.masked, idx, after)

    def extract_symbols(self) -> None:
        n = len(self.raw)
        for idx in range(n):
            ml = self.lines_for_regex[idx]
            if not ml or ml.isspace():
                continue
            try:
                self._symbols_on_line(idx, ml)
            except Exception as exc:  # noqa: BLE001 - une ligne exotique ne doit pas arrêter l'analyse
                self.fa.warnings.append("ligne %d : symbole ignoré (%s)" % (idx + 1, exc))

    def _symbols_on_line(self, idx: int, ml: str) -> None:
        cls = self.enclosing_class(idx)
        if cls is not None and self.depth_start[idx] == cls[3] + 1:
            self._class_member(idx, ml, cls)
            return
        # pré-filtres par sous-chaîne : évite d'exécuter chaque motif sur chaque ligne
        m = _FUNC_DECL_RE.match(ml) if "function" in ml else None
        if m:
            self._function_decl(idx, ml, m)
            return
        m = _CLASS_RE.match(ml) if "class" in ml else None
        if m and re.match(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\b", ml):
            self._class_decl(idx, ml, m)
            return
        m = _INTERFACE_RE.match(ml) if "interface" in ml else None
        if m:
            self._interface_decl(idx, ml, m)
            return
        m = _TYPE_RE.match(ml) if "type" in ml else None
        if m:
            self._type_decl(idx, ml, m)
            return
        m = _ENUM_RE.match(ml) if "enum" in ml else None
        if m:
            open_pos, end = self.function_end(idx, m.end())
            exported = ml.lstrip().startswith("export")
            members: List[str] = []
            if open_pos is not None:
                for piece in self._object_body_pieces(open_pos, end):
                    pm = re.match(r"^\s*(" + _IDENT + r"|'[^']*'|\"[^\"]*\")\s*(?:=|$)", piece.strip())
                    if pm:
                        members.append(pm.group(1).strip("'\""))
            self.add_symbol(m.group(1), "enum", idx, end, _signature(self.raw, idx, open_pos, end),
                            meta={"exported": exported, "members": members[:50]})
            return
        m = _NAMESPACE_RE.match(ml) if ("namespace" in ml or "module" in ml) else None
        if m:
            open_pos, end = self.function_end(idx, m.end() - 1)
            self.add_symbol(m.group(1), "module", idx, end, _signature(self.raw, idx, open_pos, end),
                            meta={"exported": ml.lstrip().startswith("export")})
            return
        has_decl = "const" in ml or "let" in ml or "var" in ml
        m = _VAR_DECL_RE.match(ml) if has_decl else None
        if m:
            self._var_decl(idx, ml, m)
            return
        m = _VAR_NOINIT_RE.match(ml) if has_decl else None
        if m and self.depth_start[idx] == 0:
            self.add_symbol(m.group(2), "constant" if m.group(1) == "const" else "variable", idx, idx,
                            _collapse(_strip_cr(self.raw[idx]).strip()),
                            meta={"exported": ml.lstrip().startswith("export"), "default": False, "declaration": m.group(1)})
            return
        if self.depth_start[idx] > 0 and "(" in ml and ml.rstrip().endswith("{"):
            m = _OBJ_METHOD_RE.match(ml)
            if m and m.group(1) not in _JS_KEYWORDS:
                open_pos, end = self.function_end(idx, m.end() - 1)
                self.add_symbol(m.group(1), "method", idx, end, _signature(self.raw, idx, open_pos, end),
                                meta={"exported": False, "object_method": True, "async": "async" in ml[:m.start(1)]})

    def _function_decl(self, idx: int, ml: str, m: "re.Match[str]") -> None:
        name = m.group(1) or "default"
        head = ml[:m.start(1)] if m.group(1) else ml[:m.end()]
        open_pos, end = self.function_end(idx, m.end() - 1)
        kind = "component" if self.looks_like_component(name, idx, end) else "function"
        meta: Dict[str, Any] = {
            "exported": bool(re.match(r"^\s*export\b", ml)),
            "default": bool(re.match(r"^\s*export\s+default\b", ml)),
            "async": bool(re.search(r"\basync\b", head)),
        }
        if self.depth_start[idx] > 0:
            meta["nested"] = True
        if open_pos is None:
            meta["declaration_only"] = True
        self.add_symbol(name, kind, idx, end, _signature(self.raw, idx, open_pos, end), meta=meta)

    def _var_decl(self, idx: int, ml: str, m: "re.Match[str]") -> None:
        keyword, name, rest = m.group(1), m.group(2), m.group(3)
        exported = bool(re.match(r"^\s*export\b", ml))
        init_col = m.start(3)
        func_like = False
        wrapped = False
        if _FUNC_INIT_RE.match(rest):
            func_like = True
            open_pos, end = self.function_end(idx, init_col)
        elif _ARROW_INIT_RE.match(rest) or (re.match(r"^(?:async\s+)?[(<]", rest) and "=>" in rest):
            func_like = True
            open_pos, end = self.arrow_end(idx, init_col)
        elif _WRAPPED_INIT_RE.match(rest):
            func_like = wrapped = True
            open_pos, end = None, _find_expression_end(self.masked, idx, init_col)
        else:
            open_pos, end = None, _find_expression_end(self.masked, idx, init_col)
        if func_like:
            kind = "component" if (self.looks_like_component(name, idx, end) or
                                   (wrapped and _PASCAL_CASE.match(name))) else "function"
            meta: Dict[str, Any] = {"exported": exported, "default": False, "async": bool(re.match(r"^async\b", rest)),
                                    "declaration": keyword}
            if wrapped:
                meta["wrapper"] = rest.split("(", 1)[0].strip()
            if self.depth_start[idx] > 0:
                meta["nested"] = True
            self.add_symbol(name, kind, idx, end, _signature(self.raw, idx, open_pos, end), meta=meta)
            return
        if self.depth_start[idx] == 0 and not re.match(r"^(?:await\s+)?require\s*\(", rest):
            kind = "constant" if keyword == "const" else "variable"
            meta = {"exported": exported, "default": False, "declaration": keyword}
            raw_rest = _strip_cr(self.raw[idx])[init_col:init_col + 200].strip()
            ic = re.match(r"^(?:axios|ky|got)\.(?:create|extend)\s*\(", raw_rest)
            if ic:
                meta["http_client"] = raw_rest.split(".", 1)[0]
            self.add_symbol(name, kind, idx, end, _collapse(_strip_cr(self.raw[idx]).strip()), meta=meta)

    def _class_decl(self, idx: int, ml: str, m: "re.Match[str]") -> None:
        name = m.group(1) or "default"
        base = m.group(2)
        open_pos, end = self.function_end(idx, m.end())
        decorators = self.decorators_before(idx)
        kind = "component" if (base and _REACT_BASE_RE.match(base)) else "class"
        meta: Dict[str, Any] = {
            "exported": bool(re.match(r"^\s*export\b", ml)),
            "default": bool(re.match(r"^\s*export\s+default\b", ml)),
        }
        if base:
            meta["extends"] = base
        im = re.search(r"\bimplements\s+([^{]{1,200})", ml)
        if im:
            meta["implements"] = [s.strip() for s in im.group(1).split(",") if s.strip()]
        if decorators:
            meta["decorators"] = [d[1] for d in decorators]
        if re.match(r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?abstract\b", ml):
            meta["abstract"] = True
        self.add_symbol(name, kind, idx, end, _signature(self.raw, idx, open_pos, end), meta=meta)
        self.class_ranges.append((idx, end, name, self.depth_start[idx]))
        k = len(self.class_ranges) - 1
        for i in range(idx + 1, min(end, len(self.class_owner) - 1) + 1):  # la ligne ``class`` elle-même est exclue
            self.class_owner[i] = k

    def _class_member(self, idx: int, ml: str, cls: Tuple[int, int, str, int]) -> None:
        class_name = cls[2]
        stripped = ml.strip()
        if not stripped or stripped[0] in "@}/)" or stripped.startswith("case ") or stripped.startswith("default:"):
            return
        m = _METHOD_RE.match(ml)
        if m and m.group(1) not in _JS_KEYWORDS and "=" not in ml[:m.start(1)]:
            name = m.group(1)
            open_pos, end = self.function_end(idx, m.end() - 1)
            meta: Dict[str, Any] = {"exported": False, "async": bool(re.search(r"\basync\b", ml[:m.start(1)]))}
            mods = re.findall(r"\b(public|private|protected|static|readonly|override|abstract|get|set)\b", ml[:m.start(1)])
            if mods:
                meta["modifiers"] = mods
            if open_pos is None:
                meta["declaration_only"] = True
            decorators = self.decorators_before(idx)
            if decorators:
                meta["decorators"] = [d[1] for d in decorators]
                meta["decorator_lines"] = [d[0] + 1 for d in decorators]
            self.add_symbol(name, "method", idx, end, _signature(self.raw, idx, open_pos, end), parent=class_name, meta=meta)
            return
        m = _ARROW_PROP_RE.match(ml)
        if m and m.group(1) not in _JS_KEYWORDS:
            name, rest = m.group(1), m.group(2)
            if _ARROW_INIT_RE.match(rest) or _FUNC_INIT_RE.match(rest) or (rest.startswith("(") and "=>" in rest):
                if _FUNC_INIT_RE.match(rest):
                    open_pos, end = self.function_end(idx, m.start(2))
                else:
                    open_pos, end = self.arrow_end(idx, m.start(2))
                meta = {"exported": False, "arrow": True, "async": bool(re.match(r"^async\b", rest))}
                self.add_symbol(name, "method", idx, end, _signature(self.raw, idx, open_pos, end), parent=class_name, meta=meta)
                return
            end = _find_expression_end(self.masked, idx, m.start(2))
            self.add_symbol(name, "property", idx, end, _collapse(_strip_cr(self.raw[idx]).strip()), parent=class_name,
                            meta={"exported": False})
            return
        m = _PROPERTY_RE.match(ml)
        if m and m.group(1) not in _JS_KEYWORDS:
            end = _find_expression_end(self.masked, idx, m.end())
            decorators = self.decorators_before(idx)
            meta = {"exported": False}
            if decorators:
                meta["decorators"] = [d[1] for d in decorators]
            self.add_symbol(m.group(1), "property", idx, end, _collapse(_strip_cr(self.raw[idx]).strip()),
                            parent=class_name, meta=meta)

    # ----------------------------------------------------------------- interfaces, types, modèles

    def _interface_decl(self, idx: int, ml: str, m: "re.Match[str]") -> None:
        name = m.group(1)
        open_pos, end = self.function_end(idx, m.end())
        meta: Dict[str, Any] = {
            "exported": bool(re.match(r"^\s*export\b", ml)),
            "default": bool(re.match(r"^\s*export\s+default\b", ml)),
        }
        if m.group(2):
            meta["extends"] = [s.strip() for s in _collapse(m.group(2)).split(",") if s.strip()]
        self.add_symbol(name, "interface", idx, end, _signature(self.raw, idx, open_pos, end), meta=meta)
        if open_pos is not None:
            self._object_model(name, "ts_interface", idx, open_pos, end, meta)

    def _type_decl(self, idx: int, ml: str, m: "re.Match[str]") -> None:
        name, rest = m.group(1), m.group(2)
        meta: Dict[str, Any] = {"exported": bool(re.match(r"^\s*export\b", ml)), "default": False}
        open_pos = None
        if rest.lstrip().startswith("{") or re.match(r"^[\w$.<>\[\], |&]{0,200}&\s*\{", rest):
            open_pos = _find_body_open(self.masked, idx, m.start(2))
        if open_pos is not None:
            end = _find_block_end(self.masked, open_pos[0], open_pos[1])
            end = end if end is not None else len(self.masked) - 1
            self.add_symbol(name, "type", idx, end, _signature(self.raw, idx, open_pos, end), meta=meta)
            self._object_model(name, "ts_type", idx, open_pos, end, meta)
            return
        end = _find_expression_end(self.masked, idx, m.start(2))
        raw_rhs = _collapse("\n".join(self.raw[idx:end + 1]), 400)
        raw_rhs = raw_rhs.split("=", 1)[1].strip().rstrip(";").strip() if "=" in raw_rhs else raw_rhs
        values = _literal_union(raw_rhs)
        if values is not None:
            meta["union_values"] = values
        self.add_symbol(name, "type", idx, end, _collapse(_strip_cr(self.raw[idx]).strip().rstrip(";")), meta=meta)

    def _object_model(self, name: str, kind: str, decl_idx: int, open_pos: Tuple[int, int], end: int,
                      sym_meta: Dict[str, Any]) -> None:
        pieces = self._object_body_pieces(open_pos, end)
        fields: List[ModelField] = []
        for piece in pieces:
            f = _parse_field(piece)
            if f is not None:
                fields.append(f)
        meta: Dict[str, Any] = {"exported": bool(sym_meta.get("exported"))}
        if sym_meta.get("extends"):
            meta["extends"] = sym_meta["extends"]
        self.fa.models.append(ExtractedModel(name=name, kind=kind, fields=fields, line=decl_idx + 1, meta=meta))

    def _object_body_pieces(self, open_pos: Tuple[int, int], end: int, max_lines: int = 2000) -> List[str]:
        """Découpe le corps d'un objet (entre ses accolades) en morceaux de profondeur 1 séparés par ; , ou \\n.

        Le texte masqué pilote la profondeur et élimine les commentaires ; le contenu brut est repris
        à l'intérieur des chaînes (valeurs d'énumération) et pour le reste du code.
        """
        pieces: List[str] = []
        cur: List[str] = []
        depth = 0
        quote = ""
        oli, oci = open_pos
        for li in range(oli, min(end, oli + max_lines) + 1):
            ml = self.masked[li]
            raw = self.raw[li]
            start = oci if li == oli else 0
            for i in range(start, len(ml)):
                mc = ml[i]
                rc = raw[i] if i < len(raw) else mc
                if quote:
                    cur.append(rc)
                    if mc == quote:
                        quote = ""
                    continue
                if mc in "'\"`":
                    quote = mc
                    cur.append(mc)
                    continue
                if mc == "{":
                    depth += 1
                    if depth == 1:
                        continue
                elif mc == "}":
                    depth -= 1
                    if depth == 0:
                        pieces.append("".join(cur))
                        return [p for p in pieces if p.strip()]
                if depth == 1 and mc in ";,\n":
                    pieces.append("".join(cur))
                    cur = []
                else:
                    cur.append(mc)
            if depth == 1:  # fin de ligne à profondeur 1 : un champ sans ; ni , se termine ici
                pieces.append("".join(cur))
                cur = []
            elif depth > 1 and (not cur or cur[-1] != "\n"):
                cur.append("\n")
        pieces.append("".join(cur))
        return [p for p in pieces if p.strip()]

    # ----------------------------------------------------------------- exports différés

    def apply_late_exports(self) -> None:
        for idx, ml in enumerate(self.lines_for_regex):
            if "export" not in ml:
                continue
            m = _EXPORT_LIST_RE.match(ml)
            if m and " from " not in ml:
                for item in m.group(1).split(","):
                    item = item.strip()
                    if not item:
                        continue
                    item = re.sub(r"^type\s+", "", item)
                    parts = re.split(r"\s+as\s+", item)
                    self._mark_exported(parts[0].strip(), default=(len(parts) > 1 and parts[1].strip() == "default"))
                continue
            m = _EXPORT_DEFAULT_NAME_RE.match(ml)
            if m:
                self._mark_exported(m.group(1), default=True)
                continue
            m = _MODULE_EXPORTS_RE.match(ml)
            if m:
                end = _find_expression_end(self.masked, idx, m.start(1), max_lines=40)
                expr = " ".join(self.lines_for_regex[idx:end + 1])
                expr = expr.split("=", 1)[1] if "=" in expr else expr
                for ident in re.findall(r"\b" + _IDENT + r"\b", expr):
                    if ident not in _JS_KEYWORDS:
                        self._mark_exported(ident, commonjs=True)
                continue
            m = _EXPORTS_PROP_RE.match(ml)
            if m:
                self._mark_exported(m.group(1), commonjs=True)
                rhs = ml[m.end():].strip().rstrip(";").strip()
                if re.match(r"^" + _IDENT + r"$", rhs):
                    self._mark_exported(rhs, commonjs=True)

    def _mark_exported(self, name: str, default: bool = False, commonjs: bool = False) -> None:
        for sym in self.symbols_by_name.get(name, []):
            if sym.parent is None:
                sym.meta["exported"] = True
                if default:
                    sym.meta["default"] = True
                if commonjs:
                    sym.meta["commonjs"] = True

    # ----------------------------------------------------------------- imports

    def extract_imports(self) -> None:
        n = len(self.raw)
        idx = 0
        while idx < n:
            ml = self.lines_for_regex[idx]
            consumed = 1
            try:
                if "import" in ml or "require" in ml or "export" in ml:
                    consumed = self._imports_on_line(idx, ml) or 1
            except Exception as exc:  # noqa: BLE001
                self.fa.warnings.append("ligne %d : import ignoré (%s)" % (idx + 1, exc))
            idx += consumed

    def _statement_window(self, idx: int, max_lines: int = 40) -> Tuple[str, int]:
        """Texte brut d'une instruction import/export pouvant s'étendre sur plusieurs lignes."""
        parts: List[str] = []
        for j in range(idx, min(len(self.raw), idx + max_lines)):
            parts.append(_strip_cr(self.raw[j]))
            ml = self.lines_for_regex[j]
            if re.search(r"\bfrom\s*['\"]", ml) or re.search(r"['\"]\s*;?\s*$", ml) or ";" in ml:
                return "\n".join(parts), j - idx + 1
        return "\n".join(parts), 1

    def _imports_on_line(self, idx: int, ml: str) -> int:
        raw = _strip_cr(self.raw[idx])
        consumed = 1
        if _IMPORT_START_RE.match(ml) and self.is_code_at(idx, ml.find("import")):
            m = _IMPORT_EQ_RE.match(raw)
            if m:
                self._add_import(m.group(3), [m.group(1)], idx)
                return 1
            m = _IMPORT_SIDE_RE.match(raw)
            if m:
                self._add_import(m.group(2), [], idx)
                return 1
            text, consumed = self._statement_window(idx)
            m = _IMPORT_FROM_RE.match(text)
            if m:
                self._add_import(m.group(3), _import_names(m.group(1)), idx)
                return consumed
            return 1
        if re.match(r"^\s*export\b", ml) and self.is_code_at(idx, ml.find("export")):
            text, consumed = self._statement_window(idx)
            m = _EXPORT_FROM_RE.match(text)
            if m:
                clause = m.group(1)
                names = _import_names(clause) if clause.startswith("{") else ["*"]
                self._add_import(m.group(3), names, idx)
                return consumed
            return 1
        for m in _REQUIRE_RE.finditer(raw):
            if not self.is_code_at(idx, m.start()):
                continue
            names: List[str] = []
            dm = _REQUIRE_DECL_RE.match(raw)
            if dm and m.start() < dm.end():
                if dm.group(1):
                    names = [dm.group(1)]
                else:
                    names = [re.split(r"\s*:\s*", s.strip())[0].strip() for s in dm.group(2).split(",") if s.strip()]
            self._add_import(m.group(2), names, idx)
        for m in _DYNAMIC_IMPORT_RE.finditer(raw):
            if self.is_code_at(idx, m.start()):
                self._add_import(m.group(2), [], idx)
        return consumed

    def _add_import(self, module: str, names: List[str], idx: int) -> None:
        module = module.strip()
        if not module:
            return
        is_relative = module.startswith("./") or module.startswith("../") or module in (".", "..")
        self.fa.imports.append(ExtractedImport(module=module, names=names, line=idx + 1, is_relative=is_relative))


# --------------------------------------------------------------------------- motifs d'imports

_IMPORT_START_RE = re.compile(r"^\s*import\b(?!\s*\()")
_IMPORT_FROM_RE = re.compile(r"^\s*import\s+(?:type\s+)?(.*?)\s*from\s*(['\"])([^'\"\n]+)\2", re.S)
_IMPORT_SIDE_RE = re.compile(r"^\s*import\s*(['\"])([^'\"\n]+)\1")
_IMPORT_EQ_RE = re.compile(r"^\s*import\s+(" + _IDENT + r")\s*=\s*require\s*\(\s*(['\"])([^'\"\n]+)\2")
_EXPORT_FROM_RE = re.compile(
    r"^\s*export\s+(?:type\s+)?(\*(?:\s+as\s+" + _IDENT + r")?|\{[^}]{0,2000}\})\s*from\s*(['\"])([^'\"\n]+)\2", re.S
)
_REQUIRE_RE = re.compile(r"\brequire\s*\(\s*(['\"])([^'\"\n]+)\1\s*\)")
_REQUIRE_DECL_RE = re.compile(r"^\s*(?:const|let|var)\s+(?:(" + _IDENT + r")|\{([^}]{0,500})\})\s*=\s*require\s*\(")
_DYNAMIC_IMPORT_RE = re.compile(r"\bimport\s*\(\s*(['\"])([^'\"\n]+)\1\s*\)")


def _import_names(clause: str) -> List[str]:
    """Noms importés d'une clause ``X, { a, b as c, type D }, * as ns`` (noms source, ``*`` pour un espace de noms)."""
    names: List[str] = []
    clause = clause.strip()
    if not clause:
        return names
    bm = re.search(r"\{([^}]*)\}", clause)
    head = clause[:bm.start()] if bm else clause
    tail = clause[bm.end():] if bm else ""
    for part in (head + "," + tail).split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("*"):
            names.append("*")
        elif re.match(r"^" + _IDENT + r"$", part):
            names.append(part)
    if bm:
        for item in bm.group(1).split(","):
            item = re.sub(r"^\s*type\s+", "", item.strip()).strip()
            if not item:
                continue
            src = re.split(r"\s+as\s+", item)[0].strip()
            if src:
                names.append(src)
    return names


# --------------------------------------------------------------------------- champs de modèles

_FIELD_RE = re.compile(r"^\s*(?:readonly\s+)?(" + _IDENT + r"|'[^']*'|\"[^\"]*\")\s*(\?)?\s*:\s*(.+)$", re.S)
_LITERAL_RE = re.compile(r"^(['\"`])(.*)\1$", re.S)


def _split_union(type_text: str) -> List[str]:
    """Découpe un type sur les ``|`` de premier niveau (parenthèses, crochets, accolades, chevrons ignorés)."""
    parts: List[str] = []
    depth = 0
    cur: List[str] = []
    quote = ""
    for ch in type_text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"`":
            quote = ch
        elif ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "|" and depth <= 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


def _literal_union(type_text: str) -> Optional[List[str]]:
    """Valeurs d'une union de littéraux (``"a" | "b"``, ``null`` toléré) ; None si ce n'est pas le cas."""
    values: List[str] = []
    for part in _split_union(type_text):
        if part in ("null", "undefined"):
            continue
        lm = _LITERAL_RE.match(part)
        if lm is None:
            return None
        values.append(lm.group(2))
    return values if values else None


def _parse_field(piece: str) -> Optional[ModelField]:
    piece = piece.strip()
    if not piece or piece[0] in "[(<":
        return None
    m = _FIELD_RE.match(piece)
    if m is None:
        return None
    name = m.group(1).strip("'\"")
    optional = m.group(2) == "?"
    type_text = _collapse(m.group(3), 300).rstrip(";,").strip()
    parts = _split_union(type_text)
    nullable = optional or any(p in ("null", "undefined") for p in parts)
    meta: Dict[str, Any] = {}
    values = _literal_union(type_text)
    if values is not None:
        meta["enum_values"] = values
    if optional:
        meta["optional"] = True
    if piece.startswith("readonly"):
        meta["readonly"] = True
    return ModelField(name=name, type=type_text, nullable=nullable, meta=meta)


# --------------------------------------------------------------------------- endpoints

_ROUTE_CALL_RE = re.compile(
    r"(?<![\w$])((?:this\.)?" + _IDENT + r"(?:\.[\w$]+){0,2})\.(get|post|put|patch|delete|del|head|options|all|request)"
    r"\s*(?:<[^>()\n]{0,200}>)?\s*\("
)
_FETCH_CALL_RE = re.compile(r"(?<![\w$.])(fetch|\$fetch|useFetch|useSWR|useSWRImmutable|ky|got|superagent)\s*(?:<[^>()\n]{0,100}>)?\s*\(")
_AXIOS_CALL_RE = re.compile(r"(?<![\w$.])axios\s*(?:<[^>()\n]{0,100}>)?\s*\(")
_ROUTE_CHAIN_RE = re.compile(r"(?<![\w$])(" + _IDENT + r")\.route\s*\(\s*(['\"`])([^'\"`\n]*)\2\s*\)")
_ROUTE_OBJECT_RE = re.compile(r"(?<![\w$])(" + _IDENT + r")\.route\s*\(\s*\{")
_CHAIN_METHOD_RE = re.compile(r"\.(get|post|put|patch|delete|del|all)\s*\(")
_MOUNT_RE = re.compile(r"(?<![\w$])(" + _IDENT + r")\.use\s*\(\s*(['\"`])([^'\"`\n]*)\2\s*,\s*(" + _IDENT + r")\b")
_OPTION_METHOD_RE = re.compile(r"\bmethod\s*:\s*(?:\[\s*)?(['\"])([A-Za-z]+)\1")
_OPTION_URL_RE = re.compile(r"\b(?:url|path)\s*:\s*")
_OPTION_BASE_RE = re.compile(r"\bbaseURL\s*:\s*(?:(['\"`])([^'\"`\n]*)\1|(" + _IDENT + r"(?:\.[\w$]+)*))")
_CLIENT_INSTANCE_RE = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(" + _IDENT + r")\s*(?::[^=]{0,100})?=\s*(axios|ky|got)\.(?:create|extend)\s*\(")
_SERVER_INSTANCE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(" + _IDENT + r")\s*(?::[^=]{0,100})?=\s*"
    r"(express\s*\(|express\.Router\s*\(|Router\s*\(|new\s+Router\b|new\s+KoaRouter\b|fastify\s*\(|Fastify\s*\(|"
    r"new\s+Hono\b|new\s+Koa\b|polka\s*\(|restify\.createServer\s*\(|Hapi\.server\s*\(|new\s+Elysia\b|express\.Router\b)"
)
# ``const app = require("express")()`` / ``const fastify = require("fastify")()`` : instance créée à l'import
# (motif appliqué à la ligne brute, le module étant une chaîne masquée ailleurs)
_SERVER_REQUIRE_INSTANCE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+(" + _IDENT + r")\s*=\s*require\s*\(\s*(['\"])([^'\"\n]+)\2\s*\)\s*\("
)
_SERVER_MODULES = {
    "express": "express", "koa": "koa", "@koa/router": "koa", "koa-router": "koa", "fastify": "fastify",
    "hono": "hono", "restify": "restify", "polka": "polka", "@hapi/hapi": "hapi", "elysia": "elysia",
}
# un récepteur nommé comme son framework (``fastify.get``, ``hono.post``) le désigne sans ambiguïté
_FRAMEWORK_RECEIVER_NAMES = {
    "express": "express", "fastify": "fastify", "koa": "koa", "hono": "hono", "polka": "polka",
    "restify": "restify", "hapi": "hapi", "elysia": "elysia",
}
_SERVER_RECEIVER_NAMES = frozenset(["app", "router", "server", "fastify", "koa", "hono", "api", "routes", "r"])
_CLIENT_RECEIVER_NAMES = {
    "axios": "axios", "ky": "ky", "got": "got", "superagent": "http", "$http": "http", "http": "http",
    "httpClient": "http", "client": "http", "apiClient": "http", "instance": "axios", "request": "http",
    "agent": "http", "fetcher": "http", "api": "http",
}
_ROUTE_HINTS = (".get", ".post", ".put", ".patch", ".delete", ".del", ".head", ".options", ".all", ".request")
_FETCH_HINTS = ("fetch", "useFetch", "useSWR", "ky", "got", "superagent")
_HANDLER_START_RE = re.compile(r"^\s*(?:async\b|function\b|\(|\[|" + _IDENT + r"\s*=>)")
_HANDLER_REF_RE = re.compile(r"^\s*(" + _IDENT + r"(?:\.[\w$]+){0,3})\s*$")
_HTTP_DECORATOR_RE = re.compile(r"^@(Get|Post|Put|Patch|Delete|All|Options|Head)\s*\(\s*(?:\[\s*)?(?:(['\"`])([^'\"`]*)\2)?")
_CONTROLLER_DECORATOR_RE = re.compile(
    r"^@(Controller|JsonController|Route)\s*\(\s*(?:(['\"`])([^'\"`]*)\2|\{[^}]*?\bpath\s*:\s*(['\"`])([^'\"`]*)\4)?"
)
_SCHEME_HOST_RE = re.compile(r"^[a-z][a-z0-9+.-]*://[^/\s`]*", re.I)
_NEXT_PAGES_API_RE = re.compile(r"(?:^|/)pages/api/(.+?)\.(?:[cm]?[jt]sx?)$")
_NEXT_APP_API_RE = re.compile(r"(?:^|/)app/api(?:/(.*?))?/route\.(?:[cm]?[jt]sx?)$")
_SVELTEKIT_SERVER_RE = re.compile(r"(?:^|/)routes(?:/(.*?))?/\+server\.[cm]?[jt]s$")
_NUXT_SERVER_RE = re.compile(r"(?:^|/)server/(api|routes)/(.+?)(?:\.(get|post|put|patch|delete|head|options))?\.[cm]?[jt]s$")


def _split_call_args(text: str, pos: int, max_chars: int = 2000) -> Tuple[List[str], bool]:
    """Sépare les arguments de premier niveau d'un appel dont la parenthèse ouvrante précède ``pos``.

    Renvoie (arguments, parenthèse fermante trouvée). Chaînes, gabarits et imbrications sont respectés.
    """
    args: List[str] = []
    cur: List[str] = []
    depth = 0
    quote = ""
    i = pos
    end = min(len(text), pos + max_chars)
    while i < end:
        ch = text[i]
        if quote:
            cur.append(ch)
            if ch == "\\" and i + 1 < end:
                cur.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "'\"`":
            quote = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            if depth == 0:
                args.append("".join(cur))
                return [a for a in args if a.strip()], True
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    args.append("".join(cur))
    return [a for a in args if a.strip()], False


def _url_from_arg(arg: str) -> Tuple[Optional[str], bool]:
    """Chemin « brut » d'un argument URL : littéraux concaténés, gabarits conservés, expressions en ``${…}``.

    Renvoie (chemin, dynamique) ; dynamique = aucune partie littérale (ex. ``fetch(url)``).
    """
    s = arg.strip()
    pieces: List[Tuple[str, str]] = []
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        ch = s[i]
        if ch in "'\"":
            j = i + 1
            buf: List[str] = []
            while j < n and s[j] != ch:
                if s[j] == "\\" and j + 1 < n:
                    buf.append(s[j + 1])
                    j += 2
                    continue
                buf.append(s[j])
                j += 1
            pieces.append(("lit", "".join(buf)))
            i = j + 1
        elif ch == "`":
            j = i + 1
            depth = 0
            while j < n:
                if s[j] == "\\":
                    j += 2
                    continue
                if s[j] == "$" and s[j + 1:j + 2] == "{":
                    depth += 1
                    j += 2
                    continue
                if s[j] == "}" and depth > 0:
                    depth -= 1
                elif s[j] == "`" and depth == 0:
                    break
                j += 1
            pieces.append(("tpl", s[i + 1:j]))
            i = j + 1
        else:
            j = i
            depth = 0
            while j < n:
                c = s[j]
                if c in "([{":
                    depth += 1
                elif c in ")]}":
                    depth -= 1
                    if depth < 0:
                        break
                elif c in "+," and depth == 0:
                    break
                elif c in "'\"`" and depth == 0:
                    break
                j += 1
            expr = s[i:j].strip()
            if expr:
                pieces.append(("expr", expr))
            i = j
        while i < n and s[i].isspace():
            i += 1
        if i < n and s[i] == "+":
            i += 1
            continue
        break
    if not pieces:
        return None, True
    dynamic = all(kind == "expr" for kind, _ in pieces)
    out: List[str] = []
    for kind, value in pieces:
        if kind == "expr":
            out.append("${%s}" % _collapse(value, 80))
        else:
            out.append(value)
    path = "".join(out)
    return (path if path else None), dynamic


def _base_of(path: str) -> Optional[str]:
    if path.startswith("${"):
        close = path.find("}")
        return path[2:close] if close > 0 else None
    m = _SCHEME_HOST_RE.match(path)
    return m.group(0) if m else None


def _join_route(prefix: str, sub: str) -> str:
    p = prefix.strip().strip("/")
    s = sub.strip().strip("/")
    joined = "/".join(x for x in (p, s) if x)
    return "/" + joined if joined else "/"


def _norm_method(name: str) -> str:
    return _METHOD_ALIASES.get(name.lower(), name.upper())


def _method_from_options(text: str, default: str = "GET") -> str:
    m = _OPTION_METHOD_RE.search(text)
    return m.group(2).upper() if m else default


class _EndpointPass:
    """Détection des routes serveur et des appels HTTP client (mixin logique sur ``_Analysis``)."""

    @staticmethod
    def scan_receivers(a: _Analysis) -> None:
        # récepteurs créés par un ``Router()`` / ``new Router()`` générique : leur framework est celui du fichier
        generic: List[str] = []
        for idx, ml in enumerate(a.lines_for_regex):
            if "=" not in ml and "require" not in ml and "import" not in ml:
                continue
            raw = _strip_cr(a.raw[idx])
            m = _CLIENT_INSTANCE_RE.match(ml)
            if m:
                a.receivers[m.group(1)] = ("client", m.group(2))
                continue
            m = _SERVER_INSTANCE_RE.match(ml)
            if m:
                init = m.group(2)
                fw = "express"
                for key in ("koa", "fastify", "hono", "polka", "restify", "hapi", "elysia"):
                    if key in init.lower():
                        fw = key
                if init.startswith("new Koa") or "KoaRouter" in init:
                    fw = "koa"
                if init.startswith(("Router", "new Router")):
                    # un routeur générique ne révèle pas le framework : il prendra celui du fichier
                    generic.append(m.group(1))
                    a.receivers[m.group(1)] = ("server", a.server_framework or fw)
                    continue
                a.receivers[m.group(1)] = ("server", fw)
                a.server_framework = a.server_framework or fw
                continue
            m = _SERVER_REQUIRE_INSTANCE_RE.match(raw)
            if m and m.group(3) in _SERVER_MODULES:
                fw = _SERVER_MODULES[m.group(3)]
                a.receivers[m.group(1)] = ("server", fw)
                a.server_framework = a.server_framework or fw
        for imp in a.fa.imports:
            fw = _SERVER_MODULES.get(imp.module)
            if fw and not a.server_framework:
                a.server_framework = fw
        # les ``Router()`` génériques créés avant la détection du framework (import plus bas) héritent de celui-ci
        if a.server_framework:
            for name in generic:
                a.receivers[name] = ("server", a.server_framework)

    @staticmethod
    def scan_mounts(a: _Analysis) -> None:
        for idx, ml in enumerate(a.lines_for_regex):
            if ".use" not in ml:
                continue
            for m in _MOUNT_RE.finditer(_strip_cr(a.raw[idx])):
                if a.is_code_at(idx, m.start()):
                    a.mounts[m.group(4)] = m.group(3)

    @staticmethod
    def full_mount(a: _Analysis, receiver: str) -> Optional[str]:
        if receiver not in a.mounts:
            return None
        parts: List[str] = []
        seen = set()
        cur = receiver
        while cur in a.mounts and cur not in seen and len(parts) < 6:
            seen.add(cur)
            parts.insert(0, a.mounts[cur])
            cur = _EndpointPass._mounter(a, cur)
        joined = "/".join(p.strip("/") for p in parts if p.strip("/"))
        return "/" + joined if joined else "/"

    @staticmethod
    def _mounter(a: _Analysis, child: str) -> str:
        """Nom du routeur sur lequel ``child`` est monté (recherche textuelle bornée)."""
        for idx, ml in enumerate(a.lines_for_regex):
            if ".use" not in ml or child not in ml:
                continue
            for m in _MOUNT_RE.finditer(_strip_cr(a.raw[idx])):
                if m.group(4) == child:
                    return m.group(1)
        return ""

    @staticmethod
    def server_framework_for(a: _Analysis, receiver: str, default: str = "express") -> str:
        """Framework serveur d'un récepteur : instance connue, nom du récepteur, sinon celui du fichier."""
        known = a.receivers.get(receiver)
        if known and known[0] == "server":
            return known[1]
        base = receiver.split(".")[-1]
        if base in _FRAMEWORK_RECEIVER_NAMES:
            return _FRAMEWORK_RECEIVER_NAMES[base]
        return a.server_framework or default

    @staticmethod
    def classify(a: _Analysis, receiver: str, args: List[str]) -> Tuple[str, str]:
        known = a.receivers.get(receiver)
        if known:
            return known
        base = receiver.split(".")[-1] if receiver.startswith("this.") else receiver
        if receiver.startswith("this.") or receiver.startswith("$"):
            return "client", _CLIENT_RECEIVER_NAMES.get(base, "http")
        if base in ("axios", "ky", "got", "superagent"):
            return "client", _CLIENT_RECEIVER_NAMES[base]
        if len(args) >= 2 and _HANDLER_START_RE.match(args[1]):
            return "server", _EndpointPass.server_framework_for(a, receiver)
        if base in _FRAMEWORK_RECEIVER_NAMES:
            return "server", _FRAMEWORK_RECEIVER_NAMES[base]
        if a.server_framework and base in _SERVER_RECEIVER_NAMES:
            return "server", a.server_framework
        if base in _CLIENT_RECEIVER_NAMES:
            return "client", _CLIENT_RECEIVER_NAMES[base]
        if base in _SERVER_RECEIVER_NAMES:
            return "server", a.server_framework or "express"
        return "client", "http"

    @staticmethod
    def server_handler(a: _Analysis, idx: int, args: List[str], closed: bool) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        rest = args[1:]
        refs = [r.strip() for r in rest if _HANDLER_REF_RE.match(r)]
        inline = any(_HANDLER_START_RE.match(r) for r in rest)
        if refs and (closed or not inline):
            handler = refs[-1]
            if len(refs) > 1:
                meta["middleware"] = refs[:-1]
            return handler, meta
        if refs:
            meta["middleware"] = refs
        meta["inline"] = True
        return a.enclosing_function(idx), meta

    @staticmethod
    def add(a: _Analysis, side: str, method: str, path: str, handler: str, framework: str, idx: int,
            meta: Dict[str, Any]) -> ExtractedEndpoint:
        ep = ExtractedEndpoint(side=side, method=method, path=path, handler=handler, framework=framework,
                               line=idx + 1, meta=meta)
        a.fa.endpoints.append(ep)
        return ep

    @staticmethod
    def extract(a: _Analysis) -> None:
        seen = set()
        for idx, ml in enumerate(a.lines_for_regex):
            if "(" not in ml:
                continue
            try:
                _EndpointPass._line(a, idx, ml, seen)
            except Exception as exc:  # noqa: BLE001
                a.fa.warnings.append("ligne %d : endpoint ignoré (%s)" % (idx + 1, exc))
        _EndpointPass.decorated_routes(a)
        _EndpointPass.path_derived(a)

    @staticmethod
    def _line(a: _Analysis, idx: int, ml: str, seen: set) -> None:
        raw = _strip_cr(a.raw[idx])
        window = ""
        route_hint = any(tok in ml for tok in _ROUTE_HINTS)
        for m in (_ROUTE_CALL_RE.finditer(ml) if route_hint else ()):
            if not a.is_code_at(idx, m.start()):
                continue
            receiver, verb = m.group(1), m.group(2)
            window = window or a.window(idx)
            args, closed = _split_call_args(window, m.end())
            if not args:
                continue
            side, framework = _EndpointPass.classify(a, receiver, args)
            if side == "server":
                path, dynamic = _url_from_arg(args[0])
                if path is None or dynamic or (not path.startswith("/") and "/" not in path and path != "*"):
                    continue
                handler, meta = _EndpointPass.server_handler(a, idx, args, closed)
                meta["receiver"] = receiver
                mount = _EndpointPass.full_mount(a, receiver)
                if mount:
                    meta["mount"] = mount
                key = ("server", verb, path, idx)
                if key in seen:
                    continue
                seen.add(key)
                _EndpointPass.add(a, "server", _norm_method(verb), path, handler, framework, idx, meta)
            else:
                first = args[0].strip()
                if first.startswith("{"):
                    um = _OPTION_URL_RE.search(first)
                    path, dynamic = _url_from_arg(first[um.end():]) if um else (None, True)
                    options_text = first
                else:
                    path, dynamic = _url_from_arg(first)
                    options_text = args[1] if len(args) > 1 else ""
                if path is None:
                    continue
                if not (path.startswith("/") or "/" in path or path.startswith("${")):
                    continue
                if dynamic and receiver.split(".")[-1] not in _CLIENT_RECEIVER_NAMES and receiver not in a.receivers:
                    continue
                method = _norm_method(verb)
                if method == "ANY":
                    method = _method_from_options(options_text, "GET")
                _EndpointPass._add_client(a, idx, method, path, dynamic, framework, receiver, options_text, seen)
        fetch_hint = any(tok in ml for tok in _FETCH_HINTS)
        for m in (_FETCH_CALL_RE.finditer(ml) if fetch_hint else ()):
            if not a.is_code_at(idx, m.start()):
                continue
            callee = m.group(1)
            window = window or a.window(idx)
            args, _closed = _split_call_args(window, m.end())
            if not args:
                continue
            path, dynamic = _url_from_arg(args[0])
            if path is None:
                continue
            options_text = args[1] if len(args) > 1 else ""
            method = _method_from_options(options_text, "GET")
            framework = {"useSWR": "swr", "useSWRImmutable": "swr", "ky": "ky", "got": "got",
                         "superagent": "http"}.get(callee, "fetch")
            extra: Dict[str, Any] = {"call": callee}
            context = " ".join(a.lines_for_regex[max(0, idx - 2):idx + 1])
            if "useQuery(" in context or "useMutation(" in context:
                extra["via"] = "useQuery" if "useQuery(" in context else "useMutation"
            _EndpointPass._add_client(a, idx, method, path, dynamic, framework, callee, options_text, seen, extra)
        for m in (_AXIOS_CALL_RE.finditer(ml) if "axios" in ml else ()):
            if not a.is_code_at(idx, m.start()):
                continue
            window = window or a.window(idx)
            args, _closed = _split_call_args(window, m.end())
            if not args:
                continue
            first = args[0].strip()
            if first.startswith("{"):
                um = _OPTION_URL_RE.search(first)
                if not um:
                    continue
                path, dynamic = _url_from_arg(first[um.end():])
                options_text = first
            else:
                path, dynamic = _url_from_arg(first)
                options_text = args[1] if len(args) > 1 else ""
            if path is None:
                continue
            _EndpointPass._add_client(a, idx, _method_from_options(options_text, "GET"), path, dynamic, "axios",
                                      "axios", options_text, seen, {"call": "axios"})
        if ".route" in ml:
            _EndpointPass._route_forms(a, idx, ml, raw, seen)

    @staticmethod
    def _add_client(a: _Analysis, idx: int, method: str, path: str, dynamic: bool, framework: str, receiver: str,
                    options_text: str, seen: set, extra: Optional[Dict[str, Any]] = None) -> None:
        key = ("client", method, path, idx)
        if key in seen:
            return
        seen.add(key)
        meta: Dict[str, Any] = {"receiver": receiver}
        if extra:
            meta.update(extra)
        base = _base_of(path)
        if base:
            meta["base"] = base
        else:
            bm = _OPTION_BASE_RE.search(options_text)
            if bm:
                meta["base"] = bm.group(2) or bm.group(3)
        if dynamic:
            meta["dynamic"] = True
        _EndpointPass.add(a, "client", method, path, a.enclosing_function(idx), framework, idx, meta)

    @staticmethod
    def _route_forms(a: _Analysis, idx: int, ml: str, raw: str, seen: set) -> None:
        """``app.route('/x').get(h).post(h)`` (Express) et ``server.route({ method, url|path })`` (Fastify, Hapi)."""
        for m in _ROUTE_CHAIN_RE.finditer(raw):
            if not a.is_code_at(idx, m.start()):
                continue
            receiver, path = m.group(1), m.group(3)
            chain = a.window(idx, extra=12, limit=4000)
            stop = chain.find(";")
            chain = chain if stop < 0 else chain[:stop + 1]
            for cm in _CHAIN_METHOD_RE.finditer(chain, m.end() - (len(raw) - len(_strip_cr(a.raw[idx])))):
                line_idx = idx + chain.count("\n", 0, cm.start())
                args, closed = _split_call_args(chain, cm.end())
                handler, meta = _EndpointPass.server_handler(a, line_idx, [path] + args, closed)
                meta["receiver"] = receiver
                mount = _EndpointPass.full_mount(a, receiver)
                if mount:
                    meta["mount"] = mount
                key = ("server", cm.group(1), path, line_idx)
                if key in seen:
                    continue
                seen.add(key)
                _EndpointPass.add(a, "server", _norm_method(cm.group(1)), path, handler,
                                  _EndpointPass.server_framework_for(a, receiver), line_idx, meta)
        for m in _ROUTE_OBJECT_RE.finditer(raw):
            if not a.is_code_at(idx, m.start()):
                continue
            receiver = m.group(1)
            if a.receivers.get(receiver, ("server", ""))[0] != "server":
                continue
            block = a.window(idx, extra=12, limit=4000)
            um = _OPTION_URL_RE.search(block)
            if not um:
                continue
            path, dynamic = _url_from_arg(block[um.end():])
            if path is None or dynamic:
                continue
            mm = _OPTION_METHOD_RE.search(block)
            methods = []
            if mm:
                seg = block[mm.start():mm.start() + 200]
                methods = [x.upper() for x in re.findall(r"['\"]([A-Za-z]+)['\"]", seg.split("]")[0] if "[" in seg[:12] else seg.split(",")[0])]
            hm = re.search(r"\bhandler\s*:\s*(" + _IDENT + r"(?:\.[\w$]+){0,3})\s*[,}]", block)
            handler = hm.group(1) if hm else a.enclosing_function(idx)
            meta: Dict[str, Any] = {"receiver": receiver, "route_object": True}
            if not hm:
                meta["inline"] = True
            fw = _EndpointPass.server_framework_for(a, receiver, default="fastify")
            for method in (methods or ["ANY"]):
                key = ("server", method, path, idx)
                if key in seen:
                    continue
                seen.add(key)
                _EndpointPass.add(a, "server", method, path, handler, fw, idx, dict(meta))

    @staticmethod
    def decorated_routes(a: _Analysis) -> None:
        """NestJS (et routing-controllers / tsoa) : ``@Controller('x')`` + ``@Get(':id')`` → ``Classe@méthode``."""
        methods_by_parent: Dict[str, List[ExtractedSymbol]] = {}
        for sym in a.fa.symbols:
            if sym.kind == "method" and sym.parent and sym.meta.get("decorators"):
                methods_by_parent.setdefault(sym.parent, []).append(sym)
        for cls in a.fa.symbols:
            if cls.kind not in ("class", "component") or not cls.meta.get("decorators"):
                continue
            prefix = None
            framework = "nestjs"
            for deco in cls.meta["decorators"]:
                cm = _CONTROLLER_DECORATOR_RE.match(deco)
                if cm:
                    prefix = cm.group(3) if cm.group(3) is not None else (cm.group(5) or "")
                    framework = {"Controller": "nestjs", "JsonController": "routing-controllers", "Route": "tsoa"}[cm.group(1)]
                    break
            if prefix is None:
                continue
            for sym in methods_by_parent.get(cls.name, []):
                if not (cls.line_start <= sym.line_start <= cls.line_end):
                    continue
                for deco, dline in zip(sym.meta["decorators"], sym.meta.get("decorator_lines", [])):
                    hm = _HTTP_DECORATOR_RE.match(deco)
                    if not hm:
                        continue
                    sub = hm.group(3) or ""
                    meta = {"prefix": prefix, "route": sub, "controller": cls.name, "decorator": deco}
                    _EndpointPass.add(a, "server", _norm_method(hm.group(1)), _join_route(prefix, sub),
                                      "%s@%s" % (cls.name, sym.name), framework, dline - 1, meta)

    @staticmethod
    def path_derived(a: _Analysis) -> None:
        """Routes déduites du chemin de fichier : Next.js (pages/api, app/api/**/route), SvelteKit, Nuxt."""
        p = a.path
        exported = [s for s in a.fa.symbols if s.kind in ("function", "component", "constant", "variable")
                    and s.parent is None and s.meta.get("exported")]
        m = _NEXT_APP_API_RE.search(p)
        if m:
            rel = "/".join(seg for seg in (m.group(1) or "").split("/") if seg and not seg.startswith("("))
            route = "/api" + ("/" + rel if rel else "")
            _EndpointPass._emit_exported(a, route, exported, "nextjs", {"router": "app"})
            return
        m = _NEXT_PAGES_API_RE.search(p)
        if m:
            rel = m.group(1)
            if rel == "index":
                rel = ""
            elif rel.endswith("/index"):
                rel = rel[:-6]
            route = "/api" + ("/" + rel if rel else "")
            default = [s for s in a.fa.symbols if s.parent is None and s.meta.get("default") and s.kind in ("function", "constant", "variable")]
            handler = default[0].name if default else "default"
            line = (default[0].line_start - 1) if default else 0
            _EndpointPass.add(a, "server", "ANY", route, handler, "nextjs", line,
                              {"derived_from_path": True, "file": p, "router": "pages"})
            return
        m = _SVELTEKIT_SERVER_RE.search(p)
        if m:
            rel = "/".join(seg for seg in (m.group(1) or "").split("/") if seg and not seg.startswith("("))
            _EndpointPass._emit_exported(a, "/" + rel if rel else "/", exported, "sveltekit", {})
            return
        m = _NUXT_SERVER_RE.search(p)
        if m:
            rel = m.group(2)
            rel = "" if rel == "index" else (rel[:-6] if rel.endswith("/index") else rel)
            route = ("/api/" + rel if m.group(1) == "api" else "/" + rel).rstrip("/") or "/"
            method = m.group(3).upper() if m.group(3) else "ANY"
            default = [s for s in a.fa.symbols if s.parent is None and s.meta.get("default")]
            _EndpointPass.add(a, "server", method, route, default[0].name if default else "default", "nuxt",
                              (default[0].line_start - 1) if default else 0,
                              {"derived_from_path": True, "file": p})

    @staticmethod
    def _emit_exported(a: _Analysis, route: str, exported: List[ExtractedSymbol], framework: str,
                       extra: Dict[str, Any]) -> None:
        handlers = [s for s in exported if s.name in _HTTP_EXPORT_NAMES]
        meta = {"derived_from_path": True, "file": a.path}
        meta.update(extra)
        if not handlers:
            _EndpointPass.add(a, "server", "ANY", route, "", framework, 0, dict(meta))
            return
        for s in handlers:
            _EndpointPass.add(a, "server", s.name, route, s.name, framework, s.line_start - 1, dict(meta))


# --------------------------------------------------------------------------- Vue / Svelte

_SCRIPT_OPEN_RE = re.compile(r"<script\b[^>]{0,300}>", re.I)
_SCRIPT_CLOSE_RE = re.compile(r"</script\s*>", re.I)


def extract_script_blocks(text: str) -> Tuple[str, int]:
    """Ne conserve que le contenu des blocs ``<script>`` ; les autres lignes sont vidées (numéros de ligne conservés)."""
    out: List[str] = []
    inside = False
    count = 0
    for line in text.split("\n"):
        if not inside:
            m = _SCRIPT_OPEN_RE.search(line)
            if not m:
                out.append("")
                continue
            inside = True
            count += 1
            rest = line[m.end():]
            c = _SCRIPT_CLOSE_RE.search(rest)
            if c:
                out.append(" " * m.end() + rest[:c.start()])
                inside = False
            else:
                out.append(" " * m.end() + rest)
        else:
            c = _SCRIPT_CLOSE_RE.search(line)
            if c:
                out.append(line[:c.start()])
                inside = False
            else:
                out.append(line)
    return "\n".join(out), count


# --------------------------------------------------------------------------- point d'entrée

_EXT_LANGUAGE = {
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx", ".jsx": "jsx",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".vue": "vue", ".svelte": "svelte",
}


def language_from_path(path: str) -> str:
    lower = (path or "").lower()
    for ext, lang in _EXT_LANGUAGE.items():
        if lower.endswith(ext):
            return lang
    return "javascript"


class JavaScriptExtractor:
    """Extracteur JS/TS/JSX/TSX/Vue/Svelte fondé sur des expressions régulières ligne par ligne."""

    languages: List[str] = list(LANGUAGES)

    def analyze(self, text: str, path: str) -> FileAnalysis:
        language = language_from_path(path)
        if text is None:
            text = ""
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        if not text.strip():
            return FileAnalysis(language=language)
        warnings: List[str] = []
        try:
            if language in ("vue", "svelte"):
                text, count = extract_script_blocks(text)
                if count == 0:
                    fa = FileAnalysis(language=language)
                    fa.warnings.append("aucun bloc <script> dans %s" % (path or "le fichier"))
                    return fa
            analysis = _Analysis(text, path or "", language)
        except Exception as exc:  # noqa: BLE001
            fa = FileAnalysis(language=language)
            fa.warnings.append("échec du prétraitement de %s : %s" % (path, exc))
            return fa
        fa = analysis.fa
        steps = (
            ("symboles", analysis.extract_symbols),
            ("exports", analysis.apply_late_exports),
            ("imports", analysis.extract_imports),
            ("récepteurs HTTP", lambda: _EndpointPass.scan_receivers(analysis)),
            ("montages de routeurs", lambda: _EndpointPass.scan_mounts(analysis)),
            ("endpoints", lambda: _EndpointPass.extract(analysis)),
        )
        for label, step in steps:
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - résultat partiel plutôt qu'aucun résultat
                warnings.append("%s : analyse partielle de %s (%s)" % (label, path, exc))
        fa.warnings.extend(warnings)
        return fa


EXTRACTOR = JavaScriptExtractor()
