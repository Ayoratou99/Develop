"""Extracteur Java / Kotlin (Spring, JPA, JAX-RS, Micronaut, Feign, Retrofit, Ktor, clients HTTP).

Analyse par expressions régulières ligne par ligne, avec équilibrage des accolades hors chaînes et
commentaires. Aucune dépendance externe. Objectif : une carte citable (fichier:ligne) des symboles,
endpoints, modèles de données et imports, jamais une analyse exhaustive. Toute entrée bizarre
(vide, tronquée, minifiée, invalide) produit au pire des avertissements, jamais d'exception.
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
    ModelRelation,
)

# --------------------------------------------------------------------------- limites de sécurité
_MAX_DECL_LINE = 2000  # longueur max d'une ligne soumise aux regex de déclaration
_MAX_PENDING_LINES = 80  # lignes max pour une signature multi-lignes
_MAX_CTOR_LINES = 200  # lignes max pour un constructeur primaire Kotlin / record Java
_MAX_WINDOW_LINES = 12  # lignes max d'une instruction client HTTP (chaînage de builder)
_MAX_ANN_ARGS = 3000  # caractères max des arguments d'une annotation
_LONG_LINE = 5000  # au-delà : avertissement "fichier minifié ?"
_MAX_SEGMENTS = 64  # instructions max analysées sur une même ligne (au-delà : seules les accolades comptent)
_SNIFF_LINES = 400  # lignes examinées pour deviner Java/Kotlin quand l'extension n'aide pas

# --------------------------------------------------------------------------- pré-traitement
# Chaînes (texte brut """...""" d'abord), caractères, commentaires. Toutes les alternatives sont bornées
# (pas de .* gourmand) : une chaîne ne traverse pas une fin de ligne, un bloc """ ou /* non fermé
# n'échoue qu'une seule fois (balayage linéaire).
_LEXEME = re.compile(
    r'"""[\s\S]*?"""'
    r'|"(?:\\.|[^"\\\n])*"'
    r"|'(?:\\.|[^'\\\n])*'"
    r"|/\*[\s\S]*?\*/"
    r"|//[^\n]*"
)
_NON_NL = re.compile(r"[^\n]")


def _blank(s: str) -> str:
    """Remplace tout sauf les retours à la ligne par des espaces (préserve les numéros de ligne)."""
    return _NON_NL.sub(" ", s)


def _preprocess(text: str) -> Tuple[str, str]:
    """Renvoie (code, masked) : ``code`` = texte sans commentaires (chaînes conservées), ``masked`` =
    idem avec le contenu des chaînes/caractères effacé (pour compter accolades et parenthèses)."""
    code_parts: List[str] = []
    masked_parts: List[str] = []
    pos = 0
    for m in _LEXEME.finditer(text):
        seg = text[pos:m.start()]
        code_parts.append(seg)
        masked_parts.append(seg)
        tok = m.group(0)
        if tok.startswith("/"):
            blank = _blank(tok)
            code_parts.append(blank)
            masked_parts.append(blank)
        else:
            code_parts.append(tok)
            if len(tok) > 2:
                masked_parts.append(tok[0] + _blank(tok[1:-1]) + tok[-1])
            else:
                masked_parts.append(tok)
        pos = m.end()
    tail = text[pos:]
    code_parts.append(tail)
    masked_parts.append(tail)
    return "".join(code_parts), "".join(masked_parts)


# --------------------------------------------------------------------------- annotations
_ANN_NAME = re.compile(r"@(?:\w+\s*:\s*)?([\w.]+)")  # @Column, @field:Column, @javax.persistence.Id
_STR_LIT = re.compile(r'"((?:\\.|[^"\\\n]){0,500})"')
_KW_ARG = re.compile(r"^\s*(\w+)\s*=(?!=)\s*([\s\S]*)$")


class _Ann:
    """Annotation analysée : nom court, arguments positionnels et nommés (texte brut)."""

    __slots__ = ("name", "positional", "kwargs", "line")

    def __init__(self, name: str, line: int) -> None:
        self.name = name.rsplit(".", 1)[-1]
        self.positional: List[str] = []
        self.kwargs: Dict[str, str] = {}
        self.line = line

    def strings(self, *keys: str) -> List[str]:
        """Chaînes littérales des arguments ``keys`` (ou positionnels si ``keys`` est vide ou absent)."""
        raw: List[str] = []
        for k in keys:
            if k in self.kwargs:
                raw.append(self.kwargs[k])
        if not raw and (not keys or True):
            raw.extend(self.positional)
        out: List[str] = []
        for r in raw:
            out.extend(s for s in _STR_LIT.findall(r))
        return out

    def value(self, key: str) -> Optional[str]:
        v = self.kwargs.get(key)
        return v.strip() if v is not None else None


_RE_SPLIT_TOK = re.compile(r'"(?:\\.|[^"\\\n])*"|[()\[\]{}]|,|[^"()\[\]{},]+')


def _split_top_level(s: str) -> List[str]:
    """Découpe sur les virgules de premier niveau (hors parenthèses/accolades/crochets/chaînes)."""
    parts: List[str] = []
    depth = 0
    buf: List[str] = []
    for m in _RE_SPLIT_TOK.finditer(s):
        tok = m.group(0)
        if tok == ",":
            if depth <= 0:
                parts.append("".join(buf))
                buf = []
                continue
        elif tok in "([{":
            depth += 1
        elif tok in ")]}":
            depth -= 1
        buf.append(tok)
    if buf:
        parts.append("".join(buf))
    return [p for p in parts if p.strip()]


_RE_PAREN_TOK = re.compile(r'"(?:\\.|[^"\\\n])*"|[()]')


def _matching_paren(s: str, start: int, limit: int = _MAX_ANN_ARGS) -> int:
    """Index de la parenthèse fermante correspondant à ``s[start] == '('`` (chaînes ignorées), -1 sinon."""
    depth = 0
    end = min(len(s), start + limit)
    for m in _RE_PAREN_TOK.finditer(s, start, end):
        tok = m.group(0)
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
            if depth == 0:
                return m.start()
    return -1


def _parse_annotations(text: str, first_line: int) -> List[_Ann]:
    """Extrait les annotations d'un texte (une ou plusieurs lignes, commentaires déjà retirés)."""
    anns: List[_Ann] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _ANN_NAME.search(text, pos)
        if not m:
            break
        if m.group(1) == "interface":  # déclaration @interface, pas une annotation
            pos = m.end()
            continue
        line = first_line + text.count("\n", 0, m.start())
        ann = _Ann(m.group(1), line)
        j = m.end()
        while j < n and text[j] in " \t":
            j += 1
        if j < n and text[j] == "(":
            close = _matching_paren(text, j)
            if close < 0:
                close = min(n, j + _MAX_ANN_ARGS)
            inner = text[j + 1:close]
            for part in _split_top_level(inner):
                km = _KW_ARG.match(part)
                if km:
                    ann.kwargs[km.group(1)] = km.group(2).strip()
                else:
                    ann.positional.append(part.strip())
            pos = close + 1
        else:
            pos = m.end()
        anns.append(ann)
    return anns


def _strip_leading_annotations(line: str) -> Tuple[str, str]:
    """Sépare les annotations en tête de ligne du reste (``@Override public void x()`` → ("@Override ", "public void x()"))."""
    i = 0
    n = len(line)
    while True:
        j = i
        while j < n and line[j] in " \t":
            j += 1
        if j >= n or line[j] != "@":
            break
        m = _ANN_NAME.match(line, j)
        if not m or m.group(1) == "interface":
            break
        k = m.end()
        while k < n and line[k] in " \t":
            k += 1
        if k < n and line[k] == "(":
            close = _matching_paren(line, k)
            if close < 0:
                return line[: n], ""  # annotation ouverte : pas de déclaration sur cette ligne
            k = close + 1
        i = k
    return line[:i], line[i:]


_INLINE_ANN = re.compile(r"@(?:\w+\s*:\s*)?[\w.]+(?:\((?:[^()]|\([^()]*\))*\))?")


def _remove_inline_annotations(decl: str) -> str:
    """Retire les annotations intercalées (``public @Nullable String x()``), bornées."""
    if "@" not in decl:
        return decl
    return _INLINE_ANN.sub("", decl)


# --------------------------------------------------------------------------- regex de déclaration
_JAVA_MODS = r"(?:(?:public|private|protected|static|final|abstract|sealed|non-sealed|strictfp|synchronized|native|default|transient|volatile)\s+)*"
_KT_MODS = (
    r"(?:(?:public|private|internal|protected|open|abstract|final|sealed|data|enum|annotation|inner|value|inline|"
    r"companion|override|suspend|operator|infix|tailrec|external|actual|expect|const|lateinit|fun|vararg)\s+)*"
)
_GENERIC = r"<(?:[^<>;]|<(?:[^<>;]|<[^<>;]*>)*>)*>"  # 3 niveaux max, alternatives disjointes
_JAVA_TYPE = r"[\w.$]+(?:\s*" + _GENERIC + r")?(?:\s*\[\s*\])*"
_KT_TYPE = r"[\w.]+(?:\s*" + _GENERIC + r")?\??(?:\s*\.\s*[\w.]+(?:" + _GENERIC + r")?\??)*"

_RE_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;?\s*$")
_RE_IMPORT_JAVA = re.compile(r"^\s*import\s+(static\s+)?([\w.]+(?:\.\*)?)\s*;")
_RE_IMPORT_KT = re.compile(r"^\s*import\s+([\w.`]+(?:\.\*)?)(?:\s+as\s+(\w+))?\s*;?\s*$")
_RE_TYPE_JAVA = re.compile(
    r"^\s*(" + _JAVA_MODS + r")(class|interface|enum|record|@interface)\s+(\w+)"
)
_RE_TYPE_KT = re.compile(r"^\s*(" + _KT_MODS + r")(class|interface|object)\b\s*(\w+)?")
_RE_METHOD_JAVA = re.compile(
    r"^\s*(" + _JAVA_MODS + r")(?:" + _GENERIC + r"\s*)?(" + _JAVA_TYPE + r")\s+(\w+)\s*\("
)
_RE_CTOR_JAVA = re.compile(r"^\s*(" + _JAVA_MODS + r")(\w+)\s*\(")
_RE_FUN_KT = re.compile(
    r"^\s*(" + _KT_MODS + r")fun\s+(?:" + _GENERIC + r"\s*)?(?:(" + _KT_TYPE + r")\s*\.\s*)?(\w+|`[^`\n]{1,100}`)\s*\("
)
_RE_CTOR_KT = re.compile(r"^\s*(" + _KT_MODS + r")constructor\s*\(")
_RE_FIELD_JAVA = re.compile(
    r"^\s*(" + _JAVA_MODS + r")(" + _JAVA_TYPE + r")\s+(\w+)\s*(?:\[\s*\]\s*)*(=|;|,)"
)
_RE_PROP_KT = re.compile(
    r"^\s*(" + _KT_MODS + r")(val|var)\s+(?:(" + _KT_TYPE + r")\s*\.\s*)?(\w+|`[^`\n]{1,100}`)\s*"
    r"(?::\s*(" + _KT_TYPE + r"))?\s*(?:(=)\s*(.{0,200}?))?\s*;?\s*$"
)
# Indices Java/Kotlin pour un fichier sans extension reconnue (bornés, une ligne à la fois).
_RE_KT_HINT = re.compile(
    r"^\s*(?:(?:public|private|internal|protected|open|suspend|override|inline|const|lateinit|abstract|final|"
    r"companion|data|sealed|value|inner|enum|annotation)\s+)*"
    r"(?:fun|val|var|object|typealias|(?:data|sealed|value|inner|annotation|enum)\s+class)\b"
)
_RE_SEMI_END = re.compile(r";\s*$")
_RE_TYPEALIAS_KT = re.compile(r"^\s*(" + _KT_MODS + r")typealias\s+(\w+)")
_RE_ENUM_CONST = re.compile(r"([A-Z_][A-Z0-9_]*)\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*(?=,|;|\{|$)")
_RE_ENUM_LINE = re.compile(r"^\s*[A-Z_][A-Z0-9_]*\s*(?:\(|,|;|\{|$)")
_RE_KT_EXPR_BODY = re.compile(r"^\s*(?::\s*[\w.<>?,\s]{0,200}?)?=(?!=)")
_RE_ROUTE_KT = re.compile(r"^\s*route\s*\(\s*\"([^\"\n]{0,300})\"")
_RE_KTOR_VERB = re.compile(
    r"^\s*(get|post|put|patch|delete|head|options)\s*(?:<[^>\n]{0,100}>)?\s*(?:\(\s*(?:\"([^\"\n]{0,300})\")?\s*\))?\s*\{"
)
_RE_KTOR_ROUTING = re.compile(r"^\s*(?:routing|install\s*\(\s*Routing\s*\))\s*\{|\bRoute\s*\.\s*\w+\s*\(")

_NOT_A_TYPE = frozenset(
    "return throw new case else goto package import break continue assert yield throws extends implements "
    "if for while switch do try catch finally super this instanceof var "
    "public private protected static final abstract sealed non-sealed strictfp synchronized native default "
    "transient volatile".split()
)
_NOT_A_NAME = frozenset(
    "if for while switch catch synchronized return new else try do super this throw".split()
)
_KT_NOT_A_NAME = frozenset("if for while when catch return else try do super this throw".split())

_SPRING_MAPPINGS = {
    "GetMapping": "GET", "PostMapping": "POST", "PutMapping": "PUT", "PatchMapping": "PATCH",
    "DeleteMapping": "DELETE", "RequestMapping": "ANY",
}
_JAXRS_VERBS = frozenset(["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
_MICRONAUT_VERBS = {"Get": "GET", "Post": "POST", "Put": "PUT", "Patch": "PATCH", "Delete": "DELETE",
                    "Head": "HEAD", "Options": "OPTIONS", "Trace": "TRACE"}
_JPA_RELATIONS = {"ManyToOne": "belongs_to", "OneToMany": "has_many", "OneToOne": "has_one",
                  "ManyToMany": "many_to_many"}
_RE_REQUEST_METHOD = re.compile(r"(?:RequestMethod\s*\.\s*)?\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE)\b")

# --------------------------------------------------------------------------- petits utilitaires
_RE_SNAKE1 = re.compile(r"(.)([A-Z][a-z]+)")
_RE_SNAKE2 = re.compile(r"([a-z0-9])([A-Z])")
_RE_COLLECTION = re.compile(r"^(?:java\.util\.)?(?:List|Set|Collection|Iterable|MutableList|MutableSet|ArrayList|LinkedList|HashSet|Optional|Flux|Mono)\s*<\s*([\w.]+)")
_RE_WS = re.compile(r"\s+")


def _snake_case(name: str) -> str:
    return _RE_SNAKE2.sub(r"\1_\2", _RE_SNAKE1.sub(r"\1_\2", name)).lower()


def _element_type(type_str: str) -> str:
    """``List<OrderLine>`` → ``OrderLine`` ; ``Customer`` → ``Customer`` ; ``Set<Tag>?`` → ``Tag``."""
    t = type_str.strip().rstrip("?").strip()
    m = _RE_COLLECTION.match(t)
    if m:
        return m.group(1).rsplit(".", 1)[-1]
    return t.rsplit("<", 1)[0].rsplit(".", 1)[-1] if "<" in t else t.rsplit(".", 1)[-1]


def _collapse(s: str, limit: int = 300) -> str:
    s = _RE_WS.sub(" ", s).strip()
    return s if len(s) <= limit else s[:limit] + "…"


def _modifiers(group: Optional[str]) -> List[str]:
    return group.split() if group else []


def _signature(parts: List[str]) -> str:
    """Signature lisible : annotations de tête retirées, espaces compactés, longueur bornée."""
    if not parts:
        return ""
    first = _strip_leading_annotations(parts[0])[1]
    return _collapse(" ".join([first] + list(parts[1:])))


def _stmt_end(ml: str, semicolons: bool = True) -> int:
    """Index du ``;`` de premier niveau (si ``semicolons``) ou du ``}`` fermant un bloc englobant qui termine
    la première instruction de ``ml`` (ligne masquée) ; -1 si l'instruction court jusqu'à la fin de la ligne.
    Permet de traiter plusieurs déclarations écrites sur une même ligne (``class A { int x; void m() {} }``)."""
    paren = 0
    brace = 0
    for m in _RE_BRACKETS.finditer(ml):
        c = m.group(0)
        if c == "(":
            paren += 1
        elif c == ")":
            paren -= 1
        elif c == "{":
            brace += 1
        elif c == "}":
            if brace <= 0:
                return m.start()
            brace -= 1
        elif semicolons and paren <= 0 and brace <= 0:  # ``;`` de premier niveau
            return m.start()
    return -1


def _sniff_language(text: str) -> str:
    """Devine Java ou Kotlin d'après le contenu (extension absente ou inconnue) : des ``fun``/``val``/``var``
    sans ``;`` final penchent pour Kotlin, des lignes terminées par ``;`` pour Java."""
    kt = java = 0
    for line in text.split("\n", _SNIFF_LINES)[:_SNIFF_LINES]:
        if len(line) > _MAX_DECL_LINE:
            line = line[:_MAX_DECL_LINE]
        if _RE_SEMI_END.search(line):
            java += 1
        elif _RE_KT_HINT.match(line):
            kt += 1
    return "kotlin" if kt > java else "java"


def _join_path(prefix: str, path: str) -> str:
    """Concatène préfixe de classe et chemin de méthode en respectant les ``/``."""
    if not prefix:
        return path or "/"
    if not path:
        return prefix
    return prefix.rstrip("/") + "/" + path.lstrip("/")


_RE_URL_TOKEN = re.compile(r'"((?:\\.|[^"\\\n]){0,500})"|([A-Za-z_$][\w$.]*(?:\(\s*\))?)')
_RE_KT_TEMPLATE_BASE = re.compile(r"^\$(?:\{\s*([\w.]+)[^}]*\}|([\w.]+))")
_RE_FORMAT_SPEC = re.compile(r"%[0-9.]*[sdif]")


def _first_argument(s: str, open_idx: int) -> str:
    """Texte du premier argument de l'appel dont la parenthèse ouvrante est ``s[open_idx]`` (borné)."""
    depth = 0
    in_str = False
    i = open_idx
    end = min(len(s), open_idx + 600)
    buf: List[str] = []
    while i < end:
        c = s[i]
        if in_str:
            buf.append(c)
            if c == "\\" and i + 1 < end:
                buf.append(s[i + 1])
                i += 2
                continue
            if c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            buf.append(c)
        elif c in "([{":
            depth += 1
            if depth > 1:
                buf.append(c)
        elif c in ")]}":
            depth -= 1
            if depth <= 0:
                break
            buf.append(c)
        elif c == "," and depth == 1:
            break
        else:
            buf.append(c)
        i += 1
    return "".join(buf).strip()


def _url_from_expr(expr: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Reconstruit un chemin à partir d'une expression d'URL : ``baseUrl + "/orders/" + id`` →
    (``${baseUrl}/orders/${id}``, ``baseUrl``, False). Renvoie (path, base, dynamic)."""
    expr = expr.strip()
    if not expr:
        return None, None, False
    inner = re.match(r"^(?:java\.net\.)?(?:URI\.create|new\s+URI|URI|Uri\.parse|URL|new\s+URL)\s*\((.*)\)\s*$", expr, re.S)
    if inner:
        expr = inner.group(1).strip()
    fmt = re.match(r"^(?:String|MessageFormat)\.format\s*\(", expr)
    if fmt:
        lit = _STR_LIT.search(expr)
        if not lit:
            return None, None, False
        s = lit.group(1)
        tb = _RE_KT_TEMPLATE_BASE.match(s)
        return _RE_FORMAT_SPEC.sub("{param}", re.sub(r"\{\d+\}", "{param}", s)), (tb.group(1) or tb.group(2)) if tb else None, False
    parts: List[str] = []
    base: Optional[str] = None
    has_literal = False
    for m in _RE_URL_TOKEN.finditer(expr):
        if m.group(1) is not None:
            lit = m.group(1)
            has_literal = True
            if not parts:
                tb = _RE_KT_TEMPLATE_BASE.match(lit)
                if tb:
                    base = tb.group(1) or tb.group(2)
            parts.append(_RE_FORMAT_SPEC.sub("{param}", lit))
        else:
            ident = m.group(2)
            if ident in ("String", "format", "URI", "create", "new", "toString", "class", "java", "lang", "net"):
                continue
            if not parts and base is None:
                base = ident.rstrip("()")
            parts.append("${" + ident.rstrip("()") + "}")
    if not parts:
        return None, None, False
    return "".join(parts), base, not has_literal


# --------------------------------------------------------------------------- état d'analyse
class _Block:
    """Un bloc ``{ ... }`` ouvert : porte éventuellement un symbole (type/méthode) et son contexte."""

    __slots__ = ("symbol", "kind", "annotations", "model", "prefixes", "server_framework", "feign",
                 "route_prefix", "ktor_routing", "enum_done", "is_enum", "is_interface", "ctor_done")

    def __init__(self, symbol: Optional[ExtractedSymbol] = None, kind: str = "other") -> None:
        self.symbol = symbol
        self.kind = kind  # 'type' | 'method' | 'other'
        self.annotations: List[_Ann] = []
        self.model: Optional[ExtractedModel] = None
        self.prefixes: List[str] = [""]
        self.server_framework: Optional[str] = None
        self.feign: Optional[Dict[str, Any]] = None
        self.route_prefix: str = ""
        self.ktor_routing: bool = False
        self.enum_done: bool = False
        self.is_enum: bool = False
        self.is_interface: bool = False
        self.ctor_done: bool = False


class _Pending:
    """Déclaration dont le corps (``{``) ou la fin (``;``) n'est pas encore atteint."""

    __slots__ = ("block", "paren", "sig_parts", "start_line", "last_line", "ready", "is_type", "language")

    def __init__(self, block: _Block, paren: int, first_sig: str, line: int, language: str) -> None:
        self.block = block
        self.paren = paren
        self.sig_parts: List[str] = [first_sig]
        self.start_line = line
        self.last_line = line
        self.ready = False  # Kotlin : en-tête complet, un ``{`` peut encore suivre sur la ligne suivante
        self.is_type = block.kind == "type"
        self.language = language


_RE_BRACKETS = re.compile(r"[{}();]")
_RE_PARENS = re.compile(r"[()]")
_KT_CONTINUATION_ENDS = (",", ":", "(", "<", ".", "->", "=", "where", "&&", "||", "+")
_RE_BASE_HINT = re.compile(
    r"(?:\.baseUrl|\.rootUri|WebClient\.create|RestClient\.create|HttpClient\s*\(|\.url)\s*\(\s*\"((?:https?://|/)[^\"\n]{0,300})\""
)


class _ScanResult:
    __slots__ = ("attached", "terminated", "paren", "cut", "balance_idx")

    def __init__(self) -> None:
        self.attached = False
        self.terminated = False
        self.paren = 0
        self.cut = -1  # index du ``{`` ou ``;`` qui clôt la signature
        self.balance_idx = -1  # index juste après la ``)`` qui ramène la profondeur de parenthèses à 0


class _Analyzer:
    """Analyse d'un fichier : une instance par appel (état de pile, annotations, déclarations en attente)."""

    def __init__(self, text: str, path: str, language: str, fa: FileAnalysis) -> None:
        self.text = text
        self.path = path
        self.lang = language
        self.kt = language == "kotlin"
        self.fa = fa
        self.stack: List[_Block] = []
        self.pending: Optional[_Pending] = None
        self.ann_buf: List[str] = []
        self.ann_first_line = 0
        self.ann_paren = 0
        self.package: Optional[str] = None
        self.base_hint: Optional[str] = None
        self.consumed: Dict[str, int] = {}  # framework client -> dernière ligne consommée par une fenêtre
        self.expr_fun: Optional[Tuple[int, ExtractedSymbol]] = None  # fonction à corps-expression de la ligne courante
        self.stray_close = 0
        self.code_lines: List[str] = []
        self.masked_lines: List[str] = []

    # ------------------------------------------------------------------ pilotage
    def run(self) -> None:
        text = self.text
        if not text or not text.strip():
            self.fa.warnings.append("fichier vide")
            return
        if "\x00" in text:
            self.fa.warnings.append("contenu binaire ou encodage inattendu : fichier ignoré")
            return
        code, masked = _preprocess(text)
        self.code_lines = code.split("\n")
        self.masked_lines = masked.split("\n")
        if len(self.code_lines) != len(self.masked_lines):  # ne devrait jamais arriver
            self.fa.warnings.append("pré-traitement incohérent : analyse abandonnée")
            return
        if max(len(line) for line in self.code_lines) > _LONG_LINE:
            self.fa.warnings.append(
                "ligne de plus de %d caractères (fichier minifié ?) : déclarations analysées partiellement" % _LONG_LINE
            )
        bh = _RE_BASE_HINT.search(code)
        self.base_hint = bh.group(1) if bh else None
        for idx, (cl, ml) in enumerate(zip(self.code_lines, self.masked_lines)):
            lineno = idx + 1
            try:
                self._process_line(lineno, cl, ml)
            except Exception as exc:  # noqa: BLE001 - une ligne exotique ne doit pas arrêter l'analyse
                self.fa.warnings.append("ligne %d : erreur interne ignorée (%s)" % (lineno, exc))
        self._finish(len(self.code_lines))

    def _finish(self, nlines: int) -> None:
        p = self.pending
        if p is not None:
            sym = p.block.symbol
            if sym is not None:
                sym.line_end = max(sym.line_start, p.last_line)
                sym.signature = _signature(p.sig_parts)
            if not p.ready:
                self.fa.warnings.append("déclaration incomplète à la ligne %d (fichier tronqué ?)" % p.start_line)
            self.pending = None
        if self.stack:
            open_syms = 0
            for blk in self.stack:
                if blk.symbol is not None:
                    blk.symbol.line_end = max(blk.symbol.line_start, nlines)
                    open_syms += 1
            self.fa.warnings.append("%d accolade(s) non fermée(s) en fin de fichier" % len(self.stack))
            self.stack = []
        if self.stray_close:
            self.fa.warnings.append("%d accolade(s) fermante(s) en trop" % self.stray_close)

    # ------------------------------------------------------------------ pile de blocs
    def _type_block(self) -> Optional[_Block]:
        for blk in reversed(self.stack):
            if blk.kind == "type":
                return blk
        return None

    def _enclosing_handler(self, lineno: int = 0) -> str:
        if self.expr_fun is not None and self.expr_fun[0] == lineno:
            s = self.expr_fun[1]
            return "%s@%s" % (s.parent, s.name) if s.parent else s.name
        for blk in reversed(self.stack):
            if blk.kind == "method" and blk.symbol is not None:
                s = blk.symbol
                return "%s@%s" % (s.parent, s.name) if s.parent else s.name
        tb = self._type_block()
        return tb.symbol.name if tb is not None and tb.symbol is not None else ""

    def _close_block(self, blk: _Block, lineno: int) -> None:
        if blk.symbol is not None:
            blk.symbol.line_end = lineno

    def _scan_braces(self, lineno: int, ml: str, attach: Optional[_Block], paren: int = 0,
                     end: int = -1) -> _ScanResult:
        """Parcourt les ``{ } ( ) ;`` d'une ligne masquée (jusqu'à ``end`` si fourni). Le premier ``{``
        hors parenthèses est associé à ``attach`` (ouverture du corps) ; un ``;`` hors parenthèses avant
        tout ``{`` termine la déclaration. Dans les deux cas le balayage s'arrête à cette coupure
        (``res.cut``) : le reste de la ligne est traité ensuite comme un nouveau segment."""
        res = _ScanResult()
        res.paren = paren
        opened = paren > 0
        if end < 0:
            end = len(ml)
        for m in _RE_BRACKETS.finditer(ml, 0, end):
            c = m.group(0)
            if c == "(":
                res.paren += 1
                opened = True
            elif c == ")":
                res.paren -= 1
                if opened and res.paren <= 0 and res.balance_idx < 0:
                    res.balance_idx = m.end()
            elif c == "{":
                if attach is not None and res.paren <= 0:
                    self.stack.append(attach)
                    res.attached = True
                    res.cut = m.start()
                    return res
                self.stack.append(_Block())
            elif c == "}":
                if self.stack:
                    self._close_block(self.stack.pop(), lineno)
                else:
                    self.stray_close += 1
            elif c == ";":
                if attach is not None and res.paren <= 0:
                    res.terminated = True
                    res.cut = m.start()
                    return res
        return res

    # ------------------------------------------------------------------ traitement d'une ligne
    def _process_line(self, lineno: int, cl: str, ml: str) -> None:
        """Une ligne = une déclaration en attente à poursuivre, puis une suite de segments (instructions)."""
        offset = 0
        if self.pending is not None:
            consumed, offset, calls = self._continue_pending(lineno, cl, ml)
            if consumed and offset < 0:  # ligne entièrement absorbée par la déclaration en attente
                if calls:
                    self._client_calls(lineno, cl)
                return
            if not consumed:
                offset = 0
        self._process_segments(lineno, cl, ml, offset)
        self._client_calls(lineno, cl)

    def _process_segments(self, lineno: int, cl: str, ml: str, offset: int) -> None:
        """Traite la ligne à partir de ``offset`` segment par segment (``class A { int x; void m() {} }``).
        Au-delà de ``_MAX_SEGMENTS`` (ligne minifiée), seules les accolades du reste sont comptées."""
        count = 0
        n = len(cl)
        while offset < n:
            seg_cl = cl[offset:] if offset else cl
            seg_ml = ml[offset:] if offset else ml
            if count >= _MAX_SEGMENTS:
                self._scan_braces(lineno, seg_ml, None)
                return
            cut = self._process_segment(lineno, seg_cl, seg_ml, offset)
            if cut <= 0:
                return
            offset += cut
            count += 1

    def _process_segment(self, lineno: int, cl: str, ml: str, offset: int) -> int:
        """Traite un segment de ligne (annotations, déclaration, accolades). Renvoie la longueur consommée
        si le reste de la ligne doit être traité comme un nouveau segment, -1 si tout a été consommé."""
        stripped = cl.strip()
        if self.ann_paren > 0:  # annotation multi-lignes en cours
            self.ann_buf.append(cl[:_MAX_DECL_LINE])
            self.ann_paren += ml.count("(") - ml.count(")")
            self._scan_braces(lineno, ml, None)
            return -1
        if not stripped:
            return -1
        cl_decl = cl[:_MAX_DECL_LINE]
        ann_head = ""
        rest = cl_decl
        if stripped.startswith("@") and not stripped.startswith("@interface"):
            ann_head, rest = _strip_leading_annotations(cl_decl)
            if not rest.strip():  # ligne(s) d'annotations seules
                if not self.ann_buf:
                    self.ann_first_line = lineno
                self.ann_buf.append(cl_decl)
                self.ann_paren = max(0, ml.count("(") - ml.count(")"))
                self._scan_braces(lineno, ml, None)
                return -1
        anns: List[_Ann] = []
        if self.ann_buf:
            anns.extend(_parse_annotations("\n".join(self.ann_buf), self.ann_first_line))
        if ann_head:
            anns.extend(_parse_annotations(ann_head, lineno))
        self.ann_buf = []
        self.ann_paren = 0
        # Fin de la première instruction du segment. Les ``;`` ne séparent des déclarations que dans un corps
        # de type ou au niveau fichier ; dans un corps de méthode seul le ``}`` qui le ferme coupe la ligne
        # (``int get() { return x; } void m() {}``).
        top = self.stack[-1] if self.stack else None
        stmt_end = _stmt_end(ml, semicolons=(top is None or top.kind == "type"))
        if stmt_end >= 0:
            decl_end = stmt_end + 1 if ml[stmt_end] == ";" else stmt_end  # le ``;`` reste (regex de champ)
            rest = rest[:max(0, decl_end - len(ann_head))]
        decl = rest
        if "@" in rest and not rest.lstrip().startswith("@interface"):
            decl = _remove_inline_annotations(rest)
        attach = self._handle_declaration(lineno, decl, ml, anns, offset)
        if attach is not None:
            res = self._scan_braces(lineno, ml, attach)
            self._settle(lineno, attach, [cl_decl], res, ml, stripped)
            return res.cut + 1 if res.cut >= 0 else -1
        if stmt_end >= 0:
            self._scan_braces(lineno, ml, None, end=stmt_end + 1)
            return stmt_end + 1
        self._scan_braces(lineno, ml, None)
        return -1

    def _continue_pending(self, lineno: int, cl: str, ml: str) -> Tuple[bool, int, bool]:
        """Poursuit une déclaration multi-lignes. Renvoie (consommée, offset du reste de la ligne à traiter
        en segments ou -1, appels HTTP clients à chercher sur la ligne)."""
        p = self.pending
        assert p is not None
        stripped = cl.strip()
        if p.ready:  # Kotlin : en-tête complet ; seul un ``{`` en début de ligne peut encore ouvrir le corps
            if stripped.startswith("{"):
                res = self._scan_braces(lineno, ml, p.block)
                self.pending = None
                return True, (res.cut + 1 if res.attached else -1), res.attached
            self._finalize_pending(p.last_line)
            return False, 0, True
        if not stripped:
            return True, -1, False
        if lineno - p.start_line > _MAX_PENDING_LINES:
            self._finalize_pending(p.start_line)
            self.fa.warnings.append("déclaration incomplète à la ligne %d (signature trop longue ?)" % p.start_line)
            return False, 0, True
        p.last_line = lineno
        cl_decl = cl[:_MAX_DECL_LINE]
        res = self._scan_braces(lineno, ml, p.block, paren=p.paren)
        p.paren = res.paren
        p.sig_parts.append(cl_decl)
        self.pending = None
        self._settle(lineno, p.block, p.sig_parts, res, ml, stripped, resumed=p)
        if res.attached or res.terminated:
            return True, res.cut + 1, True
        if self.pending is None:  # corps-expression Kotlin : la ligne entière est le corps
            return True, -1, True
        return True, -1, False

    def _finalize_pending(self, line_end: int) -> None:
        p = self.pending
        if p is None:
            return
        sym = p.block.symbol
        if sym is not None:
            sym.line_end = max(sym.line_start, line_end)
            sym.signature = _signature(p.sig_parts)
        self.pending = None

    def _settle(self, lineno: int, block: _Block, sig_parts: List[str], res: _ScanResult, ml: str,
                stripped: str, resumed: Optional[_Pending] = None) -> None:
        """Décide, après le balayage d'une ligne, si la déclaration est ouverte (``{``), terminée (``;`` ou
        corps-expression Kotlin) ou toujours en attente."""
        sym = block.symbol
        last = sig_parts[-1]
        head = " ".join(sig_parts[:-1])
        expr_cut = -1  # Kotlin : position du ``=`` d'un corps-expression sur la dernière ligne
        if self.kt and res.balance_idx >= 0 and block.kind == "method":
            rem = ml[res.balance_idx:]
            em = _RE_KT_EXPR_BODY.match(rem)
            if em:
                expr_cut = res.balance_idx + em.end() - 1
        if res.attached or res.terminated:
            cut = res.cut if res.cut >= 0 else len(last)
            if 0 <= expr_cut < cut:  # ``fun x() = run { ... }`` : le bloc est le lambda, la signature s'arrête au ``=``
                cut = expr_cut
                if sym is not None:
                    sym.meta["expression_body"] = True
            if sym is not None:
                sym.signature = _signature(sig_parts[:-1] + [last[:cut]])
                if res.terminated:
                    sym.line_end = lineno
            return
        paren = res.paren
        if self.kt and paren <= 0 and (expr_cut >= 0 or (stripped.endswith("=") and block.kind == "method")):
            if sym is not None:
                cut = expr_cut if expr_cut >= 0 else len(last.rstrip().rstrip("="))
                sym.signature = _signature(sig_parts[:-1] + [last[:cut]])
                sym.line_end = lineno
                sym.meta["expression_body"] = True
                self.expr_fun = (lineno, sym)
            return
        if resumed is not None:
            p = resumed
            p.paren = paren
        else:
            p = _Pending(block, paren, last, lineno, self.lang)
        p.last_line = lineno
        if self.kt and paren <= 0 and not stripped.endswith(_KT_CONTINUATION_ENDS):
            p.ready = True
            if sym is not None:
                sym.signature = _signature(p.sig_parts)
                sym.line_end = lineno
        self.pending = p

    # ------------------------------------------------------------------ déclarations
    def _handle_declaration(self, lineno: int, decl: str, ml: str, anns: List[_Ann],
                            offset: int = 0) -> Optional[_Block]:
        """Reconnaît une déclaration sur le segment de ligne (``offset`` = colonne du segment) ; renvoie le
        bloc à associer au prochain ``{`` (types, méthodes, blocs de routage Ktor) ou None."""
        s = decl.strip()
        if not s:
            return None
        top = self.stack[-1] if self.stack else None
        in_type = top is not None and top.kind == "type"
        if s.startswith("package"):
            m = _RE_PACKAGE.match(decl)
            if m:
                self.package = m.group(1)
                self.fa.symbols.append(ExtractedSymbol(m.group(1), "module", lineno, lineno, signature=_collapse(s),
                                                       meta={"package": m.group(1)}))
            return None
        if s.startswith("import"):
            self._declare_import(lineno, decl)
            return None
        if ("class" in s or "interface" in s or "enum" in s or "record" in s or "object" in s):
            m = (_RE_TYPE_KT if self.kt else _RE_TYPE_JAVA).match(decl)
            if m:
                blk = self._declare_type(lineno, decl, ml, m, anns, offset)
                if blk is not None:
                    return blk
        if self.kt:
            blk = self._ktor_server(lineno, decl)
            if blk is not None:
                return blk
            m = _RE_TYPEALIAS_KT.match(decl)
            if m:
                self.fa.symbols.append(ExtractedSymbol(m.group(2), "type", lineno, lineno, signature=_collapse(s),
                                                       meta={"modifiers": _modifiers(m.group(1)), "typealias": True}))
                return None
        if "(" in s:
            if self.kt:
                m = _RE_FUN_KT.match(decl)
                if m and m.group(3) not in _KT_NOT_A_NAME:
                    return self._declare_function(lineno, decl, anns, m.group(3).strip("`"), _modifiers(m.group(1)),
                                                  receiver=m.group(2), returns=None)
                if in_type:
                    m = _RE_CTOR_KT.match(decl)
                    if m:
                        assert top is not None and top.symbol is not None
                        return self._declare_function(lineno, decl, anns, top.symbol.name, _modifiers(m.group(1)),
                                                      constructor=True)
            elif in_type:
                assert top is not None and top.symbol is not None
                m = _RE_METHOD_JAVA.match(decl)
                if m:
                    rtype = m.group(2).strip()
                    name = m.group(3)
                    if rtype.split("<", 1)[0].split("[", 1)[0] not in _NOT_A_TYPE and name not in _NOT_A_NAME:
                        top.enum_done = True
                        return self._declare_function(lineno, decl, anns, name, _modifiers(m.group(1)), returns=rtype)
                m = _RE_CTOR_JAVA.match(decl)
                if m and m.group(2) == top.symbol.name:
                    top.enum_done = True
                    return self._declare_function(lineno, decl, anns, m.group(2), _modifiers(m.group(1)),
                                                  constructor=True)
        if in_type:
            assert top is not None
            if top.is_enum and not top.enum_done and _RE_ENUM_LINE.match(decl):
                self._declare_enum_constants(lineno, decl, top)
                return None
            if self.kt:
                m = _RE_PROP_KT.match(decl)
                if m:
                    top.enum_done = True
                    self._declare_property(lineno, decl, m, anns, top)
            else:
                m = _RE_FIELD_JAVA.match(decl)
                if m and m.group(2).split("<", 1)[0].split("[", 1)[0] not in _NOT_A_TYPE:
                    top.enum_done = True
                    self._declare_field(lineno, decl, m, anns, top)
        elif self.kt and top is None:
            m = _RE_PROP_KT.match(decl)
            if m:
                self._declare_property(lineno, decl, m, anns, None)
        return None

    def _declare_import(self, lineno: int, decl: str) -> None:
        if self.kt:
            m = _RE_IMPORT_KT.match(decl)
            if not m:
                return
            module = m.group(1).replace("`", "")
            names = [] if module.endswith(".*") else [module.rsplit(".", 1)[-1]]
            self.fa.imports.append(ExtractedImport(module=module, names=names, line=lineno, is_relative=False))
            return
        m = _RE_IMPORT_JAVA.match(decl)
        if not m:
            return
        module = m.group(2)
        names = [] if module.endswith(".*") else [module.rsplit(".", 1)[-1]]
        self.fa.imports.append(ExtractedImport(module=module, names=names, line=lineno, is_relative=False))

    def _declare_type(self, lineno: int, decl: str, ml: str, m: "re.Match[str]", anns: List[_Ann],
                      offset: int = 0) -> Optional[_Block]:
        mods = _modifiers(m.group(1))
        keyword = m.group(2)
        name = m.group(3)
        if not name and keyword != "object":
            return None  # ``class`` sans nom : fichier tronqué ou ``Foo::class`` isolé
        meta: Dict[str, Any] = {"modifiers": mods, "annotations": [a.name for a in anns]}
        if self.package:
            meta["package"] = self.package
        if keyword == "object":
            if not name:
                if "companion" not in mods:
                    return None  # expression ``object : Foo { }`` anonyme
                name = "Companion"
            meta["object"] = True
            kind = "class"
        elif keyword == "interface" or keyword == "@interface":
            kind = "interface"
            if keyword == "@interface":
                meta["annotation_type"] = True
        elif keyword == "enum" or "enum" in mods:
            kind = "enum"
        else:
            kind = "class"
            if keyword == "record":
                meta["record"] = True
        if "companion" in mods:
            meta["companion"] = True
        if "data" in mods:
            meta["data"] = True
        if "annotation" in mods:
            meta["annotation_class"] = True
        if "sealed" in mods:
            meta["sealed"] = True
        parent_blk = self._type_block()
        parent = parent_blk.symbol.name if parent_blk is not None and parent_blk.symbol is not None else None
        sym = ExtractedSymbol(name, kind, lineno, lineno, signature=_collapse(decl.strip()), parent=parent, meta=meta)
        blk = _Block(sym, "type")
        blk.annotations = anns
        blk.is_enum = kind == "enum"
        blk.is_interface = kind == "interface"
        ann_names = set(a.name for a in anns)
        # --- modèle de données
        if ann_names & {"Entity", "MappedSuperclass", "Embeddable", "Document"}:
            blk.model = self._jpa_model(lineno, name, anns, ann_names)
        elif "data" in mods or keyword == "record":
            blk.model = ExtractedModel(name=name, kind="dataclass", line=lineno,
                                       meta={"language": self.lang, "record": keyword == "record"})
        self.fa.symbols.append(sym)
        if blk.model is not None:
            self.fa.models.append(blk.model)
        # --- constructeur primaire (Kotlin) / composants (record Java)
        if (self.kt and keyword == "class") or keyword == "record":
            self._primary_constructor(lineno, ml, name, blk, anns, offset)
        # --- contrôleurs / clients déclaratifs
        self._type_routing(blk, anns, ann_names)
        return blk

    def _declare_function(self, lineno: int, decl: str, anns: List[_Ann], name: str, mods: List[str],
                          receiver: Optional[str] = None, returns: Optional[str] = None,
                          constructor: bool = False) -> _Block:
        tblock = self._type_block()
        parent = tblock.symbol.name if tblock is not None and tblock.symbol is not None else None
        meta: Dict[str, Any] = {"modifiers": mods, "annotations": [a.name for a in anns]}
        if "static" in mods:
            meta["static"] = True
        if returns:
            meta["returns"] = _collapse(returns, 120)
        if receiver:
            meta["receiver"] = _collapse(receiver, 120)
        if constructor:
            meta["constructor"] = True
        if "suspend" in mods:
            meta["async"] = True
        kind = "method" if parent else "function"
        sym = ExtractedSymbol(name, kind, lineno, lineno, signature=_collapse(decl.strip()), parent=parent, meta=meta)
        blk = _Block(sym, "method")
        if self.kt and (receiver in ("Route", "Routing", "Application") or re.search(r"rout(e|es|ing)", name, re.I)):
            blk.ktor_routing = True
        self.fa.symbols.append(sym)
        if tblock is not None:
            self._server_endpoints(sym, anns, tblock)
        return blk

    def _declare_field(self, lineno: int, decl: str, m: "re.Match[str]", anns: List[_Ann], tblock: _Block) -> None:
        mods = _modifiers(m.group(1))
        ftype = _collapse(m.group(2), 120)
        name = m.group(3)
        kind = "constant" if "static" in mods and "final" in mods else "property"
        sig = decl.strip()
        default = None
        if m.group(4) == "=":
            head, _, tail = decl.partition("=")
            sig = head.strip()
            default = _collapse(tail.strip().rstrip(";"), 100) or None
        parent = tblock.symbol.name if tblock.symbol is not None else None
        meta: Dict[str, Any] = {"type": ftype, "modifiers": mods, "annotations": [a.name for a in anns]}
        self.fa.symbols.append(ExtractedSymbol(name, kind, lineno, lineno, signature=_collapse(sig), parent=parent, meta=meta))
        if tblock.model is not None and "static" not in mods:
            self._model_field(tblock.model, name, ftype, anns, default=default)

    def _declare_property(self, lineno: int, decl: str, m: "re.Match[str]", anns: List[_Ann],
                          tblock: Optional[_Block]) -> None:
        mods = _modifiers(m.group(1))
        name = m.group(4).strip("`")
        ptype = _collapse(m.group(5), 120) if m.group(5) else ""
        default = _collapse(m.group(7), 100) if m.group(6) else None
        if tblock is not None:
            kind = "constant" if "const" in mods else "property"
        else:
            kind = "constant" if "const" in mods else "variable"
        parent = tblock.symbol.name if tblock is not None and tblock.symbol is not None else None
        sig = decl.partition("=")[0].strip() if m.group(6) else decl.strip()
        meta: Dict[str, Any] = {"type": ptype, "modifiers": mods, "annotations": [a.name for a in anns],
                                "mutable": m.group(2) == "var"}
        if m.group(3):
            meta["receiver"] = m.group(3)
        self.fa.symbols.append(ExtractedSymbol(name, kind, lineno, lineno, signature=_collapse(sig), parent=parent, meta=meta))
        if tblock is not None and tblock.model is not None and not m.group(3):
            self._model_field(tblock.model, name, ptype, anns, default=default,
                              nullable=(ptype.endswith("?") if ptype else None), mutable=(m.group(2) == "var"))

    def _declare_enum_constants(self, lineno: int, decl: str, tblock: _Block) -> None:
        parent = tblock.symbol.name if tblock.symbol is not None else None
        for m in _RE_ENUM_CONST.finditer(decl[:_MAX_DECL_LINE]):
            self.fa.symbols.append(ExtractedSymbol(m.group(1), "constant", lineno, lineno, signature=_collapse(m.group(0)),
                                                   parent=parent, meta={"enum_constant": True}))
        if ";" in decl:
            tblock.enum_done = True

    def _ktor_server(self, lineno: int, decl: str) -> Optional[_Block]:
        """Routage Ktor : ``routing { }``, ``route("/x") { }``, ``get("/y") { }``."""
        if _RE_KTOR_ROUTING.match(decl):
            blk = _Block()
            blk.ktor_routing = True
            return blk
        m = _RE_ROUTE_KT.match(decl)
        if m:
            blk = _Block()
            blk.ktor_routing = True
            blk.route_prefix = m.group(1)
            return blk
        m = _RE_KTOR_VERB.match(decl)
        if m and any(b.ktor_routing for b in self.stack):
            prefix = ""
            for b in self.stack:
                if b.route_prefix:
                    prefix = _join_path(prefix, b.route_prefix)
            path = m.group(2) or ""
            self.fa.endpoints.append(ExtractedEndpoint(
                side="server", method=m.group(1).upper(), path=_join_path(prefix, path), handler=self._enclosing_handler(lineno),
                framework="ktor", line=lineno, meta={"prefix": prefix, "route": path}))
        return None

    # ------------------------------------------------------------------ modèles
    def _jpa_model(self, lineno: int, name: str, anns: List[_Ann], ann_names: "set[str]") -> ExtractedModel:
        model = ExtractedModel(name=name, kind="jpa", line=lineno, meta={"language": self.lang})
        table: Optional[str] = None
        for a in anns:
            if a.name == "Table":
                strs = a.strings("name", "value")
                if strs:
                    table = strs[0]
                schema = a.strings("schema")
                if schema:
                    model.meta["schema"] = schema[0]
            elif a.name == "Entity":
                strs = a.strings("name")
                if strs:
                    model.meta["entity_name"] = strs[0]
            elif a.name == "Document":  # Spring Data MongoDB
                strs = a.strings("collection", "value")
                if strs:
                    table = strs[0]
                model.meta["document"] = True
        if "MappedSuperclass" in ann_names:
            model.meta["mapped_superclass"] = True
        elif "Embeddable" in ann_names:
            model.meta["embeddable"] = True
        else:
            model.table_name = table or _snake_case(name)
        return model

    def _model_field(self, model: ExtractedModel, name: str, ftype: str, anns: List[_Ann],
                     default: Optional[str] = None, nullable: Optional[bool] = None,
                     mutable: Optional[bool] = None) -> None:
        names = set(a.name for a in anns)
        if "Transient" in names:
            return
        f = ModelField(name=name, type=ftype, nullable=nullable, default=default)
        if mutable is not None:
            f.meta["mutable"] = mutable
        for a in anns:
            if a.name == "Column":
                col = a.strings("name", "value")
                if col and col[0] != name:
                    f.meta["column"] = col[0]
                nv = a.value("nullable")
                if nv in ("false", "true"):
                    f.nullable = nv == "true"
                if a.value("unique") == "true":
                    f.meta["unique"] = True
                length = a.value("length")
                if length and length.isdigit():
                    f.meta["length"] = int(length)
            elif a.name == "JoinColumn":
                col = a.strings("name", "value")
                if col:
                    f.meta["join_column"] = col[0]
                nv = a.value("nullable")
                if nv in ("false", "true"):
                    f.nullable = nv == "true"
            elif a.name in ("Id", "EmbeddedId"):
                f.meta["primary"] = True
                if f.nullable is None:
                    f.nullable = False
            elif a.name == "GeneratedValue":
                f.meta["generated"] = True
                strat = a.value("strategy")
                if strat:
                    f.meta["strategy"] = strat.rsplit(".", 1)[-1]
            elif a.name == "Enumerated":
                f.meta["enum"] = True
            elif a.name in _JPA_RELATIONS:
                kind = _JPA_RELATIONS[a.name]
                target = a.value("targetEntity")
                target = target.replace(".class", "").replace("::class", "").rsplit(".", 1)[-1] if target else _element_type(ftype)
                model.relations.append(ModelRelation(kind=kind, target=target, field=name))
                f.meta["relation"] = kind
                mapped = a.strings("mappedBy")
                if mapped:
                    f.meta["mapped_by"] = mapped[0]
        model.fields.append(f)

    def _primary_constructor(self, lineno: int, ml: str, name: str, blk: _Block, class_anns: List[_Ann],
                             offset: int = 0) -> None:
        """Propriétés du constructeur primaire Kotlin (``val``/``var``) ou composants d'un record Java.
        ``ml`` est le segment de ligne masqué commençant à la colonne ``offset``."""
        m = re.search(r"\b(?:class|record)\s+" + re.escape(name) + r"\b[^({\n]{0,300}?\(", ml)
        if not m:
            return
        start_idx = lineno - 1
        open_col = offset + m.end() - 1  # colonne absolue dans la ligne complète
        stop = min(len(self.code_lines), start_idx + _MAX_CTOR_LINES)
        depth = 0
        close_line = -1
        close_col = -1
        for i in range(start_idx, stop):
            for pm in _RE_PARENS.finditer(self.masked_lines[i], open_col if i == start_idx else 0):
                if pm.group(0) == "(":
                    depth += 1
                else:
                    depth -= 1
                    if depth == 0:
                        close_line, close_col = i, pm.start()
                        break
            if close_line >= 0:
                break
        if close_line < 0:
            return
        if close_line == start_idx:
            inner = self.code_lines[start_idx][open_col + 1:close_col]
        else:
            inner = "\n".join([self.code_lines[start_idx][open_col + 1:]] + self.code_lines[start_idx + 1:close_line]
                              + [self.code_lines[close_line][:close_col]])
        if len(inner) > _MAX_ANN_ARGS * 8:
            inner = inner[: _MAX_ANN_ARGS * 8]
        cursor = 0
        for part in _split_top_level(inner):
            off = inner.find(part, cursor)
            if off >= 0:
                cursor = off + len(part)
            off = max(off, 0) + (len(part) - len(part.lstrip()))
            aline = lineno + inner.count("\n", 0, off)  # ligne de la première annotation du paramètre
            anns = _parse_annotations(part, aline)
            clean = _INLINE_ANN.sub("", part).strip()
            nm = re.search(r"\b(?:val|var)\s", part) if self.kt else None
            name_off = (off - (len(part) - len(part.lstrip())) + nm.start()) if nm else (off - (len(part) - len(part.lstrip())) + len(part.rstrip()) - 1)
            pline = lineno + inner.count("\n", 0, max(name_off, off))  # ligne du nom de la propriété
            if self.kt:
                pm2 = _RE_KT_PARAM.match(clean)
                if not pm2 or not pm2.group(2):
                    continue  # paramètre simple (sans val/var) : pas une propriété
                mut, pname, ptype = pm2.group(2), pm2.group(3).strip("`"), _collapse(pm2.group(4), 120)
                pdefault = _collapse(pm2.group(5), 100) if pm2.group(5) else None
                meta: Dict[str, Any] = {"type": ptype, "modifiers": _modifiers(pm2.group(1)),
                                        "annotations": [a.name for a in anns], "mutable": mut == "var",
                                        "constructor_property": True}
                self.fa.symbols.append(ExtractedSymbol(pname, "property", pline, pline, signature=_collapse(clean),
                                                       parent=name, meta=meta))
                if blk.model is not None:
                    self._model_field(blk.model, pname, ptype, anns, default=pdefault, nullable=ptype.endswith("?"),
                                      mutable=(mut == "var"))
            else:
                pm3 = _RE_JAVA_PARAM.match(clean)
                if not pm3:
                    continue
                ptype, pname = _collapse(pm3.group(1), 120), pm3.group(2)
                self.fa.symbols.append(ExtractedSymbol(pname, "property", pline, pline, signature=_collapse(clean),
                                                       parent=name, meta={"type": ptype, "annotations": [a.name for a in anns],
                                                                          "record_component": True}))
                if blk.model is not None:
                    self._model_field(blk.model, pname, ptype, anns)
        blk.ctor_done = True

    # ------------------------------------------------------------------ endpoints
    def _type_routing(self, blk: _Block, anns: List[_Ann], ann_names: "set[str]") -> None:
        """Préfixe de routes et framework portés par une classe (contrôleur Spring/JAX-RS/Micronaut, Feign)."""
        for a in anns:
            if a.name == "FeignClient":
                base = a.strings("url")
                cname = a.strings("name", "value")
                blk.feign = {"base": base[0] if base else None, "client": cname[0] if cname else None}
                p = a.strings("path")
                if p:
                    blk.prefixes = p
            elif a.name == "RequestMapping":
                p = a.strings("value", "path")
                if p:
                    blk.prefixes = p
                blk.server_framework = blk.server_framework or "spring"
            elif a.name == "Path":
                p = a.strings()
                if p:
                    blk.prefixes = p
                blk.server_framework = "jaxrs"
            elif a.name == "Controller" and a.positional:
                p = a.strings()
                if p and p[0].startswith("/"):  # Micronaut : @Controller("/api") ; Spring : nom de bean
                    blk.prefixes = p
                    blk.server_framework = "micronaut"
            elif a.name in ("RestController", "Controller"):
                blk.server_framework = blk.server_framework or "spring"

    def _emit(self, side: str, method: str, path: str, handler: str, framework: str, line: int,
              meta: Dict[str, Any]) -> None:
        clean = dict((k, v) for k, v in meta.items() if v not in (None, "", []))
        self.fa.endpoints.append(ExtractedEndpoint(side=side, method=method, path=path, handler=handler,
                                                   framework=framework, line=line, meta=clean))

    def _server_endpoints(self, sym: ExtractedSymbol, anns: List[_Ann], tblock: _Block) -> None:
        """Endpoints déclarés par annotations de méthode (Spring, JAX-RS, Micronaut, Feign, Retrofit)."""
        if not anns or tblock.symbol is None:
            return
        handler = "%s@%s" % (tblock.symbol.name, sym.name)
        feign = tblock.feign
        jaxrs_path = [a for a in anns if a.name == "Path"]
        extra: Dict[str, Any] = {}
        for a in anns:
            if a.name in ("Produces", "Consumes"):
                extra[a.name.lower()] = a.strings()
        for a in anns:
            if a.name in _SPRING_MAPPINGS:
                paths = a.strings("value", "path") or [""]
                if a.name == "RequestMapping":
                    mv = a.value("method")
                    methods = _RE_REQUEST_METHOD.findall(mv) if mv else []
                    methods = methods or ["ANY"]
                else:
                    methods = [_SPRING_MAPPINGS[a.name]]
                if feign is not None:
                    side, framework = "client", "feign"
                    extra_meta = {"base": feign.get("base"), "client": feign.get("client")}
                else:
                    side, framework = "server", "spring"
                    extra_meta = {}
                for attr in ("consumes", "produces"):  # @PostMapping(consumes = "application/json")
                    vals = a.strings(attr)
                    if vals:
                        extra_meta[attr] = vals
                for prefix in tblock.prefixes:
                    for p in paths:
                        for meth in methods:
                            meta = {"prefix": prefix, "route": p, "annotations": [x.name for x in anns]}
                            meta.update(extra)
                            meta.update(extra_meta)
                            self._emit(side, meth, _join_path(prefix, p), handler, framework, a.line, meta)
            elif a.name in _MICRONAUT_VERBS:
                paths = a.strings("uri", "value") or [""]
                for prefix in tblock.prefixes:
                    for p in paths:
                        meta = {"prefix": prefix, "route": p, "annotations": [x.name for x in anns]}
                        meta.update(extra)
                        self._emit("server", _MICRONAUT_VERBS[a.name], _join_path(prefix, p), handler, "micronaut", a.line, meta)
            elif a.name in _JAXRS_VERBS:
                literal = a.strings()
                if literal:  # Retrofit : @GET("users/{id}")
                    for p in literal:
                        self._emit("client", a.name, p, handler, "retrofit", a.line, {"route": p})
                else:  # JAX-RS : @GET + @Path("...")
                    paths = jaxrs_path[0].strings() if jaxrs_path else [""]
                    for prefix in tblock.prefixes:
                        for p in paths or [""]:
                            meta = {"prefix": prefix, "route": p, "annotations": [x.name for x in anns]}
                            meta.update(extra)
                            self._emit("server", a.name, _join_path(prefix, p), handler, "jaxrs", a.line, meta)
            elif a.name == "HTTP":  # Retrofit @HTTP(method = "DELETE", path = "x", hasBody = true)
                meth = a.strings("method")
                for p in a.strings("path") or [""]:
                    self._emit("client", (meth[0].upper() if meth else "ANY"), p, handler, "retrofit", a.line, {"route": p})

    # ------------------------------------------------------------------ appels HTTP clients
    def _window(self, lineno: int) -> str:
        """Texte (borné) de l'instruction qui commence à ``lineno`` : chaînage de builder sur plusieurs lignes."""
        start = lineno - 1
        parts: List[str] = []
        end = min(len(self.code_lines), start + _MAX_WINDOW_LINES)
        for i in range(start, end):
            cl = self.code_lines[i]
            parts.append(cl[:_MAX_DECL_LINE])
            ml = self.masked_lines[i]
            if ";" in ml:
                break
            st = ml.rstrip()
            if st.endswith((".", "(", ",", "{", "->", "?.")):
                continue
            nxt = self.masked_lines[i + 1].lstrip() if i + 1 < len(self.masked_lines) else ""
            if nxt.startswith((".", "?.", ")")):
                continue
            if "(" in ml and ml.count("(") > ml.count(")"):
                continue
            break
        return "\n".join(parts)

    def _consumed(self, framework: str, lineno: int) -> bool:
        return self.consumed.get(framework, 0) >= lineno

    def _consume(self, framework: str, lineno: int, window: str) -> None:
        self.consumed[framework] = lineno + window.count("\n")

    def _emit_client(self, framework: str, method: str, expr: str, lineno: int, extra: Optional[Dict[str, Any]] = None) -> bool:
        path, base, dynamic = _url_from_expr(expr)
        if path is None:
            return False
        meta: Dict[str, Any] = {"base": base}
        if base is None and self.base_hint and not path.startswith(("http://", "https://")):
            meta["base"] = self.base_hint
        if dynamic:
            meta["dynamic"] = True
        if extra:
            meta.update(extra)
        self._emit("client", method.upper(), path, self._enclosing_handler(lineno), framework, lineno, meta)
        return True

    def _client_calls(self, lineno: int, cl: str) -> None:
        if "(" not in cl:
            return
        line = cl[:_MAX_DECL_LINE]
        # RestTemplate
        m = _RE_RT.search(line)
        if m and not self._consumed("resttemplate", lineno):
            call = m.group(1)
            ok = call not in ("put", "delete") or _RE_RT_RECEIVER.search(line) is not None
            if ok:
                window = self._window(lineno)
                arg = _first_argument(window, m.end() - 1)
                if call == "exchange":
                    hm = re.search(r"HttpMethod\s*\.\s*(\w+)", window)
                    method = hm.group(1) if hm else "ANY"
                else:
                    method = _RT_METHODS.get(call, "ANY")
                if self._emit_client("resttemplate", method, arg, lineno, {"call": call}):
                    self._consume("resttemplate", lineno, window)
        # WebClient / RestClient (API fluide)
        m = _RE_WC.search(line)
        if m and not self._consumed("webclient", lineno):
            window = self._window(lineno)
            um = re.search(r"\.uri\s*\(", window)
            if um:
                arg = _first_argument(window, um.end() - 1)
            else:
                pm = re.search(r"\.path\s*\(", window)
                arg = _first_argument(window, pm.end() - 1) if pm else ""
            if arg:
                method = m.group(1).upper()
                if method == "METHOD":
                    hm = re.search(r"HttpMethod\s*\.\s*(\w+)", window)
                    method = hm.group(1) if hm else "ANY"
                framework = "restclient" if re.search(r"restclient", window, re.I) else "webclient"
                if self._emit_client(framework, method, arg, lineno):
                    self._consume("webclient", lineno, window)
        # OkHttp
        if "Request.Builder" in line or ".url(" in line:
            if not self._consumed("okhttp", lineno):
                window = self._window(lineno)
                um = re.search(r"\.url\s*\(", window)
                if um and ("Request.Builder" in window or ".build()" in window):
                    arg = _first_argument(window, um.end() - 1)
                    vm = re.search(r"\.(post|put|patch|delete|head)\s*\(", window)
                    mm = re.search(r"\.method\s*\(\s*\"(\w+)\"", window)
                    method = mm.group(1) if mm else (vm.group(1) if vm else "GET")
                    if self._emit_client("okhttp", method, arg, lineno):
                        self._consume("okhttp", lineno, window)
        # java.net.http.HttpClient (Java 11)
        if "HttpRequest.newBuilder" in line and not self._consumed("httpclient", lineno):
            window = self._window(lineno)
            arg = ""
            for pat in (r"URI\.create\s*\(", r"\.uri\s*\(", r"new\s+URI\s*\(", r"HttpRequest\.newBuilder\s*\("):
                um = re.search(pat, window)
                if um:
                    arg = _first_argument(window, um.end() - 1)
                    if arg:
                        break
            vm = re.search(r"\.(GET|POST|PUT|DELETE|PATCH|HEAD)\s*\(", window)
            mm = re.search(r"\.method\s*\(\s*\"(\w+)\"", window)
            method = mm.group(1) if mm else (vm.group(1) if vm else "GET")
            if arg and self._emit_client("httpclient", method, arg, lineno):
                self._consume("httpclient", lineno, window)
        # Ktor client
        if self.kt and not self._consumed("ktor-client", lineno):
            m = _RE_KTOR_CLIENT.search(line)
            if m:
                window = self._window(lineno)
                arg = _first_argument(window, m.end() - 1)
                if not arg or not arg.startswith('"'):
                    um = re.search(r"\burl\s*\(", window)
                    arg = _first_argument(window, um.end() - 1) if um else ""
                lit = _STR_LIT.search(arg) if arg else None
                if lit and (lit.group(1).startswith(("/", "http", "$")) or "/" in lit.group(1)):
                    if self._emit_client("ktor-client", m.group(1), arg, lineno):
                        self._consume("ktor-client", lineno, window)
        # Spring WebFlux / WebMvc.fn : route(GET("/x"), handler::list)
        if "::" in line or "route" in line:
            for fm in _RE_FN_ROUTE.finditer(line):
                hm = re.search(r"(\w+)\s*::\s*(\w+)", line)
                handler = "%s@%s" % (hm.group(1), hm.group(2)) if hm else self._enclosing_handler(lineno)
                self._emit("server", fm.group(1), fm.group(2), handler, "spring", lineno, {"functional": True})


_RE_RT = re.compile(
    r"\.(getForObject|getForEntity|postForObject|postForEntity|postForLocation|exchange|patchForObject|headForHeaders|optionsForAllow|put|delete)\s*\("
)
_RE_RT_RECEIVER = re.compile(r"(?i)(?:template|rest)\w*\s*\.\s*(?:put|delete)\s*\(")
_RT_METHODS = {"getForObject": "GET", "getForEntity": "GET", "postForObject": "POST", "postForEntity": "POST",
               "postForLocation": "POST", "patchForObject": "PATCH", "headForHeaders": "HEAD", "optionsForAllow": "OPTIONS",
               "put": "PUT", "delete": "DELETE"}
_RE_WC = re.compile(r"\.(get|post|put|patch|delete|head|options|method)\s*\(\s*(?:HttpMethod\s*\.\s*\w+\s*)?\)")
_RE_KTOR_CLIENT = re.compile(r"\.(get|post|put|patch|delete|head|options|request)\s*(?:<[^>\n]{0,100}>)?\s*\(\s*(?=\"|\w)")
_RE_FN_ROUTE = re.compile(r"(?<![\w.@])(GET|POST|PUT|PATCH|DELETE)\s*\(\s*\"([^\"\n]{0,300})\"")
_RE_KT_PARAM = re.compile(
    r"^\s*((?:(?:private|protected|internal|public|override|open|final|vararg|crossinline|noinline)\s+)*)(val|var)?\s*"
    r"(\w+|`[^`\n]{1,100}`)\s*:\s*(" + _KT_TYPE + r")\s*(?:=\s*([\s\S]{0,200}))?\s*$"
)
_RE_JAVA_PARAM = re.compile(r"^\s*(?:final\s+)?(" + _JAVA_TYPE + r")(?:\s*\.\.\.)?\s+(\w+)\s*$")


# --------------------------------------------------------------------------- extracteur
class JavaKotlinExtractor:
    """Extracteur Java/Kotlin : ``analyze(text, path)`` ne lève jamais d'exception."""

    languages: List[str] = ["java", "kotlin"]

    def analyze(self, text: str, path: str) -> FileAnalysis:
        """Le langage vient de l'extension (``.java`` / ``.kt`` / ``.kts``), sinon du contenu."""
        lower = (path or "").lower()
        language = ""
        try:
            if not isinstance(text, str):
                text = bytes(text).decode("utf-8", errors="replace")
            if lower.endswith((".kt", ".kts")):
                language = "kotlin"
            elif lower.endswith(".java"):
                language = "java"
            else:
                language = _sniff_language(text)
        except Exception:  # noqa: BLE001 - entrée illisible : on retombe sur Java
            language = language or "java"
            text = text if isinstance(text, str) else ""
        fa = FileAnalysis(language=language)
        try:
            _Analyzer(text, path, language, fa).run()
        except Exception as exc:  # noqa: BLE001 - jamais d'exception vers l'appelant
            fa.warnings.append("échec partiel de l'analyse de %s : %s" % (path, exc))
        return fa


EXTRACTOR = JavaKotlinExtractor()
