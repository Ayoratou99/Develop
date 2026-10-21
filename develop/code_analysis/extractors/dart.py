"""Extracteur Dart / Flutter : symboles, modèles de données, endpoints HTTP et imports.

Analyse purement lexicale (stdlib + ``re``), sans parseur : un scanner masque d'abord les
chaînes et les commentaires (en gardant la longueur et les sauts de ligne), puis des
expressions régulières bornées (ligne par ligne) repèrent les déclarations, et un
équilibrage d'accolades/parenthèses calculé sur le texte masqué donne les fins de
définition. Les chaînes avec interpolation (``$x``, ``${expr}``), les chaînes brutes
(``r'...'``) et les chaînes multi-lignes (``'''`` / ``\"\"\"``) sont gérées par le scanner.

Heuristiques couvertes :
- symboles : classes (dont widgets Flutter → ``component``), mixins, enums, extensions,
  typedefs, fonctions de haut niveau (y compris closures nommées ``final f = () async {…}``),
  méthodes/constructeurs/opérateurs, getters/setters, constantes ;
- modèles : classes « données » (``fromJson``/``fromMap``, ``@JsonSerializable``,
  ``@freezed`` ou uniquement des champs ``final`` + constructeur) → kind ``dart_class`` ;
- endpoints client : package ``http`` (``http.get(Uri.parse(…))``, ``http.Request('POST', …)``),
  ``dio`` (``dio.get<T>('/x')``, ``request(options: Options(method: …))``), ``Uri.parse``/``Uri.https``
  autonomes, annotations retrofit/chopper ; endpoints serveur : shelf_router (appels, cascade,
  ``@Route.get``), dart_frog (route dérivée du chemin ``routes/x/[id].dart``) ;
- imports : ``import`` / ``export`` / ``part`` / ``part of`` (alias, show, deferred, conditionnels).
"""
from __future__ import annotations

import bisect
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

# --------------------------------------------------------------------------- constantes

_IDENT = r"[A-Za-z_$][\w$]*"
_QIDENT = r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*"
# Type Dart : identifiant qualifié + génériques (2 niveaux d'imbrication) + '?' optionnel.
_GENERIC = r"<[^<>;={}()]*(?:<[^<>;={}()]*(?:<[^<>;={}()]*>[^<>;={}()]*)*>[^<>;={}()]*)*>"
# Suffixe de type fonction : ``void Function(int)`` / ``T Function<T>(T a, {int b})`` (parenthèses bornées).
_FUNC_TYPE_SUFFIX = r"(?:\s+Function\s*(?:" + _GENERIC + r")?\s*\([^()]{0,300}(?:\([^()]{0,100}\)[^()]{0,100})*\))?"
_TYPE = r"(?:" + _QIDENT + r")(?:\s*" + _GENERIC + r")?" + _FUNC_TYPE_SUFFIX + r"\??"
_STRING = r"(?:r?'[^'\n]*'|r?\"[^\"\n]*\")"
_STRING_CAPTURE = r"r?(?P<q>['\"])(?P<s>[^'\"\n]*)(?P=q)"

_KEYWORDS = frozenset(
    "if else for while do switch case default return await throw try catch finally new assert "
    "yield super this import export part library class enum mixin extension typedef with "
    "extends implements on in is as var const final late static abstract void get set "
    "operator async sync external factory covariant required break continue rethrow".split()
)
_NAME_BLACKLIST = frozenset(
    "if else for while do switch case default return await throw try catch finally new assert "
    "yield super this import export part library class enum mixin extension typedef with "
    "extends implements in is as var const final late break continue rethrow".split()
)

_WIDGET_BASES = frozenset(
    "StatelessWidget StatefulWidget State HookWidget StatefulHookWidget ConsumerWidget "
    "ConsumerStatefulWidget ConsumerState HookConsumerWidget InheritedWidget "
    "InheritedNotifier RenderObjectWidget PreferredSizeWidget".split()
)
_DATA_METHODS = frozenset("toJson toMap copyWith toString == hashCode props".split())
_HTTP_VERBS = frozenset("get post put patch delete head".split())
_MAX_SCAN_TOKENS = 400000  # borne de sécurité du scanner (fichiers pathologiques)
_MAX_ARGS_LEN = 4000  # longueur maximale d'arguments d'appel examinés
_MAX_LOOKBACK = 60  # lignes remontées pour résoudre une variable d'URL
_MAX_CALLS = 5000  # nombre maximal de sites d'appel / Uri examinés par fichier

# --------------------------------------------------------------------------- scanner (masquage)

_TOKEN_RE = re.compile(r"//|/\*|r?'''|r?\"\"\"|r?'|r?\"")
_BLOCK_RE = re.compile(r"/\*|\*/")
_INTERP_RE = re.compile(r"[{}]|'''|\"\"\"|'|\"")
# Fin de chaîne (échappement, début d'interpolation, fin de ligne pour les chaînes simples, guillemet fermant).
_STRING_END_RE: Dict[str, "re.Pattern[str]"] = {
    "'": re.compile(r"\\.|\$\{|\n|'"),
    '"': re.compile(r"\\.|\$\{|\n|\""),
    "'''": re.compile(r"\\.|\$\{|'''", re.DOTALL),
    '"""': re.compile(r"\\.|\$\{|\"\"\"", re.DOTALL),
}


def _blank(s: str) -> str:
    """Remplace tout sauf les sauts de ligne par des espaces (conserve longueur et lignes)."""
    return re.sub(r"[^\n]", " ", s)


class _Scanner:
    """Produit deux variantes du texte, de même longueur que l'original :

    - ``masked`` : chaînes ET commentaires blanchis (pour la structure : accolades, déclarations) ;
    - ``code`` : commentaires blanchis, chaînes conservées (pour les URLs, imports, valeurs).
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.warnings: List[str] = []
        self._masked: List[str] = []
        self._code: List[str] = []
        self._tokens = 0

    def run(self) -> Tuple[str, str]:
        text = self.text
        n = len(text)
        pos = 0
        while pos < n:
            self._tokens += 1
            if self._tokens > _MAX_SCAN_TOKENS:
                self.warnings.append("scanner interrompu (trop de chaînes/commentaires) : reste traité comme du code")
                self._emit(text[pos:], text[pos:])
                break
            m = _TOKEN_RE.search(text, pos)
            if m is None:
                self._emit(text[pos:], text[pos:])
                break
            self._emit(text[pos:m.start()], text[pos:m.start()])
            tok = m.group()
            start = m.start()
            if tok == "//":
                end = text.find("\n", m.end())
                end = n if end < 0 else end
                self._emit(_blank(text[start:end]), _blank(text[start:end]))
                pos = end
            elif tok == "/*":
                end = self._block_comment_end(m.end())
                self._emit(_blank(text[start:end]), _blank(text[start:end]))
                pos = end
            else:
                raw = tok.startswith("r")
                if raw and start > 0 and (text[start - 1].isalnum() or text[start - 1] in "_$"):
                    # 'r' fait partie d'un identifiant (ex. ``var'``) : ce n'est pas un préfixe brut
                    self._emit("r", "r")
                    raw = False
                    tok = tok[1:]
                    start += 1
                quote = tok[1:] if tok.startswith("r") else tok
                content_start = start + len(tok)
                end = _string_end(text, content_start, quote, raw, 0)
                terminated = end >= 0
                if not terminated:
                    if len(self.warnings) < 20:
                        self.warnings.append("chaîne non terminée à partir de la ligne %d" % (text.count("\n", 0, start) + 1))
                    if len(quote) == 1:
                        # chaîne simple : elle ne peut pas dépasser sa ligne, on ne masque que jusqu'au saut de ligne
                        end = text.find("\n", content_start)
                        end = n if end < 0 else end
                    else:
                        end = n
                chunk = text[start:end]
                self._emit(_mask_string(chunk, len(tok), quote, terminated), chunk)
                pos = end
        return "".join(self._masked), "".join(self._code)

    def _emit(self, masked: str, code: str) -> None:
        self._masked.append(masked)
        self._code.append(code)

    def _block_comment_end(self, pos: int) -> int:
        """Fin (exclusive) d'un commentaire bloc, imbriquable comme en Dart."""
        depth = 1
        text = self.text
        while depth > 0:
            m = _BLOCK_RE.search(text, pos)
            if m is None:
                return len(text)
            if m.group() == "/*":
                depth += 1
            else:
                depth -= 1
            pos = m.end()
        return pos


def _mask_string(chunk: str, prefix_len: int, quote: str, terminated: bool) -> str:
    """Blanchit l'intérieur d'un littéral de chaîne en conservant ses guillemets."""
    qlen = len(quote) if terminated and len(chunk) >= prefix_len + len(quote) else 0
    head = chunk[:prefix_len]
    tail = chunk[len(chunk) - qlen:] if qlen else ""
    body = chunk[prefix_len:len(chunk) - qlen] if qlen else chunk[prefix_len:]
    return head + _blank(body) + tail


def _string_end(text: str, pos: int, quote: str, raw: bool, depth: int) -> int:
    """Index juste après le guillemet fermant d'une chaîne commencée en ``pos`` ; -1 si non terminée."""
    n = len(text)
    if raw:
        if len(quote) == 1:  # chaîne brute simple : bornée par sa ligne
            nl = text.find("\n", pos)
            end = text.find(quote, pos, n if nl < 0 else nl)
        else:
            end = text.find(quote, pos)
        return -1 if end < 0 else end + len(quote)
    pat = _STRING_END_RE[quote]
    while pos < n:
        m = pat.search(text, pos)
        if m is None:
            return -1
        tok = m.group()
        if tok == quote:
            return m.end()
        if tok == "\n":  # chaîne simple non terminée sur sa ligne : on s'arrête là (code invalide)
            return -1
        if tok == "${":
            pos = _interp_end(text, m.end(), depth + 1)
            if pos < 0:
                return -1
            continue
        pos = m.end()  # séquence d'échappement
    return -1


def _interp_end(text: str, pos: int, depth: int) -> int:
    """Index juste après le ``}`` fermant une interpolation ``${...}`` (gère chaînes imbriquées)."""
    if depth > 24:
        end = text.find("}", pos)
        return -1 if end < 0 else end + 1
    level = 1
    n = len(text)
    while pos < n:
        m = _INTERP_RE.search(text, pos)
        if m is None:
            return -1
        tok = m.group()
        if tok == "{":
            level += 1
            pos = m.end()
        elif tok == "}":
            level -= 1
            pos = m.end()
            if level == 0:
                return pos
        else:
            raw = m.start() > 0 and text[m.start() - 1] == "r"
            pos = _string_end(text, m.end(), tok, raw, depth)
            if pos < 0:
                return -1
    return -1


# --------------------------------------------------------------------------- structure (lignes, accolades)


class _Layout:
    """Index structurel du texte masqué : lignes, accolades/parenthèses appariées, profondeur par ligne."""

    def __init__(self, masked: str) -> None:
        self.masked = masked
        self.line_starts: List[int] = [0]
        for m in re.finditer(r"\n", masked):
            self.line_starts.append(m.end())
        self.brace_close: Dict[int, int] = {}  # index '{' -> index '}'
        self.paren_close: Dict[int, int] = {}  # index '(' -> index ')'
        self.balanced = True
        self.depth_at_line: List[int] = []
        self._build()

    def _build(self) -> None:
        braces: List[int] = []
        parens: List[int] = []
        depth = 0
        starts = self.line_starts
        li = 0
        nlines = len(starts)
        depth_at_line = [0] * nlines
        for m in re.finditer(r"[{}()]", self.masked):
            idx = m.start()
            while li < nlines and starts[li] <= idx:
                depth_at_line[li] = depth
                li += 1
            ch = m.group()
            if ch == "{":
                braces.append(idx)
                depth += 1
            elif ch == "}":
                if braces:
                    self.brace_close[braces.pop()] = idx
                    depth -= 1
                else:
                    self.balanced = False
            elif ch == "(":
                parens.append(idx)
            else:
                if parens:
                    self.paren_close[parens.pop()] = idx
                else:
                    self.balanced = False
        while li < nlines:
            depth_at_line[li] = depth
            li += 1
        if braces or parens:
            self.balanced = False
        self.depth_at_line = depth_at_line

    def line_of(self, idx: int) -> int:
        """Numéro de ligne (1-based) de l'index de caractère ``idx``."""
        return bisect.bisect_right(self.line_starts, idx)

    def line_text(self, lineno: int) -> str:
        start = self.line_starts[lineno - 1]
        end = self.line_starts[lineno] - 1 if lineno < len(self.line_starts) else len(self.masked)
        return self.masked[start:end]


_BODY_START_RE = re.compile(r"\s*(?:(?P<mod>async\*?|sync\*)\s*)?(?P<tok>\{|=>|;|:|=)")
_SEMI_RE = re.compile(r";")
_OPEN_RE = re.compile(r"[{;=]")


def _collapse(s: str, limit: int = 240) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


# --------------------------------------------------------------------------- expressions de déclaration

_ANNOT_PREFIX = r"(?:@" + _QIDENT + r"(?:\([^()]*(?:\([^()]*\)[^()]*)*\))?\s*)*"
_ANNOT_LINE_RE = re.compile(r"^\s*@" + _QIDENT + r"(?:\([^()]*(?:\([^()]*\)[^()]*)*\))?\s*$")
_ANNOT_ITEM_RE = re.compile(r"@(" + _QIDENT + r")(\([^()]*(?:\([^()]*\)[^()]*)*\))?")

_CLASS_RE = re.compile(
    r"^(?P<indent>\s*)" + _ANNOT_PREFIX
    + r"(?P<mods>(?:(?:abstract|base|interface|final|sealed|mixin|macro)\s+)*)"
    + r"(?P<kw>class|mixin|enum|extension(?:\s+type)?)\s+"
    + r"(?:(?P<name>" + _IDENT + r")\s*(?P<gen>" + _GENERIC + r")?)?"
    + r"(?P<rest>[^{;=]*)(?P<end>[{;=]|$)"
)
_TYPEDEF_RE = re.compile(
    r"^\s*" + _ANNOT_PREFIX + r"typedef\s+(?:(?P<ret>" + _TYPE + r")\s+)?(?P<name>" + _IDENT + r")\s*(?:" + _GENERIC + r")?\s*(?:=|\()"
)
_LIBRARY_RE = re.compile(r"^\s*library\s+(?P<name>" + _QIDENT + r")?\s*;")
_IMPORT_RE = re.compile(
    r"^\s*(?P<kw>import|export|part\s+of|part)\s+" + _STRING_CAPTURE
    + r"(?:\s+if\s*\([^()]*\)\s*" + _STRING + r")*"  # imports conditionnels : if (dart.library.io) 'x.dart'
    + r"(?:\s+(?:deferred\s+)?as\s+(?P<alias>" + _IDENT + r"))?"
    + r"(?P<combinators>(?:\s+(?:show|hide)\s+[^;]+)*)\s*;"
)
_PART_OF_NAME_RE = re.compile(r"^\s*part\s+of\s+(?P<name>" + _QIDENT + r")\s*;")
_SHOW_RE = re.compile(r"\bshow\s+([^;]+?)(?=\s+hide\b|\s*$)")

_FUNC_RE = re.compile(
    r"^(?P<indent>\s*)" + _ANNOT_PREFIX
    + r"(?P<mods>(?:(?:static|external|abstract|const|factory|covariant)\s+)*)"
    + r"(?:(?P<ret>" + _TYPE + r")\s+)?"
    + r"(?:(?P<acc>set)\s+)?"
    + r"(?P<name>" + _IDENT + r"(?:\.(?:" + _IDENT + r"))?)\s*(?P<gen>" + _GENERIC + r")?\s*\("
)
_OPERATOR_RE = re.compile(
    r"^(?P<indent>\s*)" + _ANNOT_PREFIX
    + r"(?:(?P<ret>" + _TYPE + r")\s+)?operator\s*(?P<op>==|\[\]=?|<=|>=|<<|>>>?|[-+*/%<>&|^~]|~/|unary-)\s*\("
)
_GETTER_RE = re.compile(
    r"^(?P<indent>\s*)" + _ANNOT_PREFIX
    + r"(?P<mods>(?:(?:static|external|abstract)\s+)*)"
    + r"(?:(?P<ret>" + _TYPE + r")\s+)?get\s+(?P<name>" + _IDENT + r")\s*(?P<tok>=>|\{|;|$)"
)
_VAR_RE = re.compile(
    r"^(?P<indent>\s*)" + _ANNOT_PREFIX
    + r"(?P<mods>(?:(?:static|final|late|const|covariant|var|external|abstract)\s+)*)"
    + r"(?:(?P<type>" + _TYPE + r")\s+)?"
    + r"(?P<names>" + _IDENT + r"(?:\s*,\s*" + _IDENT + r")*)\s*"
    + r"(?:=\s*(?P<default>[^;]*))?;"
)
# Variable/champ dont l'initialiseur continue sur les lignes suivantes (``= '''``, ``= () async {``, ``= Foo(``).
_VAR_OPEN_RE = re.compile(
    r"^(?P<indent>\s*)" + _ANNOT_PREFIX
    + r"(?P<mods>(?:(?:static|final|late|const|covariant|var|external|abstract)\s+)*)"
    + r"(?:(?P<type>" + _TYPE + r")\s+)?"
    + r"(?P<names>" + _IDENT + r")\s*=\s*(?P<default>[^;]*?)\s*$"
)
_OPEN_TAIL_CHARS = frozenset("({['\"=+-*/%?:|&!<>")  # fins de ligne plausibles d'une expression inachevée
# Closure ``(params) async => …`` / ``(params) { … }`` (parenthèses bornées à deux niveaux).
_CLOSURE_RE = re.compile(r"^\s*(?:async\s+)?\([^()]*(?:\([^()]*\)[^()]*)*\)\s*(?P<mod>async\*?|sync\*)?\s*(?P<tok>=>|\{)")
_EXTENDS_RE = re.compile(r"\bextends\s+(" + _TYPE + r")")
_WITH_RE = re.compile(r"\bwith\s+([^{]*?)(?=\s+implements\b|\s+on\b|\s*$)")
_IMPLEMENTS_RE = re.compile(r"\bimplements\s+([^{]*?)(?=\s+with\b|\s+on\b|\s*$)")
_ON_RE = re.compile(r"\bon\s+([^{]*?)(?=\s+with\b|\s+implements\b|\s*$)")
_ENUM_VALUE_RE = re.compile(r"(?:^|,)\s*" + _ANNOT_PREFIX + r"(" + _IDENT + r")\s*(?:" + _GENERIC + r")?\s*(?:\([^()]*(?:\([^()]*\)[^()]*)*\))?\s*(?=,|$)")


def _split_types(s: str) -> List[str]:
    """Découpe ``A, B<C, D>, E`` en ``['A', 'B<C, D>', 'E']`` (virgules de premier niveau)."""
    out: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth <= 0:
            item = "".join(cur).strip()
            if item:
                out.append(item)
            cur = []
        else:
            cur.append(ch)
    item = "".join(cur).strip()
    if item:
        out.append(item)
    return out


def _split_top(s: str) -> List[Tuple[int, int]]:
    """Positions (start, end) des éléments séparés par des virgules de premier niveau dans ``s``."""
    spans: List[Tuple[int, int]] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth <= 0:
            spans.append((start, i))
            start = i + 1
    spans.append((start, len(s)))
    return spans


_PARAM_RE = re.compile(
    r"^\s*(?P<annots>(?:@" + _QIDENT + r"(?:\([^()]*(?:\([^()]*\)[^()]*)*\))?\s*)*)"
    r"(?P<mods>(?:(?:required|final|covariant|var|const)\s+)*)"
    r"(?:(?P<type>" + _TYPE + r")\s+)?"
    r"(?:(?P<this>this|super)\.)?(?P<name>" + _IDENT + r")\s*(?:" + _GENERIC + r")?\s*(?:\([^)]*\))?\s*\??\s*(?:=\s*(?P<default>.*))?$",
    re.DOTALL,
)


def _strip_param_groups(masked: str) -> Tuple[str, int, int]:
    """Efface les accolades/crochets de premier niveau délimitant les paramètres nommés/optionnels.

    Renvoie (texte, position du ``{`` de groupe ou -1, position du ``[`` de groupe ou -1).
    """
    chars = list(masked)
    depth = 0
    named_at = opt_at = -1
    group_close = ""
    for i, ch in enumerate(chars):
        if depth == 0 and not group_close and ch in "{[":
            if ch == "{":
                named_at = i
                group_close = "}"
            else:
                opt_at = i
                group_close = "]"
            chars[i] = " "
            continue
        if depth == 0 and group_close and ch == group_close:
            chars[i] = " "
            group_close = ""
            continue
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
    return "".join(chars), named_at, opt_at


def _parse_params(masked: str, code: str) -> List[Dict[str, Any]]:
    """Analyse une liste de paramètres (texte entre parenthèses, hors parenthèses).

    ``masked`` sert au découpage (chaînes blanchies), ``code`` (même longueur) aux valeurs par défaut.
    """
    params: List[Dict[str, Any]] = []
    masked, named_at, opt_at = _strip_param_groups(masked)
    for a, b in _split_top(masked):
        seg = masked[a:b]
        seg_code = code[a:b]
        if not seg.strip():
            continue
        named = named_at >= 0 and a >= named_at
        positional_opt = opt_at >= 0 and a >= opt_at
        m = _PARAM_RE.match(seg)
        if not m:
            continue
        default = None
        if m.group("default") is not None:
            default = _collapse(seg_code[m.start("default"):m.end("default")], 80) or None
        annots = m.group("annots") or ""
        dm = re.search(r"@Default\(", annots)
        if dm and default is None:
            # @Default(valeur) de freezed : valeur dans le code (chaînes conservées)
            raw = seg_code[m.start("annots"):m.end("annots")]
            dm2 = re.search(r"@Default\((.*)\)\s*(?:@|$)", raw, re.DOTALL)
            if dm2:
                default = _collapse(dm2.group(1), 80)
        mods = m.group("mods") or ""
        params.append({
            "name": m.group("name"),
            "type": (m.group("type") or "").strip(),
            "required": "required" in mods or not (named or positional_opt),
            "named": named,
            "default": default,
            "this": bool(m.group("this")),
            "annotations": [x for x in re.findall(r"@(" + _QIDENT + r")", annots)],
        })
    return params


# --------------------------------------------------------------------------- analyseur


class _Container:
    """Déclaration englobante (class / mixin / enum / extension) et ce qu'on y a trouvé."""

    def __init__(self, symbol: ExtractedSymbol, kw: str, depth: int, body_open: int, body_close: int) -> None:
        self.symbol = symbol
        self.kw = kw  # class | mixin | enum | extension | extension type
        self.depth = depth
        self.body_open = body_open
        self.body_close = body_close
        self.fields: List[Dict[str, Any]] = []
        self.methods: List[ExtractedSymbol] = []
        self.constructors: List[Dict[str, Any]] = []
        self.base_url: Optional[str] = None  # @RestApi(baseUrl: …) / @ChopperApi(baseUrl: …)
        self.members_from: int = body_open  # index à partir duquel les membres commencent (après les valeurs d'enum)

    @property
    def name(self) -> str:
        return self.symbol.name


class _DartAnalyzer:
    """Analyse d'un fichier Dart : une instance par fichier."""

    def __init__(self, text: str, path: str) -> None:
        self.path = path
        self.text = text
        scanner = _Scanner(text)
        self.masked, self.code = scanner.run()
        self.warnings: List[str] = list(scanner.warnings)
        self.layout = _Layout(self.masked)
        self.masked_lines = self.masked.split("\n")
        self.code_lines = self.code.split("\n")
        self.symbols: List[ExtractedSymbol] = []
        self.endpoints: List[ExtractedEndpoint] = []
        self.models: List[ExtractedModel] = []
        self.imports: List[ExtractedImport] = []
        self.containers: List[_Container] = []
        self._container_opens: List[int] = []  # body_open de chaque conteneur, croissant (pour bisect)
        self._assign_index: Optional[Dict[str, List[int]]] = None  # nom -> lignes d'affectation ``nom = …``
        self.func_spans: List[Tuple[int, int, str]] = []  # (line_start, line_end, label) des fonctions/méthodes
        self._span_starts: Optional[List[int]] = None
        self.annotated: List[Tuple[ExtractedSymbol, Optional[_Container], List[Tuple[str, str]]]] = []
        self.consumed_uri: set = set()
        self.file_base: Optional[str] = None
        self._dio_names: set = set()
        self._http_names: set = set()
        if not self.layout.balanced:
            self.warnings.append("accolades/parenthèses déséquilibrées : fins de définition approximatives")

    # ------------------------------------------------------------------ utilitaires

    def _line_index(self, lineno: int) -> int:
        return self.layout.line_starts[lineno - 1]

    def _container_at(self, lineno: int) -> Optional[_Container]:
        """Conteneur (class/mixin/enum/extension) dont le corps contient la ligne ; les corps ne s'imbriquent pas."""
        idx = self._line_index(lineno)
        i = bisect.bisect_right(self._container_opens, idx) - 1
        if i >= 0:
            c = self.containers[i]
            if c.body_open < idx <= c.body_close:
                return c
        return None

    def _body_end(self, pos: int, line_start: int) -> Tuple[int, str, str]:
        """(line_end, jeton de corps, modificateur async/sync) pour une déclaration dont l'en-tête finit en ``pos``."""
        m = _BODY_START_RE.match(self.masked, pos)
        if not m:
            return line_start, "", ""
        tok, mod = m.group("tok"), m.group("mod") or ""
        idx = m.start("tok")
        if tok == ":":  # liste d'initialisation de constructeur → corps réel ensuite
            m2 = re.compile(r"[{;]").search(self.masked, idx + 1)
            if not m2:
                return line_start, tok, mod
            tok, idx = m2.group(), m2.start()
        if tok == "{":
            close = self.layout.brace_close.get(idx)
            return (self.layout.line_of(close) if close is not None else line_start), "{", mod
        if tok in ("=>", "="):
            m3 = _SEMI_RE.search(self.masked, idx)
            return (self.layout.line_of(m3.start()) if m3 else line_start), tok, mod
        return self.layout.line_of(idx), ";", mod

    def _annotations(self, pending: List[int], lineno: int, upto: int) -> List[Tuple[str, str]]:
        """Annotations (nom, texte des arguments) des lignes en attente + début de la ligne courante."""
        items: List[Tuple[str, str]] = []
        for ln in pending:
            for m in _ANNOT_ITEM_RE.finditer(self.code_lines[ln - 1]):
                items.append((m.group(1), m.group(2) or ""))
        for m in _ANNOT_ITEM_RE.finditer(self.code_lines[lineno - 1][:upto]):
            items.append((m.group(1), m.group(2) or ""))
        return items

    def _signature(self, lineno: int, col: int, end_idx: int) -> str:
        start = self._line_index(lineno) + col
        return _collapse(self.code[start:end_idx])

    # ------------------------------------------------------------------ passe principale

    def run(self) -> FileAnalysis:
        self._scan_declarations()
        self._build_models()
        self._scan_imports()
        self._scan_endpoints()
        return FileAnalysis(
            language="dart",
            symbols=self.symbols,
            endpoints=self.endpoints,
            models=self.models,
            imports=self.imports,
            warnings=self.warnings,
        )

    def _scan_declarations(self) -> None:
        pending: List[int] = []
        nlines = len(self.masked_lines)
        skip_until = 0  # dernière ligne d'une annotation multi-ligne en cours
        for ln in range(1, nlines + 1):
            if ln <= skip_until:
                continue
            ml = self.masked_lines[ln - 1]
            if not ml.strip():
                continue
            if _ANNOT_LINE_RE.match(ml):
                pending.append(ln)
                continue
            stripped = ml.lstrip()
            if stripped.startswith("@") and "(" in stripped:
                # annotation dont les arguments s'étalent sur plusieurs lignes : @JsonSerializable(\n …\n)
                open_idx = self._line_index(ln) + ml.index("(")
                close = self.layout.paren_close.get(open_idx)
                if close is not None and self.layout.line_of(close) > ln:
                    end_ln = self.layout.line_of(close)
                    if not self.masked[close + 1:self.layout.line_starts[end_ln] - 1 if end_ln < nlines else len(self.masked)].strip():
                        pending.append(ln)
                        skip_until = end_ln
                        continue
            depth = self.layout.depth_at_line[ln - 1]
            container = self._container_at(ln)
            if depth == 0 or (container is None and not self.layout.balanced and ml[0] not in " \t"):
                self._top_level(ln, ml, pending)
            elif container is not None and depth == container.depth + 1 and self._line_index(ln) >= container.members_from:
                self._member(ln, ml, pending, container)
            pending = []

    def _top_level(self, ln: int, ml: str, pending: List[int]) -> bool:
        first = ml.lstrip().split(" ", 1)[0] if ml.strip() else ""
        if first in ("import", "export", "part"):
            return True
        m = _LIBRARY_RE.match(ml)
        if m:
            self.symbols.append(ExtractedSymbol(name=m.group("name") or "", kind="module", line_start=ln, line_end=ln,
                                                signature=_collapse(self.code_lines[ln - 1]), meta={"declaration": "library"}))
            return True
        m = _CLASS_RE.match(ml)
        if m:
            self._class_like(ln, m, pending)
            return True
        m = _TYPEDEF_RE.match(ml)
        if m:
            m3 = _SEMI_RE.search(self.masked, self._line_index(ln))
            end_idx = m3.start() if m3 else self._line_index(ln) + len(ml)
            self.symbols.append(ExtractedSymbol(
                name=m.group("name"), kind="type", line_start=ln,
                line_end=self.layout.line_of(end_idx) if m3 else ln,
                signature=self._signature(ln, len(ml) - len(ml.lstrip()), end_idx),
                meta={"declaration": "typedef", "exported": not m.group("name").startswith("_"),
                      **({"annotations": [a for a, _ in self._annotations(pending, ln, m.start("name"))]} if pending else {})},
            ))
            return True
        m = _GETTER_RE.match(ml)
        if m:
            self._accessor(ln, m, pending, None)
            return True
        m = _FUNC_RE.match(ml)
        if m and self._function(ln, m, pending, None):
            return True
        m = _VAR_RE.match(ml)
        if m and self._variable(ln, m, pending, None):
            return True
        m = _VAR_OPEN_RE.match(ml)
        if m and self._variable(ln, m, pending, None, open_ended=True):
            return True
        return False

    def _member(self, ln: int, ml: str, pending: List[int], container: _Container) -> bool:
        m = _GETTER_RE.match(ml)
        if m:
            self._accessor(ln, m, pending, container)
            return True
        m = _OPERATOR_RE.match(ml)
        if m:
            self._operator(ln, m, pending, container)
            return True
        m = _FUNC_RE.match(ml)
        if m and self._function(ln, m, pending, container):
            return True
        m = _VAR_RE.match(ml)
        if m and self._variable(ln, m, pending, container):
            return True
        m = _VAR_OPEN_RE.match(ml)
        if m and self._variable(ln, m, pending, container, open_ended=True):
            return True
        return False

    # ------------------------------------------------------------------ classes, mixins, enums, extensions

    def _class_like(self, ln: int, m: "re.Match[str]", pending: List[int]) -> None:
        kw = re.sub(r"\s+", " ", m.group("kw"))
        name = m.group("name")
        rest = m.group("rest") or ""
        if kw.startswith("extension") and name == "on":
            name, rest = None, "on " + rest
        if name is None and not kw.startswith("extension"):
            return  # ``class {`` sans nom : code invalide, rien à citer
        line_idx = self._line_index(ln)
        head_end = line_idx + m.start("end")
        tok = m.group("end")
        if tok == "":  # clauses sur plusieurs lignes : chercher le début du corps plus loin
            m2 = _OPEN_RE.search(self.masked, head_end)
            if m2:
                tok, head_end = m2.group(), m2.start()
                rest = rest + " " + self.masked[line_idx + m.end("rest"):head_end]
        body_open = body_close = -1
        if tok == "{":
            body_open = head_end
            close = self.layout.brace_close.get(head_end)
            if close is None:
                line_end = ln
                body_close = len(self.masked)
            else:
                body_close = close
                line_end = self.layout.line_of(close)
        else:  # `class A = B with C;` ou déclaration tronquée
            m3 = _SEMI_RE.search(self.masked, head_end)
            line_end = self.layout.line_of(m3.start()) if m3 else ln
            if tok == "=" and m3:
                rest = rest + " " + self.masked[head_end + 1:m3.start()]
        clauses = _collapse(rest)
        annots = self._annotations(pending, ln, m.start("mods"))
        mods = (m.group("mods") or "").split()
        meta: Dict[str, Any] = {"declaration": kw, "exported": not (name or "_").startswith("_")}
        if mods:
            meta["modifiers"] = mods
        if "abstract" in mods:
            meta["abstract"] = True
        if annots:
            meta["annotations"] = [a for a, _ in annots]
        if m.group("gen"):
            meta["generics"] = _collapse(m.group("gen"))
        ext = _EXTENDS_RE.search(clauses)
        if ext:
            meta["extends"] = ext.group(1).strip()
        w = _WITH_RE.search(clauses)
        if w:
            meta["with"] = _split_types(w.group(1))
        impl = _IMPLEMENTS_RE.search(clauses)
        if impl:
            meta["implements"] = _split_types(impl.group(1))
        on = _ON_RE.search(clauses)
        if on:
            meta["on"] = _split_types(on.group(1))
        kind = "class"
        if kw == "mixin":
            kind = "trait"
        elif kw == "enum":
            kind = "enum"
        elif kw.startswith("extension"):
            kind = "type"
        base = re.sub(r"<.*", "", meta.get("extends", "")).strip()
        if kind == "class" and (base in _WIDGET_BASES or base.endswith("Widget")):
            meta["widget"] = True
            kind = "component"
        values_end = body_open
        if kind == "enum" and body_open >= 0:
            body = self.masked[body_open + 1:body_close]
            semi = body.find(";")
            values_end = body_close if semi < 0 else body_open + 1 + semi
            meta["values"] = [v for v in _ENUM_VALUE_RE.findall(_collapse(body[:semi] if semi >= 0 else body, 100000))]
        display = name or ("extension on %s" % " ".join(meta.get("on", ["?"])))
        sig_end = head_end if tok in ("{", "=") else head_end
        symbol = ExtractedSymbol(name=display, kind=kind, line_start=ln, line_end=line_end,
                                 signature=self._signature(ln, m.start("mods"), sig_end), meta=meta)
        self.symbols.append(symbol)
        container = _Container(symbol, kw, self.layout.depth_at_line[ln - 1], body_open, body_close)
        container.members_from = values_end
        for aname, args in annots:
            bm = re.search(r"baseUrl\s*:\s*" + _STRING_CAPTURE, args)
            if bm and aname in ("RestApi", "ChopperApi"):
                container.base_url = bm.group("s")
        if body_open >= 0:
            self.containers.append(container)
            self._container_opens.append(body_open)

    # ------------------------------------------------------------------ fonctions, méthodes, accesseurs

    def _function(self, ln: int, m: "re.Match[str]", pending: List[int], container: Optional[_Container]) -> bool:
        name = m.group("name")
        ret = m.group("ret")
        acc = m.group("acc")
        mods = (m.group("mods") or "").split()
        if ret == "set" and acc is None:
            acc, ret = "set", None
        if ret and ret.split("<", 1)[0].rstrip("?") in _NAME_BLACKLIST:
            return False
        if ret in ("return", "await", "throw", "new", "yield"):
            return False
        base_name = name.split(".", 1)[0]
        if base_name in _NAME_BLACKLIST or (ret is None and acc is None and base_name in _KEYWORDS and base_name != "main"):
            return False
        is_ctor = False
        if "." in name:
            if container is None or base_name != container.name:
                return False  # appel qualifié (ex. http.get(...)) : pas une déclaration
            is_ctor = True
            name = name.split(".", 1)[1]
        elif container is not None and name == container.name and ret is None:
            is_ctor = True
        line_idx = self._line_index(ln)
        open_idx = line_idx + m.end() - 1
        close = self.layout.paren_close.get(open_idx)
        if close is None:
            close = open_idx + len(ml_rest(self.masked_lines[ln - 1], m.end()))
            self.warnings.append("parenthèses non fermées ligne %d (%s)" % (ln, name))
        line_end, tok, mod = self._body_end(close + 1, ln)
        if tok == "" and not ("abstract" in mods or "external" in mods):
            return False
        params = _parse_params(self.masked[open_idx + 1:close], self.code[open_idx + 1:close])
        annots = self._annotations(pending, ln, m.start("mods"))
        meta: Dict[str, Any] = {
            "exported": not name.startswith("_"),
            "async": mod.startswith("async"),
        }
        if mod.endswith("*"):
            meta["generator"] = True
        if ret:
            meta["return_type"] = ret
        if params:
            meta["params"] = [p["name"] for p in params]
        if annots:
            meta["annotations"] = [a for a, _ in annots]
        if "static" in mods:
            meta["static"] = True
        if tok == ";" and not is_ctor and "external" not in mods:
            meta["abstract"] = True
        if is_ctor:
            meta["constructor"] = True
            if "factory" in mods:
                meta["factory"] = True
            if "const" in mods:
                meta["const"] = True
        if acc == "set":
            kind = "property"
            meta["accessor"] = "set"
        else:
            kind = "method" if container is not None else "function"
        body_idx = self._body_index(close + 1)
        symbol = ExtractedSymbol(name=name, kind=kind, line_start=ln, line_end=line_end,
                                 signature=self._signature(ln, m.start("mods"), body_idx),
                                 parent=container.name if container else None, meta=meta)
        self.symbols.append(symbol)
        self._register_function(symbol, container, annots)
        if container is not None:
            container.methods.append(symbol)
            if is_ctor:
                container.constructors.append({"name": name, "params": params, "symbol": symbol})
        return True

    def _operator(self, ln: int, m: "re.Match[str]", pending: List[int], container: _Container) -> None:
        line_idx = self._line_index(ln)
        open_idx = line_idx + m.end() - 1
        close = self.layout.paren_close.get(open_idx, open_idx)
        line_end, tok, mod = self._body_end(close + 1, ln)
        name = "operator " + m.group("op")
        annots = self._annotations(pending, ln, m.start("op") - len("operator"))
        meta: Dict[str, Any] = {"exported": True, "async": False, "operator": True}
        if m.group("ret"):
            meta["return_type"] = m.group("ret")
        if annots:
            meta["annotations"] = [a for a, _ in annots]
        symbol = ExtractedSymbol(name=name, kind="method", line_start=ln, line_end=line_end,
                                 signature=self._signature(ln, len(m.group("indent")), self._body_index(close + 1)),
                                 parent=container.name, meta=meta)
        self.symbols.append(symbol)
        self._register_function(symbol, container, annots)
        container.methods.append(symbol)

    def _accessor(self, ln: int, m: "re.Match[str]", pending: List[int], container: Optional[_Container]) -> None:
        name = m.group("name")
        line_idx = self._line_index(ln)
        line_end, tok, mod = self._body_end(line_idx + m.start("tok"), ln)
        mods = (m.group("mods") or "").split()
        annots = self._annotations(pending, ln, m.start("mods"))
        meta: Dict[str, Any] = {"exported": not name.startswith("_"), "async": mod.startswith("async"), "accessor": "get"}
        if m.group("ret"):
            meta["return_type"] = m.group("ret")
        if "static" in mods:
            meta["static"] = True
        if tok == ";":
            meta["abstract"] = True
        if annots:
            meta["annotations"] = [a for a, _ in annots]
        symbol = ExtractedSymbol(name=name, kind="property", line_start=ln, line_end=line_end,
                                 signature=self._signature(ln, m.start("mods"), self._body_index(line_idx + m.start("tok"))),
                                 parent=container.name if container else None, meta=meta)
        self.symbols.append(symbol)
        self._register_function(symbol, container, annots)
        if container is not None:
            container.methods.append(symbol)

    def _body_index(self, pos: int) -> int:
        m = _BODY_START_RE.match(self.masked, pos)
        return m.start("tok") if m else pos

    def _register_function(self, symbol: ExtractedSymbol, container: Optional[_Container],
                           annots: List[Tuple[str, str]]) -> None:
        label = "%s@%s" % (container.name, symbol.name) if container else symbol.name
        self.func_spans.append((symbol.line_start, symbol.line_end, label))
        if annots:
            self.annotated.append((symbol, container, annots))

    # ------------------------------------------------------------------ variables et champs

    def _variable(self, ln: int, m: "re.Match[str]", pending: List[int], container: Optional[_Container],
                  open_ended: bool = False) -> bool:
        """Déclaration de variable/champ ; ``open_ended`` : l'initialiseur continue sur les lignes suivantes."""
        mods = (m.group("mods") or "").split()
        typ = (m.group("type") or "").strip()
        if not mods and not typ:
            return False
        if typ and typ.split("<", 1)[0].rstrip("?") in _KEYWORDS:
            return False
        names = [n.strip() for n in m.group("names").split(",")]
        if any(n in _NAME_BLACKLIST for n in names):
            return False
        line_idx = self._line_index(ln)
        default = None
        default_masked = ""
        if open_ended:
            tail = m.group("default").rstrip()
            if tail and tail[-1] not in _OPEN_TAIL_CHARS:
                return False
            default_start = line_idx + m.start("default")
            search_from = default_start
            if tail and tail[-1] in "{(":
                # le ``;`` cherché est celui qui suit la fermeture du bloc/appel ouvert en fin de ligne
                bracket_idx = line_idx + len(m.group(0).rstrip()) - 1
                close = (self.layout.brace_close if tail[-1] == "{" else self.layout.paren_close).get(bracket_idx)
                if close is not None:
                    search_from = close + 1
            m3 = _SEMI_RE.search(self.masked, search_from)
            if not m3:
                return False
            line_end = self.layout.line_of(m3.start())
            default_masked = self.masked[default_start:m3.start()]
            default = _collapse(self.code[default_start:m3.start()], 120) or None
            sig_end = line_idx + m.group(0).rfind("=", m.end("names"), m.start("default"))
        else:
            if m.group("default") is not None:
                default_masked = m.group("default")
                default = _collapse(self.code_lines[ln - 1][m.start("default"):m.end("default")], 120) or None
            m3 = _SEMI_RE.search(self.masked, line_idx + m.start("names"))
            line_end = self.layout.line_of(m3.start()) if m3 else ln
            sig_end = line_idx + (self.masked_lines[ln - 1].rfind("=", m.end("names"), m.start("default")) if m.group("default") is not None else m.end("names"))
        annots = self._annotations(pending, ln, m.start("mods"))
        closure = _CLOSURE_RE.match(default_masked) if default_masked else None
        for name in names:
            info = {
                "name": name, "type": typ.rstrip("?") if typ else "", "nullable": (typ.endswith("?") if typ else None),
                "default": default, "mods": mods, "annotations": annots, "line": ln,
            }
            if container is not None:
                container.fields.append(info)
            is_const = "const" in mods
            if container is None or (is_const and "static" in mods) or closure:
                kind = "constant" if is_const else "variable"
                meta: Dict[str, Any] = {"exported": not name.startswith("_"), "type": typ, "modifiers": mods}
                if annots:
                    meta["annotations"] = [a for a, _ in annots]
                if closure:
                    # closure nommée : citable comme fonction et englobante pour les appels HTTP
                    kind = "function" if container is None else "method"
                    meta["closure"] = True
                    meta["async"] = (closure.group("mod") or "").startswith("async")
                symbol = ExtractedSymbol(name=name, kind=kind, line_start=ln, line_end=line_end,
                                         signature=self._signature(ln, m.start("mods"), sig_end),
                                         parent=container.name if container else None, meta=meta)
                self.symbols.append(symbol)
                if closure:
                    self._register_function(symbol, container, [])
        return True


def ml_rest(line: str, col: int) -> str:
    """Reste de ligne à partir d'une colonne (secours quand une parenthèse n'est pas fermée)."""
    return line[col:]


# --------------------------------------------------------------------------- modèles de données

_JSON_INDEX_RE = re.compile(r"\b[A-Za-z_$][\w$]*\s*\[\s*(['\"])([^'\"\n]+)\1\s*\]")
_MAP_KEY_RE = re.compile(r"(['\"])([A-Za-z_][\w.\-]*)\1\s*:")
_JSONKEY_NAME_RE = re.compile(r"name\s*:\s*" + _STRING_CAPTURE)
_MODEL_ANNOTATIONS = frozenset("JsonSerializable freezed Freezed unfreezed".split())


def _unique(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


class _ModelMixin:
    """Détection des classes « données » (mélangé dans l'analyseur pour la lisibilité)."""

    def _keys_in_range(self: Any, symbol: Optional[ExtractedSymbol], pattern: "re.Pattern[str]", group: int) -> List[str]:
        if symbol is None:
            return []
        keys: List[str] = []
        for ln in range(symbol.line_start, min(symbol.line_end, len(self.code_lines)) + 1):
            keys.extend(m.group(group) for m in pattern.finditer(self.code_lines[ln - 1]))
        return _unique(keys)

    def _build_models(self: Any) -> None:
        for c in self.containers:
            if c.kw != "class":
                continue
            sym = c.symbol
            meta = sym.meta
            if meta.get("widget"):
                continue
            supers = [re.sub(r"<.*", "", meta.get("extends", ""))] + [re.sub(r"<.*", "", t) for t in meta.get("implements", [])]
            if any(s in ("Exception", "Error") or s.endswith("Exception") or s.endswith("Error") for s in supers):
                continue
            annots = meta.get("annotations", [])
            ctor_names = [k["name"] for k in c.constructors]
            from_json = None
            for m_ in c.methods:
                if m_.name in ("fromJson", "fromMap") and (m_.meta.get("constructor") or m_.meta.get("static")):
                    from_json = m_
                    break
            to_json = next((m_ for m_ in c.methods if m_.name in ("toJson", "toMap")), None)
            has_annot = any(a in _MODEL_ANNOTATIONS for a in annots)
            instance_fields = [f for f in c.fields if "static" not in f["mods"]]
            all_final = bool(instance_fields) and all("final" in f["mods"] or "const" in f["mods"] for f in instance_fields)
            # les constructeurs, les méthodes « données » et les getters (valeurs dérivées) ne disqualifient pas
            other_methods = [m_ for m_ in c.methods
                             if not m_.meta.get("constructor") and m_.kind != "property"
                             and m_.name.replace("operator ", "") not in _DATA_METHODS]
            plain_data = all_final and bool(c.constructors) and not other_methods and not meta.get("abstract")
            if from_json is not None:
                reason = "from_json"
            elif has_annot:
                reason = "annotation"
            elif plain_data:
                reason = "final_fields"
            else:
                continue
            fields: List[ModelField] = []
            for f in instance_fields:
                fmeta: Dict[str, Any] = {"final": "final" in f["mods"]}
                if "late" in f["mods"]:
                    fmeta["late"] = True
                for aname, args in f["annotations"]:
                    if aname == "JsonKey":
                        km = _JSONKEY_NAME_RE.search(args)
                        if km:
                            fmeta["json_key"] = km.group("s")
                fields.append(ModelField(name=f["name"], type=f["type"], nullable=f["nullable"], default=f["default"], meta=fmeta))
            if not fields:
                # freezed / classes sans champs explicites : les paramètres du constructeur décrivent les données
                for ctor in c.constructors:
                    typed = [p for p in ctor["params"] if p["type"] and not p["this"]]
                    if typed:
                        for p in typed:
                            t = p["type"]
                            fmeta = {"final": True, "required": p["required"]}
                            if p["named"]:
                                fmeta["named"] = True
                            fields.append(ModelField(name=p["name"], type=t.rstrip("?"), nullable=t.endswith("?"),
                                                     default=p["default"], meta=fmeta))
                        break
            json_keys = self._keys_in_range(from_json, _JSON_INDEX_RE, 2)
            if not json_keys and to_json is not None:
                json_keys = self._keys_in_range(to_json, _MAP_KEY_RE, 2)
            if not json_keys:
                json_keys = [f.meta["json_key"] for f in fields if f.meta.get("json_key")]
            mmeta: Dict[str, Any] = {
                "reason": reason,
                "json_keys": json_keys,
                "has_from_json": from_json is not None,
                "has_to_json": to_json is not None,
                "constructors": _unique(ctor_names),
            }
            if annots:
                mmeta["annotations"] = annots
            for key in ("extends", "with", "implements"):
                if meta.get(key):
                    mmeta[key] = meta[key]
            if "freezed" in annots or "Freezed" in annots:
                mmeta["freezed"] = True
            if "JsonSerializable" in annots:
                mmeta["json_serializable"] = True
            self.models.append(ExtractedModel(name=sym.name, kind="dart_class", fields=fields, relations=[],
                                              line=sym.line_start, meta=mmeta))


# --------------------------------------------------------------------------- imports

_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*:")


class _ImportMixin:
    def _scan_imports(self: Any) -> None:
        n = len(self.code_lines)
        ln = 1
        while ln <= n:
            s = self.masked_lines[ln - 1].lstrip()
            if s.startswith(("import ", "export ", "part ", "import'", "import\"", "export'", "export\"", "part'", "part\"")):
                text = self.code_lines[ln - 1]
                k = ln
                while ";" not in self.masked_lines[k - 1] and k < n and k - ln < 12:
                    k += 1
                    text += " " + self.code_lines[k - 1]
                m = _IMPORT_RE.match(text)
                if m:
                    module = m.group("s")
                    names: List[str] = []
                    if m.group("alias"):
                        names = [m.group("alias")]
                    else:
                        sm = _SHOW_RE.search(m.group("combinators") or "")
                        if sm:
                            names = [x.strip() for x in sm.group(1).split(",") if x.strip()]
                    self.imports.append(ExtractedImport(module=module, names=names, line=ln,
                                                        is_relative=not _SCHEME_RE.match(module)))
                else:
                    m2 = _PART_OF_NAME_RE.match(text)
                    if m2:
                        self.imports.append(ExtractedImport(module=m2.group("name"), names=[], line=ln, is_relative=False))
                ln = k
            ln += 1


# --------------------------------------------------------------------------- endpoints

# Ancré sur le point (jamais sur un identifiant optionnel : cela ferait rétro-dérouler sur tout le fichier) ;
# le receveur est lu ensuite dans une fenêtre bornée de 200 caractères à gauche.
_CALL_RE = re.compile(
    r"(?P<dots>\.\.|\.)\s*"
    r"(?P<verb>getUri|postUri|putUri|patchUri|deleteUri|headUri|get|post|put|patch|delete|head|readBytes|read|request|download|all|mount)"
    r"\s*(?:<(?:[^<>()\n]|<[^<>()\n]{0,80}>){0,120}>)?\s*\("  # dio.get<Map<String, dynamic>>(…)
)
_PREV_IDENT_RE = re.compile(r"(?<![\w$.])(" + _QIDENT + r"(?:\(\s*\))?)\s*$")
_URI_HTTPS_RE = re.compile(r"Uri\.(?P<scheme>https?)\s*\(\s*(?P<q1>['\"])(?P<host>[^'\"\n]*)(?P=q1)\s*,\s*(?P<q2>['\"])(?P<path>[^'\"\n]*)(?P=q2)")
_URI_PARSE_RE = re.compile(r"\bUri\.parse\s*\(")
_URI_CTOR_RE = re.compile(r"\bUri\s*\(")
# http.Request('POST', uri) / http.MultipartRequest('POST', Uri.parse(…)) / StreamedRequest
_REQUEST_CTOR_RE = re.compile(r"\b(?:Multipart|Streamed)?Request\s*\(\s*r?(['\"])(?P<m>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\1\s*,")
# Instances nommées : ``final dio = Dio()`` / ``Dio client = Dio()`` / ``final client = http.Client()``
_DIO_NAME_RE = re.compile(r"\b(?:Dio\s+(" + _IDENT + r")\s*[=;]|(" + _IDENT + r")\s*=\s*Dio\s*[.(])")
_HTTP_NAME_RE = re.compile(r"\b(?:(?:http\.)?Client\s+(" + _IDENT + r")\s*[=;]|(" + _IDENT + r")\s*=\s*(?:http\.)?(?:Client|IOClient|RetryClient)\s*\()")
_NAMED_STRING_RE = r"\b%s\s*:\s*r?(['\"])(?P<s>[^'\"\n]*)\1"
_LEADING_STRING_RE = re.compile(r"^\s*r?(['\"])(?P<s>[^'\"\n]*)\1")
_CONCAT_RE = re.compile(r"^\s*(?P<var>" + _QIDENT + r")\s*\+\s*r?(['\"])(?P<s>[^'\"\n]*)\2")
_LEADING_IDENT_RE = re.compile(r"^\s*(?P<var>" + _QIDENT + r")\s*(?:,|\)|$)")
_METHOD_OPT_RE = re.compile(r"\bmethod\s*:\s*(?:r?(['\"])(?P<s>[A-Za-z]+)\1|HttpMethod\.(?P<m>[a-z]+))")
_URI_ANY_RE = re.compile(r"\bUri\.(?:parse|https|http)\s*\(")
_BASEURL_RE = re.compile(r"\bbaseUrl\s*[:=]\s*(?:Uri\.parse\s*\(\s*)?r?(['\"])(?P<s>[^'\"\n]*)\1")
_HTTPMETHOD_RE = re.compile(r"HttpMethod\.(get|post|put|patch|delete|head|options)\b")
_SERVER_RECEIVERS = frozenset("router app _router _app routes pipeline handler".split())
_VERB_METHODS = {
    "get": "GET", "getUri": "GET", "read": "GET", "readBytes": "GET", "download": "GET",
    "post": "POST", "postUri": "POST", "put": "PUT", "putUri": "PUT", "patch": "PATCH", "patchUri": "PATCH",
    "delete": "DELETE", "deleteUri": "DELETE", "head": "HEAD", "headUri": "HEAD", "all": "ANY", "mount": "ANY",
    "request": "ANY",
}


def _base_of(path: str) -> Optional[str]:
    """Base visible en tête d'un chemin : ``${Env.api}``, ``$base``, ou ``https://host``."""
    m = re.match(r"\$\{([^}]*)\}", path)
    if m:
        return m.group(1).strip()
    m = re.match(r"\$([A-Za-z_$][\w$.]*)", path)
    if m:
        return m.group(1)
    m = re.match(r"([a-z][a-z0-9+.\-]*://[^/\s?#]*)", path, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _looks_like_path(s: str) -> bool:
    return s.startswith(("/", "$", "http://", "https://", "ws://", "wss://")) or "/" in s


class _EndpointMixin:
    def _enclosing_span(self: Any, lineno: int) -> Optional[Tuple[int, int, str]]:
        """Fonction/méthode englobant la ligne (les fonctions nommées ne s'imbriquent pas : la dernière commencée suffit)."""
        if self._span_starts is None or len(self._span_starts) != len(self.func_spans):
            self._span_starts = [sp[0] for sp in self.func_spans]
        i = bisect.bisect_right(self._span_starts, lineno) - 1
        for j in range(i, max(-1, i - 3), -1):  # tolérance : quelques déclarations sur une même ligne
            sp = self.func_spans[j]
            if sp[0] <= lineno <= sp[1]:
                return sp
        return None

    def _enclosing(self: Any, lineno: int) -> str:
        sp = self._enclosing_span(lineno)
        if sp:
            return sp[2]
        c = self._container_at(lineno)
        return c.name if c else ""

    def _in_code(self: Any, idx: int) -> bool:
        """Vrai si l'index n'est ni dans une chaîne ni dans un commentaire."""
        return idx < len(self.masked) and self.masked[idx] == self.code[idx]

    def _extract_url(self: Any, args: str, lineno: int, base_idx: int, depth: int = 0) -> Optional[Dict[str, Any]]:
        """Trouve l'URL/le chemin dans un texte d'arguments ; ``base_idx`` = index absolu de ``args`` dans le code."""
        if depth > 3:
            return None
        args = args[:_MAX_ARGS_LEN]
        m = _URI_HTTPS_RE.search(args)
        if m:
            self.consumed_uri.add(base_idx + m.start())
            return {"path": m.group("path"), "host": m.group("host"), "scheme": m.group("scheme"), "source": "Uri." + m.group("scheme")}
        m = _URI_PARSE_RE.search(args)
        if m:
            # Uri.parse(<expr>) : on résout l'expression intérieure (chaîne, base + '/x', variable…)
            self.consumed_uri.add(base_idx + m.start())
            open_abs = base_idx + m.end() - 1
            close_abs = self.layout.paren_close.get(open_abs)
            inner_end = close_abs if close_abs is not None else base_idx + len(args)
            inner = self.code[open_abs + 1:min(inner_end, open_abs + 1 + _MAX_ARGS_LEN)]
            info = self._extract_url(inner, lineno, open_abs + 1, depth + 1)
            if info and info.get("source") in ("string", "concat"):
                info["source"] = "Uri.parse"
            return info
        m = _URI_CTOR_RE.search(args)
        if m:
            pm = re.search(_NAMED_STRING_RE % "path", args)
            if pm:
                info: Dict[str, Any] = {"path": pm.group("s"), "source": "Uri()"}
                hm = re.search(_NAMED_STRING_RE % "host", args)
                if hm:
                    info["host"] = hm.group("s")
                return info
        m = _LEADING_STRING_RE.match(args)
        if m:
            return {"path": m.group("s"), "source": "string"}
        m = _CONCAT_RE.match(args)
        if m:
            return {"path": m.group("s"), "base": m.group("var"), "source": "concat"}
        m = _LEADING_IDENT_RE.match(args)
        if m:
            return self._resolve_var(m.group("var"), lineno, depth)
        return None

    def _resolve_var(self: Any, name: str, lineno: int, depth: int) -> Optional[Dict[str, Any]]:
        """Cherche ``name = <expr>`` en remontant dans la fonction englobante, puis dans tout le fichier."""
        short = name.rsplit(".", 1)[-1]
        pat = re.compile(r"\b" + re.escape(short) + r"\s*=\s*(?P<rhs>.+)$")
        sp = self._enclosing_span(lineno)
        lo = max(sp[0] if sp else 1, lineno - _MAX_LOOKBACK)
        candidates = list(range(lineno, lo - 1, -1))
        if self._assign_index is None:  # index global construit une seule fois (regex en C)
            self._assign_index = {}
            for gm in re.finditer(r"\b(" + _IDENT + r")\s*=[^=]", self.masked):
                self._assign_index.setdefault(gm.group(1), []).append(self.layout.line_of(gm.start()))
        glob = self._assign_index.get(short, [])
        i = bisect.bisect_right(glob, lineno)
        for ln in glob[max(0, i - 4):i + 4]:  # affectations les plus proches, avant puis après
            if ln not in candidates:
                candidates.append(ln)
        for ln in candidates:
            mline = self.masked_lines[ln - 1]
            if short not in mline:
                continue
            m = pat.search(mline)
            if not m or mline[m.start()] != self.code_lines[ln - 1][m.start()]:
                continue
            rhs = self.code_lines[ln - 1][m.start("rhs"):]
            if ln != lineno and not rhs.strip():
                continue
            info = self._extract_url(rhs, ln, self._line_index(ln) + m.start("rhs"), depth + 1)
            if info:
                info["source"] = "variable:%s" % name
                return info
        return None

    def _emit_client(self: Any, method: str, info: Dict[str, Any], lineno: int, framework: str, extra: Dict[str, Any]) -> None:
        path = info.get("path", "")
        meta: Dict[str, Any] = {}
        base = info.get("base") or _base_of(path)
        if not base and info.get("host"):
            base = "%s://%s" % (info.get("scheme") or "https", info["host"])
        if base:
            meta["base"] = base
        elif self.file_base and framework == "dio":
            meta["base"] = self.file_base
        for key in ("host", "scheme", "source"):
            if info.get(key):
                meta[key] = info[key]
        meta.update(extra)
        self.endpoints.append(ExtractedEndpoint(side="client", method=method, path=path, handler=self._enclosing(lineno),
                                                framework=framework, line=lineno, meta=meta))

    def _scan_endpoints(self: Any) -> None:
        bm = _BASEURL_RE.search(self.code)
        if bm and self._in_code(bm.start()):
            self.file_base = bm.group("s")
        # noms des instances Dio / http.Client déclarées dans le fichier (pour qualifier le framework)
        self._dio_names = {a or b for a, b in _DIO_NAME_RE.findall(self.masked)}
        self._http_names = {a or b for a, b in _HTTP_NAME_RE.findall(self.masked)}
        self._scan_calls()
        self._scan_request_ctors()
        self._scan_annotated()
        self._scan_standalone_uris()
        self._scan_dart_frog()

    def _scan_request_ctors(self: Any) -> None:
        """``http.Request('POST', uri)`` et variantes Multipart/Streamed."""
        seen = 0
        for m in _REQUEST_CTOR_RE.finditer(self.code):
            seen += 1
            if seen > _MAX_CALLS:
                break
            if not self._in_code(m.start()):
                continue
            open_idx = self.code.index("(", m.start())
            close = self.layout.paren_close.get(open_idx)
            if close is None:
                continue
            args = self.code[m.end():min(close, m.end() + _MAX_ARGS_LEN)]
            lineno = self.layout.line_of(m.start())
            info = self._extract_url(args, lineno, m.end())
            if not info or not info.get("path"):
                continue
            self._emit_client(m.group("m"), info, lineno, "http", {"call": "Request"})

    def _scan_calls(self: Any) -> None:
        seen = 0
        for m in _CALL_RE.finditer(self.code):
            seen += 1
            if seen > _MAX_CALLS:
                self.warnings.append("trop de sites d'appel HTTP : analyse des endpoints tronquée")
                break
            vstart = m.start("verb")
            if not self._in_code(vstart):
                continue
            verb = m.group("verb")
            recv = ""
            if m.group("dots") == ".":
                window = self.code[max(0, m.start() - 200):m.start()]
                pm = _PREV_IDENT_RE.search(window)
                recv = pm.group(1) if pm else ""
                if pm and pm.start() > 0 and window[pm.start() - 1] == "@":
                    continue  # annotation ``@Route.get('/x')`` : traitée par _scan_annotated
            lineno = self.layout.line_of(vstart)
            open_idx = m.end() - 1
            close = self.layout.paren_close.get(open_idx)
            if close is None:
                nl = self.code.find("\n", open_idx)
                close = len(self.code) if nl < 0 else nl
            args = self.code[open_idx + 1:min(close, open_idx + 1 + _MAX_ARGS_LEN)]
            recv_name = re.sub(r"\(\s*\)$", "", recv)
            low = recv_name.lower()
            is_server = m.group("dots") == ".." or low in _SERVER_RECEIVERS or low.endswith("router")
            if is_server:
                sm = _LEADING_STRING_RE.match(args)
                if not sm:
                    continue
                rest = args[sm.end():].lstrip(" ,\t\n")
                hm = re.match(_QIDENT + r"\b(?!\s*\()", rest)
                handler = hm.group(0) if hm else self._enclosing(lineno)
                self.endpoints.append(ExtractedEndpoint(
                    side="server", method=_VERB_METHODS.get(verb, "ANY"), path=sm.group("s"), handler=handler,
                    framework="shelf", line=lineno, meta={"receiver": recv_name or ".."},
                ))
                continue
            if verb in ("all", "mount"):
                continue
            info = self._extract_url(args, lineno, open_idx + 1)
            if not info:
                continue
            path = info.get("path", "")
            short = recv_name.rsplit(".", 1)[-1]
            dio_like = "dio" in low or verb.endswith("Uri") or verb in ("request", "download") or short in self._dio_names
            http_like = (low in ("http", "client", "_client") or low.endswith("client") or "http" in low
                         or short in self._http_names)
            if not (dio_like or http_like or "api" in low or "rest" in low or "service" in low) and not _looks_like_path(path):
                continue
            method = _VERB_METHODS.get(verb, "ANY")
            if verb == "request":
                om = _METHOD_OPT_RE.search(args)
                if om:
                    method = (om.group("s") or om.group("m")).upper()
            framework = "dio" if dio_like else "http"
            self._emit_client(method, info, lineno, framework, {"receiver": recv_name, "call": verb})

    def _scan_annotated(self: Any) -> None:
        for symbol, container, annots in self.annotated:
            label = "%s@%s" % (container.name, symbol.name) if container else symbol.name
            for aname, args in annots:
                up = aname.upper()
                side, framework = "client", "retrofit"
                if up in ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"):
                    method = up
                elif aname == "Method":
                    hm = re.search(r"HttpMethod\.([A-Za-z]+)|(['\"])([A-Za-z]+)\2", args)
                    method = (hm.group(1) or hm.group(3)).upper() if hm else "ANY"
                elif aname.startswith("Route."):
                    side, framework = "server", "shelf"
                    verb = aname.split(".", 1)[1]
                    method = _VERB_METHODS.get(verb, verb.upper() if verb.isalpha() else "ANY")
                else:
                    continue
                pm = re.search(r"\(\s*(?:path\s*:\s*)?r?(['\"])(?P<s>[^'\"\n]*)\1", args)
                if not pm and aname == "Method":
                    pm = re.search(r",\s*r?(['\"])(?P<s>[^'\"\n]*)\1", args)
                path = pm.group("s") if pm else ""
                meta: Dict[str, Any] = {"annotation": aname}
                base = _base_of(path) or (container.base_url if container else None)
                if base:
                    meta["base"] = base
                self.endpoints.append(ExtractedEndpoint(side=side, method=method, path=path, handler=label,
                                                        framework=framework, line=symbol.line_start, meta=meta))

    def _scan_standalone_uris(self: Any) -> None:
        seen = 0
        for m in _URI_ANY_RE.finditer(self.code):
            seen += 1
            if seen > _MAX_CALLS:
                self.warnings.append("trop de constructions Uri : analyse des endpoints tronquée")
                break
            if m.start() in self.consumed_uri or not self._in_code(m.start()):
                continue
            open_idx = m.end() - 1
            close = self.layout.paren_close.get(open_idx)
            if close is None:
                continue
            args = self.code[m.start():min(close + 1, m.start() + _MAX_ARGS_LEN)]
            lineno = self.layout.line_of(m.start())
            info = self._extract_url(args, lineno, m.start())
            if not info or not info.get("path") or not _looks_like_path(info["path"]):
                continue
            self._emit_client("ANY", info, lineno, "http", {"standalone": True})

    def _scan_dart_frog(self: Any) -> None:
        norm = "/" + self.path.replace("\\", "/")
        if "/routes/" not in norm or not norm.endswith(".dart"):
            return
        rel = norm.split("/routes/", 1)[1][:-5]
        segments = [s for s in rel.split("/") if s]
        if not segments or segments[-1].startswith("_"):
            return
        on_request = next((s for s in self.symbols if s.name == "onRequest" and s.kind == "function"), None)
        if on_request is None:
            return
        if segments[-1] == "index":
            segments.pop()
        route = "/" + "/".join(segments)
        methods = sorted({x.upper() for x in _HTTPMETHOD_RE.findall(self.code)}) or ["ANY"]
        for method in methods:
            self.endpoints.append(ExtractedEndpoint(side="server", method=method, path=route, handler="onRequest",
                                                    framework="dart_frog", line=on_request.line_start,
                                                    meta={"derived_from_path": True, "file": self.path}))


class _FullAnalyzer(_ModelMixin, _ImportMixin, _EndpointMixin, _DartAnalyzer):
    """Analyseur complet : déclarations + modèles + imports + endpoints."""


# --------------------------------------------------------------------------- point d'entrée


class DartExtractor:
    """Extracteur enregistré sous le langage ``dart``."""

    languages: List[str] = ["dart"]

    def analyze(self, text: str, path: str = "") -> FileAnalysis:
        """Analyse un fichier Dart ; ne lève jamais : les erreurs deviennent des avertissements."""
        try:
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
            return _FullAnalyzer(text, path or "").run()
        except Exception as exc:  # noqa: BLE001 - robustesse avant tout
            return FileAnalysis(language="dart", warnings=["échec de l'analyse Dart de %s : %r" % (path, exc)])


EXTRACTOR = DartExtractor()
