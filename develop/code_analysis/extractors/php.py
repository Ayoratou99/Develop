"""Extracteur PHP / Laravel (expressions régulières + équilibrage d'accolades, sans dépendance).

Couvre : namespace, classes / interfaces / traits / enums, fonctions, méthodes, propriétés,
constantes ; modèles Eloquent ; migrations (``Schema::create`` / ``Schema::table``) ; routes
Laravel (``Route::``) ; appels HTTP sortants (``Http::``, Guzzle, curl, ``file_get_contents`` ; les
variables PHP du chemin deviennent des paramètres nommés, ``documents/{$id}/`` → ``documents/{id}/``, et un
chemin qui n'est qu'une variable, ``Http::post($url)``, n'est pas indexé) ; imports ``use`` / ``require``. Pour les vues Blade (``*.blade.php``), seuls les appels
``route()`` / ``url()`` / ``axios`` / ``fetch`` des blocs ``<script>`` sont extraits.

Principe : le source est d'abord « masqué » en deux variantes de même longueur que l'original
(``code`` : chaînes, commentaires, heredoc et HTML remplacés par des espaces ; ``noc`` : seuls les
commentaires et le HTML sont masqués). Toutes les positions sont donc communes aux trois textes,
ce qui permet d'équilibrer les accolades / parenthèses sur ``code`` et de lire les valeurs
littérales sur ``noc``. Tous les motifs sont bornés (pas de ``.*`` gourmand sur le fichier).
"""
from __future__ import annotations

import copy
import re
from bisect import bisect_left, bisect_right
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

# --------------------------------------------------------------------------- masquage du source

_NON_NL = re.compile(r"[^\n]")
_OPEN_TAG = re.compile(r"<\?(?:php\b|=)?")
_TOKEN_RE = re.compile(
    r"""(?P<sq>'(?:[^'\\]|\\.)*')
      |(?P<dq>"(?:[^"\\]|\\.)*")
      |(?P<lc>//[^\n]*|\#(?!\[)[^\n]*)
      |(?P<bc>/\*[\s\S]*?(?:\*/|\Z))
      |(?P<hd><<<[ \t]*(?P<hq>['"]?)(?P<hl>[A-Za-z_]\w*)(?P=hq)[ \t]*\r?\n)
      |(?P<html>\?>)
      |(?P<other>[^'"/\#<?]+|.)
    """,
    re.VERBOSE,
)
_HEREDOC_END_CACHE: Dict[str, Any] = {}


def _heredoc_end(label: str) -> Any:
    """Motif (mis en cache) de la ligne de fermeture d'un heredoc/nowdoc : ``\n  LABEL`` (indentation tolérée)."""
    pat = _HEREDOC_END_CACHE.get(label)
    if pat is None:
        if len(_HEREDOC_END_CACHE) > 256:
            _HEREDOC_END_CACHE.clear()
        pat = re.compile(r"\n[ \t]*" + re.escape(label) + r"\b")
        _HEREDOC_END_CACHE[label] = pat
    return pat


def _blank(s: str) -> str:
    """Remplace tout sauf les sauts de ligne par des espaces (conserve longueur et lignes)."""
    return _NON_NL.sub(" ", s)


def _scan(text: str) -> Tuple[str, str, bool, List[str]]:
    """Renvoie ``(code, noc, has_php, notes)`` ; ``code`` et ``noc`` ont la longueur de ``text``.

    Chaque jeton est reconnu à sa position courante (``match``), sans jamais rebalayer le texte :
    un heredoc ou un bloc HTML (``?>``) est fermé par une recherche linéaire de son terminateur et,
    s'il n'en a pas, s'étend jusqu'à la fin du fichier (sémantique PHP) avec une note.
    """
    m = _OPEN_TAG.search(text)
    if m is None:
        b = _blank(text)
        return b, b, False, []
    code: List[str] = []
    noc: List[str] = []
    notes: List[str] = []
    if m.start() > 0:
        pre = _blank(text[: m.start()])
        code.append(pre)
        noc.append(pre)
    pos = m.start()
    n = len(text)
    while pos < n:
        t = _TOKEN_RE.match(text, pos)
        if t is None:  # impossible en théorie (``other`` consomme toujours un caractère) : filet de sécurité
            b = _blank(text[pos:])
            code.append(b)
            noc.append(b)
            break
        kind = t.lastgroup
        s = t.group()
        end = t.end()
        if kind == "other":
            code.append(s)
            noc.append(s)
        elif kind in ("sq", "dq"):
            code.append(s[0] + _blank(s[1:-1]) + s[-1])
            noc.append(s)
        elif kind == "hd":
            cm = _heredoc_end(t.group("hl")).search(text, end - 1)
            if cm is None:
                notes.append("heredoc %s non terminé (ligne %d) : le reste du fichier est ignoré"
                             % (t.group("hl"), text.count("\n", 0, pos) + 1))
                end = n
            else:
                end = cm.end()
            s = text[pos:end]
            code.append(_blank(s))
            noc.append(s)
        elif kind == "html":
            om = _OPEN_TAG.search(text, end)
            end = om.end() if om else n
            b = _blank(text[pos:end])
            code.append(b)
            noc.append(b)
        else:  # commentaires
            b = _blank(s)
            code.append(b)
            noc.append(b)
        pos = end
    return "".join(code), "".join(noc), True, notes


def _pair_index(code: str, oc: str, cc: str) -> Dict[int, int]:
    """Appariement global ``ouvrant → fermant`` en une passe (pile) ; équivalent à un balayage local
    depuis chaque ouvrant, mais en O(1) par requête, sans jamais rebalayer le fichier."""
    out: Dict[int, int] = {}
    stack: List[int] = []
    for m in re.finditer("[%s%s]" % (re.escape(oc), re.escape(cc)), code):
        p = m.start()
        if code[p] == oc:
            stack.append(p)
        elif stack:
            out[stack.pop()] = p
    return out


class _Src:
    """Source masqué + index des lignes et des accolades appariées."""

    def __init__(self, text: str, code: str, noc: str) -> None:
        self.text = text
        self.code = code
        self.noc = noc
        self.starts = [0] + [m.end() for m in re.finditer(r"\n", text)]
        self.nlines = len(self.starts)
        self.bpos: List[int] = []
        self.bdepth: List[int] = []
        self.bmatch: Dict[int, int] = {}
        stack: List[int] = []
        depth = 0
        for m in re.finditer(r"[{}]", code):
            p = m.start()
            if code[p] == "{":
                stack.append(p)
                depth += 1
            else:
                depth -= 1
                if stack:
                    self.bmatch[stack.pop()] = p
            self.bpos.append(p)
            self.bdepth.append(depth)
        self.pmatch = _pair_index(code, "(", ")")
        self.kmatch = _pair_index(code, "[", "]")

    def line(self, off: int) -> int:
        """Numéro de ligne (1-based) de l'offset."""
        return bisect_right(self.starts, off)

    def depth_at(self, off: int) -> int:
        i = bisect_left(self.bpos, off)
        return self.bdepth[i - 1] if i else 0

    def close_of(self, open_pos: int) -> Optional[int]:
        return self.bmatch.get(open_pos)

    def block_end_line(self, open_pos: Optional[int]) -> int:
        """Ligne de l'accolade fermante appariée, sinon dernière ligne du fichier (au mieux)."""
        if open_pos is not None:
            c = self.close_of(open_pos)
            if c is not None:
                return self.line(c)
        return self.nlines

    def match_paren(self, open_pos: int, oc: str = "(", cc: str = ")", limit: int = 0) -> Optional[int]:
        """Position du délimiteur fermant apparié dans ``code`` (délimiteurs hors chaînes/commentaires).

        ``limit`` est conservé pour compatibilité : l'appariement est pré-calculé, la requête est O(1).
        """
        if oc == "(":
            return self.pmatch.get(open_pos)
        if oc == "[":
            return self.kmatch.get(open_pos)
        return self.bmatch.get(open_pos)

    def next_brace(self, pos: int, limit: int = 2000) -> Optional[int]:
        i = self.code.find("{", pos, pos + limit)
        return i if i >= 0 else None


# --------------------------------------------------------------------------- utilitaires

_STRING_RE = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")
_WS = re.compile(r"\s+")


def _strings(raw: str) -> List[str]:
    """Toutes les chaînes littérales d'un extrait (contenu sans guillemets)."""
    out = []
    for m in _STRING_RE.finditer(raw):
        out.append(m.group(1) if m.group(1) is not None else m.group(2))
    return out


def _first_string(raw: str) -> Optional[str]:
    m = _STRING_RE.search(raw)
    if not m:
        return None
    return m.group(1) if m.group(1) is not None else m.group(2)


def _collapse(s: str) -> str:
    return _WS.sub(" ", s).strip()


def _short(name: str) -> str:
    """Dernier segment d'un nom qualifié PHP (``App\\Models\\Order`` → ``Order``)."""
    return name.strip("\\ ").rsplit("\\", 1)[-1]


_SPLIT_RES: Dict[str, Any] = {}


def _split_top(code: str, raw: str, sep: str = ",") -> List[Tuple[str, str]]:
    """Découpe un extrait au niveau 0 (hors parenthèses/crochets/accolades) sur ``sep``.

    ``code`` est la version masquée de ``raw`` (même longueur). Renvoie des couples (code, raw).
    Seuls les délimiteurs sont visités (regex), pas chaque caractère.
    """
    pat = _SPLIT_RES.get(sep)
    if pat is None:
        pat = _SPLIT_RES[sep] = re.compile(r"[()\[\]{}%s]" % re.escape(sep))
    parts: List[Tuple[str, str]] = []
    depth = 0
    start = 0
    for m in pat.finditer(code):
        ch = m.group()
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0:
            i = m.start()
            parts.append((code[start:i], raw[start:i]))
            start = i + 1
    parts.append((code[start:], raw[start:]))
    return [(c.strip(), r.strip()) for c, r in parts if r.strip()]


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


_UNCOUNTABLE = {
    "audio", "bison", "cattle", "chassis", "data", "deer", "education", "equipment", "evidence", "feedback",
    "fish", "furniture", "gold", "hardware", "information", "knowledge", "metadata", "money", "moose", "news",
    "offspring", "police", "rice", "series", "sheep", "software", "species", "swine", "traffic",
}
_PLURAL_RULES: List[Tuple[Any, str]] = [
    (re.compile(r"(s)tatus$", re.I), r"\1tatuses"),
    (re.compile(r"(quiz)$", re.I), r"\1zes"),
    (re.compile(r"^(ox)$", re.I), r"\1en"),
    (re.compile(r"([ml])ouse$", re.I), r"\1ice"),
    (re.compile(r"(matr|vert|ind)(ix|ex)$", re.I), r"\1ices"),
    (re.compile(r"(x|ch|ss|sh)$", re.I), r"\1es"),
    (re.compile(r"([^aeiouy]|qu)y$", re.I), r"\1ies"),
    (re.compile(r"(hive|gulf)$", re.I), r"\1s"),
    (re.compile(r"(?:([^f])fe|([lr])f)$", re.I), r"\1\2ves"),
    (re.compile(r"sis$", re.I), "ses"),
    (re.compile(r"([ti])um$", re.I), r"\1a"),
    (re.compile(r"(c)riterion$", re.I), r"\1riteria"),
    (re.compile(r"(p)erson$", re.I), r"\1eople"),
    (re.compile(r"(m)an$", re.I), r"\1en"),
    (re.compile(r"(c)hild$", re.I), r"\1hildren"),
    (re.compile(r"(f)oot$", re.I), r"\1eet"),
    (re.compile(r"(t)ooth$", re.I), r"\1eeth"),
    (re.compile(r"(g)oose$", re.I), r"\1eese"),
    (re.compile(r"(buffal|her|potat|tomat|volcan)o$", re.I), r"\1oes"),
    (re.compile(r"(alumn|bacill|cact|foc|fung|nucle|radi|stimul|syllab|termin|vir)us$", re.I), r"\1i"),
    (re.compile(r"us$", re.I), "uses"),
    (re.compile(r"(alias)$", re.I), r"\1es"),
    (re.compile(r"(analys|ax|cris|test|thes)is$", re.I), r"\1es"),
    (re.compile(r"s$", re.I), "s"),
    (re.compile(r"$"), "s"),
]


def pluralize(word: str) -> str:
    """Pluriel anglais simplifié (règles usuelles de Laravel/Doctrine + exceptions courantes)."""
    if not word:
        return word
    if word.lower() in _UNCOUNTABLE:
        return word
    for pat, rep in _PLURAL_RULES:
        if pat.search(word):
            return pat.sub(rep, word, count=1)
    return word + "s"


_SINGULAR_RULES: List[Tuple[Any, str]] = [
    (re.compile(r"(p)eople$", re.I), r"\1erson"),
    (re.compile(r"(c)hildren$", re.I), r"\1hild"),
    (re.compile(r"(m)en$", re.I), r"\1an"),
    (re.compile(r"(s)tatuses$", re.I), r"\1tatus"),
    (re.compile(r"(quiz)zes$", re.I), r"\1"),
    (re.compile(r"(matr|vert|ind)ices$", re.I), r"\1ix"),
    (re.compile(r"([ml])ice$", re.I), r"\1ouse"),
    (re.compile(r"(x|ch|ss|sh)es$", re.I), r"\1"),
    (re.compile(r"([^aeiouy]|qu)ies$", re.I), r"\1y"),
    (re.compile(r"([lr])ves$", re.I), r"\1f"),
    (re.compile(r"([^f])ves$", re.I), r"\1fe"),
    (re.compile(r"(alias)es$", re.I), r"\1"),
    (re.compile(r"(analys|ax|cris|test|thes)es$", re.I), r"\1is"),
    (re.compile(r"ses$", re.I), "s"),
    (re.compile(r"([ti])a$", re.I), r"\1um"),
    (re.compile(r"s$", re.I), ""),
]


def singularize(word: str) -> str:
    """Singulier anglais simplifié (paramètre de route des ressources : ``customers`` → ``customer``)."""
    if not word or word.lower() in _UNCOUNTABLE:
        return word
    for pat, rep in _SINGULAR_RULES:
        if pat.search(word):
            return pat.sub(rep, word, count=1)
    return word


def table_name_for(class_name: str) -> str:
    """Nom de table Eloquent par défaut : snake_case pluriel (``OrderLine`` → ``order_lines``)."""
    snake = _snake(class_name)
    head, _, last = snake.rpartition("_")
    plural = pluralize(last)
    return head + "_" + plural if head else plural


def _join_path(*parts: str) -> str:
    """Concatène des segments d'URL en évitant les ``/`` doubles (``'v1'``, ``'/health'`` → ``/v1/health``)."""
    segs = [p.strip("/") for p in parts if p and p.strip("/")]
    return "/" + "/".join(segs)


# --------------------------------------------------------------------------- symboles

# Les motifs commencent par un littéral (ou une alternative de littéraux) pour que ``re`` utilise sa
# recherche rapide de préfixe ; la frontière gauche est vérifiée ensuite par ``_bounded``.
_NS_RE = re.compile(r"namespace\s+([\w\\]+)\s*([;{])")
_CLASS_RE = re.compile(
    r"(class|interface|trait|enum)\s+(\w+)"
    r"(?:\s*:\s*(\w+))?"
    r"(?:\s+extends\s+([\w\\]+(?:\s*,\s*[\w\\]+)*))?"
    r"(?:\s+implements\s+([\w\\]+(?:\s*,\s*[\w\\]+)*))?"
    r"\s*\{"
)
_ANON_RE = re.compile(
    r"new\s+(?:readonly\s+)?class\b(?:\s*\((?:[^()]|\([^()]*\)){0,5000}\))?"
    r"(?:\s+extends\s+([\w\\]+))?(?:\s+implements\s+([\w\\]+(?:\s*,\s*[\w\\]+)*))?\s*\{"
)
_FUNC_RE = re.compile(r"function\s+(&\s*)?(\w+)\s*\(")
_AFTER_PARAMS_RE = re.compile(r"\s*(?::\s*(\??[\w\\]+(?:\s*[|&]\s*\(?\??[\w\\]+\)?)*))?\s*([{;])?")
_MODIFIERS = "public|protected|private|static|abstract|final|readonly|var"
_MODS_TAIL_RE = re.compile(r"((?:(?<![\w$])(?:" + _MODIFIERS + r")\s+)*)$")
_PROP_RE = re.compile(
    r"(public|protected|private|static|readonly|var)\s+((?:(?:public|protected|private|static|readonly|var)\s+)*)"
    r"(\??[\w\\|&]+\s+)?\$(\w+)"
)
_CONST_RE = re.compile(r"const\s+(?:[\w\\|?]+\s+)?(\w+)\s*=")
_DEFINE_RE = re.compile(r"define\s*\(\s*(?:'(\w+)'|\"(\w+)\")")
_CASE_RE = re.compile(r"case\s+(\w+)\s*[=;]")
_TRAIT_USE_RE = re.compile(r"use\s+([\w\\]+(?:\s*,\s*[\w\\]+)*)\s*[;{]")
_USE_RE = re.compile(r"^[ \t]*use\s+(function\s+|const\s+)?(?!\()([^;{}\n]{0,500}?)(\{[^{}]{0,3000}\})?\s*;", re.M)
_PROP_STOP_RE = re.compile(r"[=;,]")
_SIG_MAX = 1500  # longueur maximale d'une signature (fonction non fermée, fichier minifié…)
_REQUIRE_RE = re.compile(r"(?:require|include)(?:_once)?\s*\(?\s*(__DIR__\s*\.\s*)?('[^'\n]*'|\"[^\"\n]*\")")


def _bounded(code: str, pos: int, extra: str = "") -> bool:
    """Vrai si ``pos`` est une frontière gauche de mot : pas de lettre/chiffre/``_``/``$``/``\\`` avant."""
    if pos <= 0:
        return True
    ch = code[pos - 1]
    return not (ch.isalnum() or ch in "_$\\" or (extra and ch in extra))


def _mods_before(code: str, pos: int) -> Tuple[List[str], int]:
    """Modificateurs (public, static, abstract…) précédant ``pos`` → (liste, offset du premier)."""
    m = _MODS_TAIL_RE.search(code, max(0, pos - 100), pos)
    if not m or not m.group(1):
        return [], pos
    return m.group(1).split(), m.start(1)
_CLASS_LIKE_KINDS = ("class", "interface", "trait", "enum")


class _Container:
    """Corps d'une classe/interface/trait/enum : bornes des accolades et profondeur du corps."""

    __slots__ = ("name", "kind", "open", "close", "depth", "symbol")

    def __init__(self, name: str, kind: str, open_pos: int, close: int, depth: int, symbol: ExtractedSymbol) -> None:
        self.name = name
        self.kind = kind
        self.open = open_pos
        self.close = close
        self.depth = depth
        self.symbol = symbol


def _prev_char(code: str, pos: int) -> str:
    """Caractère non blanc précédant ``pos`` (les commentaires sont déjà des espaces)."""
    i = pos - 1
    while i >= 0 and code[i] in " \t\r\n":
        i -= 1
    return code[i] if i >= 0 else ""


class _AnalyzerBase:
    """Analyse d'un fichier PHP : accumule symboles, imports, modèles et endpoints."""

    def __init__(self, text: str, path: str) -> None:
        self.path = path.replace("\\", "/")
        code, noc, self.has_php, self.scan_notes = _scan(text)
        self.src = _Src(text, code, noc)
        self.result = FileAnalysis(language="php")
        self.namespaces: List[List[Any]] = []  # [début, nom, fin]
        self.containers: List[_Container] = []
        self._opens: List[int] = []
        self._cmax: List[int] = []
        self._nstarts: List[int] = []
        self.funcs: List[Tuple[int, int, str, Optional[str]]] = []  # (début, fin, nom, classe parente)
        self._fstarts: List[int] = []
        self._fmax: List[int] = []

    # ----- aides de position (recherche dichotomique + maximum préfixe des fins pour couper court)
    def _ns(self, off: int) -> str:
        i = bisect_right(self._nstarts, off) - 1
        if i >= 0 and off < self.namespaces[i][2]:
            return self.namespaces[i][1]
        return ""

    def _index_containers(self) -> None:
        self.containers.sort(key=lambda c: c.open)
        self._opens = [c.open for c in self.containers]
        self._cmax: List[int] = []
        mx = -1
        for c in self.containers:
            mx = max(mx, c.close)
            self._cmax.append(mx)

    def _container_at(self, off: int) -> Optional[_Container]:
        i = bisect_left(self._opens, off) - 1
        while i >= 0:
            if self._cmax[i] <= off:
                return None
            c = self.containers[i]
            if c.open < off < c.close:
                return c
            i -= 1
        return None

    def _ensure_func_index(self) -> None:
        if len(self._fstarts) != len(self.funcs):
            self._fstarts = [f[0] for f in self.funcs]
            self._fmax = []
            mx = -1
            for f in self.funcs:
                mx = max(mx, f[1])
                self._fmax.append(mx)

    def _funcs_between(self, lo: int, hi: int) -> List[Tuple[int, int, str, Optional[str]]]:
        """Fonctions / méthodes dont le début est dans ``]lo, hi[`` (tranche par dichotomie)."""
        self._ensure_func_index()
        return self.funcs[bisect_right(self._fstarts, lo): bisect_left(self._fstarts, hi)]

    def _func_at(self, off: int) -> Optional[Tuple[int, int, str, Optional[str]]]:
        """Fonction / méthode nommée la plus interne contenant l'offset."""
        self._ensure_func_index()
        i = bisect_right(self._fstarts, off) - 1
        while i >= 0:
            if self._fmax[i] <= off:
                return None
            f = self.funcs[i]
            if f[0] <= off < f[1]:
                return f
            i -= 1
        return None

    def handler_at(self, off: int) -> str:
        f = self._func_at(off)
        if f is None:
            return ""
        return "%s@%s" % (f[3], f[2]) if f[3] else f[2]

    # ----- symboles
    def extract_symbols(self) -> None:
        src = self.src
        code = src.code
        noc = src.noc
        n = len(code)
        for m in _NS_RE.finditer(code):
            if not _bounded(code, m.start()):
                continue
            end = n
            if m.group(2) == "{":
                c = src.close_of(m.end() - 1)
                end = c if c is not None else n
            self.namespaces.append([m.start(), m.group(1), end])
        for i in range(len(self.namespaces) - 1):
            if self.namespaces[i][2] == n:
                self.namespaces[i][2] = self.namespaces[i + 1][0]
        self._nstarts = [ns[0] for ns in self.namespaces]

        syms: List[ExtractedSymbol] = []
        for m in _CLASS_RE.finditer(code):
            if not _bounded(code, m.start(), ":>"):
                continue
            mods_l, start = _mods_before(code, m.start())
            if re.search(r"\bnew\s*$", code[max(0, start - 12): start]):
                continue
            kw, name, backing, extends, implements = m.groups()
            open_pos = m.end() - 1
            meta: Dict[str, Any] = {"namespace": self._ns(m.start())}
            for flag in ("abstract", "final", "readonly"):
                if flag in mods_l:
                    meta[flag] = True
            if extends:
                ex = [e.strip() for e in extends.split(",")]
                meta["extends"] = ex if kw == "interface" else ex[0]
            if implements:
                meta["implements"] = [i.strip() for i in implements.split(",")]
            if backing:
                meta["backing_type"] = backing
            sym = ExtractedSymbol(name=name, kind=kw, line_start=src.line(start),
                                  line_end=src.block_end_line(open_pos),
                                  signature=_collapse(code[start:open_pos]), parent=None, meta=meta)
            syms.append(sym)
            close = src.close_of(open_pos)
            self.containers.append(_Container(name, kw, open_pos, close if close is not None else n,
                                              src.depth_at(open_pos + 1), sym))
        for m in _ANON_RE.finditer(code):
            if not _bounded(code, m.start()):
                continue
            open_pos = m.end() - 1
            meta = {"namespace": self._ns(m.start()), "anonymous": True}
            if m.group(1):
                meta["extends"] = m.group(1)
            if m.group(2):
                meta["implements"] = [i.strip() for i in m.group(2).split(",")]
            sym = ExtractedSymbol(name="class@anonymous", kind="class", line_start=src.line(m.start()),
                                  line_end=src.block_end_line(open_pos),
                                  signature=_collapse(code[m.start():open_pos]), parent=None, meta=meta)
            syms.append(sym)
            close = src.close_of(open_pos)
            self.containers.append(_Container("class@anonymous", "class", open_pos, close if close is not None else n,
                                              src.depth_at(open_pos + 1), sym))
        self._index_containers()

        for m in _FUNC_RE.finditer(code):
            if not _bounded(code, m.start(), ">"):
                continue
            byref, name = m.group(1), m.group(2)
            mods_l, _ = _mods_before(code, m.start())
            paren_open = m.end() - 1
            paren_close = src.match_paren(paren_open)
            ret = None
            term = None
            body_open = None
            if paren_close is None:
                eol = code.find("\n", m.start(), m.start() + _SIG_MAX)
                sig_end = eol if eol >= 0 else min(n, m.start() + _SIG_MAX)
                line_end = src.line(m.start())
            else:
                am = _AFTER_PARAMS_RE.match(code, paren_close + 1)
                ret = am.group(1)
                term = am.group(2)
                sig_end = am.end(1) if ret else paren_close + 1
                if term == "{":
                    body_open = am.end() - 1
                    line_end = src.block_end_line(body_open)
                else:
                    line_end = src.line(am.end() - 1 if term else paren_close)
            cont = self._container_at(m.start())
            if cont is not None and src.depth_at(m.start()) == cont.depth:
                kind, parent = "method", cont.name
            else:
                kind, parent = "function", None
            meta = {"namespace": self._ns(m.start())}
            if kind == "method":
                vis = [v for v in mods_l if v in ("public", "protected", "private")]
                meta["visibility"] = vis[0] if vis else "public"
                meta["static"] = "static" in mods_l
            for flag in ("abstract", "final"):
                if flag in mods_l:
                    meta[flag] = True
            if byref:
                meta["by_ref"] = True
            if ret:
                meta["return_type"] = _collapse(ret)
            syms.append(ExtractedSymbol(name=name, kind=kind, line_start=src.line(m.start()), line_end=line_end,
                                        signature=_collapse(noc[m.start():min(sig_end, m.start() + _SIG_MAX)]),
                                        parent=parent, meta=meta))
            body_end = n
            if body_open is not None:
                c = src.close_of(body_open)
                body_end = c + 1 if c is not None else n
            elif paren_close is not None:
                body_end = paren_close + 1
            self.funcs.append((m.start(), body_end, name, parent))

        for m in _PROP_RE.finditer(code):
            if not _bounded(code, m.start()):
                continue
            cont = self._container_at(m.start())
            if cont is None or src.depth_at(m.start()) != cont.depth:
                continue
            if _prev_char(code, m.start()) not in ("{", "}", ";", "]"):
                continue
            mods_l = [m.group(1)] + m.group(2).split()
            meta = {"namespace": self._ns(m.start())}
            vis = [v for v in mods_l if v in ("public", "protected", "private")]
            meta["visibility"] = vis[0] if vis else "public"
            meta["static"] = "static" in mods_l
            if "readonly" in mods_l:
                meta["readonly"] = True
            if m.group(3):
                meta["type"] = m.group(3).strip()
            stop = _PROP_STOP_RE.search(code, m.end(), m.end() + 500)
            sig_end = stop.start() if stop else m.end()
            syms.append(ExtractedSymbol(name=m.group(4), kind="property", line_start=src.line(m.start()),
                                        line_end=src.line(m.start()), signature=_collapse(code[m.start():sig_end]),
                                        parent=cont.name, meta=meta))

        for m in _CONST_RE.finditer(code):
            if not _bounded(code, m.start(), ">"):
                continue
            cont = self._container_at(m.start())
            if cont is not None and src.depth_at(m.start()) != cont.depth:
                continue
            mods_l, start = _mods_before(code, m.start())
            if _prev_char(code, start) not in ("{", "}", ";", "]", "", ">"):
                continue
            meta = {"namespace": self._ns(m.start())}
            if cont is not None:
                vis = [v for v in mods_l if v in ("public", "protected", "private")]
                meta["visibility"] = vis[0] if vis else "public"
            syms.append(ExtractedSymbol(name=m.group(1), kind="constant", line_start=src.line(start),
                                        line_end=src.line(start), signature=_collapse(code[start:m.end() - 1]),
                                        parent=cont.name if cont else None, meta=meta))
        for m in _DEFINE_RE.finditer(noc):
            if not _bounded(noc, m.start(), ">"):
                continue
            name = m.group(1) or m.group(2)
            syms.append(ExtractedSymbol(name=name, kind="constant", line_start=src.line(m.start()),
                                        line_end=src.line(m.start()), signature="define('%s')" % name,
                                        parent=None, meta={"namespace": self._ns(m.start())}))

        for c in self.containers:
            body = code[c.open + 1: c.close]
            if c.kind == "enum":
                cases = [cm.group(1) for cm in _CASE_RE.finditer(body)
                         if _bounded(body, cm.start()) and src.depth_at(c.open + 1 + cm.start()) == c.depth]
                if cases:
                    c.symbol.meta["cases"] = cases
            if c.kind in ("class", "enum", "trait"):
                traits: List[str] = []
                for tm in _TRAIT_USE_RE.finditer(body):
                    if _bounded(body, tm.start()) and src.depth_at(c.open + 1 + tm.start()) == c.depth:
                        traits.extend(t.strip() for t in tm.group(1).split(","))
                if traits:
                    c.symbol.meta["traits"] = traits

        syms.sort(key=lambda s: s.line_start)
        self.result.symbols.extend(syms)

    # ----- imports
    def extract_imports(self) -> None:
        src = self.src
        imports: List[ExtractedImport] = []
        for m in _USE_RE.finditer(src.code):
            if self._container_at(m.start()) is not None:
                continue  # `use Trait;` dans un corps de classe
            base = _collapse(m.group(2))
            line = src.line(m.start())
            if m.group(3):  # use App\Models\{Order, Customer as C};
                names = []
                for item in m.group(3).strip("{}").split(","):
                    pth = re.split(r"\s+as\s+", item.strip())[0].strip()
                    pth = re.sub(r"^(?:function|const)\s+", "", pth)
                    if pth:
                        names.append(_short(pth))
                imports.append(ExtractedImport(module=base.replace(" ", "").strip("\\"), names=names, line=line,
                                               is_relative=False))
                continue
            for item in base.split(","):  # use A\B, C\D as E;
                pth = re.split(r"\s+as\s+", item.strip())[0].strip().strip("\\")
                if pth:
                    imports.append(ExtractedImport(module=pth, names=[_short(pth)], line=line, is_relative=False))
        for m in _REQUIRE_RE.finditer(src.noc):
            if not _bounded(src.noc, m.start(), ">"):
                continue
            target = m.group(2)[1:-1]
            if m.group(1):
                target = "." + target if target.startswith("/") else "./" + target
            imports.append(ExtractedImport(module=target, names=[], line=src.line(m.start()),
                                           is_relative=target.startswith("./") or target.startswith("../")))
        imports.sort(key=lambda i: i.line)
        self.result.imports.extend(imports)


# --------------------------------------------------------------------------- modèles Eloquent

_ELOQUENT_BASES = {"Model", "Authenticatable", "Pivot", "MorphPivot", "Eloquent"}
_MODEL_PROPS = ("table", "fillable", "guarded", "casts", "hidden", "visible", "dates", "appends", "with",
                "primaryKey", "keyType", "incrementing", "timestamps", "connection")
_MODEL_PROP_RE = re.compile(r"\$(" + "|".join(_MODEL_PROPS) + r")\s*=\s*")
_CASTS_METHOD_RE = re.compile(r"function\s+casts\s*\(\s*\)[^{;]{0,80}\{")
_RETURN_ARRAY_RE = re.compile(r"return\s*(\[|array\s*\()")
_DICT_ITEM_RE = re.compile(
    r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*=>\s*([\w\\]+::class|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[\w\\:.\-]+)"
)
_RELATION_RE = re.compile(
    r"\$this\s*->\s*(belongsTo|hasMany|hasOne|belongsToMany|morphTo|morphMany|morphOne|morphToMany|morphedByMany"
    r"|hasManyThrough|hasOneThrough)\s*\(\s*(?:([\w\\]+)::class|'([^'\n]*)'|\"([^\"\n]*)\")?"
)
_REL_KINDS = {
    "belongsTo": "belongs_to", "hasMany": "has_many", "hasOne": "has_one", "belongsToMany": "many_to_many",
    "morphTo": "morph_to", "morphMany": "morph_many", "morphOne": "morph_one", "morphToMany": "morph_to_many",
    "morphedByMany": "morphed_by_many", "hasManyThrough": "has_many_through", "hasOneThrough": "has_one_through",
}


_SCALAR_RE = re.compile(r"[^;]{0,500}")


def _unquote(raw: str) -> str:
    s = _first_string(raw)
    return s if s is not None else _collapse(raw)


def _parse_dict(raw: str) -> Dict[str, str]:
    """``['a' => 'integer', 'b' => X::class]`` → ``{'a': 'integer', 'b': 'X'}``."""
    out: Dict[str, str] = {}
    for m in _DICT_ITEM_RE.finditer(raw):
        key = m.group(1)[1:-1]
        val = m.group(2)
        if val.endswith("::class"):
            val = _short(val[:-7])
        elif val[:1] in "'\"":
            val = val[1:-1]
        out[key] = val
    return out


class _ModelMixin:
    """Partie Eloquent de l'analyseur (mélangée dans ``_Analyzer`` via héritage ci-dessous)."""

    src: _Src
    result: FileAnalysis
    containers: List[_Container]
    funcs: List[Tuple[int, int, str, Optional[str]]]
    _funcs_between: Any

    def _literal_at(self, pos: int) -> Tuple[Optional[str], int]:
        """Extrait brut (``noc``) de la valeur littérale commençant à ``pos`` : tableau, chaîne ou scalaire."""
        code = self.src.code
        n = len(code)
        while pos < n and code[pos] in " \t\r\n":
            pos += 1
        if pos >= n:
            return None, pos
        ch = code[pos]
        if ch == "[":
            close = self.src.match_paren(pos, "[", "]", 100000)
            if close is None:
                return None, pos
            return self.src.noc[pos: close + 1], close + 1
        if code.startswith("array", pos):
            p = code.find("(", pos, pos + 20)
            if p < 0:
                return None, pos
            close = self.src.match_paren(p, "(", ")", 100000)
            if close is None:
                return None, pos
            return self.src.noc[pos: close + 1], close + 1
        m = _SCALAR_RE.match(code, pos)
        end = m.end() if m else pos
        return self.src.noc[pos:end], end

    def extract_models(self) -> None:
        src = self.src
        for c in self.containers:
            if c.kind != "class" or c.name == "class@anonymous":
                continue
            extends = c.symbol.meta.get("extends") or ""
            body = src.code[c.open + 1: c.close]
            props: Dict[str, str] = {}
            for m in _MODEL_PROP_RE.finditer(body):
                abs_pos = c.open + 1 + m.start()
                if src.depth_at(abs_pos) != c.depth:
                    continue
                raw, _ = self._literal_at(c.open + 1 + m.end())
                if raw is not None:
                    props[m.group(1)] = raw
            inferred = bool(set(props) & {"table", "fillable", "guarded", "casts"})
            if _short(extends) not in _ELOQUENT_BASES and not inferred:
                continue
            meta: Dict[str, Any] = {"namespace": c.symbol.meta.get("namespace", ""), "class": c.name,
                                    "extends": extends}
            if _short(extends) not in _ELOQUENT_BASES:
                meta["inferred"] = True
            casts = _parse_dict(props["casts"]) if "casts" in props else {}
            cm = _CASTS_METHOD_RE.search(body)
            if cm and not casts:
                mclose = src.close_of(c.open + 1 + cm.end() - 1)
                if mclose is not None:
                    rm = _RETURN_ARRAY_RE.search(src.code, c.open + 1 + cm.end(), mclose)
                    if rm:
                        raw, _ = self._literal_at(rm.start(1))
                        if raw:
                            casts = _parse_dict(raw)
            table = _first_string(props["table"]) if "table" in props else None
            fields: List[ModelField] = []
            for name in _strings(props.get("fillable", "")):
                fmeta: Dict[str, Any] = {}
                if name in casts:
                    fmeta["cast"] = casts[name]
                fields.append(ModelField(name=name, type=casts.get(name, ""), meta=fmeta))
            for key in ("guarded", "hidden", "visible", "dates", "appends", "with"):
                if key in props:
                    meta[key] = _strings(props[key])
            if casts:
                meta["casts"] = casts
            if "primaryKey" in props:
                meta["primary_key"] = _unquote(props["primaryKey"])
            if "timestamps" in props:
                meta["timestamps"] = "true" in props["timestamps"].lower()
            if "incrementing" in props:
                meta["incrementing"] = "true" in props["incrementing"].lower()
            for key in ("connection", "keyType"):
                if key in props:
                    meta[_snake(key)] = _unquote(props[key])
            relations: List[ModelRelation] = []
            for start, end, fname, parent in self._funcs_between(c.open, c.close):
                if parent != c.name:
                    continue
                rm = _RELATION_RE.search(src.noc, start, end)
                if not rm:
                    continue
                target = rm.group(2) or rm.group(3) or rm.group(4) or ""
                relations.append(ModelRelation(kind=_REL_KINDS[rm.group(1)], target=_short(target), field=fname))
            self.result.models.append(ExtractedModel(
                name=c.name, kind="eloquent", table_name=table or table_name_for(c.name), fields=fields,
                relations=relations, line=c.symbol.line_start, meta=meta))


# --------------------------------------------------------------------------- migrations

_SCHEMA_RE = re.compile(
    r"Schema\s*::\s*(?:connection\s*\((?:[^()]|\([^()]*\)){0,2000}\)\s*->\s*)?(create|table)\s*\("
)
_BLUEPRINT_PARAM_RE = re.compile(r"(?<![\w$])(?:function|fn)\s*\(\s*(?:\??[\w\\]+\s+)?\$(\w+)")
_ARROW_FN_RE = re.compile(r"(?<![\w$])fn\s*\((?:[^()]|\([^()]*\)){0,500}\)\s*(?::\s*\??[\w\\|]+\s*)?=>")
_DOWN_RE = re.compile(r"function\s+down\s*\(")
_FOREACH_RE = re.compile(r"foreach\s*\(")
_FOREACH_HEAD_RE = re.compile(r"^\s*(.*?)\s+as\s+(?:\$(\w+)\s*=>\s*)?&?\s*\$(\w+)\s*$", re.S)
_ARG_DELIM_RE = re.compile(r"[()\[\]{},]")
_CLASS_CONST_REF_RE = re.compile(r"(?:self|static|parent|[\w\\]+)\s*::\s*(\w+)$")
_THIS_PROP_REF_RE = re.compile(r"\$this\s*->\s*(\w+)$")
_VAR_REF_RE = re.compile(r"\$(\w+)$")
_CHAIN_CALL_RE = re.compile(r"\s*->\s*(\w+)\s*\(")
_INCREMENTS = {"id", "bigIncrements", "increments", "mediumIncrements", "smallIncrements", "tinyIncrements"}
_DEFAULT_NAME = {"id": "id", "uuid": "uuid", "ulid": "ulid", "ipAddress": "ip_address", "macAddress": "mac_address",
                 "rememberToken": "remember_token", "softDeletes": "deleted_at", "softDeletesTz": "deleted_at",
                 "softDeletesDatetime": "deleted_at"}
_TIMESTAMPS = {"timestamps", "nullableTimestamps", "timestampsTz"}
_MORPHS = {"morphs", "nullableMorphs", "uuidMorphs", "nullableUuidMorphs", "ulidMorphs", "nullableUlidMorphs"}
_INDEX_CALLS = {"primary", "unique", "index", "fullText", "spatialIndex", "rawIndex"}
_IGNORED_CALLS = {"engine", "charset", "collation", "comment", "temporary", "dropForeign", "dropIndex",
                  "dropUnique", "dropPrimary", "dropFullText", "dropSpatialIndex", "renameIndex", "foreignUuid"}
_ON_DELETE = {"cascadeOnDelete": "cascade", "nullOnDelete": "set null", "restrictOnDelete": "restrict",
              "noActionOnDelete": "no action"}


_Call = Tuple[str, str, str, int]  # (nom, args_code, args_raw, offset du début des arguments dans code_s)


def _local_close(code_s: str, open_pos: int, end: int) -> Optional[int]:
    """Parenthèse fermante appariée par balayage local borné par ``end``."""
    depth = 0
    for i in range(open_pos, end):
        ch = code_s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _parse_chain(code_s: str, raw_s: str, pos: int, close_fn: Any = None, end: Optional[int] = None) -> List[_Call]:
    """Suite d'appels ``->nom(args)`` à partir de ``pos`` → [(nom, args_code, args_raw, args_start)].

    ``close_fn(ouvrant) -> fermant`` (appariement pré-calculé de ``_Src``) évite tout rebalayage ;
    ``end`` borne la chaîne (fin d'instruction). Un appel non fermé prend le reste de la fenêtre.
    """
    calls: List[_Call] = []
    n = len(code_s) if end is None else min(end, len(code_s))
    while pos < n:
        m = _CHAIN_CALL_RE.match(code_s, pos, n)
        if not m:
            break
        p = m.end() - 1
        close = close_fn(p) if close_fn is not None else _local_close(code_s, p, n)
        if close is None or close >= n:
            calls.append((m.group(1), code_s[p + 1: n], raw_s[p + 1: n], p + 1))
            break
        calls.append((m.group(1), code_s[p + 1: close], raw_s[p + 1: close], p + 1))
        pos = close + 1
    return calls


def _int_arg(raw: str) -> Optional[int]:
    m = re.match(r"\s*(\d+)\s*$", raw)
    return int(m.group(1)) if m else None


class _MigrationMixin:
    """Partie migrations (``Schema::create`` / ``Schema::table``)."""

    src: _Src
    result: FileAnalysis

    def _column_from_chain(self, calls: List[_Call], model: ExtractedModel) -> None:
        name, args_code, args_raw, _ = calls[0]
        parts = _split_top(args_code, args_raw)
        first = _first_string(parts[0][1]) if parts else None
        modifiers = calls[1:]
        fields: List[ModelField] = []
        rel_field: Optional[str] = None
        rel_target: Optional[str] = None
        if name in _INCREMENTS:
            fields.append(ModelField(name=first or "id", type=name, nullable=False,
                                     meta={"primary": True, "auto_increment": True, "unsigned": True}))
        elif name in _TIMESTAMPS:
            for col in ("created_at", "updated_at"):
                fields.append(ModelField(name=col, type="timestamp", nullable=True))
        elif name in _MORPHS:
            base = first or "morph"
            nullable = name.startswith("nullable")
            id_type = "uuid" if "uuid" in name.lower() else ("ulid" if "ulid" in name.lower() else "unsignedBigInteger")
            fields.append(ModelField(name=base + "_type", type="string", nullable=nullable))
            fields.append(ModelField(name=base + "_id", type=id_type, nullable=nullable))
        elif name in _DEFAULT_NAME:
            col = first or _DEFAULT_NAME[name]
            ftype = {"rememberToken": "string", "softDeletes": "timestamp", "softDeletesTz": "timestamp",
                     "softDeletesDatetime": "datetime", "ipAddress": "string", "macAddress": "string"}.get(name, name)
            nullable = name in ("rememberToken", "softDeletes", "softDeletesTz", "softDeletesDatetime")
            fields.append(ModelField(name=col, type=ftype, nullable=nullable))
        elif name == "foreignIdFor":
            cls = re.search(r"([\w\\]+)::class", args_code)
            model_name = _short(cls.group(1)) if cls else "model"
            col = first or (_snake(model_name) + "_id")
            fields.append(ModelField(name=col, type="foreignId", nullable=False, meta={"unsigned": True}))
            rel_field, rel_target = col, table_name_for(model_name)
        elif name == "foreign":
            cols = _strings(parts[0][1]) if parts else []
            rel_field = ",".join(cols)
        elif name in ("dropColumn", "dropColumns"):
            model.meta.setdefault("dropped", []).extend(_strings(args_raw))
        elif name == "dropSoftDeletes":
            model.meta.setdefault("dropped", []).append(first or "deleted_at")
        elif name == "dropTimestamps":
            model.meta.setdefault("dropped", []).extend(["created_at", "updated_at"])
        elif name == "renameColumn":
            strs = _strings(args_raw)
            if len(strs) >= 2:
                model.meta.setdefault("renamed", []).append([strs[0], strs[1]])
        elif name in _INDEX_CALLS:
            model.meta.setdefault("indexes", []).append({"type": name, "columns": _strings(parts[0][1]) if parts else []})
        elif name in _IGNORED_CALLS or first is None:
            return
        else:
            fmeta: Dict[str, Any] = {}
            if name in ("string", "char", "varchar") and len(parts) > 1:
                ln = _int_arg(parts[1][1])
                if ln is not None:
                    fmeta["length"] = ln
            elif name in ("decimal", "float", "double", "unsignedDecimal") and len(parts) > 1:
                prec = _int_arg(parts[1][1])
                if prec is not None:
                    fmeta["precision"] = prec
                if len(parts) > 2:
                    sc = _int_arg(parts[2][1])
                    if sc is not None:
                        fmeta["scale"] = sc
            elif name in ("enum", "set") and len(parts) > 1:
                fmeta["enum_values"] = _strings(parts[1][1])
            if name.startswith("unsigned") or name in ("foreignId", "foreignUlid"):
                fmeta["unsigned"] = True
            fields.append(ModelField(name=first, type=name, nullable=False, meta=fmeta))
            if name in ("foreignId", "foreignUlid", "foreignUuid"):
                rel_field = first

        for mname, mcode, mraw, _ in modifiers:
            if mname == "nullable":
                for f in fields:
                    f.nullable = "false" not in mcode.lower()
            elif mname == "default":
                for f in fields:
                    f.default = _unquote(mraw)
            elif mname == "useCurrent":
                for f in fields:
                    f.default = "CURRENT_TIMESTAMP"
            elif mname == "useCurrentOnUpdate":
                for f in fields:
                    f.meta["on_update"] = "CURRENT_TIMESTAMP"
            elif mname in ("unique", "primary", "index", "unsigned", "autoIncrement", "change", "invisible"):
                for f in fields:
                    f.meta[_snake(mname)] = True
            elif mname == "comment":
                for f in fields:
                    f.meta["comment"] = _unquote(mraw)
            elif mname == "constrained":
                tbl = _first_string(mraw)
                if not tbl and rel_field:
                    tbl = rel_target or pluralize(re.sub(r"_id$", "", rel_field))
                rel_target = tbl
            elif mname == "on":
                rel_target = _first_string(mraw) or rel_target
            elif mname == "references":
                for f in fields:
                    f.meta["references"] = _first_string(mraw)
            elif mname in _ON_DELETE:
                for f in fields:
                    f.meta["on_delete"] = _ON_DELETE[mname]
            elif mname == "onDelete":
                for f in fields:
                    f.meta["on_delete"] = _unquote(mraw)
            elif mname in ("cascadeOnUpdate", "onUpdate"):
                for f in fields:
                    f.meta["on_update"] = "cascade" if mname == "cascadeOnUpdate" else _unquote(mraw)
            elif mname in ("storedAs", "virtualAs"):
                for f in fields:
                    f.meta["computed"] = _unquote(mraw)
        model.fields.extend(fields)
        if rel_field and rel_target:
            model.relations.append(ModelRelation(kind="foreign_key", target=rel_target, field=rel_field))

    # ---- résolution du nom de table (premier argument de ``Schema::create`` / ``Schema::table``)

    def _down_spans(self) -> List[Tuple[int, int]]:
        """Bornes des corps de ``down()`` : ils défont ``up()`` et ne décrivent pas le schéma résultant."""
        src, code = self.src, self.src.code
        spans: List[Tuple[int, int]] = []
        for dm in _DOWN_RE.finditer(code):
            if not _bounded(code, dm.start()):
                continue
            close = src.match_paren(dm.end() - 1)
            after = close if close is not None else dm.end()
            bo = src.next_brace(after, 300)
            if bo is None or code.find(";", after, bo) >= 0:  # ``function down();`` (abstraite) : pas de corps
                continue
            bc = src.close_of(bo)
            if bc is not None:
                spans.append((bo, bc))
        return spans

    def _array_strings(self, start: int, end: int, want_key: bool = False) -> Optional[List[str]]:
        """Chaînes d'un tableau littéral ``[...]`` / ``array(...)`` (valeurs, ou clés si ``want_key``)."""
        src, code, noc = self.src, self.src.code, self.src.noc
        o = code.find("[" if code[start] == "[" else "(", start, end)
        if o < 0:
            return None
        c = src.match_paren(o, "[", "]") if code[o] == "[" else src.match_paren(o)
        if c is None or c > end:
            return None
        out: List[str] = []
        for ec, er in _split_top(code[o + 1: c], noc[o + 1: c]):
            k = ec.find("=>")
            if k >= 0:
                s = _first_string(er[:k] if want_key else er[k + 2:])
            else:
                s = _first_string(er)
            if s is not None:
                out.append(s)
        return out or None

    def _resolve_names(self, start: int, end: int, want_key: bool = False, depth: int = 0) -> Optional[List[str]]:
        """Chaînes désignées par l'expression ``code[start:end]`` : littéral (chaîne, tableau), constante de
        classe (``self::TABLES``), propriété (``$this->tables``) ou variable affectée / itérée par un
        ``foreach`` englobant. ``None`` si l'expression est dynamique (appel, concaténation sans littéral…)."""
        src, code, noc = self.src, self.src.code, self.src.noc
        while start < end and code[start].isspace():
            start += 1
        while end > start and code[end - 1].isspace():
            end -= 1
        if start >= end or depth > 4:
            return None
        expr = code[start:end]
        if expr[0] in "'\"":
            s = _first_string(noc[start:end])
            return [s] if s else None
        if expr[0] == "[" or expr.startswith("array"):
            return self._array_strings(start, end, want_key)
        cm = _CLASS_CONST_REF_RE.match(expr)
        if cm:
            dm = re.search(r"const\s+(?:[\w\\|?]+\s+)?" + re.escape(cm.group(1)) + r"\s*=", code)
            if dm and _bounded(code, dm.start()):
                return self._resolve_names(dm.end(), self._stmt_end(dm.end()), want_key, depth + 1)
            return None
        pm = _THIS_PROP_REF_RE.match(expr)
        if pm:
            name = re.escape(pm.group(1))
            dm = re.search(r"(?:public|protected|private|var|static|readonly)\s+(?:\??[\w\\|&]+\s+)?\$" + name
                           + r"\s*=(?![=>])", code) or re.search(r"\$this\s*->\s*" + name + r"\s*=(?![=>])", code)
            if dm:
                return self._resolve_names(dm.end(), self._stmt_end(dm.end()), want_key, depth + 1)
            return None
        vm = _VAR_REF_RE.match(expr)
        if vm:
            found = self._foreach_values(vm.group(1), start, depth)
            if found is not None:
                return found
            last = None
            for am in re.compile(r"\$" + re.escape(vm.group(1)) + r"\s*=(?![=>])").finditer(code, 0, start):
                last = am
            if last is not None:
                return self._resolve_names(last.end(), self._stmt_end(last.end()), want_key, depth + 1)
        return None

    def _stmt_end(self, pos: int) -> int:
        i = self.src.code.find(";", pos)
        return i if i >= 0 else len(self.src.code)

    def _foreach_values(self, var: str, before: int, depth: int) -> Optional[List[str]]:
        """Valeurs itérées par le ``foreach (... as $var)`` le plus interne dont le corps contient ``before``."""
        src, code = self.src, self.src.code
        best: Optional[Tuple[int, int, bool]] = None
        for fm in _FOREACH_RE.finditer(code, 0, before):
            if not _bounded(code, fm.start()):
                continue
            p = fm.end() - 1
            c = src.match_paren(p)
            if c is None or c > before:
                continue
            bo = src.next_brace(c, 50)
            bc = src.close_of(bo) if bo is not None else None
            if bo is None or bc is None or not (bo < before < bc) or code.find(";", c, bo) >= 0:
                continue
            hm = _FOREACH_HEAD_RE.match(code[p + 1: c])
            if not hm:
                continue
            if hm.group(3) == var:
                best = (p + 1 + hm.start(1), p + 1 + hm.end(1), False)
            elif hm.group(2) == var:
                best = (p + 1 + hm.start(1), p + 1 + hm.end(1), True)
        if best is None:
            return None
        return self._resolve_names(best[0], best[1], best[2], depth + 1)

    def _schema_tables(self, paren: int, close: Optional[int]) -> Tuple[List[str], int]:
        """Tables visées par l'appel dont la parenthèse ouvrante est ``paren`` → (noms, fin du 1er argument)."""
        code = self.src.code
        if close is None:  # parenthèse jamais fermée : premier littéral dans une fenêtre bornée, comme avant
            s = _first_string(self.src.noc[paren: paren + 300])
            return ([s] if s else []), paren + 300
        first_end = close
        depth = 0
        for dm in _ARG_DELIM_RE.finditer(code, paren + 1, close):
            ch = dm.group()
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif depth == 0:
                first_end = dm.start()
                break
        return (self._resolve_names(paren + 1, first_end) or []), first_end

    def extract_migrations(self) -> None:
        src = self.src
        code = src.code
        down_spans = self._down_spans()
        for m in _SCHEMA_RE.finditer(code):
            if not _bounded(code, m.start()):
                continue
            if any(a < m.start() < b for a, b in down_spans):
                continue
            paren = m.end() - 1
            close = src.match_paren(paren)
            op = m.group(1)
            line = src.line(m.start())
            tables, first_end = self._schema_tables(paren, close)
            if not tables:
                self.result.warnings.append("migration : nom de table dynamique non résolu pour Schema::%s (ligne %d)"
                                            % (op, line))
                continue
            models = []
            for table in tables:
                model = ExtractedModel(name=table, kind="migration", table_name=table, line=line,
                                       meta={"operation": op})
                if op == "table":
                    model.meta["alter"] = True
                models.append(model)
            model = models[0]
            body_open = src.next_brace(paren)
            if close is not None and body_open is not None and body_open > close:
                body_open = None  # accolade au-delà de l'appel : pas une closure de cet appel
            body_close = src.close_of(body_open) if body_open is not None else None
            if body_open is None and close is not None:
                am = _ARROW_FN_RE.search(code, first_end, close)
                if am:  # ``fn (Blueprint $t) => $t->...`` : le corps est l'expression jusqu'à la parenthèse fermante
                    body_open, body_close = am.end() - 1, close
            if body_open is None or body_close is None:
                self.result.warnings.append("migration %s : corps de Schema::%s introuvable (ligne %d)"
                                            % (model.name, op, line))
                self.result.models.extend(models)
                continue
            pm = _BLUEPRINT_PARAM_RE.search(code, paren, body_open + 1)
            var = pm.group(1) if pm else "table"
            var_re = re.compile(r"\$" + re.escape(var) + r"(?=\s*->)")
            start = body_open + 1
            while start < body_close:  # une instruction = jusqu'au prochain ``;`` (hors chaînes, déjà masquées)
                stmt_end = code.find(";", start, body_close)
                if stmt_end < 0:
                    stmt_end = body_close
                vm = var_re.search(code, start, stmt_end)
                if vm:
                    calls = _parse_chain(code, src.noc, vm.end(), src.match_paren, stmt_end)
                    if calls:
                        self._column_from_chain(calls, model)
                start = stmt_end + 1
            for extra in models[1:]:  # même closure appliquée à plusieurs tables (``foreach``)
                extra.fields = copy.deepcopy(model.fields)
                extra.relations = copy.deepcopy(model.relations)
                extra.meta = copy.deepcopy(model.meta)
            self.result.models.extend(models)


# --------------------------------------------------------------------------- routes Laravel

_ROUTE_RE = re.compile(r"Route\s*::\s*(\w+)\s*\(")
_VERB_CALLS = {"get": "GET", "post": "POST", "put": "PUT", "patch": "PATCH", "delete": "DELETE",
               "options": "OPTIONS", "any": "ANY"}
_RESOURCE_ACTIONS: List[Tuple[str, str, str]] = [
    ("index", "GET", ""), ("create", "GET", "/create"), ("store", "POST", ""), ("show", "GET", "/{p}"),
    ("edit", "GET", "/{p}/edit"), ("update", "PUT", "/{p}"), ("destroy", "DELETE", "/{p}"),
]
_API_ACTIONS = {"index", "store", "show", "update", "destroy"}
_GROUP_STR_KEYS = re.compile(r"'(prefix|as|namespace|domain|name)'\s*=>\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")
_GROUP_MW_KEY = re.compile(r"'middleware'\s*=>\s*(\[[^\]]*\]|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")
_GROUP_CTRL_KEY = re.compile(r"'controller'\s*=>\s*([\w\\]+)::class")
_API_FILE_RE = re.compile(r"(?:^|/)routes/api\.php$")


def _route_handler(code_arg: str, raw_arg: str, controller: Optional[str]) -> str:
    """Normalise le gestionnaire d'une route en ``Classe@méthode`` / ``closure``."""
    c = code_arg.strip()
    if not c:
        return ""
    if c.startswith("["):
        strs = _strings(raw_arg)
        uses = [s for s in strs if "@" in s]
        if uses:
            cls, meth = uses[0].split("@", 1)
            return _short(cls) + "@" + meth
        cls_m = re.search(r"([\w\\]+)::class", raw_arg)
        if cls_m and strs:
            return _short(cls_m.group(1)) + "@" + strs[-1]
        if cls_m:
            return _short(cls_m.group(1)) + "@__invoke"
        if len(strs) >= 2:
            return _short(strs[0]) + "@" + strs[1]
        return ""
    if re.match(r"(?:static\s+)?(?:fn|function)\b", c):
        return "closure"
    cm = re.match(r"([\w\\]+)::class$", c)
    if cm:
        return _short(cm.group(1)) + "@__invoke"
    if c[:1] in "'\"":
        s = _first_string(raw_arg) or ""
        if "@" in s:
            cls, meth = s.split("@", 1)
            return _short(cls) + "@" + meth
        if controller:
            return _short(controller) + "@" + s
        return _short(s) + "@__invoke"
    return _collapse(raw_arg)[:80]


class _RouteMixin:
    """Routes ``Route::`` avec pile de groupes (préfixes, middleware, préfixes de nom, contrôleur)."""

    src: _Src
    result: FileAnalysis
    path: str

    def _route_chain(self, m: Any) -> List[_Call]:
        """Chaîne ``Route::verbe(args)->mod(args)...`` ; les offsets d'arguments sont absolus."""
        src = self.src
        paren = m.end() - 1
        close = src.match_paren(paren)
        if close is None:
            return [(m.group(1), src.code[paren + 1: paren + 800], src.noc[paren + 1: paren + 800], paren + 1)]
        calls: List[_Call] = [(m.group(1), src.code[paren + 1: close], src.noc[paren + 1: close], paren + 1)]
        calls.extend(_parse_chain(src.code, src.noc, close + 1, src.match_paren))
        return calls

    def extract_routes(self) -> None:
        src = self.src
        is_api = bool(_API_FILE_RE.search(self.path))
        frames: List[Dict[str, Any]] = []  # pile de groupes : {start, end, prefix, middleware, name, controller}
        for m in _ROUTE_RE.finditer(src.code):
            if not _bounded(src.code, m.start(), ">"):
                continue
            while frames and frames[-1]["end"] <= m.start():  # groupes refermés avant cette route
                frames.pop()
            calls = self._route_chain(m)
            names = [c[0] for c in calls]
            # contexte hérité des groupes englobants
            prefix_parts: List[str] = []
            middleware: List[str] = []
            name_prefix = ""
            controller: Optional[str] = None
            for fr in frames:
                if fr["start"] < m.start() < fr["end"]:
                    if fr["prefix"]:
                        prefix_parts.append(fr["prefix"])
                    middleware.extend(fr["middleware"])
                    name_prefix += fr["name"]
                    controller = fr["controller"] or controller
            if "group" in names:
                self._push_group(calls, frames)
                continue
            own_mw: List[str] = []
            own_name: Optional[str] = None
            only: Optional[List[str]] = None
            except_: Optional[List[str]] = None
            domain: Optional[str] = None
            for cname, ccode, craw, _ in calls[1:]:
                if cname == "middleware":
                    own_mw.extend(_strings(craw))
                elif cname == "name":
                    own_name = _first_string(craw) or ""
                elif cname == "only":
                    only = _strings(craw)
                elif cname == "except":
                    except_ = _strings(craw)
                elif cname == "domain":
                    domain = _first_string(craw)
            verb, acode, araw, _ = calls[0]
            args = _split_top(acode, araw)
            line = src.line(m.start())
            prefix = _join_path(*prefix_parts) if prefix_parts else ""
            specs: List[Tuple[str, str, str, Optional[str]]] = []  # (méthode, chemin brut, handler, nom)
            if verb in _VERB_CALLS and args:
                path = _first_string(args[0][1]) if args[0][0][:1] in "'\"" else _collapse(args[0][1])
                handler = _route_handler(args[1][0], args[1][1], controller) if len(args) > 1 else ""
                specs.append((_VERB_CALLS[verb], path or "", handler, own_name))
            elif verb == "match" and len(args) >= 2:
                path = _first_string(args[1][1]) or _collapse(args[1][1])
                handler = _route_handler(args[2][0], args[2][1], controller) if len(args) > 2 else ""
                for meth in _strings(args[0][1]) or ["ANY"]:
                    specs.append((meth.upper(), path, handler, own_name))
            elif verb in ("redirect", "permanentRedirect") and args:
                path = _first_string(args[0][1]) or ""
                to = _first_string(args[1][1]) if len(args) > 1 else ""
                specs.append(("ANY", path, "redirect:%s" % (to or ""), own_name))
            elif verb == "view" and args:
                path = _first_string(args[0][1]) or ""
                view = _first_string(args[1][1]) if len(args) > 1 else ""
                specs.append(("GET", path, "view:%s" % (view or ""), own_name))
            elif verb == "fallback":
                handler = _route_handler(args[0][0], args[0][1], controller) if args else ""
                specs.append(("ANY", "/{fallback}", handler, own_name))
            elif verb in ("resource", "apiResource") and args:
                res = _first_string(args[0][1]) or ""
                ctrl = re.search(r"([\w\\]+)::class", args[1][1]) if len(args) > 1 else None
                ctrl_name = _short(ctrl.group(1)) if ctrl else (_short(_first_string(args[1][1]) or "") if len(args) > 1 else "")
                segs = res.strip("/").split(".")
                base = ""
                for seg in segs[:-1]:
                    base += "/%s/{%s}" % (seg, singularize(seg.split("/")[-1]).replace("-", "_"))
                last = segs[-1]
                base += "/" + last
                param = singularize(last.split("/")[-1]).replace("-", "_")
                for action, meth, suffix in _RESOURCE_ACTIONS:
                    if verb == "apiResource" and action not in _API_ACTIONS:
                        continue
                    if only is not None and action not in only:
                        continue
                    if except_ is not None and action in except_:
                        continue
                    rname = "%s.%s" % (res.replace("/", "."), action)
                    specs.append((meth, base + suffix.replace("{p}", "{%s}" % param),
                                  "%s@%s" % (ctrl_name, action) if ctrl_name else action, rname))
            else:
                continue
            for meth, path, handler, rname in specs:
                meta: Dict[str, Any] = {"prefix": prefix}
                full = _join_path("/api" if is_api else "", prefix, path)
                meta["full_path"] = full
                if is_api:
                    meta["api_prefix"] = "/api"
                mws = middleware + own_mw
                if mws:
                    meta["middleware"] = mws
                if rname is not None:
                    meta["name"] = name_prefix + rname
                elif name_prefix:
                    meta["name_prefix"] = name_prefix
                if domain:
                    meta["domain"] = domain
                if controller:
                    meta["controller"] = _short(controller)
                self.result.endpoints.append(ExtractedEndpoint(side="server", method=meth, path=path,
                                                               handler=handler, framework="laravel", line=line,
                                                               meta=meta))

    def _push_group(self, calls: List[_Call], frames: List[Dict[str, Any]]) -> None:
        """Enregistre un ``->group(...)`` / ``Route::group([...], ...)`` comme cadre (bornes du closure)."""
        src = self.src
        frame: Dict[str, Any] = {"prefix": "", "middleware": [], "name": "", "controller": None}
        open_pos: Optional[int] = None
        for cname, ccode, craw, cstart in calls:
            if cname == "prefix":
                frame["prefix"] = _first_string(craw) or ""
            elif cname == "middleware":
                frame["middleware"].extend(_strings(craw))
            elif cname in ("name", "as"):
                frame["name"] = _first_string(craw) or ""
            elif cname == "controller":
                cm = re.search(r"([\w\\]+)::class", craw)
                frame["controller"] = cm.group(1) if cm else (_first_string(craw) or None)
            elif cname == "group":
                if ccode.lstrip().startswith("["):
                    for km in _GROUP_STR_KEYS.finditer(craw):
                        key = km.group(1)
                        val = km.group(2)[1:-1]
                        if key == "prefix":
                            frame["prefix"] = val
                        elif key in ("as", "name"):
                            frame["name"] = val
                    mm = _GROUP_MW_KEY.search(craw)
                    if mm:
                        frame["middleware"].extend(_strings(mm.group(1)))
                    cm = _GROUP_CTRL_KEY.search(craw)
                    if cm:
                        frame["controller"] = cm.group(1)
                rel = ccode.find("{")
                if rel < 0:
                    return  # group(chemin de fichier) : pas de bloc
                open_pos = cstart + rel
        if open_pos is None:
            return
        close = src.close_of(open_pos)
        frame["start"] = open_pos
        frame["end"] = close if close is not None else len(src.code)
        frames.append(frame)


# --------------------------------------------------------------------------- appels HTTP sortants

_HTTP_FACADE_RE = re.compile(r"Http\s*::\s*(\w+)\s*\(")
_VAR_CALL_RE = re.compile(
    r"(\$\w+(?:\s*->\s*\w+){0,8})\s*->\s*(get|post|put|patch|delete|head|options|request"
    r"|getAsync|postAsync|putAsync|patchAsync|deleteAsync|requestAsync)\s*\("
)
_CURL_INIT_RE = re.compile(r"curl_init\s*\(")
_CURL_URL_RE = re.compile(r"curl_setopt\s*\(\s*\$\w+\s*,\s*CURLOPT_URL\s*,")
_FGC_RE = re.compile(r"file_get_contents\s*\(")
_HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options"}
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://[^/]*", re.I)
_CLIENT_HINT_RE = re.compile(r"client|http|guzzle|api", re.I)
_URL_HINT_RE = re.compile(r"url|api|base|host|endpoint", re.I)
# Variables PHP d'un chemin d'appel client (« documents/{$id}/ », « /items/$id », « {$this->base}/users ») : une route
# s'écrit avec des paramètres nommés (« documents/{id}/ », « {base}/users »), pas avec la syntaxe PHP.
_PHP_VAR_RE = re.compile(r"\$([A-Za-z_]\w*)((?:->[A-Za-z_]\w*|\[[^\[\]{}]*\])*)")
_BRACED_VAR_RE = re.compile(r"\{([^{}]*\$[^{}]*)\}")
_PROP_CHAIN_RE = re.compile(r"->([A-Za-z_]\w*)")
# Partie dynamique d'un chemin : variable, interpolation ou expression ({$this->baseUrl}, ${id}, {config('x')}, {url}).
_DYNAMIC_PART_RE = re.compile(r"\$\{[^{}]*\}|\$[A-Za-z_]\w*|\{[^{}]*\}")


def _only_dynamic(path: str) -> bool:
    """Vrai si le chemin n'est fait que de variables ou d'expressions (``$url``, ``{$this->baseUrl}{$endpoint}``,
    ``{self::PARTNER_URL}``, ``fetch(url)`` → ``{url}``) : il ne désigne aucune route, il n'est pas indexé."""
    s, prev = path, None
    while s != prev:
        prev, s = s, _DYNAMIC_PART_RE.sub("", s)
    return not s.strip()


def _var_placeholder(expr: str) -> Optional[str]:
    """Nom de paramètre d'une expression PHP : ``$id`` → ``id``, ``$this->baseUrl`` → ``baseUrl``, ``$ids[0]`` → ``ids``."""
    m = _PHP_VAR_RE.search(expr)
    if m is None:
        return None
    props = _PROP_CHAIN_RE.findall(m.group(2))
    return props[-1] if props else m.group(1)


def _php_placeholders(path: str) -> str:
    """Remplace chaque variable PHP d'un chemin par un paramètre nommé : ``documents/{$id}/`` → ``documents/{id}/``,
    ``/items/$id`` → ``/items/{id}``, ``{$this->base}/users`` → ``{base}/users`` ; une expression sans variable
    (``{config('services.api.url')}``) est conservée telle quelle."""
    path = _BRACED_VAR_RE.sub(lambda m: "{%s}" % (_var_placeholder(m.group(1)) or m.group(1)), path)
    return _PHP_VAR_RE.sub(lambda m: "{%s}" % (_var_placeholder(m.group(0)) or m.group(1)), path)


def _url_expr(code_arg: str, raw_arg: str) -> Tuple[str, Optional[str], bool]:
    """Expression d'URL → (chemin brut reconstruit, base visible, commence par un littéral).

    ``'http://x/' . $id`` → ``http://x/$id`` ; ``$base . '/orders'`` → ``$base/orders`` (base ``$base``) ;
    ``config('a.b') . '/x'`` → ``{config('a.b')}/x``.
    """
    parts = _split_top(code_arg, raw_arg, ".")
    path_parts: List[str] = []
    base: Optional[str] = None
    literal_first = False
    for i, (c, r) in enumerate(parts):
        strs = _strings(r)
        if c[:1] in "'\"" and c.rstrip()[-1:] in "'\"" and len(strs) == 1:
            path_parts.append(strs[0])
            if i == 0:
                literal_first = True
        else:
            expr = _collapse(r)
            if i == 0:
                base = expr
            path_parts.append(expr if re.match(r"^\$\w+$", expr) else "{" + expr + "}")
    path = "".join(path_parts)
    if base is None:
        sm = _SCHEME_RE.match(path)
        if sm:
            base = sm.group(0)
    return path, base, literal_first


class _ClientMixin:
    """Appels HTTP émis : ``Http::``, ``$client->get()``, ``curl_init``, ``file_get_contents``."""

    src: _Src
    result: FileAnalysis

    handler_at: Any
    _func_at: Any

    def _add_client(self, method: str, code_arg: str, raw_arg: str, framework: str, off: int,
                    extra: Optional[Dict[str, Any]] = None) -> None:
        path, base, _ = _url_expr(code_arg, raw_arg)
        if not path or _only_dynamic(path):
            return  # « Http::post($url) » : aucune route désignée
        path = _php_placeholders(path)
        meta: Dict[str, Any] = dict(extra or {})
        if base:
            meta["base"] = base
        self.result.endpoints.append(ExtractedEndpoint(side="client", method=method.upper(), path=path,
                                                       handler=self.handler_at(off), framework=framework,
                                                       line=self.src.line(off), meta=meta))

    def _call_args(self, paren: int) -> Optional[List[Tuple[str, str]]]:
        close = self.src.match_paren(paren)
        if close is None:
            return None
        return _split_top(self.src.code[paren + 1: close], self.src.noc[paren + 1: close])

    def extract_client_calls(self) -> None:
        src = self.src
        code = src.code
        for m in _HTTP_FACADE_RE.finditer(code):
            if not _bounded(code, m.start(), ">"):
                continue
            paren = m.end() - 1
            close = src.match_paren(paren)
            if close is None:
                continue
            calls: List[_Call] = [(m.group(1), code[paren + 1: close], src.noc[paren + 1: close], paren + 1)]
            calls.extend(_parse_chain(code, src.noc, close + 1, src.match_paren))
            extra: Dict[str, Any] = {}
            for cname, ccode, craw, _ in calls:
                if cname == "baseUrl":
                    extra["base"] = _first_string(craw) or _collapse(craw)
            for cname, ccode, craw, _ in calls:
                if cname in _HTTP_VERBS:
                    args = _split_top(ccode, craw)
                    if args:
                        self._add_client(cname, args[0][0], args[0][1], "laravel-http", m.start(), extra)
                    break
                if cname == "send":
                    args = _split_top(ccode, craw)
                    if len(args) >= 2:
                        self._add_client(_first_string(args[0][1]) or "ANY", args[1][0], args[1][1],
                                         "laravel-http", m.start(), extra)
                    break
        for m in _VAR_CALL_RE.finditer(code):
            var = _collapse(m.group(1)).replace(" ", "")
            verb = m.group(2)
            args = self._call_args(m.end() - 1)
            if not args:
                continue
            method = verb.replace("Async", "")
            url_idx = 0
            if method == "request":
                if len(args) < 2:
                    continue
                method = _first_string(args[0][1]) or "ANY"
                url_idx = 1
            path, base, literal_first = _url_expr(args[url_idx][0], args[url_idx][1])
            hinted = bool(_CLIENT_HINT_RE.search(var))
            looks_url = path.startswith("/") or bool(_SCHEME_RE.match(path)) or \
                (not literal_first and "/" in path)
            if not path or not (looks_url or (hinted and "/" in path)):
                continue
            framework = "laravel-test" if var == "$this" else "guzzle"
            self._add_client(method, args[url_idx][0], args[url_idx][1], framework, m.start(),
                             {"client": var})
        for m in _CURL_INIT_RE.finditer(code):
            args = self._call_args(m.end() - 1) if _bounded(code, m.start(), ">") else None
            if args:
                self._add_client(self._curl_method(m.start()), args[0][0], args[0][1], "curl", m.start())
        for m in _CURL_URL_RE.finditer(code):
            args = self._call_args(code.find("(", m.start(), m.end())) if _bounded(code, m.start(), ">") else None
            if args and len(args) >= 3:
                self._add_client(self._curl_method(m.start()), args[2][0], args[2][1], "curl", m.start())
        for m in _FGC_RE.finditer(code):
            args = self._call_args(m.end() - 1) if _bounded(code, m.start(), ">") else None
            if not args:
                continue
            path, base, _ = _url_expr(args[0][0], args[0][1])
            if _SCHEME_RE.match(path) or (base and _URL_HINT_RE.search(base) and "/" in path):
                self._add_client("GET", args[0][0], args[0][1], "file_get_contents", m.start())

    def _curl_method(self, off: int) -> str:
        """Méthode HTTP d'un appel curl : CURLOPT_CUSTOMREQUEST / CURLOPT_POST dans la fonction englobante
        (une seule recherche par fonction, mise en cache), sinon dans une fenêtre bornée autour de l'appel."""
        f = self._func_at(off)
        lo, hi = (f[0], f[1]) if f else (max(0, off - 4000), off + 4000)
        cache: Dict[Tuple[int, int], str] = self.__dict__.setdefault("_curl_cache", {})
        key = (lo, hi)
        if key not in cache:
            cache[key] = _curl_method_in(self.src.noc, lo, hi)
        return cache[key]


_CURL_CUSTOM_RE = re.compile(r"CURLOPT_CUSTOMREQUEST\s*,\s*['\"](\w+)['\"]")
_CURL_POST_RE = re.compile(r"CURLOPT_POST\s*,\s*(?:true|1)\b|CURLOPT_POSTFIELDS", re.I)


def _curl_method_in(noc: str, lo: int, hi: int) -> str:
    cm = _CURL_CUSTOM_RE.search(noc, lo, hi)
    if cm:
        return cm.group(1).upper()
    if _CURL_POST_RE.search(noc, lo, hi):
        return "POST"
    return "GET"


class _Analyzer(_AnalyzerBase, _ModelMixin, _MigrationMixin, _RouteMixin, _ClientMixin):
    """Analyseur complet d'un fichier PHP."""


# --------------------------------------------------------------------------- vues Blade

_SCRIPT_OPEN_RE = re.compile(r"<script\b[^>\n]{0,500}>", re.I)
_SCRIPT_CLOSE_RE = re.compile(r"</script\s*>", re.I)
_JS_MASK_RE = re.compile(r"'(?:[^'\\\n]|\\.)*'|\"(?:[^\"\\\n]|\\.)*\"|`(?:[^`\\]|\\.)*`|//[^\n]*|/\*[\s\S]*?\*/")
_JS_CALL_RE = re.compile(
    r"(?<![\w.$])(?:(fetch)|axios\s*\.\s*(get|post|put|patch|delete|head|options)|(axios)|\$\s*\.\s*(ajax|get|post))\s*\("
)
_JS_ROUTE_RE = re.compile(r"(?<![\w.$])(route|url)\s*\(\s*(['\"])([^'\"\n]*)\2")
_JS_FUNC_RE = re.compile(
    r"(?<![\w$])function\s+(\w+)\s*\(|(?<![\w$.])(\w+)\s*=\s*(?:async\s*)?(?:\([^()\n]*\)|\w+)\s*=>"
    r"|(?<![\w$.])(?:async\s+)?(\w+)\s*\([^()\n]*\)\s*\{"
)
_JS_METHOD_OPT_RE = re.compile(r"(?<![\w$])(?:method|type)\s*:\s*['\"](\w+)['\"]")
_JS_URL_OPT_RE = re.compile(r"(?<![\w$])url\s*:\s*")
_JS_KEYWORDS = {"if", "for", "while", "switch", "catch", "function", "return"}


def _js_mask(script: str) -> str:
    def repl(m: Any) -> str:
        s = m.group()
        return s[0] + _blank(s[1:-1]) + s[-1] if s[0] in "'\"`" else _blank(s)
    return _JS_MASK_RE.sub(repl, script)


def _js_functions(masked: str, bmatch: Dict[int, int]) -> List[Tuple[int, int, str]]:
    """Fonctions JS nommées (déclaration, fléchée affectée, méthode) avec les bornes de leur bloc."""
    out: List[Tuple[int, int, str]] = []
    for m in _JS_FUNC_RE.finditer(masked):
        name = m.group(1) or m.group(2) or m.group(3)
        if not name or name in _JS_KEYWORDS:
            continue
        open_pos = masked.find("{", m.end() - 1, m.end() + 300)
        if open_pos < 0:
            continue
        out.append((m.start(), bmatch.get(open_pos, len(masked)), name))
    return out


class _SpanIndex:
    """Intervalle le plus interne contenant un offset (intervalles triés par début ; dichotomie)."""

    def __init__(self, spans: List[Tuple[int, int, str]]) -> None:
        self.spans = spans
        self.starts = [s[0] for s in spans]
        self.maxend: List[int] = []
        mx = -1
        for s in spans:
            mx = max(mx, s[1])
            self.maxend.append(mx)

    def at(self, off: int) -> Optional[Tuple[int, int, str]]:
        i = bisect_right(self.starts, off) - 1
        while i >= 0:
            if self.maxend[i] <= off:
                return None
            s = self.spans[i]
            if s[0] <= off < s[1]:
                return s
            i -= 1
        return None


def _js_value(code: str, raw: str) -> Tuple[str, str]:
    """Valeur JS jusqu'à la prochaine ``,`` de niveau 0 ou la fermeture de l'objet/appel englobant."""
    depth = 0
    for i, ch in enumerate(code):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth < 0:
                return code[:i], raw[:i]
        elif ch == "," and depth == 0:
            return code[:i], raw[:i]
    return code, raw


def _js_arg_path(raw: str, code: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """Argument d'URL JS → chemin brut : route('x') → ``route:x``, url('/x') → ``/x``, ``'/a/' + id`` → ``/a/{id}``."""
    rm = _JS_ROUTE_RE.search(raw)
    if rm:
        if rm.group(1) == "route":
            return "route:" + rm.group(3), {"route_name": rm.group(3)}
        return rm.group(3), {}
    code = code if code is not None else _js_mask(raw)
    parts: List[str] = []
    for c, r in _split_top(code, raw, "+"):
        s = r.strip()
        if s[:1] in "'\"`" and s[-1:] == s[:1] and len(s) >= 2:
            parts.append(s[1:-1])
        else:
            expr = _collapse(s)
            parts.append("{" + expr + "}" if expr else "")
    return "".join(parts), {}


def _analyze_blade(text: str, path: str) -> FileAnalysis:
    """Vue Blade : uniquement les appels route()/url()/axios/fetch/$.ajax des blocs ``<script>``."""
    fa = FileAnalysis(language="blade")
    starts = [0] + [m.end() for m in re.finditer(r"\n", text)]
    pos = 0
    while True:
        om = _SCRIPT_OPEN_RE.search(text, pos)
        if om is None:
            break
        cm_ = _SCRIPT_CLOSE_RE.search(text, om.end())
        script_end = cm_.start() if cm_ else len(text)
        pos = cm_.end() if cm_ else len(text)
        script = text[om.end(): script_end]
        base_off = om.end()
        masked = _js_mask(script)
        pmatch = _pair_index(masked, "(", ")")
        funcs = _SpanIndex(_js_functions(masked, _pair_index(masked, "{", "}")))
        covered: List[Tuple[int, int]] = []

        def handler_at(off: int, _funcs: _SpanIndex = funcs) -> str:
            f = _funcs.at(off)
            return f[2] if f else ""

        for cm in _JS_CALL_RE.finditer(masked):
            paren = cm.end() - 1
            close = pmatch.get(paren)
            if close is None:
                continue
            covered.append((paren, close))
            args_code = masked[paren + 1: close]
            args_raw = script[paren + 1: close]
            args = _split_top(args_code, args_raw)
            if not args:
                continue
            meta: Dict[str, Any] = {}
            if cm.group(1):  # fetch(url, {method})
                framework = "fetch"
                p, meta = _js_arg_path(args[0][1], args[0][0])
                mm = _JS_METHOD_OPT_RE.search(args[1][1]) if len(args) > 1 else None
                method = mm.group(1).upper() if mm else "GET"
            elif cm.group(2):  # axios.get(url)
                framework = "axios"
                p, meta = _js_arg_path(args[0][1], args[0][0])
                method = cm.group(2).upper()
            elif cm.group(3) or cm.group(4) == "ajax":  # axios({url, method}) / $.ajax({url, type})
                framework = "axios" if cm.group(3) else "jquery"
                um = _JS_URL_OPT_RE.search(args_code)
                if not um:
                    continue
                vcode, vraw = _js_value(args_code[um.end():], args_raw[um.end():])
                p, meta = _js_arg_path(vraw, vcode)
                mm = _JS_METHOD_OPT_RE.search(args_raw)
                method = mm.group(1).upper() if mm else "GET"
            else:  # $.get(url) / $.post(url)
                framework = "jquery"
                p, meta = _js_arg_path(args[0][1], args[0][0])
                method = cm.group(4).upper()
            if not p or _only_dynamic(p):
                continue  # « fetch(url) » : aucune route désignée
            off = base_off + cm.start()
            fa.endpoints.append(ExtractedEndpoint(side="client", method=method, path=p, handler=handler_at(cm.start()),
                                                  framework=framework, line=bisect_right(starts, off), meta=meta))
        covered.sort()
        cov_starts = [lo for lo, _ in covered]
        for rm in _JS_ROUTE_RE.finditer(script):
            i = bisect_right(cov_starts, rm.start()) - 1
            if i >= 0 and covered[i][1] >= rm.start():
                continue
            p, meta = _js_arg_path(rm.group(0))
            off = base_off + rm.start()
            fa.endpoints.append(ExtractedEndpoint(side="client", method="ANY", path=p, handler=handler_at(rm.start()),
                                                  framework="blade", line=bisect_right(starts, off), meta=meta))
    fa.endpoints.sort(key=lambda e: e.line)
    return fa


# --------------------------------------------------------------------------- point d'entrée

_MAX_CHARS = 4_000_000


def _analyze_php(text: str, path: str) -> FileAnalysis:
    fa = FileAnalysis(language="php")
    if not text or not text.strip():
        fa.warnings.append("fichier vide")
        return fa
    if len(text) > _MAX_CHARS:
        fa.warnings.append("fichier tronqué à %d caractères pour l'analyse" % _MAX_CHARS)
        text = text[:_MAX_CHARS]
    try:
        an = _Analyzer(text, path)
    except Exception as exc:  # noqa: BLE001 - jamais d'exception vers l'appelant
        fa.warnings.append("échec du masquage du source : %s" % exc)
        return fa
    fa = an.result
    fa.warnings.extend(an.scan_notes)
    if not an.has_php:
        fa.warnings.append("aucune balise <?php : rien à extraire")
        return fa
    phases = (("symboles", an.extract_symbols), ("imports", an.extract_imports), ("modèles", an.extract_models),
              ("migrations", an.extract_migrations), ("routes", an.extract_routes),
              ("appels HTTP", an.extract_client_calls))
    for label, fn in phases:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            fa.warnings.append("échec de l'extraction (%s) : %s" % (label, exc))
    return fa


class PhpExtractor:
    """Extracteur PHP / Laravel ; ``languages`` = ['php', 'blade'] (Blade détecté via ``*.blade.php``)."""

    languages: List[str] = ["php", "blade"]

    def analyze(self, text: str, path: str = "") -> FileAnalysis:
        if not isinstance(text, str):
            try:
                text = bytes(text).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                text = str(text or "")
        path = path or ""
        try:
            if path.lower().endswith(".blade.php"):
                return _analyze_blade(text, path)
            return _analyze_php(text, path)
        except Exception as exc:  # noqa: BLE001 - filet de sécurité ultime
            lang = "blade" if path.lower().endswith(".blade.php") else "php"
            return FileAnalysis(language=lang, warnings=["échec de l'analyse de %s : %s" % (path, exc)])


EXTRACTOR = PhpExtractor()
