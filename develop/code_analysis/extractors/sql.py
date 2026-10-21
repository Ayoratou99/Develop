"""Extracteur SQL (DDL) : tables, vues, index, routines et contraintes.

Dialectes couverts au mieux : MySQL/MariaDB (backticks, ``ENUM('a','b')``, ``UNSIGNED``,
``AUTO_INCREMENT``, ``DELIMITER``), PostgreSQL (identifiants entre guillemets doubles, ``SERIAL``,
schémas ``public.x``, corps ``$$ … $$``), SQLite et SQL Server (crochets ``[x]``, ``GO``).

Principe : aucun parseur complet. Le fichier est d'abord balayé une seule fois (expression
régulière linéaire) pour neutraliser les commentaires hors chaînes et découper les instructions
sur ``;`` (ou le délimiteur courant) hors chaînes et hors parenthèses, en conservant le numéro de
ligne de chaque instruction. Seules les instructions ``CREATE`` / ``ALTER`` sont ensuite
découpées en jetons imbriqués (un groupe entre parenthèses = un jeton) et interprétées ;
``INSERT``/``UPDATE``/``SET``… sont ignorées silencieusement.

Résultats :

* ``CREATE TABLE`` → :class:`ExtractedModel` de kind ``sql_table`` (champs, relations ``foreign_key``,
  ``meta.indexes`` = nombre d'index déclarés dans la table, ``meta.line_end``, ``meta.schema``…) ;
* ``ALTER TABLE … ADD`` → modèle ``sql_table`` avec ``meta.alter=True`` ne portant que ce qui est ajouté ;
* ``CREATE VIEW`` → symbole ``view`` ; ``CREATE INDEX`` → symbole ``variable`` avec ``meta.index=True`` ;
  ``CREATE FUNCTION/PROCEDURE/TRIGGER`` → symboles ``function`` ; ``CREATE TYPE … AS ENUM`` → ``enum`` ;
* directives d'inclusion (``\\i``, ``SOURCE``, ``.read``) → :class:`ExtractedImport`.

Toute entrée est tolérée (vide, tronquée, invalide, énorme) : les anomalies sont signalées dans
``FileAnalysis.warnings`` et jamais levées.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from develop.code_analysis.model import (
    ExtractedImport,
    ExtractedModel,
    ExtractedSymbol,
    FileAnalysis,
    ModelField,
    ModelRelation,
)

# --------------------------------------------------------------------------- balayage global du fichier

# Corps de l'expression de balayage : chaînes, identifiants quotés, commentaires, directives de ligne,
# chaînes dollar PostgreSQL et parenthèses. Chaque alternative est linéaire (pas de .* imbriqué) et
# une chaîne/un commentaire non terminé absorbe simplement la fin du fichier (``'?``, ``\\Z``, ``.*``).
_SCAN_BODY = r"""
    (?P<str>'(?:[^'\\]+|\\.|'')*'?)
  | (?P<dq>"(?:[^"\\]+|\\.|"")*"?)
  | (?P<bt>`[^`\n]*`?)
  | (?P<br>\[[^\]\n]*\]?)
  | (?P<lc>--[^\n]*|^[ \t]*\#[^\n]*)
  | (?P<bc>/\*.*?(?:\*/|\Z))
  | (?P<directive>^[ \t]*(?:DELIMITER[ \t]+(?P<delim>\S+)
                          |\\(?:ir?|include|include_relative)[ \t]+(?P<pginc>\S+)
                          |SOURCE[ \t]+(?P<myinc>\S+\.sql)
                          |\.read[ \t]+(?P<sqinc>\S+)
                          |(?P<go>GO))[ \t]*;?[ \t]*(?=\n|\Z))
  | (?P<dollar>\$(?P<tag>[A-Za-z_][A-Za-z0-9_]*|)\$(?:.*?\$(?P=tag)\$|.*))
  | (?P<open>\()
  | (?P<close>\))
  | (?P<skip>[^'"`\[\-/$();\n%s]+)
"""
_SCAN_FLAGS = re.X | re.S | re.M | re.I
_SCAN_CACHE: Dict[str, "re.Pattern[str]"] = {}


def _scan_regex(delim: str) -> "re.Pattern[str]":
    """Expression de balayage pour un délimiteur d'instruction donné (``;`` par défaut).

    Un délimiteur personnalisé (``DELIMITER $$``) est placé en tête des alternatives pour primer sur
    la chaîne dollar ; le ``;`` standard est placé en queue (aucune autre alternative ne le capture).
    """
    rx = _SCAN_CACHE.get(delim)
    if rx is None:
        sep = r"(?P<sep>%s)" % re.escape(delim)
        # le premier caractère d'un délimiteur personnalisé ne doit pas être absorbé par ``skip``
        body = _SCAN_BODY % (re.escape(delim[0]) if delim[0] not in "'\"`[-/$();\n" else "")
        src = (sep + "|" + body) if delim != ";" else (body + "|" + sep)
        rx = re.compile(src, _SCAN_FLAGS)
        if len(_SCAN_CACHE) < 32:
            _SCAN_CACHE[delim] = rx
    return rx


# Mots-clés pouvant ouvrir une instruction : après un ``;`` rencontré dans des parenthèses non fermées,
# leur présence déclenche un découpage forcé (fichier tronqué ou parenthèse orpheline).
_STMT_KW_RE = re.compile(
    r"\s*(?:CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SET|USE|BEGIN|COMMIT|START|LOCK|UNLOCK|GRANT|REVOKE"
    r"|COPY|SELECT|TRUNCATE|PRAGMA|WITH|EXPLAIN|REPLACE|ANALYZE|VACUUM|COMMENT|DELIMITER|GO)\b",
    re.I,
)
_NON_NL_RE = re.compile(r"[^\n]+")
_NL_RE = re.compile(r"\n")
_LEAD_RE = re.compile(r"[^A-Za-z_]*")  # blancs, BOM, octets parasites avant le mot-clé
_FIRST_WORD_RE = re.compile(r"[A-Za-z_]+")


def _blank(s: str) -> str:
    """Remplace tout sauf les retours à la ligne par des espaces (les décalages restent identiques)."""
    return _NON_NL_RE.sub(lambda m: " " * (m.end() - m.start()), s)


@dataclass
class _Scan:
    """Résultat du balayage : texte nettoyé (même longueur que l'original), instructions, imports."""

    cleaned: str
    statements: List[Tuple[int, int]]  # (début, fin exclusive) dans ``cleaned``
    imports: List[ExtractedImport]
    warnings: List[str]


class _LineIndex:
    """Numéro de ligne (1-based) d'un décalage, par dichotomie sur les positions des ``\\n``."""

    def __init__(self, text: str) -> None:
        self._nl = [m.start() for m in _NL_RE.finditer(text)]

    def line_of(self, offset: int) -> int:
        return bisect_left(self._nl, offset) + 1


def _scan(text: str, lines: _LineIndex) -> _Scan:
    """Neutralise les commentaires et découpe le texte en instructions (une seule passe linéaire)."""
    chunks: List[str] = []
    last = 0
    statements: List[Tuple[int, int]] = []
    imports: List[ExtractedImport] = []
    warnings: List[str] = []
    depth = 0
    stmt_start = 0
    pos = 0
    n = len(text)
    rx = _scan_regex(";")
    std_delim = True  # False après ``DELIMITER xx`` : le délimiteur coupe alors quelle que soit la profondeur
    unterminated_reported = False

    while pos <= n:
        restart = False
        for m in rx.finditer(text, pos):
            kind = m.lastgroup
            if kind == "skip":
                continue
            if kind in ("str", "dq", "bt", "br", "dollar"):
                if m.end() == n and not unterminated_reported:
                    unterminated_reported = True
                    warnings.append("chaîne non terminée à la ligne %d" % lines.line_of(m.start()))
                continue
            if kind == "open":
                depth += 1
                continue
            if kind == "close":
                if depth > 0:
                    depth -= 1
                continue
            if kind in ("lc", "bc"):
                if kind == "bc" and not m.group().endswith("*/") and not unterminated_reported:
                    unterminated_reported = True
                    warnings.append("commentaire non terminé à la ligne %d" % lines.line_of(m.start()))
                chunks.append(text[last:m.start()])
                chunks.append(_blank(m.group()))
                last = m.end()
                continue
            if kind == "sep":
                if depth > 0 and std_delim:
                    if not _STMT_KW_RE.match(text, m.end()):
                        continue  # ``;`` à l'intérieur de parenthèses : pas une fin d'instruction
                    warnings.append(
                        "parenthèses non équilibrées avant la ligne %d : découpage forcé"
                        % lines.line_of(m.start())
                    )
                statements.append((stmt_start, m.start()))
                stmt_start = m.end()
                depth = 0
                continue
            # directive de ligne : neutralisée et terminant l'instruction en cours
            chunks.append(text[last:m.start()])
            chunks.append(_blank(m.group()))
            last = m.end()
            statements.append((stmt_start, m.start()))
            stmt_start = m.end()
            depth = 0
            inc = m.group("pginc") or m.group("myinc") or m.group("sqinc")
            if inc:
                inc = inc.strip("'\"`;")
                imports.append(
                    ExtractedImport(
                        module=inc,
                        names=[],
                        line=lines.line_of(m.start()),
                        is_relative=inc.startswith("./") or inc.startswith("../"),
                    )
                )
            delim = m.group("delim")
            if delim:
                rx = _scan_regex(delim)
                std_delim = delim == ";"
                pos = m.end()
                restart = True
                break
        if not restart:
            break
    statements.append((stmt_start, n))
    chunks.append(text[last:])
    return _Scan("".join(chunks), statements, imports, warnings)


# --------------------------------------------------------------------------- jetons d'une instruction

_FLAT_RE = re.compile(
    r"""
    (?P<str>'(?:[^'\\]+|\\.|'')*'?)
  | (?P<dq>"(?:[^"\\]+|\\.|"")*"?)
  | (?P<qid>`[^`\n]*`?|\[[^\]\n]*\]?)
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<word>[^\W\d][\w$]*)
  | (?P<open>\()
  | (?P<close>\))
  | (?P<op>::|<>|!=|<=|>=|\|\||[^\s])
""",
    re.X | re.S,
)  # les blancs ne produisent aucun jeton : le moteur C les saute sans itération Python


class _Tok:
    """Jeton : mot, nombre, chaîne (``str``/``dq``), identifiant quoté (``qid``), groupe ``paren`` ou opérateur.

    Classe à ``__slots__`` (et non dataclass) : des centaines de milliers d'instances sont créées
    pour un gros fichier et le coût d'instanciation domine le temps d'analyse.
    """

    __slots__ = ("kind", "text", "start", "space_before", "items", "closed", "up", "is_ident")

    def __init__(self, kind: str, text: str, start: int, space_before: bool = False) -> None:
        self.kind = kind
        self.text = text
        self.start = start  # décalage absolu dans le texte nettoyé
        self.space_before = space_before
        self.items: Sequence["_Tok"] = [] if kind == "paren" else ()  # enfants d'un groupe entre parenthèses
        self.closed = True  # False pour un groupe fermé implicitement en fin de texte
        self.up = text.upper() if kind == "word" else ""  # majuscules pour comparer les mots-clés
        self.is_ident = kind in ("word", "qid", "dq")

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def _tokenize(text: str, base: int = 0) -> List[_Tok]:
    """Découpe ``text`` en jetons imbriqués ; ``base`` est le décalage absolu du début de ``text``."""
    root: List[_Tok] = []
    stack: List[_Tok] = []
    current = root
    for m in _FLAT_RE.finditer(text):
        kind = m.lastgroup
        start = m.start()
        space = start > 0 and text[start - 1] in " \t\r\n"
        if kind == "open":
            tok = _Tok("paren", "", base + start, space)
            current.append(tok)
            stack.append(tok)
            current = tok.items  # type: ignore[assignment]
        elif kind == "close":
            if stack:
                tok = stack.pop()
                tok.text = text[tok.start - base:m.end()]
                current = stack[-1].items if stack else root  # type: ignore[assignment]
            # parenthèse fermante orpheline : ignorée
        else:
            current.append(_Tok(kind, m.group(), base + start, space))
    while stack:  # groupes non fermés (fichier tronqué) : fermés en fin de texte
        tok = stack.pop()
        tok.text = text[tok.start - base:]
        tok.closed = False
    return root


class _Cursor:
    """Curseur de lecture sur une liste de jetons ; hors limites → ``None``, jamais d'exception."""

    def __init__(self, toks: List[_Tok]) -> None:
        self.toks = toks
        self.i = 0

    def peek(self, k: int = 0) -> Optional[_Tok]:
        try:
            return self.toks[self.i + k]
        except IndexError:
            return None

    def next(self) -> Optional[_Tok]:
        try:
            t = self.toks[self.i]
        except IndexError:
            t = None
        self.i += 1
        return t

    def at_end(self) -> bool:
        return self.i >= len(self.toks)

    def rest(self) -> List[_Tok]:
        return self.toks[self.i:]

    def is_word(self, *words: str, k: int = 0) -> bool:
        try:
            t = self.toks[self.i + k]
        except IndexError:
            return False
        return t.kind == "word" and t.up in words

    def is_paren(self, k: int = 0) -> bool:
        t = self.peek(k)
        return t is not None and t.kind == "paren"

    def is_op(self, text: str, k: int = 0) -> bool:
        t = self.peek(k)
        return t is not None and t.kind == "op" and t.text == text

    def skip_words(self, *seq: str) -> bool:
        """Consomme la séquence de mots-clés si elle est présente ; renvoie True dans ce cas."""
        for k, w in enumerate(seq):
            if not self.is_word(w, k=k):
                return False
        self.i += len(seq)
        return True

    def take_ident(self) -> Optional[str]:
        t = self.peek()
        if t is not None and t.is_ident:
            self.i += 1
            return _unquote(t.text)
        return None

    def take_qualified(self) -> List[str]:
        """Identifiant éventuellement qualifié (``schema.table``, ``` `db`.`t` ```, ``[dbo].[t]``)."""
        parts: List[str] = []
        first = self.take_ident()
        if first is None:
            return parts
        parts.append(first)
        while self.is_op(".") and self.peek(1) is not None and self.peek(1).is_ident:  # type: ignore[union-attr]
            self.i += 1
            parts.append(_unquote(self.next().text))  # type: ignore[union-attr]
        return parts


def _unquote(text: str) -> str:
    """Retire les délimiteurs d'un identifiant ou d'une chaîne et dédouble les guillemets internes."""
    if len(text) >= 2 and text[0] in "`\"'[" and text[-1] in "`\"']":
        inner = text[1:-1]
        q = text[0]
        if q == "'":
            return inner.replace("''", "'").replace("\\'", "'")
        if q == '"':
            return inner.replace('""', '"')
        if q == "`":
            return inner.replace("``", "`")
        return inner
    if text and text[0] in "`\"'[":  # délimiteur ouvrant sans fermeture (fin de fichier)
        return text[1:]
    return text


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _split_commas(toks: List[_Tok]) -> List[List[_Tok]]:
    """Découpe une liste de jetons sur les virgules de ce niveau (les groupes restent entiers)."""
    out: List[List[_Tok]] = []
    cur: List[_Tok] = []
    for t in toks:
        if t.kind == "op" and t.text == ",":
            out.append(cur)
            cur = []
        else:
            cur.append(t)
    out.append(cur)
    return [seg for seg in out if seg]


def _ident_list(paren: Optional[_Tok]) -> List[str]:
    """Premier identifiant de chaque segment d'une liste de colonnes ``(a, b(10) DESC, c)``."""
    if paren is None or paren.kind != "paren":
        return []
    cols: List[str] = []
    for seg in _split_commas(paren.items):
        for t in seg:
            if t.is_ident:
                cols.append(_unquote(t.text))
                break
    return cols


def _segment_texts(paren: Optional[_Tok], cleaned: str) -> List[str]:
    """Texte brut (blancs compactés) de chaque segment d'un groupe ``(lower(email), created_at DESC)``."""
    if paren is None or paren.kind != "paren":
        return []
    out: List[str] = []
    for seg in _split_commas(paren.items):
        out.append(_collapse(cleaned[seg[0].start:seg[-1].end]))
    return out


# --------------------------------------------------------------------------- colonnes et contraintes

# Mots-clés qui terminent le type d'une colonne et ouvrent ses options.
_COL_OPTION_KW = frozenset(
    "NOT NULL DEFAULT PRIMARY UNIQUE REFERENCES CHECK COLLATE CHARACTER CHARSET AUTO_INCREMENT "
    "AUTOINCREMENT COMMENT GENERATED CONSTRAINT ON IDENTITY KEY COLUMN_FORMAT STORAGE VIRTUAL STORED "
    "INVISIBLE VISIBLE SRID FIRST AFTER ENCODE COMPRESSION ROWGUIDCOL SPARSE FILESTREAM MASKED "
    "PERSISTED CLUSTERED NONCLUSTERED WITH ENGINE_ATTRIBUTE AS DEFERRABLE INITIALLY MATCH USING".split()
)
# Mots pouvant prolonger un type (``DOUBLE PRECISION``, ``TIMESTAMP WITH TIME ZONE``, ``INT UNSIGNED``…).
_TYPE_CONT = frozenset(
    "PRECISION VARYING UNSIGNED SIGNED ZEROFILL WITH WITHOUT LOCAL TIME ZONE TO SECOND MINUTE HOUR "
    "DAY MONTH YEAR LARGE OBJECT BINARY".split()
)
_SERIAL_TYPES = frozenset("SERIAL BIGSERIAL SMALLSERIAL SERIAL2 SERIAL4 SERIAL8".split())
_FK_ACTIONS = frozenset("CASCADE RESTRICT SET NO".split())


class _Ctx:
    """Contexte d'analyse d'un fichier : texte nettoyé, index des lignes, résultat."""

    def __init__(self, cleaned: str, lines: _LineIndex, fa: FileAnalysis) -> None:
        self.cleaned = cleaned
        self.lines = lines
        self.fa = fa

    def span(self, toks: List[_Tok]) -> str:
        """Texte brut couvert par une suite de jetons (blancs compactés)."""
        if not toks:
            return ""
        return _collapse(self.cleaned[toks[0].start:toks[-1].end])


def _field_by_name(model: ExtractedModel, name: str) -> Optional[ModelField]:
    for f in model.fields:
        if f.name == name:
            return f
    return None


def _capture_expr(cur: _Cursor, ctx: _Ctx) -> str:
    """Expression (valeur par défaut, type…) : jetons jusqu'au prochain mot-clé d'option ou la fin."""
    toks: List[_Tok] = []
    first = True
    while not cur.at_end():
        t = cur.peek()
        if t.kind == "word" and t.up in _COL_OPTION_KW and not (first and t.up == "NULL"):
            break
        if t.kind == "op" and t.text == ",":
            break
        toks.append(cur.next())  # type: ignore[arg-type]
        first = False
    return ctx.span(toks)


def _take_fk_action(cur: _Cursor) -> str:
    """Action référentielle : ``CASCADE`` | ``SET NULL`` | ``SET DEFAULT`` | ``RESTRICT`` | ``NO ACTION``."""
    t = cur.next()
    if t is None or t.kind != "word":
        return ""
    if t.up in ("SET", "NO") and cur.peek() is not None and cur.peek().kind == "word":  # type: ignore[union-attr]
        return t.up + " " + cur.next().up  # type: ignore[union-attr]
    return t.up


def _parse_references(cur: _Cursor) -> Optional[Dict[str, Any]]:
    """``REFERENCES [schema.]table [(cols)] [MATCH x] [ON DELETE a] [ON UPDATE b] [DEFERRABLE…]``."""
    if not cur.skip_words("REFERENCES"):
        return None
    parts = cur.take_qualified()
    if not parts:
        return None
    ref: Dict[str, Any] = {"table": parts[-1], "columns": []}
    if len(parts) > 1:
        ref["schema"] = ".".join(parts[:-1])
    if cur.is_paren():
        ref["columns"] = _ident_list(cur.next())
    while not cur.at_end():
        if cur.skip_words("MATCH"):
            cur.next()
        elif cur.is_word("ON") and cur.is_word("DELETE", "UPDATE", k=1):
            cur.next()
            which = cur.next().up.lower()  # type: ignore[union-attr]
            ref["on_" + which] = _take_fk_action(cur)
        elif cur.skip_words("NOT", "DEFERRABLE") or cur.skip_words("DEFERRABLE"):
            pass
        elif cur.skip_words("INITIALLY"):
            cur.next()
        else:
            break
    return ref


def _apply_fk(model: ExtractedModel, cols: List[str], ref: Dict[str, Any], cname: Optional[str]) -> None:
    """Ajoute une relation ``foreign_key`` par colonne, annote les champs et ``meta.foreign_keys``."""
    for c in cols:
        model.relations.append(ModelRelation(kind="foreign_key", target=ref["table"], field=c))
        f = _field_by_name(model, c)
        if f is not None:
            f.meta["references"] = {k: v for k, v in ref.items() if k in ("table", "schema", "columns")}
            for k in ("on_delete", "on_update"):
                if ref.get(k):
                    f.meta[k] = ref[k]
    fk: Dict[str, Any] = {"columns": cols, "table": ref["table"], "ref_columns": ref.get("columns", [])}
    for k in ("schema", "on_delete", "on_update"):
        if ref.get(k):
            fk[k] = ref[k]
    if cname:
        fk["constraint"] = cname
    model.meta.setdefault("foreign_keys", []).append(fk)


def _parse_type(cur: _Cursor, f: ModelField) -> None:
    """Type complet d'une colonne (``VARCHAR(255)``, ``DECIMAL(10,2)``, ``ENUM(...)``, ``text[]``…)."""
    t = cur.peek()
    if t is None or not t.is_ident:
        return
    if t.kind == "word" and t.up in _COL_OPTION_KW:
        # ``character varying(255)`` est un type ; ``CHARACTER SET utf8`` est une option
        if t.up != "CHARACTER" or cur.is_word("SET", k=1):
            return
    base = [cur.next().text]  # type: ignore[union-attr]
    while cur.is_op(".") and cur.peek(1) is not None and cur.peek(1).is_ident:  # type: ignore[union-attr]
        cur.next()
        base.append(cur.next().text)  # type: ignore[union-attr]
    first_up = base[-1].upper()
    parts = [".".join(base)]
    if cur.is_paren():
        p = cur.next()
        parts[-1] += _collapse(p.text)  # type: ignore[union-attr]
        if first_up in ("ENUM", "SET"):
            f.meta["enum_values"] = [_unquote(it.text) for it in p.items if it.kind in ("str", "dq")]  # type: ignore[union-attr]
    while True:  # suffixes de tableau PostgreSQL : integer[], text[][]
        nt = cur.peek()
        if nt is not None and nt.kind == "qid" and nt.text.startswith("[") and not nt.space_before:
            parts[-1] += cur.next().text  # type: ignore[union-attr]
        else:
            break
    while True:  # mots de continuation du type
        nt = cur.peek()
        if nt is None or nt.kind != "word":
            break
        if nt.up in _TYPE_CONT:
            pass
        elif nt.up in ("CHARACTER", "CHAR", "NATIONAL", "NCHAR") and cur.is_word("VARYING", "LARGE", "CHARACTER", "CHAR", k=1):
            pass
        else:
            break
        cur.next()
        parts.append(nt.text)
        if nt.up == "UNSIGNED":
            f.meta["unsigned"] = True
        if cur.is_paren():
            parts[-1] += _collapse(cur.next().text)  # type: ignore[union-attr]
    f.type = " ".join(parts)
    if first_up in _SERIAL_TYPES:
        f.meta["auto_increment"] = True


def _parse_column(cur: _Cursor, model: ExtractedModel, ctx: _Ctx, cname: Optional[str] = None,
                  alter: bool = False) -> Optional[ModelField]:
    """Définition de colonne ``name type [options…]`` → :class:`ModelField` ajouté au modèle."""
    nt = cur.next()
    if nt is None or not nt.is_ident:
        return None
    f = ModelField(name=_unquote(nt.text), meta={"line": ctx.lines.line_of(nt.start)})
    if alter:
        f.meta["alter"] = True
    _parse_type(cur, f)
    model.fields.append(f)
    guard = 0
    while not cur.at_end() and guard < 10000:
        guard += 1
        t = cur.next()
        if t.kind != "word":
            continue
        up = t.up
        if up == "NOT":
            if cur.skip_words("NULL"):
                f.nullable = False
        elif up == "NULL":
            if f.nullable is None:
                f.nullable = True
        elif up == "DEFAULT":
            f.default = _capture_expr(cur, ctx) or None
        elif up == "PRIMARY":
            cur.skip_words("KEY")
            f.meta["primary"] = True
            f.nullable = False
        elif up == "FOREIGN":  # SQL Server : ``CONSTRAINT x FOREIGN KEY REFERENCES t(c)``
            cur.skip_words("KEY")
        elif up == "KEY":  # MySQL : ``col INT KEY`` équivaut à PRIMARY KEY
            f.meta["primary"] = True
            f.nullable = False
        elif up == "UNIQUE":
            cur.skip_words("KEY") or cur.skip_words("INDEX")
            f.meta["unique"] = True
        elif up in ("AUTO_INCREMENT", "AUTOINCREMENT"):
            f.meta["auto_increment"] = True
        elif up == "IDENTITY":  # SQL Server : IDENTITY(1,1)
            f.meta["auto_increment"] = True
            f.meta["identity"] = True
            if cur.is_paren():
                cur.next()
        elif up == "GENERATED":
            while not cur.at_end() and not cur.is_word("AS"):
                cur.next()
            cur.skip_words("AS")
            if cur.skip_words("IDENTITY"):
                f.meta["auto_increment"] = True
                f.meta["identity"] = True
                if cur.is_paren():
                    cur.next()
            elif cur.is_paren():
                f.meta["generated"] = _collapse(cur.next().text[1:-1])  # type: ignore[union-attr]
        elif up == "AS" and cur.is_paren():
            f.meta["generated"] = _collapse(cur.next().text[1:-1])  # type: ignore[union-attr]
        elif up == "REFERENCES":
            cur.i -= 1
            ref = _parse_references(cur)
            if ref is not None:
                _apply_fk(model, [f.name], ref, cname)
        elif up == "ON" and cur.skip_words("UPDATE"):  # MySQL : ON UPDATE CURRENT_TIMESTAMP
            f.meta["on_update_expr"] = _capture_expr(cur, ctx)
        elif up == "CHECK" and cur.is_paren():
            f.meta["check"] = _collapse(cur.next().text[1:-1])  # type: ignore[union-attr]
        elif up == "COMMENT":
            ct = cur.peek()
            if ct is not None and ct.kind in ("str", "dq"):
                f.meta["comment"] = _unquote(cur.next().text)  # type: ignore[union-attr]
        elif up == "COLLATE":
            ct = cur.peek()
            if ct is not None and ct.is_ident:
                f.meta["collation"] = _unquote(cur.next().text)  # type: ignore[union-attr]
        elif up == "CONSTRAINT":
            cname = cur.take_ident()
    return f


def _looks_like_index(cur: _Cursor) -> bool:
    """``KEY idx (a, b)`` (index) ou ``key INT(11)`` (colonne nommée key) ? Vrai si la première
    parenthèse contient un identifiant (liste de colonnes) plutôt que des nombres/chaînes seuls."""
    for t in cur.rest():
        if t.kind == "paren":
            return any(it.is_ident for it in t.items)
        if t.kind == "word" and t.up in _COL_OPTION_KW and t.up not in ("USING", "KEY"):
            return False
    return False


def _parse_table_element(toks: List[_Tok], model: ExtractedModel, ctx: _Ctx, alter: bool = False) -> None:
    """Un élément de ``CREATE TABLE ( … )`` ou une clause ``ALTER TABLE … ADD`` : colonne ou contrainte."""
    cur = _Cursor(toks)
    cname: Optional[str] = None
    if cur.skip_words("CONSTRAINT"):
        if not cur.is_word("PRIMARY", "UNIQUE", "FOREIGN", "CHECK", "EXCLUDE"):
            cname = cur.take_ident()
    t = cur.peek()
    if t is None:
        return
    if t.kind == "word":
        up = t.up
        if up == "PRIMARY" and cur.is_word("KEY", k=1):
            cur.i += 2
            while not cur.at_end() and not cur.is_paren():
                cur.next()
            cols = _ident_list(cur.next())
            for c in cols:
                f = _field_by_name(model, c)
                if f is None and alter:
                    f = ModelField(name=c, meta={"alter": True, "line": ctx.lines.line_of(t.start)})
                    model.fields.append(f)
                if f is not None:
                    f.meta["primary"] = True
                    f.nullable = False
            if cols:
                model.meta["primary_key"] = cols
            return
        if up == "UNIQUE":
            cur.next()
            is_index = cur.skip_words("KEY") or cur.skip_words("INDEX")
            while not cur.at_end() and not cur.is_paren():
                cur.next()  # nom d'index, USING BTREE…
            cols = _ident_list(cur.next())
            for c in cols:
                f = _field_by_name(model, c)
                if f is None and alter:
                    f = ModelField(name=c, meta={"alter": True, "line": ctx.lines.line_of(t.start)})
                    model.fields.append(f)
                if f is not None:
                    f.meta["unique"] = True
            if cols:
                model.meta.setdefault("unique_keys", []).append(cols)
            if is_index:
                model.meta["indexes"] = model.meta.get("indexes", 0) + 1
            return
        if up == "FOREIGN" and cur.is_word("KEY", k=1):
            cur.i += 2
            while not cur.at_end() and not cur.is_paren() and not cur.is_word("REFERENCES"):
                cur.next()  # nom d'index MySQL
            cols = _ident_list(cur.next()) if cur.is_paren() else []
            ref = _parse_references(cur)
            if ref is not None and cols:
                _apply_fk(model, cols, ref, cname)
            return
        if up in ("INDEX", "KEY", "FULLTEXT", "SPATIAL") and _looks_like_index(cur):
            model.meta["indexes"] = model.meta.get("indexes", 0) + 1
            return
        if up == "CHECK":
            model.meta["checks"] = model.meta.get("checks", 0) + 1
            return
        if up in ("EXCLUDE", "LIKE", "PERIOD"):
            return
    _parse_column(cur, model, ctx, cname, alter)


# --------------------------------------------------------------------------- instructions CREATE / ALTER

_OBJECT_KW = frozenset(
    "TABLE VIEW INDEX FUNCTION PROCEDURE TRIGGER TYPE SEQUENCE SCHEMA DATABASE EXTENSION DOMAIN EVENT "
    "ROLE USER AGGREGATE OPERATOR SERVER RULE POLICY PUBLICATION SUBSCRIPTION TABLESPACE COLLATION "
    "CAST CONVERSION LANGUAGE STATISTICS SYNONYM PACKAGE".split()
)
_ENGINE_RE = re.compile(r"\bENGINE\s*=?\s*([A-Za-z0-9_]+)", re.I)
_TABLE_COMMENT_RE = re.compile(r"\bCOMMENT\s*=?\s*('(?:[^'\\]+|\\.|'')*')", re.I)
_INHERITS_RE = re.compile(r"\bINHERITS\s*\(([^)]{0,500})\)", re.I)
_IDENT_SRC = r"(?:`[^`\n]+`|\"[^\"\n]+\"|\[[^\]\n]+\]|[^\W\d][\w$]*)"
_QUALIFIED_SRC = _IDENT_SRC + r"(?:\s*\.\s*" + _IDENT_SRC + r")*"
_FROM_JOIN_RE = re.compile(r"\b(?:FROM|JOIN)\s+(" + _QUALIFIED_SRC + r")", re.I)
_LANGUAGE_RE = re.compile(r"\bLANGUAGE\s+(?:'|\")?([A-Za-z0-9_]+)", re.I)
_EXECUTE_RE = re.compile(r"\bEXECUTE\s+(?:FUNCTION|PROCEDURE)\s+(" + _QUALIFIED_SRC + r")", re.I)
_SIG_MAX = 200


def _split_qualified_text(text: str) -> List[str]:
    """``public."Orders"`` → ``['public', 'Orders']`` (version regex, pour les méta des vues/triggers)."""
    return [_unquote(p.strip()) for p in re.split(r"\s*\.\s*", text.strip()) if p.strip()]


def _handle_create(cur: _Cursor, stmt: str, start: int, end: int, line: int, ctx: _Ctx) -> None:
    mods: List[str] = []
    obj: Optional[str] = None
    while not cur.at_end() and len(mods) < 24:
        t = cur.next()
        if t.kind == "word" and t.up in _OBJECT_KW:
            obj = t.up
            break
        if t.kind == "word":
            mods.append(t.up)
    line_end = ctx.lines.line_of(max(start, end - 1))
    if obj == "TABLE":
        _create_table(cur, mods, stmt, start, line, line_end, ctx)
    elif obj == "VIEW":
        _create_view(cur, mods, stmt, line, line_end, ctx)
    elif obj == "INDEX":
        _create_index(cur, mods, line, line_end, ctx)
    elif obj in ("FUNCTION", "PROCEDURE"):
        _create_routine(cur, obj.lower(), stmt, line, line_end, ctx)
    elif obj == "TRIGGER":
        _create_trigger(cur, stmt, line, line_end, ctx)
    elif obj == "TYPE":
        _create_type(cur, line, line_end, ctx)


def _create_table(cur: _Cursor, mods: List[str], stmt: str, start: int, line: int, line_end: int, ctx: _Ctx) -> None:
    cur.skip_words("IF", "NOT", "EXISTS")
    parts = cur.take_qualified()
    if not parts:
        ctx.fa.warnings.append("CREATE TABLE sans nom lisible à la ligne %d" % line)
        return
    name = parts[-1]
    model = ExtractedModel(name=name, kind="sql_table", table_name=name, line=line, meta={})
    meta = model.meta
    if len(parts) > 1:
        meta["schema"] = ".".join(parts[:-1])
    if "TEMPORARY" in mods or "TEMP" in mods:
        meta["temporary"] = True
    if cur.skip_words("PARTITION", "OF"):
        meta["partition_of"] = ".".join(cur.take_qualified())
    body: Optional[_Tok] = None
    if cur.is_paren():
        body = cur.next()
    elif cur.skip_words("LIKE"):
        meta["like"] = ".".join(cur.take_qualified())
    if cur.is_word("AS"):
        meta["as_select"] = True
    if body is not None:
        for element in _split_commas(body.items):
            _parse_table_element(element, model, ctx)
        if not body.closed:
            ctx.fa.warnings.append("CREATE TABLE %s (ligne %d) : corps tronqué" % (name, line))
        options = stmt[body.end - start:]
        m = _ENGINE_RE.search(options)
        if m:
            meta["engine"] = m.group(1)
        m = _TABLE_COMMENT_RE.search(options)
        if m:
            meta["comment"] = _unquote(m.group(1))
        m = _INHERITS_RE.search(options)
        if m:
            meta["inherits"] = [p.strip() for p in m.group(1).split(",") if p.strip()]
    meta.setdefault("indexes", 0)
    meta["line_end"] = line_end
    ctx.fa.models.append(model)


def _handle_alter(cur: _Cursor, line: int, ctx: _Ctx) -> None:
    if not cur.skip_words("TABLE"):
        return
    cur.skip_words("ONLY")
    cur.skip_words("IF", "EXISTS")
    parts = cur.take_qualified()
    if not parts:
        return
    name = parts[-1]
    model = ExtractedModel(name=name, kind="sql_table", table_name=name, line=line, meta={"alter": True})
    if len(parts) > 1:
        model.meta["schema"] = ".".join(parts[:-1])
    for action in _split_commas(cur.rest()):
        _alter_action(action, model, ctx)
    if model.fields or model.relations:
        ctx.fa.models.append(model)


def _alter_action(toks: List[_Tok], model: ExtractedModel, ctx: _Ctx) -> None:
    cur = _Cursor(toks)
    if cur.skip_words("ADD"):
        if cur.is_paren():  # MySQL : ADD (a INT, b INT)
            for element in _split_commas(cur.next().items):  # type: ignore[union-attr]
                _parse_table_element(element, model, ctx, alter=True)
            return
        cur.skip_words("COLUMN")
        cur.skip_words("IF", "NOT", "EXISTS")
        _parse_table_element(cur.rest(), model, ctx, alter=True)
    elif cur.skip_words("MODIFY"):
        cur.skip_words("COLUMN")
        f = _parse_column(cur, model, ctx, alter=True)
        if f is not None:
            f.meta["modify"] = True
    elif cur.skip_words("CHANGE"):
        cur.skip_words("COLUMN")
        old = cur.take_ident()
        f = _parse_column(cur, model, ctx, alter=True)
        if f is not None:
            f.meta["modify"] = True
            if old:
                f.meta["renamed_from"] = old
    elif cur.skip_words("ALTER"):
        cur.skip_words("COLUMN")
        t = cur.peek()
        col = cur.take_ident()
        if not col or t is None:
            return
        f = ModelField(name=col, meta={"alter": True, "modify": True, "line": ctx.lines.line_of(t.start)})
        if cur.skip_words("TYPE") or cur.skip_words("SET", "DATA", "TYPE"):
            _parse_type(cur, f)
        elif cur.skip_words("SET", "DEFAULT"):
            f.default = _capture_expr(cur, ctx) or None
        elif cur.skip_words("SET", "NOT", "NULL"):
            f.nullable = False
        elif cur.skip_words("DROP", "NOT", "NULL"):
            f.nullable = True
        else:
            return
        model.fields.append(f)


def _create_view(cur: _Cursor, mods: List[str], stmt: str, line: int, line_end: int, ctx: _Ctx) -> None:
    cur.skip_words("IF", "NOT", "EXISTS")
    parts = cur.take_qualified()
    if not parts:
        return
    name = parts[-1]
    meta: Dict[str, Any] = {}
    if len(parts) > 1:
        meta["schema"] = ".".join(parts[:-1])
    if "MATERIALIZED" in mods:
        meta["materialized"] = True
    tables: List[str] = []
    for m in _FROM_JOIN_RE.finditer(stmt):
        t = ".".join(_split_qualified_text(m.group(1)))
        if t.upper() in ("SELECT", "LATERAL", "ONLY", "VALUES", "UNNEST"):
            continue
        if t not in tables:
            tables.append(t)
        if len(tables) >= 64:
            break
    if tables:
        meta["tables"] = tables
    sig = _collapse(stmt)[:_SIG_MAX]
    ctx.fa.symbols.append(ExtractedSymbol(name=name, kind="view", line_start=line, line_end=line_end,
                                          signature=sig, meta=meta))


def _create_index(cur: _Cursor, mods: List[str], line: int, line_end: int, ctx: _Ctx) -> None:
    cur.skip_words("CONCURRENTLY")
    cur.skip_words("IF", "NOT", "EXISTS")
    name_parts = [] if cur.is_word("ON") else cur.take_qualified()
    if not cur.skip_words("ON"):
        return
    cur.skip_words("ONLY")
    table_parts = cur.take_qualified()
    if not table_parts:
        return
    if cur.skip_words("USING"):
        cur.next()
    while not cur.at_end() and not cur.is_paren():
        cur.next()
    columns = _segment_texts(cur.next(), ctx.cleaned) if cur.is_paren() else []
    table = table_parts[-1]
    name = name_parts[-1] if name_parts else "%s_idx" % table
    meta: Dict[str, Any] = {"index": True, "table": table, "columns": columns, "unique": "UNIQUE" in mods}
    if len(table_parts) > 1:
        meta["schema"] = ".".join(table_parts[:-1])
    if not name_parts:
        meta["anonymous"] = True
    sig = "CREATE %sINDEX %s ON %s (%s)" % ("UNIQUE " if meta["unique"] else "", name, table, ", ".join(columns))
    ctx.fa.symbols.append(ExtractedSymbol(name=name, kind="variable", line_start=line, line_end=line_end,
                                          signature=sig[:_SIG_MAX], meta=meta))


def _create_routine(cur: _Cursor, routine: str, stmt: str, line: int, line_end: int, ctx: _Ctx) -> None:
    cur.skip_words("IF", "NOT", "EXISTS")
    parts = cur.take_qualified()
    if not parts:
        return
    name = parts[-1]
    meta: Dict[str, Any] = {"routine": routine}
    if len(parts) > 1:
        meta["schema"] = ".".join(parts[:-1])
    params = _collapse(cur.next().text) if cur.is_paren() else "()"  # type: ignore[union-attr]
    if cur.skip_words("RETURNS"):
        ret: List[str] = []
        while not cur.at_end() and len(ret) < 12:
            t = cur.peek()
            if t.kind == "word" and t.up in ("LANGUAGE", "AS", "BEGIN", "DETERMINISTIC", "NOT", "CONTAINS", "NO",
                                              "READS", "MODIFIES", "SQL", "COMMENT", "RETURN", "IMMUTABLE",
                                              "STABLE", "VOLATILE", "CALLED", "STRICT", "SECURITY", "PARALLEL",
                                              "COST", "ROWS", "SET", "WITH", "EXTERNAL", "IS"):
                break
            if t.kind == "op" and t.text in ("$", ";"):
                break
            ret.append(_collapse(cur.next().text))  # type: ignore[union-attr]
        if ret:
            meta["returns"] = " ".join(ret)
    m = _LANGUAGE_RE.search(stmt)
    if m:
        meta["language"] = m.group(1).lower()
    sig = ("%s%s" % (name, params))
    if meta.get("returns"):
        sig += " RETURNS " + meta["returns"]
    ctx.fa.symbols.append(ExtractedSymbol(name=name, kind="function", line_start=line, line_end=line_end,
                                          signature=sig[:_SIG_MAX], meta=meta))


def _create_trigger(cur: _Cursor, stmt: str, line: int, line_end: int, ctx: _Ctx) -> None:
    cur.skip_words("IF", "NOT", "EXISTS")
    parts = cur.take_qualified()
    if not parts:
        return
    name = parts[-1]
    meta: Dict[str, Any] = {"routine": "trigger"}
    if len(parts) > 1:
        meta["schema"] = ".".join(parts[:-1])
    if cur.skip_words("INSTEAD", "OF"):
        meta["timing"] = "INSTEAD OF"
    elif cur.is_word("BEFORE", "AFTER"):
        meta["timing"] = cur.next().up  # type: ignore[union-attr]
    events: List[str] = []
    guard = 0
    while not cur.at_end() and guard < 64:
        guard += 1
        if cur.is_word("INSERT", "UPDATE", "DELETE", "TRUNCATE"):
            events.append(cur.next().up)  # type: ignore[union-attr]
            if cur.skip_words("OF"):  # UPDATE OF col1, col2
                while not cur.at_end() and not cur.is_word("OR", "ON"):
                    cur.next()
        elif cur.skip_words("OR"):
            continue
        else:
            break
    if events:
        meta["events"] = events
    if cur.skip_words("ON"):
        tparts = cur.take_qualified()
        if tparts:
            meta["table"] = tparts[-1]
    m = _EXECUTE_RE.search(stmt)
    if m:
        meta["function"] = ".".join(_split_qualified_text(m.group(1)))
    sig = "TRIGGER %s %s %s ON %s" % (name, meta.get("timing", ""), " OR ".join(events), meta.get("table", "?"))
    ctx.fa.symbols.append(ExtractedSymbol(name=name, kind="function", line_start=line, line_end=line_end,
                                          signature=_collapse(sig)[:_SIG_MAX], meta=meta))


def _create_type(cur: _Cursor, line: int, line_end: int, ctx: _Ctx) -> None:
    parts = cur.take_qualified()
    if not parts:
        return
    name = parts[-1]
    meta: Dict[str, Any] = {}
    if len(parts) > 1:
        meta["schema"] = ".".join(parts[:-1])
    kind = "type"
    if cur.skip_words("AS", "ENUM") and cur.is_paren():
        kind = "enum"
        meta["values"] = [_unquote(it.text) for it in cur.next().items if it.kind in ("str", "dq")]  # type: ignore[union-attr]
    sig = "TYPE %s" % name
    if kind == "enum":
        sig += " AS ENUM (%s)" % ", ".join(meta["values"])
    ctx.fa.symbols.append(ExtractedSymbol(name=name, kind=kind, line_start=line, line_end=line_end,
                                          signature=sig[:_SIG_MAX], meta=meta))


# --------------------------------------------------------------------------- extracteur

class SqlExtractor:
    """Extracteur pour les fichiers ``.sql`` (voir la docstring du module)."""

    languages: List[str] = ["sql"]

    def analyze(self, text: str, path: str) -> FileAnalysis:
        fa = FileAnalysis(language="sql")
        try:
            self._analyze(text if isinstance(text, str) else str(text or ""), fa)
        except Exception as exc:  # noqa: BLE001 - jamais d'exception sur une entrée exotique
            fa.warnings.append("échec de l'analyse SQL de %s : %s" % (path, exc))
        return fa

    @staticmethod
    def _analyze(text: str, fa: FileAnalysis) -> None:
        if not text or not text.strip():
            return
        lines = _LineIndex(text)
        scan = _scan(text, lines)
        fa.imports.extend(scan.imports)
        fa.warnings.extend(scan.warnings)
        ctx = _Ctx(scan.cleaned, lines, fa)
        cleaned = scan.cleaned
        for raw_start, raw_end in scan.statements:
            lead = _LEAD_RE.match(cleaned, raw_start, raw_end)
            start = lead.end() if lead else raw_start
            if start >= raw_end:
                continue
            kw = _FIRST_WORD_RE.match(cleaned, start, raw_end)
            if kw is None:
                continue
            word = kw.group().upper()
            if word not in ("CREATE", "ALTER"):
                continue  # INSERT, UPDATE, SET, DROP… : ignorés silencieusement
            stmt = cleaned[start:raw_end].rstrip()
            end = start + len(stmt)
            line = lines.line_of(start)
            try:
                cur = _Cursor(_tokenize(stmt, start))
                cur.next()  # CREATE / ALTER
                if word == "CREATE":
                    _handle_create(cur, stmt, start, end, line, ctx)
                else:
                    _handle_alter(cur, line, ctx)
            except Exception as exc:  # noqa: BLE001 - une instruction illisible n'empêche pas les autres
                fa.warnings.append("instruction illisible à la ligne %d : %s" % (line, exc))


EXTRACTOR = SqlExtractor()
