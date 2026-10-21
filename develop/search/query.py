"""Analyse d'une question utilisateur avant la recherche.

On extrait, avec des règles lisibles (aucun modèle de langage) :

* les **identifiants** (camelCase, PascalCase, snake_case, kebab-case avec chiffres, tokens entre
  accents graves, noms de fichiers avec extension) ;
* les **chemins** (segments séparés par ``/`` ou nom de fichier avec extension) ;
* les **routes** HTTP (``/api/orders``, ``POST /orders/{id}``, URL complète) ;
* les **tables / modèles** cités (« la table orders », « le modèle Order », snake_case pluriels) ;
* l'**intention** de la question (``factual``, ``fields``, ``who_calls``, ``impact``, ``decision``,
  ``broad``, ``howto``) ;
* un **intervalle de dates** (« hier », « la semaine dernière », « en septembre », « depuis le 1er
  mars », « 2026-09-01 », « entre X et Y ») ;
* les **mots-clés** significatifs (sans mots vides français/anglais, accents conservés) et la
  requête lexicale correspondante.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

INTENTS = ("factual", "fields", "who_calls", "impact", "decision", "broad", "howto")

HTTP_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "ANY")


@dataclass
class QueryAnalysis:
    """Résultat de :func:`analyze_query`."""

    text: str
    identifiers: List[str] = field(default_factory=list)
    paths: List[str] = field(default_factory=list)
    routes: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    intent: str = "factual"
    date_from: Optional[str] = None  # AAAA-MM-JJ inclusif
    date_to: Optional[str] = None  # AAAA-MM-JJ inclusif
    keywords: List[str] = field(default_factory=list)
    lexical_query: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- normalisation

_QUOTE_MAP = {"’": "'", "‘": "'", "´": "'", "“": '"', "”": '"'}


def fold(text: str) -> str:
    """Minuscules sans accents, **en conservant la longueur** : les positions trouvées dans le texte
    replié correspondent exactement à celles du texte d'origine."""
    out: List[str] = []
    for ch in text or "":
        ch = _QUOTE_MAP.get(ch, ch)
        decomposed = unicodedata.normalize("NFD", ch)
        base = decomposed[0] if decomposed else ch
        low = base.lower()
        out.append(low if len(low) == 1 else base)
    return "".join(out)


# --------------------------------------------------------------------------- mots vides

_STOPWORDS_FR = """
le la les l un une des du de d et ou où mais donc or ni car que qui quoi dont quand comment pourquoi
est sont a ai as avons avez ont être etre été ete avoir fait faire fais font je tu il elle on nous vous ils
elles me te se ce cet cette ces mon ma mes ton ta tes son sa ses notre votre nos vos leur leurs y en au aux
dans sur sous par pour avec sans vers chez entre depuis pendant pas ne plus moins très tres trop peu tout
tous toute toutes quel quelle quels quelles lequel laquelle lesquels lesquelles ceci cela ça ca là la ici si
oui non aussi comme même meme autre autres encore déjà deja alors ainsi bien puis dès des lors hier
aujourd hui demain été soit sera serait était etait étaient etaient ayant étant etant va vont peut peux
peuvent pouvez pouvons doit doivent dois devons devez veux veut voulez voulons faut fallait existe
existent moi toi lui eux celui celle ceux celles cet cette qu quoi donc ne ni
""".split()

_STOPWORDS_EN = """
the a an and or but of to in on at by for with from as is are was were be been being this that these
those it its i you he she we they them their there here what which who whom whose when where why how do
does did done have has had having not no yes can could should would will shall may might must about into
over under again further then once all any both each few more most other some such only own same so than
too very just now also me my mine your yours our ours his her hers if else than via per
""".split()

# Bruit fréquent dans les questions sur du code : trop courts ou trop génériques pour classer.
_NOISE = {"id", "ok", "etc", "ex", "cf", "nb", "ps", "svp", "stp", "please", "merci", "thanks", "bonjour", "salut",
          "hello", "hi", "hey"}

STOPWORDS = set(_STOPWORDS_FR) | set(_STOPWORDS_EN) | _NOISE
_STOPWORDS_FOLDED = {fold(w) for w in STOPWORDS}


def is_stopword(word: str) -> bool:
    return fold(word) in _STOPWORDS_FOLDED


# --------------------------------------------------------------------------- intentions

# Motifs appliqués au texte replié (minuscules, sans accents). Premier motif qui correspond, dans l'ordre.
_INTENT_RULES: List[Tuple[str, List[str]]] = [
    ("impact", [
        r"qu'?\s*est[- ]ce que (je|j'|on|ca|ça) cass",
        r"\bje casse\b", r"\bon casse\b", r"\bcasse[- ]t[- ]", r"\bva casser\b",
        r"\bimpacts?\b", r"\bimpact[ée]e?s?\b", r"\banalyse d'impact",
        r"\bsi (je|j'|on|l'on) (modifi|renomm|supprim|chang|retir|deplac|enlev|refactor|touche)",
        r"\bif (i|we) (change|rename|delete|remove|modify|refactor|touch)",
        r"\bwhat (would )?break", r"\bbreak(s|ing)?\b.*\bif\b", r"\baffect[ée]?s?\b", r"\baffected\b",
        r"\bconsequences?\b", r"\brepercussions?\b", r"\beffets? de bord",
    ]),
    ("who_calls", [
        r"\bqui appelle", r"\bqui utilise", r"\bqui importe", r"\bqui depend", r"\bqui invoque", r"\bqui consomme",
        r"\bou (est|sont)[- ]?(il |elle |ils |elles )?utilis", r"\bou est[- ]ce (que c'est |qu'on l')?utilis",
        r"\bappelants?\b", r"\bappelee? (par|depuis)\b", r"\butilisee? (par|dans|depuis)\b",
        r"\bwho calls", r"\bwho uses", r"\bcallers?\b", r"\bwhere is .* used", r"\bused by\b", r"\busages? (de|du|des|of)\b",
        r"\breferences? (a|vers|to)\b", r"\bqui fait appel",
    ]),
    ("howto", [
        r"\bcomment (faire|ajouter|creer|implementer|mettre en place|configurer|ecrire|modifier|tester|lancer|installer|"
        r"brancher|appeler|utiliser|declarer|definir|generer|deployer|migrer|activer|desactiver|afficher|envoyer|recuperer|"
        r"construire|integrer|gerer|supprimer|rajouter|je|on|puis-je|peut-on|dois-je)\b",
        r"\bhow (to|do i|do we|can i|can we|should i|should we|would i)\b", r"\bprocedure pour\b", r"\betapes pour\b",
        r"\bmarche a suivre\b", r"\btutoriel\b", r"\bguide pour\b",
    ]),
    ("decision", [
        r"\bpourquoi\b", r"\bpour quelle raison", r"\bdecid", r"\bdecision", r"\breunion", r"\bmeeting",
        r"\bquand a[- ]t[- ]on", r"\bquand est[- ]ce qu'?on a", r"\bquand (avons|avez|ont)[- ]", r"\bhistorique",
        r"\bhistory\b", r"\bwhy\b", r"\bwhen did we", r"\bchoisi\b", r"\bchoix\b", r"\barbitrage", r"\bcompte[- ]rendu",
        r"\bjournal\b", r"\bnotes? (de|du|des)\b", r"\ba[- ]t[- ]on (decide|choisi|valide|acte)", r"\bqui a (decide|choisi)",
    ]),
    ("broad", [
        r"\baudit", r"\barchitecture (globale|generale|du projet|de l'application|de l'app)\b", r"\bvue d'ensemble",
        r"\bvue globale", r"\bexpliqu\w*\s+(moi\s+)?(le\s+|ce\s+|l')?projet", r"\boverview", r"\bbig picture",
        r"\bexplain (the )?(project|codebase|architecture|whole)", r"\bcomment (fonctionne|marche) (le |l')?(projet|application|app|code|tout)\b",
        r"\bresume\w*\s+(moi\s+)?(le\s+|ce\s+|du\s+|l')?projet", r"\bpresent\w*\s+(moi\s+)?(le\s+|ce\s+|l')?projet",
        r"\bcartographie", r"\btour d'horizon", r"\bde quoi (est fait|se compose|parle) (le|ce) projet",
        r"\bc'est quoi (ce|le) projet", r"\bque fait (ce|le) projet", r"\bqualite (du|de) (code|projet)",
    ]),
    ("fields", [
        r"\bquels? (sont )?(les )?champs", r"\bquelles? (sont )?(les )?colonnes", r"\bquelles? (sont )?(les )?proprietes",
        r"\bquels? (sont )?(les )?attributs", r"\bstructure (de|du|des|d')", r"\bchamps (de|du|des|d')",
        r"\bcolonnes (de|du|des|d')", r"\bschema (de|du|des|d')", r"\battributs (de|du|des|d')", r"\bproprietes (de|du|des|d')",
        r"\bwhat (are )?(the )?(fields|columns|attributes|properties)", r"\bfields (of|in|for)\b", r"\bcolumns (of|in|for)\b",
        r"\bstructure of\b", r"\bschema of\b", r"\bliste des champs", r"\bcontient quels champs", r"\bcontient quoi\b",
        r"\bdefinition (de la|du|de l') (table|modele|entite)", r"\btype (de|du) champ",
    ]),
]
_INTENT_COMPILED = [(intent, [re.compile(p) for p in pats]) for intent, pats in _INTENT_RULES]


def detect_intent(text: str) -> str:
    """Intention d'une question (texte brut) selon des motifs français/anglais ; ``factual`` par défaut."""
    folded = fold(text or "")
    for intent, patterns in _INTENT_COMPILED:
        for pat in patterns:
            if pat.search(folded):
                return intent
    return "factual"


# --------------------------------------------------------------------------- dates

_MONTHS: Dict[str, int] = {
    "janvier": 1, "janv": 1, "jan": 1, "january": 1,
    "fevrier": 2, "fev": 2, "feb": 2, "february": 2,
    "mars": 3, "mar": 3, "march": 3,
    "avril": 4, "avr": 4, "apr": 4, "april": 4,
    "mai": 5, "may": 5,
    "juin": 6, "jun": 6, "june": 6,
    "juillet": 7, "juil": 7, "jul": 7, "july": 7,
    "aout": 8, "aug": 8, "august": 8,
    "septembre": 9, "sept": 9, "sep": 9, "september": 9,
    "octobre": 10, "oct": 10, "october": 10,
    "novembre": 11, "nov": 11, "november": 11,
    "decembre": 12, "dec": 12, "december": 12,
}
_MONTH_ALT = "(?:%s)" % "|".join(sorted(_MONTHS, key=len, reverse=True))
_YEAR = r"(?:19|20)\d{2}"
_ISO = r"\d{4}-\d{2}-\d{2}"
_DMY = r"\d{1,2}/\d{1,2}/\d{4}"
_DAY = r"\d{1,2}(?:er|ere|eme|e|st|nd|rd|th)?"
_REL_WORDS = r"(?:avant-hier|hier|aujourd'hui|yesterday|today)"
# Atome « complet » : une date identifiable seule.
_ATOM_FULL = (r"(?:%s|%s|(?:le\s+)?%s\s+(?:de\s+|of\s+)?%s(?:\s+%s)?|%s\s+%s|%s(?:\s+%s)?|%s|%s)"
              % (_ISO, _DMY, _DAY, _MONTH_ALT, _YEAR, _MONTH_ALT, _DAY, _MONTH_ALT, _YEAR, _YEAR, _REL_WORDS))
# Atome « gauche » d'un intervalle : accepte aussi un jour nu (« entre le 1er et le 15 mars »).
_ATOM_LEFT = r"(?:%s|(?:le\s+)?%s)" % (_ATOM_FULL, _DAY)

_RANGE_RULES = [
    re.compile(r"\bentre\s+(?P<a>%s)\s+et\s+(?P<b>%s)" % (_ATOM_LEFT, _ATOM_FULL)),
    re.compile(r"\bbetween\s+(?P<a>%s)\s+and\s+(?P<b>%s)" % (_ATOM_LEFT, _ATOM_FULL)),
    re.compile(r"\bdu\s+(?P<a>%s)\s+au\s+(?P<b>%s)" % (_ATOM_LEFT, _ATOM_FULL)),
    re.compile(r"\bfrom\s+(?P<a>%s)\s+(?:to|until|till)\s+(?P<b>%s)" % (_ATOM_LEFT, _ATOM_FULL)),
]
_FROM_RULES = [
    re.compile(r"\b(?:depuis|since|a partir (?:du|de|de la)|apres le|apres|after|des le)\s+(?P<a>%s)" % _ATOM_FULL),
]
_TO_RULES = [
    re.compile(r"\b(?:jusqu'au|jusqu'a|jusqu'en|avant le|avant|until|before|up to|till)\s+(?P<a>%s)" % _ATOM_FULL),
]
_IN_RULES = [
    re.compile(r"\b(?:en|in|durant|pendant|au cours de|au mois de|au mois d'|courant)\s+(?P<a>%s(?:\s+%s)?|%s)\b"
               % (_MONTH_ALT, _YEAR, _YEAR)),
]
_BARE_RULES = [
    re.compile(r"(?<![\w/-])(?P<a>%s)(?![\w/-])" % _ISO),
    re.compile(r"(?<![\w/])(?P<a>%s)(?![\w/])" % _DMY),
    re.compile(r"\b(?P<a>(?:le\s+)?%s\s+(?:de\s+|of\s+)?%s(?:\s+%s)?)\b" % (_DAY, _MONTH_ALT, _YEAR)),
    re.compile(r"\b(?P<a>%s\s+%s)\b" % (_MONTH_ALT, _DAY)),  # « September 3 »
    re.compile(r"\b(?P<a>%s\s+%s)\b" % (_MONTH_ALT, _YEAR)),
]
_RELATIVE_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bla semaine derniere\b|\blast week\b|\bsemaine passee\b"), "last_week"),
    (re.compile(r"\bcette semaine\b|\bthis week\b"), "this_week"),
    (re.compile(r"\ble mois dernier\b|\blast month\b|\bmois passe\b"), "last_month"),
    (re.compile(r"\bce mois(?:-ci)?\b|\bthis month\b"), "this_month"),
    (re.compile(r"\bl'annee derniere\b|\blast year\b|\bannee passee\b"), "last_year"),
    (re.compile(r"\bcette annee\b|\bthis year\b"), "this_year"),
    (re.compile(r"\bavant-hier\b"), "day_before_yesterday"),
    (re.compile(r"\bhier\b|\byesterday\b"), "yesterday"),
    (re.compile(r"\baujourd'hui\b|\btoday\b"), "today"),
    (re.compile(r"\bil y a (?P<n>\d+) (?P<u>jours?|semaines?|mois|ans?|annees?)\b"), "ago"),
    (re.compile(r"\b(?P<n>\d+) (?P<u>days?|weeks?|months?|years?) ago\b"), "ago"),
    (re.compile(r"\b(?:les|ces) (?P<n>\d+) derni[eè]r(?:e)?s (?P<u>jours|semaines|mois|annees|ans)\b"), "last_n"),
    (re.compile(r"\b(?:last|past) (?P<n>\d+) (?P<u>days|weeks|months|years)\b"), "last_n"),
    (re.compile(r"\b(?:ces|les) derni[eè]r(?:e)?s (?P<u>jours|semaines|mois)\b"), "last_few"),
    (re.compile(r"\b(?:ces|les) derni[eè]re?s? (?P<u>temps)\b"), "last_few"),
    (re.compile(r"\brecemment\b|\brecently\b|\bdernierement\b|\blately\b"), "recent"),
]


def _month_bounds(year: int, month: int) -> Tuple[dt.date, dt.date]:
    last = calendar.monthrange(year, month)[1]
    return dt.date(year, month, 1), dt.date(year, month, last)


def _shift_months(d: dt.date, months: int) -> dt.date:
    m = d.month - 1 + months
    y = d.year + m // 12
    m = m % 12 + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _parse_atom(raw: str, today: dt.date, hint: Optional[Tuple[int, Optional[int]]] = None
                ) -> Optional[Tuple[dt.date, dt.date]]:
    """Convertit un atome de date (texte replié) en intervalle ``(début, fin)`` inclusif.

    ``hint`` = (mois, année) emprunté à la borne droite d'un intervalle pour un jour nu.
    """
    s = re.sub(r"\s+", " ", (raw or "").strip())
    s = re.sub(r"^(le|de|of) ", "", s)
    if not s:
        return None
    if s == "hier" or s == "yesterday":
        d = today - dt.timedelta(days=1)
        return d, d
    if s == "avant-hier":
        d = today - dt.timedelta(days=2)
        return d, d
    if s == "aujourd'hui" or s == "today":
        return today, today
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
        return d, d
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        try:
            d = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
        return d, d
    m = re.fullmatch(r"(\d{1,2})(?:er|ere|eme|e|st|nd|rd|th)? (?:de |of )?(%s)(?: (\d{4}))?" % _MONTH_ALT, s)
    if not m:
        m2 = re.fullmatch(r"(%s) (\d{1,2})(?:er|ere|eme|e|st|nd|rd|th)?" % _MONTH_ALT, s)
        if m2:
            return _parse_atom("%s %s" % (m2.group(2), m2.group(1)), today, hint)
    if m:
        day, month = int(m.group(1)), _MONTHS[m.group(2)]
        year = int(m.group(3)) if m.group(3) else None
        if year is None:
            year = today.year
            try:
                if dt.date(year, month, day) > today:
                    year -= 1
            except ValueError:
                return None
        try:
            d = dt.date(year, month, day)
        except ValueError:
            return None
        return d, d
    m = re.fullmatch(r"(%s)(?: (\d{4}))?" % _MONTH_ALT, s)
    if m:
        month = _MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else (today.year if month <= today.month else today.year - 1)
        return _month_bounds(year, month)
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        return dt.date(y, 1, 1), dt.date(y, 12, 31)
    m = re.fullmatch(r"(\d{1,2})(?:er|ere|eme|e|st|nd|rd|th)?", s)
    if m and hint:
        month, year = hint
        if year is None:
            year = today.year if month <= today.month else today.year - 1
        try:
            d = dt.date(year, month, int(m.group(1)))
        except ValueError:
            return None
        return d, d
    return None


def _relative_interval(kind: str, m: "re.Match[str]", today: dt.date) -> Optional[Tuple[Optional[dt.date], Optional[dt.date]]]:
    if kind == "last_week":
        monday = today - dt.timedelta(days=today.weekday() + 7)
        return monday, monday + dt.timedelta(days=6)
    if kind == "this_week":
        return today - dt.timedelta(days=today.weekday()), today
    if kind == "last_month":
        prev = _shift_months(today.replace(day=1), -1)
        return _month_bounds(prev.year, prev.month)
    if kind == "this_month":
        return today.replace(day=1), today
    if kind == "last_year":
        return dt.date(today.year - 1, 1, 1), dt.date(today.year - 1, 12, 31)
    if kind == "this_year":
        return dt.date(today.year, 1, 1), today
    if kind == "day_before_yesterday":
        d = today - dt.timedelta(days=2)
        return d, d
    if kind == "yesterday":
        d = today - dt.timedelta(days=1)
        return d, d
    if kind == "today":
        return today, today
    if kind == "recent":
        return today - dt.timedelta(days=30), today
    unit = (m.groupdict().get("u") or "").lower()
    n_raw = m.groupdict().get("n")
    if kind == "last_few":
        n = {"jours": 7, "semaines": 4, "mois": 3, "temps": 30}.get(unit, 7)
        if unit == "semaines":
            return today - dt.timedelta(days=7 * n), today
        if unit == "mois":
            return _shift_months(today, -n), today
        return today - dt.timedelta(days=n), today
    try:
        n = max(0, int(n_raw or 0))
    except ValueError:
        return None
    if unit.startswith(("jour", "day")):
        delta = dt.timedelta(days=n)
        if kind == "ago":
            d = today - delta
            return d, d
        return today - delta, today
    if unit.startswith(("semaine", "week")):
        if kind == "ago":
            d = today - dt.timedelta(days=7 * n)
            monday = d - dt.timedelta(days=d.weekday())
            return monday, monday + dt.timedelta(days=6)
        return today - dt.timedelta(days=7 * n), today
    if unit.startswith(("mois", "month")):
        if kind == "ago":
            d = _shift_months(today, -n)
            return _month_bounds(d.year, d.month)
        return _shift_months(today, -n), today
    if unit.startswith(("an", "year")):
        if kind == "ago":
            return dt.date(today.year - n, 1, 1), dt.date(today.year - n, 12, 31)
        return dt.date(today.year - n, today.month, min(today.day, 28)), today
    return None


def _month_hint(raw: str) -> Optional[Tuple[int, Optional[int]]]:
    m = re.search(r"(%s)(?:\s+(\d{4}))?" % _MONTH_ALT, raw)
    if not m:
        m2 = re.search(r"(\d{4})-(\d{2})-\d{2}", raw)
        if m2:
            return int(m2.group(2)), int(m2.group(1))
        m3 = re.search(r"\d{1,2}/(\d{1,2})/(\d{4})", raw)
        if m3:
            return int(m3.group(1)), int(m3.group(2))
        return None
    return _MONTHS[m.group(1)], (int(m.group(2)) if m.group(2) else None)


def extract_dates(text: str, today: Optional[str] = None) -> Tuple[Optional[str], Optional[str], List[Tuple[int, int]]]:
    """``(date_from, date_to, spans)`` : intervalle inclusif AAAA-MM-JJ (bornes optionnelles) et
    positions (début, fin) des expressions de date consommées dans ``text``."""
    base = _today(today)
    folded = fold(text or "")
    spans: List[Tuple[int, int]] = []
    date_from: Optional[dt.date] = None
    date_to: Optional[dt.date] = None

    def consume(m: "re.Match[str]") -> None:
        nonlocal folded
        spans.append((m.start(), m.end()))
        folded = folded[:m.start()] + " " * (m.end() - m.start()) + folded[m.end():]

    for rule in _RANGE_RULES:
        m = rule.search(folded)
        if not m:
            continue
        right = _parse_atom(m.group("b"), base)
        left = _parse_atom(m.group("a"), base, hint=_month_hint(m.group("b")))
        if left and right:
            consume(m)
            date_from, date_to = left[0], right[1]
            if date_from > date_to:
                date_from, date_to = date_to, date_from
            break
    if date_from is None and date_to is None:
        for rule in _FROM_RULES:
            m = rule.search(folded)
            if m:
                parsed = _parse_atom(m.group("a"), base)
                if parsed:
                    consume(m)
                    date_from = parsed[0]
                    break
        for rule in _TO_RULES:
            m = rule.search(folded)
            if m:
                parsed = _parse_atom(m.group("a"), base)
                if parsed:
                    consume(m)
                    date_to = parsed[1]
                    break
    if date_from is None and date_to is None:
        for rule in _IN_RULES:
            m = rule.search(folded)
            if m:
                parsed = _parse_atom(m.group("a"), base)
                if parsed:
                    consume(m)
                    date_from, date_to = parsed
                    break
    if date_from is None and date_to is None:
        for rule in _BARE_RULES:
            m = rule.search(folded)
            if m:
                parsed = _parse_atom(m.group("a"), base)
                if parsed:
                    consume(m)
                    date_from, date_to = parsed
                    break
    if date_from is None and date_to is None:
        for rule, kind in _RELATIVE_RULES:
            m = rule.search(folded)
            if m:
                interval = _relative_interval(kind, m, base)
                if interval:
                    consume(m)
                    date_from, date_to = interval
                    break
    return (date_from.isoformat() if date_from else None, date_to.isoformat() if date_to else None, spans)


def _today(today: Optional[str]) -> dt.date:
    if today:
        try:
            return dt.date.fromisoformat(str(today)[:10])
        except ValueError:
            pass
    return dt.date.today()


# --------------------------------------------------------------------------- identifiants / chemins / routes / tables

_EXTENSIONS = ("blade.php|php|tsx|ts|jsx|js|mjs|cjs|py|rb|go|rs|java|kt|kts|dart|swift|sql|md|markdown|txt|csv|tsv|json|"
               "yaml|yml|toml|ini|env|xml|html|htm|css|scss|less|vue|svelte|prisma|graphql|gql|sh|pdf|docx|proto")
_FILENAME_RE = re.compile(r"(?<![\w/.-])([A-Za-z_][\w.-]*?)\.(%s)(?![\w/])" % _EXTENSIONS)
_FILENAME_IN_PATH_RE = re.compile(r"\.(%s)$" % _EXTENSIONS)
_BACKTICK_RE = re.compile(r"`([^`\n]{1,200})`")
_QUALIFIED_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w*)(?:::|->|\.)([A-Za-z_]\w*)\b")
_CALL_RE = re.compile(r"(?<![\w.])([A-Za-z_]\w{1,})\(")
_CAMEL_RE = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b")
_PASCAL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_UPPER_SNAKE_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_KEBAB_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")
_CAPITALIZED_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_TABLE_TRIGGER_RE = re.compile(
    r"\b(?:tables?|mod[eè]les?|models?|entit[eé]s?|entity|entities|collections?)\s+"
    r"(?:(?:la|le|les|l'|de|du|des|d'|the|of|a|an|un|une|nomm[eé]e?|appel[eé]e?|called|named)\s+)*"
    r"[`\"'«]?\s*([A-Za-z_][\w]*)", re.IGNORECASE)
_METHOD_RE = re.compile(r"^(%s)$" % "|".join(HTTP_METHODS), re.IGNORECASE)
_STRIP_CHARS = "\"'`«»“”‘’(),;!?."
_IDENT_CHARS_RE = re.compile(r"^[A-Za-z_][\w]*$")

# Mots capitalisés très courants en tête de proposition qu'on ne veut pas prendre pour des identifiants.
_CAPITALIZED_SKIP = {fold(w) for w in (
    "Quel Quelle Quels Quelles Comment Pourquoi Quand Est Peux Peut Donne Montre Explique Liste Cherche Trouve "
    "Dans Avec Pour Sur Sous Les Des Une Mais Donc Alors Ainsi Oui Non The What Which Where When Why How Show "
    "List Find Give Explain Please Merci Bonjour Salut Hello Also Then This That Here There Develop").split()}


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def _clean_token(tok: str) -> str:
    return tok.strip(_STRIP_CHARS).rstrip(":")


def _extract_slashy(text: str) -> Tuple[List[str], List[str]]:
    """Chemins et routes : jetons contenant ``/`` (ou nom de fichier avec extension)."""
    paths: List[str] = []
    routes: List[str] = []
    tokens = text.split()
    for i, raw in enumerate(tokens):
        tok = _clean_token(raw)
        if not tok:
            continue
        if re.match(r"^https?://", tok, re.IGNORECASE):
            routes.append(tok)
            continue
        if "/" not in tok:
            if _FILENAME_IN_PATH_RE.search(tok) and _FILENAME_RE.fullmatch(tok):
                paths.append(tok)
            continue
        if tok in ("/", "//") or tok.startswith("//"):
            continue
        if _FILENAME_IN_PATH_RE.search(tok) and "{" not in tok and ":" not in tok:
            p = tok[2:] if tok.startswith("./") else tok
            paths.append(p)
            continue
        looks_route = tok.startswith("/") or "{" in tok or "${" in tok or re.search(r"/:[A-Za-z_]", tok) is not None
        if looks_route:
            method = ""
            if i > 0:
                prev = _clean_token(tokens[i - 1])
                if _METHOD_RE.match(prev):
                    method = prev.upper()
            route = tok if tok.startswith("/") else "/" + tok
            routes.append(("%s %s" % (method, route)) if method else route)
            continue
        if tok.startswith("~") or tok.startswith(".."):
            paths.append(tok)
            continue
        p = tok[2:] if tok.startswith("./") else tok
        if re.match(r"^[\w.-]+(?:/[\w.-]+)+/?$", p):
            paths.append(p)
    return _dedupe(paths), _dedupe(routes)


def _sentence_start(text: str, pos: int) -> bool:
    j = pos - 1
    while j >= 0 and text[j] in " \t\"'«»(":
        j -= 1
    return j < 0 or text[j] in ".!?:\n;"


def _mask_slashy(text: str) -> Tuple[str, List[str]]:
    """Remplace par des espaces les jetons contenant ``/`` (chemins, routes, URL) et renvoie les
    radicaux de nom de fichier qu'ils contenaient (``app/Models/Order.php`` → ``Order``)."""
    stems: List[str] = []
    pieces: List[str] = []
    for raw in re.split(r"(\s+)", text):
        tok = _clean_token(raw)
        if "/" in tok and len(tok) > 1:
            base = tok.rstrip("/").rsplit("/", 1)[-1]
            m = _FILENAME_RE.fullmatch(base)
            if m and len(m.group(1)) >= 3 and _IDENT_CHARS_RE.match(m.group(1)) and not is_stopword(m.group(1)):
                stems.append(m.group(1))
            pieces.append(" " * len(raw))
        else:
            pieces.append(raw)
    return "".join(pieces), stems


def extract_identifiers(text: str) -> List[str]:
    """Identifiants probables : camelCase, PascalCase, snake_case, kebab-case avec chiffres, tokens
    entre accents graves, noms de fichiers, membres qualifiés (``A::b``, ``a.b()``) et appels ``f()``."""
    out: List[str] = []
    for m in _BACKTICK_RE.finditer(text):
        inner = m.group(1).strip()
        if "/" in inner or " " in inner:
            continue
        inner = _clean_token(inner)
        if inner and len(inner) >= 2:
            out.append(inner)
    text, path_stems = _mask_slashy(text)
    out.extend(path_stems)
    for m in _FILENAME_RE.finditer(text):
        out.append(m.group(0))
        stem = m.group(1)
        if len(stem) >= 3 and _IDENT_CHARS_RE.match(stem) and not is_stopword(stem):
            out.append(stem)
    for m in _QUALIFIED_RE.finditer(text):
        left, right = m.group(1), m.group(2)
        if _FILENAME_IN_PATH_RE.search(m.group(0)):
            continue
        sep = m.group(0)[len(left):len(m.group(0)) - len(right)]
        if sep == "." and (is_stopword(left) or is_stopword(right)) and "(" not in text[m.end():m.end() + 1]:
            continue  # « fin.Phrase » : ponctuation, pas un membre
        if not is_stopword(left) or sep != ".":
            out.append(m.group(0))
            if not is_stopword(left):
                out.append(left)
            if not is_stopword(right):
                out.append(right)
    for m in _CALL_RE.finditer(text):
        name = m.group(1)
        if not is_stopword(name):
            out.append(name)
    for rx in (_CAMEL_RE, _PASCAL_RE, _SNAKE_RE, _UPPER_SNAKE_RE):
        for m in rx.finditer(text):
            tok = m.group(0)
            if not is_stopword(tok):
                out.append(tok)
    for m in _KEBAB_RE.finditer(text):
        tok = m.group(0)
        if any(ch.isdigit() for ch in tok):
            out.append(tok)
    for m in _CAPITALIZED_RE.finditer(text):
        tok = m.group(0)
        if _sentence_start(text, m.start()):
            continue
        if fold(tok) in _CAPITALIZED_SKIP or is_stopword(tok) or fold(tok) in _MONTHS:
            continue
        out.append(tok)
    return _dedupe([t for t in out if len(t) >= 2])


_TABLE_POSTFIX_RE = re.compile(r"\b([A-Za-z_]\w*)\s+(?:tables?|models?|entity|entities)\b", re.IGNORECASE)
_TABLE_TRIGGER_WORDS = {"table", "tables", "modele", "modeles", "model", "models", "entite", "entites", "entity",
                        "entities", "collection", "collections"}


def extract_tables(text: str, identifiers: List[str]) -> List[str]:
    """Tables / modèles cités : mot après « table », « modèle », « model », « entité » (ou avant « table »
    en anglais : « customers table ») + snake_case pluriels."""
    out: List[str] = []
    for m in _TABLE_TRIGGER_RE.finditer(text):
        name = m.group(1)
        if name and not is_stopword(name) and len(name) >= 2 and fold(name) not in _TABLE_TRIGGER_WORDS:
            out.append(name)
    for m in _TABLE_POSTFIX_RE.finditer(text):
        name = m.group(1)
        if name and not is_stopword(name) and len(name) >= 3 and fold(name) not in _TABLE_TRIGGER_WORDS:
            out.append(name)
    for ident in identifiers:
        if _SNAKE_RE.fullmatch(ident) and ident.endswith("s"):
            out.append(ident)
        elif re.fullmatch(r"[a-z][a-z0-9_]*", ident) and ident.endswith("s") and len(ident) >= 4 and "`%s`" % ident in text:
            out.append(ident)  # `orders` entre accents graves
    return _dedupe(out)


# --------------------------------------------------------------------------- mots-clés

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _keyword_text(text: str) -> str:
    """Texte préparé pour les mots-clés : un chemin devient son nom de fichier sans extension, une route
    perd ses paramètres (``{id}``, ``:id``, ``${id}``), une URL perd son hôte."""
    pieces: List[str] = []
    for raw in (text or "").split():
        tok = _clean_token(raw)
        if "/" not in tok:
            pieces.append(raw)
            continue
        tok = re.sub(r"^https?://[^/]*", "", tok, flags=re.IGNORECASE)
        if _FILENAME_IN_PATH_RE.search(tok) and "{" not in tok and ":" not in tok:
            base = tok.rstrip("/").rsplit("/", 1)[-1]
            pieces.append(_FILENAME_IN_PATH_RE.sub("", base))
        else:
            pieces.append(re.sub(r"\$?\{[^}]*\}|:[A-Za-z_]\w*|\[[^\]]*\]", " ", tok))
    return " ".join(pieces)


def extract_keywords(text: str) -> List[str]:
    """Mots significatifs (minuscules, accents conservés), sans mots vides ni nombres purs."""
    cleaned = _FILENAME_RE.sub(lambda m: m.group(1), _keyword_text(text))  # Order.php -> Order
    out: List[str] = []
    for tok in _TOKEN_RE.findall(cleaned):
        if len(tok) < 2 or tok.isdigit():
            continue
        low = tok.lower()
        if fold(low) in _STOPWORDS_FOLDED:
            continue
        out.append(low)
    return _dedupe(out)


def _blank_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        for i in range(max(0, a), min(len(chars), b)):
            chars[i] = " "
    return "".join(chars)


# --------------------------------------------------------------------------- point d'entrée

def analyze_query(text: str, today: Optional[str] = None) -> QueryAnalysis:
    """Analyse complète d'une question (voir le module). ``today`` (AAAA-MM-JJ) sert aux dates relatives."""
    raw = text if isinstance(text, str) else str(text or "")
    norm = "".join(_QUOTE_MAP.get(ch, ch) for ch in raw).strip()
    analysis = QueryAnalysis(text=raw)
    if not norm:
        return analysis
    analysis.intent = detect_intent(norm)
    date_from, date_to, spans = extract_dates(norm, today)
    analysis.date_from, analysis.date_to = date_from, date_to
    analysis.paths, analysis.routes = _extract_slashy(norm)
    analysis.identifiers = extract_identifiers(norm)
    analysis.tables = extract_tables(norm, analysis.identifiers)
    analysis.keywords = extract_keywords(_blank_spans(norm, spans))
    analysis.lexical_query = " ".join(analysis.keywords)
    return analysis


__all__ = ["INTENTS", "HTTP_METHODS", "STOPWORDS", "QueryAnalysis", "analyze_query", "detect_intent",
           "extract_dates", "extract_identifiers", "extract_keywords", "extract_tables", "fold", "is_stopword"]
