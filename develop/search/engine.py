"""Moteur de recherche hybride multi-canaux de Develop.

Pour une question, six canaux sont interrogés puis **fusionnés** (Reciprocal Rank Fusion) :

* ``identifier`` — correspondance exacte (insensible à la casse) des identifiants, routes, tables et
  chemins extraits de la question sur les symboles, endpoints, modèles de données et chemins de
  fichiers, plus les « mots collés » (``user controller`` → ``UserController``, ``order line`` →
  ``OrderLine``, paires et triplets de mots adjacents) ; un identifiant resté sans correspondance est
  ensuite cherché à l'**orthographe approchée** parmi les noms de symboles (``confrimOrder`` →
  ``confirmOrder``, ``OrderServce::confrim`` → ``OrderService::confirm``), à un poids inférieur, et
  chaque approximation est signalée dans ``SearchResult.warnings`` ; les mots génériques (« table »,
  « modèle »…) ne sont jamais pris pour des noms nus, et un **mot courant** sans forme de code
  (« definition », « put », « user » — :data:`develop.search.query.COMMON_WORDS`) n'est accepté comme
  nom exact que pour une classe / un modèle / une table, ou un membre d'une classe citée dans la même
  question (``OrderService confirm``) : les méthodes homonymes (``UserFactory::definition``) ont un
  poids nul — sauf quand la question **emploie le mot comme un nom de code** (:func:`word_names_symbol` :
  « que fait la méthode confirm ? », « explique store », « où est défini index ? », « confirm() »), les
  actions des contrôleurs étant presque toutes des mots courants ; les **expansions** de vocabulaire
  (``utilisateur`` → ``user``) sont essayées de la même façon, restreintes aux classes et modèles ;
* ``route`` — endpoints dont le chemin ou le gestionnaire **contient** un mot-clé, un mot collé ou un
  terme d'expansion de la question (« authentification » → ``auth``, ``login`` → ``POST /auth/login``) ;
  un terme présent dans plus de :data:`ROUTE_LIMIT` routes ne désigne rien (« api », « store »), ni un
  mot hérité du tour précédent d'une question de suivi qui n'est pas un identifiant (« commande » d'un titre
  cité) : ils ne servent qu'à départager les routes trouvées par un terme plus précis de la question. Les
  routes classées d'abord par preuves fortes (terme égal à un segment / une action), puis route serveur avant appel client, puis
  nombre de termes contenus ; un appel client dont le chemin n'est qu'une variable (« POST $url ») est
  ignoré. Après la fusion, un appel client ne précède jamais la route serveur qu'il vise (même chemin
  normalisé) : il est replacé juste derrière elle, sauf pour une question d'impact (« qui appelle ») ;
* ``lexical`` — FTS5 / BM25 sur les morceaux (code, documents, notes) avec les mots-clés, puis une
  requête secondaire en OU sur les termes d'expansion, à un poids plus faible (:data:`W_EXPANSION`) ;
  pour une question de suivi, les termes hérités du tour précédent forment eux aussi une requête
  secondaire en OU (:data:`W_PREVIOUS`) au lieu d'être exigés en ET avec les mots de la question ;
  les morceaux des **fichiers générés volumineux** (plus de :data:`GENERATED_FILE_MAX_BYTES`, ou
  ``api-docs``, ``swagger.json``, ``*.lock``, ``*.min.*``, ``storage/app|framework|logs/``, ``public/build``…) voient leur
  apport divisé par deux et un seul morceau par tel fichier reste en tête du canal, les autres étant
  renvoyés en queue (raison « fichier généré volumineux ») ;
* ``semantic`` — cosinus numpy sur les embeddings locaux (ignoré, sans erreur, quand aucun vecteur
  n'est disponible) ;
* ``graph`` — expansion d'un saut dans le graphe de relations à partir des noeuds trouvés par le
  canal ``identifier`` et des premières routes du canal ``route`` (gestionnaires de route, modèles
  liés, documents, fichiers importateurs…). Le **fichier** d'une graine n'est lui-même une graine
  que pour une classe / un symbole de premier niveau cité (ses importateurs l'utilisent) ou un fichier
  nommé dans la question, jamais pour une route (le fichier de routes importe tous les contrôleurs du
  module, pas celui de la route, atteint par ``handles``) ni pour un modèle — même via sa classe
  homonyme ou un mot nu égal à son nom — hors question d'impact (``who_calls`` / ``impact``, où les
  importateurs sont la surface d'impact) ; les fichiers atteints « via imports » sont limités à
  :data:`GRAPH_IMPORTS_PER_SEED` par graine et passent après ses autres relations : leur premier
  morceau (namespace + ``use``) n'apprend rien ;
* ``temporal`` — notes datées (intervalle de dates de la question, ou notes récentes pour une
  question de décision).

Un résultat de type ``endpoint`` porte dans ``extra`` son gestionnaire (``handler``), son cadriciel
(``framework``) et, si une classe de validation ``<Action><Contrôleur>Request`` /
``Store|Update|Index<Modèle>Request`` existe dans l'index, ``request_class`` (nom) et
``request_class_id``.

Après fusion : déduplication (même morceau, ou même fichier avec lignes qui se chevauchent),
diversité (au plus :data:`MAX_PER_SOURCE` morceaux par fichier/document/note), tri décroissant,
``limit``. Chaque résultat conserve ses preuves : canaux, score fusionné et raison en français.

Quand **aucun** canal ne renvoie rien (faute de frappe dans un mot nu : « orderservce confrim »),
un dernier repli essaie les mots-clés de la question comme noms de symboles approchés, puis
relance le canal graphe à partir de ce qui a été trouvé.

Le moteur ne lève jamais sur une question « sale » ou un index vide : il renvoie un résultat
cohérent (éventuellement vide) et consigne les problèmes dans ``SearchResult.warnings``.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from develop.llm.base import LLMClient
from develop.search import query as _query
from develop.search.query import ACTION_TERMS, QueryAnalysis, analyze_query, fold, has_code_shape, is_common_word
from develop.storage import (
    SOURCE_TYPES,
    ChunkRow,
    Database,
    DocumentRow,
    EndpointRow,
    FileRow,
    FtsHit,
    ModelRow,
    NoteRow,
    RepoRow,
    SymbolRow,
    build_fts_query,
    fts_terms,
    node_key,
)

try:  # le paquet code_analysis est en cours d'écriture par d'autres agents : on tolère son absence
    from develop.code_analysis.model import normalize_route_path as _normalize_route_path
except Exception:  # noqa: BLE001 - import défensif
    _normalize_route_path = None  # type: ignore[assignment]

logger = logging.getLogger("develop.search")

CHANNELS = ("identifier", "route", "lexical", "semantic", "graph", "temporal")
HIT_SOURCE_TYPES = ("file", "document", "note", "symbol", "endpoint", "model")
RRF_K = 60
IDENTIFIER_BONUS = 1.5  # les correspondances exactes valent plus que les approximations
CHANNEL_LIMIT = 50  # candidats maximum par canal avant fusion
# Paliers du canal « identifier » : plus la mention est explicite, plus elle pèse.
W_EXPLICIT = 2.0  # Classe::membre, route avec son chemin, chemin de fichier
W_NAMED = 1.0  # identifiant nommé (camelCase, snake_case…), table citée, nom de fichier, mots collés
W_BARE = 0.6  # mot-clé nu égal à un nom de symbole / modèle, radical de nom de fichier
W_FUZZY = 0.5  # orthographe approchée d'un identifiant (repli « faute de frappe »), sous les mots nus
W_EXPANSION = 0.5  # canal lexical : morceaux trouvés par les termes d'expansion seulement (requête en OU)
# Canal lexical, question de suivi : morceaux trouvés par les termes hérités du tour précédent (requête en OU) ; le
# sujet conservé pèse plus qu'une expansion de vocabulaire, moins que les mots de la question elle-même.
W_PREVIOUS = 0.75
# Mots courants (« definition », « user »…) : seuls ces genres de symboles comptent comme nom exact.
CLASS_KINDS: FrozenSet[str] = frozenset({"class", "interface", "trait", "enum", "struct", "record", "type", "typedef",
                                         "model", "entity", "component", "dataclass", "dart_class", "mixin",
                                         "protocol", "module", "table", "eloquent", "sqlalchemy", "django", "jpa"})
# … sauf quand la question **emploie le mot comme un nom de code** (:func:`word_names_symbol`) : les actions des
# contrôleurs (confirm, store, update, index, show, login…) sont presque toutes des mots courants.
# Mots de la phrase qui annoncent une méthode / fonction juste avant le nom (« la méthode confirm », « l'action
# store », « où est défini confirm »)…
_CODE_CUES_BEFORE: FrozenSet[str] = frozenset("""
methode methodes method methods fonction fonctions function functions func def defini definie definis definies
defined implemente implementee implementes implementees implemented declare declaree declares declarees declared
action actions membre membres member members handler handlers callback callbacks hook hooks helper helpers
accessor accessors mutator mutators getter getters setter setters scope scopes procedure procedures routine
routines symbole symboles symbol symbols
""".split())
# … ou juste après (« the confirm method », « confirm function »)…
_CODE_CUES_AFTER: FrozenSet[str] = frozenset("""
methode methodes method methods fonction fonctions function functions action actions handler handlers callback
callbacks helper helpers hook hooks membre member
""".split())
# … et les verbes d'explication dont le complément direct est le nom demandé (« explique confirm », « que fait
# store ? », « what does confirm do »).
_EXPLAIN_VERBS: FrozenSet[str] = frozenset("""
explique expliquer expliques expliquez explain explains decris decrire decrit decrivez describe describes montre
montrer montrez show shows affiche afficher affichez display fait font does do did sert marche fonctionne works
resume resumer summarize detaille detailler documente documenter commente commenter define defines definir
""".split())
# Jetons sautés en remontant vers le mot précédent (« explique-moi confirm », « montre me store »).
_POLITE_TOKENS: FrozenSet[str] = frozenset({"moi", "me", "svp", "stp", "please"})
# Déterminants et pronoms sujets : un mot qui les suit est un mot de la phrase, pas un nom de code (« les logs »,
# « le template », « tu put m'expliquer » — faute de frappe pour « peux »).
_DETERMINERS: FrozenSet[str] = frozenset("""
le la les l un une des du ce cet cette ces mon ma mes ton ta tes son sa ses notre votre nos vos leur leurs
the a an this that these those my your his her its our their je tu il elle on nous vous ils elles i you he she
we they it
""".split())
# Mots qui annoncent une classe / un modèle / une table / un fichier (« le modèle User », « la table de user »,
# « le fichier config ») : le mot qui les suit, même étranger à la langue de la question, ne nomme pas une méthode.
_CLASS_CUES_BEFORE: FrozenSet[str] = frozenset("""
modele modeles model models classe classes class table tables entite entites entity entities interface interfaces
trait traits enum enums type types objet objets object objects controleur controleurs controller controllers
service services repository repositories factory factories migration migrations schema fichier fichiers file files
document documents composant composants component components module modules
""".split())
_PREPOSITIONS: FrozenSet[str] = frozenset({"de", "d", "of"})  # « le code de confirm » : le mot de tête est « code »
# Mots courants d'une seule des deux langues des listes de :mod:`develop.search.query` (« confirm », « store » :
# anglais seulement ; « calculer », « garde » : français seulement) ; employés dans une question de l'autre langue, ils
# sont des noms de code. Les mots de grammaire / de question propres à une langue révèlent celle de la question.
_COMMON_EN: FrozenSet[str] = frozenset(fold(w) for w in getattr(_query, "_COMMON_WORDS_EN", ()))
_COMMON_FR: FrozenSet[str] = frozenset(fold(w) for w in getattr(_query, "_COMMON_WORDS_FR", ())) | frozenset(
    getattr(_query, "_EXPANSION_WORDS", {}))
_COMMON_EN_ONLY: FrozenSet[str] = _COMMON_EN - _COMMON_FR
_COMMON_FR_ONLY: FrozenSet[str] = _COMMON_FR - _COMMON_EN
_MARKERS_FR_ALL: FrozenSet[str] = frozenset(
    fold(w) for w in (*getattr(_query, "_STOPWORDS_FR", ()), *getattr(_query, "_QUESTION_WORDS_FR", ())))
_MARKERS_EN_ALL: FrozenSet[str] = frozenset(
    fold(w) for w in (*getattr(_query, "_STOPWORDS_EN", ()), *getattr(_query, "_QUESTION_WORDS_EN", ())))
_LANGUAGE_MARKERS_FR: FrozenSet[str] = _MARKERS_FR_ALL - _MARKERS_EN_ALL  # « le », « est », « explique »
_LANGUAGE_MARKERS_EN: FrozenSet[str] = _MARKERS_EN_ALL - _MARKERS_FR_ALL  # « the », « does », « explain »
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def question_language(text: str) -> Optional[str]:
    """« fr », « en » ou ``None`` (indécis) : la langue d'une question, d'après ses mots de grammaire et de question
    propres à une seule langue (« le », « est », « explique » / « the », « does », « explain »)."""
    tokens = _WORD_RE.findall(fold(text or ""))
    fr = sum(1 for t in tokens if t in _LANGUAGE_MARKERS_FR)
    en = sum(1 for t in tokens if t in _LANGUAGE_MARKERS_EN)
    if fr == en:
        return None
    return "fr" if fr > en else "en"


def word_names_symbol(word: str, text: str, intent: str = "factual") -> bool:
    """Vrai si le mot courant ``word`` est **employé comme un nom de code** dans la question ``text``, et désigne
    donc aussi une méthode / fonction homonyme (``OrderController::confirm``), pas seulement une classe :

    * écrit en appel : « confirm() » ;
    * annoncé par un mot de code juste avant ou après (:data:`_CODE_CUES_BEFORE` / :data:`_CODE_CUES_AFTER`) :
      « la méthode confirm », « l'action store », « où est défini confirm », « the confirm method » ;
    * complément direct d'un verbe d'explication (:data:`_EXPLAIN_VERBS`) : « explique confirm », « que fait
      store ? », « what does confirm do » ;
    * mot d'une seule langue dans une question de l'autre (:data:`_COMMON_EN_ONLY` / :data:`_COMMON_FR_ONLY`) :
      « le code de confirm », « comment marche store » — sauf derrière un déterminant ou un pronom sujet
      (:data:`_DETERMINERS` : « les logs », « tu put m'expliquer »), derrière un mot de classe, directement ou via
      « de » (:data:`_CLASS_CUES_BEFORE` : « le modèle User », « la table de user »), et dans une question de
      structure (``intent == "fields"`` : « quels champs a User ? » vise le modèle, pas les relations ``user()``).

    « explique moi la definition des champs » (« definition », mot des deux langues, derrière « la ») et « peux-tu
    me décrire le modèle User » (« user », derrière « modèle ») ne nomment pas de méthode."""
    w = fold(word or "").strip()
    folded = fold(text or "")
    if not w or not folded:
        return False
    if re.search(r"\b%s\s*\(" % re.escape(w), folded):
        return True
    tokens = _WORD_RE.findall(folded)
    lang = question_language(folded)
    foreign = intent != "fields" and ((lang == "fr" and w in _COMMON_EN_ONLY) or (lang == "en" and w in _COMMON_FR_ONLY))
    for i, tok in enumerate(tokens):
        if tok != w:
            continue
        j = i - 1
        while j >= 0 and tokens[j] in _POLITE_TOKENS:
            j -= 1
        prev = tokens[j] if j >= 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if prev in _CODE_CUES_BEFORE or nxt in _CODE_CUES_AFTER or prev in _EXPLAIN_VERBS:
            return True
        head = tokens[j - 1] if prev in _PREPOSITIONS and j >= 1 else prev  # « le code de confirm » → « code »
        if foreign and prev not in _DETERMINERS and head not in _DETERMINERS and head not in _CLASS_CUES_BEFORE:
            return True
    return False


# Canal « route » : au plus ROUTE_LIMIT routes par terme ; au-delà le terme est trop générique (« api »,
# « store ») et ne sert qu'à départager ; termes de moins de ROUTE_MIN_TERM_LENGTH caractères ignorés.
ROUTE_LIMIT = 20
ROUTE_MIN_TERM_LENGTH = 3
ROUTE_MAX_TERMS = 16
ROUTE_GRAPH_SEEDS = 3  # premières routes trouvées qui servent aussi de graines au canal graphe
# Poids d'une route : 1 + ROUTE_STRONG_STEP par preuve forte (terme égal à un segment du chemin, à l'action, au
# contrôleur ou à la méthode HTTP), au plus ROUTE_STRONG_MAX preuves → 1,5, comme IDENTIFIER_BONUS.
ROUTE_STRONG_STEP = 0.125
ROUTE_STRONG_MAX = 4
# Fichiers générés volumineux (canal lexical) : seuil de taille et motifs de chemin. Le dossier d'exécution
# Laravel n'est reconnu que par son nom en minuscules suivi d'un de ses sous-dossiers (``storage/app/``,
# ``storage/framework/``, ``storage/logs/``…) : un module source nommé ``storage`` (``develop/storage/db.py``,
# ``src/storage/s3.ts``, ``app/Storage/FileStore.php``) n'est pas un fichier généré.
GENERATED_FILE_MAX_BYTES = 300_000
GENERATED_PENALTY = 0.5  # l'apport RRF d'un morceau d'un tel fichier est divisé par deux
_LARAVEL_STORAGE_DIRS = "app|framework|logs|debugbar|clockwork|pail"
_GENERATED_PATH_RE = re.compile(
    r"(?:^|/)api-docs(?:[./]|$)|(?:^|/)(?:swagger|openapi)\.(?:json|ya?ml)$"
    r"|(?:^|/)(?-i:storage/(?:" + _LARAVEL_STORAGE_DIRS + r")/)|(?:^|/)public/build(?:/|$)"
    r"|\.lock$|-lock\.(?:json|ya?ml)$|\.min\.[A-Za-z0-9]+$", re.IGNORECASE)
EXPANSION_MAX_TERMS = 12  # termes d'expansion au plus dans la requête lexicale secondaire
EXPANSION_LIMIT = 20  # morceaux au plus pour cette requête
# Repli « faute de frappe » : distance d'édition tolérée selon la longueur (voir :func:`max_edits`).
FUZZY_MIN_LENGTH = 4  # en dessous, trop de faux positifs : aucune approximation
FUZZY_MAX_LENGTH = 64  # au-dessus, ce n'est pas un identifiant
FUZZY_MAX_IDENTS = 8  # identifiants cherchés à l'orthographe approchée par question
FUZZY_MAX_PER_IDENT = 5  # symboles approchés retenus par identifiant
FUZZY_POOL = 3000  # symboles relus au plus par identifiant (préfiltre SQL : longueur + début / fin du nom)
# Mots génériques du vocabulaire des questions (« table », « modèle », « entity »…) : ils qualifient un nom, ils
# n'en sont pas un. Jamais essayés comme noms nus de symboles / modèles (palier 3), sinon « la table customers »
# remonte la propriété Eloquent ``protected $table`` (``Order::table``) devant les vraies sources. Source de
# vérité : les déclencheurs de tables de :mod:`develop.search.query` (repli local si le nom privé change).
BARE_EXCLUDED_WORDS: FrozenSet[str] = frozenset(
    {fold(w) for w in getattr(_query, "_TABLE_TRIGGER_WORDS", ())}
    | {"table", "tables", "modele", "modeles", "model", "models", "entite", "entites", "entity", "entities",
       "collection", "collections"})
SNIPPET_MAX_CHARS = 300
SYMBOL_MAX_LINES = 80  # extrait maximal lu sur disque pour un symbole / modèle / fichier
MODEL_WINDOW_LINES = 40
ENDPOINT_WINDOW_LINES = 3
MAX_PER_SOURCE = 3  # diversité : morceaux maximum par fichier / document / note
MAX_GRAPH_SEEDS = 8
GRAPH_PER_SEED = 10  # candidats au plus par graine au premier tour du canal graphe (aucune graine n'accapare le canal)
# Fichiers atteints « via imports » depuis le fichier d'une graine (importateurs d'une classe, fichiers importés) :
# au plus GRAPH_IMPORTS_PER_SEED par graine hors question d'impact (who_calls / impact : les importateurs sont la
# surface d'impact), après les autres relations de la graine. Leur premier morceau (namespace + use) n'apprend
# rien : les douze contrôleurs importés par un fichier de routes ou les dix-neuf importateurs d'un modèle
# rempliraient le contexte d'en-têtes.
GRAPH_IMPORTS_PER_SEED = 3
_IMPORT_RELATION_KINDS: FrozenSet[str] = frozenset({"imports", "imports_raw"})
MAX_FILE_BYTES = 2_000_000
TEMPORAL_LIMIT = 20
RECENT_NOTES_LIMIT = 10

# Pondération par canal, ajustée selon l'intention de la question.
_CHANNEL_WEIGHTS: Dict[str, float] = {"identifier": IDENTIFIER_BONUS, "route": 1.0, "lexical": 1.0, "semantic": 1.0,
                                      "graph": 1.0, "temporal": 1.0}
_INTENT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "decision": {"temporal": 1.3},
    "who_calls": {"graph": 1.2, "route": 0.5},  # les appelants sont des symboles : les routes homonymes sont secondaires
    "impact": {"graph": 1.2},
    "fields": {"identifier": 1.7},  # (le canal route est muet pour cette intention, voir ``search``)
    "howto": {"route": 1.2},  # « comment créer / appeler… » vise souvent une route
}
_CHANNEL_PRIORITY = ("identifier", "route", "graph", "lexical", "semantic", "temporal")

_RELATION_LABELS: Dict[str, str] = {
    "imports": "imports", "imports_raw": "imports", "calls": "appels", "calls_endpoint": "appel d'API",
    "handles": "gestionnaire de route", "uses_model": "utilisation du modèle", "defines_table": "table définie",
    "documented_by": "documentation", "mentions": "mention", "defines": "définition", "maps_to": "correspondance",
    "links": "lien", "extends": "héritage", "implements": "implémentation", "uses": "utilisation",
    "matches": "correspondance de route",
}

_TAG_RE = re.compile(r"<[^<>\n]{1,200}>")
_FTS_MARK_RE = re.compile(r"\[([^\[\]\n]{1,80})\]")
_WS_RE = re.compile(r"\s+")
_QUALIFIED_SPLIT_RE = re.compile(r"^([A-Za-z_]\w*)(?:::|->|\.)([A-Za-z_]\w*)$")
_FILENAME_RE = re.compile(r"^[\w.-]+\.[A-Za-z0-9]{1,12}$")
_LIKE_ESCAPE = {"\\": "\\\\", "%": "\\%", "_": "\\_"}
# Gestionnaire de route « Contrôleur@action » (préfixe de namespace toléré ; « closure », « api »… ignorés).
_HANDLER_RE = re.compile(r"^(?:[\w\\.]*[\\.])?(?P<ctl>[A-Za-z_]\w*?)(?:Controller)?\s*(?:@|::|->|\.)\s*(?P<action>[A-Za-z_]\w*)$")
_CAMEL_FIRST_RE = re.compile(r"^([A-Z][a-z0-9]*)(.*)$")
# Action → préfixe canonique des classes de validation Laravel (Store/Update/Index<Modèle>Request).
_REQUEST_PREFIXES = {"store": "Store", "create": "Store", "update": "Update", "edit": "Update", "patch": "Update",
                     "index": "Index", "list": "Index", "all": "Index", "getall": "Index", "show": "Show", "get": "Show",
                     "destroy": "Delete", "delete": "Delete"}


_FRENCH_SUFFIXES = ("er", "ir", "re", "es", "s", "e")
_CAMEL_WORDS_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def handler_words(handler: Optional[str]) -> Set[str]:
    """Mots (repliés) d'un gestionnaire de route : ``ShipmentController@storeFromOrder`` →
    {shipment, controller, store, from, order}. Un terme ne « contient » un gestionnaire que mot
    entier : « orders » ne désigne pas ``OrderService::confirm``."""
    return {fold(w) for w in _CAMEL_WORDS_RE.findall(handler or "")}


def route_contains(e: EndpointRow, term: str, path_only: bool = False) -> bool:
    """Vrai si ``term`` (replié) est contenu dans le chemin (sous-chaîne), est un mot entier du
    gestionnaire, ou est la méthode HTTP. Avec ``path_only`` (radical déduit d'un mot-clé, « orders » →
    ``order``), seul le chemin compte : un radical ne désigne pas ``OrderService::confirm``."""
    if term in fold(e.normalized_path or "") or term in fold(e.path or ""):
        return True
    if path_only:
        return False
    return term in handler_words(e.handler) or term == fold(e.method or "")


# Partie dynamique d'un chemin d'appel client : variable, interpolation ou expression ($url, ${id}, {$this->baseUrl}).
_DYNAMIC_PART_RE = re.compile(r"\$\{[^{}]*\}|\$[A-Za-z_]\w*|\{[^{}]*\}")


def is_dynamic_client_call(e: EndpointRow) -> bool:
    """Vrai pour un appel client dont le chemin n'est fait que de variables ou d'expressions (« POST $url »,
    « {$this->baseUrl}{$endpoint} », « {self::PARTNER_URL} », tous normalisés « / ») : il ne désigne aucune
    route et ne peut en être une. L'extracteur PHP ne les indexe plus ; un index plus ancien en garde."""
    if e.side != "client":
        return False
    s, prev = e.path or "", None
    while s != prev:
        prev, s = s, _DYNAMIC_PART_RE.sub("", s)
    return not s.strip()


def inherited_words(analysis: QueryAnalysis) -> Set[str]:
    """Mots (repliés) hérités du tour précédent d'une question de suivi qui ne sont **pas** des identifiants :
    :attr:`QueryAnalysis.previous_terms` moins :attr:`QueryAnalysis.identifiers`, soit les mots d'un titre de
    document cité et les mots-clés de la question précédente (« formulaires », « commande »). Ils portent le sujet
    conservé sans le nommer : le moteur ne les essaie pas comme noms nus (canal « identifier ») et ne leur fait
    pas désigner de route (canal « route », où ils ne font que départager). Un identifiant hérité (« Customer »,
    « Shipment », le radical du fichier cité) reste un nom à part entière."""
    return {fold(p) for p in analysis.previous_terms} - {fold(i) for i in analysis.identifiers}


def _french_stem(word: str) -> str:
    """Radical d'un mot replié sans sa désinence (« confirmer » → ``confirm``, « commandes » →
    ``commande``), s'il garde au moins quatre caractères ; ``""`` sinon."""
    for suffix in _FRENCH_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return ""


def is_generated_file(path: Optional[str], size: Optional[int]) -> bool:
    """Fichier généré ou volumineux qui monopolise le canal lexical sans rien expliquer : plus de
    :data:`GENERATED_FILE_MAX_BYTES` (dumps SQL, exports…) ou chemin typique (``api-docs``,
    ``swagger.json`` / ``openapi.json``, ``*.lock``, ``*.min.*``, ``public/build``, dossier d'exécution
    Laravel ``storage/app/`` / ``storage/framework/`` / ``storage/logs/``…). Un dossier source nommé
    ``storage`` (``develop/storage/db.py``, ``src/storage/s3.ts``, ``app/Storage/FileStore.php``) n'en est pas un."""
    if size is not None and int(size) > GENERATED_FILE_MAX_BYTES:
        return True
    p = (path or "").replace("\\", "/")
    return bool(p) and _GENERATED_PATH_RE.search(p) is not None


def request_class_names(handler: Optional[str]) -> List[str]:
    """Noms possibles, par priorité, de la classe de validation d'un gestionnaire « Contrôleur@action » :
    ``<Action><Base>Request``, ``<Premier mot><Base><Reste>Request`` (``storeFromOrder`` →
    ``StoreShipmentFromOrderRequest``), ``<Base><Action>Request``, le préfixe canonique
    (``Store|Update|Index|Show|Delete<Base>Request``) et ``<Base>Request``. Vide sans gestionnaire exploitable."""
    m = _HANDLER_RE.match((handler or "").strip())
    if not m:
        return []
    base, action = m.group("ctl"), m.group("action")
    if not base or not action:
        return []
    act = action[0].upper() + action[1:]
    names = [act + base + "Request"]
    cm = _CAMEL_FIRST_RE.match(act)
    if cm and cm.group(2):
        names.append(cm.group(1) + base + cm.group(2) + "Request")
    names.append(base + act + "Request")
    prefix = _REQUEST_PREFIXES.get(action.lower())
    if prefix:
        names += [prefix + base + "Request", base + prefix + "Request"]
    names.append(base + "Request")
    out: List[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def _normalize_route(path: str) -> str:
    """Normalise une route pour la comparer aux endpoints indexés (repli local si code_analysis manque)."""
    if _normalize_route_path is not None:
        try:
            return _normalize_route_path(path)
        except Exception:  # noqa: BLE001
            pass
    p = (path or "").strip().strip("'\"`")
    p = re.sub(r"^[a-z]+://[^/]*", "", p, flags=re.IGNORECASE)
    p = p.split("?", 1)[0].split("#", 1)[0]
    p = re.sub(r"\$\{[^}]*\}|\{[^}]*\}|:[A-Za-z_]\w*|<[^>]*>|\[[^\]]*\]|\$[A-Za-z_]\w*", ":param", p)
    segs = [(":param" if re.fullmatch(r"\d+", s) else s.lower()) for s in p.split("/") if s]
    return "/" + "/".join(segs)


def _like_escape(value: str) -> str:
    return "".join(_LIKE_ESCAPE.get(ch, ch) for ch in value)


_FUZZY_STRIP_RE = re.compile(r"[^a-z0-9]")


def fuzzy_key(name: str) -> str:
    """Forme comparée par le repli « faute de frappe » : minuscules sans accents, lettres et chiffres
    seulement (``confirm_order`` et ``confirmOrder`` ont la même clé)."""
    return _FUZZY_STRIP_RE.sub("", fold(name or ""))


def max_edits(length: int) -> int:
    """Fautes tolérées pour un identifiant de ``length`` caractères : aucune sous
    :data:`FUZZY_MIN_LENGTH`, une jusqu'à 9 caractères, deux jusqu'à 14, trois au-delà."""
    if length < FUZZY_MIN_LENGTH:
        return 0
    if length <= 9:
        return 1
    if length <= 14:
        return 2
    return 3


def edit_distance(a: str, b: str, cap: Optional[int] = None) -> int:
    """Distance de Damerau-Levenshtein restreinte : nombre minimal d'insertions, suppressions,
    substitutions et transpositions de deux caractères voisins (« confrim » → « confirm » vaut 1).

    Avec ``cap``, le calcul s'arrête dès que la distance dépasse ``cap`` et renvoie ``cap + 1``
    (chaque ligne de la matrice minore le résultat final).
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if cap is not None and abs(la - lb) > cap:
        return cap + 1
    if not la or not lb:
        return la or lb
    prev2: Optional[List[int]] = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            d = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
            if prev2 is not None and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                d = min(d, prev2[j - 2] + 1)
            cur[j] = d
        if cap is not None and min(cur) > cap:
            return cap + 1
        prev2, prev = prev, cur
    return prev[lb]


def make_snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    """Extrait court (≤ ``max_chars``), sans balises ni retours à la ligne."""
    t = _TAG_RE.sub(" ", text or "")
    t = _WS_RE.sub(" ", t).strip()
    if len(t) > max_chars:
        t = t[:max_chars - 1].rstrip() + "…"
    return t


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def lexical_terms_found(snippet: str, query_terms: Sequence[str]) -> List[str]:
    """Mots surlignés par FTS5 dans ``snippet`` qui correspondent bien à un terme cherché.

    FTS5 entoure chaque jeton trouvé de ``[`` et ``]``, mais le texte indexé contient aussi ses propres
    crochets (tableaux PHP ``[Ctl::class, 'index']``, tags d'une note ``[décision, commandes]``, index
    ``$items[0]``…) qu'on ne doit pas prendre pour des marques. Une marque n'est donc retenue que si elle
    correspond à un terme de ``query_terms`` : égalité après repli de la casse et des accents (comme le
    tokenizer ``unicode61``), ou préfixe du **dernier** terme, le seul que :func:`build_fts_query` cherche
    avec ``*``. Renvoie les mots tels qu'ils apparaissent dans le texte (en minuscules), sans doublon,
    dans l'ordre de l'extrait : « commandé » cherché → « commande », « commandes » trouvés.
    """
    terms = [t for t in (fold(q) for q in query_terms if q) if t]
    if not terms:
        return []
    exact = set(terms)
    last = terms[-1]
    found: List[str] = []
    for m in _FTS_MARK_RE.findall(snippet or ""):
        word = m.strip().lower()
        folded = fold(word)
        if not folded or word in found:
            continue
        if folded in exact or folded.startswith(last):
            found.append(word)
    return found


# --------------------------------------------------------------------------- résultats

@dataclass
class SearchHit:
    """Un passage ou un élément du projet trouvé par la recherche (format de ``docs/API.md``).

    ``text`` est le texte complet du morceau (pour l'assembleur de contexte), ``snippet`` un extrait
    de 300 caractères au plus, ``channels`` les canaux qui l'ont remonté et ``reason`` l'explication
    en français. ``extra`` porte les identifiants fins (``symbol_id``, ``endpoint_id``, ``model_id``,
    ``document_id``, ``note_id``, ``kind``, ``signature``…).
    """

    chunk_id: Optional[int] = None
    source_type: str = "file"  # file | document | note | symbol | endpoint | model
    source_id: Optional[int] = None
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

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["channels"] = list(self.channels)
        d["extra"] = dict(self.extra)
        return d


@dataclass
class SearchResult:
    """Résultat complet d'une recherche : analyse de la question, résultats fusionnés et diagnostics.

    ``channel_counts`` compte les candidats renvoyés par chaque canal avant fusion (un canal absent
    de la recherche vaut 0) ; ``semantic_available`` dit si le canal sémantique a réellement pu
    tourner (vecteur de question + embeddings en base).
    """

    query: QueryAnalysis
    hits: List[SearchHit] = field(default_factory=list)
    channel_counts: Dict[str, int] = field(default_factory=dict)
    semantic_available: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "hits": [h.to_dict() for h in self.hits],
            "channel_counts": dict(self.channel_counts),
            "semantic_available": self.semantic_available,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------- candidats internes

@dataclass
class _Candidate:
    """Résultat brut d'un canal, avant fusion : une clé de déduplication, une raison et de quoi
    construire le :class:`SearchHit` (``kind`` + ligne de base ou id de morceau)."""

    key: Tuple[str, int]
    reason: str
    kind: str  # chunk | symbol | endpoint | model | file | note | document
    chunk_id: Optional[int] = None
    row: Any = None
    weight: float = 1.0  # multiplicateur de l'apport RRF (paliers du canal identifier, hérité par le graphe)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Fused:
    cand: _Candidate
    score: float = 0.0
    channels: List[str] = field(default_factory=list)
    reasons: Dict[str, str] = field(default_factory=dict)
    contrib: Dict[str, float] = field(default_factory=dict)  # apport RRF de chaque canal
    extra: Dict[str, Any] = field(default_factory=dict)


class _Loader:
    """Chargement en lot (fichiers, dépôts, documents, notes, textes sur disque) le temps d'une recherche."""

    def __init__(self, db: Database):
        self.db = db
        self._repos: Optional[Dict[str, RepoRow]] = None
        self._files: Dict[int, Optional[FileRow]] = {}
        self._docs: Dict[int, Optional[DocumentRow]] = {}
        self._notes: Dict[int, Optional[NoteRow]] = {}
        self._lines: Dict[int, Optional[List[str]]] = {}
        self._request_classes: Dict[str, Optional[Tuple[int, str, Optional[int]]]] = {}

    def request_class(self, handler: Optional[str]) -> Optional[Tuple[int, str, Optional[int]]]:
        """``(id, nom, file_id)`` de la classe de validation (FormRequest) du gestionnaire, parmi
        :func:`request_class_names`, ou ``None``."""
        key = (handler or "").strip()
        if not key:
            return None
        if key in self._request_classes:
            return self._request_classes[key]
        found: Optional[Tuple[int, str, Optional[int]]] = None
        names = request_class_names(key)
        if names:
            ph = ",".join("?" * len(names))
            try:
                rows = self.db.query("SELECT id, name, file_id FROM symbols WHERE name IN (%s) "
                                     "AND lower(coalesce(kind, '')) = 'class' ORDER BY id" % ph, names)
            except Exception:  # noqa: BLE001
                rows = []
            by_name = {str(r["name"]): (int(r["id"]), str(r["name"]), _as_int(r["file_id"])) for r in rows}
            for n in names:
                if n in by_name:
                    found = by_name[n]
                    break
        self._request_classes[key] = found
        return found

    def repos(self) -> Dict[str, RepoRow]:
        if self._repos is None:
            try:
                self._repos = {r.id: r for r in self.db.list_repos()}
            except Exception:  # noqa: BLE001
                self._repos = {}
        return self._repos

    def preload_files(self, ids: Iterable[int]) -> None:
        wanted = sorted({int(i) for i in ids if i is not None} - set(self._files))
        for i in range(0, len(wanted), 400):
            batch = wanted[i:i + 400]
            try:
                rows = self.db.query("SELECT * FROM files WHERE id IN (%s)" % ",".join("?" * len(batch)), batch)
            except Exception:  # noqa: BLE001
                rows = []
            for r in rows:
                f = FileRow.from_row(r)
                self._files[int(f.id)] = f  # type: ignore[arg-type]
            for fid in batch:
                self._files.setdefault(fid, None)

    def file(self, file_id: Optional[int]) -> Optional[FileRow]:
        if file_id is None:
            return None
        if file_id not in self._files:
            try:
                self._files[file_id] = self.db.get_file_by_id(int(file_id))
            except Exception:  # noqa: BLE001
                self._files[file_id] = None
        return self._files[file_id]

    def document(self, doc_id: Optional[int]) -> Optional[DocumentRow]:
        if doc_id is None:
            return None
        if doc_id not in self._docs:
            try:
                self._docs[doc_id] = self.db.get_document(int(doc_id))
            except Exception:  # noqa: BLE001
                self._docs[doc_id] = None
        return self._docs[doc_id]

    def note(self, note_id: Optional[int]) -> Optional[NoteRow]:
        if note_id is None:
            return None
        if note_id not in self._notes:
            try:
                self._notes[note_id] = self.db.get_note(int(note_id))
            except Exception:  # noqa: BLE001
                self._notes[note_id] = None
        return self._notes[note_id]

    def lines(self, file_row: Optional[FileRow]) -> Optional[List[str]]:
        """Lignes d'un fichier lues sur disque (lecture seule, bornée, jamais hors du dépôt)."""
        if file_row is None or file_row.id is None:
            return None
        fid = int(file_row.id)
        if fid in self._lines:
            return self._lines[fid]
        self._lines[fid] = read_file_lines(self.repos().get(file_row.repo_id), file_row.path)
        return self._lines[fid]


def read_file_lines(repo: Optional[RepoRow], rel_path: str) -> Optional[List[str]]:
    """Lit un fichier d'un dépôt (``repo.path`` + chemin relatif) ; ``None`` si illisible ou hors dépôt."""
    if repo is None or not repo.path or not rel_path:
        return None
    try:
        root = Path(repo.path).resolve()
        full = (root / rel_path.replace("\\", "/").lstrip("/")).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        full.relative_to(root)
    except ValueError:
        return None  # tentative de sortie du dépôt
    try:
        if not full.is_file() or full.stat().st_size > MAX_FILE_BYTES:
            return None
        data = full.read_bytes()
    except OSError:
        return None
    return data.decode("utf-8", "replace").splitlines()


# --------------------------------------------------------------------------- moteur

class SearchEngine:
    """Recherche hybride sur la mémoire Develop d'un espace de travail.

    ``llm`` et ``embedding_model`` sont optionnels : sans eux, le canal sémantique est simplement
    ignoré (``semantic_available=False``). La matrice d'embeddings est chargée une fois et
    rafraîchie quand le contenu de la table ``embeddings`` change.
    """

    def __init__(self, db: Database, llm: Optional[LLMClient] = None, embedding_model: str = ""):
        self.db = db
        self.llm = llm
        self.embedding_model = (embedding_model or "").strip()
        self.warnings: List[str] = []  # derniers avertissements (embed_query…), bornés
        self._matrix_lock = threading.Lock()
        self._matrix_cache: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ API publique
    def search(self, query: str, limit: int = 20, source_types: Optional[Sequence[str]] = None,
               query_vector: Optional[np.ndarray] = None, channels: Optional[Sequence[str]] = None,
               date_from: Optional[str] = None, date_to: Optional[str] = None) -> SearchResult:
        """Recherche synchrone multi-canaux (voir le module). Ne lève jamais sur une question « sale »."""
        text = query if isinstance(query, str) else str(query or "")
        try:
            analysis = analyze_query(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("analyse de la question impossible: %s", exc)
            analysis = QueryAnalysis(text=text)
        if date_from:
            analysis.date_from = str(date_from)[:10]
        if date_to:
            analysis.date_to = str(date_to)[:10]
        result = SearchResult(query=analysis, channel_counts={c: 0 for c in CHANNELS})
        try:
            limit_n = int(limit)
        except (TypeError, ValueError):
            limit_n = 20
        if limit_n <= 0 or not text.strip():
            return result

        allowed = self._allowed_types(source_types)
        chunk_types = [t for t in SOURCE_TYPES if t in allowed]
        wanted = self._wanted_channels(channels)
        weights = dict(_CHANNEL_WEIGHTS)
        for ch, w in _INTENT_WEIGHTS.get(analysis.intent, {}).items():
            weights[ch] = w
        loader = _Loader(self.db)

        per_channel: Dict[str, List[_Candidate]] = {}
        ident_cands: List[_Candidate] = []
        route_cands: List[_Candidate] = []
        if "identifier" in wanted:
            ident_cands = self._run(result, "identifier", lambda: self._channel_identifier(analysis, allowed))
            per_channel["identifier"] = ident_cands
        if "route" in wanted and "endpoint" in allowed and analysis.intent != "fields":
            # une question de structure (« quels champs a la table orders ») ne vise pas une route : canal muet
            route_cands = self._run(result, "route", lambda: self._channel_route(analysis))
            per_channel["route"] = route_cands
        if "lexical" in wanted and chunk_types:
            per_channel["lexical"] = self._run(result, "lexical",
                                               lambda: self._channel_lexical(analysis, chunk_types, loader))
        if "semantic" in wanted and chunk_types:
            per_channel["semantic"] = self._run(result, "semantic",
                                                lambda: self._channel_semantic(query_vector, chunk_types, result))
        if "graph" in wanted:
            # Les premières routes sont des graines à part entière, avant le reste des identifiants : le
            # gestionnaire d'une route trouvée (« handles ») vaut autant qu'un dixième modèle homonyme.
            keep = max(0, MAX_GRAPH_SEEDS - ROUTE_GRAPH_SEEDS)
            seeds = ident_cands[:keep] + route_cands[:ROUTE_GRAPH_SEEDS] + ident_cands[keep:]
            per_channel["graph"] = self._run(result, "graph", lambda: self._channel_graph(analysis, seeds, allowed))
        if "temporal" in wanted and "note" in allowed:
            per_channel["temporal"] = self._run(result, "temporal", lambda: self._channel_temporal(analysis))
        if not any(per_channel.values()) and "identifier" in wanted and "symbol" in allowed:
            # Repli ultime : rien nulle part (faute de frappe dans un mot nu, « orderservce confrim ») ; les
            # mots-clés sont essayés comme noms de symboles approchés, puis le graphe repart de là.
            ident_cands = self._run(result, "identifier", lambda: self._channel_fuzzy_keywords(analysis))
            if ident_cands:
                per_channel["identifier"] = ident_cands
                if "graph" in wanted:
                    per_channel["graph"] = self._run(result, "graph",
                                                     lambda: self._channel_graph(analysis, ident_cands, allowed))
        self._note_approximations(result, per_channel.get("identifier") or [])

        fused = self._fuse(per_channel, weights)
        hits = self._materialize(fused, loader)
        hits = [h for h in hits if h.source_type in allowed]
        hits = self._dedupe_and_diversify(hits)
        if analysis.intent not in ("who_calls", "impact"):
            hits = self._server_routes_first(hits)
        result.hits = hits[:limit_n]
        return result

    async def embed_query(self, text: str) -> Optional[np.ndarray]:
        """Vecteur (float32, norme L2 = 1) de la question via le modèle d'embeddings ; ``None`` sans
        moteur/modèle ou en cas d'erreur (consignée dans :attr:`warnings`, jamais levée)."""
        if self.llm is None or not self.embedding_model:
            return None
        content = (text if isinstance(text, str) else str(text or "")).strip()
        if not content:
            return None
        try:
            vectors = await self.llm.embed([content[:8000]], self.embedding_model)
            if not vectors or not vectors[0]:
                raise ValueError("réponse vide")
            vec = np.nan_to_num(np.asarray(vectors[0], dtype=np.float32).ravel())
            norm = float(np.linalg.norm(vec))
            if vec.size == 0 or norm == 0.0:
                raise ValueError("vecteur nul")
            return (vec / norm).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            self._warn("Embedding de la question indisponible (%s) : recherche sans canal sémantique." % _short_error(exc))
            return None

    async def search_async(self, query: str, **kw: Any) -> SearchResult:
        """:meth:`embed_query` puis :meth:`search` ; les avertissements d'embedding sont joints au résultat."""
        before = len(self.warnings)
        vec = kw.pop("query_vector", None)
        if vec is None:
            vec = await self.embed_query(query)
        result = self.search(query, query_vector=vec, **kw)
        for w in self.warnings[before:]:
            if w not in result.warnings:
                result.warnings.append(w)
        return result

    def lookup_symbol(self, name: str, limit: int = 10) -> List[SearchHit]:
        """Symboles portant ce nom (exact, insensible à la casse, ``Classe::membre`` accepté) ; à défaut,
        les meilleures correspondances partielles (sous-chaîne) ; en dernier recours, les noms à
        l'orthographe approchée (« confrim » → ``confirm``)."""
        q = (name or "").strip().strip("`'\"")
        if not q:
            return []
        loader = _Loader(self.db)
        try:
            cands = self._symbol_candidates(q, exact_only=False, limit=limit)
            if not cands:
                cands = self._fuzzy_symbol_candidates(q, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lookup_symbol(%r) en échec: %s", name, exc)
            return []
        return self._simple_hits(cands, loader, "identifier")

    def lookup_endpoint(self, path: str, method: Optional[str] = None, limit: int = 10) -> List[SearchHit]:
        """Routes serveur et appels client correspondant à un chemin (normalisé) et une méthode optionnelle."""
        p = (path or "").strip().strip("`'\"")
        if not p:
            return []
        loader = _Loader(self.db)
        try:
            cands = self._endpoint_candidates(p, method, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lookup_endpoint(%r) en échec: %s", path, exc)
            return []
        return self._simple_hits(cands, loader, "identifier")

    def lookup_model(self, name: str, limit: int = 10) -> List[SearchHit]:
        """Modèles de données dont le nom ou la table correspond (exact, insensible à la casse)."""
        q = (name or "").strip().strip("`'\"")
        if not q:
            return []
        loader = _Loader(self.db)
        try:
            cands = self._model_candidates(q, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("lookup_model(%r) en échec: %s", name, exc)
            return []
        return self._simple_hits(cands, loader, "identifier")

    def file_snippet(self, file_id: int, start: int, end: int, max_lines: int = 120) -> Optional[Dict[str, Any]]:
        """``{path, start, end, text, total_lines}`` d'une fenêtre de fichier lue sur disque (lecture seule),
        bornée à ``max_lines`` ; ``None`` si le fichier est inconnu ou illisible."""
        try:
            file_row = self.db.get_file_by_id(int(file_id))
        except Exception:  # noqa: BLE001
            return None
        if file_row is None:
            return None
        loader = _Loader(self.db)
        lines = loader.lines(file_row)
        if lines is None:
            return None
        total = len(lines)
        if total == 0:
            return {"path": file_row.path, "start": 0, "end": 0, "text": "", "total_lines": 0}
        s = max(1, _as_int(start) or 1)
        e = _as_int(end)
        cap = max(1, int(max_lines or 120))
        if e is None or e < s:
            e = s + cap - 1
        e = min(e, s + cap - 1, total)
        s = min(s, total)
        return {"path": file_row.path, "start": s, "end": e, "text": "\n".join(lines[s - 1:e]), "total_lines": total}

    # ------------------------------------------------------------------ préparation
    @staticmethod
    def _allowed_types(source_types: Optional[Sequence[str]]) -> Set[str]:
        if not source_types:
            return set(HIT_SOURCE_TYPES)
        if isinstance(source_types, str):
            source_types = [t.strip() for t in source_types.split(",")]
        allowed: Set[str] = set()
        for t in source_types:
            t = str(t or "").strip().lower()
            if t == "file":
                allowed.update(("file", "symbol", "endpoint", "model"))
            elif t in HIT_SOURCE_TYPES:
                allowed.add(t)
        return allowed or set(HIT_SOURCE_TYPES)

    @staticmethod
    def _wanted_channels(channels: Optional[Sequence[str]]) -> Set[str]:
        if not channels:
            return set(CHANNELS)
        if isinstance(channels, str):
            channels = [c.strip() for c in channels.split(",")]
        wanted = {str(c).strip().lower() for c in channels if str(c).strip().lower() in CHANNELS}
        return wanted or set(CHANNELS)

    def _run(self, result: SearchResult, channel: str, fn) -> List[_Candidate]:
        """Exécute un canal en isolant ses erreurs (consignées, jamais propagées)."""
        try:
            cands = fn()
        except Exception as exc:  # noqa: BLE001
            logger.warning("canal %s en échec: %s", channel, exc, exc_info=True)
            result.warnings.append("Canal « %s » indisponible : %s" % (channel, _short_error(exc)))
            cands = []
        seen: Set[Tuple[str, int]] = set()
        unique: List[_Candidate] = []
        for c in cands:
            if c.key in seen:
                continue
            seen.add(c.key)
            unique.append(c)
            if len(unique) >= CHANNEL_LIMIT:
                break
        result.channel_counts[channel] = len(unique)
        return unique

    def _warn(self, message: str) -> None:
        logger.info(message)
        self.warnings.append(message)
        if len(self.warnings) > 20:
            del self.warnings[:-20]

    # ------------------------------------------------------------------ canal identifiants
    def _symbol_candidates(self, ident: str, exact_only: bool = True, limit: int = 30) -> List[_Candidate]:
        out: List[_Candidate] = []
        m = _QUALIFIED_SPLIT_RE.match(ident)
        if m:
            left, right = m.group(1), m.group(2)
            for s in self.db.search_symbols(right, limit=limit):
                if s.id is None or fold(s.name) != fold(right):
                    continue
                if s.parent and fold(s.parent) == fold(left):
                    out.append(_Candidate(("symbol", s.id), "membre exact « %s »" % ident, "symbol", row=s))
            if out or exact_only:
                return out
            ident = right
        exact: List[_Candidate] = []
        partial: List[_Candidate] = []
        for s in self.db.search_symbols(ident, limit=limit):
            if s.id is None:
                continue
            if fold(s.name) == fold(ident):
                exact.append(_Candidate(("symbol", s.id), "nom de symbole exact « %s »" % ident, "symbol", row=s))
            elif not exact_only:
                partial.append(_Candidate(("symbol", s.id), "nom de symbole proche de « %s »" % ident, "symbol", row=s))
        return out + exact + (partial if not exact else [])

    def _fuzzy_symbol_candidates(self, ident: str, limit: int = FUZZY_MAX_PER_IDENT) -> List[_Candidate]:
        """Symboles dont le nom est à l'orthographe approchée de ``ident`` (repli « faute de frappe »).

        La comparaison porte sur :func:`fuzzy_key` (casse, accents, ``_`` et ``-`` ignorés) avec la
        tolérance de :func:`max_edits` ; pour ``Classe::membre``, le nom **et** le parent doivent être
        proches. Les candidats sont triés par distance totale puis écart de longueur, bornés à ``limit``,
        et portent ``extra={"approximate": True, "approximate_for": ident, "edit_distance": d}``.
        """
        m = _QUALIFIED_SPLIT_RE.match(ident or "")
        name_q, parent_q = (m.group(2), m.group(1)) if m else (ident or "", None)
        key = fuzzy_key(name_q)
        n = len(key)
        if n < FUZZY_MIN_LENGTH or n > FUZZY_MAX_LENGTH:
            return []
        tol = max_edits(n)
        parent_key = fuzzy_key(parent_q) if parent_q is not None else ""
        if parent_q is not None and (not parent_key or len(parent_key) > FUZZY_MAX_LENGTH):
            return []
        parent_tol = max_edits(len(parent_key))
        scored: List[Tuple[int, int, str, int, SymbolRow]] = []
        for s in self._fuzzy_pool(key, tol, with_parent=parent_q is not None):
            if s.id is None:
                continue
            d = edit_distance(key, fuzzy_key(s.name), cap=tol)
            if d > tol:
                continue
            if parent_q is not None:
                dp = edit_distance(parent_key, fuzzy_key(s.parent or ""), cap=parent_tol)
                if dp > parent_tol:
                    continue
                d += dp
            scored.append((d, abs(len(fuzzy_key(s.name)) - n), fold(s.name), int(s.id), s))
        scored.sort(key=lambda t: t[:4])
        out: List[_Candidate] = []
        for d, _, _, sid, s in scored[:max(0, int(limit))]:
            label = ("%s::%s" % (s.parent, s.name)) if (parent_q is not None and s.parent) else s.name
            out.append(_Candidate(("symbol", sid), "nom de symbole approché « %s » (« %s » cherché)" % (label, ident),
                                  "symbol", row=s,
                                  extra={"approximate": True, "approximate_for": ident, "edit_distance": d}))
        return out

    def _fuzzy_pool(self, key: str, tol: int, with_parent: bool) -> List[SymbolRow]:
        """Préfiltre SQL borné du repli approché : noms de longueur compatible commençant ou finissant
        comme ``key`` (deux caractères), les plus proches en longueur d'abord ; ``parent`` renseigné si
        ``with_parent``."""
        n = len(key)
        sql = ("SELECT * FROM symbols WHERE length(name) BETWEEN ? AND ? "
               "AND (name LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')")
        params: List[Any] = [max(1, n - tol), n + tol + 8, _like_escape(key[:2]) + "%", "%" + _like_escape(key[-2:])]
        if with_parent:
            sql += " AND parent IS NOT NULL AND parent <> ''"
        sql += " ORDER BY abs(length(name) - ?), id LIMIT ?"
        params += [n, FUZZY_POOL]
        return [SymbolRow.from_row(r) for r in self.db.query(sql, params)]

    def _endpoint_candidates(self, route: str, method: Optional[str] = None, limit: int = 20) -> List[_Candidate]:
        raw = route.strip()
        m = re.match(r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|ANY)\s+(\S+)$", raw, re.IGNORECASE)
        if m:
            method = method or m.group(1)
            raw = m.group(2)
        norm = _normalize_route(raw)
        if not norm or norm == "/":
            return []
        label = ("%s %s" % (method.upper(), raw)) if method else raw
        out: List[_Candidate] = []
        for e in self.db.find_endpoints(normalized_path=norm, method=method or None):
            if e.id is not None:
                out.append(_Candidate(("endpoint", e.id), "route exacte « %s »" % label, "endpoint", row=e))
        if out:
            return out[:limit]
        # Repli : préfixe de base absent d'un côté (/api/orders vs /orders), dans les deux sens.
        sql = ("SELECT * FROM endpoints WHERE normalized_path <> '/' AND (normalized_path LIKE ? ESCAPE '\\' "
               "OR ? LIKE '%' || normalized_path)")
        params: List[Any] = ["%" + _like_escape(norm), norm]
        if method:
            sql += " AND upper(method) = ?"
            params.append(method.upper())
        sql += " ORDER BY side, length(normalized_path) DESC, id LIMIT ?"
        params.append(int(limit))
        for r in self.db.query(sql, params):
            e = EndpointRow.from_row(r)
            if e.id is not None:
                out.append(_Candidate(("endpoint", e.id), "route correspondante « %s »" % label, "endpoint", row=e))
        return out

    def _model_candidates(self, name: str, limit: int = 20) -> List[_Candidate]:
        out: List[_Candidate] = []
        for m in self.db.find_models(name=name):
            if m.id is not None:
                out.append(_Candidate(("model", m.id), "modèle exact « %s »" % name, "model", row=m))
        for m in self.db.find_models(table_name=name):
            if m.id is not None:
                out.append(_Candidate(("model", m.id), "table exacte « %s »" % name, "model", row=m))
        return out[:limit]

    def _class_candidates(self, name: str, limit: int = 20) -> List[_Candidate]:
        """Symboles de nom exact ``name`` dont le genre est une classe / interface / modèle
        (:data:`CLASS_KINDS`) : ce qu'un mot courant (« user », « document ») a le droit de désigner."""
        kinds = sorted(CLASS_KINDS)
        ph = ",".join("?" * len(kinds))
        rows = self.db.query("SELECT * FROM symbols WHERE lower(name) = ? AND lower(coalesce(kind, '')) IN (%s) "
                             "ORDER BY length(name), id LIMIT ?" % ph, [fold(name), *kinds, int(limit)])
        return [_Candidate(("symbol", s.id), "nom de classe exact « %s »" % name, "symbol", row=s)
                for s in (SymbolRow.from_row(r) for r in rows) if s.id is not None]

    def _member_candidates(self, name: str, parents: Set[str], limit: int = 20) -> List[_Candidate]:
        """Membres de nom exact ``name`` d'une des classes ``parents`` (repliées) citées dans la même
        question : « OrderService confirm » désigne bien ``OrderService::confirm``."""
        if not parents:
            return []
        owners = sorted(parents)
        ph = ",".join("?" * len(owners))
        rows = self.db.query("SELECT * FROM symbols WHERE lower(name) = ? AND lower(coalesce(parent, '')) IN (%s) "
                             "ORDER BY id LIMIT ?" % ph, [fold(name), *owners, int(limit)])
        out: List[_Candidate] = []
        for s in (SymbolRow.from_row(r) for r in rows):
            if s.id is not None:
                out.append(_Candidate(("symbol", s.id), "membre « %s » de la classe citée %s" % (name, s.parent),
                                      "symbol", row=s))
        return out

    def _bare_symbol_candidates(self, word: str, analysis: QueryAnalysis, parents: Set[str],
                                limit: int = 30) -> List[_Candidate]:
        """Symboles de nom exact ``word``, avec la règle des mots courants : un mot sans forme de code ni
        accents graves qui est un mot de la langue (:func:`is_common_word` : « definition », « put »,
        « user ») ne désigne qu'une classe / un modèle (:meth:`_class_candidates`) ou un membre d'une
        classe citée dans la question (:meth:`_member_candidates`) ; ses méthodes homonymes
        (``UserFactory::definition``, ``Cache::put``) ne sont jamais remontées (poids nul). Exceptions :
        une question d'appelants ou d'impact (« qui appelle confirm ? », « si je modifie confirm »)
        **nomme** un symbole, et une question qui **emploie le mot comme un nom de code**
        (:func:`word_names_symbol` : « que fait la méthode confirm ? », « explique confirm », « où est
        défini store ? ») demande bien la méthode — les actions des contrôleurs sont presque toutes des
        mots courants ; le mot est alors pris tel quel. Dans les deux cas, si une classe citée possède le
        membre (« la méthode store de ShipmentController »), c'est lui que la question nomme : les autres
        homonymes (trente ``store``) ne sont pas remontés."""
        names_code = analysis.intent in ("who_calls", "impact")
        if (names_code or has_code_shape(word) or ("`%s`" % word) in analysis.text or not is_common_word(word)
                or word_names_symbol(word, analysis.text, analysis.intent)):
            return self._member_candidates(word, parents, limit=limit) or self._symbol_candidates(word, limit=limit)
        return self._class_candidates(word, limit=limit) + self._member_candidates(word, parents, limit=limit)

    def _file_candidates(self, suffix: str, reason: str, limit: int = 10) -> List[_Candidate]:
        rel = suffix.replace("\\", "/").strip().lstrip("./").strip("/")
        if not rel:
            return []
        esc = _like_escape(rel)
        rows = self.db.query(
            "SELECT * FROM files WHERE path = ? OR lower(path) LIKE lower(?) ESCAPE '\\' "
            "ORDER BY length(path), repo_id, path LIMIT ?", (rel, "%/" + esc, int(limit)))
        return [self._file_candidate(FileRow.from_row(r), reason) for r in rows]

    def _file_stem_candidates(self, stem: str, limit: int = 5) -> List[_Candidate]:
        esc = _like_escape(stem)
        rows = self.db.query(
            "SELECT * FROM files WHERE lower(path) LIKE lower(?) ESCAPE '\\' OR lower(path) LIKE lower(?) ESCAPE '\\' "
            "ORDER BY length(path), repo_id, path LIMIT ?", ("%/" + esc + ".%", esc + ".%", int(limit)))
        return [self._file_candidate(FileRow.from_row(r), "nom de fichier « %s »" % stem) for r in rows]

    def _file_candidate(self, f: FileRow, reason: str) -> _Candidate:
        """Un fichier devient son premier morceau (fusionnable avec le canal lexical) ou un extrait disque ;
        ``row`` garde le fichier pour que le canal graphe puisse en partir."""
        chunks = self.db.chunks_for("file", int(f.id)) if f.id is not None else []
        if chunks and chunks[0].id is not None:
            return _Candidate(("chunk", int(chunks[0].id)), reason, "chunk", chunk_id=int(chunks[0].id), row=f)
        return _Candidate(("file", int(f.id)), reason, "file", row=f)  # type: ignore[arg-type]

    def _channel_identifier(self, analysis: QueryAnalysis, allowed: Set[str]) -> List[_Candidate]:
        """Correspondances exactes, par palier décroissant d'explicitation (voir :data:`W_EXPLICIT`…) ;
        l'ordre d'ajout fixe le rang RRF et l'ordre des graines du canal graphe."""
        out: List[_Candidate] = []
        qualified = [i for i in analysis.identifiers if _QUALIFIED_SPLIT_RE.match(i)]
        simple = [i for i in analysis.identifiers if not _QUALIFIED_SPLIT_RE.match(i) and not _FILENAME_RE.match(i)]
        filenames = [i for i in analysis.identifiers if _FILENAME_RE.match(i) and "/" not in i]
        # Mots-clés nus (« confirm », « orderservice ») essayés comme noms exacts, en nombre borné ; les mots
        # génériques (:data:`BARE_EXCLUDED_WORDS` : « table », « model »…) sont écartés, et les mots hérités du
        # tour précédent (:func:`inherited_words`) aussi : « commande » d'un titre cité n'est pas un nom que
        # l'utilisateur a tapé (il ramenait le modèle Commande et sa migration devant le document cité) ; un modèle
        # que le tour précédent a réellement cité est dans le marqueur par son libellé, un identifiant.
        tried = {fold(i) for i in analysis.identifiers} | {fold(t) for t in analysis.tables}
        inherited = inherited_words(analysis)
        bare: List[str] = []
        for k in analysis.keywords:
            fk = fold(k)
            if len(k) >= 3 and fk not in tried and fk not in BARE_EXCLUDED_WORDS and fk not in inherited and len(bare) < 8:
                tried.add(fk)
                bare.append(k)

        matched: Set[str] = set()  # identifiants (repliés) ayant trouvé au moins une correspondance exacte
        cited_classes: Set[str] = set()  # classes trouvées (repliées) : leurs membres homonymes de mots courants comptent

        def push(cands: List[_Candidate], weight: float, ident: Optional[str] = None) -> None:
            if cands and ident:
                matched.add(fold(ident))
            for c in cands:
                c.weight = weight
                out.append(c)
                if c.kind == "symbol" and isinstance(c.row, SymbolRow) and (c.row.kind or "").lower() in CLASS_KINDS:
                    cited_classes.add(fold(c.row.name))

        # --- palier 1 : mentions explicites (Classe::membre, routes, chemins)
        if "symbol" in allowed:
            for ident in qualified:
                push(self._symbol_candidates(ident), W_EXPLICIT, ident)
        if "endpoint" in allowed:
            for route in analysis.routes:
                push(self._endpoint_candidates(route), W_EXPLICIT)
        if "file" in allowed:
            for p in analysis.paths:
                push(self._file_candidates(p, "chemin de fichier « %s »" % p), W_EXPLICIT)
        # --- palier 2 : identifiants nommés, tables citées, noms de fichiers, mots collés
        seen_models: Set[str] = set()
        if "symbol" in allowed:
            for ident in simple:
                push(self._bare_symbol_candidates(ident, analysis, cited_classes), W_NAMED, ident)
        if "model" in allowed:
            for name in list(analysis.tables) + simple:
                if fold(name) not in seen_models:
                    seen_models.add(fold(name))
                    push(self._model_candidates(name), W_NAMED, name)
        if "file" in allowed:
            for name in filenames:
                push(self._file_candidates(name, "nom de fichier « %s »" % name), W_NAMED)
        glued = self._glued_names(analysis.keywords)  # user controller -> usercontroller == lower(name)
        if glued and "symbol" in allowed:
            ph = ",".join("?" * len(glued))
            rows = self.db.query(
                "SELECT * FROM symbols WHERE replace(replace(lower(name), '_', ''), '-', '') IN (%s) "
                "ORDER BY length(name), id LIMIT 20" % ph, glued)
            push([_Candidate(("symbol", s.id), "mots collés « %s »" % s.name, "symbol", row=s)
                  for s in (SymbolRow.from_row(r) for r in rows) if s.id is not None], W_NAMED)
        if glued and "model" in allowed:
            ph = ",".join("?" * len(glued))
            rows = self.db.query(
                "SELECT * FROM data_models WHERE replace(replace(lower(name), '_', ''), '-', '') IN (%s) "
                "OR replace(lower(coalesce(table_name, '')), '_', '') IN (%s) ORDER BY length(name), id LIMIT 20"
                % (ph, ph), glued + glued)
            push([_Candidate(("model", m.id), "mots collés « %s »" % m.name, "model", row=m)
                  for m in (ModelRow.from_row(r) for r in rows) if m.id is not None], W_NAMED)
        # --- palier 3 : mots nus, termes d'expansion (classes et modèles seulement) et radicaux de fichiers
        if "symbol" in allowed:
            for k in bare:
                push(self._bare_symbol_candidates(k, analysis, cited_classes), W_BARE)
        if "model" in allowed:
            for k in bare:
                if fold(k) not in seen_models:
                    seen_models.add(fold(k))
                    push(self._model_candidates(k), W_BARE)
        expansions = [t for t in analysis.expansions if len(t) >= 3 and fold(t) not in tried][:8]
        for t in expansions:
            tried.add(fold(t))
            if "symbol" in allowed:
                push(self._class_candidates(t), W_BARE)
            if "model" in allowed and fold(t) not in seen_models:
                seen_models.add(fold(t))
                # un terme déduit (« shipments ») : un modèle par fichier suffit, pas chaque Schema::table d'une migration
                seen_files: Set[Optional[int]] = set()
                cands = []
                for c in self._model_candidates(t):
                    fid = getattr(c.row, "file_id", None)
                    if fid not in seen_files:
                        seen_files.add(fid)
                        cands.append(c)
                push(cands, W_BARE)
        if "file" in allowed:
            for ident in simple:
                if len(ident) >= 3:
                    push(self._file_stem_candidates(ident), W_BARE, ident)
        # --- palier 4 : orthographe approchée (repli « faute de frappe ») des identifiants restés sans
        # correspondance : « confrimOrder » → confirmOrder, « OrderServce::confrim » → OrderService::confirm.
        # Un mot courant sans forme de code (« Garde », « Definition ») n'est pas une faute de frappe.
        if "symbol" in allowed:
            attempts = 0
            for ident in qualified + simple:
                if fold(ident) in matched or attempts >= FUZZY_MAX_IDENTS:
                    continue
                if not has_code_shape(ident) and is_common_word(ident) and analysis.intent not in ("who_calls", "impact"):
                    continue
                attempts += 1
                cands = self._fuzzy_symbol_candidates(ident)
                push(cands, W_FUZZY, ident)
                m = _QUALIFIED_SPLIT_RE.match(ident)
                if cands and m:  # membre approché trouvé : inutile de chercher la classe et le membre séparément
                    matched.update((fold(m.group(1)), fold(m.group(2))))
        return out

    def _channel_fuzzy_keywords(self, analysis: QueryAnalysis) -> List[_Candidate]:
        """Repli ultime, quand aucun canal n'a rien trouvé : les mots-clés nus de la question (hors
        identifiants déjà tentés et mots génériques) cherchés comme noms de symboles approchés."""
        out: List[_Candidate] = []
        tried = {fold(i) for i in analysis.identifiers}
        attempts = 0
        for k in analysis.keywords:
            fk = fold(k)
            if fk in tried or fk in BARE_EXCLUDED_WORDS or len(fuzzy_key(k)) < FUZZY_MIN_LENGTH:
                continue
            if is_common_word(k):
                continue  # un mot de la langue (« definition », « garde ») n'est pas une faute de frappe d'identifiant
            tried.add(fk)
            attempts += 1
            if attempts > FUZZY_MAX_IDENTS:
                break
            for c in self._fuzzy_symbol_candidates(k):
                c.weight = W_FUZZY
                out.append(c)
        return out

    @staticmethod
    def _note_approximations(result: SearchResult, cands: List[_Candidate]) -> None:
        """Signale dans ``result.warnings`` chaque identifiant servi à l'orthographe approchée, avec les
        noms retenus (« Aucune correspondance exacte pour « confrimOrder » ; noms proches : confirmOrder »)."""
        names: Dict[str, List[str]] = {}
        for c in cands:
            if not c.extra.get("approximate"):
                continue
            ident = str(c.extra.get("approximate_for") or "")
            s = c.row
            label = ("%s::%s" % (s.parent, s.name)) if isinstance(s, SymbolRow) and s.parent else getattr(s, "name", "")
            bucket = names.setdefault(ident, [])
            if label and label not in bucket:
                bucket.append(label)
        for ident, labels in names.items():
            result.warnings.append("Aucune correspondance exacte pour « %s » ; noms proches : %s."
                                   % (ident, ", ".join(labels[:FUZZY_MAX_PER_IDENT])))

    @staticmethod
    def _glued_names(keywords: List[str]) -> List[str]:
        """Candidats « mots collés » : tous les mots-clés joints (jusqu'à 4), puis chaque paire et chaque
        triplet de mots adjacents (``order line`` → ``orderline``, ``order line item`` → ``orderlineitem``)."""
        words = [re.sub(r"[^a-z0-9]", "", fold(k)) for k in keywords]
        words = [w for w in words if w]
        if len(words) < 2:
            return []
        glued: List[str] = []
        if len(words) <= 4:
            glued.append("".join(words))
        for a, b in zip(words, words[1:]):
            glued.append(a + b)
        for a, b, c in zip(words, words[1:], words[2:]):
            glued.append(a + b + c)
        out: List[str] = []
        for w in glued:
            if len(w) >= 4 and w not in out:
                out.append(w)
        return out

    # ------------------------------------------------------------------ canal lexical
    def _channel_lexical(self, analysis: QueryAnalysis, chunk_types: List[str], loader: _Loader) -> List[_Candidate]:
        """Mots-clés (ET, puis OU en repli — :meth:`Database.fts_search`), puis les termes d'expansion en
        OU à :data:`W_EXPANSION` ; dans chaque liste, les morceaux de fichiers générés volumineux sont
        pénalisés (:func:`is_generated_file`).

        Pour une question de suivi, la requête principale ne porte que les mots de la question elle-même : les
        termes hérités du tour précédent (:attr:`QueryAnalysis.previous_terms`, identifiants et libellés de
        sources compris) forment une requête secondaire en OU à :data:`W_PREVIOUS`, car exigés en ET ils
        filtraient : « explique-moi la définition des champs » après le document « Paramétrage des formulaires
        de commande » n'acceptait plus que le seul fichier contenant tous les mots du marqueur (« contenu »,
        « configuration », « commandes »…), et le document, premier quand la question est posée seule, disparaissait
        du canal. Un morceau trouvé par les deux requêtes (les mots de la question **et** le sujet précédent) garde
        son rang de la requête principale avec les deux poids additionnés (un canal ne compte un morceau qu'une
        fois, voir :meth:`_run`) : il passe devant les morceaux des seuls mots de la question. Sans mot propre à la
        question (« et alors ? », mots vides), les termes hérités restent la requête principale."""
        out: List[_Candidate] = []
        previous = [t for t in analysis.previous_terms if t]
        own = [k for k in analysis.keywords if k not in previous] if previous else list(analysis.keywords)
        q = " ".join(own) or analysis.lexical_query or analysis.text
        if q.strip():
            query_terms = fts_terms(q)  # exactement ce que ``Database.fts_search`` cherche
            hits = self.db.fts_search(q, limit=CHANNEL_LIMIT * 2, source_types=chunk_types)
            out.extend(self._lexical_candidates(hits, query_terms, loader, 1.0, "correspondance lexicale"))
        if previous and own:
            terms: List[str] = []
            for t in fts_terms(" ".join(previous)):  # un libellé à tirets donne ses mots, sans doublon
                if t not in terms:
                    terms.append(t)
            terms = terms[:EXPANSION_MAX_TERMS]
            if terms:
                hits = self._fts_or(terms, EXPANSION_LIMIT, chunk_types)
                cands = self._lexical_candidates(hits, terms, loader, W_PREVIOUS,
                                                 "correspondance lexicale avec le tour précédent")
                both = {c.key: c for c in cands}
                for c in out:
                    hit = both.pop(c.key, None)
                    if hit is not None:
                        c.weight += hit.weight
                        found = [str(t) for t in hit.extra.get("fts_terms") or []][:3]
                        c.reason += " ; avec le tour précédent" + (" sur %s" % ", ".join("« %s »" % t for t in found)
                                                                  if found else "")
                        c.extra["previous"] = True
                for c in cands:
                    if c.key in both:
                        c.extra["previous"] = True
                        out.append(c)
        expansions = [t for t in analysis.expansions if len(t) >= 2][:EXPANSION_MAX_TERMS]
        if expansions:
            hits = self._fts_or(expansions, EXPANSION_LIMIT, chunk_types)
            out.extend(self._lexical_candidates(hits, expansions, loader, W_EXPANSION,
                                                "correspondance lexicale élargie", expanded=True))
        return out

    def _fts_or(self, terms: List[str], limit: int, chunk_types: List[str]) -> List[FtsHit]:
        """Requête FTS5 en OU (sans préfixe) sur ``terms`` ; ``[]`` sur requête vide ou invalide."""
        match = build_fts_query(" ".join(terms), "OR", prefix_last=False)
        if not match or limit <= 0 or not chunk_types:
            return []
        sql = ("SELECT chunks_fts.rowid AS chunk_id, bm25(chunks_fts) AS score, "
               "snippet(chunks_fts, 0, '[', ']', '…', 12) AS snippet, c.source_type AS source_type, "
               "c.source_id AS source_id FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
               "WHERE chunks_fts MATCH ? AND c.source_type IN (%s) ORDER BY bm25(chunks_fts), chunks_fts.rowid LIMIT ?"
               % ",".join("?" * len(chunk_types)))
        try:
            rows = self.db.query(sql, [match, *chunk_types, int(limit)])
        except sqlite3.OperationalError:
            return []
        return [FtsHit(chunk_id=int(r["chunk_id"]), score=-float(r["score"]), snippet=r["snippet"] or "",
                       source_type=r["source_type"], source_id=r["source_id"]) for r in rows]

    def _lexical_candidates(self, hits: List[FtsHit], terms: Sequence[str], loader: _Loader, weight: float,
                            label: str, expanded: bool = False) -> List[_Candidate]:
        """Candidats d'une liste FTS, dans l'ordre BM25, avec la pénalité des fichiers générés volumineux :
        apport × :data:`GENERATED_PENALTY`, raison « fichier généré volumineux », et un seul morceau par
        tel fichier en tête de liste (les suivants sont renvoyés en queue, après les autres morceaux)."""
        loader.preload_files(int(h.source_id) for h in hits if h.source_type == "file" and h.source_id is not None)
        head: List[_Candidate] = []
        deferred: List[_Candidate] = []
        seen_generated: Set[int] = set()
        for h in hits:
            found = lexical_terms_found(h.snippet or "", terms)
            reason = ("%s sur %s" % (label, ", ".join("« %s »" % t for t in found[:3]))) if found else label
            extra: Dict[str, Any] = {"fts_score": round(float(h.score), 4), "fts_terms": found[:5]}
            if expanded:
                extra["expanded"] = True
            cand = _Candidate(("chunk", int(h.chunk_id)), reason, "chunk", chunk_id=int(h.chunk_id), weight=weight,
                              extra=extra)
            f = loader.file(int(h.source_id)) if h.source_type == "file" and h.source_id is not None else None
            if f is not None and f.id is not None and is_generated_file(f.path, f.size):
                cand.weight = weight * GENERATED_PENALTY
                cand.reason += " (fichier généré volumineux)"
                cand.extra["generated"] = True
                if int(f.id) in seen_generated:
                    deferred.append(cand)
                    continue
                seen_generated.add(int(f.id))
            head.append(cand)
        return head + deferred

    # ------------------------------------------------------------------ canal route
    def _route_terms(self, analysis: QueryAnalysis) -> Tuple[List[Tuple[str, bool]], List[Tuple[str, bool]]]:
        """``(termes, départage)`` : termes cherchés dans les routes, avec un drapeau « radical » : mots-clés et
        leur radical sans désinence française (« confirmer » → ``confirm``, « commandes » → ``commande`` ; un
        radical n'est cherché que dans le chemin), mots collés puis expansions (repliés, ≥ 3 caractères, sans
        doublon), hors termes déjà présents dans une route ou un chemin cité explicitement (le canal
        « identifier » les sert à l'exact, avec la méthode).

        Les mots **hérités du tour précédent** (:attr:`QueryAnalysis.previous_terms`) qui ne sont pas des
        identifiants — mots d'un titre de document cité, mots-clés de la question précédente (« formulaires »,
        « commande ») — ne désignent aucune route : avec leurs radicaux et mots collés, ils forment la seconde liste,
        qui ne sert qu'à départager les routes trouvées par la question elle-même. Sinon « explique-moi la
        définition des champs » après le document « Paramétrage des formulaires de commande » ramenait GET /commandes,
        POST /commandes… devant le document. Un identifiant hérité (« Customer », « Shipment ») reste un terme à part
        entière : c'est le sujet conservé."""
        explicit = [fold(r) for r in list(analysis.routes) + list(analysis.paths)]
        inherited = inherited_words(analysis)
        own = [k for k in analysis.keywords if fold(k) not in inherited]
        previous = [k for k in analysis.keywords if fold(k) in inherited]
        seen: Set[str] = set()

        def collect(keywords: List[str], expansions: List[str]) -> List[Tuple[str, bool]]:
            raw_terms: List[Tuple[str, bool]] = []
            for k in keywords:
                raw_terms.append((k, False))
                stem = _french_stem(fold(k))
                if stem:
                    raw_terms.append((stem, True))
            raw_terms += [(g, False) for g in self._glued_names(keywords)]
            raw_terms += [(x, False) for x in expansions]
            out: List[Tuple[str, bool]] = []
            for raw, derived in raw_terms:
                t = re.sub(r"[^a-z0-9_-]", "", fold(raw))
                if len(t) < ROUTE_MIN_TERM_LENGTH or t in seen or any(t in e for e in explicit):
                    continue
                seen.add(t)
                out.append((t, derived))
                if len(out) >= ROUTE_MAX_TERMS:
                    break
            return out

        terms = collect(own, list(analysis.expansions))
        return terms, collect(previous, [])

    def _channel_route(self, analysis: QueryAnalysis) -> List[_Candidate]:
        """Endpoints dont ``normalized_path``, ``path`` ou ``handler`` contient un terme (LIKE, casse
        ignorée, :data:`ROUTE_LIMIT` routes au plus par terme). Un terme présent dans plus de
        :data:`ROUTE_LIMIT` routes (« api », « store ») ne produit pas de candidat, pas plus qu'un mot hérité
        du tour précédent (voir :meth:`_route_terms`) : ils ne font que départager ceux trouvés par un terme
        précis. Un appel client dont le chemin n'est qu'une variable (« POST $url », voir
        :func:`is_dynamic_client_call`) est ignoré. Classement : preuves fortes (terme égal à un segment du
        chemin, à l'action, au contrôleur ou à la méthode), puis route serveur avant appel client (un appel
        consomme une route, il ne la définit pas), puis terme égal à un segment, nombre de termes contenus,
        chemin le plus court."""
        terms, inherited = self._route_terms(analysis)
        if not terms:
            return []
        rows: Dict[int, EndpointRow] = {}
        matched: Dict[int, List[str]] = {}
        qualifiers: List[Tuple[str, bool]] = list(inherited)
        for t, derived in terms:
            like = "%" + _like_escape(t) + "%"
            # préfiltre SQL (sous-chaîne), puis règle exacte de :func:`route_contains` (mot entier du gestionnaire)
            found = [e for e in (EndpointRow.from_row(r) for r in self.db.query(
                "SELECT * FROM endpoints WHERE lower(coalesce(normalized_path, '')) LIKE ? ESCAPE '\\' "
                "OR lower(coalesce(path, '')) LIKE ? ESCAPE '\\' OR lower(coalesce(handler, '')) LIKE ? ESCAPE '\\' "
                "ORDER BY side, length(coalesce(normalized_path, '')), id LIMIT ?",
                (like, like, like, 2 * ROUTE_LIMIT + 1)))
                     if e.id is not None and route_contains(e, t, derived) and not is_dynamic_client_call(e)]
            if len(found) > ROUTE_LIMIT:
                qualifiers.append((t, derived))
                continue
            for e in found[:ROUTE_LIMIT]:
                rows.setdefault(int(e.id), e)  # type: ignore[arg-type]
                matched.setdefault(int(e.id), []).append(t)  # type: ignore[arg-type]
        # Un mot d'action seul (« crée » → create/store/post/new, « lister » → index…) trouve les routes de création ou
        # de listage de TOUTES les entités (GET /persons/create, GET /documents/create…). Dès qu'un terme précis a
        # trouvé des routes, celles qui ne tiennent qu'à un mot d'action sont écartées ; sans terme précis, elles restent.
        if any(any(t not in ACTION_TERMS for t in found) for found in matched.values()):
            for eid in [eid for eid, found in matched.items() if all(t in ACTION_TERMS for t in found)]:
                rows.pop(eid, None)
                matched.pop(eid, None)
        if not rows:
            return []

        def parts(e: EndpointRow) -> Set[str]:
            """Preuves fortes possibles : segments du chemin, mots du gestionnaire, méthode HTTP (« crée » → POST)."""
            segs = {s for s in fold(e.normalized_path or "").split("/") if s}
            segs |= handler_words(e.handler)
            if e.method:
                segs.add(fold(e.method))
            return segs

        ranked: List[Tuple[int, int, int, int, int, int, EndpointRow, List[str], int]] = []
        for eid, e in rows.items():
            found = list(matched[eid]) + [q for q, derived in qualifiers if route_contains(e, q, derived)]
            # un radical et son mot entier (« shipment » / « shipments ») sont une seule preuve : on garde le plus long
            found = [t for t in found if not any(o != t and o.startswith(t) for o in found)]
            strong = len(parts(e) & set(found))
            # à preuves fortes égales, la route serveur avant l'appel client (le fetch() d'une vue qui vise
            # POST login n'est pas l'API d'authentification), puis un terme qui est un segment du chemin (l'identité
            # de la route) prime sur un mot du gestionnaire : « auth login » désigne /auth/login avant
            # PartnerAuthController@login
            in_path = len({s for s in fold(e.normalized_path or "").split("/") if s} & set(found))
            ranked.append((-strong, 0 if e.side == "server" else 1, -in_path, -len(found), len(e.normalized_path or ""),
                           eid, e, found, strong))
        ranked.sort(key=lambda t: t[:6])
        out: List[_Candidate] = []
        for _, _, _, _, _, eid, e, found, strong in ranked:
            reason = "route contenant %s" % ", ".join("« %s »" % t for t in found[:3])
            # jusqu'à quatre preuves fortes : une route désignée par son segment, son action et sa méthode pèse 1,5
            weight = 1.0 + ROUTE_STRONG_STEP * min(strong, ROUTE_STRONG_MAX)
            out.append(_Candidate(("endpoint", eid), reason, "endpoint", row=e, weight=weight,
                                  extra={"route_terms": found[:5], "route_strong": strong}))
        return out

    # ------------------------------------------------------------------ canal sémantique
    def _resolve_embedding_model(self) -> str:
        if self.embedding_model:
            return self.embedding_model
        try:
            models = self.db.embedding_models()
        except Exception:  # noqa: BLE001
            return ""
        return str(models[0]["model"]) if models else ""

    def _embedding_matrix(self, model: str) -> Tuple[np.ndarray, np.ndarray]:
        """``(ids, matrice normalisée)`` en cache ; rechargée quand la table ``embeddings`` change."""
        rows = self.db.query("SELECT count(*) AS n, coalesce(max(chunk_id), 0) AS mx, coalesce(sum(chunk_id), 0) AS sm "
                             "FROM embeddings WHERE model = ?", (model,))
        signature = (model, int(rows[0]["n"]), int(rows[0]["mx"]), int(rows[0]["sm"])) if rows else (model, 0, 0, 0)
        with self._matrix_lock:
            cache = self._matrix_cache
            if cache is not None and cache["signature"] == signature:
                return cache["ids"], cache["matrix"]
            ids, matrix = self.db.load_embeddings(model)
            if matrix.size:
                matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32))
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                norms[norms == 0.0] = 1.0
                matrix = np.ascontiguousarray(matrix / norms, dtype=np.float32)
            self._matrix_cache = {"signature": signature, "ids": ids, "matrix": matrix}
            return ids, matrix

    def _channel_semantic(self, query_vector: Optional[np.ndarray], chunk_types: List[str],
                          result: SearchResult) -> List[_Candidate]:
        if query_vector is None:
            return []
        model = self._resolve_embedding_model()
        if not model:
            result.warnings.append("Aucun embedding en base : canal sémantique ignoré.")
            return []
        q = np.nan_to_num(np.asarray(query_vector, dtype=np.float32).ravel())
        norm = float(np.linalg.norm(q))
        if q.size == 0 or norm == 0.0:
            result.warnings.append("Vecteur de question vide : canal sémantique ignoré.")
            return []
        q = q / norm
        ids, matrix = self._embedding_matrix(model)
        if matrix.size == 0:
            result.warnings.append("Aucun embedding en base pour le modèle « %s » : canal sémantique ignoré." % model)
            return []
        if matrix.shape[1] != q.shape[0]:
            result.warnings.append("Dimension du vecteur de question (%d) différente des embeddings « %s » (%d) : "
                                   "canal sémantique ignoré." % (q.shape[0], model, matrix.shape[1]))
            return []
        result.semantic_available = True
        sims = matrix @ q
        k = min(CHANNEL_LIMIT * 2, sims.shape[0])
        if k < sims.shape[0]:
            top = np.argpartition(-sims, k - 1)[:k]
        else:
            top = np.arange(sims.shape[0])
        top = top[np.argsort(-sims[top], kind="stable")]
        ordered = [(int(ids[i]), float(sims[i])) for i in top if float(sims[i]) > 0.0]
        if not ordered:
            return []
        types_ok = set(chunk_types)
        chunks = {c.id: c for c in self.db.get_chunks([cid for cid, _ in ordered])}
        out: List[_Candidate] = []
        for cid, sim in ordered:
            c = chunks.get(cid)
            if c is None or c.source_type not in types_ok:
                continue
            out.append(_Candidate(("chunk", cid), "proche sémantiquement (similarité %s)" % _fr_number(sim), "chunk",
                                  chunk_id=cid, extra={"similarity": round(sim, 4)}))
            if len(out) >= CHANNEL_LIMIT:
                break
        return out

    # ------------------------------------------------------------------ canal graphe
    def _channel_graph(self, analysis: QueryAnalysis, seeds: List[_Candidate], allowed: Set[str]) -> List[_Candidate]:
        seed_nodes: List[Tuple[str, str, float]] = []  # (clé de noeud, libellé, poids) — éléments exacts d'abord
        file_seeds: List[Tuple[str, str, float]] = []  # fichiers qui les contiennent (imports, documents…)
        seen_nodes: Set[str] = set()

        def add_seed(key: str, label: str, weight: float, target: List[Tuple[str, str, float]]) -> None:
            if key not in seen_nodes and len(target) < MAX_GRAPH_SEEDS:
                seen_nodes.add(key)
                target.append((key, label, min(1.0, weight)))

        # Question d'impact (« qui appelle / utilise… ») : les relations entrantes d'abord, et les importateurs
        # du fichier d'un modèle ou d'une classe sont la surface d'impact (aucune limite « via imports »).
        impact = analysis.intent in ("who_calls", "impact")
        # Fichiers des modèles trouvés : ils ne sont des graines que pour une question d'impact, qu'ils soient
        # atteints par le modèle, par sa classe homonyme ou par un mot nu égal à leur nom (« User » → User.php).
        model_files: Set[int] = {int(c.row.file_id) for c in seeds
                                 if c.kind == "model" and isinstance(c.row, ModelRow) and c.row.file_id is not None}
        for c in seeds:
            if c.kind == "symbol" and isinstance(c.row, SymbolRow):
                label = ("%s::%s" % (c.row.parent, c.row.name)) if c.row.parent else c.row.name
                add_seed(node_key("symbol", c.key[1]), label, c.weight, seed_nodes)
                # Seul un symbole de premier niveau (classe, fonction) apporte son fichier : ses importateurs
                # l'utilisent. Le fichier d'une méthode n'est relié qu'à sa classe, déjà graine si elle est citée.
                if c.row.file_id is not None and not c.row.parent and (impact or c.row.file_id not in model_files):
                    add_seed(node_key("file", c.row.file_id), label, c.weight, file_seeds)
            elif c.kind == "endpoint" and isinstance(c.row, EndpointRow):
                label = ("%s %s" % (c.row.method or "", c.row.path or "")).strip()
                add_seed(node_key("endpoint", c.key[1]), label, c.weight, seed_nodes)
                # Jamais le fichier de routes : ses imports sont tous les contrôleurs du module, pas celui de
                # la route (atteint par ``handles`` depuis le noeud endpoint).
            elif c.kind == "model" and isinstance(c.row, ModelRow):
                add_seed(node_key("model", c.key[1]), c.row.name, c.weight, seed_nodes)
                if c.row.table_name:
                    add_seed(node_key("table", str(c.row.table_name).strip().lower()), c.row.table_name, c.weight,
                             seed_nodes)
                # Le fichier du modèle (→ requêtes, politiques, fournisseurs, tests qui l'importent) ne sert
                # qu'à une question d'impact ; le noeud modèle porte déjà ``uses_model``, ``documented_by``…
                if impact and c.row.file_id is not None:
                    add_seed(node_key("file", c.row.file_id), c.row.name, c.weight, file_seeds)
            elif c.kind in ("file", "chunk") and isinstance(c.row, FileRow) and c.row.id is not None:
                # Un fichier que la question nomme (chemin ou nom de fichier, poids ≥ W_NAMED) ; pas un mot nu qui
                # se trouve être le radical d'un nom de fichier (« Shipment » → Shipment.php, poids W_BARE).
                if impact or (c.weight >= W_NAMED and c.row.id not in model_files):
                    add_seed(node_key("file", c.row.id), os.path.basename(c.row.path), c.weight, file_seeds)
        for t in analysis.tables:
            add_seed(node_key("table", t.lower()), t, W_NAMED, seed_nodes)
        seed_nodes = seed_nodes + file_seeds
        if not seed_nodes:
            return []

        # Chaque graine apporte au plus GRAPH_PER_SEED candidats au premier tour (dans l'ordre des graines :
        # éléments exacts, premières routes, puis fichiers), le reste suit : un modèle utilisé par quarante
        # fichiers n'étouffe pas le gestionnaire d'une route trouvée après lui.
        per_seed: List[List[_Candidate]] = []
        total = 0
        for key, label, weight in seed_nodes:
            rels_in = self.db.relations_to(key)
            rels_out = self.db.relations_from(key)
            ordered = (rels_in + rels_out) if impact else (rels_out + rels_in)
            # confiance décroissante ; les arêtes imports d'un fichier après ses autres relations
            # (gestionnaire, modèle, documentation, mentions), et au plus GRAPH_IMPORTS_PER_SEED hors impact
            ordered.sort(key=lambda r: (r.kind in _IMPORT_RELATION_KINDS, -float(r.confidence or 0.0)))
            cands: List[_Candidate] = []
            imported = 0
            for rel in ordered:
                other = rel.src if rel.dst == key else rel.dst
                if other == key:
                    continue
                is_import = rel.kind in _IMPORT_RELATION_KINDS
                if is_import and not impact and imported >= GRAPH_IMPORTS_PER_SEED:
                    continue
                reason = "relié à %s via %s" % (label, _RELATION_LABELS.get(rel.kind, rel.kind))
                cand = self._candidate_for_node(other, reason, allowed)
                if cand is not None:
                    cand.weight = weight
                    cand.extra["relation"] = rel.kind
                    cand.extra["via"] = label
                    cands.append(cand)
                    total += 1
                    if is_import:
                        imported += 1
                if total >= CHANNEL_LIMIT * 2:
                    break
            per_seed.append(cands)
            if total >= CHANNEL_LIMIT * 2:
                break
        out: List[_Candidate] = []
        for cands in per_seed:
            out.extend(cands[:GRAPH_PER_SEED])
        for cands in per_seed:
            out.extend(cands[GRAPH_PER_SEED:])
        return out

    def _candidate_for_node(self, node: str, reason: str, allowed: Set[str]) -> Optional[_Candidate]:
        prefix, _, ident = node.partition(":")
        if prefix == "table":
            if "model" not in allowed or not ident:
                return None
            models = self.db.find_models(table_name=ident)
            if models and models[0].id is not None:
                return _Candidate(("model", models[0].id), reason, "model", row=models[0])
            return None
        num = _as_int(ident)
        if num is None:
            return None  # module:<texte brut> et autres noeuds sans cible indexée
        if prefix == "file":
            if "file" not in allowed:
                return None
            f = self.db.get_file_by_id(num)
            return self._file_candidate(f, reason) if f is not None else None
        if prefix == "symbol":
            if "symbol" not in allowed:
                return None
            s = self.db.get_symbol(num)
            return _Candidate(("symbol", num), reason, "symbol", row=s) if s is not None else None
        if prefix == "endpoint":
            if "endpoint" not in allowed:
                return None
            e = self.db.get_endpoint(num)
            return _Candidate(("endpoint", num), reason, "endpoint", row=e) if e is not None else None
        if prefix == "model":
            if "model" not in allowed:
                return None
            m = self.db.get_model(num)
            return _Candidate(("model", num), reason, "model", row=m) if m is not None else None
        if prefix == "document":
            if "document" not in allowed:
                return None
            chunks = self.db.chunks_for("document", num)
            if chunks and chunks[0].id is not None:
                return _Candidate(("chunk", int(chunks[0].id)), reason, "chunk", chunk_id=int(chunks[0].id))
            d = self.db.get_document(num)
            return _Candidate(("document", num), reason, "document", row=d) if d is not None else None
        if prefix == "note":
            if "note" not in allowed:
                return None
            return self._note_candidate(num, reason)
        return None

    def _note_candidate(self, note_id: int, reason: str, note: Optional[NoteRow] = None) -> Optional[_Candidate]:
        chunks = self.db.chunks_for("note", note_id)
        if chunks and chunks[0].id is not None:
            return _Candidate(("chunk", int(chunks[0].id)), reason, "chunk", chunk_id=int(chunks[0].id))
        n = note if note is not None else self.db.get_note(note_id)
        return _Candidate(("note", note_id), reason, "note", row=n) if n is not None else None

    # ------------------------------------------------------------------ canal temporel
    def _channel_temporal(self, analysis: QueryAnalysis) -> List[_Candidate]:
        if analysis.date_from or analysis.date_to:
            notes = self.db.list_notes(date_from=analysis.date_from, date_to=analysis.date_to, limit=TEMPORAL_LIMIT)
        elif analysis.intent == "decision":
            notes = self.db.list_notes(limit=RECENT_NOTES_LIMIT)
        else:
            return []
        if not notes:
            return []
        keywords = [fold(k) for k in analysis.keywords if len(k) >= 3]

        def matches(n: NoteRow) -> int:
            if not keywords:
                return 0
            hay = fold((n.title or "") + "\n" + (n.body or "") + "\n" + " ".join(str(t) for t in n.tags or []))
            return sum(1 for k in keywords if k in hay)

        # Notes citant les mots-clés d'abord, puis les plus récentes (deux tris stables).
        ranked = sorted(notes, key=lambda n: (n.note_date or "", n.id or 0), reverse=True)
        ranked.sort(key=lambda n: -matches(n))
        out: List[_Candidate] = []
        for n in ranked:
            if n.id is None:
                continue
            reason = "note du %s" % (n.note_date or "sans date")
            if matches(n):
                reason += " (mots-clés présents)"
            cand = self._note_candidate(int(n.id), reason, note=n)
            if cand is not None:
                out.append(cand)
        return out

    # ------------------------------------------------------------------ fusion RRF
    @staticmethod
    def _fuse(per_channel: Dict[str, List[_Candidate]], weights: Dict[str, float]) -> List[_Fused]:
        fused: Dict[Tuple[str, int], _Fused] = {}
        order: List[Tuple[str, int]] = []
        for channel in _CHANNEL_PRIORITY:
            cands = per_channel.get(channel)
            if not cands:
                continue
            w = float(weights.get(channel, 1.0))
            for rank, cand in enumerate(cands, start=1):
                entry = fused.get(cand.key)
                if entry is None:
                    entry = _Fused(cand=cand)
                    fused[cand.key] = entry
                    order.append(cand.key)
                part = w * float(cand.weight or 1.0) / (RRF_K + rank)
                entry.score += part
                entry.contrib[channel] = entry.contrib.get(channel, 0.0) + part
                if channel not in entry.channels:
                    entry.channels.append(channel)
                entry.reasons.setdefault(channel, cand.reason)
                for k, v in cand.extra.items():
                    entry.extra.setdefault(k, v)
        out = [fused[k] for k in order]
        out.sort(key=lambda f: -f.score)
        return out

    # ------------------------------------------------------------------ construction des hits
    def _materialize(self, fused: List[_Fused], loader: _Loader) -> List[SearchHit]:
        chunk_ids = [f.cand.chunk_id for f in fused if f.cand.kind == "chunk" and f.cand.chunk_id is not None]
        chunks: Dict[int, ChunkRow] = {}
        if chunk_ids:
            try:
                chunks = {int(c.id): c for c in self.db.get_chunks(chunk_ids) if c.id is not None}
            except Exception as exc:  # noqa: BLE001
                logger.warning("chargement des morceaux impossible: %s", exc)
        file_ids: List[int] = []
        for f in fused:
            if f.cand.kind == "chunk":
                c = chunks.get(f.cand.chunk_id or -1)
                if c is not None and c.source_type == "file" and c.source_id is not None:
                    file_ids.append(int(c.source_id))
            elif f.cand.kind in ("symbol", "endpoint", "model") and f.cand.row is not None:
                fid = getattr(f.cand.row, "file_id", None)
                if fid is not None:
                    file_ids.append(int(fid))
        loader.preload_files(file_ids)

        hits: List[SearchHit] = []
        for f in fused:
            try:
                hit = self._build_hit(f.cand, chunks, loader)
            except Exception as exc:  # noqa: BLE001
                logger.warning("construction d'un résultat impossible (%s): %s", f.cand.key, exc)
                hit = None
            if hit is None:
                continue
            if hit.repo_id and "repo_name" not in hit.extra:
                # comme les morceaux : l'en-tête d'un passage du chat montre « dépôt/chemin », pas l'identifiant du dépôt
                repo = loader.repos().get(hit.repo_id)
                if repo is not None:
                    hit.extra["repo_name"] = repo.name
            hit.score = round(float(f.score), 6)
            hit.channels = list(f.channels)
            hit.reason = " ; ".join(f.reasons[c] for c in _CHANNEL_PRIORITY if c in f.reasons)
            for k, v in f.extra.items():
                hit.extra.setdefault(k, v)
            hit.extra["channel_scores"] = {c: round(v, 6) for c, v in f.contrib.items()}
            hits.append(hit)
        return hits

    def _simple_hits(self, cands: List[_Candidate], loader: _Loader, channel: str) -> List[SearchHit]:
        fused = [_Fused(cand=c, score=1.0 / (RRF_K + i + 1), channels=[channel], reasons={channel: c.reason},
                        contrib={channel: 1.0 / (RRF_K + i + 1)}, extra=dict(c.extra))
                 for i, c in enumerate(cands)]
        return self._materialize(fused, loader)

    def _build_hit(self, cand: _Candidate, chunks: Dict[int, ChunkRow], loader: _Loader) -> Optional[SearchHit]:
        if cand.kind == "chunk":
            c = chunks.get(cand.chunk_id or -1)
            return self._hit_from_chunk(c, loader) if c is not None else None
        if cand.kind == "symbol":
            return self._hit_from_symbol(cand.row, loader)
        if cand.kind == "endpoint":
            return self._hit_from_endpoint(cand.row, loader)
        if cand.kind == "model":
            return self._hit_from_model(cand.row, loader)
        if cand.kind == "file":
            return self._hit_from_file(cand.row, loader)
        if cand.kind == "note":
            return self._hit_from_note(cand.row)
        if cand.kind == "document":
            return self._hit_from_document(cand.row)
        return None

    def _hit_from_chunk(self, c: ChunkRow, loader: _Loader) -> Optional[SearchHit]:
        meta = c.meta if isinstance(c.meta, dict) else {}
        text = c.text if isinstance(c.text, str) else str(c.text or "")
        body_offset = _as_int(meta.get("body_offset")) or 0
        body = text[body_offset:] if 0 < body_offset < len(text) else text
        hit = SearchHit(chunk_id=c.id, source_type=c.source_type or "file", source_id=c.source_id,
                        line_start=c.line_start, line_end=c.line_end, page=_as_int(meta.get("page")),
                        snippet=make_snippet(body), text=text, extra={"ordinal": c.ordinal})
        if c.sha1:
            hit.extra["sha1"] = str(c.sha1)  # empreinte du texte : même passage indexé sous deux sources
        if c.source_type == "file":
            f = loader.file(_as_int(c.source_id))
            if f is not None:
                hit.path, hit.repo_id, hit.file_id = f.path, f.repo_id, f.id
                hit.title = os.path.basename(f.path)
                hit.extra["language"] = f.language or ""
                repo = loader.repos().get(f.repo_id)
                if repo is not None:
                    hit.extra["repo_name"] = repo.name
            else:
                hit.file_id = _as_int(c.source_id)
                hit.title = "fichier %s" % c.source_id
            if meta.get("symbol_hint"):
                hit.extra["symbol_hint"] = meta.get("symbol_hint")
        elif c.source_type == "document":
            d = loader.document(_as_int(c.source_id))
            hit.title = (d.title if d is not None and d.title else "document %s" % c.source_id)
            hit.extra["document_id"] = c.source_id
            if d is not None:
                hit.extra["format"] = d.format or ""
                hit.extra["source_path"] = d.source_path or d.stored_path or ""
            section = meta.get("section") or meta.get("heading") or meta.get("title")
            if section and section != hit.title:
                hit.extra["section"] = section
        elif c.source_type == "note":
            n = loader.note(_as_int(c.source_id))
            hit.title = (n.title if n is not None and n.title else str(meta.get("title") or "note %s" % c.source_id))
            hit.extra["note_id"] = c.source_id
            if n is not None:
                hit.extra["note_date"] = n.note_date
                hit.extra["tags"] = list(n.tags or [])
            else:
                hit.extra["note_date"] = meta.get("note_date")
        return hit

    def _disk_window(self, file_row: Optional[FileRow], loader: _Loader, start: Optional[int], end: Optional[int],
                     max_lines: int = SYMBOL_MAX_LINES) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        lines = loader.lines(file_row)
        if lines is None or not lines:
            return None, start, end
        total = len(lines)
        s = max(1, min(_as_int(start) or 1, total))
        e = _as_int(end)
        if e is None or e < s:
            e = min(s + max_lines - 1, total)
        e = min(e, s + max_lines - 1, total)
        return "\n".join(lines[s - 1:e]), s, e

    def _hit_from_symbol(self, s: Optional[SymbolRow], loader: _Loader) -> Optional[SearchHit]:
        if s is None or s.id is None:
            return None
        f = loader.file(s.file_id)
        title = ("%s::%s" % (s.parent, s.name)) if s.parent else s.name
        text, ls, le = self._disk_window(f, loader, s.line_start, s.line_end)
        signature = s.signature or ""
        hit = SearchHit(chunk_id=None, source_type="symbol", source_id=s.id, title=title,
                        path=f.path if f else None, repo_id=f.repo_id if f else None, file_id=s.file_id,
                        line_start=ls if text is not None else s.line_start, line_end=le if text is not None else s.line_end,
                        snippet=make_snippet(text) if text else make_snippet(signature or title),
                        text=text if text is not None else (signature or title),
                        extra={"symbol_id": s.id, "kind": s.kind, "signature": signature, "parent": s.parent or ""})
        if f is not None:
            hit.extra["language"] = f.language or ""
        return hit

    def _hit_from_endpoint(self, e: Optional[EndpointRow], loader: _Loader) -> Optional[SearchHit]:
        if e is None or e.id is None:
            return None
        f = loader.file(e.file_id)
        title = ("%s %s" % ((e.method or "").upper(), e.path or e.normalized_path or "")).strip()
        end = (e.line + ENDPOINT_WINDOW_LINES - 1) if e.line else None
        text, ls, le = self._disk_window(f, loader, e.line, end)
        fallback = "%s (%s, %s)" % (title, e.side, e.handler or e.framework or "")
        hit = SearchHit(chunk_id=None, source_type="endpoint", source_id=e.id, title=title,
                        path=f.path if f else None, repo_id=f.repo_id if f else None, file_id=e.file_id,
                        line_start=ls if text is not None else e.line, line_end=le if text is not None else e.line,
                        snippet=make_snippet(text) if text else make_snippet(fallback),
                        text=text if text is not None else fallback,
                        extra={"endpoint_id": e.id, "side": e.side, "method": (e.method or "").upper(), "route": e.path,
                               "normalized_path": e.normalized_path, "handler": e.handler or "",
                               "framework": e.framework or "", "request_class": "", "request_class_id": None})
        req = loader.request_class(e.handler) if e.side == "server" else None
        if req is not None:
            hit.extra["request_class_id"], hit.extra["request_class"] = req[0], req[1]
            if req[2] is not None:
                hit.extra["request_file_id"] = req[2]
        return hit

    def _hit_from_model(self, m: Optional[ModelRow], loader: _Loader) -> Optional[SearchHit]:
        if m is None or m.id is None:
            return None
        f = loader.file(m.file_id)
        title = m.name + (" (table %s)" % m.table_name if m.table_name and m.table_name != m.name else "")
        end = (m.line + MODEL_WINDOW_LINES - 1) if m.line else None
        text, ls, le = self._disk_window(f, loader, m.line, end)
        field_names = [str(fd.get("name") if isinstance(fd, dict) else fd) for fd in (m.fields or [])]
        fallback = "%s : %s" % (title, ", ".join(field_names) if field_names else "aucun champ connu")
        hit = SearchHit(chunk_id=None, source_type="model", source_id=m.id, title=title,
                        path=f.path if f else None, repo_id=f.repo_id if f else None, file_id=m.file_id,
                        line_start=ls if text is not None else m.line, line_end=le if text is not None else m.line,
                        snippet=make_snippet(text) if text else make_snippet(fallback),
                        text=text if text is not None else fallback,
                        extra={"model_id": m.id, "kind": m.kind or "", "table_name": m.table_name or "",
                               "fields": field_names[:50], "field_count": len(m.fields or []),
                               "relation_count": len(m.relations or [])})
        return hit

    def _hit_from_file(self, f: Optional[FileRow], loader: _Loader) -> Optional[SearchHit]:
        if f is None or f.id is None:
            return None
        text, ls, le = self._disk_window(f, loader, 1, None)
        return SearchHit(chunk_id=None, source_type="file", source_id=f.id, title=os.path.basename(f.path), path=f.path,
                         repo_id=f.repo_id, file_id=f.id, line_start=ls if text is not None else None,
                         line_end=le if text is not None else None, snippet=make_snippet(text or f.path),
                         text=text or "", extra={"language": f.language or ""})

    @staticmethod
    def _hit_from_note(n: Optional[NoteRow]) -> Optional[SearchHit]:
        if n is None or n.id is None:
            return None
        body = n.body or ""
        text = ("%s\n\n%s" % (n.title, body)) if n.title else body
        return SearchHit(chunk_id=None, source_type="note", source_id=n.id, title=n.title or "note %s" % n.id,
                         line_start=1, line_end=max(1, len(body.splitlines())), snippet=make_snippet(body or n.title),
                         text=text, extra={"note_id": n.id, "note_date": n.note_date, "tags": list(n.tags or [])})

    @staticmethod
    def _hit_from_document(d: Optional[DocumentRow]) -> Optional[SearchHit]:
        if d is None or d.id is None:
            return None
        return SearchHit(chunk_id=None, source_type="document", source_id=d.id, title=d.title or "document %s" % d.id,
                         snippet=make_snippet(d.title), text=d.title or "",
                         extra={"document_id": d.id, "format": d.format or "", "source_path": d.source_path or ""})

    # ------------------------------------------------------------------ déduplication / diversité
    @staticmethod
    def _dedupe_and_diversify(hits: List[SearchHit]) -> List[SearchHit]:
        """Déduplication puis diversité.

        Deux résultats du même fichier dont les lignes se chevauchent décrivent le même code : on n'en
        garde qu'un. Un élément structuré (symbole, endpoint, modèle — trouvé par ``identifier``,
        ``route`` ou ``graph``) l'emporte sur un simple morceau, car il porte sa signature, son
        gestionnaire ou ses champs ; entre deux morceaux, le mieux classé gagne. Le gagnant cumule les
        preuves (score, canaux, raison). Deux éléments structurés distincts (classe et méthode…) sont
        tous deux conservés. Un même texte indexé sous deux sources (le PDF d'un dépôt et le même PDF
        téléversé comme document : même ``chunks.sha1``) n'est gardé qu'une fois, sous sa source la mieux
        classée, qui cumule les preuves de l'autre. Enfin, au plus :data:`MAX_PER_SOURCE` résultats par
        fichier / document / note.
        """
        kept: List[SearchHit] = []
        by_file: Dict[int, List[SearchHit]] = {}
        by_sha: Dict[str, SearchHit] = {}  # empreinte du texte -> morceau conservé (toutes sources confondues)
        for hit in hits:
            sha = SearchEngine._content_key(hit)
            if sha is not None:
                twin = by_sha.get(sha)
                if twin is not None:
                    SearchEngine._merge_evidence(twin, hit)
                    continue
            fid = hit.file_id if hit.source_type in ("file", "symbol", "endpoint", "model") else None
            if fid is not None and hit.line_start is not None and hit.line_end is not None:
                duplicate = None
                for other in by_file.get(fid, []):
                    if other.line_start is None or other.line_end is None:
                        continue
                    if hit.source_type != "file" and other.source_type != "file":
                        continue  # deux éléments structurés distincts
                    if hit.line_start <= other.line_end and other.line_start <= hit.line_end:
                        duplicate = other
                        break
                if duplicate is not None:
                    winner, loser = SearchEngine._pick_winner(duplicate, hit)
                    SearchEngine._merge_evidence(winner, loser)
                    if winner is not duplicate:
                        kept[kept.index(duplicate)] = winner
                        bucket = by_file[fid]
                        bucket[bucket.index(duplicate)] = winner
                        for key, value in by_sha.items():
                            if value is duplicate:
                                by_sha[key] = winner  # les jumeaux suivants cumulent sur le gagnant
                    continue
            kept.append(hit)
            if fid is not None:
                by_file.setdefault(fid, []).append(hit)
            if sha is not None:
                by_sha[sha] = hit
        kept.sort(key=lambda h: -h.score)
        out: List[SearchHit] = []
        per_source: Dict[Tuple[str, int], int] = {}
        for hit in kept:
            group = SearchEngine._source_group(hit)
            if group is not None:
                n = per_source.get(group, 0)
                if n >= MAX_PER_SOURCE:
                    continue
                per_source[group] = n + 1
            out.append(hit)
        return out

    @staticmethod
    def _targets(client: SearchHit, server: SearchHit) -> bool:
        """Vrai si l'appel client vise cette route serveur : même chemin normalisé, ou chemin de l'appel égal à
        la fin du chemin serveur (``/orders/:param/confirm`` vise ``/api/orders/:param/confirm``), et méthode
        compatible (égale, ou ``ANY`` d'un côté)."""
        cp = str(client.extra.get("normalized_path") or "")
        sp = str(server.extra.get("normalized_path") or "")
        if len(cp) <= 1 or not (sp == cp or sp.endswith(cp)):
            return False
        cm = str(client.extra.get("method") or "ANY").upper()
        sm = str(server.extra.get("method") or "ANY").upper()
        return cm == sm or "ANY" in (cm, sm)

    @staticmethod
    def _server_routes_first(hits: List[SearchHit]) -> List[SearchHit]:
        """Un appel client (``fetch()`` d'une vue Blade, ``Http::post`` d'un service) est un usage d'une route, pas
        la route. La route serveur qu'il vise est une graine du graphe, qui ne crédite que ses voisins : l'appel
        cumule alors le crédit du canal route, du graphe (voisin « appel d'API ») et du morceau de sa vue, et
        passe devant la route elle-même. S'il précède une route serveur qu'il vise (:meth:`_targets`), il est
        replacé juste derrière elle, à son score, et porte ``after_server_route`` (identifiant de la route).
        Un appel vers un service tiers, sans route serveur dans les résultats, ne bouge pas."""
        out = list(hits)
        for client in hits:
            if client.source_type != "endpoint" or client.extra.get("side") != "client":
                continue
            i = next(k for k, h in enumerate(out) if h is client)
            twin = next((h for h in out[i + 1:] if h.source_type == "endpoint" and h.extra.get("side") == "server"
                         and SearchEngine._targets(client, h)), None)
            if twin is None:
                continue
            del out[i]
            j = next(k for k, h in enumerate(out) if h is twin)
            out.insert(j + 1, client)
            client.score = twin.score
            client.extra["after_server_route"] = twin.extra.get("endpoint_id")
        return out

    @staticmethod
    def _is_structured(hit: SearchHit) -> bool:
        return hit.source_type in ("symbol", "endpoint", "model")

    @staticmethod
    def _content_key(hit: SearchHit) -> Optional[str]:
        """Empreinte ``chunks.sha1`` d'un morceau (``None`` pour un élément structuré ou un hit sans morceau)."""
        if hit.chunk_id is None:
            return None
        sha = hit.extra.get("sha1") if isinstance(hit.extra, dict) else None
        return str(sha) if sha else None

    @staticmethod
    def _pick_winner(first: SearchHit, second: SearchHit) -> Tuple[SearchHit, SearchHit]:
        """``first`` est le mieux classé ; un élément structuré l'emporte toutefois sur un morceau."""
        if SearchEngine._is_structured(second) and not SearchEngine._is_structured(first):
            return second, first
        return first, second

    @staticmethod
    def _merge_evidence(winner: SearchHit, loser: SearchHit) -> None:
        """Même région de code trouvée deux fois : par canal on garde le meilleur apport (jamais de
        double compte), le score devient la somme de ces apports."""
        scores: Dict[str, float] = dict(winner.extra.get("channel_scores") or {})
        for ch, value in (loser.extra.get("channel_scores") or {}).items():
            scores[ch] = max(scores.get(ch, 0.0), float(value))
        if scores:
            winner.extra["channel_scores"] = scores
            winner.score = round(sum(scores.values()), 6)
        else:
            winner.score = round(max(winner.score, loser.score), 6)
        for ch in loser.channels:
            if ch not in winner.channels:
                winner.channels.append(ch)
        if loser.reason and loser.reason not in winner.reason:
            winner.reason = (winner.reason + " ; " + loser.reason) if winner.reason else loser.reason
        for k, v in loser.extra.items():
            winner.extra.setdefault(k, v)
        if winner.chunk_id is None and loser.chunk_id is not None:
            winner.extra.setdefault("merged_chunk_id", loser.chunk_id)

    @staticmethod
    def _source_group(hit: SearchHit) -> Optional[Tuple[str, int]]:
        if hit.source_type in ("file", "symbol", "endpoint", "model"):
            return ("file", int(hit.file_id)) if hit.file_id is not None else None
        if hit.source_id is not None:
            return (hit.source_type, int(hit.source_id))
        return None


# --------------------------------------------------------------------------- utilitaires

def _fr_number(value: float) -> str:
    return ("%.2f" % value).replace(".", ",")


def _short_error(exc: BaseException) -> str:
    msg = str(exc).strip() or exc.__class__.__name__
    return msg if len(msg) <= 160 else msg[:159] + "…"


__all__ = ["CHANNELS", "CLASS_KINDS", "GENERATED_FILE_MAX_BYTES", "HIT_SOURCE_TYPES", "MAX_PER_SOURCE", "ROUTE_LIMIT",
           "RRF_K", "SearchEngine", "SearchHit", "SearchResult", "edit_distance", "fuzzy_key", "handler_words",
           "inherited_words", "is_dynamic_client_call", "is_generated_file", "lexical_terms_found", "make_snippet", "max_edits",
           "question_language", "read_file_lines", "request_class_names", "route_contains", "word_names_symbol"]
