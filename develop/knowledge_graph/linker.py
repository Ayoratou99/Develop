"""Graphe de relations dérivées de la mémoire Develop.

L'indexeur enregistre, fichier par fichier, des relations **brutes** : ``imports_raw``
(``file:<id>`` → ``module:<module tel qu'écrit>``), ``defines_table`` (``model:<id>`` →
``table:<nom>``) et ``handles`` (``endpoint:<id>`` serveur → ``symbol:<id>``). Ce module les
transforme, à la fin de chaque run d'indexation, en relations **dérivées**, recalculées
globalement (chaque genre est supprimé puis recréé, donc idempotent) :

======================  ======================================  ==========================================
genre                   src → dst                               règle
======================  ======================================  ==========================================
``imports``             ``file:A`` → ``file:B``                 import résolu vers un fichier du même dépôt
``calls``               ``symbol:A`` / ``file:A`` → ``symbol:B``  ``B(`` dans le corps de A, B appelable du
                                                                même fichier ou d'un fichier importé (0.5–0.8)
``calls_endpoint``      ``endpoint:client`` → ``endpoint:srv``  appel HTTP ↔ route, confiance graduée
``same_entity``         ``model:X`` → ``model:Y``               même table / même entité (0.9, 0.8, 0.7)
``uses_model``          ``file:A`` → ``model:M``                import du fichier du modèle, relation déclarée
                                                                par un modèle, nom exact cité dans le dépôt
                                                                de M (0.6)
``uses_model``          ``endpoint:E`` → ``model:M``            route serveur gérée par ``<M>Controller`` ou
                                                                par un contrôleur qui importe M (0.6)
``documented_by``       noeud → ``document:D`` / ``note:N``     identifiant cité sous forme code, ou mot
                                                                ordinaire répété, dans le texte (0.9)
``mentions``            ``document:D`` / ``note:N`` → ``file:F``  fichier dont des éléments sont cités (0.7)
======================  ======================================  ==========================================

Les fichiers de documentation des dépôts (``files.kind == 'doc'`` : README, ``docs/*.md``…) sont
traités comme des documents : ils reçoivent les mêmes relations avec ``file:<id>`` pour destination
(``documented_by``) ou source (``mentions``). Les fichiers texte destinés aux machines
(:data:`NON_DOCUMENTARY_FILES` : ``robots.txt``…) sont ignorés, et un PDF / DOCX ne compte comme
documentation qu'à la racine du dépôt ou sous un dossier de documentation
(:data:`DOCUMENTATION_DIR_NAMES` : ``docs/``, ``public/documentation/``…) : une attestation générée
dans ``public/``, une page web enregistrée ou un export sous ``storage/`` citent des modèles et des
tables sans les documenter (voir :func:`_is_documentary_file`).

Un texte ne documente le code que s'il cite **au moins un identifiant sous forme code** présent dans
l'index (accents graves, CamelCase, snake_case, ``Classe::méthode``, route, chemin) : une fiche, un
formulaire, un PDF générique ou ``robots.txt`` (« User-agent ») ne mentionnent que des mots ordinaires
et ne créent aucune relation. Un mot ordinaire homonyme d'une table ou d'un modèle (« customers »,
« User ») ne compte qu'à partir de :data:`MIN_WEAK_MENTIONS` occurrences et ne relie que la table,
pas chacune des migrations qui la définissent. Un nom porté par plus de :data:`MAX_HOMONYMS` noeuds
d'un même genre (``created_at`` → des dizaines de champs, ``map``) n'en documente aucun.

Le genre brut ``handles`` posé par l'indexeur (gestionnaire du même fichier) est conservé tel quel ;
:func:`link_handlers` le complète quand le gestionnaire d'une route est défini dans un autre fichier
du dépôt (``OrderController@confirm`` → méthode ``confirm`` de la classe ``OrderController``) et
recalcule ses propres relations (``meta.rule == 'linker'``) à chaque passe.

Les **préfixes d'URL de base** d'un dépôt (``/api``…) sont appris en lecture seule depuis
``.env*``, ``axios.create({ baseURL })``, ``Dio(BaseOptions(baseUrl:))``, la configuration de
proxy de Vite / webpack / Angular et la convention Laravel ``routes/api.php`` ; ils sont
stockés dans ``meta`` (``repo:<id>:base_prefixes``, JSON). Aucun accès réseau, aucune
exécution de code, aucune écriture dans les dépôts.

Robustesse : une erreur sur un import, un endpoint, un modèle ou un document est consignée
(:attr:`LinkStats.errors`) et n'arrête jamais la passe. Performance : tout est fait avec des
index en mémoire construits par quelques requêtes en lot, puis écrit en transactions
groupées (jamais une requête par import).
"""
from __future__ import annotations

import json
import math
import os
import posixpath
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from develop.code_analysis.model import normalize_route_path
from develop.config import RepoConfig, WorkspaceConfig
from develop.knowledge_graph.mentions import (MAX_MENTIONS, Mention, NameIndex, match_candidates, scan_text,
                                              source_text)
from develop.project_scanner import read_text
from develop.project_scanner.languages import TEXTLESS_LANGUAGES
from develop.storage import Database, EndpointRow, FileRow, ModelRow, RelationRow, node_key

__all__ = [
    "CALLABLE_KINDS",
    "DERIVED_KINDS",
    "DOCUMENTATION_DIR_NAMES",
    "GENERIC_PREFIXES",
    "LinkStats",
    "MAX_HOMONYMS",
    "MIN_WEAK_MENTIONS",
    "NON_DOCUMENTARY_FILES",
    "describe_relation",
    "learn_base_prefixes",
    "link_all",
    "link_calls",
    "link_documents",
    "link_handlers",
    "link_models",
    "match_endpoints",
    "neighbors_graph",
    "resolve_imports",
]

#: Genres de relations produits (et supprimés avant recalcul) par ce module.
DERIVED_KINDS: Tuple[str, ...] = ("imports", "calls", "calls_endpoint", "same_entity", "uses_model", "documented_by",
                                  "mentions")
#: Genres de symboles appelables : cibles (et sources) des relations ``calls``, gestionnaires de routes.
CALLABLE_KINDS = frozenset({"function", "method", "component", "constructor", "view"})
#: Au-delà de ce nombre de candidats homonymes, un appel (ou un gestionnaire) n'est pas relié.
MAX_CALL_CANDIDATES = 3
MAX_HANDLER_CANDIDATES = 3
#: Gestionnaires anonymes (pas de symbole à relier).
_ANONYMOUS_HANDLERS = frozenset({"closure", "lambda", "inline", "anonymous", "function", "fn", "callback"})
#: ``nom(`` dans le corps d'un symbole (précédé ni d'une lettre, ni d'un ``$``, ni d'un ``.`` collé → ``a.b(`` donne ``b``).
_CALL_RE = re.compile(r"(?<![A-Za-z0-9_$])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
#: Définitions (``function f(``, ``def f(``…) retirées avant la recherche des appels.
_DEF_RE = re.compile(r"\b(?:function|def|fn|func|sub|proc|procedure|constructor)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(")
#: Receveur désignant la classe courante, juste avant le nom appelé (``$this->index(``, ``this.index(``, ``self.index(``).
_OWN_RECEIVER_RE = re.compile(r"(?<![A-Za-z0-9_$])(?:\$this->|this\??\.|self\.|self::|static::|cls\.)$")
#: Mots-clés et constructions qui ressemblent à un appel sans en être un.
_CALL_KEYWORDS = frozenset({
    "if", "elif", "else", "elseif", "for", "foreach", "while", "do", "until", "switch", "match", "case", "when",
    "catch", "except", "try", "finally", "return", "yield", "await", "async", "function", "fn", "def", "lambda",
    "class", "new", "delete", "typeof", "instanceof", "sizeof", "void", "with", "assert", "raise", "throw", "echo",
    "print", "require", "require_once", "include", "include_once", "import", "use", "declare", "isset", "unset",
    "empty", "list", "array", "and", "or", "not", "in", "is", "del", "super", "self", "this", "static", "unless",
    "defined", "exit", "die", "global", "namespace",
})
#: Préfixes d'URL génériques, acceptés avec une confiance réduite quand rien n'a été appris.
GENERIC_PREFIXES: Tuple[str, ...] = ("/api", "/api/v1", "/api/v2", "/v1", "/v2", "/backend")
#: Fichiers d'environnement lus (petits, à la racine du dépôt) pour apprendre les préfixes.
ENV_FILES: Tuple[str, ...] = (".env", ".env.example", ".env.local", ".env.sample", ".env.dist", ".env.template",
                              ".env.development")
#: Fichiers de configuration de proxy de développement lus à la racine du dépôt.
PROXY_CONFIG_FILES: Tuple[str, ...] = ("vite.config.ts", "vite.config.js", "vite.config.mjs", "vite.config.cjs",
                                       "vite.config.mts", "webpack.config.js", "webpack.config.ts",
                                       "proxy.conf.json", "proxy.conf.js", "package.json")
#: Taille maximale (octets) d'un fichier de configuration lu sur disque.
MAX_SMALL_FILE = 64 * 1024
#: Genres de modèles qui *sont* une table (par opposition à une classe).
TABLE_LIKE_KINDS = frozenset({"sql_table", "migration"})
#: Confiance minimale et nombre maximal de routes candidates conservées par appel client.
MIN_ENDPOINT_CONFIDENCE = 0.4
MAX_ENDPOINT_CANDIDATES = 3
#: Occurrences minimales d'un mot ordinaire (sans forme code : « customers », « User ») pour qu'il
#: documente la table ou le modèle homonyme.
MIN_WEAK_MENTIONS = 2
#: Au-delà de ce nombre de noeuds homonymes d'un même genre, une mention n'en documente aucun.
MAX_HOMONYMS = 5
#: Fichiers texte destinés aux machines (``files.kind == 'doc'`` par extension), jamais reliés comme documentation.
NON_DOCUMENTARY_FILES = frozenset({"robots.txt", "humans.txt", "security.txt", "ads.txt", "app-ads.txt",
                                   "sitemap.txt"})
#: Dossiers (segment de chemin, insensible à la casse) sous lesquels un PDF / DOCX d'un dépôt est de la
#: documentation du projet (``docs/guide.pdf``, ``public/documentation/parametrage.pdf``, ``wiki/``…) ;
#: ailleurs, un document binaire est un contenu produit ou déposé par l'application, jamais relié.
DOCUMENTATION_DIR_NAMES = frozenset({
    "doc", "docs", "documentation", "wiki", "guide", "guides", "manual", "manuals", "handbook", "spec", "specs",
    "specification", "specifications", "rfc", "rfcs", "adr", "adrs", "design", "notes",
})
_UNKNOWN_METHODS = frozenset({"", "ANY", "*"})
_LARGE_LIMIT = 10_000_000
_MIN_MODEL_NAME = 4
_MAX_ERRORS = 200

_ENV_LINE_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
_ENV_VAR_WANTED_RE = re.compile(r"URL|API|BASE|HOST|ENDPOINT|BACKEND|SERVER", re.IGNORECASE)
_ENV_VAR_EXCLUDED_RE = re.compile(
    r"DATABASE|DB_|_DB\b|REDIS|MAIL|SMTP|MONGO|POSTGRES|PGSQL|MYSQL|SQL|CACHE|QUEUE|SESSION|BROADCAST|PUSHER|"
    r"SENTRY|STRIPE|AWS|S3_|GOOGLE|FACEBOOK|GITHUB|OAUTH|VAPID|MEMCACHED|ELASTIC|KAFKA|RABBIT|AMQP|SOCKET",
    re.IGNORECASE,
)
_URL_LITERAL_RE = re.compile(r"https?://[^\s'\"`<>)]+", re.IGNORECASE)
_PATH_LITERAL_RE = re.compile(r"['\"`](/[A-Za-z0-9_\-./]*)['\"`]")
_ENV_REF_RE = re.compile(r"(?:import\.meta\.env|process\.env)\.([A-Za-z_][A-Za-z0-9_]*)|env\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]")
_BASEURL_RE = re.compile(r"\bbase_?[Uu][Rr][Ll]\s*[:=]\s*(?:['\"`]([^'\"`\n]+)['\"`]|([A-Za-z_$][A-Za-z0-9_$.]*))")
_AXIOS_RECEIVER_RE = re.compile(r"(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*axios\.create\s*\(\s*\{([^}]*)\}")
_DIO_RECEIVER_RE = re.compile(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*Dio\s*\(\s*BaseOptions\s*\(\s*([^)]*)\)")
_PROXY_BLOCK_RE = re.compile(r"\bproxy\s*:\s*\{")
_PROXY_KEY_RE = re.compile(r"['\"](/[A-Za-z0-9_\-/]*)['\"]\s*:")
_SCHEME_HOST_RE = re.compile(r"^[a-z][a-z0-9+.-]*://[^/]*", re.IGNORECASE)
_PHP_SUFFIXES = ("Model", "Entity", "Dto", "DTO", "Record", "Row", "Schema", "Table", "Type", "Interface", "Resource")
#: Suffixes d'une classe gestionnaire de routes, retirés pour retrouver l'entité qu'elle sert
#: (``ShipmentController`` → ``Shipment``, ``UsersResource`` → ``Users``) ; du plus long au plus court.
_HANDLER_SUFFIXES: Tuple[str, ...] = ("ApiController", "Controller", "Controllers", "Resources", "Resource", "Handlers",
                                      "Handler", "Endpoints", "Endpoint", "Router", "Routes", "ViewSet", "Views", "View",
                                      "Api")


# --------------------------------------------------------------------------- statistiques

@dataclass
class LinkStats:
    """Bilan d'une passe de liaison (:func:`link_all`)."""

    imports_resolved: int = 0
    imports_unresolved: int = 0
    handlers_linked: int = 0  # relations ``handles`` complétées entre fichiers
    calls_linked: int = 0  # relations ``calls`` dérivées du texte indexé
    endpoints_matched: int = 0
    endpoints_unmatched_client: int = 0
    endpoints_unmatched_server: int = 0
    models_linked: int = 0
    model_uses: int = 0
    documents_linked: int = 0
    base_prefixes: Dict[str, List[str]] = field(default_factory=dict)
    duration_s: float = 0.0
    total: int = 0  # nombre total de relations dérivées présentes après la passe
    errors: List[str] = field(default_factory=list)

    @property
    def relations(self) -> int:
        """Alias de :attr:`total` (clé lue par l'indexeur dans un résultat sous forme de dict)."""
        return self.total

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["relations"] = self.total
        return data


class _Errors:
    """Journal borné des erreurs non bloquantes d'une passe."""

    def __init__(self, sink: Optional[List[str]] = None) -> None:
        self.items: List[str] = sink if sink is not None else []

    def add(self, message: str) -> None:
        if len(self.items) < _MAX_ERRORS:
            self.items.append(str(message))
        elif len(self.items) == _MAX_ERRORS:
            self.items.append("… d'autres erreurs ont été tronquées.")


# --------------------------------------------------------------------------- utilitaires communs

def _meta(row: Any) -> Dict[str, Any]:
    m = getattr(row, "meta", None)
    return m if isinstance(m, dict) else {}


def _delete_kinds(db: Database, kinds: Sequence[str]) -> int:
    """Supprime toutes les relations des genres donnés (via l'API publique, en une transaction)."""
    n = 0
    with db.transaction():
        for kind in kinds:
            rows = db.query("SELECT DISTINCT src FROM relations WHERE kind = ?", (kind,))
            for r in rows:
                n += db.delete_relations_from(r["src"], kind)
    return n


def _count_kinds(db: Database, kinds: Sequence[str]) -> int:
    if not kinds:
        return 0
    ph = ",".join("?" * len(kinds))
    rows = db.query("SELECT count(*) AS n FROM relations WHERE kind IN (%s)" % ph, list(kinds))
    return int(rows[0]["n"]) if rows else 0


def _files_by_id(db: Database) -> Dict[int, FileRow]:
    return {int(f.id): f for f in db.list_files() if f.id is not None}


def _repo_root(workspace: Optional[WorkspaceConfig], db: Database, repo_id: str) -> Optional[str]:
    cfg = workspace.repo_by_id(repo_id) if workspace is not None else None
    if cfg is not None and cfg.path:
        return cfg.path
    row = db.get_repo(repo_id)
    return row.path if row is not None and row.path else None


def _read_small(path: str) -> Optional[str]:
    """Lit un petit fichier texte (lecture seule, borné) ; ``None`` s'il est absent, binaire ou trop gros."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > MAX_SMALL_FILE:
            return None
    except OSError:
        return None
    return read_text(path, MAX_SMALL_FILE)


def _url_path(url: str) -> str:
    """Partie chemin d'une URL, normalisée (``http://localhost:8000/api/`` → ``/api`` ; sans chemin → ``""``)."""
    p = _SCHEME_HOST_RE.sub("", (url or "").strip().strip("'\"`"))
    p = p.split("?", 1)[0].split("#", 1)[0]
    return _clean_prefix(p)


def _clean_prefix(path: str) -> str:
    """``/api/`` → ``/api`` ; ``""`` ou ``/`` → ``""`` (aucun préfixe)."""
    p = (path or "").strip().strip("'\"`")
    if not p or p == "/":
        return ""
    if not p.startswith("/"):
        p = "/" + p
    norm = normalize_route_path(p)
    return "" if norm == "/" else norm


def _var_name(expr: str) -> str:
    """Nom de variable d'une expression de base : ``${API_BASE}`` → ``API_BASE``, ``process.env.X`` → ``X``."""
    e = (expr or "").strip().strip("'\"`")
    e = e.strip("${}")
    e = e.strip("{}")
    for prefix in ("import.meta.env.", "process.env.", "this.", "self.", "env."):
        if e.startswith(prefix):
            e = e[len(prefix):]
    return e.strip("$").strip()


def _segments(path: str) -> List[str]:
    return [s for s in (path or "").split("/") if s]


def _join_prefix(prefix: str, path: str) -> str:
    if not prefix:
        return path
    if path == "/" or not path:
        return prefix
    return prefix + path


# --------------------------------------------------------------------------- préfixes de base

def _prefix_keys(repo_id: str) -> Tuple[str, str]:
    return "repo:%s:base_prefixes" % repo_id, "repo:%s:base_vars" % repo_id


def _load_prefixes(db: Database, repo_id: str) -> Tuple[List[str], Dict[str, str]]:
    key_p, key_v = _prefix_keys(repo_id)
    prefixes: List[str] = []
    variables: Dict[str, str] = {}
    try:
        raw = db.get_meta(key_p)
        if raw:
            value = json.loads(raw)
            if isinstance(value, list):
                prefixes = [str(v) for v in value if isinstance(v, str) and v]
    except (ValueError, TypeError):
        prefixes = []
    try:
        raw = db.get_meta(key_v)
        if raw:
            value = json.loads(raw)
            if isinstance(value, dict):
                variables = {str(k): str(v) for k, v in value.items() if isinstance(v, str)}
    except (ValueError, TypeError):
        variables = {}
    return prefixes, variables


class _PrefixLearner:
    """Accumule préfixes et variables de base d'un dépôt ; résout les variables par chaînage."""

    def __init__(self, db: Database, repo: RepoConfig) -> None:
        self.db = db
        self.repo = repo
        self.prefixes: Set[str] = set()
        self.variables: Dict[str, str] = {}
        # variable → (variable d'environnement dont elle dépend, préfixe de repli si celle-ci est inconnue)
        self._pending: Dict[str, Tuple[str, Optional[str]]] = {}
        self._file_text: Dict[int, Optional[str]] = {}
        self.errors: List[str] = []

    # ----- enregistrement -----
    def add_prefix(self, prefix: str) -> None:
        p = _clean_prefix(prefix)
        if p and len(_segments(p)) <= 4:
            self.prefixes.add(p)

    def set_var(self, name: str, prefix: str) -> None:
        if not name:
            return
        self.variables[name] = prefix
        if prefix:
            self.add_prefix(prefix)

    def resolve_rhs(self, name: str, rhs: str) -> None:
        """Interprète la partie droite d'une affectation : variable d'environnement (prioritaire,
        avec l'URL ou le chemin littéral de repli : ``env.VITE_API_URL ?? "http://…/api"``),
        sinon URL littérale, sinon chemin littéral."""
        fallback: Optional[str] = None
        m = _URL_LITERAL_RE.search(rhs)
        if m:
            fallback = _url_path(m.group(0))
        else:
            m = _PATH_LITERAL_RE.search(rhs)
            if m:
                fallback = _clean_prefix(m.group(1))
        m = _ENV_REF_RE.search(rhs)
        if m:
            env_var = m.group(1) or m.group(2)
            if env_var:
                if env_var in self.variables:
                    self.set_var(name, self.variables[env_var])
                else:
                    self._pending[name] = (env_var, fallback)
                return
        if fallback is not None:
            self.set_var(name, fallback)

    def settle(self) -> None:
        """Résout les variables définies à partir d'autres variables (``API_BASE = env.VITE_API_URL``) ;
        une variable d'environnement inconnue laisse place au littéral de repli."""
        for _ in range(3):
            progressed = False
            for name, (dep, _fallback) in list(self._pending.items()):
                if dep in self.variables:
                    self.set_var(name, self.variables[dep])
                    del self._pending[name]
                    progressed = True
            if not progressed:
                break
        for name, (_dep, fallback) in list(self._pending.items()):
            if fallback is not None:
                self.set_var(name, fallback)
            del self._pending[name]

    # ----- sources -----
    def from_env_files(self) -> None:
        root = self.repo.path
        if not root or not os.path.isdir(root):
            return
        for name in ENV_FILES:
            text = _read_small(os.path.join(root, name))
            if not text:
                continue
            for line in text.splitlines():
                m = _ENV_LINE_RE.match(line)
                if not m:
                    continue
                var, value = m.group(1), m.group(2).strip()
                if not _ENV_VAR_WANTED_RE.search(var) or _ENV_VAR_EXCLUDED_RE.search(var):
                    continue
                if value.startswith(("'", '"')):
                    value = value[1:].split(value[0], 1)[0]
                else:
                    value = value.split(" #", 1)[0].strip()
                if value.lower().startswith(("http://", "https://")):
                    self.set_var(var, _url_path(value))
                elif value.startswith("/"):
                    self.set_var(var, _clean_prefix(value))

    def from_proxy_configs(self) -> None:
        root = self.repo.path
        if not root or not os.path.isdir(root):
            return
        for name in PROXY_CONFIG_FILES:
            text = _read_small(os.path.join(root, name))
            if not text:
                continue
            if name == "package.json":
                try:
                    data = json.loads(text)
                except ValueError:
                    continue
                proxy = data.get("proxy") if isinstance(data, dict) else None
                if isinstance(proxy, str) and "://" in proxy:
                    self.add_prefix(_url_path(proxy))
                continue
            if name.startswith("proxy.conf"):
                for m in _PROXY_KEY_RE.finditer(text):
                    self.add_prefix(m.group(1))
                continue
            for m in _PROXY_BLOCK_RE.finditer(text):
                block = text[m.end():m.end() + 4000]
                for km in _PROXY_KEY_RE.finditer(block):
                    self.add_prefix(km.group(1))

    def _repo_endpoints(self) -> List[EndpointRow]:
        rows = self.db.query("SELECT e.* FROM endpoints e JOIN files f ON f.id = e.file_id WHERE f.repo_id = ?",
                             (self.repo.id,))
        return [EndpointRow.from_row(r) for r in rows]

    def _text_of_file(self, file_id: int) -> Optional[str]:
        if file_id not in self._file_text:
            try:
                self._file_text[file_id] = source_text(self.db, "file", file_id) or None
            except Exception as exc:  # noqa: BLE001
                self.errors.append("lecture des morceaux du fichier %s impossible : %s" % (file_id, exc))
                self._file_text[file_id] = None
        return self._file_text[file_id]

    def _resolve_identifier_in_file(self, name: str, file_id: Optional[int]) -> None:
        """Cherche ``name = …`` dans le texte indexé du fichier et interprète la partie droite."""
        if not name or file_id is None or name in self.variables or name in self._pending:
            return
        text = self._text_of_file(int(file_id))
        if not text:
            return
        pattern = re.compile(r"(?<![\w.$])%s\s*(?::\s*[^=\n]+?)?\s*=\s*([^;\n]+)" % re.escape(name))
        m = pattern.search(text)
        if m:
            self.resolve_rhs(name, m.group(1))

    def from_endpoints(self) -> None:
        """Bases des appels client (``meta.base``) et préfixe API des routes serveur (``meta.api_prefix``)."""
        for ep in self._repo_endpoints():
            meta = _meta(ep)
            if ep.side == "server":
                api_prefix = meta.get("api_prefix")
                if isinstance(api_prefix, str):
                    self.add_prefix(api_prefix)
                continue
            base = meta.get("base")
            if isinstance(base, str) and base.strip():
                if "://" in base:
                    self.add_prefix(_url_path(base))  # URL absolue : seul le chemin compte, jamais l'hôte
                else:
                    self._resolve_identifier_in_file(_var_name(base), ep.file_id)
            receiver = meta.get("receiver")
            if isinstance(receiver, str) and receiver not in ("fetch", "axios", "window", "this", "self", "Http"):
                self._resolve_identifier_in_file(receiver, ep.file_id)

    def from_code_search(self, limit: int = 30) -> None:
        """``axios.create({ baseURL })`` / ``Dio(BaseOptions(baseUrl:))`` via une recherche FTS ciblée."""
        try:
            hits = self.db.fts_search("baseURL", limit=limit, source_types=["file"])
        except Exception as exc:  # noqa: BLE001
            self.errors.append("recherche FTS « baseURL » impossible : %s" % exc)
            return
        seen_files: Set[int] = set()
        for hit in hits:
            if hit.source_id is None or hit.source_id in seen_files:
                continue
            seen_files.add(int(hit.source_id))
            f = self.db.get_file_by_id(int(hit.source_id))
            if f is None or f.repo_id != self.repo.id or f.id is None:
                continue
            text = self._text_of_file(f.id)
            if not text:
                continue
            self._scan_base_urls(text, f.id)

    def _scan_base_urls(self, text: str, file_id: int) -> None:
        for m in _AXIOS_RECEIVER_RE.finditer(text):
            receiver, body = m.group(1), m.group(2)
            bm = _BASEURL_RE.search(body)
            if bm:
                self._record_base(receiver, bm, file_id)
        for m in _DIO_RECEIVER_RE.finditer(text):
            receiver, body = m.group(1), m.group(2)
            bm = _BASEURL_RE.search(body)
            if bm:
                self._record_base(receiver, bm, file_id)
        for bm in _BASEURL_RE.finditer(text):
            self._record_base(None, bm, file_id)

    def _record_base(self, receiver: Optional[str], bm: "re.Match[str]", file_id: int) -> None:
        literal, ident = bm.group(1), bm.group(2)
        prefix: Optional[str] = None
        if literal:
            prefix = _url_path(literal) if "://" in literal else _clean_prefix(literal)
            self.add_prefix(prefix)
        elif ident:
            name = _var_name(ident)
            self._resolve_identifier_in_file(name, file_id)
            prefix = self.variables.get(name)
            if prefix is None and name in self._pending and receiver:
                self._pending[receiver] = self._pending[name]  # résolu plus tard, avec la variable d'environnement
        if receiver and prefix is not None:
            self.set_var(receiver, prefix)

    def from_server_conventions(self) -> None:
        """Laravel : ``routes/api.php`` → ``/api``."""
        root = self.repo.path
        on_disk = bool(root) and os.path.isfile(os.path.join(root, "routes", "api.php"))
        in_db = self.db.get_file(self.repo.id, "routes/api.php") is not None
        if on_disk or in_db:
            self.add_prefix("/api")

    # ----- bilan -----
    def store(self) -> List[str]:
        self.settle()
        key_p, key_v = _prefix_keys(self.repo.id)
        result = sorted(self.prefixes)
        with self.db.transaction():
            self.db.set_meta(key_p, json.dumps(result, ensure_ascii=False))
            self.db.set_meta(key_v, json.dumps(self.variables, ensure_ascii=False, sort_keys=True))
        return result


def learn_base_prefixes(db: Database, repo: RepoConfig) -> List[str]:
    """Préfixes d'URL de base d'un dépôt (ex. ``['/api']``), appris en lecture seule.

    Sources : ``.env`` / ``.env.example`` / ``.env.local``… (variables contenant URL / API /
    BASE / HOST, hors bases de données et services tiers : seul le *chemin* après l'hôte est
    retenu, jamais l'hôte ni une valeur secrète), ``axios.create({ baseURL })`` et
    ``Dio(BaseOptions(baseUrl:))`` dans les fichiers indexés (via ``meta.base`` des appels
    client et une recherche plein texte ciblée), la configuration de proxy (Vite, webpack,
    Angular, CRA) et, côté serveur, Laravel ``routes/api.php`` → ``/api`` ainsi que
    ``meta.api_prefix`` des routes. Le résultat est stocké dans ``meta`` sous
    ``repo:<id>:base_prefixes`` (JSON) ; les variables résolues (``VITE_API_URL`` → ``/api``,
    ``API_BASE`` → ``/api``, instance ``http`` → ``/api``) sous ``repo:<id>:base_vars``.
    """
    learner = _PrefixLearner(db, repo)
    for step in (learner.from_env_files, learner.from_proxy_configs, learner.from_endpoints,
                 learner.from_code_search, learner.from_server_conventions):
        try:
            step()
        except Exception as exc:  # noqa: BLE001 - une source illisible n'empêche pas les autres
            learner.errors.append("%s : %s" % (step.__name__, exc))
    return learner.store()


# --------------------------------------------------------------------------- résolution des imports

_JS_LANGUAGES = frozenset({"javascript", "jsx", "typescript", "tsx", "vue", "svelte"})
_JS_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".d.ts", ".json")
_JS_INDEX = tuple("/index" + e for e in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue"))
_JS_ALIASES = (("@/", "src/"), ("~/", "src/"), ("@/", ""), ("~/", ""), ("src/", "src/"))
_JS_BARE_ROOTS = ("", "src/", "packages/", "libs/", "apps/", "lib/", "modules/")
#: Sections de ``package.json`` déclarant des paquets ; les spécifications locales (``workspace:``…) sont ignorées.
_NPM_DEP_SECTIONS = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")
_NPM_LOCAL_SPECS = ("workspace:", "file:", "link:", "portal:")


class _RepoPaths:
    """Index en mémoire des chemins d'un dépôt : recherche exacte O(1), insensible à la casse en secours."""

    def __init__(self) -> None:
        self.ids: Dict[str, int] = {}
        self.lower: Dict[str, int] = {}
        self.by_name: Dict[str, List[str]] = {}

    def add(self, path: str, file_id: int) -> None:
        self.ids[path] = file_id
        self.lower.setdefault(path.lower(), file_id)
        self.by_name.setdefault(path.rsplit("/", 1)[-1], []).append(path)

    def find(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if path in self.ids:
            return path
        fid = self.lower.get(path.lower())
        if fid is None:
            return None
        for p, i in self.ids.items():  # rare : seulement en cas de casse différente
            if i == fid:
                return p
        return None

    def first(self, candidates: Iterable[str]) -> Optional[str]:
        for c in candidates:
            hit = self.find(c)
            if hit is not None:
                return hit
        return None


def _dirname(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def _join(base: str, rel: str) -> Optional[str]:
    """Jointure POSIX normalisée, relative à la racine du dépôt ; ``None`` si elle sort du dépôt."""
    joined = posixpath.normpath(posixpath.join(base, rel) if base else rel)
    if joined.startswith("../") or joined == ".." or joined.startswith("/"):
        return None
    if joined == ".":
        return ""
    return joined


def _js_targets(base: str) -> Iterable[str]:
    yield base
    for e in _JS_EXTS:
        yield base + e
    for e in _JS_INDEX:
        yield base + e
    if base.endswith((".js", ".jsx", ".mjs", ".cjs")):
        stem = base.rsplit(".", 1)[0]
        yield stem + ".ts"
        yield stem + ".tsx"


def _js_dir_targets(folder: str) -> Iterable[str]:
    """Points d'entrée d'un dossier : ``<dossier>/index.*`` puis ``<dossier>/src/index.*`` (jamais un fichier homonyme)."""
    for e in _JS_INDEX:
        yield folder + e
    for e in _JS_INDEX:
        yield folder + "/src" + e


def _package_name(module: str) -> str:
    """Nom de paquet npm d'un spécificateur nu : ``lodash/fp`` → ``lodash``, ``@acme/utils/x`` → ``@acme/utils``."""
    parts = module.split("/")
    if module.startswith("@") and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def _resolve_js(importer: str, module: str, files: _RepoPaths, packages: Set[str] = frozenset()) -> List[str]:
    if module.startswith(("./", "../")) or module in (".", ".."):
        base = _join(_dirname(importer), module)
        if base is None:
            return []
        hit = files.first(_js_targets(base))
        return [hit] if hit else []
    for alias, target in _JS_ALIASES:
        if module.startswith(alias):
            hit = files.first(_js_targets(target + module[len(alias):]))
            if hit:
                return [hit]
    if _package_name(module) in packages:
        return []  # dépendance déclarée dans package.json : jamais un fichier du dépôt (src/axios.ts ≠ axios)
    if module.startswith("@") and "/" in module:  # paquet d'un espace de noms (@acme/utils) → packages/utils
        short = module.split("/", 1)[1]
        for root in ("packages/", "libs/", "apps/", "modules/"):
            hit = files.first(list(_js_targets(root + short)) + list(_js_dir_targets(root + short)))
            if hit:
                return [hit]
        return []
    # module nu : résolu seulement si un dossier du dépôt porte le nom de son premier segment (``shared`` →
    # ``src/shared/index.ts``, ``shared/format`` → ``src/shared/format.ts``) ; un fichier homonyme (``src/axios.ts``
    # pour ``axios``) n'est jamais une résolution
    head, _sep, tail = module.partition("/")
    for root in _JS_BARE_ROOTS:
        folder = root + head
        if tail:
            hit = files.first(list(_js_targets(folder + "/" + tail)) + list(_js_dir_targets(folder + "/" + tail)))
        else:
            hit = files.first(_js_dir_targets(folder))
        if hit:
            return [hit]
    return []


def _py_targets(base: str) -> Iterable[str]:
    yield base + ".py"
    yield base + "/__init__.py"
    yield base + ".pyi"


def _resolve_python(importer: str, module: str, is_relative: bool, names: Sequence[str], files: _RepoPaths) -> List[str]:
    out: List[str] = []
    if is_relative or module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        rest = module[dots:]
        base: Optional[str] = _dirname(importer)
        for _ in range(max(0, dots - 1)):
            if not base:
                return []
            base = _dirname(base)
        target = _join(base or "", rest.replace(".", "/")) if rest else (base or "")
        if target is None:
            return []
        hit = files.first(_py_targets(target)) if target else files.find("__init__.py")
        if hit:
            out.append(hit)
        for n in names:
            sub = _join(target, n) if target else n
            hit = files.first(_py_targets(sub)) if sub is not None else None
            if hit and hit not in out:
                out.append(hit)
        return out
    rel = module.replace(".", "/")
    roots: List[str] = ["", "src/"]
    d = _dirname(importer)
    if d:
        roots.append(d + "/")
    for root in roots:
        hit = files.first(_py_targets(root + rel))
        if hit:
            out.append(hit)
            for n in names:
                sub = files.first(_py_targets(root + rel + "/" + n))
                if sub and sub not in out:
                    out.append(sub)
            return out
    return out


def _resolve_php(importer: str, module: str, files: _RepoPaths, psr4: Dict[str, List[str]]) -> List[str]:
    if module.startswith(("./", "../")) or module.endswith(".php"):
        target = _join(_dirname(importer), module)
        hit = files.find(target) if target else None
        return [hit] if hit else []
    ns = module.strip("\\")
    if not ns:
        return []
    for prefix in sorted(psr4, key=len, reverse=True):
        if ns == prefix.rstrip("\\") or ns.startswith(prefix):
            rest = ns[len(prefix):].strip("\\")
            for d in psr4[prefix]:
                base = d.strip("/").replace("\\", "/")
                cand = (base + "/" if base else "") + rest.replace("\\", "/") + ".php"
                hit = files.find(cand)
                if hit:
                    return [hit]
    parts = ns.split("\\")
    if len(parts) < 2:
        return []
    root, rest = parts[0], "/".join(parts[1:]) + ".php"
    # repli sans PSR-4 : seulement un dossier portant le nom de la racine de l'espace de noms (``App`` → ``app/``,
    # ``Acme`` → ``acme/`` ou ``src/Acme/``) ; ``src/`` nu uniquement pour l'espace de noms ``App`` du projet
    # (Symfony). Un espace de noms vendor (``Illuminate\Http\Request``) ne tombe jamais sur ``app/Http/Request.php``.
    bases = [root.lower(), root, "src/" + root]
    if root.lower() == "app":
        bases.append("src")
    for base in bases:
        hit = files.find(base + "/" + rest)
        if hit:
            return [hit]
    return []


def _resolve_java(module: str, files: _RepoPaths) -> List[str]:
    if not module or module.endswith(".*") or "." not in module:
        return []
    parts = module.split(".")
    for i in range(len(parts), 1, -1):
        cls = parts[i - 1]
        suffix = "/".join(parts[:i])
        for ext in (".java", ".kt"):
            for path in files.by_name.get(cls + ext, []):
                if path == suffix + ext or path.endswith("/" + suffix + ext):
                    return [path]
    return []


def _resolve_dart(importer: str, module: str, files: _RepoPaths) -> List[str]:
    if module.startswith("dart:"):
        return []
    if module.startswith("package:"):
        rest = module[len("package:"):]
        _pkg, _sep, path = rest.partition("/")
        hit = files.find("lib/" + path) if path else None
        return [hit] if hit else []
    target = _join(_dirname(importer), module)
    hit = files.find(target) if target else None
    return [hit] if hit else []


def _resolve_generic(importer: str, module: str, files: _RepoPaths) -> List[str]:
    if "/" not in module and "." not in module:
        return []
    target = _join(_dirname(importer), module)
    hit = files.find(target) if target else None
    if hit is None:
        hit = files.find(module.lstrip("./"))
    return [hit] if hit else []


def _composer_psr4(root: Optional[str]) -> Dict[str, List[str]]:
    """Table PSR-4 de ``composer.json`` (``autoload`` + ``autoload-dev``) ; vide si absent ou illisible."""
    if not root:
        return {}
    text = _read_small(os.path.join(root, "composer.json"))
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    out: Dict[str, List[str]] = {}
    if not isinstance(data, dict):
        return out
    for section in ("autoload", "autoload-dev"):
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        psr4 = block.get("psr-4")
        if not isinstance(psr4, dict):
            continue
        for ns, dirs in psr4.items():
            if not isinstance(ns, str):
                continue
            targets = dirs if isinstance(dirs, list) else [dirs]
            out.setdefault(ns, []).extend(str(d) for d in targets if isinstance(d, str))
    return out


def _package_dependencies(root: Optional[str]) -> Set[str]:
    """Paquets déclarés dans ``package.json`` (toutes sections de dépendances), hors paquets locaux
    (``workspace:``, ``file:``, ``link:``) ; vide si le fichier est absent ou illisible."""
    if not root:
        return set()
    text = _read_small(os.path.join(root, "package.json"))
    if not text:
        return set()
    try:
        data = json.loads(text)
    except ValueError:
        return set()
    out: Set[str] = set()
    if not isinstance(data, dict):
        return out
    for section in _NPM_DEP_SECTIONS:
        block = data.get(section)
        if not isinstance(block, dict):
            continue
        for name, spec in block.items():
            if isinstance(name, str) and name and not (isinstance(spec, str) and spec.startswith(_NPM_LOCAL_SPECS)):
                out.add(name)
    return out


def _resolve_module(importer: FileRow, module: str, names: Sequence[str], is_relative: bool,
                    files: _RepoPaths, psr4: Dict[str, List[str]], packages: Set[str] = frozenset()) -> List[str]:
    lang = (importer.language or "").lower()
    path = importer.path
    if lang in _JS_LANGUAGES:
        return _resolve_js(path, module, files, packages)
    if lang == "python":
        return _resolve_python(path, module, is_relative, names, files)
    if lang in ("php", "blade"):
        return _resolve_php(path, module, files, psr4)
    if lang in ("java", "kotlin"):
        return _resolve_java(module, files)
    if lang == "dart":
        return _resolve_dart(path, module, files)
    return _resolve_generic(path, module, files)


def resolve_imports(db: Database, workspace: WorkspaceConfig, errors: Optional[List[str]] = None) -> Tuple[int, int]:
    """Transforme les ``imports_raw`` en relations ``imports`` (``file:A`` → ``file:B``, même dépôt).

    JS/TS : relatifs avec extensions implicites et ``index.*``, alias ``@/`` → ``src/``, modules
    nus résolus seulement si un dossier du dépôt porte ce nom (``shared`` → ``src/shared/index.ts``,
    jamais un fichier homonyme ``src/axios.ts``, jamais un paquet déclaré dans ``package.json``) ;
    Python : ``a.b`` → ``a/b.py`` ou ``a/b/__init__.py`` (racine, ``src/`` ou dossier du fichier),
    relatifs ``..a`` ; PHP : PSR-4 via ``composer.json`` sinon dossier portant le nom de la racine de
    l'espace de noms (``App\\`` → ``app/`` ou ``src/``, ``Acme\\`` → ``acme/``), jamais un espace de
    noms vendor vers ``app/`` ; Java/Kotlin : ``a.b.C`` → ``…/a/b/C.java`` ; Dart : ``package:x/y.dart``
    → ``lib/y.dart``, relatifs directs. Les ``imports_raw`` sont conservés. Renvoie ``(résolus, non
    résolus)`` ; O(n) grâce à un index des chemins par dépôt.
    """
    log = _Errors(errors)
    files = _files_by_id(db)
    by_repo: Dict[str, _RepoPaths] = {}
    for f in files.values():
        by_repo.setdefault(f.repo_id, _RepoPaths()).add(f.path, int(f.id))  # type: ignore[arg-type]
    psr4_cache: Dict[str, Dict[str, List[str]]] = {}
    packages_cache: Dict[str, Set[str]] = {}

    raw_rows = db.query("SELECT src, dst, meta FROM relations WHERE kind = 'imports_raw' ORDER BY id")
    resolved = 0
    unresolved = 0
    merged: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in raw_rows:
        try:
            src, dst = r["src"] or "", r["dst"] or ""
            if not src.startswith("file:") or not dst.startswith("module:"):
                continue
            importer = files.get(int(src[len("file:"):]))
            module = dst[len("module:"):]
            if importer is None or not module:
                unresolved += 1
                continue
            try:
                meta = json.loads(r["meta"]) if r["meta"] else {}
            except (ValueError, TypeError):
                meta = {}
            if not isinstance(meta, dict):
                meta = {}
            names = [str(n) for n in (meta.get("names") or []) if isinstance(n, str)]
            is_relative = bool(meta.get("is_relative"))
            repo_files = by_repo.get(importer.repo_id)
            if repo_files is None:
                unresolved += 1
                continue
            if importer.repo_id not in psr4_cache:
                root = _repo_root(workspace, db, importer.repo_id)
                psr4_cache[importer.repo_id] = _composer_psr4(root)
                packages_cache[importer.repo_id] = _package_dependencies(root)
            targets = _resolve_module(importer, module, names, is_relative, repo_files, psr4_cache[importer.repo_id],
                                      packages_cache[importer.repo_id])
            targets = [t for t in targets if t and repo_files.ids.get(t) != importer.id]
            if not targets:
                unresolved += 1
                continue
            resolved += 1
            for t in targets:
                key = (int(importer.id), repo_files.ids[t])  # type: ignore[arg-type]
                entry = merged.setdefault(key, {"names": [], "modules": []})
                for n in names:
                    if n not in entry["names"]:
                        entry["names"].append(n)
                if module not in entry["modules"]:
                    entry["modules"].append(module)
        except Exception as exc:  # noqa: BLE001 - un import illisible n'arrête pas la passe
            unresolved += 1
            log.add("import illisible (%s → %s) : %s" % (r["src"], r["dst"], exc))

    rows = [RelationRow(src=node_key("file", a), dst=node_key("file", b), kind="imports", confidence=1.0, meta=m)
            for (a, b), m in merged.items()]
    with db.transaction():
        _delete_kinds(db, ["imports"])
        if rows:
            db.add_relations(rows)
    return resolved, unresolved


# --------------------------------------------------------------------------- gestionnaires et appels

def _short_name(name: Optional[str]) -> str:
    """``App\\Http\\Controllers\\OrderController`` ou ``a.b.OrderController`` → ``OrderController``."""
    return (name or "").split("\\")[-1].split(".")[-1].strip()


def _import_targets(db: Database) -> Dict[int, List[Tuple[int, List[str]]]]:
    """``{fichier A: [(fichier B importé, noms importés)]}`` d'après les relations ``imports``."""
    out: Dict[int, List[Tuple[int, List[str]]]] = {}
    for r in db.query("SELECT src, dst, meta FROM relations WHERE kind = 'imports'"):
        src, dst = r["src"] or "", r["dst"] or ""
        if not src.startswith("file:") or not dst.startswith("file:"):
            continue
        try:
            a, b = int(src[len("file:"):]), int(dst[len("file:"):])
            meta = json.loads(r["meta"]) if r["meta"] else {}
        except (ValueError, TypeError):
            continue
        names = [str(n) for n in (meta.get("names") or []) if isinstance(n, str)] if isinstance(meta, dict) else []
        out.setdefault(a, []).append((b, names))
    return out


def link_handlers(db: Database, workspace: WorkspaceConfig, errors: Optional[List[str]] = None) -> int:
    """Complète les relations ``handles`` (route serveur → symbole gestionnaire) entre fichiers.

    L'indexeur ne relie une route qu'à un gestionnaire défini dans le **même** fichier. Ici, pour
    chaque route serveur sans relation ``handles`` dont le gestionnaire est nommé
    (``OrderController@confirm``, ``UsersController.findAll``, ``list_users``), on cherche dans le
    même dépôt la méthode ``confirm`` d'une classe ``OrderController`` (ou la fonction de module
    ``list_users``) ; plusieurs candidats sont départagés par les imports du fichier de la route.
    Confiance 0.9 (0.6 si ambigu, trois candidats au plus), ``meta={handler, rule: 'linker',
    ambiguous}``. Les relations posées par l'indexeur (sans ``rule``) sont conservées ; celles de ce
    module (``rule == 'linker'``) sont supprimées puis recalculées à chaque passe, de sorte qu'une
    ambiguïté disparue (homonyme supprimé) ramène la confiance à 0.9 même si ni la route ni le
    gestionnaire restant n'ont changé. Idempotent. Renvoie le nombre de relations écrites.
    """
    log = _Errors(errors)
    files = _files_by_id(db)
    by_repo_name: Dict[Tuple[str, str], List[Any]] = {}
    for s in db.list_symbols(limit=_LARGE_LIMIT):
        if s.id is None or s.file_id is None or not s.name:
            continue
        f = files.get(int(s.file_id))
        if f is None:
            continue
        by_repo_name.setdefault((f.repo_id, s.name.lower()), []).append(s)
    kept: Dict[str, List[RelationRow]] = {}  # relations brutes de l'indexeur, par route
    stale: Set[str] = set()  # routes portant une relation de ce module, à recalculer
    for r in db.query("SELECT * FROM relations WHERE kind = 'handles'"):
        rel = RelationRow.from_row(r)
        if _meta(rel).get("rule") == "linker":
            stale.add(rel.src)
        else:
            kept.setdefault(rel.src, []).append(rel)
    existing = set(kept)
    imports_of: Optional[Dict[int, List[Tuple[int, List[str]]]]] = None
    rows: List[RelationRow] = []
    for ep in db.list_endpoints(side="server"):
        try:
            if ep.id is None or ep.file_id is None or not ep.handler:
                continue
            key = node_key("endpoint", ep.id)
            if key in existing:
                continue
            cls, meth = _split_handler(ep.handler)
            if not meth or meth.lower() in _ANONYMOUS_HANDLERS:
                continue
            f = files.get(int(ep.file_id))
            if f is None:
                continue
            cands = list(by_repo_name.get((f.repo_id, meth.lower()), []))
            if cls:
                cands = [s for s in cands if _short_name(s.parent).lower() == cls.lower()
                         and (s.kind or "").lower() in CALLABLE_KINDS]
            else:
                cands = [s for s in cands if not s.parent and (s.kind or "").lower() in CALLABLE_KINDS | {"class"}]
            if not cands:
                continue
            if len(cands) > 1:
                if imports_of is None:
                    imports_of = _import_targets(db)
                imported = {b for b, _names in imports_of.get(int(ep.file_id), [])}
                narrowed = [s for s in cands if int(s.file_id) in imported]
                if len(narrowed) == 1:
                    cands = narrowed
                elif len(cands) > MAX_HANDLER_CANDIDATES:
                    continue
            conf = 0.9 if len(cands) == 1 else 0.6
            for s in cands:
                rows.append(RelationRow(src=key, dst=node_key("symbol", int(s.id)), kind="handles", confidence=conf,
                                        meta={"handler": ep.handler, "rule": "linker", "ambiguous": len(cands) > 1}))
        except Exception as exc:  # noqa: BLE001 - une route illisible n'arrête pas la passe
            log.add("gestionnaire illisible (%s %s) : %s" % (ep.method, ep.path, exc))
    if stale or rows:
        with db.transaction():
            for src in sorted(stale):
                db.delete_relations_from(src, "handles")
                if src in kept:
                    db.add_relations(kept[src])  # relation brute partageant la route (rare) : remise en place
            if rows:
                db.add_relations(rows)
    return len(rows)


def _file_lines(db: Database, file_id: int) -> Dict[int, str]:
    """``{numéro de ligne: texte}`` d'un fichier indexé, reconstitué depuis ses morceaux (sans en-tête)."""
    lines: Dict[int, str] = {}
    for chunk in db.chunks_for("file", file_id):
        text = chunk.text or ""
        offset = (chunk.meta or {}).get("body_offset")
        if isinstance(offset, int) and 0 < offset <= len(text):
            text = text[offset:]
        start = chunk.line_start or 1
        for i, line in enumerate(text.splitlines()):
            lines.setdefault(start + i, line)
    return lines


def link_calls(db: Database, workspace: WorkspaceConfig, errors: Optional[List[str]] = None) -> int:
    """Dérive les relations ``calls`` (``symbol:A`` ou ``file:A`` → ``symbol:B``) du texte indexé.

    Aucun extracteur ne produit de graphe d'appels ; on repère dans le corps de chaque fonction,
    méthode ou composant (lignes des morceaux indexés, ligne de définition exclue) les noms
    suivis d'une parenthèse ouvrante, et on les confronte aux symboles appelables
    (:data:`CALLABLE_KINDS`) du même fichier et des fichiers importés (relations ``imports``) :

    * nom (ou classe parente) explicitement importé → 0.8 (``rule='import'``) ;
    * méthode de la classe courante appelée sur ``$this->`` / ``this.`` / ``self.`` / ``self::`` /
      ``static::`` → 0.7 (``'same_class'``), seule retenue même si des homonymes existent dans
      d'autres classes du fichier ou des fichiers importés (``index``, ``show``… sont partout) ;
    * symbole du même fichier → 0.7 (``'same_file'``) ;
    * fichier importé sans ce nom dans la liste d'import (``store.add``) → 0.6 (``'name'``) ;
    * plusieurs candidats (trois au plus) → 0.5 chacun (``'ambiguous'``) ; au-delà, rien.

    Le code hors de tout symbole (module, routes Express en ligne) est rattaché au fichier
    (``file:A``). Le genre ``calls`` est entièrement dérivé : supprimé puis recréé. Renvoie le
    nombre de relations écrites.
    """
    log = _Errors(errors)
    files = _files_by_id(db)
    callable_by_file: Dict[int, List[Any]] = {}
    for s in db.list_symbols(limit=_LARGE_LIMIT):
        if s.id is None or s.file_id is None or not s.name or len(s.name) < 2:
            continue
        if (s.kind or "").lower() in CALLABLE_KINDS and s.name not in _CALL_KEYWORDS:
            callable_by_file.setdefault(int(s.file_id), []).append(s)
    imports_of = _import_targets(db)
    rows: Dict[Tuple[str, str], RelationRow] = {}
    for fid, f in files.items():
        if (f.kind or "") != "code":
            continue
        callers = callable_by_file.get(fid, [])
        imported = imports_of.get(fid, [])
        if not callers and not imported:
            continue
        try:
            targets: Dict[str, List[Tuple[Any, str, float]]] = {}
            for s in callers:
                targets.setdefault(s.name, []).append((s, "same_file", 0.7))
            for b, names in imported:
                for s in callable_by_file.get(b, []):
                    explicit = s.name in names or (bool(s.parent) and _short_name(s.parent) in names)
                    targets.setdefault(s.name, []).append((s, "import" if explicit else "name", 0.8 if explicit else 0.6))
            if not targets:
                continue
            lines = _file_lines(db, fid)
            if not lines:
                continue
            owner: Dict[int, Any] = {}
            definition_lines: Set[int] = set()
            for s in sorted(callers, key=lambda x: ((x.line_end or 0) - (x.line_start or 0), x.line_start or 0)):
                if s.line_start is None:
                    continue
                end = s.line_end if s.line_end is not None and s.line_end >= s.line_start else s.line_start
                if end > s.line_start:
                    definition_lines.add(s.line_start)
                for ln in range(s.line_start, end + 1):
                    owner.setdefault(ln, s)
            file_key = node_key("file", fid)
            for ln in sorted(lines):
                text = lines[ln]
                if "(" not in text:
                    continue
                if ln in definition_lines:
                    continue
                caller = owner.get(ln)
                if caller is not None and caller.line_start == caller.line_end:
                    text = text.split("(", 1)[1] if "(" in text else ""  # fonction d'une ligne : après sa signature
                text = _DEF_RE.sub(" ", text)
                for m in _CALL_RE.finditer(text):
                    name = m.group(1)
                    if name in _CALL_KEYWORDS:
                        continue
                    cands = targets.get(name)
                    if not cands:
                        continue
                    if caller is not None:
                        cands = [c for c in cands if c[0].id != caller.id]
                        if caller.parent and _OWN_RECEIVER_RE.search(text, 0, m.start()):
                            # ``$this->index()`` : la méthode de la classe courante prime sur tout homonyme ;
                            # sans méthode locale (héritée), les autres candidats restent possibles
                            own = [(s, "same_class", conf) for s, rule, conf in cands
                                   if rule == "same_file" and s.parent == caller.parent]
                            if own:
                                cands = own
                    if not cands or len(cands) > MAX_CALL_CANDIDATES:
                        continue
                    src = node_key("symbol", int(caller.id)) if caller is not None else file_key
                    for target, rule, conf in cands:
                        if len(cands) > 1:
                            rule, conf = "ambiguous", min(conf, 0.5)
                        key = (src, node_key("symbol", int(target.id)))
                        current = rows.get(key)
                        if current is None or conf > float(current.confidence or 0.0):
                            rows[key] = RelationRow(src=src, dst=key[1], kind="calls", confidence=conf,
                                                    meta={"rule": rule, "name": name, "line": ln})
        except Exception as exc:  # noqa: BLE001 - un fichier illisible n'arrête pas la passe
            log.add("appels de %s : %s" % (f.path, exc))
    with db.transaction():
        _delete_kinds(db, ["calls"])
        if rows:
            db.add_relations(list(rows.values()))
    return len(rows)


# --------------------------------------------------------------------------- appariement des endpoints

@dataclass
class _Server:
    id: int
    path: str
    method: str
    repo_id: Optional[str]
    file_id: Optional[int]


class _ServerIndex:
    def __init__(self) -> None:
        self.by_path: Dict[str, List[_Server]] = {}
        self.by_suffix: Dict[str, List[_Server]] = {}
        self.by_shape: Dict[Tuple[int, str], List[_Server]] = {}
        self.all: List[_Server] = []

    def add(self, s: _Server) -> None:
        self.all.append(s)
        self.by_path.setdefault(s.path, []).append(s)
        segs = _segments(s.path)
        for i in range(1, len(segs) - 1):
            self.by_suffix.setdefault("/" + "/".join(segs[i:]), []).append(s)
        if segs:
            self.by_shape.setdefault((len(segs), segs[0]), []).append(s)


def _method_factor(client: str, server: str) -> float:
    c, s = (client or "").upper(), (server or "").upper()
    if c in _UNKNOWN_METHODS or s in _UNKNOWN_METHODS:
        return 0.9
    if c == s or {c, s} == {"GET", "HEAD"}:
        return 1.0
    return 0.0


def _param_absorbs(a: str, b: str) -> bool:
    """Vrai si les deux chemins ont le même nombre de segments et ne diffèrent que par des ``:param``."""
    sa, sb = _segments(a), _segments(b)
    if len(sa) != len(sb) or not sa:
        return False
    literal_equal = 0
    for x, y in zip(sa, sb):
        if x == y:
            if x != ":param":
                literal_equal += 1
            continue
        if x != ":param" and y != ":param":
            return False
    return literal_equal >= 1


def _client_paths(ep: EndpointRow, variables: Dict[str, str]) -> List[Tuple[str, float]]:
    """Chemins candidats d'un appel client : ``[(chemin normalisé, plafond de confiance)]``.

    Si la base (``meta.base`` : URL ou variable résolue, ou instance ``meta.receiver`` connue)
    apporte un préfixe, le chemin « effectif » (préfixe + chemin) est privilégié ; le chemin
    brut reste candidat avec un plafond de 0.75.
    """
    raw = normalize_route_path(ep.path or "")
    if raw == "/" or not raw:
        return []
    if "://" in (ep.path or ""):
        return [(raw, 1.0)]
    meta = _meta(ep)
    base_path: Optional[str] = None
    base = meta.get("base")
    if isinstance(base, str) and base.strip():
        if "://" in base:
            base_path = _url_path(base)
        else:
            base_path = variables.get(_var_name(base))
    if base_path is None:
        receiver = meta.get("receiver")
        if isinstance(receiver, str) and receiver:
            base_path = variables.get(receiver)
    if base_path and not raw.startswith(base_path + "/") and raw != base_path:
        eff = _join_prefix(base_path, raw)
        return [(eff, 1.0), (raw, 0.75)]
    return [(raw, 1.0)]


def _candidates_for(c: str, idx: _ServerIndex, learned: Sequence[str],
                    generic: Sequence[str]) -> Dict[int, Tuple[float, str, str]]:
    """Routes serveur candidates pour un chemin client normalisé : ``{server_id: (score, règle, préfixe)}``."""
    found: Dict[int, Tuple[float, str, str]] = {}

    def offer(servers: Sequence[_Server], score: float, rule: str, prefix: str) -> None:
        for s in servers:
            cur = found.get(s.id)
            if cur is None or score > cur[0]:
                found[s.id] = (score, rule, prefix)

    offer(idx.by_path.get(c, []), 1.0, "exact", "")
    for p in learned:
        offer(idx.by_path.get(_join_prefix(p, c), []), 1.0, "learned_prefix", p)
        if c.startswith(p + "/"):
            offer(idx.by_path.get(c[len(p):], []), 0.85, "learned_prefix", p)
    for g in generic:
        if g in learned:
            continue
        offer(idx.by_path.get(_join_prefix(g, c), []), 0.75, "generic_prefix", g)
        if c.startswith(g + "/"):
            offer(idx.by_path.get(c[len(g):], []), 0.75, "generic_prefix", g)
    segs = _segments(c)
    if len(segs) >= 2:
        for i in range(1, len(segs) - 1):  # le chemin client se termine par le chemin serveur (≥ 2 segments)
            offer(idx.by_path.get("/" + "/".join(segs[i:]), []), 0.65, "suffix", "")
        offer(idx.by_suffix.get(c, []), 0.6, "suffix", "")  # le chemin serveur se termine par le chemin client
    if not found or max(v[0] for v in found.values()) < 0.65:
        pool: List[_Server] = []
        if segs:
            pool += idx.by_shape.get((len(segs), segs[0]), [])
            if segs[0] != ":param":
                pool += idx.by_shape.get((len(segs), ":param"), [])
            else:
                pool = [s for s in idx.all if len(_segments(s.path)) == len(segs)]
        offer([s for s in pool if s.path != c and _param_absorbs(c, s.path)], 0.5, "param", "")
    return found


def match_endpoints(db: Database, workspace: WorkspaceConfig, errors: Optional[List[str]] = None) -> Tuple[int, int, int]:
    """Apparie chaque appel client (``side='client'``) aux routes serveur de tous les dépôts.

    Chemins normalisés (``normalize_route_path`` ; côté serveur ``meta.full_path`` est déjà pris
    en compte par l'indexeur). Règles, de la plus sûre à la moins sûre : chemin identique et
    méthode identique → 1.0 ; préfixe appris du dépôt (client ou serveur) + chemin client =
    chemin serveur → 1.0, l'inverse → 0.85 ; préfixe générique (``/api``, ``/api/v1``, ``/v1``,
    ``/backend``) → 0.75 ; le chemin client se termine par le chemin serveur (≥ 2 segments) →
    0.65 (inverse 0.6) ; un ``:param`` absorbant un segment littéral → 0.5. Méthode inconnue
    d'un côté → × 0.9 ; méthodes contradictoires → pas d'arête ; ``k`` candidats ex æquo →
    ÷ √k ; au plus 3 candidats de confiance ≥ 0.4 par appel. Relations
    ``calls_endpoint`` (``endpoint:<client>`` → ``endpoint:<serveur>``,
    ``meta={rule, prefix, method, client_path, server_path}``).
    Renvoie ``(appels appariés, appels sans correspondance, routes sans appelant)``.
    """
    log = _Errors(errors)
    files = _files_by_id(db)
    endpoints = db.list_endpoints()
    repo_ids = {f.repo_id for f in files.values()}
    learned: Dict[str, List[str]] = {}
    variables: Dict[str, Dict[str, str]] = {}
    for rid in repo_ids:
        p, v = _load_prefixes(db, rid)
        learned[rid] = p
        variables[rid] = v

    idx = _ServerIndex()
    clients: List[EndpointRow] = []
    for ep in endpoints:
        if ep.id is None:
            continue
        f = files.get(int(ep.file_id)) if ep.file_id is not None else None
        repo_id = f.repo_id if f else None
        if ep.side == "server":
            path = (ep.normalized_path or "").strip() or normalize_route_path(ep.path or "")
            if not path or path == "/":
                continue
            idx.add(_Server(id=int(ep.id), path=path, method=(ep.method or "").upper(), repo_id=repo_id,
                            file_id=ep.file_id))
        elif ep.side == "client":
            clients.append(ep)
    servers_by_id = {s.id: s for s in idx.all}
    # préfixes appris sur tous les dépôts (côté client : baseURL / .env ; côté serveur : Laravel /api…)
    all_learned = sorted({p for plist in learned.values() for p in plist})
    generic = tuple(g for g in GENERIC_PREFIXES if g not in all_learned)

    rows: List[RelationRow] = []
    matched_clients = 0
    servers_called: Set[int] = set()
    for ep in clients:
        try:
            f = files.get(int(ep.file_id)) if ep.file_id is not None else None
            repo_id = f.repo_id if f else None
            client_vars = variables.get(repo_id or "", {})
            best: Dict[int, Tuple[float, str, str, str]] = {}
            for c, cap in _client_paths(ep, client_vars):
                for sid, (score, rule, prefix) in _candidates_for(c, idx, all_learned, generic).items():
                    capped = min(score, cap)
                    cur = best.get(sid)
                    if cur is None or capped > cur[0]:
                        best[sid] = (capped, rule, prefix, c)
            if not best:
                continue
            scored: List[Tuple[float, int, str, str, str]] = []
            for sid, (score, rule, prefix, cpath) in best.items():
                factor = _method_factor(ep.method or "", servers_by_id[sid].method)
                if factor <= 0:
                    continue  # méthodes contradictoires (GET vs POST explicites) : pas d'arête
                scored.append((round(score * factor, 4), sid, rule, prefix, cpath))
            if not scored:
                continue
            counts: Dict[float, int] = {}
            for sc, _sid, _rule, _prefix, _cpath in scored:
                counts[sc] = counts.get(sc, 0) + 1
            final: List[Tuple[float, int, str, str, str, int]] = []
            for sc, sid, rule, prefix, cpath in scored:
                k = counts[sc]
                conf = sc / math.sqrt(k) if k > 1 else sc  # k candidats ex æquo → ÷ √k
                if conf >= MIN_ENDPOINT_CONFIDENCE:
                    final.append((round(conf, 4), sid, rule, prefix, cpath, k))
            final.sort(key=lambda t: (-t[0], t[1]))
            final = final[:MAX_ENDPOINT_CANDIDATES]
            if not final:
                continue
            matched_clients += 1
            for conf, sid, rule, prefix, cpath, k in final:
                srv = servers_by_id[sid]
                servers_called.add(sid)
                rows.append(RelationRow(src=node_key("endpoint", ep.id), dst=node_key("endpoint", sid),
                                        kind="calls_endpoint", confidence=conf,
                                        meta={"rule": rule, "prefix": prefix, "method": srv.method,
                                              "client_path": cpath, "server_path": srv.path, "tied": k}))
        except Exception as exc:  # noqa: BLE001
            log.add("appariement impossible pour l'appel %s %s : %s" % (ep.method, ep.path, exc))

    with db.transaction():
        _delete_kinds(db, ["calls_endpoint"])
        if rows:
            db.add_relations(rows)
    unmatched_clients = len(clients) - matched_clients
    unmatched_servers = len([s for s in idx.all if s.id not in servers_called])
    return matched_clients, unmatched_clients, unmatched_servers


# --------------------------------------------------------------------------- modèles

def _snake(name: str) -> str:
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name or "")
    return s.replace("-", "_").lower()


def _plural(word: str) -> str:
    if not word:
        return word
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _table_candidates(name: str) -> List[Tuple[str, float]]:
    """Tables qu'un nom de classe peut désigner : ``OrderLine`` → ``order_lines`` (0.8), ``order_line`` (0.8)…"""
    out: List[Tuple[str, float]] = []
    snake = _snake(name)
    if snake:
        out.append((_plural(snake), 0.8))
        out.append((snake, 0.8))
    stripped = name
    for suffix in _PHP_SUFFIXES:
        if name.endswith(suffix) and len(name) - len(suffix) >= 3:
            stripped = name[:-len(suffix)]
            break
    if stripped != name:
        s2 = _snake(stripped)
        out.append((_plural(s2), 0.6))
        out.append((s2, 0.6))
    return out


def _model_table(m: ModelRow) -> str:
    table = (m.table_name or "").strip().strip("`\"'[]")
    if not table and (m.kind or "") in TABLE_LIKE_KINDS:
        table = m.name
    return table.lower()


#: Familles de langages : un fichier « utilise » par son nom, de préférence, un modèle défini dans la même famille.
_LANGUAGE_FAMILIES: Tuple[Tuple[str, frozenset], ...] = (
    ("js", _JS_LANGUAGES),
    ("php", frozenset({"php", "blade"})),
    ("jvm", frozenset({"java", "kotlin"})),
)


def _language_family(language: Optional[str]) -> str:
    """``tsx`` / ``vue`` → ``js`` ; ``blade`` → ``php`` ; ``kotlin`` → ``jvm`` ; sinon le langage lui-même."""
    lang = (language or "").strip().lower()
    for family, members in _LANGUAGE_FAMILIES:
        if lang in members:
            return family
    return lang


def _name_candidates(models: Sequence[ModelRow], owner: FileRow, files: Dict[int, FileRow],
                     exclude_own: bool = True) -> List[ModelRow]:
    """Modèles homonymes qu'un fichier peut « utiliser » par leur nom.

    Seuls les modèles du **même dépôt** que le fichier sont retenus (une interface TS du frontend
    n'est jamais « utilisée » par un fichier PHP du backend parce que les noms coïncident) ; dans
    un monodépôt où frontend et backend cohabitent, les modèles définis dans la même famille de
    langage que le fichier (:data:`_LANGUAGE_FAMILIES`) l'emportent quand il y en a. Les modèles
    définis dans le fichier lui-même sont écartés sauf si ``exclude_own`` est faux (route d'un
    fichier qui définit aussi le modèle).
    """
    same_repo: List[Tuple[ModelRow, Optional[str]]] = []
    for m in models:
        model_file = files.get(int(m.file_id)) if m.file_id is not None else None
        if model_file is None or model_file.repo_id != owner.repo_id:
            continue
        if exclude_own and model_file.id == owner.id:
            continue
        same_repo.append((m, model_file.language))
    family = _language_family(owner.language)
    close = [m for m, language in same_repo if _language_family(language) == family]
    return close if close else [m for m, _language in same_repo]


def _handler_entity(cls: str) -> str:
    """Entité servie par une classe gestionnaire : ``ShipmentController`` → ``Shipment``,
    ``Api\\V1\\UsersResource`` → ``Users`` ; le nom lui-même s'il ne porte aucun suffixe connu."""
    name = _short_name(cls)
    for suffix in _HANDLER_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)]
    return name


def _split_handler(handler: str) -> Tuple[Optional[str], str]:
    h = (handler or "").strip().strip("'\"")
    for sep in ("@", "::", "#"):
        if sep in h:
            cls, meth = h.rsplit(sep, 1)
            return cls.split("\\")[-1].split(".")[-1] or None, meth
    if "." in h and not h.endswith("."):
        cls, meth = h.rsplit(".", 1)
        return cls.split("\\")[-1].split(".")[-1] or None, meth
    return None, h.split("\\")[-1]


def link_models(db: Database, workspace: WorkspaceConfig, errors: Optional[List[str]] = None) -> Tuple[int, int]:
    """Relie les modèles de données entre eux et aux fichiers qui les utilisent.

    (a) ``same_entity`` (``model:X`` → ``model:Y``, X < Y) : même ``table_name`` (eloquent ↔
    migration ↔ sql_table) → 0.9 ; nom de classe ↔ table par snake_case pluriel (``OrderLine`` ↔
    ``order_lines``) → 0.8 (0.6 après retrait d'un suffixe ``Model``/``Dto``/``Row``…) ;
    interface ou type TS (ou toute classe) de même nom qu'un modèle serveur d'un autre genre →
    0.7. (b) ``uses_model`` (``file:A`` → ``model:M``, 0.6) : A importe le fichier définissant M
    (relations ``imports``, en respectant les noms importés, ``rule='import'``) ; un modèle
    défini dans A déclare une relation vers M (``hasOne(Shipment::class)``, ``'relation'``) ; ou
    le nom **exact** de M (≥ 4 caractères, mot entier délimité par ``[A-Za-z0-9_$]``, sensible
    à la casse : ``Shipment`` ne correspond ni à ``shipments``, ni à ``ShipmentController``, ni
    à la variable ``$shipment``) apparaît dans un symbole, une signature ou un gestionnaire
    d'endpoint de A (``'name'``). (c) ``uses_model`` (``endpoint:E`` → ``model:M``, 0.6,
    ``rule='handler'``) : la route serveur E est gérée par une classe nommée d'après M
    (``<M>Controller``, ``<M>sResource``… : ``meta.via == 'name'``) ou par une classe dont le
    fichier, du même dépôt, importe M (``meta.via == 'import'``) ; la relation porte sur la route
    elle-même, jamais sur le fichier de routes (qui en déclare des dizaines pour d'autres
    modèles). La règle du nom reste dans le **dépôt de M** : les modèles homonymes d'autres
    dépôts (``Order`` eloquent du backend ↔ interface TS ``Order`` du frontend) sont déjà reliés
    par ``same_entity`` et ne sont jamais « utilisés » par simple coïncidence de nom ; un fichier
    n'utilise pas non plus un homonyme d'un modèle qu'il définit lui-même (son propre nom
    explique la correspondance), et dans un monodépôt les homonymes de la même famille de
    langage l'emportent. Renvoie ``(same_entity, uses_model)``.
    """
    log = _Errors(errors)
    models = [m for m in db.list_models() if m.id is not None]
    files = _files_by_id(db)
    pairs: Dict[Tuple[int, int], Tuple[float, str, str]] = {}

    def propose(a: ModelRow, b: ModelRow, conf: float, rule: str, detail: str = "") -> None:
        if a.id is None or b.id is None or a.id == b.id:
            return
        key = (min(a.id, b.id), max(a.id, b.id))
        cur = pairs.get(key)
        if cur is None or conf > cur[0]:
            pairs[key] = (conf, rule, detail)

    try:
        by_table: Dict[str, List[ModelRow]] = {}
        by_name: Dict[str, List[ModelRow]] = {}
        for m in models:
            t = _model_table(m)
            if t:
                by_table.setdefault(t, []).append(m)
            by_name.setdefault(m.name.lower(), []).append(m)
        for table, group in by_table.items():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    propose(group[i], group[j], 0.9, "same_table", table)
        for m in models:
            if (m.kind or "") in TABLE_LIKE_KINDS:
                continue
            seen_tables: Set[str] = set()
            for table, conf in _table_candidates(m.name):
                if table in seen_tables:
                    continue
                seen_tables.add(table)
                for other in by_table.get(table, []):
                    if other.id != m.id:
                        propose(m, other, conf, "name_to_table", table)
        for name, group in by_name.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if (a.kind or "") == (b.kind or "") and (a.kind or "") not in TABLE_LIKE_KINDS:
                        continue  # deux interfaces TS homonymes : pas une preuve de même entité
                    propose(a, b, 0.7, "same_name", name)
    except Exception as exc:  # noqa: BLE001
        log.add("liaison same_entity interrompue : %s" % exc)

    same_rows = [RelationRow(src=node_key("model", a), dst=node_key("model", b), kind="same_entity", confidence=conf,
                             meta={"rule": rule, "detail": detail}) for (a, b), (conf, rule, detail) in pairs.items()]

    # ----- uses_model -----
    uses: Dict[Tuple[int, int], str] = {}  # (fichier, modèle) → règle
    endpoint_uses: Dict[Tuple[int, int], Dict[str, Any]] = {}  # (route serveur, modèle) → meta
    try:
        models_by_file: Dict[int, List[ModelRow]] = {}
        class_like: List[ModelRow] = []
        for m in models:
            if m.file_id is not None:
                models_by_file.setdefault(int(m.file_id), []).append(m)
            if (m.kind or "") not in TABLE_LIKE_KINDS and len(m.name) >= _MIN_MODEL_NAME:
                class_like.append(m)
        imports: Dict[int, List[Tuple[int, List[str]]]] = {}
        for r in db.query("SELECT src, dst, meta FROM relations WHERE kind = 'imports'"):
            try:
                a = int((r["src"] or "")[len("file:"):])
                b = int((r["dst"] or "")[len("file:"):])
            except ValueError:
                continue
            try:
                meta = json.loads(r["meta"]) if r["meta"] else {}
            except (ValueError, TypeError):
                meta = {}
            names = [str(n) for n in (meta.get("names") or [])] if isinstance(meta, dict) else []
            imports.setdefault(a, []).append((b, names))
        for a, targets in imports.items():
            for b, names in targets:
                for m in models_by_file.get(b, []):
                    if names and m.name not in names:
                        continue
                    uses.setdefault((a, int(m.id)), "import")  # type: ignore[arg-type]

        by_exact: Dict[str, List[ModelRow]] = {}
        for m in class_like:
            by_exact.setdefault(m.name, []).append(m)
        # relation déclarée par un modèle (``hasOne(Shipment::class)``) : son fichier utilise le modèle cible
        for m in models:
            owner = files.get(int(m.file_id)) if m.file_id is not None else None
            if owner is None:
                continue
            for rel in (m.relations or []):
                target = _short_name(rel.get("target")) if isinstance(rel, dict) else ""
                if not target or target == m.name:
                    continue
                for t in _name_candidates(by_exact.get(target, []), owner, files):
                    uses.setdefault((int(m.file_id), int(t.id)), "relation")  # type: ignore[arg-type]

        blobs: Dict[int, List[str]] = {}
        class_files: Dict[str, List[int]] = {}
        symbol_files: Dict[int, int] = {}
        for r in db.query("SELECT id, file_id, name, kind, parent, signature FROM symbols"):
            fid = r["file_id"]
            if fid is None:
                continue
            if r["id"] is not None:
                symbol_files[int(r["id"])] = int(fid)
            blobs.setdefault(int(fid), []).extend(x for x in (r["name"], r["parent"], r["signature"]) if x)
            if (r["kind"] or "") in ("class", "interface", "trait", "enum") and r["name"]:
                class_files.setdefault(str(r["name"]).split("\\")[-1], []).append(int(fid))
        endpoints_by_file: Dict[int, List[EndpointRow]] = {}
        for ep in db.list_endpoints():
            if ep.file_id is None:
                continue
            endpoints_by_file.setdefault(int(ep.file_id), []).append(ep)
            if ep.handler:
                blobs.setdefault(int(ep.file_id), []).append(ep.handler)
        if class_like:
            names_sorted = sorted(by_exact, key=len, reverse=True)
            # nom exact, mot entier : ni ``shipments``, ni ``ShipmentController``, ni la variable ``$shipment``
            pattern = re.compile(r"(?<![A-Za-z0-9_$])(%s)(?![A-Za-z0-9_])" % "|".join(re.escape(n) for n in names_sorted))
            for fid, parts in blobs.items():
                owner = files.get(fid)
                if owner is None:
                    continue
                own_names = {m.name.lower() for m in models_by_file.get(fid, [])}
                text = " ".join(parts)
                for found in set(pattern.findall(text)):
                    if found.lower() in own_names:
                        continue  # le fichier définit un modèle de ce nom : la correspondance n'est pas un usage
                    for m in _name_candidates(by_exact.get(found, []), owner, files):
                        uses.setdefault((fid, int(m.id)), "name")  # type: ignore[arg-type]
        # route serveur gérée par une classe nommée d'après le modèle (``ShipmentController``) ou par une classe
        # dont le fichier (même dépôt) importe le modèle : la relation porte sur la route, jamais sur le fichier de
        # routes, qui déclare des dizaines de routes d'autres modèles
        uses_by_file: Dict[int, Dict[int, str]] = {}
        for (fid, mid), rule in uses.items():
            uses_by_file.setdefault(fid, {})[mid] = rule
        handled_files: Dict[int, Set[int]] = {}  # route → fichiers de ses symboles gestionnaires (``handles``)
        for r in db.query("SELECT src, dst FROM relations WHERE kind = 'handles'"):
            src, dst = r["src"] or "", r["dst"] or ""
            if not src.startswith("endpoint:") or not dst.startswith("symbol:"):
                continue
            try:
                eid, sid = int(src[len("endpoint:"):]), int(dst[len("symbol:"):])
            except ValueError:
                continue
            if sid in symbol_files:
                handled_files.setdefault(eid, set()).add(symbol_files[sid])
        by_entity: Dict[str, List[ModelRow]] = {}  # nom ou pluriel du modèle, en minuscules
        for m in class_like:
            by_entity.setdefault(m.name.lower(), []).append(m)
            by_entity.setdefault(_plural(m.name).lower(), []).append(m)
        for fid, eps in endpoints_by_file.items():
            owner = files.get(fid)
            if owner is None:
                continue
            for ep in eps:
                if ep.side != "server" or not ep.handler or ep.id is None:
                    continue
                cls, _meth = _split_handler(ep.handler)
                if not cls:
                    continue
                found_models: Dict[int, str] = {}
                entity = _handler_entity(cls)
                if entity:
                    for m in _name_candidates(by_entity.get(entity.lower(), []), owner, files, exclude_own=False):
                        found_models.setdefault(int(m.id), "name")  # type: ignore[arg-type]
                for gid in set(class_files.get(cls, [])) | handled_files.get(int(ep.id), set()):
                    handler_file = files.get(gid)
                    if handler_file is None or handler_file.repo_id != owner.repo_id:
                        continue
                    for mid, rule in uses_by_file.get(gid, {}).items():
                        if rule == "import":
                            found_models.setdefault(mid, "import")
                for mid, via in found_models.items():
                    endpoint_uses[(int(ep.id), mid)] = {"rule": "handler", "handler": ep.handler, "via": via}
    except Exception as exc:  # noqa: BLE001
        log.add("liaison uses_model interrompue : %s" % exc)

    use_rows = [RelationRow(src=node_key("file", a), dst=node_key("model", b), kind="uses_model", confidence=0.6,
                            meta={"rule": rule}) for (a, b), rule in uses.items()]
    use_rows += [RelationRow(src=node_key("endpoint", e), dst=node_key("model", b), kind="uses_model", confidence=0.6,
                             meta=meta) for (e, b), meta in endpoint_uses.items()]
    with db.transaction():
        _delete_kinds(db, ["same_entity", "uses_model"])
        if same_rows:
            db.add_relations(same_rows)
        if use_rows:
            db.add_relations(use_rows)
    return len(same_rows), len(use_rows)


# --------------------------------------------------------------------------- documents et notes

def _is_documentary_file(path: str, language: Optional[str] = None) -> bool:
    """Vrai si un fichier ``kind == 'doc'`` d'un dépôt documente le projet (cible légitime de ``documented_by``).

    * texte rédigé (Markdown, reStructuredText, ``.txt``) : partout (README, ``docs/*.md``, CHANGELOG…),
      sauf les fichiers destinés aux machines (:data:`NON_DOCUMENTARY_FILES` : ``robots.txt``…) ;
    * PDF / DOCX (:data:`TEXTLESS_LANGUAGES`) : seulement à la racine du dépôt ou sous un dossier de
      documentation (:data:`DOCUMENTATION_DIR_NAMES`). Ailleurs (``public/``, ``storage/``, ``exports/``…)
      ce sont des contenus produits ou déposés — attestations générées avec des noms réels, pages web
      enregistrées avec des codes de récupération — qui citent des modèles sans les documenter : reliés
      comme « documentation », ils remonteraient dans le contexte de chaque question sur ces modèles.
    """
    norm = (path or "").replace("\\", "/").strip("/").lower()
    if not norm:
        return False
    segments = norm.split("/")
    if segments[-1] in NON_DOCUMENTARY_FILES:
        return False
    if (language or "").lower() in TEXTLESS_LANGUAGES:
        return len(segments) == 1 or any(seg in DOCUMENTATION_DIR_NAMES for seg in segments[:-1])
    return True


def _documented_nodes(index: NameIndex, mention: Mention) -> List[Tuple[str, str, Optional[int]]]:
    """Noeuds qu'une mention documente vraiment.

    Un mot ordinaire qui nomme une table (« customers ») désigne la table, pas chacune des migrations
    ou DDL qui la définissent (:attr:`NameIndex.definition_models`) ; une forme code (``customers`` en
    accents graves) garde ses modèles. Un nom porté par plus de :data:`MAX_HOMONYMS` noeuds d'un même
    genre (``created_at`` dans des dizaines de modèles, ``map``) est trop ambigu pour en documenter un.
    """
    nodes = list(mention.nodes)
    if not mention.strong and any(kind == "table" for kind, _key, _fid in nodes):
        nodes = [n for n in nodes
                 if not (n[0] == "model" and n[1].startswith("model:") and n[1][len("model:"):].isdigit()
                         and int(n[1][len("model:"):]) in index.definition_models)]
    per_kind: Dict[str, int] = {}
    for kind, _key, _fid in nodes:
        per_kind[kind] = per_kind.get(kind, 0) + 1
    return [n for n in nodes if per_kind[n[0]] <= MAX_HOMONYMS]


def _document_relations(index: NameIndex, source_type: str, source_id: int, text: str) -> List[RelationRow]:
    """Relations ``documented_by`` et ``mentions`` d'un texte (document, note ou fichier de documentation).

    Rien si le texte ne cite aucun identifiant sous forme code présent dans l'index (fiche, formulaire,
    PDF générique, ``robots.txt``) ; un mot ordinaire ne compte qu'à partir de :data:`MIN_WEAK_MENTIONS`
    occurrences ; voir :func:`_documented_nodes` pour les noeuds retenus par mention.
    """
    dst = node_key(source_type, source_id)
    self_file = int(source_id) if source_type == "file" else None
    mentions = match_candidates(index, scan_text(text), limit=MAX_MENTIONS)
    if not any(m.exists and m.strong for m in mentions):
        return []  # aucun identifiant sous forme code : ce texte ne documente pas le code
    rows: List[RelationRow] = []
    seen: Set[str] = set()
    files_hit: Dict[int, Dict[str, Any]] = {}

    def hit_file(file_id: Optional[int], name: str, count: int, direct: bool = False) -> None:
        if file_id is None or int(file_id) == self_file:
            return  # un document de dépôt ne se mentionne pas lui-même
        entry = files_hit.setdefault(int(file_id), {"names": [], "count": 0, "direct": False})
        entry["direct"] = entry["direct"] or direct
        entry["count"] += count
        if name not in entry["names"]:
            entry["names"].append(name)

    for m in mentions:
        if not m.exists:
            continue
        if not m.strong and m.count < MIN_WEAK_MENTIONS:
            continue  # « User » dans « User-agent », « customers » une fois dans une fiche : pas une documentation
        for kind, key, file_id in _documented_nodes(index, m):
            if kind == "file":
                hit_file(int(key[len("file:"):]), m.name, m.count, direct=True)
                continue
            if kind == "field":
                hit_file(file_id, m.name, m.count)
                continue
            if key in seen or (file_id is not None and int(file_id) == self_file):
                continue
            seen.add(key)
            rows.append(RelationRow(src=key, dst=dst, kind="documented_by", confidence=0.9,
                                    meta={"mentions": m.count, "name": m.name, "kind": kind, "shape": m.shape}))
            hit_file(file_id, m.name, m.count)
    for fid, entry in files_hit.items():
        rows.append(RelationRow(src=dst, dst=node_key("file", fid), kind="mentions",
                                confidence=0.9 if entry["direct"] else 0.7,
                                meta={"names": entry["names"][:20], "count": entry["count"]}))
    return rows


def link_documents(db: Database, workspace: WorkspaceConfig, errors: Optional[List[str]] = None) -> int:
    """Relie documents et notes aux éléments de code qu'ils citent.

    Pour chaque document, chaque note et chaque fichier de documentation d'un dépôt
    (``kind == 'doc'`` : README, ``docs/*.md``…) : identifiants candidats du texte (voir
    :mod:`develop.knowledge_graph.mentions`), correspondance exacte insensible à la casse avec
    les symboles, modèles, tables, endpoints et fichiers de l'index → ``documented_by``
    (``symbol:|model:|endpoint:|table:`` → ``document:|note:|file:``, 0.9, ``meta={mentions: n, name,
    kind, shape}``) et ``mentions`` (``document:|note:|file:`` → ``file:``, 0.7, ou 0.9 si le chemin du
    fichier est cité). Au plus 200 mentions par document ; l'index des noms est construit une fois.

    Garde-fous contre le bruit (voir :func:`_document_relations`) : un texte sans identifiant sous
    forme code ne crée rien ; un mot ordinaire doit être répété (:data:`MIN_WEAK_MENTIONS`) et ne
    relie que la table ; un nom trop partagé (:data:`MAX_HOMONYMS`) ne relie rien ; les fichiers
    machine (:data:`NON_DOCUMENTARY_FILES`) et les PDF / DOCX hors des dossiers de documentation
    (:data:`DOCUMENTATION_DIR_NAMES`) sont ignorés (voir :func:`_is_documentary_file`).
    Renvoie le nombre de documents, notes et fichiers de documentation reliés à au moins un élément.
    """
    log = _Errors(errors)
    index = NameIndex.build(db)
    rows: List[RelationRow] = []
    linked = 0
    sources: List[Tuple[str, int]] = []
    sources += [("document", int(d.id)) for d in db.list_documents(limit=_LARGE_LIMIT) if d.id is not None]
    sources += [("note", int(n.id)) for n in db.list_notes(limit=_LARGE_LIMIT) if n.id is not None]
    sources += [("file", int(f.id)) for f in db.list_files(kind="doc")
                if f.id is not None and _is_documentary_file(f.path or "", f.language)]
    for source_type, source_id in sources:
        try:
            text = source_text(db, source_type, source_id)
            if not text.strip():
                continue
            found = _document_relations(index, source_type, source_id, text)
            if any(r.kind == "documented_by" for r in found):
                linked += 1
            rows.extend(found)
        except Exception as exc:  # noqa: BLE001
            log.add("%s:%s : liaison impossible : %s" % (source_type, source_id, exc))
    with db.transaction():
        _delete_kinds(db, ["documented_by", "mentions"])
        if rows:
            db.add_relations(rows)
    return linked


# --------------------------------------------------------------------------- orchestration

def link_all(db: Database, workspace: WorkspaceConfig,
             progress: Optional[Callable[[str], None]] = None) -> LinkStats:
    """Recalcule tout le graphe dérivé : préfixes par dépôt → imports → gestionnaires → appels →
    endpoints → modèles → documents.

    Chaque étape écrit en transaction groupée et consigne ses erreurs dans
    :attr:`LinkStats.errors` sans interrompre les suivantes. ``progress`` reçoit des messages
    courts en français. Idempotent : deux exécutions successives donnent le même graphe.
    """
    started = time.time()
    stats = LinkStats()
    log = _Errors(stats.errors)

    def say(message: str) -> None:
        if progress is None:
            return
        try:
            progress(message)
        except Exception:  # noqa: BLE001 - un rappel d'interface ne casse jamais la passe
            pass

    say("Apprentissage des préfixes d'URL…")
    for repo in workspace.repos:
        if not repo.enabled:
            continue
        try:
            stats.base_prefixes[repo.id] = learn_base_prefixes(db, repo)
        except Exception as exc:  # noqa: BLE001
            log.add("préfixes du dépôt %s : %s" % (repo.name or repo.id, exc))
    say("Résolution des imports…")
    try:
        stats.imports_resolved, stats.imports_unresolved = resolve_imports(db, workspace, stats.errors)
    except Exception as exc:  # noqa: BLE001
        log.add("résolution des imports : %s" % exc)
    say("Liaison des gestionnaires de routes…")
    try:
        stats.handlers_linked = link_handlers(db, workspace, stats.errors)
    except Exception as exc:  # noqa: BLE001
        log.add("liaison des gestionnaires : %s" % exc)
    say("Dérivation des appels entre symboles…")
    try:
        stats.calls_linked = link_calls(db, workspace, stats.errors)
    except Exception as exc:  # noqa: BLE001
        log.add("dérivation des appels : %s" % exc)
    say("Appariement des appels HTTP et des routes…")
    try:
        stats.endpoints_matched, stats.endpoints_unmatched_client, stats.endpoints_unmatched_server = \
            match_endpoints(db, workspace, stats.errors)
    except Exception as exc:  # noqa: BLE001
        log.add("appariement des endpoints : %s" % exc)
    say("Liaison des modèles de données…")
    try:
        stats.models_linked, stats.model_uses = link_models(db, workspace, stats.errors)
    except Exception as exc:  # noqa: BLE001
        log.add("liaison des modèles : %s" % exc)
    say("Liaison des documents et des notes…")
    try:
        stats.documents_linked = link_documents(db, workspace, stats.errors)
    except Exception as exc:  # noqa: BLE001
        log.add("liaison des documents : %s" % exc)
    try:
        stats.total = _count_kinds(db, DERIVED_KINDS)
    except Exception as exc:  # noqa: BLE001
        log.add("comptage des relations : %s" % exc)
    stats.duration_s = round(time.time() - started, 3)
    say("Graphe de relations à jour (%d relations dérivées)." % stats.total)
    return stats


# --------------------------------------------------------------------------- explorateur

def _batches(items: Sequence[Any], size: int = 400) -> Iterable[Sequence[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _endpoint_label(method: Optional[str], path: Optional[str], normalized: Optional[str], side: Optional[str]) -> str:
    m = (method or "").upper()
    p = (normalized or "").strip() or normalize_route_path(path or "")
    label = ("%s %s" % (m, p)).strip() if m and m not in _UNKNOWN_METHODS else p
    if side == "client":
        return "appel " + label
    return label


def _node_details(db: Database, keys: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Libellés et positions d'un lot de noeuds, par type, en quelques requêtes en lot."""
    out: Dict[str, Dict[str, Any]] = {}
    by_type: Dict[str, List[int]] = {}
    for key in keys:
        if ":" not in key:
            continue
        prefix, ident = key.split(":", 1)
        if prefix == "table":
            out[key] = {"key": key, "type": "table", "label": "table %s" % ident, "file": None, "path": None,
                        "line": None, "repo_id": None}
            continue
        if prefix == "module":
            out[key] = {"key": key, "type": "module", "label": ident, "file": None, "path": None, "line": None,
                        "repo_id": None}
            continue
        try:
            by_type.setdefault(prefix, []).append(int(ident))
        except ValueError:
            continue
    queries = {
        "file": "SELECT id, repo_id, path FROM files WHERE id IN (%s)",
        "symbol": "SELECT s.id, s.name, s.parent, s.kind, s.file_id, s.line_start, f.path, f.repo_id FROM symbols s "
                  "LEFT JOIN files f ON f.id = s.file_id WHERE s.id IN (%s)",
        "endpoint": "SELECT e.id, e.side, e.method, e.path, e.normalized_path, e.handler, e.file_id, e.line, f.path AS fpath, "
                    "f.repo_id FROM endpoints e LEFT JOIN files f ON f.id = e.file_id WHERE e.id IN (%s)",
        "model": "SELECT m.id, m.name, m.kind, m.table_name, m.file_id, m.line, f.path, f.repo_id FROM data_models m "
                 "LEFT JOIN files f ON f.id = m.file_id WHERE m.id IN (%s)",
        "document": "SELECT id, title, source_path, stored_path, format FROM documents WHERE id IN (%s)",
        "note": "SELECT id, title, note_date FROM notes WHERE id IN (%s)",
    }
    for prefix, ids in by_type.items():
        sql = queries.get(prefix)
        if sql is None:
            continue
        for batch in _batches(sorted(set(ids))):
            rows = db.query(sql % ",".join("?" * len(batch)), list(batch))
            for r in rows:
                key = node_key(prefix, int(r["id"]))
                if prefix == "file":
                    out[key] = {"key": key, "type": "file", "label": r["path"], "file": int(r["id"]), "path": r["path"],
                                "line": None, "repo_id": r["repo_id"]}
                elif prefix == "symbol":
                    parent = (r["parent"] or "").split("\\")[-1]
                    label = "%s.%s" % (parent, r["name"]) if parent else r["name"]
                    out[key] = {"key": key, "type": "symbol", "label": label, "file": r["file_id"], "path": r["path"],
                                "line": r["line_start"], "repo_id": r["repo_id"], "kind": r["kind"]}
                elif prefix == "endpoint":
                    out[key] = {"key": key, "type": "endpoint",
                                "label": _endpoint_label(r["method"], r["path"], r["normalized_path"], r["side"]),
                                "file": r["file_id"], "path": r["fpath"], "line": r["line"], "repo_id": r["repo_id"],
                                "side": r["side"], "method": r["method"], "handler": r["handler"]}
                elif prefix == "model":
                    out[key] = {"key": key, "type": "model", "label": r["name"], "file": r["file_id"], "path": r["path"],
                                "line": r["line"], "repo_id": r["repo_id"], "kind": r["kind"],
                                "table_name": r["table_name"]}
                elif prefix == "document":
                    title = r["title"] or r["source_path"] or ("document %s" % r["id"])
                    out[key] = {"key": key, "type": "document", "label": title, "file": None,
                                "path": r["source_path"] or r["stored_path"], "line": None, "repo_id": None}
                elif prefix == "note":
                    title = r["title"] or ("note %s" % r["id"])
                    if r["note_date"]:
                        title = "%s (%s)" % (title, r["note_date"])
                    out[key] = {"key": key, "type": "note", "label": title, "file": None, "path": None, "line": None,
                                "repo_id": None}
    return out


def neighbors_graph(db: Database, node: str, depth: int = 1, limit: int = 200) -> Dict[str, Any]:
    """Sous-graphe autour d'un noeud pour l'explorateur : parcours en largeur borné.

    ``{nodes: [{key, type, label, file, path, line, repo_id}], edges: [{src, dst, kind, confidence}]}``.
    Les libellés sont lisibles (nom qualifié du symbole, ``GET /orders/:param``, nom du modèle,
    titre du document). Les relations brutes ``imports_raw`` (vers des modules non résolus) sont
    ignorées ; ``limit`` borne le nombre de noeuds, ``depth`` le nombre de sauts (0 à 4).
    """
    start = (node or "").strip()
    result: Dict[str, Any] = {"node": start, "depth": 0, "nodes": [], "edges": [], "truncated": False}
    if not start or ":" not in start:
        return result
    depth = max(0, min(int(depth), 4))
    limit = max(1, int(limit))
    result["depth"] = depth
    order: List[str] = [start]
    seen: Set[str] = {start}
    frontier: List[str] = [start]
    truncated = False
    for _ in range(depth):
        if not frontier or len(order) >= limit:
            break
        discovered: List[str] = []
        for batch in _batches(frontier):
            ph = ",".join("?" * len(batch))
            rows = db.query("SELECT src, dst FROM relations WHERE (src IN (%s) OR dst IN (%s)) "
                            "AND kind != 'imports_raw' ORDER BY kind, src, dst" % (ph, ph), list(batch) + list(batch))
            for r in rows:
                for other in (r["src"], r["dst"]):
                    if other in seen:
                        continue
                    if len(order) + len(discovered) >= limit:
                        truncated = True
                        continue
                    seen.add(other)
                    discovered.append(other)
        order.extend(discovered)
        frontier = discovered
    details = _node_details(db, order)
    nodes: List[Dict[str, Any]] = []
    kept: Set[str] = set()
    for key in order:
        info = details.get(key)
        if info is None:
            continue  # noeud orphelin (fichier supprimé entre-temps)
        kept.add(key)
        nodes.append(info)
    # toutes les arêtes entre les noeuds conservés (y compris entre voisins du dernier niveau)
    edges: Dict[Tuple[str, str, str], float] = {}
    kept_list = sorted(kept)
    for batch in _batches(kept_list):
        ph = ",".join("?" * len(batch))
        rows = db.query("SELECT src, dst, kind, confidence FROM relations WHERE src IN (%s) AND kind != 'imports_raw'"
                        % ph, list(batch))
        for r in rows:
            if r["dst"] in kept:
                edges[(r["src"], r["dst"], r["kind"])] = float(r["confidence"] if r["confidence"] is not None else 1.0)
    result["nodes"] = nodes
    result["edges"] = [{"src": s, "dst": d, "kind": k, "confidence": round(c, 4)} for (s, d, k), c in sorted(edges.items())]
    result["truncated"] = truncated
    return result


# --------------------------------------------------------------------------- descriptions

def _what(info: Optional[Dict[str, Any]], key: str) -> str:
    """« la table orders », « le modèle Order », « le symbole OrderService », « la route GET /orders »…"""
    if info is None:
        prefix = key.split(":", 1)[0]
        return {"table": "la table", "model": "le modèle", "symbol": "le symbole", "endpoint": "la route",
                "file": "le fichier", "document": "le document", "note": "la note"}.get(prefix, "l'élément") + " " + key
    t = info["type"]
    label = info["label"]
    if t == "table":
        return "la table %s" % label[len("table "):] if label.startswith("table ") else "la table %s" % label
    if t == "model":
        return "le modèle %s" % label
    if t == "symbol":
        return "le symbole %s" % label
    if t == "endpoint":
        return ("l'appel %s" % label[len("appel "):]) if label.startswith("appel ") else "la route %s" % label
    if t == "file":
        return "le fichier %s" % label
    if t == "document":
        return "le document « %s »" % label
    if t == "note":
        return "la note « %s »" % label
    return label


def _format_confidence(conf: float) -> str:
    """``0.85`` → ``0,85`` ; ``0.9`` → ``0,9`` (virgule décimale, sans zéros inutiles)."""
    text = ("%.2f" % conf).rstrip("0").rstrip(".")
    return (text or "0").replace(".", ",")


def describe_relation(rel: RelationRow, db: Database) -> str:
    """Phrase française courte décrivant une relation, pour l'interface et l'analyse d'impact.

    Exemples : « importe src/api/client.ts », « appelle POST /orders/:param/confirm (0,85) »,
    « documente la table orders », « même entité que orders (0,9) », « utilise le modèle Order ».
    Le sujet est la source de la relation, sauf pour ``documented_by`` (le sujet est le
    document) et ``handles`` (la route est gérée par le symbole). La confiance n'est affichée
    que si elle est inférieure à 1.
    """
    kind = rel.kind or ""
    details = _node_details(db, [rel.src, rel.dst])
    src_info, dst_info = details.get(rel.src), details.get(rel.dst)
    conf = float(rel.confidence) if rel.confidence is not None else 1.0
    suffix = " (%s)" % _format_confidence(conf) if conf < 0.999 else ""
    dst_label = dst_info["label"] if dst_info else rel.dst
    src_label = src_info["label"] if src_info else rel.src
    if kind == "imports":
        return "importe %s" % dst_label
    if kind == "imports_raw":
        module = rel.dst[len("module:"):] if rel.dst.startswith("module:") else rel.dst
        return "importe le module %s" % module
    if kind == "calls_endpoint":
        label = dst_label[len("appel "):] if dst_label.startswith("appel ") else dst_label
        return "appelle %s%s" % (label, suffix)
    if kind == "handles":
        return "route gérée par %s" % dst_label
    if kind == "defines_table":
        return "définit %s" % _what(dst_info, rel.dst)
    if kind == "same_entity":
        return "même entité que %s%s" % (dst_label, suffix)
    if kind == "uses_model":
        return "utilise %s%s" % (_what(dst_info, rel.dst), suffix)
    if kind == "documented_by":
        return "documente %s" % _what(src_info, rel.src)
    if kind == "mentions":
        return "mentionne %s%s" % (_what(dst_info, rel.dst), suffix)
    if kind == "calls":
        return "appelle %s%s" % (dst_label, suffix)
    return "%s → %s%s" % (kind or "relation", dst_label if dst_info else src_label, suffix)
