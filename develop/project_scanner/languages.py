"""Détection du langage, du type de fichier et des exclusions par défaut.

Tout repose sur le nom du fichier (extension ou nom spécial) : aucun contenu n'est
lu, sauf par :func:`is_probably_binary` qui lit un court échantillon d'octets.
Les comparaisons de motifs sont sensibles à la casse (``fnmatchcase``).
"""
from __future__ import annotations

import codecs
import fnmatch
import os
from typing import Dict, FrozenSet, Optional, Tuple

# --------------------------------------------------------------------------- langages

#: Extension (minuscule, avec le point) → langage.
EXTENSION_LANGUAGES: Dict[str, str] = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
    ".vue": "vue",
    ".svelte": "svelte",
    ".php": "php", ".phtml": "php",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".dart": "dart",
    ".go": "go",
    ".rb": "ruby", ".rake": "ruby",
    ".rs": "rust",
    ".cs": "csharp",
    ".swift": "swift",
    ".sql": "sql",
    ".md": "markdown", ".markdown": "markdown", ".mdx": "markdown",
    ".txt": "text",
    ".rst": "rst",
    ".json": "json", ".jsonc": "json", ".json5": "json",
    ".yml": "yaml", ".yaml": "yaml",
    ".toml": "toml",
    ".xml": "xml", ".xsd": "xml", ".xsl": "xml", ".xslt": "xml", ".plist": "xml",
    ".html": "html", ".htm": "html",
    ".css": "css",
    ".scss": "scss", ".sass": "scss",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell",
    ".dockerfile": "dockerfile",
    ".env": "env",
    ".ini": "ini", ".cfg": "ini", ".conf": "ini", ".properties": "ini",
    ".graphql": "graphql", ".gql": "graphql",
    ".proto": "proto",
    ".csv": "csv", ".tsv": "csv",
    ".pdf": "pdf",
    ".docx": "docx",
}

#: Noms de fichiers spéciaux (minuscules) sans extension significative → langage.
SPECIAL_FILENAMES: Dict[str, str] = {
    "dockerfile": "dockerfile",
    "containerfile": "dockerfile",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "artisan": "php",
    ".editorconfig": "ini",
}

#: Langages dont le contenu est documentaire (chunking « prose »).
DOC_LANGUAGES: FrozenSet[str] = frozenset({"markdown", "text", "rst", "pdf", "docx"})
#: Langages de configuration (manifestes, réglages, variables d'environnement).
CONFIG_LANGUAGES: FrozenSet[str] = frozenset({"json", "yaml", "toml", "ini", "env", "xml", "dockerfile"})
#: Langages de données / schémas.
DATA_LANGUAGES: FrozenSet[str] = frozenset({"sql", "csv", "graphql", "proto"})
#: Langages de code source.
CODE_LANGUAGES: FrozenSet[str] = frozenset({
    "python", "javascript", "jsx", "typescript", "tsx", "vue", "svelte", "php", "blade", "java",
    "kotlin", "dart", "go", "ruby", "rust", "csharp", "swift", "html", "css", "scss", "shell",
})
#: Formats binaires dont le texte est extrait par un parseur dédié (jamais écartés comme « binaires »).
TEXTLESS_LANGUAGES: FrozenSet[str] = frozenset({"pdf", "docx"})

#: Extensions (minuscules) toujours considérées comme binaires, sans lire le fichier.
BINARY_EXTENSIONS: FrozenSet[str] = frozenset({
    # images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".icns", ".webp", ".tif", ".tiff", ".psd",
    ".heic", ".heif", ".avif",
    # polices
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # archives
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".7z", ".rar", ".jar", ".war", ".ear", ".whl",
    # exécutables, bibliothèques, bytecode
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".lib", ".obj", ".class", ".pyc", ".pyo",
    ".pyd", ".wasm", ".dSYM", ".app", ".dmg", ".pkg", ".deb", ".rpm", ".msi", ".apk", ".aab", ".ipa",
    # bases de données
    ".sqlite", ".sqlite3", ".db", ".mdb", ".realm",
    # médias
    ".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".flac", ".avi", ".mov", ".mkv", ".webm",
    # bureautique non prise en charge (le .docx et le .pdf sont des langages à part)
    ".doc", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp",
    # divers
    ".pdb", ".pak", ".dat", ".blend", ".fbx", ".glb", ".keystore", ".jks", ".p12", ".pfx",
})

# --------------------------------------------------------------------------- exclusions par défaut

#: Dossiers jamais parcourus. Un nom simple s'applique à tout dossier portant ce nom, à
#: n'importe quelle profondeur ; un chemin (``storage/framework``) s'applique à cette
#: séquence de segments, où qu'elle apparaisse. Outre les dépendances et les artefacts de
#: construction, les dossiers de *contenus* déposés ou produits par l'application (fichiers
#: d'utilisateurs, pièces jointes, exports, sauvegardes : ``media/``, ``uploads/``…) sont
#: écartés : ils ne contiennent jamais de code et souvent des données personnelles
#: (attestations générées, pièces d'identité, dumps) qui n'ont rien à faire dans la mémoire.
DEFAULT_EXCLUDED_DIRS: FrozenSet[str] = frozenset({
    "node_modules", "vendor", ".git", ".hg", ".svn", "dist", "build", "out", ".next", ".nuxt",
    ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache", "target", "coverage",
    ".idea", ".vscode", ".gradle", "Pods", "DerivedData", "bower_components", ".cache",
    "storage/framework", "storage/logs", "public/build", ".dart_tool", ".terraform",
    "media", "uploads", "upload", "attachments", "backups", "dumps",
})

#: Motifs (sur le nom de fichier) des fichiers de secrets : clés privées, certificats, identifiants.
#: Jamais parcourus, lus ni indexés (ils font partie de :data:`DEFAULT_EXCLUDED_FILES`) ; voir
#: :func:`is_secret_file` pour le cas des ``.env`` réels, qui sont listés mais jamais lus.
SECRET_FILE_PATTERNS: FrozenSet[str] = frozenset({
    "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore", "id_rsa", "id_rsa.*", "id_ed25519", "id_ed25519.*",
    ".npmrc", ".pypirc", ".netrc", "*.secret", "*.secrets",
})

#: Un fichier ``.env`` dont le nom contient l'un de ces mots est un gabarit sans secret (``.env.example``).
ENV_TEMPLATE_MARKERS: Tuple[str, ...] = ("example", "sample", "template", "dist")

#: Fichiers jamais indexés (noms exacts ou motifs glob sur le nom de fichier).
DEFAULT_EXCLUDED_FILES: FrozenSet[str] = SECRET_FILE_PATTERNS | frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock", "poetry.lock",
    "Pipfile.lock", "Cargo.lock", "Gemfile.lock", "go.sum",
    "*.min.js", "*.min.css", "*.map", "*.log", ".DS_Store", "Thumbs.db",
})

_UTF16_BOMS: Tuple[bytes, bytes] = (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def detect_language(path: str) -> Optional[str]:
    """Langage d'après l'extension ou le nom du fichier, ``None`` si inconnu.

    Cas particuliers : ``*.blade.php`` → ``blade`` ; ``Dockerfile``, ``Dockerfile.dev``,
    ``*.dockerfile`` → ``dockerfile`` ; ``.env``, ``.env.local``, ``.env.example`` → ``env``.
    """
    name = _basename(path)
    if not name:
        return None
    lower = name.lower()
    if lower.endswith(".blade.php"):
        return "blade"
    ext = os.path.splitext(lower)[1]
    if ext and ext in EXTENSION_LANGUAGES:
        return EXTENSION_LANGUAGES[ext]
    if lower in SPECIAL_FILENAMES:
        return SPECIAL_FILENAMES[lower]
    if lower.startswith("dockerfile."):
        return "dockerfile"
    if lower.startswith(".env"):
        return "env"
    return None


def is_env_template(path: str) -> bool:
    """Vrai pour un gabarit ``.env`` sans secret : ``.env.example``, ``.env.sample``, ``.env.dist``…

    Un fichier de langage ``env`` dont le nom contient l'un des :data:`ENV_TEMPLATE_MARKERS`.
    """
    name = _basename(path).lower()
    return detect_language(name) == "env" and any(marker in name for marker in ENV_TEMPLATE_MARKERS)


def is_secret_file(path: str) -> bool:
    """Vrai si le fichier contient, par nature, des secrets : il ne doit jamais être lu ni indexé.

    Décision sur le nom seul, sans ouvrir le fichier :

    * un ``.env`` réel (``.env``, ``.env.local``, ``.env.production``, ``prod.env``…), c'est-à-dire
      un fichier de langage ``env`` qui n'est pas un gabarit (:func:`is_env_template`) ;
    * un nom correspondant à :data:`SECRET_FILE_PATTERNS` (``*.pem``, ``*.key``, ``id_rsa``, ``.npmrc``…).

    Les seconds sont écartés dès le parcours (:data:`DEFAULT_EXCLUDED_FILES`) ; les premiers sont
    listés par le scanner pour que la mémoire connaisse leur existence, mais leur contenu n'est
    jamais renvoyé par l'explorateur ni indexé.
    """
    name = _basename(path)
    if not name:
        return False
    if any(fnmatch.fnmatchcase(name, pat) for pat in SECRET_FILE_PATTERNS):
        return True
    return detect_language(name) == "env" and not is_env_template(name)


def _is_text_config(lower_name: str) -> bool:
    """``requirements*.txt``, ``constraints*.txt`` et ``CMakeLists.txt`` sont des manifestes, pas des documents."""
    return (lower_name.startswith(("requirements", "constraints")) and lower_name.endswith(".txt")) \
        or lower_name == "cmakelists.txt"


def classify_kind(path: str, language: Optional[str]) -> str:
    """Famille du fichier : ``code``, ``doc``, ``config``, ``data`` ou ``other``.

    Décisions documentées : tout fichier JSON est classé ``config`` (manifestes, tsconfig,
    réglages…), y compris un JSON « de contenu » comme un glossaire : il reste indexé et
    cherchable, simplement sans le chunking « données » réservé au SQL/CSV/GraphQL/Proto.
    De même ``requirements*.txt`` / ``constraints*.txt`` / ``CMakeLists.txt`` sont ``config``
    malgré leur extension ``.txt``.
    """
    if not language:
        return "other"
    if language == "text" and _is_text_config(_basename(path).lower()):
        return "config"
    if language in DOC_LANGUAGES:
        return "doc"
    if language in CONFIG_LANGUAGES:
        return "config"
    if language in DATA_LANGUAGES:
        return "data"
    if language in CODE_LANGUAGES:
        return "code"
    return "other"


def has_binary_extension(path: str) -> bool:
    """Vrai si l'extension est une extension binaire connue (aucune lecture du fichier)."""
    return os.path.splitext(_basename(path).lower())[1] in BINARY_EXTENSIONS


def sniff_binary(data: bytes) -> bool:
    """Vrai si l'échantillon contient un octet nul, sauf s'il commence par un BOM UTF-16."""
    if not data:
        return False
    if data.startswith(_UTF16_BOMS):
        return False
    return b"\0" in data


def is_probably_binary(path: str, sample: int = 8192) -> bool:
    """Vrai si l'extension est binaire ou si l'échantillon lu contient un octet nul.

    Un fichier illisible n'est pas considéré comme binaire (le scanner le compte à part).
    """
    if has_binary_extension(path):
        return True
    try:
        with open(path, "rb") as fh:
            head = fh.read(max(1, sample))
    except OSError:
        return False
    return sniff_binary(head)
