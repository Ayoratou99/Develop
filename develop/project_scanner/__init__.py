"""Parcours en lecture seule des dépôts : fichiers indexables, langages, pile technique."""
from develop.project_scanner.languages import (  # noqa: F401
    BINARY_EXTENSIONS,
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    ENV_TEMPLATE_MARKERS,
    SECRET_FILE_PATTERNS,
    classify_kind,
    detect_language,
    has_binary_extension,
    is_env_template,
    is_probably_binary,
    is_secret_file,
    sniff_binary,
)
from develop.project_scanner.scanner import (  # noqa: F401
    SKIP_REASONS,
    FileEntry,
    ScanResult,
    compute_sha1,
    decode_bytes,
    is_excluded_dir_path,
    is_excluded_file_name,
    nfc,
    parse_gitignore,
    path_is_within,
    read_text,
    scan_repo,
)
from develop.project_scanner.stack import (  # noqa: F401
    PROJECT_MARKER_NAMES,
    DetectedStack,
    SubProject,
    detect_stack,
    discover_subprojects,
    nested_subproject_paths,
)
