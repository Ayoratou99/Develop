"""Détection de la pile technique (frameworks, gestionnaires de paquets, type de dépôt).

Seuls de petits fichiers *marqueurs* sont lus (``package.json``, ``composer.json``,
``pyproject.toml``…), jamais le dépôt entier. Les comptes de langages viennent de la
liste de :class:`~develop.project_scanner.scanner.FileEntry` fournie par le scanner.

Choix du ``suggested_kind`` :

1. frameworks trouvés dans les marqueurs **à la racine** → priorité
   ``mobile`` > ``backend`` > ``frontend`` > ``database`` > ``other`` (un monolithe Laravel
   avec un ``package.json`` Vue reste un ``backend``) ;
2. sinon, frameworks trouvés dans des marqueurs **imbriqués** (profondeur ≤ 3) : un seul
   type → ce type ; plusieurs types différents → ``other`` (monodépôt) ;
3. sinon, d'après les langages : SQL majoritaire → ``database`` ; Markdown/texte
   majoritaire → ``docs`` ; sinon la famille du langage de code dominant.
"""
from __future__ import annotations

import json
import os
import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from develop.project_scanner.languages import CODE_LANGUAGES, DOC_LANGUAGES
from develop.project_scanner.scanner import FileEntry, is_excluded_dir_path, read_text

_MARKER_MAX_BYTES = 512_000
_MAX_NESTED_DEPTH = 3  # profondeur (en dossiers) des marqueurs imbriqués pris en compte
_MAX_NESTED_DIRS = 40

#: Fichiers dont la présence fait d'un dossier un projet (:func:`discover_subprojects`).
PROJECT_MARKER_NAMES: FrozenSet[str] = frozenset({
    "package.json", "composer.json", "pyproject.toml", "requirements.txt", "pom.xml", "build.gradle",
    "build.gradle.kts", "pubspec.yaml", "go.mod", "Gemfile", "Cargo.toml", ".git",
})
#: Fichiers marqueurs lus par :func:`detect_stack` (aussi cherchés dans les sous-dossiers de ``files``).
STACK_MARKER_NAMES: FrozenSet[str] = frozenset({
    "package.json", "composer.json", "pyproject.toml", "requirements.txt", "Pipfile", "setup.py",
    "setup.cfg", "pom.xml", "build.gradle", "build.gradle.kts", "pubspec.yaml", "go.mod", "Gemfile",
    "Cargo.toml", "schema.prisma", "angular.json",
})

_KIND_PRIORITY: Tuple[str, ...] = ("mobile", "backend", "frontend", "database", "other")

# (clé de dépendance, framework, type)
_NODE_FRAMEWORKS: Tuple[Tuple[str, str, str], ...] = (
    ("next", "next", "frontend"),
    ("nuxt", "nuxt", "frontend"),
    ("nuxt3", "nuxt", "frontend"),
    ("@angular/core", "angular", "frontend"),
    ("@sveltejs/kit", "sveltekit", "frontend"),
    ("svelte", "svelte", "frontend"),
    ("vue", "vue", "frontend"),
    ("react-native", "react-native", "mobile"),
    ("expo", "expo", "mobile"),
    ("react", "react", "frontend"),
    ("@nestjs/core", "nestjs", "backend"),
    ("express", "express", "backend"),
    ("fastify", "fastify", "backend"),
    ("koa", "koa", "backend"),
    ("@hapi/hapi", "hapi", "backend"),
    ("hapi", "hapi", "backend"),
    ("electron", "electron", "other"),
    ("prisma", "prisma", "database"),
    ("@prisma/client", "prisma", "database"),
)
_PHP_FRAMEWORKS: Tuple[Tuple[str, str, str], ...] = (
    ("laravel/framework", "laravel", "backend"),
    ("symfony/framework-bundle", "symfony", "backend"),
    ("symfony/symfony", "symfony", "backend"),
    ("slim/slim", "slim", "backend"),
    ("cakephp/cakephp", "cakephp", "backend"),
    ("codeigniter4/framework", "codeigniter", "backend"),
)
_PYTHON_FRAMEWORKS: Tuple[Tuple[str, str, str], ...] = (
    ("django", "django", "backend"),
    ("fastapi", "fastapi", "backend"),
    ("flask", "flask", "backend"),
)
_JVM_FRAMEWORKS: Tuple[Tuple[str, str, str], ...] = (
    ("spring-boot", "spring-boot", "backend"),
    ("org.springframework.boot", "spring-boot", "backend"),
    ("io.quarkus", "quarkus", "backend"),
    ("com.android.application", "android", "mobile"),
    ("com.android.library", "android", "mobile"),
)
_GO_FRAMEWORKS: Tuple[Tuple[str, str, str], ...] = (
    ("github.com/gin-gonic/gin", "gin", "backend"),
    ("github.com/labstack/echo", "echo", "backend"),
    ("github.com/gofiber/fiber", "fiber", "backend"),
    ("github.com/go-chi/chi", "chi", "backend"),
)
_RUBY_FRAMEWORKS: Tuple[Tuple[str, str, str], ...] = (
    ("rails", "rails", "backend"),
    ("sinatra", "sinatra", "backend"),
)
_RUST_FRAMEWORKS: Tuple[Tuple[str, str, str], ...] = (
    ("actix-web", "actix", "backend"),
    ("axum", "axum", "backend"),
    ("rocket", "rocket", "backend"),
)

#: Type suggéré d'après le langage de code dominant quand aucun framework n'est reconnu.
_LANGUAGE_KIND_HINTS: Dict[str, str] = {
    "python": "backend", "php": "backend", "blade": "backend", "java": "backend", "go": "backend",
    "ruby": "backend", "csharp": "backend", "kotlin": "backend", "rust": "backend",
    "javascript": "frontend", "jsx": "frontend", "typescript": "frontend", "tsx": "frontend",
    "vue": "frontend", "svelte": "frontend", "html": "frontend", "css": "frontend", "scss": "frontend",
    "dart": "mobile", "swift": "mobile",
}


@dataclass
class DetectedStack:
    frameworks: List[str] = field(default_factory=list)
    languages: Dict[str, int] = field(default_factory=dict)  # nombre de fichiers par langage
    package_managers: List[str] = field(default_factory=list)
    suggested_kind: str = "other"  # 'frontend' | 'backend' | 'mobile' | 'docs' | 'database' | 'other'
    markers: List[str] = field(default_factory=list)  # fichiers (relatifs) qui ont servi d'indice


@dataclass
class SubProject:
    path: str  # absolu
    name: str  # chemin relatif à la racine explorée ("frontend", "services/notifier"), ou nom de la racine
    suggested_kind: str
    frameworks: List[str] = field(default_factory=list)
    #: Chemins relatifs (POSIX, triés) des autres sous-projets découverts *sous* ce dossier, à
    #: reporter dans ``RepoConfig.exclude`` : indexés comme dépôts à part entière, ils sont ainsi
    #: écartés du parcours de celui-ci et chaque fichier n'est indexé qu'une fois. Vide sauf pour
    #: la racine d'un monodépôt.
    exclude: List[str] = field(default_factory=list)


@dataclass
class _DirFindings:
    frameworks: Dict[str, str] = field(default_factory=dict)  # framework → type
    package_managers: Set[str] = field(default_factory=set)
    markers: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- lecture des marqueurs

def _read_marker(abs_dir: str, name: str) -> Optional[str]:
    path = os.path.join(abs_dir, name)
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > _MARKER_MAX_BYTES:
            return None
    except OSError:
        return None
    return read_text(path, max_bytes=_MARKER_MAX_BYTES)


def _load_json(text: Optional[str]) -> Dict:
    if not text:
        return {}
    try:
        data = json.loads(text)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _dep_keys(manifest: Dict, *sections: str) -> Set[str]:
    keys: Set[str] = set()
    for section in sections:
        deps = manifest.get(section)
        if isinstance(deps, dict):
            keys.update(str(k).lower() for k in deps)
    return keys


def _match_text(text: str, table: Sequence[Tuple[str, str, str]], findings: _DirFindings) -> None:
    lower = text.lower()
    for needle, framework, kind in table:
        if re.search(r"(?<![\w.])" + re.escape(needle) + r"(?!\w)", lower):
            findings.frameworks[framework] = kind


def _inspect_dir(abs_dir: str) -> _DirFindings:
    """Lit les marqueurs présents directement dans ``abs_dir`` (jamais les sous-dossiers)."""
    f = _DirFindings()
    try:
        names = set(os.listdir(abs_dir))
    except OSError:
        return f

    def has(name: str) -> bool:
        return name in names

    # ----- Node -----
    if has("package.json"):
        f.markers.append("package.json")
        manifest = _load_json(_read_marker(abs_dir, "package.json"))
        deps = _dep_keys(manifest, "dependencies", "devDependencies")
        for key, framework, kind in _NODE_FRAMEWORKS:
            if key in deps:
                f.frameworks[framework] = kind
        pm = str(manifest.get("packageManager") or "").split("@")[0].strip().lower()
        if has("pnpm-lock.yaml"):
            f.package_managers.add("pnpm")
        elif has("yarn.lock"):
            f.package_managers.add("yarn")
        elif has("bun.lockb") or has("bun.lock"):
            f.package_managers.add("bun")
        elif pm in ("npm", "pnpm", "yarn", "bun"):
            f.package_managers.add(pm)
        else:
            f.package_managers.add("npm")
    for cfg, framework in (("next.config", "next"), ("nuxt.config", "nuxt"), ("svelte.config", "svelte"),
                           ("vite.config", "vite")):
        hits = sorted(n for n in names if n.startswith(cfg + "."))
        if hits:
            f.markers.append(hits[0])
            f.frameworks.setdefault(framework, "frontend")
    if has("angular.json"):
        f.markers.append("angular.json")
        f.frameworks["angular"] = "frontend"

    # ----- PHP -----
    if has("composer.json"):
        f.markers.append("composer.json")
        f.package_managers.add("composer")
        manifest = _load_json(_read_marker(abs_dir, "composer.json"))
        deps = _dep_keys(manifest, "require", "require-dev")
        for key, framework, kind in _PHP_FRAMEWORKS:
            if key in deps:
                f.frameworks[framework] = kind
    if has("artisan") and os.path.isdir(os.path.join(abs_dir, "routes")):
        f.markers.extend(["artisan", "routes/"])
        f.frameworks["laravel"] = "backend"

    # ----- Python -----
    py_markers = [n for n in ("pyproject.toml", "requirements.txt", "Pipfile", "setup.py", "setup.cfg") if has(n)]
    if py_markers:
        f.markers.extend(py_markers)
        texts = {n: _read_marker(abs_dir, n) or "" for n in py_markers}
        _match_text("\n".join(texts.values()), _PYTHON_FRAMEWORKS, f)
        if has("poetry.lock") or "[tool.poetry]" in texts.get("pyproject.toml", ""):
            f.package_managers.add("poetry")
        elif has("uv.lock"):
            f.package_managers.add("uv")
        elif has("Pipfile"):
            f.package_managers.add("pipenv")
        else:
            f.package_managers.add("pip")

    # ----- JVM -----
    jvm_markers = [n for n in ("pom.xml", "build.gradle", "build.gradle.kts") if has(n)]
    if jvm_markers:
        f.markers.extend(jvm_markers)
        f.package_managers.add("maven" if "pom.xml" in jvm_markers else "gradle")
        _match_text("\n".join(_read_marker(abs_dir, n) or "" for n in jvm_markers), _JVM_FRAMEWORKS, f)

    # ----- Dart / Flutter -----
    if has("pubspec.yaml"):
        f.markers.append("pubspec.yaml")
        f.package_managers.add("pub")
        text = _read_marker(abs_dir, "pubspec.yaml") or ""
        if re.search(r"(?m)^\s*flutter\s*:", text) or re.search(r"(?m)^\s*sdk\s*:\s*flutter\b", text):
            f.frameworks["flutter"] = "mobile"

    # ----- Go -----
    if has("go.mod"):
        f.markers.append("go.mod")
        f.package_managers.add("go")
        _match_text(_read_marker(abs_dir, "go.mod") or "", _GO_FRAMEWORKS, f)

    # ----- Ruby -----
    if has("Gemfile"):
        f.markers.append("Gemfile")
        f.package_managers.add("bundler")
        text = _read_marker(abs_dir, "Gemfile") or ""
        for key, framework, kind in _RUBY_FRAMEWORKS:
            if re.search(r"""(?m)^\s*gem\s+['"]%s['"]""" % re.escape(key), text):
                f.frameworks[framework] = kind

    # ----- Rust -----
    if has("Cargo.toml"):
        f.markers.append("Cargo.toml")
        f.package_managers.add("cargo")
        _match_text(_read_marker(abs_dir, "Cargo.toml") or "", _RUST_FRAMEWORKS, f)

    # ----- Prisma -----
    if has("schema.prisma"):
        f.markers.append("schema.prisma")
        f.frameworks.setdefault("prisma", "database")
    elif os.path.isfile(os.path.join(abs_dir, "prisma", "schema.prisma")):
        f.markers.append("prisma/schema.prisma")
        f.frameworks.setdefault("prisma", "database")
    return f


# --------------------------------------------------------------------------- décision

def _pick_kind(kinds: Set[str]) -> Optional[str]:
    for k in _KIND_PRIORITY:
        if k in kinds:
            return k
    return None


def _kind_from_languages(languages: Dict[str, int]) -> str:
    docs = sum(n for lang, n in languages.items() if lang in DOC_LANGUAGES)
    database = languages.get("sql", 0)
    code = sum(n for lang, n in languages.items() if lang in CODE_LANGUAGES)
    if database > code and database >= docs:
        return "database"
    if docs > code and docs > database:
        return "docs"
    if code:
        top = max((lang for lang in languages if lang in CODE_LANGUAGES), key=lambda l: (languages[l], l))
        return _LANGUAGE_KIND_HINTS.get(top, "other")
    return "other"


def detect_stack(root: str, files: Sequence[FileEntry]) -> DetectedStack:
    """Pile technique de ``root`` d'après ses marqueurs (racine + marqueurs imbriqués de ``files``).

    ``files`` sert aux comptes de langages et à localiser des marqueurs imbriqués (profondeur
    ≤ 3, au plus 40 dossiers) ; il peut être vide pour une détection légère sur la racine seule.
    """
    root_abs = os.path.abspath(os.path.expanduser(root or "."))
    findings: Dict[str, _DirFindings] = {"": _inspect_dir(root_abs)}
    nested: Set[str] = set()
    for entry in files:
        rel = entry.path.replace("\\", "/")
        if "/" in rel and posixpath.basename(rel) in STACK_MARKER_NAMES and rel.count("/") <= _MAX_NESTED_DEPTH:
            nested.add(posixpath.dirname(rel))
    for rel_dir in sorted(nested)[:_MAX_NESTED_DIRS]:
        findings[rel_dir] = _inspect_dir(os.path.join(root_abs, *rel_dir.split("/")))

    counts = Counter(e.language for e in files if e.language)
    languages = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    frameworks: Set[str] = set()
    package_managers: Set[str] = set()
    markers: Set[str] = set()
    for rel_dir, f in findings.items():
        frameworks.update(f.frameworks)
        package_managers.update(f.package_managers)
        markers.update(m if not rel_dir else rel_dir + "/" + m for m in f.markers)

    root_kinds = set(findings[""].frameworks.values())
    kind = _pick_kind(root_kinds)
    if kind is None:
        nested_kinds = {_pick_kind(set(f.frameworks.values())) for d, f in findings.items() if d}
        nested_kinds.discard(None)
        if len(nested_kinds) == 1:
            kind = nested_kinds.pop()
        elif nested_kinds:
            kind = "other"
        else:
            kind = _kind_from_languages(languages)
    return DetectedStack(
        frameworks=sorted(frameworks), languages=languages, package_managers=sorted(package_managers),
        suggested_kind=kind or "other", markers=sorted(markers),
    )


# --------------------------------------------------------------------------- sous-projets

def nested_subproject_paths(parent: str, candidates: Sequence[str]) -> List[str]:
    """Chemins relatifs (POSIX, triés) des ``candidates`` (chemins absolus) situés strictement
    sous le dossier ``parent`` ; le dossier lui-même et les chemins extérieurs sont ignorés.

    Sert à remplir :attr:`SubProject.exclude` : ``parent`` indexé comme dépôt doit écarter de
    son parcours les dossiers indexés comme dépôts à part entière.
    """
    base = os.path.abspath(os.path.expanduser(parent or "."))
    prefix = os.path.join(base, "")
    out: List[str] = []
    for cand in candidates:
        cand_abs = os.path.abspath(os.path.expanduser(cand or "."))
        if cand_abs != base and cand_abs.startswith(prefix):
            out.append(cand_abs[len(prefix):].replace(os.sep, "/"))
    return sorted(set(out))


def discover_subprojects(root: str, max_depth: int = 3) -> List[SubProject]:
    """Sous-dossiers (profondeur ≤ ``max_depth``) contenant un marqueur de projet.

    Les dossiers exclus par défaut, cachés ou liés symboliquement ne sont pas explorés, et
    on ne descend pas dans un sous-projet déjà trouvé. La racine elle-même est listée si
    elle porte un marqueur (``.git`` compris : un monodépôt Git garde ainsi ses dossiers
    sans marqueur — docs, base de données, outils), mais reste explorée (un monodépôt avec
    un ``package.json`` de workspace expose ainsi ses paquets). Résultat trié par chemin.

    Un sous-projet qui en contient d'autres (en pratique la racine) reçoit dans
    :attr:`SubProject.exclude` leurs chemins relatifs : repris tels quels dans
    ``RepoConfig.exclude``, ils retirent ces dossiers de son parcours (:func:`scan_repo`),
    de sorte qu'un fichier n'est jamais indexé à la fois par la racine et par son
    sous-projet (doublons de symboles, de routes, ambiguïté des correspondances).
    """
    root_abs = os.path.abspath(os.path.expanduser(root or "."))
    found: List[SubProject] = []
    if not os.path.isdir(root_abs):
        return found
    stack: List[Tuple[str, str, int]] = [("", root_abs, 0)]
    while stack:
        rel_dir, abs_dir, depth = stack.pop()
        try:
            names = sorted(os.listdir(abs_dir))
        except OSError:
            continue
        if any(n in PROJECT_MARKER_NAMES for n in names):
            light = detect_stack(abs_dir, ())
            found.append(SubProject(path=abs_dir, name=rel_dir or os.path.basename(root_abs) or root_abs,
                                    suggested_kind=light.suggested_kind, frameworks=light.frameworks))
            if rel_dir:
                continue
        if depth >= max_depth:
            continue
        for n in names:
            if n.startswith("."):
                continue
            child_rel = n if not rel_dir else rel_dir + "/" + n
            if is_excluded_dir_path(child_rel):
                continue
            child_abs = os.path.join(abs_dir, n)
            try:
                if os.path.islink(child_abs) or not os.path.isdir(child_abs):
                    continue
            except OSError:
                continue
            stack.append((child_rel, child_abs, depth + 1))
    found.sort(key=lambda s: s.path)
    paths = [s.path for s in found]
    for sp in found:
        sp.exclude = nested_subproject_paths(sp.path, paths)
    return found
