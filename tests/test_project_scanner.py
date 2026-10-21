"""Tests du module ``develop.project_scanner`` (fixture en lecture seule, écritures dans tmp_path)."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import unicodedata
from pathlib import Path

import pytest

from develop.git_tools import git_available
from develop.project_scanner import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_EXCLUDED_FILES,
    ENV_TEMPLATE_MARKERS,
    SECRET_FILE_PATTERNS,
    DetectedStack,
    FileEntry,
    ScanResult,
    classify_kind,
    compute_sha1,
    decode_bytes,
    detect_language,
    detect_stack,
    discover_subprojects,
    has_binary_extension,
    is_env_template,
    is_excluded_dir_path,
    is_excluded_file_name,
    is_probably_binary,
    is_secret_file,
    nested_subproject_paths,
    parse_gitignore,
    read_text,
    scan_repo,
    sniff_binary,
)

_GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@x", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env=dict(_GIT_ENV, HOME=str(repo)))


def _write(path: Path, content: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _paths(result: ScanResult) -> list:
    return [f.path for f in result.files]


def _by_path(result: ScanResult) -> dict:
    return {f.path: f for f in result.files}


# =========================================================================== languages

@pytest.mark.parametrize("name, expected", [
    ("src/main.py", "python"), ("a.js", "javascript"), ("a.mjs", "javascript"), ("a.jsx", "jsx"),
    ("src/api/client.ts", "typescript"), ("types.d.ts", "typescript"), ("OrderList.tsx", "tsx"),
    ("App.vue", "vue"), ("App.svelte", "svelte"), ("api.php", "php"),
    ("resources/views/home.blade.php", "blade"), ("Main.java", "java"), ("Main.kt", "kotlin"),
    ("main.dart", "dart"), ("main.go", "go"), ("app.rb", "ruby"), ("Gemfile", "ruby"), ("main.rs", "rust"),
    ("Program.cs", "csharp"), ("App.swift", "swift"), ("schema.sql", "sql"), ("README.md", "markdown"),
    ("notes.txt", "text"), ("index.rst", "rst"), ("glossaire.json", "json"), ("ci.yml", "yaml"),
    ("pyproject.toml", "toml"), ("pom.xml", "xml"), ("index.html", "html"), ("a.css", "css"),
    ("a.scss", "scss"), ("run.sh", "shell"), ("Dockerfile", "dockerfile"), ("Dockerfile.dev", "dockerfile"),
    ("prod.dockerfile", "dockerfile"), (".env", "env"), (".env.example", "env"), (".env.local", "env"),
    ("setup.cfg", "ini"), ("app.ini", "ini"), ("schema.graphql", "graphql"), ("api.proto", "proto"),
    ("data.csv", "csv"), ("manuel.pdf", "pdf"), ("cahier.docx", "docx"), ("artisan", "php"),
    ("PHOTO.PNG", None), ("logo.png", None), (".gitignore", None), ("Makefile", None), ("yarn.lock", None),
    ("", None), ("archive.tar.gz", None),
])
def test_detect_language(name, expected):
    assert detect_language(name) == expected


@pytest.mark.parametrize("name, language, kind", [
    ("a.py", "python", "code"), ("a.tsx", "tsx", "code"), ("a.blade.php", "blade", "code"),
    ("index.html", "html", "code"), ("run.sh", "shell", "code"),
    ("README.md", "markdown", "doc"), ("a.txt", "text", "doc"), ("a.rst", "rst", "doc"),
    ("a.pdf", "pdf", "doc"), ("a.docx", "docx", "doc"),
    ("package.json", "json", "config"), ("composer.json", "json", "config"), ("glossaire.json", "json", "config"),
    ("ci.yml", "yaml", "config"), ("pyproject.toml", "toml", "config"), ("setup.cfg", "ini", "config"),
    (".env", "env", "config"), ("pom.xml", "xml", "config"), ("Dockerfile", "dockerfile", "config"),
    ("schema.sql", "sql", "data"), ("data.csv", "csv", "data"), ("schema.graphql", "graphql", "data"),
    ("api.proto", "proto", "data"),
    (".gitignore", None, "other"), ("x.bin", "inconnu", "other"),
    ("requirements.txt", "text", "config"), ("requirements-dev.txt", "text", "config"),
    ("constraints.txt", "text", "config"), ("CMakeLists.txt", "text", "config"), ("notes/requirements.txt", "text", "config"),
])
def test_classify_kind(name, language, kind):
    assert classify_kind(name, language) == kind


def test_has_binary_extension():
    assert has_binary_extension("logo.png") and has_binary_extension("dir/FONT.WOFF2")
    assert has_binary_extension("x.pyc") and has_binary_extension("db.sqlite") and has_binary_extension("a.zip")
    assert not has_binary_extension("a.py") and not has_binary_extension("manuel.pdf")


def test_sniff_binary():
    assert sniff_binary(b"abc\0def")
    assert not sniff_binary(b"plain text\n") and not sniff_binary(b"")
    assert not sniff_binary("héllo".encode("utf-16"))  # BOM UTF-16 : nuls légitimes


def test_is_probably_binary(tmp_path: Path, sample_project: Path):
    assert is_probably_binary(str(sample_project / "frontend" / "logo.png"))
    assert not is_probably_binary(str(sample_project / "README.md"))
    blob = tmp_path / "blob.dat2"
    blob.write_bytes(b"\x00\x01\x02text")
    assert is_probably_binary(str(blob))
    assert not is_probably_binary(str(tmp_path / "absent.txt"))


def test_default_exclusions_content():
    for d in ("node_modules", "vendor", ".git", "dist", "__pycache__", "storage/framework", ".terraform"):
        assert d in DEFAULT_EXCLUDED_DIRS
    for f in ("package-lock.json", "yarn.lock", "composer.lock", "*.min.js", "*.map", ".DS_Store"):
        assert f in DEFAULT_EXCLUDED_FILES
    assert is_excluded_dir_path("frontend/node_modules/leftpad")
    assert is_excluded_dir_path("backend/storage/framework/views")
    assert not is_excluded_dir_path("backend/storage/app") and not is_excluded_dir_path("")
    assert is_excluded_file_name("app.min.js") and is_excluded_file_name("go.sum") and is_excluded_file_name("x.log")
    assert not is_excluded_file_name("app.js") and not is_excluded_file_name("package.json")


@pytest.mark.parametrize("name, secret", [
    # .env réels : détectés (langage env) mais jamais lus
    (".env", True), ("frontend/.env", True), (".env.local", True), (".env.production", True),
    ("prod.env", True), ("config/.ENV", True),
    # gabarits : sans secret, lisibles
    (".env.example", False), (".env.sample", False), (".env.template", False), (".env.dist", False),
    ("frontend/.env.example", False),
    # clés, certificats, identifiants : jamais parcourus
    ("server.key", True), ("cert.pem", True), ("id_rsa", True), ("id_rsa.pub", True), ("id_ed25519", True),
    (".npmrc", True), (".netrc", True), ("deploy.secret", True), ("keystore.jks", True),
    # fichiers ordinaires
    ("package.json", False), ("README.md", False), ("env.ts", False), ("environment.py", False),
    ("src/env/config.yaml", False), ("", False),
])
def test_is_secret_file(name, secret):
    assert is_secret_file(name) is secret


def test_secret_patterns_and_env_templates():
    assert SECRET_FILE_PATTERNS <= DEFAULT_EXCLUDED_FILES
    assert all(is_excluded_file_name(n) for n in ("server.key", "cert.pem", "id_rsa", ".npmrc"))
    for marker in ENV_TEMPLATE_MARKERS:
        assert is_env_template(".env." + marker) and not is_secret_file(".env." + marker)
    assert not is_env_template(".env") and not is_env_template("example.ts")
    # un .env réel n'est pas exclu du parcours : il est listé (détecté) mais jamais lu
    assert not is_excluded_file_name(".env") and detect_language(".env") == "env"


def test_scan_lists_real_env_but_never_keys(tmp_path: Path):
    """Non-régression « Secrets » (README) : ``.env`` réel listé sans être exclu, ``server.key`` jamais listé."""
    _write(tmp_path / "app" / ".env", "SECRET_TOKEN=sk-verif-123456\n")
    _write(tmp_path / "app" / ".env.example", "SECRET_TOKEN=\n")
    _write(tmp_path / "app" / "server.key", "-----BEGIN PRIVATE KEY-----\nabc\n")
    _write(tmp_path / "app" / "main.py", "x = 1\n")
    res = scan_repo(str(tmp_path), "t", use_git=False)
    by = _by_path(res)
    assert set(by) == {"app/.env", "app/.env.example", "app/main.py"}
    assert by["app/.env"].language == "env" and by["app/.env"].kind == "config"
    assert res.skipped["excluded_file"] == 1
    assert [p for p in by if is_secret_file(p)] == ["app/.env"]


# =========================================================================== scanner : fixture (os.walk)

@pytest.fixture(scope="module")
def fixture_scan(sample_project: Path) -> ScanResult:
    return scan_repo(str(sample_project), "fx", use_git=False)


def test_scan_fixture_basics(fixture_scan: ScanResult, sample_project: Path):
    assert fixture_scan.used_git is False
    assert fixture_scan.warnings == []
    assert fixture_scan.root == str(sample_project.resolve()) or fixture_scan.root == str(sample_project)
    paths = _paths(fixture_scan)
    assert paths == sorted(paths) and len(paths) == len(set(paths))
    # 30 fichiers - node_modules (dossier) - logo.png (binaire) - .gitignore (sans langage)
    assert len(paths) == 27
    assert all(isinstance(f, FileEntry) and f.repo_id == "fx" and f.size > 0 and f.mtime > 0 for f in fixture_scan.files)
    assert all(os.path.isabs(f.abs_path) and f.abs_path.endswith(f.path) for f in fixture_scan.files)


def test_scan_fixture_exclusions(fixture_scan: ScanResult):
    paths = _paths(fixture_scan)
    assert not any(p.startswith("frontend/node_modules") for p in paths)
    assert fixture_scan.skipped["excluded_dir"] >= 1
    assert "frontend/logo.png" not in paths
    assert fixture_scan.skipped["binary"] == 1
    assert ".gitignore" not in paths and fixture_scan.skipped["no_language"] == 1
    assert fixture_scan.skipped["too_large"] == 0 and fixture_scan.skipped["unreadable"] == 0


def test_scan_fixture_languages(fixture_scan: ScanResult):
    by = _by_path(fixture_scan)
    assert by["frontend/src/api/client.ts"].language == "typescript"
    assert by["frontend/src/components/OrderList.tsx"].language == "tsx"
    assert by["frontend/src/components/Notifications.jsx"].language == "jsx"
    assert by["backend/routes/api.php"].language == "php" and by["backend/routes/api.php"].kind == "code"
    assert by["services/notifier/src/server.js"].language == "javascript"
    assert by["database/schema.sql"].language == "sql" and by["database/schema.sql"].kind == "data"
    assert by["tools/report.py"].language == "python"
    assert by["frontend/.env.example"].language == "env" and by["frontend/.env.example"].kind == "config"
    for p in ("README.md", "docs/dictionnaire-donnees.md", "docs/decisions/2026-09-01-reunion-statuts.md"):
        assert by[p].language == "markdown" and by[p].kind == "doc"
    # décision documentée : le JSON est toujours "config", même un glossaire de contenu
    assert by["docs/glossaire.json"].language == "json" and by["docs/glossaire.json"].kind == "config"
    assert by["frontend/package.json"].kind == "config"
    assert sum(1 for f in fixture_scan.files if f.language == "php") == 9
    assert not any(f.is_binary for f in fixture_scan.files)


def test_scan_include_unknown(sample_project: Path):
    res = scan_repo(str(sample_project), "fx", use_git=False, include_unknown=True)
    by = _by_path(res)
    assert ".gitignore" in by and by[".gitignore"].language is None and by[".gitignore"].kind == "other"
    assert res.skipped["no_language"] == 0
    assert "frontend/logo.png" not in by  # toujours binaire


def test_scan_custom_exclusions(sample_project: Path):
    res = scan_repo(str(sample_project), "fx", use_git=False, exclude=["docs", "*.sql", "backend/database", "*.tsx"])
    paths = _paths(res)
    assert not any(p.startswith("docs/") for p in paths)
    assert not any(p.endswith(".sql") or p.endswith(".tsx") for p in paths)
    assert not any(p.startswith("backend/database/") for p in paths)
    assert "backend/routes/api.php" in paths and "frontend/src/api/client.ts" in paths
    assert res.skipped["excluded_dir"] >= 3 and res.skipped["excluded_file"] >= 3


def test_scan_too_large(tmp_path: Path):
    _write(tmp_path / "big.py", "x" * 5000)
    _write(tmp_path / "small.py", "y = 1\n")
    res = scan_repo(str(tmp_path), "r", use_git=False, max_file_size=1000)
    assert _paths(res) == ["small.py"] and res.skipped["too_large"] == 1


def test_scan_missing_dir(tmp_path: Path):
    res = scan_repo(str(tmp_path / "nulle-part"), "r")
    assert res.files == [] and res.used_git is False
    assert len(res.warnings) == 1 and "nulle-part" in res.warnings[0]
    assert all(v == 0 for v in res.skipped.values())
    file_as_root = _write(tmp_path / "fichier.py")
    assert scan_repo(str(file_as_root), "r").warnings


def test_scan_binary_and_textless(tmp_path: Path):
    (tmp_path / "blob.py").write_bytes(b"print(1)\x00\x01")
    (tmp_path / "manuel.pdf").write_bytes(b"%PDF-1.4\n\x00\x00stream")
    (tmp_path / "cahier.docx").write_bytes(b"PK\x03\x04\x00\x00")
    (tmp_path / "notes.txt").write_bytes("texte UTF-16".encode("utf-16"))
    (tmp_path / "image.jpg").write_bytes(b"not really")
    res = scan_repo(str(tmp_path), "r", use_git=False)
    by = _by_path(res)
    assert "blob.py" not in by and "image.jpg" not in by and res.skipped["binary"] == 2
    assert by["manuel.pdf"].is_binary and by["manuel.pdf"].kind == "doc" and by["manuel.pdf"].language == "pdf"
    assert by["cahier.docx"].is_binary and by["cahier.docx"].language == "docx"
    assert "notes.txt" in by and not by["notes.txt"].is_binary


def test_scan_gitignore_without_git(tmp_path: Path):
    _write(tmp_path / ".gitignore", "# commentaire\nbuild/\n*.tmp.py\nsecret.txt\n/racine.py\n!keep.tmp.py\n")
    _write(tmp_path / "sub" / ".gitignore", "generated/\n")
    _write(tmp_path / "app.py")
    _write(tmp_path / "racine.py")
    _write(tmp_path / "deep" / "racine.py")  # motif ancré : seulement à la racine
    _write(tmp_path / "secret.txt")
    _write(tmp_path / "a.tmp.py")
    _write(tmp_path / "keep.tmp.py")  # négation ignorée → exclu quand même
    _write(tmp_path / "build" / "out.py")
    _write(tmp_path / "sub" / "generated" / "x.py")
    _write(tmp_path / "sub" / "src" / "y.py")
    _write(tmp_path / "other" / "generated" / "z.py")  # la règle de sub/ ne s'applique pas ici
    res = scan_repo(str(tmp_path), "r", use_git=False)
    assert _paths(res) == ["app.py", "deep/racine.py", "other/generated/z.py", "sub/src/y.py"]
    assert res.used_git is False
    assert res.skipped["excluded_dir"] == 2 and res.skipped["excluded_file"] == 4


def test_scan_symlinks_not_followed(tmp_path: Path):
    _write(tmp_path / "real" / "a.py")
    os.symlink(tmp_path / "real", tmp_path / "linkdir")
    os.symlink(tmp_path / "real" / "a.py", tmp_path / "link.py")
    res = scan_repo(str(tmp_path), "r", use_git=False)
    assert _paths(res) == ["real/a.py"] and res.skipped["symlink"] == 2


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0), reason="droits POSIX requis")
def test_scan_unreadable_file(tmp_path: Path):
    ok = _write(tmp_path / "ok.py")
    locked = _write(tmp_path / "locked.py", "secret\n")
    locked.chmod(0)
    try:
        res = scan_repo(str(tmp_path), "r", use_git=False)
        assert _paths(res) == ["ok.py"] and res.skipped["unreadable"] == 1
        assert read_text(str(locked)) is None and compute_sha1(str(locked)) == ""
    finally:
        locked.chmod(0o644)
        ok.chmod(0o644)


# =========================================================================== scanner : mode git

needs_git = pytest.mark.skipif(not git_available(), reason="git absent")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _write(tmp_path / ".gitignore", "ignored/\n*.log\n")
    _write(tmp_path / "src" / "a.py", "print('a')\n")
    _write(tmp_path / "src" / "b.ts", "export const b = 1;\n")
    _write(tmp_path / "README.md", "# Projet\n")
    _write(tmp_path / "vendor" / "lib.php", "<?php\n")  # suivi par git mais exclu par défaut
    _write(tmp_path / "yarn.lock", "# lock\n")
    _write(tmp_path / "ignored" / "junk.py")
    _write(tmp_path / "debug.log")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


@needs_git
def test_scan_git_repo(git_repo: Path):
    _write(git_repo / "src" / "new.py", "print('new')\n")  # non suivi mais pas ignoré
    _write(git_repo / "node_modules" / "x" / "index.js")  # non suivi, non ignoré par git : exclu par défaut
    os.symlink(git_repo / "src" / "a.py", git_repo / "src" / "link.py")
    res = scan_repo(str(git_repo), "g", use_git=True)
    assert res.used_git is True and res.warnings == []
    paths = _paths(res)
    assert paths == ["README.md", "src/a.py", "src/b.ts", "src/new.py"]
    assert "ignored/junk.py" not in paths and "debug.log" not in paths
    assert res.skipped["excluded_dir"] == 2  # vendor/lib.php, node_modules/x/index.js
    assert res.skipped["excluded_file"] == 1  # yarn.lock
    assert res.skipped["symlink"] == 1 and res.skipped["no_language"] == 1  # .gitignore


@needs_git
def test_scan_git_repo_custom_exclude_and_size(git_repo: Path):
    _write(git_repo / "src" / "big.py", "#" * 3000)
    res = scan_repo(str(git_repo), "g", exclude=["src/*.ts", "README.md"], max_file_size=2000)
    assert _paths(res) == ["src/a.py"]
    assert res.skipped["excluded_file"] >= 2 and res.skipped["too_large"] == 1


@needs_git
def test_scan_git_disabled_or_subdir(git_repo: Path):
    res = scan_repo(str(git_repo), "g", use_git=False)
    assert res.used_git is False
    assert _paths(res) == ["README.md", "src/a.py", "src/b.ts"]  # .gitignore simple appliqué quand même
    sub = scan_repo(str(git_repo / "src"), "g", use_git=True)  # pas la racine du dépôt → parcours fichier
    assert sub.used_git is False and _paths(sub) == ["a.py", "b.ts"]


def test_scan_fixture_use_git_true_falls_back(sample_project: Path):
    """La fixture n'est pas une racine git : use_git=True doit retomber sur le parcours fichier."""
    res = scan_repo(str(sample_project), "fx", use_git=True)
    assert res.used_git is False and len(res.files) == 27


# =========================================================================== lecture

def test_compute_sha1(tmp_path: Path):
    data = os.urandom(200_000)
    p = tmp_path / "blob.bin"
    p.write_bytes(data)
    assert compute_sha1(str(p)) == hashlib.sha1(data).hexdigest()
    assert compute_sha1(str(tmp_path / "absent")) == ""


def test_read_text_encodings(tmp_path: Path):
    utf8 = tmp_path / "u8.txt"
    utf8.write_bytes("héllo wörld".encode("utf-8"))
    assert read_text(str(utf8)) == "héllo wörld"
    sig = tmp_path / "sig.txt"
    sig.write_bytes(b"\xef\xbb\xbf" + "bom é".encode("utf-8"))
    assert read_text(str(sig)) == "bom é"
    u16 = tmp_path / "u16.txt"
    u16.write_bytes("seize é".encode("utf-16"))
    assert read_text(str(u16)) == "seize é"
    latin = tmp_path / "latin.txt"
    latin.write_bytes(b"caf\xe9")
    assert read_text(str(latin)) == "café"
    binary = tmp_path / "bin.dat"
    binary.write_bytes(b"abc\x00def")
    assert read_text(str(binary)) is None
    assert read_text(str(tmp_path / "absent.txt")) is None


def test_read_text_truncation_keeps_utf8(tmp_path: Path):
    p = tmp_path / "long.txt"
    p.write_bytes(("é" * 100).encode("utf-8"))  # 200 octets
    assert read_text(str(p), max_bytes=3) == "é"  # séquence coupée abandonnée, pas de latin-1
    assert read_text(str(p), max_bytes=4) == "éé"
    assert len(read_text(str(p), max_bytes=10_000)) == 100


def test_decode_bytes():
    assert decode_bytes(b"") == ""
    assert decode_bytes("é".encode("utf-8")) == "é"
    assert decode_bytes(b"\xe9t\xe9") == "été"
    assert decode_bytes(b"\x00") is None
    assert decode_bytes("x".encode("utf-16") + b"\x00") == "x"  # UTF-16 avec octet orphelin final


def test_parse_gitignore():
    rules = parse_gitignore("# c\n\nbuild/\n*.log\n/root.txt\nsrc/gen/\n**/deep\n!keep\n\\#literal\n", base="sub")
    assert [(r.pattern, r.dir_only, r.anchored, r.base) for r in rules] == [
        ("build", True, False, "sub"), ("*.log", False, False, "sub"), ("root.txt", False, True, "sub"),
        ("src/gen", True, True, "sub"), ("deep", False, False, "sub"), ("#literal", False, False, "sub"),
    ]


# =========================================================================== stack

def _scan(root: Path) -> ScanResult:
    return scan_repo(str(root), "x", use_git=False)


def test_detect_stack_fixture_subprojects(sample_project: Path):
    fe = detect_stack(str(sample_project / "frontend"), _scan(sample_project / "frontend").files)
    assert isinstance(fe, DetectedStack)
    assert fe.frameworks == ["react"] and fe.suggested_kind == "frontend"
    assert fe.package_managers == ["npm"] and fe.markers == ["package.json"]
    assert fe.languages["typescript"] == 2 and fe.languages["tsx"] == 2 and fe.languages["jsx"] == 1
    assert "javascript" not in fe.languages  # node_modules exclu

    be = detect_stack(str(sample_project / "backend"), _scan(sample_project / "backend").files)
    assert be.frameworks == ["laravel"] and be.suggested_kind == "backend"
    assert be.package_managers == ["composer"] and be.markers == ["composer.json"]
    assert be.languages == {"php": 9, "json": 1}

    nt = detect_stack(str(sample_project / "services" / "notifier"), _scan(sample_project / "services" / "notifier").files)
    assert nt.frameworks == ["express"] and nt.suggested_kind == "backend" and nt.languages["javascript"] == 2


def test_detect_stack_fixture_root(sample_project: Path, fixture_scan: ScanResult):
    st = detect_stack(str(sample_project), fixture_scan.files)
    assert st.frameworks == ["express", "laravel", "react"]
    assert st.package_managers == ["composer", "npm"]
    assert st.markers == ["backend/composer.json", "frontend/package.json", "services/notifier/package.json"]
    assert st.suggested_kind == "other"  # monodépôt : frontend + backend imbriqués, rien à la racine
    assert st.languages["php"] == 9 and st.languages["markdown"] == 4 and st.languages["sql"] == 1
    assert list(st.languages)[0] == "php"  # trié par fréquence décroissante


def test_detect_stack_language_fallbacks(tmp_path: Path):
    docs = tmp_path / "docs"
    for i in range(3):
        _write(docs / ("page%d.md" % i), "# titre\n")
    _write(docs / "meta.json", "{}")
    st = detect_stack(str(docs), _scan(docs).files)
    assert st.suggested_kind == "docs" and st.frameworks == [] and st.markers == []

    db = tmp_path / "db"
    for i in range(3):
        _write(db / ("migr%d.sql" % i), "select 1;\n")
    _write(db / "README.md", "# schéma\n")
    assert detect_stack(str(db), _scan(db).files).suggested_kind == "database"

    php = tmp_path / "php"
    _write(php / "index.php", "<?php\n")
    _write(php / "README.md", "# x\n")
    assert detect_stack(str(php), _scan(php).files).suggested_kind == "backend"

    assert detect_stack(str(tmp_path / "vide"), []).suggested_kind == "other"
    assert detect_stack(str(tmp_path / "absent"), []) == DetectedStack()


def test_detect_stack_markers_by_ecosystem(tmp_path: Path):
    py = tmp_path / "py"
    _write(py / "pyproject.toml", '[project]\nname = "api"\ndependencies = ["fastapi>=0.110", "httpx"]\n')
    _write(py / "poetry.lock", "")
    st = detect_stack(str(py), [])
    assert st.frameworks == ["fastapi"] and st.suggested_kind == "backend"
    assert st.package_managers == ["poetry"] and st.markers == ["pyproject.toml"]

    req = tmp_path / "req"
    _write(req / "requirements.txt", "Django==4.2\ndjango-cors-headers\n")
    assert detect_stack(str(req), []).frameworks == ["django"]
    assert detect_stack(str(req), []).package_managers == ["pip"]

    lar = tmp_path / "laravel"
    _write(lar / "artisan", "#!/usr/bin/env php\n")
    (lar / "routes").mkdir()
    _write(lar / "package.json", '{"dependencies": {"vue": "^3"}}')
    st = detect_stack(str(lar), [])
    assert st.frameworks == ["laravel", "vue"] and st.suggested_kind == "backend"  # backend prime sur frontend
    assert "artisan" in st.markers and "routes/" in st.markers

    fl = tmp_path / "flutter"
    _write(fl / "pubspec.yaml", "name: app\ndependencies:\n  flutter:\n    sdk: flutter\n")
    st = detect_stack(str(fl), [])
    assert st.frameworks == ["flutter"] and st.suggested_kind == "mobile" and st.package_managers == ["pub"]

    spring = tmp_path / "spring"
    _write(spring / "pom.xml", "<project><parent><artifactId>spring-boot-starter-parent</artifactId></parent></project>")
    st = detect_stack(str(spring), [])
    assert st.frameworks == ["spring-boot"] and st.package_managers == ["maven"] and st.suggested_kind == "backend"

    nxt = tmp_path / "next"
    _write(nxt / "package.json", '{"dependencies": {"next": "14", "react": "18"}, "packageManager": "pnpm@8.1.0"}')
    _write(nxt / "next.config.js", "module.exports = {};\n")
    st = detect_stack(str(nxt), [])
    assert st.frameworks == ["next", "react"] and st.package_managers == ["pnpm"] and st.suggested_kind == "frontend"
    assert st.markers == ["next.config.js", "package.json"]

    go = tmp_path / "go"
    _write(go / "go.mod", "module x\n\nrequire github.com/gin-gonic/gin v1.9.0\n")
    assert detect_stack(str(go), []).frameworks == ["gin"]

    rails = tmp_path / "rails"
    _write(rails / "Gemfile", "source 'https://rubygems.org'\ngem 'rails', '~> 7.1'\n")
    st = detect_stack(str(rails), [])
    assert st.frameworks == ["rails"] and st.package_managers == ["bundler"]

    prisma = tmp_path / "prisma"
    _write(prisma / "prisma" / "schema.prisma", "datasource db {}\n")
    st = detect_stack(str(prisma), [])
    assert st.frameworks == ["prisma"] and st.suggested_kind == "database" and st.markers == ["prisma/schema.prisma"]

    broken = tmp_path / "broken"
    _write(broken / "package.json", "{ pas du json")
    _write(broken / "composer.json", "[1, 2]")
    st = detect_stack(str(broken), [])
    assert st.frameworks == [] and sorted(st.markers) == ["composer.json", "package.json"]
    assert st.package_managers == ["composer", "npm"]


def test_discover_subprojects_fixture(sample_project: Path):
    subs = discover_subprojects(str(sample_project))
    by_name = {s.name: s for s in subs}
    assert set(by_name) == {"frontend", "backend", "services/notifier"}  # tools/ sans marqueur, racine non plus
    assert by_name["frontend"].suggested_kind == "frontend" and by_name["frontend"].frameworks == ["react"]
    assert by_name["backend"].suggested_kind == "backend" and by_name["backend"].frameworks == ["laravel"]
    assert by_name["services/notifier"].suggested_kind == "backend" and by_name["services/notifier"].frameworks == ["express"]
    assert all(os.path.isabs(s.path) and os.path.isdir(s.path) for s in subs)
    assert [s.path for s in subs] == sorted(s.path for s in subs)
    assert all(s.exclude == [] for s in subs)  # aucun sous-projet n'en contient un autre
    assert {s.name for s in discover_subprojects(str(sample_project), max_depth=1)} == {"frontend", "backend"}


def test_discover_subprojects_root_and_nesting(tmp_path: Path):
    _write(tmp_path / "package.json", '{"workspaces": ["packages/*"], "dependencies": {"express": "4"}}')
    _write(tmp_path / "packages" / "web" / "package.json", '{"dependencies": {"react": "18"}}')
    _write(tmp_path / "packages" / "web" / "tools" / "package.json", "{}")  # dans un sous-projet trouvé : ignoré
    _write(tmp_path / "node_modules" / "dep" / "package.json", "{}")  # exclu
    _write(tmp_path / ".hidden" / "package.json", "{}")  # dossier caché : non exploré
    _write(tmp_path / "docs" / "README.md", "# docs\n")
    subs = discover_subprojects(str(tmp_path))
    assert [s.name for s in subs] == [tmp_path.name, "packages/web"]
    assert subs[0].path == str(tmp_path) and subs[0].suggested_kind == "backend" and subs[0].frameworks == ["express"]
    assert subs[1].suggested_kind == "frontend"
    assert subs[0].exclude == ["packages/web"] and subs[1].exclude == []  # la racine n'indexe pas son paquet
    assert discover_subprojects(str(tmp_path / "absent")) == []
    assert discover_subprojects(str(tmp_path / "docs")) == []


def test_nested_subproject_paths(tmp_path: Path):
    root = str(tmp_path / "mono")
    inside = [os.path.join(root, "backend"), os.path.join(root, "services", "notifier")]
    outside = [root, str(tmp_path / "mono2" / "x"), str(tmp_path / "ailleurs")]
    assert nested_subproject_paths(root, inside + outside) == ["backend", "services/notifier"]
    assert nested_subproject_paths(root + os.sep, inside) == ["backend", "services/notifier"]  # barre finale tolérée
    assert nested_subproject_paths(os.path.join(root, "backend"), [root] + inside) == []
    assert nested_subproject_paths(root, []) == []


def test_discover_subprojects_git_monorepo_root_excludes_nested(tmp_path: Path, sample_project: Path):
    """Non-régression : sur un monodépôt Git (racine marquée par ``.git``), la racine est proposée
    en plus de ses sous-projets mais porte leur exclusion ; indexer tous les candidats ne parcourt
    chaque fichier qu'une fois (plus de symboles, routes ou modèles en double)."""
    project = tmp_path / "project"
    shutil.copytree(sample_project, project)
    (project / ".git").mkdir()  # marqueur de dépôt suffisant pour la découverte (git n'est pas appelé)
    subs = discover_subprojects(str(project))
    assert [s.name for s in subs] == ["project", "backend", "frontend", "services/notifier"]
    root, children = subs[0], subs[1:]
    assert root.path == str(project) and root.suggested_kind == "other"
    assert root.exclude == ["backend", "frontend", "services/notifier"]
    assert all(c.exclude == [] for c in children)

    # ce que l'assistant configure par défaut (toutes les cases cochées) : un dépôt par candidat, avec son exclude
    whole = scan_repo(str(project), "all", use_git=False)
    parts = [scan_repo(sp.path, sp.name, exclude=sp.exclude, use_git=False) for sp in subs]
    root_part = parts[0]
    assert root_part.skipped["excluded_dir"] >= 3 and root_part.warnings == []
    assert not any(p.startswith(("backend/", "frontend/", "services/notifier/")) for p in _paths(root_part))
    assert "README.md" in _paths(root_part) and "docs/dictionnaire-donnees.md" in _paths(root_part)
    assert "database/schema.sql" in _paths(root_part) and "tools/report.py" in _paths(root_part)
    seen = [os.path.normpath(f.abs_path) for part in parts for f in part.files]
    assert len(seen) == len(set(seen)) == len(whole.files)  # chaque fichier une seule fois, aucun oublié
    assert set(seen) == {os.path.normpath(f.abs_path) for f in whole.files}
    assert sum(1 for p in seen if p.endswith("OrderService.php")) == 1

    # sans l'exclusion (ancien comportement), la racine réindexait tout
    assert len(scan_repo(root.path, "root", use_git=False).files) == len(whole.files)
    # une fois les sous-projets décochés, rien n'est à exclure (c'est au client de recalculer)
    assert nested_subproject_paths(root.path, [root.path]) == []


# =========================================================================== Unicode NFC

_NFD_DIR = unicodedata.normalize("NFD", "Décomposé")
_NFD_FILE = unicodedata.normalize("NFD", "Résumé.php")
_NFC_REL = unicodedata.normalize("NFC", "app/Décomposé/Résumé.php")


def test_scan_walk_normalizes_paths_to_nfc(tmp_path: Path):
    """Nom accentué décomposé (NFD, forme écrite par le Finder/HFS+) : clé indexée en NFC, fichier lisible."""
    assert _NFD_FILE != unicodedata.normalize("NFC", _NFD_FILE)
    _write(tmp_path / "app" / _NFD_DIR / _NFD_FILE, "<?php\nclass Resume {}\n")
    _write(tmp_path / "plain.py", "x = 1\n")
    res = scan_repo(str(tmp_path), "n", use_git=False)
    paths = _paths(res)
    assert paths == [_NFC_REL, "plain.py"]
    assert unicodedata.normalize("NFD", _NFC_REL) not in paths
    entry = _by_path(res)[_NFC_REL]
    assert os.path.isfile(entry.abs_path) and read_text(entry.abs_path) == "<?php\nclass Resume {}\n"
    assert entry.abs_path.startswith(str(tmp_path))
    # un motif d'exclusion saisi en NFD exclut quand même le dossier indexé en NFC
    res = scan_repo(str(tmp_path), "n", use_git=False, exclude=[_NFD_DIR])
    assert _paths(res) == ["plain.py"] and res.skipped["excluded_dir"] == 1
    # idem pour une règle .gitignore écrite en NFD
    _write(tmp_path / ".gitignore", _NFD_FILE + "\n")
    res = scan_repo(str(tmp_path), "n", use_git=False)
    assert _paths(res) == ["plain.py"] and res.skipped["excluded_file"] == 1


@needs_git
def test_scan_git_repo_normalizes_paths_to_nfc(git_repo: Path):
    """Mode ``git ls-files`` : clé NFC même quand git renvoie les octets NFD du disque (precomposeunicode désactivé)."""
    _git(git_repo, "config", "core.precomposeunicode", "false")
    _write(git_repo / "src" / _NFD_DIR / _NFD_FILE, "<?php\nclass Resume {}\n")
    _git(git_repo, "add", "-A")
    _git(git_repo, "commit", "-q", "-m", "nfd")
    res = scan_repo(str(git_repo), "g", use_git=True)
    assert res.used_git is True
    nfc_rel = unicodedata.normalize("NFC", "src/Décomposé/Résumé.php")
    paths = _paths(res)
    assert nfc_rel in paths and unicodedata.normalize("NFD", nfc_rel) not in paths
    entry = _by_path(res)[nfc_rel]
    assert os.path.isfile(entry.abs_path) and read_text(entry.abs_path) == "<?php\nclass Resume {}\n"
    # et quand git recompose lui-même (défaut macOS) : même clé
    _git(git_repo, "config", "core.precomposeunicode", "true")
    assert nfc_rel in _paths(scan_repo(str(git_repo), "g", use_git=True))


# =========================================================================== fichiers et dossiers illisibles

needs_posix_perms = pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                                       reason="droits POSIX requis")


@needs_posix_perms
def test_scan_names_unreadable_files_and_dirs(tmp_path: Path):
    """Les fichiers comptés ``unreadable`` et les dossiers impossibles à lister sont nommés, pas seulement comptés
    (l'indexeur s'en sert pour ne pas supprimer un fichier simplement devenu inaccessible)."""
    _write(tmp_path / "ok.py")
    locked = _write(tmp_path / "locked.py", "secret\n")
    closed = tmp_path / "closed"
    _write(closed / "a.py")
    _write(closed / "sub" / "b.py")
    locked.chmod(0)
    closed.chmod(0)
    try:
        res = scan_repo(str(tmp_path), "r", use_git=False)
        assert _paths(res) == ["ok.py"]
        assert res.skipped["unreadable"] == 1 and res.unreadable == ["locked.py"]
        assert res.unreadable_dirs == ["closed"]
        assert len(res.warnings) == 1 and "closed" in res.warnings[0] and "illisible" in res.warnings[0]
        # la racine elle-même fermée : aucun fichier, dossier "" listé, un seul avertissement, compteurs à zéro
        tmp_path.chmod(0)
        try:
            root = scan_repo(str(tmp_path), "r", use_git=False)
        finally:
            tmp_path.chmod(0o755)
        assert root.files == [] and root.unreadable == [] and root.unreadable_dirs == [""]
        assert len(root.warnings) == 1 and all(v == 0 for v in root.skipped.values())
    finally:
        tmp_path.chmod(0o755)
        closed.chmod(0o755)
        locked.chmod(0o644)
    # sans incident : listes vides
    clean = scan_repo(str(tmp_path), "r", use_git=False)
    assert clean.unreadable == [] and clean.unreadable_dirs == [] and clean.skipped["unreadable"] == 0
    assert _paths(clean) == ["closed/a.py", "closed/sub/b.py", "locked.py", "ok.py"]


@needs_git
@needs_posix_perms
def test_scan_git_names_unreadable_paths(git_repo: Path):
    """Mode git : un fichier suivi illisible est nommé ; un dossier fermé est signalé (par git ou par le parcours
    de repli), jamais passé sous silence."""
    (git_repo / "README.md").chmod(0)
    try:
        res = scan_repo(str(git_repo), "g", use_git=True)
        assert res.used_git is True
        assert _paths(res) == ["src/a.py", "src/b.ts"]
        assert res.unreadable == ["README.md"] and res.skipped["unreadable"] == 1 and res.unreadable_dirs == []
    finally:
        (git_repo / "README.md").chmod(0o644)
    (git_repo / "src").chmod(0)
    try:
        res = scan_repo(str(git_repo), "g", use_git=True)
        assert not any(p.startswith("src/") for p in _paths(res))
        covered = {"src/a.py", "src/b.ts"} <= set(res.unreadable) or "src" in res.unreadable_dirs
        assert covered, (res.unreadable, res.unreadable_dirs, res.warnings)
    finally:
        (git_repo / "src").chmod(0o755)
