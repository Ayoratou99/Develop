"""Tests des scripts d'installation ``install.sh`` et ``install.ps1``.

Syntaxe (``sh -n``, ``bash -n``, ``dash -n`` ; PowerShell si ``pwsh`` est présent), aide, et parcours complet
d'``install.sh`` sans réseau : un faux ``python3`` (script shell) joue ``-m venv``, ``-m pip`` et
``-m develop --version`` et journalise ses appels.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install.sh"
INSTALL_PS1 = ROOT / "install.ps1"

FAKE_PYTHON = """#!/bin/sh
# Faux Python : journalise ses arguments et imite venv / pip / develop.
printf '%s\\n' "$*" >> "${FAKE_PY_LOG:?}"
case "$1" in
  -c)
    case "$2" in *print*) echo "3.12.1" ;; esac
    exit 0 ;;
  -m)
    case "$2" in
      venv) mkdir -p "$3/bin" && cp "$0" "$3/bin/python" && chmod 755 "$3/bin/python" \\
            && echo "home = fake" > "$3/pyvenv.cfg" && exit 0 ;;
      pip) exit 0 ;;
      develop) echo "develop 0.1.0" && exit 0 ;;
    esac ;;
esac
exit 1
"""

OLD_PYTHON = """#!/bin/sh
exit 1
"""

TOOLS = ("sh", "cat", "cp", "mkdir", "rm", "chmod", "grep", "uname", "basename", "dirname")

# install.sh est le script POSIX : sous Windows (même avec le sh de Git), c'est install.ps1 qui compte.
needs_sh = pytest.mark.skipif(sys.platform == "win32" or shutil.which("sh") is None,
                              reason="install.sh est réservé à macOS/Linux (sous Windows : install.ps1)")


def _tools_dir(tmp_path: Path) -> Path:
    """Un PATH minimal : les utilitaires du script, sans aucun Python."""
    d = tmp_path / "tools"
    d.mkdir()
    for name in TOOLS:
        real = shutil.which(name)
        assert real, name
        os.symlink(real, d / name)
    return d


@pytest.fixture
def env(tmp_path: Path) -> SimpleNamespace:
    home = tmp_path / "home"
    home.mkdir()
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    python = fake_dir / "python3"
    python.write_text(FAKE_PYTHON, encoding="utf-8")
    python.chmod(0o755)
    old = fake_dir / "python-old"
    old.write_text(OLD_PYTHON, encoding="utf-8")
    old.chmod(0o755)
    log = tmp_path / "python.log"
    variables = {
        "HOME": str(home),
        "PATH": str(_tools_dir(tmp_path)),
        "FAKE_PY_LOG": str(log),
        "SHELL": "/bin/zsh",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    return SimpleNamespace(home=home, env=variables, log=log, python=python, old_python=old, tmp=tmp_path)


def run_install(args, env: dict, cwd: Path = ROOT) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    return subprocess.run(["sh", str(INSTALL_SH), *args], capture_output=True, text=True, env=env, cwd=str(cwd))


# --------------------------------------------------------------------------- syntaxe et aide

@needs_sh
@pytest.mark.parametrize("shell", ["sh", "bash", "dash"])
def test_install_sh_syntax(shell: str):
    if shutil.which(shell) is None:
        pytest.skip("%s absent" % shell)
    proc = subprocess.run([shell, "-n", str(INSTALL_SH)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bit exécutable sans objet sous Windows")
def test_install_sh_is_executable_and_posix():
    assert os.access(INSTALL_SH, os.X_OK)
    first = INSTALL_SH.read_text(encoding="utf-8").splitlines()[0]
    assert first == "#!/bin/sh"


@needs_sh
def test_install_sh_help_and_unknown_option(env):
    proc = run_install(["--help"], env.env)
    assert proc.returncode == 0 and "--desktop" in proc.stdout and "--from" in proc.stdout
    proc = run_install(["--bogus"], env.env)
    assert proc.returncode == 2 and "Option inconnue" in proc.stderr
    proc = run_install(["--python"], env.env)
    assert proc.returncode == 1 and "attend une valeur" in proc.stderr
    assert not env.log.exists()  # aucun Python lancé


# --------------------------------------------------------------------------- parcours complet (faux Python)

@needs_sh
def test_install_sh_full_run_is_idempotent(env):
    proc = run_install(["--python", str(env.python), "--desktop"], env.env)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    venv = env.home / ".develop" / "venv"
    assert (venv / "pyvenv.cfg").is_file() and (venv / "bin" / "python").is_file()
    shim = env.home / ".local" / "bin" / "develop"
    assert shim.is_file() and os.access(shim, os.X_OK)
    text = shim.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n") and str(venv / "bin" / "python") in text and '-m develop "$@"' in text
    calls = env.log.read_text(encoding="utf-8").splitlines()
    assert any(c.startswith("-m venv ") for c in calls)
    pip = [c for c in calls if c.startswith("-m pip install")]
    assert len(pip) == 1 and pip[0].endswith("%s[desktop]" % ROOT)
    assert "Develop est installé" in proc.stdout and "develop doctor" in proc.stdout
    assert "n'est pas dans votre PATH" in proc.stdout and ".zshrc" in proc.stdout and str(shim) in proc.stdout
    assert "(création)" in proc.stdout and "développ" not in proc.stderr

    # Deuxième passage : environnement réutilisé, pas de nouveau venv, paquet réinstallé sans l'extra.
    proc = run_install(["--python", str(env.python)], env.env)
    assert proc.returncode == 0, proc.stderr
    assert "réutilisé" in proc.stdout
    calls = env.log.read_text(encoding="utf-8").splitlines()
    assert sum(1 for c in calls if c.startswith("-m venv ")) == 1
    pip = [c for c in calls if c.startswith("-m pip install")]
    assert len(pip) == 2 and pip[1].endswith(str(ROOT)) and "[desktop]" not in pip[1]
    assert "--desktop pour l'activer" in proc.stdout


@needs_sh
def test_install_sh_honours_bin_dir_and_develop_home(env):
    env.env["DEVELOP_HOME"] = str(env.tmp / "elsewhere")
    env.env["PATH"] = env.env["PATH"] + os.pathsep + str(env.tmp / "bin")
    proc = run_install(["--python", str(env.python), "--bin-dir", str(env.tmp / "bin")], env.env)
    assert proc.returncode == 0, proc.stderr
    assert (env.tmp / "elsewhere" / "venv" / "pyvenv.cfg").is_file()
    assert (env.tmp / "bin" / "develop").is_file()
    assert not (env.home / ".develop").exists()
    assert "n'est pas dans votre PATH" not in proc.stdout  # --bin-dir était dans le PATH


@needs_sh
def test_install_sh_refuses_to_overwrite_a_foreign_develop_command(env):
    shim = env.home / ".local" / "bin" / "develop"
    shim.parent.mkdir(parents=True)
    shim.write_text("#!/bin/sh\necho autre outil\n", encoding="utf-8")
    proc = run_install(["--python", str(env.python)], env.env)
    assert proc.returncode == 1 and "n'a pas été créé par ce script" in proc.stderr
    assert shim.read_text(encoding="utf-8") == "#!/bin/sh\necho autre outil\n"


@needs_sh
def test_install_sh_recreates_a_broken_venv_but_not_a_foreign_dir(env):
    venv = env.home / ".develop" / "venv"
    venv.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = cassé\n", encoding="utf-8")  # pas de bin/python
    proc = run_install(["--python", str(env.python)], env.env)
    assert proc.returncode == 0 and "recréé" in proc.stdout and (venv / "bin" / "python").is_file()

    shutil.rmtree(venv)
    venv.mkdir()
    (venv / "notes.txt").write_text("à moi", encoding="utf-8")
    proc = run_install(["--python", str(env.python)], env.env)
    assert proc.returncode == 1 and "n'est pas un environnement Python" in proc.stderr
    assert (venv / "notes.txt").is_file()


@needs_sh
def test_install_sh_rejects_old_python_and_explains_when_none_found(env):
    proc = run_install(["--python", str(env.old_python)], env.env)
    assert proc.returncode == 1 and "3.9" in proc.stderr
    proc = run_install([], env.env)  # PATH sans aucun Python
    assert proc.returncode == 1
    assert "Aucun Python 3.9" in proc.stderr and "python.org" in proc.stderr
    assert not (env.home / ".develop").exists()


@needs_sh
def test_install_sh_validates_the_source(env):
    proc = run_install(["--python", str(env.python), "--from", str(env.tmp / "absent")], env.env)
    assert proc.returncode == 1 and "introuvable" in proc.stderr
    empty = env.tmp / "vide"
    empty.mkdir()
    proc = run_install(["--python", str(env.python), "--from", str(empty)], env.env)
    assert proc.returncode == 1 and "pyproject.toml" in proc.stderr
    archive = env.tmp / "develop_local-0.1.0.tar.gz"
    archive.write_bytes(b"")
    proc = run_install(["--python", str(env.python), "--from", str(archive), "--desktop"], env.env)
    assert proc.returncode == 0, proc.stderr
    pip = [c for c in env.log.read_text(encoding="utf-8").splitlines() if c.startswith("-m pip install")]
    assert pip[-1].endswith("%s[desktop]" % archive)


# --------------------------------------------------------------------------- install.ps1

def test_install_ps1_is_utf8_with_bom():
    """Windows PowerShell 5.1 lit un .ps1 sans BOM en ANSI : les accents seraient altérés."""
    raw = INSTALL_PS1.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    raw.decode("utf-8")


def test_install_ps1_static_consistency():
    """Relecture statique d'install.ps1 (jamais exécuté sur macOS/Linux) : les blocs attendus sont présents et
    dans l'ordre : garde Windows, codes de retour natifs maîtrisés, détection de Python (py -3, python3,
    python ; alias du Microsoft Store refusé), venv, pip, shim ``develop.cmd``, PATH utilisateur (registre,
    sans doublon), ``DEVELOP_HOME``, message final ; politique d'exécution documentée dans l'en-tête."""
    text = INSTALL_PS1.read_text(encoding="utf-8-sig")
    ordered = [
        "-ExecutionPolicy Bypass -File install.ps1",  # en-tête : commande d'installation documentée
        "[CmdletBinding()]",
        '$ErrorActionPreference = "Stop"',
        "$PSNativeCommandUseErrorActionPreference = $false",  # PowerShell 7.4+ : codes de retour testés à la main
        '$env:OS -ne "Windows_NT"',  # garde : pas pwsh sous macOS/Linux
        "$env:DEVELOP_HOME",
        "function Invoke-Native",
        "WindowsApps",  # Test-StoreAlias : l'alias du Microsoft Store n'est pas un Python
        "function Test-PythonOk",
        '"py -3", "python3", "python"',  # ordre de détection
        'Test-StoreAlias "python"',  # … et l'alias est expliqué à l'utilisateur quand rien n'est trouvé
        'Fail "la création de l\'environnement a échoué."',
        '"-m", "pip", "install"',
        '"-m", "develop", "--version"',
        'Join-Path $BinDir "develop.cmd"',
        "-m develop %*",
        'HKCU:\\Environment',  # PATH utilisateur : registre, valeur ExpandString conservée
        "-notcontains",  # pas de doublon
        "ExpandString",
        "SendMessageTimeout",  # WM_SETTINGCHANGE : nouveau PATH sans redémarrer la session
        "Develop est installé",
        "develop doctor",
    ]
    position = -1
    for block in ordered:
        found = text.find(block, position + 1)
        assert found >= 0, "bloc absent ou hors séquence : %s" % block
        position = found
    assert "shell=True" not in text and "Invoke-Expression" not in text
    assert "\t" not in text, "tabulations : l'indentation est en espaces"
    assert text.count("{") == text.count("}"), "accolades déséquilibrées"
    assert text.count("(") == text.count(")"), "parenthèses déséquilibrées"
    assert "Set-ExecutionPolicy" not in text  # le script documente -ExecutionPolicy Bypass, il ne modifie rien
    assert "$ShimMark" in text and "n'a pas été créé par ce script" in text  # idempotence et refus d'écraser
    assert "(existant, réutilisé)" in text and "(inutilisable, recréé)" in text


def test_install_ps1_parses_with_powershell():
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        pytest.skip("PowerShell absent")
    script = ("$errors = $null; [System.Management.Automation.Language.Parser]::ParseFile('%s', [ref]$null, [ref]$errors)"
              " | Out-Null; if ($errors) { $errors | ForEach-Object { Write-Output $_.Message }; exit 1 }"
              % str(INSTALL_PS1).replace("'", "''"))
    proc = subprocess.run([pwsh, "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
