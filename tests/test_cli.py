"""Tests de la CLI ``develop`` (``develop.cli.main``) avec ``DEVELOP_HOME`` temporaire."""
from __future__ import annotations

import json
import shutil
import socket
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from develop.cli import main, pick_port

FIXTURE = Path(__file__).parent / "fixtures" / "sample_project"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.setenv("DEVELOP_HOME", str(tmp_path / "home"))
    project = tmp_path / "sample_project"
    shutil.copytree(FIXTURE, project)
    return SimpleNamespace(home=tmp_path / "home", project=project)


def _json_out(capsys: pytest.CaptureFixture) -> dict:  # type: ignore[type-arg]
    out = capsys.readouterr().out
    return json.loads(out)


def _init(home: SimpleNamespace, capsys: pytest.CaptureFixture) -> dict:  # type: ignore[type-arg]
    rc = main(["init", "Projet de test", "--json",
               "--repo", "backend=%s" % (home.project / "backend"),
               "--repo", "frontend=%s" % (home.project / "frontend"),
               "--repo", "docs=%s" % (home.project / "docs"),
               "--repo", "database=%s" % (home.project / "database")])
    assert rc == 0
    return _json_out(capsys)


# --------------------------------------------------------------------------- status / doctor

def test_status_json_without_workspace(home, capsys):
    assert main(["status", "--json"]) == 0
    data = _json_out(capsys)
    assert data["needs_setup"] is True and data["workspace"] is None
    assert data["engine"]["engine"] == "none" and data["offline"]["mode"] == "strict"


def test_status_text_without_workspace(home, capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Aucun espace de travail" in out and "Configuration requise" in out


def test_doctor_json_and_text(home, capsys):
    assert main(["doctor", "--json"]) == 0
    data = _json_out(capsys)
    assert {"profile", "engines", "advice"} <= set(data)
    assert data["profile"]["summary"] and isinstance(data["advice"]["recommendations"], list)
    assert {"recommended_engine", "ollama"} <= set(data["engines"])
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Machine :" in out and "Modèles" in out


def test_status_verify_offline_blocks_and_logs(home, capsys):
    assert main(["status", "--verify-offline", "--json"]) == 0
    data = _json_out(capsys)
    v = data["offline_verification"]
    assert v["blocked"] is True and v["logged"] is True and v["ok"] is True and v["error"] == ""
    assert data["offline"]["blocked_count"] >= 1
    assert main(["status", "--verify-offline"]) == 0
    assert "example.com" in capsys.readouterr().out


# --------------------------------------------------------------------------- init / repos / workspaces

def test_init_creates_and_activates_workspace(home, capsys):
    ws = _init(home, capsys)
    assert ws["id"] == "projet-de-test" and [r["kind"] for r in ws["repos"]] == ["backend", "frontend", "docs", "database"]
    assert (home.home / "workspaces" / ws["id"] / "workspace.json").is_file()
    settings = json.loads((home.home / "settings.json").read_text(encoding="utf-8"))
    assert settings["active_workspace"] == ws["id"]
    assert main(["status", "--json"]) == 0
    assert _json_out(capsys)["needs_setup"] is False
    assert main(["repos", "list"]) == 0
    out = capsys.readouterr().out
    assert "backend" in out and str(home.project / "backend") in out
    assert main(["workspaces", "list"]) == 0
    assert "* projet-de-test" in capsys.readouterr().out


def test_init_rejects_unknown_kind_or_missing_dir(home, capsys):
    assert main(["init", "X", "--repo", "serveur=%s" % home.project]) == 1
    assert "Erreur" in capsys.readouterr().err
    assert main(["init", "X", "--repo", "backend=/nonexistent/dir"]) == 1
    assert "n'existe pas" in capsys.readouterr().err


def test_repos_add_rm_and_workspaces_use_rm(home, capsys):
    ws = _init(home, capsys)
    assert main(["repos", "add", str(home.project / "services" / "notifier"), "--kind", "backend", "--name", "notifier"]) == 0
    assert "notifier" in capsys.readouterr().out
    assert main(["repos", "list", "--json"]) == 0
    repos = _json_out(capsys)
    assert [r["name"] for r in repos][-1] == "notifier" and len(repos) == 5
    assert main(["repos", "rm", repos[-1]["id"]]) == 0
    capsys.readouterr()
    assert main(["repos", "list", "--json"]) == 0 and len(_json_out(capsys)) == 4
    assert main(["repos", "rm", "inconnu"]) == 1
    capsys.readouterr()
    assert main(["init", "Autre"]) == 0
    assert main(["workspaces", "use", ws["id"]]) == 0
    capsys.readouterr()
    assert main(["workspaces", "list", "--json"]) == 0
    data = _json_out(capsys)
    assert data["active_workspace"] == ws["id"] and len(data["workspaces"]) == 2
    assert main(["workspaces", "rm", "autre"]) == 0
    assert main(["workspaces", "rm", "autre"]) == 1


# --------------------------------------------------------------------------- index / search / ask / impact

def test_index_search_ask_impact(home, capsys):
    ws = _init(home, capsys)
    backend_id = next(r["id"] for r in ws["repos"] if r["kind"] == "backend")

    assert main(["index"]) == 0
    out = capsys.readouterr().out
    assert "Indexation terminée" in out and "symboles" in out
    assert main(["index", "--json", "--quiet"]) == 0
    run = _json_out(capsys)
    assert run["status"] == "ok" and run["stats"]["files_indexed"] == 0 and run["stats"]["files_unchanged"] > 0

    assert main(["search", "orders", "--json"]) == 0
    data = _json_out(capsys)
    assert set(data) >= {"query", "identifiers", "results"} and data["results"]
    assert main(["search", "OrderService::confirm", "--limit", "3"]) == 0
    out = capsys.readouterr().out
    assert "Identifiants repérés" in out and "OrderService" in out

    assert main(["ask", "Qui appelle OrderService::confirm ?", "--search-only", "--json"]) == 0
    data = _json_out(capsys)
    assert [e["type"] for e in data["events"]] == ["status", "sources", "done"] and data["sources"]
    assert main(["ask", "Qui appelle OrderService::confirm ?", "--search-only"]) == 0
    assert "Sources :" in capsys.readouterr().out
    assert main(["ask", "Que fait Order ?"]) == 2  # aucun moteur configuré : erreur llm_unavailable
    assert "llm_unavailable" in capsys.readouterr().out

    assert main(["impact", "--file", "app/Services/OrderService.php", "--json"]) == 0
    report = _json_out(capsys)
    assert report["changed"][0]["path"] == "app/Services/OrderService.php" and report["affected"]
    assert main(["impact", "--file", "app/Services/OrderService.php", "--repo", backend_id]) == 0
    out = capsys.readouterr().out
    assert "Changements analysés : 1" in out
    assert main(["impact", "--file", "inconnu.php"]) == 1
    assert "introuvable" in capsys.readouterr().err
    assert main(["impact"]) == 1  # aucun dépôt Git
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Espace actif : Projet de test" in out and "fichiers" in out


def test_commands_require_a_workspace(home, capsys):
    assert main(["index"]) == 1 and "Aucun espace" in capsys.readouterr().err
    assert main(["search", "x"]) == 1
    assert main(["repos", "list"]) == 1


# --------------------------------------------------------------------------- divers

def test_main_without_command_prints_help(home, capsys):
    assert main([]) == 1
    assert "commande" in capsys.readouterr().out


def test_pick_port_skips_busy_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        assert pick_port("127.0.0.1", port) != port
        assert pick_port("127.0.0.1", port) in range(port + 1, port + 11)


# --------------------------------------------------------------------------- app (fenêtre native)

def test_app_without_pywebview_explains_and_falls_back_to_serve(home, capsys, monkeypatch):
    import develop.cli as cli

    monkeypatch.setitem(sys.modules, "webview", None)  # « import webview » lève ImportError
    calls = []
    monkeypatch.setattr(cli, "cmd_serve", lambda args: calls.append(args) or 0)
    assert main(["app", "--port", "8799"]) == 0
    out = capsys.readouterr().out
    assert "pywebview" in out and "develop-local[desktop]" in out and "develop serve" in out
    assert len(calls) == 1
    served = calls[0]
    assert served.host == "127.0.0.1" and served.port == 8799 and served.no_browser is False
    assert served.workspace is None


def test_app_fallback_propagates_serve_errors(home, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "webview", None)
    assert main(["app", "--workspace", "inconnu"]) == 1
    assert "introuvable" in capsys.readouterr().err


def test_app_serves_until_the_window_closes(home, capsys, monkeypatch):
    """Faux pywebview : la fenêtre « s'ouvre » quand /api/status répond, et sa fermeture arrête le serveur."""
    import urllib.request

    from develop.cli import _port_free

    seen = {}
    fake = types.ModuleType("webview")

    def create_window(title, url, **options):
        seen["title"], seen["url"], seen["options"] = title, url, options

    def start(**options):
        seen["start"] = options
        with urllib.request.urlopen(seen["url"] + "/api/status", timeout=5) as resp:
            seen["status"] = resp.status
            seen["body"] = json.loads(resp.read().decode("utf-8"))

    fake.create_window, fake.start = create_window, start
    monkeypatch.setitem(sys.modules, "webview", fake)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    assert main(["app", "--port", str(port)]) == 0
    assert seen["title"] == "Develop" and seen["url"] == "http://127.0.0.1:%d" % port
    assert seen["status"] == 200 and seen["body"]["needs_setup"] is True
    assert seen["options"]["text_select"] is True and seen["start"]["private_mode"] is False
    assert Path(seen["start"]["storage_path"]) == home.home / "webview"
    assert not any(t.name == "develop-server" and t.is_alive() for t in threading.enumerate())
    assert _port_free("127.0.0.1", port)
    out = capsys.readouterr().out
    assert "fenêtre ouverte" in out and "serveur arrêté" in out


def test_app_falls_back_to_browser_when_window_cannot_open(home, capsys, monkeypatch):
    """pywebview importable mais sans moteur (GTK/Qt/WebView2) : navigateur, serveur arrêté à Ctrl+C."""
    import develop.cli as cli

    fake = types.ModuleType("webview")
    fake.create_window = lambda *a, **k: None

    def start(**options):
        raise RuntimeError("You must have either QT or GTK with Python extensions installed")

    fake.start = start
    monkeypatch.setitem(sys.modules, "webview", fake)
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", lambda url: opened.append(url) or True)

    class FakeThread:
        """Thread déjà terminé : la boucle d'attente du Ctrl+C ne bloque pas."""

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    class FakeServer:
        should_exit = False

    monkeypatch.setattr(cli, "_start_server_thread", lambda app, host, port: (FakeServer(), FakeThread()))
    monkeypatch.setattr(cli, "_wait_for_server", lambda url, timeout, give_up=None: True)
    assert main(["app", "--port", "8798"]) == 0
    captured = capsys.readouterr()
    assert "Impossible d'ouvrir la fenêtre" in captured.err and "RuntimeError" in captured.err
    assert opened == ["http://127.0.0.1:8798"] and "Ctrl+C" in captured.out


def test_wait_for_server_gives_up_when_asked():
    from develop.cli import _wait_for_server

    assert _wait_for_server("http://127.0.0.1:9", timeout=5.0, give_up=lambda: True) is False
