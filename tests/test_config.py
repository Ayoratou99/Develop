from __future__ import annotations

import json
from pathlib import Path

import pytest

from develop import config


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DEVELOP_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_paths_follow_develop_home(isolated_home: Path):
    assert config.develop_home() == isolated_home
    assert config.db_path("x") == isolated_home / "workspaces" / "x" / "memory.sqlite3"


def test_settings_roundtrip_and_defaults():
    s = config.load_settings()
    assert s.port == 8765 and s.active_workspace is None
    s.port = 9000
    config.save_settings(s)
    assert config.load_settings().port == 9000


def test_create_list_load_delete_workspace(isolated_home: Path):
    ws = config.create_workspace("Mon Projet")
    assert ws.id == "mon-projet"
    ws2 = config.create_workspace("Mon Projet")
    assert ws2.id == "mon-projet-2"
    assert [w.id for w in config.list_workspaces()] == ["mon-projet", "mon-projet-2"]
    assert config.load_settings().active_workspace == "mon-projet"
    ws.repos.append(config.RepoConfig(name="front", path="~/code/front", kind="frontend"))
    ws.llm.engine = "ollama"
    ws.llm.model = "qwen2.5-coder:7b"
    config.save_workspace(ws)
    loaded = config.load_workspace("mon-projet")
    assert loaded is not None and loaded.repos[0].kind == "frontend" and loaded.llm.model == "qwen2.5-coder:7b"
    assert "~" not in loaded.repos[0].path
    assert (isolated_home / "workspaces" / "mon-projet" / "documents").is_dir()
    assert config.delete_workspace("mon-projet") is True
    assert config.load_workspace("mon-projet") is None
    assert config.load_settings().active_workspace is None


def test_corrupt_workspace_is_skipped(isolated_home: Path):
    d = isolated_home / "workspaces" / "broken"
    d.mkdir(parents=True)
    (d / "workspace.json").write_text("{not json")
    assert config.list_workspaces() == []
    assert config.load_workspace("broken") is None


def test_workspace_json_is_plain_and_readable(isolated_home: Path):
    ws = config.create_workspace("Démo Été")
    raw = json.loads((config.workspace_dir(ws.id) / "workspace.json").read_text(encoding="utf-8"))
    assert raw["name"] == "Démo Été" and raw["mode"] == "strict"
