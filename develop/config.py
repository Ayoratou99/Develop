"""Configuration de Develop : réglages globaux et espaces de travail.

Disposition sur disque (jamais dans les dépôts de l'utilisateur, qui restent en lecture seule) ::

    ~/.develop/                         (ou $DEVELOP_HOME)
    ├── settings.json                   réglages globaux (port, dernier espace ouvert…)
    ├── venv/                           environnement créé par install.sh / install.ps1
    ├── webview/                        données de la fenêtre native « develop app » (selon la plateforme)
    └── workspaces/<id>/
        ├── workspace.json              dépôts, LLM, mode réseau
        ├── memory.sqlite3              index, relations, notes, conversations
        └── documents/                  documents déposés (copie locale)
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

RepoKind = Literal["frontend", "backend", "docs", "database", "mobile", "shared", "other"]
EngineKind = Literal["none", "ollama", "openai_compat"]
NetworkMode = Literal["strict", "local_network"]


class RepoConfig(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    path: str
    kind: RepoKind = "other"
    enabled: bool = True
    exclude: List[str] = Field(default_factory=list)  # motifs glob supplémentaires à exclure

    @field_validator("path")
    @classmethod
    def _expand(cls, v: str) -> str:
        return str(Path(os.path.expanduser(v)))


class LLMConfig(BaseModel):
    engine: EngineKind = "none"
    base_url: str = "http://127.0.0.1:11434"
    model: str = ""
    embedding_model: str = ""  # vide = recherche lexicale (FTS5) uniquement
    temperature: float = 0.2
    num_ctx: int = 8192  # fenêtre de contexte demandée au moteur
    think: bool = False  # réflexion interne (« thinking ») des modèles qui la supportent : plus précis mais bien plus lent
    max_context_chars: int = 24_000  # budget de contexte RAG injecté dans le prompt
    api_key: str = ""  # certains serveurs compatibles OpenAI exigent une clé factice


class WorkspaceConfig(BaseModel):
    id: str
    name: str
    repos: List[RepoConfig] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mode: NetworkMode = "strict"
    allowed_hosts: List[str] = Field(default_factory=list)
    language: str = "fr"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def repo_by_id(self, repo_id: str) -> Optional[RepoConfig]:
        for r in self.repos:
            if r.id == repo_id:
                return r
        return None


class GlobalSettings(BaseModel):
    active_workspace: Optional[str] = None
    host: str = "127.0.0.1"
    port: int = 8765
    open_browser: bool = True
    setup_completed: bool = False


# --------------------------------------------------------------------------- chemins

def develop_home() -> Path:
    root = os.environ.get("DEVELOP_HOME")
    return Path(root).expanduser() if root else Path.home() / ".develop"


def workspaces_dir() -> Path:
    return develop_home() / "workspaces"


def workspace_dir(workspace_id: str) -> Path:
    return workspaces_dir() / workspace_id


def db_path(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "memory.sqlite3"


def documents_dir(workspace_id: str) -> Path:
    return workspace_dir(workspace_id) / "documents"


def _settings_path() -> Path:
    return develop_home() / "settings.json"


def _atomic_write(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- réglages globaux

def load_settings() -> GlobalSettings:
    p = _settings_path()
    if not p.exists():
        return GlobalSettings()
    try:
        return GlobalSettings.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return GlobalSettings()


def save_settings(settings: GlobalSettings) -> None:
    _atomic_write(_settings_path(), settings.model_dump())


# --------------------------------------------------------------------------- espaces de travail

def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "projet"


def list_workspaces() -> List[WorkspaceConfig]:
    out: List[WorkspaceConfig] = []
    root = workspaces_dir()
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        cfg = d / "workspace.json"
        if cfg.is_file():
            try:
                out.append(WorkspaceConfig.model_validate(json.loads(cfg.read_text(encoding="utf-8"))))
            except (ValueError, OSError):
                continue
    return out


def load_workspace(workspace_id: str) -> Optional[WorkspaceConfig]:
    cfg = workspace_dir(workspace_id) / "workspace.json"
    if not cfg.is_file():
        return None
    try:
        return WorkspaceConfig.model_validate(json.loads(cfg.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return None


def save_workspace(ws: WorkspaceConfig) -> WorkspaceConfig:
    ws.updated_at = time.time()
    d = workspace_dir(ws.id)
    (d / "documents").mkdir(parents=True, exist_ok=True)
    _atomic_write(d / "workspace.json", ws.model_dump())
    return ws


def create_workspace(name: str) -> WorkspaceConfig:
    base = slugify(name)
    wid = base
    n = 2
    while workspace_dir(wid).exists():
        wid = "%s-%d" % (base, n)
        n += 1
    ws = WorkspaceConfig(id=wid, name=name)
    save_workspace(ws)
    settings = load_settings()
    if settings.active_workspace is None:
        settings.active_workspace = wid
        save_settings(settings)
    return ws


def delete_workspace(workspace_id: str) -> bool:
    """Supprime la mémoire Develop d'un espace (jamais les dépôts de l'utilisateur)."""
    import shutil

    d = workspace_dir(workspace_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    settings = load_settings()
    if settings.active_workspace == workspace_id:
        settings.active_workspace = None
        save_settings(settings)
    return True
