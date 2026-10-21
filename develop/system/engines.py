"""Détection des moteurs d'IA présents sur la machine (Ollama, Docker, serveurs compatibles OpenAI).

Toutes les sondes sont locales (loopback), rapides (timeouts courts) et passent par la garde
hors ligne : aucune requête ne peut sortir de la machine.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from develop.llm import offline_guard
from develop.llm.base import ModelInfo
from develop.llm.ollama import OllamaClient
from develop.llm.openai_compat import OpenAICompatClient

# Ports usuels des serveurs locaux compatibles OpenAI
OPENAI_COMPAT_CANDIDATES = [
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("llama.cpp (llama-server)", "http://127.0.0.1:8080/v1"),
    ("vLLM / LocalAI", "http://127.0.0.1:8000/v1"),
    ("Jan", "http://127.0.0.1:1337/v1"),
]


@dataclass
class EngineProbe:
    engine: str  # "ollama" | "openai_compat"
    label: str
    base_url: str
    reachable: bool = False
    version: str = ""
    models: List[ModelInfo] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"engine": self.engine, "label": self.label, "base_url": self.base_url, "reachable": self.reachable,
                "version": self.version, "models": [m.to_dict() for m in self.models], "error": self.error}


@dataclass
class EnginesReport:
    ollama_binary: Optional[str]
    ollama_models_dir: str
    ollama: EngineProbe
    openai_servers: List[EngineProbe]
    docker_available: bool
    docker_running: bool
    docker_ollama_container: Optional[str]
    recommended_engine: str  # "ollama" | "openai_compat" | "none"
    recommended_base_url: str

    def to_dict(self) -> Dict[str, Any]:
        return {"ollama_binary": self.ollama_binary, "ollama_models_dir": self.ollama_models_dir,
                "ollama": self.ollama.to_dict(), "openai_servers": [p.to_dict() for p in self.openai_servers],
                "docker_available": self.docker_available, "docker_running": self.docker_running,
                "docker_ollama_container": self.docker_ollama_container,
                "recommended_engine": self.recommended_engine, "recommended_base_url": self.recommended_base_url}


# Sous Windows, un processus enfant ne doit jamais faire apparaître de console (fenêtre native « develop app »).
_NO_WINDOW: Dict[str, int] = ({"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
                              if hasattr(subprocess, "CREATE_NO_WINDOW") else {})


def _run(cmd: List[str], timeout: float = 4.0) -> Optional[str]:
    """Sortie standard d'une commande (décodée en UTF-8, caractères invalides remplacés : la console Windows
    n'est pas en UTF-8) ; ``None`` si la commande est absente, échoue, dépasse ``timeout`` ou est refusée."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=timeout, check=False, **_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def ollama_models_dir() -> str:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    return str(Path.home() / ".ollama" / "models")


def detect_docker() -> Dict[str, Any]:
    """Docker est-il installé, démarré, et un conteneur Ollama tourne-t-il ?"""
    available = shutil.which("docker") is not None
    running = False
    container: Optional[str] = None
    if available:
        out = _run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Ports}}"], timeout=5.0)
        if out is not None:
            running = True
            for line in out.splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and "ollama" in (parts[0] + " " + parts[1]).lower():
                    container = parts[0]
                    break
    return {"available": available, "running": running, "ollama_container": container}


async def probe_ollama(base_url: str = "http://127.0.0.1:11434", guard: Optional[offline_guard.OfflineGuard] = None,
                       transport: Optional[httpx.AsyncBaseTransport] = None, timeout: float = 1.5) -> EngineProbe:
    probe = EngineProbe(engine="ollama", label="Ollama", base_url=base_url)
    client = OllamaClient(base_url, guard=guard, transport=transport)
    try:
        status = await asyncio.wait_for(client.health(), timeout=timeout + 2.0)
        probe.reachable = status.ok
        probe.version = status.version
        probe.models = status.models
        probe.error = status.error
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 - une sonde ne plante jamais
        probe.error = "Ollama ne répond pas sur %s (%s)" % (base_url, exc.__class__.__name__)
    finally:
        await client.aclose()
    return probe


async def probe_openai_compat(label: str, base_url: str, guard: Optional[offline_guard.OfflineGuard] = None,
                              transport: Optional[httpx.AsyncBaseTransport] = None, timeout: float = 1.5) -> EngineProbe:
    probe = EngineProbe(engine="openai_compat", label=label, base_url=base_url)
    client = OpenAICompatClient(base_url, guard=guard, transport=transport)
    try:
        status = await asyncio.wait_for(client.health(), timeout=timeout + 2.0)
        probe.reachable = status.ok
        probe.models = status.models
        probe.error = status.error
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        probe.error = "%s ne répond pas sur %s (%s)" % (label, base_url, exc.__class__.__name__)
    finally:
        await client.aclose()
    return probe


async def detect_engines(guard: Optional[offline_guard.OfflineGuard] = None,
                         transport: Optional[httpx.AsyncBaseTransport] = None,
                         ollama_url: Optional[str] = None,
                         extra_openai_urls: Optional[List[str]] = None,
                         probe_docker: bool = True) -> EnginesReport:
    """Sonde tout en parallèle et recommande un moteur."""
    ollama_base = ollama_url or os.environ.get("OLLAMA_HOST", "").strip() or "http://127.0.0.1:11434"
    if ollama_base and not ollama_base.startswith("http"):
        ollama_base = "http://" + ollama_base
    candidates = list(OPENAI_COMPAT_CANDIDATES) + [("Serveur configuré", u) for u in (extra_openai_urls or [])]
    tasks = [probe_ollama(ollama_base, guard=guard, transport=transport)]
    tasks += [probe_openai_compat(label, url, guard=guard, transport=transport) for label, url in candidates]
    results = await asyncio.gather(*tasks)
    ollama_probe = results[0]
    openai_probes = list(results[1:])
    docker = detect_docker() if probe_docker else {"available": False, "running": False, "ollama_container": None}

    if ollama_probe.reachable:
        rec, rec_url = "ollama", ollama_base
    else:
        reachable = [p for p in openai_probes if p.reachable]
        if reachable:
            rec, rec_url = "openai_compat", reachable[0].base_url
        else:
            rec, rec_url = "none", ""
    return EnginesReport(
        ollama_binary=shutil.which("ollama"), ollama_models_dir=ollama_models_dir(), ollama=ollama_probe,
        openai_servers=openai_probes, docker_available=bool(docker["available"]), docker_running=bool(docker["running"]),
        docker_ollama_container=docker["ollama_container"], recommended_engine=rec, recommended_base_url=rec_url,
    )
