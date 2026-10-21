"""Construction du client LLM d'après la configuration de l'espace de travail."""
from __future__ import annotations

from typing import Optional

import httpx

from develop.config import LLMConfig
from develop.llm import offline_guard
from develop.llm.base import LLMClient, NullClient
from develop.llm.ollama import OllamaClient
from develop.llm.openai_compat import OpenAICompatClient


def create_client(cfg: LLMConfig, guard: Optional[offline_guard.OfflineGuard] = None,
                  transport: Optional[httpx.AsyncBaseTransport] = None) -> LLMClient:
    if cfg.engine == "ollama":
        return OllamaClient(cfg.base_url or "http://127.0.0.1:11434", guard=guard, transport=transport)
    if cfg.engine == "openai_compat":
        return OpenAICompatClient(cfg.base_url or "http://127.0.0.1:1234/v1", api_key=cfg.api_key, guard=guard, transport=transport)
    return NullClient()
