"""Interface commune des moteurs LLM locaux.

Un moteur = un serveur HTTP local (Ollama, LM Studio, llama.cpp, vLLM…). Develop ne
dépend que de cette interface, ce qui permet de changer de moteur sans toucher à la
mémoire du projet.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence


class LLMError(RuntimeError):
    """Erreur générique du moteur (réseau, réponse invalide…)."""


class LLMUnavailable(LLMError):
    """Aucun moteur configuré ou moteur injoignable : l'UI doit guider l'utilisateur."""


class ModelNotFound(LLMError):
    """Le modèle demandé n'est pas installé sur le moteur."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def to_openai(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    id: str = ""


@dataclass
class ChatMessage:
    role: str  # system | user | assistant | tool
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""  # pour role == "tool"
    name: str = ""  # nom de l'outil pour role == "tool"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ChatChunk:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    done: bool = False
    usage: Dict[str, int] = field(default_factory=dict)  # prompt_tokens, completion_tokens


@dataclass
class ModelInfo:
    name: str
    size_bytes: Optional[int] = None
    family: str = ""
    parameter_size: str = ""  # ex: "7.6B"
    quantization: str = ""  # ex: "Q4_K_M"
    context_length: Optional[int] = None
    capabilities: List[str] = field(default_factory=list)  # "completion", "tools", "embedding", "vision"
    embedding_dim: Optional[int] = None

    @property
    def supports_tools(self) -> bool:
        return "tools" in self.capabilities

    @property
    def is_embedding(self) -> bool:
        return "embedding" in self.capabilities

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "size_bytes": self.size_bytes, "family": self.family,
                "parameter_size": self.parameter_size, "quantization": self.quantization,
                "context_length": self.context_length, "capabilities": list(self.capabilities),
                "embedding_dim": self.embedding_dim, "supports_tools": self.supports_tools,
                "is_embedding": self.is_embedding}


@dataclass
class EngineStatus:
    ok: bool
    engine: str
    base_url: str
    version: str = ""
    error: str = ""
    models: List[ModelInfo] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "engine": self.engine, "base_url": self.base_url, "version": self.version,
                "error": self.error, "models": [m.to_dict() for m in self.models]}


class LLMClient:
    """Interface abstraite (asynchrone) d'un moteur local."""

    engine: str = "abstract"
    base_url: str = ""

    async def health(self) -> EngineStatus:
        raise NotImplementedError

    async def list_models(self) -> List[ModelInfo]:
        raise NotImplementedError

    async def model_info(self, name: str) -> ModelInfo:
        raise NotImplementedError

    def chat_stream(self, messages: Sequence[ChatMessage], model: str, *, temperature: float = 0.2,
                    num_ctx: Optional[int] = None, tools: Optional[Sequence[ToolSpec]] = None,
                    max_tokens: Optional[int] = None, think: Optional[bool] = None) -> AsyncIterator[ChatChunk]:
        raise NotImplementedError

    async def embed(self, texts: Sequence[str], model: str) -> List[List[float]]:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


class NullClient(LLMClient):
    """Moteur « aucun » : toutes les opérations expliquent qu'il faut configurer un moteur."""

    engine = "none"

    def __init__(self, base_url: str = ""):
        self.base_url = base_url

    async def health(self) -> EngineStatus:
        return EngineStatus(ok=False, engine="none", base_url="",
                            error="Aucun moteur d'IA configuré. Ouvrez les réglages pour choisir Ollama ou un serveur local.")

    async def list_models(self) -> List[ModelInfo]:
        return []

    async def model_info(self, name: str) -> ModelInfo:
        raise LLMUnavailable("Aucun moteur d'IA configuré.")

    async def chat_stream(self, messages, model, **kwargs):  # type: ignore[override]
        raise LLMUnavailable("Aucun moteur d'IA configuré : la recherche fonctionne, mais pas le chat.")
        yield ChatChunk()  # pragma: no cover - rend la fonction asynchrone-génératrice

    async def embed(self, texts, model):  # type: ignore[override]
        raise LLMUnavailable("Aucun moteur d'IA configuré : embeddings indisponibles.")


# --------------------------------------------------------------------------- protocole textuel d'outils

_TOOL_LINE = re.compile(r'^\s*\{\s*"tool"\s*:\s*"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"\s*,\s*"args"\s*:\s*(?P<args>\{.*\})\s*\}\s*$')


def parse_text_tool_calls(text: str) -> List[ToolCall]:
    """Repli pour les modèles sans appels d'outils natifs.

    Le modèle émet une ligne JSON de la forme ``{"tool": "search", "args": {"query": "…"}}``
    (éventuellement dans un bloc ```json). On en extrait les appels valides.
    """
    calls: List[ToolCall] = []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        m = _TOOL_LINE.match(line)
        if not m:
            continue
        try:
            args = json.loads(m.group("args"))
        except ValueError:
            continue
        if isinstance(args, dict):
            calls.append(ToolCall(name=m.group("name"), arguments=args, id="text-%d" % (len(calls) + 1)))
    return calls


def context_length_from_model_info(model_info: Dict[str, Any]) -> Optional[int]:
    """Ollama renvoie ``model_info`` avec une clé ``<architecture>.context_length``."""
    for key, value in (model_info or {}).items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return value
    return None
