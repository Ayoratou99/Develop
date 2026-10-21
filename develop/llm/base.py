"""Interface commune des moteurs LLM locaux.

Un moteur = un serveur HTTP local (Ollama, LM Studio, llama.cpp, vLLM…). Develop ne
dépend que de cette interface, ce qui permet de changer de moteur sans toucher à la
mémoire du projet.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence


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


def canonical_model_name(name: str) -> str:
    """Nom de modèle comparable : un nom sans étiquette vaut ``:latest`` (``qwen3`` ↔ ``qwen3:latest``)."""
    name = (name or "").strip()
    return name if ":" in name else name + ":latest"


def model_is_loaded(model: str, loaded: Sequence[str]) -> bool:
    """Vrai si ``model`` figure dans la liste ``loaded`` du moteur (``loaded_models()``) ; un nom sans étiquette
    vaut ``:latest`` (``qwen3`` ↔ ``qwen3:latest``), mais ``qwen3:14b`` ne se confond pas avec ``qwen3:8b``.
    Même règle pour ``GET /api/llm/state`` et pour l'annonce « Chargement du modèle… » du chat."""
    if not model or not loaded:
        return False
    wanted = canonical_model_name(model)
    return any(canonical_model_name(m) == wanted for m in loaded)


def loaded_context_length(model: str, contexts: Mapping[str, Optional[int]]) -> Optional[int]:
    """Fenêtre (``num_ctx``) avec laquelle le moteur a chargé ``model`` d'après ``loaded_contexts()`` ; ``None`` si le
    modèle n'est pas en mémoire ou si le moteur ne dit pas sa fenêtre. Même règle de nom que :func:`model_is_loaded`."""
    if not model or not contexts:
        return None
    wanted = canonical_model_name(model)
    for name, ctx in contexts.items():
        if canonical_model_name(name) == wanted:
            try:
                value = int(ctx) if ctx is not None else 0
            except (TypeError, ValueError):
                return None
            return value if value > 0 else None
    return None


class LLMClient:
    """Interface abstraite (asynchrone) d'un moteur local."""

    engine: str = "abstract"
    base_url: str = ""
    # Vrai quand ``loaded_models()`` reflète réellement la mémoire du moteur (Ollama : /api/ps) ; faux quand
    # le moteur ne le dit pas (serveur compatible OpenAI, aucun) — l'interface ne bloque alors rien.
    reports_loaded_models: bool = False

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

    async def loaded_models(self) -> Optional[List[str]]:
        """Modèles actuellement chargés en mémoire par le moteur : ``[]`` quand il dit n'en avoir aucun, ``None``
        quand il ne sait pas le dire (serveur compatible OpenAI, aucun moteur) ; un moteur qui devrait le dire mais ne
        répond pas lève ``LLMError``. Le chat n'annonce « Chargement du modèle… » (et n'écarte sa mesure de vitesse
        de lecture) que sur une liste, vide ou non — jamais sur ``None`` ni sur une erreur."""
        return None

    async def loaded_contexts(self) -> Optional[Dict[str, Optional[int]]]:
        """``{modèle en mémoire: fenêtre (num_ctx) avec laquelle il a été chargé}`` ; ``None`` pour une fenêtre
        inconnue. Même contrat que :meth:`loaded_models` (``{}`` = rien en mémoire, ``None`` = le moteur ne sait pas
        le dire, ``LLMError`` = il devrait mais ne répond pas) ; par défaut ses modèles, sans fenêtre."""
        names = await self.loaded_models()
        return None if names is None else {name: None for name in names}

    async def warm(self, model: str, num_ctx: Optional[int] = None) -> bool:
        """Précharge un modèle en mémoire pour éviter l'attente au premier message (False si non supporté).

        ``num_ctx`` est la fenêtre que le chat demandera ensuite : un moteur qui recharge le modèle dès que
        les options diffèrent (Ollama) doit le charger avec cette même fenêtre, sinon le premier message
        paie un second chargement."""
        return False

    async def unload(self, model: str) -> bool:
        """Décharge un modèle de la mémoire du moteur (libère la RAM) ; False si non supporté ou refusé."""
        return False

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
