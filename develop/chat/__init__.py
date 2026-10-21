"""Orchestrateur du chat de Develop : budget de contexte, assembleur de sources, outils, événements SSE.

Le modèle ne lit jamais tout le projet : :mod:`develop.chat.budget` calcule ce qui peut lui être
envoyé, :mod:`develop.chat.context_builder` assemble les passages cités [S1], [S2]…,
:mod:`develop.chat.tools` expose des outils locaux bornés et :mod:`develop.chat.orchestrator`
enchaîne recherche → contexte → boucle d'outils → réponse sous forme d'événements SSE.
"""
from develop.chat.budget import Budget, compute_budget, estimate_tokens  # noqa: F401
from develop.chat.context_builder import (  # noqa: F401
    ContextPack,
    LocalHit,
    Source,
    add_tool_sources,
    build_context,
    render_sources_as_text,
    render_sources_for_model,
)
from develop.chat.orchestrator import ChatEvent, ChatOrchestrator, find_invalid_citations  # noqa: F401
from develop.chat.prompts import (  # noqa: F401
    HISTORY_SUMMARY_PROMPT,
    SYSTEM_PROMPT,
    TOOLS_TEXT_PROTOCOL,
    build_system_prompt,
)
from develop.chat.tools import TOOL_SPECS, ToolExecutor, ToolResult  # noqa: F401

__all__ = [
    "Budget",
    "ChatEvent",
    "ChatOrchestrator",
    "ContextPack",
    "HISTORY_SUMMARY_PROMPT",
    "LocalHit",
    "SYSTEM_PROMPT",
    "Source",
    "TOOLS_TEXT_PROTOCOL",
    "TOOL_SPECS",
    "ToolExecutor",
    "ToolResult",
    "add_tool_sources",
    "build_context",
    "build_system_prompt",
    "compute_budget",
    "estimate_tokens",
    "find_invalid_citations",
    "render_sources_as_text",
    "render_sources_for_model",
]
