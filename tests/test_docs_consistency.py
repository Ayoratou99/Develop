"""La documentation suit le moteur : nombre de canaux de recherche (README.md, docs/ARCHITECTURE.md),
textes de l'événement ``status`` du chat, sources de fiche et clés de ``done.budget`` (docs/API.md §6)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Set

from develop.chat.budget import Budget
from develop.chat.context_builder import CARD_CHANNEL, CARD_REASON
from develop.chat.orchestrator import STATUS_FOLLOWUP, STATUS_SEARCHING, STATUS_SMALLTALK, ChatOrchestrator
from develop.search.engine import CHANNELS

ROOT = Path(__file__).resolve().parent.parent
FRENCH_NUMBERS = {3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept", 8: "huit", 9: "neuf"}
# mot français qui désigne chaque canal du moteur dans la liste de l'étape 2 du README
README_CHANNEL_WORDS = {"identifier": "identifiants exacts", "route": "routes", "lexical": "lexicale",
                        "semantic": "sémantique", "graph": "graphe", "temporal": "notes par date"}
# même chose pour l'énumération de la question Q5 de docs/ARCHITECTURE.md (« canal (détail) »)
ARCHITECTURE_CHANNEL_WORDS = {"identifier": "identifiants exacts (", "route": "routes (", "lexical": "lexical (",
                              "semantic": "sémantique (", "graph": "structurel (", "temporal": "temporel ("}


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _channel_counts(text: str) -> Set[int]:
    """Nombres annoncés devant « canaux » (chiffres ou lettres), quelle que soit la casse."""
    words = {v: k for k, v in FRENCH_NUMBERS.items()}
    found: Set[int] = set()
    for m in re.finditer(r"\b(\d+|%s) canaux\b" % "|".join(words), text, re.IGNORECASE):
        token = m.group(1).lower()
        found.add(int(token) if token.isdigit() else words[token])
    return found


def _chat_section(api_md: str) -> str:
    start = api_md.index("## 6. Conversation (chat)")
    return api_md[start:api_md.index("## 7.", start)]


def test_architecture_counts_the_engine_channels():
    text = _read("docs/ARCHITECTURE.md")
    n = len(CHANNELS)
    assert "`search.hybrid` (%d canaux, RRF)" % n in text  # flux principal
    head = "%s canaux (spec §2)" % FRENCH_NUMBERS[n].capitalize()
    assert head in text  # Q5
    assert _channel_counts(text) == {n}
    enumeration = text[text.index(head):text.index("Dédup par fichier", text.index(head))]
    assert set(ARCHITECTURE_CHANNEL_WORDS) == set(CHANNELS)
    for channel in CHANNELS:
        assert ARCHITECTURE_CHANNEL_WORDS[channel] in enumeration, channel


def test_readme_lists_every_engine_channel():
    text = _read("README.md")
    n = len(CHANNELS)
    assert _channel_counts(text) == {n}
    m = re.search(r"2\. cherche sur %s canaux : (.*?) ;\n3\." % FRENCH_NUMBERS[n], text, re.DOTALL)
    assert m, "étape 2 du README introuvable"
    step = " ".join(m.group(1).split())
    assert set(README_CHANNEL_WORDS) == set(CHANNELS)
    for channel in CHANNELS:
        assert README_CHANNEL_WORDS[channel] in step, channel


def test_api_documents_status_texts_and_card_sources():
    chat = _chat_section(_read("docs/API.md"))
    assert STATUS_SEARCHING in chat
    assert STATUS_SMALLTALK in chat
    for part in STATUS_FOLLOWUP.split("%s"):  # « Question de suivi : contexte « … » conservé »
        assert part in chat, part
    assert "channels:['%s']" % CARD_CHANNEL in chat
    assert "« %s »" % CARD_REASON in chat


def test_api_documents_every_done_budget_key():
    chat = _chat_section(_read("docs/API.md"))
    m = re.search(r"`budget = \{([^}]*)\}`", chat)
    assert m, "description de done.budget introuvable"
    documented = {item.split(":")[0].strip() for item in m.group(1).split(",")}
    budget = Budget(num_ctx=8192, system_tokens=900, history_tokens_max=800, context_tokens_max=3000,
                    answer_reserve=1600, tools_mode="none", source="num_ctx 8192 (configuration)")
    emitted = ChatOrchestrator._budget_dict(object.__new__(ChatOrchestrator), budget, None, 0, 0)
    assert documented == set(emitted)
    # les clés que l'UI lit (views/chat.js : budgetFooter, toolsUsage) sont émises et documentées
    ui = _read("develop/web/static/views/chat.js")
    read_by_ui = {key for key in emitted if re.search(r"\b(?:b|budget)\.%s\b" % key, ui)}
    assert {"num_ctx", "context_tokens_used", "max_context_tokens", "source", "tools_tokens_used",
            "tools_tokens_max", "tool_rounds"} <= read_by_ui
