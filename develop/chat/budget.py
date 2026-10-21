"""Budget de contexte du chat : la fenêtre du modèle est un budget à répartir, pas un plafond à remplir.

Le LLM ne lit jamais tout le projet. Pour chaque question, Develop calcule :

- un ``num_ctx`` effectif = min(fenêtre réelle du modèle, valeur configurée, plafond dérivé de la
  RAM de la machine) — remplir 128k sur un portable prend des minutes et sature la mémoire ;
- une répartition de ce ``num_ctx`` entre consignes système (définitions d'outils natifs
  comprises), question, historique de conversation, contexte récupéré (code, documents, notes),
  résultats d'outils et réserve pour la réponse.

Invariant : ``system_tokens + question_tokens + history_tokens_max + context_tokens_max +
tools_tokens_max + answer_reserve <= num_ctx``. Si les consignes avec outils ne laissent pas de
place à au moins un résultat d'outil, les outils sont désactivés (``tools_mode = "none"``) et
leur part revient au contexte. Les consignes système sont mesurées sur les vrais gabarits de
:mod:`develop.chat.prompts` (ou fournies, déjà mesurées, par l'orchestrateur), jamais devinées.

Tout est estimé en tokens ≈ caractères / 4 (code ≈ caractères / 3,5), sans tokenizer.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

DEFAULT_CONTEXT_LENGTH = 8192  # fenêtre supposée quand le moteur ne la communique pas
MIN_NUM_CTX = 2048
MIN_ANSWER_RESERVE = 1024
MAX_ANSWER_RESERVE = 6000
ANSWER_SHARE = 0.30
HISTORY_SHARE = 0.12
MIN_HISTORY_TOKENS = 1024
MIN_CONTEXT_TOKENS = 256  # plancher des plafonds (max_context_chars, demande de l'UI), jamais de la partition
CONTEXT_FIRST_TOKENS = 1024  # le contexte récupéré prend jusqu'à 1024 tokens avant de partager avec les outils
MIN_TOOL_BUDGET = 200  # en dessous, un résultat d'outil ne vaut pas son coût : outil non exécuté / outils désactivés
DEFAULT_TOOL_ROUNDS = 4
SYSTEM_MARGIN_TOKENS = 64  # marge de l'estimation des consignes (nom du projet, liste des dépôts, tokenizer réel)
TOOL_SPEC_OVERHEAD_TOKENS = 4  # cadre ajouté par le gabarit du moteur autour de chaque définition d'outil
TOOLS_MODES = ("native", "text", "none")

# Part du « reste » (après système, question, historique et réserve de réponse) consacrée au
# contexte récupéré, selon le type de question. Le reste sert aux résultats d'outils.
INTENT_SHARES = {
    "factual": 0.35,
    "fields": 0.35,
    "howto": 0.35,
    "who_calls": 0.40,
    "impact": 0.50,
    "decision": 0.35,
    "broad": 0.45,
}
DEFAULT_SHARE = 0.35

# Plafond de num_ctx selon la RAM totale (Go) : (borne exclusive, plafond).
RAM_CAPS = ((8, 4096), (16, 8192), (32, 16384), (64, 32768))
MAX_RAM_CAP = 65536
ACCELERATOR_BONUS_GB = 24  # au-delà, le cache KV tient sur l'accélérateur : plafond ×2


def estimate_tokens(text: Any, is_code: bool = False) -> int:
    """Estimation grossière du nombre de tokens (prose ≈ 4 caractères/token, code ≈ 3,5)."""
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    per_token = 3.5 if is_code else 4.0
    return int(math.ceil(len(text) / per_token))


def tool_specs_tokens(tool_specs: Optional[Sequence[Any]] = None) -> int:
    """Coût estimé des définitions d'outils natifs telles qu'envoyées au moteur à chaque appel (JSON OpenAI).

    Ces définitions ne sont dans aucun message mais occupent bien la fenêtre : le gabarit du
    moteur les insère dans le prompt système. ``None`` = :data:`develop.chat.tools.TOOL_SPECS`.
    """
    if tool_specs is None:
        from develop.chat.tools import TOOL_SPECS  # import paresseux (tools importe budget)
        tool_specs = TOOL_SPECS
    payload: List[Any] = []
    for spec in tool_specs:
        to_openai = getattr(spec, "to_openai", None)
        payload.append(to_openai() if callable(to_openai) else spec)
    if not payload:
        return 0
    return estimate_tokens(json.dumps(payload, ensure_ascii=False, default=str)) + TOOL_SPEC_OVERHEAD_TOKENS * len(payload)


def system_prompt_tokens(tools_mode: str, num_ctx: int = DEFAULT_CONTEXT_LENGTH, tool_rounds: int = DEFAULT_TOOL_ROUNDS,
                         workspace_name: str = "", repos_summary: str = "", today: str = "") -> int:
    """Coût estimé des consignes système pour un mode d'outils, définitions natives comprises.

    Mesuré sur le vrai gabarit (:func:`develop.chat.prompts.build_system_prompt`) plus une marge
    pour les parties variables (nom du projet, dépôts) quand elles ne sont pas fournies.
    """
    from develop.chat.prompts import build_system_prompt  # import paresseux (prompts → tools → budget)
    mode = tools_mode if tools_mode in TOOLS_MODES else "none"
    shape = SimpleNamespace(num_ctx=int(num_ctx or 0), tool_rounds_max=int(tool_rounds or DEFAULT_TOOL_ROUNDS))
    text = build_system_prompt(mode, shape, workspace_name, repos_summary, today or "2026-01-01")
    tokens = estimate_tokens(text) + SYSTEM_MARGIN_TOKENS
    if mode == "native":
        tokens += tool_specs_tokens()
    return tokens


@dataclass
class Budget:
    """Répartition (en tokens) de la fenêtre de contexte pour une question.

    ``system_tokens`` comprend les définitions d'outils natifs quand ``tools_mode == "native"`` ;
    ``tools_tokens_max`` est la part réservée aux résultats d'outils (0 sans outils).
    """

    num_ctx: int
    system_tokens: int
    history_tokens_max: int
    context_tokens_max: int
    answer_reserve: int
    tool_rounds_max: int = DEFAULT_TOOL_ROUNDS
    source: str = ""  # explication en français du calcul (affichée dans l'UI / les journaux)
    tools_mode: str = "native"  # mode d'outils effectif : native, text ou none
    question_tokens: int = 0
    tools_tokens_max: int = 0

    @property
    def total(self) -> int:
        """Somme des parts : toujours ≤ ``num_ctx``."""
        return (self.system_tokens + self.question_tokens + self.history_tokens_max + self.context_tokens_max
                + self.tools_tokens_max + self.answer_reserve)

    def to_dict(self) -> dict:
        return {
            "num_ctx": self.num_ctx,
            "system_tokens": self.system_tokens,
            "question_tokens": self.question_tokens,
            "history_tokens_max": self.history_tokens_max,
            "context_tokens_max": self.context_tokens_max,
            "tools_tokens_max": self.tools_tokens_max,
            "answer_reserve": self.answer_reserve,
            "tool_rounds_max": self.tool_rounds_max,
            "tools_mode": self.tools_mode,
            "source": self.source,
        }


def ram_cap(ram_total_gb: Optional[float]) -> Optional[int]:
    """Plafond de ``num_ctx`` dérivé de la RAM totale ; ``None`` si la RAM est inconnue."""
    if ram_total_gb is None:
        return None
    try:
        ram = float(ram_total_gb)
    except (TypeError, ValueError):
        return None
    if ram <= 0:
        return None
    for limit, cap in RAM_CAPS:
        if ram < limit:
            return cap
    return MAX_RAM_CAP


def intent_share(intent: Optional[str]) -> float:
    return INTENT_SHARES.get((intent or "").strip().lower(), DEFAULT_SHARE)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def effective_num_ctx(cfg: Any, model_info: Any = None, profile: Any = None) -> Tuple[int, List[str]]:
    """``num_ctx`` effectif = max(2048, min(fenêtre du modèle, configuration, plafond RAM)) + explication."""
    parts: List[str] = []
    model_window = getattr(model_info, "context_length", None) if model_info is not None else None
    model_window = _as_int(model_window)
    window = model_window if model_window > 0 else DEFAULT_CONTEXT_LENGTH
    if model_window > 0:
        parts.append("fenêtre du modèle %d" % window)
    else:
        parts.append("fenêtre du modèle inconnue → %d par défaut" % window)
    candidates: List[int] = [window]

    cfg_ctx = _as_int(getattr(cfg, "num_ctx", 0))
    if cfg_ctx > 0:
        candidates.append(cfg_ctx)
        parts.append("configuration %d" % cfg_ctx)

    if profile is not None:
        ram = getattr(profile, "ram_total_gb", None)
        cap = ram_cap(ram)
        if cap is not None:
            desc = "plafond RAM %d (%g Go)" % (cap, float(ram))
            accel = getattr(profile, "accelerator_memory_gb", None)
            try:
                accel_val = float(accel) if accel is not None else None
            except (TypeError, ValueError):
                accel_val = None
            if accel_val is not None and accel_val >= ACCELERATOR_BONUS_GB:
                boosted = min(cap * 2, window)
                if boosted > cap:
                    desc += ", ×2 grâce à l'accélérateur (%g Go) → %d" % (accel_val, boosted)
                cap = boosted
            candidates.append(cap)
            parts.append(desc)
        else:
            parts.append("RAM inconnue : pas de plafond matériel")

    return max(MIN_NUM_CTX, min(candidates)), parts


def _split(num_ctx: int, system: int, question: int, answer: int, history_default: int, share: float,
           with_tools: bool) -> Tuple[int, int, int, int]:
    """Répartit ce qui reste après système, question et réponse ; renvoie (historique, reste, contexte, outils).

    L'historique cède en premier sur une petite fenêtre : il garde au plus la moitié de ce qui
    reste au-delà de :data:`MIN_CONTEXT_TOKENS`. Sans outils, tout le reste va au contexte.
    """
    free = num_ctx - system - question - answer
    history = min(history_default, max(0, free - MIN_CONTEXT_TOKENS) // 2)
    remaining = max(0, free - history)
    if not with_tools:
        return history, remaining, remaining, 0
    context = max(min(CONTEXT_FIRST_TOKENS, remaining), int(remaining * share))
    return history, remaining, context, remaining - context


def compute_budget(cfg: Any, model_info: Any = None, profile: Any = None, intent: str = "factual",
                   override_context_tokens: Optional[int] = None, tools_mode: str = "native",
                   system_tokens: Optional[int] = None, question_tokens: int = 0) -> Budget:
    """Calcule le budget pour une question.

    ``cfg`` est un :class:`develop.config.LLMConfig` (``num_ctx``, ``max_context_chars``),
    ``model_info`` un :class:`develop.llm.base.ModelInfo` (``context_length``) ou ``None``,
    ``profile`` un :class:`develop.system.profile.MachineProfile` ou ``None``.

    ``tools_mode`` est le mode d'outils souhaité (``native``, ``text`` ou ``none``) ;
    ``system_tokens`` le coût mesuré des consignes système dans ce mode (définitions d'outils
    natifs comprises), sinon il est estimé sur les gabarits ; ``question_tokens`` le coût de la
    question de l'utilisateur. Si les outils ne laisseraient pas :data:`MIN_TOOL_BUDGET` tokens à
    leurs résultats, le budget revient en mode ``none`` (``tool_rounds_max = 0``) et le dit dans
    ``source`` ; un ``system_tokens`` fourni pour un mode avec outils est alors remplacé par
    l'estimation sans outils (l'appelant remesure s'il veut l'exactitude).
    """
    num_ctx, parts = effective_num_ctx(cfg, model_info, profile)
    rounds = DEFAULT_TOOL_ROUNDS
    mode = tools_mode if tools_mode in TOOLS_MODES else "none"
    question = max(0, _as_int(question_tokens))
    share = intent_share(intent)

    # ----- parts fixes -----
    answer_reserve = max(MIN_ANSWER_RESERVE, min(MAX_ANSWER_RESERVE, int(num_ctx * ANSWER_SHARE)))
    history_default = min(max(MIN_HISTORY_TOKENS, int(num_ctx * HISTORY_SHARE)), num_ctx // 4)
    measured = system_tokens is not None and _as_int(system_tokens) >= 0
    system = _as_int(system_tokens) if measured else system_prompt_tokens(mode, num_ctx, rounds)
    system_desc = "mesurées" if measured else "estimées"

    # ----- répartition, avec repli sans outils si la fenêtre ne les porte pas -----
    history, remaining, context, tools = _split(num_ctx, system, question, answer_reserve, history_default, share,
                                                mode != "none")
    downgraded = ""
    if mode != "none" and tools < MIN_TOOL_BUDGET:
        downgraded = ("outils désactivés : avec leurs consignes (%d tokens) il resterait %d token(s) pour leurs résultats"
                      % (system, tools))
        mode = "none"
        system = system_prompt_tokens(mode, num_ctx, 0)
        system_desc = "estimées"
        history, remaining, context, tools = _split(num_ctx, system, question, answer_reserve, history_default, share,
                                                    False)
    if mode == "none":
        rounds = 0

    if mode == "none":
        context_desc = "contexte ≤ %d (tout le reste)" % context
    else:
        context_desc = "contexte ≤ %d (%s : %d %% du reste %d)" % (context, intent or "factual", int(share * 100), remaining)

    max_chars = _as_int(getattr(cfg, "max_context_chars", 0))
    if max_chars > 0:
        chars_cap = max(MIN_CONTEXT_TOKENS, max_chars // 4)
        if chars_cap < context:
            context = chars_cap
            context_desc += ", borné par max_context_chars (%d caractères)" % max_chars
    if override_context_tokens is not None:
        override = _as_int(override_context_tokens)
        if override > 0:
            override = max(MIN_CONTEXT_TOKENS, override)
            if override < context:
                context = override
                context_desc += ", borné par la demande (%d)" % override

    if num_ctx - system - question - answer_reserve < 0:
        parts.append("question trop longue pour la fenêtre : le moteur tronquera")
    source = ("num_ctx %d = min(%s) ; système %d (%s, mode %s), question %d, historique ≤ %d, %s, outils ≤ %d, "
              "réponse %d, %d tours d'outils max" % (num_ctx, ", ".join(parts), system, system_desc, mode, question,
                                                     history, context_desc, tools, answer_reserve, rounds))
    if downgraded:
        source += " ; " + downgraded
    return Budget(num_ctx=num_ctx, system_tokens=system, history_tokens_max=history, context_tokens_max=context,
                  answer_reserve=answer_reserve, tool_rounds_max=rounds, source=source, tools_mode=mode,
                  question_tokens=question, tools_tokens_max=tools)
