"""Budget de contexte du chat : la fenêtre du modèle est un budget à répartir, pas un plafond à remplir.

Le LLM ne lit jamais tout le projet. Pour chaque question, Develop calcule :

- un ``num_ctx`` effectif = min(fenêtre réelle du modèle, valeur configurée, plafond dérivé de la
  RAM de la machine) — remplir 128k sur un portable prend des minutes et sature la mémoire ;
- une répartition de ce ``num_ctx`` entre consignes système (définitions d'outils natifs
  comprises), question, historique de conversation, contexte récupéré (code, documents, notes),
  résultats d'outils et réserve pour la réponse.

Répartition (le contexte récupéré est la part qui fait la qualité des réponses, elle est servie
en premier) :

- consignes système = tokens estimés du vrai gabarit (:func:`develop.chat.prompts.build_system_prompt`)
  + 10 % de marge, plafonnés à :data:`SYSTEM_TOKENS_CAP` ; en mode natif s'ajoutent les définitions
  d'outils, réellement envoyées à chaque appel ;
- historique ≤ 10 % de ``num_ctx``, entre :data:`MIN_HISTORY_TOKENS` et :data:`MAX_HISTORY_TOKENS` ;
- réserve de réponse = 20 % de ``num_ctx``, entre :data:`MIN_ANSWER_RESERVE` et
  :data:`MAX_ANSWER_RESERVE` (une réponse citée tient en 3k tokens ; le modèle peut être relancé) ;
- contexte récupéré = 80 % de ce qui reste (90 % pour une analyse d'impact ou une question large),
  le solde allant aux résultats d'outils ; sans outils, tout le reste va au contexte.

Repères (mode natif, question factuelle) : ≈ 3,3k tokens de contexte à ``num_ctx`` 8192, ≈ 8k à
16384, ≈ 20k à 32768.

Invariant : ``system_tokens + question_tokens + history_tokens_max + context_tokens_max +
tools_tokens_max + answer_reserve <= num_ctx``. Si les consignes avec outils ne laissent pas de
place à au moins un résultat d'outil, les outils sont désactivés (``tools_mode = "none"``) et
leur part revient au contexte. Les consignes système sont mesurées sur les vrais gabarits de
:mod:`develop.chat.prompts` (ou fournies, déjà mesurées, par l'orchestrateur), jamais devinées.

Borne de latence : la fenêtre n'est pas le seul plafond, le temps de lecture l'est aussi. Le
modèle lit le prompt à ``prefill_tps`` tokens/s (mesurés par l'orchestrateur, sinon estimés par
:func:`develop.llm.advisor.estimate_prefill_tps`) et la première lettre de la réponse doit
arriver en ``cfg.first_token_target_s`` (25 s par défaut) : le contexte récupéré ne dépasse jamais
``prefill_tps × first_token_target_s − consignes − question − historique``, avec un plancher de
:data:`LATENCY_CONTEXT_FLOOR` tokens (en dessous, la qualité prime sur le délai) et pour plafond
le contexte calculé ci-dessus. Sans ``prefill_tps``, ou avec ``first_token_target_s`` à 0 (objectif
désactivé), aucune borne de latence.

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
ANSWER_SHARE = 0.20  # réserve de réponse : 20 % de num_ctx…
MIN_ANSWER_RESERVE = 1024  # … jamais moins (une réponse citée tient rarement en moins)…
MAX_ANSWER_RESERVE = 3072  # … jamais plus (au-delà, la fenêtre sert mieux au contexte ; le modèle peut être relancé)
HISTORY_SHARE = 0.10  # historique : 10 % de num_ctx…
MIN_HISTORY_TOKENS = 768  # … au moins deux tours courts…
MAX_HISTORY_TOKENS = 2048  # … au plus 2k (le résumé en une ligne couvre le reste)
MIN_CONTEXT_TOKENS = 256  # plancher des plafonds (max_context_chars, demande de l'UI), jamais de la partition
CONTEXT_FIRST_TOKENS = 1024  # le contexte récupéré prend jusqu'à 1024 tokens avant de partager avec les outils
MIN_TOOL_BUDGET = 200  # en dessous, un résultat d'outil ne vaut pas son coût : outil non exécuté / outils désactivés
DEFAULT_TOOL_ROUNDS = 4
SYSTEM_MARGIN_SHARE = 0.10  # marge de l'estimation des consignes (nom du projet, liste des dépôts, tokenizer réel)
SYSTEM_TOKENS_CAP = 1500  # plafond des consignes textuelles estimées (hors définitions d'outils natifs)
TOOL_SPEC_OVERHEAD_TOKENS = 4  # cadre ajouté par le gabarit du moteur autour de chaque définition d'outil
TOOLS_MODES = ("native", "text", "none")
DEFAULT_FIRST_TOKEN_TARGET_S = 25.0  # même défaut que LLMConfig.first_token_target_s (cfg absente ou ancienne)
LATENCY_CONTEXT_FLOOR = 1500  # la borne de latence ne descend jamais le contexte sous 1 500 tokens

# Part du « reste » (après système, question, historique et réserve de réponse) consacrée au
# contexte récupéré, selon le type de question. Le solde sert aux résultats d'outils.
INTENT_SHARES = {
    "factual": 0.80,
    "fields": 0.80,
    "howto": 0.80,
    "who_calls": 0.85,
    "impact": 0.90,
    "decision": 0.80,
    "broad": 0.90,
}
DEFAULT_SHARE = 0.80

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


def system_text_tokens(tools_mode: str, num_ctx: int = DEFAULT_CONTEXT_LENGTH, tool_rounds: int = DEFAULT_TOOL_ROUNDS,
                       workspace_name: str = "", repos_summary: str = "", today: str = "") -> int:
    """Coût estimé des consignes textuelles seules : vrai gabarit + :data:`SYSTEM_MARGIN_SHARE`, ≤ :data:`SYSTEM_TOKENS_CAP`.

    La marge couvre les parties variables (nom du projet, dépôts) quand elles ne sont pas fournies
    et l'écart entre l'estimateur et le tokenizer réel. Les définitions d'outils natifs ne sont pas
    comprises (voir :func:`system_prompt_tokens`).
    """
    from develop.chat.prompts import build_system_prompt  # import paresseux (prompts → tools → budget)
    mode = tools_mode if tools_mode in TOOLS_MODES else "none"
    shape = SimpleNamespace(num_ctx=int(num_ctx or 0), tool_rounds_max=int(tool_rounds or DEFAULT_TOOL_ROUNDS))
    text = build_system_prompt(mode, shape, workspace_name, repos_summary, today or "2026-01-01")
    tokens = estimate_tokens(text)
    # marge arrondie par excès en arithmétique entière : 890 × 1,1 vaut 979,0000000001 en flottant, ce qui ferait
    # un token de plus avec math.ceil (et une estimation qui dépend du bruit du flottant)
    margin = (tokens * int(SYSTEM_MARGIN_SHARE * 100) + 99) // 100
    return min(SYSTEM_TOKENS_CAP, tokens + margin)


def system_prompt_tokens(tools_mode: str, num_ctx: int = DEFAULT_CONTEXT_LENGTH, tool_rounds: int = DEFAULT_TOOL_ROUNDS,
                         workspace_name: str = "", repos_summary: str = "", today: str = "") -> int:
    """Coût estimé des consignes système pour un mode d'outils, définitions natives comprises.

    = :func:`system_text_tokens` (gabarit mesuré + 10 % de marge, plafonné) + :func:`tool_specs_tokens`
    en mode ``native`` (les définitions sont réellement envoyées, elles ne sont jamais plafonnées).
    """
    mode = tools_mode if tools_mode in TOOLS_MODES else "none"
    tokens = system_text_tokens(mode, num_ctx, tool_rounds, workspace_name, repos_summary, today)
    if mode == "native":
        tokens += tool_specs_tokens()
    return tokens


@dataclass
class Budget:
    """Répartition (en tokens) de la fenêtre de contexte pour une question.

    ``system_tokens`` comprend les définitions d'outils natifs quand ``tools_mode == "native"`` ;
    ``tools_tokens_max`` est la part réservée aux résultats d'outils (0 sans outils) ; ``prefill_tps`` la
    vitesse de lecture (tokens/s) retenue pour la borne de latence (``None`` : pas de borne) et
    ``latency_capped`` dit si cette borne a réduit le contexte.
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
    prefill_tps: Optional[float] = None
    latency_capped: bool = False
    first_token_target_s: float = DEFAULT_FIRST_TOKEN_TARGET_S

    @property
    def total(self) -> int:
        """Somme des parts : toujours ≤ ``num_ctx``."""
        return (self.system_tokens + self.question_tokens + self.history_tokens_max + self.context_tokens_max
                + self.tools_tokens_max + self.answer_reserve)

    @property
    def prompt_tokens_max(self) -> int:
        """Ce que le modèle lit avant la première lettre : consignes, question, historique et contexte (plafonds)."""
        return self.system_tokens + self.question_tokens + self.history_tokens_max + self.context_tokens_max

    @property
    def expected_first_token_s(self) -> Optional[float]:
        """Délai estimé (s) avant la première lettre = :attr:`prompt_tokens_max` ÷ ``prefill_tps`` ; ``None`` sans vitesse."""
        return read_time_s(self.prompt_tokens_max, self.prefill_tps)

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
            "prefill_tps": self.prefill_tps,
            "latency_capped": self.latency_capped,
        }


def read_time_s(tokens: int, prefill_tps: Optional[float]) -> Optional[float]:
    """Secondes nécessaires pour lire ``tokens`` à ``prefill_tps`` tokens/s (``None`` sans vitesse valable)."""
    try:
        tps = float(prefill_tps) if prefill_tps is not None else 0.0
    except (TypeError, ValueError):
        return None
    if tps <= 0:
        return None
    return max(0, int(tokens or 0)) / tps


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


def answer_reserve_for(num_ctx: int) -> int:
    """Réserve de réponse : :data:`ANSWER_SHARE` de ``num_ctx`` entre :data:`MIN_ANSWER_RESERVE` et :data:`MAX_ANSWER_RESERVE`."""
    return max(MIN_ANSWER_RESERVE, min(MAX_ANSWER_RESERVE, int(num_ctx * ANSWER_SHARE)))


def history_tokens_for(num_ctx: int) -> int:
    """Historique : :data:`HISTORY_SHARE` de ``num_ctx`` entre :data:`MIN_HISTORY_TOKENS` et :data:`MAX_HISTORY_TOKENS`."""
    return min(MAX_HISTORY_TOKENS, max(MIN_HISTORY_TOKENS, int(num_ctx * HISTORY_SHARE)))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def latency_context_cap(prefill_tps: Optional[float], target_s: float, system: int, question: int,
                        history: int) -> Optional[int]:
    """Tokens de contexte lisibles en ``target_s`` secondes à ``prefill_tps`` tokens/s une fois consignes, question
    et historique lus, jamais moins que :data:`LATENCY_CONTEXT_FLOOR` ; ``None`` sans vitesse ni objectif valables."""
    tps = _as_float(prefill_tps)
    target = _as_float(target_s)
    if tps <= 0 or target <= 0:
        return None
    return max(LATENCY_CONTEXT_FLOOR, int(tps * target) - system - question - history)


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
    reste au-delà de :data:`MIN_CONTEXT_TOKENS`. Sans outils, tout le reste va au contexte ; avec
    outils, le contexte prend ``share`` du reste (et jamais moins que :data:`CONTEXT_FIRST_TOKENS`
    si le reste le permet), le solde allant aux résultats d'outils.
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
                   system_tokens: Optional[int] = None, question_tokens: int = 0,
                   prefill_tps: Optional[float] = None) -> Budget:
    """Calcule le budget pour une question.

    ``cfg`` est un :class:`develop.config.LLMConfig` (``num_ctx``, ``max_context_chars``,
    ``first_token_target_s``), ``model_info`` un :class:`develop.llm.base.ModelInfo`
    (``context_length``) ou ``None``, ``profile`` un :class:`develop.system.profile.MachineProfile`
    ou ``None``.

    ``tools_mode`` est le mode d'outils souhaité (``native``, ``text`` ou ``none``) ;
    ``system_tokens`` le coût mesuré des consignes système dans ce mode (définitions d'outils
    natifs comprises), utilisé tel quel, sinon il est estimé sur les gabarits avec 10 % de marge
    (:func:`system_prompt_tokens`) ; ``question_tokens`` le coût de la question de l'utilisateur.
    Si les outils ne laisseraient pas :data:`MIN_TOOL_BUDGET` tokens à leurs résultats, le budget
    revient en mode ``none`` (``tool_rounds_max = 0``) et le dit dans ``source`` ; un
    ``system_tokens`` fourni pour un mode avec outils est alors remplacé par l'estimation sans
    outils (l'appelant remesure s'il veut l'exactitude).

    ``prefill_tps`` est la vitesse de lecture du modèle (tokens/s, mesurée ou estimée) : le contexte
    est alors borné pour que le prompt entier soit lu en ``cfg.first_token_target_s`` secondes
    (:func:`latency_context_cap`, plancher :data:`LATENCY_CONTEXT_FLOOR`) ; ``source`` l'explique
    (« plafonné à 3200 tokens pour rester sous 25 s (lecture estimée 60 tokens/s) ») et
    ``Budget.latency_capped`` le signale. ``None`` : pas de borne de latence.
    """
    num_ctx, parts = effective_num_ctx(cfg, model_info, profile)
    rounds = DEFAULT_TOOL_ROUNDS
    mode = tools_mode if tools_mode in TOOLS_MODES else "none"
    question = max(0, _as_int(question_tokens))
    share = intent_share(intent)
    target_s = _as_float(getattr(cfg, "first_token_target_s", None), DEFAULT_FIRST_TOKEN_TARGET_S)
    tps = _as_float(prefill_tps)
    tps = tps if tps > 0 else None

    # ----- parts fixes -----
    answer_reserve = answer_reserve_for(num_ctx)
    history_default = history_tokens_for(num_ctx)
    measured = system_tokens is not None and _as_int(system_tokens) >= 0
    system = _as_int(system_tokens) if measured else system_prompt_tokens(mode, num_ctx, rounds)
    system_desc = "mesurées" if measured else "estimées, gabarit + %d %% de marge" % int(SYSTEM_MARGIN_SHARE * 100)
    if mode == "native":
        system_desc += ", définitions d'outils natifs comprises"

    # ----- répartition, avec repli sans outils si la fenêtre ne les porte pas -----
    history, remaining, context, tools = _split(num_ctx, system, question, answer_reserve, history_default, share,
                                                mode != "none")
    downgraded = ""
    if mode != "none" and tools < MIN_TOOL_BUDGET:
        downgraded = ("outils désactivés : avec leurs consignes (%d tokens) il resterait %d token(s) pour leurs résultats"
                      % (system, tools))
        mode = "none"
        system = system_prompt_tokens(mode, num_ctx, 0)
        system_desc = "estimées, gabarit + %d %% de marge" % int(SYSTEM_MARGIN_SHARE * 100)
        history, remaining, context, tools = _split(num_ctx, system, question, answer_reserve, history_default, share,
                                                    False)
    if mode == "none":
        rounds = 0

    if mode == "none":
        context_desc = "contexte ≤ %d (tout le reste)" % context
    else:
        context_desc = "contexte ≤ %d (%s : %d %% du reste)" % (context, intent or "factual", int(share * 100))

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
    latency_capped = False
    latency_cap = latency_context_cap(tps, target_s, system, question, history)
    if latency_cap is not None and latency_cap < context:
        context = latency_cap
        latency_capped = True
        context_desc += (", plafonné à %d tokens pour rester sous %g s (lecture estimée %d tokens/s)"
                         % (context, target_s, int(round(tps or 0))))

    if num_ctx - system - question - answer_reserve < 0:
        parts.append("question trop longue pour la fenêtre : le moteur tronquera")
    source = ("num_ctx %d = min(%s) ; système %d (%s, mode %s), question %d, "
              "historique ≤ %d (%d %% de num_ctx, entre %d et %d), réponse %d (%d %% de num_ctx, entre %d et %d), "
              "reste %d : %s, outils ≤ %d, %d tours d'outils max"
              % (num_ctx, ", ".join(parts), system, system_desc, mode, question,
                 history, int(HISTORY_SHARE * 100), MIN_HISTORY_TOKENS, MAX_HISTORY_TOKENS,
                 answer_reserve, int(ANSWER_SHARE * 100), MIN_ANSWER_RESERVE, MAX_ANSWER_RESERVE,
                 remaining, context_desc, tools, rounds))
    if downgraded:
        source += " ; " + downgraded
    return Budget(num_ctx=num_ctx, system_tokens=system, history_tokens_max=history, context_tokens_max=context,
                  answer_reserve=answer_reserve, tool_rounds_max=rounds, source=source, tools_mode=mode,
                  question_tokens=question, tools_tokens_max=tools, prefill_tps=tps, latency_capped=latency_capped,
                  first_token_target_s=target_s)
