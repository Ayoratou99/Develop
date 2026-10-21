"""Orchestrateur du chat : recherche → budget → contexte → boucle d'outils bornée → événements SSE.

Pour chaque question, :meth:`ChatOrchestrator.respond` produit une suite d'événements
(:class:`ChatEvent`) : ``status``, ``sources``, ``tool_call``/``tool_result`` (au plus
``Budget.tool_rounds_max`` tours), ``delta`` puis ``done`` — ou ``error`` si le moteur est
injoignable (les ``sources`` ont alors déjà été émises). Chaque événement reste petit : les
textes complets des sources ne quittent jamais le serveur, seules leurs métadonnées sont envoyées.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Set, Tuple

from develop.chat.budget import MIN_TOOL_BUDGET, Budget, compute_budget, estimate_tokens, tool_specs_tokens
from develop.chat.context_builder import (ContextPack, Source, add_tool_sources, build_context,
                                          render_sources_as_text, render_sources_for_model)
from develop.chat.prompts import HISTORY_SUMMARY_PREFIX, build_system_prompt
from develop.chat.tools import TOOL_SPECS, ToolExecutor, ToolResult
from develop.llm.base import (ChatMessage, LLMClient, LLMError, LLMUnavailable, ModelInfo, ModelNotFound, ToolCall,
                              parse_text_tool_calls)
from develop.llm.offline_guard import OfflineViolation
from develop.storage import Database, StorageError

logger = logging.getLogger("develop.chat")

MAX_CALLS_PER_ROUND = 3
MESSAGE_OVERHEAD_TOKENS = 8  # rôle, délimiteurs du gabarit du moteur, identifiant d'appel
USER_FRAME_TOKENS = 32  # « Sources disponibles (cite-les avec [Sn]) : », sauts de ligne entre blocs et arrondis
SOURCES_FRAME_TOKENS = 16  # « Nouvelles sources citables : » et sauts de ligne entre les blocs
TEXT_RESULT_PREFIX = "Résultat de l'outil %s :\n"
INTERRUPTED_PREFIX = "[réponse interrompue] "
TOOLS_EXHAUSTED_TEXT = "Budget d'outils épuisé : réponds maintenant avec ce que tu as, en citant les sources disponibles."
NO_ANSWER_TEXT = "Le modèle n'a renvoyé aucune réponse."
STATUS_SEARCHING = "Recherche dans le projet…"
USEFUL_TEXT_MIN_CHARS = 160

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_TOOL_LINE_RE = re.compile(r'^\s*`*\s*\{\s*"tool"\s*:.*\}\s*`*\s*$')  # ligne d'appel d'outil complète
_TOOL_HEAD_RE = re.compile(r'^\s*`*\s*\{\s*"tool"\s*:')  # ligne d'appel d'outil inachevée (flux coupé)
_TOOL_START_RE = re.compile(r"^\s*`*\s*\{")  # ligne entamée qui peut encore devenir un appel d'outil
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$")  # clôture de bloc de code
_FENCE_HEAD_RE = re.compile(r"^\s*`{1,3}[a-zA-Z]*\s*$")  # clôture éventuellement inachevée


@dataclass
class ChatEvent:
    """Un événement SSE du chat (``type`` + ``data`` sérialisable en JSON)."""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        return "event: %s\ndata: %s\n\n" % (self.type, payload)


def _event(type_: str, **data: Any) -> ChatEvent:
    return ChatEvent(type_, dict(data))


def _remove_tool_lines(text: str, drop_incomplete_tail: bool = False, in_fence: bool = False) -> Tuple[str, bool]:
    """Texte sans ses lignes d'appel d'outil ni les clôtures ``` qui les encadrent, mise en page conservée.

    Les blocs de code légitimes restent intacts : seules la clôture qui ouvre un bloc juste avant une ligne
    d'outil et celle qui le referme juste après sont retirées. ``in_fence`` dit si le texte commence à
    l'intérieur d'un bloc de code (état rendu en second, pour filtrer un flux morceau par morceau). Avec
    ``drop_incomplete_tail``, une dernière ligne d'outil inachevée (flux coupé) est retirée aussi.
    """
    lines = (text or "").splitlines(keepends=True)
    kept: List[str] = []
    close_pending = False  # la clôture ouvrant un bloc d'outil a été retirée : sa fermante le sera aussi
    for i, line in enumerate(lines):
        is_tool = bool(_TOOL_LINE_RE.match(line))
        if not is_tool and drop_incomplete_tail and i == len(lines) - 1:
            is_tool = bool(_TOOL_HEAD_RE.match(line))
        if is_tool:
            if in_fence and kept and _FENCE_RE.match(kept[-1]):
                kept.pop()
                in_fence = False
                close_pending = True
            continue
        if _FENCE_RE.match(line):
            if close_pending and line.strip() == "```":
                close_pending = False
                continue
            in_fence = not in_fence
        if line.strip():
            close_pending = False
        kept.append(line)
    return "".join(kept), in_fence


def _strip_tool_lines(text: str) -> str:
    """Texte restant une fois retirées les lignes d'appel d'outil (et leurs clôtures), espaces de bord ôtés."""
    return _remove_tool_lines(text)[0].strip()


def _may_be_tool_block(line: str, partial: bool) -> bool:
    """Vrai si la ligne peut appartenir à un bloc d'appel d'outil : ligne JSON, clôture ``` ou ligne vide.

    Pour une ligne encore inachevée (``partial``), on regarde si elle peut le devenir.
    """
    if not line.strip():
        return True
    if partial:
        return bool(_TOOL_START_RE.match(line) or _FENCE_HEAD_RE.match(line))
    return bool(_TOOL_LINE_RE.match(line) or _FENCE_RE.match(line))


class _ToolLineFilter:
    """Filtre du flux en mode texte : aucune ligne d'appel d'outil (ni sa clôture ```) n'est émise.

    Le texte part au fil de l'eau, sauf sa fin tant qu'elle peut encore former un bloc d'appel d'outil
    (ligne JSON entamée, clôture, lignes vides) : elle est retenue jusqu'à ce qu'une ligne ordinaire la
    suive ou que le flux se termine. Le modèle qui écrit une réponse utile puis la ligne d'outil voit donc
    sa réponse affichée et l'outil exécuté, sans que la ligne JSON n'atteigne l'utilisateur.
    """

    def __init__(self) -> None:
        self.pending = ""
        self.in_fence = False

    def feed(self, text: str, release: bool = True) -> str:
        """Ajoute ``text`` ; renvoie la partie devenue sûre à émettre (lignes d'outil retirées), sinon ``""``."""
        self.pending += text
        if not release or not self.pending:
            return ""
        lines = self.pending.split("\n")
        k = len(lines)
        while k > 0 and _may_be_tool_block(lines[k - 1], partial=(k == len(lines))):
            k -= 1
        if k == 0:
            return ""
        if k == len(lines):
            decided, self.pending = self.pending, ""
        else:
            decided = "\n".join(lines[:k]) + "\n"
            self.pending = "\n".join(lines[k:])
        out, self.in_fence = _remove_tool_lines(decided, in_fence=self.in_fence)
        return out

    def flush(self) -> str:
        """Fin du flux : ce qui restait retenu, sans ligne d'outil (même inachevée)."""
        rest, self.pending = self.pending, ""
        out, self.in_fence = _remove_tool_lines(rest, drop_incomplete_tail=True, in_fence=self.in_fence)
        return out

    def discard(self) -> None:
        """Oublie ce qui était retenu (tour d'outil pur, éventuellement précédé d'une courte préface)."""
        self.pending = ""


def _has_useful_text(text: str) -> bool:
    rest = _strip_tool_lines(text)
    return bool(_CITATION_RE.search(rest)) or len(rest) >= USEFUL_TEXT_MIN_CHARS


def _could_be_tool_round(text: str) -> bool:
    """Tant que c'est vrai, les deltas sont retenus (mode texte) : la réponse peut être un appel d'outil."""
    s = (text or "").lstrip()
    if not s:
        return True
    if s[0] in "{`":
        return True
    return not _has_useful_text(s)


def _message_tokens(msg: ChatMessage) -> int:
    n = estimate_tokens(msg.content or "")
    if msg.tool_calls:
        n += sum(estimate_tokens(json.dumps(c.arguments, ensure_ascii=False)) + 8 for c in msg.tool_calls)
    return n


def find_invalid_citations(text: str, valid_refs: Sequence[str]) -> List[str]:
    """Références [Sn] citées mais absentes des sources, dans l'ordre d'apparition, sans doublon."""
    valid = set(valid_refs)
    out: List[str] = []
    for m in _CITATION_RE.finditer(text or ""):
        ref = "S%s" % m.group(1)
        if ref not in valid and ref not in out:
            out.append(ref)
    return out


class ChatOrchestrator:
    """Produit les réponses du chat sous forme de flux d'événements."""

    def __init__(self, db: Database, workspace: Any, llm: LLMClient, search_engine: Any = None,
                 impact_analyzer: Any = None, profile: Any = None, model_info: Optional[ModelInfo] = None,
                 today: Optional[str] = None, git: Any = None):
        self.db = db
        self.workspace = workspace
        self.llm = llm
        self.search_engine = search_engine
        self.impact_analyzer = impact_analyzer
        self.profile = profile
        self.model_info = model_info
        self.today = today or time.strftime("%Y-%m-%d")
        self.tools = ToolExecutor(db, workspace, search_engine, impact_analyzer, git)

    # ------------------------------------------------------------------ aides
    @property
    def llm_cfg(self) -> Any:
        return getattr(self.workspace, "llm", None)

    def _repos_summary(self) -> str:
        parts: List[str] = []
        for repo in getattr(self.workspace, "repos", []) or []:
            if not getattr(repo, "enabled", True):
                continue
            parts.append("%s (%s)" % (getattr(repo, "name", "?"), getattr(repo, "kind", "other")))
        return ", ".join(parts)

    def _tools_mode(self, mode: str) -> str:
        if mode in ("no_tools", "search_only"):
            return "none"
        if self.model_info is not None and getattr(self.model_info, "supports_tools", False):
            return "native"
        return "text"

    def _plan(self, tools_mode: str, intent: str, max_context_tokens: Optional[int],
              user_text: str) -> Tuple[Budget, str]:
        """Budget de la question avec les consignes système réellement mesurées ; renvoie ``(budget, consignes)``.

        Les consignes dépendent du budget (``num_ctx``, tours d'outils) et le budget de leur coût :
        une première passe fixe ``num_ctx`` et le mode, les consignes sont construites puis mesurées
        (définitions d'outils natifs comprises) et le budget recalculé avec cette mesure. Si cette
        mesure ne laisse pas de place aux outils, le budget revient sans outils et les consignes sont
        reconstruites et remesurées dans ce mode. Le mode effectif est ``budget.tools_mode``.
        """
        cfg, info, prof = self.llm_cfg, self.model_info, self.profile
        name, repos = getattr(self.workspace, "name", ""), self._repos_summary()
        question_tokens = estimate_tokens(user_text.strip()) + USER_FRAME_TOKENS

        def measured(text: str, mode: str) -> int:
            return estimate_tokens(text) + (tool_specs_tokens() if mode == "native" else 0)

        def downgrade_reason(b: Budget) -> str:
            idx = b.source.find("outils désactivés")
            return b.source[idx:] if idx >= 0 else ""

        budget = compute_budget(cfg, info, prof, intent, max_context_tokens, tools_mode=tools_mode,
                                question_tokens=question_tokens)
        reason = downgrade_reason(budget)
        system_text = ""
        for _ in range(2):  # au plus : mesure → repli sans outils → nouvelle mesure
            mode = budget.tools_mode
            system_text = build_system_prompt(mode, budget, name, repos, self.today)
            budget = compute_budget(cfg, info, prof, intent, max_context_tokens, tools_mode=mode,
                                    system_tokens=measured(system_text, mode), question_tokens=question_tokens)
            if budget.tools_mode == mode:
                break
            reason = downgrade_reason(budget) or reason
        else:
            system_text = build_system_prompt(budget.tools_mode, budget, name, repos, self.today)
        if reason and "outils désactivés" not in budget.source:
            budget.source += " ; " + reason  # le repli a eu lieu à une passe précédente : on garde l'explication
        return budget, system_text

    async def _search(self, text: str) -> Tuple[List[Any], str]:
        """Analyse + embedding + recherche multi-canaux ; renvoie ``(hits, intent)`` sans jamais lever."""
        engine = self.search_engine
        intent = "factual"
        if engine is None:
            return [], intent
        analysis: Any = None
        analyzed = False
        for attr in ("analyze", "analyze_query"):
            fn = getattr(engine, attr, None)
            if callable(fn):
                analyzed = True
                try:
                    analysis = await _maybe_await(fn(text))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("analyse de la question impossible: %s", exc)
                    analysis = None
                break
        if not analyzed:
            # le moteur n'expose pas d'analyse : on utilise celle de develop.search.query (import paresseux)
            try:
                from develop.search.query import analyze_query
                analysis = analyze_query(text, self.today)
            except Exception as exc:  # noqa: BLE001 - module absent ou en cours d'écriture
                logger.debug("analyze_query indisponible: %s", exc)
                analysis = None
        kwargs: Dict[str, Any] = {"limit": 30}
        if analysis is not None:
            intent = str(getattr(analysis, "intent", None) or intent)
            for key in ("date_from", "date_to"):
                value = getattr(analysis, key, None)
                if value:
                    kwargs[key] = value
        embed = getattr(engine, "embed_query", None)
        if callable(embed):
            try:
                vec = await _maybe_await(embed(text))
                if vec is not None:
                    kwargs["query_vector"] = vec
            except Exception as exc:  # noqa: BLE001
                logger.info("embedding de la question indisponible: %s", exc)
        try:
            result = await _maybe_await(engine.search(text, **kwargs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("recherche en échec, réponse sans contexte: %s", exc, exc_info=True)
            return [], intent
        if isinstance(result, list):
            hits = result
        else:
            hits = list(getattr(result, "hits", []) or [])
            query = getattr(result, "query", None)
            if query is not None and getattr(query, "intent", None):
                intent = str(query.intent)
        return hits, intent

    def _history(self, conversation_id: int, exclude_message_id: int, budget: Budget) -> Tuple[List[ChatMessage], str, int]:
        """Derniers messages tenant dans ``history_tokens_max`` + résumé en une ligne des plus anciens.

        Renvoie ``(messages, résumé, tokens des messages)`` ; le résumé (ajouté aux consignes par
        l'appelant) tient dans ce qui reste du budget d'historique, de sorte que messages + résumé
        ne dépassent jamais ``history_tokens_max``.
        """
        conv = self.db.get_conversation(conversation_id) or {}
        messages = [m for m in conv.get("messages", []) if m.get("id") != exclude_message_id
                    and m.get("role") in ("user", "assistant")]
        limit = max(0, int(budget.history_tokens_max))
        # on remonte depuis le message le plus récent tant que le budget le permet
        kept_from = len(messages)
        used = 0
        while kept_from > 0:
            cost = estimate_tokens(str(messages[kept_from - 1].get("content") or ""))
            if kept_from < len(messages) and used + cost > limit:
                break
            kept_from -= 1
            used += cost
        # pas de réponse orpheline en tête d'historique : sa question est passée dans le résumé
        while kept_from < len(messages) - 1 and messages[kept_from].get("role") == "assistant":
            kept_from += 1
        kept: List[ChatMessage] = []
        for m in messages[kept_from:]:
            content = str(m.get("content") or "")
            if len(messages) - kept_from == 1 and estimate_tokens(content) > limit:
                # un seul message, trop long : on le tronque plutôt que de perdre tout l'historique
                content = content[: max(0, limit * 4 - 20)].rstrip() + " […]"
            kept.append(ChatMessage(role=str(m.get("role")), content=content))
        used = sum(estimate_tokens(m.content) for m in kept)
        older = messages[:kept_from]
        summary = ""
        if older:
            titles = []
            for m in older:
                if m.get("role") == "user":
                    first_line = str(m.get("content") or "").strip().splitlines()
                    if first_line:
                        titles.append(first_line[0][:80])
            if titles:
                summary = HISTORY_SUMMARY_PREFIX + " ; ".join("Q : " + t for t in titles)
                max_chars = max(0, limit - used) * 4  # le résumé prend ce que les messages n'ont pas consommé
                if max_chars < len(HISTORY_SUMMARY_PREFIX) + 8:
                    summary = ""  # pas la place d'un seul titre : l'historique gardé suffit
                elif len(summary) > max_chars:
                    summary = summary[:max_chars - 1].rstrip() + "…"
        return kept, summary, used

    def _store(self, conversation_id: int, content: str, pack: Optional[ContextPack]) -> Optional[int]:
        sources = [s.to_dict() for s in pack.sources] if pack is not None else []
        try:
            return self.db.add_message(conversation_id, "assistant", content, sources=sources)
        except StorageError as exc:
            logger.error("enregistrement du message impossible: %s", exc)
            return None

    def _budget_dict(self, budget: Budget, pack: Optional[ContextPack], rounds: int, context_tokens: int,
                     tools_tokens_used: int = 0, tools_tokens_max: int = 0) -> Dict[str, Any]:
        """``budget`` de l'événement ``done`` : le contexte initial et les tours d'outils sont comptés à part.

        - ``context_tokens_used`` ≤ ``max_context_tokens`` : les passages retenus par la recherche initiale et
          leur budget (les sources ajoutées ensuite par les outils ne s'y cumulent pas) ;
        - ``tools_tokens_used`` / ``tools_tokens_max`` : ce que les tours d'outils ont consommé (appels,
          résultats, sources ajoutées, avis d'épuisement) et ce que la fenêtre leur laissait une fois le prompt
          réel et la réserve de réponse comptés ; 0 sans outils ;
        - ``sources_tokens`` : tous les blocs de sources citables, contexte initial et outils compris.
        """
        return {"num_ctx": budget.num_ctx, "context_tokens_used": max(0, int(context_tokens or 0)),
                "max_context_tokens": budget.context_tokens_max, "sources_tokens": pack.tokens_used if pack else 0,
                "tools_tokens_used": max(0, int(tools_tokens_used or 0)),
                "tools_tokens_max": max(0, int(tools_tokens_max or 0)), "tool_rounds": rounds,
                "answer_reserve": budget.answer_reserve, "tools_mode": budget.tools_mode, "source": budget.source}

    # ------------------------------------------------------------------ flux principal
    async def respond(self, conversation_id: int, user_text: str, mode: str = "auto",
                      max_context_tokens: Optional[int] = None,
                      cancel: Optional[asyncio.Event] = None) -> AsyncIterator[ChatEvent]:
        """Répond à ``user_text`` dans la conversation ; produit les événements SSE du chat."""
        mode = mode if mode in ("auto", "search_only", "no_tools") else "auto"
        user_text = str(user_text or "")
        emitted: List[str] = []  # fragments de réponse déjà envoyés à l'UI (= contenu enregistré)
        pack: Optional[ContextPack] = None
        stored = False
        rounds = 0
        try:
            # (1) message utilisateur
            try:
                user_message_id = self.db.add_message(conversation_id, "user", user_text)
            except StorageError as exc:
                stored = True
                yield _event("error", code="conversation_not_found", message=str(exc))
                return

            # (2) recherche
            yield _event("status", text=STATUS_SEARCHING)
            hits, intent = await self._search(user_text)

            # (3) budget : consignes système mesurées (définitions d'outils natifs comprises), mode d'outils
            #     effectif (``none`` si la fenêtre ne porte pas les outils), question comptée
            budget, system_text = self._plan(self._tools_mode(mode), intent, max_context_tokens, user_text)
            tools_mode = budget.tools_mode

            # (4) contexte
            pack = build_context(hits, budget.context_tokens_max, intent)
            context_tokens = pack.tokens_used  # contexte initial seul : les outils ajoutent ensuite au pack
            yield _event("sources", items=[s.to_dict() for s in pack.sources])

            # (5) recherche seule : pas de LLM
            if mode == "search_only":
                content = render_sources_as_text(pack)
                message_id = self._store(conversation_id, content, pack)
                stored = True
                yield _event("done", message_id=message_id, usage={},
                             budget=self._budget_dict(budget, pack, 0, context_tokens),
                             truncated_sources=[s.to_dict() for s in pack.truncated], invalid_citations=[],
                             cancelled=False, mode=mode)
                return

            # (6) historique (messages + résumé ≤ history_tokens_max)
            history, summary, history_tokens = self._history(conversation_id, user_message_id, budget)

            # (7) messages
            if summary:
                system_text += "\n\n" + summary
            user_content = user_text.strip() + "\n\n" + render_sources_for_model(pack)
            messages: List[ChatMessage] = [ChatMessage(role="system", content=system_text)]
            messages += history
            messages.append(ChatMessage(role="user", content=user_content))
            # Tout ce qui occupe la fenêtre est compté : messages, définitions d'outils natifs (hors messages
            # mais insérées par le gabarit du moteur à chaque appel tant que les outils sont proposés), réserve
            # de réponse. ``tools_available()`` est ce qui reste pour les résultats d'outils.
            prompt_tokens_est = estimate_tokens(system_text) + history_tokens + estimate_tokens(user_content)
            spec_tokens = tool_specs_tokens(TOOL_SPECS) if tools_mode == "native" else 0
            tools_tokens_used = 0

            def tools_available() -> int:
                return budget.num_ctx - prompt_tokens_est - spec_tokens - budget.answer_reserve - tools_tokens_used

            # ce que la fenêtre laisse aux tours d'outils (rapporté dans ``done.budget``) ; 0 sans outils
            tools_tokens_max = max(0, tools_available()) if tools_mode != "none" else 0

            # (8) boucle de génération / outils
            model = str(getattr(self.llm_cfg, "model", "") or "")
            temperature = float(getattr(self.llm_cfg, "temperature", 0.2) or 0.2)
            usage: Dict[str, int] = {}
            cancelled = False
            force_final = False
            seen_call_ids: Set[str] = set()
            while True:
                allow_tools = tools_mode != "none" and not force_final and rounds < budget.tool_rounds_max
                round_text = ""
                # Mode texte : (a) retenue initiale tant que la réponse peut n'être qu'un appel d'outil (rien
                # n'est montré d'une courte préface « Je vais regarder. ») ; (b) filtre de flux qui écarte
                # toute ligne d'outil du texte émis, y compris au tour final forcé où le modèle en écrit parfois.
                hold = allow_tools and tools_mode == "text"
                text_filter: Optional[_ToolLineFilter] = _ToolLineFilter() if tools_mode == "text" else None
                native_calls: List[ToolCall] = []
                extra_llm_kwargs: Dict[str, Any] = {}
                if self.model_info is not None and "thinking" in (getattr(self.model_info, "capabilities", None) or []):
                    # Réflexion interne : coûteuse (minutes sur un 27B) ; désactivée sauf demande explicite.
                    extra_llm_kwargs["think"] = bool(getattr(self.llm_cfg, "think", False))
                stream = self.llm.chat_stream(messages, model, temperature=temperature, num_ctx=budget.num_ctx,
                                              tools=TOOL_SPECS if (allow_tools and tools_mode == "native") else None,
                                              max_tokens=budget.answer_reserve, **extra_llm_kwargs)
                try:
                    async for chunk in stream:
                        if cancel is not None and cancel.is_set():
                            cancelled = True
                            break
                        text = getattr(chunk, "text", "") or ""
                        if text:
                            round_text += text
                            if text_filter is not None:
                                if hold and not _could_be_tool_round(round_text):
                                    hold = False
                                text = text_filter.feed(text, release=not hold)
                            if text:
                                emitted.append(text)
                                yield _event("delta", text=text)
                        for call in getattr(chunk, "tool_calls", None) or []:
                            native_calls.append(call)
                        chunk_usage = getattr(chunk, "usage", None) or {}
                        for key, value in chunk_usage.items():
                            try:
                                usage[key] = usage.get(key, 0) + int(value)
                            except (TypeError, ValueError):
                                continue
                finally:
                    aclose = getattr(stream, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception:  # noqa: BLE001
                            pass
                if cancelled:
                    rest = text_filter.flush() if text_filter is not None else ""
                    if rest:
                        emitted.append(rest)
                        yield _event("delta", text=rest)
                    break

                calls: List[ToolCall] = []
                if allow_tools:
                    # mode texte : l'outil est exécuté même si une réponse utile précède la ligne JSON (comme
                    # un modèle natif qui écrit du texte et appelle un outil dans le même tour)
                    calls = list(native_calls) if tools_mode == "native" else parse_text_tool_calls(round_text)
                if text_filter is not None:
                    if calls and hold:
                        text_filter.discard()  # appel d'outil pur : la courte préface éventuelle n'est pas montrée
                    else:
                        rest = text_filter.flush()
                        if rest:
                            emitted.append(rest)
                            yield _event("delta", text=rest)
                if not calls:
                    break

                # tour d'outils (copies : les objets du client ne sont jamais mutés, identifiants uniques)
                rounds += 1
                calls = [ToolCall(name=str(c.name or ""), arguments=dict(c.arguments) if isinstance(c.arguments, dict)
                                  else c.arguments, id=str(c.id or "")) for c in calls[:MAX_CALLS_PER_ROUND]]
                for n, call in enumerate(calls):
                    if not call.id or call.id in seen_call_ids:
                        call.id = "call-%d-%d" % (rounds, n + 1)
                    seen_call_ids.add(call.id)
                if tools_mode == "native":
                    assistant_msg = ChatMessage(role="assistant", content=round_text, tool_calls=calls)
                else:
                    assistant_msg = ChatMessage(role="assistant", content=round_text)
                messages.append(assistant_msg)
                tools_tokens_used += _message_tokens(assistant_msg) + MESSAGE_OVERHEAD_TOKENS
                text_results: List[str] = []
                for call in calls:
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        break
                    yield _event("tool_call", name=call.name, arguments=call.arguments, id=call.id)
                    available = tools_available()
                    per_call = 0
                    if available < MIN_TOOL_BUDGET:
                        result = ToolResult(name=call.name, summary="Budget de contexte épuisé : outil non exécuté.",
                                            text="Budget de contexte épuisé : réponds avec les sources déjà fournies.",
                                            error="budget épuisé")
                        force_final = True
                    else:
                        per_call = min(available, max(MIN_TOOL_BUDGET, budget.context_tokens_max))
                        result = await self.tools.execute(call, per_call)
                    added: List[Source] = []
                    if result.hits and not result.error:
                        # Un seul budget par appel : le texte de l'outil et ses nouvelles sources se le partagent.
                        # Pour read_file/symbol, le texte est remplacé par le résumé dès qu'une source est ajoutée
                        # (son contenu est dans les blocs) : tout le budget va alors aux sources.
                        text_part = result.summary if result.name in ("read_file", "symbol") else result.text
                        sources_budget = per_call - estimate_tokens(text_part) - SOURCES_FRAME_TOKENS
                        added = add_tool_sources(pack, result.hits, max(0, sources_budget))
                    content = self._tool_message_content(result, pack, added)
                    if tools_mode != "native":
                        content = TEXT_RESULT_PREFIX % call.name + content
                    tools_tokens_used += estimate_tokens(content) + MESSAGE_OVERHEAD_TOKENS
                    data: Dict[str, Any] = {"name": call.name, "summary": result.summary,
                                            "added_sources": [s.to_dict() for s in added], "id": call.id}
                    if result.error:
                        data["error"] = result.error
                    yield _event("tool_result", **data)
                    if tools_mode == "native":
                        messages.append(ChatMessage(role="tool", content=content, tool_call_id=call.id, name=call.name))
                    else:
                        text_results.append(content)
                if cancelled:
                    break
                if tools_mode != "native" and text_results:
                    messages.append(ChatMessage(role="user", content="\n\n".join(text_results)))
                if rounds >= budget.tool_rounds_max or tools_available() < MIN_TOOL_BUDGET:
                    force_final = True
                    messages.append(ChatMessage(role="user", content=TOOLS_EXHAUSTED_TEXT))
                    tools_tokens_used += estimate_tokens(TOOLS_EXHAUSTED_TEXT) + MESSAGE_OVERHEAD_TOKENS

            # (9) fin
            answer = "".join(emitted)
            budget_info = self._budget_dict(budget, pack, rounds, context_tokens, tools_tokens_used, tools_tokens_max)
            if cancelled:
                message_id = self._store(conversation_id, answer, pack) if answer.strip() else None
                stored = True
                yield _event("done", message_id=message_id, usage=usage, budget=budget_info,
                             truncated_sources=[s.to_dict() for s in pack.truncated],
                             invalid_citations=find_invalid_citations(answer, list(pack.hits_by_ref.keys())),
                             cancelled=True, mode=mode)
                return
            if not answer.strip():
                answer = NO_ANSWER_TEXT
                emitted.append(answer)
                yield _event("delta", text=answer)
            invalid = find_invalid_citations(answer, list(pack.hits_by_ref.keys()))
            message_id = self._store(conversation_id, answer, pack)
            stored = True
            yield _event("done", message_id=message_id, usage=usage, budget=budget_info,
                         truncated_sources=[s.to_dict() for s in pack.truncated], invalid_citations=invalid,
                         cancelled=False, mode=mode)
        except (LLMUnavailable, ModelNotFound, OfflineViolation, LLMError) as exc:
            code = _error_code(exc)
            logger.info("chat interrompu (%s): %s", code, exc)
            message_id = self._store_partial(conversation_id, emitted, pack)
            stored = True
            yield _event("error", code=code, message=str(exc), message_id=message_id)
        except asyncio.CancelledError:
            if not stored:
                self._store_partial(conversation_id, emitted, pack)
                stored = True
            raise
        except Exception as exc:  # noqa: BLE001 - jamais de 500 silencieux dans un flux SSE
            logger.exception("erreur interne du chat")
            message_id = self._store_partial(conversation_id, emitted, pack)
            stored = True
            yield _event("error", code="internal_error", message="Erreur interne du chat : %s" % exc, message_id=message_id)
        finally:
            if not stored:
                # le client a fermé le flux : on garde la réponse partielle
                self._store_partial(conversation_id, emitted, pack)

    def _store_partial(self, conversation_id: int, emitted: List[str], pack: Optional[ContextPack]) -> Optional[int]:
        text = "".join(emitted)
        if not text.strip():
            return None
        return self._store(conversation_id, INTERRUPTED_PREFIX + text, pack)

    @staticmethod
    def _tool_message_content(result: ToolResult, pack: ContextPack, added: List[Source]) -> str:
        """Contenu renvoyé au modèle : résultat de l'outil + blocs des nouvelles sources citables."""
        parts: List[str] = []
        covered_by_sources = result.name in ("read_file", "symbol") and added and not result.error
        if result.text and not covered_by_sources:
            parts.append(result.text)
        elif covered_by_sources:
            parts.append(result.summary)
        if added:
            parts.append(render_sources_for_model(pack, [s.ref for s in added]))
        if result.error and not result.text:
            parts.append("Erreur : %s" % result.error)
        return "\n\n".join(p for p in parts if p)


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, ModelNotFound):
        return "model_not_found"
    if isinstance(exc, LLMUnavailable):
        return "llm_unavailable"
    if isinstance(exc, OfflineViolation):
        return "offline_blocked"
    return "llm_error"


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or hasattr(value, "__await__"):
        return await value
    return value
