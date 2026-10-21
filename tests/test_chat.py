"""Tests de l'orchestrateur du chat (develop.chat) : budget, contexte, outils, flux d'événements."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from develop.chat import (
    TOOL_SPECS,
    Budget,
    ChatEvent,
    ChatOrchestrator,
    Source,
    ToolExecutor,
    add_tool_sources,
    build_context,
    build_system_prompt,
    compute_budget,
    estimate_tokens,
    render_sources_for_model,
)
from develop.chat.budget import (LATENCY_CONTEXT_FLOOR, MAX_ANSWER_RESERVE, MAX_HISTORY_TOKENS, MIN_ANSWER_RESERVE,
                                 MIN_HISTORY_TOKENS, MIN_TOOL_BUDGET, SYSTEM_TOKENS_CAP, answer_reserve_for,
                                 history_tokens_for, system_prompt_tokens, system_text_tokens, tool_specs_tokens)
from develop.chat.context_builder import (GRAPH_FILE_PREVIEW_LINES, IMPORTS_OMITTED_MARKER, LocalHit, _CardLoader,
                                          _fit_sections, _repo_prefix, _request_section, build_cards, format_block,
                                          format_header, render_sources_as_text)
from develop.chat.orchestrator import (INTERRUPTED_PREFIX, SOURCES_FRAME_TOKENS, TOOLS_EXHAUSTED_TEXT, _message_tokens,
                                       find_invalid_citations, followup_terms, is_smalltalk, smalltalk_reply)
from develop.chat.prompts import HISTORY_SUMMARY_PREFIX, SYSTEM_PROMPT, TOOLS_TEXT_PROTOCOL
from develop.chat.tools import TRUNCATED_NOTE, ToolResult, fit_text
from develop.config import LLMConfig, RepoConfig, WorkspaceConfig
from develop.llm.base import (ChatChunk, LLMClient, LLMError, LLMUnavailable, ModelInfo, ModelNotFound, NullClient,
                              ToolCall)
from develop.llm.offline_guard import OfflineViolation
from develop.storage import (ChunkRow, Database, DocumentRow, EndpointRow, FileRow, ModelRow, NoteRow, RelationRow,
                             RepoRow, SymbolRow, node_key)
from develop.system.profile import MachineProfile

# --------------------------------------------------------------------------- faux objets


@dataclass
class Hit:
    """Même forme que ``develop.search.engine.SearchHit``."""

    chunk_id: Optional[int] = None
    source_type: str = "file"
    source_id: Optional[int] = None
    title: str = ""
    path: Optional[str] = None
    repo_id: Optional[str] = None
    file_id: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    page: Optional[int] = None
    snippet: str = ""
    text: str = ""
    score: float = 0.0
    channels: List[str] = field(default_factory=list)
    reason: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeQuery:
    intent: str = "factual"
    identifiers: List[str] = field(default_factory=list)
    date_from: Optional[str] = None
    date_to: Optional[str] = None


@dataclass
class FakeResult:
    query: FakeQuery
    hits: List[Hit]
    semantic_available: bool = False


class FakeSearchEngine:
    """Moteur de recherche scripté : ``hits`` pour la première recherche, ``tool_hits`` pour les suivantes."""

    def __init__(self, hits: Sequence[Hit] = (), tool_hits: Optional[Sequence[Hit]] = None, intent: str = "factual",
                 analysis: Optional[FakeQuery] = None, snippet: Optional[Dict[str, Any]] = None):
        self.hits = list(hits)
        self.tool_hits = list(tool_hits) if tool_hits is not None else None
        self.intent = intent
        self.analysis = analysis
        self.snippet = snippet
        self.calls: List[Dict[str, Any]] = []
        self.embed_calls = 0
        if analysis is not None:
            self.analyze = lambda text: analysis  # type: ignore[assignment]

    def search(self, query: str, limit: int = 20, source_types: Optional[List[str]] = None, query_vector: Any = None,
               date_from: Optional[str] = None, date_to: Optional[str] = None) -> FakeResult:
        self.calls.append({"query": query, "limit": limit, "source_types": source_types, "query_vector": query_vector,
                           "date_from": date_from, "date_to": date_to})
        hits = self.hits if (len(self.calls) == 1 or self.tool_hits is None) else self.tool_hits
        return FakeResult(FakeQuery(intent=self.intent, date_from=date_from, date_to=date_to), list(hits))

    async def embed_query(self, text: str) -> Any:
        self.embed_calls += 1
        return None

    def lookup_symbol(self, name: str) -> List[Hit]:
        return []

    def lookup_endpoint(self, path: str, method: Optional[str] = None) -> List[Hit]:
        return []

    def lookup_model(self, name: str) -> List[Hit]:
        return []

    def file_snippet(self, file_id: int, start: int, end: int, max_lines: int = 120) -> Optional[Dict[str, Any]]:
        return self.snippet


class FakeReport:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    def to_dict(self) -> Dict[str, Any]:
        return self.data


class FakeImpactAnalyzer:
    """Analyseur d'impact scripté (même signatures que ``develop.impact_analysis.analyzer.ImpactAnalyzer``)."""

    def __init__(self) -> None:
        self.calls: List[Any] = []

    def _report(self, path: str) -> FakeReport:
        return FakeReport({
            "changed": [{"path": path, "status": "modified", "repo_id": "back", "symbols_changed": ["confirm"]}],
            "affected": [
                {"node": "symbol:2", "type": "symbol", "name": "OrderController.confirm",
                 "path": "app/Http/Controllers/OrderController.php", "line": 25, "repo_id": "back", "file_id": 2,
                 "reason": "appelle OrderService.confirm", "confidence": 0.9, "hops": 1},
                {"node": "endpoint:1", "type": "endpoint", "name": "POST /orders/{order}/confirm", "path": "routes/api.php",
                 "line": 13, "repo_id": "back", "file_id": 3, "reason": "route gérée par OrderController.confirm",
                 "confidence": 0.7, "hops": 2},
            ],
            "endpoints": [{"name": "POST /orders/{order}/confirm"}], "models": [], "frontend_calls": [{"name": "confirmOrder"}],
            "documents": [{"title": "Dictionnaire de données"}], "notes": [], "stats": {"affected": 2},
            "warnings": ["index partiel : le dépôt frontend n'a pas été réanalysé"],
        })

    def from_file(self, repo_id: str, path: str, max_hops: int = 2) -> FakeReport:
        self.calls.append(("file", repo_id, path, max_hops))
        return self._report(path)

    def from_symbol(self, symbol_id: int, max_hops: int = 2) -> FakeReport:
        self.calls.append(("symbol", symbol_id, max_hops))
        return self._report("symbol:%d" % symbol_id)

    def from_git(self, repo_id: str, rev_range: Optional[str] = None, max_hops: int = 2) -> FakeReport:
        self.calls.append(("git", repo_id, rev_range, max_hops))
        return self._report(rev_range or "HEAD")


class FakeLLM(LLMClient):
    """Client LLM en mémoire : ``rounds[i]`` est la liste de ChatChunk renvoyée au i-ième appel."""

    engine = "fake"

    def __init__(self, rounds: Sequence[Sequence[ChatChunk]], delay: float = 0.0):
        self.rounds = [list(r) for r in rounds]
        self.delay = delay
        self.calls: List[Dict[str, Any]] = []

    async def chat_stream(self, messages, model, *, temperature=0.2, num_ctx=None, tools=None, max_tokens=None):  # type: ignore[override]
        self.calls.append({"messages": list(messages), "model": model, "temperature": temperature, "num_ctx": num_ctx,
                           "tools": list(tools) if tools else None, "max_tokens": max_tokens})
        idx = min(len(self.calls) - 1, len(self.rounds) - 1)
        for chunk in self.rounds[idx]:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield chunk

    async def embed(self, texts, model):  # type: ignore[override]
        return [[0.0] for _ in texts]


class FailingLLM(LLMClient):
    """Émet ``prefix`` puis lève ``exc`` (simule un moteur qui tombe en pleine génération)."""

    engine = "failing"

    def __init__(self, exc: BaseException, prefix: str = ""):
        self.exc = exc
        self.prefix = prefix

    async def chat_stream(self, messages, model, **kwargs):  # type: ignore[override]
        if self.prefix:
            yield ChatChunk(text=self.prefix)
        raise self.exc


def profile(ram_gb: float, unified: bool = False) -> MachineProfile:
    return MachineProfile(os="macos", os_version="15", arch="arm64", cpu_model="test", cpu_cores_physical=8,
                          cpu_cores_logical=8, ram_total_gb=ram_gb, ram_available_gb=ram_gb / 2,
                          apple_silicon=unified, unified_memory=unified)


def info(context_length: Optional[int], tools: bool = True) -> ModelInfo:
    return ModelInfo(name="fake:7b", context_length=context_length, capabilities=["completion"] + (["tools"] if tools else []))


def text_chunks(*parts: str, usage: Optional[Dict[str, int]] = None) -> List[ChatChunk]:
    chunks = [ChatChunk(text=p) for p in parts]
    chunks.append(ChatChunk(done=True, usage=usage or {"prompt_tokens": 100, "completion_tokens": 20}))
    return chunks


def tool_chunks(*calls: ToolCall) -> List[ChatChunk]:
    return [ChatChunk(tool_calls=list(calls)), ChatChunk(done=True, usage={"prompt_tokens": 50, "completion_tokens": 5})]


def file_hit(**kw: Any) -> Hit:
    base = dict(chunk_id=1, source_type="file", source_id=1, title="OrderService.php", path="app/Services/OrderService.php",
                repo_id="back", file_id=1, line_start=10, line_end=23, score=0.9, channels=["identifier", "lexical"],
                reason="identifiant OrderService.confirm", extra={"language": "php", "repo_name": "backend",
                                                                  "symbol": "OrderService.confirm"},
                text="    public function confirm(Order $order): Order\n    {\n        $order->status = 'confirmed';\n"
                     "        $order->save();\n        return $order;\n    }")
    base.update(kw)
    return Hit(**base)


def doc_hit(**kw: Any) -> Hit:
    base = dict(chunk_id=2, source_type="document", source_id=7, title="Dictionnaire de données", score=0.6,
                channels=["lexical"], reason="mentionne la table orders", page=3,
                extra={"section": "Table orders", "document_id": 7},
                text="| status | enum | draft, confirmed, shipped, cancelled |\n| confirmed_at | timestamp | renseignée par OrderService::confirm |")
    base.update(kw)
    return Hit(**base)


def note_hit(**kw: Any) -> Hit:
    base = dict(chunk_id=3, source_type="note", source_id=4, title="Réunion statuts", score=0.5, channels=["temporal"],
                reason="note datée du 2026-09-01", extra={"note_date": "2026-09-01", "tags": ["décision", "commandes"]},
                text="Décision : on n'ajoute PAS de statut paid pour l'instant.\nLe paiement sera géré par paid_at.")
    base.update(kw)
    return Hit(**base)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db():
    with Database(":memory:") as d:
        yield d


@pytest.fixture
def workspace(sample_project: Path) -> WorkspaceConfig:
    return WorkspaceConfig(id="ws", name="Boutique", repos=[
        RepoConfig(id="back", name="backend", path=str(sample_project / "backend"), kind="backend"),
        RepoConfig(id="front", name="frontend", path=str(sample_project / "frontend"), kind="frontend"),
    ], llm=LLMConfig(engine="ollama", model="fake:7b", num_ctx=8192))


def seed_index(db: Database, sample_project: Path) -> Dict[str, Any]:
    """Peuple la mémoire avec quelques artefacts du mini-projet (sans toucher à la fixture)."""
    db.upsert_repo(RepoRow(id="back", name="backend", path=str(sample_project / "backend"), kind="backend"))
    db.upsert_repo(RepoRow(id="front", name="frontend", path=str(sample_project / "frontend"), kind="frontend"))
    service = FileRow(repo_id="back", path="app/Services/OrderService.php", language="php", kind="code", size=700)
    controller = FileRow(repo_id="back", path="app/Http/Controllers/OrderController.php", language="php", kind="code", size=500)
    routes = FileRow(repo_id="back", path="routes/api.php", language="php", kind="code", size=400)
    model = FileRow(repo_id="back", path="app/Models/Order.php", language="php", kind="code", size=500)
    client = FileRow(repo_id="front", path="src/api/client.ts", language="typescript", kind="code", size=900)
    for f in (service, controller, routes, model, client):
        db.upsert_file(f)
    service_text = (sample_project / "backend/app/Services/OrderService.php").read_text(encoding="utf-8")
    controller_text = (sample_project / "backend/app/Http/Controllers/OrderController.php").read_text(encoding="utf-8")
    svc = db.replace_file_artifacts(service.id, symbols=[
        SymbolRow(name="OrderService", kind="class", line_start=8, line_end=29),
        SymbolRow(name="confirm", kind="method", parent="OrderService", line_start=10, line_end=23,
                  signature="confirm(Order $order): Order"),
        SymbolRow(name="totalFor", kind="method", parent="OrderService", line_start=25, line_end=28),
    ], chunks=[ChunkRow(text=service_text, line_start=1, line_end=29)])
    ctl = db.replace_file_artifacts(controller.id, symbols=[
        SymbolRow(name="OrderController", kind="class", line_start=9, line_end=29),
        SymbolRow(name="confirm", kind="method", parent="OrderController", line_start=25, line_end=28),
    ], chunks=[ChunkRow(text=controller_text, line_start=1, line_end=29)])
    rt = db.replace_file_artifacts(routes.id, endpoints=[
        EndpointRow(side="server", method="GET", path="/orders", normalized_path="/orders", handler="OrderController@index",
                    framework="laravel", line=11),
        EndpointRow(side="server", method="POST", path="/orders/{order}/confirm", normalized_path="/orders/:param/confirm",
                    handler="OrderController@confirm", framework="laravel", line=13),
    ])
    md = db.replace_file_artifacts(model.id, models=[
        ModelRow(name="Order", table_name="orders", kind="eloquent", line=9,
                 fields=[{"name": "reference", "type": "string"},
                         {"name": "status", "type": "enum", "meta": {"enum_values": ["draft", "confirmed"]}},
                         {"name": "confirmed_at", "type": "datetime", "nullable": True}],
                 relations=[{"kind": "belongsTo", "target": "Customer", "field": "customer_id"}]),
    ])
    cl = db.replace_file_artifacts(client.id, endpoints=[
        EndpointRow(side="client", method="POST", path="/orders/${id}/confirm", normalized_path="/orders/:param/confirm",
                    handler="confirmOrder", framework="axios", line=33),
    ])
    svc_confirm, ctl_confirm = svc["symbols"][1], ctl["symbols"][1]
    db.add_relations([
        RelationRow(src=node_key("symbol", ctl_confirm), dst=node_key("symbol", svc_confirm), kind="calls", confidence=0.9),
        RelationRow(src=node_key("file", controller.id), dst=node_key("file", service.id), kind="imports"),
        RelationRow(src=node_key("endpoint", rt["endpoints"][1]), dst=node_key("symbol", ctl_confirm), kind="handles"),
    ])
    doc_id = db.add_document(DocumentRow(title="Dictionnaire de données", format="md", kind="upload", sha1="doc1"))
    db.replace_chunks("document", doc_id, [ChunkRow(
        text="Table orders : status enum draft, confirmed, shipped, cancelled ; confirmed_at renseigné par OrderService::confirm.",
        meta={"section": "Table orders", "page": 3})])
    note_recent = db.add_note(NoteRow(title="Réunion statuts", note_date="2026-09-01", tags=["décision", "commandes"],
                                      body="Décision : on n'ajoute PAS de statut paid pour l'instant. Le paiement sera géré par paid_at."))
    db.replace_chunks("note", note_recent, [ChunkRow(text="Réunion statuts. Décision : pas de statut paid pour l'instant ; paid_at plus tard.")])
    note_old = db.add_note(NoteRow(title="Brainstorm paiement", note_date="2026-06-15", tags=["commandes"],
                                   body="On envisage un statut paid sur les commandes."))
    db.replace_chunks("note", note_old, [ChunkRow(text="Brainstorm paiement : on envisage un statut paid sur les commandes.")])
    return {"service": service, "controller": controller, "routes": routes, "model": model, "client": client,
            "svc_confirm": svc_confirm, "ctl_confirm": ctl_confirm, "model_id": md["models"][0],
            "server_endpoint": rt["endpoints"][1], "client_endpoint": cl["endpoints"][0], "doc_id": doc_id,
            "note_recent": note_recent, "note_old": note_old}


@pytest.fixture
def seeded(db: Database, sample_project: Path) -> Dict[str, Any]:
    return seed_index(db, sample_project)


def without_model_phase_statuses(events: List[ChatEvent]) -> List[ChatEvent]:
    """Retire les ``status`` informatifs émis après ``sources`` (chargement du modèle, lecture des sources) :
    leur présence dépend du moteur et du mode, pas de la logique testée ; les indices restent stables."""
    out: List[ChatEvent] = []
    seen_sources = False
    for e in events:
        if e.type == "sources":
            seen_sources = True
        if e.type == "status" and seen_sources:
            continue
        out.append(e)
    return out


async def collect(orch: ChatOrchestrator, conversation_id: int, text: str, **kwargs: Any) -> List[ChatEvent]:
    return without_model_phase_statuses([ev async for ev in orch.respond(conversation_id, text, **kwargs)])


def types(events: List[ChatEvent]) -> List[str]:
    return [e.type for e in events]


def last(events: List[ChatEvent], type_: str) -> ChatEvent:
    found = [e for e in events if e.type == type_]
    assert found, "aucun événement %s dans %s" % (type_, types(events))
    return found[-1]


# =========================================================================== budget


def test_estimate_tokens_prose_and_code():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0  # type: ignore[arg-type]
    assert estimate_tokens("a" * 400) == 100
    assert estimate_tokens("a" * 350, is_code=True) == 100
    assert estimate_tokens("abc") == 1  # arrondi vers le haut


def test_compute_budget_unknown_window_defaults_to_8192():
    b = compute_budget(LLMConfig(num_ctx=131072), None, None)
    assert b.num_ctx == 8192
    assert "inconnue" in b.source and "8192" in b.source
    assert b.tool_rounds_max == 4


@pytest.mark.parametrize("ram,cap", [(4, 4096), (12, 8192), (24, 16384), (48, 32768), (128, 65536)])
def test_compute_budget_ram_caps(ram, cap):
    b = compute_budget(LLMConfig(num_ctx=131072), info(131072), profile(ram))
    assert b.num_ctx == cap
    assert "plafond RAM" in b.source


def test_compute_budget_accelerator_doubles_cap_bounded_by_window():
    # 32 Go de mémoire unifiée Apple → accélérateur ≈ 24 Go → plafond 32768 × 2
    b = compute_budget(LLMConfig(num_ctx=131072), info(131072), profile(32, unified=True))
    assert b.num_ctx == 65536
    assert "accélérateur" in b.source
    # borné par la fenêtre du modèle
    b2 = compute_budget(LLMConfig(num_ctx=131072), info(40000), profile(32, unified=True))
    assert b2.num_ctx == 40000
    # sans accélérateur suffisant (24 Go unifiés → 18 Go), pas de bonus
    assert compute_budget(LLMConfig(num_ctx=131072), info(131072), profile(32)).num_ctx == 32768
    assert compute_budget(LLMConfig(num_ctx=131072), info(131072), profile(24, unified=True)).num_ctx == 16384


def test_compute_budget_is_min_of_model_config_and_ram():
    b = compute_budget(LLMConfig(num_ctx=16384), info(32768), profile(64))
    assert b.num_ctx == 16384
    b = compute_budget(LLMConfig(num_ctx=65536), info(32768), profile(64))
    assert b.num_ctx == 32768
    b = compute_budget(LLMConfig(num_ctx=65536), info(65536), profile(12))
    assert b.num_ctx == 8192


def test_compute_budget_partition_is_consistent():
    for ram in (4, 16, 64):
        b = compute_budget(LLMConfig(num_ctx=65536, max_context_chars=1_000_000), info(65536), profile(ram), intent="impact")
        assert b.system_tokens + b.history_tokens_max + b.context_tokens_max + b.answer_reserve <= b.num_ctx
        assert b.total <= b.num_ctx
        assert MIN_ANSWER_RESERVE <= b.answer_reserve <= MAX_ANSWER_RESERVE
        assert b.history_tokens_max >= min(MIN_HISTORY_TOKENS, b.num_ctx // 4)
        assert b.history_tokens_max <= min(MAX_HISTORY_TOKENS, max(MIN_HISTORY_TOKENS, b.num_ctx // 10))
        assert b.context_tokens_max > 0
    big = compute_budget(LLMConfig(num_ctx=16384, max_context_chars=1_000_000), info(16384), None, intent="factual")
    # les consignes sont mesurées sur le vrai gabarit (définitions d'outils natifs comprises), pas devinées ;
    # la réserve de réponse est 20 % de la fenêtre, plafonnée à 3k (une réponse citée n'en demande pas plus)
    assert big.system_tokens == system_prompt_tokens("native", 16384) and big.answer_reserve == MAX_ANSWER_RESERVE == 3072
    assert big.tools_mode == "native" and big.tool_rounds_max == 4 and big.tools_tokens_max >= MIN_TOOL_BUDGET


@pytest.mark.parametrize("ctx,history,answer", [(2048, 768, 1024), (4096, 768, 1024), (8192, 819, 1638),
                                                (16384, 1638, 3072), (32768, 2048, 3072), (65536, 2048, 3072)])
def test_compute_budget_history_and_answer_follow_window(ctx, history, answer):
    # historique = min(max(768, 10 % de num_ctx), 2048) ; réponse = clamp(20 % de num_ctx, 1024, 3072)
    assert history_tokens_for(ctx) == history and answer_reserve_for(ctx) == answer
    b = compute_budget(LLMConfig(num_ctx=ctx), info(ctx), None, tools_mode="none")
    assert b.answer_reserve == answer and b.history_tokens_max <= history
    if ctx >= 4096:
        assert b.history_tokens_max == history  # la fenêtre porte l'historique entier ; à 2048 il cède au contexte
    assert "10 %% de num_ctx, entre %d et %d" % (MIN_HISTORY_TOKENS, MAX_HISTORY_TOKENS) in b.source
    assert "20 %% de num_ctx, entre %d et %d" % (MIN_ANSWER_RESERVE, MAX_ANSWER_RESERVE) in b.source


@pytest.mark.parametrize("ctx,floor", [(8192, 3000), (16384, 7500), (32768, 15000)])
def test_compute_budget_context_floor_per_window(ctx, floor):
    # Régression : à 8192 en natif le contexte récupéré ne faisait que 1081 tokens (forfait système 1622 + réponse 30 %)
    # alors que la fenêtre contenait tout ce qu'il fallait ; objectifs : ≥ 3k à 8k, ≥ 7,5k à 16k, ≥ 15k à 32k.
    cfg = LLMConfig(num_ctx=ctx)  # max_context_chars par défaut (96 000 ≈ 24k tokens) : ne bride pas ces paliers
    for mode in ("native", "text", "none"):
        for intent in ("factual", "fields", "howto", "decision"):
            b = compute_budget(cfg, info(ctx), profile(64), intent=intent, tools_mode=mode, question_tokens=40)
            assert b.context_tokens_max >= floor, (ctx, mode, intent, b.source)
            assert b.total <= ctx and b.tools_mode == mode and b.tool_rounds_max == (0 if mode == "none" else 4)
        factual = compute_budget(cfg, info(ctx), profile(64), intent="factual", tools_mode=mode).context_tokens_max
        for wide in ("impact", "broad"):
            w = compute_budget(cfg, info(ctx), profile(64), intent=wide, tools_mode=mode)
            assert w.context_tokens_max >= factual and w.total <= ctx
            if mode != "none":
                assert w.context_tokens_max > factual and "90 % du reste" in w.source
    native = compute_budget(cfg, info(ctx), profile(64), intent="factual", tools_mode="native")
    assert "80 % du reste" in native.source and native.tools_tokens_max >= MIN_TOOL_BUDGET


def test_compute_budget_system_tokens_are_prompt_plus_margin_not_a_flat_fee():
    # consignes = gabarit mesuré + 10 % de marge (plafond 1500), définitions d'outils natifs en sus (jamais plafonnées)
    shape = compute_budget(LLMConfig(num_ctx=8192), info(8192), None, tools_mode="none")
    for mode in ("native", "text", "none"):
        real = estimate_tokens(build_system_prompt(mode, shape, "", "", "2026-01-01"))
        text = system_text_tokens(mode, 8192)
        assert text == min(SYSTEM_TOKENS_CAP, int(real * 1.1 + 0.999)) and real < text <= SYSTEM_TOKENS_CAP
        assert system_prompt_tokens(mode, 8192) == text + (tool_specs_tokens() if mode == "native" else 0)
    b = compute_budget(LLMConfig(num_ctx=8192), info(8192), None, tools_mode="text")
    assert b.system_tokens == system_prompt_tokens("text", 8192) < 1100 and "10 % de marge" in b.source
    # une liste de dépôts démesurée ne mange pas la fenêtre : l'estimation des consignes textuelles est plafonnée
    assert system_text_tokens("text", 8192, repos_summary="x" * 20_000) == SYSTEM_TOKENS_CAP


def test_eight_sources_of_350_tokens_fit_in_context_at_8192():
    # À 8192 (natif, factuel) : au moins 8 passages de ≈ 350 tokens tiennent dans le contexte récupéré.
    b = compute_budget(LLMConfig(num_ctx=8192), info(8192), None, intent="factual", tools_mode="native", question_tokens=30)
    assert b.context_tokens_max >= 8 * 350
    line = "        $this->validate($request, ['status' => 'required|in:draft,confirmed']); // champ %02d\n"
    body = "".join(line % j for j in range(13))  # ≈ 1 200 caractères de code ≈ 350 tokens
    assert 330 <= estimate_tokens(body, is_code=True) <= 370
    hits = [file_hit(chunk_id=10 + i, source_id=10 + i, file_id=10 + i, path="app/Models/Model%d.php" % i,
                     title="Model%d.php" % i, line_start=1, line_end=13, text=body, score=0.9 - i * 0.01)
            for i in range(8)]
    pack = build_context(hits, budget_tokens=b.context_tokens_max, intent="factual")
    assert len(pack.sources) == 8 and not pack.truncated and pack.tokens_used <= b.context_tokens_max
    # chaque passage est entier : dernière ligne présente, aucune coupure
    assert all("champ 12" in pack.blocks[s.ref] and "tronqué" not in pack.blocks[s.ref] for s in pack.sources)


def test_compute_budget_partition_never_exceeds_window():
    # Régression : num_ctx=2048 donnait système 600 + historique 512 + contexte 256 + réponse 1024 = 2392 > 2048
    b = compute_budget(LLMConfig(num_ctx=2048), None, None)
    assert b.num_ctx == 2048
    assert b.system_tokens + b.history_tokens_max + b.context_tokens_max + b.answer_reserve <= 2048
    for ctx in (2048, 2560, 3072, 4096, 6144, 8192, 16384, 32768, 65536):
        for mode in ("native", "text", "none"):
            for intent in ("factual", "impact", "broad"):
                b = compute_budget(LLMConfig(num_ctx=ctx, max_context_chars=10**9), info(ctx), profile(64), intent,
                                   tools_mode=mode, question_tokens=40)
                assert b.num_ctx == ctx and b.question_tokens == 40
                assert b.total <= b.num_ctx, (ctx, mode, intent, b.source)
                assert b.answer_reserve >= MIN_ANSWER_RESERVE and b.context_tokens_max >= 0 and b.history_tokens_max >= 0
                assert b.tools_mode in ("native", "text", "none")
                if b.tools_mode == "none":
                    assert b.tools_tokens_max == 0 and b.tool_rounds_max == 0
                else:
                    assert b.tools_mode == mode and b.tools_tokens_max >= MIN_TOOL_BUDGET and b.tool_rounds_max == 4
    # une question énorme ne fait jamais déborder la partition : historique et contexte s'effacent
    huge = compute_budget(LLMConfig(num_ctx=4096), info(4096), None, question_tokens=3500)
    assert huge.total <= 4096 + 3500 and huge.history_tokens_max == 0 and huge.context_tokens_max == 0
    assert "question trop longue" in huge.source


def test_compute_budget_system_tokens_follow_real_prompt():
    # Régression : système = min(1500, max(600, num_ctx // 8)) sous-estimait le vrai prompt (890 tokens en mode
    # texte avec la liste des outils, 573 + 972 de définitions JSON en natif)
    shape = compute_budget(LLMConfig(num_ctx=8192), info(8192), None, tools_mode="none")
    specs = tool_specs_tokens()
    assert specs >= estimate_tokens(json.dumps([t.to_openai() for t in TOOL_SPECS], ensure_ascii=False))
    for mode in ("native", "text", "none"):
        real = estimate_tokens(build_system_prompt(mode, shape, "Boutique", "backend (backend), frontend (frontend)", "2026-10-17"))
        if mode == "native":
            real += specs
        assert system_prompt_tokens(mode, 8192) >= real
        b = compute_budget(LLMConfig(num_ctx=8192, max_context_chars=10**9), info(8192), None, tools_mode=mode)
        assert b.system_tokens >= real and b.tools_mode == mode
    assert system_prompt_tokens("native") > system_prompt_tokens("text") > system_prompt_tokens("none")
    # une mesure fournie est utilisée telle quelle…
    b = compute_budget(LLMConfig(num_ctx=8192), info(8192), None, tools_mode="text", system_tokens=777)
    assert b.system_tokens == 777 and "mesurées" in b.source
    # … sauf si elle ne laisse pas de place aux outils : repli sans outils, consignes ré-estimées dans ce mode
    small = compute_budget(LLMConfig(num_ctx=2048), info(2048), None, tools_mode="native", system_tokens=1600)
    assert small.tools_mode == "none" and small.tool_rounds_max == 0 and small.tools_tokens_max == 0
    assert small.system_tokens == system_prompt_tokens("none", 2048) and "outils désactivés" in small.source
    assert small.total <= 2048


def test_compute_budget_intents_change_context_share():
    cfg = LLMConfig(num_ctx=16384, max_context_chars=1_000_000)
    shares = {intent: compute_budget(cfg, info(16384), None, intent=intent).context_tokens_max
              for intent in ("factual", "fields", "howto", "who_calls", "impact", "decision", "broad", "inconnu")}
    assert shares["factual"] == shares["fields"] == shares["howto"] == shares["decision"] == shares["inconnu"]
    # 80 % du reste pour une question ciblée, 85 % pour les appelants, 90 % pour l'impact et les questions larges
    assert shares["factual"] < shares["who_calls"] < shares["broad"] == shares["impact"]
    assert "impact" in compute_budget(cfg, info(16384), None, intent="impact").source


def test_compute_budget_override_and_max_context_chars():
    cfg = LLMConfig(num_ctx=16384, max_context_chars=2000)
    b = compute_budget(cfg, info(16384), None)
    assert b.context_tokens_max == 500  # 2000 caractères / 4
    assert "max_context_chars" in b.source
    b = compute_budget(LLMConfig(num_ctx=16384), info(16384), None, override_context_tokens=900)
    assert b.context_tokens_max == 900 and "demande" in b.source
    # une demande plus grande que le calcul n'augmente pas le budget ; une demande absurde est bornée
    assert compute_budget(LLMConfig(num_ctx=16384), info(16384), None, override_context_tokens=10**7).context_tokens_max \
        == compute_budget(LLMConfig(num_ctx=16384), info(16384), None).context_tokens_max
    assert compute_budget(LLMConfig(num_ctx=16384), info(16384), None, override_context_tokens=1).context_tokens_max == 256


def test_max_context_chars_default_and_legacy_value_upgraded():
    # 96 000 caractères ≈ 24k tokens : le plafond ne bride plus le contexte récupéré avant 32k de fenêtre…
    assert LLMConfig().max_context_chars == 96_000
    b = compute_budget(LLMConfig(num_ctx=32768), info(32768), profile(64), intent="factual")
    assert b.context_tokens_max > 15_000 and "max_context_chars" not in b.source
    # … et l'ancien défaut (24 000, écrit en clair dans chaque workspace.json) est relu comme le nouveau défaut,
    # une autre valeur explicite est respectée
    assert LLMConfig(max_context_chars=24_000).max_context_chars == 96_000
    assert LLMConfig.model_validate({"max_context_chars": 24000}).max_context_chars == 96_000
    assert LLMConfig(max_context_chars=20_000).max_context_chars == 20_000
    assert compute_budget(LLMConfig(num_ctx=32768, max_context_chars=20_000), info(32768), profile(64)).context_tokens_max == 5_000


def test_budget_to_dict_and_tolerant_inputs():
    b = compute_budget(None, None, None)  # configuration absente : valeurs par défaut
    assert isinstance(b, Budget) and b.num_ctx == 8192
    d = b.to_dict()
    assert set(d) == {"num_ctx", "system_tokens", "question_tokens", "history_tokens_max", "context_tokens_max",
                      "tools_tokens_max", "answer_reserve", "tool_rounds_max", "tools_mode", "source", "prefill_tps",
                      "latency_capped"}
    assert d["tools_mode"] == "native" and d["question_tokens"] == 0
    assert d["prefill_tps"] is None and d["latency_capped"] is False and b.expected_first_token_s is None
    assert compute_budget(None, None, None, tools_mode="turbo").tools_mode == "none"  # mode inconnu = sans outils


def test_compute_budget_latency_cap_bounds_context_by_reading_speed():
    # Régression : avec num_ctx 40 960 l'UI laissait le modèle lire 11,7k tokens de contexte à 60 tokens/s (192 s avant
    # la première lettre). Le contexte est borné par ce que le modèle lit en first_token_target_s (25 s par défaut).
    assert LLMConfig().first_token_target_s == 25.0
    cfg = LLMConfig(num_ctx=32768)
    free = compute_budget(cfg, info(32768), profile(64))  # sans vitesse : aucune borne de latence
    assert free.prefill_tps is None and free.latency_capped is False and free.context_tokens_max > LATENCY_CONTEXT_FLOOR
    # 60 tok/s × 25 s = 1 500 tokens lisibles, moins que les consignes seules : le plancher s'applique
    slow = compute_budget(cfg, info(32768), profile(64), prefill_tps=60)
    assert slow.latency_capped is True and slow.context_tokens_max == LATENCY_CONTEXT_FLOOR == 1500
    assert slow.prefill_tps == 60 and slow.total <= slow.num_ctx
    assert "plafonné à 1500 tokens pour rester sous 25 s (lecture estimée 60 tokens/s)" in slow.source
    assert slow.expected_first_token_s == pytest.approx(slow.prompt_tokens_max / 60) and slow.expected_first_token_s > 25
    # 400 tok/s × 25 s = 10 000 tokens : consignes, question et historique lus d'abord, le reste au contexte
    mid = compute_budget(cfg, info(32768), profile(64), prefill_tps=400, question_tokens=40)
    assert mid.latency_capped is True
    assert mid.context_tokens_max == 10_000 - mid.system_tokens - mid.question_tokens - mid.history_tokens_max
    assert LATENCY_CONTEXT_FLOOR < mid.context_tokens_max < free.context_tokens_max
    assert mid.expected_first_token_s == pytest.approx(25.0)
    # un modèle rapide n'est pas bridé (2 000 tok/s × 25 s = 50 000 tokens > contexte libre)
    fast = compute_budget(cfg, info(32768), profile(64), prefill_tps=2000)
    assert fast.latency_capped is False and fast.context_tokens_max == free.context_tokens_max
    assert "plafonné" not in fast.source and fast.prefill_tps == 2000
    # l'objectif est configurable : 80 s à 400 tok/s = 32 000 tokens, plus que le prompt entier → plus de borne
    patient = compute_budget(LLMConfig(num_ctx=32768, first_token_target_s=80), info(32768), profile(64), prefill_tps=400)
    assert patient.latency_capped is False and patient.first_token_target_s == 80
    assert patient.context_tokens_max == free.context_tokens_max and patient.expected_first_token_s < 80
    # le plancher ne relève jamais un contexte déjà plus petit (fenêtre de 2 048 : tout tient sous 1 500)
    small = compute_budget(LLMConfig(num_ctx=2048), info(2048), None, prefill_tps=60)
    assert small.latency_capped is False
    assert small.context_tokens_max == compute_budget(LLMConfig(num_ctx=2048), info(2048), None).context_tokens_max < 1500
    # la borne s'ajoute aux autres plafonds (demande de l'UI, max_context_chars) : le plus petit gagne
    assert compute_budget(cfg, info(32768), profile(64), prefill_tps=400, override_context_tokens=900).context_tokens_max == 900
    # vitesses inexploitables : pas de borne
    for tps in (0, -5, "lent", float("nan")):
        assert compute_budget(cfg, info(32768), profile(64), prefill_tps=tps).latency_capped is False  # type: ignore[arg-type]


def test_estimate_prefill_tps_follows_generation_speed_then_model_size():
    from develop.llm.advisor import (PREFILL_PER_GENERATION, PREFILL_TPS_MAX, PREFILL_TPS_MIN, estimate_prefill_tps,
                                     estimate_speed, model_params_b)

    # taille du modèle : parameter_size d'Ollama (« 7.6B », « 137M ») ou le nom (« qwen3:14b », « qwen3.5:9b »)
    assert model_params_b("qwen3:14b") == 14 and model_params_b("qwen3.5:9b") == 9 and model_params_b("qwen3-embedding:0.6b") == 0.6
    assert model_params_b(ModelInfo(name="x", parameter_size="7.6B")) == 7.6 and model_params_b({"name": "gemma3:27b"}) == 27
    assert model_params_b(ModelInfo(name="nomic-embed-text", parameter_size="137M")) == 0.137
    assert model_params_b("qwen3:30b-a3b") == 30 and model_params_b("mistral:latest") is None and model_params_b(None) is None
    # machine connue : ≈ 6 × la vitesse de génération estimée, bornée [40, 2000]
    m1max = MachineProfile(os="macos", os_version="15", arch="arm64", cpu_model="Apple M1 Max", cpu_cores_physical=10,
                           cpu_cores_logical=10, ram_total_gb=32, ram_available_gb=20, apple_silicon=True, unified_memory=True)
    gen = estimate_speed({"params_b": 14}, m1max)
    assert gen and estimate_prefill_tps("qwen3:14b", m1max) == pytest.approx(PREFILL_PER_GENERATION * gen)
    assert PREFILL_TPS_MIN < estimate_prefill_tps("qwen3.8:27b", m1max) < estimate_prefill_tps("qwen3.5:9b", m1max) < PREFILL_TPS_MAX
    assert estimate_prefill_tps("all-minilm:33m", m1max) == PREFILL_TPS_MAX  # minuscule : plafond
    assert estimate_prefill_tps("llama3:70b", profile(32)) == PREFILL_TPS_MIN  # CPU seul : < 40 tok/s → plancher
    # machine inconnue : repli par taille (150 < 10B, 100 < 20B, 60 au-delà), 100 si la taille manque aussi
    assert estimate_prefill_tps("qwen3.5:9b", None) == 150 and estimate_prefill_tps("qwen3:14b", None) == 100
    assert estimate_prefill_tps(ModelInfo(name="big", parameter_size="27B"), None) == 60
    assert estimate_prefill_tps("mistral:latest", None) == 100 and estimate_prefill_tps(None, None) == 100


# =========================================================================== assembleur de contexte


def test_build_context_keeps_score_order_and_numbers_sources():
    hits = [doc_hit(score=0.6), file_hit(score=0.9), note_hit(score=0.5)]
    pack = build_context(hits, budget_tokens=4000, intent="factual")
    assert [s.ref for s in pack.sources] == ["S1", "S2", "S3"]
    assert [s.source_type for s in pack.sources] == ["file", "document", "note"]
    assert pack.hits_by_ref["S1"] is hits[1]
    assert pack.truncated == []
    assert 0 < pack.tokens_used <= 4000
    assert pack.text.index("[S1]") < pack.text.index("[S2]") < pack.text.index("[S3]")


def test_build_context_block_formats():
    pack = build_context([file_hit(), doc_hit(), note_hit()], budget_tokens=4000)
    blocks = pack.text.split("\n\n")
    assert blocks[0].startswith("[S1] backend/app/Services/OrderService.php:L10-23 (symbole OrderService.confirm) — identifiant OrderService.confirm\n```php\n")
    assert blocks[0].endswith("\n```")
    assert "public function confirm" in blocks[0]
    assert blocks[1].startswith("[S2] Document « Dictionnaire de données » §Table orders, page 3 — mentionne la table orders\n```text\n")
    assert blocks[2].startswith("[S3] Note du 2026-09-01 — Réunion statuts [décision, commandes] — note datée du 2026-09-01\n```text\n")


def test_build_context_keeps_one_hit_per_channel():
    big = "x" * 1500 + "\n" + "y" * 1500  # ≈ 860 tokens chacun en code
    hits = [
        file_hit(chunk_id=1, source_id=1, path="a.php", score=0.9, channels=["lexical"], text=big),
        file_hit(chunk_id=2, source_id=2, path="b.php", score=0.8, channels=["lexical"], text=big),
        file_hit(chunk_id=3, source_id=3, path="c.php", score=0.7, channels=["lexical"], text=big),
        file_hit(chunk_id=4, source_id=4, path="d.php", score=0.2, channels=["semantic"], text="petit passage sémantique"),
        note_hit(chunk_id=5, score=0.1, channels=["temporal"]),
        file_hit(chunk_id=6, source_id=6, path="e.php", score=0.05, channels=["graph"], text="voisin du graphe"),
    ]
    pack = build_context(hits, budget_tokens=1100)
    channels = [set(s.channels) for s in pack.sources]
    assert {"semantic"} in channels and {"temporal"} in channels and {"graph"} in channels
    assert any("lexical" in c for c in channels)
    assert pack.sources[0].path == "a.php"  # le meilleur score reste S1
    excluded = {s.path or s.title for s in pack.truncated}
    assert "c.php" in excluded or "b.php" in excluded
    assert pack.tokens_used <= 1100


def test_build_context_truncates_last_block_between_lines():
    lines = ["ligne numéro %03d : du contenu pour remplir" % i for i in range(1, 201)]
    hit = file_hit(text="\n".join(lines), line_start=1, line_end=200)
    pack = build_context([hit], budget_tokens=600)
    assert len(pack.sources) == 1 and pack.truncated == []
    block = pack.blocks["S1"]
    assert block.startswith("[S1] ")
    assert "non affichées)" in block
    body = block.split("```php\n", 1)[1].rsplit("\n```", 1)[0]
    kept = body.splitlines()[:-1]
    assert kept == lines[: len(kept)]  # jamais de coupure au milieu d'une ligne
    assert 3 <= len(kept) < 200
    assert pack.tokens_used <= 600
    # un budget trop petit pour tout : seconde source écartée et listée dans truncated
    hit.chunk_id, hit.source_id = 50, 5
    pack2 = build_context([file_hit(chunk_id=1, score=0.9), hit], budget_tokens=120)
    assert len(pack2.sources) >= 1 and len(pack2.truncated) >= 1
    assert all(s.ref == "" for s in pack2.truncated)


def test_build_context_dedupes_and_caps_per_file():
    hits = [file_hit(chunk_id=i, line_start=i * 10, line_end=i * 10 + 5, score=1 - i / 10, text="bloc %d" % i) for i in range(1, 7)]
    hits.append(file_hit(chunk_id=1, score=0.95, text="doublon"))  # même chunk_id → ignoré
    pack = build_context(hits, budget_tokens=5000, max_per_file=3)
    assert len(pack.sources) == 3
    assert [s.line_start for s in pack.sources] == [10, 20, 30]
    empty = build_context([file_hit(text="", snippet="")], budget_tokens=1000)
    assert empty.sources == [] and "Aucune source" in render_sources_for_model(empty)
    assert build_context([], 0).text == ""


def test_build_context_previews_files_reached_by_the_graph_only():
    """Projet réel : un fichier relié par le graphe seul (importateur d'une classe, fichier utilisant un modèle)
    entrait par son premier morceau — namespace, dix lignes ``use``, déclaration — soit 300 à 400 tokens par
    fichier et jusqu'à 70 % du contexte. La relation est la preuve : le passage devient un aperçu (en-tête omis,
    GRAPH_FILE_PREVIEW_LINES lignes, reste compté) ; un fichier trouvé aussi par un autre canal reste entier."""
    header = ("# backend/app/Http/Controllers/EmpruntController.php L1-40\n<?php\n\nnamespace Modules\\Bibliotheque\\Http;\n\n"
              + "".join("use Modules\\Bibliotheque\\Models\\Model%d;\n" % i for i in range(10))
              + "\nclass EmpruntController extends Controller\n{\n    public function __construct(EmpruntService $service)\n"
              "    {\n        parent::__construct($service);\n    }\n\n"
              + "".join("    public function action%d(Request $request): JsonResponse\n    {\n"
                        "        return $this->paginateModel($request->only(['sort']));\n    }\n\n" % i for i in range(5)))
    linked = file_hit(chunk_id=1, source_id=1, file_id=1, path="app/Http/Controllers/EmpruntController.php",
                      title="EmpruntController.php", line_start=1, line_end=40, channels=["graph"],
                      reason="relié à Emprunt via utilisation du modèle", text=header,
                      extra={"language": "php", "repo_name": "backend", "relation": "uses_model", "via": "Emprunt"})
    matched = file_hit(chunk_id=2, source_id=2, file_id=2, path="app/Http/Controllers/LivreController.php", score=0.8,
                       title="LivreController.php", line_start=1, line_end=40, channels=["graph", "lexical"],
                       reason="relié à Emprunt via imports ; correspondance lexicale sur « livre »", text=header,
                       extra={"language": "php", "repo_name": "backend", "relation": "imports"})
    short = file_hit(chunk_id=3, source_id=3, file_id=3, path="app/Http/Requests/IndexEmpruntRequest.php", score=0.7,
                     title="IndexEmpruntRequest.php", line_start=1, line_end=7, channels=["graph"],
                     reason="relié à Emprunt via utilisation du modèle", extra={"language": "php", "repo_name": "backend"},
                     text="class IndexEmpruntRequest extends FormRequest\n{\n    public function authorize(): bool\n"
                          "    {\n        return true;\n    }\n}")
    pack = build_context([linked, matched, short], budget_tokens=4000)
    assert [s.ref for s in pack.sources] == ["S1", "S2", "S3"] and pack.truncated == []
    b1, b2, b3 = pack.blocks["S1"], pack.blocks["S2"], pack.blocks["S3"]
    # aperçu : en-tête définitif, 13 lignes d'en-tête omises (ligne du morceau, <?php, namespace, dix use), puis
    # GRAPH_FILE_PREVIEW_LINES lignes non vides (déclaration, constructeur, première action), le reste compté
    assert b1.startswith("[S1] backend/app/Http/Controllers/EmpruntController.php:L1-40 — relié à Emprunt via utilisation du modèle\n```php\n")
    body = b1.split("```php\n", 1)[1].rsplit("\n```", 1)[0].splitlines()
    assert body[0] == IMPORTS_OMITTED_MARKER % 13 and "use Modules" not in b1 and "namespace Modules" not in b1
    assert body[1] == "class EmpruntController extends Controller" and "parent::__construct($service);" in b1
    assert len([l for l in body[1:-1] if l.strip()]) == GRAPH_FILE_PREVIEW_LINES and "action0" in b1 and "action1" not in b1
    assert body[-1] == "… (tronqué : 22 lignes non affichées)"  # 31 lignes utiles après l'en-tête, 9 montrées (dont une vide)
    assert estimate_tokens(b1, is_code=True) * 3 < estimate_tokens(b2, is_code=True)
    # trouvé aussi par un mot de la question : montré entier ; passage court sans en-tête : rien à réduire
    assert "use Modules\\Bibliotheque\\Models\\Model9;" in b2 and "action4" in b2 and "omises" not in b2
    assert "omises" not in b3 and "non affichées" not in b3 and "return true;" in b3
    assert pack.tokens_used == sum(estimate_tokens(b, is_code=True) for b in (b1, b2, b3))
    # l'aperçu entre là où le passage entier ne tenait pas ; trop petit pour lui, il est écarté (jamais tronqué)
    tight = build_context([linked], budget_tokens=estimate_tokens(b1, is_code=True) + 5)
    assert tight.blocks["S1"] == b1 and tight.truncated == []
    assert build_context([linked], budget_tokens=20).sources == []


def test_build_context_leaves_out_generated_file_passages_unless_cited_or_alone():
    """Projet réel : ``storage/api-docs/api-docs.json`` (851 morceaux, étiquette « fichier généré volumineux » du moteur,
    classé en queue) entrait tout de même à 16k — trois schémas OpenAPI tronqués, 1 358 tokens, rien pour la réponse.
    Un tel passage n'entre que si la question cite le fichier (canal identifier) ou s'il n'y a aucun autre passage ;
    écarté, il reste listé dans ``truncated`` avec sa raison."""
    schema = '"/api/v1/login": {\n  "post": {\n    "summary": "authentification",\n    "tags": ["Auth"]\n  }\n}'
    generated = [file_hit(chunk_id=10 + i, source_id=9, file_id=9, path="storage/api-docs/api-docs.json", title="api-docs.json",
                          line_start=19063 + i * 300, line_end=19099 + i * 300, score=0.008 - i / 1000, channels=["lexical"],
                          reason="correspondance lexicale sur « authentification » (fichier généré volumineux)",
                          extra={"language": "json", "generated": True, "fts_terms": ["authentification"]}, text=schema)
                 for i in range(3)]
    normal = [file_hit(chunk_id=1, source_id=1, file_id=1, path="routes/api.php", title="api.php", score=0.9, channels=["lexical"],
                       reason="correspondance lexicale sur « login »", text="Route::post('login', [AuthController::class, 'login']);"),
              file_hit(chunk_id=2, source_id=2, file_id=2, path="app/Services/AuthService.php", title="AuthService.php", score=0.5,
                       channels=["lexical"], reason="correspondance lexicale sur « authentification »",
                       text="    public function login(array $credentials): array\n    {\n        return $this->attempt($credentials);\n    }")]
    pack = build_context(normal + generated, budget_tokens=20000)  # budget large : rien n'est écarté faute de place
    assert [s.path for s in pack.sources] == ["routes/api.php", "app/Services/AuthService.php"]
    assert [s.path for s in pack.truncated] == ["storage/api-docs/api-docs.json"] * 3
    assert all(s.ref == "" and "fichier généré volumineux" in s.reason for s in pack.truncated)
    assert "api-docs" not in pack.text and pack.tokens_used == build_context(normal, budget_tokens=20000).tokens_used
    # chemin typique remonté par un autre canal, sans l'étiquette du canal lexical : même règle
    lock = file_hit(chunk_id=20, source_id=8, file_id=8, path="composer.lock", title="composer.lock", score=0.3, channels=["semantic"],
                    reason="proximité sémantique", extra={"language": "json"}, text='"name": "laravel/sanctum",\n"version": "v3.2.1"')
    pack = build_context(normal + [lock], budget_tokens=20000)
    assert [s.path for s in pack.sources] == ["routes/api.php", "app/Services/AuthService.php"]
    assert [s.path for s in pack.truncated] == ["composer.lock"]
    # la question cite le fichier (canal identifier) : il entre, à son rang
    cited = file_hit(chunk_id=11, source_id=9, file_id=9, path="storage/api-docs/api-docs.json", title="api-docs.json", score=0.95,
                     line_start=19363, line_end=19399, channels=["identifier", "lexical"], extra={"language": "json", "generated": True},
                     reason="chemin de fichier « storage/api-docs/api-docs.json » ; correspondance lexicale (fichier généré volumineux)",
                     text=schema)
    pack = build_context(normal + generated + [cited], budget_tokens=20000)
    assert [s.path for s in pack.sources] == ["storage/api-docs/api-docs.json", "routes/api.php", "app/Services/AuthService.php"]
    assert pack.sources[0].line_start == 19363 and len(pack.truncated) == 3  # les morceaux non cités restent écartés
    # aucun autre passage : mieux vaut un passage faible qu'un contexte vide
    pack = build_context(generated, budget_tokens=20000)
    assert [s.path for s in pack.sources] == ["storage/api-docs/api-docs.json"] * 3 and pack.truncated == []
    # un autre passage, mais vide : même repli (rien d'autre à montrer)
    pack = build_context([file_hit(chunk_id=30, source_id=3, file_id=3, path="x.php", text="", snippet="")] + generated, budget_tokens=20000)
    assert len(pack.sources) == 3 and pack.truncated == []


def test_build_context_keeps_one_copy_of_a_passage_indexed_under_two_sources():
    # même PDF indexé comme fichier du dépôt (file 571) et comme document téléversé (document 2) : mêmes
    # morceaux, même ``chunks.sha1`` transmis par le moteur dans extra → un seul passage dans le contexte
    text = "Paramétrage des formulaires\nChaque formulaire déclare ses champs : libellé, type, obligatoire."
    as_file = file_hit(chunk_id=11, source_id=571, file_id=571, path="public/documentation/parametrage.pdf",
                       title="parametrage.pdf", line_start=1, line_end=2, score=0.8, channels=["lexical"], text=text,
                       extra={"language": "pdf", "sha1": "a" * 40})
    as_doc = doc_hit(chunk_id=22, source_id=2, title="Paramétrage des formulaires de commande", line_start=1, line_end=2,
                     page=1, score=0.79, text=text, extra={"document_id": 2, "sha1": "a" * 40})
    other = doc_hit(chunk_id=23, source_id=2, title="Paramétrage des formulaires de commande", line_start=3, line_end=4,
                    page=2, score=0.7, text="Autre page : les valeurs par défaut.", extra={"document_id": 2, "sha1": "b" * 40})
    pack = build_context([as_file, as_doc, other], budget_tokens=4000, max_per_file=1)
    assert [(s.ref, s.source_type, s.line_start) for s in pack.sources] == [("S1", "file", 1), ("S2", "document", 3)]
    assert pack.text.count("Chaque formulaire déclare") == 1 and pack.truncated == []
    # le doublon écarté n'a pas consommé la place par source du document (S2 est bien entré malgré max_per_file=1)
    assert pack.hits_by_ref["S2"] is other
    # un outil qui ramène le même passage sous l'autre source ne l'ajoute pas ; un texte nouveau, si
    twin = LocalHit(source_type="document", source_id=2, chunk_id=22, title="Paramétrage des formulaires de commande",
                    line_start=1, line_end=2, text=text, channels=["tool"], extra={"sha1": "a" * 40})
    fresh = LocalHit(source_type="note", source_id=9, chunk_id=30, title="Note", text="Les champs obligatoires sont marqués *.",
                     channels=["tool"], extra={"sha1": "c" * 40})
    assert [s.source_type for s in add_tool_sources(pack, [twin, fresh], budget_tokens=500)] == ["note"]
    assert len(pack.sources) == 3 and pack.text.count("Chaque formulaire déclare") == 1
    # sans empreinte, rien n'est rapproché par le texte : deux fichiers distincts au contenu identique restent deux sources
    plain = [file_hit(chunk_id=i, source_id=i, file_id=i, path="app/Models/M%d.php" % i, text="<?php // identique") for i in (1, 2)]
    assert len(build_context(plain, budget_tokens=4000).sources) == 2


def test_add_tool_sources_respects_budget_and_skips_covered():
    pack = build_context([file_hit()], budget_tokens=2000)
    covered = LocalHit(source_type="file", source_id=1, path="app/Services/OrderService.php", line_start=12, line_end=20,
                       text="inclus dans S1", channels=["tool"])
    fresh = LocalHit(source_type="file", source_id=2, path="routes/api.php", line_start=11, line_end=13,
                     text="Route::post('/orders/{order}/confirm', ...);", channels=["tool"], reason="lu via read_file",
                     extra={"language": "php"})
    assert add_tool_sources(pack, [covered, fresh], budget_tokens=0) == []
    added = add_tool_sources(pack, [covered, fresh], budget_tokens=500)
    assert [s.ref for s in added] == ["S2"] and added[0].path == "routes/api.php"
    assert len(pack.sources) == 2 and "[S2] routes/api.php:L11-13" in pack.text
    rendered = render_sources_for_model(pack, ["S2"])
    assert rendered.startswith("Nouvelles sources citables") and "[S2]" in rendered and "[S1]" not in rendered
    # un hit trop gros pour le budget restant est ignoré (ou tronqué), jamais ajouté en entier
    huge = LocalHit(source_type="file", source_id=3, path="big.php", line_start=1, line_end=2, text="z" * 4000, channels=["tool"])
    before = pack.tokens_used
    added2 = add_tool_sources(pack, [huge], budget_tokens=30)
    assert added2 == [] and pack.tokens_used == before


def test_source_to_dict_matches_api_contract():
    src = Source.from_hit("S1", doc_hit())
    d = src.to_dict()
    assert set(d) == {"ref", "source_type", "path", "title", "line_start", "line_end", "page", "repo_id", "file_id",
                      "document_id", "note_id", "symbol_id", "endpoint_id", "model_id", "score", "channels", "reason"}
    assert d["ref"] == "S1" and d["document_id"] == 7 and d["page"] == 3 and "text" not in d
    assert Source.from_hit("S2", note_hit()).note_id == 4
    assert Source.from_hit("S3", {"source_type": "symbol", "source_id": 9, "text": "x"}).symbol_id == 9


def test_render_sources_as_text_lists_everything():
    pack = build_context([file_hit(), doc_hit(), note_hit()], budget_tokens=4000)
    text = render_sources_as_text(pack)
    assert text.startswith("Sources trouvées (3)")
    assert "[S1] backend/app/Services/OrderService.php:L10-23" in text or "[S1] app/Services/OrderService.php:L10-23" in text
    assert "[S2] Document « Dictionnaire de données » (page 3)" in text
    assert "[S3] Note « Réunion statuts »" in text


# =========================================================================== prompts


def test_build_system_prompt_modes():
    budget = compute_budget(LLMConfig(), info(8192), None)
    native = build_system_prompt("native", budget, "Boutique", "backend (backend), frontend (frontend)", "2026-10-17")
    text = build_system_prompt("text", budget, "Boutique", "", "2026-10-17")
    none = build_system_prompt("none", budget, "Boutique", "", "2026-10-17")
    for p in (native, text, none):
        assert p.startswith(SYSTEM_PROMPT) and "Boutique" in p and "2026-10-17" in p and "[S1]" in p
    assert "Dépôts indexés : backend (backend), frontend (frontend)." in native
    assert "appeler les outils" in native and '{"tool"' not in native
    assert '{"tool": "<nom>", "args": {...}}' in text and "4 tours" in text
    for spec in TOOL_SPECS:
        assert "- %s(" % spec.name in text
    assert "Aucun outil" in none and "tool" not in none.lower().split("aucun outil")[1][:5]
    assert TOOLS_TEXT_PROTOCOL.startswith("Outils (protocole textuel)")


def test_system_prompt_asks_for_complete_answers_and_stays_under_900_tokens():
    # consigne de longueur : complète mais sans délayage, ordre de grandeur donné pour une question factuelle
    assert "sans délayage" in SYSTEM_PROMPT and "150-300 mots" in SYSTEM_PROMPT
    # le prompt système (hors définitions d'outils natifs) reste ≤ 900 tokens dans tous les modes, protocole textuel
    # (le plus long, avec la liste des outils) compris : chaque token de consigne est pris au contexte récupéré
    budget = compute_budget(LLMConfig(num_ctx=32768), info(32768), None)
    for mode in ("native", "text", "none"):
        for repos in ("", "backend (backend), frontend (frontend)"):
            text = build_system_prompt(mode, budget, "Boutique", repos, "2026-10-17")
            assert estimate_tokens(text) <= 900, (mode, repos, estimate_tokens(text))


# =========================================================================== outils


@pytest.fixture
def executor(db: Database, workspace: WorkspaceConfig, seeded: Dict[str, Any]) -> ToolExecutor:
    return ToolExecutor(db, workspace, search_engine=None, impact_analyzer=FakeImpactAnalyzer())


async def test_tool_search_fallback_fts_and_engine(executor: ToolExecutor, db: Database, workspace: WorkspaceConfig):
    res = await executor.execute(ToolCall(name="search", arguments={"query": "confirm commande brouillon"}), 2000)
    assert res.error is None and res.hits and res.name == "search"
    assert res.hits[0].channels == ["lexical"] and res.hits[0].source_type == "file"
    assert "résultat(s) pour « confirm commande brouillon »" in res.text and "OrderService.php" in res.text
    assert len(res.summary) <= 200
    engine = FakeSearchEngine([file_hit(), doc_hit()])
    ex2 = ToolExecutor(db, workspace, search_engine=engine)
    res2 = await ex2.execute(ToolCall(name="search", arguments='{"query": "status", "source_types": "file,document"}'), 2000)
    assert engine.calls[0]["query"] == "status" and engine.calls[0]["source_types"] == ["file", "document"]
    assert engine.calls[0]["limit"] <= 10 and engine.embed_calls == 1
    assert [h.source_type for h in res2.hits] == ["file", "document"]
    assert "1. app/Services/OrderService.php:L10-23" in res2.text and "2. Document « Dictionnaire de données » p.3" in res2.text
    none = await ex2.execute(ToolCall(name="search", arguments={}), 2000)
    assert none.error and "query" in none.error
    engine.hits = []
    nothing = await ex2.execute(ToolCall(name="search", arguments={"query": "zzz"}), 2000)
    assert nothing.error is None and nothing.hits == [] and "Aucun résultat" in nothing.summary


async def test_tool_read_file_suffix_bounds_and_errors(executor: ToolExecutor, db: Database, workspace: WorkspaceConfig, tmp_path: Path):
    res = await executor.execute(ToolCall(name="read_file", arguments={"path": "Services/OrderService.php", "start": 10, "end": 500}), 4000)
    assert res.error is None
    assert res.text.startswith("app/Services/OrderService.php:L10-29 (sur 29 lignes)\n")
    assert "public function confirm" in res.text and "<?php" not in res.text
    assert len(res.hits) == 1 and res.hits[0].line_start == 10 and res.hits[0].line_end == 29
    assert res.hits[0].extra["language"] == "php" and res.hits[0].file_id is not None
    # nom de base seul et préfixe « backend/ » (nom du dépôt) acceptés
    for path in ("OrderService.php", "backend/app/Services/OrderService.php", "./app/Services/OrderService.php"):
        r = await executor.execute(ToolCall(name="read_file", arguments={"path": path}), 4000)
        assert r.error is None and r.hits[0].path == "app/Services/OrderService.php", path
    # start au-delà de la fin → borné, pas d'erreur
    r = await executor.execute(ToolCall(name="read_file", arguments={"path": "routes/api.php", "start": "9999"}), 4000)
    assert r.error is None and r.hits[0].line_start == 14
    missing = await executor.execute(ToolCall(name="read_file", arguments={"path": "nope/Inexistant.php"}), 4000)
    assert missing.error == "fichier introuvable" and "introuvable" in missing.summary
    no_arg = await executor.execute(ToolCall(name="read_file", arguments={}), 4000)
    assert no_arg.error and "path" in no_arg.error
    # fenêtre de 120 lignes au plus (dépôt temporaire)
    repo = tmp_path / "big"
    repo.mkdir()
    (repo / "long.py").write_text("\n".join("line %d" % i for i in range(1, 301)), encoding="utf-8")
    db.upsert_repo(RepoRow(id="tmp", name="tmp", path=str(repo), kind="other"))
    db.upsert_file(FileRow(repo_id="tmp", path="long.py", language="python", kind="code"))
    r = await executor.execute(ToolCall(name="read_file", arguments={"path": "long.py", "start": 1, "end": 300}), 10000)
    assert r.hits[0].line_start == 1 and r.hits[0].line_end == 120 and r.text.rstrip().endswith("line 120")
    # fichier indexé mais absent du disque → erreur propre
    db.upsert_file(FileRow(repo_id="tmp", path="ghost.py", language="python", kind="code"))
    r = await executor.execute(ToolCall(name="read_file", arguments={"path": "ghost.py"}), 4000)
    assert r.error == "lecture impossible"
    # le moteur de recherche peut fournir la fenêtre (file_snippet)
    snip = {"path": "app/Services/OrderService.php", "start": 10, "end": 12, "text": "a\nb\nc", "total_lines": 29}
    ex3 = ToolExecutor(db, workspace, search_engine=FakeSearchEngine(snippet=snip))
    r = await ex3.execute(ToolCall(name="read_file", arguments={"path": "OrderService.php", "start": 10, "end": 12}), 4000)
    assert r.text.endswith("a\nb\nc") and r.hits[0].line_end == 12


async def test_tool_symbol_and_callers(executor: ToolExecutor, seeded: Dict[str, Any]):
    res = await executor.execute(ToolCall(name="symbol", arguments={"name": "confirm"}), 4000)
    assert res.error is None and res.text.startswith("2 symbole(s) pour « confirm »")
    assert "OrderService.confirm [method] app/Services/OrderService.php:L10-23 — confirm(Order $order): Order" in res.text
    assert "OrderController.confirm [method]" in res.text
    assert len(res.hits) == 2 and all(h.source_type == "symbol" for h in res.hits)
    assert res.hits[0].extra["symbol_id"] == seeded["svc_confirm"] and "public function confirm" in res.hits[0].text
    assert res.hits[0].line_start == 10 and res.hits[0].line_end == 23
    unknown = await executor.execute(ToolCall(name="symbol", arguments={"name": "nope_zzz"}), 4000)
    assert unknown.error is None and "introuvable" in unknown.summary and unknown.hits == []
    callers = await executor.execute(ToolCall(name="callers", arguments={"name": "OrderService"}), 4000)
    assert callers.error is None
    callers = await executor.execute(ToolCall(name="callers", arguments={"name": "confirm"}), 4000)
    assert "OrderController.confirm (app/Http/Controllers/OrderController.php:L25) [calls, confiance 0.90]" in callers.text
    assert "POST /orders/{order}/confirm [server]" in callers.text  # relation handles
    assert callers.hits and callers.hits[0].extra["symbol_id"] == seeded["ctl_confirm"]
    assert callers.hits[0].reason == "appelant de confirm"
    lonely = await executor.execute(ToolCall(name="callers", arguments={"name": "totalFor"}), 4000)
    assert "Aucune relation entrante" in lonely.text and "0 appelant(s)" in lonely.summary


async def test_tool_endpoint_matches_server_and_client(executor: ToolExecutor):
    res = await executor.execute(ToolCall(name="endpoint", arguments={"path": "/orders/5/confirm", "method": "post"}), 4000)
    assert res.error is None and res.text.startswith("2 correspondance(s) pour POST /orders/5/confirm (normalisé : /orders/:param/confirm)")
    assert "[server] POST /orders/{order}/confirm → OrderController@confirm (laravel) routes/api.php:L13" in res.text
    assert "[client] POST /orders/${id}/confirm → confirmOrder (axios) src/api/client.ts:L33" in res.text
    assert {h.source_type for h in res.hits} == {"endpoint"} and {h.repo_id for h in res.hits} == {"back", "front"}
    server = [h for h in res.hits if h.repo_id == "back"][0]
    assert "Route::post('/orders/{order}/confirm'" in server.text and server.line_start <= 13 <= server.line_end
    assert server.extra["endpoint_id"] and server.title == "POST /orders/{order}/confirm"
    only_get = await executor.execute(ToolCall(name="endpoint", arguments={"path": "/orders", "method": "GET"}), 4000)
    assert "1 correspondance(s)" in only_get.text and "OrderController@index" in only_get.text
    none = await executor.execute(ToolCall(name="endpoint", arguments={"path": "/customers/42/invoices"}), 4000)
    assert none.error is None and "Aucune route" in none.summary
    assert (await executor.execute(ToolCall(name="endpoint", arguments={}), 4000)).error


async def test_tool_model_by_name_table_or_substring(executor: ToolExecutor, seeded: Dict[str, Any]):
    res = await executor.execute(ToolCall(name="model", arguments={"name": "Order"}), 4000)
    assert res.error is None
    assert res.text.startswith("Modèle Order (eloquent, table orders) — app/Models/Order.php:L9")
    assert "- status : enum (valeurs : draft, confirmed)" in res.text
    assert "- confirmed_at : datetime (nullable)" in res.text
    assert "- belongsTo → Customer (customer_id)" in res.text
    assert len(res.hits) == 1 and res.hits[0].source_type == "model" and res.hits[0].extra["model_id"] == seeded["model_id"]
    by_table = await executor.execute(ToolCall(name="model", arguments={"table": "orders"}), 4000)
    assert "Modèle Order" in by_table.text
    partial = await executor.execute(ToolCall(name="model", arguments={"name": "ord"}), 4000)
    assert "Modèle Order" in partial.text
    missing = await executor.execute(ToolCall(name="model", arguments={"name": "Invoice"}), 4000)
    assert missing.error is None and "introuvable" in missing.summary


async def test_tool_notes_query_and_dates(executor: ToolExecutor, seeded: Dict[str, Any]):
    res = await executor.execute(ToolCall(name="notes", arguments={"query": "statut paid"}), 4000)
    assert res.error is None and res.text.startswith("2 note(s) pour « statut paid »")
    assert "Note du 2026-09-01 — Réunion statuts [décision, commandes]" in res.text
    assert {h.extra["note_date"] for h in res.hits} == {"2026-09-01", "2026-06-15"}
    assert all(h.source_type == "note" for h in res.hits)
    recent = await executor.execute(ToolCall(name="notes", arguments={"query": "paid", "date_from": "2026-08-01"}), 4000)
    assert recent.text.startswith("1 note(s)") and "Réunion statuts" in recent.text and "Brainstorm" not in recent.text
    window = await executor.execute(ToolCall(name="notes", arguments={"date_from": "2026-06-01", "date_to": "2026-06-30"}), 4000)
    assert window.text.startswith("1 note(s)") and "Brainstorm paiement" in window.text
    latest = await executor.execute(ToolCall(name="notes", arguments={}), 4000)
    assert latest.text.index("Réunion statuts") < latest.text.index("Brainstorm paiement")  # plus récente d'abord
    nothing = await executor.execute(ToolCall(name="notes", arguments={"query": "kubernetes", "date_to": "2020-01-01"}), 4000)
    assert nothing.error is None and "Aucune note" in nothing.summary and nothing.hits == []
    # repli sans FTS : recherche dans titre/corps/étiquettes
    tag = await executor.execute(ToolCall(name="notes", arguments={"query": "décision"}), 4000)
    assert "Réunion statuts" in tag.text


async def test_tool_impact_with_and_without_analyzer(executor: ToolExecutor, db: Database, workspace: WorkspaceConfig, seeded):
    res = await executor.execute(ToolCall(name="impact", arguments={"path": "OrderService.php"}), 4000)
    assert res.error is None and res.text.startswith("Analyse d'impact pour app/Services/OrderService.php :")
    assert "Changé : app/Services/OrderService.php" in res.text
    assert "- symbol OrderController.confirm (app/Http/Controllers/OrderController.php:L25) — appelle OrderService.confirm [confiance 0.90, 1 saut(s)]" in res.text
    assert "Routes (1) : POST /orders/{order}/confirm" in res.text and "Avertissement : index partiel" in res.text
    assert "2 élément(s) potentiellement affecté(s)" in res.summary
    assert len(res.hits) == 2 and res.hits[0].channels == ["graph"] and res.hits[0].path == "app/Http/Controllers/OrderController.php"
    analyzer = executor.impact_analyzer
    assert analyzer.calls[0] == ("file", "back", "app/Services/OrderService.php", 2)
    by_symbol = await executor.execute(ToolCall(name="impact", arguments={"symbol": "OrderService", "max_hops": 3}), 4000)
    assert by_symbol.error is None and analyzer.calls[-1][0] == "symbol" and analyzer.calls[-1][2] == 3
    assert (await executor.execute(ToolCall(name="impact", arguments={"path": "nope.php"}), 4000)).error == "fichier introuvable"
    assert (await executor.execute(ToolCall(name="impact", arguments={"symbol": "zzz"}), 4000)).error == "symbole introuvable"
    assert (await executor.execute(ToolCall(name="impact", arguments={}), 4000)).error
    without = ToolExecutor(db, workspace)
    res2 = await without.execute(ToolCall(name="impact", arguments={"path": "OrderService.php"}), 4000)
    assert res2.error == "analyseur absent" and "indisponible" in res2.summary


async def test_tool_list_dir(executor: ToolExecutor):
    root = await executor.execute(ToolCall(name="list_dir", arguments={}), 4000)
    assert root.error is None and root.text.startswith("Contenu de « / » (5 fichier(s) indexé(s))")
    assert "- backend/app/ (3 fichier(s))" in root.text and "- backend/routes/ (1 fichier(s))" in root.text
    assert "- frontend/src/ (1 fichier(s))" in root.text
    app = await executor.execute(ToolCall(name="list_dir", arguments={"path": "app"}), 4000)
    assert "- backend/app/Services/ (1 fichier(s))" in app.text and "- backend/app/Http/ (1 fichier(s))" in app.text
    services = await executor.execute(ToolCall(name="list_dir", arguments={"path": "backend/app/Services/"}), 4000)
    assert "- backend/app/Services/OrderService.php (php, 700 o)" in services.text
    assert "0 dossier(s), 1 fichier(s)" in services.summary
    empty = await executor.execute(ToolCall(name="list_dir", arguments={"path": "nowhere"}), 4000)
    assert empty.error is None and "vide ou inconnu" in empty.summary


async def test_tool_executor_never_raises_and_truncates(executor: ToolExecutor, db: Database, monkeypatch):
    unknown = await executor.execute(ToolCall(name="teleport", arguments={"x": 1}), 4000)
    assert unknown.error == "Outil inconnu : teleport" and "search" in unknown.text and unknown.hits == []

    def boom(*args, **kwargs):
        raise RuntimeError("base verrouillée")

    monkeypatch.setattr(db, "search_symbols", boom)
    res = await executor.execute(ToolCall(name="symbol", arguments={"name": "confirm"}), 4000)
    assert res.error == "base verrouillée" and res.text.startswith("Erreur de l'outil symbol")
    assert len(res.summary) <= 200
    monkeypatch.undo()
    # arguments sous forme de texte brut → interprétés comme requête
    res = await executor.execute(ToolCall(name="search", arguments="confirm"), 4000)
    assert res.error is None and res.hits
    # troncature au budget avec mention explicite (coupure entre deux lignes)
    full = await executor.execute(ToolCall(name="impact", arguments={"path": "OrderService.php"}), 4000)
    small = await executor.execute(ToolCall(name="impact", arguments={"path": "OrderService.php"}), 64)
    assert estimate_tokens(full.text) > 64
    assert small.text.endswith(TRUNCATED_NOTE) and estimate_tokens(small.text) <= 64
    assert full.text.startswith(small.text[: -len(TRUNCATED_NOTE)]) and small.text.startswith("Analyse d'impact pour")
    assert len(small.summary) <= 200 and small.hits  # les sources sont conservées même si le texte est tronqué
    assert ToolResult(name="x").to_dict()["hits"] == 0


def test_fit_text_never_cuts_a_line_in_the_middle():
    note = TRUNCATED_NOTE
    assert fit_text("", 10) == "" and fit_text("a\nb", 10) == "a\nb"  # vide ou dans le budget : inchangé
    # non-régression : dernier retour à la ligne dans la première moitié de la fenêtre → on y recule quand même
    # (cas réel : en-tête court puis ligne longue d'une note ou d'une liste de valeurs enum)
    head = "2 note(s) pour « commande » :"
    out = fit_text(head + "\n- Note du 2026-01-05 — Réunion : " + "v" * 2000 + "\n- Note du 2026-01-06 — x", 100)
    assert out == head + note and estimate_tokens(out) <= 100
    # aucune ligne entière ne tient (première ligne plus longue que la fenêtre) : mention seule, pas de ligne coupée
    assert fit_text("x" * 900 + "\n" + "y" * 900, 100) == note.strip()
    assert fit_text("x" * 900, 1) == note.strip()  # fenêtre plus courte que la mention
    # retour à la ligne exactement à la limite de la fenêtre : la ligne qui le précède tient, on la garde
    allowed = int(100 * 4.0) - len(note)
    out = fit_text("a" * allowed + "\n" + "b" * 500, 100)
    assert out == "a" * allowed + note and estimate_tokens(out) <= 100
    # plusieurs lignes (code) : uniquement des lignes entières, préfixe du texte, budget respecté
    lines = ["ligne %02d : %s" % (i, "w" * 37) for i in range(40)]
    text = "\n".join(lines)
    out = fit_text(text, 150, is_code=True)
    assert out.endswith(note)
    kept = out[: -len(note)].splitlines()
    assert 1 < len(kept) < 40 and kept == lines[: len(kept)]
    assert estimate_tokens(out, True) <= 150


# =========================================================================== orchestrateur


def test_chat_event_to_sse_format():
    ev = ChatEvent("delta", {"text": "é « ok »"})
    assert ev.to_sse() == 'event: delta\ndata: {"text": "é « ok »"}\n\n'
    assert ChatEvent("done").to_sse() == "event: done\ndata: {}\n\n"
    payload = json.loads(ChatEvent("x", {"p": Path("/tmp")}).to_sse().split("data: ", 1)[1])
    assert payload == {"p": "/tmp"}


def test_find_invalid_citations():
    assert find_invalid_citations("Voir [S1] et [S2][S9], encore [S9] et [S12].", ["S1", "S2"]) == ["S9", "S12"]
    assert find_invalid_citations("rien", []) == []


def make_orchestrator(db: Database, workspace: WorkspaceConfig, llm: LLMClient, engine: Any = None,
                      model_info: Optional[ModelInfo] = None, **kw: Any) -> ChatOrchestrator:
    return ChatOrchestrator(db, workspace, llm, search_engine=engine, impact_analyzer=FakeImpactAnalyzer(),
                            profile=profile(32), model_info=model_info, today="2026-10-17", **kw)


async def test_respond_auto_native_tools_sequence(db: Database, workspace: WorkspaceConfig, seeded):
    engine = FakeSearchEngine([file_hit()], tool_hits=[file_hit(), doc_hit()])
    llm = FakeLLM([
        tool_chunks(ToolCall(name="search", arguments={"query": "statut orders"})),
        text_chunks("D'après [S1], seule une commande brouillon ", "peut être confirmée ; le statut est documenté [S2]. Voir aussi [S9].",
                    usage={"prompt_tokens": 400, "completion_tokens": 30}),
    ])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(32768, tools=True))
    conv = db.create_conversation("test")
    events = await collect(orch, conv, "Quand une commande peut-elle être confirmée ?")
    assert types(events) == ["status", "sources", "tool_call", "tool_result", "delta", "delta", "done"]
    assert events[0].data == {"text": "Recherche dans le projet…"}
    sources = events[1].data["items"]
    assert [s["ref"] for s in sources] == ["S1"] and sources[0]["path"] == "app/Services/OrderService.php"
    assert all("text" not in s for s in sources)  # événements petits : métadonnées seulement
    assert events[2].data["name"] == "search" and events[2].data["arguments"] == {"query": "statut orders"}
    result = events[3].data
    assert result["name"] == "search" and len(result["summary"]) <= 200 and "error" not in result
    assert [s["ref"] for s in result["added_sources"]] == ["S2"] and result["added_sources"][0]["source_type"] == "document"
    assert "".join(e.data["text"] for e in events if e.type == "delta").endswith("[S2]. Voir aussi [S9].")
    done = events[-1].data
    assert done["invalid_citations"] == ["S9"] and done["cancelled"] is False
    assert done["usage"] == {"prompt_tokens": 450, "completion_tokens": 35}
    assert done["budget"]["tool_rounds"] == 1 and done["budget"]["num_ctx"] == 8192
    assert set(done["budget"]) >= {"num_ctx", "context_tokens_used", "max_context_tokens", "tool_rounds"}
    assert done["truncated_sources"] == []
    # recherche : limite 30, embedding tenté, intention lue par analyze_query
    assert engine.calls[0]["limit"] == 30 and engine.embed_calls >= 1
    # appels LLM : outils natifs au premier tour, message tool au second
    assert len(llm.calls) == 2
    first, second = llm.calls
    assert first["tools"] == TOOL_SPECS and first["num_ctx"] == 8192 and first["model"] == "fake:7b"
    assert first["messages"][0].role == "system" and "appeler les outils" in first["messages"][0].content
    assert first["messages"][-1].role == "user" and "[S1]" in first["messages"][-1].content
    roles = [m.role for m in second["messages"]]
    assert roles[-3:] == ["user", "assistant", "tool"]
    assert second["messages"][-2].tool_calls[0].name == "search" and second["messages"][-2].tool_calls[0].id
    assert second["messages"][-1].tool_call_id == second["messages"][-2].tool_calls[0].id
    assert second["messages"][-1].name == "search" and "[S2]" in second["messages"][-1].content
    # messages enregistrés
    conv_data = db.get_conversation(conv)
    msgs = conv_data["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "Quand une commande peut-elle être confirmée ?"
    assert msgs[1]["id"] == done["message_id"] and msgs[1]["content"].startswith("D'après [S1]")
    assert [s["ref"] for s in msgs[1]["sources"]] == ["S1", "S2"]


async def test_respond_text_protocol_without_native_tools(db: Database, workspace: WorkspaceConfig, seeded):
    engine = FakeSearchEngine([file_hit()], tool_hits=[note_hit()])
    llm = FakeLLM([
        text_chunks('{"tool": "notes", ', '"args": {"query": "paid"}}'),
        text_chunks("Réponse : pas de statut paid [S1][S2]."),
    ])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(32768, tools=False))
    conv = db.create_conversation()
    events = await collect(orch, conv, "A-t-on un statut paid ?")
    assert types(events) == ["status", "sources", "tool_call", "tool_result", "delta", "done"]
    assert events[2].data["name"] == "notes" and events[2].data["arguments"] == {"query": "paid"}
    assert events[3].data["added_sources"] and events[3].data["added_sources"][0]["ref"] == "S2"
    assert events[4].data["text"] == "Réponse : pas de statut paid [S1][S2]."
    assert events[-1].data["invalid_citations"] == []
    assert len(llm.calls) == 2 and llm.calls[0]["tools"] is None and llm.calls[1]["tools"] is None
    system = llm.calls[0]["messages"][0].content
    assert '{"tool": "<nom>", "args": {...}}' in system
    second = llm.calls[1]["messages"]
    assert [m.role for m in second[-3:]] == ["user", "assistant", "user"]
    assert second[-1].content.startswith("Résultat de l'outil notes :") and "[S2]" in second[-1].content
    assert db.get_conversation(conv)["messages"][-1]["content"] == "Réponse : pas de statut paid [S1][S2]."


async def test_respond_text_protocol_keeps_useful_text(db: Database, workspace: WorkspaceConfig, seeded):
    # une « vraie » réponse contenant une citation n'est pas prise pour un appel d'outil
    llm = FakeLLM([text_chunks("La confirmation passe par OrderService::confirm [S1].\n")])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=None)
    conv = db.create_conversation()
    events = await collect(orch, conv, "Comment confirmer ?")
    assert types(events) == ["status", "sources", "delta", "done"]
    assert len(llm.calls) == 1


PREFACE = ("La confirmation passe par OrderService::confirm : si la commande n'est pas en statut draft, abort(422) ; "
           "sinon le statut devient confirmed et confirmed_at est renseigné [S1].")
CALLERS_LINE = '{"tool": "callers", "args": {"name": "confirm"}}'


async def test_respond_text_protocol_useful_text_then_tool_line(db: Database, workspace: WorkspaceConfig, seeded):
    # non-régression : réponse utile (citation, ≥ 160 caractères) PUIS ligne d'outil dans le même tour →
    # l'outil est exécuté et la ligne JSON n'atteint ni l'UI ni le message enregistré
    llm = FakeLLM([text_chunks(PREFACE + "\n" + CALLERS_LINE), text_chunks("Elle est appelée par le contrôleur [S2].")])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(32768, tools=False))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Comment se passe la confirmation ?")
    assert types(events) == ["status", "sources", "delta", "tool_call", "tool_result", "delta", "done"]
    assert events[2].data["text"] == PREFACE + "\n"
    assert events[3].data["name"] == "callers" and events[3].data["arguments"] == {"name": "confirm"}
    assert "error" not in events[4].data and events[4].data["added_sources"][0]["ref"] == "S2"
    assert events[5].data["text"] == "Elle est appelée par le contrôleur [S2]."
    assert all('"tool"' not in e.data["text"] for e in events if e.type == "delta")
    assert len(llm.calls) == 2 and events[-1].data["budget"]["tool_rounds"] == 1
    # le modèle revoit son propre tour (texte + ligne d'outil) puis le résultat de l'outil
    second = llm.calls[1]["messages"]
    assert second[-2].role == "assistant" and second[-2].content == PREFACE + "\n" + CALLERS_LINE
    assert second[-1].role == "user" and second[-1].content.startswith("Résultat de l'outil callers :")
    stored = db.get_conversation(conv)["messages"][-1]["content"]
    assert stored == PREFACE + "\nElle est appelée par le contrôleur [S2]."
    assert events[-1].data["invalid_citations"] == []


async def test_respond_text_protocol_streams_text_and_never_leaks_tool_lines(db: Database, workspace: WorkspaceConfig,
                                                                             seeded):
    # flux réel : la préface arrive en morceaux et est émise avant l'appel d'outil ; la ligne d'outil (coupée
    # en deux, dans un bloc ```json) n'est jamais émise ; au tour final forcé (tours épuisés) une ligne d'outil
    # parasite est retirée sans être exécutée et un bloc de code légitime est conservé
    tool_round = text_chunks(PREFACE[:60], PREFACE[60:], "\n```json\n", CALLERS_LINE[:20], CALLERS_LINE[20:], "\n```")
    final = text_chunks("Bilan [S1] :\n```php\n", "$order->status = 'confirmed';\n```\n", CALLERS_LINE)
    llm = FakeLLM([tool_round, tool_round, tool_round, tool_round, final])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(32768, tools=False))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Comment se passe la confirmation ?")
    kinds = types(events)
    assert kinds[:4] == ["status", "sources", "delta", "delta"]  # la préface part dès qu'elle est utile, avant l'outil
    assert kinds.count("tool_call") == 4 and kinds.count("tool_result") == 4 and kinds[-1] == "done"
    assert len(llm.calls) == 5 and llm.calls[4]["messages"][-1].content == TOOLS_EXHAUSTED_TEXT
    deltas = [e.data["text"] for e in events if e.type == "delta"]
    assert all('"tool"' not in d and "```json" not in d for d in deltas)
    answer = "".join(deltas)
    assert answer.count(PREFACE) == 4
    assert answer.endswith("Bilan [S1] :\n```php\n$order->status = 'confirmed';\n```\n")
    assert db.get_conversation(conv)["messages"][-1]["content"] == answer


async def test_respond_text_protocol_short_preface_stays_hidden(db: Database, workspace: WorkspaceConfig, seeded):
    # une courte préface avant l'appel d'outil (« Je vais regarder. ») n'est toujours pas montrée
    llm = FakeLLM([text_chunks("Je vais regarder.\n", CALLERS_LINE), text_chunks("Appelée par le contrôleur [S2].")])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(32768, tools=False))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Qui appelle confirm ?")
    assert types(events) == ["status", "sources", "tool_call", "tool_result", "delta", "done"]
    assert events[4].data["text"] == "Appelée par le contrôleur [S2]."
    assert db.get_conversation(conv)["messages"][-1]["content"] == "Appelée par le contrôleur [S2]."


async def test_respond_search_only_never_calls_llm(db: Database, workspace: WorkspaceConfig, seeded):
    orch = make_orchestrator(db, workspace, NullClient(), FakeSearchEngine([file_hit(), doc_hit()]))
    conv = db.create_conversation()
    events = await collect(orch, conv, "status orders", mode="search_only")
    assert types(events) == ["status", "sources", "done"]
    assert [s["ref"] for s in events[1].data["items"]] == ["S1", "S2"]
    done = events[-1].data
    assert done["message_id"] and done["usage"] == {} and done["budget"]["tool_rounds"] == 0
    msgs = db.get_conversation(conv)["messages"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["content"].startswith("Sources trouvées (2)")
    assert "[S2] Document « Dictionnaire de données » (page 3)" in msgs[-1]["content"]
    assert [s["ref"] for s in msgs[-1]["sources"]] == ["S1", "S2"]
    # sans moteur de recherche du tout : réponse propre
    orch2 = make_orchestrator(db, workspace, NullClient(), None)
    events = await collect(orch2, conv, "rien", mode="search_only")
    assert types(events) == ["status", "sources", "done"] and events[1].data["items"] == []


async def test_respond_no_tools_mode_ignores_tool_calls(db: Database, workspace: WorkspaceConfig, seeded):
    llm = FakeLLM([[ChatChunk(text="Réponse directe [S1].", tool_calls=[ToolCall(name="search", arguments={"query": "x"})]),
                    ChatChunk(done=True)]])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(32768, tools=True))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Question", mode="no_tools")
    assert types(events) == ["status", "sources", "delta", "done"]
    assert llm.calls[0]["tools"] is None and "Aucun outil" in llm.calls[0]["messages"][0].content
    assert events[-1].data["budget"]["tool_rounds"] == 0


async def test_respond_llm_unavailable_after_sources(db: Database, workspace: WorkspaceConfig, seeded):
    orch = make_orchestrator(db, workspace, NullClient(), FakeSearchEngine([file_hit()]))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Question")
    assert types(events) == ["status", "sources", "error"]
    assert events[-1].data["code"] == "llm_unavailable" and "moteur" in events[-1].data["message"].lower()
    assert events[-1].data["message_id"] is None
    assert [m["role"] for m in db.get_conversation(conv)["messages"]] == ["user"]


@pytest.mark.parametrize("exc,code", [
    (ModelNotFound("modèle absent"), "model_not_found"),
    (LLMError("réponse invalide"), "llm_error"),
    (OfflineViolation("api.example.com", "strict", "https://api.example.com/v1"), "offline_blocked"),
    (LLMUnavailable("hors ligne"), "llm_unavailable"),
])
async def test_respond_error_codes_and_partial_message(db: Database, workspace: WorkspaceConfig, seeded, exc, code):
    orch = make_orchestrator(db, workspace, FailingLLM(exc, prefix="Début de réponse "), FakeSearchEngine([file_hit()]),
                             model_info=info(8192, tools=True))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Question")
    assert types(events) == ["status", "sources", "delta", "error"]
    err = events[-1].data
    assert err["code"] == code and err["message"]
    msgs = db.get_conversation(conv)["messages"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["id"] == err["message_id"]
    assert msgs[-1]["content"] == INTERRUPTED_PREFIX + "Début de réponse "


async def test_respond_cancellation_stores_partial_answer(db: Database, workspace: WorkspaceConfig, seeded):
    llm = FakeLLM([text_chunks("Première partie [S1]. ", "Deuxième partie.", "Troisième partie.")], delay=0.01)
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(8192, tools=True))
    conv = db.create_conversation()
    cancel = asyncio.Event()
    events: List[ChatEvent] = []
    async for ev in orch.respond(conv, "Question", cancel=cancel):
        events.append(ev)
        if ev.type == "delta":
            cancel.set()
    events = without_model_phase_statuses(events)
    assert types(events) == ["status", "sources", "delta", "done"]
    done = events[-1].data
    assert done["cancelled"] is True and done["message_id"]
    assert db.get_conversation(conv)["messages"][-1]["content"] == "Première partie [S1]. "
    # annulation avant tout texte : aucun message assistant enregistré
    cancel2 = asyncio.Event()
    cancel2.set()
    events2 = await collect(orch, conv, "Encore", cancel=cancel2)
    assert types(events2) == ["status", "sources", "done"] and events2[-1].data["cancelled"] is True
    assert events2[-1].data["message_id"] is None
    assert db.get_conversation(conv)["messages"][-1]["content"] == "Encore"


async def test_tool_rounds_and_calls_per_round_are_bounded(db: Database, workspace: WorkspaceConfig, seeded):
    many = tool_chunks(*[ToolCall(name="symbol", arguments={"name": "confirm"}) for _ in range(5)])
    llm = FakeLLM([many, many, many, many, text_chunks("Fin [S1].")])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(32768, tools=True))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Qui appelle confirm ?")
    assert types(events)[:2] == ["status", "sources"] and types(events)[-1] == "done"
    tool_calls = [e for e in events if e.type == "tool_call"]
    assert len(tool_calls) == 12  # 4 tours × 3 appels au plus
    assert len([e for e in events if e.type == "tool_result"]) == 12
    assert events[-1].data["budget"]["tool_rounds"] == 4
    assert len(llm.calls) == 5
    assert llm.calls[4]["tools"] is None  # plus d'outils au dernier tour
    assert llm.calls[4]["messages"][-1].role == "user" and llm.calls[4]["messages"][-1].content == TOOLS_EXHAUSTED_TEXT
    assert all(c["tools"] == TOOL_SPECS for c in llm.calls[:4])
    # les identifiants d'appel sont uniques et transmis aux messages « tool »
    ids = [e.data["id"] for e in tool_calls]
    assert len(set(ids)) == 12
    assert db.get_conversation(conv)["messages"][-1]["content"] == "Fin [S1]."


async def test_history_is_bounded_and_summarized(db: Database, workspace: WorkspaceConfig, seeded):
    workspace.llm.num_ctx = 4096  # historique ≤ 768 tokens (10 % de num_ctx, plancher 768) ; chaque message ≈ 250-285 tokens
    limit = history_tokens_for(4096)
    assert limit == 768
    conv = db.create_conversation()
    for i in range(6):
        db.add_message(conv, "user", "Question %d : %s" % (i, "détail " * 141))
        db.add_message(conv, "assistant", "Réponse %d : %s" % (i, "contenu " * 141))
    llm = FakeLLM([text_chunks("ok")])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(8192, tools=True))
    events = await collect(orch, conv, "Nouvelle question")
    assert types(events)[-1] == "done"
    messages = llm.calls[0]["messages"]
    system, history, user = messages[0], messages[1:-1], messages[-1]
    assert user.role == "user" and user.content.startswith("Nouvelle question")
    assert sum(estimate_tokens(m.content) for m in history) <= limit
    # la dernière paire question/réponse entière, jamais une réponse orpheline en tête
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].content.startswith("Question 5") and history[1].content.startswith("Réponse 5")
    assert HISTORY_SUMMARY_PREFIX in system.content and "Q : Question 0" in system.content and "Q : Question 4" in system.content
    assert "Q : Question 5" not in system.content  # la dernière question est dans l'historique, pas dans le résumé
    assert "Réponse" not in system.content.split(HISTORY_SUMMARY_PREFIX, 1)[1]
    # un seul message plus long que le budget : tronqué, pas perdu
    lone = db.create_conversation()
    db.add_message(lone, "user", "Longue question : " + "mot " * 3000)
    llm3 = FakeLLM([text_chunks("ok")])
    orch3 = make_orchestrator(db, workspace, llm3, FakeSearchEngine([file_hit()]), model_info=info(8192, tools=True))
    await collect(orch3, lone, "Suite")
    hist = llm3.calls[0]["messages"][1:-1]
    assert len(hist) == 1 and hist[0].content.endswith(" […]") and estimate_tokens(hist[0].content) <= limit
    # une conversation courte : pas de résumé
    short = db.create_conversation()
    db.add_message(short, "user", "Bonjour")
    db.add_message(short, "assistant", "Salut")
    llm2 = FakeLLM([text_chunks("ok")])
    orch2 = make_orchestrator(db, workspace, llm2, FakeSearchEngine([file_hit()]), model_info=info(8192, tools=True))
    await collect(orch2, short, "Suite")
    msgs = llm2.calls[0]["messages"]
    assert [m.role for m in msgs] == ["system", "user", "assistant", "user"]
    assert HISTORY_SUMMARY_PREFIX not in msgs[0].content


async def test_intent_dates_and_context_override_flow(db: Database, workspace: WorkspaceConfig, seeded):
    analysis = FakeQuery(intent="decision", date_from="2026-09-01", date_to="2026-09-30")
    engine = FakeSearchEngine([note_hit(), file_hit()], intent="decision", analysis=analysis)
    llm = FakeLLM([text_chunks("Décision du 2026-09-01 [S2].")])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(32768, tools=True))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Pourquoi pas de statut paid en septembre ?", max_context_tokens=300)
    assert engine.calls[0]["date_from"] == "2026-09-01" and engine.calls[0]["date_to"] == "2026-09-30"
    done = events[-1].data
    assert done["budget"]["max_context_tokens"] == 300 and done["budget"]["context_tokens_used"] <= 300
    assert done["invalid_citations"] == []
    # un moteur en panne n'empêche pas de répondre (sans contexte)
    broken = FakeSearchEngine([file_hit()])
    broken.search = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("index corrompu"))  # type: ignore[assignment]
    llm2 = FakeLLM([text_chunks("Je n'ai rien trouvé.")])
    orch2 = make_orchestrator(db, workspace, llm2, broken, model_info=info(32768, tools=True))
    events = await collect(orch2, conv, "Question")
    assert types(events) == ["status", "sources", "delta", "done"] and events[1].data["items"] == []
    assert "Aucune source" in llm2.calls[0]["messages"][-1].content


async def test_respond_unknown_conversation_and_empty_answer(db: Database, workspace: WorkspaceConfig, seeded):
    orch = make_orchestrator(db, workspace, FakeLLM([[ChatChunk(done=True)]]), FakeSearchEngine([file_hit()]),
                             model_info=info(8192, tools=True))
    events = await collect(orch, 9999, "Question")
    assert types(events) == ["error"] and events[0].data["code"] == "conversation_not_found"
    conv = db.create_conversation()
    events = await collect(orch, conv, "Question")
    assert types(events) == ["status", "sources", "delta", "done"]
    assert "aucune réponse" in events[2].data["text"].lower()
    assert db.get_conversation(conv)["messages"][-1]["role"] == "assistant"


async def test_sse_stream_is_well_formed_end_to_end(db: Database, workspace: WorkspaceConfig, seeded):
    llm = FakeLLM([text_chunks("Bonjour [S1]")])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(8192, tools=True))
    conv = db.create_conversation()
    raw = "".join([ev.to_sse() async for ev in orch.respond(conv, "Salut")])
    frames = [f for f in raw.split("\n\n") if f]
    assert [f.split("\n")[0] for f in frames] == ["event: status", "event: sources", "event: delta", "event: done"]
    for f in frames:
        head, data = f.split("\n", 1)
        assert data.startswith("data: ")
        json.loads(data[len("data: "):])


# =========================================================================== budget respecté à chaque appel au moteur


def sent_tokens(call: Dict[str, Any]) -> int:
    """Tokens (estimateur du projet) réellement envoyés au moteur : messages + définitions d'outils natifs."""
    n = sum(_message_tokens(m) for m in call["messages"])
    if call["tools"]:
        n += tool_specs_tokens(call["tools"])
    return n


def big_tool_hits(count: int = 8) -> List[Hit]:
    return [file_hit(chunk_id=100 + i, source_id=100 + i, path="app/Big%d.php" % i, line_start=1, line_end=60,
                     text="\n".join("ligne %02d du fichier %d : du code pour remplir la fenêtre ;" % (j, i) for j in range(60)))
            for i in range(count)]


async def test_prompt_plus_reserve_fits_small_windows(db: Database, workspace: WorkspaceConfig, seeded):
    # Régression : à num_ctx=2048 le prompt envoyé dépassait num_ctx (natif) ou num_ctx − réserve (texte) ;
    # à 4096 natif les définitions d'outils (≈ 1000 tokens) faisaient déborder dès le premier appel.
    conv = db.create_conversation()
    for i in range(3):
        db.add_message(conv, "user", "Question précédente %d : %s" % (i, "détail " * 150))
        db.add_message(conv, "assistant", "Réponse précédente %d : %s" % (i, "contenu " * 150))
    for ctx, tools in ((2048, True), (2048, False), (4096, True), (4096, False), (8192, True), (8192, False)):
        workspace.llm.num_ctx = ctx
        llm = FakeLLM([text_chunks("Réponse courte [S1].")])
        orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit(), doc_hit(), note_hit()]),
                                 model_info=info(ctx, tools=tools))
        events = await collect(orch, conv, "Quels champs a la table orders ?")
        assert types(events)[-1] == "done"
        done = events[-1].data["budget"]
        assert done["num_ctx"] == ctx and done["context_tokens_used"] <= done["max_context_tokens"]
        call = llm.calls[-1]
        assert sent_tokens(call) + call["max_tokens"] <= ctx, (ctx, tools, sent_tokens(call), done["source"])
        system = call["messages"][0].content
        if ctx <= 2048 or (ctx <= 4096 and tools):
            # la fenêtre ne porte pas les outils (2k en tout, ou 4k avec ≈ 1000 tokens de définitions natives) :
            # ils sont désactivés, leurs définitions ne partent pas
            assert done["tools_mode"] == "none" and call["tools"] is None and "Aucun outil" in system
            assert "outils désactivés" in done["source"]
        else:
            # 4k en protocole textuel : la réserve de réponse (1k) et l'historique (768) laissent la place d'un résultat
            assert done["tools_mode"] == ("native" if tools else "text")
            assert (call["tools"] == TOOL_SPECS) is tools
        # le résumé de l'historique tient dans le budget d'historique avec les messages gardés. Ce budget est celui
        # que l'orchestrateur calcule sur les consignes SANS le résumé (il mesure les consignes, dimensionne
        # l'historique, puis ajoute le résumé aux consignes) : le résumé est compté dans l'historique, pas deux fois.
        # (Avant la refonte du budget, l'historique cédait moins vite sur une petite fenêtre et l'ancienne mesure,
        # consignes + résumé, restait sous le plafond par hasard.)
        history = call["messages"][1:-1]
        system_only, _, summary = system.partition("\n\n" + HISTORY_SUMMARY_PREFIX)
        summary_tokens = estimate_tokens(HISTORY_SUMMARY_PREFIX + summary) if summary else 0
        measured = estimate_tokens(system_only) + (tool_specs_tokens() if done["tools_mode"] == "native" else 0)
        assert sum(_message_tokens(m) for m in history) + summary_tokens \
            <= compute_budget(workspace.llm, info(ctx, tools=tools), profile(32), tools_mode=done["tools_mode"],
                              system_tokens=measured).history_tokens_max + 1
    workspace.llm.num_ctx = 8192


async def test_native_tool_definitions_count_in_tool_budget(db: Database, workspace: WorkspaceConfig, seeded):
    # Régression : les définitions d'outils natifs (envoyées à chaque tour) n'étaient comptées ni dans le prompt
    # ni dans le budget des outils ; après deux tours le prompt dépassait num_ctx − réserve.
    engine = FakeSearchEngine([file_hit()], tool_hits=big_tool_hits())
    calls = tool_chunks(*[ToolCall(name="search", arguments={"query": "statut %d" % i}) for i in range(3)])
    llm = FakeLLM([calls, calls, calls, calls, text_chunks("Fin [S1].")])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(8192, tools=True))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Qu'est-ce que je casse si je modifie OrderService::confirm ?")
    assert types(events)[-1] == "done"
    done = events[-1].data
    assert done["budget"]["tools_mode"] == "native" and done["budget"]["tool_rounds"] >= 1
    assert llm.calls[0]["tools"] == TOOL_SPECS and llm.calls[-1]["tools"] is None
    for n, call in enumerate(llm.calls):
        assert sent_tokens(call) + call["max_tokens"] <= 8192, (n, sent_tokens(call), done["budget"]["source"])
    # le budget s'épuise proprement : message d'épuisement, outils non exécutés signalés, jamais de débordement
    results = [e.data for e in events if e.type == "tool_result"]
    assert results and any(r.get("error") == "budget épuisé" for r in results) or done["budget"]["tool_rounds"] == 4
    assert llm.calls[-1]["messages"][-1].content == TOOLS_EXHAUSTED_TEXT
    # l'épuisement est atteint avant que les sources ajoutées ne dépassent ce qui restait aux outils
    budget = compute_budget(workspace.llm, info(8192, tools=True), profile(32), "impact", tools_mode="native",
                            system_tokens=estimate_tokens(llm.calls[0]["messages"][0].content) + tool_specs_tokens())
    added = sum(estimate_tokens(m.content) for m in llm.calls[-1]["messages"] if m.role == "tool")
    assert added <= budget.tools_tokens_max + budget.history_tokens_max + budget.context_tokens_max


async def test_tool_text_and_sources_share_one_budget(db: Database, workspace: WorkspaceConfig, seeded):
    # Régression : un appel d'outil pouvait coûter result.text (≤ budget de l'appel) PLUS des sources
    # (≤ context_tokens_max), soit jusqu'à deux fois le budget de l'appel.
    engine = FakeSearchEngine([file_hit()], tool_hits=big_tool_hits())
    llm = FakeLLM([tool_chunks(ToolCall(name="search", arguments={"query": "fichiers volumineux"})), text_chunks("Fin [S2].")])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(8192, tools=True))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Question")
    done = events[-1].data
    assert types(events)[-1] == "done" and done["budget"]["tools_mode"] == "native"
    per_call = done["budget"]["max_context_tokens"]  # à 8192, la part des outils dépasse context_tokens_max
    tool_msg = [m for m in llm.calls[1]["messages"] if m.role == "tool"][0]
    assert "résultat(s) pour « fichiers volumineux »" in tool_msg.content  # le texte de l'outil est là…
    added = last(events, "tool_result").data["added_sources"]
    assert added and "[%s]" % added[0]["ref"] in tool_msg.content  # … et des sources aussi…
    assert estimate_tokens(tool_msg.content) <= per_call + SOURCES_FRAME_TOKENS  # … dans un seul budget
    for call in llm.calls:
        assert sent_tokens(call) + call["max_tokens"] <= 8192
    # mode texte : même partage, enveloppe « Résultat de l'outil … » comprise
    llm2 = FakeLLM([text_chunks('{"tool": "search", "args": {"query": "fichiers volumineux"}}'), text_chunks("Fin [S2].")])
    orch2 = make_orchestrator(db, workspace, llm2, FakeSearchEngine([file_hit()], tool_hits=big_tool_hits()),
                              model_info=info(8192, tools=False))
    conv2 = db.create_conversation()
    events2 = await collect(orch2, conv2, "Question")
    done2 = events2[-1].data
    user_msg = llm2.calls[1]["messages"][-1]
    assert user_msg.role == "user" and user_msg.content.startswith("Résultat de l'outil search :")
    assert estimate_tokens(user_msg.content) <= done2["budget"]["max_context_tokens"] + SOURCES_FRAME_TOKENS + 8
    for call in llm2.calls:
        assert sent_tokens(call) + call["max_tokens"] <= 8192


# =========================================================================== budget affiché (événement done)


async def test_done_budget_counts_initial_context_and_tool_rounds_apart(db: Database, workspace: WorkspaceConfig, seeded):
    # Régression : ``context_tokens_used`` cumulait les sources ajoutées par chaque tour d'outils (jusqu'à
    # context_tokens_max de plus par appel) alors que ``max_context_tokens`` reste le budget de la recherche
    # initiale ; l'UI affichait « Contexte : 1927 / 1290 » dès le premier outil. Le contexte initial et les
    # tours d'outils sont maintenant rapportés à part, chacun sous son propre plafond.
    search = tool_chunks(ToolCall(name="search", arguments={"query": "fichiers volumineux"}))
    llm = FakeLLM([search, text_chunks("Fin [S2].")])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()], tool_hits=big_tool_hits()),
                             model_info=info(8192, tools=True))
    conv = db.create_conversation()
    events = await collect(orch, conv, "Quels champs a la table orders ?")
    assert types(events)[-1] == "done"
    b = events[-1].data["budget"]
    assert last(events, "tool_result").data["added_sources"] and b["tool_rounds"] == 1
    # contexte initial : ce que la recherche a retenu dans son budget, inchangé par l'outil
    assert b["context_tokens_used"] == build_context([file_hit()], b["max_context_tokens"]).tokens_used
    assert 0 < b["context_tokens_used"] <= b["max_context_tokens"]
    # les sources ajoutées par l'outil comptent dans les tours d'outils, pas dans le contexte initial
    assert b["sources_tokens"] > b["context_tokens_used"]
    tool_msg = [m for m in llm.calls[1]["messages"] if m.role == "tool"][0]
    assert estimate_tokens(tool_msg.content) <= b["tools_tokens_used"] <= b["tools_tokens_max"]
    # ce que la fenêtre laissait aux outils : au plus num_ctx − prompt réel du premier appel − réserve de réponse
    assert 0 < b["tools_tokens_max"] <= 8192 - sent_tokens(llm.calls[0]) - b["answer_reserve"]
    # trois tours : le contexte initial ne bouge pas, la consommation des outils se cumule sous son plafond
    # (à 16k : à 8k le contexte récupéré vaut ≈ 3,3k tokens, qu'un seul appel peut rapporter, et trois tours de
    # huit gros passages ne tiennent plus dans ce que la fenêtre laisse aux outils)
    workspace.llm.num_ctx = 16384
    llm3 = FakeLLM([search, search, search, text_chunks("Fin [S2].")])
    orch3 = make_orchestrator(db, workspace, llm3, FakeSearchEngine([file_hit()], tool_hits=big_tool_hits()),
                              model_info=info(16384, tools=True))
    b3 = (await collect(orch3, db.create_conversation(), "Quels champs a la table orders ?"))[-1].data["budget"]
    workspace.llm.num_ctx = 8192
    assert b3["tool_rounds"] == 3 and b3["context_tokens_used"] == b["context_tokens_used"]
    assert b["tools_tokens_used"] < b3["tools_tokens_used"] <= b3["tools_tokens_max"]
    # sans outils (recherche seule, mode sans outils) : rien n'est compté ni réservé pour les outils
    orch_s = make_orchestrator(db, workspace, FakeLLM([text_chunks("Réponse [S1].")]), FakeSearchEngine([file_hit()]),
                               model_info=info(8192, tools=True))
    bs = (await collect(orch_s, db.create_conversation(), "Question", mode="search_only"))[-1].data["budget"]
    assert bs["tools_tokens_used"] == 0 and bs["tools_tokens_max"] == 0 and bs["tool_rounds"] == 0
    assert bs["sources_tokens"] == bs["context_tokens_used"] <= bs["max_context_tokens"]
    bn = (await collect(orch_s, db.create_conversation(), "Question", mode="no_tools"))[-1].data["budget"]
    assert bn["tools_mode"] == "none" and bn["tools_tokens_used"] == 0 and bn["tools_tokens_max"] == 0


def test_chat_view_shows_tool_rounds_apart_from_context():
    # l'interface lit les compteurs séparés : contexte initial / budget, puis tours d'outils / leur plafond
    js = (Path(__file__).resolve().parents[1] / "develop" / "web" / "static" / "views" / "chat.js").read_text(encoding="utf-8")
    assert "context_tokens_used" in js and "max_context_tokens" in js
    assert "tools_tokens_used" in js and "tools_tokens_max" in js and "tool_rounds" in js


# =========================================================================== fiches structurées


REQUEST_RULES_BODY = [
    "    public function rules(): array",
    "    {",
    "        $adherentChecker = app(AdherentExistsCheckerInterface::class);",
    "",
    "        return [",
    "            'adherent_id' => ['required', 'uuid', function ($attr, $value, $fail) use ($adherentChecker) {",
    "                if (!$adherentChecker->exists($value)) $fail('Cet adhérent est inconnu.');",
    "            }],",
    "            'date_emprunt' => 'required|date',",
    "            'agence_id' => 'required|uuid',",
    "            'motif' => [",
    "                'nullable', Rule::in(Emprunt::MOTIFS),",
    "            ],",
    "            'prolongation' => 'nullable|string|max:255',",
    "            'status' => ['sometimes', 'string', 'in:A RENDRE,RENDU'],",
    "            'date_debut' => 'required|date',",
    "            'exemplaire_id' => ['required', 'uuid', fn ($a, $v, $fail) => $fail('L\\'exemplaire est invalide.')],",
    "        ];",
    "    }",
]


def php_lines(total: int, at: Dict[int, Sequence[str]]) -> List[str]:
    """``total`` lignes de PHP factice, avec des blocs placés à des numéros de ligne précis (1-based)."""
    lines = ["// ligne %d" % (i + 1) for i in range(total)]
    for start, block in at.items():
        for k, text in enumerate(block):
            lines[start - 1 + k] = text
    lines[0] = "<?php"
    return lines


def seed_emprunt(db: Database, root: Path, write_request: bool = True) -> Dict[str, Any]:
    """Entité Laravel complète dans une base vide : modèle Eloquent, migrations (same_entity), FormRequest,
    contrôleur, routes (handles), Swagger ; ``write_request`` écrit la classe de requête sur le disque."""
    db.upsert_repo(RepoRow(id="biblio", name="bibliotheque", path=str(root), kind="backend"))
    files = {
        "model": FileRow(repo_id="biblio", path="app/Models/Emprunt.php", language="php", kind="code"),
        "mig1": FileRow(repo_id="biblio", path="database/migrations/2026_02_04_create_emprunts_table.php", language="php", kind="code"),
        "mig2": FileRow(repo_id="biblio", path="database/migrations/2026_03_11_update_emprunt_motif_to_enum.php", language="php", kind="code"),
        "request": FileRow(repo_id="biblio", path="app/Http/Requests/StoreEmpruntRequest.php", language="php", kind="code"),
        "controller": FileRow(repo_id="biblio", path="app/Http/Controllers/Api/V1/EmpruntController.php", language="php", kind="code"),
        "routes": FileRow(repo_id="biblio", path="routes/api.php", language="php", kind="code"),
        "swagger": FileRow(repo_id="biblio", path="Documentation/Swagger/EmpruntSwagger.php", language="php", kind="code"),
    }
    for f in files.values():
        db.upsert_file(f)
    model_lines = php_lines(41, {11: ["class Emprunt extends Model", "{", "    use HasUuids, SoftDeletes;",
                                     "    protected $fillable = ['livre_id', 'status', 'date_debut', 'date_fin', 'motif', 'commentaire'];"]})
    md = db.replace_file_artifacts(files["model"].id, symbols=[
        SymbolRow(name="Emprunt", kind="class", line_start=11, line_end=41, signature="class Emprunt extends Model"),
        SymbolRow(name="livre", kind="method", parent="Emprunt", line_start=36, line_end=39, signature="function livre(): BelongsTo"),
    ], models=[ModelRow(name="Emprunt", table_name="emprunts", kind="eloquent", line=11, fields=[
        {"name": "livre_id", "type": ""}, {"name": "status", "type": ""},
        {"name": "date_debut", "type": "date", "meta": {"cast": "date"}}, {"name": "date_fin", "type": "date"},
        {"name": "motif", "type": ""}, {"name": "commentaire", "type": ""}],
        relations=[{"kind": "belongs_to", "target": "Livre", "field": "livre"}])],
        chunks=[ChunkRow(text="\n".join(model_lines), line_start=1, line_end=41)])
    m1 = db.replace_file_artifacts(files["mig1"].id, models=[
        ModelRow(name="emprunts", table_name="emprunts", kind="migration", line=11, fields=[
            {"name": "id", "type": "uuid", "nullable": False, "meta": {"primary": True}},
            {"name": "livre_id", "type": "uuid", "nullable": False, "meta": {"index": True}},
            {"name": "date_debut", "type": "date", "nullable": False}, {"name": "date_fin", "type": "date", "nullable": True},
            {"name": "motif", "type": "string", "nullable": False}, {"name": "commentaire", "type": "text", "nullable": True},
            {"name": "created_at", "type": "timestamp", "nullable": True}]),
        ModelRow(name="emprunts", table_name="emprunts", kind="migration", line=30, fields=[
            {"name": "status", "type": "enum", "nullable": False, "default": "A RENDRE", "meta": {"enum_values": ["A RENDRE", "RENDU"]}}]),
    ])
    m2 = db.replace_file_artifacts(files["mig2"].id, models=[
        ModelRow(name="emprunts", table_name="emprunts", kind="migration", line=15, fields=[
            {"name": "motif", "type": "enum", "nullable": True, "meta": {"enum_values": ["Retour anticipé", "Perte déclarée"]}}]),
        ModelRow(name="emprunts", table_name="emprunts", kind="migration", line=28,  # down() : ignoré
                 fields=[{"name": "motif", "type": "string", "nullable": False}]),
    ])
    request_lines = php_lines(55, {11: ["class StoreEmpruntRequest extends FormRequest", "{"], 28: REQUEST_RULES_BODY})
    header = "# app/Http/Requests/StoreEmpruntRequest.php L28-55 rules\n"
    rq = db.replace_file_artifacts(files["request"].id, symbols=[
        SymbolRow(name="StoreEmpruntRequest", kind="class", line_start=11, line_end=55,
                  signature="class StoreEmpruntRequest extends FormRequest", meta={"extends": "FormRequest"}),
        SymbolRow(name="authorize", kind="method", parent="StoreEmpruntRequest", line_start=15, line_end=18,
                  signature="function authorize(): bool", meta={"visibility": "public"}),
        SymbolRow(name="prepareForValidation", kind="method", parent="StoreEmpruntRequest", line_start=20, line_end=26,
                  signature="function prepareForValidation(): void", meta={"visibility": "protected"}),
        SymbolRow(name="rules", kind="method", parent="StoreEmpruntRequest", line_start=28, line_end=46,
                  signature="function rules(): array", meta={"visibility": "public"}),
    ], chunks=[ChunkRow(text="\n".join(request_lines[:26]), line_start=1, line_end=26),
               ChunkRow(text=header + "\n".join(request_lines[27:55]), line_start=28, line_end=55, meta={"body_offset": len(header)})])
    if write_request:
        target = root / "app" / "Http" / "Requests" / "StoreEmpruntRequest.php"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(request_lines) + "\n", encoding="utf-8")
    ctl = db.replace_file_artifacts(files["controller"].id, symbols=[
        SymbolRow(name="EmpruntController", kind="class", line_start=16, line_end=82,
                  signature="class EmpruntController extends Controller", meta={"extends": "Controller"}),
        SymbolRow(name="__construct", kind="method", parent="EmpruntController", line_start=18, line_end=21,
                  signature="function __construct(EmpruntService $service)", meta={"visibility": "public"}),
        SymbolRow(name="index", kind="method", parent="EmpruntController", line_start=35, line_end=40,
                  signature="function index(Request $request): JsonResponse", meta={"visibility": "public"}),
        SymbolRow(name="store", kind="method", parent="EmpruntController", line_start=48, line_end=53,
                  signature="function store(StoreEmpruntRequest $request): JsonResponse", meta={"visibility": "public"}),
        SymbolRow(name="update", kind="method", parent="EmpruntController", line_start=55, line_end=60,
                  signature="function update(Emprunt $emprunt, UpdateEmpruntRequest $request): JsonResponse",
                  meta={"visibility": "public"}),
        SymbolRow(name="guard", kind="method", parent="EmpruntController", line_start=70, line_end=75,
                  signature="function guard(): void", meta={"visibility": "private"}),
    ])
    meta = {"prefix": "/v1/bibliotheque", "api_prefix": "/api", "middleware": ["auth:api", "verified"]}
    rt = db.replace_file_artifacts(files["routes"].id, endpoints=[
        EndpointRow(side="server", method="GET", path="/emprunts", normalized_path="/api/v1/bibliotheque/emprunts",
                    handler="EmpruntController@index", framework="laravel", line=81,
                    meta=dict(meta, full_path="/api/v1/bibliotheque/emprunts", name="emprunts.index")),
        EndpointRow(side="server", method="POST", path="/emprunts", normalized_path="/api/v1/bibliotheque/emprunts",
                    handler="EmpruntController@store", framework="laravel", line=81,
                    meta=dict(meta, full_path="/api/v1/bibliotheque/emprunts", name="emprunts.store")),
        EndpointRow(side="server", method="PUT", path="/emprunts/{emprunt}", normalized_path="/api/v1/bibliotheque/emprunts/:param",
                    handler="EmpruntController@update", framework="laravel", line=81,
                    meta=dict(meta, full_path="/api/v1/bibliotheque/emprunts/{emprunt}", name="emprunts.update")),
        EndpointRow(side="server", method="DELETE", path="/emprunts/{emprunt}", normalized_path="/api/v1/bibliotheque/emprunts/:param",
                    handler="EmpruntController@destroy", framework="laravel", line=81,
                    meta=dict(meta, full_path="/api/v1/bibliotheque/emprunts/{emprunt}", name="emprunts.destroy")),
        EndpointRow(side="server", method="POST", path="emprunts/{emprunt}/prolonger", normalized_path="/api/v1/bibliotheque/emprunts/:param/prolonger",
                    handler="EmpruntController@prolongerEmprunt", framework="laravel", line=80,
                    meta=dict(meta, full_path="/api/v1/bibliotheque/emprunts/{emprunt}/prolonger", name="bibliotheque.emprunt.prolonger")),
        EndpointRow(side="client", method="POST", path="/api/v1/bibliotheque/emprunts", normalized_path="/api/v1/bibliotheque/emprunts",
                    handler="createEmprunt", framework="axios", line=12),
    ])
    sw = db.replace_file_artifacts(files["swagger"].id, symbols=[SymbolRow(name="EmpruntSwagger", kind="class", line_start=22, line_end=300)])
    eloquent, migrations = md["models"][0], m1["models"] + m2["models"]
    db.add_relations([RelationRow(src=node_key("model", eloquent), dst=node_key("model", mid), kind="same_entity",
                                  meta={"rule": "same_table"}) for mid in migrations])
    db.add_relations([
        RelationRow(src=node_key("file", files["controller"].id), dst=node_key("model", eloquent), kind="uses_model", meta={"rule": "import"}),
        RelationRow(src=node_key("endpoint", rt["endpoints"][1]), dst=node_key("symbol", ctl["symbols"][3]), kind="handles"),
        RelationRow(src=node_key("endpoint", rt["endpoints"][0]), dst=node_key("symbol", ctl["symbols"][2]), kind="handles"),
    ])
    return {"files": files, "model_id": eloquent, "migration_ids": migrations, "model_class": md["symbols"][0],
            "request_class": rq["symbols"][0], "controller_class": ctl["symbols"][0], "store_method": ctl["symbols"][3],
            "post_endpoint": rt["endpoints"][1], "get_endpoint": rt["endpoints"][0], "client_endpoint": rt["endpoints"][5],
            "swagger_class": sw["symbols"][0], "model_lines": model_lines}


def emprunt_hits(g: Dict[str, Any]) -> Dict[str, Hit]:
    """Hits au format du moteur pour l'entité ``seed_emprunt`` (modèle, migration, classe, fichier, méthode, route)."""
    files = g["files"]
    return {
        "model": Hit(source_type="model", source_id=g["model_id"], title="Emprunt (table emprunts)", path=files["model"].path,
                     repo_id="biblio", file_id=files["model"].id, line_start=11, line_end=41, score=0.9, channels=["identifier"],
                     reason="mots collés « Emprunt »", text="\n".join(g["model_lines"][10:41]),
                     extra={"model_id": g["model_id"], "kind": "eloquent", "table_name": "emprunts"}),
        "migration": Hit(source_type="model", source_id=g["migration_ids"][0], title="emprunts", path=files["mig1"].path,
                         repo_id="biblio", file_id=files["mig1"].id, line_start=11, line_end=28, score=0.5, channels=["graph"],
                         reason="relié à Emprunt via same_entity", text="Schema::create('emprunts', function (Blueprint $table) {",
                         extra={"model_id": g["migration_ids"][0], "kind": "migration", "table_name": "emprunts"}),
        "class": Hit(source_type="symbol", source_id=g["model_class"], title="Emprunt", path=files["model"].path, repo_id="biblio",
                     file_id=files["model"].id, line_start=11, line_end=41, score=0.6, channels=["identifier"],
                     reason="mots collés « Emprunt »", text="\n".join(g["model_lines"][10:41]),
                     extra={"symbol_id": g["model_class"], "kind": "class", "signature": "class Emprunt extends Model"}),
        "model_file": Hit(chunk_id=1, source_type="file", source_id=files["model"].id, title="Emprunt.php", path=files["model"].path,
                          repo_id="biblio", file_id=files["model"].id, line_start=1, line_end=41, score=0.4, channels=["lexical"],
                          reason="correspondance lexicale sur « emprunt »", text="\n".join(g["model_lines"]), extra={"language": "php"}),
        "store": Hit(source_type="symbol", source_id=g["store_method"], title="EmpruntController::store", path=files["controller"].path,
                     repo_id="biblio", file_id=files["controller"].id, line_start=48, line_end=53, score=0.3, channels=["graph"],
                     reason="relié à Emprunt via utilisation du modèle",
                     text="    public function store(StoreEmpruntRequest $request): JsonResponse\n    {\n        return $this->service->create($request->validated());\n    }",
                     extra={"symbol_id": g["store_method"], "kind": "method", "parent": "EmpruntController"}),
        "post": Hit(source_type="endpoint", source_id=g["post_endpoint"], title="POST /emprunts", path=files["routes"].path, repo_id="biblio",
                    file_id=files["routes"].id, line_start=81, line_end=83, score=0.7, channels=["identifier"], reason="route POST /emprunts",
                    text="    Route::apiResource('emprunts', EmpruntController::class);",
                    extra={"endpoint_id": g["post_endpoint"], "side": "server", "method": "POST", "route": "/emprunts",
                           "handler": "EmpruntController@store"}),
        "controller": Hit(source_type="symbol", source_id=g["controller_class"], title="EmpruntController", path=files["controller"].path,
                          repo_id="biblio", file_id=files["controller"].id, line_start=16, line_end=82, score=0.65, channels=["identifier"],
                          reason="nom de symbole exact", text="class EmpruntController extends Controller\n{\n}",
                          extra={"symbol_id": g["controller_class"], "kind": "class"}),
        "client": Hit(source_type="endpoint", source_id=g["client_endpoint"], title="POST /api/v1/bibliotheque/emprunts", path="src/api.ts",
                      repo_id="biblio", file_id=None, line_start=12, line_end=14, score=0.2, channels=["identifier"], reason="appel client",
                      text="axios.post('/api/v1/bibliotheque/emprunts', data)", extra={"endpoint_id": g["client_endpoint"], "side": "client"}),
    }


def test_build_cards_model_card_merges_migrations_routes_and_validation(db: Database, tmp_path: Path):
    g = seed_emprunt(db, tmp_path)
    h = emprunt_hits(g)
    # modèle + migration + classe du modèle + fichier du modèle : une seule fiche, celle de l'entité
    cards = build_cards(db, [h["model"], h["migration"], h["class"], h["model_file"]])
    assert len(cards) == 1 and cards[0].kind == "model" and cards[0].title == "Modèle Emprunt"
    text = cards[0].text
    assert text.startswith("Modèle Emprunt (table emprunts, bibliotheque/app/Models/Emprunt.php:11)\n· champs : ")
    # types / nullable / défaut / énumération viennent des migrations (up() de la plus récente, jamais son down())
    assert "livre_id (uuid, requis)" in text and "date_fin (date, nullable)" in text and "commentaire (text, nullable)" in text
    assert "status (enum A RENDRE|RENDU, requis, défaut A RENDRE)" in text
    assert "motif (enum Retour anticipé|Perte déclarée, nullable)" in text and "motif (string" not in text
    assert "· autres colonnes (migrations) : id, created_at" in text
    assert "· relations : belongs_to Livre (livre)" in text
    assert "· migrations : 2026_02_04_create_emprunts_table.php, 2026_03_11_update_emprunt_motif_to_enum.php" in text
    # routes : préfixe commun factorisé, contrôleur répété seulement quand il change, ordre GET/POST/PUT/DELETE
    assert ("· routes : (préfixe /api/v1/bibliotheque) GET /emprunts → EmpruntController@index, POST /emprunts → @store, "
            "PUT /emprunts/{emprunt} → @update, DELETE /emprunts/{emprunt} → @destroy, POST /emprunts/{emprunt}/prolonger → @prolongerEmprunt") in text
    # validation : rules() lu sur le disque, compacté, closures signalées
    assert ("· validation (création) : StoreEmpruntRequest (bibliotheque/app/Http/Requests/StoreEmpruntRequest.php:28) rules() : "
            "adherent_id : required|uuid|(closure) ; date_emprunt : required|date ; agence_id : required|uuid ; "
            "motif : nullable|Rule::in(Emprunt::MOTIFS) ; prolongation : nullable|string|max:255 ; "
            "status : sometimes|string|in:A RENDRE,RENDU ; date_debut : required|date ; "
            "exemplaire_id : required|uuid|(closure)") in text  # tableau multi-lignes recollé, message d'erreur de closure ignoré
    assert "UpdateEmpruntRequest" not in text  # classe absente de l'index : pas inventée
    assert "· documentation Swagger : bibliotheque/Documentation/Swagger/EmpruntSwagger.php" in text
    assert cards[0].tokens == estimate_tokens(text) <= 350
    # ce que la fiche couvre : le modèle, ses migrations, sa classe, ses routes affichées et les morceaux bruts de leurs fichiers
    routes = {("endpoint", e.id) for e in db.list_endpoints(side="server") if (e.handler or "").startswith("EmpruntController@")}
    assert cards[0].covers == {("model", g["model_id"])} | {("model", m) for m in g["migration_ids"]} | {("symbol", g["model_class"])} | routes
    assert cards[0].file_ids == {g["files"]["model"].id, g["files"]["mig1"].id, g["files"]["mig2"].id}
    # la même fiche part d'un hit de migration (l'entité est retrouvée via same_entity) ou de la classe du modèle
    assert build_cards(db, [h["migration"]])[0].text == text and build_cards(db, [h["class"]])[0].text == text
    # jamais plus de max_cards, jamais de fiche pour un fichier ou un document
    assert build_cards(db, [h["model_file"], doc_hit(), note_hit()]) == [] and build_cards(db, [h["model"]], max_cards=0) == []


def test_build_context_puts_cards_first_and_dedupes_covered_passages(db: Database, tmp_path: Path):
    g = seed_emprunt(db, tmp_path)
    h = emprunt_hits(g)
    hits = [h["model"], h["migration"], h["class"], h["model_file"], h["store"], doc_hit(score=0.25)]
    pack = build_context(hits, 1200, "howto", db=db)
    # la fiche est S1, en tête, section « Fiches » puis « Passages » ; les passages couverts ne sont pas répétés
    assert [s.ref for s in pack.sources] == ["S1", "S2", "S3"] and len(pack.cards) == 1 and pack.cards[0].ref == "S1"
    assert pack.text.startswith("Fiches (synthèse exacte de l'index : champs, règles de validation, routes) :\n\n[S1] Modèle Emprunt (table emprunts")
    assert "\n\nPassages :\n\n[S2] " in pack.text and pack.text.count("[S1]") == 1
    s1 = pack.sources[0]
    assert s1.source_type == "model" and s1.model_id == g["model_id"] and s1.path == "app/Models/Emprunt.php" and s1.line_start == 11
    assert s1.reason == "fiche structurée (synthèse de l'index)" and s1.channels == ["card"] and s1.file_id == g["files"]["model"].id
    assert [s.source_type for s in pack.sources[1:]] == ["symbol", "document"] and pack.sources[1].symbol_id == g["store_method"]
    assert pack.hits_by_ref["S1"].extra["card"] == "model" and pack.blocks["S1"] == "[S1] " + pack.cards[0].text
    assert "StoreEmpruntRequest" in pack.text and "livre_id (uuid, requis)" in pack.text and "POST /emprunts → @store" in pack.text
    assert pack.tokens_used <= 1200 and pack.truncated == []
    assert pack.sources == [Source.from_hit("S1", pack.hits_by_ref["S1"])] + pack.sources[1:]  # sérialisable comme les autres
    # recherche seule : la fiche est listée comme une source ordinaire
    assert "- [S1] app/Models/Emprunt.php:L11 — fiche structurée (synthèse de l'index)" in render_sources_as_text(pack)
    # sans base : comportement d'origine, aucune fiche ; fiches fournies : utilisées telles quelles
    plain = build_context(hits, 1200, "howto")
    assert plain.cards == [] and not plain.text.startswith("Fiches") and plain.sources[0].source_type == "model"
    given = build_context(hits, 1200, "howto", cards=pack.cards)
    assert given.text == pack.text
    # budget trop petit pour la fiche entière : elle est écartée en bloc, les passages reprennent la place
    small = build_context(hits, 150, "howto", db=db)
    assert small.cards == [] and small.sources and small.tokens_used <= 150 and not small.text.startswith("Fiches")
    # plusieurs fiches : elles ne prennent jamais plus des deux tiers du budget (la première passe toujours)
    big_card, small_card = pack.cards[0], build_cards(db, [h["controller"]])[0]
    many = build_context(hits, 600, "howto", cards=[big_card, small_card, small_card])
    assert [c.title for c in many.cards] == ["Modèle Emprunt"] and many.sources[1].source_type != "model"
    assert [c.title for c in build_context(hits, 1500, "howto", cards=[big_card, small_card]).cards] == ["Modèle Emprunt", "Classe EmpruntController"]
    # les outils ajoutent après les fiches, sans doublon de ce que la fiche couvre (modèle, route affichée)
    added = add_tool_sources(pack, [h["model"], h["post"], h["controller"]], 500)
    assert [s.ref for s in added] == ["S4"] and added[0].source_type == "symbol" and added[0].symbol_id == g["controller_class"]


def test_build_context_keeps_one_passage_per_identical_span():
    # plusieurs routes déclarées sur la même ligne (apiResource) donnent des hits distincts au texte identique
    hits = [file_hit(source_type="endpoint", source_id=i, chunk_id=None, title="GET /orders", path="routes/web.php", line_start=7,
                     line_end=9, text="Route::resource('orders', OrderController::class);", extra={"endpoint_id": i})
            for i in range(1, 5)]
    hits.append(file_hit(source_type="endpoint", source_id=9, chunk_id=None, path="routes/web.php", line_start=7, line_end=9,
                         text="Route::get('/orders/export', [OrderController::class, 'export']);", extra={"endpoint_id": 9}))
    pack = build_context(hits, 800)
    assert [s.endpoint_id for s in pack.sources] == [1, 9] and pack.truncated == []


def test_build_cards_endpoint_and_class_cards(db: Database, tmp_path: Path, monkeypatch):
    g = seed_emprunt(db, tmp_path)
    h = emprunt_hits(g)
    cards = build_cards(db, [h["post"], h["controller"], h["client"]])
    assert [c.kind for c in cards] == ["endpoint", "symbol"]  # un appel client n'a pas de fiche
    ep = cards[0].text
    assert ep.startswith("Route POST /emprunts (bibliotheque/routes/api.php:81) → EmpruntController@store · chemin complet : /api/v1/bibliotheque/emprunts"
                         " · nom : emprunts.store · middleware : auth:api, verified")
    assert "· contrôleur : store(StoreEmpruntRequest $request): JsonResponse (bibliotheque/app/Http/Controllers/Api/V1/EmpruntController.php:48-53)" in ep
    assert "· classe de requête : StoreEmpruntRequest (bibliotheque/app/Http/Requests/StoreEmpruntRequest.php:28) rules() : adherent_id : required|uuid|(closure)" in ep
    assert "· modèle principal : Emprunt (table emprunts, bibliotheque/app/Models/Emprunt.php:11) — champs : livre_id (uuid, requis), status (enum" in ep
    assert "· documentation Swagger : bibliotheque/Documentation/Swagger/EmpruntSwagger.php" in ep
    assert cards[0].covers == {("endpoint", g["post_endpoint"])} and cards[0].source is None and cards[0].hit.extra["endpoint_id"] == g["post_endpoint"]
    cls = cards[1].text
    assert cls.startswith("Classe EmpruntController (bibliotheque/app/Http/Controllers/Api/V1/EmpruntController.php:16-82) — class EmpruntController extends Controller")
    assert "· parent : Controller" in cls
    assert ("· méthodes publiques (4) : __construct(EmpruntService $service) L18 ; index(Request $request): JsonResponse L35 ; "
            "store(StoreEmpruntRequest $request): JsonResponse L48 ; update(Emprunt $emprunt, UpdateEmpruntRequest $request): JsonResponse L55") in cls
    assert "guard" not in cls  # méthode privée
    assert all(estimate_tokens(c.text) <= 350 for c in cards)
    # une classe de requête vue comme symbole : ses règles sont dans la fiche
    rq_hit = Hit(source_type="symbol", source_id=g["request_class"], title="StoreEmpruntRequest", path=g["files"]["request"].path,
                 repo_id="biblio", file_id=g["files"]["request"].id, line_start=11, line_end=55, score=0.5, channels=["identifier"],
                 text="class StoreEmpruntRequest extends FormRequest", extra={"symbol_id": g["request_class"], "kind": "class"})
    rq_card = build_cards(db, [rq_hit])[0]
    assert rq_card.text.startswith("Classe StoreEmpruntRequest (bibliotheque/app/Http/Requests/StoreEmpruntRequest.php:11-55)")
    assert "· règles de validation (rules(), bibliotheque/app/Http/Requests/StoreEmpruntRequest.php:28) : adherent_id : required|uuid|(closure) ;" in rq_card.text
    assert "prepareForValidation" not in rq_card.text and "authorize(): bool L15" in rq_card.text
    # une route listée sur la fiche du modèle n'a pas de fiche à elle, quel que soit l'ordre des hits ;
    # une seule fiche de route par contrôleur
    assert [c.kind for c in build_cards(db, [h["model"], h["post"], h["post"]])] == ["model"]
    assert [c.kind for c in build_cards(db, [h["post"], h["class"], h["model"]])] == ["model"]
    get_hit = Hit(source_type="endpoint", source_id=g["get_endpoint"], title="GET /emprunts", path="routes/api.php",
                  file_id=g["files"]["routes"].id, text="Route::get", extra={"endpoint_id": g["get_endpoint"]})
    assert [c.title for c in build_cards(db, [h["post"], get_hit])] == ["Route POST /emprunts"]
    # route hors des routes affichées de la fiche du modèle : fiche de route sans répéter champs ni règles,
    # et la fiche du modèle passe devant même si la route était le premier hit
    monkeypatch.setattr("develop.chat.context_builder.CARD_MAX_ROUTES", 1)
    both = build_cards(db, [h["post"], h["model"]])
    assert [c.kind for c in both] == ["model", "endpoint"] and "GET /emprunts → EmpruntController@index, … (+4)" in both[0].text
    assert "· modèle principal : Emprunt (champs : voir sa fiche)" in both[1].text
    assert "· classe de requête : StoreEmpruntRequest (règles : voir la fiche du modèle)" in both[1].text
    assert "adherent_id" not in both[1].text and "livre_id" not in both[1].text
    monkeypatch.setattr("develop.chat.context_builder.CARD_MAX_ROUTES", 8)
    # route sans relation handles ni classe de requête dans la signature : convention Store<Modèle>Request, méthode retrouvée par le handler
    db.delete_relations_touching([node_key("endpoint", g["get_endpoint"])])
    get_card = build_cards(db, [Hit(source_type="endpoint", source_id=g["get_endpoint"], title="GET /emprunts", path="routes/api.php",
                                    file_id=g["files"]["routes"].id, text="Route::get", extra={"endpoint_id": g["get_endpoint"]})])[0]
    assert "· contrôleur : index(Request $request): JsonResponse (bibliotheque/app/Http/Controllers/Api/V1/EmpruntController.php:35-40)" in get_card.text
    assert "classe de requête" not in get_card.text and "· modèle principal : Emprunt (table emprunts" in get_card.text


def test_card_paths_carry_the_repo_name_like_passage_headers(db: Database, tmp_path: Path):
    """Les fiches citent « dépôt/chemin », le libellé de l'en-tête des passages (:func:`format_header`) : sur un projet
    découpé en modules (un dépôt indexé par module), « app/Models/Emprunt.php » n'existe pas à la racine du projet et
    deux modules ont chacun leur ``routes/api.php`` ; le préfixe rend chaque citation résoluble et non ambiguë."""
    root = tmp_path / "Modules" / "Bibliotheque"
    g = seed_emprunt(db, root)
    db.upsert_repo(RepoRow(id="biblio", name="Modules/Bibliotheque", path=str(root), kind="backend"))
    db.upsert_repo(RepoRow(id="sec", name="Modules/Securite", path=str(tmp_path / "Modules" / "Securite"), kind="backend"))
    user = FileRow(repo_id="sec", path="app/Models/User.php", language="php", kind="code")
    routes = FileRow(repo_id="sec", path="routes/api.php", language="php", kind="code")  # même chemin relatif que dans Bibliotheque
    for f in (user, routes):
        db.upsert_file(f)
    um = db.replace_file_artifacts(user.id, models=[ModelRow(name="User", table_name="users", kind="eloquent", line=42,
                                                           fields=[{"name": "email", "type": "string"}])])["models"][0]
    login = db.replace_file_artifacts(routes.id, endpoints=[EndpointRow(
        side="server", method="POST", path="/login", normalized_path="/api/login", handler="AuthController@login",
        framework="laravel", line=11)])["endpoints"][0]
    h = emprunt_hits(g)
    model_card = build_cards(db, [h["model"]])[0].text
    assert model_card.startswith("Modèle Emprunt (table emprunts, Modules/Bibliotheque/app/Models/Emprunt.php:11)\n")
    assert "· validation (création) : StoreEmpruntRequest (Modules/Bibliotheque/app/Http/Requests/StoreEmpruntRequest.php:28) rules()" in model_card
    assert "· documentation Swagger : Modules/Bibliotheque/Documentation/Swagger/EmpruntSwagger.php" in model_card
    route_card = build_cards(db, [h["post"]])[0].text
    assert route_card.startswith("Route POST /emprunts (Modules/Bibliotheque/routes/api.php:81) → EmpruntController@store")
    assert "(Modules/Bibliotheque/app/Http/Controllers/Api/V1/EmpruntController.php:48-53)" in route_card
    assert "· modèle principal : Emprunt (table emprunts, Modules/Bibliotheque/app/Models/Emprunt.php:11) — champs" in route_card
    assert build_cards(db, [h["controller"]])[0].text.startswith(
        "Classe EmpruntController (Modules/Bibliotheque/app/Http/Controllers/Api/V1/EmpruntController.php:16-82)")
    # l'autre module : son routes/api.php et son modèle se distinguent par leur préfixe
    user_hit = Hit(source_type="model", source_id=um, title="User", path=user.path, repo_id="sec", file_id=user.id, line_start=42,
                   text="class User", extra={"model_id": um, "kind": "eloquent"})
    login_hit = Hit(source_type="endpoint", source_id=login, title="POST /login", path=routes.path, repo_id="sec", file_id=routes.id,
                    line_start=11, text="Route::post", extra={"endpoint_id": login, "handler": "AuthController@login"})
    cards = build_cards(db, [user_hit, login_hit])
    assert cards[0].text.startswith("Modèle User (table users, Modules/Securite/app/Models/User.php:42)\n· champs : email")
    assert cards[1].text.startswith("Route POST /login (Modules/Securite/routes/api.php:11) → AuthController@login")
    # même libellé que l'en-tête d'un passage du même fichier ; le hit d'une fiche porte repo_name comme ceux du moteur,
    # sa source garde le chemin relatif au dépôt
    passage = Hit(**dict(h["model_file"].__dict__, extra={"language": "php", "repo_name": "Modules/Bibliotheque"}))
    assert format_header("S3", passage).startswith("[S3] Modules/Bibliotheque/app/Models/Emprunt.php:L1-41")
    assert _repo_prefix(cards[0].hit) == "Modules/Securite" and cards[0].hit.path == "app/Models/User.php"
    loader = _CardLoader(db)
    assert loader.where(user.id) == "Modules/Securite/app/Models/User.php" and loader.path(user.id) == "app/Models/User.php"
    assert loader.where(None) == "" and loader.where(10_000) == ""


def test_cards_read_rules_from_index_when_disk_is_unavailable_and_stay_under_350_tokens(db: Database, tmp_path: Path):
    g = seed_emprunt(db, tmp_path, write_request=False)  # fichier absent du disque : repli sur les morceaux indexés
    h = emprunt_hits(g)
    card = build_cards(db, [h["model"]])[0]
    assert "rules() : adherent_id : required|uuid|(closure) ; date_emprunt : required|date" in card.text
    # fichier hors du dépôt : jamais lu, règles « non lues » signalées plutôt qu'inventées
    evil = FileRow(repo_id="biblio", path="../../../etc/EvilRequest.php", language="php", kind="code")
    db.upsert_file(evil)
    ev = db.replace_file_artifacts(evil.id, symbols=[
        SymbolRow(name="StoreEvilRequest", kind="class", line_start=1, line_end=20),
        SymbolRow(name="rules", kind="method", parent="StoreEvilRequest", line_start=5, line_end=15, meta={"visibility": "public"})])
    evil_hit = Hit(source_type="symbol", source_id=ev["symbols"][0], title="StoreEvilRequest", path=evil.path, file_id=evil.id,
                   line_start=1, line_end=20, text="class StoreEvilRequest", extra={"symbol_id": ev["symbols"][0], "kind": "class"})
    assert "rules(), bibliotheque/../../../etc/EvilRequest.php:5) : non lues (ouvrir le fichier)" in build_cards(db, [evil_hit])[0].text
    # modèle énorme (80 champs, 30 relations, 40 routes) : la fiche reste ≤ 350 tokens, les champs d'abord
    big = FileRow(repo_id="biblio", path="app/Models/Claim.php", language="php", kind="code")
    db.upsert_file(big)
    bm = db.replace_file_artifacts(big.id, models=[ModelRow(name="Claim", table_name="claims", kind="eloquent", line=9,
                                                            fields=[{"name": "champ_numero_%02d" % i, "type": "string", "nullable": False} for i in range(80)],
                                                            relations=[{"kind": "belongs_to", "target": "Cible%d" % i, "field": "cible%d" % i} for i in range(30)])])
    routes = FileRow(repo_id="biblio", path="routes/claims.php", language="php", kind="code")
    db.upsert_file(routes)
    db.replace_file_artifacts(routes.id, endpoints=[EndpointRow(side="server", method="GET", path="/claims/variante-%02d/{id}" % i,
                                                               normalized_path="/claims/variante-%02d/:param" % i,
                                                               handler="ClaimController@variante%d" % i, framework="laravel", line=i + 1)
                                                   for i in range(40)])
    big_hit = Hit(source_type="model", source_id=bm["models"][0], title="Claim", path=big.path, file_id=big.id, line_start=9,
                  text="class Claim", extra={"model_id": bm["models"][0], "kind": "eloquent"})
    big_card = build_cards(db, [big_hit])[0]
    assert estimate_tokens(big_card.text) <= 350 and big_card.tokens <= 350
    assert big_card.text.startswith("Modèle Claim (table claims, bibliotheque/app/Models/Claim.php:9)\n· champs : champ_numero_00 (string, requis)")
    fields_line = big_card.text.splitlines()[1]
    assert fields_line.endswith("…") and len(fields_line) <= 702 and "champ_numero_79" not in fields_line  # ligne bornée, jamais coupée en mots
    assert "· routes : " in big_card.text and big_card.text.endswith("… (+32)")  # 8 routes montrées, le reste compté
    pack = build_context([big_hit], 400, "fields", db=db)
    assert pack.cards and pack.tokens_used <= 400


def test_model_card_shortens_routes_before_dropping_validation_rules(db: Database, tmp_path: Path):
    # Régression : une fiche de modèle qui débordait de CARD_MAX_TOKENS retirait d'un bloc la section « validation
    # (création) » (priorité 3) en gardant ses 8 routes (priorité 2) ; sur un index réel, 7 des 39 modèles ayant un
    # Store<Modèle>Request::rules() n'affichaient aucune règle — précisément la
    # « définition des champs » demandée. Désormais une section est raccourcie (variante suivante) avant d'être
    # retirée, les règles de création passent avant les routes, et la place restante revient par priorité.
    header = "Modèle X (app/Models/X.php:1)"
    champs = "champs : " + "a" * 400
    routes = ["routes : " + "r" * 400, "routes : " + "r" * 200 + ", … (+4)", "routes : " + "r" * 100 + ", … (+6)"]
    rules = ["validation (création) : " + "v" * 400, "validation (création) : " + "v" * 200 + " ; … (+5)"]
    policy = "policy : XPolicy (app/Policies/XPolicy.php:9)"
    sections = [(1, champs), (3, routes), (2, rules), (8, policy)]
    assert _fit_sections(header, sections, 10_000) == "\n".join([header, "· " + champs, "· " + routes[0], "· " + rules[0], "· " + policy])
    # 300 tokens : les routes se raccourcissent (les règles, plus prioritaires, restent entières) et la policy,
    # retirée en premier, revient dans la place ainsi libérée
    at300 = _fit_sections(header, sections, 300)
    assert at300 == "\n".join([header, "· " + champs, "· " + routes[1], "· " + rules[0], "· " + policy]) and estimate_tokens(at300) <= 300
    # 200 tokens : les règles se raccourcissent plutôt que de disparaître, et deux routes reviennent après elles
    at200 = _fit_sections(header, sections, 200)
    assert at200 == "\n".join([header, "· " + champs, "· " + routes[2], "· " + rules[1]]) and estimate_tokens(at200) <= 200
    assert _fit_sections(header, [(1, ["même", "même", ""]), (2, [])], 10_000) == header + "\n· même"  # variantes inutiles ignorées
    # entité Laravel réelle : 20 champs typés, 12 routes, 14 règles de création — la fiche entière dépasse 350 tokens
    db.upsert_repo(RepoRow(id="biblio", name="bibliotheque", path=str(tmp_path), kind="backend"))
    model = FileRow(repo_id="biblio", path="app/Models/Dossier.php", language="php", kind="code")
    routes_file = FileRow(repo_id="biblio", path="routes/dossiers.php", language="php", kind="code")
    request = FileRow(repo_id="biblio", path="app/Http/Requests/StoreDossierRequest.php", language="php", kind="code")
    for f in (model, routes_file, request):
        db.upsert_file(f)
    dm = db.replace_file_artifacts(model.id, models=[ModelRow(
        name="Dossier", table_name="dossiers", kind="eloquent", line=9,
        fields=[{"name": "champ_%02d" % i, "type": "string", "nullable": False} for i in range(20)])])
    rt = db.replace_file_artifacts(routes_file.id, endpoints=[
        EndpointRow(side="server", method="GET", path="/dossiers/variante-%02d/{id}" % i, normalized_path="/dossiers/variante-%02d/:param" % i,
                    handler="DossierController@variante%d" % i, framework="laravel", line=i + 1) for i in range(12)])
    body = (["    public function rules(): array", "    {", "        return ["]
            + ["            'champ_%02d' => 'required|string|max:255'," % i for i in range(14)] + ["        ];", "    }"])
    db.replace_file_artifacts(request.id, symbols=[
        SymbolRow(name="StoreDossierRequest", kind="class", line_start=3, line_end=30),
        SymbolRow(name="rules", kind="method", parent="StoreDossierRequest", line_start=6, line_end=5 + len(body), meta={"visibility": "public"})])
    target = tmp_path / "app" / "Http" / "Requests" / "StoreDossierRequest.php"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(php_lines(30, {3: ["class StoreDossierRequest extends FormRequest", "{"], 6: body})) + "\n", encoding="utf-8")
    # les variantes des règles gardent des entrées entières et comptent le reste, jamais de coupe au milieu d'une règle
    variants = _request_section(_CardLoader(db), "StoreDossierRequest", "validation (création)")
    prefix = "validation (création) : StoreDossierRequest (bibliotheque/app/Http/Requests/StoreDossierRequest.php:6) rules() : champ_00 : required|string|max:255 ; "
    assert [v.startswith(prefix) for v in variants] == [True, True, True]
    assert variants[0].endswith("champ_13 : required|string|max:255") and variants[1].endswith("champ_07 : required|string|max:255 ; … (+6)")
    assert variants[2].endswith("champ_04 : required|string|max:255 ; … (+9)")
    hit = Hit(source_type="model", source_id=dm["models"][0], title="Dossier", path=model.path, file_id=model.id, line_start=9,
              text="class Dossier", extra={"model_id": dm["models"][0], "kind": "eloquent"})
    card = build_cards(db, [hit])[0]
    assert card.tokens == estimate_tokens(card.text) <= 350
    lines = card.text.splitlines()
    assert lines[0] == "Modèle Dossier (table dossiers, bibliotheque/app/Models/Dossier.php:9)" and lines[1].startswith("· champs : champ_00 (string, requis), ")
    assert "champ_19 (string, requis)" in lines[1]
    assert lines[2] == "· routes : GET /dossiers/variante-00/{id} → DossierController@variante0, GET /dossiers/variante-01/{id} → @variante1, … (+10)"
    assert lines[3] == "· " + variants[0]  # les 14 règles, entières, avant les routes raccourcies
    assert len(lines) == 4
    # seules les routes affichées sont couvertes par la fiche : les autres restent des passages citables
    assert card.covers == {("model", dm["models"][0]), ("endpoint", rt["endpoints"][0]), ("endpoint", rt["endpoints"][1])}


LONG_QUESTION = ("Comment OrderService::confirm vérifie-t-il le statut draft avant de passer en confirmed, renseigner confirmed_at, "
                 "notifier le client, journaliser la décision et déclencher la facture ?")


def test_followup_terms_come_from_previous_question_and_sources():
    previous = "que dit le document paramétrage des formulaires"
    sources = [{"path": "public/documentation/specification-formulaires-commandes.pdf"}, {"title": "Paramétrage des formulaires de commande"},
               {"path": "scripts/documentation/contenu_configuration.py"}, {"path": "app/Models/Ignored.php"}]
    terms = followup_terms("explique moi la définition des champs avec exemple", previous, sources)
    # identifiants, titre du document (3 premières sources) puis mots-clés ≥ 4 lettres ; un nom de fichier composé
    # (« contenu_configuration », « specification-formulaires-commandes ») n'est jamais un terme : FTS5 le cherche comme une
    # phrase exacte que seul ce fichier contient et le canal lexical (tous les termes exigés) se réduirait à lui
    assert terms == ["Paramétrage des formulaires de commande", "paramétrage", "formulaires"]
    assert "dit" not in terms and "Ignored" not in terms
    # un nom nu de classe reste un terme (« Emprunt »), pas un nom générique (« api » de routes/api.php)
    terms = followup_terms("et ses champs ?", "décris-moi ce modèle",
                           [{"path": "app/Models/Emprunt.php", "source_type": "model", "title": "Emprunt"},
                            {"path": "routes/api.php", "source_type": "endpoint", "title": "POST /emprunts"}])
    assert terms[0] == "Emprunt" and "api" not in terms
    assert len(followup_terms("et ?", "Order OrderService OrderController confirm statut commande client facture", [])) <= 8
    # les termes déjà dans la question ne sont pas répétés ; question longue avec identifiant : pas de suivi
    assert followup_terms("et le document paramétrage des formulaires ?", previous, []) == []
    assert followup_terms("et le document paramétrage ?", previous, [])[-1] == "formulaires"
    assert followup_terms(LONG_QUESTION, previous, sources) == []
    assert followup_terms("et les champs ?", "", sources) == [] and followup_terms("et les champs ?", "bonjour", []) == []


def test_followup_needs_an_opener_or_a_missing_subject():
    """Régression : une question n'était autonome qu'avec ≥ 8 mots-clés ET un identifiant, donc presque toute question
    française était un « suivi » et héritait des termes du tour précédent. Sur l'index réel, après « que dit le document
    paramétrage des formulaires », « quel est l'api d'authentification et un exemple d'appel » perdait toutes ses sources
    d'authentification (top 5 = contenu_configuration.py, GET /commandes, POST /commandes, le PDF du paramétrage). Désormais une
    question est un suivi seulement si elle s'ouvre sur un mot de liaison ou ne nomme aucun sujet à elle (ni identifiant,
    chemin, route ou table, ni mot-clé autre qu'un mot d'aspect : champs, validation, définition…)."""
    from develop.chat.orchestrator import FOLLOWUP_ASPECT_WORDS, FOLLOWUP_OPENERS, is_followup_question
    previous = "que dit le document paramétrage des formulaires"
    sources = [{"path": "public/documentation/specification-formulaires-commandes.pdf"}, {"path": "scripts/documentation/contenu_configuration.py"}]
    # une question qui nomme son sujet est autonome, même courte et sans identifiant
    for text in ("quel est l'api d'authentification et un exemple d'appel", "comment confirmer une commande via l'api ?",
                 "comment fonctionne le paramétrage", "qui appelle OrderService::confirm ?", "peux-tu me décrire le modèle User",
                 "quels champs a la table orders ?", "ouvre OrderService.php", "quelle est l'api pour rechercher un client ?",
                 "comment est calculé le montant total d'une commande ?", LONG_QUESTION):
        assert not is_followup_question(text), text
        assert followup_terms(text, previous, sources) == [], text
    # mot de liaison en tête, ou aucun sujet à elle (mots d'aspect seulement, pronoms) : suivi
    for text in ("et les champs ?", "Et ses commandes ?", "ok mais pourquoi ?", "et ?", "d'accord, et la validation ?",
                 "explique moi la définition des champs avec exemple", "quelles sont les règles de validation ?",
                 "montre moi le code", "il sert à quoi ?", "quelles tables sont liées ?", "et pour les utilisateurs ?"):
        assert is_followup_question(text), text
        assert followup_terms(text, previous, sources), text
    assert {"et", "mais", "ok", "donc", "d'accord"} <= FOLLOWUP_OPENERS and {"champs", "validation", "definition"} <= FOLLOWUP_ASPECT_WORDS
    assert is_followup_question("") and is_followup_question("   ?")  # rien à chercher : le tour précédent fait la question
    # une analyse déjà calculée est réutilisée telle quelle (pas de seconde analyse)
    from develop.search.query import analyze_query
    assert not is_followup_question("quel est l'api d'authentification", analyze_query("quel est l'api d'authentification"))
    assert is_followup_question("quels sont les champs ?", analyze_query("quels sont les champs ?"))


async def test_followup_question_enriches_search_but_not_the_prompt(db: Database, workspace: WorkspaceConfig, seeded):
    engine = FakeSearchEngine([doc_hit()])
    llm = FakeLLM([text_chunks("Le document décrit les champs [S1].")])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(32768, tools=False))
    conv = db.create_conversation()
    first = await collect(orch, conv, "que dit le document paramétrage des formulaires")
    assert first[0].data == {"text": "Recherche dans le projet…"}
    assert engine.calls[0]["query"] == "que dit le document paramétrage des formulaires"  # première question : rien à conserver
    events = await collect(orch, conv, "explique moi la définition des champs avec exemple")
    assert types(events) == ["status", "sources", "delta", "done"]
    status = events[0].data["text"]  # sources de la réponse précédente (titre du document) puis mots-clés de la question
    assert status.startswith("Question de suivi : contexte « Dictionnaire de données ") and status.endswith("paramétrage formulaires » conservé")
    context = status[len("Question de suivi : contexte « "):-len(" » conservé")]
    assert engine.calls[1]["query"] == "explique moi la définition des champs avec exemple [contexte précédent : %s]" % context
    # le modèle reçoit la question d'origine, sans le marqueur ; le message enregistré aussi
    user_msg = llm.calls[1]["messages"][-1].content
    assert user_msg.startswith("explique moi la définition des champs avec exemple\n\n") and "contexte précédent" not in user_msg
    assert db.get_conversation(conv)["messages"][-2]["content"] == "explique moi la définition des champs avec exemple"
    # question longue avec identifiants : recherche telle quelle
    events = await collect(orch, conv, LONG_QUESTION)
    assert events[0].data == {"text": "Recherche dans le projet…"} and engine.calls[2]["query"] == LONG_QUESTION
    # recherche seule : jamais enrichie ; après une petite conversation : rien à conserver non plus
    events = await collect(orch, conv, "et les champs ?", mode="search_only")
    assert types(events) == ["status", "sources", "done"] and engine.calls[3]["query"] == "et les champs ?"
    await collect(orch, conv, "merci")
    events = await collect(orch, conv, "et les champs ?")
    assert events[0].data == {"text": "Recherche dans le projet…"} and engine.calls[-1]["query"] == "et les champs ?"
    # les crochets du marqueur sont tolérés par l'analyse : les termes deviennent mots-clés / identifiants
    from develop.search.query import analyze_query
    analysis = analyze_query("et les champs ? [contexte précédent : Emprunt StoreEmpruntRequest paramétrage formulaires]")
    assert {"emprunt", "storeempruntrequest", "paramétrage", "formulaires"} <= set(analysis.keywords)
    assert "StoreEmpruntRequest" in analysis.identifiers and "contexte" not in analysis.identifiers


async def test_smalltalk_answers_without_search_or_llm(db: Database, workspace: WorkspaceConfig, seeded):
    engine = FakeSearchEngine([file_hit()])
    llm = FakeLLM([text_chunks("ne doit jamais être appelé")])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(8192, tools=True))
    conv = db.create_conversation()
    answers: List[str] = []
    for text in ("bonjour", "Merci beaucoup !", "ok", "Salut", "ça va ?", "d'accord, merci"):
        events = await collect(orch, conv, text)
        assert types(events) == ["status", "sources", "delta", "done"], text
        assert events[0].data["text"].startswith("Petite conversation") and events[1].data["items"] == []
        answer = events[2].data["text"]
        assert "Develop" in answer and "« Boutique »" in answer and "décris-moi le modèle Order" in answer and "GET /orders" in answer
        done = events[-1].data
        assert done["message_id"] and done["usage"] == {} and done["budget"]["tools_mode"] == "none" and done["budget"]["num_ctx"] == 8192
        assert done["invalid_citations"] == [] and done["truncated_sources"] == []
        answers.append(answer)
    assert engine.calls == [] and engine.embed_calls == 0 and llm.calls == []
    assert answers[0].startswith("Bonjour !") and answers[1].startswith("Avec plaisir !") and answers[2].startswith("Très bien.")
    assert answers[4].startswith("Très bien, merci !")
    msgs = db.get_conversation(conv)["messages"]
    assert [m["role"] for m in msgs][-2:] == ["user", "assistant"] and msgs[-1]["content"] == answers[-1] and msgs[-1]["sources"] == []
    # pas une petite conversation : un terme du projet, une vraie question, plus de trois mots
    for text in ("bonjour, décris le modèle Order", "merci pour la réponse précédente", "Order", "ok et les champs ?", ""):
        assert not is_smalltalk(text), text
    for text in ("Bonjour !", "merci", "MERCI BEAUCOUP", "ok ok", "Ça va", "bonne journée", "non"):
        assert is_smalltalk(text), text
    # mode recherche seule inchangé : « bonjour » est cherché comme avant
    events = await collect(orch, conv, "bonjour", mode="search_only")
    assert types(events) == ["status", "sources", "done"] and engine.calls[0]["query"] == "bonjour"
    assert [s["ref"] for s in events[1].data["items"]] == ["S1"]
    # sans index : exemples génériques, jamais d'erreur
    with Database(":memory:") as empty:
        text = smalltalk_reply(empty, "", "bonjour")
        assert "modèle <Nom>" in text and "POST /<ressource>" in text and "Je suis l'assistant de Develop pour l'espace de travail." in text


async def test_short_reply_to_the_model_reaches_it(db: Database, workspace: WorkspaceConfig, seeded):
    """Régression : « oui », « non », « Non merci », « ok top » étaient absorbés comme petites conversations (réponse
    générique « Très bien. Je suis l'assistant… ») même quand le modèle venait de poser une question : sa question
    restait sans réponse et « non » recevait « Très bien. ». Désormais un acquiescement qui suit une question du modèle,
    et un oui / non dès que le modèle a répondu, lui parviennent (question de suivi, historique compris) ; la question
    de suivi d'après garde le sujet d'avant le « oui » ; « non » sans rien à contester ouvre sur « Entendu. »."""
    from develop.chat.orchestrator import (SMALLTALK_ACKS, SMALLTALK_ANSWERS, SMALLTALK_SIGNATURE, asks_a_question,
                                           is_reply_to_model)

    engine = FakeSearchEngine([file_hit()])
    question = "Le modèle Order a trois champs [S1]. Voulez-vous le détail des routes ?"
    llm = FakeLLM([text_chunks(question), text_chunks("Les routes sont GET /orders et POST /orders [S1]."),
                   text_chunks("La validation est dans StoreOrderRequest [S1]."), text_chunks("Précisez ce qui ne convient pas [S1].")])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(32768, tools=False))
    conv = db.create_conversation()
    await collect(orch, conv, "décris-moi le modèle Order")
    # « oui » répond à la question du modèle : suivi (termes du tour précédent), historique, modèle — pas de réponse générique
    events = await collect(orch, conv, "oui")
    status = events[0].data["text"]
    assert status.startswith("Question de suivi : contexte « ") and "Order" in status, status
    assert engine.calls[-1]["query"].startswith("oui [contexte précédent : ") and len(llm.calls) == 2
    sent = llm.calls[-1]["messages"]
    assert [(m.role, m.content) for m in sent[1:3]] == [("user", "décris-moi le modèle Order"), ("assistant", question)]
    assert sent[-1].role == "user" and sent[-1].content.startswith("oui\n\n")
    assert "Les routes sont GET /orders" in db.get_conversation(conv)["messages"][-1]["content"]
    # la question de suivi d'après garde le sujet : la question précédente est celle d'avant le « oui », pas « oui »
    events = await collect(orch, conv, "et la validation ?")
    assert events[0].data["text"].startswith("Question de suivi : contexte « ") and "Order" in events[0].data["text"]
    assert orch._previous_turn(conv, None)[0] == "et la validation ?"
    # « non » après une affirmation la conteste : il parvient au modèle ; « ok top » après une affirmation reste une
    # petite conversation ; « oui » après une réponse de petite conversation (qui n'appelle rien) aussi
    await collect(orch, conv, "non")
    assert len(llm.calls) == 4 and llm.calls[-1]["messages"][-1].content.startswith("non\n\n")
    events = await collect(orch, conv, "ok top")
    assert events[0].data["text"].startswith("Petite conversation") and events[2].data["text"].startswith("Très bien.")
    events = await collect(orch, conv, "oui")
    assert events[0].data["text"].startswith("Petite conversation") and len(llm.calls) == 4
    assert engine.calls[-1]["query"].startswith("non [contexte précédent : ")  # dernière recherche : celle du « non »
    # fonctions pures : acquiescement et oui / non après une question du modèle ; politesse inchangée
    for text in ("oui", "non", "Non merci", "ok top", "bien", "d'accord", "Oui !"):
        assert is_reply_to_model(text, question) and not is_smalltalk(text, question), text
    for text in ("bonjour", "merci", "ça va ?", "bonne journée", "merci beaucoup"):
        assert is_smalltalk(text, question) and not is_reply_to_model(text, question), text
    statement = "Le modèle Order a trois champs [S1]."
    assert is_smalltalk("ok top", statement) and is_smalltalk("ok", statement) and is_smalltalk("bien", statement)
    assert not is_smalltalk("oui", statement) and not is_smalltalk("non", statement) and not is_smalltalk("Non merci", statement)
    assert is_smalltalk("oui") and is_smalltalk("non") and is_smalltalk("oui", "") and not is_reply_to_model("oui", "")
    assert not is_reply_to_model("ok mais pourquoi ?", question) and not is_reply_to_model("", question)
    canned = smalltalk_reply(db, "Boutique", "bonjour")
    assert SMALLTALK_SIGNATURE in canned and is_smalltalk("oui", canned) and is_smalltalk("ok", canned)
    assert SMALLTALK_ANSWERS.isdisjoint(SMALLTALK_ACKS) and {"oui", "non", "yes", "no"} <= SMALLTALK_ANSWERS
    # la question du modèle : dernière ligne finissant par « ? » hors citation, emphase ou guillemets
    assert asks_a_question("**Voulez-vous le détail ?**") and asks_a_question("Oui.\nVoulez-vous le détail des routes ? [S1]\n")
    assert asks_a_question("Souhaitez-vous la suite (routes, validation) ?") and not asks_a_question("Fin de la réponse [S1].")
    assert not asks_a_question("- « qui appelle <fonction> ? » / « quel est l'impact ? »") and not asks_a_question("") and not asks_a_question(canned)
    # ouverture : une négation n'est pas accueillie par « Très bien. »
    assert smalltalk_reply(db, "Boutique", "non").startswith("Entendu. Je suis l'assistant de Develop pour l'espace de travail « Boutique ».")
    assert smalltalk_reply(db, "Boutique", "Non merci").startswith("Entendu.") and smalltalk_reply(db, "Boutique", "ok").startswith("Très bien.")
    # dernière réponse du modèle : seulement si le dernier message est bien de l'assistant
    assert orch._previous_answer(conv, None).startswith("Très bien.")
    db.add_message(conv, "user", "question sans réponse")
    assert orch._previous_answer(conv, None) == ""


def test_system_prompt_tells_the_model_to_use_cards_verbatim():
    assert "10. Les fiches donnent champs, règles de validation et routes exacts : reprends-les tels quels." in SYSTEM_PROMPT
    assert "N'invente jamais un champ" in SYSTEM_PROMPT and "quel fichier ouvrir" in SYSTEM_PROMPT  # règles 4 et 5 inchangées


# --------------------------------------------------------------------------- intégration recherche ↔ contexte

def test_route_channel_hit_is_a_representative_in_context():
    """Le canal « route » du moteur a son représentant dans le contexte, comme les autres canaux, dans l'ordre du moteur."""
    from develop.chat.context_builder import CHANNEL_PRIORITY
    from develop.search.engine import _CHANNEL_PRIORITY
    assert CHANNEL_PRIORITY == _CHANNEL_PRIORITY == ("identifier", "route", "graph", "lexical", "semantic", "temporal")
    body = "\n".join("        ligne %02d : du code pour remplir le budget de contexte ;" % j for j in range(30))
    hits = [file_hit(chunk_id=10 + i, source_id=10 + i, file_id=10 + i, path="app/Big%d.php" % i, title="Big%d.php" % i,
                     line_start=1, line_end=30, text=body, score=0.9 - i * 0.01, channels=["lexical"]) for i in range(4)]
    route = file_hit(chunk_id=None, source_type="endpoint", source_id=5, file_id=50, path="routes/api.php", title="POST /orders/{order}/confirm",
                     line_start=13, line_end=13, score=0.05, channels=["route"], reason="route contenant « confirm »",
                     text="Route::post('/orders/{order}/confirm', [OrderController::class, 'confirm']);",
                     extra={"endpoint_id": 5, "handler": "OrderController@confirm", "language": "php"})
    big = estimate_tokens(format_block("S1", hits[0]), is_code=True)
    pack = build_context(hits + [route], big + 80)  # un seul gros passage tient : le second représentant est la route
    assert [s.source_type for s in pack.sources] == ["file", "endpoint"] and pack.sources[1].endpoint_id == 5
    assert len(pack.truncated) == 3


def test_endpoint_card_uses_the_request_class_resolved_by_the_search_engine(db: Database, tmp_path: Path):
    """Le canal « route » du moteur résout la classe de validation (``extra.request_class``) ; la fiche de route s'en
    sert quand la signature du gestionnaire n'en nomme pas, et n'invente rien sans indication."""
    g = seed_emprunt(db, tmp_path)
    rq = FileRow(repo_id="biblio", path="app/Http/Requests/IndexEmpruntRequest.php", language="php", kind="code")
    db.upsert_file(rq)
    rq_id = db.replace_file_artifacts(rq.id, symbols=[SymbolRow(name="IndexEmpruntRequest", kind="class", line_start=9, line_end=30)])["symbols"][0]
    base = dict(source_type="endpoint", source_id=g["get_endpoint"], title="GET /emprunts", path="routes/api.php",
                file_id=g["files"]["routes"].id, text="Route::get", channels=["route"], reason="route contenant « emprunts »")
    extra = {"endpoint_id": g["get_endpoint"], "handler": "EmpruntController@index", "request_class": "IndexEmpruntRequest",
             "request_class_id": rq_id, "request_file_id": rq.id}
    card = build_cards(db, [Hit(extra=extra, **base)])[0]
    assert "· classe de requête : IndexEmpruntRequest (bibliotheque/app/Http/Requests/IndexEmpruntRequest.php:9)" in card.text
    plain = build_cards(db, [Hit(extra={"endpoint_id": g["get_endpoint"], "handler": "EmpruntController@index"}, **base)])[0]
    assert "classe de requête" not in plain.text
    # la signature du gestionnaire (store(StoreEmpruntRequest …)) prime sur l'indication du moteur
    post = emprunt_hits(g)["post"]
    post = Hit(**dict(post.__dict__, extra=dict(post.extra, request_class="App\\Http\\Requests\\AutreRequest")))
    assert "· classe de requête : StoreEmpruntRequest (bibliotheque/app/Http/Requests/StoreEmpruntRequest.php:28) rules()" in build_cards(db, [post])[0].text


def test_model_card_lists_routes_linked_to_the_model_by_the_graph(db: Database, tmp_path: Path):
    """Le graphe porte désormais ``uses_model`` sur la route elle-même (``endpoint:E`` → ``model:M``) : une route gérée
    par un autre contrôleur mais reliée au modèle figure sur sa fiche, après les routes de ``<Modèle>Controller``."""
    g = seed_emprunt(db, tmp_path)
    routes = FileRow(repo_id="biblio", path="routes/livres.php", language="php", kind="code")
    db.upsert_file(routes)
    ep = db.replace_file_artifacts(routes.id, endpoints=[EndpointRow(
        side="server", method="POST", path="/livres/{livre}/emprunt", normalized_path="/api/v1/bibliotheque/livres/:param/emprunt",
        handler="LivreController@emprunt", framework="laravel", line=5)])["endpoints"][0]
    before = build_cards(db, [emprunt_hits(g)["model"]])[0]
    assert "LivreController@emprunt" not in before.text and ("endpoint", ep) not in before.covers
    db.add_relations([RelationRow(src=node_key("endpoint", ep), dst=node_key("model", g["model_id"]), kind="uses_model",
                                  meta={"rule": "handler", "handler": "LivreController@emprunt", "via": "name"})])
    after = build_cards(db, [emprunt_hits(g)["model"]])[0]
    line = next(l for l in after.text.splitlines() if l.startswith("· routes : "))
    assert line.endswith("POST /livres/{livre}/emprunt → LivreController@emprunt")  # après les routes du contrôleur du modèle
    assert "GET /emprunts → EmpruntController@index, POST /emprunts → @store" in line and ("endpoint", ep) in after.covers


async def test_followup_passages_bring_the_requested_section(db: Database, workspace: WorkspaceConfig, seeded):
    """« explique-moi la définition des champs » après « que dit le document … » : la recherche ne ramène que le
    sommaire (il contient tous les mots de la question) ; les morceaux du document cité au tour précédent sont
    relus et la section dont le titre porte le mot-clé (« 4. Les champs… ») passe en tête du contexte."""
    from develop.chat.orchestrator import followup_passages

    doc_id = db.add_document(DocumentRow(title="Paramétrage des formulaires", format="pdf", kind="upload", sha1="pdf-param"))
    sommaire = ("Paramétrage des formulaires de commande\nNote — Ce document décrit la définition d'un formulaire de commande et de ses champs.\n"
                "Sommaire\n1. Vue d'ensemble\n2. Modèle de données\n3. Le template DOCX\n4. Les champs (custom et partagés)\n"
                "5. Les champs de type référence")
    section3 = "3. Le template DOCX et ses placeholders\nLe template est un fichier Word avec des marqueurs {{nom:type}}."
    section4 = ("4. Les champs (custom et partagés)\nUn champ décrit une donnée à saisir. Attributs : nom_champ, type, obligatoire.\n"
                "POST /api/v1/shop/form-fields")
    section5 = "5. Les champs de type référence\nUn champ peut pointer vers une table (public.pays)."
    ids = db.replace_chunks("document", doc_id, [
        ChunkRow(text=sommaire, line_start=1, line_end=9, meta={"page": 1}),
        ChunkRow(text=section3, line_start=11, line_end=12, meta={"page": 2}),
        ChunkRow(text=section4, line_start=14, line_end=16, meta={"page": 3, "section": "4. Les champs (custom et partagés)",
                                                                   "heading_path": ["4. Les champs (custom et partagés)"]}),
        ChunkRow(text=section5, line_start=18, line_end=19, meta={"page": 4}),  # sans métadonnées : titre lu dans le texte
    ])
    sommaire_hit = doc_hit(chunk_id=ids[0], source_id=doc_id, title="Paramétrage des formulaires", page=1, text=sommaire,
                           extra={"document_id": doc_id}, reason="correspondance lexicale sur « champs »")
    engine = FakeSearchEngine([sommaire_hit])
    llm = FakeLLM([text_chunks("Le document décrit les champs [S1]."), text_chunks("Un champ décrit une donnée [S1].")])
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(32768, tools=False))
    conv = db.create_conversation()
    first = await collect(orch, conv, "que dit le document paramétrage des formulaires")
    assert [s["document_id"] for s in first[1].data["items"]] == [doc_id] and first[1].data["items"][0]["channels"] == ["lexical"]
    events = await collect(orch, conv, "explique moi la définition des champs avec exemple")
    assert types(events) == ["status", "sources", "delta", "done"]
    assert events[0].data["text"].startswith("Question de suivi : contexte « Paramétrage des formulaires ")
    items = events[1].data["items"]
    # §4 (mot-clé « champs » dans le titre de section) puis §5 (titre lu dans le texte) puis le sommaire (mots du
    # corps seulement) ; la section 3, sans mot-clé, n'est pas relue
    assert [(s["ref"], s["line_start"], s["page"], s["channels"]) for s in items[:3]] == [
        ("S1", 14, 3, ["followup"]), ("S2", 18, 4, ["followup"]), ("S3", 1, 1, ["followup"])]
    assert items[0]["reason"] == "suite de la conversation : §4. Les champs (custom et partagés) de « Paramétrage des formulaires »"
    assert items[1]["reason"] == "suite de la conversation : §5. Les champs de type référence de « Paramétrage des formulaires »"
    assert all(s["line_start"] != 11 for s in items) and sum(1 for s in items if s["line_start"] == 1) == 1  # sommaire une fois
    user_msg = llm.calls[1]["messages"][-1].content
    assert ("[S1] Document « Paramétrage des formulaires » §4. Les champs (custom et partagés), page 3 — suite de la "
            "conversation : §4. Les champs (custom et partagés) de « Paramétrage des formulaires »") in user_msg
    assert "Un champ décrit une donnée à saisir" in user_msg and "{{nom:type}}" not in user_msg
    # le passage relu porte l'empreinte du morceau (jumeaux fichier / document rapprochés par l'assembleur)
    passages = followup_passages(db, "explique moi la définition des champs avec exemple", first[1].data["items"])
    assert [p.chunk_id for p in passages] == [ids[2], ids[3], ids[0]] and passages[0].extra["sha1"] and passages[0].score == 0.0
    # question sans mot-clé exploitable, sources précédentes absentes ou sans morceaux : aucun passage
    assert followup_passages(db, "et ensuite ?", first[1].data["items"]) == []
    assert followup_passages(db, "explique les champs", []) == []
    assert followup_passages(db, "explique les champs", [{"document_id": doc_id + 100}, {"file_id": 99999}]) == []
    # une question longue avec identifiant n'est pas un suivi : rien de relu
    events = await collect(orch, conv, LONG_QUESTION)
    assert events[0].data == {"text": "Recherche dans le projet…"}
    assert all("followup" not in s["channels"] for s in events[1].data["items"])



async def test_model_phase_statuses_announce_loading_then_reading(tmp_path):
    """Après ``sources`` : « Chargement du modèle … » si le moteur ne l'a pas en mémoire, puis « Le modèle lit … ».
    Un appel qui a dû charger le modèle n'est pas pris pour une mesure de vitesse de lecture. ``/api/ps`` vide (``[]``)
    veut dire « rien en mémoire » : chargement annoncé, pas de mesure ; ``None`` « le moteur ne le dit pas » (serveur
    compatible OpenAI, Ollama injoignable) : rien d'annoncé, la mesure est conservée."""
    from develop.chat.orchestrator import ChatOrchestrator, STATUS_LOADING, STATUS_READING, _prefill_samples

    class _Llm(NullClient):
        engine = "ollama"

        def __init__(self, loaded):
            super().__init__()
            self._loaded = loaded

        async def loaded_models(self):
            return self._loaded

        async def chat_stream(self, messages, model, **kwargs):  # type: ignore[override]
            yield ChatChunk(text="ok", done=True, usage={"prompt_tokens": 4000, "completion_tokens": 1})

    db = Database(":memory:")
    ws = WorkspaceConfig(id="w", name="w", llm=LLMConfig(engine="ollama", model="qwen3.8:27b"))
    samples = 0
    # Même règle que GET /api/llm/state : une autre taille du même modèle (qwen3.8:9b) n'est pas « chargé ».
    for loaded, expect_loading in ((None, False), ([], True), (["qwen3.8:27b"], False), (["autre:1b"], True),
                                   (["qwen3.8:9b"], True)):
        orch = ChatOrchestrator(db, ws, _Llm(loaded), FakeSearchEngine([]))
        conv = db.create_conversation("t")
        events = [e async for e in orch.respond(conv, "question ?", mode="no_tools")]
        statuses = [e.data.get("text", "") for e in events if e.type == "status"]
        assert any(t.startswith(STATUS_READING.split(" %d")[0]) for t in statuses)
        assert any(t == STATUS_LOADING % "qwen3.8:27b" for t in statuses) is expect_loading
        samples += 0 if expect_loading else 1
        assert len(_prefill_samples(db, "ollama", "qwen3.8:27b")) == samples


async def test_prefill_measure_ignores_calls_where_the_engine_loaded_the_model():
    """Ollama rapporte ``load_duration`` (→ ``usage.load_duration_ms``) : au-delà de ``PREFILL_LOAD_MAX_S`` l'appel a
    (re)chargé le modèle (keep_alive expiré entre deux sondages de l'UI, autre ``num_ctx``, ``develop ask`` sans
    préchauffage) et son délai n'est pas une vitesse de lecture, même si ``/api/ps`` le disait en mémoire juste avant.
    ``first_token_s`` reste le délai perçu et ``done.usage`` rapporte ``load_duration_ms``."""
    from develop.chat.orchestrator import ChatOrchestrator, PREFILL_LOAD_MAX_S, _prefill_samples

    class _Llm(NullClient):
        engine = "ollama"

        def __init__(self, load_ms):
            super().__init__()
            self.load_ms = load_ms

        async def loaded_models(self):
            return ["qwen3:14b"]

        async def chat_stream(self, messages, model, **kwargs):  # type: ignore[override]
            yield ChatChunk(text="ok", done=True,
                            usage={"prompt_tokens": 4000, "completion_tokens": 1, "load_duration_ms": self.load_ms})

    db = Database(":memory:")
    ws = WorkspaceConfig(id="w", name="w", llm=LLMConfig(engine="ollama", model="qwen3:14b"))
    assert PREFILL_LOAD_MAX_S == 1.0
    samples = 0
    for load_ms, measured in ((12_000, False), (1_001, False), (40, True), (0, True)):
        orch = ChatOrchestrator(db, ws, _Llm(load_ms), FakeSearchEngine([]))
        conv = db.create_conversation("t")
        events = [e async for e in orch.respond(conv, "question ?", mode="no_tools")]
        done = [e for e in events if e.type == "done"][-1].data
        assert done["usage"]["load_duration_ms"] == load_ms and done["budget"]["first_token_s"] is not None
        samples += 1 if measured else 0
        assert len(_prefill_samples(db, "ollama", "qwen3:14b")) == samples


async def test_prefill_not_measured_when_first_fragment_is_a_tool_call():
    """Ollama n'émet un appel d'outil qu'une fois son JSON entièrement généré : un premier appel qui commence par un
    ``tool_call`` ne mesure pas la vitesse de lecture (le délai mêlerait lecture et génération : ×4 sur un 14B, la
    moyenne glisserait vers le plancher et réduirait le contexte sans nécessité). ``first_token_s`` reste le délai perçu.
    Un premier fragment de texte, lui, est mesuré, même s'il précède un appel d'outil dans le même tour."""
    from develop.chat.orchestrator import ChatOrchestrator, _prefill_samples

    search_call = ToolCall(name="search", arguments={"query": "orders"}, id="c1")
    usage = {"prompt_tokens": 4000, "completion_tokens": 60}

    class _Llm(NullClient):
        engine = "ollama"

        def __init__(self, first_round):
            super().__init__()
            self.first_round = list(first_round)
            self.calls = 0

        async def loaded_models(self):
            return ["fake:7b"]

        async def chat_stream(self, messages, model, **kwargs):  # type: ignore[override]
            self.calls += 1
            if self.calls > 1:  # tour suivant l'outil
                yield ChatChunk(text="Réponse.", done=True, usage=usage)
                return
            await asyncio.sleep(0.02)  # lecture du prompt
            for chunk in self.first_round:
                if chunk.tool_calls:
                    await asyncio.sleep(0.06)  # génération du JSON de l'appel avant son émission
                yield chunk

    db = Database(":memory:")
    ws = WorkspaceConfig(id="w", name="w", llm=LLMConfig(engine="ollama", model="fake:7b"))

    async def run(first_round, mode):
        orch = ChatOrchestrator(db, ws, _Llm(first_round), FakeSearchEngine([]), model_info=info(32768))
        events = [e async for e in orch.respond(db.create_conversation("t"), "Quels champs a la table orders ?", mode=mode)]
        return [e for e in events if e.type == "done"][-1].data["budget"], types(events)

    # premier fragment = appel d'outil : délai perçu rapporté, aucune mesure enregistrée
    budget, kinds = await run([ChatChunk(tool_calls=[search_call], done=True, usage=usage)], "auto")
    assert kinds == ["status", "sources", "status", "tool_call", "tool_result", "delta", "done"]
    assert budget["first_token_s"] >= 0.08 and _prefill_samples(db, "ollama", "fake:7b") == []
    # réponse texte directe : mesurée (4 000 tokens en ≈ 0,02 s → plafond 2 000 tok/s)
    budget, kinds = await run([ChatChunk(text="Réponse.", done=True, usage=usage)], "no_tools")
    assert kinds == ["status", "sources", "status", "delta", "done"]
    assert budget["first_token_s"] < 0.08 and _prefill_samples(db, "ollama", "fake:7b") == [2000.0]
    # texte puis appel d'outil dans le même tour : le premier fragment de texte date la fin de la lecture
    budget, kinds = await run([ChatChunk(text="Je regarde. "), ChatChunk(tool_calls=[search_call], done=True, usage=usage)],
                              "auto")
    assert kinds == ["status", "sources", "status", "delta", "tool_call", "tool_result", "delta", "done"]
    assert budget["first_token_s"] < 0.08 and _prefill_samples(db, "ollama", "fake:7b") == [2000.0, 2000.0]


async def test_prefill_measure_discounts_the_prompt_prefix_cached_by_the_engine(db: Database, workspace: WorkspaceConfig, seeded):
    """``usage.prompt_tokens`` (``prompt_eval_count`` d'Ollama) compte tout le prompt alors que le moteur ne relit que
    ce qui n'est pas dans son cache de préfixe (préfixe commun avec ses requêtes précédentes) : le délai jusqu'au
    premier fragment ne couvre que cette part. Une question reposée telle quelle dans une nouvelle conversation
    (prompt identique, première lettre immédiate) ne fournit donc aucune mesure — au lieu d'un échantillon à
    2 000 tok/s qui relâchait la borne de latence (11,7k tokens relus en 192 s au tour suivant) — et un tour ordinaire
    (consignes et début de l'historique en cache) est mesuré sur la part réellement lue."""
    from develop.chat.orchestrator import PREFILL_MIN_PROMPT_TOKENS, _prefill_samples

    class CachedPromptLLM(LLMClient):
        """Moteur simulé avec cache de préfixe : lit ``chars_per_s`` caractères/s de ce qui n'est pas dans le préfixe
        commun avec l'appel précédent, rapporte ``prompt_tokens`` = tout le prompt (4 caractères/token)."""
        engine = "fake"

        def __init__(self, chars_per_s: float):
            self.chars_per_s = chars_per_s
            self.last = ""
            self.calls: List[Dict[str, int]] = []

        async def chat_stream(self, messages, model, **kwargs):  # type: ignore[override]
            text = "".join("%s\n%s\n" % (m.role, m.content) for m in messages)
            cached = 0
            for a, b in zip(self.last, text):
                if a != b:
                    break
                cached += 1
            self.last = text
            self.calls.append({"total": len(text), "cached": cached})
            await asyncio.sleep((len(text) - cached) / self.chars_per_s)
            yield ChatChunk(text="Réponse [S1]. " + "Le statut est documenté. " * 60)
            yield ChatChunk(done=True, usage={"prompt_tokens": len(text) // 4, "completion_tokens": 5})

    engine = FakeSearchEngine([file_hit(), doc_hit(), note_hit()])
    llm = CachedPromptLLM(6000.0)  # vitesse réelle 1 500 tok/s : sous le plafond 2 000, une mesure gonflée se verrait
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(32768, tools=True))
    question = "Quels champs a la table orders ?"
    # (1) première question : rien en cache, le prompt entier est lu → mesure ≈ 1 500 tok/s
    first = db.create_conversation()
    d1 = (await collect(orch, first, question, mode="no_tools"))[-1].data["budget"]
    assert llm.calls[-1]["cached"] == 0 and llm.calls[-1]["total"] // 4 >= PREFILL_MIN_PROMPT_TOKENS
    assert d1["first_token_s"] >= 0.3 and 1100 <= d1["prefill_tps"] <= 1700
    assert _prefill_samples(db, "fake", "fake:7b") == [d1["prefill_tps"]]
    # (2) la même question dans une nouvelle conversation : prompt identique, entièrement en cache, première lettre
    #     immédiate → aucune mesure (le compte total ÷ délai aurait valu le plafond 2 000 tok/s)
    second = db.create_conversation()
    d2 = (await collect(orch, second, question, mode="no_tools"))[-1].data["budget"]
    assert llm.calls[-1]["cached"] == llm.calls[-1]["total"] and d2["first_token_s"] < 0.1
    assert _prefill_samples(db, "fake", "fake:7b") == [d1["prefill_tps"]] and d2["prefill_tps"] == d1["prefill_tps"]
    # (3) une autre question dans cette conversation : consignes et début de l'historique en cache (≈ la moitié du
    #     prompt), mesure sur la part relue → toujours ≈ 1 500 tok/s, et non 2 000
    d3 = (await collect(orch, second, "Quelles routes manipulent la table orders ?", mode="no_tools"))[-1].data["budget"]
    call = llm.calls[-1]
    assert 0.3 < call["cached"] / call["total"] < 0.7 and (call["total"] - call["cached"]) // 4 >= PREFILL_MIN_PROMPT_TOKENS
    samples = _prefill_samples(db, "fake", "fake:7b")
    assert len(samples) == 2 and abs(samples[1] - samples[0]) <= 0.25 * samples[0] and samples[1] < 1800
    assert d3["prefill_tps"] == min(samples)  # médiane basse des deux mesures
    # (4) la part en cache retirée du compte : trop peu de tokens relus → pas de mesure ; sinon mesure sur le reste
    assert orch._cached_fraction("autre:1b", "x") == 0.0 and orch._cached_fraction("fake:7b", "") == 0.0
    assert orch._record_prefill(4000, 1.0, cached_fraction=0.95) is None  # 200 tokens relus < 256
    assert orch._record_prefill(4000, 1.0, cached_fraction=0.5) == sorted(samples)[1]  # 2 000 tok/s : médiane basse
    assert _prefill_samples(db, "fake", "fake:7b") == samples + [2000.0]


def test_prefill_tps_median_is_persisted_per_model():
    from develop.chat.orchestrator import (PREFILL_META_KEY, PREFILL_META_VERSION, PREFILL_SAMPLES, load_prefill_tps,
                                           record_prefill_tps)

    with Database(":memory:") as mem:
        assert load_prefill_tps(mem, "ollama", "qwen3:14b") is None
        retained = None
        for tps in (300, 50, 60, 70, 80, 90, 100):  # sept mesures : seules les cinq dernières comptent
            retained = record_prefill_tps(mem, "ollama", "qwen3:14b", tps)
        assert retained == 80.0 == load_prefill_tps(mem, "ollama", "qwen3:14b")
        stored = json.loads(mem.get_meta(PREFILL_META_KEY % "qwen3:14b"))
        assert stored["engine"] == "ollama" and stored["samples"] == [60, 70, 80, 90, 100] and stored["avg"] == 80
        assert stored["v"] == PREFILL_META_VERSION == 2 and stored["median"] == 80
        assert len(stored["samples"]) == PREFILL_SAMPLES and stored["updated_at"] > 0
        # la valeur retenue est la médiane basse : deux mesures gonflées (prompt en cache non détecté : cache d'un
        # autre slot, redémarrage de l'outil) ne la déplacent pas, la moyenne aurait relâché la borne de latence
        assert record_prefill_tps(mem, "ollama", "qwen3:14b", 2000) == 90.0  # [70, 80, 90, 100, 2000]
        assert record_prefill_tps(mem, "ollama", "qwen3:14b", 2000) == 100.0  # [80, 90, 100, 2000, 2000]
        assert json.loads(mem.get_meta(PREFILL_META_KEY % "qwen3:14b"))["avg"] == 854  # la moyenne, elle, aurait cédé
        # nombre pair : la plus petite des deux centrales (surestimer coûte des minutes, sous-estimer des sources)
        assert record_prefill_tps(mem, "ollama", "pair:7b", 60) == 60.0 and record_prefill_tps(mem, "ollama", "pair:7b", 2000) == 60.0
        # les mesures d'avant la correction (sans version : prompt entier compté, cache compris) sont ignorées puis remplacées
        mem.set_meta(PREFILL_META_KEY % "ancien:14b", json.dumps({"engine": "ollama", "samples": [2000, 2000, 60], "avg": 1353}))
        assert load_prefill_tps(mem, "ollama", "ancien:14b") is None
        assert record_prefill_tps(mem, "ollama", "ancien:14b", 60) == 60.0
        assert json.loads(mem.get_meta(PREFILL_META_KEY % "ancien:14b"))["samples"] == [60]
        # bornes [40, 2000] ; un autre modèle n'a rien ; le même nom sur un autre moteur n'est pas comparable
        assert record_prefill_tps(mem, "ollama", "petit:1b", 1_000_000) == 2000.0
        assert record_prefill_tps(mem, "ollama", "lent:70b", 3) == 40.0
        assert load_prefill_tps(mem, "ollama", "autre:7b") is None and load_prefill_tps(mem, "openai_compat", "qwen3:14b") is None
        # un autre moteur qui mesure repart de zéro ; mesures invalides refusées ; méta corrompue tolérée
        assert record_prefill_tps(mem, "openai_compat", "qwen3:14b", 500) == 500.0
        assert record_prefill_tps(mem, "openai_compat", "qwen3:14b", 0) is None
        assert record_prefill_tps(mem, "openai_compat", "", 50) is None and record_prefill_tps(None, "ollama", "x", 50) is None
        mem.set_meta(PREFILL_META_KEY % "qwen3:14b", "{pas du json")
        assert load_prefill_tps(mem, "openai_compat", "qwen3:14b") is None


async def test_measured_reading_speed_is_reported_and_bounds_the_next_budget(db: Database, workspace: WorkspaceConfig, seeded):
    import re as _re

    from develop.chat.orchestrator import PREFILL_META_KEY, load_prefill_tps, record_prefill_tps

    engine = FakeSearchEngine([file_hit(), doc_hit(), note_hit()])
    reading = _re.compile(r"Le modèle lit (\d+) sources \(≈ (\d+) tokens, ~(\d+) s estimées\) avant de répondre…")
    # (1) aucune mesure : estimation (fake:7b sur une machine sans GPU → plancher 40 tok/s) ; le contexte est plafonné
    #     au plancher (40 × 25 s = 1 000 tokens < consignes) et le délai annoncé compte le prompt entier à cette vitesse
    llm = FakeLLM([text_chunks("Réponse [S1].", usage={"prompt_tokens": 4000, "completion_tokens": 5})], delay=0.02)
    orch = make_orchestrator(db, workspace, llm, engine, model_info=info(32768, tools=True))
    assert orch.prefill_tps() == (40.0, False)
    conv = db.create_conversation()
    events = [e async for e in orch.respond(conv, "Quels champs a la table orders ?", mode="no_tools")]
    status = [e.data["text"] for e in events if e.type == "status"][-1]
    m = reading.fullmatch(status)
    assert m, status
    n_sources, context_tokens, seconds = (int(g) for g in m.groups())
    done = events[-1].data["budget"]
    assert n_sources == len(events[1].data["items"]) and context_tokens == done["context_tokens_used"]
    assert seconds >= (context_tokens + estimate_tokens(llm.calls[0]["messages"][0].content)) / 40
    assert done["latency_capped"] is True and done["max_context_tokens"] == 1500
    assert "plafonné à 1500 tokens pour rester sous 25 s (lecture estimée 40 tokens/s)" in done["source"]
    # le premier appel est chronométré : 4 000 tokens de prompt en ≈ 20 ms → borné à 2 000 tok/s, moyenne persistée
    assert done["first_token_s"] >= 0.02 and done["prefill_tps"] == 2000.0 == load_prefill_tps(db, "fake", "fake:7b")
    # (2) la moyenne mesurée remplace l'estimation : à 2 000 tok/s le contexte n'est plus bridé ; un prompt court
    #     (100 tokens) ne fournit pas de mesure
    llm2 = FakeLLM([text_chunks("Réponse [S1].")])
    orch2 = make_orchestrator(db, workspace, llm2, engine, model_info=info(32768, tools=True))
    assert orch2.prefill_tps() == (2000.0, True)
    done2 = (await collect(orch2, conv, "Quelles routes manipulent la table orders ?", mode="no_tools"))[-1].data["budget"]
    assert done2["latency_capped"] is False and done2["max_context_tokens"] > 1500 and done2["prefill_tps"] == 2000.0
    assert done2["first_token_s"] is not None and load_prefill_tps(db, "fake", "fake:7b") == 2000.0
    # (3) un modèle mesuré lent (60 tok/s) : plancher, explication et délai annoncé ≈ prompt ÷ 60
    db.set_meta(PREFILL_META_KEY % "fake:7b", None)
    record_prefill_tps(db, "fake", "fake:7b", 60)
    orch3 = make_orchestrator(db, workspace, FakeLLM([text_chunks("Réponse [S1].")]), engine, model_info=info(32768, tools=True))
    events3 = [e async for e in orch3.respond(conv, "Quels champs a la table orders ?", mode="no_tools")]
    m3 = reading.fullmatch([e.data["text"] for e in events3 if e.type == "status"][-1])
    assert m3 and int(m3.group(3)) >= int(m3.group(2)) / 60
    done3 = events3[-1].data["budget"]
    assert done3["max_context_tokens"] == 1500 and done3["latency_capped"] is True and done3["prefill_tps"] == 60.0
    assert "lecture estimée 60 tokens/s" in done3["source"]
    # recherche seule : aucun appel, donc aucun délai mesuré, mais la borne du budget est rapportée
    done_s = (await collect(orch3, conv, "Question", mode="search_only"))[-1].data["budget"]
    assert done_s["first_token_s"] is None and done_s["prefill_tps"] == 60.0 and done_s["latency_capped"] is True



async def test_nothink_is_always_sent_to_ollama_even_without_model_info():
    """« /set nothink » : think=False part vers Ollama même quand les capacités du modèle sont inconnues ;
    think=True n'est envoyé que si le modèle supporte la réflexion."""
    from develop.chat.orchestrator import ChatOrchestrator

    class _Llm(NullClient):
        engine = "ollama"

        def __init__(self):
            super().__init__()
            self.kwargs = []

        async def chat_stream(self, messages, model, **kwargs):  # type: ignore[override]
            self.kwargs.append(kwargs)
            yield ChatChunk(text="ok", done=True)

    db = Database(":memory:")
    for think, info, expected in ((False, None, False), (True, None, "absent"),
                                  (True, info_with_caps(["completion", "thinking"]), True),
                                  (False, info_with_caps(["completion"]), False)):
        ws = WorkspaceConfig(id="w", name="w", llm=LLMConfig(engine="ollama", model="qwen3:14b", think=think))
        llm = _Llm()
        orch = ChatOrchestrator(db, ws, llm, FakeSearchEngine([]), model_info=info)
        conv = db.create_conversation("t")
        async for _ in orch.respond(conv, "question ?", mode="no_tools"):
            pass
        sent = llm.kwargs[0].get("think", "absent")
        assert sent == expected, (think, info, sent)


def info_with_caps(caps):
    return ModelInfo(name="qwen3:14b", context_length=40960, capabilities=list(caps))
