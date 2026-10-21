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
from develop.chat.budget import (MAX_ANSWER_RESERVE, MIN_ANSWER_RESERVE, MIN_TOOL_BUDGET, system_prompt_tokens,
                                 tool_specs_tokens)
from develop.chat.context_builder import LocalHit, render_sources_as_text
from develop.chat.orchestrator import (INTERRUPTED_PREFIX, SOURCES_FRAME_TOKENS, TOOLS_EXHAUSTED_TEXT, _message_tokens,
                                       find_invalid_citations)
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


async def collect(orch: ChatOrchestrator, conversation_id: int, text: str, **kwargs: Any) -> List[ChatEvent]:
    return [ev async for ev in orch.respond(conversation_id, text, **kwargs)]


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
        assert b.history_tokens_max >= min(768, b.num_ctx // 4)
        assert b.history_tokens_max <= max(1024, int(b.num_ctx * 0.12))
        assert b.context_tokens_max > 0
    big = compute_budget(LLMConfig(num_ctx=16384, max_context_chars=1_000_000), info(16384), None, intent="factual")
    # les consignes sont mesurées sur le vrai gabarit (définitions d'outils natifs comprises), pas devinées
    assert big.system_tokens == system_prompt_tokens("native", 16384) and big.answer_reserve == int(16384 * 0.30)
    assert big.tools_mode == "native" and big.tool_rounds_max == 4 and big.tools_tokens_max >= MIN_TOOL_BUDGET


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
    assert shares["factual"] < shares["who_calls"] < shares["broad"] < shares["impact"]
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


def test_budget_to_dict_and_tolerant_inputs():
    b = compute_budget(None, None, None)  # configuration absente : valeurs par défaut
    assert isinstance(b, Budget) and b.num_ctx == 8192
    d = b.to_dict()
    assert set(d) == {"num_ctx", "system_tokens", "question_tokens", "history_tokens_max", "context_tokens_max",
                      "tools_tokens_max", "answer_reserve", "tool_rounds_max", "tools_mode", "source"}
    assert d["tools_mode"] == "native" and d["question_tokens"] == 0
    assert compute_budget(None, None, None, tools_mode="turbo").tools_mode == "none"  # mode inconnu = sans outils


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
    workspace.llm.num_ctx = 4096  # historique ≤ 1024 tokens ≈ 4096 caractères ; chaque message ≈ 354 tokens
    conv = db.create_conversation()
    for i in range(6):
        db.add_message(conv, "user", "Question %d : %s" % (i, "détail " * 200))
        db.add_message(conv, "assistant", "Réponse %d : %s" % (i, "contenu " * 200))
    llm = FakeLLM([text_chunks("ok")])
    orch = make_orchestrator(db, workspace, llm, FakeSearchEngine([file_hit()]), model_info=info(8192, tools=True))
    events = await collect(orch, conv, "Nouvelle question")
    assert types(events)[-1] == "done"
    messages = llm.calls[0]["messages"]
    system, history, user = messages[0], messages[1:-1], messages[-1]
    assert user.role == "user" and user.content.startswith("Nouvelle question")
    assert sum(estimate_tokens(m.content) for m in history) <= 1024
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
    assert len(hist) == 1 and hist[0].content.endswith(" […]") and estimate_tokens(hist[0].content) <= 1024
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
        if ctx <= 4096:
            # la fenêtre ne porte pas les outils : ils sont désactivés, leurs définitions ne partent pas
            assert done["tools_mode"] == "none" and call["tools"] is None and "Aucun outil" in system
            assert "outils désactivés" in done["source"]
        else:
            assert done["tools_mode"] == ("native" if tools else "text")
            assert (call["tools"] == TOOL_SPECS) is tools
        # le résumé de l'historique tient dans le budget d'historique avec les messages gardés
        history = call["messages"][1:-1]
        summary = system.split(HISTORY_SUMMARY_PREFIX, 1)[1] if HISTORY_SUMMARY_PREFIX in system else ""
        assert sum(_message_tokens(m) for m in history) + estimate_tokens(HISTORY_SUMMARY_PREFIX + summary) \
            <= compute_budget(workspace.llm, info(ctx, tools=tools), profile(32), tools_mode=done["tools_mode"],
                              system_tokens=estimate_tokens(system)).history_tokens_max + 1
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
    llm3 = FakeLLM([search, search, search, text_chunks("Fin [S2].")])
    orch3 = make_orchestrator(db, workspace, llm3, FakeSearchEngine([file_hit()], tool_hits=big_tool_hits()),
                              model_info=info(8192, tools=True))
    b3 = (await collect(orch3, db.create_conversation(), "Quels champs a la table orders ?"))[-1].data["budget"]
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
