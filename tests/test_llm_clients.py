from __future__ import annotations

import json

import httpx
import pytest

from develop.config import LLMConfig
from develop.llm import base, factory
from develop.llm.ollama import OllamaClient
from develop.llm.openai_compat import OpenAICompatClient

# ------------------------------------------------------------------ faux Ollama

def fake_ollama(request: httpx.Request) -> httpx.Response:
    p = request.url.path
    if p == "/api/version":
        return httpx.Response(200, json={"version": "0.12.1"})
    if p == "/api/tags":
        return httpx.Response(200, json={"models": [
            {"name": "qwen2.5-coder:7b", "size": 4_700_000_000, "details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M"}},
            {"name": "nomic-embed-text:latest", "size": 274_000_000, "details": {"family": "nomic-bert", "parameter_size": "137M", "quantization_level": "F16"}},
        ]})
    if p == "/api/show":
        body = json.loads(request.content)
        if body.get("model") == "missing:latest":
            return httpx.Response(404, json={"error": "model not found"})
        return httpx.Response(200, json={"details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
                                         "capabilities": ["completion", "tools"],
                                         "model_info": {"qwen2.context_length": 32768, "qwen2.embedding_length": 3584}})
    if p == "/api/chat":
        body = json.loads(request.content)
        assert body["stream"] is True and body["options"]["num_ctx"] == 8192
        if body["model"] == "missing:latest":
            return httpx.Response(404, json={"error": "model not found"})
        if body.get("tools"):
            lines = [{"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "search", "arguments": {"query": "Order"}}}]}, "done": False},
                     {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 12, "eval_count": 3}]
        else:
            lines = [{"message": {"role": "assistant", "content": "Bon"}, "done": False},
                     {"message": {"role": "assistant", "content": "jour"}, "done": False},
                     {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 10, "eval_count": 2}]
        return httpx.Response(200, content="\n".join(json.dumps(l) for l in lines).encode())
    if p == "/api/embed":
        body = json.loads(request.content)
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3] for _ in body["input"]]})
    return httpx.Response(404)


async def test_ollama_health_models_and_info():
    c = OllamaClient("http://127.0.0.1:11434", transport=httpx.MockTransport(fake_ollama))
    st = await c.health()
    assert st.ok and st.version == "0.12.1" and [m.name for m in st.models] == ["qwen2.5-coder:7b", "nomic-embed-text:latest"]
    assert st.models[1].is_embedding and not st.models[0].is_embedding
    info = await c.model_info("qwen2.5-coder:7b")
    assert info.context_length == 32768 and info.supports_tools and info.quantization == "Q4_K_M"
    with pytest.raises(base.ModelNotFound):
        await c.model_info("missing:latest")
    await c.aclose()


async def test_ollama_chat_stream_text_and_usage():
    c = OllamaClient(transport=httpx.MockTransport(fake_ollama))
    chunks = [ch async for ch in c.chat_stream([base.ChatMessage("user", "salut")], "qwen2.5-coder:7b", num_ctx=8192)]
    assert "".join(ch.text for ch in chunks) == "Bonjour"
    assert chunks[-1].done and chunks[-1].usage == {"prompt_tokens": 10, "completion_tokens": 2}


async def test_ollama_chat_stream_tool_calls_and_missing_model():
    c = OllamaClient(transport=httpx.MockTransport(fake_ollama))
    tools = [base.ToolSpec("search", "cherche", {"type": "object", "properties": {"query": {"type": "string"}}})]
    chunks = [ch async for ch in c.chat_stream([base.ChatMessage("user", "q")], "qwen2.5-coder:7b", num_ctx=8192, tools=tools)]
    calls = [tc for ch in chunks for tc in ch.tool_calls]
    assert calls and calls[0].name == "search" and calls[0].arguments == {"query": "Order"}
    with pytest.raises(base.ModelNotFound):
        async for _ in c.chat_stream([base.ChatMessage("user", "q")], "missing:latest", num_ctx=8192):
            pass


async def test_ollama_think_flag_is_only_sent_when_given():
    seen = []

    def handler(request):
        if request.url.path == "/api/chat":
            seen.append(json.loads(request.content).get("think", "absent"))
            return httpx.Response(200, content=json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}).encode())
        return fake_ollama(request)
    c = OllamaClient(transport=httpx.MockTransport(handler))
    async for _ in c.chat_stream([base.ChatMessage("user", "q")], "qwen3.8:27b", num_ctx=8192, think=False):
        pass
    async for _ in c.chat_stream([base.ChatMessage("user", "q")], "qwen3.8:27b", num_ctx=8192):
        pass
    assert seen == [False, "absent"]


async def test_ollama_embed():
    c = OllamaClient(transport=httpx.MockTransport(fake_ollama))
    vecs = await c.embed(["a", "b"], "nomic-embed-text")
    assert vecs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert await c.embed([], "x") == []


async def test_ollama_unreachable_gives_clear_message():
    def down(request):
        raise httpx.ConnectError("refused")
    c = OllamaClient(transport=httpx.MockTransport(down))
    st = await c.health()
    assert not st.ok and "ollama serve" in st.error


async def test_ollama_blocked_by_offline_guard():
    from develop.llm.offline_guard import OfflineGuard, OfflineViolation
    g = OfflineGuard()
    c = OllamaClient("http://api.remote.example", guard=g, transport=httpx.MockTransport(fake_ollama))
    # OfflineViolation n'est pas une LLMError : elle remonte telle quelle (et est journalisée) sans jamais atteindre le réseau
    with pytest.raises(OfflineViolation):
        await c.list_models()
    assert g.blocked_attempts()[0].host == "api.remote.example"
    st = await c.health()  # health() ne doit pas planter non plus : il explique le blocage
    assert not st.ok and "hors ligne" in st.error


# ------------------------------------------------------------------ faux serveur OpenAI-compatible

def fake_openai(request: httpx.Request) -> httpx.Response:
    p = request.url.path
    assert request.headers.get("authorization", "").startswith("Bearer ")
    if p == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "qwen2.5-coder-7b-instruct"}, {"id": "text-embedding-nomic"}]})
    if p == "/v1/chat/completions":
        body = json.loads(request.content)
        if body.get("tools"):
            events = [
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "search", "arguments": "{\"qu"}}]}, "finish_reason": None}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ery\": \"Order\"}"}}]}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        else:
            events = [{"choices": [{"delta": {"content": "Bon"}, "finish_reason": None}]},
                      {"choices": [{"delta": {"content": "jour"}, "finish_reason": None}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}]
        sse = "".join("data: %s\n\n" % json.dumps(e) for e in events) + "data: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})
    if p == "/v1/embeddings":
        body = json.loads(request.content)
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(i), 1.0]} for i, _ in enumerate(body["input"])]})
    return httpx.Response(404)


async def test_openai_compat_models_chat_and_embed():
    c = OpenAICompatClient("http://localhost:1234", transport=httpx.MockTransport(fake_openai))
    assert c.base_url.endswith("/v1")
    st = await c.health()
    assert st.ok and len(st.models) == 2
    chunks = [ch async for ch in c.chat_stream([base.ChatMessage("user", "salut")], "qwen2.5-coder-7b-instruct")]
    assert "".join(ch.text for ch in chunks) == "Bonjour"
    assert any(ch.usage.get("prompt_tokens") == 7 for ch in chunks)
    tools = [base.ToolSpec("search", "cherche")]
    chunks = [ch async for ch in c.chat_stream([base.ChatMessage("user", "q")], "qwen2.5-coder-7b-instruct", tools=tools)]
    calls = [tc for ch in chunks for tc in ch.tool_calls]
    assert calls == [base.ToolCall(name="search", arguments={"query": "Order"}, id="call_1")]
    assert await c.embed(["a", "b"], "text-embedding-nomic") == [[0.0, 1.0], [1.0, 1.0]]
    await c.aclose()


# ------------------------------------------------------------------ divers

def test_factory_and_null_client():
    assert isinstance(factory.create_client(LLMConfig(engine="none")), base.NullClient)
    assert isinstance(factory.create_client(LLMConfig(engine="ollama")), OllamaClient)
    assert isinstance(factory.create_client(LLMConfig(engine="openai_compat", base_url="http://127.0.0.1:8080")), OpenAICompatClient)


async def test_null_client_explains():
    c = base.NullClient()
    st = await c.health()
    assert not st.ok and "moteur" in st.error.lower()
    with pytest.raises(base.LLMUnavailable):
        async for _ in c.chat_stream([], "x"):
            pass


def test_parse_text_tool_calls():
    text = 'Je vais chercher.\n```json\n{"tool": "search", "args": {"query": "Order total"}}\n```\n{"tool":"read_file","args":{"path":"a.py","start":1,"end":40}}\n{"tool": "bad"}\n'
    calls = base.parse_text_tool_calls(text)
    assert [c.name for c in calls] == ["search", "read_file"]
    assert calls[1].arguments == {"path": "a.py", "start": 1, "end": 40}


def test_context_length_helper():
    assert base.context_length_from_model_info({"llama.context_length": 131072, "x": 1}) == 131072
    assert base.context_length_from_model_info({}) is None


async def test_ollama_keeps_models_warm_and_bounds_embedding_context():
    """keep_alive sur chat et embeddings, num_ctx réduit pour l'embedder, /api/ps et préchauffage."""
    seen = {}

    def handler(request):
        p = request.url.path
        body = json.loads(request.content) if request.content else {}
        seen[p] = body
        if p == "/api/chat":
            return httpx.Response(200, content=json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True}).encode())
        if p == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
        if p == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3.8:27b", "size": 1}]})
        if p == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return fake_ollama(request)
    c = OllamaClient(transport=httpx.MockTransport(handler))
    async for _ in c.chat_stream([base.ChatMessage("user", "q")], "qwen3.8:27b", num_ctx=8192):
        pass
    assert seen["/api/chat"]["keep_alive"] == "30m"
    await c.embed(["x"], "qwen3-embedding:0.6b")
    assert seen["/api/embed"]["keep_alive"] == "30m" and seen["/api/embed"]["options"]["num_ctx"] == 4096
    assert await c.loaded_models() == ["qwen3.8:27b"]
    assert await c.warm("qwen3.8:27b") is True and seen["/api/generate"] == {"model": "qwen3.8:27b", "keep_alive": "30m"}
    assert await base.NullClient().loaded_models() == [] and await base.NullClient().warm("x") is False
