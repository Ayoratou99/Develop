"""Adaptateur pour tout serveur local compatible API OpenAI (LM Studio, llama.cpp, vLLM, LocalAI, Jan…)."""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

import httpx

from develop.llm import offline_guard
from develop.llm.base import (ChatChunk, ChatMessage, EngineStatus, LLMClient, LLMError, LLMUnavailable,
                              ModelInfo, ModelNotFound, ToolCall, ToolSpec)


def _msg_to_openai(m: ChatMessage) -> Dict[str, Any]:
    d: Dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [{"id": c.id or "call-%d" % (i + 1), "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                           for i, c in enumerate(m.tool_calls)]
    if m.role == "tool":
        d["tool_call_id"] = m.tool_call_id or "call-1"
        if m.name:
            d["name"] = m.name
    return d


class OpenAICompatClient(LLMClient):
    engine = "openai_compat"

    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", api_key: str = "",
                 guard: Optional[offline_guard.OfflineGuard] = None,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.base_url = base
        headers = {"Authorization": "Bearer %s" % (api_key or "local")}
        self._client = offline_guard.make_async_client(guard=guard, inner=transport, base_url=self.base_url, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Any:
        try:
            r = await self._client.request(method, path, json=payload, timeout=timeout)
        except offline_guard.OfflineViolation:
            raise
        except httpx.HTTPError as exc:
            raise LLMUnavailable("Serveur compatible OpenAI injoignable sur %s (%s)." % (self.base_url, exc.__class__.__name__)) from exc
        if r.status_code == 404 and payload and payload.get("model"):
            raise ModelNotFound("Modèle « %s » introuvable sur %s" % (payload["model"], self.base_url))
        if r.status_code >= 400:
            raise LLMError("Le serveur a répondu %d sur %s : %s" % (r.status_code, path, r.text[:200]))
        try:
            return r.json()
        except ValueError as exc:
            raise LLMError("Réponse invalide sur %s" % path) from exc

    async def health(self) -> EngineStatus:
        try:
            models = await self.list_models()
        except (LLMError, offline_guard.OfflineViolation) as exc:
            return EngineStatus(ok=False, engine=self.engine, base_url=self.base_url, error=str(exc))
        return EngineStatus(ok=True, engine=self.engine, base_url=self.base_url, models=models)

    async def list_models(self) -> List[ModelInfo]:
        data = await self._request("GET", "/models", timeout=5.0)
        out: List[ModelInfo] = []
        for m in (data or {}).get("data", []):
            name = str(m.get("id", ""))
            caps = ["embedding"] if "embed" in name.lower() else ["completion", "tools"]
            out.append(ModelInfo(name=name, capabilities=caps, context_length=m.get("context_length") or m.get("max_context_length")))
        return out

    async def model_info(self, name: str) -> ModelInfo:
        for m in await self.list_models():
            if m.name == name:
                return m
        raise ModelNotFound("Modèle « %s » introuvable sur %s" % (name, self.base_url))

    async def chat_stream(self, messages: Sequence[ChatMessage], model: str, *, temperature: float = 0.2,
                          num_ctx: Optional[int] = None, tools: Optional[Sequence[ToolSpec]] = None,
                          max_tokens: Optional[int] = None, think: Optional[bool] = None) -> AsyncIterator[ChatChunk]:
        payload: Dict[str, Any] = {"model": model, "messages": [_msg_to_openai(m) for m in messages],
                                   "stream": True, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if tools:
            payload["tools"] = [t.to_openai() for t in tools]
        pending: Dict[int, Dict[str, Any]] = {}  # assemblage des tool_calls diffusés par fragments
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload, timeout=httpx.Timeout(600.0, connect=3.0)) as r:
                if r.status_code == 404:
                    raise ModelNotFound("Modèle « %s » introuvable sur %s" % (model, self.base_url))
                if r.status_code >= 400:
                    body = await r.aread()
                    raise LLMError("Le serveur a répondu %d : %s" % (r.status_code, body.decode("utf-8", "replace")[:300]))
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except ValueError:
                        continue
                    if data.get("error"):
                        raise LLMError("Serveur : %s" % data["error"])
                    choices = data.get("choices") or []
                    usage_raw = data.get("usage") or {}
                    usage = {k: int(v) for k, v in (("prompt_tokens", usage_raw.get("prompt_tokens")),
                                                    ("completion_tokens", usage_raw.get("completion_tokens"))) if isinstance(v, int)}
                    if not choices:
                        if usage:
                            yield ChatChunk(usage=usage)
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    for tc in delta.get("tool_calls") or []:
                        idx = int(tc.get("index", 0))
                        slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                        slot["id"] = tc.get("id") or slot["id"]
                        fn = tc.get("function") or {}
                        slot["name"] = fn.get("name") or slot["name"]
                        slot["args"] += fn.get("arguments") or ""
                    finish = choice.get("finish_reason")
                    text = delta.get("content") or ""
                    calls: List[ToolCall] = []
                    if finish:
                        for idx in sorted(pending):
                            slot = pending[idx]
                            try:
                                args = json.loads(slot["args"] or "{}")
                            except ValueError:
                                args = {"_raw": slot["args"]}
                            calls.append(ToolCall(name=slot["name"], arguments=args if isinstance(args, dict) else {"value": args},
                                                  id=slot["id"] or "call-%d" % (idx + 1)))
                        pending.clear()
                    yield ChatChunk(text=text, tool_calls=calls, done=bool(finish), usage=usage)
        except offline_guard.OfflineViolation:
            raise
        except httpx.HTTPError as exc:
            raise LLMUnavailable("Serveur compatible OpenAI injoignable sur %s (%s)." % (self.base_url, exc.__class__.__name__)) from exc
        yield ChatChunk(done=True)

    async def embed(self, texts: Sequence[str], model: str) -> List[List[float]]:
        if not texts:
            return []
        data = await self._request("POST", "/embeddings", {"model": model, "input": list(texts)}, timeout=300.0)
        items = sorted((data or {}).get("data", []), key=lambda d: d.get("index", 0))
        if len(items) != len(texts):
            raise LLMError("Le serveur a renvoyé %d embeddings pour %d textes" % (len(items), len(texts)))
        return [[float(x) for x in it.get("embedding", [])] for it in items]
