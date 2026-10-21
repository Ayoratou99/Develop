from __future__ import annotations

import json

import httpx

from develop.system import engines


def _router(request: httpx.Request) -> httpx.Response:
    host, port, path = request.url.host, request.url.port, request.url.path
    if port == 11434:
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.12.0"})
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:7b", "size": 1, "details": {}}]})
    if port == 1234 and path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "lmstudio-model"}]})
    raise httpx.ConnectError("refused")


async def test_detect_engines_prefers_ollama_when_reachable():
    report = await engines.detect_engines(transport=httpx.MockTransport(_router), probe_docker=False)
    assert report.ollama.reachable and report.ollama.version == "0.12.0"
    assert report.recommended_engine == "ollama"
    lm = [p for p in report.openai_servers if p.base_url.startswith("http://127.0.0.1:1234")][0]
    assert lm.reachable and lm.models[0].name == "lmstudio-model"
    others = [p for p in report.openai_servers if not p.base_url.startswith("http://127.0.0.1:1234")]
    assert all(not p.reachable and p.error for p in others)
    json.dumps(report.to_dict())


async def test_detect_engines_falls_back_to_openai_compat_then_none():
    def only_lm(request):
        if request.url.port == 1234 and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        raise httpx.ConnectError("refused")
    report = await engines.detect_engines(transport=httpx.MockTransport(only_lm), probe_docker=False)
    assert report.recommended_engine == "openai_compat" and report.recommended_base_url == "http://127.0.0.1:1234/v1"

    def nothing(request):
        raise httpx.ConnectError("refused")
    report = await engines.detect_engines(transport=httpx.MockTransport(nothing), probe_docker=False)
    assert report.recommended_engine == "none" and not report.ollama.reachable


async def test_ollama_host_env_is_used(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11435")
    seen = []

    def router(request):
        seen.append(request.url.port)
        raise httpx.ConnectError("refused")
    report = await engines.detect_engines(transport=httpx.MockTransport(router), probe_docker=False)
    assert 11435 in seen and report.ollama.base_url == "http://127.0.0.1:11435"


def test_detect_docker_never_raises():
    d = engines.detect_docker()
    assert set(d) == {"available", "running", "ollama_container"}
