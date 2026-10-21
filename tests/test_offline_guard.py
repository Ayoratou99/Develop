from __future__ import annotations

import httpx
import pytest

from develop.llm import offline_guard as og


def _ok_transport():
    return httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True, "host": request.url.host}))


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "[::1]", "127.0.0.9", "app.localhost", "LOCALHOST"])
def test_strict_allows_loopback(host):
    g = og.OfflineGuard()
    assert g.is_allowed(host)


@pytest.mark.parametrize("host", ["api.openai.com", "192.168.1.10", "10.0.0.5", "8.8.8.8", "ollama.lan", "host.docker.internal", "", None])
def test_strict_blocks_everything_else(host):
    g = og.OfflineGuard()
    assert not g.is_allowed(host)


def test_local_network_mode_allows_private_ips_but_not_public_or_unresolved_names():
    g = og.OfflineGuard(mode=og.MODE_LOCAL_NETWORK)
    assert g.is_allowed("192.168.1.10")
    assert g.is_allowed("10.0.0.5")
    assert not g.is_allowed("8.8.8.8")
    assert not g.is_allowed("ollama.lan")  # un nom doit être autorisé explicitement
    g.configure(og.MODE_LOCAL_NETWORK, ["ollama.lan"])
    assert g.is_allowed("ollama.lan")
    assert g.is_allowed("host.docker.internal") and g.is_allowed("nas.local")


def test_allowed_hosts_are_ignored_in_strict_mode():
    g = og.OfflineGuard()
    g.configure(og.MODE_STRICT, ["ollama.lan"])
    assert not g.is_allowed("ollama.lan")


def test_client_blocks_before_sending_and_logs():
    g = og.OfflineGuard()
    sent = []

    def handler(request):
        sent.append(str(request.url))
        return httpx.Response(200)

    client = og.make_client(guard=g, inner=httpx.MockTransport(handler))
    with pytest.raises(og.OfflineViolation):
        client.get("https://api.openai.com/v1/models")
    assert sent == []
    log = g.blocked_attempts()
    assert len(log) == 1 and log[0].host == "api.openai.com" and log[0].mode == og.MODE_STRICT
    r = client.get("http://127.0.0.1:11434/api/tags")
    assert r.status_code == 200 and sent == ["http://127.0.0.1:11434/api/tags"]


async def test_async_client_blocks():
    g = og.OfflineGuard()
    client = og.make_async_client(guard=g, inner=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(og.OfflineViolation):
        await client.get("http://example.com/")
    r = await client.get("http://localhost:1234/v1/models")
    assert r.status_code == 200
    await client.aclose()


def test_configure_rejects_unknown_mode():
    with pytest.raises(ValueError):
        og.OfflineGuard().configure("cloud")
