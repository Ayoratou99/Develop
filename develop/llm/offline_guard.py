"""Garde hors ligne : aucune requête HTTP ne peut quitter la machine en mode strict.

Principe : *tout* client HTTP de Develop passe par :func:`make_client`, qui installe un
transport httpx enveloppé. Avant chaque requête, l'hôte est vérifié ; s'il n'est pas
local (ou explicitement autorisé en mode « local avec services réseau internes »), la
requête est refusée **avant** toute résolution DNS ou connexion, et le blocage est
journalisé pour être affiché dans l'interface.
"""
from __future__ import annotations

import ipaddress
import threading
import time
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Set
from collections import deque

import httpx

MODE_STRICT = "strict"  # aucun hôte non loopback
MODE_LOCAL_NETWORK = "local_network"  # loopback + réseaux privés + hôtes autorisés

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}
_LOCAL_NETWORK_NAMES = {"host.docker.internal", "gateway.docker.internal"}


class OfflineViolation(httpx.TransportError):
    """Levée quand une requête tente de joindre un hôte interdit."""

    def __init__(self, host: str, mode: str, url: str):
        self.host = host
        self.mode = mode
        self.url = url
        super().__init__(
            "Requête bloquée par le mode hors ligne (%s) : %s n'est pas un hôte local autorisé." % (mode, host)
        )


@dataclass
class BlockedAttempt:
    timestamp: float
    host: str
    url: str
    mode: str


@dataclass
class OfflineGuard:
    """Politique réseau partagée par tous les clients HTTP de Develop."""

    mode: str = MODE_STRICT
    allowed_hosts: Set[str] = field(default_factory=set)
    max_log: int = 200
    _log: Deque[BlockedAttempt] = field(default_factory=lambda: deque(maxlen=200), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ----- politique -----
    def is_allowed(self, host: Optional[str]) -> bool:
        if not host:
            return False
        h = host.strip().lower().rstrip(".")
        if h.startswith("[") and h.endswith("]"):
            h = h[1:-1]
        if h in _LOOPBACK_NAMES or h.endswith(".localhost"):
            return True
        if self.mode == MODE_LOCAL_NETWORK and (h in self.allowed_hosts or h in _LOCAL_NETWORK_NAMES or h.endswith(".local")):
            return True
        try:
            ip = ipaddress.ip_address(h)
        except ValueError:
            # Nom d'hôte non loopback : jamais en strict ; en réseau local seulement s'il est
            # explicitement autorisé (évite toute résolution DNS vers l'extérieur).
            return False
        if ip.is_loopback:
            return True
        if self.mode == MODE_LOCAL_NETWORK:
            return bool(ip.is_private or ip.is_link_local)
        return False

    def check(self, url: str) -> None:
        """Lève :class:`OfflineViolation` si l'URL n'est pas autorisée, et journalise."""
        try:
            parsed = httpx.URL(url)
            host = parsed.host
        except Exception:  # noqa: BLE001 - URL invalide = refusée
            host = ""
        if self.is_allowed(host):
            return
        with self._lock:
            self._log.append(BlockedAttempt(time.time(), host or "?", str(url), self.mode))
        raise OfflineViolation(host or "?", self.mode, str(url))

    def blocked_attempts(self) -> List[BlockedAttempt]:
        with self._lock:
            return list(self._log)

    def clear_log(self) -> None:
        with self._lock:
            self._log.clear()

    def configure(self, mode: str, allowed_hosts: Optional[Iterable[str]] = None) -> None:
        if mode not in (MODE_STRICT, MODE_LOCAL_NETWORK):
            raise ValueError("mode inconnu: %s" % mode)
        self.mode = mode
        self.allowed_hosts = {h.strip().lower() for h in (allowed_hosts or []) if h and h.strip()}


class GuardedTransport(httpx.BaseTransport):
    """Transport synchrone qui vérifie l'hôte avant de déléguer."""

    def __init__(self, guard: OfflineGuard, inner: Optional[httpx.BaseTransport] = None):
        self.guard = guard
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.guard.check(str(request.url))
        return self.inner.handle_request(request)

    def close(self) -> None:
        self.inner.close()


class AsyncGuardedTransport(httpx.AsyncBaseTransport):
    """Transport asynchrone qui vérifie l'hôte avant de déléguer."""

    def __init__(self, guard: OfflineGuard, inner: Optional[httpx.AsyncBaseTransport] = None):
        self.guard = guard
        self.inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.guard.check(str(request.url))
        return await self.inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self.inner.aclose()


# Garde globale par défaut (strict). L'application la reconfigure d'après la configuration.
GUARD = OfflineGuard()


def make_client(guard: Optional[OfflineGuard] = None, inner: Optional[httpx.BaseTransport] = None,
                **kwargs) -> httpx.Client:
    """Client httpx synchrone protégé. ``inner`` permet d'injecter un MockTransport en test."""
    kwargs.setdefault("timeout", httpx.Timeout(30.0, connect=3.0))
    kwargs.setdefault("trust_env", False)  # ignore HTTP_PROXY & co : pas de proxy sortant
    return httpx.Client(transport=GuardedTransport(guard or GUARD, inner), **kwargs)


def make_async_client(guard: Optional[OfflineGuard] = None, inner: Optional[httpx.AsyncBaseTransport] = None,
                      **kwargs) -> httpx.AsyncClient:
    """Client httpx asynchrone protégé (utilisé pour le streaming du chat)."""
    kwargs.setdefault("timeout", httpx.Timeout(120.0, connect=3.0))
    kwargs.setdefault("trust_env", False)
    return httpx.AsyncClient(transport=AsyncGuardedTransport(guard or GUARD, inner), **kwargs)
