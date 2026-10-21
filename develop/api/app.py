"""Application FastAPI de Develop : API locale (``docs/API.md``) et interface web statique.

* ``create_app(settings_override=None, llm_transport=None)`` construit l'application ; le
  ``lifespan`` lance, sans bloquer, l'interrogation du modèle configuré (``model_info``) et ferme
  proprement la mémoire à l'arrêt ;
* ``/static`` sert ``develop/web/static``, ``/`` sert ``index.html`` (routage par ``#`` côté client) ;
* un middleware ASGI ajoute les en-têtes de sécurité (CSP, Referrer-Policy, nosniff) et refuse,
  pour les méthodes mutantes, toute requête dont ``Origin`` / ``Sec-Fetch-Site`` n'est pas de
  même origine ou locale (anti-CSRF ; les clients sans ces en-têtes — CLI, curl — passent) ;
* toutes les erreurs sont rendues ``{"error": {"code", "message"}}`` en français.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from develop import __version__, config
from develop.api.routes import all_routers, api_error
from develop.api.state import MODEL_INFO_TIMEOUT_S, AppState
from develop.config import GlobalSettings

logger = logging.getLogger("develop.api")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"
INDEX_HTML = WEB_DIR / "index.html"

# L'interface n'écrit jamais d'attribut ``style`` : elle assigne ``element.style`` (CSSOM), ce que la
# CSP n'interdit pas. ``'unsafe-inline'`` pour ``style-src`` reste autorisé par prudence (contrat).
CONTENT_SECURITY_POLICY = ("default-src 'self'; connect-src 'self'; img-src 'self' data:; "
                           "style-src 'self' 'unsafe-inline'")
SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}
MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})

DEFAULT_CODES: Dict[int, str] = {
    400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
    409: "conflict", 413: "payload_too_large", 415: "unsupported_media_type", 422: "validation_error",
    500: "internal_error", 501: "not_implemented", 503: "service_unavailable",
}
DEFAULT_MESSAGES: Dict[int, str] = {
    404: "Ressource introuvable.", 405: "Méthode non autorisée sur cette route.",
    500: "Erreur interne du serveur Develop.",
}


def error_payload(code: str, message: str) -> Dict[str, Any]:
    return {"error": {"code": code, "message": message}}


# --------------------------------------------------------------------------- middleware

def _host_only(value: str) -> str:
    """Hôte sans port ni crochets IPv6, en minuscules."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        return v[1:v.find("]")] if "]" in v else v
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def is_loopback_host(host: str) -> bool:
    h = _host_only(host)
    return h in LOOPBACK_HOSTS or h.endswith(".localhost")


def same_origin(headers: Headers) -> bool:
    """Vrai si la requête vient de la même origine (ou d'aucun navigateur)."""
    site = headers.get("sec-fetch-site")
    if site is not None and site.strip().lower() not in ("same-origin", "none"):
        return False
    origin = headers.get("origin")
    if origin is None:
        return True
    origin = origin.strip()
    if not origin or origin.lower() == "null":
        return False
    try:
        origin_host = _host_only(urlsplit(origin).netloc)
    except ValueError:
        return False
    if not origin_host:
        return False
    request_host = _host_only(headers.get("host", ""))
    return origin_host == request_host or is_loopback_host(origin_host)


class SecurityMiddleware:
    """En-têtes de sécurité sur chaque réponse + refus des requêtes mutantes d'une autre origine."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if scope.get("method", "GET").upper() in MUTATING_METHODS and not same_origin(headers):
            response = JSONResponse(status_code=403, content=error_payload(
                "same_origin_required",
                "Requête refusée : seules les requêtes de même origine (interface locale) peuvent modifier l'état."))
            for key, value in SECURITY_HEADERS.items():
                response.headers[key] = value
            await response(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                hdrs = MutableHeaders(scope=message)
                for key, value in SECURITY_HEADERS.items():
                    hdrs[key] = value
            await send(message)

        await self.app(scope, receive, send_with_headers)


# --------------------------------------------------------------------------- erreurs

def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc: StarletteHTTPException):  # type: ignore[no-untyped-def]
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            code = str(detail.get("code"))
            message = str(detail.get("message") or DEFAULT_MESSAGES.get(exc.status_code, ""))
        else:
            code = DEFAULT_CODES.get(exc.status_code, "http_%d" % exc.status_code)
            message = str(detail) if detail else DEFAULT_MESSAGES.get(exc.status_code, "Erreur HTTP %d." % exc.status_code)
            if exc.status_code == 404 and message == "Not Found":
                message = "Route ou ressource introuvable : %s" % request.url.path
        headers = getattr(exc, "headers", None) or None
        return JSONResponse(status_code=exc.status_code, content=error_payload(code, message), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc: RequestValidationError):  # type: ignore[no-untyped-def]
        parts = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            parts.append("%s : %s" % (loc or "corps", err.get("msg", "invalide")))
        message = "Requête invalide — " + " ; ".join(parts) if parts else "Requête invalide."
        return JSONResponse(status_code=422, content=error_payload("validation_error", message))

    @app.exception_handler(Exception)
    async def _unexpected(request, exc: Exception):  # type: ignore[no-untyped-def]
        logger.exception("erreur interne sur %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content=error_payload(
            "internal_error", "Erreur interne de Develop : %s (%s)." % (exc, exc.__class__.__name__)))


# --------------------------------------------------------------------------- application

def create_app(settings_override: Optional[GlobalSettings] = None,
               llm_transport: Optional[httpx.AsyncBaseTransport] = None) -> FastAPI:
    """Construit l'application. ``llm_transport`` (tests) remplace le réseau des clients LLM."""
    settings = settings_override if settings_override is not None else config.load_settings()
    state = AppState(settings, llm_transport=llm_transport)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        task = asyncio.create_task(state.refresh_model_info(MODEL_INFO_TIMEOUT_S))
        try:
            yield
        finally:
            if not task.done():
                task.cancel()
            await state.aclose()

    app = FastAPI(title="Develop", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)
    app.state.app_state = state
    app.add_middleware(SecurityMiddleware)
    register_error_handlers(app)
    for router in all_routers():
        app.include_router(router)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    def index() -> FileResponse:  # type: ignore[no-untyped-def]
        if not INDEX_HTML.is_file():
            raise api_error(404, "ui_missing", "L'interface web n'est pas installée (develop/web/index.html absent).")
        return FileResponse(str(INDEX_HTML), media_type="text/html")

    return app
