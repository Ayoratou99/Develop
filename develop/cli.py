"""Ligne de commande ``develop`` (argparse, sans dépendance supplémentaire).

Commandes : ``serve``, ``app``, ``init``, ``repos``, ``workspaces``, ``index``, ``search``, ``ask``,
``impact``, ``status``, ``doctor``. Toutes partagent l':class:`~develop.api.state.AppState` de l'API (même
configuration, même mémoire, même garde hors ligne). Sortie en français ; ``--json`` donne la forme
brute de l'API pour les scripts. ``main(argv=None) -> int``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from typing import Any, Callable, Dict, List, Optional, Sequence

from develop import __version__, config
from develop.config import RepoConfig

REPO_KINDS = ("frontend", "backend", "docs", "database", "mobile", "shared", "other")
PORT_ATTEMPTS = 10
BROWSER_WAIT_S = 30.0
APP_HOST = "127.0.0.1"  # la fenêtre native ne sert jamais ailleurs que la boucle locale
APP_WINDOW = (1280, 840)
APP_MIN_WINDOW = (900, 600)
SERVER_STOP_WAIT_S = 10.0


class CliError(Exception):
    """Erreur utilisateur affichée sans trace (code de sortie 1)."""


# --------------------------------------------------------------------------- aides

def _out(text: str = "") -> None:
    print(text)


def _json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _state(args: argparse.Namespace):  # type: ignore[no-untyped-def]
    from develop.api.state import AppState

    return AppState(workspace_id=getattr(args, "workspace", None))


def _require_ws(state):  # type: ignore[no-untyped-def]
    from develop.api.state import NoWorkspace

    try:
        return state.require_workspace()
    except NoWorkspace as exc:
        raise CliError(str(exc))


def _fmt_ts(value: Optional[float]) -> str:
    if not value:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(value)))


def _parse_repo_spec(spec: str) -> RepoConfig:
    """``kind=/chemin`` ou ``/chemin`` (type ``other``)."""
    kind, sep, path = spec.partition("=")
    if not sep:
        kind, path = "other", spec
    kind = kind.strip().lower()
    path = os.path.abspath(os.path.expanduser(path.strip()))
    if kind not in REPO_KINDS:
        raise CliError("Type de dépôt inconnu « %s » (attendus : %s)." % (kind, ", ".join(REPO_KINDS)))
    if not os.path.isdir(path):
        raise CliError("Le dossier « %s » n'existe pas." % path)
    return RepoConfig(name=os.path.basename(path.rstrip("/\\")) or path, path=path, kind=kind)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- serve

def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def pick_port(host: str, port: int, attempts: int = PORT_ATTEMPTS) -> int:
    """Le port demandé, sinon le premier libre parmi ``port+1 … port+attempts``."""
    if _port_free(host, port):
        return port
    for candidate in range(port + 1, port + attempts + 1):
        if _port_free(host, candidate):
            _out("Port %d occupé : utilisation du port %d." % (port, candidate))
            return candidate
    raise CliError("Aucun port libre entre %d et %d sur %s." % (port, port + attempts, host))


def _ephemeral_port(host: str) -> int:
    """Un port libre choisi par le système (quand toute la plage ``pick_port`` est occupée)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_server(url: str, timeout: float = BROWSER_WAIT_S,
                     give_up: Optional[Callable[[], bool]] = None) -> bool:
    """Vrai dès que ``GET /api/status`` répond 200 ; faux après ``timeout`` ou si ``give_up()`` devient vrai.

    Boucle locale uniquement : les variables de proxy de l'environnement sont ignorées.
    """
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not (give_up is not None and give_up()):
        try:
            with opener.open(url + "/api/status", timeout=1.0) as resp:  # noqa: S310 - boucle locale
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - serveur pas encore prêt
            pass
        time.sleep(0.25)
    return False


def _open_browser_when_ready(url: str, timeout: float = BROWSER_WAIT_S) -> bool:
    """Attend que le serveur réponde, puis ouvre le navigateur (depuis un thread)."""
    if not _wait_for_server(url, timeout):
        return False
    webbrowser.open(url)
    return True


def _activate_workspace(settings: config.GlobalSettings, workspace_id: Optional[str]) -> None:
    """``--workspace`` : vérifie l'espace et le rend actif (persisté dans settings.json)."""
    if not workspace_id:
        return
    if config.load_workspace(workspace_id) is None:
        raise CliError("Espace de travail « %s » introuvable." % workspace_id)
    settings.active_workspace = workspace_id
    config.save_settings(settings)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from develop.api import create_app

    settings = config.load_settings()
    _activate_workspace(settings, args.workspace)
    host = args.host or settings.host or "127.0.0.1"
    port = pick_port(host, int(args.port or settings.port or 8765))
    app = create_app(settings)
    url = "http://%s:%d" % (host, port)
    if not args.no_browser and settings.open_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True, name="develop-browser").start()
    print("Develop %s — interface : %s (Ctrl+C pour arrêter)" % (__version__, url), flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


# --------------------------------------------------------------------------- app (fenêtre native)

def _load_webview():  # type: ignore[no-untyped-def]
    """Le module ``webview`` (paquet pywebview, extra ``desktop``) ou ``None`` s'il n'est pas installé."""
    try:
        import webview  # type: ignore[import-not-found]
    except ImportError:
        return None
    return webview


def _start_server_thread(app, host: str, port: int):  # type: ignore[no-untyped-def]
    """Lance uvicorn dans un thread démon ; renvoie ``(server, thread)``.

    ``server.should_exit = True`` déclenche l'arrêt propre (lifespan compris). Boucle ``asyncio`` pure :
    le thread ne doit pas changer la politique d'événements du processus, dont le thread principal
    appartient à la fenêtre.
    """
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio",
                                           timeout_graceful_shutdown=int(SERVER_STOP_WAIT_S)))
    thread = threading.Thread(target=server.run, daemon=True, name="develop-server")
    thread.start()
    return server, thread


def _stop_server_thread(server, thread) -> None:  # type: ignore[no-untyped-def]
    server.should_exit = True
    thread.join(SERVER_STOP_WAIT_S + 5.0)


def cmd_app(args: argparse.Namespace) -> int:
    """Serveur sur la boucle locale dans un thread, interface dans une fenêtre native (pywebview).

    Sans pywebview : message explicite, puis repli sur ``develop serve`` (navigateur). Fermer la
    fenêtre arrête le serveur ; si la fenêtre ne peut pas s'ouvrir (GTK/Qt/WebView2 absents), le
    serveur déjà lancé reste disponible dans le navigateur jusqu'à Ctrl+C.
    """
    webview = _load_webview()
    if webview is None:
        _out("La fenêtre Develop nécessite pywebview, qui n'est pas installé :")
        _out("  pip install \"develop-local[desktop]\"   (ou : sh install.sh --desktop)")
        _out("Repli : ouverture dans le navigateur, comme « develop serve ».")
        return cmd_serve(argparse.Namespace(host=APP_HOST, port=args.port, no_browser=False,
                                            workspace=args.workspace))

    from develop.api import create_app

    settings = config.load_settings()
    _activate_workspace(settings, args.workspace)
    try:
        port = pick_port(APP_HOST, int(args.port or settings.port or 8765))
    except CliError:
        port = _ephemeral_port(APP_HOST)
    url = "http://%s:%d" % (APP_HOST, port)
    server, thread = _start_server_thread(create_app(settings), APP_HOST, port)
    try:
        if not _wait_for_server(url, BROWSER_WAIT_S, give_up=lambda: not thread.is_alive()):
            raise CliError("Le serveur Develop ne répond pas sur %s." % url)
        print("Develop %s — fenêtre ouverte sur %s (fermer la fenêtre arrête le serveur)" % (__version__, url),
              flush=True)
        storage = config.develop_home() / "webview"
        storage.mkdir(parents=True, exist_ok=True)
        try:
            webview.create_window("Develop", url, width=APP_WINDOW[0], height=APP_WINDOW[1],
                                  min_size=APP_MIN_WINDOW, text_select=True)
            webview.start(private_mode=False, storage_path=str(storage))
        except KeyboardInterrupt:
            pass
        except Exception as exc:  # noqa: BLE001 - moteur de fenêtre absent (GTK, Qt, WebView2…)
            print("Impossible d'ouvrir la fenêtre (%s : %s)." % (exc.__class__.__name__, exc), file=sys.stderr)
            print("Repli : interface dans le navigateur sur %s (Ctrl+C pour arrêter)." % url, flush=True)
            webbrowser.open(url)
            try:
                while thread.is_alive():
                    thread.join(0.5)
            except KeyboardInterrupt:
                pass
    finally:
        _stop_server_thread(server, thread)
    _out("Fenêtre fermée : serveur arrêté.")
    return 0


# --------------------------------------------------------------------------- espaces et dépôts

def cmd_init(args: argparse.Namespace) -> int:
    repos = [_parse_repo_spec(s) for s in (args.repo or [])]
    ws = config.create_workspace(args.name)
    ws.repos = repos
    config.save_workspace(ws)
    settings = config.load_settings()
    settings.active_workspace = ws.id
    config.save_settings(settings)
    if args.json:
        _json(ws.model_dump())
    else:
        _out("Espace de travail « %s » créé (identifiant %s) et activé." % (ws.name, ws.id))
        for r in repos:
            _out("  dépôt %-10s %s  [%s]" % (r.kind, r.path, r.id))
        if not repos:
            _out("  Ajoutez des dépôts : develop repos add /chemin --kind backend")
    return 0


def cmd_workspaces(args: argparse.Namespace) -> int:
    settings = config.load_settings()
    if args.action == "list":
        items = config.list_workspaces()
        if args.json:
            _json({"active_workspace": settings.active_workspace, "workspaces": [w.model_dump() for w in items]})
            return 0
        if not items:
            _out("Aucun espace de travail. Créez-en un : develop init NOM --repo backend=/chemin")
            return 0
        for w in items:
            flag = "*" if w.id == settings.active_workspace else " "
            _out("%s %-24s %-30s %d dépôt(s)  modifié %s" % (flag, w.id, w.name, len(w.repos), _fmt_ts(w.updated_at)))
        return 0
    if args.action == "use":
        if config.load_workspace(args.id) is None:
            raise CliError("Espace « %s » introuvable." % args.id)
        settings.active_workspace = args.id
        config.save_settings(settings)
        _out("Espace « %s » activé." % args.id)
        return 0
    if args.action == "rm":
        if not config.delete_workspace(args.id):
            raise CliError("Espace « %s » introuvable." % args.id)
        _out("Mémoire Develop de l'espace « %s » supprimée (vos dépôts sont intacts)." % args.id)
        return 0
    raise CliError("Action inconnue : %s" % args.action)


def cmd_repos(args: argparse.Namespace) -> int:
    state = _state(args)
    ws = _require_ws(state)
    if args.action == "list":
        if args.json:
            _json([r.model_dump() for r in ws.repos])
            return 0
        if not ws.repos:
            _out("Aucun dépôt dans l'espace « %s »." % ws.name)
        for r in ws.repos:
            _out("%-10s %-10s %-24s %s%s" % (r.id, r.kind, r.name, r.path, "" if r.enabled else "  (désactivé)"))
        return 0
    if args.action == "add":
        repo = _parse_repo_spec("%s=%s" % (args.kind, args.path))
        if args.name:
            repo.name = args.name
        if any(os.path.abspath(r.path) == repo.path for r in ws.repos):
            raise CliError("Ce dossier est déjà un dépôt de l'espace.")
        ws.repos.append(repo)
        config.save_workspace(ws)
        _out("Dépôt « %s » ajouté (%s, identifiant %s). Lancez « develop index »." % (repo.name, repo.kind, repo.id))
        return 0
    if args.action == "rm":
        before = len(ws.repos)
        ws.repos = [r for r in ws.repos if r.id != args.id]
        if len(ws.repos) == before:
            raise CliError("Dépôt « %s » introuvable." % args.id)
        config.save_workspace(ws)
        _out("Dépôt « %s » retiré de la configuration (son index sera purgé à la prochaine indexation)." % args.id)
        return 0
    raise CliError("Action inconnue : %s" % args.action)


# --------------------------------------------------------------------------- index

def cmd_index(args: argparse.Namespace) -> int:
    from develop.indexing import Indexer, IndexProgress

    state = _state(args)
    ws = _require_ws(state)
    if args.repo and ws.repo_by_id(args.repo) is None:
        raise CliError("Dépôt « %s » inconnu dans l'espace « %s »." % (args.repo, ws.name))
    last_line = [""]

    def on_progress(p: IndexProgress) -> None:
        parts = [p.phase]
        if p.repo:
            parts.append("dépôt %s" % p.repo)
        if p.total:
            parts.append("%d/%d" % (p.current, p.total))
        if p.message:
            parts.append(str(p.message))
        line = " — ".join(parts)
        if line != last_line[0] and not args.quiet and not args.json:
            _out("  " + line)
            last_line[0] = line

    embedding_model = ws.llm.embedding_model if (ws.llm.engine != "none" and not args.no_embeddings) else ""
    if not args.json:
        _out("Indexation %s de l'espace « %s » (%d dépôt(s))…" % ("complète" if args.full else "incrémentale", ws.name,
                                                                  len(ws.repos)))
    indexer = Indexer(state.db, ws, state.llm, embedding_model, progress=on_progress)
    row = indexer.run(full=args.full, repo_id=args.repo, with_embeddings=not args.no_embeddings)
    stats = row.stats or {}
    if args.json:
        _json({"status": row.status, "started_at": row.started_at, "finished_at": row.finished_at, "stats": stats,
               "error": row.error})
    else:
        labels = {"ok": "terminée", "partial": "terminée avec des erreurs", "cancelled": "annulée", "failed": "échouée"}
        _out("Indexation %s en %.1f s : %d fichier(s) parcourus, %d indexés, %d inchangés, %d supprimés ; "
             "%d symboles, %d endpoints, %d modèles, %d relations, %d documents, %d notes, %d embeddings."
             % (labels.get(row.status, row.status), float(stats.get("duration_s", 0.0)), stats.get("files_scanned", 0),
                stats.get("files_indexed", 0), stats.get("files_unchanged", 0), stats.get("files_deleted", 0),
                stats.get("symbols", 0), stats.get("endpoints", 0), stats.get("models", 0), stats.get("relations", 0),
                stats.get("documents", 0), stats.get("notes", 0), stats.get("embeddings", 0)))
        for err in list(stats.get("errors") or [])[:10]:
            _out("  erreur : %s (%s) : %s" % (err.get("path") or "?", err.get("stage"), err.get("message")))
        for w in list(stats.get("warnings") or [])[:10]:
            _out("  avertissement : %s" % w)
        if row.error:
            _out("  %s" % row.error)
    return 0 if row.status in ("ok", "partial") else 1


# --------------------------------------------------------------------------- recherche et chat

def cmd_search(args: argparse.Namespace) -> int:
    from develop.api.routes.search import search_result_dict

    state = _state(args)
    _require_ws(state)
    types = [t.strip() for t in args.types.split(",")] if args.types else None
    result = asyncio.run(state.search_engine.search_async(args.query, limit=args.limit, source_types=types))
    data = search_result_dict(result)
    if args.json:
        _json(data)
        return 0
    if data["identifiers"]:
        _out("Identifiants repérés : %s" % ", ".join(data["identifiers"]))
    if not data["results"]:
        _out("Aucun résultat. L'index est-il construit ? (develop index)")
        return 0
    for i, hit in enumerate(data["results"], 1):
        where = hit.get("path") or hit.get("title") or "?"
        if hit.get("line_start"):
            where += ":%s" % hit["line_start"]
            if hit.get("line_end") and hit["line_end"] != hit["line_start"]:
                where += "-%s" % hit["line_end"]
        elif hit.get("page"):
            where += " (p. %s)" % hit["page"]
        _out("%2d. [%s] %s — %s" % (i, hit.get("source_type"), hit.get("title") or where, where))
        if hit.get("reason") or hit.get("channels"):
            _out("    %s%s" % (", ".join(hit.get("channels") or []), (" · " + hit["reason"]) if hit.get("reason") else ""))
        if hit.get("snippet"):
            _out("    %s" % hit["snippet"][:200])
    for w in data.get("warnings") or []:
        _out("avertissement : %s" % w)
    return 0


def _source_line(src: Dict[str, Any]) -> str:
    where = src.get("path") or src.get("title") or "?"
    if src.get("line_start"):
        where += ":%s" % src["line_start"]
        if src.get("line_end") and src["line_end"] != src["line_start"]:
            where += "-%s" % src["line_end"]
    elif src.get("page"):
        where += " (p. %s)" % src["page"]
    reason = src.get("reason") or ""
    return "[%s] %s %s%s" % (src.get("ref"), src.get("source_type"), where, (" — " + reason) if reason else "")


def cmd_ask(args: argparse.Namespace) -> int:
    state = _state(args)
    _require_ws(state)
    mode = "search_only" if args.search_only else ("no_tools" if args.no_tools else "auto")
    orchestrator = state.orchestrator
    assert orchestrator is not None and state.db is not None

    async def run() -> Dict[str, Any]:
        await state.refresh_model_info()
        conv_id = int(args.conversation) if args.conversation else state.db.create_conversation(args.question[:60])
        collected: Dict[str, Any] = {"conversation_id": conv_id, "answer": "", "sources": [], "events": [],
                                     "done": None, "error": None}
        stream = orchestrator.respond(conv_id, args.question, mode, args.max_context_tokens)
        try:
            async for ev in stream:
                collected["events"].append({"type": ev.type, "data": ev.data})
                if ev.type == "status" and not args.json:
                    print("… %s" % ev.data.get("text", ""), file=sys.stderr)
                elif ev.type == "sources":
                    collected["sources"] = list(ev.data.get("items") or [])
                elif ev.type == "tool_call" and not args.json:
                    print("… outil %s %s" % (ev.data.get("name"), json.dumps(ev.data.get("arguments", {}), ensure_ascii=False)),
                          file=sys.stderr)
                elif ev.type == "tool_result":
                    for s in ev.data.get("added_sources") or []:
                        if not any(x.get("ref") == s.get("ref") for x in collected["sources"]):
                            collected["sources"].append(s)
                elif ev.type == "delta":
                    collected["answer"] += ev.data.get("text", "")
                    if not args.json:
                        sys.stdout.write(ev.data.get("text", ""))
                        sys.stdout.flush()
                elif ev.type == "done":
                    collected["done"] = ev.data
                elif ev.type == "error":
                    collected["error"] = ev.data
        finally:
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                await aclose()
        return collected

    result = asyncio.run(run())
    if args.json:
        _json(result)
    else:
        if result["answer"]:
            _out()
        if result["sources"]:
            _out("\nSources :")
            for s in result["sources"]:
                _out("  " + _source_line(s))
        elif mode == "search_only":
            _out("Aucune source trouvée.")
        if result["error"]:
            _out("Erreur (%s) : %s" % (result["error"].get("code"), result["error"].get("message")))
        done = result.get("done") or {}
        budget = done.get("budget") or {}
        if budget:
            _out("\nContexte : %s / %s tokens (num_ctx %s)" % (budget.get("context_tokens_used"),
                                                             budget.get("max_context_tokens"), budget.get("num_ctx")))
    return 2 if result["error"] else 0


# --------------------------------------------------------------------------- impact

def cmd_impact(args: argparse.Namespace) -> int:
    from fastapi import HTTPException

    from develop.api.routes.impact import compute_impact

    state = _state(args)
    _require_ws(state)
    if args.file:
        source = "file"
    elif args.symbol:
        source = "symbol"
    else:
        source = "git"
    try:
        data = compute_impact(state, source, repo_id=args.repo, rev_range=args.range, path=args.file,
                              symbol_id=args.symbol, max_hops=args.max_hops)
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": str(exc.detail)}
        raise CliError(str(detail.get("message")))
    if args.json:
        _json(data)
        return 0
    changed = data.get("changed") or []
    _out("Changements analysés : %d" % len(changed))
    for c in changed[:30]:
        syms = c.get("symbols_changed") or []
        names = ", ".join(s if isinstance(s, str) else str(s.get("name")) for s in syms)
        _out("  %-9s %s%s" % (c.get("status") or "modifié", c.get("path"), (" : " + names) if names else ""))
    for w in data.get("warnings") or []:
        _out("avertissement : %s" % w)
    groups = (("endpoints", "Endpoints"), ("models", "Modèles"), ("frontend_calls", "Appels frontend"),
              ("documents", "Documents"), ("notes", "Notes"))
    shown = set()
    for key, label in groups:
        items = data.get(key) or []
        if not items:
            continue
        _out("\n%s (%d) :" % (label, len(items)))
        for it in items:
            shown.add(it.get("node"))
            _out("  %3d %%  %s%s — %s" % (round(float(it.get("confidence", 0)) * 100), it.get("name") or it.get("path"),
                                           (" (%s)" % it.get("path")) if it.get("path") and it.get("name") != it.get("path") else "",
                                           it.get("reason") or ""))
    rest = [it for it in (data.get("affected") or []) if it.get("node") not in shown]
    if rest:
        _out("\nAutres éléments affectés (%d) :" % len(rest))
        for it in rest:
            _out("  %3d %%  [%s] %s — %s" % (round(float(it.get("confidence", 0)) * 100), it.get("type"),
                                              it.get("path") or it.get("name"), it.get("reason") or ""))
    if not rest and not any(data.get(k) for k, _ in groups):
        _out("Aucun élément affecté d'après l'index actuel.")
    if data.get("stale_index"):
        _out("\nAttention : l'index n'est plus à jour (%d fichier(s) modifiés) — relancez « develop index »."
             % len(data.get("stale_files") or []))
    return 0


# --------------------------------------------------------------------------- status et doctor

def verify_offline(guard) -> Dict[str, Any]:  # type: ignore[no-untyped-def]
    """Tentative volontaire vers http://example.com à travers la garde : DOIT être bloquée et journalisée."""
    from develop.llm.offline_guard import OfflineViolation, make_client

    before = len(guard.blocked_attempts())
    blocked = False
    error = ""
    client = make_client(guard, timeout=3.0)
    try:
        client.get("http://example.com/")
    except OfflineViolation:
        blocked = True
    except Exception as exc:  # noqa: BLE001 - une erreur réseau signifie que la requête est SORTIE
        error = "%s : %s" % (exc.__class__.__name__, exc)
    finally:
        client.close()
    logged = len(guard.blocked_attempts()) > before and guard.blocked_attempts()[-1].host == "example.com"
    return {"blocked": blocked, "logged": logged, "ok": blocked and logged, "error": error, "mode": guard.mode}


def cmd_status(args: argparse.Namespace) -> int:
    state = _state(args)
    verification = verify_offline(state.guard) if args.verify_offline else None  # avant l'état : compteur à jour
    data = asyncio.run(state.status_dict())
    if verification is not None:
        data["offline_verification"] = verification
    if args.json:
        _json(data)
    else:
        _out("Develop %s — dossier %s" % (data["version"], config.develop_home()))
        ws = data.get("workspace")
        if not ws:
            _out("Aucun espace de travail : lancez « develop init NOM --repo backend=/chemin » ou « develop serve ».")
        else:
            _out("Espace actif : %s (%s), %d dépôt(s)" % (ws["name"], ws["id"], len(ws.get("repos") or [])))
            for r in ws.get("repos") or []:
                _out("  %-10s %-24s %s" % (r.get("kind"), r.get("name"), r.get("path")))
        engine = data["engine"]
        if engine.get("engine") in (None, "", "none"):
            _out("Moteur d'IA : aucun (recherche seule) — voir « develop doctor »")
        else:
            _out("Moteur d'IA : %s %s — %s" % (engine.get("engine"), engine.get("model") or "",
                                               "opérationnel" if engine.get("ok") else "indisponible : %s" % engine.get("error")))
        index = data["index"]
        counts = index.get("counts") or {}
        last = index.get("last_run") or {}
        _out("Index : %s%s — %s fichiers, %s symboles, %s endpoints, %s modèles, %s relations, %s documents, %s notes"
             % ("en cours (%s)" % index.get("phase") if index.get("running") else "au repos",
                (", dernier run %s le %s" % (last.get("status"), _fmt_ts(last.get("finished_at")))) if last else ", jamais indexé",
                counts.get("files", 0), counts.get("symbols", 0), counts.get("endpoints", 0), counts.get("models", 0),
                counts.get("relations", 0), counts.get("documents", 0), counts.get("notes", 0)))
        offline = data["offline"]
        _out("Mode réseau : %s (%d requête(s) bloquée(s))" % (
            "hors ligne strict" if offline.get("mode") == "strict" else "réseau local (%s)" % ", ".join(offline.get("allowed_hosts") or []) or "aucun hôte",
            offline.get("blocked_count", 0)))
        if data.get("needs_setup"):
            _out("Configuration requise : ouvrez l'interface (develop serve) ou ajoutez des dépôts.")
        if verification is not None:
            if verification["ok"]:
                _out("Vérification hors ligne : requête vers example.com bloquée avant tout accès réseau et journalisée.")
            else:
                _out("ÉCHEC de la vérification hors ligne : bloquée=%s journalisée=%s %s"
                     % (verification["blocked"], verification["logged"], verification["error"]))
    if verification is not None and not verification["ok"]:
        return 1
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from develop.llm import advisor, offline_guard
    from develop.system.engines import detect_engines
    from develop.system.profile import describe, detect_machine

    ws = config.load_workspace(config.load_settings().active_workspace or "") if config.load_settings().active_workspace else None
    if ws is not None:
        try:
            offline_guard.GUARD.configure(ws.mode, ws.allowed_hosts)
        except ValueError:
            pass
    profile = detect_machine()
    engines = asyncio.run(detect_engines(guard=offline_guard.GUARD))
    installed: List[Any] = list(engines.ollama.models)
    for probe in engines.openai_servers:
        installed.extend(probe.models)
    num_ctx = ws.llm.num_ctx if ws is not None and ws.llm.num_ctx else 16384
    advice = advisor.advise(profile, engines, installed_models=installed, num_ctx=num_ctx)
    if args.json:
        _json({"profile": dict(profile.to_dict(), summary=describe(profile)), "engines": engines.to_dict(),
               "advice": advice.to_dict(), "workspace": ws.model_dump() if ws is not None else None})
        return 0
    _out("Develop %s — diagnostic" % __version__)
    _out("")
    _out(advisor.format_advice_text(advice))
    for w in profile.warnings:
        _out("avertissement : %s" % w)
    if ws is not None:
        _out("")
        _out("Espace actif : %s — moteur %s %s" % (ws.name, ws.llm.engine, ws.llm.model or ""))
    return 0


# --------------------------------------------------------------------------- analyseur

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="develop",
                                     description="Develop : mémoire locale et assistant hors ligne d'un projet logiciel.")
    parser.add_argument("--version", action="version", version="develop %s" % __version__)
    sub = parser.add_subparsers(dest="command", metavar="commande")

    p = sub.add_parser("serve", help="démarre l'API et l'interface web (127.0.0.1:8765)")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-browser", action="store_true", help="n'ouvre pas le navigateur")
    p.add_argument("--workspace", default=None, help="identifiant de l'espace à activer")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("app", help="ouvre l'interface dans une fenêtre native (pywebview, extra « desktop »)")
    p.add_argument("--port", type=int, default=None, help="port du serveur local (sinon celui des réglages)")
    p.add_argument("--workspace", default=None, help="identifiant de l'espace à activer")
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("init", help="crée un espace de travail et l'active")
    p.add_argument("name", help="nom de l'espace")
    p.add_argument("--repo", action="append", metavar="KIND=/chemin", help="dépôt à ajouter (répétable)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("workspaces", help="liste, active ou supprime des espaces de travail")
    ws_sub = p.add_subparsers(dest="action", metavar="action")
    q = ws_sub.add_parser("list")
    q.add_argument("--json", action="store_true")
    q = ws_sub.add_parser("use")
    q.add_argument("id")
    q = ws_sub.add_parser("rm")
    q.add_argument("id")
    p.set_defaults(func=cmd_workspaces)

    p = sub.add_parser("repos", help="dépôts de l'espace actif")
    p.add_argument("--workspace", default=None)
    r_sub = p.add_subparsers(dest="action", metavar="action")
    q = r_sub.add_parser("list")
    q.add_argument("--json", action="store_true")
    q = r_sub.add_parser("add")
    q.add_argument("path")
    q.add_argument("--kind", default="other", choices=REPO_KINDS)
    q.add_argument("--name", default=None)
    q = r_sub.add_parser("rm")
    q.add_argument("id")
    p.set_defaults(func=cmd_repos)

    p = sub.add_parser("index", help="indexe les dépôts, documents et notes (incrémental)")
    p.add_argument("--full", action="store_true", help="réindexation complète")
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--repo", default=None, help="n'indexer que ce dépôt (identifiant)")
    p.add_argument("--workspace", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="recherche hybride dans la mémoire du projet")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--types", default=None, help="file,document,note,symbol,endpoint,model")
    p.add_argument("--workspace", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="pose une question au chat (réponse en flux, sources citées)")
    p.add_argument("question")
    p.add_argument("--search-only", action="store_true", help="sources seulement, sans appeler le modèle")
    p.add_argument("--no-tools", action="store_true", help="une seule passe, sans outils")
    p.add_argument("--conversation", default=None, help="identifiant d'une conversation existante")
    p.add_argument("--max-context-tokens", type=int, default=None)
    p.add_argument("--workspace", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("impact", help="analyse d'impact (Git, fichier ou symbole)")
    p.add_argument("--range", default=None, help="plage de commits A..B (sinon working tree)")
    p.add_argument("--file", default=None, help="chemin relatif au dépôt")
    p.add_argument("--symbol", default=None, help="identifiant du symbole")
    p.add_argument("--repo", default=None)
    p.add_argument("--max-hops", type=int, default=3)
    p.add_argument("--workspace", default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_impact)

    p = sub.add_parser("status", help="état : espace, moteur, index, mode réseau")
    p.add_argument("--json", action="store_true")
    p.add_argument("--verify-offline", action="store_true", help="vérifie que la garde bloque une requête sortante")
    p.add_argument("--workspace", default=None)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("doctor", help="profil machine, moteurs détectés et modèles recommandés")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    if getattr(args, "command", None) in ("workspaces", "repos") and not getattr(args, "action", None):
        parser.parse_args([args.command, "--help"])
        return 1
    try:
        return int(args.func(args) or 0)
    except CliError as exc:
        print("Erreur : %s" % exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrompu.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
