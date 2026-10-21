"""État partagé de l'application Develop : espace actif, mémoire, moteur LLM, indexation.

Un seul :class:`AppState` par processus (API ou CLI). Il possède :

* la configuration globale et l'espace de travail actif (``None`` → assistant de configuration) ;
* la :class:`~develop.storage.Database` de l'espace, ouverte sur ``config.db_path`` ;
* la garde hors ligne globale (:data:`develop.llm.offline_guard.GUARD`), reconfigurée selon
  ``workspace.mode`` / ``allowed_hosts`` ;
* le client LLM, le moteur de recherche, l'analyseur d'impact et l'orchestrateur du chat ;
* des caches courts : profil machine (permanent), moteurs détectés (30 s), santé du moteur (15 s),
  ``model_info`` (rafraîchi au démarrage sans bloquer, et par ``POST /api/llm/test``) ;
* l'indexation en arrière-plan : un seul thread (``ThreadPoolExecutor(1)``), un événement
  d'annulation, le dernier :class:`~develop.indexing.IndexProgress` et des files asyncio vers
  lesquelles les événements SSE sont publiés depuis le thread de travail.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import httpx

from develop import __version__, config
from develop.chat import ChatOrchestrator
from develop.config import GlobalSettings, WorkspaceConfig
from develop.impact_analysis import ImpactAnalyzer
from develop.indexing import Indexer, IndexProgress
from develop.llm import factory, offline_guard
from develop.llm.base import EngineStatus, LLMClient, ModelInfo, NullClient
from develop.search import SearchEngine
from develop.storage import Database
from develop.system.engines import EnginesReport, detect_engines
from develop.system.profile import MachineProfile, detect_machine

logger = logging.getLogger("develop.api")

ENGINES_CACHE_S = 30.0
HEALTH_CACHE_S = 15.0
HEALTH_TIMEOUT_S = 5.0
MODEL_INFO_TIMEOUT_S = 3.0


class IndexRunning(RuntimeError):
    """Une indexation est déjà en cours (→ 409 ``index_running``)."""


class NoWorkspace(RuntimeError):
    """Aucun espace de travail actif (→ 400 ``no_workspace``)."""


class AppState:
    """Objets partagés par les routes de l'API et par les commandes de la CLI."""

    def __init__(self, settings: Optional[GlobalSettings] = None,
                 llm_transport: Optional[httpx.AsyncBaseTransport] = None,
                 workspace_id: Optional[str] = None):
        self.settings: GlobalSettings = settings if settings is not None else config.load_settings()
        self.llm_transport = llm_transport
        self.workspace: Optional[WorkspaceConfig] = None
        self.db: Optional[Database] = None
        self.guard: offline_guard.OfflineGuard = offline_guard.GUARD
        self.llm: LLMClient = NullClient()
        self.model_info: Optional[ModelInfo] = None
        self.search_engine: Optional[SearchEngine] = None
        self.indexer: Optional[Indexer] = None
        self.impact: Optional[ImpactAnalyzer] = None
        self.orchestrator: Optional[ChatOrchestrator] = None
        self._profile: Optional[MachineProfile] = None
        self._engines: Optional[EnginesReport] = None
        self._engines_at = 0.0
        self._health: Optional[Dict[str, Any]] = None
        self._health_at = 0.0
        # indexation en arrière-plan
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="develop-index")
        self._index_lock = threading.Lock()
        self.index_cancel = threading.Event()
        self.index_progress: Optional[IndexProgress] = None
        self.index_run_id: Optional[str] = None
        self.index_running = False
        self.index_started_at: Optional[float] = None
        self._index_future: Optional[Future] = None
        self._subscribers: List[Tuple[asyncio.AbstractEventLoop, "asyncio.Queue[Any]"]] = []
        self._sub_lock = threading.Lock()
        self.load_workspace(workspace_id or self.settings.active_workspace)

    # ------------------------------------------------------------------ espace de travail
    @property
    def needs_setup(self) -> bool:
        """Vrai sans espace actif ou sans dépôt activé : l'interface ouvre l'assistant."""
        ws = self.workspace
        return ws is None or not any(r.enabled for r in ws.repos)

    def load_workspace(self, workspace_id: Optional[str]) -> Optional[WorkspaceConfig]:
        """Charge ``workspace_id`` (ou, à défaut, le premier espace existant) et construit les services."""
        ws = config.load_workspace(workspace_id) if workspace_id else None
        if ws is None:
            existing = config.list_workspaces()
            ws = existing[0] if existing else None
            if ws is not None and self.settings.active_workspace != ws.id:
                self.settings.active_workspace = ws.id
                try:
                    config.save_settings(self.settings)
                except OSError:
                    pass
        if ws is None:
            self.detach()
            return None
        self._build(ws)
        return ws

    def activate_workspace(self, workspace_id: str) -> WorkspaceConfig:
        """Rend ``workspace_id`` actif (réglages globaux + reconstruction complète)."""
        ws = config.load_workspace(workspace_id)
        if ws is None:
            raise KeyError(workspace_id)
        self.settings.active_workspace = ws.id
        config.save_settings(self.settings)
        self._build(ws)
        return ws

    def detach(self) -> None:
        """Ferme la mémoire de l'espace courant ; l'état passe en « configuration requise »."""
        if self.db is not None:
            try:
                self.db.close()
            except Exception:  # noqa: BLE001
                pass
        self.db = None
        self.workspace = None
        self.llm = NullClient()
        self.model_info = None
        self.search_engine = None
        self.indexer = None
        self.impact = None
        self.orchestrator = None
        self._health = None
        self._health_at = 0.0

    def _build(self, ws: WorkspaceConfig) -> None:
        if self.db is None or self.workspace is None or self.workspace.id != ws.id:
            if self.db is not None:
                try:
                    self.db.close()
                except Exception:  # noqa: BLE001
                    pass
            self.db = Database(config.db_path(ws.id))
        self.workspace = ws
        self.reconfigure(ws)

    def reconfigure(self, ws: Optional[WorkspaceConfig] = None) -> None:
        """Reconstruit garde, client LLM, recherche, impact et chat après un changement de configuration."""
        if ws is not None:
            self.workspace = ws
        ws = self.workspace
        if ws is None or self.db is None:
            return
        try:
            self.guard.configure(ws.mode, ws.allowed_hosts)
        except ValueError:
            self.guard.configure(offline_guard.MODE_STRICT, [])
        self.llm = factory.create_client(ws.llm, self.guard, transport=self.llm_transport)
        self.model_info = None
        self._health = None
        self._health_at = 0.0
        embedding_model = ws.llm.embedding_model if ws.llm.engine != "none" else ""
        self.search_engine = SearchEngine(self.db, self.llm, embedding_model)
        self.indexer = Indexer(self.db, ws, self.llm, embedding_model)
        self.impact = ImpactAnalyzer(self.db, ws, self.search_engine, stale_checker=self.indexer.stale_files)
        self.orchestrator = ChatOrchestrator(self.db, ws, self.llm, search_engine=self.search_engine,
                                             impact_analyzer=self.impact, profile=self.profile(),
                                             model_info=self.model_info)

    def require_workspace(self) -> WorkspaceConfig:
        if self.workspace is None or self.db is None:
            raise NoWorkspace("Aucun espace de travail actif : créez-en un (POST /api/workspaces ou « develop init »).")
        return self.workspace

    # ------------------------------------------------------------------ machine et moteurs
    def profile(self) -> MachineProfile:
        if self._profile is None:
            self._profile = detect_machine()
        return self._profile

    async def engines_report(self, force: bool = False) -> EnginesReport:
        """Moteurs détectés (sondes loopback), en cache 30 s."""
        now = time.time()
        if force or self._engines is None or now - self._engines_at > ENGINES_CACHE_S:
            self._engines = await detect_engines(guard=self.guard, transport=self.llm_transport)
            self._engines_at = time.time()
        return self._engines

    async def engine_status(self, force: bool = False) -> Dict[str, Any]:
        """``{ok, engine, model, error, version, base_url}`` d'après un ``health()`` en cache ≤ 15 s."""
        now = time.time()
        if not force and self._health is not None and now - self._health_at <= HEALTH_CACHE_S:
            return dict(self._health)
        ws = self.workspace
        engine = ws.llm.engine if ws is not None else "none"
        model = ws.llm.model if ws is not None else ""
        if ws is None or engine == "none":
            status = EngineStatus(ok=False, engine="none", base_url="",
                                  error="Aucun moteur d'IA configuré : la recherche fonctionne, mais pas le chat.")
        else:
            try:
                status = await asyncio.wait_for(self.llm.health(), HEALTH_TIMEOUT_S)
            except asyncio.TimeoutError:
                status = EngineStatus(ok=False, engine=engine, base_url=ws.llm.base_url,
                                      error="Le moteur %s ne répond pas (délai dépassé)." % engine)
            except Exception as exc:  # noqa: BLE001 - jamais de 500 sur /api/status
                status = EngineStatus(ok=False, engine=engine, base_url=ws.llm.base_url, error=str(exc))
            if status.ok and model and status.models and not any(m.name == model for m in status.models):
                status = EngineStatus(ok=False, engine=status.engine, base_url=status.base_url,
                                      version=status.version, models=status.models,
                                      error="Le modèle « %s » n'est pas installé sur %s." % (model, status.base_url))
        self._health = {"ok": bool(status.ok), "engine": status.engine or engine, "model": model,
                        "error": status.error or "", "version": status.version, "base_url": status.base_url}
        self._health_at = time.time()
        return dict(self._health)

    async def refresh_model_info(self, timeout: float = MODEL_INFO_TIMEOUT_S) -> Optional[ModelInfo]:
        """Interroge le moteur sur le modèle configuré (fenêtre de contexte, outils), sans jamais lever."""
        ws = self.workspace
        if ws is None or ws.llm.engine == "none" or not ws.llm.model:
            return None
        try:
            info = await asyncio.wait_for(self.llm.model_info(ws.llm.model), timeout)
        except Exception as exc:  # noqa: BLE001
            logger.info("model_info(%s) indisponible : %s", ws.llm.model, exc)
            return None
        self.set_model_info(info)
        return info

    async def warm_models(self, timeout: float = 300.0) -> bool:
        """Précharge le modèle de chat et le modèle d'embeddings pour que la première question ne paie pas
        le chargement (≈ 30 s pour un modèle de 20 Go). Ne lève jamais ; False si rien n'a été préchargé."""
        ws = self.workspace
        if ws is None or ws.llm.engine == "none" or not ws.llm.model:
            return False
        warmed = False
        try:
            warmed = bool(await asyncio.wait_for(self.llm.warm(ws.llm.model), timeout))
            if ws.llm.embedding_model:
                await asyncio.wait_for(self.llm.embed(["préchauffage"], ws.llm.embedding_model), 120.0)
        except Exception as exc:  # noqa: BLE001
            logger.info("préchauffage du modèle %s impossible : %s", ws.llm.model, exc)
        return warmed

    def set_model_info(self, info: Optional[ModelInfo]) -> None:
        self.model_info = info
        if self.orchestrator is not None:
            self.orchestrator.model_info = info

    def blocked_attempts(self) -> List[Dict[str, Any]]:
        return [dataclasses.asdict(b) for b in self.guard.blocked_attempts()]

    # ------------------------------------------------------------------ indexation
    def start_index(self, full: bool = False, repo_id: Optional[str] = None, with_embeddings: bool = True) -> str:
        """Lance un run d'indexation dans le thread dédié ; renvoie ``run_id``. 409 si déjà en cours."""
        ws = self.require_workspace()
        with self._index_lock:
            if self.index_running:
                raise IndexRunning("Une indexation est déjà en cours (run %s)." % self.index_run_id)
            if repo_id and ws.repo_by_id(repo_id) is None:
                raise KeyError(repo_id)
            self.index_running = True
            self.index_cancel = threading.Event()
            run_id = uuid.uuid4().hex[:12]
            self.index_run_id = run_id
            self.index_started_at = time.time()
            self.index_progress = IndexProgress(phase="scan", current=0, total=0, message="Démarrage…")
            embedding_model = ws.llm.embedding_model if (with_embeddings and ws.llm.engine != "none") else ""
            indexer = Indexer(self.db, ws, self.llm, embedding_model, progress=self._on_progress,
                              cancel_event=self.index_cancel)
            self._index_future = self._executor.submit(self._run_index, indexer, full, repo_id, with_embeddings, run_id)
            return run_id

    def _run_index(self, indexer: Indexer, full: bool, repo_id: Optional[str], with_embeddings: bool,
                   run_id: str) -> None:
        status = "failed"
        error: Optional[str] = None
        try:
            row = indexer.run(full=full, repo_id=repo_id, with_embeddings=with_embeddings)
            status, error = row.status, row.error
        except Exception as exc:  # noqa: BLE001 - run() ne lève pas, par sécurité
            error = "%s : %s" % (exc.__class__.__name__, exc)
            logger.exception("indexation interrompue")
        finally:
            with self._index_lock:
                self.index_running = False
        if self.search_engine is not None:
            try:  # la matrice d'embeddings en cache doit être rechargée au prochain appel
                self.search_engine._matrix_cache = None
            except Exception:  # noqa: BLE001
                pass
        payload = self.index_status()
        payload["run_id"] = run_id
        if status == "failed":
            self.publish("error", {"code": "index_failed", "message": error or "Indexation échouée.", "status": payload})
        else:
            self.publish("done", payload)

    def _on_progress(self, progress: IndexProgress) -> None:
        self.index_progress = progress
        self.publish("progress", self.index_status())

    def cancel_index(self) -> bool:
        """Demande l'arrêt du run en cours ; renvoie vrai si un run tournait."""
        with self._index_lock:
            running = self.index_running
            if running:
                self.index_cancel.set()
        return running

    def wait_index(self, timeout: Optional[float] = None) -> None:
        """Attend la fin du run courant (CLI et tests)."""
        fut = self._index_future
        if fut is not None:
            fut.result(timeout=timeout)

    def index_status(self) -> Dict[str, Any]:
        """Forme de ``GET /api/index/status`` (et de l'événement SSE ``progress``)."""
        p = self.index_progress
        last: Optional[Dict[str, Any]] = None
        counts: Dict[str, int] = {}
        if self.db is not None:
            try:
                row = self.db.last_index_run()
                if row is not None:
                    last = {"id": row.id, "status": row.status, "started_at": row.started_at,
                            "finished_at": row.finished_at, "stats": dict(row.stats or {}), "error": row.error}
                counts = self.db.stats()
            except Exception as exc:  # noqa: BLE001 - base fermée pendant un changement d'espace
                logger.debug("index_status: %s", exc)
        return {
            "running": bool(self.index_running),
            "run_id": self.index_run_id,
            "phase": p.phase if (p is not None and self.index_running) else "done",
            "current": p.current if p is not None else 0,
            "total": p.total if p is not None else 0,
            "repo": p.repo if p is not None else "",
            "message": p.message if p is not None else "",
            "started_at": self.index_started_at,
            "last_run": last,
            "counts": counts,
        }

    # ------------------------------------------------------------------ abonnés SSE
    def subscribe(self) -> "asyncio.Queue[Any]":
        """File d'événements ``(type, data)`` liée à la boucle asyncio courante."""
        loop = asyncio.get_running_loop()
        queue: "asyncio.Queue[Any]" = asyncio.Queue()
        with self._sub_lock:
            self._subscribers.append((loop, queue))
        return queue

    def unsubscribe(self, queue: "asyncio.Queue[Any]") -> None:
        with self._sub_lock:
            self._subscribers = [(l, q) for (l, q) in self._subscribers if q is not queue]

    def publish(self, event: str, data: Dict[str, Any]) -> None:
        """Publie un événement vers toutes les files (appelable depuis n'importe quel thread)."""
        with self._sub_lock:
            subscribers = list(self._subscribers)
        dead: List["asyncio.Queue[Any]"] = []
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (event, data))
            except RuntimeError:  # boucle fermée (client parti)
                dead.append(queue)
        for queue in dead:
            self.unsubscribe(queue)

    # ------------------------------------------------------------------ état global
    async def status_dict(self) -> Dict[str, Any]:
        """Forme de ``GET /api/status``."""
        ws = self.workspace
        engine = await self.engine_status()
        return {
            "version": __version__,
            "settings": self.settings.model_dump(),
            "active_workspace": ws.id if ws is not None else None,
            "workspace": ws.model_dump() if ws is not None else None,
            "engine": {"ok": engine["ok"], "engine": engine["engine"], "model": engine["model"],
                       "error": engine["error"]},
            "index": self.index_status(),
            "offline": {"mode": self.guard.mode, "allowed_hosts": sorted(self.guard.allowed_hosts),
                        "blocked_count": len(self.guard.blocked_attempts())},
            "needs_setup": self.needs_setup,
        }

    # ------------------------------------------------------------------ cycle de vie
    async def aclose(self) -> None:
        try:
            await self.llm.aclose()
        except Exception:  # noqa: BLE001
            pass
        self.close()

    def close(self) -> None:
        self.index_cancel.set()
        self._executor.shutdown(wait=False)
        if self.db is not None:
            try:
                self.db.close()
            except Exception:  # noqa: BLE001
                pass
            self.db = None
