"""Calcul des embeddings (recherche sémantique) à partir du moteur LLM local.

Principes :

* les vecteurs sont normalisés (norme L2 = 1, ``float32``) **avant** stockage : la similarité
  cosinus se réduit alors à un produit scalaire numpy côté recherche ;
* un seul modèle d'embeddings à la fois : si la mémoire contient des vecteurs d'un autre
  modèle (meta ``embedding_model``), ils sont tous supprimés puis recalculés — jamais de
  mélange de dimensions ;
* le client LLM est asynchrone alors que l'indexation est synchrone (thread de fond) : la
  coroutine est exécutée avec ``asyncio.run`` dans un nouvel event loop, ou dans un thread
  dédié si un event loop tourne déjà dans le thread courant ;
* aucune erreur ne remonte : moteur injoignable (:class:`LLMUnavailable`) → arrêt propre en
  renvoyant le nombre de vecteurs déjà calculés ; les autres erreurs sont consignées dans
  :attr:`Embedder.errors` et l'on s'arrête après quelques lots consécutifs en échec.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, TypeVar

import numpy as np

from develop.llm.base import LLMClient, LLMUnavailable
from develop.storage import ChunkRow, Database

T = TypeVar("T")

#: Clé ``meta`` qui mémorise le modèle d'embeddings dont les vecteurs sont en base.
EMBEDDING_MODEL_META_KEY = "embedding_model"
#: Longueur maximale (caractères) du texte envoyé au modèle pour un morceau.
EMBED_MAX_CHARS = 8000
#: Nombre de lots consécutifs en échec (hors moteur injoignable) avant d'abandonner le run.
MAX_CONSECUTIVE_FAILURES = 3
_MAX_ERRORS = 200


def run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Exécute une coroutine depuis du code synchrone.

    Sans event loop dans le thread courant : ``asyncio.run`` (nouvel event loop, fermé à la
    fin). Si un event loop tourne déjà (appel depuis un gestionnaire FastAPI, un test
    asynchrone…), la coroutine est exécutée dans un thread dédié avec son propre loop, et
    le résultat (ou l'exception) est rapporté dans le thread appelant.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - rapportée telle quelle dans le thread appelant
            box["error"] = exc

    thread = threading.Thread(target=_runner, name="develop-embedder", daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")  # type: ignore[return-value]


def normalize_vectors(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Matrice ``float32`` ``(n, dim)`` normalisée ligne par ligne (norme L2 = 1).

    Un vecteur nul reste nul (il ne ressemblera à rien, ce qui est le comportement voulu).
    Lève ``ValueError`` si les vecteurs sont vides ou de dimensions hétérogènes.
    """
    rows = [list(v) for v in vectors]
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    dims = {len(v) for v in rows}
    if len(dims) != 1:
        raise ValueError("vecteurs de dimensions hétérogènes : %s" % sorted(dims))
    dim = dims.pop()
    if dim == 0:
        raise ValueError("vecteurs d'embedding vides")
    matrix = np.nan_to_num(np.asarray(rows, dtype=np.float32))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray((matrix / norms).astype(np.float32, copy=False))


def text_for_chunk(chunk: ChunkRow) -> str:
    """Texte envoyé au modèle pour un morceau (tronqué à :data:`EMBED_MAX_CHARS`)."""
    text = chunk.text if isinstance(chunk.text, str) else str(chunk.text or "")
    text = text.strip()
    if len(text) > EMBED_MAX_CHARS:
        text = text[:EMBED_MAX_CHARS]
    return text or " "  # certains moteurs refusent une entrée vide


class Embedder:
    """Calcule et stocke les vecteurs manquants pour un modèle d'embeddings donné.

    ``llm`` peut valoir ``None`` (aucun moteur) : toutes les opérations renvoient alors 0.
    """

    def __init__(self, db: Database, llm: Optional[LLMClient], model: str):
        self.db = db
        self.llm = llm
        self.model = (model or "").strip()
        self.errors: List[Dict[str, str]] = []
        self.warnings: List[str] = []
        self.unavailable: bool = False  # vrai si le moteur s'est révélé injoignable pendant le dernier run
        self.purged: int = 0  # vecteurs supprimés lors d'un changement de modèle

    # ------------------------------------------------------------------ erreurs
    def _error(self, message: str, stage: str = "embeddings") -> None:
        if len(self.errors) < _MAX_ERRORS:
            self.errors.append({"path": "", "stage": stage, "message": message})

    # ------------------------------------------------------------------ cohérence du modèle
    def ensure_model_consistency(self) -> bool:
        """Garantit qu'un seul modèle est présent en base.

        Si meta ``embedding_model`` existe et diffère du modèle courant, **tous** les vecteurs
        sont supprimés (changement de modèle = recalcul complet) puis la meta est mise à jour.
        Renvoie ``True`` si une purge a eu lieu.
        """
        if not self.model:
            return False
        stored = self.db.get_meta(EMBEDDING_MODEL_META_KEY)
        purged = False
        with self.db.transaction():
            if stored and stored != self.model:
                self.purged = self.db.delete_embeddings()
                purged = True
                self.warnings.append("Modèle d'embeddings changé (%s → %s) : %d vecteur(s) supprimé(s), recalcul complet."
                                     % (stored, self.model, self.purged))
            else:
                # vecteurs d'un autre modèle sans meta (base ancienne) : on nettoie aussi
                others = [m for m in self.db.embedding_models() if m["model"] != self.model]
                for m in others:
                    self.purged += self.db.delete_embeddings(m["model"])
                    purged = True
            if stored != self.model:
                self.db.set_meta(EMBEDDING_MODEL_META_KEY, self.model)
        return purged

    # ------------------------------------------------------------------ calcul
    def embed_missing(self, batch_size: int = 32, limit: Optional[int] = None,
                      progress: Optional[Callable[[int, int], None]] = None,
                      cancel_event: Optional[threading.Event] = None) -> int:
        """Calcule les vecteurs des morceaux qui n'en ont pas pour :attr:`model` ; renvoie le nombre écrit.

        ``progress(done, total)`` est appelé après chaque lot. ``cancel_event`` interrompt
        proprement entre deux lots. Jamais d'exception : voir :attr:`errors` / :attr:`unavailable`.
        """
        self.unavailable = False
        if self.llm is None or not self.model:
            return 0
        try:
            self.ensure_model_consistency()
            ids = self.db.chunk_ids_missing_embeddings(self.model, limit)
        except Exception as exc:  # noqa: BLE001 - base indisponible : on consigne
            self._error("Lecture des morceaux à vectoriser impossible : %s" % exc)
            return 0
        if not ids:
            return 0
        size = max(1, int(batch_size))
        try:
            return run_coroutine_sync(self._embed_ids(ids, size, progress, cancel_event))
        except Exception as exc:  # noqa: BLE001 - ceinture et bretelles
            self._error("Calcul des embeddings interrompu : %s" % exc)
            return 0

    async def _embed_ids(self, ids: List[int], batch_size: int, progress: Optional[Callable[[int, int], None]],
                         cancel_event: Optional[threading.Event]) -> int:
        assert self.llm is not None
        total = len(ids)
        done = 0
        failures = 0
        for start in range(0, total, batch_size):
            if cancel_event is not None and cancel_event.is_set():
                break
            batch_ids = ids[start:start + batch_size]
            chunks = self.db.get_chunks(batch_ids)
            if not chunks:
                continue  # morceaux remplacés entre-temps
            texts = [text_for_chunk(c) for c in chunks]
            try:
                vectors = await self.llm.embed(texts, self.model)
            except LLMUnavailable as exc:
                self.unavailable = True
                self.warnings.append("Moteur d'embeddings injoignable, calcul interrompu (%d/%d) : %s" % (done, total, exc))
                break
            except Exception as exc:  # noqa: BLE001 - LLMError, erreurs réseau, réponse invalide…
                failures += 1
                self._error("Lot d'embeddings %d-%d en échec : %s" % (start + 1, start + len(chunks), exc))
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    self.warnings.append("Embeddings abandonnés après %d lots consécutifs en échec." % failures)
                    break
                continue
            failures = 0
            try:
                matrix = normalize_vectors(vectors)
                if matrix.shape[0] != len(chunks):
                    raise ValueError("%d vecteur(s) reçus pour %d texte(s)" % (matrix.shape[0], len(chunks)))
            except ValueError as exc:
                self._error("Réponse d'embedding invalide du modèle %s : %s" % (self.model, exc))
                failures += 1
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                continue
            items = [(int(c.id), self.model, matrix[i]) for i, c in enumerate(chunks) if c.id is not None]
            try:
                done += self.db.upsert_embeddings(items)
            except Exception as exc:  # noqa: BLE001
                self._error("Écriture des embeddings impossible : %s" % exc)
                break
            if progress is not None:
                try:
                    progress(done, total)
                except Exception:  # noqa: BLE001 - un rappel d'interface ne doit jamais casser le calcul
                    pass
        return done

    async def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Vecteurs normalisés ``(n, dim)`` de textes arbitraires (requêtes de recherche).

        Lève les erreurs du moteur (:class:`LLMUnavailable`…) : l'appelant décide du repli
        (recherche lexicale seule). Renvoie une matrice vide pour une liste vide.
        """
        items = [t if isinstance(t, str) else str(t or "") for t in texts]
        if not items:
            return np.zeros((0, 0), dtype=np.float32)
        if self.llm is None or not self.model:
            raise LLMUnavailable("Aucun modèle d'embeddings configuré.")
        prepared = [(t.strip()[:EMBED_MAX_CHARS] or " ") for t in items]
        vectors = await self.llm.embed(prepared, self.model)
        matrix = normalize_vectors(vectors)
        if matrix.shape[0] != len(prepared):
            raise ValueError("%d vecteur(s) reçus pour %d texte(s)" % (matrix.shape[0], len(prepared)))
        return matrix
