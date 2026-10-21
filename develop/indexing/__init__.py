"""Indexation incrémentale : fichiers des dépôts, documents, notes, relations brutes, embeddings."""
from develop.indexing.embedder import (  # noqa: F401
    EMBEDDING_MODEL_META_KEY,
    Embedder,
    normalize_vectors,
    run_coroutine_sync,
)
from develop.indexing.indexer import (  # noqa: F401
    EXTRACTOR_VERSION,
    PHASES,
    PROGRESS_EVERY,
    IndexCancelled,
    Indexer,
    IndexProgress,
)

__all__ = [
    "EMBEDDING_MODEL_META_KEY",
    "EXTRACTOR_VERSION",
    "PHASES",
    "PROGRESS_EVERY",
    "Embedder",
    "IndexCancelled",
    "IndexProgress",
    "Indexer",
    "normalize_vectors",
    "run_coroutine_sync",
]
