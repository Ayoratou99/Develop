"""Graphe de connaissances de Develop : relations dérivées de l'index et mentions documentaires.

* :mod:`develop.knowledge_graph.linker` — à la fin de chaque run d'indexation, :func:`link_all`
  recalcule globalement les relations dérivées (``imports``, ``calls``, ``calls_endpoint``,
  ``same_entity``, ``uses_model``, ``documented_by``, ``mentions``) à partir des relations brutes de
  l'indexeur et complète les ``handles`` dont le gestionnaire est défini dans un autre fichier ;
  :func:`neighbors_graph` et :func:`describe_relation` servent l'explorateur et l'analyse d'impact.
* :mod:`develop.knowledge_graph.mentions` — identifiants cités dans les documents et les notes,
  confrontés à l'index pour détecter la dérive documentation / code.

Tout est hors ligne, en lecture seule sur les dépôts, et n'écrit que dans la mémoire Develop.
"""
from develop.knowledge_graph.linker import (  # noqa: F401
    CALLABLE_KINDS,
    DERIVED_KINDS,
    GENERIC_PREFIXES,
    LinkStats,
    describe_relation,
    learn_base_prefixes,
    link_all,
    link_calls,
    link_documents,
    link_handlers,
    link_models,
    match_endpoints,
    neighbors_graph,
    resolve_imports,
)
from develop.knowledge_graph.mentions import (  # noqa: F401
    MAX_MENTIONS,
    Candidate,
    Mention,
    NameIndex,
    document_mentions,
    extract_identifiers,
    is_stale,
    match_candidates,
    scan_text,
    source_text,
)

__all__ = [
    "CALLABLE_KINDS",
    "DERIVED_KINDS",
    "GENERIC_PREFIXES",
    "MAX_MENTIONS",
    "Candidate",
    "LinkStats",
    "Mention",
    "NameIndex",
    "describe_relation",
    "document_mentions",
    "extract_identifiers",
    "is_stale",
    "learn_base_prefixes",
    "link_all",
    "link_calls",
    "link_documents",
    "link_handlers",
    "link_models",
    "match_candidates",
    "match_endpoints",
    "neighbors_graph",
    "resolve_imports",
    "scan_text",
    "source_text",
]
