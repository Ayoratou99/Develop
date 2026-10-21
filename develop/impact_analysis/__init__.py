"""Analyse d'impact déterministe : d'un diff Git, d'un fichier ou d'un symbole vers les éléments affectés.

Sans LLM : tout repose sur le graphe de relations de la mémoire Develop (imports, routes,
appels frontend, modèles, tables, documents, notes) et, en option, sur une recherche lexicale.
"""
from develop.impact_analysis.analyzer import (  # noqa: F401
    DECAY,
    HUB_THRESHOLD,
    MAX_HOPS,
    NODE_CAP,
    ImpactAnalyzer,
    ImpactItem,
    ImpactReport,
)
from develop.impact_analysis.diff import (  # noqa: F401
    Hunk,
    changed_line_ranges,
    merge_ranges,
    parse_unified_diff,
    ranges_intersect,
    symbols_touched,
)

__all__ = [
    "DECAY",
    "HUB_THRESHOLD",
    "MAX_HOPS",
    "NODE_CAP",
    "Hunk",
    "ImpactAnalyzer",
    "ImpactItem",
    "ImpactReport",
    "changed_line_ranges",
    "merge_ranges",
    "parse_unified_diff",
    "ranges_intersect",
    "symbols_touched",
]
