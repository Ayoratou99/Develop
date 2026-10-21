"""Analyse statique légère du code : symboles, endpoints, modèles de données, imports.

Aucun parseur lourd obligatoire : Python via ``ast``, les autres langages via des
expressions régulières robustes et des heuristiques par framework. L'objectif n'est pas
une analyse exhaustive, mais une carte fiable et citable (fichier:ligne) du projet.
"""
from develop.code_analysis.model import (  # noqa: F401
    ExtractedEndpoint,
    ExtractedImport,
    ExtractedModel,
    ExtractedSymbol,
    FileAnalysis,
    ModelField,
    ModelRelation,
    normalize_route_path,
)
from develop.code_analysis.registry import analyze_file, get_extractor, supported_languages  # noqa: F401
