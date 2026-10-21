"""Registre des extracteurs par langage."""
from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Protocol

from develop.code_analysis.model import FileAnalysis


class Extractor(Protocol):
    languages: List[str]

    def analyze(self, text: str, path: str) -> FileAnalysis:  # pragma: no cover - protocole
        ...


# langage -> module de l'extracteur (chargé paresseusement, un extracteur manquant n'empêche pas les autres)
_EXTRACTOR_MODULES: Dict[str, str] = {
    "python": "develop.code_analysis.extractors.python",
    "javascript": "develop.code_analysis.extractors.javascript",
    "jsx": "develop.code_analysis.extractors.javascript",
    "typescript": "develop.code_analysis.extractors.javascript",
    "tsx": "develop.code_analysis.extractors.javascript",
    "vue": "develop.code_analysis.extractors.javascript",
    "svelte": "develop.code_analysis.extractors.javascript",
    "php": "develop.code_analysis.extractors.php",
    "blade": "develop.code_analysis.extractors.php",
    "sql": "develop.code_analysis.extractors.sql",
    "java": "develop.code_analysis.extractors.java",
    "kotlin": "develop.code_analysis.extractors.java",
    "dart": "develop.code_analysis.extractors.dart",
}
_CACHE: Dict[str, Optional[Extractor]] = {}


def supported_languages() -> List[str]:
    return sorted(_EXTRACTOR_MODULES)


def get_extractor(language: Optional[str]) -> Optional[Extractor]:
    if not language or language not in _EXTRACTOR_MODULES:
        return None
    modname = _EXTRACTOR_MODULES[language]
    if modname not in _CACHE:
        try:
            mod = importlib.import_module(modname)
            _CACHE[modname] = mod.EXTRACTOR  # chaque module expose une instance EXTRACTOR
        except (ImportError, AttributeError):
            _CACHE[modname] = None
    return _CACHE[modname]


def analyze_file(text: str, language: Optional[str], path: str = "") -> FileAnalysis:
    """Analyse un fichier ; renvoie une analyse vide (avec avertissement) si aucun extracteur ne s'applique."""
    ext = get_extractor(language)
    if ext is None:
        return FileAnalysis(language=language or "", warnings=["aucun extracteur pour %s" % (language or "?")])
    try:
        return ext.analyze(text, path)
    except Exception as exc:  # noqa: BLE001 - un fichier exotique ne doit jamais arrêter l'indexation
        return FileAnalysis(language=language or "", warnings=["échec de l'analyse de %s : %s" % (path, exc)])
