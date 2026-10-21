"""Tests de non-régression de la modale « Détecter les sous-projets » (``develop/web/static/views/setup.js``).

L'interface est en JavaScript vanille, sans étape de build ni navigateur disponible dans la suite :
on vérifie statiquement les règles qui bornent la modale à la fenêtre et qui tronquent ou coupent les
chemins longs, telles qu'elles ont été vérifiées dans un navigateur à 1024 × 768.

Défaut corrigé : la piste implicite (``auto``) de la grille ``.modal-backdrop`` s'élargissait jusqu'à
la largeur min-content du contenu (un chemin de dépôt de ~170 caractères) ; la modale de 860 px,
centrée dans une piste de ~1 650 px, débordait de l'écran (sélecteurs de type coupés, bouton
« Ajouter la sélection » visible seulement par son bord). Avec ``minmax(0, 1fr)`` la piste vaut la
largeur disponible, les chemins ``.truncate`` s'abrègent avec « … » et le chemin de la phrase
d'introduction passe à la ligne n'importe où (``overflow-wrap: anywhere``).
"""
from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "develop" / "web" / "static"
APP_CSS = STATIC / "app.css"
SETUP_JS = STATIC / "views" / "setup.js"

# Corps de la fonction exportée, jusqu'à la première ligne « } » en colonne 0.
_DISCOVER_RE = re.compile(r"^export function discoverSubprojects\(path\) \{\n.*?\n\}\n", re.M | re.S)


def _css_declarations(selector: str) -> dict:
    """Déclarations ``{propriété: valeur}`` de la première règle ``selector { … }`` d'app.css (règles sur une ligne)."""
    css = APP_CSS.read_text(encoding="utf-8")
    m = re.search(r"^" + re.escape(selector) + r"\s*\{([^}]*)\}", css, re.M)
    assert m, "règle %s introuvable dans app.css" % selector
    out = {}
    for decl in m.group(1).split(";"):
        if ":" in decl:
            prop, value = decl.split(":", 1)
            out[prop.strip()] = " ".join(value.split())
    return out


def _discover_source() -> str:
    m = _DISCOVER_RE.search(SETUP_JS.read_text(encoding="utf-8"))
    assert m, "discoverSubprojects introuvable dans setup.js"
    return m.group(0)


def test_modal_backdrop_column_is_bounded_to_the_viewport():
    """Régression : la colonne de la grille ne doit plus s'élargir au contenu (minmax(0, 1fr))."""
    backdrop = _css_declarations(".modal-backdrop")
    assert backdrop.get("display") == "grid"
    assert backdrop.get("grid-template-columns") == "minmax(0, 1fr)"
    assert backdrop.get("position") == "fixed" and backdrop.get("inset") == "0"
    # La modale ne dépasse jamais la piste (100 %), le corps absorbe tout dépassement restant.
    assert _css_declarations(".modal").get("width") == "min(560px, 100%)"
    assert _css_declarations(".modal-wide").get("width") == "min(860px, 100%)"
    assert _css_declarations(".modal-body").get("overflow") == "auto"


def test_long_paths_truncate_or_wrap_in_discover_modal():
    """Les chemins des sous-projets s'abrègent (« … », chemin complet en info-bulle) et celui de l'intro se coupe."""
    # .truncate : overflow ≠ visible donne au flex-item une taille minimale automatique de 0, condition
    # pour qu'un chemin plus large que .list-item-main (min-width: 0) se réduise avec des points de suspension.
    truncate = _css_declarations(".truncate")
    assert truncate.get("overflow") == "hidden"
    assert truncate.get("text-overflow") == "ellipsis"
    assert truncate.get("white-space") == "nowrap"
    assert _css_declarations(".list-item-main").get("min-width") == "0"

    src = _discover_source()
    assert "el('span', { class: 'mono truncate', title: sp.path }, sp.path)" in src
    # Le chemin de la phrase d'introduction n'est plus interpolé brut : il est dans un nœud coupable n'importe où
    # (un chemin sans tiret n'a aucune césure naturelle et élargissait le corps de la modale).
    assert "détecté(s) dans ${path}" not in src
    assert "el('span', { class: 'mono', style: { overflowWrap: 'anywhere' } }, path)" in src
    # Le sélecteur de type garde une largeur fixe : c'est la colonne du chemin qui se réduit, pas lui.
    assert "el('div', { style: { width: '160px' } }, kindSelect(row.kind" in src
