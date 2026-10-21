"""Tests de la mise en page étroite (app.css, ``@media (max-width: 900px)``) : la réponse en cours reste joignable.

Bug d'origine : sous 900 px de large (ex. volet navigateur de 894 px), la règle responsive masquait tous
les ``<span>`` des entrées de navigation — donc la pastille « Réponse en cours » (``#nav-chat-busy``, un
span, app.js) — et la liste des conversations (``.chat-side``). Un utilisateur qui quittait le chat
pendant une réponse n'avait plus aucun chemin dans l'interface pour revenir à la conversation qui répond
(mesuré : ``getComputedStyle(#nav-chat-busy).display`` = "none" à 880 px et 894 px, "flex" à 1280 px).

Correctif (app.css uniquement) : seul le libellé (span sans classe) est masqué ; la pastille reste
affichée sous forme de point qui pulse sur l'icône (texte à 0, info-bulle et clic inchangés) ; la liste
des conversations n'est plus masquée mais réduite à 190 px.

Vérifications statiques du bloc ``@media`` d'app.css (pas de moteur CSS en test) ; mesuré dans un
navigateur à 880 px après correction : pastille 10 × 10 px en ``display: flex``, ``.chat-side`` à 190 px.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Set

APP_CSS = Path(__file__).resolve().parent.parent / "develop" / "web" / "static" / "app.css"
NARROW_MEDIA = "@media (max-width: 900px) {"

_RULE_RE = re.compile(r"([^{}]+?)\s*\{([^{}]*)\}")


def _narrow_rules() -> Dict[str, Dict[str, str]]:
    """Règles ``{sélecteur: {propriété: valeur}}`` du bloc ``@media (max-width: 900px) { … }`` d'app.css."""
    css = APP_CSS.read_text(encoding="utf-8")
    assert NARROW_MEDIA in css, "bloc %s introuvable dans app.css" % NARROW_MEDIA
    body = css.split(NARROW_MEDIA, 1)[1]
    body = body[:body.index("\n}\n")]  # fin du bloc : « } » en colonne 0
    rules: Dict[str, Dict[str, str]] = {}
    for selector, decls in _RULE_RE.findall(body):
        out = {}
        for decl in decls.split(";"):
            if ":" in decl:
                prop, value = decl.split(":", 1)
                out[prop.strip()] = " ".join(value.split())
        rules[" ".join(selector.split())] = out
    return rules


def _hidden_selectors(rules: Dict[str, Dict[str, str]]) -> Set[str]:
    """Sélecteurs (listes éclatées) des règles ``display: none`` du bloc étroit."""
    return {part.strip() for sel, decl in rules.items() if decl.get("display") == "none" for part in sel.split(",")}


def test_narrow_layout_keeps_the_running_answer_badge_visible():
    """La pastille « Réponse en cours » (span.badge de l'entrée Chat) survit au masquage des libellés : point sur l'icône."""
    rules = _narrow_rules()
    hidden = _hidden_selectors(rules)
    # Le libellé est masqué par un sélecteur qui exclut la pastille ; aucun sélecteur masqué ne l'atteint.
    assert ".nav-item > span:not(.badge)" in hidden
    assert not {".nav-item span", ".nav-item > span", "span", ".badge", ".nav-item .badge"} & hidden
    badge = rules[".nav-item .badge"]
    assert badge.get("display") != "none"
    # Point de 10 px ancré en haut à droite de l'entrée (position: relative), texte réduit à 0 : la pastille
    # reste cliquable (ouvre la conversation qui répond) et garde son info-bulle.
    assert rules[".nav-item"]["position"] == "relative"
    assert badge["position"] == "absolute" and badge["width"] == "10px" and badge["height"] == "10px"
    assert badge["border-radius"] == "50%" and badge["font-size"] == "0" and badge["overflow"] == "hidden"
    assert badge["animation"].startswith("pulse")
    # Sans réponse ailleurs, « hidden » garde la main sur ce point (display none !important, règle globale).
    assert ".hidden { display: none !important; }" in APP_CSS.read_text(encoding="utf-8")


def test_narrow_layout_keeps_the_conversation_list():
    """La liste des conversations n'est plus masquée sous 900 px : colonne réduite à 190 px."""
    rules = _narrow_rules()
    assert ".chat-side" not in _hidden_selectors(rules)
    assert ".chat-side" not in rules  # aucune règle étroite ne la retouche : elle garde son affichage (flex)
    assert rules[".chat"]["grid-template-columns"] == "190px minmax(0, 1fr)"
