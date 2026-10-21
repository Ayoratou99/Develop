"""Entrée pendant un chargement de modèle dans la vue chat (``develop/web/static/views/chat.js`` : ``send``).

Bug d'origine : message saisi puis Entrée pendant « chargement en mémoire… » : la notification disait
« votre message partira dès qu'il sera prêt » alors que rien n'était mis en file — une fois le modèle
chargé, le message n'était pas envoyé (le texte restait seulement saisi), et la même phrase ajoutait
« réessayez dans quelques secondes » ; l'utilisateur attendait un envoi automatique qui ne venait pas.

Correctif : la notification dit ce qui se passe réellement (comme sa jumelle du modèle déchargé) :
« envoyez votre message une fois le modèle chargé (le texte reste saisi) ». Rien n'est mis en file :
à la LM Studio, on ne chatte qu'une fois le modèle en mémoire.

Le banc ``tests/web_harness/chat_send_loading_harness.mjs`` (Node ≥ 22.15) monte la vraie vue dans un
DOM minimal avec un modèle en cours de chargement, saisit un texte, appuie sur Entrée puis termine le
chargement (événement ``develop:llm-state``) ; son DOM ignore ``disabled``, ce qui atteint ``send()``
même sur un champ grisé (cas d'un chargement signalé entre la frappe et l'envoi).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
CHAT_JS = ROOT / "develop" / "web" / "static" / "views" / "chat.js"
HARNESS = ROOT / "tests" / "web_harness" / "chat_send_loading_harness.mjs"


def _node_with_hooks() -> Optional[str]:
    """Chemin de Node si une version avec module.registerHooks (≥ 22.15) est disponible, sinon None."""
    exe = shutil.which("node")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, encoding="utf-8", timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.match(r"v(\d+)\.(\d+)", out)
    if not m:
        return None
    return exe if (int(m.group(1)), int(m.group(2))) >= (22, 15) else None


def _run_harness(fixtures: Dict[str, Any]) -> Dict[str, Any]:
    node = _node_with_hooks()
    if node is None:
        pytest.skip("Node ≥ 22.15 absent")
    proc = subprocess.run([node, str(HARNESS), json.dumps(fixtures)], capture_output=True, text=True, encoding="utf-8", timeout=60, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stderr, proc.stderr
    return json.loads(proc.stdout)


def test_loading_toast_does_not_promise_an_automatic_send():
    """Statique : la phrase trompeuse a disparu ; la notification du chargement dit d'envoyer une fois chargé."""
    src = CHAT_JS.read_text(encoding="utf-8")
    assert "partira dès qu" not in src
    assert "réessayez dans quelques secondes" not in src
    m = re.search(r"if \(llm\.loading\) toast\('([^']*(?:\\'[^']*)*)'\)", src)
    assert m, "notification « modèle en cours de chargement » introuvable dans send()"
    assert "envoyez votre message une fois le modèle chargé" in m.group(1).replace("\\'", "'")


def test_enter_while_loading_keeps_the_text_and_sends_nothing_once_loaded():
    """Banc : Entrée pendant le chargement → notification honnête ; fin du chargement → rien ne part, texte gardé."""
    out = _run_harness({})
    steps = out["steps"]
    assert steps["loading"]["banner"] == "loading"
    assert steps["loading"]["sendDisabled"] is True

    pressed = steps["enterWhileLoading"]
    assert pressed["sends"] == 0
    assert pressed["inputValue"] == "Qui appelle POST /orders ?"
    assert len(pressed["toasts"]) == 1, pressed["toasts"]
    toast = pressed["toasts"][0]
    assert "envoyez votre message une fois le modèle chargé" in toast
    assert "partira" not in toast and "réessayez" not in toast

    loaded = steps["loaded"]
    assert loaded["banner"] == "loaded"
    assert loaded["sends"] == 0, "rien n'est mis en file : aucun envoi automatique après le chargement"
    assert loaded["inputValue"] == "Qui appelle POST /orders ?", "le texte reste saisi pour être envoyé"
    assert loaded["userMessages"] == 1, "aucun message ajouté au fil tant que l'utilisateur n'a pas renvoyé"
    assert loaded["sendDisabled"] is False and loaded["inputDisabled"] is False
    assert not any(r.startswith("POST /api/conversations/") for r in out["requests"]), out["requests"]
