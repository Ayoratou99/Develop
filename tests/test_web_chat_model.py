"""Chargement du modèle façon LM Studio dans la vue chat (``develop/web/static/views/chat.js`` : ``initModel``).

Bug d'origine : ``initModel()`` relançait ``POST /api/llm/load`` à chaque montage de la vue dès que le
modèle n'était pas en mémoire — y compris après un « Décharger » explicite : cliquer une autre
conversation, ou aller dans Réglages puis revenir, remontait la vue et ramenait le modèle en mémoire
(10 à 40 s, RAM reprise) sans aucun clic.

Correctif : le chargement automatique n'a lieu qu'**une fois par session** (onglet), à la première
ouverture du chat où le moteur répond ; modèle vu en mémoire, chargement lancé, « Charger le modèle » ou
« Décharger » consomment cette occasion (drapeau ``sessionStorage`` ``develop.chat.modelAutoLoadDone``,
repli en mémoire du module). Ensuite un modèle hors mémoire ne revient que sur « Charger le modèle » ou
à l'envoi d'un message, jamais par une navigation.

Le banc ``tests/web_harness/chat_model_harness.mjs`` (Node ≥ 22.15) monte la vraie vue dans un DOM minimal
avec un faux moteur à mémoire (load / unload) et rejoue : première ouverture, autre conversation,
« Décharger » puis navigations, « Charger le modèle », Entrée sur modèle déchargé, nouvelle session.
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
HARNESS = ROOT / "tests" / "web_harness" / "chat_model_harness.mjs"

ENGINE_DOWN = {"engine": {"engine": "ollama", "ok": False, "model": "qwen3.5:9b", "error": "connexion refusée"},
               "workspace": {"llm": {"engine": "ollama", "embedding_model": "qwen3-embedding:0.6b"}}}


def _node_with_hooks() -> Optional[str]:
    """Chemin de Node si une version avec module.registerHooks (≥ 22.15) est disponible, sinon None."""
    exe = shutil.which("node")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
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
    proc = subprocess.run([node, str(HARNESS), json.dumps(fixtures)], capture_output=True, text=True, timeout=60, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stderr, proc.stderr
    return json.loads(proc.stdout)


def test_model_is_auto_loaded_once_per_session_and_never_after_unload():
    """Chargement automatique une seule fois par session ; après « Décharger », naviguer ne recharge jamais le modèle."""
    src = CHAT_JS.read_text(encoding="utf-8")
    assert "const MODEL_AUTOLOAD_KEY = 'develop.chat.modelAutoLoadDone';" in src
    assert "if (autoLoadDone() || !engineOk()) return;" in src  # initModel : plus de chargement sans clic une fois consommé
    assert src.count("consumeAutoLoad();") == 4  # loadModel, unloadModel, initModel (chargement en cours, modèle chargé)

    # Reproduction de l'utilisateur : le serveur a préchauffé le modèle, la vue le trouve en mémoire.
    warmed = _run_harness({"llm_state": {"loaded": True, "embedding_loaded": True, "last_load_s": 14.2}})["steps"]
    assert warmed["firstMount"]["banner"] == "loaded" and warmed["firstMount"]["loads"] == 0 and warmed["firstMount"]["flag"] == "1"
    assert warmed["unloadClicked"] is True and warmed["unloaded"]["banner"] == "unloaded" and warmed["unloaded"]["unloads"] == 1
    # autre conversation, puis Réglages → Chat : bandeau « non chargé », bouton « Charger le modèle », aucun POST /api/llm/load
    for step in ("afterUnloadRemount", "afterUnloadRemountAgain"):
        assert warmed[step]["loads"] == 0, step
        assert warmed[step]["banner"] == "unloaded" and warmed[step]["loadButton"] is True and warmed[step]["unloadButton"] is False, step
        assert warmed[step]["sendDisabled"] is True and warmed[step]["sendTitle"] == "Chargez le modèle pour envoyer un message.", step
    # « Charger le modèle » recharge explicitement ; Entrée sur modèle déchargé lance le chargement et garde le texte
    assert warmed["loadClicked"] is True and warmed["explicitLoad"]["loads"] == 1 and warmed["explicitLoad"]["banner"] == "loading"
    assert warmed["explicitLoaded"]["banner"] == "loaded" and warmed["explicitLoaded"]["sendDisabled"] is False
    assert warmed["sendWhileUnloaded"]["loads"] == 2 and warmed["sendWhileUnloaded"]["inputValue"] == "Que fait OrderService ?"
    # nouvelle session (onglet rouvert), modèle hors mémoire : chargement automatique de nouveau, une fois
    assert warmed["newSession"]["loads"] == 3 and warmed["newSession"]["banner"] == "loading"

    # Démarrage à froid : première ouverture → un seul chargement automatique ; même règle ensuite.
    cold = _run_harness({})["steps"]
    assert cold["firstMount"]["banner"] == "loading" and cold["firstMount"]["loads"] == 1 and cold["firstMount"]["flag"] == "1"
    assert cold["loaded"]["banner"] == "loaded" and cold["otherConversation"]["loads"] == 1
    assert cold["afterUnloadRemount"]["loads"] == 1 and cold["afterUnloadRemountAgain"]["loads"] == 1
    assert cold["afterUnloadRemount"]["banner"] == "unloaded" and cold["afterUnloadRemount"]["loadButton"] is True
    assert cold["explicitLoad"]["loads"] == 2 and cold["sendWhileUnloaded"]["loads"] == 3 and cold["newSession"]["loads"] == 4

    # Moteur injoignable à la première ouverture : rien ne part et l'occasion reste pour la prochaine ouverture.
    down = _run_harness({"status": ENGINE_DOWN})["steps"]
    assert down["firstMount"]["loads"] == 0 and down["firstMount"]["flag"] is None and down["firstMount"]["banner"] == "unloaded"
