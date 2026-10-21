"""Test du lien « Chat » de la barre latérale (app.js : routeur, barre latérale ; views/chat.js : création
de conversation).

Bug d'origine (bug C de l'utilisateur, après le magasin de flux) : le lien « Chat » pointait toujours sur
``#/chat`` (nouvelle conversation vide). Pendant une réponse, aller dans Réglages puis cliquer « Chat »
affichait « Nouvelle conversation » sans la réponse : le flux continuait bien en arrière-plan (pastille
« Réponse en cours »), mais il fallait un second clic pour le voir, ce que rien n'expliquait à l'écran.

Correctif : ``app.js`` garde la cible du lien « Chat » = la conversation affichée en dernier
(``rememberChatRoute``), mise à jour par le routeur à chaque route du chat et par la vue chat quand elle
crée la conversation (URL remplacée par ``history.replaceState``, sans ``hashchange``) ; « Nouvelle » ou la
suppression de la conversation affichée ramènent le lien sur ``#/chat``.

Si Node ≥ 22.15 est installé, le banc ``tests/web_harness/app_nav_harness.mjs`` charge le vrai ``app.js``
(routeur + barre latérale) avec la vraie vue chat et le vrai magasin de flux, et rejoue le scénario exact.
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
STATIC = ROOT / "develop" / "web" / "static"
CHAT_JS = STATIC / "views" / "chat.js"
APP_JS = STATIC / "app.js"
HARNESS = ROOT / "tests" / "web_harness" / "app_nav_harness.mjs"

QUESTION = "Que fait OrderService ?"
READING = "Le modèle lit 1 source (≈ 300 tokens) avant de répondre…"


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


def test_sidebar_chat_link_returns_to_last_conversation():
    """Réglages puis « Chat » pendant une réponse : la conversation qui répond est à l'écran, sans second clic."""
    # Contrat : le routeur mémorise la route du chat, la vue lui signale la conversation qu'elle crée.
    app = APP_JS.read_text(encoding="utf-8")
    assert "export function rememberChatRoute(conversationId)" in app
    assert "if (parsed.name === 'chat') rememberChatRoute(parsed.parts[0] || null);" in app
    assert "href: item.name === 'chat' ? chatHash : `#/${item.name}`," in app
    chat = CHAT_JS.read_text(encoding="utf-8")
    assert "import { state, navigate, refreshStatus, rememberChatRoute } from '/static/app.js';" in chat
    assert chat.index("state.route = { name: 'chat', parts: [conversationId], query: {} };") < chat.index("rememberChatRoute(conversationId);")

    out = _run_harness({})

    # Démarrage sur #/chat : lien « Chat » = nouvelle conversation, fil vide.
    start = out["start"]
    assert start["hash"] == "#/chat" and start["chatHref"] == "#/chat" and start["active"] == ["chat"]
    assert start["thread"]["empty"] == ["Nouvelle conversation"] and start["thread"]["users"] == []

    # Envoi : la vue crée la conversation 9 (URL remplacée sans hashchange) → le lien la suit aussitôt.
    sent = out["sent"]
    assert sent["hash"] == "#/chat/9" and sent["chatHref"] == "#/chat/9" and sent["stateRoute"]["parts"] == ["9"]
    assert sent["thread"]["users"] == [QUESTION] and sent["thread"]["assistants"][0]["status"] == [READING]
    assert sent["badgeHidden"] is True  # la conversation qui répond est affichée : pas de pastille

    # Réglages pendant la lecture du contexte : le lien vise toujours la conversation 9, la pastille la signale aussi.
    settings = out["settings"]
    assert settings["hash"] == "#/settings" and settings["active"] == ["settings"] and settings["page"] == "page page-settings"
    assert settings["chatHref"] == "#/chat/9" and settings["running"] == ["9"]
    assert settings["badgeHidden"] is False and settings["badgeConversation"] == "9"

    # Retour par le lien « Chat » : la conversation 9 et sa réponse en cours, pas « Nouvelle conversation ».
    back = out["back"]
    assert back["hash"] == "#/chat/9" and back["active"] == ["chat"] and back["chatHref"] == "#/chat/9"
    assert back["thread"]["empty"] == [] and back["thread"]["users"] == [QUESTION]
    assert back["thread"]["assistants"][0]["status"] == [READING] and back["thread"]["assistants"][0]["text"] == ""
    assert back["thread"]["sendDisabled"] is True and back["thread"]["stopHidden"] is False
    assert back["thread"]["list"] == [{"title": QUESTION, "busy": True, "active": True}]
    assert back["badgeHidden"] is True and back["running"] == ["9"]

    # Fin du flux en direct dans la vue revenue : texte, pied, boutons rétablis ; le lien reste sur la conversation.
    finished = out["finished"]
    assert finished["running"] == [] and finished["chatHref"] == "#/chat/9"
    assert finished["thread"]["assistants"][0]["text"] == "OrderService gère les commandes [S1]."
    assert finished["thread"]["assistants"][0]["status"] == [] and finished["thread"]["assistants"][0]["foot"].startswith("Entrée : 300 tokens")
    assert finished["thread"]["sendDisabled"] is False and finished["thread"]["stopHidden"] is True
    after = out["backAfterDone"]
    assert after["hash"] == "#/chat/9" and after["thread"]["users"] == [QUESTION]
    assert [a["text"] for a in after["thread"]["assistants"]] == ["OrderService gère les commandes [S1]."]

    # « Nouvelle » : conversation vide, le lien « Chat » redevient #/chat, même après un détour par Réglages.
    fresh = out["fresh"]
    assert fresh["hash"] == "#/chat" and fresh["chatHref"] == "#/chat" and fresh["thread"]["empty"] == ["Nouvelle conversation"]
    assert fresh["thread"]["list"] == [{"title": QUESTION, "busy": False, "active": False}]
    fresh_back = out["freshBack"]
    assert fresh_back["hash"] == "#/chat" and fresh_back["chatHref"] == "#/chat" and fresh_back["thread"]["empty"] == ["Nouvelle conversation"]

    # Navigation réelle : chaque clic suit le href du lien, et les autres vues sont montées puis démontées.
    assert out["hashChanges"] == ["#/settings", "#/chat/9", "#/explorer", "#/chat/9", "#/chat", "#/settings", "#/chat"]
    assert out["mounts"] == [["settings", []], ["explorer", []], ["settings", []]]
