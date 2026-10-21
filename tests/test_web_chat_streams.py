"""Tests de la réponse de chat conservée quand on change d'écran (streams.js, views/chat.js, app.js).

Bug d'origine : la vue chat consommait elle-même le flux SSE ; aller dans Réglages pendant une
réponse coupait le flux (abort au démontage) et, au retour, la conversation rechargée depuis la base
ne contenait pas la réponse — il fallait renvoyer le message.

Correctif : un magasin de flux au niveau de l'application (``develop/web/static/streams.js``) lit le
flux indépendamment des vues, conserve les événements reçus (tampon) et l'état, et les rejoue à tout
abonné tardif ; la vue s'y abonne et s'en désabonne au démontage sans le couper (seul « Arrêter »
interrompt) ; la barre latérale affiche « Réponse en cours » quand un flux tourne hors de sa
conversation ; un flux terminé est purgé au rechargement de la conversation depuis la base, qui fait
foi dès qu'elle contient la réponse (rien n'est affiché deux fois).

Deux niveaux : des vérifications statiques (contrat des modules, syntaxe via ``node --check``, aucune
URL externe) et, si Node ≥ 22.15 est installé, le banc ``tests/web_harness/chat_streams_harness.mjs``
qui simule le flux SSE avec un ReadableStream et monte la vraie vue dans un DOM minimal.
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
STREAMS_JS = STATIC / "streams.js"
CHAT_JS = STATIC / "views" / "chat.js"
APP_JS = STATIC / "app.js"
API_JS = STATIC / "api.js"
HARNESS = ROOT / "tests" / "web_harness" / "chat_streams_harness.mjs"

NBSP = re.compile("[  ]")  # séparateurs de milliers d'Intl.NumberFormat('fr-FR') : espaces insécables


def _node_exe() -> Optional[str]:
    return shutil.which("node")


def _node_with_hooks() -> Optional[str]:
    """Chemin de Node si une version avec module.registerHooks (≥ 22.15) est disponible, sinon None."""
    exe = _node_exe()
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


def _fixtures() -> Dict[str, Any]:
    """« Base » vue par la vue : les messages utilisateur sont enregistrés dès réception, pas encore les réponses."""
    return {
        "conversations": [{"id": 7, "title": "OrderService"}, {"id": 8, "title": "Facturation"}],
        "db": {
            "7": {"id": 7, "title": "OrderService", "messages": [
                {"id": 1, "role": "user", "content": "Que fait OrderService ?", "sources": [], "created_at": "2026-10-21T10:00:00"}]},
            "8": {"id": 8, "title": "Facturation", "messages": [
                {"id": 5, "role": "user", "content": "Explique la facturation", "sources": [], "created_at": "2026-10-21T10:01:00"}]},
        },
    }


def _run_harness(fixtures: Dict[str, Any]) -> Dict[str, Any]:
    node = _node_with_hooks()
    if node is None:
        pytest.skip("Node ≥ 22.15 absent")
    proc = subprocess.run([node, str(HARNESS), json.dumps(fixtures)], capture_output=True, text=True, timeout=60, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr
    assert "Warning" not in proc.stderr, proc.stderr
    return json.loads(proc.stdout)


def _plain(text: Optional[str]) -> Optional[str]:
    return NBSP.sub(" ", text) if text is not None else None


# --------------------------------------------------------------------------- statique

def test_modules_have_valid_syntax_and_no_external_url():
    """streams.js, chat.js, app.js et api.js passent ``node --check`` et ne référencent aucune adresse non locale."""
    for path in (STREAMS_JS, CHAT_JS, APP_JS, API_JS):
        for url in re.findall(r"https?://[^\s'\"()]+", path.read_text(encoding="utf-8")):
            assert re.match(r"https?://(127\.0\.0\.1|localhost)(:\d+)?", url), "URL externe dans %s : %s" % (path.name, url)
    node = _node_exe()
    if not node:
        pytest.skip("node absent : syntaxe non vérifiée")
    for path in (STREAMS_JS, CHAT_JS, APP_JS, API_JS):
        proc = subprocess.run([node, "--input-type=module", "--check"], input=path.read_text(encoding="utf-8"),
                              capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, "%s : %s" % (path.name, proc.stderr)


def test_stream_store_contract():
    """streams.js expose le magasin (start, subscribe avec rejeu, abort, release, onChange) et lit le flux via api.js."""
    src = STREAMS_JS.read_text(encoding="utf-8")
    assert "import { streamPost } from '/static/api.js';" in src
    for name in ("startStream", "subscribe", "abort", "getStream", "isRunning", "runningIds", "release", "discard", "onChange"):
        assert "export function %s(" % name in src, name
    assert "export const END_EVENT = 'end';" in src
    assert "for (const ev of entry.events.slice()) safeCall(handler, ev.name, ev.data, { at: ev.at, replay: true });" in src  # rejeu
    assert "if (existing && existing.running) throw new Error('Une réponse est déjà en cours dans cette conversation.');" in src
    assert "entry.aborted = !!(err && err.name === 'AbortError');" in src
    assert "if (!entry || entry.running) return false;" in src  # release ne purge jamais un flux en cours


def test_chat_view_delegates_to_store_and_keeps_stream_on_unmount():
    """La vue ne lit plus le flux elle-même : envoi délégué à startStream, abonnement, démontage sans abort."""
    src = CHAT_JS.read_text(encoding="utf-8")
    assert "streamPost" not in src  # plus de lecture directe du flux par la vue
    assert "from '/static/streams.js';" in src
    assert "const entry = startStream(conversationId, { content, mode });" in src
    assert "unsubscribe = subscribe(entry.conversationId, onEvent);" in src
    assert "stopBtn.addEventListener('click', () => { if (conversationId) abortStream(conversationId); });" in src
    assert "new AbortController()" not in src and "abort.abort()" not in src
    assert "detachStream(); // le flux continue dans streams.js ; seul « Arrêter » le coupe" in src
    # reprise au montage : la base fait foi si elle a la réponse, sinon le tampon est rejoué, puis purge du flux terminé
    assert "export function storedAnswerIndex(messages, entry)" in src
    assert "if (storedAnswerIndex(messages, entry) < 0) {" in src
    assert "if (!entry.running) release(conversationId);" in src
    assert "title: 'Réponse en cours' }, '…')" in src  # marque de la liste des conversations
    # pied de message et clés de done.budget inchangés (tests existants)
    assert "budgetFooter(budget, initialSources, truncated)" in src and "initialSources = sources.length;" in src


def test_sidebar_badge_shows_running_answer_outside_its_conversation():
    """app.js : pastille « Réponse en cours » sur l'entrée Chat, mise à jour à chaque route et à chaque début/fin de flux."""
    src = APP_JS.read_text(encoding="utf-8")
    assert "import { onChange as onStreamsChange, runningIds } from '/static/streams.js';" in src
    assert "id: 'nav-chat-busy'" in src and "'Réponse en cours'" in src
    assert "const elsewhere = running.filter((id) => id !== shown);" in src
    assert "badge.classList.toggle('hidden', !elsewhere.length);" in src
    assert "onStreamsChange(renderStreamBadge);" in src
    assert src.count("renderStreamBadge();") == 1  # après highlightNav dans route()
    assert src.index("highlightNav(parsed.name);") < src.index("renderStreamBadge();")


# --------------------------------------------------------------------------- fonctionnel (Node)

def test_stream_survives_view_changes_and_is_purged_after_reload():
    """Tampon rejoué à un abonné tardif, pas de coupure au démontage, « Arrêter », purge après done + rechargement."""
    out = _run_harness(_fixtures())

    # A. magasin : l'abonné précoce voit tout en direct, l'abonné tardif reçoit d'abord le tampon (replay) puis le direct
    store = out["store"]
    assert [e for e in store["early"] if e[1]] == []
    assert store["late"] == [["status", True], ["sources", True], ["status", True], ["delta", True], ["delta", False]]
    assert store["text"] == "OrderService gère les commandes [S1]." and store["sources"] == ["S1"]
    assert store["running"] is True and store["done"] is False and store["runningIds"] == ["7"]
    assert store["isRunning7"] is True and store["isRunning8"] is False and store["noStream"] is None
    assert store["events"] == ["status", "sources", "status", "delta", "delta"]

    # B. vue montée pendant la réponse : message utilisateur (de la base, une seule fois) + réponse reconstruite
    mounted = out["mounted"]
    assert mounted["users"] == ["Que fait OrderService ?"]
    assert len(mounted["assistants"]) == 1
    assert mounted["assistants"][0]["text"] == "OrderService gère les commandes [S1]."
    assert mounted["assistants"][0]["sources"] == 1 and mounted["assistants"][0]["status"] == []  # texte arrivé : ligne d'état masquée
    assert mounted["stopHidden"] is False and mounted["sendDisabled"] is True
    assert mounted["list"] == [{"title": "OrderService", "busy": True, "active": True}, {"title": "Facturation", "busy": False, "active": False}]
    # démontage (navigation vers Réglages) : rien n'est coupé, le tampon continue de se remplir
    assert out["afterUnmount"] == {"signalAborted": False, "running": True,
                                   "text": "OrderService gère les commandes [S1]. Elle valide le stock.", "subscribers": 1}
    # retour : tout le texte reçu entre-temps
    remounted = out["remounted"]
    assert remounted["users"] == ["Que fait OrderService ?"]
    assert [a["text"] for a in remounted["assistants"]] == ["OrderService gère les commandes [S1]. Elle valide le stock."]
    # fin en direct : pied de message, boutons rétablis, plus de « … » dans la liste, flux gardé jusqu'au rechargement
    after_done = out["afterDone"]
    assert after_done["running"] is False and after_done["done"] is True and after_done["messageId"] == 2 and after_done["kept"] is True
    assert _plain(after_done["assistants"][0]["foot"]) == "Entrée : 1 200 tokensRéponse : 18 tokenscontexte 300 / 3 000 tokens — 1 source sur 1 retenue"
    assert after_done["stopHidden"] is True and after_done["sendDisabled"] is False
    assert after_done["list"][0]["busy"] is False and after_done["statusRefreshes"] == 1
    # rechargement : la base contient la réponse → elle est affichée (une seule fois) et le flux est purgé
    reload = out["afterReload"]
    assert reload["purged"] is True
    assert reload["users"] == ["Que fait OrderService ?"]
    assert [a["text"] for a in reload["assistants"]] == ["OrderService gère les commandes [S1]. Elle valide le stock."]
    # message de la base : son pied revient de la mémoire locale remplie au done (voir test_footer_survives_reload…)
    assert _plain(reload["assistants"][0]["foot"]) == "Entrée : 1 200 tokensRéponse : 18 tokenscontexte 300 / 3 000 tokens — 1 source sur 1 retenue"
    assert reload["footers"] == ["/2"]

    # C. « Arrêter » : signal coupé, flux terminé `aborted`, encadré, boutons rétablis
    before = out["beforeAbort"]
    assert before["assistants"][0]["text"] == "La facturation" and before["stopHidden"] is False
    assert before["list"] == [{"title": "OrderService", "busy": False, "active": False}, {"title": "Facturation", "busy": True, "active": True}]
    abort = out["abort"]
    assert abort["signalAborted"] is True and abort["aborted"] is True and abort["running"] is False
    assert abort["lastEvent"] == "end" and abort["runningIds"] == [] and abort["text"] == "La facturation"
    assert abort["assistants"][0]["alerts"] == ["Génération arrêtée."] and abort["assistants"][0]["text"] == "La facturation"
    assert abort["stopHidden"] is True and abort["sendDisabled"] is False and abort["list"][1]["busy"] is False

    # D. rechargement d'un flux terminé que la base n'a pas : rejoué une dernière fois, puis purgé (la base fait foi ensuite)
    abort_reload = out["abortReload"]
    assert abort_reload["purged"] is True
    assert abort_reload["assistants"][0]["text"] == "La facturation" and abort_reload["assistants"][0]["alerts"] == ["Génération arrêtée."]
    assert out["abortReloadAgain"]["assistants"] == [] and out["abortReloadAgain"]["users"] == ["Explique la facturation"]

    # E. envoi depuis la vue : POST délégué au magasin, un second envoi pendant la réponse est ignoré
    send = out["send"]
    assert send["posts"] == [["/api/conversations/7/messages", {"content": "Que fait OrderService ?", "mode": "auto"}],
                             ["/api/conversations/8/messages", {"content": "Explique la facturation", "mode": "no_tools"}],
                             ["/api/conversations/7/messages", {"content": "Et ensuite ?", "mode": "auto"}]]
    assert send["users"] == ["Que fait OrderService ?", "Et ensuite ?"] and send["streamsOpened"] == 3
    assert [a["text"] for a in send["assistants"]] == ["OrderService gère les commandes [S1]. Elle valide le stock.", ""]
    assert send["assistants"][1]["status"] == ["Recherche dans le projet…"]
    assert send["runningIds"] == ["7"] and send["stopHidden"] is False and send["sendDisabled"] is True
    assert send["inputValue"] == "Encore ?" and send["list"][0]["busy"] is True
    # notifications de début / fin (pastille de la barre latérale, liste)
    assert out["changes"] == [["7"], [], ["8"], [], ["7"]]


def test_footer_with_latency_survives_reload_from_local_memory():
    """Le pied vu au done (tokens, contexte, lecture, première lettre, plafonnement) est identique après rechargement
    de la conversation depuis la base, qui n'enregistre pas usage / budget : il vient de la mémoire locale (localStorage),
    par message ; un flux arrêté avant done (pas de message_id) ne mémorise rien."""
    out = _run_harness({**_fixtures(), "done_budget": {
        "prefill_tps": 1041.0, "first_token_s": 4.06, "latency_capped": True,
        "source": "contexte ≤ 8 000, plafonné à 3 000 tokens pour rester sous 25 s (lecture 1 041 tokens/s)"}})
    expected = ("Entrée : 1 200 tokensRéponse : 18 tokenscontexte 300 / 3 000 tokens — 1 source sur 1 retenue"
                "Lecture : 1 041 tokens/sPremière lettre : 4,1 scontexte plafonné pour tenir le délai")
    assert _plain(out["afterDone"]["assistants"][0]["foot"]) == expected
    reload = out["afterReload"]
    assert reload["purged"] is True and len(reload["assistants"]) == 1
    assert _plain(reload["assistants"][0]["foot"]) == expected
    assert reload["footers"] == ["/2"]  # espace actif inconnu dans le banc → préfixe vide ; message 2
    assert out["abortReload"]["assistants"][0]["foot"] == ""  # arrêté avant done : rien à mémoriser
