"""Tests des aides de l'interface pour un projet mal compris (chat.js, setup.js, app.css).

Trois aides, toutes en JavaScript vanille sans étape de build :

1. chat.js : bandeau « Recherche lexicale seule » tant que l'espace n'a pas de modèle d'embeddings
   (lien vers les réglages, commande à copier, masquable pour la session) ;
2. setup.js : formulaire moteur avec la case « Réflexion interne du modèle (think) », l'aide
   « 16 384 conseillé sur cette machine » du champ num_ctx (POST /api/llm/test, sinon RAM) et
   l'avertissement quand le modèle n'a pas la capacité « tools » ;
3. chat.js : pied de message lisible « contexte 2 758 / 2 758 tokens — 7 sources sur 23 retenues » et
   « n sources n'ont pas tenu dans le contexte : augmentez num_ctx ».

Deux niveaux : des vérifications statiques (textes, liaison des champs, aucune URL externe, syntaxe
des modules via ``node --check``) et, si Node ≥ 22.15 est installé, le banc
``tests/web_harness/chat_help_harness.mjs`` qui monte les vraies vues dans un DOM minimal.
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
SETUP_JS = STATIC / "views" / "setup.js"
SETTINGS_JS = STATIC / "views" / "settings.js"
APP_CSS = STATIC / "app.css"
HARNESS = ROOT / "tests" / "web_harness" / "chat_help_harness.mjs"

NBSP = re.compile("[  ]")  # séparateurs de milliers d'Intl.NumberFormat('fr-FR')


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
    """Les trois vues passent ``node --input-type=module --check`` et ne référencent aucune adresse non locale."""
    for path in (CHAT_JS, SETUP_JS, SETTINGS_JS, APP_CSS):
        for url in re.findall(r"https?://[^\s'\"()]+", path.read_text(encoding="utf-8")):
            assert re.match(r"https?://(127\.0\.0\.1|localhost)(:\d+)?", url), "URL externe dans %s : %s" % (path.name, url)
    node = _node_exe()
    if not node:
        pytest.skip("node absent : syntaxe non vérifiée")
    for path in (CHAT_JS, SETUP_JS, SETTINGS_JS):
        proc = subprocess.run([node, "--input-type=module", "--check"], input=path.read_text(encoding="utf-8"),
                              capture_output=True, text=True, timeout=30)
        assert proc.returncode == 0, "%s : %s" % (path.name, proc.stderr)


def test_chat_embeddings_banner_text_link_and_session_dismissal():
    """Le bandeau lit status.workspace.llm.embedding_model, renvoie vers #/settings, propose la commande et se masque pour la session."""
    src = CHAT_JS.read_text(encoding="utf-8")
    assert "export const EMBEDDING_PULL_COMMAND = 'ollama pull qwen3-embedding:0.6b';" in src
    assert "llm.embedding_model) return null" in src
    assert "Recherche lexicale seule." in src
    assert "Pour la recherche sémantique (questions formulées autrement que le code), ajoutez un modèle d\\'embeddings dans " in src
    assert "el('a', { href: '#/settings' }, 'Réglages › Moteur d\\'IA')" in src
    assert "(0,6 Go) — puis réindexez." in src
    assert "copyButton(EMBEDDING_PULL_COMMAND, 'Copier')" in src
    assert "window.sessionStorage" in src and "Masquer pour cette session" in src
    assert "embeddingsBanner(state.status)," in src  # monté au-dessus du fil


def test_chat_model_banner_announces_a_pending_reload_with_a_reload_button():
    """Bandeau « Modèle … » (GET /api/llm/state) : chargé en N s ; chargé avec une autre fenêtre que celle du chat
    (``reload_pending``) → pastille orange, « chargé avec 32 768 ≠ 40 960 : rechargement au prochain message » et
    bouton « Recharger » (→ onLoad, comme « Charger le modèle ») à côté de « Décharger » ; sans ``loaded_num_ctx``
    (Ollama ancien) jamais d'alerte ; chargement, non chargé et moteur sans information inchangés."""
    out = _run_harness({})["modelBanner"]
    loaded = out["loaded"]
    assert loaded["class"] == "chat-model chat-model-loaded" and loaded["dots"] == ["dot dot-ok"]
    assert _plain(loaded["text"]).startswith("Modèle qwen3:14bchargé en 12,3 s") and "≠" not in loaded["text"]
    assert [b["text"] for b in loaded["buttons"]] == ["Décharger"] and loaded["note"] == []

    pending = out["reloadPending"]
    assert pending["class"] == "chat-model chat-model-loaded chat-model-reload" and pending["dots"] == ["dot dot-warn"]
    assert _plain(pending["text"]).startswith("Modèle qwen3:14bchargéavec 32 768 ≠ 40 960 : rechargement au prochain message")
    assert "chargé en" not in pending["text"]  # la durée du dernier chargement ne vaut plus : le modèle sera rechargé
    assert [b["text"] for b in pending["buttons"]] == ["Recharger", "Décharger"]
    reload_btn = pending["buttons"][0]
    assert reload_btn["primary"] and "40 960" in _plain(reload_btn["title"]) and "10 à 40 s" in reload_btn["title"]
    assert len(pending["note"]) == 1 and "32 768" in _plain(pending["note"][0]) and "rechargera le modèle au prochain message" in pending["note"][0]
    assert out["clicks"] == ["load", "unload"]  # « Recharger » = onLoad (POST /api/llm/load), « Décharger » = onUnload

    assert out["reloadWithoutWindow"]["class"] == "chat-model chat-model-loaded" and "≠" not in out["reloadWithoutWindow"]["text"]
    assert out["loading"]["progress"] == 1 and "chargement en mémoire… 10 s (en général 10 à 40 s)" in out["loading"]["text"]
    assert [b["text"] for b in out["unloaded"]["buttons"]] == ["Charger le modèle"] and "n'a pas pu charger" in out["unloaded"]["text"]
    assert out["unknown"]["class"] == "chat-model chat-model-unknown" and out["none"] is None


def test_chat_footer_reads_done_budget_in_plain_words():
    """Le pied de message garde les compteurs de done.budget et les présente en français lisible."""
    src = CHAT_JS.read_text(encoding="utf-8")
    assert "budgetFooter(budget, initialSources, truncated)" in src
    assert "initialSources = sources.length;" in src  # sources de la recherche initiale, avant les ajouts des outils
    assert "`contexte ${formatNumber(b.context_tokens_used || 0)} / ${formatNumber(b.max_context_tokens || b.num_ctx)} tokens — ${sources}`" in src
    assert "pas tenu dans le contexte : " in src and "'augmentez num_ctx'" in src
    assert "non consultée(s), budget atteint" not in src  # ancien libellé remplacé
    css = APP_CSS.read_text(encoding="utf-8")
    assert ".msg-foot .budget-truncated" in css and ".chat-main > .chat-banner" in css


def test_chat_footer_shows_reading_speed_first_letter_delay_and_latency_cap():
    """Le pied de message affiche la latence de done.budget (prefill_tps, first_token_s, latency_capped) et la garde
    en mémoire locale par message (rememberFooter / recallFooter) pour la réafficher au rechargement de la conversation."""
    src = CHAT_JS.read_text(encoding="utf-8")
    for key in ("b.prefill_tps", "b.first_token_s", "b.latency_capped"):
        assert key in src, key
    assert "`Lecture : ${formatNumber(Math.round(Number(b.prefill_tps)))} tokens/s`" in src
    assert "`Première lettre : ${formatSeconds(b.first_token_s)}`" in src
    assert "'contexte plafonné pour tenir le délai'" in src
    # en direct comme au rechargement, le même contenu de pied ; mémorisé au done (rejeu compris), relu dans assistantMessage
    assert "replace(foot, footContent(usage, budget, initialSources, truncated));" in src
    assert "rememberFooter(data.message_id, { usage, budget, retained: initialSources, truncated });" in src
    assert "const remembered = recallFooter(message.id);" in src
    assert "const FOOTERS_KEY = 'develop.chat.footers';" in src and "const FOOTERS_MAX = 200;" in src
    assert "window.localStorage.getItem(FOOTERS_KEY)" in src and "window.localStorage.setItem(FOOTERS_KEY, JSON.stringify(list))" in src

    out = _run_harness({"profile": {"ram_total_gb": 32}})
    latency = out["footerLatency"]
    assert [n["class"] for n in latency] == ["budget-line", "budget-truncated", "budget-latency", "budget-latency", "budget-capped"]
    assert [_plain(n["text"]) for n in latency] == [
        "contexte 3 469 / 3 480 tokens — 21 sources sur 22 retenues",
        "1 source n'a pas tenu dans le contexte : augmentez num_ctx",
        "Lecture : 1 041 tokens/s",
        "Première lettre : 4,1 s",
        "contexte plafonné pour tenir le délai",
    ]
    assert _plain(latency[4]["title"]) == "contexte ≤ 8 000, plafonné à 3 480 tokens pour rester sous 25 s (lecture 1 041 tokens/s)"  # explication du calcul
    assert "Vitesse de lecture du prompt" in latency[2]["title"] and "premier fragment" in latency[3]["title"]
    # sans mesure (first_token_s null, pas de plafonnement) : seule la vitesse retenue
    assert [_plain(t) for t in out["footerLatencyNoMeasure"]] == ["contexte 300 / 3 000 tokens — 1 source sur 1 retenue", "Lecture : 60 tokens/s"]
    assert out["seconds"] == ["4,1 s", "0,4 s", "10 s", "192 s", "0 s", "—", "—"]
    # mémoire locale : par identifiant de message, remplacée au second done, sans identifiant ignorée, sources écartées réduites
    memory = out["footerMemory"]
    assert memory["recalled"]["usage"] == {"prompt_tokens": 4712, "completion_tokens": 75} and memory["recalled"]["retained"] == 21
    assert memory["recalled"]["budget"]["prefill_tps"] == 1041.0 and memory["recalled"]["budget"]["first_token_s"] == 4.06
    assert memory["recalled"]["truncated"] == [] and memory["unknown"] is None and memory["none"] is None
    assert memory["keys"] == ["/42", "/43"] and memory["truncatedKeys"] == ["path", "title", "ref"]
    # bornée aux 200 derniers messages ; stockage indisponible : aucune erreur, le pied reste connu de la page en cours
    assert out["footerMemoryBounded"] == {"count": 200, "first": "/200", "last": "/399", "evicted": True}
    assert out["footerMemoryUnavailable"] is True


def test_engine_form_binds_think_num_ctx_advice_and_tools_warning():
    """Case think liée à llm.think, aide num_ctx (test puis RAM) et avertissement « sans outils » dans setup.js."""
    src = SETUP_JS.read_text(encoding="utf-8")
    assert "Réflexion interne du modèle (think) : plus précis, 2 à 3 fois plus lent" in src
    assert "llm.think = !!thinkInput.checked;" in src
    assert "think: false" in src  # brouillon neuf de l'assistant
    assert "conseillé sur cette machine" in src
    assert "testAdvice = Number(r.suggested_num_ctx)" in src
    assert "ramAdvice = adviseNumCtx(ramGb)" in src and "p.ram_total_gb" in src
    assert "if (ram >= 32) return 16384;" in src and "if (ram >= 16) return 8192;" in src
    assert "info.supports_tools !== false" in src
    assert ("Ce modèle ne sait pas appeler les outils de Develop : les réponses s\\'appuieront seulement sur la "
            "recherche initiale ; préférez un modèle avec outils (qwen3.5:9b, gemma4:12b, devstral)." in src)
    # l'écran Réglages envoie l'objet llm complet (think compris) à PUT /api/workspaces
    assert "saveButton(ws.id, () => ({ llm: draft.llm }))" in SETTINGS_JS.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- fonctionnel (Node)

def test_rendered_help_with_model_without_tools_and_test_suggestion():
    """Modèle testé sans capacité tools, suggestion 16 384 : bandeau, pied de message, case think, aide et avertissement."""
    out = _run_harness({
        "profile": {"ram_total_gb": 32},
        "llm_test": {"ok": True, "model_info": {"name": "deepseek-coder-v2:16b", "supports_tools": False, "context_length": 163840},
                     "suggested_num_ctx": 16384},
        "budget": {"num_ctx": 8192, "context_tokens_used": 2758, "max_context_tokens": 2758, "source": "num_ctx 8192 = min(…)"},
        "retained": 7,
        "truncated": [{"path": "app/Models/User.php"}] * 16,
    })
    # 3. pied de message
    footer = out["footer"]
    assert _plain(footer[0]["text"]) == "contexte 2 758 / 2 758 tokens — 7 sources sur 23 retenues"
    assert footer[0]["title"] == "num_ctx 8192 = min(…)"  # l'explication du calcul reste en info-bulle
    assert footer[1]["text"] == "16 sources n'ont pas tenu dans le contexte : augmentez num_ctx"
    assert footer[1]["links"] == ["#/settings"]
    assert [_plain(t) for t in out["footerSingular"]] == ["contexte 300 / 900 tokens — 1 source sur 2 retenue",
                                                          "1 source n'a pas tenu dans le contexte : augmentez num_ctx"]
    assert out["footerEmpty"] == 0 and _plain(out["footerNoSource"][0]) == "contexte 0 / 900 tokens — aucune source"
    # 1. bandeau
    banner = out["banner"]
    assert banner["class"] == "alert alert-info chat-banner"
    assert banner["text"].startswith("Recherche lexicale seule. Pour la recherche sémantique (questions formulées autrement que le code), "
                                     "ajoutez un modèle d'embeddings dans Réglages › Moteur d'IA — par exemple ollama pull qwen3-embedding:0.6b")
    assert banner["text"].endswith("(0,6 Go) — puis réindexez.")
    assert banner["links"] == ["#/settings"] and banner["code"] == ["ollama pull qwen3-embedding:0.6b"] and banner["copyButtons"] == 1
    assert out["bannerAfterDismiss"] == {"attached": False, "again": False, "session": {"develop.chat.embeddingsBannerDismissed": "1"}}
    assert out["bannerNoEngine"] is False and out["bannerWithEmbeddings"] is False and out["bannerNoStatus"] is False
    # 2. formulaire moteur
    assert out["advice"] == {"64": 16384, "32": 16384, "24": 8192, "16": 8192, "8": 4096, "0": None, "null": None, "x": None}
    assert out["toolsWarning"]["withTools"] is None and out["toolsWarning"]["unknown"] is None
    assert "Ce modèle ne sait pas appeler les outils de Develop" in out["toolsWarning"]["without"]
    form = out["form"]
    assert form["thinkLabel"] == "Réflexion interne du modèle (think) : plus précis, 2 à 3 fois plus lent" and form["thinkChecked"] is False
    assert _plain(form["ctxHint"]).startswith("16 384 conseillé sur cette machine (fenêtre du modèle et mémoire de la machine).")
    assert [_plain(t) for t in form["ctxUseButton"]] == ["Utiliser 16 384"]
    assert len(form["toolsWarning"]) == 1 and "deepseek-coder-v2:16b" in form["toolsWarning"][0]
    # la sonde silencieuse teste le modèle configuré sans embeddings
    assert form["llmTestRequests"] == [{"engine": "ollama", "base_url": "http://127.0.0.1:11434", "model": "deepseek-coder-v2:16b",
                                        "embedding_model": "", "api_key": ""}]
    # cocher think et « Utiliser 16 384 » modifient l'objet llm envoyé ensuite à PUT /api/workspaces
    assert out["formAfter"]["think"] is True and out["formAfter"]["num_ctx"] == 16384 and out["formAfter"]["ctxUseButton"] == 0


def test_rendered_help_falls_back_to_ram_when_engine_is_unreachable():
    """Sans suggestion du test (moteur injoignable), l'aide num_ctx vient de la RAM et aucun avertissement outils n'apparaît."""
    out = _run_harness({
        "profile": {"ram_total_gb": 16, "unified_memory": True},
        "llm_test": {"ok": False, "model_info": None, "suggested_num_ctx": None, "error": "Moteur injoignable."},
        "status": {"workspace": {"llm": {"engine": "ollama", "embedding_model": ""}}},
    })
    form = out["form"]
    assert _plain(form["ctxHint"]).startswith("8 192 conseillé sur cette machine (16 Go de mémoire).")
    assert form["ctxUseButton"] == [] and form["toolsWarning"] == []  # 8 192 déjà saisi : pas de bouton ; modèle inconnu : pas d'alerte
    assert out["banner"] is not None


def test_rendered_help_without_engine_runs_no_probe():
    """Moteur « Aucun » : pas de sonde, champs désactivés, pas de bandeau embeddings dans le chat."""
    out = _run_harness({
        "profile": {"ram_total_gb": 64},
        "llm": {"engine": "none", "base_url": "", "model": "", "embedding_model": "", "api_key": "", "num_ctx": 8192, "think": False},
        "status": {"workspace": {"llm": {"engine": "none", "embedding_model": ""}}},
    })
    assert out["form"]["llmTestRequests"] == [] and out["form"]["toolsWarning"] == []
    assert _plain(out["form"]["ctxHint"]).startswith("16 384 conseillé sur cette machine (64 Go de mémoire).")
    assert out["banner"] is None
