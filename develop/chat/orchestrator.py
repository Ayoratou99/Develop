"""Orchestrateur du chat : recherche → budget → contexte → boucle d'outils bornée → événements SSE.

Pour chaque question, :meth:`ChatOrchestrator.respond` produit une suite d'événements
(:class:`ChatEvent`) : ``status``, ``sources``, ``tool_call``/``tool_result`` (au plus
``Budget.tool_rounds_max`` tours), ``delta`` puis ``done`` — ou ``error`` si le moteur est
injoignable (les ``sources`` ont alors déjà été émises). Chaque événement reste petit : les
textes complets des sources ne quittent jamais le serveur, seules leurs métadonnées sont envoyées.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import statistics
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Set, Tuple

from develop.chat.budget import (MIN_TOOL_BUDGET, Budget, compute_budget, estimate_tokens, read_time_s,
                                 tool_specs_tokens)
from develop.chat.context_builder import (ContextPack, LocalHit, Source, add_tool_sources, build_context,
                                          render_sources_as_text, render_sources_for_model)
from develop.chat.prompts import HISTORY_SUMMARY_PREFIX, build_system_prompt
from develop.chat.tools import TOOL_SPECS, ToolExecutor, ToolResult
from develop.llm.advisor import PREFILL_TPS_MAX, PREFILL_TPS_MIN, estimate_prefill_tps, model_params_b
from develop.llm.base import (ChatMessage, LLMClient, LLMError, LLMUnavailable, ModelInfo, ModelNotFound, ToolCall,
                              model_is_loaded, parse_text_tool_calls)
from develop.llm.offline_guard import OfflineViolation
from develop.search.query import FOLLOWUP_MARKER  # le marqueur est reconnu par analyze_query : même source de vérité
from develop.storage import Database, StorageError

logger = logging.getLogger("develop.chat")

MAX_CALLS_PER_ROUND = 3
MESSAGE_OVERHEAD_TOKENS = 8  # rôle, délimiteurs du gabarit du moteur, identifiant d'appel
USER_FRAME_TOKENS = 32  # « Sources disponibles (cite-les avec [Sn]) : », sauts de ligne entre blocs et arrondis
SOURCES_FRAME_TOKENS = 16  # « Nouvelles sources citables : » et sauts de ligne entre les blocs
TEXT_RESULT_PREFIX = "Résultat de l'outil %s :\n"
INTERRUPTED_PREFIX = "[réponse interrompue] "
TOOLS_EXHAUSTED_TEXT = "Budget d'outils épuisé : réponds maintenant avec ce que tu as, en citant les sources disponibles."
NO_ANSWER_TEXT = "Le modèle n'a renvoyé aucune réponse."
STATUS_SEARCHING = "Recherche dans le projet…"
STATUS_FOLLOWUP = "Question de suivi : contexte « %s » conservé"
STATUS_SMALLTALK = "Petite conversation : réponse directe, sans recherche ni modèle."
STATUS_LOADING = "Chargement du modèle %s en mémoire (jusqu'à 30 s la première fois)…"
STATUS_READING = "Le modèle lit %d sources (≈ %d tokens, ~%d s estimées) avant de répondre…"
# Vitesse de préremplissage mesurée (tokens du prompt *non cachés* ÷ délai jusqu'au premier fragment de texte), persistée
# par modèle dans la base : meta ``llm_prefill_tps:<modèle>`` = {"engine", "v", "samples": [≤ PREFILL_SAMPLES], "median",
# "avg", "updated_at"}. Le moteur garde en cache le préfixe commun avec ses requêtes précédentes (Ollama : cache KV par
# slot) et ``usage.prompt_tokens`` compte pourtant tout le prompt : la part en cache est estimée par le plus long
# préfixe commun avec les derniers prompts envoyés à ce modèle et retirée du compte, sinon une question reposée dans
# une nouvelle conversation (prompt entièrement en cache, première lettre en < 1 s) vaudrait 2 000 tok/s et la borne de
# latence lâcherait pour les questions suivantes. La valeur retenue est la médiane (basse) des mesures : une mesure
# aberrante isolée (cache d'un autre slot, redémarrage de l'outil, machine occupée) ne la déplace pas.
PREFILL_META_KEY = "llm_prefill_tps:%s"
PREFILL_META_VERSION = 2  # les mesures d'avant la correction du cache (v1, gonflées) ne sont pas relues
PREFILL_SAMPLES = 5  # médiane des 5 dernières mesures
PREFILL_MIN_PROMPT_TOKENS = 256  # en dessous (tokens non cachés), la mesure est du bruit (latence du moteur, pas sa vitesse)
PREFILL_LOAD_MAX_S = 1.0  # au-delà, l'appel a (re)chargé le modèle (Ollama : ``load_duration``) : son délai n'est pas une lecture
PREFILL_RECENT_PROMPTS = 4  # prompts gardés par modèle pour estimer la part en cache (≈ slots parallèles du moteur)
# Question « de suivi » (enrichie du tour précédent) : elle s'ouvre sur un mot de liaison (« et les champs ? »,
# « ok mais pourquoi ? ») ou ne nomme aucun sujet à elle — ni identifiant, chemin, route ou table, ni mot-clé autre
# qu'un mot d'*aspect* (« champs », « validation », « définition »… : une facette du sujet précédent, pas un sujet).
# Une question qui nomme son sujet (« l'api d'authentification », « confirmer une commande », « OrderService::confirm »)
# est autonome, quelle que soit sa longueur : une conversation change de sujet à chaque tour ou presque.
FOLLOWUP_OPENERS = frozenset("et mais ok okay donc puis ensuite aussi alors sinon oui non encore idem pareil egalement "
                             "d'accord daccord and but also then so".split())
FOLLOWUP_ASPECT_WORDS = frozenset("""
champ champs definition definitions validation validations verification verifications controle controles regle regles
contrainte contraintes route routes endpoint endpoints url urls chemin chemins migration migrations seeder seeders
relation relations lie liee lies liees lien liens dependance dependances appelant appelants methode methodes fonction fonctions attribut
attributs propriete proprietes colonne colonnes table tables donnee donnees nom noms type types valeur valeurs format
formats taille limite limites obligatoire obligatoires requis optionnel optionnels facultatif facultatifs nullable
code structure schema implementation signature parametre parametres argument arguments retour resultat resultats
entree entrees sortie sorties requete requetes reponse reponses message messages erreur erreurs statut statuts
contenu logique comportement mecanisme processus flux etape etapes ordre sequence condition conditions cas suite
reste liste listes test tests difference differences usage documentation page pages vue vues controleur controleurs
service services classe classes fichier fichiers permission permissions droit droits autorisation autorisations
politique politiques policy
""".split())
FOLLOWUP_MAX_TERMS = 8
FOLLOWUP_SOURCES = 3  # titres / chemins des premières sources de la réponse précédente
FOLLOWUP_MAX_PASSAGES = 3  # passages des sources précédentes remontés devant la recherche pour une question de suivi
FOLLOWUP_MIN_KEYWORD_LENGTH = 3
FOLLOWUP_CHANNEL = "followup"
FOLLOWUP_REASON = "suite de la conversation : %s"
USEFUL_TEXT_MIN_CHARS = 160

# Petites conversations (≤ 3 mots) : réponse immédiate, sans recherche ni LLM.
SMALLTALK_GREETINGS = frozenset("bonjour bonsoir salut hello hi hey coucou yo re rebonjour".split())
SMALLTALK_THANKS = frozenset("merci thanks thank thx".split())
# acquiescements : ils closent un échange… sauf quand le modèle vient de poser une question (ils y répondent)
SMALLTALK_ACKS = frozenset("ok okay oki d'accord daccord dac compris parfait super top cool genial nickel bien".split())
# oui / non : des réponses, adressées au modèle dès qu'il a parlé (« non » après une réponse la conteste)
SMALLTALK_YES = frozenset("oui yes ouais yep si".split())
SMALLTALK_NO = frozenset("non no nope nan".split())
SMALLTALK_ANSWERS = SMALLTALK_YES | SMALLTALK_NO
# phrase propre aux réponses de petite conversation : une telle réponse n'appelle rien de l'utilisateur
SMALLTALK_SIGNATURE = "Je suis l'assistant de Develop pour l'espace de travail"
SMALLTALK_WELLBEING = frozenset(("ca va", "ça va", "comment ca va", "comment ça va", "comment vas tu", "comment vas-tu",
                                 "tu vas bien", "how are you"))
SMALLTALK_FAREWELLS = frozenset(("bonne journee", "bonne soiree", "bonne nuit", "bon week-end", "bon weekend", "au revoir",
                                 "a bientot", "a plus", "a plus tard", "bye", "ciao", "good bye", "goodbye"))
SMALLTALK_FILLERS = frozenset("beaucoup bien tout le la a à toi vous bonne journee soiree et ça ca va tres très "
                              "monde develop".split())
SMALLTALK_MAX_WORDS = 3

_CITATION_RE = re.compile(r"\[S(\d+)\]")
_TOOL_LINE_RE = re.compile(r'^\s*`*\s*\{\s*"tool"\s*:.*\}\s*`*\s*$')  # ligne d'appel d'outil complète
_TOOL_HEAD_RE = re.compile(r'^\s*`*\s*\{\s*"tool"\s*:')  # ligne d'appel d'outil inachevée (flux coupé)
_TOOL_START_RE = re.compile(r"^\s*`*\s*\{")  # ligne entamée qui peut encore devenir un appel d'outil
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$")  # clôture de bloc de code
_FENCE_HEAD_RE = re.compile(r"^\s*`{1,3}[a-zA-Z]*\s*$")  # clôture éventuellement inachevée


@dataclass
class ChatEvent:
    """Un événement SSE du chat (``type`` + ``data`` sérialisable en JSON)."""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        return "event: %s\ndata: %s\n\n" % (self.type, payload)


def _event(type_: str, **data: Any) -> ChatEvent:
    return ChatEvent(type_, dict(data))


def _remove_tool_lines(text: str, drop_incomplete_tail: bool = False, in_fence: bool = False) -> Tuple[str, bool]:
    """Texte sans ses lignes d'appel d'outil ni les clôtures ``` qui les encadrent, mise en page conservée.

    Les blocs de code légitimes restent intacts : seules la clôture qui ouvre un bloc juste avant une ligne
    d'outil et celle qui le referme juste après sont retirées. ``in_fence`` dit si le texte commence à
    l'intérieur d'un bloc de code (état rendu en second, pour filtrer un flux morceau par morceau). Avec
    ``drop_incomplete_tail``, une dernière ligne d'outil inachevée (flux coupé) est retirée aussi.
    """
    lines = (text or "").splitlines(keepends=True)
    kept: List[str] = []
    close_pending = False  # la clôture ouvrant un bloc d'outil a été retirée : sa fermante le sera aussi
    for i, line in enumerate(lines):
        is_tool = bool(_TOOL_LINE_RE.match(line))
        if not is_tool and drop_incomplete_tail and i == len(lines) - 1:
            is_tool = bool(_TOOL_HEAD_RE.match(line))
        if is_tool:
            if in_fence and kept and _FENCE_RE.match(kept[-1]):
                kept.pop()
                in_fence = False
                close_pending = True
            continue
        if _FENCE_RE.match(line):
            if close_pending and line.strip() == "```":
                close_pending = False
                continue
            in_fence = not in_fence
        if line.strip():
            close_pending = False
        kept.append(line)
    return "".join(kept), in_fence


def _strip_tool_lines(text: str) -> str:
    """Texte restant une fois retirées les lignes d'appel d'outil (et leurs clôtures), espaces de bord ôtés."""
    return _remove_tool_lines(text)[0].strip()


def _may_be_tool_block(line: str, partial: bool) -> bool:
    """Vrai si la ligne peut appartenir à un bloc d'appel d'outil : ligne JSON, clôture ``` ou ligne vide.

    Pour une ligne encore inachevée (``partial``), on regarde si elle peut le devenir.
    """
    if not line.strip():
        return True
    if partial:
        return bool(_TOOL_START_RE.match(line) or _FENCE_HEAD_RE.match(line))
    return bool(_TOOL_LINE_RE.match(line) or _FENCE_RE.match(line))


class _ToolLineFilter:
    """Filtre du flux en mode texte : aucune ligne d'appel d'outil (ni sa clôture ```) n'est émise.

    Le texte part au fil de l'eau, sauf sa fin tant qu'elle peut encore former un bloc d'appel d'outil
    (ligne JSON entamée, clôture, lignes vides) : elle est retenue jusqu'à ce qu'une ligne ordinaire la
    suive ou que le flux se termine. Le modèle qui écrit une réponse utile puis la ligne d'outil voit donc
    sa réponse affichée et l'outil exécuté, sans que la ligne JSON n'atteigne l'utilisateur.
    """

    def __init__(self) -> None:
        self.pending = ""
        self.in_fence = False

    def feed(self, text: str, release: bool = True) -> str:
        """Ajoute ``text`` ; renvoie la partie devenue sûre à émettre (lignes d'outil retirées), sinon ``""``."""
        self.pending += text
        if not release or not self.pending:
            return ""
        lines = self.pending.split("\n")
        k = len(lines)
        while k > 0 and _may_be_tool_block(lines[k - 1], partial=(k == len(lines))):
            k -= 1
        if k == 0:
            return ""
        if k == len(lines):
            decided, self.pending = self.pending, ""
        else:
            decided = "\n".join(lines[:k]) + "\n"
            self.pending = "\n".join(lines[k:])
        out, self.in_fence = _remove_tool_lines(decided, in_fence=self.in_fence)
        return out

    def flush(self) -> str:
        """Fin du flux : ce qui restait retenu, sans ligne d'outil (même inachevée)."""
        rest, self.pending = self.pending, ""
        out, self.in_fence = _remove_tool_lines(rest, drop_incomplete_tail=True, in_fence=self.in_fence)
        return out

    def discard(self) -> None:
        """Oublie ce qui était retenu (tour d'outil pur, éventuellement précédé d'une courte préface)."""
        self.pending = ""


def _has_useful_text(text: str) -> bool:
    rest = _strip_tool_lines(text)
    return bool(_CITATION_RE.search(rest)) or len(rest) >= USEFUL_TEXT_MIN_CHARS


def _could_be_tool_round(text: str) -> bool:
    """Tant que c'est vrai, les deltas sont retenus (mode texte) : la réponse peut être un appel d'outil."""
    s = (text or "").lstrip()
    if not s:
        return True
    if s[0] in "{`":
        return True
    return not _has_useful_text(s)


def _message_tokens(msg: ChatMessage) -> int:
    n = estimate_tokens(msg.content or "")
    if msg.tool_calls:
        n += sum(estimate_tokens(json.dumps(c.arguments, ensure_ascii=False)) + 8 for c in msg.tool_calls)
    return n


def _prefill_samples(db: Any, engine: str, model: str) -> List[float]:
    """Mesures enregistrées pour ``model`` (même moteur), les plus anciennes d'abord ; ``[]`` si rien d'exploitable."""
    if db is None or not model:
        return []
    try:
        raw = db.get_meta(PREFILL_META_KEY % model)
    except Exception as exc:  # noqa: BLE001 - base fermée : comme sans mesure
        logger.debug("mesures de préremplissage illisibles pour %s: %s", model, exc)
        return []
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    if data.get("v") != PREFILL_META_VERSION:
        return []  # mesures d'un ancien format (prompt entier compté, cache compris) : gonflées, on repart de zéro
    stored_engine = str(data.get("engine") or "")
    if engine and stored_engine and stored_engine != engine:
        return []  # même nom de modèle sur un autre moteur : rien de comparable
    samples: List[float] = []
    for value in data.get("samples") or []:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f > 0:
            samples.append(f)
    return samples[-PREFILL_SAMPLES:]


def _prefill_retained(samples: Sequence[float]) -> float:
    """Valeur retenue d'une série de mesures : la médiane basse (la plus petite des deux centrales quand le nombre
    est pair). Une mesure isolée trop haute (prompt en cache non détecté) ou trop basse (machine occupée) ne la
    déplace pas, et entre deux candidates c'est la prudente : surestimer la vitesse coûte des minutes d'attente,
    la sous-estimer quelques sources."""
    return round(float(statistics.median_low(sorted(samples))), 1)


def load_prefill_tps(db: Any, engine: str, model: str) -> Optional[float]:
    """Vitesse de préremplissage mesurée (tokens/s, médiane des :data:`PREFILL_SAMPLES` dernières mesures) de
    ``model`` sur ``engine``, lue dans la base ; ``None`` sans mesure."""
    samples = _prefill_samples(db, engine, model)
    if not samples:
        return None
    return _prefill_retained(samples)


def record_prefill_tps(db: Any, engine: str, model: str, tps: float) -> Optional[float]:
    """Ajoute une mesure (bornée entre :data:`PREFILL_TPS_MIN` et :data:`PREFILL_TPS_MAX`) aux mesures de ``model``
    dans la base ; renvoie la nouvelle valeur retenue (médiane, :func:`_prefill_retained`), ``None`` si la mesure
    est invalide ou la base absente."""
    try:
        value = float(tps)
    except (TypeError, ValueError):
        return None
    if db is None or not model or not value > 0:
        return None
    value = min(PREFILL_TPS_MAX, max(PREFILL_TPS_MIN, value))
    samples = (_prefill_samples(db, engine, model) + [round(value, 1)])[-PREFILL_SAMPLES:]
    median = _prefill_retained(samples)
    avg = round(sum(samples) / len(samples), 1)
    try:
        db.set_meta(PREFILL_META_KEY % model, json.dumps({"engine": engine, "v": PREFILL_META_VERSION, "samples": samples,
                                                         "median": median, "avg": avg, "updated_at": time.time()}))
    except Exception as exc:  # noqa: BLE001 - jamais de réponse perdue pour une statistique
        logger.debug("mesure de préremplissage non enregistrée pour %s: %s", model, exc)
        return None
    return median


def find_invalid_citations(text: str, valid_refs: Sequence[str]) -> List[str]:
    """Références [Sn] citées mais absentes des sources, dans l'ordre d'apparition, sans doublon."""
    valid = set(valid_refs)
    out: List[str] = []
    for m in _CITATION_RE.finditer(text or ""):
        ref = "S%s" % m.group(1)
        if ref not in valid and ref not in out:
            out.append(ref)
    return out


# --------------------------------------------------------------------------- petites conversations

def _fold(text: str) -> str:
    """Minuscules sans accents ni ponctuation, espaces normalisés (« Ça va ?! » → « ca va »)."""
    s = unicodedata.normalize("NFKD", str(text or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"[^\w' -]+", " ", s).replace("’", "'")
    return " ".join(s.split())


_TRAILING_DECOR_RE = re.compile(r"(\s|\[S\d+\]|[*_`)\]])+$")  # citations, emphase, parenthèses en fin de ligne


def asks_a_question(text: str) -> bool:
    """Vrai quand la dernière ligne non vide de ``text`` se termine par un point d'interrogation hors guillemets :
    « Voulez-vous le détail des routes ? [S1] » en pose une, un exemple cité « que fait GET /orders ? » non."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return False
    return _TRAILING_DECOR_RE.sub("", lines[-1]).endswith("?")


def _formula_words(text: str) -> List[str]:
    """Mots de ``text`` quand il a la forme d'une formule courte (≤ :data:`SMALLTALK_MAX_WORDS` mots, le premier
    salutation, remerciement, acquiescement ou oui / non, les autres connus ou de liaison), sinon ``[]`` :
    « ok mais pourquoi ? » n'en est pas une."""
    words = _fold(text).split()
    if not words or len(words) > SMALLTALK_MAX_WORDS:
        return []
    known = SMALLTALK_GREETINGS | SMALLTALK_THANKS | SMALLTALK_ACKS | SMALLTALK_ANSWERS
    if words[0] not in known or not all(w in known or w in SMALLTALK_FILLERS for w in words):
        return []
    return words


def is_reply_to_model(text: str, previous_answer: str) -> bool:
    """Vrai pour une formule courte qui s'adresse au modèle plutôt qu'à l'assistant : « oui », « non », « non merci »
    dès que le modèle a répondu (``previous_answer``, sa dernière réponse), « ok », « ok top », « d'accord » quand
    cette réponse se termine par une question (« voulez-vous le détail des routes ? »). Une réponse de petite
    conversation (:data:`SMALLTALK_SIGNATURE`) n'appelle rien : rien ne lui répond."""
    words = _formula_words(text)
    if not words:
        return False
    answer = str(previous_answer or "").strip()
    if not answer or SMALLTALK_SIGNATURE in answer:
        return False
    if words[0] in SMALLTALK_ANSWERS:
        return True
    return words[0] in SMALLTALK_ACKS and asks_a_question(answer)


def is_smalltalk(text: str, previous_answer: str = "") -> bool:
    """Vrai pour une formule de politesse courte (« bonjour », « merci beaucoup », « ok », « ça va ? »).

    ``previous_answer`` est la dernière réponse de l'assistant : un « oui » / « non » / « ok top » qui lui répond
    (:func:`is_reply_to_model`) n'est pas une petite conversation, il doit parvenir au modèle. Sans ce contexte,
    « oui » ou « non » seuls, sans rien à quoi répondre, restent une petite conversation.
    """
    folded = _fold(text)
    if not folded or len(folded.split()) > SMALLTALK_MAX_WORDS:
        return False
    if folded in SMALLTALK_WELLBEING or folded in SMALLTALK_FAREWELLS:
        return True
    return bool(_formula_words(text)) and not is_reply_to_model(text, previous_answer)


def _smalltalk_opening(text: str) -> str:
    folded = _fold(text)
    words = folded.split()
    first = words[0] if words else ""
    if folded in SMALLTALK_WELLBEING:
        return "Très bien, merci ! Prêt à explorer le projet."
    if folded in SMALLTALK_FAREWELLS:
        return "À bientôt !"
    if first in SMALLTALK_THANKS:
        return "Avec plaisir !"
    if first in SMALLTALK_GREETINGS:
        return "Bonjour !"
    if first in SMALLTALK_NO:
        return "Entendu."
    return "Très bien."


def smalltalk_reply(db: Any, workspace_name: str, user_text: str = "") -> str:
    """Réponse en français aux petites conversations : ce que Develop sait faire et des exemples adaptés à l'index.

    Les exemples citent jusqu'à trois modèles de données et trois routes réellement indexés ; sans index,
    des formulations génériques sont proposées.
    """
    models: List[str] = []
    routes: List[str] = []
    try:
        for m in db.list_models() or []:
            name = str(getattr(m, "name", "") or "")
            if not name or str(getattr(m, "kind", "") or "") == "migration" or name in models:
                continue
            models.append(name)
            if len(models) >= 3:
                break
    except Exception:  # noqa: BLE001
        models = []
    try:
        for e in db.list_endpoints(side="server") or []:
            handler = str(getattr(e, "handler", "") or "")
            path = str(getattr(e, "path", "") or getattr(e, "normalized_path", "") or "")
            if not path or handler.lower() == "closure":
                continue
            label = ("%s %s" % (str(getattr(e, "method", "") or "").upper(), path if path.startswith("/") else "/" + path)).strip()
            if label not in routes:
                routes.append(label)
            if len(routes) >= 3:
                break
    except Exception:  # noqa: BLE001
        routes = []
    space = (" « %s »" % workspace_name) if workspace_name else ""
    lines = [
        "%s %s%s." % (_smalltalk_opening(user_text), SMALLTALK_SIGNATURE, space),
        "Je réponds à partir de ce qui est indexé (code, symboles, routes, modèles de données, documents, notes) en citant "
        "mes sources ; je ne connais pas le reste du projet.",
        "",
        "Je peux par exemple :",
        "- décrire un modèle de données (champs, types, relations, migrations, règles de validation, routes) ;",
        "- expliquer une route ou un endpoint et comment l'appeler ;",
        "- retrouver qui appelle une fonction, l'impact d'un changement, ou ce que dit un document.",
        "",
        "Exemples de questions pour cet espace :",
    ]
    if models:
        lines.append("- « décris-moi le modèle %s »" % models[0])
        if len(models) > 1:
            lines.append("- « quels sont les champs et la validation de %s ? »" % models[1])
        if len(models) > 2:
            lines.append("- « quelles routes manipulent %s ? »" % models[2])
    else:
        lines.append("- « décris-moi le modèle <Nom> »")
    if routes:
        lines.append("- « comment appeler %s ? »" % routes[0])
        if len(routes) > 1:
            lines.append("- « que fait %s ? »" % routes[1])
        if len(routes) > 2:
            lines.append("- « qui gère %s ? »" % routes[2])
    else:
        lines.append("- « comment appeler POST /<ressource> ? »")
    lines.append("- « qui appelle <fonction> ? » / « quel est l'impact d'un changement sur <fichier> ? »")
    return "\n".join(lines)


# --------------------------------------------------------------------------- suivi de conversation

_LABEL_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{2,}$")  # nom nu : classe, modèle (« Shipment », « AuthController »)
_LABEL_GENERIC = frozenset({"api", "app", "web", "index", "main", "routes", "config", "readme", "console", "kernel",
                            "channels", "schema", "package", "composer"})


def _source_label(source: Dict[str, Any]) -> str:
    """Terme de suivi tiré d'une source enregistrée : le titre d'un document ou d'une note, le nom nu d'un fichier
    de classe (``app/Models/Shipment.php`` → ``Shipment``), rien sinon.

    Un nom de fichier composé (``contenu_configuration``, ``specification-formulaires-commandes``,
    ``2026_06_30_create_form_templates_table``) n'est pas un terme : FTS5 le cherche comme une phrase exacte que seul ce
    fichier contient, et comme tous les termes sont exigés ensemble (ET), le canal lexical ne renvoie plus que
    ce fichier (le script qui génère un PDF, jamais le PDF). Le sujet est déjà porté par le titre du document
    et les mots-clés de la question précédente.
    """
    if not isinstance(source, dict):
        return ""
    title = str(source.get("title") or "").strip()
    path = str(source.get("path") or "")
    if source.get("source_type") in ("document", "note") or source.get("document_id") is not None or not path:
        return title
    base = os.path.basename(path.rstrip("/"))
    stem = base.split(".", 1)[0] if base else ""
    if stem and _LABEL_NAME_RE.match(stem) and stem.lower() not in _LABEL_GENERIC:
        return stem
    return ""


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> Set[str]:
    return set(_WORD_RE.findall(_fold(text)))


def _keyword_in(words: Set[str], keyword: str) -> bool:
    """Le mot-clé (replié) figure parmi ``words`` (repliés), au pluriel près (« champs » ~ « champ »)."""
    if keyword in words:
        return True
    if len(keyword) < 4:
        return False
    stem = keyword[:-1] if keyword.endswith("s") else keyword
    return stem in words or (stem + "s") in words or (stem + "x") in words


def _chunk_headings(chunk: Any) -> List[str]:
    """Titres d'un morceau : ``heading_path`` / ``section`` de ses métadonnées (documents découpés par
    sections), sinon les lignes de titre numérotées de son texte suivies de texte (jamais celles d'un sommaire,
    qui se suivent)."""
    meta = getattr(chunk, "meta", None)
    meta = meta if isinstance(meta, dict) else {}
    out = [str(h) for h in (meta.get("heading_path") or []) if str(h).strip()]
    section = meta.get("section") or meta.get("heading")
    if section and str(section) not in out:
        out.append(str(section))
    if out:
        return out
    from develop.document_indexer.chunking import numbered_heading
    lines = str(getattr(chunk, "text", "") or "").splitlines()
    for i, line in enumerate(lines):
        h = numbered_heading(line)
        if h is None:
            continue
        following = next((l for l in lines[i + 1:] if l.strip()), "")
        if following and numbered_heading(following) is None:
            out.append(h[1])
    return out


def followup_passages(db: Database, user_text: str, previous_sources: Sequence[Dict[str, Any]],
                      today: Optional[str] = None, max_passages: int = FOLLOWUP_MAX_PASSAGES) -> List[LocalHit]:
    """Passages des sources citées au tour précédent qui répondent à la question de suivi elle-même.

    Une question de suivi courte (« explique-moi la définition des champs ») vise une partie précise du sujet
    précédent. La recherche enrichie des termes du tour précédent retrouve le sujet (le document, le fichier),
    pas forcément cette partie : le sommaire d'un document contient tous les mots de la question, la section
    demandée non. On relit donc les morceaux des :data:`FOLLOWUP_SOURCES` premières sources précédentes
    (documents et fichiers) et on les classe par les mots-clés de la question seule : mots-clés présents dans
    le titre de section du morceau d'abord (« 4. Les champs (custom et partagés) »), puis nombre de mots-clés
    dans le texte, puis ordre des sources et du document. Au plus ``max_passages`` passages, distincts par
    texte (le PDF d'un dépôt et le même PDF téléversé ne comptent qu'une fois), canal :data:`FOLLOWUP_CHANNEL`.
    """
    try:
        from develop.search.query import analyze_query
        raw_keywords = list(analyze_query(str(user_text or ""), today).keywords or [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("analyse de la question de suivi impossible: %s", exc)
        return []
    keywords = [k for k in dict.fromkeys(_fold(k).replace(" ", "") for k in raw_keywords)
                if len(k) >= FOLLOWUP_MIN_KEYWORD_LENGTH]
    if not keywords or max_passages <= 0:
        return []
    targets: List[Tuple[str, int]] = []
    for source in list(previous_sources or [])[:FOLLOWUP_SOURCES]:
        if not isinstance(source, dict):
            continue
        doc_id, file_id = _as_int(source.get("document_id")), _as_int(source.get("file_id"))
        key = ("document", doc_id) if doc_id is not None else (("file", file_id) if file_id is not None else None)
        if key is not None and key not in targets:
            targets.append(key)
    scored: List[Tuple[Tuple[int, int, int, int], Any, str, int, Optional[str]]] = []
    for order, (source_type, source_id) in enumerate(targets):
        try:
            chunks = db.chunks_for(source_type, source_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug("morceaux de %s:%s illisibles: %s", source_type, source_id, exc)
            continue
        for chunk in chunks:
            text = str(getattr(chunk, "text", "") or "")
            if not text.strip():
                continue
            headings = _chunk_headings(chunk)
            matched = [h for h in headings if any(_keyword_in(_words(h), k) for k in keywords)]
            in_heading = sum(1 for k in keywords if _keyword_in(_words(" ".join(headings)), k))
            in_body = sum(1 for k in keywords if _keyword_in(_words(text), k))
            if not in_heading and not in_body:
                continue
            rank = (-in_heading, -in_body, order, int(getattr(chunk, "ordinal", 0) or 0))
            scored.append((rank, chunk, source_type, source_id, matched[0] if matched else None))
    scored.sort(key=lambda item: item[0])
    out: List[LocalHit] = []
    seen_text: Set[str] = set()
    for _, chunk, source_type, source_id, heading in scored:
        key = str(getattr(chunk, "sha1", "") or "") or " ".join(str(chunk.text).split())
        if key in seen_text:
            continue
        seen_text.add(key)
        out.append(_followup_hit(db, chunk, source_type, source_id, heading))
        if len(out) >= max_passages:
            break
    return out


def _followup_hit(db: Database, chunk: Any, source_type: str, source_id: int, heading: Optional[str] = None) -> LocalHit:
    """:class:`LocalHit` d'un morceau relu pour une question de suivi (même forme que les hits du moteur) ;
    ``heading`` est le titre de section du morceau qui a répondu à la question, cité dans la raison."""
    meta = getattr(chunk, "meta", None)
    meta = meta if isinstance(meta, dict) else {}
    extra: Dict[str, Any] = {"ordinal": getattr(chunk, "ordinal", None)}
    if getattr(chunk, "sha1", None):
        extra["sha1"] = str(chunk.sha1)
    title, path, repo_id, file_id = "", None, None, None
    section: Optional[str] = str(heading) if heading else None
    if source_type == "document":
        doc = db.get_document(int(source_id))
        title = doc.title if doc is not None and doc.title else "document %s" % source_id
        extra["document_id"] = int(source_id)
        if doc is not None:
            extra["format"] = doc.format or ""
            extra["source_path"] = doc.source_path or doc.stored_path or ""
        heading_path = meta.get("heading_path") or []
        own = meta.get("section") or meta.get("heading") or (heading_path[-1] if heading_path else None)
        if own and str(own) != title:
            extra["section"] = str(own)  # l'en-tête du bloc de contexte l'affiche (« §4. Les champs… »)
            section = section or str(own)
    else:
        row = db.get_file_by_id(int(source_id))
        if row is not None:
            path, repo_id, file_id = row.path, row.repo_id, row.id
            title = os.path.basename(row.path)
            extra["language"] = row.language or ""
            repo = db.get_repo(row.repo_id) if row.repo_id else None
            if repo is not None:
                extra["repo_name"] = repo.name
        else:
            file_id, title = int(source_id), "fichier %s" % source_id
    where = ("§%s de « %s »" % (section, title)) if section else ("« %s »" % (path or title))
    text = str(getattr(chunk, "text", "") or "")
    return LocalHit(source_type=source_type, source_id=int(source_id), chunk_id=getattr(chunk, "id", None), title=title,
                    path=path, repo_id=repo_id, file_id=file_id, line_start=getattr(chunk, "line_start", None),
                    line_end=getattr(chunk, "line_end", None), page=_as_int(meta.get("page")), snippet=text[:300],
                    text=text, score=0.0, channels=[FOLLOWUP_CHANNEL], reason=FOLLOWUP_REASON % where, extra=extra)


def is_followup_question(user_text: str, analysis: Any = None, today: Optional[str] = None) -> bool:
    """Vrai quand la question a besoin du tour précédent pour être cherchée.

    Deux signes, lisibles sans modèle de langage : elle s'ouvre sur un mot de liaison (:data:`FOLLOWUP_OPENERS` :
    « et ses commandes ? », « ok mais pourquoi ? »), ou elle ne nomme aucun sujet à elle — aucun identifiant, chemin,
    route ni table, et pas un mot-clé qui ne soit un mot d'aspect (:data:`FOLLOWUP_ASPECT_WORDS` : « explique-moi la
    définition des champs »). « quel est l'api d'authentification », « comment confirmer une commande » ou « qui
    appelle OrderService::confirm ? » nomment leur sujet : autonomes. ``analysis`` est le résultat de
    :func:`develop.search.query.analyze_query` sur la question, calculé s'il manque.
    """
    text = str(user_text or "")
    words = _fold(text).split()
    if not words:
        return True
    if words[0] in FOLLOWUP_OPENERS:
        return True
    if analysis is None:
        try:
            from develop.search.query import analyze_query
            analysis = analyze_query(text, today)
        except Exception as exc:  # noqa: BLE001
            logger.debug("analyse de suivi impossible: %s", exc)
            return False
    for attr in ("identifiers", "paths", "routes", "tables"):
        if list(getattr(analysis, attr, []) or []):
            return False
    keywords = [_fold(k) for k in (getattr(analysis, "keywords", []) or [])]
    return all(k in FOLLOWUP_ASPECT_WORDS for k in keywords if k)


def followup_terms(user_text: str, previous_question: str, previous_sources: Sequence[Dict[str, Any]],
                   today: Optional[str] = None) -> List[str]:
    """Termes du tour précédent à ajouter à la recherche d'une question de suivi, ou ``[]``.

    Une question est « de suivi » au sens de :func:`is_followup_question` (mot de liaison en tête, ou aucun sujet
    à elle). Les termes viennent des identifiants puis des mots-clés de la question précédente et des titres /
    chemins (sans extension) des premières sources de la réponse précédente ; ceux déjà présents dans la question
    sont omis, le tout borné à :data:`FOLLOWUP_MAX_TERMS`. Dans une vraie question de suivi ils **sont** le sujet
    (« et ses commandes ? » → Customer) : le moteur les cherche avec les mots de la question (identifiants hérités à
    poids plein, mots-clés hérités en départage des routes) ; une question autonome n'en reçoit jamais.
    """
    if not previous_question or not str(previous_question).strip():
        return []
    try:
        from develop.search.query import analyze_query
        current = analyze_query(str(user_text or ""), today)
        previous = analyze_query(str(previous_question), today)
    except Exception as exc:  # noqa: BLE001
        logger.debug("analyse de suivi impossible: %s", exc)
        return []
    if not is_followup_question(user_text, current):
        return []
    keywords = list(getattr(current, "keywords", []) or [])
    identifiers = list(getattr(current, "identifiers", []) or [])
    present = {_fold(k) for k in keywords} | {_fold(i) for i in identifiers}
    candidates: List[str] = list(getattr(previous, "identifiers", []) or [])
    for s in list(previous_sources or [])[:FOLLOWUP_SOURCES]:
        label = _source_label(s)
        if label:
            candidates.append(label)
    candidates += [k for k in (getattr(previous, "keywords", []) or []) if len(str(k)) >= 4]  # « dit », « api »… : bruit
    out: List[str] = []
    seen: Set[str] = set()
    for term in candidates:
        term = str(term or "").strip()
        key = _fold(term)
        if not key or key in seen or key in present or len(key) < 2:
            continue
        # un titre de document est un groupe de mots : chaque mot déjà présent dans la question est inutile
        if " " in key and all(w in present for w in key.split()):
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= FOLLOWUP_MAX_TERMS:
            break
    return out


class ChatOrchestrator:
    """Produit les réponses du chat sous forme de flux d'événements."""

    def __init__(self, db: Database, workspace: Any, llm: LLMClient, search_engine: Any = None,
                 impact_analyzer: Any = None, profile: Any = None, model_info: Optional[ModelInfo] = None,
                 today: Optional[str] = None, git: Any = None):
        self.db = db
        self.workspace = workspace
        self.llm = llm
        self.search_engine = search_engine
        self.impact_analyzer = impact_analyzer
        self.profile = profile
        self.model_info = model_info
        self.today = today or time.strftime("%Y-%m-%d")
        self.tools = ToolExecutor(db, workspace, search_engine, impact_analyzer, git)
        # derniers prompts envoyés, par modèle (texte rendu par :meth:`_prompt_text`, réponse comprise) : ce que le
        # moteur a probablement encore en cache de préfixe ; voir :meth:`_cached_fraction`
        self._recent_prompts: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ aides
    @property
    def llm_cfg(self) -> Any:
        return getattr(self.workspace, "llm", None)

    def _repos_summary(self) -> str:
        parts: List[str] = []
        for repo in getattr(self.workspace, "repos", []) or []:
            if not getattr(repo, "enabled", True):
                continue
            parts.append("%s (%s)" % (getattr(repo, "name", "?"), getattr(repo, "kind", "other")))
        return ", ".join(parts)

    def _tools_mode(self, mode: str) -> str:
        if mode in ("no_tools", "search_only"):
            return "none"
        if self.model_info is not None and getattr(self.model_info, "supports_tools", False):
            return "native"
        return "text"

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm_cfg, "model", "") or "")

    @property
    def engine_name(self) -> str:
        return str(getattr(self.llm, "engine", "") or getattr(self.llm_cfg, "engine", "") or "")

    def prefill_tps(self) -> Tuple[float, bool]:
        """``(tokens/s de lecture du prompt, mesuré ?)`` pour le modèle configuré : la médiane des dernières mesures
        du chat (:func:`load_prefill_tps`) quand elle existe, sinon l'estimation du conseiller
        (:func:`develop.llm.advisor.estimate_prefill_tps` sur ``model_info`` — taille du modèle — et le profil)."""
        model = self.model_name
        measured = load_prefill_tps(self.db, self.engine_name, model)
        if measured:
            return measured, True
        # model_info porte la taille (« 27B ») quand le moteur la donne ; sinon le nom du modèle (« qwen3:14b »)
        described: Any = self.model_info if (self.model_info is not None and model_params_b(self.model_info)) else model
        return estimate_prefill_tps(described, self.profile), False

    def _plan(self, tools_mode: str, intent: str, max_context_tokens: Optional[int],
              user_text: str) -> Tuple[Budget, str]:
        """Budget de la question avec les consignes système réellement mesurées ; renvoie ``(budget, consignes)``.

        Les consignes dépendent du budget (``num_ctx``, tours d'outils) et le budget de leur coût :
        une première passe fixe ``num_ctx`` et le mode, les consignes sont construites puis mesurées
        (définitions d'outils natifs comprises) et le budget recalculé avec cette mesure. Si cette
        mesure ne laisse pas de place aux outils, le budget revient sans outils et les consignes sont
        reconstruites et remesurées dans ce mode. Le mode effectif est ``budget.tools_mode``. Le
        contexte est borné par la vitesse de lecture du modèle (:meth:`prefill_tps`, mesurée sinon
        estimée) et l'objectif ``first_token_target_s`` de la configuration.
        """
        cfg, info, prof = self.llm_cfg, self.model_info, self.profile
        name, repos = getattr(self.workspace, "name", ""), self._repos_summary()
        question_tokens = estimate_tokens(user_text.strip()) + USER_FRAME_TOKENS
        tps, _ = self.prefill_tps()

        def measured(text: str, mode: str) -> int:
            return estimate_tokens(text) + (tool_specs_tokens() if mode == "native" else 0)

        def downgrade_reason(b: Budget) -> str:
            idx = b.source.find("outils désactivés")
            return b.source[idx:] if idx >= 0 else ""

        budget = compute_budget(cfg, info, prof, intent, max_context_tokens, tools_mode=tools_mode,
                                question_tokens=question_tokens, prefill_tps=tps)
        reason = downgrade_reason(budget)
        system_text = ""
        for _ in range(2):  # au plus : mesure → repli sans outils → nouvelle mesure
            mode = budget.tools_mode
            system_text = build_system_prompt(mode, budget, name, repos, self.today)
            budget = compute_budget(cfg, info, prof, intent, max_context_tokens, tools_mode=mode,
                                    system_tokens=measured(system_text, mode), question_tokens=question_tokens,
                                    prefill_tps=tps)
            if budget.tools_mode == mode:
                break
            reason = downgrade_reason(budget) or reason
        else:
            system_text = build_system_prompt(budget.tools_mode, budget, name, repos, self.today)
        if reason and "outils désactivés" not in budget.source:
            budget.source += " ; " + reason  # le repli a eu lieu à une passe précédente : on garde l'explication
        return budget, system_text

    async def _search(self, text: str) -> Tuple[List[Any], str]:
        """Analyse + embedding + recherche multi-canaux ; renvoie ``(hits, intent)`` sans jamais lever."""
        engine = self.search_engine
        intent = "factual"
        if engine is None:
            return [], intent
        analysis: Any = None
        analyzed = False
        for attr in ("analyze", "analyze_query"):
            fn = getattr(engine, attr, None)
            if callable(fn):
                analyzed = True
                try:
                    analysis = await _maybe_await(fn(text))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("analyse de la question impossible: %s", exc)
                    analysis = None
                break
        if not analyzed:
            # le moteur n'expose pas d'analyse : on utilise celle de develop.search.query (import paresseux)
            try:
                from develop.search.query import analyze_query
                analysis = analyze_query(text, self.today)
            except Exception as exc:  # noqa: BLE001 - module absent ou en cours d'écriture
                logger.debug("analyze_query indisponible: %s", exc)
                analysis = None
        kwargs: Dict[str, Any] = {"limit": 30}
        if analysis is not None:
            intent = str(getattr(analysis, "intent", None) or intent)
            for key in ("date_from", "date_to"):
                value = getattr(analysis, key, None)
                if value:
                    kwargs[key] = value
        embed = getattr(engine, "embed_query", None)
        if callable(embed):
            try:
                vec = await _maybe_await(embed(text))
                if vec is not None:
                    kwargs["query_vector"] = vec
            except Exception as exc:  # noqa: BLE001
                logger.info("embedding de la question indisponible: %s", exc)
        try:
            result = await _maybe_await(engine.search(text, **kwargs))
        except Exception as exc:  # noqa: BLE001
            logger.warning("recherche en échec, réponse sans contexte: %s", exc, exc_info=True)
            return [], intent
        if isinstance(result, list):
            hits = result
        else:
            hits = list(getattr(result, "hits", []) or [])
            query = getattr(result, "query", None)
            if query is not None and getattr(query, "intent", None):
                intent = str(query.intent)
        return hits, intent

    def _history(self, conversation_id: int, exclude_message_id: int, budget: Budget) -> Tuple[List[ChatMessage], str, int]:
        """Derniers messages tenant dans ``history_tokens_max`` + résumé en une ligne des plus anciens.

        Renvoie ``(messages, résumé, tokens des messages)`` ; le résumé (ajouté aux consignes par
        l'appelant) tient dans ce qui reste du budget d'historique, de sorte que messages + résumé
        ne dépassent jamais ``history_tokens_max``.
        """
        conv = self.db.get_conversation(conversation_id) or {}
        messages = [m for m in conv.get("messages", []) if m.get("id") != exclude_message_id
                    and m.get("role") in ("user", "assistant")]
        limit = max(0, int(budget.history_tokens_max))
        # on remonte depuis le message le plus récent tant que le budget le permet
        kept_from = len(messages)
        used = 0
        while kept_from > 0:
            cost = estimate_tokens(str(messages[kept_from - 1].get("content") or ""))
            if kept_from < len(messages) and used + cost > limit:
                break
            kept_from -= 1
            used += cost
        # pas de réponse orpheline en tête d'historique : sa question est passée dans le résumé
        while kept_from < len(messages) - 1 and messages[kept_from].get("role") == "assistant":
            kept_from += 1
        kept: List[ChatMessage] = []
        for m in messages[kept_from:]:
            content = str(m.get("content") or "")
            if len(messages) - kept_from == 1 and estimate_tokens(content) > limit:
                # un seul message, trop long : on le tronque plutôt que de perdre tout l'historique
                content = content[: max(0, limit * 4 - 20)].rstrip() + " […]"
            kept.append(ChatMessage(role=str(m.get("role")), content=content))
        used = sum(estimate_tokens(m.content) for m in kept)
        older = messages[:kept_from]
        summary = ""
        if older:
            titles = []
            for m in older:
                if m.get("role") == "user":
                    first_line = str(m.get("content") or "").strip().splitlines()
                    if first_line:
                        titles.append(first_line[0][:80])
            if titles:
                summary = HISTORY_SUMMARY_PREFIX + " ; ".join("Q : " + t for t in titles)
                max_chars = max(0, limit - used) * 4  # le résumé prend ce que les messages n'ont pas consommé
                if max_chars < len(HISTORY_SUMMARY_PREFIX) + 8:
                    summary = ""  # pas la place d'un seul titre : l'historique gardé suffit
                elif len(summary) > max_chars:
                    summary = summary[:max_chars - 1].rstrip() + "…"
        return kept, summary, used

    def _previous_answer(self, conversation_id: int, exclude_message_id: Optional[int]) -> str:
        """Dernière réponse de l'assistant avant le message courant, ``""`` si le dernier message n'en est pas une."""
        try:
            conv = self.db.get_conversation(conversation_id) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("historique illisible pour la petite conversation: %s", exc)
            return ""
        messages = [m for m in conv.get("messages", []) if m.get("id") != exclude_message_id]
        if not messages or messages[-1].get("role") != "assistant":
            return ""
        return str(messages[-1].get("content") or "")

    def _previous_turn(self, conversation_id: int, exclude_message_id: Optional[int]) -> Tuple[str, List[Dict[str, Any]]]:
        """``(dernière question utilisateur, sources de la réponse qui l'a suivie)`` avant le message courant.

        Un « oui » / « non merci » / « ok » adressé au modèle (:func:`is_reply_to_model`) n'est pas une question :
        la question est celle d'avant, les sources celles de la dernière réponse.
        """
        try:
            conv = self.db.get_conversation(conversation_id) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug("historique illisible pour le suivi: %s", exc)
            return "", []
        messages = [m for m in conv.get("messages", []) if m.get("id") != exclude_message_id]
        question = ""
        sources: List[Dict[str, Any]] = []
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            role = m.get("role")
            if role == "assistant":
                if not sources:
                    sources = [s for s in (m.get("sources") or []) if isinstance(s, dict)]
            elif role == "user":
                content = str(m.get("content") or "")
                before = messages[i - 1] if i > 0 else {}
                answer = str(before.get("content") or "") if before.get("role") == "assistant" else ""
                if is_reply_to_model(content, answer):
                    continue
                question = content
                break
        return question, sources

    def _followup_context(self, conversation_id: int, exclude_message_id: Optional[int],
                          user_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """``(termes du tour précédent séparés par des espaces, sources de la réponse précédente)`` quand la
        question est un suivi, sinon ``("", [])``."""
        previous_question, previous_sources = self._previous_turn(conversation_id, exclude_message_id)
        if not previous_question or is_smalltalk(previous_question):
            return "", []
        terms = followup_terms(user_text, previous_question, previous_sources, self.today)
        if not terms:
            return "", []
        return " ".join(terms), previous_sources

    def _with_followup_passages(self, hits: List[Any], previous_sources: Sequence[Dict[str, Any]],
                                user_text: str) -> List[Any]:
        """Passages de suivi (:func:`followup_passages`) placés devant les résultats de la recherche, au score du
        meilleur résultat : ils passent en tête du contexte ; un morceau déjà trouvé par la recherche n'est pas
        dupliqué (l'assembleur de contexte garde la première occurrence d'un ``chunk_id``)."""
        try:
            passages = followup_passages(self.db, user_text, previous_sources, self.today)
        except Exception as exc:  # noqa: BLE001 - jamais de réponse perdue à cause du suivi
            logger.debug("passages de suivi indisponibles: %s", exc)
            return hits
        if not passages:
            return hits
        top = 0.0
        for h in hits:
            try:
                top = max(top, float(getattr(h, "score", 0.0) or 0.0))
            except (TypeError, ValueError):
                continue
        for p in passages:
            p.score = round(top, 6)
        return list(passages) + list(hits)

    def _store(self, conversation_id: int, content: str, pack: Optional[ContextPack]) -> Optional[int]:
        sources = [s.to_dict() for s in pack.sources] if pack is not None else []
        try:
            return self.db.add_message(conversation_id, "assistant", content, sources=sources)
        except StorageError as exc:
            logger.error("enregistrement du message impossible: %s", exc)
            return None

    def _budget_dict(self, budget: Budget, pack: Optional[ContextPack], rounds: int, context_tokens: int,
                     tools_tokens_used: int = 0, tools_tokens_max: int = 0, first_token_s: Optional[float] = None,
                     prefill_tps: Optional[float] = None) -> Dict[str, Any]:
        """``budget`` de l'événement ``done`` : le contexte initial et les tours d'outils sont comptés à part.

        - ``context_tokens_used`` ≤ ``max_context_tokens`` : les passages retenus par la recherche initiale et
          leur budget (les sources ajoutées ensuite par les outils ne s'y cumulent pas) ;
        - ``tools_tokens_used`` / ``tools_tokens_max`` : ce que les tours d'outils ont consommé (appels,
          résultats, sources ajoutées, avis d'épuisement) et ce que la fenêtre leur laissait une fois le prompt
          réel et la réserve de réponse comptés ; 0 sans outils ;
        - ``sources_tokens`` : tous les blocs de sources citables, contexte initial et outils compris ;
        - ``prefill_tps`` : vitesse de lecture du modèle (tokens/s) retenue après ce tour (médiane des dernières
          mesures, mise à jour par la mesure du tour, sinon la valeur — mesurée ou estimée — qui a servi au budget) ;
          ``first_token_s`` : délai mesuré jusqu'au premier fragment du premier appel (``null`` sans mesure) ;
          ``latency_capped`` : vrai si la borne de latence a réduit le contexte.
        """
        tps = prefill_tps if prefill_tps is not None else budget.prefill_tps
        return {"num_ctx": budget.num_ctx, "context_tokens_used": max(0, int(context_tokens or 0)),
                "max_context_tokens": budget.context_tokens_max, "sources_tokens": pack.tokens_used if pack else 0,
                "tools_tokens_used": max(0, int(tools_tokens_used or 0)),
                "tools_tokens_max": max(0, int(tools_tokens_max or 0)), "tool_rounds": rounds,
                "answer_reserve": budget.answer_reserve, "tools_mode": budget.tools_mode, "source": budget.source,
                "prefill_tps": round(float(tps), 1) if tps else None,
                "first_token_s": round(float(first_token_s), 2) if first_token_s is not None else None,
                "latency_capped": bool(budget.latency_capped)}

    # ------------------------------------------------------------------ flux principal
    async def _model_needs_loading(self) -> bool:
        """Vrai si le moteur dit (Ollama : /api/ps) que le modèle configuré n'est pas en mémoire — une liste vide
        (rien en mémoire : keep_alive expiré, ``develop ask`` sans préchauffage, après ``POST /api/llm/unload``) le dit
        aussi ; faux s'il l'est ou si le moteur ne le dit pas (``loaded_models()`` → ``None`` : serveur compatible
        OpenAI ; erreur ou lenteur : moteur injoignable). Jamais d'exception."""
        model = self.model_name
        if not model:
            return False
        try:
            loaded = await asyncio.wait_for(self.llm.loaded_models(), 3.0)
        except Exception:  # noqa: BLE001 - moteur muet ou lent : on ne conclut rien
            loaded = None
        if loaded is None:
            return False
        return not model_is_loaded(model, loaded)  # même règle que GET /api/llm/state (qwen3:14b ≠ qwen3:8b)

    def _announce_model_phase(self, needs_loading: bool, n_sources: int, tokens: int,
                              read_s: Optional[float]) -> List[ChatEvent]:
        """Événements ``status`` avant l'appel du modèle : chargement si le moteur n'a pas le modèle en mémoire,
        puis lecture des sources avec le délai estimé (prompt entier ÷ vitesse de lecture, au moins 1 s)."""
        events: List[ChatEvent] = []
        if needs_loading:
            events.append(_event("status", text=STATUS_LOADING % self.model_name))
        seconds = max(1, int(math.ceil(read_s))) if read_s is not None else 0
        events.append(_event("status", text=STATUS_READING % (n_sources, tokens, seconds)))
        return events

    @staticmethod
    def _prompt_text(messages: Sequence[ChatMessage], tools: Optional[Sequence[Any]] = None, answer: str = "") -> str:
        """Rendu textuel d'un appel au modèle (outils proposés, messages avec rôle et appels d'outils, puis la réponse
        générée) : la forme sous laquelle les prompts sont comparés pour estimer leur préfixe commun. Le gabarit du
        moteur ajoute ses propres marqueurs, mais le préfixe commun de deux prompts y diverge au même endroit."""
        parts = ["tools: %s\n" % ", ".join(str(getattr(t, "name", "") or "") for t in (tools or []))]
        for m in messages:
            parts.append("%s\n%s\n" % (m.role, m.content))
            for c in m.tool_calls or []:
                parts.append("%s(%s)\n" % (c.name, json.dumps(c.arguments, ensure_ascii=False, sort_keys=True)))
        if answer:
            parts.append("assistant\n%s\n" % answer)
        return "".join(parts)

    def _cached_fraction(self, model: str, prompt_text: str) -> float:
        """Part (0–1) de ``prompt_text`` que le moteur a probablement déjà en cache : le plus long préfixe commun
        avec les :data:`PREFILL_RECENT_PROMPTS` derniers prompts envoyés à ``model`` par cet orchestrateur (Ollama
        réutilise le cache KV du préfixe commun avec une requête précédente du même modèle, réponse comprise).
        0 sans prompt précédent — après un redémarrage de l'outil, le cache du moteur peut être chaud : une mesure
        gonflée isolée, que la médiane absorbe."""
        if not prompt_text:
            return 0.0
        best = 0
        for previous in self._recent_prompts.get(model, []):
            best = max(best, len(os.path.commonprefix([previous, prompt_text])))
        return min(1.0, best / len(prompt_text))

    def _remember_prompt(self, model: str, prompt_text: str) -> None:
        """Garde ``prompt_text`` parmi les derniers prompts envoyés à ``model`` (voir :meth:`_cached_fraction`)."""
        if not model or not prompt_text:
            return
        recent = [p for p in self._recent_prompts.get(model, []) if p != prompt_text]
        recent.append(prompt_text)
        self._recent_prompts[model] = recent[-PREFILL_RECENT_PROMPTS:]

    def _record_prefill(self, prompt_tokens: Optional[int], first_token_s: Optional[float],
                        load_s: Optional[float] = None, cached_fraction: float = 0.0) -> Optional[float]:
        """Enregistre la vitesse de lecture mesurée sur un appel (tokens du prompt non cachés ÷ délai jusqu'au premier
        fragment de texte) et renvoie la nouvelle valeur retenue (médiane des dernières mesures) ; ``None`` si la mesure
        n'est pas exploitable : prompt non caché trop court pour dominer la latence du moteur (moins de
        :data:`PREFILL_MIN_PROMPT_TOKENS` tokens — une question reposée telle quelle dans une nouvelle conversation
        est entièrement en cache, le délai ne mesure rien), délai nul, ou ``load_s`` — temps de chargement du modèle
        rapporté par le moteur pour cet appel, ``usage.load_duration_ms`` d'Ollama — au-delà de
        :data:`PREFILL_LOAD_MAX_S` (le délai mêlait chargement et lecture). ``prompt_tokens`` compte tout le prompt,
        cache compris (``prompt_eval_count`` d'Ollama) ; ``cached_fraction`` (:meth:`_cached_fraction`) en est retirée."""
        try:
            total = int(prompt_tokens or 0)
            delay = float(first_token_s or 0.0)
            loading = float(load_s or 0.0)
            cached = min(1.0, max(0.0, float(cached_fraction or 0.0)))
        except (TypeError, ValueError):
            return None
        if loading > PREFILL_LOAD_MAX_S:
            logger.debug("préremplissage %s : appel non mesuré, le moteur a chargé le modèle pendant %.1f s",
                         self.model_name, loading)
            return None
        tokens = int(round(total * (1.0 - cached)))
        if tokens < PREFILL_MIN_PROMPT_TOKENS or delay <= 0:
            if total >= PREFILL_MIN_PROMPT_TOKENS and delay > 0:
                logger.debug("préremplissage %s : appel non mesuré, %d tokens sur %d probablement en cache (%.2f s)",
                             self.model_name, total - tokens, total, delay)
            return None
        retained = record_prefill_tps(self.db, self.engine_name, self.model_name, tokens / delay)
        if retained is not None:
            logger.debug("préremplissage %s : %d tokens non cachés (sur %d) en %.2f s → %.0f tok/s (retenu %.0f)",
                         self.model_name, tokens, total, delay, tokens / delay, retained)
        return retained

    async def respond(self, conversation_id: int, user_text: str, mode: str = "auto",
                      max_context_tokens: Optional[int] = None,
                      cancel: Optional[asyncio.Event] = None) -> AsyncIterator[ChatEvent]:
        """Répond à ``user_text`` dans la conversation ; produit les événements SSE du chat."""
        mode = mode if mode in ("auto", "search_only", "no_tools") else "auto"
        user_text = str(user_text or "")
        emitted: List[str] = []  # fragments de réponse déjà envoyés à l'UI (= contenu enregistré)
        pack: Optional[ContextPack] = None
        stored = False
        rounds = 0
        try:
            # (1) message utilisateur
            try:
                user_message_id = self.db.add_message(conversation_id, "user", user_text)
            except StorageError as exc:
                stored = True
                yield _event("error", code="conversation_not_found", message=str(exc))
                return

            # (1 bis) petites conversations : réponse immédiate, sans recherche ni modèle (mode recherche seule inchangé) ;
            #         un « oui » / « non » / « ok » qui répond à une question du modèle lui parvient (question de suivi)
            if mode != "search_only" and is_smalltalk(user_text, self._previous_answer(conversation_id, user_message_id)):
                content = smalltalk_reply(self.db, str(getattr(self.workspace, "name", "") or ""), user_text)
                budget, _ = self._plan("none", "factual", max_context_tokens, user_text)
                yield _event("status", text=STATUS_SMALLTALK)
                yield _event("sources", items=[])
                emitted.append(content)
                yield _event("delta", text=content)
                message_id = self._store(conversation_id, content, None)
                stored = True
                yield _event("done", message_id=message_id, usage={}, budget=self._budget_dict(budget, None, 0, 0),
                             truncated_sources=[], invalid_citations=[], cancelled=False, mode=mode)
                return

            # (2) recherche : une question de suivi (mot de liaison en tête ou sans sujet à elle, voir
            #     :func:`is_followup_question`) cherche aussi avec les termes du tour précédent (un seul événement
            #     status, qui le dit) et relit les sources précédentes
            #     (:func:`followup_passages`) ; la question d'origine seule va au modèle
            search_text = user_text
            followup, followup_sources = (self._followup_context(conversation_id, user_message_id, user_text)
                                          if mode != "search_only" else ("", []))
            if followup:
                search_text = "%s %s" % (user_text.strip(), FOLLOWUP_MARKER % followup)
                yield _event("status", text=STATUS_FOLLOWUP % followup)
            else:
                yield _event("status", text=STATUS_SEARCHING)
            hits, intent = await self._search(search_text)
            if followup:
                # les passages des sources précédentes qui répondent à la question elle-même (section demandée
                # d'un document) passent devant : la recherche retrouve le sujet, pas toujours la partie visée
                hits = self._with_followup_passages(hits, followup_sources, user_text)

            # (3) budget : consignes système mesurées (définitions d'outils natifs comprises), mode d'outils
            #     effectif (``none`` si la fenêtre ne porte pas les outils), question comptée
            budget, system_text = self._plan(self._tools_mode(mode), intent, max_context_tokens, user_text)
            tools_mode = budget.tools_mode

            # (4) contexte : fiches structurées (modèles, routes, classes) en tête, puis passages
            pack = build_context(hits, budget.context_tokens_max, intent, db=self.db)
            context_tokens = pack.tokens_used  # contexte initial seul : les outils ajoutent ensuite au pack
            yield _event("sources", items=[s.to_dict() for s in pack.sources])

            # (5) recherche seule : pas de LLM
            if mode == "search_only":
                content = render_sources_as_text(pack)
                message_id = self._store(conversation_id, content, pack)
                stored = True
                yield _event("done", message_id=message_id, usage={},
                             budget=self._budget_dict(budget, pack, 0, context_tokens),
                             truncated_sources=[s.to_dict() for s in pack.truncated], invalid_citations=[],
                             cancelled=False, mode=mode)
                return

            # (6) historique (messages + résumé ≤ history_tokens_max)
            history, summary, history_tokens = self._history(conversation_id, user_message_id, budget)

            # (7) messages
            if summary:
                system_text += "\n\n" + summary
            user_content = user_text.strip() + "\n\n" + render_sources_for_model(pack)
            messages: List[ChatMessage] = [ChatMessage(role="system", content=system_text)]
            messages += history
            messages.append(ChatMessage(role="user", content=user_content))
            # Tout ce qui occupe la fenêtre est compté : messages, définitions d'outils natifs (hors messages
            # mais insérées par le gabarit du moteur à chaque appel tant que les outils sont proposés), réserve
            # de réponse. ``tools_available()`` est ce qui reste pour les résultats d'outils.
            prompt_tokens_est = estimate_tokens(system_text) + history_tokens + estimate_tokens(user_content)
            spec_tokens = tool_specs_tokens(TOOL_SPECS) if tools_mode == "native" else 0
            tools_tokens_used = 0

            def tools_available() -> int:
                return budget.num_ctx - prompt_tokens_est - spec_tokens - budget.answer_reserve - tools_tokens_used

            # ce que la fenêtre laisse aux tours d'outils (rapporté dans ``done.budget``) ; 0 sans outils
            tools_tokens_max = max(0, tools_available()) if tools_mode != "none" else 0

            # (7 bis) L'attente la plus longue est la lecture du prompt par le modèle (préremplissage), précédée
            #         du chargement du modèle s'il a été déchargé : on le dit, avec le délai estimé (prompt entier
            #         ÷ vitesse de lecture mesurée ou estimée), au lieu de laisser « Recherche… ».
            needs_loading = await self._model_needs_loading()
            for ev in self._announce_model_phase(needs_loading, len(pack.sources), pack.tokens_used,
                                                 read_time_s(prompt_tokens_est + spec_tokens, budget.prefill_tps)):
                yield ev

            # (8) boucle de génération / outils
            model = str(getattr(self.llm_cfg, "model", "") or "")
            temperature = float(getattr(self.llm_cfg, "temperature", 0.2) or 0.2)
            usage: Dict[str, int] = {}
            cancelled = False
            force_final = False
            seen_call_ids: Set[str] = set()
            first_token_s: Optional[float] = None  # délai jusqu'au premier fragment du premier appel
            prefill_avg: Optional[float] = None  # valeur retenue (médiane) mise à jour par la mesure de ce tour
            calls_made = 0
            while True:
                allow_tools = tools_mode != "none" and not force_final and rounds < budget.tool_rounds_max
                round_text = ""
                # Mode texte : (a) retenue initiale tant que la réponse peut n'être qu'un appel d'outil (rien
                # n'est montré d'une courte préface « Je vais regarder. ») ; (b) filtre de flux qui écarte
                # toute ligne d'outil du texte émis, y compris au tour final forcé où le modèle en écrit parfois.
                hold = allow_tools and tools_mode == "text"
                text_filter: Optional[_ToolLineFilter] = _ToolLineFilter() if tools_mode == "text" else None
                native_calls: List[ToolCall] = []
                extra_llm_kwargs: Dict[str, Any] = {}
                want_think = bool(getattr(self.llm_cfg, "think", False))
                caps = getattr(self.model_info, "capabilities", None) or []
                if getattr(self.llm, "engine", "") == "ollama" and not want_think:
                    # Équivalent de « /set nothink » : toujours envoyé à Ollama (accepté même par un modèle sans
                    # réflexion), sinon un modèle Qwen 3 réfléchit par défaut et multiplie le temps de réponse.
                    extra_llm_kwargs["think"] = False
                elif want_think and self.model_info is not None and "thinking" in caps:
                    # Activer la réflexion n'est possible que si le modèle la supporte (Ollama refuse sinon).
                    extra_llm_kwargs["think"] = True
                tools_sent = TOOL_SPECS if (allow_tools and tools_mode == "native") else None
                stream = self.llm.chat_stream(messages, model, temperature=temperature, num_ctx=budget.num_ctx,
                                              tools=tools_sent, max_tokens=budget.answer_reserve, **extra_llm_kwargs)
                # part du prompt que le moteur a probablement en cache (préfixe commun avec ses prompts précédents :
                # consignes, historique, sources relues, voire le prompt entier d'une question reposée) : retirée de
                # la mesure de vitesse de lecture ; le prompt est gardé après l'appel pour les comparaisons suivantes
                prompt_text = self._prompt_text(messages, tools_sent)
                cached_fraction = self._cached_fraction(model, prompt_text) if calls_made == 0 else 0.0
                # perf_counter, pas monotonic : sous Windows (Python < 3.13) monotonic() n'avance que toutes les
                # ~15,6 ms, un délai nul y passerait pour « non mesurable » et la vitesse de lecture ne serait jamais apprise
                sent_at = time.perf_counter()  # la requête part au premier parcours du flux
                first_chunk_at: Optional[float] = None  # premier fragment, quel qu'il soit (délai perçu)
                first_text_at: Optional[float] = None  # premier fragment de texte, s'il précède tout appel d'outil
                prefill_measurable = True
                call_usage: Dict[str, int] = {}
                try:
                    async for chunk in stream:
                        now = time.perf_counter()
                        if first_chunk_at is None:
                            first_chunk_at = now
                        if cancel is not None and cancel.is_set():
                            cancelled = True
                            break
                        text = getattr(chunk, "text", "") or ""
                        if prefill_measurable and first_text_at is None:
                            if getattr(chunk, "tool_calls", None):
                                # Ollama n'émet un appel d'outil qu'une fois son JSON entièrement généré (quelques
                                # secondes) : ce délai mêle lecture et génération, il ne mesure pas la lecture.
                                prefill_measurable = False
                            elif text:
                                first_text_at = now
                        if text:
                            round_text += text
                            if text_filter is not None:
                                if hold and not _could_be_tool_round(round_text):
                                    hold = False
                                text = text_filter.feed(text, release=not hold)
                            if text:
                                emitted.append(text)
                                yield _event("delta", text=text)
                        for call in getattr(chunk, "tool_calls", None) or []:
                            native_calls.append(call)
                        chunk_usage = getattr(chunk, "usage", None) or {}
                        for key, value in chunk_usage.items():
                            try:
                                usage[key] = usage.get(key, 0) + int(value)
                                call_usage[key] = call_usage.get(key, 0) + int(value)
                            except (TypeError, ValueError):
                                continue
                finally:
                    aclose = getattr(stream, "aclose", None)
                    if callable(aclose):
                        try:
                            await aclose()
                        except Exception:  # noqa: BLE001
                            pass
                calls_made += 1
                if first_chunk_at is not None:
                    # le moteur a lu ce prompt (et généré la suite) : c'est son cache de préfixe pour l'appel suivant
                    self._remember_prompt(model, self._prompt_text(messages, tools_sent, round_text))
                if calls_made == 1 and first_chunk_at is not None:
                    # Le premier appel lit le contexte frais : son délai jusqu'au premier fragment de texte mesure la
                    # vitesse de lecture du modèle (usage.prompt_tokens hors part en cache ÷ délai). Les tours
                    # d'outils suivants relisent un prompt en grande partie en cache ; un modèle qui se charge ou
                    # réfléchit avant d'écrire attend pour d'autres raisons ; un appel d'outil n'est émis qu'une fois
                    # son JSON généré : ces appels ne sont pas mesurés (``first_token_s`` reste le délai perçu jusqu'au
                    # premier fragment).
                    first_token_s = first_chunk_at - sent_at
                    if first_text_at is not None and not needs_loading and not extra_llm_kwargs.get("think"):
                        prefill_avg = self._record_prefill(call_usage.get("prompt_tokens"), first_text_at - sent_at,
                                                           load_s=call_usage.get("load_duration_ms", 0) / 1000.0,
                                                           cached_fraction=cached_fraction)
                if cancelled:
                    rest = text_filter.flush() if text_filter is not None else ""
                    if rest:
                        emitted.append(rest)
                        yield _event("delta", text=rest)
                    break

                calls: List[ToolCall] = []
                if allow_tools:
                    # mode texte : l'outil est exécuté même si une réponse utile précède la ligne JSON (comme
                    # un modèle natif qui écrit du texte et appelle un outil dans le même tour)
                    calls = list(native_calls) if tools_mode == "native" else parse_text_tool_calls(round_text)
                if text_filter is not None:
                    if calls and hold:
                        text_filter.discard()  # appel d'outil pur : la courte préface éventuelle n'est pas montrée
                    else:
                        rest = text_filter.flush()
                        if rest:
                            emitted.append(rest)
                            yield _event("delta", text=rest)
                if not calls:
                    break

                # tour d'outils (copies : les objets du client ne sont jamais mutés, identifiants uniques)
                rounds += 1
                calls = [ToolCall(name=str(c.name or ""), arguments=dict(c.arguments) if isinstance(c.arguments, dict)
                                  else c.arguments, id=str(c.id or "")) for c in calls[:MAX_CALLS_PER_ROUND]]
                for n, call in enumerate(calls):
                    if not call.id or call.id in seen_call_ids:
                        call.id = "call-%d-%d" % (rounds, n + 1)
                    seen_call_ids.add(call.id)
                if tools_mode == "native":
                    assistant_msg = ChatMessage(role="assistant", content=round_text, tool_calls=calls)
                else:
                    assistant_msg = ChatMessage(role="assistant", content=round_text)
                messages.append(assistant_msg)
                tools_tokens_used += _message_tokens(assistant_msg) + MESSAGE_OVERHEAD_TOKENS
                text_results: List[str] = []
                for call in calls:
                    if cancel is not None and cancel.is_set():
                        cancelled = True
                        break
                    yield _event("tool_call", name=call.name, arguments=call.arguments, id=call.id)
                    available = tools_available()
                    per_call = 0
                    if available < MIN_TOOL_BUDGET:
                        result = ToolResult(name=call.name, summary="Budget de contexte épuisé : outil non exécuté.",
                                            text="Budget de contexte épuisé : réponds avec les sources déjà fournies.",
                                            error="budget épuisé")
                        force_final = True
                    else:
                        per_call = min(available, max(MIN_TOOL_BUDGET, budget.context_tokens_max))
                        result = await self.tools.execute(call, per_call)
                    added: List[Source] = []
                    if result.hits and not result.error:
                        # Un seul budget par appel : le texte de l'outil et ses nouvelles sources se le partagent.
                        # Pour read_file/symbol, le texte est remplacé par le résumé dès qu'une source est ajoutée
                        # (son contenu est dans les blocs) : tout le budget va alors aux sources.
                        text_part = result.summary if result.name in ("read_file", "symbol") else result.text
                        sources_budget = per_call - estimate_tokens(text_part) - SOURCES_FRAME_TOKENS
                        added = add_tool_sources(pack, result.hits, max(0, sources_budget))
                    content = self._tool_message_content(result, pack, added)
                    if tools_mode != "native":
                        content = TEXT_RESULT_PREFIX % call.name + content
                    tools_tokens_used += estimate_tokens(content) + MESSAGE_OVERHEAD_TOKENS
                    data: Dict[str, Any] = {"name": call.name, "summary": result.summary,
                                            "added_sources": [s.to_dict() for s in added], "id": call.id}
                    if result.error:
                        data["error"] = result.error
                    yield _event("tool_result", **data)
                    if tools_mode == "native":
                        messages.append(ChatMessage(role="tool", content=content, tool_call_id=call.id, name=call.name))
                    else:
                        text_results.append(content)
                if cancelled:
                    break
                if tools_mode != "native" and text_results:
                    messages.append(ChatMessage(role="user", content="\n\n".join(text_results)))
                if rounds >= budget.tool_rounds_max or tools_available() < MIN_TOOL_BUDGET:
                    force_final = True
                    messages.append(ChatMessage(role="user", content=TOOLS_EXHAUSTED_TEXT))
                    tools_tokens_used += estimate_tokens(TOOLS_EXHAUSTED_TEXT) + MESSAGE_OVERHEAD_TOKENS

            # (9) fin
            answer = "".join(emitted)
            budget_info = self._budget_dict(budget, pack, rounds, context_tokens, tools_tokens_used, tools_tokens_max,
                                            first_token_s=first_token_s, prefill_tps=prefill_avg)
            if cancelled:
                message_id = self._store(conversation_id, answer, pack) if answer.strip() else None
                stored = True
                yield _event("done", message_id=message_id, usage=usage, budget=budget_info,
                             truncated_sources=[s.to_dict() for s in pack.truncated],
                             invalid_citations=find_invalid_citations(answer, list(pack.hits_by_ref.keys())),
                             cancelled=True, mode=mode)
                return
            if not answer.strip():
                answer = NO_ANSWER_TEXT
                emitted.append(answer)
                yield _event("delta", text=answer)
            invalid = find_invalid_citations(answer, list(pack.hits_by_ref.keys()))
            message_id = self._store(conversation_id, answer, pack)
            stored = True
            yield _event("done", message_id=message_id, usage=usage, budget=budget_info,
                         truncated_sources=[s.to_dict() for s in pack.truncated], invalid_citations=invalid,
                         cancelled=False, mode=mode)
        except (LLMUnavailable, ModelNotFound, OfflineViolation, LLMError) as exc:
            code = _error_code(exc)
            logger.info("chat interrompu (%s): %s", code, exc)
            message_id = self._store_partial(conversation_id, emitted, pack)
            stored = True
            yield _event("error", code=code, message=str(exc), message_id=message_id)
        except asyncio.CancelledError:
            if not stored:
                self._store_partial(conversation_id, emitted, pack)
                stored = True
            raise
        except Exception as exc:  # noqa: BLE001 - jamais de 500 silencieux dans un flux SSE
            logger.exception("erreur interne du chat")
            message_id = self._store_partial(conversation_id, emitted, pack)
            stored = True
            yield _event("error", code="internal_error", message="Erreur interne du chat : %s" % exc, message_id=message_id)
        finally:
            if not stored:
                # le client a fermé le flux : on garde la réponse partielle
                self._store_partial(conversation_id, emitted, pack)

    def _store_partial(self, conversation_id: int, emitted: List[str], pack: Optional[ContextPack]) -> Optional[int]:
        text = "".join(emitted)
        if not text.strip():
            return None
        return self._store(conversation_id, INTERRUPTED_PREFIX + text, pack)

    @staticmethod
    def _tool_message_content(result: ToolResult, pack: ContextPack, added: List[Source]) -> str:
        """Contenu renvoyé au modèle : résultat de l'outil + blocs des nouvelles sources citables."""
        parts: List[str] = []
        covered_by_sources = result.name in ("read_file", "symbol") and added and not result.error
        if result.text and not covered_by_sources:
            parts.append(result.text)
        elif covered_by_sources:
            parts.append(result.summary)
        if added:
            parts.append(render_sources_for_model(pack, [s.ref for s in added]))
        if result.error and not result.text:
            parts.append("Erreur : %s" % result.error)
        return "\n\n".join(p for p in parts if p)


def _error_code(exc: BaseException) -> str:
    if isinstance(exc, ModelNotFound):
        return "model_not_found"
    if isinstance(exc, LLMUnavailable):
        return "llm_unavailable"
    if isinstance(exc, OfflineViolation):
        return "offline_blocked"
    return "llm_error"


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or hasattr(value, "__await__"):
        return await value
    return value
