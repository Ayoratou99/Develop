// views/chat.js — conversations, fil de messages, réponse en flux SSE (status, sources,
// tool_call/tool_result, delta, done, error), citations [S1] cliquables.
//
// Le flux lui-même est lu par streams.js, au niveau de l'application : la vue s'y abonne (tampon
// rejoué puis direct) et s'en désabonne au démontage sans le couper. Changer d'écran pendant une
// réponse ne perd donc rien ; seul « Arrêter » interrompt la génération. Le lien « Chat » de la barre
// latérale ramène à la conversation affichée en dernier (app.js : rememberChatRoute), réponse comprise.

import { api } from '/static/api.js';
import { startStream, subscribe, abort as abortStream, getStream, isRunning, release, discard, onChange, END_EVENT } from '/static/streams.js';
import {
  el, replace, icon, toast, toastError, errorBox, infoBox, emptyState, spinner, renderMarkdown,
  confirmDialog, promptDialog, formatNumber, formatRelative, badge, collapsible, copyButton,
} from '/static/ui.js';
import { state, navigate, refreshStatus, rememberChatRoute } from '/static/app.js';
import { ENGINE_LABELS } from '/static/views/setup.js';

/** Commande proposée par le bandeau « Recherche lexicale seule » (modèle d'embeddings léger, 0,6 Go). */
export const EMBEDDING_PULL_COMMAND = 'ollama pull qwen3-embedding:0.6b';
const EMBEDDINGS_BANNER_KEY = 'develop.chat.embeddingsBannerDismissed';
/**
 * Clé sessionStorage : le chargement automatique du modèle de la session (l'onglet) est consommé — modèle vu en
 * mémoire ou chargement lancé à la première ouverture du chat, « Charger le modèle », « Décharger ». Ensuite,
 * façon LM Studio, un modèle hors mémoire ne revient que sur un clic ou à l'envoi d'un message, jamais par une
 * simple navigation (autre conversation, Réglages puis Chat : chaque retour remonte la vue). Repli en mémoire
 * de la page si sessionStorage est indisponible (navigation privée…).
 */
const MODEL_AUTOLOAD_KEY = 'develop.chat.modelAutoLoadDone';
let modelAutoLoadDone = false;
const autoLoadDone = () => modelAutoLoadDone || sessionFlag(MODEL_AUTOLOAD_KEY);
const consumeAutoLoad = () => { modelAutoLoadDone = true; sessionFlag(MODEL_AUTOLOAD_KEY, true); };

const MODES = [
  { value: 'auto', label: 'Auto', help: 'Recherche dans le projet, puis le modèle peut demander des outils (lecture de fichier, appelants…) : au plus 4 tours.' },
  { value: 'search_only', label: 'Recherche seule', help: 'Affiche uniquement les sources trouvées, sans appeler le modèle. Fonctionne sans moteur d\'IA.' },
  { value: 'no_tools', label: 'Sans outils', help: 'Une seule passe : contexte récupéré puis réponse du modèle, sans tours d\'outils.' },
];

const TYPE_LABELS = { file: 'fichier', document: 'document', note: 'note', symbol: 'symbole', endpoint: 'endpoint', model: 'modèle' };

/** Lien de navigation vers une source (fichier:ligne, document, note, symbole). */
export function sourceHref(src) {
  if (!src) return null;
  if (src.source_type === 'document' && src.document_id) return `#/documents/${encodeURIComponent(src.document_id)}`;
  if (src.source_type === 'note' && src.note_id) return `#/notes/${encodeURIComponent(src.note_id)}`;
  if (src.source_type === 'symbol' && src.symbol_id && !src.file_id) return `#/explorer/symbols/${encodeURIComponent(src.symbol_id)}`;
  if (src.file_id) return `#/explorer/file/${encodeURIComponent(src.file_id)}${src.line_start ? `?line=${src.line_start}` : ''}`;
  if (src.symbol_id) return `#/explorer/symbols/${encodeURIComponent(src.symbol_id)}`;
  if (src.document_id) return `#/documents/${encodeURIComponent(src.document_id)}`;
  if (src.note_id) return `#/notes/${encodeURIComponent(src.note_id)}`;
  return null;
}

/** Libellé court d'une source : chemin:lignes ou titre (page). */
function sourceLabel(src) {
  const where = src.path || src.title || '';
  if (src.line_start) return `${where}:${src.line_start}${src.line_end && src.line_end !== src.line_start ? `-${src.line_end}` : ''}`;
  if (src.page) return `${where} (p. ${src.page})`;
  return where;
}

/**
 * Pied de message lisible, à partir de `done.budget` : « contexte 2 758 / 2 758 tokens — 7 sources sur
 * 23 retenues » (passages retenus par la recherche initiale sur leur budget ; les sources ajoutées
 * ensuite par les outils sont comptées dans `toolsUsage`, le ratio ne dépasse donc jamais 100 %), puis,
 * si des sources ont été écartées, « 3 sources n'ont pas tenu dans le contexte : augmentez num_ctx »,
 * enfin la latence mesurée : « Lecture : 1 041 tokens/s », « Première lettre : 4,1 s » et, quand la borne
 * de latence a réduit le contexte (`latency_capped`), « contexte plafonné pour tenir le délai ».
 * `retained` : nombre de sources de la recherche initiale ; `truncated` : `done.truncated_sources`.
 * Renvoie une liste de nœuds (vide sans budget).
 */
export function budgetFooter(budget, retained, truncated) {
  const nodes = [];
  const b = budget || {};
  const dropped = Array.isArray(truncated) ? truncated : [];
  const kept = Math.max(0, Number(retained) || 0);
  if (b.num_ctx) {
    const total = kept + dropped.length;
    const sources = total
      ? `${kept} source${kept > 1 ? 's' : ''} sur ${total} retenue${kept > 1 ? 's' : ''}`
      : 'aucune source';
    nodes.push(el('span', { class: 'budget-line', title: b.source || `num_ctx ${formatNumber(b.num_ctx)}` },
      `contexte ${formatNumber(b.context_tokens_used || 0)} / ${formatNumber(b.max_context_tokens || b.num_ctx)} tokens — ${sources}`));
  }
  if (dropped.length) {
    const n = dropped.length;
    nodes.push(el('span', { class: 'budget-truncated', title: dropped.map((s) => s.path || s.title || s.ref || '?').join('\n') },
      `${n} source${n > 1 ? 's' : ''} n'${n > 1 ? 'ont' : 'a'} pas tenu dans le contexte : `,
      el('a', { href: '#/settings', title: 'Fenêtre de contexte (num_ctx) dans Réglages › Moteur d\'IA' }, 'augmentez num_ctx')));
  }
  if (Number(b.prefill_tps) > 0) {
    nodes.push(el('span', { class: 'budget-latency', title: 'Vitesse de lecture du prompt retenue pour ce modèle : moyenne des dernières mesures du chat, sinon estimation d\'après sa taille et la machine' },
      `Lecture : ${formatNumber(Math.round(Number(b.prefill_tps)))} tokens/s`));
  }
  if (b.first_token_s !== null && b.first_token_s !== undefined && Number.isFinite(Number(b.first_token_s))) {
    nodes.push(el('span', { class: 'budget-latency', title: 'Délai mesuré entre l\'envoi du prompt et le premier fragment de la réponse (hors chargement du modèle)' },
      `Première lettre : ${formatSeconds(b.first_token_s)}`));
  }
  if (b.latency_capped) {
    nodes.push(el('span', { class: 'budget-capped', title: b.source || 'La borne de latence a réduit le contexte lu par le modèle pour que la première lettre arrive dans le délai visé' },
      'contexte plafonné pour tenir le délai'));
  }
  return nodes;
}

/** Secondes lisibles : « 4,1 s » sous 10 s, « 192 s » au-delà ; « — » sans valeur. */
export function formatSeconds(seconds) {
  const n = seconds === null || seconds === undefined || seconds === '' ? NaN : Number(seconds);
  if (!Number.isFinite(n)) return '—';
  return `${new Intl.NumberFormat('fr-FR', { maximumFractionDigits: n < 10 ? 1 : 0 }).format(n)} s`;
}

// Mémoire locale des pieds de message : la base n'enregistre pas `usage` / `budget` avec la réponse, le
// pied (tokens, contexte, latence) est donc gardé ici à l'événement done, par message, pour être réaffiché
// au rechargement de la conversation. Portée : l'espace actif (les identifiants se répètent d'un espace à
// l'autre) ; 200 derniers messages ; stockage indisponible → pied perdu au rechargement, rien de plus.
const FOOTERS_KEY = 'develop.chat.footers';
const FOOTERS_MAX = 200;
let footersCache = null;

function footerKey(messageId) {
  const ws = state.status && state.status.active_workspace;
  return `${ws === null || ws === undefined ? '' : ws}/${messageId}`;
}

function readFooters() {
  if (footersCache) return footersCache;
  let list = [];
  try {
    const raw = window.localStorage.getItem(FOOTERS_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    if (Array.isArray(parsed)) list = parsed.filter((e) => e && typeof e.key === 'string');
  } catch (_) { /* stockage indisponible ou illisible : aucun pied mémorisé */ }
  footersCache = list;
  return list;
}

/**
 * Mémorise le pied du message enregistré `messageId` : `{usage, budget, retained, truncated}` tels que
 * reçus à l'événement done (`truncated` réduit à path / title / ref). Sans identifiant : rien.
 */
export function rememberFooter(messageId, info) {
  if (messageId === null || messageId === undefined) return;
  const key = footerKey(messageId);
  const entry = {
    key,
    usage: info.usage || {},
    budget: info.budget || {},
    retained: Number(info.retained) || 0,
    truncated: (Array.isArray(info.truncated) ? info.truncated : []).map((s) => ({ path: s.path, title: s.title, ref: s.ref })),
  };
  const list = readFooters().filter((e) => e.key !== key).concat([entry]).slice(-FOOTERS_MAX);
  footersCache = list;
  try { window.localStorage.setItem(FOOTERS_KEY, JSON.stringify(list)); } catch (_) { /* stockage indisponible : pied perdu au rechargement */ }
}

/** Pied mémorisé pour le message `messageId` (`{usage, budget, retained, truncated}`), ou null. */
export function recallFooter(messageId) {
  if (messageId === null || messageId === undefined) return null;
  const key = footerKey(messageId);
  return readFooters().find((e) => e.key === key) || null;
}

/** Drapeau de session (sessionStorage) ; lecture si `value` est omis. Indisponible (navigation privée…) → false. */
function sessionFlag(key, value) {
  try {
    if (value === undefined) return window.sessionStorage.getItem(key) === '1';
    window.sessionStorage.setItem(key, value ? '1' : '');
  } catch (_) { /* stockage indisponible : le bandeau reviendra à la prochaine ouverture de la vue */ }
  return false;
}

/**
 * Bandeau « Recherche lexicale seule » tant que l'espace n'a pas de modèle d'embeddings
 * (`status.workspace.llm.embedding_model` vide) alors qu'un moteur est configuré : lien vers les réglages,
 * commande d'installation à copier, masquable pour la session (sessionStorage). Null sinon.
 * Sans moteur du tout, l'encadré « Aucun moteur configuré » du fil suffit (les embeddings passent par le moteur).
 */
export function embeddingsBanner(status) {
  const llm = status && status.workspace && status.workspace.llm;
  if (!llm || !llm.engine || llm.engine === 'none' || llm.embedding_model) return null;
  if (sessionFlag(EMBEDDINGS_BANNER_KEY)) return null;
  const banner = el('div', { class: 'alert alert-info chat-banner', role: 'status' }, icon('info', 18),
    el('div', { class: 'alert-body' },
      el('strong', 'Recherche lexicale seule.'),
      ' Pour la recherche sémantique (questions formulées autrement que le code), ajoutez un modèle d\'embeddings dans ',
      el('a', { href: '#/settings' }, 'Réglages › Moteur d\'IA'),
      ' — par exemple ', el('code', { class: 'md-code' }, EMBEDDING_PULL_COMMAND), ' ', copyButton(EMBEDDING_PULL_COMMAND, 'Copier'),
      ' (0,6 Go) — puis réindexez.'),
    el('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', title: 'Masquer pour cette session', 'aria-label': 'Masquer ce bandeau pour cette session',
      onClick: () => { sessionFlag(EMBEDDINGS_BANNER_KEY, true); banner.remove(); } }, icon('close', 14)));
  return banner;
}

/** Secondes écoulées depuis `since` (epoch en secondes, horloge du serveur local) ; 0 si inconnu. */
export function loadingSeconds(since, now = Date.now()) {
  const s = Number(since);
  if (!s) return 0;
  return Math.max(0, Math.round(now / 1000 - s));
}

/**
 * Bandeau « Modèle qwen3:14b » façon LM Studio, d'après `GET /api/llm/state` :
 * chargé (pastille verte, bouton « Décharger »), en cours de chargement (barre indéterminée, secondes
 * écoulées dans `.chat-model-elapsed`, « en général 10 à 40 s »), non chargé (bouton « Charger le modèle »,
 * erreur éventuelle) ; moteur sans information (`known:false`) : état inconnu, rien n'est bloqué.
 * Chargé avec une autre fenêtre que celle du chat (`reload_pending`, `loaded_num_ctx` ≠ `num_ctx`) : pastille
 * orange, « chargé avec 32 768 ≠ 40 960 : rechargement au prochain message » et bouton « Recharger » (le moteur
 * recharge le modèle dès que les options diffèrent ; autant le faire maintenant qu'au premier message).
 * Null sans moteur configuré (l'encadré « Aucun moteur » du fil suffit) ou sans état.
 */
export function modelBanner(llm, { onLoad, onUnload, busy = false, now = Date.now() } = {}) {
  if (!llm || !llm.engine || llm.engine === 'none' || !llm.model) return null;
  const name = el('span', { class: 'chat-model-name' }, `Modèle ${llm.model}`);
  const dot = (kind) => el('span', { class: `dot dot-${kind}`, 'aria-hidden': 'true' });
  if (!llm.known) {
    return el('div', { class: 'chat-model chat-model-unknown', role: 'status' },
      el('div', { class: 'chat-model-line' }, dot('neutral'), name,
        el('span', { class: 'muted small' }, 'ce moteur ne dit pas si le modèle est en mémoire : rien n\'est bloqué.')));
  }
  if (llm.loading) {
    const elapsed = el('span', { class: 'chat-model-elapsed', dataset: { since: String(llm.loading_since || '') } },
      `${loadingSeconds(llm.loading_since, now)} s`);
    return el('div', { class: 'chat-model chat-model-loading', role: 'status', 'aria-busy': 'true' },
      el('div', { class: 'chat-model-line' }, dot('busy'), name,
        el('span', { class: 'muted' }, 'chargement en mémoire… ', elapsed, ' (en général 10 à 40 s)')),
      el('div', { class: 'progress' }, el('div', { class: 'progress-bar indeterminate' })));
  }
  if (llm.loaded) {
    const reload = !!llm.reload_pending && !!llm.loaded_num_ctx && !!llm.num_ctx;
    return el('div', { class: `chat-model chat-model-loaded${reload ? ' chat-model-reload' : ''}`, role: 'status' },
      el('div', { class: 'chat-model-line' }, dot(reload ? 'warn' : 'ok'), name,
        el('span', { class: 'muted' }, llm.last_load_s && !reload ? `chargé en ${formatNumber(llm.last_load_s)} s` : 'chargé'),
        reload ? el('span', { class: 'muted chat-model-reload-note', title: `Le moteur a chargé ${llm.model} avec une fenêtre de ${formatNumber(llm.loaded_num_ctx)} tokens alors que le chat demande ${formatNumber(llm.num_ctx)} : il rechargera le modèle au prochain message (10 à 40 s). « Recharger » le fait tout de suite.` },
          `avec ${formatNumber(llm.loaded_num_ctx)} ≠ ${formatNumber(llm.num_ctx)} : rechargement au prochain message`) : null,
        llm.embedding_model && !llm.embedding_loaded ? el('span', { class: 'muted small', title: `Modèle d'embeddings ${llm.embedding_model} hors mémoire : rechargé à la prochaine recherche sémantique` }, '(embeddings à recharger)') : null,
        el('span', { class: 'grow' }),
        reload ? el('button', { class: 'btn btn-primary btn-sm', type: 'button', disabled: busy, title: `Recharge le modèle avec la fenêtre du chat (${formatNumber(llm.num_ctx)} tokens, 10 à 40 s) pour que le prochain message parte aussitôt`, onClick: onLoad }, 'Recharger') : null,
        el('button', { class: 'btn btn-sm', type: 'button', disabled: busy, title: 'Libère la mémoire du moteur ; le prochain message devra recharger le modèle (10 à 40 s)', onClick: onUnload }, 'Décharger')));
  }
  return el('div', { class: 'chat-model chat-model-unloaded', role: 'status' },
    el('div', { class: 'chat-model-line' }, dot('neutral'), name,
      el('span', { class: 'muted' }, 'non chargé'),
      el('span', { class: 'grow' }),
      el('button', { class: 'btn btn-primary btn-sm', type: 'button', disabled: busy, title: 'Charge le modèle en mémoire (en général 10 à 40 s) ; le chat est disponible ensuite', onClick: onLoad }, 'Charger le modèle')),
    llm.error ? el('div', { class: 'chat-model-error' }, llm.error) : null);
}

/** Pied de message « Outils : 1 542 / 3 765 tokens (1 tour) » : consommation des tours d'outils, null sans tour. */
function toolsUsage(budget) {
  const rounds = Number(budget.tool_rounds || 0);
  if (!rounds) return null;
  const max = budget.tools_tokens_max ? ` / ${formatNumber(budget.tools_tokens_max)}` : '';
  return el('span', { title: 'Tokens consommés par les tours d\'outils (appels, résultats, sources ajoutées), sur ce que la fenêtre leur laissait' },
    `Outils : ${formatNumber(budget.tools_tokens_used || 0)}${max} tokens (${rounds} tour${rounds > 1 ? 's' : ''})`);
}

/**
 * Contenu du pied de message (`.msg-foot`), le même en direct (événement done) et au rechargement (mémoire
 * locale) : « Entrée : 1 200 tokens », « Réponse : 18 tokens », les lignes de `budgetFooter`, « Outils : … ».
 * `initialSources` : sources retenues par la recherche initiale (avant les ajouts des outils).
 */
function footContent(usage, budget, initialSources, truncated) {
  const u = usage || {};
  return [
    u.prompt_tokens !== undefined ? el('span', `Entrée : ${formatNumber(u.prompt_tokens)} tokens`) : null,
    u.completion_tokens !== undefined ? el('span', `Réponse : ${formatNumber(u.completion_tokens)} tokens`) : null,
    budgetFooter(budget, initialSources, truncated),
    toolsUsage(budget || {}),
  ];
}

/** Panneau « Sources utilisées » (liste S1… cliquable). */
function sourcesPanel(sources, open = false) {
  if (!sources || !sources.length) return null;
  const list = el('div', { class: 'sources-list' }, sources.map((src) => {
    const href = sourceHref(src);
    return el('div', { class: 'source', dataset: { ref: src.ref || '' }, onClick: () => { if (href) navigate(href); } },
      el('span', { class: 'cite' }, src.ref || '?'),
      badge(TYPE_LABELS[src.source_type] || src.source_type || '?'),
      el('span', { class: 'source-path' }, sourceLabel(src)),
      (src.channels || []).length ? el('span', { class: 'muted small' }, src.channels.join(', ')) : null,
      src.reason ? el('span', { class: 'source-reason' }, src.reason) : null);
  }));
  return collapsible(`Sources utilisées (${sources.length})`, list, open);
}

/** Rendu Markdown d'une réponse avec citations [Sn] converties en liens. */
function renderAnswer(container, text, sources) {
  const byRef = new Map((sources || []).map((s) => [s.ref, s]));
  const citation = (ref) => {
    const src = byRef.get(ref);
    const href = sourceHref(src);
    if (!src) return el('span', { class: 'cite', title: 'Source non disponible' }, `[${ref}]`);
    return el('a', { class: 'cite', href: href || '#', title: sourceLabel(src), onClick: (e) => {
      if (!href) { e.preventDefault(); return; }
      // surligne la source dans le panneau avant de naviguer
      const panel = container.closest('.msg');
      if (panel) panel.querySelectorAll('.source').forEach((n) => n.classList.toggle('target', n.dataset.ref === ref));
    } }, `[${ref}]`);
  };
  replace(container, renderMarkdown(text, { citation }));
}

function userMessage(content) {
  return el('div', { class: 'msg msg-user' }, el('div', { class: 'msg-role' }, 'Vous'), el('div', { class: 'msg-body' }, content));
}

/** Message assistant relu depuis la base ; son pied (tokens, contexte, latence) revient de la mémoire locale s'il y est. */
function assistantMessage(message) {
  const body = el('div', { class: 'msg-body md' });
  renderAnswer(body, message.content || '', message.sources || []);
  const remembered = recallFooter(message.id);
  return el('div', { class: 'msg msg-assistant' },
    el('div', { class: 'msg-role' }, 'Develop', message.created_at ? el('span', { class: 'muted' }, ` · ${formatRelative(message.created_at)}`) : null),
    sourcesPanel(message.sources || []),
    body,
    remembered ? el('div', { class: 'msg-foot' }, footContent(remembered.usage, remembered.budget, remembered.retained, remembered.truncated)) : null);
}

// --------------------------------------------------------------------------- vue

/**
 * Le message assistant de la conversation chargée depuis la base est-il déjà là pour ce flux ?
 * Par `message_id` (événements `done` / `error`) ; sinon (flux en cours, arrêté avant `done`) par
 * la présence d'un message assistant après le message utilisateur de la requête.
 */
export function storedAnswerIndex(messages, entry) {
  const list = Array.isArray(messages) ? messages : [];
  if (entry.messageId !== null && entry.messageId !== undefined) {
    return list.findIndex((m) => m.id === entry.messageId);
  }
  let userIdx = -1;
  for (let i = list.length - 1; i >= 0; i -= 1) {
    if (list[i].role === 'user' && (list[i].content || '') === entry.request.content) { userIdx = i; break; }
  }
  if (userIdx < 0) return -1;
  for (let i = userIdx + 1; i < list.length; i += 1) if (list[i].role === 'assistant') return i;
  return -1;
}

/** Le message utilisateur de ce flux est-il déjà dans la conversation chargée (le serveur l'enregistre dès réception) ? */
function hasUserMessage(messages, entry) {
  return (messages || []).some((m) => m.role === 'user' && (m.content || '') === entry.request.content);
}

export async function mount(root, route) {
  let conversationId = route.parts[0] || null;
  let alive = true;
  let sending = false; // création de conversation puis départ du flux (double clic)
  let unsubscribe = null; // désabonnement du flux affiché (le flux lui-même vit dans streams.js)
  let stopTicker = () => {};
  let lastConversations = null; // dernière liste reçue (re-rendue sans requête quand un flux commence ou finit)

  const convList = el('div', { class: 'chat-list' });
  const thread = el('div', { class: 'chat-thread' });
  const input = el('textarea', { class: 'textarea compose-input', placeholder: 'Posez une question sur le projet… (Entrée pour envoyer, Maj+Entrée pour un saut de ligne)', rows: 2 });
  const modeSelect = el('select', { class: 'select', id: 'chat-mode', style: { width: 'auto' }, title: MODES[0].help }, MODES.map((m) => el('option', { value: m.value, title: m.help }, m.label)));
  const modeHelp = el('span', { class: 'muted small' }, MODES[0].help);
  const sendBtn = el('button', { class: 'btn btn-primary', type: 'button' }, icon('send', 16), 'Envoyer');
  const stopBtn = el('button', { class: 'btn btn-danger hidden', type: 'button' }, icon('stop', 16), 'Arrêter');

  modeSelect.addEventListener('change', () => {
    const m = MODES.find((x) => x.value === modeSelect.value);
    modeSelect.title = m ? m.help : '';
    modeHelp.textContent = m ? m.help : '';
    syncControls(); // « Recherche seule » n'attend pas le modèle
  });

  const engineInfo = () => (state.status && state.status.engine) || {};
  const engineConfigured = () => { const e = engineInfo(); return !!e.engine && e.engine !== 'none'; };
  const engineOk = () => !!engineInfo().ok;
  /** Encadré : aucun moteur (à configurer) ou moteur configuré mais injoignable (à démarrer). */
  const noEngineBox = () => {
    const e = engineInfo();
    if (!engineConfigured()) {
      return infoBox(el('div', el('div', { style: { fontWeight: '600' } }, 'Aucun moteur configuré : la recherche fonctionne, le chat non.'),
        el('div', 'Le mode « Recherche seule » est sélectionné. Pour obtenir des réponses, ', el('a', { href: '#/settings' }, 'configurez un moteur d\'IA'), '.')), 'warn');
    }
    return infoBox(el('div',
      el('div', { style: { fontWeight: '600' } }, `${ENGINE_LABELS[e.engine] || e.engine} injoignable${e.model ? ` (modèle ${e.model})` : ''} : le chat passe en mode « Recherche seule ».`),
      e.error ? el('div', { class: 'muted small' }, e.error) : null,
      el('div', e.engine === 'ollama' ? 'Démarrez le service (« ollama serve » ou l\'application Ollama) ' : 'Démarrez votre serveur local ',
        'ou vérifiez son adresse, puis ', el('a', { href: '#/settings' }, 'vérifiez la connexion dans les réglages'), '. Dès que le moteur répond, choisissez le mode « Auto ».')), 'warn');
  };

  /** Une réponse est-elle en cours dans la conversation affichée ? */
  const busy = () => conversationId !== null && isRunning(conversationId);

  // ---------------------------------------------------------------- modèle en mémoire (façon LM Studio)
  let llm = state.llm || null; // dernier état connu de /api/llm/state (null : inconnu, rien n'est bloqué)
  let modelBusy = false; // requête load / unload en cours
  let modelTicker = null; // secondes écoulées pendant un chargement
  const modelHost = el('div', { class: 'chat-model-host' });
  /** Envoi bloqué tant que le modèle n'est pas en mémoire (hors mode « Recherche seule », qui ne l'appelle pas). */
  const modelBlocked = () => modeSelect.value !== 'search_only' && !!llm && llm.known && (llm.loading || !llm.loaded);

  /** Boutons Envoyer / Arrêter selon le flux de la conversation affichée et l'état du modèle. */
  const syncControls = () => {
    const blocked = modelBlocked();
    sendBtn.disabled = sending || busy() || blocked;
    sendBtn.title = !blocked ? '' : (llm.loading ? 'Le modèle se charge…' : 'Chargez le modèle pour envoyer un message.');
    stopBtn.classList.toggle('hidden', !busy());
    // Champ grisé pendant le chargement seulement : non chargé, on peut déjà écrire en attendant.
    input.disabled = blocked && !!llm.loading;
    input.title = input.disabled ? 'Le modèle se charge…' : '';
  };

  const stopModelTicker = () => { clearInterval(modelTicker); modelTicker = null; };
  const renderModelBar = () => {
    replace(modelHost, modelBanner(llm, { onLoad: () => loadModel(), onUnload: () => unloadModel(), busy: modelBusy }));
    syncControls();
    stopModelTicker();
    if (llm && llm.loading) {
      const elapsed = modelHost.querySelector('.chat-model-elapsed');
      modelTicker = setInterval(() => { if (elapsed) elapsed.textContent = `${loadingSeconds(llm.loading_since)} s`; }, 1000);
    }
  };
  const applyLlm = (next) => { llm = next || null; renderModelBar(); };

  const loadModel = async () => {
    if (modelBusy) return;
    consumeAutoLoad(); // automatique (première ouverture) ou explicite : plus de chargement sans clic ensuite
    modelBusy = true;
    renderModelBar();
    try {
      applyLlm(await api.post('/api/llm/load'));
    } catch (e) {
      toastError(e);
    } finally {
      modelBusy = false;
      renderModelBar();
    }
    refreshStatus(); // l'application suit le chargement (/api/llm/events) et rediffuse chaque état
  };
  const unloadModel = async () => {
    if (modelBusy) return;
    modelBusy = true;
    renderModelBar();
    try {
      applyLlm(await api.post('/api/llm/unload'));
      consumeAutoLoad(); // déchargement explicite mémorisé : aucune navigation ne le rechargera
      toast('Modèle déchargé : la mémoire est libérée. Cliquez sur « Charger le modèle » pour reprendre.');
    } catch (e) {
      toastError(e);
    } finally {
      modelBusy = false;
      renderModelBar();
    }
    refreshStatus();
  };
  /**
   * À l'ouverture : état réel, puis chargement automatique **une seule fois par session** (onglet), à la première
   * ouverture du chat où le moteur répond et n'a pas le modèle en mémoire. Ensuite, façon LM Studio, un modèle
   * hors mémoire (« Décharger », ou libéré par le moteur) ne revient que sur « Charger le modèle » ou à l'envoi
   * d'un message : naviguer (autre conversation, Réglages puis Chat) ne relance jamais un chargement.
   */
  const initModel = async () => {
    try {
      applyLlm(await api.get('/api/llm/state'));
    } catch (_) { /* état inconnu : rien n'est bloqué */ }
    if (!alive || !llm || !llm.known) return;
    if (llm.loading) { // un chargement déjà lancé (démarrage, réglages) : s'assurer qu'il est suivi
      consumeAutoLoad();
      refreshStatus();
      return;
    }
    if (llm.loaded) { consumeAutoLoad(); return; }
    if (autoLoadDone() || !engineOk()) return; // déjà consommé (ou moteur injoignable : l'occasion reste pour la prochaine ouverture)
    await loadModel();
  };
  const onLlmState = (e) => { if (alive) applyLlm(e && e.detail ? e.detail : null); };
  window.addEventListener('develop:llm-state', onLlmState);

  // ---------------------------------------------------------------- conversations
  const renderConversations = (items) => {
    lastConversations = items;
    if (!items.length) {
      replace(convList, el('div', { class: 'muted small', style: { padding: '12px' } }, 'Aucune conversation. Posez votre première question.'));
      return;
    }
    replace(convList, items.map((c) => el('div', { class: `conv-item${String(c.id) === String(conversationId) ? ' active' : ''}`, onClick: () => navigate(`#/chat/${encodeURIComponent(c.id)}`) },
      el('span', { class: 'conv-title', title: c.title || '' }, c.title || 'Sans titre'),
      isRunning(c.id) ? el('span', { class: 'conv-busy muted', title: 'Réponse en cours' }, '…') : null,
      el('span', { class: 'conv-tools' },
        el('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', title: 'Renommer', onClick: async (e) => {
          e.stopPropagation();
          const title = await promptDialog('Renommer la conversation', { value: c.title || '' });
          if (title === null || !title) return;
          try { await api.patch(`/api/conversations/${encodeURIComponent(c.id)}`, { title }); loadConversations(); } catch (err) { toastError(err); }
        } }, icon('notes', 14)),
        el('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', title: 'Supprimer', onClick: async (e) => {
          e.stopPropagation();
          if (!(await confirmDialog(`Supprimer la conversation « ${c.title || 'Sans titre'} » ?`, { danger: true, okLabel: 'Supprimer' }))) return;
          try {
            discard(c.id); // une réponse en cours n'a plus de destination : coupée avant la suppression
            await api.del(`/api/conversations/${encodeURIComponent(c.id)}`);
            if (c.id === conversationId) navigate('#/chat'); else loadConversations();
          } catch (err) { toastError(err); }
        } }, icon('trash', 14))))));
  };

  const loadConversations = async () => {
    if (!lastConversations) replace(convList, spinner('Conversations…'));
    try {
      const items = await api.get('/api/conversations');
      if (!alive) return;
      renderConversations(items);
    } catch (e) {
      replace(convList, errorBox(e));
    }
  };

  const createConversation = async (title) => {
    const created = await api.post('/api/conversations', { title: title || undefined });
    conversationId = String(created.id);
    // Mise à jour de l'URL sans remonter la vue (pas d'événement hashchange) : le routeur ne la voit pas,
    // on lui signale la conversation pour que le lien « Chat » de la barre latérale y ramène.
    history.replaceState(null, '', `#/chat/${encodeURIComponent(conversationId)}`);
    state.route = { name: 'chat', parts: [conversationId], query: {} };
    rememberChatRoute(conversationId);
    loadConversations();
  };

  // ---------------------------------------------------------------- fil
  const scrollToEnd = () => { thread.scrollTop = thread.scrollHeight; };

  /** Se désabonne du flux affiché (sans le couper : seul « Arrêter » le fait). */
  const detachStream = () => {
    if (unsubscribe) unsubscribe();
    unsubscribe = null;
    stopTicker();
    stopTicker = () => {};
  };

  /**
   * Affiche la réponse d'un flux (en cours ou terminé) : squelette du message assistant rempli au fil
   * des événements, le tampon étant rejoué d'abord (retour sur la conversation pendant la réponse).
   */
  const attachStream = (entry) => {
    detachStream();
    const mode = entry.request.mode;
    const statusLine = el('div', { class: 'msg-status' }, el('span', { class: 'spinner' }), el('span', 'Envoi…'));
    const sourcesHost = el('div', { class: 'sources' });
    const toolsHost = el('div');
    const body = el('div', { class: 'msg-body md' });
    const foot = el('div', { class: 'msg-foot' });
    const errorHost = el('div');
    const msg = el('div', { class: 'msg msg-assistant' }, el('div', { class: 'msg-role' }, 'Develop'), statusLine, sourcesHost, toolsHost, body, errorHost, foot);
    thread.appendChild(msg);
    scrollToEnd();

    let sources = [];
    let initialSources = 0; // sources retenues par la recherche initiale (avant les ajouts des outils)
    let text = '';
    let pendingRender = false;
    const rerender = () => { renderAnswer(body, text, sources); pendingRender = false; scrollToEnd(); };

    // Compteur de secondes sur la ligne d'état : l'attente la plus longue est la lecture du contexte par le modèle.
    // Les horodatages viennent du tampon : au retour sur la conversation, le compteur reprend où il en est.
    let statusSince = entry.startedAt;
    const elapsedEl = el('span', { class: 'muted' }, '');
    const tickElapsed = () => {
      const s = Math.round((Date.now() - statusSince) / 1000);
      elapsedEl.textContent = s >= 3 ? ` (${s} s)` : '';
    };
    const ticker = setInterval(tickElapsed, 1000);
    stopTicker = () => { clearInterval(ticker); elapsedEl.textContent = ''; };
    const finish = () => {
      statusLine.classList.add('hidden');
      stopTicker();
      stopTicker = () => {};
      // Dernier rendu seulement s'il y a du texte : ne pas effacer le message de remplacement
      // (mode « Recherche seule ») que l'événement done a placé dans `body`.
      if (text || pendingRender) rerender();
      syncControls();
    };

    const onEvent = (name, data, meta) => {
      if (name === 'status') {
        statusSince = meta.at;
        replace(statusLine, el('span', { class: 'spinner' }), el('span', data.text || ''), elapsedEl);
        tickElapsed();
      } else if (name === 'sources') {
        sources = data.items || [];
        initialSources = sources.length;
        replace(sourcesHost, sourcesPanel(sources, mode === 'search_only'));
      } else if (name === 'tool_call') {
        toolsHost.appendChild(collapsible(el('span', { class: 'row' }, badge('outil', 'accent'), `${data.name || '?'}`),
          el('pre', { class: 'small' }, JSON.stringify(data.arguments || {}, null, 2))));
        replace(statusLine, el('span', { class: 'spinner' }), el('span', `Outil ${data.name || ''}…`));
      } else if (name === 'tool_result') {
        const added = data.added_sources || [];
        const failed = !!data.error; // l'orchestrateur ajoute `error` quand l'outil a échoué
        if (added.length) {
          sources = sources.concat(added.filter((s) => !sources.some((x) => x.ref === s.ref)));
          replace(sourcesHost, sourcesPanel(sources));
        }
        toolsHost.appendChild(collapsible(el('span', { class: 'row' }, badge(failed ? 'erreur' : 'résultat', failed ? 'error' : 'ok'), `${data.name || '?'}`),
          el('div',
            failed ? el('div', { class: 'text-danger' }, `L'outil a échoué : ${data.error}`) : null,
            data.summary ? el('div', data.summary) : null,
            added.length ? el('div', { class: 'muted small' }, `${added.length} source(s) ajoutée(s)`) : null), failed));
      } else if (name === 'delta') {
        text += data.text || '';
        // Rendu Markdown à la fin de chaque paragraphe pour limiter le coût (rejeu : un seul rendu à la fin).
        if (!meta.replay && (data.text || '').includes('\n\n')) rerender();
        else if (!pendingRender) { pendingRender = true; setTimeout(() => { if (pendingRender) rerender(); }, 250); }
        statusLine.classList.add('hidden'); stopTicker();
      } else if (name === 'done') {
        const usage = data.usage || {};
        const budget = data.budget || {};
        const truncated = data.truncated_sources || [];
        replace(foot, footContent(usage, budget, initialSources, truncated));
        // Mémorisé aussi au rejeu : si done est arrivé pendant que la vue était démontée (Réglages…), c'est le
        // seul moment où il passe ici ; la mémoire dédoublonne par message.
        rememberFooter(data.message_id, { usage, budget, retained: initialSources, truncated });
        if (mode === 'search_only' && !text) replace(body, el('p', { class: 'muted' }, sources.length ? 'Sources trouvées ci-dessus (aucun appel au modèle).' : 'Aucune source trouvée pour cette question.'));
      } else if (name === 'error') {
        const err = { code: data.code, message: data.message || 'Erreur inconnue' };
        if (data.code === 'llm_unavailable') {
          const configured = engineConfigured();
          replace(errorHost, infoBox(el('div',
            el('div', { style: { fontWeight: '600' } }, configured ? 'Le moteur d\'IA n\'a pas répondu' : 'Configurez un moteur d\'IA'),
            el('div', data.message || 'Le modèle n\'a pas répondu.'),
            el('div', 'Les sources trouvées restent affichées ci-dessus. ',
              configured ? 'Démarrez le moteur ou ' : '',
              el('a', { href: '#/settings' }, configured ? 'vérifiez la connexion dans les réglages' : 'Ouvrir les réglages'), configured ? '.' : '')), 'warn'));
        } else {
          replace(errorHost, errorBox(err));
        }
        if (!meta.replay) toastError(err);
      } else if (name === END_EVENT) {
        if (data.aborted) {
          replace(errorHost, infoBox('Génération arrêtée.', 'info'));
        } else if (data.error) {
          replace(errorHost, errorBox(data.error));
          if (!meta.replay) toastError(data.error);
        }
        finish();
        if (!meta.replay) {
          refreshStatus();
          loadConversations();
          input.focus();
        }
      }
    };

    unsubscribe = subscribe(entry.conversationId, onEvent);
    if (pendingRender) rerender(); // texte rejoué : rendu immédiat
    syncControls();
  };

  /**
   * Au chargement de la conversation : reprend le flux qui la concerne, en cours ou terminé mais pas
   * encore relu depuis la base. Un flux terminé est purgé ici : si la base a déjà la réponse, elle fait
   * foi (rien n'est affiché deux fois) ; sinon le tampon est rejoué une dernière fois.
   */
  const resumeStream = (messages) => {
    const entry = getStream(conversationId);
    if (!entry) return;
    if (storedAnswerIndex(messages, entry) < 0) {
      thread.querySelectorAll('.chat-empty').forEach((n) => n.remove());
      if (!hasUserMessage(messages, entry)) thread.appendChild(userMessage(entry.request.content));
      attachStream(entry); // rejoue le tampon : l'entrée doit encore être dans le magasin
    }
    if (!entry.running) release(conversationId);
  };

  const loadThread = async () => {
    detachStream();
    if (!conversationId) {
      replace(thread, el('div', { class: 'chat-empty' },
        engineOk() ? null : noEngineBox(),
        emptyState('Nouvelle conversation', 'Posez une question : « Que fait OrderService ? », « Qui appelle POST /orders ? », « Quelles notes parlent de la facturation ? »')));
      syncControls();
      return;
    }
    replace(thread, spinner('Chargement de la conversation…'));
    try {
      const conv = await api.get(`/api/conversations/${encodeURIComponent(conversationId)}`);
      if (!alive) return;
      const messages = conv.messages || [];
      const nodes = messages.map((m) => (m.role === 'user' ? userMessage(m.content || '') : assistantMessage(m)));
      replace(thread, engineOk() ? null : noEngineBox(),
        nodes.length ? nodes : el('div', { class: 'chat-empty' }, emptyState('Conversation vide', 'Posez votre première question ci-dessous.')));
      resumeStream(messages);
      syncControls();
      scrollToEnd();
    } catch (e) {
      replace(thread, errorBox(e, [el('button', { class: 'btn btn-sm', type: 'button', onClick: () => navigate('#/chat') }, 'Nouvelle conversation')]));
    }
  };

  // ---------------------------------------------------------------- envoi en flux
  const send = async () => {
    const content = input.value.trim();
    if (!content || sending || busy()) return;
    if (modelBlocked()) { // Entrée pendant un chargement ou modèle déchargé : le texte reste saisi, rien n'est mis en file
      if (llm.loading) toast('Le modèle se charge… envoyez votre message une fois le modèle chargé (le texte reste saisi).');
      else { toast('Chargement du modèle lancé : envoyez votre message une fois le modèle chargé.'); loadModel(); }
      return;
    }
    const mode = modeSelect.value;
    input.value = '';
    sending = true;
    syncControls();
    thread.querySelectorAll('.chat-empty').forEach((n) => n.remove());
    thread.appendChild(userMessage(content));
    scrollToEnd();
    try {
      if (!conversationId) await createConversation(content.slice(0, 60));
      // Le flux vit dans streams.js : changer d'écran ne le coupe pas, la vue s'y réabonne au retour.
      const entry = startStream(conversationId, { content, mode });
      sending = false;
      attachStream(entry);
    } catch (e) {
      sending = false;
      thread.appendChild(el('div', { class: 'msg msg-assistant' }, el('div', { class: 'msg-role' }, 'Develop'), errorBox(e)));
      toastError(e);
      syncControls();
    }
  };

  sendBtn.addEventListener('click', send);
  stopBtn.addEventListener('click', () => { if (conversationId) abortStream(conversationId); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(220, input.scrollHeight)}px`;
  });
  // Début ou fin d'un flux (ici ou ailleurs) : boutons et marque « … » de la liste, sans requête.
  const offChange = onChange(() => {
    if (!alive) return;
    syncControls();
    if (lastConversations) renderConversations(lastConversations);
  });

  // Message pré-rempli (depuis l'analyse d'impact).
  if (state.chatPrefill) {
    input.value = state.chatPrefill;
    state.chatPrefill = null;
    setTimeout(() => input.dispatchEvent(new Event('input')), 0);
  }
  if (!engineOk()) modeSelect.value = 'search_only';
  modeSelect.dispatchEvent(new Event('change'));

  replace(root, el('div', { class: 'chat' },
    el('aside', { class: 'chat-side' },
      el('div', { class: 'chat-side-head' }, el('strong', 'Conversations'),
        el('button', { class: 'btn btn-sm', type: 'button', title: 'Nouvelle conversation', onClick: () => navigate('#/chat') }, icon('plus', 14), 'Nouvelle')),
      convList),
    el('section', { class: 'chat-main' },
      embeddingsBanner(state.status),
      modelHost,
      thread,
      el('div', { class: 'chat-compose' }, el('div', { class: 'compose-box' },
        input,
        el('div', { class: 'compose-tools' },
          el('div', { class: 'row' }, el('label', { class: 'label', for: 'chat-mode' }, 'Mode'), modeSelect, modeHelp),
          el('div', { class: 'row' }, stopBtn, sendBtn)))))));

  renderModelBar();
  await Promise.all([loadConversations(), loadThread(), initModel()]);
  input.focus();

  return () => {
    alive = false;
    offChange();
    window.removeEventListener('develop:llm-state', onLlmState);
    stopModelTicker();
    detachStream(); // le flux continue dans streams.js ; seul « Arrêter » le coupe
  };
}
