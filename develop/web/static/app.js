// app.js — point d'entrée : état global, routeur par hash, barre latérale et indicateur d'état.
//
// Chaque vue exporte `mount(root, route)` et renvoie (éventuellement) une fonction de
// démontage, appelée avant de changer de vue (fermeture des EventSource, minuteurs…).

import { api, watchLlm } from '/static/api.js';
import { el, icon, replace, errorBox, toast } from '/static/ui.js';
import { onChange as onStreamsChange, runningIds } from '/static/streams.js';
import * as setupView from '/static/views/setup.js';
import * as chatView from '/static/views/chat.js';
import * as explorerView from '/static/views/explorer.js';
import * as documentsView from '/static/views/documents.js';
import * as notesView from '/static/views/notes.js';
import * as impactView from '/static/views/impact.js';
import * as settingsView from '/static/views/settings.js';

/** État global unique de l'application (pas de variables globales éparses). */
export const state = {
  status: null, // dernier GET /api/status
  statusError: null, // erreur réseau sur /api/status
  llm: null, // dernier GET /api/llm/state (modèle en mémoire ? chargement en cours ?) ; null si inconnu
  route: { name: 'chat', parts: [], query: {} },
  chatPrefill: null, // message pré-rempli (depuis l'analyse d'impact)
  impactPick: null, // fichier ou symbole choisi dans l'explorateur pour l'analyse d'impact
  explorerQuery: '', // dernière recherche globale (restaurée au retour dans l'explorateur)
};

const VIEWS = {
  setup: setupView,
  chat: chatView,
  explorer: explorerView,
  documents: documentsView,
  notes: notesView,
  impact: impactView,
  settings: settingsView,
};

const NAV = [
  { name: 'chat', label: 'Chat', icon: 'chat' },
  { name: 'explorer', label: 'Explorateur', icon: 'explorer' },
  { name: 'documents', label: 'Documents', icon: 'documents' },
  { name: 'notes', label: 'Journal', icon: 'notes' },
  { name: 'impact', label: 'Impact', icon: 'impact' },
  { name: 'settings', label: 'Réglages', icon: 'settings' },
];

let current = null; // { name, unmount }
let routeSeq = 0;
let statusTimer = null;
let llmWatch = null; // stop() du suivi /api/llm/events pendant un chargement de modèle
let chatHash = '#/chat'; // cible du lien « Chat » : la conversation affichée en dernier (voir rememberChatRoute)

/** Nom de l'événement DOM (sur `window`) diffusé à chaque nouvel état du modèle ; `detail` = état ou null. */
export const LLM_STATE_EVENT = 'develop:llm-state';

// --------------------------------------------------------------------------- navigation

/** Change de route (le routeur réagit à `hashchange`). */
export function navigate(hash) {
  const target = hash.startsWith('#') ? hash : `#${hash}`;
  if (window.location.hash === target) {
    route();
  } else {
    window.location.hash = target;
  }
}

/** Analyse `#/explorer/file/12?line=3` → {name:'explorer', parts:['file','12'], query:{line:'3'}}. */
export function parseHash(hash = window.location.hash) {
  const raw = hash.replace(/^#\/?/, '');
  const [pathPart, queryPart = ''] = raw.split('?');
  const parts = pathPart.split('/').filter(Boolean).map(decodeURIComponent);
  const query = {};
  for (const [key, value] of new URLSearchParams(queryPart)) query[key] = value;
  const name = parts.shift() || 'chat';
  return { name, parts, query };
}

async function route() {
  const seq = ++routeSeq;
  const parsed = parseHash();
  if (!state.status && !state.statusError) await refreshStatus();
  if (seq !== routeSeq) return;

  // Premier lancement : l'assistant de configuration est obligatoire.
  if (state.status && state.status.needs_setup && parsed.name !== 'setup') {
    window.location.hash = '#/setup';
    return;
  }
  if (!VIEWS[parsed.name]) {
    window.location.hash = '#/chat';
    return;
  }

  if (current && typeof current.unmount === 'function') {
    try { current.unmount(); } catch (e) { console.error(e); }
  }
  current = null;
  state.route = parsed;
  highlightNav(parsed.name);
  if (parsed.name === 'chat') rememberChatRoute(parsed.parts[0] || null);
  renderStreamBadge();

  const main = document.getElementById('main');
  main.replaceChildren();
  main.scrollTop = 0;
  document.title = `Develop — ${NAV.find((n) => n.name === parsed.name)?.label || 'Configuration'}`;
  try {
    const unmount = await VIEWS[parsed.name].mount(main, parsed);
    if (seq !== routeSeq) {
      if (typeof unmount === 'function') unmount();
      return;
    }
    current = { name: parsed.name, unmount };
  } catch (e) {
    console.error(e);
    replace(main, el('div', { class: 'page' }, errorBox(e)));
  }
}

function highlightNav(name) {
  document.querySelectorAll('.nav-item').forEach((node) => node.classList.toggle('active', node.dataset.name === name));
}

/**
 * Le lien « Chat » de la barre latérale ramène à la conversation affichée en dernier (`#/chat/<id>`,
 * qu'elle réponde encore ou non) et non à une conversation vide : aller dans Réglages pendant une
 * réponse puis revenir par ce lien retrouve la réponse en cours, sans second clic. `#/chat` après
 * « Nouvelle » ou la suppression de la conversation affichée. Appelé par le routeur à chaque route
 * du chat et par la vue chat quand elle crée la conversation (URL remplacée sans hashchange).
 */
export function rememberChatRoute(conversationId) {
  chatHash = conversationId ? `#/chat/${encodeURIComponent(String(conversationId))}` : '#/chat';
  document.querySelectorAll('.nav-item').forEach((node) => { if (node.dataset.name === 'chat') node.setAttribute('href', chatHash); });
}

/**
 * Pastille « Réponse en cours » sur l'entrée Chat de la barre latérale : visible quand une réponse
 * est en flux (streams.js) alors que la conversation affichée n'est pas celle qui répond (autre
 * vue, ou autre conversation). Un clic ouvre la conversation qui répond.
 */
function renderStreamBadge() {
  const badge = document.getElementById('nav-chat-busy');
  if (!badge) return;
  const running = runningIds();
  const shown = state.route.name === 'chat' ? String(state.route.parts[0] || '') : '';
  const elsewhere = running.filter((id) => id !== shown);
  badge.classList.toggle('hidden', !elsewhere.length);
  badge.dataset.conversation = elsewhere[0] || '';
}

// --------------------------------------------------------------------------- état serveur

/**
 * Recharge /api/status (et /api/llm/state : modèle en mémoire, chargement en cours) puis met à jour
 * l'indicateur de la barre latérale. Pendant un chargement, le suivi SSE `/api/llm/events` prend le
 * relais seconde par seconde ; chaque état est diffusé aux vues (`LLM_STATE_EVENT`, ex. bandeau du chat).
 */
export async function refreshStatus() {
  try {
    const [status, llm] = await Promise.all([
      api.get('/api/status'),
      api.get('/api/llm/state').catch(() => null), // état inconnu : rien n'est bloqué
    ]);
    state.status = status;
    state.statusError = null;
    state.llm = llm;
  } catch (e) {
    state.statusError = e;
  }
  renderStatus();
  broadcastLlmState();
  ensureLlmWatch();
  scheduleStatus();
  return state.status;
}

function broadcastLlmState() {
  try {
    window.dispatchEvent(new CustomEvent(LLM_STATE_EVENT, { detail: state.llm }));
  } catch (_) { /* environnement sans CustomEvent : la vue relira /api/llm/state elle-même */ }
}

function applyLlmState(llm) {
  state.llm = llm;
  renderStatus();
  broadcastLlmState();
}

/** Suit un chargement en cours (un seul suivi à la fois) jusqu'au dernier état `loading:false`. */
function ensureLlmWatch() {
  if (llmWatch || !state.llm || !state.llm.loading) return;
  llmWatch = watchLlm({
    onState: applyLlmState,
    onDone: (llm) => {
      llmWatch = null;
      if (llm) applyLlmState(llm);
      ensureLlmWatch(); // un nouveau chargement a pu commencer entre-temps (changement de modèle)
    },
  });
}

function scheduleStatus() {
  clearTimeout(statusTimer);
  const running = state.status && state.status.index && state.status.index.running;
  statusTimer = setTimeout(refreshStatus, running ? 3000 : 15000);
}

function dot(kind) {
  return el('span', { class: `dot dot-${kind}`, 'aria-hidden': 'true' });
}

/** Indicateur moteur / index / hors ligne dans le pied de la barre latérale. */
function renderStatus() {
  const host = document.getElementById('status-panel');
  if (!host) return;
  const s = state.status;
  if (!s) {
    replace(host, el('div', { class: 'status-line' }, dot('error'), el('span', 'Serveur injoignable')),
      el('button', { class: 'btn btn-sm', type: 'button', onClick: refreshStatus }, 'Réessayer'));
    return;
  }
  const ws = s.workspace;
  const engine = s.engine || {};
  const index = s.index || {};
  const offline = s.offline || {};
  const counts = index.counts || {};
  const files = counts.files || 0;

  const engineLine = !engine.engine || engine.engine === 'none'
    ? [dot('neutral'), el('span', 'Aucun moteur d\'IA')]
    : engine.ok
      ? [dot('ok'), el('span', { title: `${engine.engine} — ${engine.model || ''}` }, `Moteur : ${engine.model || engine.engine}`)]
      : [dot('error'), el('span', { title: engine.error || '' }, 'Moteur indisponible')];

  // Modèle en mémoire (seulement quand le moteur le dit : Ollama) : chargé / chargement / non chargé.
  const llm = state.llm;
  const modelLine = !(llm && llm.known && engine.ok)
    ? null
    : llm.loading
      ? [dot('busy'), el('span', { title: `Chargement de ${llm.model} en mémoire (en général 10 à 40 s)` }, 'Chargement…')]
      : llm.loaded
        ? [dot('ok'), el('span', { title: llm.last_load_s ? `${llm.model} chargé en ${llm.last_load_s} s` : `${llm.model} en mémoire` }, 'Modèle chargé')]
        : [dot('neutral'), el('span', { title: llm.error || `${llm.model} n'est pas en mémoire : le chat le chargera` }, 'Modèle non chargé')];

  const indexLine = index.running
    ? [dot('busy'), el('span', 'Indexation en cours…')]
    : files > 0
      ? [dot('ok'), el('span', `Index : ${files} fichier${files > 1 ? 's' : ''}`)]
      : [dot('neutral'), el('span', 'Index vide')];

  const offlineLine = offline.mode === 'local_network'
    ? [dot('warn'), el('span', 'Réseau local autorisé')]
    : [dot('ok'), el('span', 'Hors ligne strict')];
  if (offline.blocked_count) offlineLine.push(el('span', { class: 'badge badge-warn', title: 'Requêtes bloquées' }, String(offline.blocked_count)));

  replace(host,
    el('div', { class: 'status-ws', title: ws ? ws.name : '' }, ws ? ws.name : 'Aucun espace de travail'),
    el('div', { class: 'status-line' }, engineLine),
    modelLine ? el('div', { class: 'status-line' }, modelLine) : null,
    el('div', { class: 'status-line' }, indexLine),
    el('div', { class: 'status-line' }, offlineLine));
}

// --------------------------------------------------------------------------- barre latérale

function renderSidebar() {
  const nav = document.getElementById('nav');
  replace(nav, NAV.map((item) => el('a', {
    class: 'nav-item',
    href: item.name === 'chat' ? chatHash : `#/${item.name}`, // « Chat » : dernière conversation affichée
    dataset: { name: item.name },
  }, icon(item.icon), el('span', item.label),
  item.name === 'chat' ? el('span', {
    class: 'badge badge-accent hidden', id: 'nav-chat-busy', title: 'Ouvrir la conversation qui répond',
    onClick: (e) => {
      const id = e.currentTarget.dataset.conversation;
      if (!id) return;
      e.preventDefault();
      e.stopPropagation();
      navigate(`#/chat/${encodeURIComponent(id)}`);
    },
  }, 'Réponse en cours') : null)));
}

// --------------------------------------------------------------------------- démarrage

function start() {
  renderSidebar();
  window.addEventListener('hashchange', route);
  window.addEventListener('online', () => renderStatus());
  window.addEventListener('offline', () => renderStatus());
  onStreamsChange(renderStreamBadge); // une réponse commence ou finit hors de sa conversation
  if (!window.location.hash) window.location.hash = '#/chat';
  route();
}

start();
export { toast };
