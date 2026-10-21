// streams.js — magasin des réponses en flux du chat, au niveau de l'application (indépendant des vues).
//
// Avant, la vue chat consommait elle-même le flux SSE de POST /api/conversations/{id}/messages :
// quitter l'écran pendant une réponse (Réglages…) coupait le flux et, au retour, la conversation
// rechargée depuis la base ne contenait pas encore la réponse — il fallait renvoyer le message.
// Ici le flux est lu ici, quelle que soit la vue affichée :
//  - startStream(conversationId, request) lance la requête, conserve tous les événements reçus
//    (tampon) et l'état dérivé (running, done, error, texte accumulé, sources, outils, budget…) ;
//  - subscribe(conversationId, handler) rejoue d'abord le tampon puis diffuse en direct ; le
//    gestionnaire reçoit (name, data, {at, replay}) pour chaque événement SSE, puis un événement
//    synthétique `end` {aborted, error} quand le flux se termine (fin normale, « Arrêter », coupure) ;
//  - abort(conversationId) interrompt la génération (bouton « Arrêter ») : démonter une vue ne coupe rien ;
//  - un flux terminé reste disponible jusqu'à ce que la vue recharge la conversation depuis la base
//    (release(conversationId) le purge alors : la base fait foi dès qu'elle contient la réponse) ;
//  - onChange(listener) prévient (pastille de la barre latérale, liste des conversations) à chaque
//    début et fin de flux.

import { streamPost } from '/static/api.js';

/** Nom de l'événement synthétique de fin de flux : data = {aborted, error}. */
export const END_EVENT = 'end';

const streams = new Map(); // id de conversation (chaîne) → flux
const changeListeners = new Set();

function key(conversationId) {
  return String(conversationId);
}

function safeCall(fn, ...args) {
  try { fn(...args); } catch (e) { console.error(e); }
}

function notifyChange() {
  for (const fn of [...changeListeners]) safeCall(fn, runningIds());
}

/** Met à jour l'état dérivé d'un flux à partir d'un événement SSE. */
function apply(entry, name, data) {
  const d = data || {};
  if (name === 'status') {
    entry.status = d.text || '';
  } else if (name === 'sources') {
    entry.sources = (d.items || []).slice();
    entry.initialSources = entry.sources.length;
  } else if (name === 'tool_call') {
    entry.tools.push({ call: d });
  } else if (name === 'tool_result') {
    const added = (d.added_sources || []).filter((s) => !entry.sources.some((x) => x.ref === s.ref));
    if (added.length) entry.sources = entry.sources.concat(added);
    entry.tools.push({ result: d });
  } else if (name === 'delta') {
    entry.text += d.text || '';
    entry.status = '';
  } else if (name === 'done') {
    entry.done = true;
    entry.messageId = d.message_id === undefined ? null : d.message_id;
    entry.usage = d.usage || {};
    entry.budget = d.budget || {};
    entry.truncated = d.truncated_sources || [];
    entry.cancelled = !!d.cancelled;
  } else if (name === 'error') {
    entry.error = { code: d.code || 'error', message: d.message || 'Erreur inconnue' };
    if (d.message_id !== undefined && d.message_id !== null) entry.messageId = d.message_id;
  }
}

/**
 * Lance la réponse à `request` ({content, mode}) dans la conversation et renvoie le flux.
 * Lève une erreur si une réponse est déjà en cours dans cette conversation ; un flux terminé
 * est remplacé.
 */
export function startStream(conversationId, request) {
  const id = key(conversationId);
  const existing = streams.get(id);
  if (existing && existing.running) throw new Error('Une réponse est déjà en cours dans cette conversation.');
  const controller = new AbortController();
  const entry = {
    conversationId: id,
    request: { ...request },
    startedAt: Date.now(),
    endedAt: null,
    events: [], // {name, data, at} dans l'ordre de réception (tampon rejoué aux abonnés tardifs)
    running: true,
    done: false, // événement `done` reçu
    cancelled: false, // `done.cancelled` (interruption vue par le serveur)
    aborted: false, // « Arrêter » côté client
    error: null, // événement SSE `error` {code, message}
    failure: null, // échec de transport (ApiError réseau / HTTP) : pas d'événement serveur
    status: '',
    text: '',
    sources: [],
    initialSources: 0,
    tools: [],
    usage: null,
    budget: null,
    truncated: [],
    messageId: null,
    subscribers: new Set(),
    controller,
  };
  streams.set(id, entry);

  const emit = (name, data) => {
    const at = Date.now();
    entry.events.push({ name, data, at });
    apply(entry, name, data);
    for (const handler of [...entry.subscribers]) safeCall(handler, name, data, { at, replay: false });
  };
  const finish = (err) => {
    entry.running = false;
    entry.endedAt = Date.now();
    entry.aborted = !!(err && err.name === 'AbortError');
    entry.failure = entry.aborted || !err ? null : err;
    const data = { aborted: entry.aborted, error: entry.failure };
    entry.events.push({ name: END_EVENT, data, at: entry.endedAt });
    for (const handler of [...entry.subscribers]) safeCall(handler, END_EVENT, data, { at: entry.endedAt, replay: false });
    entry.subscribers.clear();
    notifyChange();
  };
  entry.promise = streamPost(`/api/conversations/${encodeURIComponent(id)}/messages`, entry.request,
    { onEvent: emit, signal: controller.signal })
    .then(() => finish(null), (err) => finish(err || new Error('Flux interrompu.')));
  notifyChange();
  return entry;
}

/**
 * Abonne `handler(name, data, {at, replay})` au flux de la conversation : rejoue d'abord le tampon
 * (replay = true, `end` compris si le flux est terminé) puis diffuse en direct.
 * Renvoie la fonction de désabonnement, ou null sans flux pour cette conversation.
 */
export function subscribe(conversationId, handler) {
  const entry = streams.get(key(conversationId));
  if (!entry) return null;
  for (const ev of entry.events.slice()) safeCall(handler, ev.name, ev.data, { at: ev.at, replay: true });
  if (!entry.running) return () => {};
  entry.subscribers.add(handler);
  return () => { entry.subscribers.delete(handler); };
}

/** Interrompt la génération en cours (bouton « Arrêter »). Renvoie true si un flux a été coupé. */
export function abort(conversationId) {
  const entry = streams.get(key(conversationId));
  if (!entry || !entry.running) return false;
  entry.controller.abort();
  return true;
}

/** Le flux (en cours ou terminé) de la conversation, ou null. */
export function getStream(conversationId) {
  return streams.get(key(conversationId)) || null;
}

/** Une réponse est-elle en cours dans cette conversation ? */
export function isRunning(conversationId) {
  const entry = streams.get(key(conversationId));
  return !!(entry && entry.running);
}

/** Identifiants (chaînes) des conversations dont une réponse est en cours. */
export function runningIds() {
  return [...streams.values()].filter((e) => e.running).map((e) => e.conversationId);
}

/**
 * Purge un flux terminé (après rechargement de la conversation depuis la base). Un flux en cours
 * n'est jamais purgé (abort() d'abord). Renvoie true si quelque chose a été retiré.
 */
export function release(conversationId) {
  const id = key(conversationId);
  const entry = streams.get(id);
  if (!entry || entry.running) return false;
  streams.delete(id);
  return true;
}

/** Oublie le flux d'une conversation supprimée : coupé s'il est en cours, puis retiré du magasin. */
export function discard(conversationId) {
  const id = key(conversationId);
  const entry = streams.get(id);
  if (!entry) return false;
  if (entry.running) entry.controller.abort();
  streams.delete(id);
  return true;
}

/** Écoute les débuts et fins de flux : listener(runningIds). Renvoie la fonction de retrait. */
export function onChange(listener) {
  changeListeners.add(listener);
  return () => { changeListeners.delete(listener); };
}
