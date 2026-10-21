// api.js — client HTTP de l'API locale de Develop et lecteurs de flux SSE.
//
// Deux manières de recevoir des événements serveur :
//  - streamPost() : POST + lecture du corps `text/event-stream` via ReadableStream
//    (utilisé par le chat, qui envoie un message et reçoit la réponse en flux) ;
//  - subscribe()  : EventSource classique sur une route GET (progression d'indexation).
//
// Toutes les erreurs HTTP sont converties en ApiError {code, message, status} à partir
// du corps `{error:{code,message}}` renvoyé par le serveur (ou `{detail}` de FastAPI).

export class ApiError extends Error {
  constructor(code, message, status = 0) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
  }
}

/** Construit une URL absolue (même origine) avec des paramètres de requête facultatifs. */
function buildUrl(path, query) {
  const url = new URL(path, window.location.origin);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value === undefined || value === null || value === '') continue;
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

/** Message lisible à partir du champ `detail` des réponses par défaut de FastAPI (422, 404…). */
function detailMessage(detail) {
  if (detail === null || detail === undefined) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (!item || typeof item !== 'object') return String(item);
      const loc = (item.loc || []).filter((part) => part !== 'body' && part !== 'query' && part !== 'path').join('.');
      return loc ? `${loc} : ${item.msg || JSON.stringify(item)}` : (item.msg || JSON.stringify(item));
    }).join(' ; ');
  }
  if (typeof detail === 'object') return detail.msg || detail.message || JSON.stringify(detail);
  return String(detail);
}

/** Transforme une réponse HTTP en échec en ApiError lisible. */
async function toApiError(res) {
  let payload = null;
  try {
    payload = await res.json();
  } catch (_) {
    /* corps non JSON : on garde le message générique */
  }
  const err = (payload && typeof payload === 'object' && payload.error && typeof payload.error === 'object') ? payload.error : {};
  let message = err.message || '';
  if (!message && payload && typeof payload === 'object' && payload.detail !== undefined) {
    const detail = detailMessage(payload.detail);
    if (detail) message = res.status === 422 ? `Requête invalide : ${detail}` : detail;
  }
  if (!message && typeof payload === 'string') message = payload;
  if (!message) message = `Erreur HTTP ${res.status}`;
  const code = err.code || (res.status === 422 ? 'validation_error' : `http_${res.status}`);
  return new ApiError(code, message, res.status);
}

/** Erreur réseau (serveur injoignable ou connexion coupée). */
function networkError(during = false) {
  return new ApiError('network', during
    ? 'La connexion avec le serveur Develop a été coupée pendant la réponse. Vérifiez qu\'il est toujours démarré, puis réessayez.'
    : 'Impossible de joindre le serveur Develop. Est-il démarré ?', 0);
}

/** Requête générique. `form` prend le pas sur `body` (envoi multipart). */
async function request(method, path, { query, body, form, signal } = {}) {
  const init = { method, signal, headers: {} };
  if (form) {
    init.body = form;
  } else if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(buildUrl(path, query), init);
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    throw networkError();
  }
  if (!res.ok) throw await toApiError(res);
  if (res.status === 204) return null;
  const type = res.headers.get('content-type') || '';
  if (type.includes('application/json')) return res.json();
  return res.text();
}

export const api = {
  get: (path, query, opts = {}) => request('GET', path, { ...opts, query }),
  post: (path, body, opts = {}) => request('POST', path, { ...opts, body }),
  put: (path, body, opts = {}) => request('PUT', path, { ...opts, body }),
  patch: (path, body, opts = {}) => request('PATCH', path, { ...opts, body }),
  del: (path, opts = {}) => request('DELETE', path, opts),
  upload: (path, form, opts = {}) => request('POST', path, { ...opts, form }),
};

// --------------------------------------------------------------------------- SSE

/** Analyse un bloc SSE (lignes `event:` / `data:`) et renvoie {event, data} ou null. */
function parseSseBlock(raw) {
  let event = 'message';
  const dataLines = [];
  for (const line of raw.split('\n')) {
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  if (!dataLines.length) return null;
  const text = dataLines.join('\n');
  let data;
  try {
    data = JSON.parse(text);
  } catch (_) {
    data = { text };
  }
  return { event, data };
}

/**
 * POST dont la réponse est un flux SSE lu via ReadableStream.
 * `onEvent(eventName, data)` est appelé pour chaque événement ; la promesse se résout à la fin du flux.
 * Un AbortSignal permet d'interrompre la lecture (bouton « Arrêter »).
 */
export async function streamPost(path, body, { onEvent, signal } = {}) {
  let res;
  try {
    res = await fetch(buildUrl(path), {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    throw networkError();
  }
  if (!res.ok) throw await toApiError(res);
  if (!res.body) throw new ApiError('stream', 'Le navigateur ne sait pas lire ce flux.');

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  const emit = (raw) => {
    const evt = parseSseBlock(raw);
    if (evt && onEvent) onEvent(evt.event, evt.data);
  };
  // Les fins de ligne (\n, \r\n, \r) sont normalisées sur le tampon accumulé, jamais morceau par
  // morceau : un \r final reste en attente jusqu'au morceau suivant (une coupure entre \r et \n
  // ne doit pas fabriquer une ligne vide qui séparerait `event:` de `data:`).
  const drain = (final) => {
    let carry = '';
    if (!final && buffer.endsWith('\r')) { carry = '\r'; buffer = buffer.slice(0, -1); }
    buffer = buffer.replace(/\r\n?/g, '\n');
    let idx;
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      emit(buffer.slice(0, idx));
      buffer = buffer.slice(idx + 2);
    }
    buffer += carry;
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      drain(false);
    }
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    // cancel() d'un flux déjà en erreur renvoie une promesse rejetée : on l'absorbe.
    try { reader.cancel().catch(() => {}); } catch (_) { /* déjà fermé */ }
    throw networkError(true);
  }
  buffer += decoder.decode();
  drain(true);
  if (buffer.trim()) emit(buffer);
}

/**
 * Abonnement EventSource sur une route GET. `handlers` associe un nom d'événement serveur à une
 * fonction recevant les données JSON ; `onError` est appelé si la connexion échoue ou se ferme.
 * Un événement serveur nommé `error` (contrat `/api/index/events`) ne se confond pas avec
 * l'événement natif `error` d'EventSource : seuls les MessageEvent (porteurs de `data`)
 * atteignent les gestionnaires nommés, les événements de connexion vont à `onError`.
 * Renvoie une fonction de fermeture (à appeler en quittant la vue : pas de fuite).
 */
export function subscribe(path, handlers = {}) {
  const source = new EventSource(buildUrl(path));
  for (const [name, fn] of Object.entries(handlers)) {
    if (name === 'onError' || typeof fn !== 'function') continue;
    source.addEventListener(name, (e) => {
      if (typeof e.data !== 'string') return; // Event natif (connexion) : traité par onerror
      let data = null;
      try {
        data = JSON.parse(e.data);
      } catch (_) {
        data = { text: e.data };
      }
      fn(data);
    });
  }
  source.onerror = () => {
    if (handlers.onError) handlers.onError();
  };
  return () => source.close();
}

/**
 * Suit une indexation : SSE `/api/index/events` d'abord, repli sur un sondage de
 * `/api/index/status` toutes les 1,5 s si le flux échoue. Renvoie stop().
 * `onDone(status)` est appelé au plus une fois (status peut être null si la relecture finale échoue).
 */
export function watchIndex({ onProgress, onDone, onError } = {}) {
  let stopped = false;
  let finished = false;
  let closeSource = null;
  let timer = null;

  const stop = () => {
    stopped = true;
    if (closeSource) closeSource();
    closeSource = null;
    clearTimeout(timer);
    timer = null;
  };

  const finish = async () => {
    if (finished) return;
    finished = true;
    // Fermeture synchrone AVANT la relecture : la fin du flux côté serveur ne doit pas
    // déclencher onerror → sondage parallèle → second onDone.
    stop();
    let status = null;
    try {
      status = await api.get('/api/index/status');
      if (onProgress) onProgress(status);
    } catch (e) {
      if (onError) onError(e);
    }
    if (onDone) onDone(status);
  };

  const poll = async () => {
    if (stopped) return;
    try {
      const status = await api.get('/api/index/status');
      if (stopped) return;
      if (onProgress) onProgress(status);
      if (status.running === false) {
        finished = true;
        stop();
        if (onDone) onDone(status);
        return;
      }
    } catch (e) {
      if (onError) onError(e);
    }
    if (!stopped) timer = setTimeout(poll, 1500);
  };

  try {
    closeSource = subscribe('/api/index/events', {
      progress: (data) => { if (!stopped && onProgress) onProgress(data); },
      done: () => finish(),
      error: (data) => {
        if (stopped) return;
        if (onError) onError(new ApiError((data && data.code) || 'index_error', (data && data.message) || "Erreur pendant l'indexation."));
        finish();
      },
      onError: () => {
        // Connexion refusée, route absente ou flux fermé : repli sur le sondage.
        if (stopped) return;
        if (closeSource) closeSource();
        closeSource = null;
        poll();
      },
    });
  } catch (_) {
    poll();
  }
  return stop;
}
