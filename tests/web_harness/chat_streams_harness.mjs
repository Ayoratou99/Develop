// Banc d'essai Node du magasin de flux du chat (develop/web/static/streams.js) et de la vue chat
// (views/chat.js) : la réponse en cours survit au changement d'écran.
//
// Charge les vrais modules streams.js, api.js, ui.js et views/chat.js dans un DOM minimal (sans
// navigateur ni dépendance). `fetch` est remplacé : POST …/messages renvoie un corps
// `text/event-stream` sous forme de ReadableStream piloté par le scénario (push / close, et
// erreur AbortError quand le signal est coupé), les GET renvoient les fixtures (« base »).
//
// Scénario (imprimé en JSON) :
//  A. magasin seul : un abonné tardif reçoit d'abord le tampon (replay) puis le direct ;
//  B. vue montée pendant le flux : message reconstruit ; démontage sans coupure ; remontage ;
//     done en direct ; purge après rechargement quand la base contient la réponse ;
//  C. « Arrêter » : signal coupé, flux terminé `aborted`, encadré « Génération arrêtée. » ;
//  D. rechargement d'un flux terminé absent de la base : rejoué une dernière fois puis purgé ;
//  E. envoi depuis la vue : délégué à startStream (requête POST), boutons Envoyer / Arrêter.
//
// Usage : node chat_streams_harness.mjs '<fixtures JSON>'
//   { "conversations": [...GET /api/conversations], "db": { "7": {...GET /api/conversations/7}, "8": {...} }, "status": {...},
//     "llm_state": {...GET /api/llm/state, modèle chargé par défaut} }
import * as nodeModule from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

// fileURLToPath : valable aussi sous Windows (« new URL(...).pathname » y donne « /C:/... » et laisse les %20).
const STATIC_DIR = fileURLToPath(new URL('../../develop/web/static', import.meta.url));

// ------------------------------------------------------------------ DOM minimal
class StubClassList {
  constructor() { this.set = new Set(); }
  add(...names) { names.forEach((n) => this.set.add(n)); }
  remove(...names) { names.forEach((n) => this.set.delete(n)); }
  contains(name) { return this.set.has(name); }
  toggle(name, force) {
    const on = force === undefined ? !this.set.has(name) : Boolean(force);
    if (on) this.set.add(name); else this.set.delete(name);
    return on;
  }
  toString() { return [...this.set].join(' '); }
}

class StubNode {
  constructor(tagName, text = null) {
    this.tagName = tagName ? tagName.toUpperCase() : '#text';
    this.nodeValue = text;
    this.childNodes = [];
    this.parentNode = null;
    this.attributes = {};
    this.listeners = {};
    this.classList = new StubClassList();
    this.style = {};
    this.dataset = {};
    this.id = '';
    this._value = undefined;
  }
  get className() { return this.classList.toString(); }
  set className(value) { this.classList.set = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get firstChild() { return this.childNodes[0] || null; }
  get children() { return this.childNodes.filter((c) => c.tagName !== '#text'); }
  get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
  get value() {
    if (this._value === undefined && this.tagName === 'SELECT') {
      const chosen = this.options.find((o) => o.selected) || this.options[0];
      return chosen ? chosen.value : '';
    }
    return this._value === undefined ? '' : this._value;
  }
  set value(v) { this._value = v; }
  get textContent() {
    if (this.tagName === '#text') return this.nodeValue;
    return this.childNodes.map((c) => c.textContent).join('');
  }
  set textContent(value) { this.childNodes = [new StubNode(null, String(value))]; }
  appendChild(child) {
    if (child.tagName === '#FRAGMENT') { // un fragment se vide dans son parent, comme dans un navigateur
      for (const c of child.childNodes.slice()) this.appendChild(c);
      child.childNodes = [];
      return child;
    }
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  removeChild(child) {
    const i = this.childNodes.indexOf(child);
    if (i >= 0) this.childNodes.splice(i, 1);
    child.parentNode = null;
    return child;
  }
  replaceChildren(...nodes) { this.childNodes.forEach((c) => { c.parentNode = null; }); this.childNodes = []; nodes.forEach((n) => this.appendChild(n)); }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  setAttribute(name, value) { this.attributes[name] = String(value); if (name === 'id') this.id = String(value); }
  getAttribute(name) { return name in this.attributes ? this.attributes[name] : null; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn); }
  /** Déclenche les écouteurs et renvoie une promesse résolue quand les écouteurs async ont fini. */
  dispatch(type) {
    const evt = { type, target: this, currentTarget: this, preventDefault() {}, stopPropagation() {} };
    return Promise.all((this.listeners[type] || []).map((fn) => fn(evt)));
  }
  dispatchEvent(evt) { this.dispatch(evt.type); return true; }
  focus() {}
  scrollIntoView() {}
  matches(selector) {
    const s = selector.trim();
    if (s.startsWith('.')) return this.classList.contains(s.slice(1));
    if (s.startsWith('#')) return this.id === s.slice(1);
    return this.tagName === s.toUpperCase();
  }
  querySelectorAll(selector) {
    const wanted = selector.split(',');
    return this.findAll((n) => wanted.some((s) => n.matches(s)));
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  closest(selector) {
    for (let n = this; n; n = n.parentNode) if (n.matches && n.matches(selector)) return n;
    return null;
  }
  /** Tous les descendants (éléments) satisfaisant le prédicat. */
  findAll(pred, out = []) {
    for (const child of this.childNodes) {
      if (child.tagName !== '#text' && pred(child)) out.push(child);
      child.findAll(pred, out);
    }
    return out;
  }
}

globalThis.Node = StubNode;
globalThis.Element = StubNode;
globalThis.document = {
  createElement: (tag) => new StubNode(tag),
  createElementNS: (_ns, tag) => new StubNode(tag),
  createTextNode: (text) => new StubNode(null, String(text)),
  createDocumentFragment: () => new StubNode('#fragment'),
  body: new StubNode('body'),
};
globalThis.window = globalThis;
// window.addEventListener / dispatchEvent (événements applicatifs comme `develop:llm-state`)
const windowEvents = new EventTarget();
globalThis.addEventListener = (type, fn) => windowEvents.addEventListener(type, fn);
globalThis.removeEventListener = (type, fn) => windowEvents.removeEventListener(type, fn);
globalThis.dispatchEvent = (evt) => windowEvents.dispatchEvent(evt);
globalThis.location = { origin: 'http://127.0.0.1:8765', hash: '#/chat/7' };
globalThis.history = { replaceState() {} };
const sessionStore = {};
globalThis.sessionStorage = {
  getItem: (k) => (k in sessionStore ? sessionStore[k] : null),
  setItem: (k, v) => { sessionStore[k] = String(v); },
};
const localStore = {}; // mémoire locale des pieds de message (views/chat.js : rememberFooter / recallFooter)
globalThis.localStorage = {
  getItem: (k) => (k in localStore ? localStore[k] : null),
  setItem: (k, v) => { localStore[k] = String(v); },
};

// ------------------------------------------------------------------ fixtures et faux fetch
const fixtures = JSON.parse(process.argv[2] || '{}');
const db = fixtures.db || {};
const conversations = fixtures.conversations || [];
const requests = [];
const sseStreams = []; // flux SSE ouverts par POST …/messages, dans l'ordre

/** Flux SSE piloté : push(texte), close(), et erreur AbortError quand le signal est coupé. */
function makeSse(signal) {
  let controller = null;
  const encoder = new TextEncoder();
  const stream = new ReadableStream({ start(c) { controller = c; } });
  const handle = {
    stream,
    signal,
    closed: false,
    push(text) { controller.enqueue(encoder.encode(text)); },
    close() { if (!handle.closed) { handle.closed = true; controller.close(); } },
  };
  if (signal) {
    signal.addEventListener('abort', () => {
      if (handle.closed) return;
      handle.closed = true;
      controller.error(new DOMException('The operation was aborted.', 'AbortError'));
    });
  }
  return handle;
}

const jsonResponse = (payload, status = 200) => ({
  ok: status < 400, status, headers: { get: (h) => (h.toLowerCase() === 'content-type' ? 'application/json' : null) },
  json: async () => payload, text: async () => JSON.stringify(payload),
});

globalThis.fetch = async (url, init = {}) => {
  const method = init.method || 'GET';
  const p = new URL(url).pathname;
  const body = init.body ? JSON.parse(init.body) : null;
  requests.push([method, p, body]);
  let m;
  if (method === 'POST' && (m = p.match(/^\/api\/conversations\/([^/]+)\/messages$/))) {
    const sse = makeSse(init.signal);
    sse.conversationId = decodeURIComponent(m[1]);
    sseStreams.push(sse);
    return { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body: sse.stream };
  }
  if (method === 'GET' && p === '/api/conversations') return jsonResponse(conversations);
  if (method === 'GET' && (m = p.match(/^\/api\/conversations\/([^/]+)$/))) {
    const conv = db[decodeURIComponent(m[1])];
    return conv ? jsonResponse(conv) : jsonResponse({ error: { code: 'conversation_not_found', message: 'introuvable' } }, 404);
  }
  if (method === 'POST' && p === '/api/conversations') return jsonResponse({ id: 9, title: (body && body.title) || '' }, 201);
  if (method === 'GET' && p === '/api/llm/state') return jsonResponse(fixtures.llm_state || { engine: 'ollama', model: 'qwen3.5:9b', known: true, loaded: true, loading: false });
  return jsonResponse({});
};

globalThis.__harness = {
  state: { status: fixtures.status || { engine: { engine: 'ollama', ok: true, model: 'qwen3.5:9b' }, workspace: { llm: { engine: 'ollama', embedding_model: 'qwen3-embedding:0.6b' } } },
    chatPrefill: null, route: { name: 'chat', parts: ['7'], query: {} } },
  navigated: [],
  statusRefreshes: 0,
};

const STUBS = {
  '/static/app.js': `export const state = globalThis.__harness.state;
export function navigate(hash) { globalThis.__harness.navigated.push(hash); }
export function rememberChatRoute() {}
export async function refreshStatus() { globalThis.__harness.statusRefreshes += 1; return globalThis.__harness.state.status; }`,
  '/static/views/setup.js': `export const ENGINE_LABELS = { ollama: 'Ollama', openai_compat: 'Serveur compatible OpenAI', none: 'Aucun moteur' };`,
};

const STATIC_URL = pathToFileURL(STATIC_DIR).href;
function resolveStatic(specifier, context, next) {
  if (specifier in STUBS) return { url: 'data:text/javascript,' + encodeURIComponent(STUBS[specifier]), shortCircuit: true };
  if (specifier.startsWith('/static/')) return { url: STATIC_URL + specifier.slice('/static'.length), shortCircuit: true };
  return next(specifier, context);
}
if (typeof nodeModule.registerHooks === 'function') {
  nodeModule.registerHooks({ resolve: resolveStatic }); // Node ≥ 22.15 / 23.5
} else {
  const hooks = `const STUBS = ${JSON.stringify(STUBS)}; const STATIC_URL = ${JSON.stringify(STATIC_URL)};
export ${resolveStatic.toString().replace('resolveStatic', 'resolve')}`;
  nodeModule.register('data:text/javascript,' + encodeURIComponent(hooks), import.meta.url);
}

// ------------------------------------------------------------------ scénario
const streams = await import('/static/streams.js');
const chat = await import('/static/views/chat.js');
const settle = () => new Promise((r) => setTimeout(r, 15));
const text = (node) => (node ? node.textContent : null);
const sse = (name, data) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
const out = { requests };

/** Ce que la vue affiche : messages du fil, boutons, liste des conversations. */
function snapshot(root) {
  const thread = root.findAll((n) => n.classList.contains('chat-thread'))[0];
  const bodyOf = (n) => text(n.findAll((c) => c.classList.contains('msg-body'))[0]);
  const stop = root.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Arrêter'))[0];
  const send = root.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Envoyer'))[0];
  return {
    users: thread.findAll((n) => n.classList.contains('msg-user')).map(bodyOf),
    assistants: thread.findAll((n) => n.classList.contains('msg-assistant')).map((n) => ({
      text: bodyOf(n),
      status: n.findAll((c) => c.classList.contains('msg-status') && !c.classList.contains('hidden')).map(text),
      sources: n.findAll((c) => c.classList.contains('source')).length,
      tools: n.findAll((c) => c.tagName === 'DETAILS' && !text(c).startsWith('Sources utilisées')).length,
      foot: text(n.findAll((c) => c.classList.contains('msg-foot'))[0]) || '',
      alerts: n.findAll((c) => c.classList.contains('alert')).map(text),
    })),
    empty: thread.findAll((n) => n.classList.contains('chat-empty')).length,
    stopHidden: stop ? stop.classList.contains('hidden') : null,
    sendDisabled: send ? !!send.disabled : null,
    list: root.findAll((n) => n.classList.contains('conv-item')).map((n) => ({
      title: text(n.findAll((c) => c.classList.contains('conv-title'))[0]),
      busy: n.findAll((c) => c.classList.contains('conv-busy')).length > 0,
      active: n.classList.contains('active'),
    })),
  };
}

const changes = [];
streams.onChange((ids) => changes.push(ids.slice()));

// A. magasin seul : tampon rejoué à un abonné tardif, puis direct
const question = 'Que fait OrderService ?';
const entry7 = streams.startStream(7, { content: question, mode: 'auto' });
await settle();
const sse7 = sseStreams[0];
const early = [];
streams.subscribe(7, (name, data, meta) => early.push([name, meta.replay]));
sse7.push(sse('status', { text: 'Recherche dans le projet…' }));
sse7.push(sse('sources', { items: [{ ref: 'S1', source_type: 'file', path: 'app/Services/OrderService.php', file_id: 3, line_start: 10, line_end: 40 }] }));
sse7.push(sse('status', { text: 'Le modèle lit 1 source (≈ 300 tokens) avant de répondre…' }));
await settle();
sse7.push(sse('delta', { text: 'OrderService gère ' }));
await settle();
const late = [];
const offLate = streams.subscribe(7, (name, data, meta) => late.push([name, meta.replay]));
sse7.push(sse('delta', { text: 'les commandes [S1].' }));
await settle();
offLate();
out.store = {
  early, late,
  text: entry7.text, status: entry7.status, sources: entry7.sources.map((s) => s.ref), running: entry7.running, done: entry7.done,
  runningIds: streams.runningIds(), isRunning7: streams.isRunning(7), isRunning8: streams.isRunning(8),
  events: entry7.events.map((e) => e.name), noStream: streams.subscribe(99, () => {}),
};

// B. vue montée pendant le flux : reconstruction depuis le tampon (la base n'a que le message utilisateur)
const root = new StubNode('div');
const mountChat = (id) => chat.mount(root, { name: 'chat', parts: id ? [String(id)] : [], query: {} });
let unmount = await mountChat(7);
await settle();
out.mounted = snapshot(root);
// démontage : le flux n'est pas coupé et continue de remplir le tampon
unmount();
sse7.push(sse('delta', { text: ' Elle valide le stock.' }));
await settle();
out.afterUnmount = { signalAborted: sse7.signal.aborted, running: entry7.running, text: entry7.text, subscribers: entry7.subscribers.size };
// remontage : tout le texte reçu entre-temps, puis la fin en direct
unmount = await mountChat(7);
await settle();
out.remounted = snapshot(root);
const done7 = { message_id: 2, usage: { prompt_tokens: 1200, completion_tokens: 18 },
  budget: { num_ctx: 8192, context_tokens_used: 300, max_context_tokens: 3000, source: 'num_ctx 8192', ...(fixtures.done_budget || {}) },
  truncated_sources: [], invalid_citations: [], cancelled: false, mode: 'auto' };
sse7.push(sse('done', done7));
sse7.close();
await settle();
out.afterDone = { ...snapshot(root), running: entry7.running, done: entry7.done, messageId: entry7.messageId,
  kept: streams.getStream(7) !== null, statusRefreshes: globalThis.__harness.statusRefreshes };
// la base contient maintenant la réponse : au rechargement, elle fait foi et le flux est purgé
db['7'].messages.push({ id: 2, role: 'assistant', content: entry7.text, sources: entry7.sources, created_at: '2026-10-21T10:00:05' });
unmount();
unmount = await mountChat(7);
await settle();
out.afterReload = { ...snapshot(root), purged: streams.getStream(7) === null,
  footers: JSON.parse(localStore['develop.chat.footers'] || '[]').map((e) => e.key) };
unmount();

// C. « Arrêter » pendant une réponse affichée
const entry8 = streams.startStream(8, { content: 'Explique la facturation', mode: 'no_tools' });
await settle();
const sse8 = sseStreams[1];
sse8.push(sse('status', { text: 'Recherche dans le projet…' }));
sse8.push(sse('sources', { items: [] }));
sse8.push(sse('delta', { text: 'La facturation' }));
await settle();
unmount = await mountChat(8);
await settle();
out.beforeAbort = snapshot(root);
const stopBtn = root.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Arrêter'))[0];
await stopBtn.dispatch('click');
await settle();
out.abort = { ...snapshot(root), signalAborted: sse8.signal.aborted, aborted: entry8.aborted, running: entry8.running,
  text: entry8.text, lastEvent: entry8.events[entry8.events.length - 1].name, runningIds: streams.runningIds() };
// D. rechargement : la base n'a pas (encore) la réponse partielle → rejouée une dernière fois, puis purgée
unmount();
unmount = await mountChat(8);
await settle();
out.abortReload = { ...snapshot(root), purged: streams.getStream(8) === null };
unmount();
unmount = await mountChat(8);
await settle();
out.abortReloadAgain = snapshot(root);
unmount();

// E. envoi depuis la vue : délégué à startStream ; une seconde conversation peut répondre en même temps
unmount = await mountChat(7);
await settle();
const input = root.findAll((n) => n.tagName === 'TEXTAREA')[0];
const sendBtn = root.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Envoyer'))[0];
input.value = 'Et ensuite ?';
await sendBtn.dispatch('click');
await settle();
const sse7b = sseStreams[2];
sse7b.push(sse('status', { text: 'Recherche dans le projet…' }));
await settle();
input.value = 'Encore ?';
await sendBtn.dispatch('click'); // ignoré : une réponse est déjà en cours ici
await settle();
out.send = { ...snapshot(root), posts: requests.filter(([m, p]) => m === 'POST' && p.endsWith('/messages')).map(([, p, b]) => [p, b]),
  runningIds: streams.runningIds(), streamsOpened: sseStreams.length, inputValue: input.value };
unmount();
out.changes = changes;

process.stdout.write(JSON.stringify(out, null, 1), () => process.exit(0));
