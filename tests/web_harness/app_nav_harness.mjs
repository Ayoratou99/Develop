// Banc d'essai Node du routeur et de la barre latérale (develop/web/static/app.js) avec la vraie vue
// chat (views/chat.js) et le vrai magasin de flux (streams.js) : le lien « Chat » de la barre latérale
// ramène à la conversation affichée en dernier, réponse en cours comprise.
//
// Scénario de l'utilisateur (imprimé en JSON) : Chat → envoyer une question (la conversation est créée
// par la vue, URL remplacée sans hashchange) → pendant « Le modèle lit N sources… », cliquer
// « Réglages » puis « Chat » dans la barre latérale → la réponse en cours doit être à l'écran, sans
// second clic ; puis fin du flux en direct ; puis « Nouvelle » ramène le lien sur une conversation vide.
//
// Charge les vrais modules app.js, views/chat.js, streams.js, api.js et ui.js dans un DOM minimal (sans
// navigateur ni dépendance) ; les autres vues sont des doublures qui notent leur montage. `location.hash`
// déclenche `hashchange` de façon asynchrone comme un navigateur ; `history.replaceState` ne le fait pas.
// `fetch` est remplacé : POST …/messages renvoie un flux SSE piloté par le scénario, POST /api/conversations
// crée la conversation dans la « base » (qui enregistre aussi le message utilisateur dès réception).
//
// Usage : node app_nav_harness.mjs '<fixtures JSON>'
//   { "status": {...GET /api/status}, "llm_state": {...GET /api/llm/state, modèle chargé par défaut} }
import * as nodeModule from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const STATIC_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../develop/web/static');

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
    if (child.tagName === '#FRAGMENT') {
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
  findAll(pred, out = []) {
    for (const child of this.childNodes) {
      if (child.tagName !== '#text' && pred(child)) out.push(child);
      child.findAll(pred, out);
    }
    return out;
  }
}

// Coquille de index.html : barre latérale (#nav, #status-panel) et zone centrale (#main).
const body = new StubNode('body');
const navHost = new StubNode('nav'); navHost.setAttribute('id', 'nav');
const statusHost = new StubNode('div'); statusHost.setAttribute('id', 'status-panel');
const main = new StubNode('main'); main.setAttribute('id', 'main');
body.appendChild(navHost); body.appendChild(statusHost); body.appendChild(main);

globalThis.Node = StubNode;
globalThis.Element = StubNode;
globalThis.document = {
  title: 'Develop',
  createElement: (tag) => new StubNode(tag),
  createElementNS: (_ns, tag) => new StubNode(tag),
  createTextNode: (text) => new StubNode(null, String(text)),
  createDocumentFragment: () => new StubNode('#fragment'),
  body,
  getElementById: (id) => body.findAll((n) => n.id === id)[0] || null,
  querySelectorAll: (selector) => body.querySelectorAll(selector),
  querySelector: (selector) => body.querySelector(selector),
};
globalThis.window = globalThis;
const windowEvents = new EventTarget();
globalThis.addEventListener = (type, fn) => windowEvents.addEventListener(type, fn);
globalThis.removeEventListener = (type, fn) => windowEvents.removeEventListener(type, fn);
globalThis.dispatchEvent = (evt) => windowEvents.dispatchEvent(evt);

// location.hash : changer la valeur déclenche `hashchange` (tâche asynchrone, comme un navigateur) ;
// history.replaceState change l'URL sans événement.
let currentHash = '#/chat';
const hashChanges = [];
globalThis.location = {
  origin: 'http://127.0.0.1:8765',
  get hash() { return currentHash; },
  set hash(value) {
    const target = String(value).startsWith('#') ? String(value) : `#${value}`;
    if (target === currentHash) return;
    currentHash = target;
    hashChanges.push(target);
    setTimeout(() => windowEvents.dispatchEvent(new Event('hashchange')), 0);
  },
};
globalThis.history = { replaceState(_state, _title, url) { currentHash = String(url); } };
const sessionStore = {};
globalThis.sessionStorage = {
  getItem: (k) => (k in sessionStore ? sessionStore[k] : null),
  setItem: (k, v) => { sessionStore[k] = String(v); },
};
const localStore = {}; // mémoire des pieds de message de la vue chat (évite le localStorage expérimental de Node)
globalThis.localStorage = {
  getItem: (k) => (k in localStore ? localStore[k] : null),
  setItem: (k, v) => { localStore[k] = String(v); },
};

// ------------------------------------------------------------------ fixtures et faux fetch
const fixtures = JSON.parse(process.argv[2] || '{}');
const status = fixtures.status || {
  needs_setup: false,
  engine: { engine: 'ollama', ok: true, model: 'qwen3.5:9b' },
  index: { running: false, counts: { files: 12 } },
  offline: { mode: 'strict', blocked_count: 0 },
  workspace: { name: 'demo', llm: { engine: 'ollama', embedding_model: 'qwen3-embedding:0.6b' } },
};
const llmState = fixtures.llm_state || { engine: 'ollama', model: 'qwen3.5:9b', known: true, loaded: true, loading: false };
const db = {}; // « base » : conversations créées par la vue, messages utilisateur enregistrés dès réception
const conversations = [];
const requests = [];
const sseStreams = [];
let nextId = 9;

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

const jsonResponse = (payload, code = 200) => ({
  ok: code < 400, status: code, headers: { get: (h) => (h.toLowerCase() === 'content-type' ? 'application/json' : null) },
  json: async () => payload, text: async () => JSON.stringify(payload),
});

globalThis.fetch = async (url, init = {}) => {
  const method = init.method || 'GET';
  const p = new URL(url).pathname;
  const reqBody = init.body ? JSON.parse(init.body) : null;
  requests.push([method, p, reqBody]);
  let m;
  if (method === 'POST' && (m = p.match(/^\/api\/conversations\/([^/]+)\/messages$/))) {
    const id = decodeURIComponent(m[1]);
    if (db[id]) db[id].messages.push({ id: db[id].messages.length + 1, role: 'user', content: reqBody.content, sources: [], created_at: '2026-10-21T10:00:00' });
    const sse = makeSse(init.signal);
    sse.conversationId = id;
    sseStreams.push(sse);
    return { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body: sse.stream };
  }
  if (method === 'POST' && p === '/api/conversations') {
    const created = { id: nextId, title: (reqBody && reqBody.title) || '' };
    nextId += 1;
    db[String(created.id)] = { ...created, messages: [] };
    conversations.unshift(created);
    return jsonResponse(created, 201);
  }
  if (method === 'GET' && p === '/api/conversations') return jsonResponse(conversations);
  if (method === 'GET' && (m = p.match(/^\/api\/conversations\/([^/]+)$/))) {
    const conv = db[decodeURIComponent(m[1])];
    return conv ? jsonResponse(conv) : jsonResponse({ error: { code: 'conversation_not_found', message: 'introuvable' } }, 404);
  }
  if (method === 'GET' && p === '/api/status') return jsonResponse(status);
  if (method === 'GET' && p === '/api/llm/state') return jsonResponse(llmState);
  return jsonResponse({});
};

// Doublures des autres vues : elles notent leur montage et leur démontage.
globalThis.__harness = { mounts: [], unmounts: [] };
const stubView = (name) => `export async function mount(root, route) {
  globalThis.__harness.mounts.push([${JSON.stringify(name)}, route.parts]);
  const node = document.createElement('div'); node.className = 'page page-${name}'; root.appendChild(node);
  return () => { globalThis.__harness.unmounts.push(${JSON.stringify(name)}); };
}`;
const STUBS = {
  '/static/views/setup.js': `export const ENGINE_LABELS = { ollama: 'Ollama', openai_compat: 'Serveur compatible OpenAI', none: 'Aucun moteur' };\n${stubView('setup')}`,
  '/static/views/explorer.js': stubView('explorer'),
  '/static/views/documents.js': stubView('documents'),
  '/static/views/notes.js': stubView('notes'),
  '/static/views/impact.js': stubView('impact'),
  '/static/views/settings.js': stubView('settings'),
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
const settle = () => new Promise((r) => setTimeout(r, 15));
const text = (node) => (node ? node.textContent : null);
const sse = (name, data) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
const out = { requests, hashChanges, mounts: globalThis.__harness.mounts };

const navLink = (name) => navHost.findAll((n) => n.classList.contains('nav-item') && n.dataset.name === name)[0];
/** Clic sur une entrée de la barre latérale : le navigateur suit son href. */
const clickNav = async (name) => { globalThis.location.hash = navLink(name).getAttribute('href'); await settle(); await settle(); };

/** Ce qui est à l'écran : URL, vue montée, cible du lien « Chat », pastille, fil du chat. */
function snapshot() {
  const thread = main.findAll((n) => n.classList.contains('chat-thread'))[0];
  const bodyOf = (n) => text(n.findAll((c) => c.classList.contains('msg-body'))[0]);
  const badge = navHost.findAll((n) => n.id === 'nav-chat-busy')[0];
  const stop = main.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Arrêter'))[0];
  const send = main.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Envoyer'))[0];
  const active = navHost.findAll((n) => n.classList.contains('nav-item') && n.classList.contains('active')).map((n) => n.dataset.name);
  return {
    hash: globalThis.location.hash,
    title: document.title,
    active,
    chatHref: navLink('chat').getAttribute('href'),
    badgeHidden: badge ? badge.classList.contains('hidden') : null,
    badgeConversation: badge ? badge.dataset.conversation : null,
    page: main.findAll((n) => n.classList.contains('page'))[0] ? main.findAll((n) => n.classList.contains('page'))[0].className : null,
    thread: thread ? {
      users: thread.findAll((n) => n.classList.contains('msg-user')).map(bodyOf),
      assistants: thread.findAll((n) => n.classList.contains('msg-assistant')).map((n) => ({
        text: bodyOf(n),
        status: n.findAll((c) => c.classList.contains('msg-status') && !c.classList.contains('hidden')).map(text),
        foot: text(n.findAll((c) => c.classList.contains('msg-foot'))[0]) || '',
      })),
      empty: thread.findAll((n) => n.classList.contains('chat-empty')).map((n) => text(n.findAll((c) => c.tagName === 'H2' || c.tagName === 'H3' || c.classList.contains('empty-title'))[0]) || text(n)),
      stopHidden: stop ? stop.classList.contains('hidden') : null,
      sendDisabled: send ? !!send.disabled : null,
      list: main.findAll((n) => n.classList.contains('conv-item')).map((n) => ({
        title: text(n.findAll((c) => c.classList.contains('conv-title'))[0]),
        busy: n.findAll((c) => c.classList.contains('conv-busy')).length > 0,
        active: n.classList.contains('active'),
      })),
    } : null,
  };
}

// Démarrage sur #/chat (nouvelle conversation) : app.js monte la vue chat au chargement.
const app = await import('/static/app.js');
const streams = await import('/static/streams.js');
await settle(); await settle();
out.start = snapshot();

// Envoi d'une question : la vue crée la conversation (URL remplacée sans hashchange) et lance le flux.
const question = 'Que fait OrderService ?';
const input = main.findAll((n) => n.tagName === 'TEXTAREA')[0];
const sendBtn = main.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Envoyer'))[0];
input.value = question;
await sendBtn.dispatch('click');
await settle();
const sseA = sseStreams[0];
sseA.push(sse('status', { text: 'Recherche dans le projet…' }));
sseA.push(sse('sources', { items: [{ ref: 'S1', source_type: 'file', path: 'app/Services/OrderService.php', file_id: 3, line_start: 10, line_end: 40 }] }));
sseA.push(sse('status', { text: 'Le modèle lit 1 source (≈ 300 tokens) avant de répondre…' }));
await settle();
out.sent = { ...snapshot(), stateRoute: app.state.route, runningIds: streams.runningIds() };

// Pendant la lecture du contexte : « Réglages » dans la barre latérale, puis « Chat ».
await clickNav('settings');
out.settings = { ...snapshot(), running: streams.runningIds() };
await clickNav('chat');
out.back = { ...snapshot(), running: streams.runningIds() };

// Fin du flux en direct, telle que la vue l'affiche après le retour.
sseA.push(sse('delta', { text: 'OrderService gère les commandes [S1].' }));
await settle();
sseA.push(sse('done', { message_id: 2, usage: { prompt_tokens: 300, completion_tokens: 8 },
  budget: { num_ctx: 8192, context_tokens_used: 300, max_context_tokens: 3000, source: 'num_ctx 8192' },
  truncated_sources: [], invalid_citations: [], cancelled: false, mode: 'auto' }));
sseA.close();
await settle();
out.finished = { ...snapshot(), running: streams.runningIds() };

// Un autre écran, puis retour : la conversation terminée reste la cible du lien.
await clickNav('explorer');
await clickNav('chat');
out.backAfterDone = snapshot();

// « Nouvelle » : le lien « Chat » redevient une conversation vide, le retour depuis Réglages aussi.
const newBtn = main.findAll((n) => n.tagName === 'BUTTON' && n.getAttribute('title') === 'Nouvelle conversation')[0];
await newBtn.dispatch('click');
await settle(); await settle();
out.fresh = snapshot();
await clickNav('settings');
await clickNav('chat');
out.freshBack = snapshot();

process.stdout.write(JSON.stringify(out, null, 1), () => process.exit(0));
