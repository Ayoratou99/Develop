// Banc d'essai Node de l'envoi pendant un chargement de modèle dans la vue chat (views/chat.js : send,
// modelBlocked) : ce que dit la notification et ce qu'il advient du message une fois le modèle chargé.
//
// Charge la vraie vue et le vrai ui.js dans un DOM minimal (sans navigateur ni dépendance), remplace app.js
// par une doublure et `fetch` par un faux moteur : GET /api/llm/state renvoie un modèle **en cours de
// chargement**, le banc termine le chargement en diffusant `develop:llm-state` comme app.js le fait d'après
// /api/llm/events. Le DOM minimal ignore `disabled` : Entrée atteint `send()` même sur un champ grisé, ce qui
// couvre aussi le cas d'un chargement signalé entre la saisie et l'envoi.
//
// Scénario : modèle en cours de chargement, texte saisi, Entrée → notification ; fin du chargement → rien ne doit
// partir tout seul (le texte reste saisi, le bouton Envoyer redevient actif) et la notification ne doit pas
// avoir promis un envoi automatique.
//
// Usage : node chat_send_loading_harness.mjs '<fixtures JSON>'
//   { "llm_state": {...GET /api/llm/state initial}, "status": {...GET /api/status} }
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
  /** Déclenche les écouteurs (avec les champs de `extra`, ex. `key`) ; promesse résolue quand les écouteurs async ont fini. */
  dispatch(type, extra = {}) {
    const evt = { type, target: this, currentTarget: this, preventDefault() {}, stopPropagation() {}, ...extra };
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

// ------------------------------------------------------------------ fixtures et faux moteur
const fixtures = JSON.parse(process.argv[2] || '{}');
const requests = [];
const conversations = [{ id: 7, title: 'OrderService' }];
const db = {
  7: { id: 7, title: 'OrderService', messages: [{ id: 1, role: 'user', content: 'Que fait OrderService ?', sources: [], created_at: '2026-10-21T10:00:00' }] },
};
/** État du moteur, forme de GET /api/llm/state : le modèle est en cours de chargement à l'ouverture. */
let llm = { engine: 'ollama', model: 'qwen3.5:9b', embedding_model: 'qwen3-embedding:0.6b', known: true, loaded: false, embedding_loaded: false,
  loading: true, loading_since: Math.floor(Date.now() / 1000) - 3, last_load_s: null, error: null, ...(fixtures.llm_state || {}) };
const llmState = () => ({ ...llm });
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
  if (method === 'GET' && p === '/api/llm/state') return jsonResponse(llmState());
  if (method === 'POST' && p === '/api/llm/load') {
    if (!llm.loaded) llm = { ...llm, loading: true, loading_since: Math.floor(Date.now() / 1000), error: null };
    return jsonResponse(llmState(), 202);
  }
  if (method === 'GET' && p === '/api/conversations') return jsonResponse(conversations);
  if (method === 'GET' && (m = p.match(/^\/api\/conversations\/([^/]+)$/))) {
    const conv = db[decodeURIComponent(m[1])];
    return conv ? jsonResponse(conv) : jsonResponse({ error: { code: 'conversation_not_found', message: 'introuvable' } }, 404);
  }
  if (method === 'POST' && (m = p.match(/^\/api\/conversations\/([^/]+)\/messages$/))) {
    return { ok: true, status: 200, headers: { get: () => 'text/event-stream' }, body: new ReadableStream({ start() {} }) };
  }
  return jsonResponse({});
};

/** Fin du chargement côté moteur, diffusée comme app.js le fait d'après /api/llm/events. */
function finishLoad() {
  llm = { ...llm, loaded: true, embedding_loaded: true, loading: false, loading_since: null, last_load_s: 12.3 };
  globalThis.dispatchEvent(new CustomEvent('develop:llm-state', { detail: llmState() }));
}

globalThis.__harness = {
  state: { status: fixtures.status || { engine: { engine: 'ollama', ok: true, model: 'qwen3.5:9b' }, workspace: { llm: { engine: 'ollama', embedding_model: 'qwen3-embedding:0.6b' } } },
    llm: null, chatPrefill: null, route: { name: 'chat', parts: ['7'], query: {} } },
  navigated: [],
  statusRefreshes: 0,
};

const STUBS = {
  '/static/app.js': `export const state = globalThis.__harness.state;
export function navigate(hash) { globalThis.__harness.navigated.push(hash); }
export async function refreshStatus() { globalThis.__harness.statusRefreshes += 1; return globalThis.__harness.state.status; }
export function rememberChatRoute() {}`,
  '/static/views/setup.js': `export const ENGINE_LABELS = { ollama: 'Ollama', openai_compat: 'Serveur compatible OpenAI', none: 'Aucun moteur' };`,
};

const STATIC_URL = pathToFileURL(STATIC_DIR).href;
function resolveStatic(specifier, context, next) {
  if (specifier in STUBS) return { url: 'data:text/javascript,' + encodeURIComponent(STUBS[specifier]), shortCircuit: true };
  if (specifier.startsWith('/static/')) return { url: STATIC_URL + specifier.slice('/static'.length), shortCircuit: true };
  return next(specifier, context);
}
if (typeof nodeModule.registerHooks === 'function') {
  nodeModule.registerHooks({ resolve: resolveStatic });
} else {
  const hooks = `const STUBS = ${JSON.stringify(STUBS)}; const STATIC_URL = ${JSON.stringify(STATIC_URL)};
export ${resolveStatic.toString().replace('resolveStatic', 'resolve')}`;
  nodeModule.register('data:text/javascript,' + encodeURIComponent(hooks), import.meta.url);
}

// ------------------------------------------------------------------ scénario
const chat = await import('/static/views/chat.js');
const settle = () => new Promise((r) => setTimeout(r, 15));
const out = { steps: {} };
const sends = () => requests.filter(([m, p]) => m === 'POST' && /^\/api\/conversations\/[^/]+\/messages$/.test(p)).length;
/** Textes des notifications affichées (ui.js : toasts dans document.body). */
const toasts = () => document.body.findAll((n) => n.classList.contains('toast-text')).map((n) => n.textContent);

const root = new StubNode('div');
const button = (label) => root.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes(label))[0] || null;
function snapshot() {
  const banner = root.findAll((n) => n.classList.contains('chat-model'))[0];
  const send = button('Envoyer');
  const input = root.findAll((n) => n.tagName === 'TEXTAREA')[0];
  const kind = banner ? [...banner.classList.set].find((c) => c.startsWith('chat-model-') && c !== 'chat-model-line') : null;
  return {
    banner: kind ? kind.replace('chat-model-', '') : null, // loading | loaded | unloaded | unknown
    sendDisabled: send ? !!send.disabled : null, inputDisabled: input ? !!input.disabled : null,
    inputValue: input ? input.value : null,
    userMessages: root.findAll((n) => n.classList.contains('msg-user')).length,
    sends: sends(), toasts: toasts(),
  };
}

const unmount = await chat.mount(root, { name: 'chat', parts: ['7'], query: {} });
await settle();
out.steps.loading = snapshot();

// Texte saisi puis Entrée pendant le chargement
const input = root.findAll((n) => n.tagName === 'TEXTAREA')[0];
input.value = 'Qui appelle POST /orders ?';
await input.dispatch('keydown', { key: 'Enter', shiftKey: false });
await settle();
out.steps.enterWhileLoading = snapshot();

// Fin du chargement : rien ne doit partir tout seul
finishLoad();
await settle();
await settle();
out.steps.loaded = snapshot();
unmount();

out.requests = requests.map(([m, p]) => `${m} ${p}`);
process.stdout.write(JSON.stringify(out, null, 1), () => process.exit(0));
