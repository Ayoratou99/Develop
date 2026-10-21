// Banc d'essai Node des aides de l'interface (chat.js : bandeau « Recherche lexicale seule », pied de
// message budget ; setup.js : formulaire moteur avec case « think », num_ctx conseillé, avertissement
// « modèle sans outils »).
//
// Charge les vraies vues et le vrai ui.js dans un DOM minimal (sans navigateur ni dépendance), remplace
// app.js / api.js par des doublures, exécute les scénarios et imprime en JSON ce qui est affiché.
//
// Usage : node chat_help_harness.mjs '<réponses des doublures en JSON>'
//   { "profile": {...GET /api/system/profile}, "llm_test": {...POST /api/llm/test}, "status": {...GET /api/status} }
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
  /** Un <select> sans valeur explicite vaut son option sélectionnée (sinon la première), comme dans un navigateur. */
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
  dispatch(type) { return Promise.all((this.listeners[type] || []).map((fn) => fn({ type, target: this, preventDefault() {}, stopPropagation() {} }))); }
  focus() {}
  scrollIntoView() {}
  querySelector(selector) {
    const wanted = selector.split(',').map((s) => s.trim().toUpperCase());
    return this.findAll((n) => wanted.includes(n.tagName))[0] || null;
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
  body: new StubNode('body'),
};
globalThis.window = globalThis;
const sessionStore = {};
globalThis.sessionStorage = {
  getItem: (k) => (k in sessionStore ? sessionStore[k] : null),
  setItem: (k, v) => { sessionStore[k] = String(v); },
};

// ------------------------------------------------------------------ doublures
const fixtures = JSON.parse(process.argv[2] || '{}');
globalThis.__harness = {
  state: { status: fixtures.status || null, chatPrefill: null },
  requests: [],
  fixtures,
};

const STUBS = {
  '/static/app.js': `export const state = globalThis.__harness.state;
export function navigate() {}
export async function refreshStatus() { return globalThis.__harness.state.status; }`,
  '/static/api.js': `const f = globalThis.__harness.fixtures;
export const api = {
  async get(p, params) {
    globalThis.__harness.requests.push(['GET', p, params || null]);
    if (p === '/api/system/profile') return f.profile || {};
    if (p === '/api/llm/models') return f.models || [];
    if (p === '/api/index/status') return { running: false };
    return {};
  },
  async post(p, body) {
    globalThis.__harness.requests.push(['POST', p, body]);
    if (p === '/api/llm/test') return f.llm_test || { ok: false };
    return {};
  },
  async put() { return {}; }, async patch() { return {}; }, async del() { return null; }, async upload() { return {}; },
};
export async function streamPost() {}
export function watchIndex() { return () => {}; }`,
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

// ------------------------------------------------------------------ scénarios
const chat = await import('/static/views/chat.js');
const setup = await import('/static/views/setup.js');
const tick = () => new Promise((r) => setTimeout(r, 0));
const text = (node) => (node ? node.textContent : null);
const out = {};

// 1. pied de message budget
const budget = fixtures.budget || { num_ctx: 8192, context_tokens_used: 2758, max_context_tokens: 2758, source: 'num_ctx 8192 = …' };
const truncated = fixtures.truncated || [];
out.footer = chat.budgetFooter(budget, fixtures.retained ?? 7, truncated).map((n) => ({ class: n.className, text: text(n), title: n.attributes.title || null,
  links: n.findAll((c) => c.tagName === 'A').map((a) => a.attributes.href) }));
out.footerSingular = chat.budgetFooter({ num_ctx: 4096, context_tokens_used: 300, max_context_tokens: 900 }, 1, [{ path: 'a.php' }]).map(text);
out.footerEmpty = chat.budgetFooter({}, 0, []).length;
out.footerNoSource = chat.budgetFooter({ num_ctx: 4096, context_tokens_used: 0, max_context_tokens: 900 }, 0, []).map(text);

// 2. bandeau embeddings : affiché, masqué pour la session, absent sans moteur ou avec embeddings
const status = fixtures.status || { workspace: { llm: { engine: 'ollama', embedding_model: '' } } };
const banner = chat.embeddingsBanner(status);
out.banner = banner ? {
  class: banner.className,
  text: text(banner),
  links: banner.findAll((n) => n.tagName === 'A').map((a) => a.attributes.href),
  code: banner.findAll((n) => n.tagName === 'CODE').map(text),
  copyButtons: banner.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Copier')).length,
} : null;
if (banner) {
  const host = new StubNode('section');
  host.appendChild(banner);
  const dismiss = banner.findAll((n) => n.tagName === 'BUTTON' && (n.attributes['aria-label'] || '').includes('Masquer'))[0];
  if (dismiss) await dismiss.dispatch('click');
  out.bannerAfterDismiss = { attached: banner.parentNode === host, again: chat.embeddingsBanner(status) !== null, session: { ...sessionStore } };
}
out.bannerNoEngine = chat.embeddingsBanner({ workspace: { llm: { engine: 'none', embedding_model: '' } } }) !== null;
out.bannerWithEmbeddings = chat.embeddingsBanner({ workspace: { llm: { engine: 'ollama', embedding_model: 'nomic-embed-text' } } }) !== null;
out.bannerNoStatus = chat.embeddingsBanner(null) !== null;

// 3. num_ctx conseillé d'après la RAM et avertissement « sans outils »
out.advice = Object.fromEntries([64, 32, 24, 16, 8, 0, null, 'x'].map((ram) => [String(ram), setup.adviseNumCtx(ram)]));
out.toolsWarning = {
  without: text(setup.toolsWarning({ name: 'deepseek-coder-v2:16b', supports_tools: false })),
  withTools: setup.toolsWarning({ name: 'qwen3.5:9b', supports_tools: true }),
  unknown: setup.toolsWarning(null),
};

// 4. formulaire moteur monté avec un modèle déjà configuré : sonde silencieuse, case think, aide num_ctx
const llm = fixtures.llm || { engine: 'ollama', base_url: 'http://127.0.0.1:11434', model: 'deepseek-coder-v2:16b', embedding_model: '', api_key: '', num_ctx: 8192, think: false };
const form = setup.engineForm(llm);
await tick(); await tick(); await tick();
const checkbox = form.findAll((n) => n.tagName === 'INPUT' && n.attributes.type === 'checkbox')[0];
const checkboxLabel = checkbox ? checkbox.parentNode : null;
const ctxInput = form.findAll((n) => n.tagName === 'INPUT' && n.attributes.type === 'number')[0];
const ctxHint = form.findAll((n) => n.classList.contains('hint') && n.textContent.includes('contexte'))[0];
out.form = {
  thinkLabel: checkboxLabel ? text(checkboxLabel) : null,
  thinkChecked: checkbox ? !!checkbox.checked : null,
  ctxValue: ctxInput ? ctxInput.value : null,
  ctxHint: text(ctxHint),
  ctxUseButton: ctxHint ? ctxHint.findAll((n) => n.tagName === 'BUTTON').map(text) : [],
  toolsWarning: form.findAll((n) => n.classList.contains('alert-warn')).map(text),
  llmTestRequests: globalThis.__harness.requests.filter(([m, p]) => m === 'POST' && p === '/api/llm/test').map(([, , body]) => body),
};
if (checkbox) { checkbox.checked = true; await checkbox.dispatch('change'); }
const useBtn = ctxHint ? ctxHint.findAll((n) => n.tagName === 'BUTTON')[0] : null;
if (useBtn) await useBtn.dispatch('click');
out.formAfter = { think: llm.think, num_ctx: llm.num_ctx, ctxHint: text(ctxHint), ctxUseButton: ctxHint ? ctxHint.findAll((n) => n.tagName === 'BUTTON').length : null };

process.stdout.write(JSON.stringify(out, null, 1));
