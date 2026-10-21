// Banc d'essai Node de la vue « Analyse d'impact » (develop/web/static/views/impact.js).
//
// Charge la vraie vue et le vrai ui.js dans un DOM minimal (sans navigateur ni dépendance),
// remplace app.js / api.js / explorer.js par des doublures, simule « Source : un fichier »
// puis « Analyser » avec le rapport reçu en argument, et imprime en JSON ce que la carte
// « Changements » affiche ainsi que le texte envoyé au chat par « Expliquer avec l'IA ».
//
// Usage : node impact_view_harness.mjs '<rapport ImpactReport en JSON>' [source=file|symbol|git]
import * as nodeModule from 'node:module';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const STATIC_DIR = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../develop/web/static');

// ------------------------------------------------------------------ DOM minimal
class StubClassList {
  constructor(owner) { this.owner = owner; this.set = new Set(); }
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
    this.classList = new StubClassList(this);
    this.style = {};
    this.dataset = {};
    this.id = '';
  }
  get className() { return this.classList.toString(); }
  set className(value) { this.classList.set = new Set(String(value).split(/\s+/).filter(Boolean)); }
  get firstChild() { return this.childNodes[0] || null; }
  get children() { return this.childNodes.filter((c) => c.tagName !== '#text'); }
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
  dispatch(type) { return Promise.all((this.listeners[type] || []).map((fn) => fn({ type, target: this, preventDefault() {} }))); }
  focus() {}
  querySelector(selector) {
    const wanted = selector.split(',').map((s) => s.trim().toUpperCase());
    const walk = (node) => {
      for (const child of node.childNodes) {
        if (wanted.includes(child.tagName)) return child;
        const found = walk(child);
        if (found) return found;
      }
      return null;
    };
    return walk(this);
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

// ------------------------------------------------------------------ doublures partagées avec les modules remplacés
const report = JSON.parse(process.argv[2] || '{}');
const sourceKind = process.argv[3] || 'file';
globalThis.__harness = {
  state: { impactPick: null, chatPrefill: null },
  navigated: [],
  requests: [],
  report,
  // Résultats renvoyés par GET /api/symbols (source « Un symbole » : le choix se fait dans cette liste).
  symbols: [{ id: 42, name: 'confirm', parent: 'OrderService', kind: 'method', line_start: 10,
    file: { path: (report.changed && report.changed[0] && report.changed[0].path) || 'app/Services/OrderService.php' } }],
};

const STUBS = {
  '/static/app.js': `export const state = globalThis.__harness.state;
export function navigate(hash) { globalThis.__harness.navigated.push(hash); }`,
  '/static/api.js': `export const api = {
  async get(p, params) { globalThis.__harness.requests.push(['GET', p, params || null]); return p === '/api/symbols' ? globalThis.__harness.symbols : []; },
  async post(p, body) { globalThis.__harness.requests.push(['POST', p, body]); return globalThis.__harness.report; },
};`,
  '/static/views/explorer.js': `export function fileHref(fileId, line) { return '#/explorer/file/' + fileId + (line ? '?line=' + line : ''); }`,
};

// Résolution des imports absolus « /static/… » : doublure (module data:) ou vrai fichier.
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
const { mount } = await import('/static/views/impact.js');
const root = new StubNode('div');
await mount(root);
await new Promise((r) => setTimeout(r, 0)); // laisse GET /api/repos se résoudre

const selects = root.findAll((n) => n.tagName === 'SELECT');
const source = selects[0];
source.value = sourceKind === 'git' ? 'git' : sourceKind;
await source.dispatch('change');
if (sourceKind === 'file') {
  const pathInput = root.findAll((n) => n.tagName === 'INPUT' && n.attributes.placeholder && n.attributes.placeholder.includes('chemin'))[0];
  pathInput.value = (report.changed && report.changed[0] && report.changed[0].path) || 'app/Services/OrderService.php';
} else if (sourceKind === 'symbol') {
  // Recherche (anti-rebond de 300 ms) puis clic sur le premier résultat.
  const search = root.findAll((n) => n.tagName === 'INPUT' && n.attributes.type === 'search')[0];
  search.value = 'confirm';
  await search.dispatch('input');
  await new Promise((r) => setTimeout(r, 400));
  const hit = root.findAll((n) => n.classList.contains('clickable'))[0];
  if (!hit) throw new Error('aucun résultat de symbole affiché');
  await hit.dispatch('click');
}
const analyze = root.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Analyser'))[0];
await analyze.dispatch('click');
await new Promise((r) => setTimeout(r, 0));

const changesCard = root.findAll((n) => n.classList.contains('card') && n.firstChild && n.firstChild.textContent === 'Changements')[0];
const rows = changesCard ? changesCard.findAll((n) => n.classList.contains('list-item')) : [];
const changes = rows.map((row) => {
  const badge = row.findAll((n) => n.classList.contains('badge'))[0];
  return { badge: badge ? badge.textContent : null, badgeClass: badge ? badge.className : null, text: row.textContent };
});
const explain = root.findAll((n) => n.tagName === 'BUTTON' && n.textContent.includes('Expliquer'))[0];
if (explain) await explain.dispatch('click');

process.stdout.write(JSON.stringify({
  changes,
  prefill: globalThis.__harness.state.chatPrefill,
  navigated: globalThis.__harness.navigated,
  requests: globalThis.__harness.requests,
  errors: root.findAll((n) => n.classList.contains('alert-error')).map((n) => n.textContent),
}, null, 1));
