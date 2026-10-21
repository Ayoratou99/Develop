// ui.js — petits utilitaires d'interface sans dépendance : création d'éléments,
// rendu Markdown sûr (jamais d'innerHTML sur du contenu externe), toasts, modales,
// formatage de dates et de tailles, icônes SVG.

// --------------------------------------------------------------------------- éléments

const PROPERTY_KEYS = new Set(['value', 'checked', 'disabled', 'selected', 'readOnly', 'multiple', 'indeterminate', 'open']);

/** Ajoute des enfants (texte, nœud, tableau, null) à un nœud. */
export function append(node, children) {
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    if (Array.isArray(child)) append(node, child);
    else if (child instanceof Node) node.appendChild(child);
    else node.appendChild(document.createTextNode(String(child)));
  }
  return node;
}

/**
 * el('div', {class:'x', onClick: fn, dataset:{id:1}}, 'texte', autreNœud, [liste])
 * Les chaînes deviennent des nœuds texte : aucun HTML n'est interprété.
 */
export function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  if (attrs === null || attrs === undefined) attrs = {};
  if (typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs)) {
    children.unshift(attrs);
    attrs = {};
  }
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class' || key === 'className') node.className = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === 'text') node.textContent = value;
    else if (PROPERTY_KEYS.has(key)) node[key] = value;
    else node.setAttribute(key, value === true ? '' : value);
  }
  return append(node, children);
}

/** Vide un conteneur puis y place les enfants donnés. */
export function replace(container, ...children) {
  container.replaceChildren();
  return append(container, children);
}

/** Échappe une chaîne pour un usage HTML (utile pour les attributs construits à la main). */
export function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Anti-rebond simple. */
export function debounce(fn, delay = 300) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

// --------------------------------------------------------------------------- icônes

const ICONS = {
  chat: 'M4 4h16v11H8l-4 4V4z',
  explorer: 'M3 5h6l2 2h10v12H3V5z',
  documents: 'M6 2h8l5 5v15H6V2zm8 0v5h5',
  notes: 'M5 3h14v18H5V3zm3 5h8M8 12h8M8 16h5',
  impact: 'M12 3l9 16H3L12 3zm0 6v4m0 3v1',
  settings: 'M12 8a4 4 0 100 8 4 4 0 000-8zm8 4l2-1-1-3-2 .3a7 7 0 00-1.7-1.7L17.6 4l-3-1-1 2a7 7 0 00-2.4 0l-1-2-3 1 .3 2.3A7 7 0 005.8 8L3.5 7.7l-1 3L4.5 12l-2 1 1 3 2.3-.3A7 7 0 007.5 17.4L7.2 20l3 1 1-2a7 7 0 002.4 0l1 2 3-1-.3-2.3a7 7 0 001.7-1.7l2.3.3 1-3-2-1.3z',
  search: 'M10 4a6 6 0 100 12 6 6 0 000-12zm5 11l5 5',
  copy: 'M8 8h12v12H8V8zm-4 8V4h12',
  close: 'M5 5l14 14M19 5L5 19',
  folder: 'M3 6h6l2 2h10v11H3V6z',
  file: 'M6 2h8l5 5v15H6V2z',
  git: 'M12 3a3 3 0 100 6 3 3 0 000-6zm0 12a3 3 0 100 6 3 3 0 000-6zm0-6v6',
  check: 'M4 12l5 5L20 6',
  plus: 'M12 5v14M5 12h14',
  trash: 'M4 7h16M9 7V4h6v3m-7 4v7m4-7v7M6 7l1 14h10l1-14',
  back: 'M15 5l-7 7 7 7',
  stop: 'M6 6h12v12H6z',
  send: 'M3 12l18-8-6 18-3-8-9-2z',
  refresh: 'M4 12a8 8 0 0114-5l2 2M20 4v5h-5M20 12a8 8 0 01-14 5l-2-2M4 20v-5h5',
  chevron: 'M9 5l7 7-7 7',
  warn: 'M12 4l9 16H3L12 4zm0 6v4m0 3v1',
  info: 'M12 3a9 9 0 100 18 9 9 0 000-18zm0 7v6m0-9v1',
  upload: 'M12 16V4m-5 5l5-5 5 5M4 20h16',
  link: 'M10 14l4-4m-6 6l-2 2a3 3 0 01-4-4l4-4m8-4l2-2a3 3 0 014 4l-4 4',
};

/** Icône SVG construite par createElementNS (aucune chaîne HTML). */
export function icon(name, size = 18) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.8');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  svg.classList.add('icon');
  const path = document.createElementNS(ns, 'path');
  path.setAttribute('d', ICONS[name] || ICONS.info);
  svg.appendChild(path);
  return svg;
}

// --------------------------------------------------------------------------- presse-papiers

/** Copie du texte, avec repli sur execCommand pour les contextes non sécurisés. */
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (_) {
    const area = el('textarea', { value: text, style: { position: 'fixed', opacity: '0' } });
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try {
      ok = document.execCommand('copy');
    } catch (__) {
      ok = false;
    }
    area.remove();
    return ok;
  }
}

/** Bouton « Copier » qui confirme visuellement la copie. */
export function copyButton(text, label = 'Copier') {
  const button = el('button', { class: 'btn btn-ghost btn-sm', type: 'button', title: 'Copier dans le presse-papiers' }, icon('copy', 14), ' ', label);
  button.addEventListener('click', async () => {
    const ok = await copyText(typeof text === 'function' ? text() : text);
    const previous = button.textContent;
    replace(button, icon('check', 14), ' ', ok ? 'Copié' : 'Échec');
    setTimeout(() => replace(button, icon('copy', 14), ' ', label), 1500);
    void previous;
  });
  return button;
}

// --------------------------------------------------------------------------- Markdown

const SAFE_URL = /^(https?:\/\/|mailto:|#)/i;

/** Rend le contenu « inline » (gras, italique, code, liens, citations [S1]) en nœuds DOM. */
export function renderInline(text, options = {}) {
  const nodes = [];
  const pattern = /(`+)([\s\S]*?[^`])\1(?!`)|\*\*([^*]+?)\*\*|__([^_]+?)__|(?<![\w*])\*([^*\n]+?)\*(?!\w)|(?<![\w_])_([^_\n]+?)_(?!\w)|\[(S\d+)\]|\[([^\]\n]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)|\bhttps?:\/\/[^\s<>"')\]]+/g;
  let last = 0;
  let match;
  const pushText = (chunk) => {
    const parts = chunk.split('\n');
    parts.forEach((part, i) => {
      if (i > 0) nodes.push(el('br'));
      if (part) nodes.push(document.createTextNode(part));
    });
  };
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) pushText(text.slice(last, match.index));
    last = pattern.lastIndex;
    if (match[2] !== undefined) nodes.push(el('code', { class: 'md-code' }, match[2]));
    else if (match[3] !== undefined) nodes.push(el('strong', renderInline(match[3], options)));
    else if (match[4] !== undefined) nodes.push(el('strong', renderInline(match[4], options)));
    else if (match[5] !== undefined) nodes.push(el('em', renderInline(match[5], options)));
    else if (match[6] !== undefined) nodes.push(el('em', renderInline(match[6], options)));
    else if (match[7] !== undefined) {
      const ref = match[7];
      const custom = options.citation ? options.citation(ref) : null;
      nodes.push(custom || el('span', { class: 'cite' }, `[${ref}]`));
    } else if (match[8] !== undefined) {
      const href = match[9];
      if (SAFE_URL.test(href)) nodes.push(makeLink(href, renderInline(match[8], options)));
      else nodes.push(document.createTextNode(`${match[8]} (${href})`));
    } else {
      nodes.push(makeLink(match[0], [match[0]]));
    }
  }
  if (last < text.length) pushText(text.slice(last));
  return nodes;
}

/**
 * Lien Markdown. Une URL externe (http/https) n'est jamais ouverte directement : Develop fonctionne
 * hors ligne et l'adresse peut avoir été inventée par le modèle ; un clic demande confirmation.
 */
function makeLink(href, children) {
  const external = /^https?:/i.test(href);
  if (!external) return el('a', { href, class: 'md-link' }, children);
  const open = (e) => {
    e.preventDefault();
    confirmDialog(el('div',
      el('p', 'Ouvrir ce lien externe dans votre navigateur ?'),
      el('p', { class: 'mono small', style: { wordBreak: 'break-all' } }, href),
      el('p', { class: 'muted small' }, 'Develop fonctionne hors ligne et n\'a pas vérifié cette adresse : elle peut provenir d\'une réponse du modèle.')),
    { title: 'Lien externe', okLabel: 'Ouvrir' }).then((ok) => { if (ok) window.open(href, '_blank', 'noopener'); });
  };
  return el('a', { href, class: 'md-link md-link-external', rel: 'noopener noreferrer', title: 'Lien externe (confirmation avant ouverture)', onClick: open },
    children, el('span', { class: 'md-ext', 'aria-label': 'lien externe' }, '↗'));
}

/** Bloc de code avec en-tête (langage) et bouton copier. */
export function codeBlock(code, lang = '') {
  return el('div', { class: 'codeblock' },
    el('div', { class: 'codeblock-bar' }, el('span', { class: 'codeblock-lang' }, lang || 'code'), copyButton(code)),
    el('pre', el('code', { class: 'codeblock-body' }, code)));
}

const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
const LIST_ITEM = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;

function splitTableRow(line) {
  let row = line.trim();
  if (row.startsWith('|')) row = row.slice(1);
  if (row.endsWith('|')) row = row.slice(0, -1);
  return row.split('|').map((cell) => cell.trim());
}

/** Construit une liste (ul/ol) en gérant un niveau d'imbrication par indentation. */
function renderList(lines, options) {
  const first = lines[0].match(LIST_ITEM);
  const ordered = /\d/.test(first[2]);
  const baseIndent = first[1].length;
  const list = el(ordered ? 'ol' : 'ul', { class: 'md-list' });
  let current = null;
  let nested = [];
  const flushNested = () => {
    if (current && nested.length) current.appendChild(renderList(nested, options));
    nested = [];
  };
  for (const line of lines) {
    const m = line.match(LIST_ITEM);
    if (m && m[1].length <= baseIndent) {
      flushNested();
      current = el('li', renderInline(m[3], options));
      list.appendChild(current);
    } else if (m) {
      nested.push(line);
    } else if (current) {
      // ligne de continuation
      if (nested.length) nested.push(line);
      else current.appendChild(el('span', ' ', renderInline(line.trim(), options)));
    }
  }
  flushNested();
  return list;
}

/**
 * Mini-rendu Markdown sûr → DocumentFragment. Pris en charge : titres, paragraphes, gras,
 * italique, listes (un niveau imbriqué), citations, règles, blocs de code avec bouton copier,
 * code inline, liens http(s), tableaux simples. `options.citation(ref)` personnalise [S1].
 */
export function renderMarkdown(text, options = {}) {
  const frag = document.createDocumentFragment();
  const lines = String(text ?? '').replace(/\r\n?/g, '\n').split('\n');
  let i = 0;
  const isBlockStart = (line) => /^\s*(```|~~~)/.test(line) || /^#{1,6}\s/.test(line) || /^\s*>/.test(line) || LIST_ITEM.test(line) || /^(-{3,}|\*{3,}|_{3,})\s*$/.test(line);

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i += 1; continue; }

    // bloc de code délimité
    const fence = line.match(/^\s*(```+|~~~+)\s*([\w+#.-]*)/);
    if (fence) {
      const buf = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith(fence[1][0].repeat(3))) { buf.push(lines[i]); i += 1; }
      i += 1;
      frag.appendChild(codeBlock(buf.join('\n'), fence[2]));
      continue;
    }
    // titre
    const heading = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (heading) {
      const level = Math.min(heading[1].length + 1, 6);
      frag.appendChild(el(`h${level}`, { class: `md-h md-h${heading[1].length}` }, renderInline(heading[2], options)));
      i += 1;
      continue;
    }
    // règle horizontale
    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { frag.appendChild(el('hr')); i += 1; continue; }
    // citation
    if (/^\s*>/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) { buf.push(lines[i].replace(/^\s*>\s?/, '')); i += 1; }
      frag.appendChild(el('blockquote', { class: 'md-quote' }, renderMarkdown(buf.join('\n'), options)));
      continue;
    }
    // tableau
    if (line.includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      const head = splitTableRow(line);
      const rows = [];
      i += 2;
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) { rows.push(splitTableRow(lines[i])); i += 1; }
      frag.appendChild(el('div', { class: 'table-wrap' }, el('table', { class: 'md-table' },
        el('thead', el('tr', head.map((cell) => el('th', renderInline(cell, options))))),
        el('tbody', rows.map((row) => el('tr', head.map((_, c) => el('td', renderInline(row[c] || '', options)))))))));
      continue;
    }
    // liste
    if (LIST_ITEM.test(line)) {
      const buf = [];
      while (i < lines.length && (LIST_ITEM.test(lines[i]) || (lines[i].trim() && /^\s{2,}/.test(lines[i])))) { buf.push(lines[i]); i += 1; }
      frag.appendChild(renderList(buf, options));
      continue;
    }
    // paragraphe
    const buf = [];
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i]) && !(lines[i].includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]))) {
      buf.push(lines[i]);
      i += 1;
    }
    if (!buf.length) { buf.push(lines[i]); i += 1; }
    frag.appendChild(el('p', { class: 'md-p' }, renderInline(buf.join('\n'), options)));
  }
  return frag;
}

// --------------------------------------------------------------------------- toasts

let toastHost = null;

/** Affiche une notification éphémère. type: info | success | error | warn. */
export function toast(message, { type = 'info', timeout = 4500 } = {}) {
  if (!toastHost) {
    toastHost = el('div', { class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.appendChild(toastHost);
  }
  const node = el('div', { class: `toast toast-${type}` }, el('span', { class: 'toast-text' }, message),
    el('button', { class: 'toast-close', type: 'button', 'aria-label': 'Fermer', onClick: () => node.remove() }, icon('close', 14)));
  toastHost.appendChild(node);
  if (timeout > 0) setTimeout(() => node.remove(), timeout);
  return node;
}

/** Affiche une erreur API (toast) et renvoie son message. */
export function toastError(err) {
  const message = err && err.message ? err.message : String(err);
  toast(message, { type: 'error', timeout: 7000 });
  return message;
}

/** Encadré d'erreur inline. */
export function errorBox(err, actions = []) {
  const message = err && err.message ? err.message : String(err);
  const code = err && err.code ? err.code : null;
  return el('div', { class: 'alert alert-error', role: 'alert' }, icon('warn', 18),
    el('div', { class: 'alert-body' }, el('div', message), code ? el('div', { class: 'muted small' }, `Code : ${code}`) : null, actions.length ? el('div', { class: 'alert-actions' }, actions) : null));
}

/** Encadré d'information. */
export function infoBox(children, type = 'info') {
  return el('div', { class: `alert alert-${type}` }, icon(type === 'warn' ? 'warn' : 'info', 18), el('div', { class: 'alert-body' }, children));
}

/** État vide explicite. */
export function emptyState(title, hint = null, actions = []) {
  return el('div', { class: 'empty' }, el('div', { class: 'empty-title' }, title), hint ? el('div', { class: 'empty-hint' }, hint) : null, actions.length ? el('div', { class: 'empty-actions' }, actions) : null);
}

export function spinner(label = 'Chargement…') {
  return el('div', { class: 'spinner-row' }, el('span', { class: 'spinner', 'aria-hidden': 'true' }), el('span', { class: 'muted' }, label));
}

// --------------------------------------------------------------------------- modales

/**
 * Ouvre une modale. Renvoie {root, body, close}. Fermeture par Échap, clic sur le voile ou close().
 */
export function modal({ title, body, actions = [], wide = false, onClose = null }) {
  const bodyNode = el('div', { class: 'modal-body' }, body);
  const root = el('div', { class: 'modal-backdrop', role: 'dialog', 'aria-modal': 'true', 'aria-label': title || 'Fenêtre' });
  const close = () => {
    document.removeEventListener('keydown', onKey);
    root.remove();
    if (onClose) onClose();
  };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const panel = el('div', { class: `modal${wide ? ' modal-wide' : ''}` },
    el('div', { class: 'modal-head' }, el('h2', { class: 'modal-title' }, title || ''), el('button', { class: 'btn btn-ghost btn-icon', type: 'button', 'aria-label': 'Fermer', onClick: close }, icon('close'))),
    bodyNode,
    actions.length ? el('div', { class: 'modal-actions' }, actions) : null);
  root.appendChild(panel);
  root.addEventListener('click', (e) => { if (e.target === root) close(); });
  document.addEventListener('keydown', onKey);
  document.body.appendChild(root);
  return { root, body: bodyNode, close };
}

/** Demande de confirmation. Résout true si l'utilisateur confirme. `message` : texte ou nœud. */
export function confirmDialog(message, { title = 'Confirmer', okLabel = 'Confirmer', danger = false } = {}) {
  return new Promise((resolve) => {
    let dialog = null;
    let settled = false;
    // `settled` avant close() : modal.close() exécute onClose, qui ne doit plus résoudre false.
    const finish = (value) => { settled = true; dialog.close(); resolve(value); };
    dialog = modal({
      title,
      body: message instanceof Node ? message : el('p', message),
      actions: [
        el('button', { class: 'btn', type: 'button', onClick: () => finish(false) }, 'Annuler'),
        el('button', { class: `btn ${danger ? 'btn-danger' : 'btn-primary'}`, type: 'button', onClick: () => finish(true) }, okLabel),
      ],
      onClose: () => { if (!settled) resolve(false); },
    });
  });
}

/** Demande une valeur texte. Résout la chaîne ou null si annulé. */
export function promptDialog(title, { label = '', value = '', okLabel = 'Valider', placeholder = '' } = {}) {
  return new Promise((resolve) => {
    const input = el('input', { class: 'input', type: 'text', value, placeholder, id: `prompt-${++fieldSeq}` });
    let dialog = null;
    let settled = false;
    const finish = (val) => { settled = true; dialog.close(); resolve(val); };
    const form = el('form', { class: 'form-stack', onSubmit: (e) => { e.preventDefault(); finish(input.value.trim()); } },
      label ? el('label', { class: 'label', for: input.id }, label) : null, input);
    dialog = modal({
      title,
      body: form,
      actions: [
        el('button', { class: 'btn', type: 'button', onClick: () => finish(null) }, 'Annuler'),
        el('button', { class: 'btn btn-primary', type: 'button', onClick: () => finish(input.value.trim()) }, okLabel),
      ],
      onClose: () => { if (!settled) resolve(null); },
    });
    setTimeout(() => input.focus(), 0);
  });
}

// --------------------------------------------------------------------------- formatage

function toDate(value) {
  if (value === null || value === undefined || value === '') return null;
  if (value instanceof Date) return value;
  if (typeof value === 'number') return new Date(value < 1e12 ? value * 1000 : value);
  const asNumber = Number(value);
  if (!Number.isNaN(asNumber) && /^\d+(\.\d+)?$/.test(String(value))) return toDate(asNumber);
  // AAAA-MM-JJ (note_date) : date civile locale, pas minuit UTC (sinon la veille à l'ouest de Greenwich).
  const ymd = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value).trim());
  if (ymd) return new Date(Number(ymd[1]), Number(ymd[2]) - 1, Number(ymd[3]));
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

/** Date lisible en français (secondes epoch, millisecondes, ISO ou AAAA-MM-JJ). */
export function formatDate(value, { time = true } = {}) {
  const d = toDate(value);
  if (!d) return '—';
  const opts = time ? { dateStyle: 'medium', timeStyle: 'short' } : { dateStyle: 'medium' };
  return new Intl.DateTimeFormat('fr-FR', opts).format(d);
}

/** Durée relative (« il y a 3 min »). */
export function formatRelative(value) {
  const d = toDate(value);
  if (!d) return '—';
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return "à l'instant";
  if (diff < 3600) return `il y a ${Math.round(diff / 60)} min`;
  if (diff < 86400) return `il y a ${Math.round(diff / 3600)} h`;
  if (diff < 86400 * 30) return `il y a ${Math.round(diff / 86400)} j`;
  return formatDate(d, { time: false });
}

/** Taille lisible (octets → Ko, Mo, Go). */
export function formatSize(bytes) {
  if (bytes === null || bytes === undefined || Number.isNaN(Number(bytes))) return '—';
  const n = Number(bytes);
  if (n < 1024) return `${n} o`;
  const units = ['Ko', 'Mo', 'Go', 'To'];
  let value = n / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value < 10 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

export function formatNumber(n) {
  if (n === null || n === undefined) return '—';
  return new Intl.NumberFormat('fr-FR').format(Number(n));
}

/** Date du jour au format AAAA-MM-JJ (fuseau local). */
export function todayIso() {
  const d = new Date();
  const pad = (x) => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

// --------------------------------------------------------------------------- composants

/** Barre de confiance (0..1). */
export function confidenceBar(value) {
  const pct = Math.max(0, Math.min(100, Math.round((Number(value) || 0) * 100)));
  const level = pct >= 70 ? 'high' : pct >= 40 ? 'mid' : 'low';
  return el('div', { class: 'confidence', title: `Confiance : ${pct} %` },
    el('div', { class: 'confidence-track' }, el('div', { class: `confidence-bar confidence-${level}`, style: { width: `${pct}%` } })),
    el('span', { class: 'confidence-label' }, `${pct} %`));
}

/** Pastille. */
export function badge(text, kind = 'neutral') {
  return el('span', { class: `badge badge-${kind}` }, text);
}

/** Onglets simples : tabs = [{id,label}], onSelect(id). Renvoie {root, select}. */
export function tabs(items, active, onSelect) {
  const root = el('div', { class: 'tabs', role: 'tablist' });
  const buttons = new Map();
  const select = (id) => {
    for (const [key, button] of buttons) button.classList.toggle('active', key === id);
  };
  for (const item of items) {
    const button = el('button', { class: 'tab', type: 'button', role: 'tab', onClick: () => { select(item.id); onSelect(item.id); } }, item.label);
    buttons.set(item.id, button);
    root.appendChild(button);
  }
  select(active);
  return { root, select };
}

let fieldSeq = 0;

/** Premier contrôle de formulaire (input/select/textarea) parmi `control` (nœud, tableau ou conteneur). */
function findControl(control) {
  const list = Array.isArray(control) ? control : [control];
  for (const node of list) {
    if (!(node instanceof Element)) continue;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(node.tagName)) return node;
    const inner = node.querySelector ? node.querySelector('input, select, textarea') : null;
    if (inner) return inner;
  }
  return null;
}

/** Champ de formulaire étiqueté : le <label> est associé au contrôle (for/id), l'aide par aria-describedby. */
export function field(label, control, hint = null) {
  const target = findControl(control);
  const labelNode = el('label', { class: 'label' }, label);
  const hintNode = hint ? el('div', { class: 'hint' }, hint) : null;
  if (target) {
    if (!target.id) target.id = `field-${++fieldSeq}`;
    labelNode.htmlFor = target.id;
    if (hintNode) {
      if (!hintNode.id) hintNode.id = `${target.id}-hint`;
      target.setAttribute('aria-describedby', hintNode.id);
    }
  }
  return el('div', { class: 'field' }, labelNode, control, hintNode);
}

/** Section repliable (details/summary). */
export function collapsible(summary, content, open = false) {
  return el('details', { class: 'collapsible', open }, el('summary', summary), el('div', { class: 'collapsible-body' }, content));
}

/** Liste clé → valeur. */
export function dl(entries) {
  return el('dl', { class: 'dl' }, entries.filter((e) => e).map(([k, v]) => [el('dt', k), el('dd', v)]));
}
