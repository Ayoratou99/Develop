// views/explorer.js — recherche globale, fichiers (arbre + vue fichier avec numéros de ligne),
// endpoints, modèles de données, symboles. Deep-link : #/explorer/file/{id}?line=12.

import { api } from '/static/api.js';
import {
  el, replace, icon, errorBox, emptyState, spinner, debounce, badge, tabs, dl,
  confidenceBar, codeBlock, formatSize, formatNumber,
} from '/static/ui.js';
import { state, navigate } from '/static/app.js';

const CHUNK = 400;
const TABS = [
  { id: 'files', label: 'Fichiers' },
  { id: 'endpoints', label: 'Endpoints' },
  { id: 'models', label: 'Modèles' },
  { id: 'symbols', label: 'Symboles' },
];

/** Lien explorateur vers un fichier (+ ligne). */
export function fileHref(fileId, line) {
  return `#/explorer/file/${encodeURIComponent(fileId)}${line ? `?line=${line}` : ''}`;
}

/** Résout un résultat de recherche en route. */
function hitHref(hit) {
  switch (hit.source_type) {
    case 'document': return `#/documents/${encodeURIComponent(hit.source_id)}`;
    case 'note': return `#/notes/${encodeURIComponent(hit.source_id)}`;
    case 'symbol': return `#/explorer/symbols/${encodeURIComponent(hit.source_id)}`;
    case 'model': return `#/explorer/models/${encodeURIComponent(hit.source_id)}`;
    case 'endpoint': return `#/explorer/endpoints?q=${encodeURIComponent(hit.path || hit.title || '')}`;
    default: return fileHref(hit.source_id, hit.line_start);
  }
}

/**
 * Référence de fichier d'un objet (endpoint, modèle, appelant, source…) : {id, path, repo_id}.
 * L'identifiant de fichier vient UNIQUEMENT de `file_id` ou de `file.id` (objet) : `obj.id` est
 * l'identifiant de l'objet lui-même (endpoint, modèle, symbole) et ne désigne jamais un fichier.
 * Sans identifiant, fileLink() affiche le chemin en texte simple.
 */
function fileRef(obj) {
  if (!obj) return null;
  if (typeof obj === 'string') return { path: obj };
  if (obj.file && typeof obj.file === 'object') return { id: obj.file.id || obj.file.file_id || null, path: obj.file.path, repo_id: obj.file.repo_id || obj.repo_id };
  const path = typeof obj.file === 'string' ? obj.file : (obj.file_path || obj.path);
  return { id: obj.file_id || null, path, repo_id: obj.repo_id };
}

/**
 * Référence de fichier d'une relation (`relations.out[]` / `relations.in[]` de /api/files/{id}) :
 * objet fichier explicite (dst_file / src_file : {id, path}), identifiant (dst_file_id / src_file_id)
 * ou clé de nœud du graphe ('file:12').
 */
function relationFileRef(r, dir) {
  const out = dir === 'out';
  const fileObj = out ? (r.dst_file || r.target_file) : (r.src_file || r.source_file);
  if (fileObj && typeof fileObj === 'object') return { id: fileObj.id || fileObj.file_id || null, path: fileObj.path, repo_id: fileObj.repo_id };
  const node = out ? (r.dst || r.target) : (r.src || r.source);
  const key = typeof node === 'string' ? /^file:(\d+)$/.exec(node) : null;
  const id = (out ? (r.dst_file_id || r.target_file_id) : (r.src_file_id || r.source_file_id))
    || (key ? key[1] : null) || (node && typeof node === 'object' ? node.file_id : null) || null;
  const path = (node && typeof node === 'object') ? (node.path || node.file) : (typeof node === 'string' && !key ? node : null);
  return { id, path: path || (out ? r.dst_path : r.src_path) || null };
}

/** Bouton/lien vers un fichier:ligne quand l'identifiant est connu, sinon texte. */
function fileLink(ref, line, label) {
  const text = label || `${ref && ref.path ? ref.path : '?'}${line ? `:${line}` : ''}`;
  if (ref && ref.id) return el('a', { href: fileHref(ref.id, line), class: 'mono small' }, text);
  return el('span', { class: 'mono small' }, text);
}

/** Libellé lisible d'un objet de relation ou d'import de forme inconnue. */
function describe(item) {
  if (item === null || item === undefined) return '';
  if (typeof item !== 'object') return String(item);
  return item.label || item.name || item.path || item.module || item.target || item.dst || item.src || item.key || JSON.stringify(item);
}

/**
 * Texte d'une cellule du tableau « Champs » d'un modèle.
 * null / undefined / chaîne vide / objet ou tableau vide → « — » (jamais la chaîne « null » ni « {} »),
 * booléen → oui / non, objet non vide → paires « clé: valeur », tableau → valeurs séparées par des virgules.
 */
export function formatFieldValue(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'boolean') return value ? 'oui' : 'non';
  if (Array.isArray(value)) {
    return value.length ? value.map((v) => (v && typeof v === 'object' ? JSON.stringify(v) : formatFieldValue(v))).join(', ') : '—';
  }
  if (typeof value === 'object') {
    const keys = Object.keys(value);
    return keys.length ? keys.map((k) => `${k}: ${formatFieldValue(value[k])}`).join(', ') : '—';
  }
  return String(value);
}

/** Cellule du tableau des champs : valeur formatée, grisée quand elle est absente. */
function fieldCell(value) {
  const text = formatFieldValue(value);
  return el('td', text === '—' ? { class: 'muted' } : null, text);
}

// --------------------------------------------------------------------------- recherche globale

function searchBar(initial = '') {
  const input = el('input', { class: 'input', type: 'search', value: initial, placeholder: 'Rechercher dans le code, les documents et les notes… (noms de symboles, routes, tables, texte)' });
  const results = el('div', { class: 'search-results' });
  let seq = 0;
  const run = debounce(async () => {
    const q = input.value.trim();
    const mine = ++seq;
    state.explorerQuery = q; // conservé : les résultats reviennent après un clic (fichier, document…)
    if (!q) { results.replaceChildren(); return; }
    replace(results, spinner('Recherche…'));
    try {
      const data = await api.get('/api/search', { q, limit: 20 });
      if (mine !== seq) return;
      const hits = data.results || [];
      replace(results,
        (data.identifiers || []).length ? el('div', { class: 'row small muted', style: { marginBottom: '6px' } }, 'Identifiants repérés :', data.identifiers.map((id) => badge(id, 'accent'))) : null,
        hits.length ? el('div', { class: 'list' }, hits.map((hit) => el('div', { class: 'hit', onClick: () => navigate(hitHref(hit)) },
          badge(hit.source_type || '?'),
          el('div', { class: 'grow' },
            el('div', { class: 'row' }, el('strong', hit.title || hit.path || '?'), hit.path && hit.title !== hit.path ? el('span', { class: 'mono small muted' }, `${hit.path}${hit.line_start ? `:${hit.line_start}` : ''}${hit.page ? ` p.${hit.page}` : ''}`) : null),
            hit.snippet ? el('div', { class: 'hit-snippet' }, hit.snippet) : null,
            el('div', { class: 'row small muted' }, (hit.channels || []).map((c) => badge(c)), hit.reason ? el('span', hit.reason) : null))))) : emptyState('Aucun résultat', 'Essayez un autre terme, ou lancez l\'indexation si l\'index est vide.'));
    } catch (e) {
      if (mine === seq) replace(results, errorBox(e));
    }
  }, 300);
  input.addEventListener('input', run);
  if (initial) run();
  return el('div', { class: 'explorer-search' }, el('div', { class: 'row' }, icon('search', 18), input), results);
}

// --------------------------------------------------------------------------- arbre de fichiers

/** Construit un arbre imbriqué à partir d'une liste plate de fichiers. */
function buildTree(files) {
  const root = { dirs: new Map(), files: [] };
  for (const f of files) {
    const parts = f.path.split('/');
    let node = root;
    for (let i = 0; i < parts.length - 1; i += 1) {
      if (!node.dirs.has(parts[i])) node.dirs.set(parts[i], { dirs: new Map(), files: [] });
      node = node.dirs.get(parts[i]);
    }
    node.files.push({ ...f, name: parts[parts.length - 1] });
  }
  return root;
}

function renderTree(node, prefix, expanded, activePath, activeFileId) {
  const container = el('div', { class: 'tree-children' });
  const dirs = [...node.dirs.keys()].sort((a, b) => a.localeCompare(b));
  for (const name of dirs) {
    const path = prefix ? `${prefix}/${name}` : name;
    const isOpen = expanded.has(path) || (activePath && activePath.startsWith(`${path}/`));
    if (isOpen) expanded.add(path);
    const children = isOpen ? renderTree(node.dirs.get(name), path, expanded, activePath, activeFileId) : null;
    const row = el('div', { class: 'tree-node', onClick: () => {
      if (expanded.has(path)) expanded.delete(path); else expanded.add(path);
      const fresh = renderTree(node, prefix, expanded, activePath, activeFileId);
      container.replaceWith(fresh);
    } }, el('span', { class: 'tree-caret' }, isOpen ? '▾' : '▸'), icon('folder', 14), el('span', { class: 'truncate' }, name));
    container.appendChild(row);
    if (children) container.appendChild(children);
  }
  for (const f of node.files.sort((a, b) => a.name.localeCompare(b.name))) {
    container.appendChild(el('div', { class: `tree-node${String(f.id) === String(activeFileId) ? ' active' : ''}`, title: `${f.path} (${f.language || f.kind || ''})`, onClick: () => navigate(fileHref(f.id)) },
      el('span', { class: 'tree-caret' }), icon('file', 14), el('span', { class: 'truncate' }, f.name)));
  }
  return container;
}

/** Colonne des dépôts et de leurs arbres (chargement paresseux par dépôt). */
function repoTrees(active) {
  const host = el('div', { class: 'tree' }, spinner('Dépôts…'));
  const expandedRepos = new Set();
  const expandedDirs = new Map(); // repo_id → Set(chemins)
  const cache = new Map(); // repo_id → liste de fichiers
  let repos = [];

  const render = () => {
    if (!repos.length) { replace(host, emptyState('Aucun dépôt', 'Ajoutez des dépôts dans les réglages puis lancez l\'indexation.')); return; }
    replace(host, repos.map((repo) => {
      const open = expandedRepos.has(repo.id);
      const counts = repo.counts || {};
      const head = el('div', { class: 'repo-head', onClick: () => { if (open) expandedRepos.delete(repo.id); else expandedRepos.add(repo.id); render(); } },
        el('span', { class: 'tree-caret' }, open ? '▾' : '▸'), icon('git', 15), el('span', { class: 'truncate grow' }, repo.name),
        el('span', { class: 'muted small nowrap' }, formatNumber(counts.files || 0)));
      const body = el('div');
      if (open) {
        if (cache.has(repo.id)) {
          const files = cache.get(repo.id);
          if (!files.length) replace(body, el('div', { class: 'muted small', style: { padding: '4px 12px' } }, 'Index vide : lancez l\'indexation.'));
          else {
            if (!expandedDirs.has(repo.id)) expandedDirs.set(repo.id, new Set());
            replace(body, renderTree(buildTree(files), '', expandedDirs.get(repo.id), active && active.repo_id === repo.id ? active.path : null, active ? active.id : null));
          }
        } else {
          replace(body, spinner('Fichiers…'));
          api.get('/api/files', { repo_id: repo.id, limit: 50000 }).then((files) => { cache.set(repo.id, files || []); render(); }).catch((e) => replace(body, errorBox(e)));
        }
      }
      return el('div', head, body);
    }));
  };

  api.get('/api/repos').then((list) => {
    repos = list || [];
    if (active && active.repo_id) expandedRepos.add(active.repo_id);
    else if (repos.length === 1) expandedRepos.add(repos[0].id);
    render();
  }).catch((e) => replace(host, errorBox(e)));

  return { root: host, setActive: (file) => { active = file; if (file && file.repo_id) expandedRepos.add(file.repo_id); render(); } };
}

// --------------------------------------------------------------------------- vue fichier

/** Vue d'un fichier : contenu par tranches de 400 lignes + encadrés d'analyse. */
function fileView(fileId, targetLine) {
  const root = el('div', spinner('Chargement du fichier…'));
  const codeHost = el('div', { class: 'code-view' });
  let nextStart = 1;
  let totalLines = null;
  let loading = false;
  let info = null;

  const moreBtn = el('button', { class: 'btn btn-sm', type: 'button', onClick: () => loadChunk() }, 'Charger les 400 lignes suivantes');
  const moreRow = el('div', { class: 'code-more hidden' }, moreBtn);

  const appendLines = (data) => {
    const lines = data.text === '' ? [] : String(data.text ?? '').replace(/\n$/, '').split('\n');
    const first = Number(data.start) || nextStart;
    lines.forEach((text, i) => {
      const n = first + i;
      codeHost.appendChild(el('div', { class: 'code-line', id: `L${n}`, dataset: { line: n } }, el('span', { class: 'code-ln' }, String(n)), el('span', { class: 'code-text' }, text)));
    });
    nextStart = first + lines.length;
    totalLines = Number(data.total_lines) || totalLines;
    moreRow.classList.toggle('hidden', !(totalLines && nextStart <= totalLines));
    moreBtn.disabled = false;
  };

  const loadChunk = async () => {
    if (loading) return null;
    loading = true;
    moreBtn.disabled = true;
    try {
      const data = await api.get(`/api/files/${encodeURIComponent(fileId)}/content`, { start: nextStart, end: nextStart + CHUNK - 1 });
      appendLines(data);
      return data;
    } catch (e) {
      codeHost.appendChild(errorBox(e));
      return null;
    } finally {
      loading = false;
    }
  };

  const scrollToLine = (n) => {
    codeHost.querySelectorAll('.code-line.highlight').forEach((x) => x.classList.remove('highlight'));
    const node = codeHost.querySelector(`#L${n}`);
    if (!node) return false;
    node.classList.add('highlight');
    node.scrollIntoView({ block: 'center' });
    return true;
  };

  /** Charge les tranches nécessaires jusqu'à la ligne demandée puis la surligne. */
  const goToLine = async (n) => {
    n = Number(n);
    if (!n) return;
    let guard = 0;
    while (!codeHost.querySelector(`#L${n}`) && guard < 50) {
      if (totalLines && nextStart > totalLines) break;
      const data = await loadChunk();
      if (!data) break;
      guard += 1;
    }
    scrollToLine(n);
  };

  const sideList = (title, items, renderItem) => el('div', { class: 'card' }, el('div', { class: 'card-title' }, title, badge(String(items.length))),
    items.length ? el('div', { class: 'stack', style: { gap: '4px' } }, items.map(renderItem)) : el('div', { class: 'muted small' }, 'Aucun'));

  api.get(`/api/files/${encodeURIComponent(fileId)}`).then(async (data) => {
    info = data;
    const relations = data.relations || {};
    replace(root,
      el('div', { class: 'row-between', style: { marginBottom: '10px' } },
        el('div', el('h2', { class: 'mono', style: { margin: 0 } }, data.path), el('div', { class: 'muted small' }, [data.language, data.kind, data.size !== undefined ? formatSize(data.size) : null, data.repo_id ? `dépôt ${data.repo_id}` : null].filter(Boolean).join(' · '))),
        el('div', { class: 'row' }, el('button', { class: 'btn btn-sm', type: 'button', onClick: () => { state.impactPick = { file_id: data.id, path: data.path, repo_id: data.repo_id }; navigate('#/impact'); } }, icon('impact', 14), 'Analyser l\'impact'))),
      codeHost, moreRow,
      el('div', { class: 'panel-grid', style: { marginTop: '14px' } },
        sideList('Symboles', data.symbols || [], (s) => el('div', { class: 'row small' }, badge(s.kind || '?'), el('button', { class: 'btn-link mono', type: 'button', onClick: () => goToLine(s.line_start) }, s.name), s.line_start ? el('span', { class: 'muted' }, `l. ${s.line_start}`) : null, s.id ? el('a', { href: `#/explorer/symbols/${encodeURIComponent(s.id)}`, class: 'small' }, 'détail') : null)),
        sideList('Endpoints', data.endpoints || [], (e) => el('div', { class: 'row small' }, badge(e.side || '?', e.side === 'server' ? 'ok' : 'accent'), el('span', { class: 'mono' }, `${e.method || ''} ${e.path || ''}`), e.handler ? el('span', { class: 'muted' }, e.handler) : null, e.line ? el('button', { class: 'btn-link', type: 'button', onClick: () => goToLine(e.line) }, `l. ${e.line}`) : null)),
        sideList('Modèles', data.models || [], (m) => el('div', { class: 'row small' }, badge(m.kind || '?'), m.id ? el('a', { href: `#/explorer/models/${encodeURIComponent(m.id)}` }, m.name) : el('span', m.name), m.table_name ? el('span', { class: 'muted mono' }, m.table_name) : null)),
        sideList('Imports', data.imports || [], (imp) => el('div', { class: 'small mono' }, describe(imp), imp && imp.resolved_file_id ? el('a', { href: fileHref(imp.resolved_file_id), style: { marginLeft: '6px' } }, '→') : null)),
        sideList('Relations', [...(relations.out || []).map((r) => ({ ...r, dir: 'out' })), ...(relations.in || []).map((r) => ({ ...r, dir: 'in' }))], (r) => el('div', { class: 'row small' },
          badge(r.dir === 'out' ? 'sortante' : 'entrante', r.dir === 'out' ? 'accent' : 'ok'), r.kind ? el('span', { class: 'muted' }, r.kind) : null,
          fileLink(relationFileRef(r, r.dir), r.line, describe(r.dir === 'out' ? (r.dst_file || r.target || r.dst || r) : (r.src_file || r.source || r.src || r))),
          r.confidence !== undefined ? confidenceBar(r.confidence) : null)),
        sideList('Documents qui en parlent', data.documents_mentioning || [], (d) => el('a', { href: `#/documents/${encodeURIComponent(d.id || d.document_id)}`, class: 'small' }, d.title || d.path || String(d.id)))));
    await loadChunk();
    if (targetLine) await goToLine(targetLine);
  }).catch((e) => replace(root, errorBox(e)));

  return { root, get info() { return info; } };
}

// --------------------------------------------------------------------------- endpoints

function endpointsTab(initialQuery = '') {
  const side = el('select', { class: 'select', style: { width: 'auto' } }, el('option', { value: '' }, 'Serveur et client'), el('option', { value: 'server' }, 'Serveur'), el('option', { value: 'client' }, 'Client'));
  const q = el('input', { class: 'input', type: 'search', value: initialQuery, placeholder: 'Filtrer par chemin, méthode, gestionnaire…' });
  const host = el('div');
  const load = debounce(async () => {
    replace(host, spinner('Endpoints…'));
    try {
      const items = await api.get('/api/endpoints', { side: side.value, q: q.value.trim() });
      if (!items.length) { replace(host, emptyState('Aucun endpoint', 'Aucune route ou appel HTTP détecté avec ce filtre.')); return; }
      replace(host, el('div', { class: 'table-wrap' }, el('table',
        el('thead', el('tr', ['Côté', 'Méthode', 'Chemin', 'Gestionnaire', 'Fichier', 'Correspondances'].map((h) => el('th', h)))),
        el('tbody', items.map((e) => el('tr',
          el('td', badge(e.side === 'server' ? 'serveur' : 'client', e.side === 'server' ? 'ok' : 'accent'), e.framework ? el('div', { class: 'muted small' }, e.framework) : null),
          el('td', el('span', { class: 'mono' }, e.method || '')),
          el('td', el('span', { class: 'mono' }, e.path || ''), e.normalized_path && e.normalized_path !== e.path ? el('div', { class: 'muted small mono' }, e.normalized_path) : null),
          el('td', e.handler || ''),
          el('td', fileLink(fileRef(e), e.line)),
          el('td', (e.matches || []).length ? el('div', { class: 'stack', style: { gap: '4px' } }, e.matches.map((m) => el('div', { class: 'row small' },
            badge(m.side === 'server' ? 'serveur' : 'client', m.side === 'server' ? 'ok' : 'accent'), el('span', { class: 'mono' }, m.path || ''), m.handler ? el('span', { class: 'muted' }, m.handler) : null,
            fileLink(fileRef(m), m.line), confidenceBar(m.confidence)))) : el('span', { class: 'muted small' }, 'aucune'))))))));
    } catch (e) { replace(host, errorBox(e)); }
  }, 300);
  side.addEventListener('change', load);
  q.addEventListener('input', load);
  load();
  return el('div', el('div', { class: 'row', style: { marginBottom: '10px' } }, side, q), host);
}

// --------------------------------------------------------------------------- modèles

function modelsTab(selectedId) {
  const q = el('input', { class: 'input', type: 'search', placeholder: 'Filtrer les modèles / tables…' });
  const listHost = el('div');
  const detail = el('div');
  const load = debounce(async () => {
    replace(listHost, spinner('Modèles…'));
    try {
      const items = await api.get('/api/models', { q: q.value.trim() });
      if (!items.length) { replace(listHost, emptyState('Aucun modèle', 'Aucun modèle de données (migration, ORM, table SQL, interface) détecté.')); return; }
      replace(listHost, el('div', { class: 'list' }, items.map((m) => el('div', { class: `list-item clickable${String(m.id) === String(selectedId) ? ' active' : ''}`, onClick: () => navigate(`#/explorer/models/${encodeURIComponent(m.id)}`) },
        el('div', { class: 'list-item-main' }, el('div', { class: 'list-item-title row' }, m.name, badge(m.kind || '?')),
          el('div', { class: 'list-item-meta' }, m.table_name ? el('span', { class: 'mono' }, m.table_name) : null, el('span', `${m.field_count ?? 0} champ(s)`), m.file ? el('span', { class: 'mono truncate' }, m.file) : null))))));
    } catch (e) { replace(listHost, errorBox(e)); }
  }, 300);
  q.addEventListener('input', load);
  load();

  if (selectedId) {
    replace(detail, spinner('Modèle…'));
    api.get(`/api/models/${encodeURIComponent(selectedId)}`).then((m) => {
      const fields = m.fields || [];
      const cols = fields.length ? Object.keys(fields[0]).filter((k) => !['id', 'model_id'].includes(k)) : [];
      replace(detail,
        el('h2', m.name, ' ', badge(m.kind || '?')),
        dl([['Table', m.table_name || '—'], ['Fichier', fileLink(fileRef(m), m.line)]]),
        el('div', { class: 'card', style: { marginTop: '12px' } }, el('div', { class: 'card-title' }, 'Champs', badge(String(fields.length))),
          fields.length ? el('div', { class: 'table-wrap' }, el('table', el('thead', el('tr', cols.map((c) => el('th', c)))), el('tbody', fields.map((f) => el('tr', cols.map((c) => fieldCell(f[c]))))))) : el('div', { class: 'muted small' }, 'Aucun champ détecté')),
        el('div', { class: 'panel-grid' },
          el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Relations'), (m.relations || []).length ? el('div', { class: 'stack', style: { gap: '4px' } }, m.relations.map((r) => el('div', { class: 'small row' }, badge(r.kind || r.type || 'relation'), el('span', describe(r.target || r.related || r.model || r.name || r)), r.confidence !== undefined ? confidenceBar(r.confidence) : null))) : el('div', { class: 'muted small' }, 'Aucune')),
          el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Éléments liés'), (m.linked || []).length ? el('div', { class: 'stack', style: { gap: '4px' } }, m.linked.map((l) => el('div', { class: 'small row' }, badge(l.kind || '?'), l.model_id ? el('a', { href: `#/explorer/models/${encodeURIComponent(l.model_id)}` }, l.name) : el('span', l.name), l.file ? el('span', { class: 'mono muted' }, l.file) : null))) : el('div', { class: 'muted small' }, 'Aucun (migration ↔ ORM ↔ table ↔ interface)')),
          el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Endpoints qui l\'utilisent'), (m.endpoints_using || []).length ? el('div', { class: 'stack', style: { gap: '4px' } }, m.endpoints_using.map((e) => el('div', { class: 'small row' }, el('span', { class: 'mono' }, `${e.method || ''} ${e.path || ''}`), fileLink(fileRef(e), e.line)))) : el('div', { class: 'muted small' }, 'Aucun')),
          el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Documents qui en parlent'), (m.documents_mentioning || []).length ? el('div', { class: 'stack', style: { gap: '4px' } }, m.documents_mentioning.map((d) => el('a', { href: `#/documents/${encodeURIComponent(d.id || d.document_id)}`, class: 'small' }, d.title || String(d.id)))) : el('div', { class: 'muted small' }, 'Aucun'))));
    }).catch((e) => replace(detail, errorBox(e)));
  } else {
    replace(detail, emptyState('Sélectionnez un modèle', 'Champs, relations, éléments liés et documents apparaîtront ici.'));
  }
  return el('div', { class: 'split' }, el('div', q, el('div', { style: { height: '8px' } }), listHost), detail);
}

// --------------------------------------------------------------------------- symboles

function symbolsTab(selectedId) {
  const q = el('input', { class: 'input', type: 'search', placeholder: 'Nom de fonction, classe, méthode…' });
  const listHost = el('div', el('div', { class: 'muted small' }, 'Saisissez un nom pour rechercher des symboles.'));
  const detail = el('div');
  const load = debounce(async () => {
    const term = q.value.trim();
    if (!term) return;
    replace(listHost, spinner('Symboles…'));
    try {
      const items = await api.get('/api/symbols', { q: term, limit: 60 });
      if (!items.length) { replace(listHost, emptyState('Aucun symbole')); return; }
      replace(listHost, el('div', { class: 'list' }, items.map((s) => el('div', { class: `list-item clickable${String(s.id) === String(selectedId) ? ' active' : ''}`, onClick: () => navigate(`#/explorer/symbols/${encodeURIComponent(s.id)}`) },
        el('div', { class: 'list-item-main' }, el('div', { class: 'list-item-title row' }, el('span', { class: 'mono' }, s.parent ? `${s.parent}.${s.name}` : s.name), badge(s.kind || '?')),
          el('div', { class: 'list-item-meta' }, s.file ? el('span', { class: 'mono truncate' }, `${s.file.path}:${s.line_start || ''}`) : null))))));
    } catch (e) { replace(listHost, errorBox(e)); }
  }, 300);
  q.addEventListener('input', load);

  if (selectedId) {
    replace(detail, spinner('Symbole…'));
    api.get(`/api/symbols/${encodeURIComponent(selectedId)}`).then((s) => {
      const file = fileRef(s);
      const fullName = s.parent ? `${s.parent}.${s.name}` : s.name;
      replace(detail,
        el('div', { class: 'row-between', style: { marginBottom: '8px' } },
          el('h2', { class: 'mono', style: { margin: 0 } }, fullName, ' ', badge(s.kind || '?')),
          el('button', { class: 'btn btn-sm', type: 'button', onClick: () => {
            state.impactPick = { symbol_id: s.id, name: fullName, path: file ? file.path : null, line_start: s.line_start, repo_id: file ? file.repo_id : null };
            navigate('#/impact');
          } }, icon('impact', 14), 'Analyser l\'impact')),
        dl([['Signature', el('code', s.signature || '—')], ['Fichier', fileLink(file, s.line_start)], ['Lignes', s.line_start ? `${s.line_start}–${s.line_end || s.line_start}` : '—']]),
        s.snippet ? codeBlock(s.snippet, (s.file && s.file.language) || '') : null,
        el('div', { class: 'panel-grid' },
          el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Appelants', badge(String((s.callers || []).length))), (s.callers || []).length ? el('div', { class: 'stack', style: { gap: '4px' } }, s.callers.map((c) => el('div', { class: 'small row' }, c.symbol_id || c.id ? el('a', { href: `#/explorer/symbols/${encodeURIComponent(c.symbol_id || c.id)}`, class: 'mono' }, c.name || describe(c)) : el('span', { class: 'mono' }, c.name || describe(c)), fileLink(fileRef(c), c.line || c.line_start), c.kind ? el('span', { class: 'muted' }, c.kind) : null))) : el('div', { class: 'muted small' }, 'Aucun appelant connu')),
          el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Utilise', badge(String((s.uses || []).length))), (s.uses || []).length ? el('div', { class: 'stack', style: { gap: '4px' } }, s.uses.map((u) => el('div', { class: 'small row' }, u.symbol_id || u.id ? el('a', { href: `#/explorer/symbols/${encodeURIComponent(u.symbol_id || u.id)}`, class: 'mono' }, u.name || describe(u)) : el('span', { class: 'mono' }, u.name || describe(u)), fileLink(fileRef(u), u.line || u.line_start)))) : el('div', { class: 'muted small' }, 'Aucune dépendance connue'))));
      if (!q.value) { q.value = s.name; load(); }
    }).catch((e) => replace(detail, errorBox(e)));
  } else {
    replace(detail, emptyState('Recherchez un symbole', 'Le détail affiche la signature, l\'extrait de code et les appelants.'));
  }
  return el('div', { class: 'split' }, el('div', q, el('div', { style: { height: '8px' } }), listHost), detail);
}

// --------------------------------------------------------------------------- vue

export async function mount(root, route) {
  const [first, second] = route.parts;
  let tab = 'files';
  let fileId = null;
  let selectedId = null;
  if (first === 'file') { tab = 'files'; fileId = second; } else if (TABS.some((t) => t.id === first)) { tab = first; selectedId = second || null; }

  const content = el('div');
  const tabBar = tabs(TABS, tab, (id) => navigate(`#/explorer/${id}`));
  const page = el('div', { class: 'page' },
    el('div', { class: 'page-head' }, el('div', el('h1', 'Explorateur'), el('div', { class: 'page-sub' }, 'Fichiers, routes, modèles et symboles de tous les dépôts, avec leurs relations.'))),
    // Sur l'onglet Endpoints, `q` est le filtre de routes, pas la recherche globale.
    searchBar(tab === 'endpoints' ? (state.explorerQuery || '') : (route.query.q || state.explorerQuery || '')), tabBar.root, content);
  replace(root, page);

  if (tab === 'files') {
    const trees = repoTrees(null);
    const right = el('div');
    if (fileId) {
      const view = fileView(fileId, route.query.line);
      replace(right, view.root);
      // Ouvre le dépôt et le dossier du fichier actif une fois ses métadonnées connues.
      api.get(`/api/files/${encodeURIComponent(fileId)}`).then((f) => trees.setActive({ id: f.id, path: f.path, repo_id: f.repo_id })).catch(() => {});
    } else {
      replace(right, emptyState('Choisissez un fichier', 'Le contenu s\'affiche avec ses symboles, endpoints, modèles, imports et relations.'));
    }
    replace(content, el('div', { class: 'split' }, trees.root, right));
  } else if (tab === 'endpoints') {
    replace(content, endpointsTab(route.query.q || ''));
  } else if (tab === 'models') {
    replace(content, modelsTab(selectedId));
  } else if (tab === 'symbols') {
    replace(content, symbolsTab(selectedId));
  }
  return () => {};
}
