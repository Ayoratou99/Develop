// views/impact.js — analyse d'impact : changements Git en cours, plage de commits,
// un fichier ou un symbole → éléments affectés groupés avec raison, confiance et sauts.

import { api } from '/static/api.js';
import {
  el, replace, icon, toastError, errorBox, infoBox, emptyState, spinner, badge, confidenceBar, field, dl, formatNumber, debounce,
} from '/static/ui.js';
import { state, navigate } from '/static/app.js';
import { fileHref } from '/static/views/explorer.js';

const GROUPS = [
  ['endpoints', 'Endpoints'],
  ['models', 'Modèles'],
  ['frontend_calls', 'Appels frontend'],
  ['files', 'Fichiers'],
  ['documents', 'Documents'],
  ['notes', 'Notes'],
];

/** Statistiques du rapport (ImpactAnalyzer) → libellés ; les clés inconnues ne sont pas affichées. */
const STAT_LABELS = {
  changed_files: 'fichiers modifiés', seeds: 'points de départ', affected_total: 'éléments affectés',
  hubs_skipped: 'nœuds très connectés ignorés', max_hops: 'profondeur (sauts)', nodes_visited: 'nœuds visités',
  truncated: 'résultat tronqué',
};
const TYPE_LABELS = {
  file: 'fichier', symbol: 'symbole', endpoint: 'endpoint', model: 'modèle', table: 'table',
  document: 'document', note: 'note', frontend_call: 'appel frontend',
};
/** Statut d'un fichier du rapport (ImpactAnalyzer : statuts Git traduits, « selected » pour un fichier ou un symbole choisi) → libellé. */
const STATUS_LABELS = {
  selected: 'sélectionné', modified: 'modifié', added: 'ajouté', deleted: 'supprimé', renamed: 'renommé', untracked: 'non suivi',
};
/** Style de pastille par statut ; les statuts inconnus restent neutres. */
const STATUS_KINDS = { selected: 'neutral', modified: 'accent', added: 'ok', deleted: 'error', renamed: 'accent', untracked: 'warn' };

/** Libellé français d'un statut ; un statut inconnu est affiché tel quel, un statut absent vaut « modifié ». */
function statusLabel(status) {
  if (!status) return STATUS_LABELS.modified;
  return STATUS_LABELS[status] || String(status);
}

/** Pastille de statut d'un fichier changé. */
function statusBadge(status) {
  return badge(statusLabel(status), STATUS_KINDS[status] || (status ? 'neutral' : 'accent'));
}

/** Ligne de statistiques lisible (« fichiers modifiés : 2 · nœuds visités : 42 · par type : … »). */
function statsLine(stats) {
  const parts = [];
  for (const [k, v] of Object.entries(stats || {})) {
    const label = STAT_LABELS[k];
    if (!label) continue;
    if (typeof v === 'boolean') parts.push(`${label} : ${v ? 'oui' : 'non'}`);
    else if (typeof v === 'number') parts.push(`${label} : ${formatNumber(v)}`);
    else if (typeof v === 'string') parts.push(`${label} : ${v}`);
  }
  const byType = stats && stats.by_type && typeof stats.by_type === 'object'
    ? Object.entries(stats.by_type).map(([t, n]) => `${TYPE_LABELS[t] || t} ${formatNumber(n)}`).join(', ') : '';
  if (byType) parts.push(`par type : ${byType}`);
  return parts.join(' · ');
}

/** Route de navigation pour un élément affecté. */
function itemHref(item) {
  const type = item.type || '';
  if (type === 'document' && (item.document_id || item.id)) return `#/documents/${encodeURIComponent(item.document_id || item.id)}`;
  if (type === 'note' && (item.note_id || item.id)) return `#/notes/${encodeURIComponent(item.note_id || item.id)}`;
  if (type === 'model' && (item.model_id || item.id)) return `#/explorer/models/${encodeURIComponent(item.model_id || item.id)}`;
  if (type === 'symbol' && (item.symbol_id || item.id)) return `#/explorer/symbols/${encodeURIComponent(item.symbol_id || item.id)}`;
  if (item.file_id) return fileHref(item.file_id, item.line);
  if (item.node && /^file:/.test(item.node)) return fileHref(item.node.slice(5), item.line);
  if (type === 'endpoint' && item.path) return `#/explorer/endpoints?q=${encodeURIComponent(item.name || item.path)}`;
  if (item.path) return `#/explorer/files`;
  return null;
}

function affectedRow(item) {
  const href = itemHref(item);
  return el('div', { class: 'impact-row', onClick: () => { if (href) navigate(href); } },
    el('div',
      el('div', { class: 'row' }, badge(TYPE_LABELS[item.type] || item.type || '?'), el('strong', item.name || item.path || item.node || '?'), item.path && item.name !== item.path ? el('span', { class: 'mono small muted' }, `${item.path}${item.line ? `:${item.line}` : ''}`) : null),
      item.reason ? el('div', { class: 'muted small' }, item.reason) : null),
    confidenceBar(item.confidence),
    el('span', { class: 'muted small nowrap' }, item.hops !== undefined ? `${item.hops} saut${item.hops > 1 ? 's' : ''}` : ''));
}

/** Regroupe les éléments affectés : groupes dédiés du rapport, puis fichiers/autres depuis `affected`. */
function groupedItems(report) {
  const out = new Map(GROUPS.map(([k]) => [k, []]));
  for (const [key] of GROUPS) {
    if (key === 'files') continue;
    for (const item of report[key] || []) out.get(key).push(item);
  }
  const known = new Set();
  for (const [key] of GROUPS) for (const item of out.get(key)) known.add(`${item.type}:${item.node || item.name}:${item.path}`);
  for (const item of report.affected || []) {
    const sig = `${item.type}:${item.node || item.name}:${item.path}`;
    if (known.has(sig)) continue;
    const bucket = item.type === 'endpoint' ? 'endpoints' : item.type === 'model' || item.type === 'table' ? 'models' : item.type === 'document' ? 'documents' : item.type === 'note' ? 'notes' : item.type === 'frontend_call' ? 'frontend_calls' : 'files';
    out.get(bucket).push(item);
    known.add(sig);
  }
  return out;
}

/** Texte résumant le rapport, envoyé au chat par « Expliquer avec l'IA ». */
function summarize(report, request, symbolLabel = '') {
  const lines = ['Explique-moi l\'impact de ces changements et les risques de régression à vérifier.', ''];
  const origin = request.source === 'git'
    ? (request.rev_range ? `commits ${request.rev_range}` : 'modifications Git en cours')
    : request.source === 'file' ? `fichier ${request.path}` : `symbole ${symbolLabel || request.symbol_id}`;
  lines.push(`Source : ${origin}${request.repo_id ? ` (dépôt ${request.repo_id})` : ''}.`);
  const changed = report.changed || [];
  if (changed.length) {
    lines.push('', 'Fichiers concernés :');
    changed.slice(0, 20).forEach((c) => lines.push(`- ${c.path} (${statusLabel(c.status)})${(c.symbols_changed || []).length ? ` : ${c.symbols_changed.map((s) => (typeof s === 'string' ? s : s.name)).join(', ')}` : ''}`));
    if (changed.length > 20) lines.push(`- … et ${changed.length - 20} autres`);
  }
  const groups = groupedItems(report);
  for (const [key, label] of GROUPS) {
    const items = groups.get(key);
    if (!items.length) continue;
    lines.push('', `${label} affectés :`);
    items.slice(0, 12).forEach((i) => lines.push(`- ${i.name || i.path || i.node}${i.path && i.name !== i.path ? ` (${i.path})` : ''} — ${i.reason || ''} [confiance ${Math.round((i.confidence || 0) * 100)} %]`));
    if (items.length > 12) lines.push(`- … et ${items.length - 12} autres`);
  }
  return lines.join('\n');
}

function renderReport(report, request, symbolLabel = '') {
  const changed = report.changed || [];
  const groups = groupedItems(report);
  const total = [...groups.values()].reduce((n, arr) => n + arr.length, 0);
  const statsText = statsLine(report.stats || {});
  return el('div',
    (report.warnings || []).map((w) => infoBox(w, 'warn')),
    el('div', { class: 'row-between', style: { marginBottom: '10px' } },
      el('div', { class: 'row' }, el('h2', { style: { margin: 0 } }, 'Rapport d\'impact'), badge(`${changed.length} changement(s)`), badge(`${total} élément(s) affecté(s)`, total ? 'warn' : 'ok')),
      el('button', { class: 'btn btn-primary', type: 'button', onClick: () => { state.chatPrefill = summarize(report, request, symbolLabel); navigate('#/chat'); } }, icon('chat', 16), 'Expliquer avec l\'IA')),
    statsText ? el('div', { class: 'muted small', style: { marginBottom: '10px' } }, statsText) : null,
    el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Changements'),
      changed.length ? el('div', { class: 'list' }, changed.map((c) => el('div', { class: 'list-item' },
        statusBadge(c.status),
        el('div', { class: 'list-item-main' },
          c.file_id ? el('a', { href: fileHref(c.file_id), class: 'mono' }, c.path) : el('span', { class: 'mono' }, c.path),
          (c.symbols_changed || []).length ? el('div', { class: 'list-item-meta' }, 'Symboles : ', c.symbols_changed.map((s) => badge(typeof s === 'string' ? s : s.name))) : null)))) : el('div', { class: 'muted small' }, 'Aucun changement détecté.')),
    total === 0 ? emptyState('Aucun élément affecté', 'Rien ne dépend des changements analysés, d\'après l\'index actuel.') : null,
    GROUPS.map(([key, label]) => {
      const items = groups.get(key);
      if (!items.length) return null;
      return el('div', { class: 'card impact-group' }, el('div', { class: 'card-title' }, label, badge(String(items.length))), el('div', items.map(affectedRow)));
    }));
}

export async function mount(root) {
  const pick = state.impactPick;
  state.impactPick = null;

  const source = el('select', { class: 'select' },
    el('option', { value: 'git' }, 'Modifications Git en cours'),
    el('option', { value: 'range' }, 'Plage de commits'),
    el('option', { value: 'file' }, 'Un fichier'),
    el('option', { value: 'symbol' }, 'Un symbole'));
  const repo = el('select', { class: 'select' }, el('option', { value: '' }, 'Tous les dépôts'));
  const range = el('input', { class: 'input mono', type: 'text', placeholder: 'ex. HEAD~3..HEAD ou main..feature' });
  const path = el('input', { class: 'input mono', type: 'text', placeholder: 'chemin/relatif/au/depot.php' });
  const hops = el('input', { class: 'input', type: 'number', min: 1, max: 3, value: 3 });
  const rangeField = field('Plage de commits', range, 'Syntaxe Git : « A..B ».');
  const pathField = field('Fichier', path, el('span', 'Sélectionnez-le depuis l\'explorateur (bouton « Analyser l\'impact ») ou saisissez son chemin. ', el('a', { href: '#/explorer' }, 'Ouvrir l\'explorateur')));
  const result = el('div');

  // ---------------------------------------------------------------- sélecteur de symbole
  // Recherche par nom (GET /api/symbols) ; le choix mémorise l'identifiant, jamais saisi à la main.
  let symbolId = '';
  let symbolLabel = '';
  const symbolSearch = el('input', { class: 'input', type: 'search', placeholder: 'Nom de fonction, classe, méthode…', autocomplete: 'off' });
  const symbolResults = el('div', { class: 'list hidden', style: { marginTop: '6px', maxHeight: '240px', overflow: 'auto' } });
  const symbolChosen = el('div', { class: 'small', style: { marginTop: '6px' } });
  const clearSymbol = () => { symbolId = ''; symbolLabel = ''; symbolChosen.replaceChildren(); symbolSearch.focus(); };
  const chooseSymbol = (s) => {
    symbolId = String(s.id);
    symbolLabel = s.parent ? `${s.parent}.${s.name}` : (s.name || String(s.id));
    const filePath = s.file && typeof s.file === 'object' ? s.file.path : (s.path || '');
    const where = filePath ? `${filePath}${s.line_start ? `:${s.line_start}` : ''}` : '';
    replace(symbolChosen, 'Symbole choisi : ', el('strong', { class: 'mono' }, symbolLabel),
      where ? el('span', { class: 'mono muted' }, ` — ${where}`) : null, ' ',
      el('button', { class: 'btn-link small', type: 'button', onClick: clearSymbol }, 'changer'));
    symbolResults.classList.add('hidden');
    symbolSearch.value = '';
  };
  let searchSeq = 0;
  const searchSymbols = debounce(async () => {
    const term = symbolSearch.value.trim();
    const mine = ++searchSeq;
    if (!term) { symbolResults.classList.add('hidden'); return; }
    replace(symbolResults, spinner('Recherche…'));
    symbolResults.classList.remove('hidden');
    try {
      const items = await api.get('/api/symbols', { q: term, limit: 15 });
      if (mine !== searchSeq) return;
      replace(symbolResults, items.length ? items.map((s) => el('div', { class: 'list-item clickable', onClick: () => chooseSymbol(s) },
        el('div', { class: 'list-item-main' },
          el('div', { class: 'list-item-title row' }, el('span', { class: 'mono' }, s.parent ? `${s.parent}.${s.name}` : s.name), badge(s.kind || '?')),
          s.file && s.file.path ? el('div', { class: 'list-item-meta' }, el('span', { class: 'mono truncate' }, `${s.file.path}:${s.line_start || ''}`)) : null)))
        : el('div', { class: 'muted small', style: { padding: '8px 12px' } }, 'Aucun symbole ne correspond.'));
    } catch (e) {
      if (mine === searchSeq) replace(symbolResults, errorBox(e));
    }
  }, 300);
  symbolSearch.addEventListener('input', searchSymbols);
  const symbolField = field('Symbole', el('div', symbolSearch, symbolResults, symbolChosen),
    el('span', 'Tapez un nom pour le retrouver, ou ouvrez-le dans l\'explorateur (bouton « Analyser l\'impact »). ', el('a', { href: '#/explorer/symbols' }, 'Rechercher dans l\'explorateur')));

  const syncFields = () => {
    rangeField.classList.toggle('hidden', source.value !== 'range');
    pathField.classList.toggle('hidden', source.value !== 'file');
    symbolField.classList.toggle('hidden', source.value !== 'symbol');
  };
  source.addEventListener('change', syncFields);

  if (pick && pick.symbol_id) {
    source.value = 'symbol';
    chooseSymbol({ id: pick.symbol_id, name: pick.name, path: pick.path, line_start: pick.line_start });
  } else if (pick) {
    source.value = 'file';
    path.value = pick.path || '';
  }
  syncFields();

  api.get('/api/repos').then((repos) => {
    for (const r of repos) repo.appendChild(el('option', { value: r.id, selected: pick && pick.repo_id === r.id }, `${r.name}${r.git && r.git.branch ? ` (${r.git.branch}${r.git.dirty ? ', modifié' : ''})` : ''}`));
  }).catch((e) => toastError(e));

  const analyzeBtn = el('button', { class: 'btn btn-primary', type: 'button', onClick: async () => {
    const request = { source: source.value === 'range' ? 'git' : source.value, max_hops: Number(hops.value) || 3 };
    if (repo.value) request.repo_id = repo.value;
    if (source.value === 'range') request.rev_range = range.value.trim();
    if (source.value === 'file') request.path = path.value.trim();
    if (source.value === 'symbol') request.symbol_id = /^\d+$/.test(symbolId) ? Number(symbolId) : symbolId;
    if (source.value === 'range' && !request.rev_range) { replace(result, errorBox({ message: 'Indiquez une plage de commits.' })); return; }
    if (source.value === 'file' && !request.path) { replace(result, errorBox({ message: 'Indiquez un fichier.' })); return; }
    if (source.value === 'symbol' && !symbolId) { replace(result, errorBox({ message: 'Choisissez un symbole dans la liste (tapez son nom).' })); return; }
    analyzeBtn.disabled = true;
    replace(result, spinner('Analyse en cours…'));
    try {
      const report = await api.post('/api/impact', request);
      replace(result, renderReport(report, request, symbolLabel));
    } catch (e) {
      replace(result, errorBox(e));
      toastError(e);
    } finally {
      analyzeBtn.disabled = false;
    }
  } }, icon('impact', 16), 'Analyser');

  replace(root, el('div', { class: 'page' },
    el('div', { class: 'page-head' }, el('div', el('h1', 'Analyse d\'impact'), el('div', { class: 'page-sub' }, 'Quels endpoints, modèles, appels frontend, documents et notes dépendent de ce qui change ?'))),
    el('div', { class: 'card' },
      el('div', { class: 'form-grid' }, field('Source', source), field('Dépôt', repo)),
      rangeField, pathField, symbolField,
      el('div', { class: 'form-grid' }, field('Profondeur (sauts dans le graphe)', hops, '3 par défaut : service → contrôleur → route → appel frontend.')),
      el('div', { class: 'row' }, analyzeBtn)),
    result));
  return () => {};
}
