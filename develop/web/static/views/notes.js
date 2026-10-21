// views/notes.js — journal de notes internes datées : filtres, liste par date décroissante,
// éditeur (titre, date, tags, statut, note remplacée, corps) avec aperçu Markdown.

import { api } from '/static/api.js';
import {
  el, replace, icon, toast, toastError, errorBox, emptyState, spinner, badge, confirmDialog,
  renderMarkdown, formatDate, todayIso, field, debounce,
} from '/static/ui.js';
import { navigate } from '/static/app.js';

const STATUS_LABELS = { valid: 'Valide', superseded: 'Remplacée', historical: 'Historique' };
const STATUS_KIND = { valid: 'ok', superseded: 'warn', historical: 'neutral' };

function statusBadge(status) {
  return badge(STATUS_LABELS[status] || status || '?', STATUS_KIND[status] || 'neutral');
}

/** Libellé d'une note dans le sélecteur « Remplace la note » : titre — date (statut). */
function noteOptionLabel(n) {
  const status = n.status && n.status !== 'valid' ? ` (${(STATUS_LABELS[n.status] || n.status).toLowerCase()})` : '';
  return `${n.title || 'Sans titre'} — ${formatDate(n.note_date, { time: false })}${status}`;
}

/** Identifiant tel qu'attendu par l'API (numérique si possible), ou null. */
function idValue(raw) {
  const v = String(raw === null || raw === undefined ? '' : raw).trim();
  if (!v) return null;
  return /^\d+$/.test(v) ? Number(v) : v;
}

/** Éditeur d'une note (nouvelle si `note` est null). */
function noteEditor(note, { onSaved, onDeleted }) {
  const isNew = !note;
  const data = note ? { ...note } : { title: '', body: '', tags: [], note_date: todayIso(), status: 'valid' };
  const title = el('input', { class: 'input', type: 'text', value: data.title || '', placeholder: 'Titre de la note' });
  const date = el('input', { class: 'input', type: 'date', value: data.note_date || todayIso() });
  const tags = el('input', { class: 'input', type: 'text', value: (data.tags || []).join(', '), placeholder: 'facturation, api, réunion' });
  const status = el('select', { class: 'select' }, Object.entries(STATUS_LABELS).map(([v, label]) => el('option', { value: v, selected: data.status === v }, label)));
  const body = el('textarea', { class: 'textarea mono', value: data.body || '', placeholder: 'Contenu en Markdown : décisions, contexte, liens vers des fichiers…', style: { minHeight: '280px' } });
  const preview = el('div', { class: 'note-preview md' });
  const error = el('div');

  // « Remplace la note » : sélecteur (titre + date) alimenté par GET /api/notes, toujours visible
  // (le champ `supersedes` du contrat est facultatif et concerne la note qui prend la place d'une autre).
  const currentSup = idValue(data.supersedes);
  const supersedes = el('select', { class: 'select' }, el('option', { value: '', selected: currentSup === null }, 'Aucune'));
  if (currentSup !== null) supersedes.appendChild(el('option', { value: String(currentSup), selected: true }, `Note ${currentSup}`));
  api.get('/api/notes', { limit: 200 }).then((notes) => {
    for (const n of notes || []) {
      if (note && String(n.id) === String(note.id)) continue;
      const existing = [...supersedes.options].find((o) => o.value === String(n.id));
      if (existing) { existing.textContent = noteOptionLabel(n); continue; }
      supersedes.appendChild(el('option', { value: String(n.id) }, noteOptionLabel(n)));
    }
  }).catch(() => { /* la liste est un confort : la valeur courante reste sélectionnable */ });

  const refreshPreview = debounce(() => replace(preview, body.value.trim() ? renderMarkdown(body.value) : el('span', { class: 'muted' }, 'Aperçu Markdown')), 200);
  body.addEventListener('input', refreshPreview);
  refreshPreview();

  const payload = () => {
    const p = {
      title: title.value.trim(),
      body: body.value,
      tags: tags.value.split(',').map((t) => t.trim()).filter(Boolean),
      note_date: date.value || todayIso(),
      status: status.value,
    };
    const sup = idValue(supersedes.value);
    if (sup !== null) p.supersedes = sup;
    else if (!isNew && currentSup !== null) p.supersedes = null; // effacement explicite à la mise à jour
    return p;
  };

  const saveBtn = el('button', { class: 'btn btn-primary', type: 'button', onClick: async () => {
    const p = payload();
    if (!p.title) { replace(error, errorBox({ message: 'Le titre est obligatoire.' })); return; }
    saveBtn.disabled = true;
    error.replaceChildren();
    try {
      const saved = isNew ? await api.post('/api/notes', p) : await api.put(`/api/notes/${encodeURIComponent(note.id)}`, p);
      toast(isNew ? 'Note créée et indexée.' : 'Note mise à jour.', { type: 'success' });
      onSaved(saved);
    } catch (e) {
      replace(error, errorBox(e));
      toastError(e);
    } finally {
      saveBtn.disabled = false;
    }
  } }, isNew ? 'Créer la note' : 'Enregistrer');

  const deleteBtn = isNew ? null : el('button', { class: 'btn btn-danger', type: 'button', onClick: async () => {
    if (!(await confirmDialog(`Supprimer la note « ${note.title} » ?`, { danger: true, okLabel: 'Supprimer' }))) return;
    try { await api.del(`/api/notes/${encodeURIComponent(note.id)}`); toast('Note supprimée.'); onDeleted(); } catch (e) { toastError(e); }
  } }, icon('trash', 14), 'Supprimer');

  return el('div',
    el('h2', isNew ? 'Nouvelle note' : 'Modifier la note'),
    el('div', { class: 'form-grid' }, field('Titre', title), field('Date', date)),
    el('div', { class: 'form-grid' },
      field('Tags (séparés par des virgules)', tags),
      field('Statut', status, 'Une note « remplacée » ou « historique » reste consultable avec sa date.'),
      field('Remplace la note', supersedes, 'Facultatif : la note dont celle-ci prend la place (décision révisée, règle mise à jour).')),
    el('div', { class: 'editor-split' }, field('Contenu', body), field('Aperçu', preview)),
    error,
    el('div', { class: 'row' }, saveBtn, deleteBtn, note ? el('span', { class: 'muted small' }, `Créée le ${formatDate(note.created_at)}`) : null));
}

export async function mount(root, route) {
  const selectedId = route.parts[0] || null; // 'new' ou identifiant
  const from = el('input', { class: 'input', type: 'date' });
  const to = el('input', { class: 'input', type: 'date' });
  const tag = el('input', { class: 'input', type: 'text', placeholder: 'Tag' });
  const q = el('input', { class: 'input', type: 'search', placeholder: 'Texte…' });
  const listHost = el('div');
  const editorHost = el('div');

  const loadList = async () => {
    replace(listHost, spinner('Notes…'));
    try {
      const notes = await api.get('/api/notes', { from: from.value, to: to.value, tag: tag.value.trim(), q: q.value.trim(), limit: 200 });
      if (!notes.length) { replace(listHost, emptyState('Aucune note', 'Consignez vos décisions et comptes rendus : ils serviront au chat et à l\'analyse d\'impact.')); return; }
      replace(listHost, el('div', { class: 'list' }, notes.map((n) => el('div', { class: `list-item clickable${String(n.id) === String(selectedId) ? ' active' : ''}`, onClick: () => navigate(`#/notes/${encodeURIComponent(n.id)}`) },
        el('div', { class: 'list-item-main' },
          el('div', { class: 'list-item-title row' }, n.title || 'Sans titre', statusBadge(n.status)),
          el('div', { class: 'list-item-meta' }, el('span', formatDate(n.note_date, { time: false })), (n.tags || []).length ? el('span', { class: 'note-tags' }, n.tags.map((t) => badge(t))) : null),
          n.excerpt ? el('div', { class: 'muted small' }, n.excerpt) : null)))));
    } catch (e) {
      replace(listHost, errorBox(e));
    }
  };
  const reload = debounce(loadList, 300);
  [from, to, tag, q].forEach((n) => { n.addEventListener('input', reload); n.addEventListener('change', reload); });

  const openEditor = async () => {
    if (!selectedId) { replace(editorHost, emptyState('Sélectionnez une note', 'Ou créez-en une nouvelle.', [el('button', { class: 'btn btn-primary', type: 'button', onClick: () => navigate('#/notes/new') }, icon('plus', 14), 'Nouvelle note')])); return; }
    if (selectedId === 'new') {
      replace(editorHost, noteEditor(null, { onSaved: (saved) => navigate(`#/notes/${encodeURIComponent(saved.id)}`), onDeleted: () => navigate('#/notes') }));
      return;
    }
    replace(editorHost, spinner('Note…'));
    try {
      const note = await api.get(`/api/notes/${encodeURIComponent(selectedId)}`);
      replace(editorHost, noteEditor(note, { onSaved: () => loadList(), onDeleted: () => navigate('#/notes') }));
    } catch (e) {
      replace(editorHost, errorBox(e));
    }
  };

  replace(root, el('div', { class: 'page' },
    el('div', { class: 'page-head' },
      el('div', el('h1', 'Journal'), el('div', { class: 'page-sub' }, 'Notes internes datées : décisions, réunions, règles métier. Les plus récentes passent en premier.')),
      el('button', { class: 'btn btn-primary', type: 'button', onClick: () => navigate('#/notes/new') }, icon('plus', 16), 'Nouvelle note')),
    el('div', { class: 'notes-layout' },
      el('div',
        el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Filtres'),
          el('div', { class: 'form-grid' }, field('Du', from), field('Au', to)),
          el('div', { class: 'form-grid' }, field('Tag', tag), field('Texte', q))),
        listHost),
      editorHost)));
  await Promise.all([loadList(), openEditor()]);
  return () => {};
}
