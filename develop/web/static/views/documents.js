// views/documents.js — dépôt de documents (glisser-déposer, fichier, chemin local),
// liste avec badge « peut-être obsolète », détail (contenu, mentions) et suppression.

import { api } from '/static/api.js';
import {
  el, replace, icon, toast, toastError, errorBox, emptyState, spinner, badge, confirmDialog,
  formatSize, formatDate, formatNumber, dl,
} from '/static/ui.js';
import { navigate } from '/static/app.js';

// Libellés de la colonne « Type » : un par type de mention que produit develop.knowledge_graph.mentions
// (MENTION_KINDS) ; « inconnu » = identifiant qui ressemble à du code mais absent de l'index (dérive doc / code).
const MENTION_LABELS = { symbol: 'symbole', model: 'modèle', table: 'table', endpoint: 'endpoint', field: 'champ', file: 'fichier', unknown: 'inconnu' };

/** Zone de dépôt + bouton fichier + import par chemin. `onAdded()` recharge la liste. */
function uploadPanel(onAdded) {
  const fileInput = el('input', { type: 'file', multiple: true, class: 'hidden', accept: '.pdf,.md,.markdown,.txt,.sql,.csv,.json,.yaml,.yml,.html,.docx' });
  const status = el('div');
  const drop = el('div', { class: 'dropzone' },
    icon('upload', 22),
    el('div', 'Glissez-déposez vos documents ici (PDF, Markdown, SQL, TXT, CSV, JSON…)'),
    el('div', { class: 'row', style: { justifyContent: 'center', marginTop: '8px' } },
      el('button', { class: 'btn', type: 'button', onClick: () => fileInput.click() }, 'Choisir des fichiers')),
    fileInput);

  const uploadFiles = async (files) => {
    const list = [...files];
    if (!list.length) return;
    let ok = 0;
    for (const file of list) {
      replace(status, spinner(`Envoi de ${file.name} (${ok + 1}/${list.length})…`));
      const form = new FormData();
      form.append('file', file, file.name);
      try {
        await api.upload('/api/documents', form);
        ok += 1;
      } catch (e) {
        toastError(e);
      }
    }
    status.replaceChildren();
    if (ok) toast(`${ok} document(s) ajouté(s) et indexé(s).`, { type: 'success' });
    onAdded();
  };

  fileInput.addEventListener('change', () => { uploadFiles(fileInput.files); fileInput.value = ''; });
  ['dragenter', 'dragover'].forEach((evt) => drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((evt) => drop.addEventListener(evt, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { if (e.dataTransfer && e.dataTransfer.files) uploadFiles(e.dataTransfer.files); });

  const pathInput = el('input', { class: 'input mono', type: 'text', placeholder: '/chemin/vers/dictionnaire-donnees.pdf' });
  const importBtn = el('button', { class: 'btn', type: 'button', onClick: async () => {
    const path = pathInput.value.trim();
    if (!path) return;
    importBtn.disabled = true;
    replace(status, spinner('Import du fichier…'));
    try {
      await api.post('/api/documents', { path });
      toast('Document importé (copié dans l\'espace de travail).', { type: 'success' });
      pathInput.value = '';
      onAdded();
    } catch (e) {
      toastError(e);
    } finally {
      importBtn.disabled = false;
      status.replaceChildren();
    }
  } }, 'Importer par chemin');
  pathInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') importBtn.click(); });

  return el('div', drop, el('div', { class: 'row', style: { marginBottom: '10px' } }, pathInput, importBtn), status);
}

/** Détail d'un document : métadonnées, mentions (dérive), contenu extrait. */
function documentDetail(id, onDeleted) {
  const root = el('div', spinner('Document…'));
  Promise.all([
    api.get(`/api/documents/${encodeURIComponent(id)}`),
    api.get(`/api/documents/${encodeURIComponent(id)}/content`).catch((e) => ({ error: e })),
  ]).then(([doc, content]) => {
    const mentions = doc.mentions || [];
    const missing = mentions.filter((m) => !m.exists);
    const text = content && content.error ? null : (typeof content === 'string' ? content : (content.text || content.content || JSON.stringify(content, null, 2)));
    replace(root,
      el('div', { class: 'row-between', style: { marginBottom: '10px' } },
        el('div', el('h2', { style: { margin: 0 } }, doc.title || 'Sans titre'), el('div', { class: 'muted small' }, [doc.format, doc.kind, formatSize(doc.size), `ajouté le ${formatDate(doc.added_at)}`].filter(Boolean).join(' · '))),
        el('div', { class: 'row' },
          doc.stale ? badge('peut-être obsolète', 'warn') : badge('à jour', 'ok'),
          el('button', { class: 'btn btn-danger btn-sm', type: 'button', onClick: async () => {
            if (!(await confirmDialog(`Supprimer le document « ${doc.title || id} » ? Cette action retire sa copie et son index.`, { danger: true, okLabel: 'Supprimer' }))) return;
            try { await api.del(`/api/documents/${encodeURIComponent(id)}`); toast('Document supprimé.'); onDeleted(); } catch (e) { toastError(e); }
          } }, icon('trash', 14), 'Supprimer'))),
      dl([['Morceaux indexés', formatNumber(doc.chunk_count)], ['Mis à jour', formatDate(doc.updated_at)]]),
      el('div', { class: 'card', style: { marginTop: '12px' } },
        el('div', { class: 'card-title' }, 'Mentions du code', badge(String(mentions.length)), missing.length ? badge(`${missing.length} introuvable(s)`, 'warn') : null),
        mentions.length
          ? el('div', { class: 'table-wrap' }, el('table', el('thead', el('tr', el('th', 'Type'), el('th', 'Nom'), el('th', 'État'))),
            el('tbody', mentions.map((m) => el('tr', el('td', MENTION_LABELS[m.kind] || m.kind || '?'), el('td', el('span', { class: 'mono' }, m.name)), el('td', m.exists ? badge('existe', 'ok') : badge('n\'existe plus', 'error')))))))
          : el('div', { class: 'muted small' }, 'Aucun symbole, modèle, table, endpoint, champ ou fichier reconnu dans ce document.'),
        missing.length ? el('div', { class: 'hint', style: { marginTop: '6px' } }, 'Ce document cite des éléments absents de l\'index : il ne correspond peut-être plus au code. Develop ne modifie jamais vos documents.') : null),
      el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Contenu extrait'),
        text !== null ? el('div', { class: 'doc-content' }, text) : errorBox(content.error)));
  }).catch((e) => replace(root, errorBox(e)));
  return root;
}

export async function mount(root, route) {
  const selectedId = route.parts[0] || null;
  const listHost = el('div');
  const detailHost = el('div');

  const loadList = async () => {
    replace(listHost, spinner('Documents…'));
    try {
      const docs = await api.get('/api/documents');
      if (!docs.length) { replace(listHost, emptyState('Aucun document', 'Déposez un dictionnaire de données, une spécification ou un schéma SQL.')); return; }
      replace(listHost, el('div', { class: 'list' }, docs.map((d) => el('div', { class: `list-item clickable${String(d.id) === String(selectedId) ? ' active' : ''}`, onClick: () => navigate(`#/documents/${encodeURIComponent(d.id)}`) },
        icon('documents', 16),
        el('div', { class: 'list-item-main' },
          el('div', { class: 'list-item-title row' }, d.title || 'Sans titre', d.stale ? badge('peut-être obsolète', 'warn') : null),
          el('div', { class: 'list-item-meta' }, badge(d.format || d.kind || '?'), el('span', formatSize(d.size)), el('span', formatDate(d.added_at, { time: false })), el('span', `${formatNumber(d.chunk_count)} morceau(x)`)))))));
    } catch (e) {
      replace(listHost, errorBox(e));
    }
  };

  replace(detailHost, selectedId ? documentDetail(selectedId, () => navigate('#/documents')) : emptyState('Sélectionnez un document', 'Le contenu extrait et les mentions du code apparaîtront ici.'));

  replace(root, el('div', { class: 'page' },
    el('div', { class: 'page-head' }, el('div', el('h1', 'Documents'), el('div', { class: 'page-sub' }, 'Dictionnaire de données, spécifications, schémas : copiés dans l\'espace de travail et indexés pour la recherche et le chat.'))),
    uploadPanel(loadList),
    el('div', { class: 'split' }, listHost, detailHost)));
  await loadList();
  return () => {};
}
