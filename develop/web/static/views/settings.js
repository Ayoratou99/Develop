// views/settings.js — réglages : espaces de travail, dépôts, moteur d'IA, mode réseau,
// réindexation complète, journal des requêtes bloquées, infos machine, suppression.

import { api } from '/static/api.js';
import {
  el, replace, icon, toast, toastError, errorBox, emptyState, spinner, badge, confirmDialog, promptDialog,
  formatDate, dl, field,
} from '/static/ui.js';
import { state, navigate, refreshStatus } from '/static/app.js';
import { repoEditor, engineForm, modeForm, indexPanel, KIND_LABELS, OS_LABELS } from '/static/views/setup.js';

/** Bouton « Enregistrer » qui envoie un corps partiel à PUT /api/workspaces/{id}. */
function saveButton(wsId, buildPayload, label = 'Enregistrer') {
  const button = el('button', { class: 'btn btn-primary', type: 'button' }, label);
  button.addEventListener('click', async () => {
    button.disabled = true;
    try {
      await api.put(`/api/workspaces/${encodeURIComponent(wsId)}`, buildPayload());
      toast('Réglages enregistrés.', { type: 'success' });
      await refreshStatus();
    } catch (e) {
      toastError(e);
    } finally {
      button.disabled = false;
    }
  });
  return button;
}

/** Panneau des espaces de travail : liste, activation, création, suppression. */
function workspacesPanel(activeId) {
  const host = el('div', spinner('Espaces…'));
  const load = async () => {
    try {
      const list = await api.get('/api/workspaces');
      replace(host, list.length ? el('div', { class: 'list' }, list.map((ws) => el('div', { class: 'list-item' },
        el('div', { class: 'list-item-main' },
          el('div', { class: 'list-item-title row' }, ws.name, ws.id === activeId ? badge('actif', 'ok') : null),
          el('div', { class: 'list-item-meta' }, el('span', { class: 'mono' }, ws.id), el('span', `${(ws.repos || []).length} dépôt(s)`), el('span', `modifié ${formatDate(ws.updated_at)}`))),
        ws.id !== activeId ? el('button', { class: 'btn btn-sm', type: 'button', onClick: async () => {
          try { await api.post(`/api/workspaces/${encodeURIComponent(ws.id)}/activate`, {}); toast(`Espace « ${ws.name} » activé.`, { type: 'success' }); await refreshStatus(); navigate('#/settings'); } catch (e) { toastError(e); }
        } }, 'Activer') : null,
        el('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', title: 'Supprimer la mémoire Develop de cet espace', onClick: async () => {
          if (!(await confirmDialog(`Supprimer l'espace « ${ws.name} » ? Son index, ses notes, documents et conversations seront effacés. Vos dépôts ne sont jamais touchés.`, { danger: true, okLabel: 'Supprimer' }))) return;
          try { await api.del(`/api/workspaces/${encodeURIComponent(ws.id)}`); toast('Espace supprimé.'); await refreshStatus(); navigate(state.status && state.status.needs_setup ? '#/setup' : '#/settings'); } catch (e) { toastError(e); }
        } }, icon('trash', 15))))) : emptyState('Aucun espace de travail'));
    } catch (e) {
      replace(host, errorBox(e));
    }
  };
  load();
  return el('div', { class: 'card' },
    el('div', { class: 'card-title' }, 'Espaces de travail',
      el('button', { class: 'btn btn-sm', type: 'button', onClick: async () => {
        const name = await promptDialog('Nouvel espace de travail', { label: 'Nom', placeholder: 'ex. Boutique en ligne', okLabel: 'Créer' });
        if (!name) return;
        try { await api.post('/api/workspaces', { name }); toast('Espace créé et activé.', { type: 'success' }); await refreshStatus(); navigate('#/setup'); } catch (e) { toastError(e); }
      } }, icon('plus', 14), 'Nouvel espace')),
    host);
}

/** Journal des requêtes réseau bloquées par la garde hors ligne. */
function blockedPanel() {
  const host = el('div', spinner('Journal…'));
  const load = async () => {
    try {
      const items = await api.get('/api/offline/blocked');
      replace(host, items.length ? el('div', { class: 'table-wrap' }, el('table',
        el('thead', el('tr', el('th', 'Date'), el('th', 'Hôte'), el('th', 'URL'), el('th', 'Mode'))),
        el('tbody', items.map((b) => el('tr', el('td', { class: 'nowrap' }, formatDate(b.timestamp)), el('td', { class: 'mono' }, b.host), el('td', { class: 'mono small' }, b.url), el('td', b.mode)))))) : el('div', { class: 'muted small' }, 'Aucune requête bloquée : rien n\'a tenté de sortir de la machine.'));
    } catch (e) {
      replace(host, errorBox(e));
    }
  };
  load();
  return el('div', { class: 'card' },
    el('div', { class: 'card-title' }, 'Requêtes réseau bloquées',
      el('div', { class: 'row' },
        el('button', { class: 'btn btn-sm', type: 'button', onClick: load }, icon('refresh', 14), 'Actualiser'),
        el('button', { class: 'btn btn-sm', type: 'button', onClick: async () => {
          try { await api.del('/api/offline/blocked'); toast('Journal vidé.'); load(); refreshStatus(); } catch (e) { toastError(e); }
        } }, icon('trash', 14), 'Vider'))),
    el('p', { class: 'muted small' }, 'Toute tentative de joindre un service distant est bloquée et consignée ici.'),
    host);
}

function machinePanel() {
  const host = el('div', spinner('Machine…'));
  Promise.all([api.get('/api/system/profile'), api.get('/api/system/engines').catch(() => null)]).then(([p, engines]) => {
    replace(host, el('p', p.summary || ''), dl([
      ['Système', el('span', { title: p.os_version || '' }, `${OS_LABELS[p.os] || p.os || '?'} (${p.arch || '?'})`)],
      ['Processeur', `${p.cpu_model || '?'}${p.cpu_cores_logical ? ` — ${p.cpu_cores_logical} cœurs` : ''}`],
      ['Mémoire', p.ram_total_gb ? `${p.ram_total_gb} Go${p.unified_memory ? ' (unifiée)' : ''}` : '—'],
      ['GPU', (p.gpus || []).map((g) => g.name).join(', ') || (p.apple_silicon ? 'Apple Silicon' : 'aucun')],
      ['Disque libre', p.disk_free_gb ? `${p.disk_free_gb} Go` : '—'],
      ['Python', p.python_version || '—'],
      engines ? ['Ollama', engines.ollama && engines.ollama.reachable ? `joignable (${engines.ollama.version || '?'})` : 'injoignable'] : null,
      engines ? ['Docker', engines.docker_available ? (engines.docker_running ? 'en marche' : 'installé, arrêté') : 'absent'] : null,
      state.status && state.status.version ? ['Develop', state.status.version] : null,
    ]));
  }).catch((e) => replace(host, errorBox(e)));
  return el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Votre machine'), host);
}

export async function mount(root) {
  const status = state.status || (await refreshStatus());
  const ws = status && status.workspace;
  const page = el('div', { class: 'page' });
  replace(root, page);

  if (!ws) {
    replace(page,
      el('div', { class: 'page-head' }, el('h1', 'Réglages')),
      emptyState('Aucun espace de travail', 'Créez un espace de travail pour configurer vos dépôts et votre moteur d\'IA.', [el('button', { class: 'btn btn-primary', type: 'button', onClick: () => navigate('#/setup') }, 'Lancer l\'assistant')]),
      workspacesPanel(null), blockedPanel(), machinePanel());
    return () => {};
  }

  const draft = { name: ws.name, repos: (ws.repos || []).map((r) => ({ ...r })), llm: { ...(ws.llm || {}) }, mode: ws.mode || 'strict', allowed_hosts: [...(ws.allowed_hosts || [])] };
  const nameInput = el('input', { class: 'input', type: 'text', value: draft.name, onInput: (e) => { draft.name = e.target.value; } });
  const index = indexPanel({ full: true, label: 'Réindexer (complet)' });

  replace(page,
    el('div', { class: 'page-head' }, el('div', el('h1', 'Réglages'), el('div', { class: 'page-sub' }, `Espace actif : ${ws.name}`)), el('a', { href: '#/setup', class: 'btn' }, 'Relancer l\'assistant')),

    el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Espace de travail et dépôts'),
      field('Nom', nameInput),
      el('div', { class: 'label', style: { marginBottom: '6px' } }, 'Dépôts'),
      repoEditor(draft.repos, () => {}),
      el('div', { class: 'hint', style: { margin: '8px 0' } }, `Types : ${Object.values(KIND_LABELS).join(', ')}. Après un changement de dépôts, relancez une indexation.`),
      el('div', { class: 'row' }, saveButton(ws.id, () => ({ name: draft.name.trim() || ws.name, repos: draft.repos })))),

    el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Moteur d\'IA'),
      engineForm(draft.llm),
      el('div', { class: 'row', style: { marginTop: '8px' } }, saveButton(ws.id, () => ({ llm: draft.llm })))),

    el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Mode d\'exécution'),
      modeForm(draft),
      el('div', { class: 'row', style: { marginTop: '10px' } }, saveButton(ws.id, () => ({ mode: draft.mode, allowed_hosts: draft.allowed_hosts })))),

    el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Index'),
      el('p', { class: 'muted small' }, 'La réindexation complète reconstruit l\'index du code et des documents ; les notes, documents déposés et conversations sont conservés.'),
      index.root),

    blockedPanel(),
    machinePanel(),
    workspacesPanel(ws.id),

    el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Zone sensible'),
      el('p', { class: 'muted small' }, 'Supprime la mémoire Develop de l\'espace actif (index, notes, documents, conversations). Vos dépôts restent intacts.'),
      el('button', { class: 'btn btn-danger', type: 'button', onClick: async () => {
        if (!(await confirmDialog(`Supprimer définitivement l'espace « ${ws.name} » ?`, { danger: true, okLabel: 'Supprimer l\'espace' }))) return;
        try { await api.del(`/api/workspaces/${encodeURIComponent(ws.id)}`); toast('Espace supprimé.'); await refreshStatus(); navigate('#/setup'); } catch (e) { toastError(e); }
      } }, icon('trash', 16), 'Supprimer cet espace de travail')));

  return () => index.destroy();
}
