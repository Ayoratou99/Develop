// views/chat.js — conversations, fil de messages, envoi en flux SSE (status, sources,
// tool_call/tool_result, delta, done, error), citations [S1] cliquables.

import { api, streamPost } from '/static/api.js';
import {
  el, replace, icon, toast, toastError, errorBox, infoBox, emptyState, spinner, renderMarkdown,
  confirmDialog, promptDialog, formatNumber, formatRelative, badge, collapsible,
} from '/static/ui.js';
import { state, navigate, refreshStatus } from '/static/app.js';
import { ENGINE_LABELS } from '/static/views/setup.js';

const MODES = [
  { value: 'auto', label: 'Auto', help: 'Recherche dans le projet, puis le modèle peut demander des outils (lecture de fichier, appelants…) : au plus 4 tours.' },
  { value: 'search_only', label: 'Recherche seule', help: 'Affiche uniquement les sources trouvées, sans appeler le modèle. Fonctionne sans moteur d\'IA.' },
  { value: 'no_tools', label: 'Sans outils', help: 'Une seule passe : contexte récupéré puis réponse du modèle, sans tours d\'outils.' },
];

const TYPE_LABELS = { file: 'fichier', document: 'document', note: 'note', symbol: 'symbole', endpoint: 'endpoint', model: 'modèle' };

/** Lien de navigation vers une source (fichier:ligne, document, note, symbole). */
export function sourceHref(src) {
  if (!src) return null;
  if (src.source_type === 'document' && src.document_id) return `#/documents/${encodeURIComponent(src.document_id)}`;
  if (src.source_type === 'note' && src.note_id) return `#/notes/${encodeURIComponent(src.note_id)}`;
  if (src.source_type === 'symbol' && src.symbol_id && !src.file_id) return `#/explorer/symbols/${encodeURIComponent(src.symbol_id)}`;
  if (src.file_id) return `#/explorer/file/${encodeURIComponent(src.file_id)}${src.line_start ? `?line=${src.line_start}` : ''}`;
  if (src.symbol_id) return `#/explorer/symbols/${encodeURIComponent(src.symbol_id)}`;
  if (src.document_id) return `#/documents/${encodeURIComponent(src.document_id)}`;
  if (src.note_id) return `#/notes/${encodeURIComponent(src.note_id)}`;
  return null;
}

/** Libellé court d'une source : chemin:lignes ou titre (page). */
function sourceLabel(src) {
  const where = src.path || src.title || '';
  if (src.line_start) return `${where}:${src.line_start}${src.line_end && src.line_end !== src.line_start ? `-${src.line_end}` : ''}`;
  if (src.page) return `${where} (p. ${src.page})`;
  return where;
}

/**
 * Pied de message « Contexte : 1 081 / 1 081 (num_ctx 8 192) » : passages retenus par la recherche initiale
 * et leur budget. Les sources ajoutées ensuite par les outils ne s'y cumulent pas (elles sont comptées
 * dans `toolsUsage`), donc le ratio ne dépasse jamais 100 %.
 */
function contextUsage(budget) {
  if (!budget.num_ctx) return null;
  return el('span', { title: 'Passages retenus par la recherche initiale, sur le budget de contexte de la question' },
    `Contexte : ${formatNumber(budget.context_tokens_used || 0)} / ${formatNumber(budget.max_context_tokens || budget.num_ctx)} (num_ctx ${formatNumber(budget.num_ctx)})`);
}

/** Pied de message « Outils : 1 542 / 3 765 tokens (1 tour) » : consommation des tours d'outils, null sans tour. */
function toolsUsage(budget) {
  const rounds = Number(budget.tool_rounds || 0);
  if (!rounds) return null;
  const max = budget.tools_tokens_max ? ` / ${formatNumber(budget.tools_tokens_max)}` : '';
  return el('span', { title: 'Tokens consommés par les tours d\'outils (appels, résultats, sources ajoutées), sur ce que la fenêtre leur laissait' },
    `Outils : ${formatNumber(budget.tools_tokens_used || 0)}${max} tokens (${rounds} tour${rounds > 1 ? 's' : ''})`);
}

/** Panneau « Sources utilisées » (liste S1… cliquable). */
function sourcesPanel(sources, open = false) {
  if (!sources || !sources.length) return null;
  const list = el('div', { class: 'sources-list' }, sources.map((src) => {
    const href = sourceHref(src);
    return el('div', { class: 'source', dataset: { ref: src.ref || '' }, onClick: () => { if (href) navigate(href); } },
      el('span', { class: 'cite' }, src.ref || '?'),
      badge(TYPE_LABELS[src.source_type] || src.source_type || '?'),
      el('span', { class: 'source-path' }, sourceLabel(src)),
      (src.channels || []).length ? el('span', { class: 'muted small' }, src.channels.join(', ')) : null,
      src.reason ? el('span', { class: 'source-reason' }, src.reason) : null);
  }));
  return collapsible(`Sources utilisées (${sources.length})`, list, open);
}

/** Rendu Markdown d'une réponse avec citations [Sn] converties en liens. */
function renderAnswer(container, text, sources) {
  const byRef = new Map((sources || []).map((s) => [s.ref, s]));
  const citation = (ref) => {
    const src = byRef.get(ref);
    const href = sourceHref(src);
    if (!src) return el('span', { class: 'cite', title: 'Source non disponible' }, `[${ref}]`);
    return el('a', { class: 'cite', href: href || '#', title: sourceLabel(src), onClick: (e) => {
      if (!href) { e.preventDefault(); return; }
      // surligne la source dans le panneau avant de naviguer
      const panel = container.closest('.msg');
      if (panel) panel.querySelectorAll('.source').forEach((n) => n.classList.toggle('target', n.dataset.ref === ref));
    } }, `[${ref}]`);
  };
  replace(container, renderMarkdown(text, { citation }));
}

function userMessage(content) {
  return el('div', { class: 'msg msg-user' }, el('div', { class: 'msg-role' }, 'Vous'), el('div', { class: 'msg-body' }, content));
}

function assistantMessage(message) {
  const body = el('div', { class: 'msg-body md' });
  renderAnswer(body, message.content || '', message.sources || []);
  return el('div', { class: 'msg msg-assistant' },
    el('div', { class: 'msg-role' }, 'Develop', message.created_at ? el('span', { class: 'muted' }, ` · ${formatRelative(message.created_at)}`) : null),
    sourcesPanel(message.sources || []),
    body);
}

// --------------------------------------------------------------------------- vue

export async function mount(root, route) {
  let conversationId = route.parts[0] || null;
  let abort = null; // AbortController de la génération en cours
  let alive = true;

  const convList = el('div', { class: 'chat-list' });
  const thread = el('div', { class: 'chat-thread' });
  const input = el('textarea', { class: 'textarea compose-input', placeholder: 'Posez une question sur le projet… (Entrée pour envoyer, Maj+Entrée pour un saut de ligne)', rows: 2 });
  const modeSelect = el('select', { class: 'select', id: 'chat-mode', style: { width: 'auto' }, title: MODES[0].help }, MODES.map((m) => el('option', { value: m.value, title: m.help }, m.label)));
  const modeHelp = el('span', { class: 'muted small' }, MODES[0].help);
  const sendBtn = el('button', { class: 'btn btn-primary', type: 'button' }, icon('send', 16), 'Envoyer');
  const stopBtn = el('button', { class: 'btn btn-danger hidden', type: 'button' }, icon('stop', 16), 'Arrêter');

  modeSelect.addEventListener('change', () => {
    const m = MODES.find((x) => x.value === modeSelect.value);
    modeSelect.title = m ? m.help : '';
    modeHelp.textContent = m ? m.help : '';
  });

  const engineInfo = () => (state.status && state.status.engine) || {};
  const engineConfigured = () => { const e = engineInfo(); return !!e.engine && e.engine !== 'none'; };
  const engineOk = () => !!engineInfo().ok;
  /** Encadré : aucun moteur (à configurer) ou moteur configuré mais injoignable (à démarrer). */
  const noEngineBox = () => {
    const e = engineInfo();
    if (!engineConfigured()) {
      return infoBox(el('div', el('div', { style: { fontWeight: '600' } }, 'Aucun moteur configuré : la recherche fonctionne, le chat non.'),
        el('div', 'Le mode « Recherche seule » est sélectionné. Pour obtenir des réponses, ', el('a', { href: '#/settings' }, 'configurez un moteur d\'IA'), '.')), 'warn');
    }
    return infoBox(el('div',
      el('div', { style: { fontWeight: '600' } }, `${ENGINE_LABELS[e.engine] || e.engine} injoignable${e.model ? ` (modèle ${e.model})` : ''} : le chat passe en mode « Recherche seule ».`),
      e.error ? el('div', { class: 'muted small' }, e.error) : null,
      el('div', e.engine === 'ollama' ? 'Démarrez le service (« ollama serve » ou l\'application Ollama) ' : 'Démarrez votre serveur local ',
        'ou vérifiez son adresse, puis ', el('a', { href: '#/settings' }, 'vérifiez la connexion dans les réglages'), '. Dès que le moteur répond, choisissez le mode « Auto ».')), 'warn');
  };

  // ---------------------------------------------------------------- conversations
  const loadConversations = async () => {
    replace(convList, spinner('Conversations…'));
    try {
      const items = await api.get('/api/conversations');
      if (!alive) return;
      if (!items.length) {
        replace(convList, el('div', { class: 'muted small', style: { padding: '12px' } }, 'Aucune conversation. Posez votre première question.'));
        return;
      }
      replace(convList, items.map((c) => el('div', { class: `conv-item${c.id === conversationId ? ' active' : ''}`, onClick: () => navigate(`#/chat/${encodeURIComponent(c.id)}`) },
        el('span', { class: 'conv-title', title: c.title || '' }, c.title || 'Sans titre'),
        el('span', { class: 'conv-tools' },
          el('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', title: 'Renommer', onClick: async (e) => {
            e.stopPropagation();
            const title = await promptDialog('Renommer la conversation', { value: c.title || '' });
            if (title === null || !title) return;
            try { await api.patch(`/api/conversations/${encodeURIComponent(c.id)}`, { title }); loadConversations(); } catch (err) { toastError(err); }
          } }, icon('notes', 14)),
          el('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', title: 'Supprimer', onClick: async (e) => {
            e.stopPropagation();
            if (!(await confirmDialog(`Supprimer la conversation « ${c.title || 'Sans titre'} » ?`, { danger: true, okLabel: 'Supprimer' }))) return;
            try {
              await api.del(`/api/conversations/${encodeURIComponent(c.id)}`);
              if (c.id === conversationId) navigate('#/chat'); else loadConversations();
            } catch (err) { toastError(err); }
          } }, icon('trash', 14))))));
    } catch (e) {
      replace(convList, errorBox(e));
    }
  };

  const createConversation = async (title) => {
    const created = await api.post('/api/conversations', { title: title || undefined });
    conversationId = created.id;
    // Mise à jour de l'URL sans remonter la vue (pas d'événement hashchange).
    history.replaceState(null, '', `#/chat/${encodeURIComponent(conversationId)}`);
    state.route = { name: 'chat', parts: [conversationId], query: {} };
    loadConversations();
  };

  // ---------------------------------------------------------------- fil
  const scrollToEnd = () => { thread.scrollTop = thread.scrollHeight; };

  const loadThread = async () => {
    if (!conversationId) {
      replace(thread, el('div', { class: 'chat-empty' },
        engineOk() ? null : noEngineBox(),
        emptyState('Nouvelle conversation', 'Posez une question : « Que fait OrderService ? », « Qui appelle POST /orders ? », « Quelles notes parlent de la facturation ? »')));
      return;
    }
    replace(thread, spinner('Chargement de la conversation…'));
    try {
      const conv = await api.get(`/api/conversations/${encodeURIComponent(conversationId)}`);
      if (!alive) return;
      const nodes = (conv.messages || []).map((m) => (m.role === 'user' ? userMessage(m.content || '') : assistantMessage(m)));
      replace(thread, engineOk() ? null : noEngineBox(),
        nodes.length ? nodes : el('div', { class: 'chat-empty' }, emptyState('Conversation vide', 'Posez votre première question ci-dessous.')));
      scrollToEnd();
    } catch (e) {
      replace(thread, errorBox(e, [el('button', { class: 'btn btn-sm', type: 'button', onClick: () => navigate('#/chat') }, 'Nouvelle conversation')]));
    }
  };

  // ---------------------------------------------------------------- envoi en flux
  const send = async () => {
    const content = input.value.trim();
    if (!content || abort) return;
    const mode = modeSelect.value;
    input.value = '';
    sendBtn.disabled = true;
    stopBtn.classList.remove('hidden');
    thread.querySelectorAll('.chat-empty').forEach((n) => n.remove());
    thread.appendChild(userMessage(content));

    // Squelette du message assistant, rempli au fil des événements.
    const statusLine = el('div', { class: 'msg-status' }, el('span', { class: 'spinner' }), el('span', 'Envoi…'));
    const sourcesHost = el('div', { class: 'sources' });
    const toolsHost = el('div');
    const body = el('div', { class: 'msg-body md' });
    const foot = el('div', { class: 'msg-foot' });
    const errorHost = el('div');
    const msg = el('div', { class: 'msg msg-assistant' }, el('div', { class: 'msg-role' }, 'Develop'), statusLine, sourcesHost, toolsHost, body, errorHost, foot);
    thread.appendChild(msg);
    scrollToEnd();

    let sources = [];
    let text = '';
    let pendingRender = false;
    const rerender = () => { renderAnswer(body, text, sources); pendingRender = false; scrollToEnd(); };

    const onEvent = (name, data) => {
      if (name === 'status') {
        replace(statusLine, el('span', { class: 'spinner' }), el('span', data.text || ''));
      } else if (name === 'sources') {
        sources = data.items || [];
        replace(sourcesHost, sourcesPanel(sources, mode === 'search_only'));
      } else if (name === 'tool_call') {
        toolsHost.appendChild(collapsible(el('span', { class: 'row' }, badge('outil', 'accent'), `${data.name || '?'}`),
          el('pre', { class: 'small' }, JSON.stringify(data.arguments || {}, null, 2))));
        replace(statusLine, el('span', { class: 'spinner' }), el('span', `Outil ${data.name || ''}…`));
      } else if (name === 'tool_result') {
        const added = data.added_sources || [];
        const failed = !!data.error; // l'orchestrateur ajoute `error` quand l'outil a échoué
        if (added.length) {
          sources = sources.concat(added.filter((s) => !sources.some((x) => x.ref === s.ref)));
          replace(sourcesHost, sourcesPanel(sources));
        }
        toolsHost.appendChild(collapsible(el('span', { class: 'row' }, badge(failed ? 'erreur' : 'résultat', failed ? 'error' : 'ok'), `${data.name || '?'}`),
          el('div',
            failed ? el('div', { class: 'text-danger' }, `L'outil a échoué : ${data.error}`) : null,
            data.summary ? el('div', data.summary) : null,
            added.length ? el('div', { class: 'muted small' }, `${added.length} source(s) ajoutée(s)`) : null), failed));
      } else if (name === 'delta') {
        text += data.text || '';
        // Rendu Markdown à la fin de chaque paragraphe pour limiter le coût.
        if ((data.text || '').includes('\n\n')) rerender();
        else if (!pendingRender) { pendingRender = true; setTimeout(rerender, 250); }
        statusLine.classList.add('hidden');
      } else if (name === 'done') {
        const usage = data.usage || {};
        const budget = data.budget || {};
        const truncated = data.truncated_sources || [];
        replace(foot,
          usage.prompt_tokens !== undefined ? el('span', `Entrée : ${formatNumber(usage.prompt_tokens)} tokens`) : null,
          usage.completion_tokens !== undefined ? el('span', `Réponse : ${formatNumber(usage.completion_tokens)} tokens`) : null,
          contextUsage(budget),
          toolsUsage(budget),
          truncated.length ? el('span', { class: 'muted', title: truncated.map((s) => s.path || s.title || s.ref).join('\n') }, `${truncated.length} source(s) non consultée(s), budget atteint`) : null);
        if (mode === 'search_only' && !text) replace(body, el('p', { class: 'muted' }, sources.length ? 'Sources trouvées ci-dessus (aucun appel au modèle).' : 'Aucune source trouvée pour cette question.'));
      } else if (name === 'error') {
        const err = { code: data.code, message: data.message || 'Erreur inconnue' };
        if (data.code === 'llm_unavailable') {
          const configured = engineConfigured();
          replace(errorHost, infoBox(el('div',
            el('div', { style: { fontWeight: '600' } }, configured ? 'Le moteur d\'IA n\'a pas répondu' : 'Configurez un moteur d\'IA'),
            el('div', data.message || 'Le modèle n\'a pas répondu.'),
            el('div', 'Les sources trouvées restent affichées ci-dessus. ',
              configured ? 'Démarrez le moteur ou ' : '',
              el('a', { href: '#/settings' }, configured ? 'vérifiez la connexion dans les réglages' : 'Ouvrir les réglages'), configured ? '.' : '')), 'warn'));
        } else {
          replace(errorHost, errorBox(err));
        }
        toastError(err);
      }
    };

    abort = new AbortController();
    try {
      if (!conversationId) await createConversation(content.slice(0, 60));
      await streamPost(`/api/conversations/${encodeURIComponent(conversationId)}/messages`, { content, mode }, { onEvent, signal: abort.signal });
    } catch (e) {
      if (e && e.name === 'AbortError') {
        replace(errorHost, infoBox('Génération arrêtée.', 'info'));
      } else {
        replace(errorHost, errorBox(e));
        toastError(e);
      }
    } finally {
      abort = null;
      statusLine.classList.add('hidden');
      // Dernier rendu seulement s'il y a du texte : ne pas effacer le message de remplacement
      // (mode « Recherche seule ») que l'événement done a placé dans `body`.
      if (text || pendingRender) rerender();
      sendBtn.disabled = false;
      stopBtn.classList.add('hidden');
      refreshStatus();
      loadConversations();
      input.focus();
    }
  };

  sendBtn.addEventListener('click', send);
  stopBtn.addEventListener('click', () => { if (abort) abort.abort(); });
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(220, input.scrollHeight)}px`;
  });

  // Message pré-rempli (depuis l'analyse d'impact).
  if (state.chatPrefill) {
    input.value = state.chatPrefill;
    state.chatPrefill = null;
    setTimeout(() => input.dispatchEvent(new Event('input')), 0);
  }
  if (!engineOk()) modeSelect.value = 'search_only';
  modeSelect.dispatchEvent(new Event('change'));

  replace(root, el('div', { class: 'chat' },
    el('aside', { class: 'chat-side' },
      el('div', { class: 'chat-side-head' }, el('strong', 'Conversations'),
        el('button', { class: 'btn btn-sm', type: 'button', title: 'Nouvelle conversation', onClick: () => navigate('#/chat') }, icon('plus', 14), 'Nouvelle')),
      convList),
    el('section', { class: 'chat-main' },
      thread,
      el('div', { class: 'chat-compose' }, el('div', { class: 'compose-box' },
        input,
        el('div', { class: 'compose-tools' },
          el('div', { class: 'row' }, el('label', { class: 'label', for: 'chat-mode' }, 'Mode'), modeSelect, modeHelp),
          el('div', { class: 'row' }, stopBtn, sendBtn)))))));

  await Promise.all([loadConversations(), loadThread()]);
  input.focus();

  return () => {
    alive = false;
    if (abort) abort.abort();
  };
}
