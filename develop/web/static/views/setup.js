// views/setup.js — assistant de configuration en 4 étapes (espace + dépôts, moteur d'IA,
// mode d'exécution, indexation). Les composants sont exportés pour l'écran Réglages.

import { api, watchIndex } from '/static/api.js';
import {
  el, replace, icon, toast, toastError, errorBox, infoBox, emptyState, spinner, modal, field, badge,
  copyButton, collapsible, dl, formatSize, formatNumber, formatDate,
} from '/static/ui.js';
import { state, navigate, refreshStatus } from '/static/app.js';

export const KIND_LABELS = {
  frontend: 'Frontend',
  backend: 'Backend',
  docs: 'Documentation',
  database: 'Base de données',
  mobile: 'Mobile',
  shared: 'Partagé',
  other: 'Autre',
};

/** Phases d'un run (PHASES de develop/indexing/indexer.py). */
const PHASE_LABELS = {
  scan: 'Parcours des fichiers',
  analyze: 'Analyse du code',
  documents: 'Documents',
  notes: 'Notes',
  relations: 'Relations',
  embeddings: 'Embeddings',
  done: 'Terminé',
};

export const ENGINE_LABELS = { ollama: 'Ollama', openai_compat: 'Serveur compatible OpenAI', none: 'Aucun moteur' };
export const OS_LABELS = { macos: 'macOS', linux: 'Linux', windows: 'Windows' };
const CATEGORY_LABELS = { chat_code: 'chat / code', chat: 'chat', code: 'code', embedding: 'embeddings' };

/** Statuts d'un run d'indexation (Indexer.run : ok | partial | cancelled | failed). */
const RUN_STATUS = {
  ok: { kind: 'success', icon: 'check', title: 'Indexation terminée' },
  partial: { kind: 'warn', icon: 'warn', title: 'Indexation terminée avec des erreurs' },
  cancelled: { kind: 'warn', icon: 'warn', title: 'Indexation interrompue' },
  failed: { kind: 'error', icon: 'warn', title: 'Indexation échouée' },
  error: { kind: 'error', icon: 'warn', title: 'Indexation échouée' },
};
/** Statistiques scalaires d'un run (Indexer._new_stats) → libellés ; les objets (repos, totals…) ne sont pas affichés. */
const STAT_LABELS = {
  full: 'indexation complète', files_scanned: 'fichiers parcourus', files_indexed: 'fichiers indexés',
  files_unchanged: 'fichiers inchangés', files_deleted: 'fichiers supprimés', symbols: 'symboles',
  endpoints: 'endpoints', models: 'modèles', chunks: 'morceaux', documents: 'documents', notes: 'notes',
  relations: 'relations', embeddings: 'embeddings', embeddings_pending: 'embeddings en attente',
  errors_total: 'erreurs', duration_s: 'durée',
};
const STAGE_LABELS = { scan: 'parcours', analyze: 'analyse', documents: 'documents', notes: 'notes', relations: 'relations', embeddings: 'embeddings', run: 'exécution' };
/** Compteurs de l'index (Database.stats). */
const COUNT_LABELS = {
  repos: 'Dépôts', files: 'Fichiers', symbols: 'Symboles', endpoints: 'Endpoints', models: 'Modèles',
  relations: 'Relations', documents: 'Documents', notes: 'Notes', chunks: 'Morceaux', embeddings: 'Embeddings',
  conversations: 'Conversations', messages: 'Messages',
};

const DEFAULT_URLS = { ollama: 'http://127.0.0.1:11434', openai_compat: 'http://127.0.0.1:1234/v1', none: '' };

function kindSelect(value, onChange) {
  return el('select', { class: 'select', value, onChange: (e) => onChange(e.target.value) },
    Object.entries(KIND_LABELS).map(([k, label]) => el('option', { value: k, selected: k === value }, label)));
}

function baseName(path) {
  const parts = String(path).replace(/[\\/]+$/, '').split(/[\\/]/);
  return parts[parts.length - 1] || path;
}

// --------------------------------------------------------------------------- dépôts imbriqués

/** Chemin de `child` relatif à `parent` (séparateur « / »), ou null si `child` n'est pas strictement sous `parent`. */
export function relativeUnder(child, parent) {
  const norm = (p) => String(p || '').replace(/\\/g, '/').replace(/\/+$/, '');
  const c = norm(child);
  const p = norm(parent);
  if (!c || !p || c === p || !c.startsWith(`${p}/`)) return null;
  return c.slice(p.length + 1);
}

/**
 * Ajoute à chaque dépôt de la liste l'exclusion (`exclude`, motifs de RepoConfig) des autres dépôts
 * situés sous son dossier : la racine d'un monodépôt n'indexe pas ce que ses sous-projets indexent
 * déjà, chaque fichier n'est donc indexé qu'une fois. Les motifs existants sont conservés.
 */
export function excludeNestedRepos(repos) {
  for (const repo of repos) {
    const exclude = [...(Array.isArray(repo.exclude) ? repo.exclude : [])];
    for (const other of repos) {
      const rel = relativeUnder(other.path, repo.path);
      if (rel && !exclude.includes(rel)) exclude.push(rel);
    }
    repo.exclude = exclude;
  }
  return repos;
}

/** Retire des dépôts de la liste l'exclusion qui visait `repo` (à appeler quand il quitte la liste). */
export function unexcludeNestedRepo(repo, repos) {
  for (const other of repos) {
    const rel = relativeUnder(repo.path, other.path);
    if (rel && Array.isArray(other.exclude) && other.exclude.includes(rel)) other.exclude = other.exclude.filter((p) => p !== rel);
  }
  return repos;
}

/** Rappel « hors a, b » des motifs exclus d'un dépôt, ou null s'il n'y en a pas. */
function excludeHint(exclude) {
  if (!exclude || !exclude.length) return null;
  return el('span', { class: 'muted', title: 'Dossiers ou motifs exclus du parcours de ce dépôt (indexés comme dépôts à part entière quand ce sont des sous-projets).' }, `hors ${exclude.join(', ')}`);
}

// --------------------------------------------------------------------------- sélecteur de dossier

/** Ouvre le sélecteur de dossiers côté serveur ; résout le chemin choisi ou null. */
export function browseFolder(startPath = '') {
  return new Promise((resolve) => {
    let dialog = null;
    let currentPath = startPath;
    let settled = false;
    const finish = (value) => { settled = true; dialog.close(); resolve(value); };

    const crumbs = el('div', { class: 'browser-crumbs' });
    const list = el('div', { class: 'browser-list' });
    const pathInput = el('input', { class: 'input mono', type: 'text', placeholder: 'Chemin du dossier…' });
    const chooseBtn = el('button', { class: 'btn btn-primary', type: 'button', onClick: () => finish(pathInput.value.trim() || currentPath || null) }, 'Choisir ce dossier');

    const load = async (path) => {
      replace(list, spinner('Lecture du dossier…'));
      try {
        const data = await api.post('/api/fs/browse', path ? { path } : {});
        currentPath = data.path || '';
        pathInput.value = currentPath;
        renderCrumbs(data);
        renderEntries(data);
      } catch (e) {
        replace(list, errorBox(e));
      }
    };

    const renderCrumbs = (data) => {
      // Chemins POSIX (/a/b) et Windows (C:\a\b ou C:/a/b) : découpe sur les deux séparateurs,
      // reconstruction avec le séparateur d'origine et le préfixe de lecteur.
      const raw = String(data.path || '');
      const sep = raw.includes('\\') && !raw.includes('/') ? '\\' : '/';
      const drive = /^[A-Za-z]:/.exec(raw);
      const rest = drive ? raw.slice(drive[0].length) : raw;
      const segments = rest.split(/[\\/]/).filter(Boolean);
      const nodes = [el('button', { class: 'crumb', type: 'button', onClick: () => load('') }, 'Racines')];
      let acc = drive ? drive[0] : '';
      if (drive) nodes.push(el('span', { class: 'muted' }, sep), el('button', { class: 'crumb', type: 'button', onClick: () => load(`${drive[0]}${sep}`) }, drive[0]));
      segments.forEach((seg) => {
        acc += `${sep}${seg}`;
        const target = acc;
        nodes.push(el('span', { class: 'muted' }, sep), el('button', { class: 'crumb', type: 'button', onClick: () => load(target) }, seg));
      });
      replace(crumbs, nodes);
    };

    const renderEntries = (data) => {
      const entries = data.entries || [];
      const nodes = [];
      if (data.parent) nodes.push(el('div', { class: 'browser-entry', onClick: () => load(data.parent) }, icon('back', 14), el('span', '..'), el('span', { class: 'muted small' }, 'dossier parent')));
      for (const entry of entries) {
        nodes.push(el('div', { class: 'browser-entry', onClick: () => load(entry.path), title: entry.path },
          icon(entry.is_git ? 'git' : 'folder', 15),
          el('span', { class: 'grow truncate' }, entry.name),
          entry.is_git ? badge('git', 'accent') : null,
          entry.has_marker ? badge('projet', 'ok') : null));
      }
      if (!nodes.length) nodes.push(el('div', { class: 'muted', style: { padding: '12px' } }, 'Aucun sous-dossier.'));
      replace(list, nodes);
    };

    pathInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); load(pathInput.value.trim()); } });

    dialog = modal({
      title: 'Choisir un dossier',
      wide: true,
      body: el('div', { class: 'stack' },
        el('div', { class: 'row' }, pathInput, el('button', { class: 'btn', type: 'button', onClick: () => load(pathInput.value.trim()) }, 'Aller')),
        crumbs, list,
        el('div', { class: 'hint' }, 'Les dossiers marqués « git » sont des dépôts ; « projet » signale un fichier de projet (package.json, composer.json, pyproject…).')),
      actions: [el('button', { class: 'btn', type: 'button', onClick: () => finish(null) }, 'Annuler'), chooseBtn],
      onClose: () => { if (!settled) resolve(null); },
    });
    load(startPath);
  });
}

// --------------------------------------------------------------------------- détection des sous-projets

/** Lance la détection sur un dossier et propose une liste cochable ; résout les dépôts choisis ou null. */
export function discoverSubprojects(path) {
  return new Promise((resolve) => {
    let dialog = null;
    let settled = false;
    const finish = (value) => { settled = true; dialog.close(); resolve(value); };
    const body = el('div', spinner('Détection des sous-projets…'));
    const addBtn = el('button', { class: 'btn btn-primary', type: 'button', disabled: true }, 'Ajouter la sélection');
    const rows = [];

    dialog = modal({
      title: 'Détecter les sous-projets',
      wide: true,
      body,
      actions: [el('button', { class: 'btn', type: 'button', onClick: () => finish(null) }, 'Annuler'), addBtn],
      onClose: () => { if (!settled) resolve(null); },
    });

    api.post('/api/repos/discover', { path }).then((data) => {
      const subprojects = data.subprojects || [];
      const root = data.root || {};
      const candidates = subprojects.length ? subprojects : [{ path, name: baseName(path), suggested_kind: root.suggested_kind || root.kind || 'other', frameworks: root.frameworks || [] }];
      const nestedCount = subprojects.filter((sp) => relativeUnder(sp.path, path)).length;
      const rootListed = subprojects.length > nestedCount; // la racine (dépôt Git, workspace…) est proposée pour le reste de son contenu
      const list = el('div', { class: 'list' });
      const hints = [];
      // Un candidat qui en contient d'autres (la racine d'un monodépôt) exclut de son parcours ceux
      // qui sont cochés : ils sont indexés comme dépôts à part entière, jamais deux fois.
      const nestedChecked = (row) => rows.filter((o) => o !== row && o.checked && relativeUnder(o.path, row.path)).map((o) => relativeUnder(o.path, row.path));
      const renderHints = () => { for (const { row, node } of hints) replace(node, excludeHint(nestedChecked(row))); };
      for (const sp of candidates) {
        const row = { checked: true, name: sp.name || baseName(sp.path), path: sp.path, kind: sp.suggested_kind || 'other' };
        rows.push(row);
        const check = el('input', { type: 'checkbox', checked: true, onChange: (e) => { row.checked = e.target.checked; renderHints(); } });
        const hint = el('span');
        hints.push({ row, node: hint });
        list.appendChild(el('div', { class: 'list-item' },
          check,
          el('div', { class: 'list-item-main' },
            el('div', { class: 'list-item-title' }, row.name),
            el('div', { class: 'list-item-meta' }, el('span', { class: 'mono truncate', title: sp.path }, sp.path),
              (sp.frameworks || []).length ? el('span', (sp.frameworks || []).join(', ')) : null,
              hint)),
          el('div', { style: { width: '160px' } }, kindSelect(row.kind, (v) => { row.kind = v; }))));
      }
      renderHints();
      replace(body,
        subprojects.length
          // Le chemin peut être coupé n'importe où (overflow-wrap) : un chemin long sans tiret n'a aucune
          // césure naturelle et élargirait sinon le corps de la modale au-delà de la fenêtre.
          ? el('p', `${nestedCount} sous-projet(s) détecté(s) dans `, el('span', { class: 'mono', style: { overflowWrap: 'anywhere' } }, path),
            `.${rootListed ? ' Le dossier lui-même est proposé pour le reste de son contenu, hors sous-projets cochés.' : ''} Ajustez le type si besoin.`)
          : infoBox('Aucun sous-projet détecté : le dossier sera ajouté comme un seul dépôt.', 'info'),
        list);
      addBtn.disabled = false;
      addBtn.addEventListener('click', () => finish(excludeNestedRepos(rows.filter((r) => r.checked).map((r) => ({ name: r.name, path: r.path, kind: r.kind, enabled: true, exclude: [] })))));
    }).catch((e) => replace(body, errorBox(e)));
  });
}

// --------------------------------------------------------------------------- éditeur de dépôts

/**
 * Liste des dépôts : un seul bouton d'ajout (sélecteur de dossier puis détection des sous-projets),
 * type modifiable, suppression, et « Sous-projets » sur un dépôt déjà listé (détection sans rouvrir le sélecteur).
 * Un dépôt situé sous un autre dépôt de la liste en est exclu (`exclude`) à l'ajout, et réintégré au retrait.
 */
export function repoEditor(repos, onChange) {
  const root = el('div');
  const render = () => {
    const addRepos = (items) => {
      let added = 0;
      for (const item of items) {
        if (repos.some((r) => r.path === item.path)) continue;
        repos.push(item);
        added += 1;
      }
      if (!added && items.length) toast('Ces dépôts sont déjà dans la liste.', { type: 'info' });
      excludeNestedRepos(repos);
      onChange();
      render();
    };
    const removeRepo = (index) => {
      const [removed] = repos.splice(index, 1);
      if (removed) unexcludeNestedRepo(removed, repos);
      onChange();
      render();
    };
    const detectInto = async (path) => {
      const picked = await discoverSubprojects(path);
      if (picked) addRepos(picked);
    };

    const rows = repos.length
      ? el('div', { class: 'list' }, repos.map((repo, index) => el('div', { class: 'list-item' },
        icon('git', 16),
        el('div', { class: 'list-item-main' },
          el('input', { class: 'input', type: 'text', value: repo.name, placeholder: 'Nom', 'aria-label': 'Nom du dépôt', onInput: (e) => { repo.name = e.target.value; onChange(); } }),
          el('div', { class: 'list-item-meta' }, el('span', { class: 'mono truncate', title: repo.path }, repo.path), excludeHint(repo.exclude))),
        el('div', { style: { width: '160px' } }, kindSelect(repo.kind || 'other', (v) => { repo.kind = v; onChange(); })),
        el('button', { class: 'btn btn-ghost btn-sm', type: 'button', title: 'Détecter les sous-projets de ce dossier et les ajouter à la liste', onClick: () => detectInto(repo.path) }, icon('search', 14), 'Sous-projets'),
        el('button', { class: 'btn btn-ghost btn-icon', type: 'button', title: 'Retirer ce dépôt', 'aria-label': 'Retirer', onClick: () => removeRepo(index) }, icon('trash', 16)))))
      : emptyState('Aucun dépôt', 'Ajoutez le dossier de votre projet : Develop le lira sans jamais le modifier et y détectera les sous-projets (frontend, backend, documentation…).');

    replace(root,
      rows,
      el('div', { class: 'row', style: { marginTop: '10px' } },
        el('button', { class: 'btn', type: 'button', onClick: async () => {
          const path = await browseFolder(repos.length ? repos[0].path : '');
          if (path) await detectInto(path);
        } }, icon('folder', 16), 'Ajouter un dossier'),
        el('span', { class: 'muted small' }, 'Develop détecte les sous-projets du dossier choisi et vous laisse cocher ceux à indexer.')));
  };
  render();
  return root;
}

// --------------------------------------------------------------------------- moteur d'IA

/**
 * num_ctx conseillé d'après la RAM totale (Go), à défaut de la suggestion de POST /api/llm/test :
 * ≥ 32 Go → 16 384 ; ≥ 16 Go → 8 192 ; en dessous → 4 096 ; null si la RAM est inconnue.
 */
export function adviseNumCtx(ramTotalGb) {
  const ram = Number(ramTotalGb);
  if (!ram || Number.isNaN(ram) || ram <= 0) return null;
  if (ram >= 32) return 16384;
  if (ram >= 16) return 8192;
  return 4096;
}

/** Avertissement quand le modèle testé n'a pas la capacité « tools » (`model_info.supports_tools` false) ; null sinon. */
export function toolsWarning(info) {
  if (!info || info.supports_tools !== false) return null;
  return infoBox(el('div',
    info.name ? el('div', { style: { fontWeight: '600' } }, info.name) : null,
    el('div', 'Ce modèle ne sait pas appeler les outils de Develop : les réponses s\'appuieront seulement sur la recherche initiale ; préférez un modèle avec outils (qwen3.5:9b, gemma4:12b, devstral).')), 'warn');
}

/** Carte « Votre machine » ; `profilePromise` : GET /api/system/profile partagé avec le formulaire moteur. */
function machineCard(profilePromise) {
  const card = el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Votre machine'), spinner('Analyse de la machine…'));
  profilePromise.then((p) => {
    const gpus = (p.gpus || []).map((g) => g.name + (g.vram_gb ? ` (${g.vram_gb} Go)` : '')).join(', ');
    replace(card,
      el('div', { class: 'card-title' }, 'Votre machine'),
      el('p', p.summary || ''),
      dl([
        ['Système', el('span', { title: p.os_version || '' }, `${OS_LABELS[p.os] || p.os || '?'} (${p.arch || '?'})`)],
        ['Processeur', `${p.cpu_model || '?'}${p.cpu_cores_logical ? ` — ${p.cpu_cores_logical} cœurs` : ''}`],
        ['Mémoire', p.ram_total_gb ? `${p.ram_total_gb} Go${p.unified_memory ? ' (unifiée)' : ''}${p.ram_available_gb ? `, ${p.ram_available_gb} Go disponibles` : ''}` : '—'],
        ['Accélérateur', gpus || (p.apple_silicon ? 'Apple Silicon' : 'Aucun GPU détecté')],
        ['Disque libre', p.disk_free_gb ? `${p.disk_free_gb} Go${p.disk_total_gb ? ` sur ${p.disk_total_gb} Go` : ''}` : '—'],
      ]),
      (p.warnings || []).length ? el('div', { style: { marginTop: '8px' } }, p.warnings.map((w) => infoBox(w, 'warn'))) : null);
  }).catch((e) => replace(card, el('div', { class: 'card-title' }, 'Votre machine'), errorBox(e)));
  return card;
}

/** Carte des moteurs détectés ; `onReport(report)` reçoit le rapport brut (pour ouvrir les étapes d'installation). */
function enginesCard(onUse, onReport = null) {
  const card = el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Moteurs détectés'), spinner('Recherche des moteurs locaux…'));
  const probeRow = (probe, label) => el('div', { class: 'list-item' },
    el('div', { class: 'list-item-main' },
      el('div', { class: 'list-item-title row' }, label, probe.reachable ? badge('joignable', 'ok') : badge('injoignable', 'error'), probe.version ? el('span', { class: 'muted small' }, `v${probe.version}`) : null),
      el('div', { class: 'list-item-meta' }, el('span', { class: 'mono' }, probe.base_url),
        probe.reachable ? el('span', `${(probe.models || []).length} modèle(s)`) : null,
        probe.error ? el('span', { class: 'muted' }, probe.error) : null)),
    probe.reachable ? el('button', { class: 'btn btn-sm', type: 'button', onClick: () => onUse(probe.engine, probe.base_url) }, 'Utiliser') : null);

  api.get('/api/system/engines').then((r) => {
    if (onReport) onReport(r);
    const items = [probeRow(r.ollama || {}, 'Ollama')];
    for (const server of r.openai_servers || []) items.push(probeRow(server, server.label || 'Serveur compatible OpenAI'));
    items.push(el('div', { class: 'list-item' }, el('div', { class: 'list-item-main' },
      el('div', { class: 'list-item-title row' }, 'Docker', r.docker_available ? badge(r.docker_running ? 'en marche' : 'installé, arrêté', r.docker_running ? 'ok' : 'warn') : badge('absent', 'neutral')),
      el('div', { class: 'list-item-meta' }, r.docker_ollama_container ? el('span', `Conteneur Ollama : ${r.docker_ollama_container}`) : el('span', 'Docker n\'est pas obligatoire.')))));
    replace(card,
      el('div', { class: 'card-title' }, 'Moteurs détectés'),
      el('div', { class: 'list' }, items),
      r.ollama_binary ? el('div', { class: 'hint', style: { marginTop: '8px' } }, `Binaire Ollama : ${r.ollama_binary} — modèles dans ${r.ollama_models_dir || '?'}`) : el('div', { class: 'hint', style: { marginTop: '8px' } }, 'Ollama n\'est pas installé : voir les instructions ci-dessous.'));
  }).catch((e) => replace(card, el('div', { class: 'card-title' }, 'Moteurs détectés'), errorBox(e)));
  return card;
}

/** Une recommandation ; `onPick(tag, isEmbedding)` reprend le tag dans le formulaire (chat ou embeddings). */
function recommendationItem(r, onPick = null, isEmbedding = false) {
  const useBtn = onPick && r.tag ? el('button', {
    class: `btn btn-sm${r.installed ? ' btn-primary' : ''}`, type: 'button',
    title: r.installed
      ? `Utiliser ${r.tag} comme modèle ${isEmbedding ? 'd\'embeddings' : 'de chat'}`
      : `Renseigne ${r.tag} dans le formulaire ; installez-le d'abord avec la commande ci-contre`,
    onClick: () => onPick(r.tag, isEmbedding),
  }, 'Utiliser ce modèle') : null;
  return el('div', { class: 'reco-item' },
    el('div',
      el('div', { class: 'row' }, el('strong', { class: 'mono' }, r.tag), r.category ? badge(CATEGORY_LABELS[r.category] || r.category) : null, r.installed ? badge('installé', 'ok') : null, r.size_gb ? el('span', { class: 'muted small' }, `${r.size_gb} Go`) : null),
      el('div', { class: 'muted small' }, r.reason || '')),
    el('div', { class: 'reco-actions' },
      r.pull_command ? el('div', { class: 'reco-cmd' }, el('span', r.pull_command), copyButton(r.pull_command, '')) : null,
      useBtn));
}

/**
 * Carte du conseiller de modèles. `advisor` (facultatif) : objet partagé avec enginesCard —
 * `advisor.needsInstall` ouvre d'emblée les étapes d'installation, `advisor.steps` reçoit le <details>.
 */
function recommendationsCard(onPick = null, advisor = null) {
  const card = el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Modèles recommandés pour cette machine'), spinner('Calcul des recommandations…'));
  api.get('/api/system/recommendations').then((data) => {
    const groups = [
      ['recommended', 'Recommandés', 'ok'],
      ['possible_slow', 'Possibles mais lents', 'warn'],
      ['too_big', 'Trop gros pour cette machine', 'error'],
    ];
    const recs = data.recommendations || [];
    const groupNodes = groups.map(([verdict, label, kind]) => {
      const items = recs.filter((r) => r.verdict === verdict);
      if (!items.length) return null;
      return el('div', { class: 'reco-group' }, el('div', { class: 'row', style: { marginBottom: '4px' } }, badge(label, kind), el('span', { class: 'muted small' }, `${items.length}`)),
        items.map((r) => recommendationItem(r, onPick, false)));
    });
    const embeddings = data.embedding || [];
    const steps = data.install_steps || [];
    const stepsNode = steps.length ? collapsible('Étapes d\'installation pour votre système', steps.map((s) => el('div', { style: { marginBottom: '10px' } },
      el('div', { style: { fontWeight: '600' } }, s.title),
      (s.commands || []).map((c) => el('div', { class: 'cmd-line' }, el('code', c), copyButton(c))),
      s.notes ? el('div', { class: 'hint' }, s.notes) : null)), !!(advisor && advisor.needsInstall)) : null;
    if (advisor) advisor.steps = stepsNode;
    replace(card,
      el('div', { class: 'card-title' }, 'Modèles recommandés pour cette machine', data.tier ? badge(data.tier, 'accent') : null),
      data.profile_summary ? el('p', { class: 'muted' }, data.profile_summary) : null,
      (data.warnings || []).map((w) => infoBox(w, 'warn')),
      recs.length ? groupNodes : emptyState('Aucune recommandation disponible'),
      embeddings.length ? el('div', { class: 'reco-group' }, el('div', { class: 'row', style: { marginBottom: '4px' } }, badge('Embeddings (recherche sémantique)', 'accent')), embeddings.map((r) => recommendationItem(r, onPick, true))) : null,
      stepsNode,
      el('div', { class: 'hint' }, 'Develop ne télécharge rien lui-même : lancez la commande dans un terminal, puis « Utiliser ce modèle » (ou « Rafraîchir » dans le formulaire) pour le sélectionner.'));
  }).catch((e) => replace(card, el('div', { class: 'card-title' }, 'Modèles recommandés'), errorBox(e)));
  return card;
}

/**
 * Formulaire moteur d'IA : moteur, URL, modèle (liste rafraîchissable), embeddings, réflexion interne
 * (think), num_ctx avec valeur conseillée, test. `llm` est modifié en place ; `onChange()` prévient le parent.
 */
export function engineForm(llm, onChange = () => {}) {
  if (!llm.engine) llm.engine = 'none';
  if (!llm.base_url) llm.base_url = DEFAULT_URLS[llm.engine] || '';
  let models = [];
  let ramGb = null; // RAM totale (Go) d'après GET /api/system/profile
  let ramAdvice = null; // num_ctx conseillé d'après la RAM
  let testAdvice = null; // num_ctx suggéré par POST /api/llm/test pour le modèle courant (prioritaire)

  const engineSelect = el('select', { class: 'select' },
    el('option', { value: 'ollama', selected: llm.engine === 'ollama' }, 'Ollama'),
    el('option', { value: 'openai_compat', selected: llm.engine === 'openai_compat' }, 'Serveur local compatible OpenAI (LM Studio, llama.cpp, vLLM, Docker…)'),
    el('option', { value: 'none', selected: llm.engine === 'none' }, 'Aucun moteur (recherche seule, pas de chat)'));
  const urlInput = el('input', { class: 'input mono', type: 'text', value: llm.base_url, placeholder: DEFAULT_URLS.ollama });
  const keyInput = el('input', { class: 'input', type: 'password', value: llm.api_key || '', placeholder: 'Facultatif (clé factice acceptée par certains serveurs)', autocomplete: 'off' });
  const modelList = el('datalist', { id: 'llm-model-list' });
  const modelInput = el('input', { class: 'input mono', type: 'text', list: 'llm-model-list', value: llm.model || '', placeholder: 'ex. qwen2.5-coder:7b' });
  const embedSelect = el('select', { class: 'select' });
  const thinkInput = el('input', { type: 'checkbox', checked: !!llm.think });
  const ctxInput = el('input', { class: 'input', type: 'number', min: 1024, step: 1024, value: llm.num_ctx || 8192 });
  const ctxAdvice = el('span'); // aide du champ num_ctx : « 16 384 conseillé sur cette machine » + rappel
  const modelsInfo = el('span', { class: 'muted small' });
  const testResult = el('div');
  const toolsHost = el('div'); // avertissement « modèle sans capacité tools »
  const keyField = field('Clé API', keyInput);

  const refreshBtn = el('button', { class: 'btn btn-sm', type: 'button', onClick: () => loadModels() }, icon('refresh', 14), 'Rafraîchir');
  // Même libellé que les étapes d'installation du conseiller (« Vérifier »).
  const testBtn = el('button', { class: 'btn', type: 'button', onClick: () => runTest() }, 'Vérifier la connexion');

  /** Aide du champ num_ctx : valeur conseillée (test du modèle, sinon RAM) avec bouton « Utiliser » si elle diffère. */
  const renderCtxAdvice = () => {
    const value = testAdvice || ramAdvice;
    const origin = testAdvice ? 'fenêtre du modèle et mémoire de la machine' : (ramGb ? `${formatNumber(ramGb)} Go de mémoire` : '');
    const current = Number(ctxInput.value) || 0;
    replace(ctxAdvice,
      value ? el('span', { class: 'ctx-advice' }, el('strong', `${formatNumber(value)} conseillé sur cette machine`), origin ? ` (${origin})` : '', '. ') : null,
      'Un contexte trop grand ralentit fortement un modèle local ; trop petit, les sources ne tiennent pas dans la réponse.',
      value && current !== value
        ? [' ', el('button', { class: 'btn btn-sm', type: 'button', disabled: llm.engine === 'none', onClick: () => { ctxInput.value = value; sync(); } }, `Utiliser ${formatNumber(value)}`)]
        : null);
  };

  const sync = () => {
    llm.engine = engineSelect.value;
    llm.base_url = urlInput.value.trim();
    llm.api_key = keyInput.value;
    llm.model = modelInput.value.trim();
    // Tant que la liste n'est pas rendue, le select n'a aucune option : ne pas écraser le
    // modèle d'embeddings enregistré par ''.
    if (embedSelect.options.length) llm.embedding_model = embedSelect.value;
    llm.think = !!thinkInput.checked;
    llm.num_ctx = Number(ctxInput.value) || 8192;
    keyField.classList.toggle('hidden', llm.engine !== 'openai_compat');
    const disabled = llm.engine === 'none';
    [urlInput, modelInput, embedSelect, thinkInput, ctxInput, refreshBtn, testBtn].forEach((n) => { n.disabled = disabled; });
    renderCtxAdvice();
    onChange();
  };

  /** Applique un résultat de POST /api/llm/test : num_ctx suggéré et avertissement « sans outils ». */
  const applyTest = (r) => {
    if (r && r.suggested_num_ctx) testAdvice = Number(r.suggested_num_ctx) || null;
    renderCtxAdvice();
    replace(toolsHost, toolsWarning(r && r.model_info));
  };

  /** Sonde silencieuse du modèle déjà configuré (au montage) : même résultat que le test, sans l'afficher. */
  const probeModel = async () => {
    if (llm.engine === 'none' || !llm.model) return;
    const model = llm.model;
    try {
      const r = await api.post('/api/llm/test', { engine: llm.engine, base_url: llm.base_url, model, embedding_model: '', api_key: llm.api_key || '' });
      if (llm.model === model) applyTest(r); // sinon le modèle a changé entre-temps
    } catch (_) { /* moteur injoignable : « Vérifier la connexion » le dira */ }
  };

  const renderEmbeddings = () => {
    const current = llm.embedding_model || '';
    const candidates = models.filter((m) => m.is_embedding || /embed/i.test(m.name));
    const names = new Set(candidates.map((m) => m.name));
    if (current && !names.has(current)) names.add(current);
    replace(embedSelect,
      el('option', { value: '', selected: !current }, 'Aucun : recherche lexicale seule'),
      [...names].map((name) => el('option', { value: name, selected: name === current }, name)));
  };

  const loadModels = async () => {
    sync();
    if (llm.engine === 'none') return;
    replace(modelsInfo, spinner('Lecture des modèles…'));
    try {
      models = await api.get('/api/llm/models', { engine: llm.engine, base_url: llm.base_url });
      replace(modelList, models.map((m) => el('option', { value: m.name }, [m.parameter_size, m.quantization].filter(Boolean).join(' '))));
      replace(modelsInfo, models.length ? `${models.length} modèle(s) disponible(s)` : 'Aucun modèle : installez-en un (voir recommandations).');
      renderEmbeddings();
    } catch (e) {
      replace(modelsInfo, el('span', { class: 'muted' }, `Modèles indisponibles : ${e.message}`));
    }
  };

  const runTest = async () => {
    sync();
    replace(testResult, spinner('Test en cours…'));
    testBtn.disabled = true;
    try {
      const r = await api.post('/api/llm/test', { engine: llm.engine, base_url: llm.base_url, model: llm.model, embedding_model: llm.embedding_model, api_key: llm.api_key || '' });
      const info = r.model_info || {};
      const status = r.engine_status || {};
      replace(testResult, el('div', { class: `alert alert-${r.ok ? 'success' : 'error'}` }, icon(r.ok ? 'check' : 'warn'),
        el('div', { class: 'alert-body' },
          el('div', { style: { fontWeight: '600' } }, r.ok ? 'Moteur opérationnel' : 'Le test a échoué'),
          r.error ? el('div', r.error) : null,
          dl([
            ['Moteur', status.ok ? `${status.engine || llm.engine} joignable${status.version ? ` (v${status.version})` : ''}` : (status.error || 'injoignable')],
            ['Fenêtre de contexte', info.context_length ? `${formatNumber(info.context_length)} tokens` : 'inconnue'],
            ['Appels d\'outils', info.supports_tools ? 'pris en charge nativement' : 'protocole textuel (repli)'],
            ['Embeddings', llm.embedding_model ? (r.embedding_ok ? `OK (dimension ${r.embedding_dim || '?'})` : 'échec') : 'désactivés (recherche lexicale seule)'],
            r.suggested_num_ctx ? ['num_ctx suggéré', el('span', { class: 'row' }, formatNumber(r.suggested_num_ctx), el('button', { class: 'btn btn-sm', type: 'button', onClick: () => { ctxInput.value = r.suggested_num_ctx; sync(); } }, 'Utiliser'))] : null,
          ]))));
      applyTest(r);
    } catch (e) {
      replace(testResult, errorBox(e));
    } finally {
      testBtn.disabled = false;
    }
  };

  engineSelect.addEventListener('change', () => {
    const previousDefault = Object.values(DEFAULT_URLS).includes(urlInput.value.trim()) || !urlInput.value.trim();
    if (previousDefault) urlInput.value = DEFAULT_URLS[engineSelect.value] || '';
    sync();
    loadModels();
  });
  [urlInput, keyInput, modelInput, ctxInput].forEach((n) => n.addEventListener('input', sync));
  // Un autre modèle : la suggestion et l'avertissement du test précédent ne le concernent plus.
  modelInput.addEventListener('input', () => { testAdvice = null; replace(toolsHost); renderCtxAdvice(); });
  [embedSelect, thinkInput].forEach((n) => n.addEventListener('change', sync));
  urlInput.addEventListener('change', loadModels);

  const form = el('div', { class: 'card' },
    el('div', { class: 'card-title' }, 'Configuration du moteur'),
    field('Moteur', engineSelect),
    field('Adresse du serveur', urlInput, 'Toujours une adresse locale : rien ne sort de la machine.'),
    keyField,
    field(el('span', { class: 'row' }, 'Modèle de chat', refreshBtn, modelsInfo), [modelInput, modelList], 'Choisissez un modèle installé ou saisissez son nom exact.'),
    field('Modèle d\'embeddings', embedSelect, 'Sans embeddings, la recherche reste fonctionnelle (lexicale + identifiants + graphe).'),
    el('div', { class: 'field' },
      el('label', { class: 'checkbox' }, thinkInput, el('span', 'Réflexion interne du modèle (think) : plus précis, 2 à 3 fois plus lent')),
      el('div', { class: 'hint' }, 'Transmise seulement aux modèles qui la proposent (qwen3, deepseek-r1…) ; sans effet sur les autres.')),
    field('Fenêtre de contexte demandée (num_ctx)', ctxInput, ctxAdvice),
    el('div', { class: 'row' }, testBtn),
    testResult,
    toolsHost);

  const useEngine = (engine, baseUrl) => {
    engineSelect.value = engine;
    urlInput.value = baseUrl;
    sync();
    loadModels();
    toast(`${ENGINE_LABELS[engine] || engine} sélectionné.`, { type: 'success' });
  };

  /** « Utiliser ce modèle » du conseiller : remplit le modèle de chat ou d'embeddings. */
  const onPick = (tag, isEmbedding) => {
    if (llm.engine === 'none') {
      // Les recommandations sont des tags Ollama : on active ce moteur s'il n'y en a aucun.
      engineSelect.value = 'ollama';
      if (!urlInput.value.trim()) urlInput.value = DEFAULT_URLS.ollama;
      sync();
    }
    if (isEmbedding) {
      llm.embedding_model = tag;
      renderEmbeddings(); // ajoute le tag à la liste s'il n'y est pas et le sélectionne
      sync();
      toast(`Modèle d'embeddings « ${tag} » sélectionné.`, { type: 'success' });
    } else {
      modelInput.value = tag;
      sync();
      toast(`Modèle de chat « ${tag} » sélectionné.`, { type: 'success' });
    }
    loadModels();
    form.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  };

  // État partagé entre la carte des moteurs et le conseiller : sans Ollama installé ou sans
  // moteur joignable, les étapes d'installation sont dépliées d'emblée.
  const advisor = { steps: null, needsInstall: false };
  const onEnginesReport = (r) => {
    const reachable = !!(r.ollama && r.ollama.reachable) || (r.openai_servers || []).some((s) => s && s.reachable);
    advisor.needsInstall = !r.ollama_binary || !reachable;
    if (advisor.steps && advisor.needsInstall) advisor.steps.open = true;
  };

  const profilePromise = api.get('/api/system/profile');
  const root = el('div',
    el('div', { class: 'card-grid' }, machineCard(profilePromise), enginesCard(useEngine, onEnginesReport)),
    el('div', { style: { height: '14px' } }),
    recommendationsCard(onPick, advisor),
    form);
  // renderEmbeddings() AVANT le premier sync() : sinon le select vide écrase embedding_model.
  renderEmbeddings();
  sync();
  // num_ctx conseillé d'après la RAM tant que le modèle n'a pas été testé.
  profilePromise.then((p) => { ramGb = p.ram_total_gb || null; ramAdvice = adviseNumCtx(ramGb); renderCtxAdvice(); }).catch(() => {});
  if (llm.engine !== 'none') { loadModels(); probeModel(); }
  return root;
}

// --------------------------------------------------------------------------- mode d'exécution

/** Choix du mode réseau : hors ligne strict ou réseau local + hôtes autorisés. */
export function modeForm(draft, onChange = () => {}) {
  if (!draft.mode) draft.mode = 'strict';
  if (!Array.isArray(draft.allowed_hosts)) draft.allowed_hosts = [];
  const hosts = el('textarea', { class: 'textarea mono', value: draft.allowed_hosts.join('\n'), placeholder: 'ollama.lan\n192.168.1.20', style: { minHeight: '80px' } });
  const hostsField = field('Hôtes autorisés (un par ligne)', hosts, 'Seuls ces hôtes pourront être contactés en plus de la machine locale.');
  const sync = () => {
    draft.allowed_hosts = hosts.value.split(/\n+/).map((h) => h.trim()).filter(Boolean);
    hostsField.classList.toggle('hidden', draft.mode !== 'local_network');
    onChange();
  };
  const radio = (value, title, text) => el('label', { class: 'radio-card' },
    el('input', { type: 'radio', name: 'mode', value, checked: draft.mode === value, onChange: () => { draft.mode = value; sync(); } }),
    el('div', el('div', { style: { fontWeight: '600' } }, title), el('div', { class: 'muted small' }, text)));
  hosts.addEventListener('input', sync);
  const root = el('div', { class: 'stack' },
    radio('strict', 'Hors ligne strict (recommandé)', 'Aucune requête réseau ne quitte la machine : toute tentative est bloquée et consignée.'),
    radio('local_network', 'Local avec services réseau internes', 'Autorise des services sur votre réseau local (par exemple un serveur Ollama sur une autre machine).'),
    hostsField);
  sync();
  return root;
}

// --------------------------------------------------------------------------- indexation

/**
 * Panneau d'indexation : bouton, progression (SSE puis repli sondage), statistiques.
 * Renvoie {root, destroy}. `destroy()` ferme le flux (à appeler en quittant la vue).
 */
export function indexPanel({ full = false, onDone = null, label = 'Indexer maintenant' } = {}) {
  let stop = null;
  let destroyed = false; // plus aucun suivi ne doit démarrer après destroy()
  let finished = false; // finish() est idempotent (onProgress et onDone peuvent tous deux l'appeler)
  const bar = el('div', { class: 'progress-bar', style: { width: '0%' } });
  const progress = el('div', { class: 'progress hidden' }, bar);
  const line = el('div', { class: 'muted small' });
  const stats = el('div');
  const startBtn = el('button', { class: 'btn btn-primary', type: 'button' }, icon('refresh', 16), label);
  const cancelBtn = el('button', { class: 'btn hidden', type: 'button', onClick: async () => {
    try { await api.post('/api/index/cancel', {}); toast('Arrêt demandé.'); } catch (e) { toastError(e); }
  } }, 'Arrêter');

  const formatStat = (key, value) => {
    if (key === 'duration_s') return `${formatNumber(Math.round(Number(value) * 10) / 10)} s`;
    if (typeof value === 'boolean') return value ? 'oui' : 'non';
    return formatNumber(value);
  };

  /** Dernier run (statut ok | partial | cancelled | failed, stats scalaires, erreurs) et compteurs. */
  const renderStats = (status) => {
    const last = status.last_run || {};
    const counts = status.counts || {};
    const countItems = Object.entries(counts)
      .filter(([, v]) => typeof v === 'number' || typeof v === 'string')
      .map(([k, v]) => [COUNT_LABELS[k] || k, formatNumber(v)]);
    let lastNode = null;
    if (last.status) {
      const runStats = last.stats && typeof last.stats === 'object' ? last.stats : {};
      const errors = Array.isArray(runStats.errors) ? runStats.errors : [];
      const warnings = Array.isArray(runStats.warnings) ? runStats.warnings : [];
      const errorCount = Number(runStats.errors_total) || errors.length;
      const spec = RUN_STATUS[last.status] || { kind: 'info', icon: 'info', title: `Indexation : ${last.status}` };
      const title = last.status === 'partial' && errorCount
        ? `Indexation terminée avec ${formatNumber(errorCount)} erreur${errorCount > 1 ? 's' : ''}`
        : spec.title;
      const statLines = Object.entries(runStats)
        .filter(([k, v]) => STAT_LABELS[k] && (typeof v === 'number' || typeof v === 'boolean') && !(k === 'errors_total' && !v))
        .map(([k, v]) => `${STAT_LABELS[k]} : ${formatStat(k, v)}`);
      lastNode = el('div', { class: `alert alert-${spec.kind}` }, icon(spec.icon),
        el('div', { class: 'alert-body' },
          el('div', { style: { fontWeight: '600' } }, title),
          last.error ? el('div', last.error) : null,
          el('div', { class: 'muted small' }, `Début : ${formatDate(last.started_at)} — fin : ${formatDate(last.finished_at)}`),
          statLines.length ? el('div', { class: 'muted small' }, statLines.join(' · ')) : null,
          errors.length ? collapsible(`${formatNumber(errors.length)} erreur(s) détaillée(s)`, el('div', { class: 'stack', style: { gap: '2px' } },
            errors.slice(0, 50).map((err) => el('div', { class: 'small' },
              err && err.path ? el('span', { class: 'mono' }, err.path, ' ') : null,
              err && err.stage ? badge(STAGE_LABELS[err.stage] || err.stage) : null, ' ',
              err && typeof err === 'object' ? (err.message || '') : String(err))),
            errors.length > 50 ? el('div', { class: 'muted small' }, `… et ${errors.length - 50} autres`) : null), last.status === 'failed') : null,
          warnings.length ? collapsible(`${formatNumber(warnings.length)} avertissement(s)`, el('div', { class: 'stack small', style: { gap: '2px' } },
            warnings.slice(0, 50).map((w) => el('div', typeof w === 'string' ? w : (w && w.message) || JSON.stringify(w))))) : null));
    }
    replace(stats, lastNode, countItems.length ? dl(countItems) : null);
  };

  const onProgress = (status) => {
    if (destroyed || !status) return;
    progress.classList.remove('hidden');
    const total = Number(status.total) || 0;
    const current = Number(status.current) || 0;
    if (total > 0) {
      bar.classList.remove('indeterminate');
      bar.style.width = `${Math.min(100, Math.round((current / total) * 100))}%`;
    } else {
      bar.classList.add('indeterminate');
    }
    const phase = PHASE_LABELS[status.phase] || status.phase || '';
    replace(line, [phase, status.repo ? `dépôt ${status.repo}` : null, total ? `${current}/${total}` : null, status.message || null].filter(Boolean).join(' — '));
    // Un événement `progress` sans champ `running` (IndexProgress brut) n'est pas une fin de run.
    if (status.running === false) finish(status);
  };

  const finish = (status) => {
    if (finished) return;
    finished = true;
    if (stop) stop();
    stop = null;
    startBtn.disabled = false;
    cancelBtn.classList.add('hidden');
    bar.classList.remove('indeterminate');
    bar.style.width = '100%';
    if (status) renderStats(status);
    refreshStatus();
    if (onDone) onDone(status);
  };

  const follow = () => {
    if (destroyed || stop) return;
    finished = false;
    startBtn.disabled = true;
    cancelBtn.classList.remove('hidden');
    progress.classList.remove('hidden');
    bar.classList.add('indeterminate');
    replace(line, 'Démarrage…');
    stop = watchIndex({ onProgress, onDone: finish, onError: (e) => { if (!destroyed) toastError(e); } });
  };

  startBtn.addEventListener('click', async () => {
    try {
      await api.post('/api/index', { full });
      follow();
    } catch (e) {
      if (destroyed) return;
      if (e.status === 409) { toast('Une indexation est déjà en cours : suivi en direct.'); follow(); } else toastError(e);
    }
  });

  // Si une indexation tourne déjà (reprise), on la suit immédiatement.
  api.get('/api/index/status').then((status) => {
    if (destroyed) return;
    if (status.running) follow();
    else renderStats(status);
  }).catch(() => {});

  const root = el('div', { class: 'stack' }, el('div', { class: 'row' }, startBtn, cancelBtn), progress, line, stats);
  return { root, destroy: () => { destroyed = true; if (stop) stop(); stop = null; } };
}

// --------------------------------------------------------------------------- assistant

const STEPS = ['Espace de travail', 'Moteur d\'IA', 'Mode d\'exécution', 'Indexation'];

function stepsBar(active) {
  return el('div', { class: 'steps' }, STEPS.map((label, i) => el('div', { class: `step${i < active ? ' done' : i === active ? ' active' : ''}` },
    el('div', { class: 'step-bar' }), el('span', `${i + 1}. ${label}`))));
}

/** Enregistre le brouillon (création puis mise à jour partielle). */
async function saveDraft(draft, payload) {
  if (!draft.id) {
    const ws = await api.post('/api/workspaces', { name: draft.name || 'Mon projet' });
    draft.id = ws.id;
  }
  const saved = await api.put(`/api/workspaces/${encodeURIComponent(draft.id)}`, payload);
  if (saved && saved.repos) draft.repos = saved.repos;
  return saved;
}

export async function mount(root, route) {
  const status = state.status || (await refreshStatus());
  const existing = status && status.workspace ? status.workspace : null;
  // Reprise : si un espace existe déjà, on pré-remplit l'assistant.
  const draft = existing
    ? { id: existing.id, name: existing.name, repos: (existing.repos || []).map((r) => ({ ...r })), llm: { ...(existing.llm || {}) }, mode: existing.mode || 'strict', allowed_hosts: [...(existing.allowed_hosts || [])] }
    : { id: null, name: '', repos: [], llm: { engine: 'none', base_url: '', model: '', embedding_model: '', api_key: '', num_ctx: 8192, think: false }, mode: 'strict', allowed_hosts: [] };

  let step = Math.max(0, Math.min(3, Number(route.query.step) || 0));
  let panel = null; // composant courant possédant un destroy()
  const page = el('div', { class: 'page page-narrow' });
  root.appendChild(page);

  const render = () => {
    if (panel && panel.destroy) panel.destroy();
    panel = null;
    const error = el('div');
    const content = el('div');
    const nextBtn = el('button', { class: 'btn btn-primary', type: 'button' }, step === 3 ? 'Ouvrir le chat' : 'Continuer');
    const backBtn = el('button', { class: 'btn', type: 'button', disabled: step === 0, onClick: () => { step -= 1; render(); } }, 'Retour');

    const wrap = async (fn) => {
      nextBtn.disabled = true;
      error.replaceChildren();
      try {
        await fn();
      } catch (e) {
        replace(error, errorBox(e));
        toastError(e);
      } finally {
        nextBtn.disabled = false;
      }
    };

    if (step === 0) {
      // L'erreur inline (« Donnez un nom… », « Ajoutez un dépôt… ») disparaît dès que l'utilisateur corrige.
      const nameInput = el('input', { class: 'input', type: 'text', value: draft.name, placeholder: 'ex. Boutique en ligne', onInput: (e) => { draft.name = e.target.value; error.replaceChildren(); } });
      replace(content,
        el('h2', 'Espace de travail et dépôts'),
        el('p', { class: 'muted' }, 'Un espace de travail regroupe les dépôts d\'un même projet (frontend, backend, documentation, base de données…). Develop les lit sans jamais les modifier.'),
        field('Nom de l\'espace de travail', nameInput),
        el('div', { class: 'label', style: { marginBottom: '6px' } }, 'Dépôts du projet'),
        repoEditor(draft.repos, () => error.replaceChildren()));
      nextBtn.onclick = () => wrap(async () => {
        if (!draft.name.trim()) throw new Error('Donnez un nom à l\'espace de travail.');
        if (!draft.repos.length) throw new Error('Ajoutez au moins un dépôt.');
        await saveDraft(draft, { name: draft.name.trim(), repos: draft.repos });
        await refreshStatus();
        step = 1;
        render();
      });
    } else if (step === 1) {
      replace(content,
        el('h2', 'Moteur d\'IA'),
        el('p', { class: 'muted' }, 'Choisissez un modèle local. Sans moteur, la recherche et l\'explorateur fonctionnent, mais pas le chat.'),
        engineForm(draft.llm));
      nextBtn.onclick = () => wrap(async () => {
        if (draft.llm.engine !== 'none' && !draft.llm.model) throw new Error('Indiquez le modèle de chat à utiliser, ou choisissez « Aucun moteur ».');
        await saveDraft(draft, { llm: draft.llm });
        step = 2;
        render();
      });
    } else if (step === 2) {
      replace(content,
        el('h2', 'Mode d\'exécution'),
        el('p', { class: 'muted' }, 'Develop fonctionne entièrement sur votre machine ; ce réglage décide si des services de votre réseau local peuvent être contactés.'),
        modeForm(draft));
      nextBtn.onclick = () => wrap(async () => {
        await saveDraft(draft, { mode: draft.mode, allowed_hosts: draft.allowed_hosts });
        step = 3;
        render();
      });
    } else {
      panel = indexPanel({});
      replace(content,
        el('h2', 'Indexation'),
        el('p', { class: 'muted' }, 'Develop va parcourir les dépôts, analyser le code, les documents et les relations. L\'indexation est incrémentale : les lancements suivants ne traitent que les changements.'),
        el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Récapitulatif'), dl([
          ['Espace', draft.name],
          ['Dépôts', draft.repos.length ? el('div', draft.repos.map((r) => el('div', `${r.name} — ${KIND_LABELS[r.kind] || r.kind} — `, el('span', { class: 'mono muted small' }, r.path)))) : 'aucun'],
          ['Moteur', draft.llm.engine === 'none' ? 'aucun (recherche seule)' : `${ENGINE_LABELS[draft.llm.engine] || draft.llm.engine} — ${draft.llm.model || '?'}${draft.llm.embedding_model ? ` + embeddings ${draft.llm.embedding_model}` : ''}`],
          ['Mode', draft.mode === 'strict' ? 'hors ligne strict' : `réseau local (${draft.allowed_hosts.join(', ') || 'aucun hôte'})`],
        ])),
        el('div', { class: 'card' }, el('div', { class: 'card-title' }, 'Lancer l\'indexation'), panel.root));
      nextBtn.onclick = async () => { await refreshStatus(); navigate('#/chat'); };
    }

    replace(page,
      el('div', { class: 'page-head' }, el('div', el('h1', 'Configuration de Develop'), el('div', { class: 'page-sub' }, existing ? `Espace « ${existing.name} » : reprise de la configuration.` : 'Quatre étapes pour commencer à travailler hors ligne.'))),
      stepsBar(step),
      content,
      error,
      el('div', { class: 'wizard-actions' }, backBtn, nextBtn));
  };

  render();
  return () => { if (panel && panel.destroy) panel.destroy(); };
}
