# Develop — contrat de l'API HTTP locale

Serveur FastAPI lié à `127.0.0.1` (port par défaut 8765). L'interface web statique est servie sur `/`
(`develop/web/index.html`, `/static/*`). Toutes les réponses sont en JSON sauf les flux SSE
(`text/event-stream`, format `event: <type>\ndata: <json>\n\n`). Les erreurs utilisent
`{"error": {"code": "...", "message": "... (en français, actionnable)"}}` avec un statut 4xx/5xx.

Un seul **espace de travail actif** à la fois ; toutes les routes sans `workspace_id` portent sur l'actif.
Si aucun espace n'existe, l'UI ouvre l'assistant de configuration.

## 1. État, espaces de travail, configuration

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/status` | `{version, settings, active_workspace, workspace, engine:{ok,engine,model,error}, index:{running,last_run,counts}, offline:{mode,blocked_count}, needs_setup}` |
| GET | `/api/workspaces` | liste `[WorkspaceConfig]` |
| POST | `/api/workspaces` | `{name}` → crée et active ; renvoie `WorkspaceConfig` |
| GET | `/api/workspaces/{id}` | configuration complète |
| PUT | `/api/workspaces/{id}` | met à jour `name, repos[], llm{}, mode, allowed_hosts[]` (corps partiel accepté) ; reconfigure la garde hors ligne et le client LLM |
| DELETE | `/api/workspaces/{id}` | supprime la mémoire Develop de l'espace (jamais les dépôts) |
| POST | `/api/workspaces/{id}/activate` | rend actif |
| POST | `/api/repos/discover` | `{path}` → `{root:{...stack}, subprojects:[{path,name,suggested_kind,frameworks}]}` |
| POST | `/api/fs/browse` | `{path?}` → `{path, parent, entries:[{name,path,is_git,has_marker}]}` (sélecteur de dossiers côté serveur ; racines = home + volumes) |

`RepoConfig`, `LLMConfig`, `WorkspaceConfig` : voir `develop/config.py`.

## 2. Machine, moteurs et conseiller de modèles

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/system/profile` | `MachineProfile.to_dict()` + `summary` (une ligne) |
| GET | `/api/system/engines` | `EnginesReport.to_dict()` (Ollama, Docker, serveurs compatibles OpenAI) |
| GET | `/api/system/recommendations` | `{profile_summary, tier, recommendations:[{tag, category, size_gb, verdict:'recommended'|'possible_slow'|'too_big', reason, pull_command, installed}], embedding:[...], install_steps:[{title, commands[], notes}], warnings[]}` |
| POST | `/api/llm/test` | `{engine, base_url, model, embedding_model, api_key}` → `{ok, engine_status, model_info:{context_length, supports_tools}, embedding_ok, embedding_dim, suggested_num_ctx, expected_first_token_s, prefill_tps, error}` — `suggested_num_ctx` = min(fenêtre du modèle, 16 384 pour un modèle ≥ 20B sinon 32 768, plafond RAM de la machine) : jamais la fenêtre entière d'un modèle 128k ; `prefill_tps` est la vitesse de lecture du prompt (tokens/s) mesurée par le chat pour ce modèle, sinon estimée d'après sa taille et la machine ; `expected_first_token_s` le délai attendu avant la première lettre avec ce `num_ctx` (prompt maximal du budget ÷ `prefill_tps`) |
| GET | `/api/llm/models?engine=&base_url=` | `[ModelInfo]` sans sauvegarder la configuration |
| GET | `/api/llm/state` | `{engine, model, embedding_model, known, loaded, embedding_loaded, loading, loading_since, last_load_s, error, num_ctx, loaded_num_ctx, reload_pending}` : le modèle de chat configuré est-il en mémoire du moteur (Ollama : `/api/ps`), un préchauffage est-il en cours (`loading`, depuis `loading_since` en secondes epoch), durée du dernier chargement réussi (`last_load_s`) ; `known: false` pour un moteur qui ne dit pas ce qu'il a en mémoire (serveur compatible OpenAI, aucun moteur) : `loaded` vaut alors `true` et l'interface ne bloque rien ; `num_ctx` est la fenêtre que le chat demande au moteur (min(fenêtre du modèle, configuration, plafond RAM), même calcul que `done.budget.num_ctx`), `loaded_num_ctx` celle avec laquelle le moteur a chargé le modèle (`context_length` de `/api/ps`, `null` s'il ne la dit pas) et `reload_pending` vaut `true` quand elles diffèrent : le modèle est en mémoire mais Ollama le rechargera au prochain message (bandeau « chargé avec 32 768 ≠ 40 960 : rechargement au prochain message », bouton « Recharger ») |
| POST | `/api/llm/load` | lance (ou rejoint) le préchauffage du modèle de chat puis du modèle d'embeddings en tâche de fond ; le modèle est chargé avec la fenêtre `num_ctx` que le chat demandera (Ollama recharge un modèle dès que ses options diffèrent : sans cela, le premier message payait un second chargement silencieux) ; 202 `{loading: true, …state}` ; 400 `no_engine` sans moteur ou modèle configuré |
| POST | `/api/llm/unload` | décharge le modèle de chat et le modèle d'embeddings (Ollama : `POST /api/generate` avec `keep_alive: 0`) ; interrompt un préchauffage en cours ; `{loaded: false, unloaded: [noms], …state}` ; 400 `unload_unsupported` pour un moteur sans cette information |
| GET | `/api/llm/events` | SSE : un événement `state` (même forme que `/api/llm/state`) par seconde tant qu'un chargement est en cours, puis un dernier `state` (`loading: false`) et fin du flux ; sans chargement, un seul `state` puis fin |
| GET | `/api/offline/blocked` | `[BlockedAttempt]` ; DELETE vide le journal |

Le préchauffage est aussi lancé au démarrage de `develop serve` et après chaque changement de moteur ou de
modèle (`PUT /api/workspaces/{id}`, activation d'un espace) ; un seul chargement à la fois, un second appel
rejoint celui en cours. L'interface du chat n'autorise l'envoi (hors mode « Recherche seule ») qu'une fois
`loaded` vrai et `loading` faux.

## 3. Indexation

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/api/index` | `{full?: bool, repo_id?: str}` → démarre en arrière-plan ; `{run_id}` ; 409 si déjà en cours |
| GET | `/api/index/status` | `{running, run_id, phase ('scan'|'analyze'|'documents'|'relations'|'embeddings'|'done'), current, total, repo, message, last_run:{status, started_at, finished_at, stats, error}, counts}` |
| GET | `/api/index/events` | SSE : `progress` (même forme que status), `done`, `error` |
| POST | `/api/index/cancel` | demande l'arrêt |

Incrémental par défaut : seuls les fichiers nouveaux/modifiés/supprimés (sha1+mtime+taille) sont retraités ;
`full=true` réinitialise l'index de l'espace (pas les notes/documents/conversations).

## 4. Explorateur et recherche

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/repos` | `[{...RepoConfig, counts:{files,symbols,endpoints,models}, head_sha, last_indexed_at, git:{branch, dirty}}]` |
| GET | `/api/files?repo_id=&q=&language=&kind=&limit=` | liste plate `[{id, repo_id, path, language, kind, size}]` (l'UI construit l'arbre) |
| GET | `/api/files/{id}` | fichier + `symbols[]`, `endpoints[]`, `models[]`, `imports[]`, `relations:{out[], in[]}`, `documents_mentioning[]` |
| GET | `/api/files/{id}/content?start=&end=` | `{path, language, start, end, total_lines, text}` (max 2000 lignes par appel, lecture seule) |
| GET | `/api/symbols?q=&kind=&repo_id=&limit=` | `[{id, name, kind, parent, signature, file:{id,path,repo_id}, line_start, line_end}]` |
| GET | `/api/symbols/{id}` | symbole + fichier + `callers[]` (relations entrantes) + `uses[]` + `snippet` |
| GET | `/api/endpoints?side=&q=` | `[{id, side, method, path, normalized_path, handler, framework, file, line, matches:[{endpoint_id, side, path, handler, file, confidence}]}]` |
| GET | `/api/models?q=` | `[{id, name, table_name, kind, file, line, field_count}]` |
| GET | `/api/models/{id}` | modèle + `fields[]`, `relations[]`, `linked:[{model_id, kind, name, file}]` (migration ↔ eloquent ↔ table SQL ↔ interface TS), `endpoints_using[]`, `documents_mentioning:[{id, kind:'document'|'note', title, relation, confidence, via}]` — toute l'entité : `via` est `null` si le document cite ce modèle, sinon `{node, type, label}` du noeud intermédiaire (table définie ou modèle `same_entity`) |
| GET | `/api/graph/neighbors?node=file:12&depth=1&limit=200` | `{nodes:[{key, type, label, file, line}], edges:[{src, dst, kind, confidence}]}` |
| GET | `/api/search?q=&types=file,document,note&limit=20` | `{query, identifiers:[...], results:[SearchHit]}` |

`SearchHit = {chunk_id, source_type ('file'|'document'|'note'|'symbol'|'endpoint'|'model'), source_id, title, path, line_start, line_end, page, snippet, score, channels:['identifier','route','lexical','semantic','graph','temporal'], reason, extra}` (dans l'événement `sources` du chat, un passage relu pour une question de suivi porte le canal `followup` et la raison « suite de la conversation : §… de « … » ») ; un hit `endpoint` porte dans `extra` `handler`, `framework`, `method`, `route`, `normalized_path` et, si une classe de validation existe dans l'index, `request_class` / `request_class_id` / `request_file_id` ; un hit du canal `route` porte `route_terms` et `route_strong`.

`query = {text, identifiers, paths, routes, tables, intent, date_from, date_to, keywords, lexical_query, expansions, previous_terms}` : `expansions` sont les termes du code déduits du vocabulaire de la question (« authentification » → `auth`, `login`…), `previous_terms` les mots-clés du tour précédent quand le chat a enrichi une question de suivi du marqueur `[contexte précédent : …]` (l'intention et les dates viennent toujours de la question seule).

## 5. Documents et journal

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/documents` | `[{id, title, format, kind, size, added_at, updated_at, chunk_count, stale:bool}]` |
| POST | `/api/documents` | multipart `file` (upload) **ou** JSON `{path}` (import d'un fichier local, copié dans `documents/`) → document indexé |
| GET | `/api/documents/{id}` | métadonnées + `chunks_preview[]` + `mentions:[{kind:'symbol'|'table'|'endpoint', name, exists:bool}]` (détection de dérive) |
| GET | `/api/documents/{id}/content` | texte extrait |
| DELETE | `/api/documents/{id}` | supprime |
| GET | `/api/notes?from=&to=&tag=&q=&limit=` | `[{id, title, note_date, tags[], status, excerpt}]` triées par date décroissante |
| POST | `/api/notes` | `{title, body, tags[], note_date (AAAA-MM-JJ, défaut aujourd'hui), status ('valid'|'superseded'|'historical'), supersedes?: id}` |
| GET / PUT / DELETE | `/api/notes/{id}` | note complète ; la mise à jour réindexe |

## 6. Conversation (chat)

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/conversations` | `[{id, title, created_at, updated_at, message_count}]` |
| POST | `/api/conversations` | `{title?}` → `{id}` |
| GET | `/api/conversations/{id}` | `{id, title, messages:[{id, role, content, sources[], created_at}]}` |
| PATCH | `/api/conversations/{id}` | `{title}` |
| DELETE | `/api/conversations/{id}` | supprime |
| POST | `/api/conversations/{id}/messages` | `{content, mode:'auto'|'search_only'|'no_tools', max_context_tokens?}` → **SSE** |

Événements SSE du chat, dans l'ordre :
1. `status` `{text}` — avant la recherche, un seul : « Recherche dans le projet… » (question ordinaire) ; « Question de suivi : contexte « … » conservé » (question courte ou sans identifiant : la recherche reprend les termes du tour précédent, cités entre les guillemets, et relit ses sources) ; « Petite conversation : réponse directe, sans recherche ni modèle. » (salutation, remerciement… : `sources` vide, puis `delta` et `done` sans appel au LLM). Après `sources`, avant l'appel du modèle (jamais en `search_only`) : « Chargement du modèle <modèle> en mémoire (jusqu'à 30 s la première fois)… » si le moteur dit ne pas l'avoir en mémoire (Ollama : `/api/ps`, liste vide comprise — keep_alive expiré, après `POST /api/llm/unload`, `develop ask` sans préchauffage ; rien quand le moteur ne le dit pas : serveur compatible OpenAI, Ollama injoignable), puis toujours « Le modèle lit 9 sources (≈ 3 200 tokens, ~20 s estimées) avant de répondre… » — le délai est le prompt entier (consignes, historique, question, sources) divisé par la vitesse de lecture `budget.prefill_tps`. La lecture d'un fichier par un outil est rapportée dans `tool_result.summary` (« Lecture de OrderService.php. »), pas dans `status`.
2. `sources` `{items:[Source]}` — contexte sélectionné avant la génération ; `Source = {ref:'S1', source_type, path|title, line_start, line_end, page, repo_id, file_id, document_id, note_id, symbol_id, endpoint_id, model_id, score, channels, reason}`. Les **fiches structurées** placées en tête du contexte (modèle : champs, migrations, règles de validation ; route ; classe), synthétisées depuis l'index, sont des sources comme les autres, avec `channels:['card']`, `reason` « fiche structurée (synthèse de l'index) » et `score` 0 ; un passage relu pour une question de suivi porte le canal `followup` (§4).
3. `tool_call` `{name, arguments}` / `tool_result` `{name, summary, added_sources:[Source]}` — tours d'outils (max 4)
4. `delta` `{text}` — fragments de la réponse
5. `done` `{message_id, usage:{prompt_tokens, completion_tokens, load_duration_ms}, budget, truncated_sources:[Source], invalid_citations:['S9'…], cancelled, mode}` — `budget = {num_ctx, context_tokens_used, max_context_tokens, sources_tokens, tools_tokens_used, tools_tokens_max, tool_rounds, answer_reserve, tools_mode:'native'|'text'|'none', source, prefill_tps, first_token_s, latency_capped}` : `context_tokens_used` / `max_context_tokens` comptent les passages retenus par la recherche initiale et leur budget (pied de message « contexte 2 758 / 2 758 tokens — 7 sources sur 23 retenues ») ; `tools_tokens_used` / `tools_tokens_max` et `tool_rounds` ce que les tours d'outils ont consommé et ce que la fenêtre leur laissait (« Outils : 1 542 / 3 765 tokens (1 tour) », 0 sans outil) ; `sources_tokens` tous les blocs de sources citables, outils compris ; `answer_reserve` la réserve de réponse ; `tools_mode` le mode d'outils effectif (`none` quand la fenêtre ne porte pas les outils) ; `source` l'explication en français du calcul de `num_ctx` et du contexte (info-bulle du pied), dont la borne de latence : « contexte ≤ 8 000 (…), plafonné à 3 200 tokens pour rester sous 25 s (lecture estimée 60 tokens/s) » ; `prefill_tps` la vitesse de lecture du prompt (tokens/s) retenue pour ce modèle après ce tour — moyenne mobile des 5 dernières mesures du chat (`usage.prompt_tokens` ÷ délai jusqu'au premier fragment de texte du premier appel ; pas de mesure quand le modèle se chargeait (`/api/ps` ne le listait pas avant l'appel, liste vide comprise, ou Ollama rapporte après coup un `load_duration` de plus de 1 s : keep_alive expiré, autre `num_ctx`), réfléchissait, ou a commencé par un appel d'outil — Ollama ne l'émet qu'une fois son JSON entièrement généré, le délai mêlerait lecture et génération ; persistée par modèle dans la mémoire de l'espace), sinon l'estimation d'après la taille du modèle et la machine ; `first_token_s` ce délai mesuré sur ce tour (`null` sans mesure : recherche seule, flux coupé avant le premier fragment) ; `latency_capped` vrai quand cette borne a réduit le contexte (objectif `LLMConfig.first_token_target_s`, 25 s par défaut, plancher 1 500 tokens). `usage` est vide sans LLM ; `usage.load_duration_ms` (Ollama seulement) est le temps passé à (re)charger le modèle pendant ce tour, en millisecondes — ≈ 0 s'il était déjà en mémoire ; `invalid_citations` liste les `[Sn]` cités par le modèle sans source correspondante (le texte est conservé tel quel) ; `cancelled` vaut `true` quand le client a interrompu le flux (le texte partiel est enregistré s'il n'est pas vide) ; `mode` reprend celui de la requête.
6. `error` `{code, message}` — ex. `llm_unavailable` (la recherche a tout de même renvoyé `sources`).

La réponse cite `[S1]`, `[S2]`… ; l'UI remplace chaque citation par un lien vers la source (fichier:ligne, document, note).
`mode='search_only'` renvoie `sources` puis `done` sans appeler le LLM (fonctionne sans moteur).

## 7. Analyse d'impact et Git

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/api/impact` | `{source:'git'|'file'|'symbol', repo_id?, rev_range?, path?, symbol_id?, max_hops?: 2}` → `ImpactReport` |
| GET | `/api/git/status?repo_id=` | `{branch, head, changes:[ChangedFile]}` |
| GET | `/api/git/log?repo_id=&limit=&path=` | `[CommitInfo]` |

`ImpactReport = {changed:[{path, status, repo_id, symbols_changed:[...]}], affected:[{node, type, name, path, line, repo_id, reason, confidence (0..1), hops}], endpoints:[...], models:[...], frontend_calls:[...], documents:[...], notes:[...], stats:{...}, warnings:[...]}`.
Chaque élément porte une `reason` lisible : « importe src/api/client.ts », « appelle POST /orders/:param/confirm », « documente la table orders ».

## 8. Conventions

- Tous les chemins renvoyés sont relatifs à la racine du dépôt (`repo_id` + `path`), séparateur `/`.
- Pagination simple par `limit` (+ `offset` quand utile) ; tri déterministe.
- Le serveur n'écrit jamais dans les dépôts configurés.
- Les longues opérations (indexation, chat) sont asynchrones ou en flux ; aucune route ne bloque plus de quelques secondes.

## 9. Précisions d'implémentation (v1)

- `GET /api/files/{id}/content` renvoie `404 file_unreadable` si le fichier a disparu ou n'est pas lisible.
- `GET /api/index/events` envoie d'abord l'état courant ; sans indexation en cours, le flux se termine aussitôt (`running=false`), sinon après `done` ou `error`.
- Notes : `status` et `supersedes` sont stockés dans `tags` sous la forme `status:<valeur>` et `supersedes:<id>`, filtrés des `tags` renvoyés ; créer une note avec `supersedes` passe l'ancienne en `superseded`.
- `POST /api/documents` indexe immédiatement les morceaux (FTS) ; les relations `documented_by` et les embeddings sont produits au prochain run d'indexation. `mentions` et `stale` sont calculés à la volée.
- `ImpactReport.documents` peut contenir des éléments de `type: 'file'` (fichiers de documentation des dépôts, `kind: 'doc'`), en plus des documents déposés et des notes.
- `stats.linker` d'un run contient `LinkStats` (dont `handlers_linked`, `calls_linked`) ; `stats.relations` inclut les relations dérivées recalculées à chaque run.
- Chat SSE : la déconnexion du client est détectée entre deux événements ; la réponse partielle est enregistrée.
- `LLMConfig.think` (défaut `false`) : réflexion interne des modèles qui la supportent ; transmise à Ollama (`think`) uniquement quand `model_info.capabilities` contient `thinking`, ignorée par les serveurs compatibles OpenAI.
- `LLMConfig.first_token_target_s` (défaut `25.0`, `0` = pas d'objectif) : objectif de délai avant la première lettre de la réponse. Le contexte récupéré ne dépasse jamais `prefill_tps × first_token_target_s − consignes − question − historique` (plancher 1 500 tokens, plafond le contexte calculé par la fenêtre) ; `prefill_tps` est mesuré par le chat (meta `llm_prefill_tps:<modèle>` de la base de l'espace, moyenne mobile sur 5 mesures, bornée entre 40 et 2 000 tokens/s) ou, avant toute mesure, estimé ≈ 6 × la vitesse de génération du conseiller (sans bande passante connue : 150 tokens/s sous 10B, 100 jusqu'à 19B, 60 à partir de 20B).
