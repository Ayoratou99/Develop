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
| POST | `/api/llm/test` | `{engine, base_url, model, embedding_model, api_key}` → `{ok, engine_status, model_info:{context_length, supports_tools}, embedding_ok, embedding_dim, suggested_num_ctx, error}` |
| GET | `/api/llm/models?engine=&base_url=` | `[ModelInfo]` sans sauvegarder la configuration |
| GET | `/api/offline/blocked` | `[BlockedAttempt]` ; DELETE vide le journal |

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

`SearchHit = {chunk_id, source_type ('file'|'document'|'note'|'symbol'|'endpoint'|'model'), source_id, title, path, line_start, line_end, page, snippet, score, channels:['lexical','semantic','identifier','graph','temporal'], reason}`.

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
1. `status` `{text}` — « Recherche dans le projet… », « Lecture de OrderService.php… »
2. `sources` `{items:[Source]}` — contexte sélectionné avant la génération ; `Source = {ref:'S1', source_type, path|title, line_start, line_end, page, repo_id, file_id, document_id, note_id, symbol_id, score, channels, reason}`
3. `tool_call` `{name, arguments}` / `tool_result` `{name, summary, added_sources:[Source]}` — tours d'outils (max 4)
4. `delta` `{text}` — fragments de la réponse
5. `done` `{message_id, usage:{prompt_tokens, completion_tokens}, budget:{num_ctx, context_tokens_used, max_context_tokens}, truncated_sources:[...]}`
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
