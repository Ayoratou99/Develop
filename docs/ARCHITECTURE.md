# Develop — Document de décisions d'architecture (v1)

> **État d'implémentation (2026-10-15).** Ce document est la synthèse du panel de conception. Il a guidé la
> construction, mais le code est la référence quand ils divergent. Écarts assumés en v1 :
> le schéma SQLite implémenté est `develop/storage/schema.sql` (v1 : embeddings clés par `chunk_id`, pas de colonne
> `terms` ; les identifiants sont retrouvés par le canal « identifiants exacts » de la recherche et par l'en-tête de
> chaque morceau de code) ; l'upload de documents reste en `multipart` (`python-multipart` conservé) ; la garde
> socket de niveau 2 et les `link_overrides` sont reportés en v1.5 ; la boucle d'outils bornée est en v1.
> Le conseiller de modèles (`develop/llm/advisor.py`, catalogue `develop/llm/catalog.json`) est en v1.


Statut : tranché. Cible : Python 3.9.6 système, macOS/Linux/Windows, SQLite 3.54 + FTS5, dépendances de base `fastapi uvicorn httpx pydantic>=2 numpy<2.1 pypdf`. Le dépôt contient déjà ~5 800 lignes (`config.py`, `storage/db.py` + `schema.sql`, `llm/*` dont `offline_guard.py`, `project_scanner/*`, `document_indexer/*`, `code_analysis/model.py` + `registry.py`, `git_tools/repo.py`, `system/profile.py` + `engines.py`, `web/static/{app,api,ui}.js`, `docs/API.md`, fixture `tests/fixtures/sample_project`) : **on construit dessus, on ne repart pas de zéro**. Le vocabulaire du code reste `workspace` (config, API) ; l'UI affiche « projet ».

## 1. Vue d'ensemble

Trois couches, trois packages racines, aucune dépendance inversée :

1. **Source** (lecture seule, jamais modifiée) : `project_scanner`, `git_tools`, `code_analysis`, `document_indexer`. Produit des faits purs (dataclasses) sans toucher à la base.
2. **Mémoire** (`~/.develop/workspaces/<id>/memory.sqlite3`) : `storage` (schéma v2), `indexing` (pipeline incrémental, linker, documents, notes, historique git), `search` (identifiants exacts + FTS5/BM25 + cosinus numpy + structurel + temporel, fusion RRF), `impact`.
3. **Moteur IA** (remplaçable sans reconstruire la mémoire) : `llm` (clients Ollama / OpenAI-compatible / Null derrière `LLMClient`, garde hors ligne, embeddings), `rag` (fiches, budget, contexte, citations), `advisor` (profil machine + catalogue + conseiller).

`server` (FastAPI, 127.0.0.1:8765) et `cli` orchestrent ; `web/` est une SPA statique vanilla servie par FastAPI.

Flux principal d'une question : `rag.chat` → détection des noms connus → **fiches** (SQL seul) → `search.hybrid` (6 canaux, RRF) → `rag.context` (budget calculé) → `LLMClient.chat_stream` → SSE `status, cards, sources, delta*, done` → persistance `messages`. Sans LLM configuré : mêmes événements, `delta` = gabarit déterministe « Aucun moteur configuré — éléments trouvés : … ».

Principes non négociables :
- **Lecture seule** : aucune écriture sous un dépôt ; git appelé avec `GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0`, sous-commandes locales uniquement, timeout.
- **Hors ligne** : un seul point de sortie HTTP (`llm.offline_guard.make_async_client`), `trust_env=False`, mode strict = loopback littéral sans résolution DNS, garde socket processus, CSP `'self'`, tout blocage journalisé et visible.
- **Dégradation sans LLM / sans embeddings** : index, explorateur, recherche BM25, fiches, impact, documents, notes fonctionnent ; une erreur par fichier est une ligne `index_errors`, une erreur de service est un état dans `/api/status`, jamais un crash.
- **Python 3.9** : `from __future__ import annotations` partout, `Optional/Union/List/Dict` dans tout ce que pydantic évalue, pas de `match`, pas de `dataclass(slots=)`, pas de `tomllib` (config JSON), `fromisoformat` sans `Z` (dates = `REAL` epoch, `note_date` = `YYYY-MM-DD`).

## 2. Décisions Q1..Q12

**Q1 Interface.** SPA sans build : `web/index.html` coquille + modules ES natifs (`app.js` routeur par hash et état unique, `api.js`, `ui.js` composants + Markdown sûr rendu par `createElement/textContent`, `views/<écran>.js` exportant `mount(root, route) -> unmount`). Chat en « SSE sur fetch » : `POST /api/conversations/{id}/messages` → `text/event-stream` lu par `ReadableStream`, `AbortController` côté client, `request.is_disconnected()` testé à chaque delta côté serveur (ferme le flux httpx amont), `: ping` toutes les 15 s. Progression d'index via `EventSource` natif. Sécurité locale : bind 127.0.0.1, CSP `default-src 'self'; connect-src 'self'; img-src 'self' data:`, `Referrer-Policy: no-referrer`, vérification `Host` loopback et `Sec-Fetch-Site: same-origin` (ou `Origin` identique) sur toute requête mutante et sur `/api/fs/browse` (anti-CSRF/DNS-rebinding, point manquant chez « robust »). localStorage = préférences seulement. Rejeté : WebSocket, EventSource pour le chat (GET sans corps), Jinja, toute bibliothèque JS vendorisée, `/api/open-in-editor` (v1.5, désactivé par défaut).

**Q2 Stockage.** `~/.develop/` (`DEVELOP_HOME`) : `settings.json`, `logs/`, `workspaces/<id>/{workspace.json, memory.sqlite3, documents/}` (disposition déjà codée dans `config.py`). Une base par workspace, WAL, `synchronous=NORMAL`, `foreign_keys=ON`, `busy_timeout=5000`, une connexion par thread, **un verrou d'écriture processus** (`Database.write_lock`) et transactions courtes (lots de 100 fichiers). Schéma v2 (section 4) : ids de fichiers stables, `relations` à clés textuelles `type:id` (existant, conservé) + `origin_file_id` pour l'invalidation, `chunks_fts(title, text, terms)` avec `terms` = identifiants éclatés (vérifié : sans cette colonne « order controller » ne trouve jamais `OrderController`), `embeddings` clés par `(text_sha1, model) WITHOUT ROWID` (dédup gratuite, plusieurs modèles côte à côte — remplace la PK `chunk_id` actuelle), matrice numpy en RAM invalidée par `meta.index_version`, `quick_check` + espace disque au démarrage, migrations par `meta.schema_version` (pré-release : une base v1 est renommée `.v1-backup` et recréée). Rejeté : `.develop/` dans les dépôts, sqlite-vec/faiss, tables d'arêtes spécialisées, table `refs` de toutes les occurrences (explose sur un monorepo), table `fields` séparée (les champs restent en JSON dans `data_models.fields`).

**Q3 Extraction.** Protocole existant `Extractor.analyze(text, path) -> FileAnalysis` (pur, sans base), registre par langage, `try/except` par fichier → `files.status='error'` + ligne `index_errors` (le chunking générique et la FTS continuent). Socle commun `code_analysis/base.py` : vue masquée (commentaires et contenus de chaînes remplacés par des espaces à longueur constante, table des littéraux), `BraceScanner`, `split_identifier`, `enclosing_symbol`, lignes > 5 000 caractères tronquées, fichiers > 1 Mo ou ligne moyenne > 400 → `status='large'`/`'minified'` sans symboles. `EXTRACTOR_VERSION` par module, stocké sur `files` → réextraction ciblée. Ordre v1 : Python (`ast` : FastAPI/Flask/Django, SQLAlchemy/Django/pydantic/dataclass), JS/TS/JSX/TSX (fonctions/classes/interfaces TS avec champs, composants React, imports résolus via `tsconfig.paths`, Express avec propagation des préfixes `app.use` par le graphe d'imports, NestJS, `fetch`/`axios`/`this.http`), PHP/Laravel (routes `Route::*`, `resource`/`apiResource` expansés, groupes `prefix/controller/group` par pile fermée au `BraceScanner`, préfixe `/api` pour `routes/api.php` lu dans `bootstrap/app.php` sinon défaut, modèles `$table/$fillable/$casts` + relations, contrôleurs, migrations `Schema::create/table` → modèle `migration`, PSR-4), SQL DDL (`CREATE TABLE`, `ALTER`, `FOREIGN KEY`), puis Java/Spring et Dart/Flutter en « basique » (annotations, `fromJson`, Dio/http). Dictionnaire de données : tableaux Markdown/CSV reconnus par `document_indexer.loaders.looks_like_data_dictionary` → `data_models(kind='dictionary')` avec descriptions et alias. OpenAPI JSON si présent → endpoints confiance 1.0. Rejeté : tree-sitter obligatoire (extra v2 derrière le même protocole), LLM pour extraire, `ProcessPoolExecutor` (spawn macOS, pickling ; threads suffisent, option ultérieure), Prisma/TypeORM/Mongoose/Next (v1.5).

**Q4 Liaison inter-dépôts.** `code_analysis.model.normalize_route_path` (existant, forme `:param`) appliqué symétriquement : côté serveur après composition des préfixes, côté client après retrait de l'hôte, de la query et des bases (`${API_BASE}`, `$baseUrl`). Préfixes de base appris par dépôt (`.env*` lus pour les URL seulement, `axios.create({baseURL})`, `import.meta.env`/`process.env`, proxy `vite.config`, `Dio(BaseOptions)`) → `repos.base_prefixes`. Scores : exact chemin + méthode 1.0 ; méthode inconnue 0.9 ; préfixe connu 0.85 ; préfixe générique `/api|/api/v1|/v1` 0.75 ; suffixe ≥ 2 segments 0.65 ; `:param` absorbant un littéral 0.5 ; k candidats ex æquo → ÷√k ; 3 candidats max ≥ 0.4 ; **méthode contradictoire → aucune arête** (tranché contre le ×0.5 de « mvp » : une méthode différente est le plus souvent une autre route ; l'appel apparaît dans « non liés »). Arêtes `calls` dérivées (origin NULL) reconstruites globalement en fin de run, `evidence` lisible. `link_overrides` à clés stables (`repo:path:METHOD:/chemin/:param`) réappliquées après chaque relink, boutons Confirmer/Rejeter dès la v1. Même mécanique : `model_table` (`$table`/`@Table` prioritaires, sinon snake pluriel), `handles` (endpoint → symbole `Controller.method`), `same_shape` (interface TS / classe Dart ↔ modèle : nom singulier/pluriel/alias + Jaccard des champs ≥ 0.5). Rejeté : OpenAPI obligatoire, `route:list`, appariement par LLM.

**Q5 Recherche et embeddings.** Chunking : code = un chunk par symbole de haut niveau (`document_indexer.chunking.chunk_for_file` existant, mode `code`), classe > 120 lignes scindée par méthode, fenêtres 60/10 sinon, ligne d'en-tête `file: repo/chemin lines a-b symbol: X`, plafond 2 000 caractères ; documents par titres puis ~1 200 caractères / 150 de recouvrement, PDF par page, CSV en-tête + 50 lignes, SQL par instruction, note = une unité (< 1 500 car.). Six canaux (spec §2) fusionnés par RRF k=60 : identifiants exacts (NOCASE sur `symbols.name/qualname`, `endpoints.normalized_path`, `data_models.name/table_name`, `files.path`), routes (endpoints dont le chemin ou le handler contient un terme de la question ou de son expansion de vocabulaire, « authentification » → `login`, `AuthController` ; preuves fortes d'abord, route serveur avant appel client ; un terme présent dans trop de routes ne fait que départager), lexical (FTS5, `build_fts_query` existant : termes cités, préfixe `*` ≥ 4 car., AND puis OR, `bm25(chunks_fts, 3.0, 1.0, 0.7)`), sémantique (cosinus numpy si modèle configuré, sinon ignoré), structurel (1 saut dans `relations` depuis les nœuds trouvés), temporel (notes : `since/until/tag`, boost récence 90 j quand la question parle de décision/réunion). Dédup par fichier + lignes chevauchantes, max 3 passages par fichier, au moins un passage par canal ayant répondu. Embeddings : passe séparée, résumable (chunks dont `sha1` n'a pas de ligne pour le modèle courant), lots de 32, priorité notes > documents > code, Ollama `/api/embed` avec repli `/api/embeddings`, OpenAI `/v1/embeddings`, préfixes `search_query:`/`search_document:` pour nomic ; changement de modèle = recalcul explicite, couverture affichée « vecteurs : 62 % (nomic-embed-text) », purge de l'ancien modèle après ≥ 95 %. Rejeté : mélange pondéré de scores, ANN, trigram FTS, reranker, recalcul automatique.

**Q6 LLM et RAG.** `llm.base.LLMClient` existant (`health`, `list_models`, `model_info`, `chat_stream`, `embed`) avec `OllamaClient`, `OpenAICompatClient` (couvre Docker/LM Studio/vLLM/llama.cpp : « Docker » n'est qu'une `base_url`), `NullClient`. `num_ctx` **toujours** envoyé = `min(context_length via /api/show, config.num_ctx, plafond RAM du profil)` (plafond : 8k < 16 Go, 16k < 32 Go, 32k au-delà ; `POST /api/llm/test` ne suggère jamais la fenêtre entière : min(fenêtre, 16k dès 20B sinon 32k, plafond RAM), avec `prefill_tps` et `expected_first_token_s`). Budget (`chat/budget.py`) : consignes système = gabarit réellement mesuré + 10 % de marge (plafond 1,5k ; définitions d'outils natifs en sus), historique = min(max(768, 10 % de `num_ctx`), 2k), réserve réponse = clamp(20 % de `num_ctx`, 1k, 3k), contexte récupéré = 80 % du reste (90 % impact/large) soit ≈ 3,3k tokens à 8k, ≈ 8k à 16k, ≈ 20k à 32k (factuel, outils natifs), plafonné par `max_context_chars` (défaut 96 000 caractères ≈ 24k tokens ; l'ancien défaut 24 000 est relu comme le nouveau) ; comptage ≈ caractères/4 (code /3,5). **Borne de latence** : la fenêtre n'est pas le seul plafond, le temps de lecture l'est aussi — première lettre attendue en `LLMConfig.first_token_target_s` (25 s), donc contexte ≤ `prefill_tps × 25 − consignes − question − historique` (plancher 1,5k tokens, `Budget.latency_capped`, expliqué dans `source`). `prefill_tps` (tokens/s de préremplissage) est **mesuré** par l'orchestrateur sur le premier appel de chaque question (`usage.prompt_tokens` ÷ délai jusqu'au premier fragment ; pas pendant un chargement du modèle ni avec la réflexion interne ; prompt ≥ 256 tokens), moyenne mobile sur 5 mesures persistée par modèle (`meta llm_prefill_tps:<modèle>`, bornée [40, 2000]) et exposée dans `done.budget` (`prefill_tps`, `first_token_s`) ; avant toute mesure il est **estimé** par `advisor.estimate_prefill_tps` ≈ 6 × la vitesse de génération (bande passante ÷ taille), repli par taille (150 < 10B, 100 < 20B, 60 au-delà). Le `status` « Le modèle lit N sources (≈ T tokens, ~S s estimées)… » annonce ce délai (prompt entier ÷ `prefill_tps`). Fiches compactes en tête, puis passages `[S1] repo/chemin:12-48 (symbole)` en fence, `sources` émis avant le premier `delta`, citations invalides retirées en fin de flux et `cited` calculé, `messages.status='partial'` si flux coupé. Prompt système FR/EN : le contexte est de la donnée (jamais des instructions), citer `[Sn]`, dire quand l'information manque, ne jamais inventer champ/route/fichier. Historique = derniers tours sans leurs sources ; question courte de suivi → requête enrichie des noms de la question précédente. `api_key` lue depuis une variable d'environnement (`api_key_env`), jamais écrite en clair. Boucle d'outils bornée (4 tours, `ToolSpec`/`parse_text_tool_calls` déjà codés, événements `tool_call/tool_result` réservés) : **v1.5** ; synthèses hiérarchiques : v2 (table réservée). Rejeté : SDK openai/ollama, LangChain, tiktoken.

**Q7 Garde hors ligne.** `llm/offline_guard.py` (existant) reste l'unique fabrique de clients httpx (`trust_env=False`, `follow_redirects=False`, timeouts connect 3 s / lecture 120 s / flux 600 s). Modifications : **strict = loopback littéral seulement** (`127.0.0.0/8`, `::1`, `localhost`, `*.localhost`) — `host.docker.internal` et `allowed_hosts` ne valent qu'en `local_network` (tranché contre « ux » qui mélangeait les deux modes) ; `local_network` = loopback + RFC1918/ULA/lien-local + `.local` + `allowed_hosts`, nom résolu par `getaddrinfo` et refusé si une adresse est publique. Refus = `OfflineViolation` + ligne `network_log` (base) + tampon mémoire + compteur `/api/status` + événement SSE `blocked` + badge ; accès autorisés non-loopback journalisés aussi. Niveau 2 (strict, par défaut, `socket_guard: true`) : `socket_guard.install()` enveloppe `socket.socket.connect/connect_ex` et `socket.getaddrinfo` (loopback, `localhost`, AF_UNIX autorisés ; uvicorn ne fait que bind/listen, vérifié par test) + `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY=*`. Niveau 3 : CSP, test grep (aucun `httpx.Client/AsyncClient`, `requests`, `urllib.request`, `aiohttp` hors `offline_guard.py` ; aucune URL externe dans `web/`), `develop status --verify-offline` et `POST /api/network/verify-offline` (tentative volontaire vers `example.com` qui doit être bloquée et journalisée), « Tester l'accès » refuse une `base_url` interdite sans la contacter.

**Q8 Indexation incrémentale.** `project_scanner.scan_repo` existant : `git ls-files -z --cached --others --exclude-standard` (sans `--recurse-submodules`, incompatible avec `--others` ; un sous-module = dépôt séparé), sinon `scandir` sans symlinks + exclusions + `.developignore` ; NUL dans 8 Ko = binaire ; chemins POSIX NFC ; décodage utf-8 → utf-8-sig → utf-16 BOM → latin-1 avec `files.encoding` visible. Détection : `(size, mtime)` identiques → rien lu ; sinon sha1 ; identique → mise à jour mtime ; différent ou `extractor_version` périmé → réindexation du fichier en une transaction (`Database.replace_file_artifacts` existant : DELETE dérivés + relations d'origine, INSERT, id stable, `generation` posée au commit) ; absents → `delete_files_not_in` (cascade dure). Fin de run : re-résolution des imports `unresolved:*`, `linker.rebuild_derived` (calls, handles, model_table, migrates, same_shape, uses_model + overrides), purge des arêtes pendantes, `commits` via `git log -n 500` depuis `repos.head_sha`, `meta.index_version++`, cache vectoriel invalidé, passe d'embeddings réveillée. Un thread indexeur (`ThreadPoolExecutor(1)`) sous verrou (409 sinon), lots de 100 fichiers, annulation coopérative, progression `index_runs` + SSE ; au démarrage un run `running` orphelin passe `interrupted` (reprise naturelle par `generation`). Déclencheurs : bouton, CLI, `--refresh`, poller stat optionnel (`auto_refresh_seconds`), extra `watch` (watchfiles, debounce 2 s). Rejeté : hash systématique, parseur `.gitignore` complet maison, watch obligatoire, ProcessPool.

**Q9 Analyse d'impact.** Entrées : working tree (`git diff --unified=0 HEAD` + non suivis entiers), `--staged`, plage `A..B`, fichier, symbole, diff collé ; champ `Modèle.champ` en v1.5. Avant analyse : réindexation ciblée des fichiers changés, sinon `stale_index=true`. Propagation déterministe : niveau 0 = fichiers + symboles intersectant un hunk (1.0, suppression ou signature changée = « fort ») ; niveau 1 = importateurs (0.9), endpoints `handles` (0.95), modèles/tables `model_table`/`migrates` (0.8), `uses_model` (0.6) ; niveau 2 = appels client via `calls` × confiance puis leurs importateurs ×0.7 ; docs/notes par arêtes `mentions` (0.9) + FTS (0.5) ; profondeur ≤ 3, décroissance 0.85/niveau, meilleur chemin conservé, 500 nœuds max, hubs > 200 dépendants signalés non développés. Sortie `ImpactReport` (section 7), seuils sûr ≥ 0.8 / probable ≥ 0.5 / possible ; rendu Markdown ; `impact_runs` persistés ; « Expliquer avec le LLM » hors chemin critique. Rejeté : impact par LLM, profondeur illimitée, exécution des tests du projet.

**Q10 Documents et journal.** `document_indexer.loaders` existant couvre md/txt/pdf (pypdf, 0 caractère → `no_text` « PDF scanné »)/csv/json/yaml/xml/html/docx (stdlib)/sql. Dépôt : `PUT /api/documents/upload?name=` en `application/octet-stream` (python-multipart retiré ; `docs/API.md` est corrigé) copié dans `documents/<sha1>-<nom>`, ou `POST /api/documents/by-path` (référence, jamais copié, `status='missing'` si disparu) ; plafonds 50 Mo / 5 Mo de texte ; `is_dictionary` auto-détecté et basculable ; fichiers documentaires des dépôts listés dans l'écran Documents (lignes `files`, `kind='doc'`). Notes : `title, body, tags, note_date` (date métier), `status valid|superseded|historical` + `superseded_by` (spec §5), `links` d'entités, `note_revisions` (20 dernières), mentions `[[chemin]]`/`[[Symbole]]`/`@Modèle` → arêtes `mentions` ; sauvegarde = chunking + FTS synchrones, embedding en fond ; « Enregistrer la réponse en note » ; export `.md`. Dérive doc/code « lite » v1 : à la lecture d'un document ou d'une note, les noms de tables/symboles/endpoints cités mais absents de l'index sont renvoyés (`mentions[].exists=false`) → bandeau « semble ne plus correspondre au code » ; passe batch en v1.5. `.env*`, `*.pem`, `*.key`, `*.p12` jamais chunkés. Rejeté : python-docx obligatoire, OCR (v2), notes en fichiers.

**Q11 Packaging et CLI.** `pyproject.toml` : dépendances `fastapi, uvicorn` (sans `[standard]`), `httpx, pydantic>=2, numpy>=1.24,<2.1, pypdf, typing_extensions` ; extras `watch=[watchfiles]`, `sysinfo=[psutil]`, `docx=[python-docx]`, `yaml=[pyyaml]`, `local-embeddings=[sentence-transformers]`, `treesitter`, `dev=[pytest, pytest-asyncio]` ; `python-multipart` retiré ; package-data `web/**`, `storage/*.sql`, `advisor/catalog.json`. CLI argparse (`develop/cli.py`, `__main__.py`) : `serve [--port 8765 repli +10] [--no-browser] [--refresh]`, `init NOM --repo kind=/chemin…`, `repos add|rm|list`, `index [--full] [--embed] [--repo]`, `search`, `ask [--search-only] [--json]` (consomme le même flux SSE), `impact [--diff|--staged|--range A..B|--file] [--json|--md]`, `status [--json] [--verify-offline]`, `doctor` (profil + moteurs + catalogue + conseiller + FTS integrity-check), `workspaces list|use|rm`, `notes export`, `net-log`. Un workspace actif par processus, bascule refusée pendant un run. Rejeté : typer/click, tomllib, un serveur par workspace, en-tête `X-Develop-Project`.

**Q12 Tests.** pytest + **pytest-asyncio `asyncio_mode=auto`** (déjà installé et utilisé par `test_llm_clients.py` ; tranché contre le plugin anyio de « mvp » pour ne pas réécrire les tests existants) + `fastapi.testclient.TestClient` (`client.stream` pour le SSE). Fixtures : `tmp_home`, `no_network` autouse (strict forcé + `socket.connect/getaddrinfo` patchés), `sample_project` (copie en `tmp_path` + `git init/commit`, skip si git absent), `indexed_db`, `fake_llm` (`httpx.MockTransport` Ollama NDJSON + OpenAI SSE + embeddings déterministes par hash), goldens JSON par fichier de fixture, dossier `pieges/`, `tools/fake_llm_server.py` pour la QA manuelle, marqueur `live`. Suite < 60 s, exécutée avec `python3.9 -W error::DeprecationWarning` (le `filterwarnings=ignore` actuel est retiré).

**Divergences des juges, tranchées.** Base = ossature « mvp » (2 juges sur 3) ; de « ux » on prend les fiches lite, les overrides, `num_ctx`, les citations validées, le middleware same-origin, `note_date` (un juge la préférait entière : son périmètre v1 est trop vaste pour livrer) ; de « robust » la Q13 en bloc, `extractor_version`/`generation`, `index_errors`, `stale_index`, `verify-offline`. Le conseiller de modèles, absent des trois, est en v1 car déjà amorcé dans `system/`.

## 3. Arborescence du package et interfaces publiques

Légende : (E) existant conservé, (M) existant modifié, (N) nouveau. Toutes les signatures sont 3.9.

```
develop/
  __init__.py (E)  __main__.py (N)  cli.py (N)  errors.py (N)  config.py (M)
  storage/   db.py (M)  schema.sql (M, v2)
  llm/       base.py (E)  ollama.py (E)  openai_compat.py (E)  factory.py (E)
             offline_guard.py (M)  socket_guard.py (N)  embeddings.py (N)
  system/    profile.py (E)  engines.py (E)
  advisor/   __init__.py  catalog.json  recommend.py (N)
  project_scanner/  scanner.py  languages.py  stack.py (E)
  git_tools/ repo.py (M : env GIT_OPTIONAL_LOCKS=0, diff_hunks)
  code_analysis/  model.py (M)  registry.py (E)  base.py (N)  resolver.py (N)
             extractors/ python.py javascript.py php.py sql.py java.py dart.py openapi.py (N)
  document_indexer/  loaders.py (E)  chunking.py (E)
  indexing/  pipeline.py  fileops.py  linker.py  documents.py  notes.py  history.py  events.py  watcher.py (N)
  search/    identifiers.py  vectors.py  hybrid.py (N)
  rag/       budget.py  cards.py  context.py  prompts.py  citations.py  chat.py (N)
  impact/    gitdiff.py  analyzer.py  report.py (N)
  server/    app.py  state.py  middleware.py  sse.py  deps.py
             routes/ system.py workspaces.py repos.py index.py search.py chat.py explorer.py
                     documents.py notes.py impact.py network.py advisor.py (N)
  web/       index.html (E)  static/ app.js api.js ui.js app.css (M)
             static/views/ setup.js chat.js explorer.js documents.js notes.js impact.js settings.js (N)
tools/fake_llm_server.py (N)
```

**`errors.py`** — `class DevelopError(Exception)`; `ConfigError`, `RepoUnavailable`, `StaleIndex`, `DatabaseCorrupt`, `IndexCancelled` ; `LLMUnavailable = llm.base.LLMUnavailable` ; `NetworkBlocked = llm.offline_guard.OfflineViolation`. Règle : erreur fichier = ligne, erreur service = état, seule `ConfigError` bloque une action (HTTP 400 avec remède).

**`config.py`** (M) — garde `RepoConfig(id, name, path, kind, enabled, exclude)` + `api_prefix: Optional[str] = None`; `LLMConfig` + `api_key_env: str = ""` (remplace `api_key` en clair), `ctx_cap_from_ram: bool = True`; `WorkspaceConfig` + `socket_guard: bool = True`, `auto_refresh_seconds: int = 0`, `embed_on_index: bool = False`; fonctions existantes `develop_home/workspace_dir/db_path/documents_dir/load_settings/save_settings/list_workspaces/load_workspace/save_workspace/create_workspace/delete_workspace` + `logs_dir() -> Path`.

**`storage/db.py`** (M) — `Database(path, migrate=True)` existant, méthodes conservées (`transaction()`, `upsert_file`, `file_signatures`, `delete_files_not_in`, `replace_file_artifacts`, `search_symbols`, `find_endpoints`, `find_models`, `add_relations`, `relations_from/to`, `neighbors`, documents/notes CRUD, `replace_chunks`, `fts_search`, conversations, `start/finish_index_run`, `stats`). Ajouts :
```python
write_lock: threading.RLock                          # pris par transaction() en écriture
def quick_check(self) -> bool
def set_file_status(self, file_id: int, status: str, error: Optional[str] = None,
                    extractor_version: Optional[int] = None, generation: Optional[int] = None) -> None
def add_index_error(self, run_id: int, stage: str, message: str, file_id: Optional[int] = None,
                    path: Optional[str] = None) -> int
def list_index_errors(self, run_id: int, limit: int = 200) -> List[IndexErrorRow]
def delete_relations_by_origin(self, file_id: int) -> int
def delete_derived_relations(self, kinds: Sequence[str]) -> int
def upsert_override(self, src_key: str, dst_key: str, verdict: str) -> int
def list_overrides(self) -> List[OverrideRow]
def upsert_embeddings(self, items: Iterable[Tuple[str, str, np.ndarray]]) -> int   # (text_sha1, model, vec)
def load_embeddings(self, model: str) -> Tuple[np.ndarray, np.ndarray]              # (chunk_ids, matrix N×dim)
def chunks_missing_embeddings(self, model: str, limit: int = 1000) -> List[Tuple[int, str, str]]  # (chunk_id, sha1, text)
def purge_orphan_embeddings(self, keep_model: Optional[str] = None) -> int
def embedding_coverage(self, model: str) -> Tuple[int, int]                          # (avec, total)
def identifier_lookup(self, idents: Sequence[str], limit: int = 50) -> List[IdentHit]
def add_note_revision(self, note_id: int) -> int
def add_network_log(self, row: NetworkLogRow) -> int
def list_network_log(self, limit: int = 200, blocked_only: bool = False) -> List[NetworkLogRow]
def upsert_commits(self, repo_id: str, commits: Sequence[CommitInfo], files: Dict[str, List[Tuple[str, str]]]) -> int
def commits_for_path(self, repo_id: str, path: str, limit: int = 20) -> List[CommitRow]
def bump_index_version(self) -> int
```
Dataclasses ajoutées : `IndexErrorRow`, `OverrideRow`, `NetworkLogRow`, `CommitRow`, `IdentHit(kind, id, name, file_id, path, line)`; `ChunkRow` + `title, terms, page, symbol_id, token_est`; `FileRow` + `status, error, encoding, extractor_version, generation`; `NoteRow` + `status, superseded_by, links, pinned`; `MessageRow` + `cards, model, status, usage`; `DocumentRow` + `mtime, pages, tags, is_dictionary, status, error`; `RelationRow` + `origin_file_id, evidence, resolved_by`.

**`llm/offline_guard.py`** (M) — `OfflineGuard(mode, allowed_hosts, socket_guard=True)`, `is_allowed(host) -> bool`, `check(url) -> None`, `configure(mode, allowed_hosts)`, `set_sink(fn: Callable[[BlockedAttempt], None]) -> None` (persistance en base par `server.state`), `blocked_attempts()`, `clear_log()`; `make_client(...)`, `make_async_client(...)` inchangés. **`llm/socket_guard.py`** (N) — `install(guard: OfflineGuard) -> None`, `uninstall() -> None`, `is_installed() -> bool`.

**`llm/embeddings.py`** (N) —
```python
class EmbeddingService:
    def __init__(self, db: Database, client: LLMClient, model: str, batch_size: int = 32) -> None
    async def embed_query(self, text: str) -> Optional[np.ndarray]
    async def embed_pending(self, progress: Optional[Callable[[int, int], None]] = None,
                            cancel: Optional[threading.Event] = None) -> int
    def coverage(self) -> Dict[str, Any]        # {model, dim, done, total, stale_models: [...]}
    @staticmethod
    def model_id(engine: str, model: str) -> str  # "ollama:nomic-embed-text"
```

**`advisor/recommend.py`** (N) —
```python
@dataclass
class ModelAdvice: tag: str; family: str; size_gb: float; min_ram_gb: float; usage: str; context: int
                   license: str; verdict: str  # recommended | possible_slow | too_big
                   reason: str; pull_command: str; installed: bool
@dataclass
class AdvisorReport: profile_summary: str; tier: str; chat: List[ModelAdvice]; embedding: List[ModelAdvice]
                     install_steps: List[Dict[str, Any]]; num_ctx_cap: int; warnings: List[str]
def load_catalog() -> List[Dict[str, Any]]
def recommend(profile: MachineProfile, engines: EnginesReport, installed: Sequence[str] = ()) -> AdvisorReport
def num_ctx_cap(profile: MachineProfile) -> int
```

**`code_analysis/base.py`** (N) — `mask_source(text, language) -> MaskedSource(masked: str, literals: List[Tuple[int, int, str]])`; `class BraceScanner(masked).block_end(pos: int) -> int`; `split_identifier(name) -> List[str]`; `line_of(text, offset) -> int`; `enclosing_symbol(symbols: Sequence[ExtractedSymbol], line: int) -> Optional[ExtractedSymbol]`; `truncate_long_lines(text, max_len=5000) -> str`. Chaque extracteur expose `EXTRACTOR = <instance>` et `EXTRACTOR_VERSION: int`. **`code_analysis/model.py`** (M) — `ExtractedSymbol.meta["uses"]: List[str]` (noms de classes/modèles référencés), `ExtractedEndpoint.meta` : `prefix, base_ref, client`; `FileAnalysis` + `extractor_version: int = 0`. **`code_analysis/resolver.py`** (N) — `class RepoContext` (`load(root, kind) -> RepoContext` : alias tsconfig, PSR-4, racine `src/`, `base_prefixes`, `api_prefix`) ; `resolve_import(ctx, from_path, imp: ExtractedImport, language) -> Optional[str]`; `detect_base_prefixes(ctx) -> List[str]`.

**`indexing/pipeline.py`** (N) —
```python
@dataclass
class IndexRunResult: run_id: int; status: str; scanned: int; indexed: int; removed: int; errors: int
                      duration_s: float; generation: int
class Indexer:
    def __init__(self, db: Database, workspace: WorkspaceConfig, events: ProgressBus) -> None
    def run(self, full: bool = False, repo_ids: Optional[Sequence[str]] = None,
            paths: Optional[Sequence[str]] = None, trigger: str = "manual",
            cancel: Optional[threading.Event] = None) -> IndexRunResult
    def reindex_paths(self, repo_id: str, rel_paths: Sequence[str]) -> int   # utilisé par impact
    def mark_interrupted_runs(self) -> int
```
`indexing/fileops.py` — `process_file(entry: FileEntry, ctx: RepoContext) -> FileArtifacts(file: FileRow, symbols, endpoints, models, chunks, imports: List[ExtractedImport], uses: List[str], status, error)` (pur, exécutable en thread). `indexing/linker.py` — `score(call: EndpointRow, route: EndpointRow, base_prefixes: Sequence[str]) -> Tuple[float, str]`; `rebuild_derived(db: Database, workspace: WorkspaceConfig) -> LinkStats(calls, handles, model_table, same_shape, unresolved_imports)`; `table_for_model(name: str, declared: Optional[str]) -> str`; `src_key(ep: EndpointRow, repo_id: str) -> str`. `indexing/history.py` — `refresh_commits(db, repo: RepoConfig, limit: int = 500) -> int`. `indexing/events.py` — `class ProgressBus: publish(event: str, data: Dict[str, Any]) -> None; subscribe() -> queue.Queue; snapshot() -> Dict[str, Any]`. `indexing/watcher.py` — `try_start(workspace, on_change: Callable[[Dict[str, List[str]]], None]) -> bool`.

**`indexing/documents.py`** — `class DocumentService(db, workspace)`: `add_upload(name: str, data: bytes, tags: Sequence[str] = ()) -> DocumentRow`; `add_by_path(path: str, tags=()) -> DocumentRow`; `refresh() -> int`; `reindex(doc_id: int) -> DocumentRow`; `remove(doc_id) -> bool`; `content(doc_id, page: Optional[int] = None) -> Dict[str, Any]`; `mentions(doc_id) -> List[Mention(kind, name, exists, line)]`. **`indexing/notes.py`** — `class NoteService(db)`: `create(title, body, tags, note_date: Optional[str] = None, status="valid", links=()) -> NoteRow`; `update(note_id, **fields) -> NoteRow` (révision archivée, réindex synchrone); `delete(note_id) -> bool`; `supersede(old_id, new_id) -> None`; `from_message(message_id: int) -> NoteRow`; `export_markdown(date_from=None, date_to=None) -> str`; `mentions(note_id) -> List[Mention]`.

**`search/hybrid.py`** (N) —
```python
@dataclass
class SearchFilters: types: Sequence[str] = ("file", "document", "note"); repo_id: Optional[str] = None
                     language: Optional[str] = None; tag: Optional[str] = None
                     since: Optional[str] = None; until: Optional[str] = None
@dataclass
class SearchHit: chunk_id: Optional[int]; source_type: str; source_id: int; title: str; path: str
                 repo_id: Optional[str]; line_start: Optional[int]; line_end: Optional[int]; page: Optional[int]
                 snippet: str; score: float; channels: List[str]; reason: str
async def search(db: Database, query: str, filters: SearchFilters, limit: int = 20,
                 embedder: Optional[EmbeddingService] = None, graph_hops: int = 1) -> List[SearchHit]
def parse_query(q: str) -> Tuple[str, SearchFilters]          # "kind:note tag:réunion since:2026-01-01 statuts"
```
`search/identifiers.py` — `extract_identifiers(question: str) -> List[str]`; `lookup(db, idents) -> List[SearchHit]`. `search/vectors.py` — `class VectorIndex: get(db, model) -> VectorIndex` (cache par `index_version`), `topk(q: np.ndarray, k: int, allowed: Optional[np.ndarray] = None) -> List[Tuple[int, float]]`, `to_blob(vec) / from_blob(blob, dim)`.

**`rag/`** — `budget.py`: `@dataclass ContextBudget(num_ctx, system, retrieved, history, reserve)`; `compute_budget(model_ctx: Optional[int], cfg: LLMConfig, cap: int, question_kind: str) -> ContextBudget`; `estimate_tokens(text, kind="text") -> int`. `cards.py`: `@dataclass Card(kind, title, data: Dict[str, Any], sources: List[Source])`; `model_card(db, model_id) -> Card` (champs fusionnés par source model/migration/ddl/dictionary/ts_interface/dart, divergences, relations, endpoints, docs) ; `endpoint_card(db, endpoint_id) -> Card` (handler, appelants + confiance, modèles, docs) ; `callers_card(db, symbol_id) -> Card`; `cards_for_question(db, question) -> List[Card]` (déclenchées par un nom connu dans la question, max 3). `context.py`: `@dataclass Source(ref, source_type, source_id, repo_id, path, title, line_start, line_end, page, chunk_id, symbol_id, score, channels, reason, cited=False)`; `build_messages(workspace, question, cards, hits, history, budget) -> Tuple[List[ChatMessage], List[Source]]`. `citations.py`: `finalize(text: str, sources: List[Source]) -> Tuple[str, List[Source]]`. `chat.py`: `class ChatService(db, workspace, client, embedder)`: `async def stream(conversation_id: int, content: str, mode: str = "auto", disconnected: Optional[Callable[[], Awaitable[bool]]] = None) -> AsyncIterator[ChatEvent(name, data)]`; `question_kind(q) -> str` (`factual|impact|decision|general`).

**`impact/`** — `gitdiff.py`: `collect(repo: GitRepo, mode: str, rev_range: Optional[str] = None, path: Optional[str] = None, diff_text: Optional[str] = None) -> DiffSummary(files: Dict[str, List[Tuple[int, int]]], deleted: List[str], renamed: Dict[str, str])`; `parse_unified_diff(text) -> DiffSummary`. `analyzer.py`: `analyze(db, workspace, repo_id, diff: DiffSummary, max_depth=3, indexer: Optional[Indexer] = None) -> ImpactReport`. `report.py`: `ImpactReport` (pydantic, section 7), `to_markdown(report) -> str`.

**`server/`** — `app.py`: `create_app(state: AppState) -> FastAPI`; `serve(host, port, open_browser, refresh) -> None`. `state.py`: `class AppState` (`settings, workspace: Optional[WorkspaceConfig], db: Optional[Database], guard, client: LLMClient, embedder: Optional[EmbeddingService], indexer, events, index_executor, index_future`) avec `activate(workspace_id) -> None`, `reload_llm() -> None`, `ensure_workspace() -> WorkspaceConfig` (lève `ConfigError` 400 si aucun). `deps.py`: `get_state()`, `get_db()`, `get_client()` (overridables en test). `sse.py`: `event(name: str, data: Any) -> bytes`; `stream(gen: AsyncIterator[Tuple[str, Any]], ping_s: float = 15.0) -> StreamingResponse`. `middleware.py`: `LocalSecurityMiddleware` (Host loopback, same-origin sur mutations et `/api/fs/browse`, CSP, no-store).

**`web/static`** — `app.js`: `state`, `navigate(hash)`, `parseHash()`, `refreshStatus()`; `api.js`: `api.get/post/put/patch/del(path, body, {query})`, `api.putRaw(path, blob, {query})`, `streamSSE(path, body, {onEvent, signal})`, `subscribe(path, handlers)`, `watchIndex(...)`; `ui.js`: `el, replace, renderMarkdown(text, {sources, onCite})`, `citationChip(source)`, `cardModel(card)`, `cardEndpoint(card)`, `fileViewer({text, language, highlight:[a,b]})`, `confidenceBar`, `badge`, `toast`, `modal`, `tabs`; chaque vue : `export function mount(root, route) { ...; return unmount; }`.

## 4. DDL SQLite v2 (`develop/storage/schema.sql`)

Rôles : `meta` clés `schema_version=2, index_version, embedding_model, workspace_id` ; `repos` dépôts ; `files` fichiers à id stable ; `symbols` déclarations ; `endpoints` routes serveur **et** appels client (`side`) ; `data_models` modèles/tables/interfaces (champs en JSON) ; `relations` graphe polymorphe ; `link_overrides` corrections utilisateur ; `documents`/`notes`/`note_revisions` ; `chunks` unité de recherche commune + `chunks_fts` ; `embeddings` vecteurs par contenu ; `conversations`/`messages` ; `commits`/`commit_files` historique ; `index_runs`/`index_errors` ; `network_log` ; `impact_runs` ; `summaries` réservée v2. Dates `REAL` epoch sauf `note_date`.

```sql
PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA synchronous = NORMAL; PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS repos (
  id TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '', path TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT 'other',            -- frontend|backend|mobile|docs|database|shared|other
  status TEXT NOT NULL DEFAULT 'ok',             -- ok|missing|unreadable
  frameworks TEXT NOT NULL DEFAULT '[]', api_prefix TEXT, base_prefixes TEXT NOT NULL DEFAULT '[]',
  head_sha TEXT, last_indexed_at REAL);

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT, repo_id TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  path TEXT NOT NULL, language TEXT, kind TEXT,  -- code|doc|config|data|other
  size INTEGER, mtime REAL, sha1 TEXT, encoding TEXT,
  status TEXT NOT NULL DEFAULT 'ok',             -- ok|error|skipped|large|minified|binary
  error TEXT, extractor_version INTEGER NOT NULL DEFAULT 0, generation INTEGER NOT NULL DEFAULT 0,
  indexed_at REAL, UNIQUE (repo_id, path));
CREATE INDEX IF NOT EXISTS idx_files_sha1 ON files(sha1);
CREATE INDEX IF NOT EXISTS idx_files_lang ON files(language);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);

CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  name TEXT NOT NULL, qualname TEXT NOT NULL DEFAULT '', kind TEXT,
  line_start INTEGER, line_end INTEGER, signature TEXT, parent TEXT, meta TEXT);  -- meta: decorators, uses[], fields[], exported
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_symbols_qualname ON symbols(qualname);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_id);
CREATE INDEX IF NOT EXISTS idx_symbols_kind ON symbols(kind);

CREATE TABLE IF NOT EXISTS endpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  side TEXT NOT NULL CHECK (side IN ('server','client')), method TEXT, path TEXT,
  normalized_path TEXT,                          -- /api/orders/:param
  handler TEXT, framework TEXT, line INTEGER,
  symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,  -- symbole englobant / méthode décorée
  meta TEXT);                                    -- prefix, base_ref, client, middleware, request_fields
CREATE INDEX IF NOT EXISTS idx_endpoints_norm ON endpoints(normalized_path, method);
CREATE INDEX IF NOT EXISTS idx_endpoints_file ON endpoints(file_id);
CREATE INDEX IF NOT EXISTS idx_endpoints_side ON endpoints(side);

CREATE TABLE IF NOT EXISTS data_models (
  id INTEGER PRIMARY KEY AUTOINCREMENT, file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  name TEXT NOT NULL, table_name TEXT, kind TEXT, -- eloquent|migration|sql_table|ts_interface|dart_class|jpa|pydantic|sqlalchemy|dictionary|openapi_schema
  fields TEXT, relations TEXT, line INTEGER,
  symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL, meta TEXT);  -- fields: [{name,type,nullable,default,description,source}]
CREATE INDEX IF NOT EXISTS idx_models_name ON data_models(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_models_table ON data_models(table_name);
CREATE INDEX IF NOT EXISTS idx_models_file ON data_models(file_id);

-- Graphe. Clés de noeud : file:<id> symbol:<id> endpoint:<id> model:<id> document:<id> note:<id> table:<nom> unresolved:<spec>
-- origin_file_id non NULL = arête par-fichier (imports, mentions) supprimée à la réindexation du fichier ;
-- NULL = arête dérivée (calls, handles, model_table, migrates, same_shape, uses_model, references) reconstruite en fin de run.
CREATE TABLE IF NOT EXISTS relations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, src TEXT NOT NULL, dst TEXT NOT NULL, kind TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 1.0, evidence TEXT,
  origin_file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
  resolved_by TEXT NOT NULL DEFAULT 'auto',      -- auto|user
  meta TEXT, UNIQUE (src, dst, kind));
CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(dst);
CREATE INDEX IF NOT EXISTS idx_relations_kind ON relations(kind);
CREATE INDEX IF NOT EXISTS idx_relations_origin ON relations(origin_file_id);

CREATE TABLE IF NOT EXISTS link_overrides (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  src_key TEXT NOT NULL,                         -- 'frontend:src/api/client.ts:GET:/orders/:param'
  dst_key TEXT NOT NULL,                         -- 'backend:routes/api.php:GET:/api/orders/:param'
  verdict TEXT NOT NULL CHECK (verdict IN ('confirm','reject')), created_at REAL,
  UNIQUE (src_key, dst_key));

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL DEFAULT '',
  kind TEXT NOT NULL DEFAULT 'upload' CHECK (kind IN ('upload','path')),
  source_path TEXT, stored_path TEXT, format TEXT, size INTEGER, sha1 TEXT, mtime REAL, pages INTEGER,
  tags TEXT NOT NULL DEFAULT '[]', is_dictionary INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'ok',             -- ok|no_text|missing|error
  error TEXT, added_at REAL, updated_at REAL, meta TEXT);
CREATE INDEX IF NOT EXISTS idx_documents_sha1 ON documents(sha1);

CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL DEFAULT '', body TEXT NOT NULL DEFAULT '',
  tags TEXT NOT NULL DEFAULT '[]', note_date TEXT,  -- YYYY-MM-DD (date métier)
  status TEXT NOT NULL DEFAULT 'valid' CHECK (status IN ('valid','superseded','historical')),
  superseded_by INTEGER REFERENCES notes(id) ON DELETE SET NULL,
  links TEXT NOT NULL DEFAULT '[]',              -- [{kind:'model', key:'model:3', label:'Order'}]
  pinned INTEGER NOT NULL DEFAULT 0, created_at REAL, updated_at REAL);
CREATE INDEX IF NOT EXISTS idx_notes_date ON notes(note_date);

CREATE TABLE IF NOT EXISTS note_revisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
  title TEXT NOT NULL, body TEXT NOT NULL, saved_at REAL);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_type TEXT NOT NULL CHECK (source_type IN ('file','document','note')), source_id INTEGER NOT NULL,
  ordinal INTEGER NOT NULL, title TEXT NOT NULL DEFAULT '',  -- 'backend/app/Models/Order.php · class Order' | 'Spec › Commandes'
  text TEXT NOT NULL, terms TEXT NOT NULL DEFAULT '',          -- identifiants éclatés + segments de chemin
  line_start INTEGER, line_end INTEGER, page INTEGER,
  symbol_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
  sha1 TEXT NOT NULL, token_est INTEGER NOT NULL DEFAULT 0, meta TEXT,
  UNIQUE (source_type, source_id, ordinal));
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_sha1 ON chunks(sha1);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  title, text, terms, content='chunks', content_rowid='id',
  tokenize = "unicode61 remove_diacritics 2 tokenchars '_'");
-- requête : SELECT rowid, bm25(chunks_fts, 3.0, 1.0, 0.7) AS s FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, title, text, terms) VALUES (new.id, new.title, new.text, new.terms); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, title, text, terms) VALUES ('delete', old.id, old.title, old.text, old.terms); END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, title, text, terms) VALUES ('delete', old.id, old.title, old.text, old.terms);
  INSERT INTO chunks_fts(rowid, title, text, terms) VALUES (new.id, new.title, new.text, new.terms); END;
CREATE TRIGGER IF NOT EXISTS files_ad_chunks AFTER DELETE ON files BEGIN
  DELETE FROM chunks WHERE source_type = 'file' AND source_id = old.id; END;
CREATE TRIGGER IF NOT EXISTS documents_ad_chunks AFTER DELETE ON documents BEGIN
  DELETE FROM chunks WHERE source_type = 'document' AND source_id = old.id; END;
CREATE TRIGGER IF NOT EXISTS notes_ad_chunks AFTER DELETE ON notes BEGIN
  DELETE FROM chunks WHERE source_type = 'note' AND source_id = old.id; END;

CREATE TABLE IF NOT EXISTS embeddings (
  text_sha1 TEXT NOT NULL, model TEXT NOT NULL,  -- 'ollama:nomic-embed-text'
  dim INTEGER NOT NULL, vec BLOB NOT NULL,        -- float32 little-endian, normalisé L2
  created_at REAL, PRIMARY KEY (text_sha1, model)) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL DEFAULT '', created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('system','user','assistant')), content TEXT NOT NULL DEFAULT '',
  sources TEXT NOT NULL DEFAULT '[]', cards TEXT NOT NULL DEFAULT '[]', model TEXT,
  status TEXT NOT NULL DEFAULT 'done',           -- done|partial|cancelled|error
  usage TEXT, created_at REAL);                  -- usage: {prompt_tokens, completion_tokens, num_ctx, duration_ms}
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS commits (
  id INTEGER PRIMARY KEY AUTOINCREMENT, repo_id TEXT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
  sha TEXT NOT NULL, author TEXT, date TEXT, subject TEXT, UNIQUE (repo_id, sha));
CREATE INDEX IF NOT EXISTS idx_commits_repo_date ON commits(repo_id, date DESC);
CREATE TABLE IF NOT EXISTS commit_files (
  commit_id INTEGER NOT NULL REFERENCES commits(id) ON DELETE CASCADE, path TEXT NOT NULL,
  change TEXT NOT NULL DEFAULT 'M', PRIMARY KEY (commit_id, path)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_commit_files_path ON commit_files(path);

CREATE TABLE IF NOT EXISTS index_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL, finished_at REAL,
  trigger TEXT NOT NULL DEFAULT 'manual',        -- manual|cli|startup|poll|watch|note|document|impact
  mode TEXT NOT NULL DEFAULT 'incremental',      -- incremental|full|partial
  generation INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'running',        -- running|ok|partial|error|cancelled|interrupted
  stats TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS index_errors (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL REFERENCES index_runs(id) ON DELETE CASCADE,
  file_id INTEGER, path TEXT, stage TEXT NOT NULL,  -- scan|read|decode|extract|chunk|link|document|note|embed|write
  message TEXT NOT NULL, created_at REAL);
CREATE INDEX IF NOT EXISTS idx_index_errors_run ON index_errors(run_id);

CREATE TABLE IF NOT EXISTS network_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, mode TEXT NOT NULL, host TEXT NOT NULL, port INTEGER,
  url TEXT NOT NULL, method TEXT, allowed INTEGER NOT NULL, reason TEXT NOT NULL, caller TEXT);
CREATE INDEX IF NOT EXISTS idx_network_log_ts ON network_log(ts DESC);

CREATE TABLE IF NOT EXISTS impact_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL, input TEXT NOT NULL, report TEXT NOT NULL,
  stale_index INTEGER NOT NULL DEFAULT 0);

-- Réservée (v2, synthèses hiérarchiques) : aucune route ne l'utilise en v1.
CREATE TABLE IF NOT EXISTS summaries (
  id INTEGER PRIMARY KEY AUTOINCREMENT, owner_type TEXT NOT NULL, owner_id INTEGER NOT NULL,
  level TEXT NOT NULL, sources_sha1 TEXT NOT NULL, text TEXT NOT NULL, model TEXT,
  status TEXT NOT NULL DEFAULT 'ok', created_at REAL, UNIQUE (owner_type, owner_id, level));

INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '2');
INSERT OR IGNORE INTO meta(key, value) VALUES ('index_version', '0');
```

## 5. Routes HTTP (FastAPI, préfixe `/api`, JSON ; erreurs `{"error":{"code","message","hint"}}`)

Toutes les routes sans `workspace_id` portent sur le workspace actif ; `needs_setup=true` tant qu'aucun n'existe. Mutations et `/api/fs/browse` exigent same-origin.

| Méthode | Chemin | Corps → réponse |
|---|---|---|
| GET | `/` , `/static/*` | `index.html` (CSP), fichiers statiques |
| GET | `/api/status` | `{version, needs_setup, workspace, repos[], engine:{ok,engine,model,error}, embeddings:{model,done,total,stale}, index:{running,run_id,last_run,stale_files,counts}, network:{mode,blocked_count,socket_guard}, disk_free_gb}` |
| GET/POST | `/api/workspaces` | liste / `{name}` → crée et active |
| GET/PUT/DELETE | `/api/workspaces/{id}` | config complète / mise à jour partielle (recharge garde + LLM ; 400 si `base_url` non locale en strict) / suppression de la mémoire |
| POST | `/api/workspaces/{id}/activate` | 409 pendant un run |
| GET/POST/PUT/DELETE | `/api/repos`, `/api/repos/{id}` | `{name,path,kind,exclude,api_prefix}` ; frameworks détectés par `stack.detect_stack` |
| POST | `/api/repos/discover` | `{path}` → `{root, subprojects[]}` |
| POST | `/api/fs/browse` | `{path?}` → `{path, parent, entries:[{name,path,is_git,has_marker}]}` |
| GET | `/api/system/profile` ; `/api/system/engines` ; `/api/system/advice` | `MachineProfile`, `EnginesReport`, `AdvisorReport` |
| POST | `/api/llm/test` | `{engine, base_url, model, embedding_model, api_key_env}` → `{ok, status, model_info, embedding_ok, embedding_dim, suggested_num_ctx, error}` (refus immédiat sans contact si hôte interdit) |
| GET | `/api/llm/models?engine=&base_url=` | `[ModelInfo]` |
| POST | `/api/index` | `{full?, repo_ids?, paths?, embed?}` → `{run_id}` ; 409 si un run est en cours |
| GET | `/api/index/status` ; `/api/index/runs` ; `/api/index/runs/{id}/errors` | état courant, historique, erreurs par fichier |
| GET | `/api/index/events` | SSE `progress {phase,current,total,repo,message}`, `done`, `error`, `blocked` |
| POST | `/api/index/cancel` ; `/api/index/embed` | annulation ; `{recompute?}` passe d'embeddings seule |
| GET | `/api/search?q=&types=&repo_id=&language=&tag=&since=&until=&limit=` | `{query, identifiers[], mode:'hybrid'|'lexical', results:[SearchHit]}` |
| GET/POST | `/api/conversations` ; GET/PATCH/DELETE `/api/conversations/{id}` | liste / `{title?}` / messages avec sources et fiches / renommer / supprimer |
| POST | `/api/conversations/{id}/messages` | `{content, mode:'auto'|'search_only', model?, filters?}` → **SSE** `status {text}` · `cards {items:[Card]}` · `sources {items:[Source]}` · `delta {text}`* · `done {message_id, usage, budget, truncated_sources[]}` \| `error {code,message}` \| `blocked {host,reason}` ; `: ping` / 15 s ; déconnexion client = annulation amont |
| POST | `/api/messages/{id}/to-note` | → `NoteRow` (citations conservées) |
| POST | `/api/ask` | `{content, mode}` → SSE identique (CLI) ou JSON si `?stream=false` |
| GET | `/api/files?repo_id=&q=&language=&kind=&status=` ; `/api/files/{id}` ; `/api/files/{id}/content?start=&end=` | liste plate ; fichier + symboles/endpoints/modèles/relations in-out/documents mentionnant/commits ; texte lu à la demande (2 000 lignes max, `safe_join`) |
| GET | `/api/symbols?q=&kind=&repo_id=` ; `/api/symbols/{id}` | recherche NOCASE ; fiche `CallersCard` |
| GET | `/api/endpoints?side=&q=&method=&repo_id=&unlinked=` ; `/api/endpoints/{id}` | table avec `matches[]` ; `EndpointCard` |
| GET | `/api/models?q=&kind=` ; `/api/models/{id}` | liste ; `ModelCard` (champs fusionnés, divergences, relations, migrations, endpoints, docs) |
| GET | `/api/graph/neighbors?node=file:12&depth=1&kinds=&limit=200` | `{nodes[], edges[]}` |
| POST | `/api/links/override` | `{src_key, dst_key, verdict}` ; relink immédiat des deux nœuds |
| POST | `/api/impact` | `{source:'git'|'file'|'symbol'|'diff', repo_id, mode?:'worktree'|'staged'|'range', rev_range?, path?, symbol_id?, diff?, max_depth?, no_index?}` → `ImpactReport` |
| GET | `/api/impact/runs/{id}` ; `/api/git/status?repo_id=` ; `/api/git/log?repo_id=&path=&limit=` | rapport persisté ; `{branch, head, changes[]}` ; commits |
| GET | `/api/documents` ; `/api/documents/{id}` ; `/api/documents/{id}/content?page=` ; `/api/documents/{id}/raw` | liste unifiée (uploads + chemins + fichiers doc des dépôts) ; métadonnées + `mentions[]` (dérive) ; texte ; original |
| PUT | `/api/documents/upload?name=&tags=` | corps `application/octet-stream` → `DocumentRow` |
| POST / DELETE | `/api/documents/by-path` ; `/api/documents/refresh` ; `/api/documents/{id}` | `{path, tags?, is_dictionary?}` ; re-vérification ; suppression |
| GET/POST | `/api/notes?q=&tag=&from=&to=&status=` | liste triée par `note_date` ; `{title, body, tags, note_date?, status?, supersedes?, links?}` |
| GET/PUT/DELETE | `/api/notes/{id}` ; GET `/api/notes/{id}/revisions` ; GET `/api/notes/tags` ; GET `/api/notes/export` | note + mentions ; révisions ; tags avec comptes ; Markdown |
| GET/DELETE | `/api/network/log?limit=&blocked_only=` | journal ; vidage |
| POST | `/api/network/verify-offline` | tente `https://example.com` via le client gardé → `{blocked:true, logged:true}` |
| GET | `/api/logs?lines=200` | fin de `~/.develop/logs/develop.log` |

## 6. Écrans de l'UI (`web/static/views/*.js`)

- **Assistant (`#/setup`)** — 4 étapes : projet et dépôts (navigateur `/api/fs/browse`, `discover`, rôle, frameworks en badges) ; Moteur d'IA (Ollama | Docker/OpenAI-compatible | aucun, `base_url`, « Tester l'accès », choix du modèle de chat et d'embeddings, **conseiller** : profil machine en une ligne, modèles classés recommandé / possible mais lent / trop gros avec raison et `ollama pull …`, instructions par OS) ; Mode réseau (strict | local avec services internes + hôtes autorisés) ; Indexer maintenant (barre SSE, accès à l'explorateur dès la fin de la phase FTS).
- **Chat (`#/chat/:id`)** — conversations à gauche ; saisie avec auto-complétion des noms de modèles/routes ; sélecteur de modèle, bascule « recherche seule », Stop ; réponse = fiches d'abord (tableau champ/type/nullable/défaut/source/description avec colonnes back/TS/Dart et ligne « divergences » ; endpoint = méthode, chemin, handler, appelants + confiance), puis Markdown avec puces `[S1]` cliquables ; panneau Sources à droite (viewer lignes surlignées, « Voir dans l'explorateur ») ; « Enregistrer en note » ; bandeaux « aucun moteur », « lexical seul », toast `blocked`.
- **Explorateur (`#/explorer/<onglet>`)** — onglets Recherche (champ unique avec syntaxe `kind:note tag:x since:`, facettes, badge lexical/hybride), Fichiers (arbre par dépôt, badges statut/encodage, viewer + outline + panneau Relations + commits), Symboles, Endpoints (table méthode/chemin/handler/appelants/confiance, détail avec preuve, Confirmer/Rejeter), Modèles (ModelCard), Liaisons (appels non liés, routes sans appelant). Chaque élément : « Analyser l'impact », « Demander au chat ».
- **Documents (`#/documents`)** — glisser-déposer (PUT octet-stream), ajout par chemin, bascule dictionnaire, statut (ok / sans texte / manquant / erreur), aperçu par page, bandeau de dérive, réindexer, supprimer.
- **Journal (`#/notes`)** — timeline par `note_date`, filtres tag/période/statut, éditeur Markdown + aperçu, `[[` auto-complétion de chemins/symboles, statut valide / remplacée par / historique, révisions, épingler, export.
- **Impact (`#/impact`)** — dépôt + source (working tree pré-rempli via `git status`, staged, plage, fichier, symbole, diff collé) ; avertissement index périmé « Réindexer puis relancer » ; groupes dépliables avec badges sûr/probable/possible et chemin « via » ; clic → viewer ; « Expliquer avec le LLM », « Enregistrer en note », copier Markdown/JSON.
- **Réglages (`#/settings`)** — Projet (sélecteur multi-workspaces), Dépôts, Moteur d'IA (+ conseiller), Embeddings (couverture, Recalculer N chunks), Réseau (mode, hôtes, garde socket, Journal réseau, « Vérifier hors ligne »), Indexation (auto-refresh, watch disponible ou non, erreurs du dernier run, Réindexer tout), Diagnostics (`doctor`, chemins, logs).
- **Barre d'état (permanente, `app.js`)** — pastilles index (dernier run, fichiers périmés, progression), LLM, embeddings, réseau « Hors ligne strict · N blocages ».

## 7. Format des citations et du rapport d'impact

**Citations.** Chaque passage injecté porte `[S<n>]` ; le modèle cite `[S1]`, `[S1][S3]`. `Source` (JSON, stocké dans `messages.sources`) :
```json
{"ref":"S1","source_type":"file","source_id":42,"repo_id":"backend","path":"app/Models/Order.php",
 "title":"backend/app/Models/Order.php · class Order","line_start":12,"line_end":48,"page":null,
 "chunk_id":913,"symbol_id":77,"score":0.031,"channels":["identifier","lexical"],
 "reason":"nom exact « Order » + BM25","cited":true}
```
`source_type ∈ file|document|note|symbol|endpoint|model` (les fiches sont des sources). Liens UI : `#/explorer/file/42?line=12`, `#/documents/7?page=3`, `#/notes/12`, `#/explorer/models/3`. En fin de flux `citations.finalize` retire les `[Sn]` inconnus et marque `cited`. L'UI et l'export ne dépendent jamais de l'existence de `chunk_id` (toujours `path:lignes`). CLI `develop ask` affiche la réponse puis `S1  backend/app/Models/Order.php:12-48`.

**ImpactReport** (pydantic, 3.9) :
```json
{"input":{"source":"git","repo_id":"backend","mode":"worktree","rev_range":null},
 "stale_index":false,"generation":17,
 "changed":[{"repo_id":"backend","path":"app/Services/OrderService.php","status":"M","hunks":[[10,18]],
             "symbols":[{"id":80,"name":"OrderService.confirm","kind":"method","lines":[9,22],"change":"modified"}]}],
 "affected":[{"type":"endpoint","key":"endpoint:5","repo_id":"backend","path":"routes/api.php","line":13,
              "label":"POST /api/orders/:param/confirm","level":1,"confidence":0.95,"tier":"certain",
              "why":["OrderService.confirm","← OrderController.confirm (imports)","← POST /api/orders/:param/confirm (handles)"]},
             {"type":"client_call","key":"endpoint:31","repo_id":"frontend","path":"src/api/client.ts","line":32,
              "label":"http.post `/orders/${id}/confirm`","level":2,"confidence":0.81,"tier":"certain","why":["…","← calls (préfixe /api, 0.85)"]}],
 "groups":{"dependents":[...],"endpoints":[...],"frontend_calls":[...],"models":[...],"docs_notes":[...]},
 "hubs":[{"key":"file:3","path":"src/utils/money.ts","dependents":240}],
 "counts":{"changed_files":1,"affected":9,"certain":4,"probable":3,"possible":2},"warnings":[]}
```
`tier` : certain ≥ 0.8, probable ≥ 0.5, possible sinon. `to_markdown` produit un tableau par groupe (chemin, ligne, confiance, pourquoi).

## 8. Modes de défaillance et signalement

| Situation | Comportement | Signalement UI |
|---|---|---|
| Aucun workspace / dépôt introuvable | `needs_setup` ; `repos.status='missing'`, index conservé | redirection assistant ; bandeau dépôt manquant |
| git absent ou dépôt sans `.git` | repli `scandir` | avertissement dans Réglages > Dépôts |
| Fichier illisible, binaire, > 1 Mo, minifié, encodage de repli | `files.status`, `encoding`, non chunké ou chunk générique | badge sur le fichier dans l'explorateur |
| Extracteur qui lève | `status='error'` + `index_errors(stage='extract')`, chunking générique continue | compteur d'erreurs du run, liste cliquable |
| Run interrompu (arrêt brutal) | marqué `interrupted` au démarrage, reprise par `generation` | pastille index « run interrompu, reprise » |
| Deux instances / run concurrent | verrou d'écriture + 409 `index_running` | toast « indexation en cours » |
| Base corrompue (`quick_check`) | renommée `.corrupt-<ts>`, recréée | bandeau « mémoire recréée, réindexer » |
| Disque < 500 Mo / `SQLITE_FULL` | run `error`, transaction annulée | bandeau avec Mo à libérer |
| Aucun LLM / moteur arrêté / modèle absent | `engine.ok=false`, chat en mode gabarit, `ModelNotFound` avec `ollama pull` | bandeau chat + bouton Tester / Conseiller |
| Pas d'embeddings ou modèle changé | BM25 seul, couverture partielle utilisée | badge « lexical seul » / « vecteurs : 62 % » + Recalculer |
| Réponse LLM coupée / annulée | `messages.status partial|cancelled`, texte conservé | mention sous le message |
| Requête réseau externe | `OfflineViolation`, `network_log`, SSE `blocked` | toast + compteur dans la barre d'état |
| `base_url` interdite en strict | `PUT /api/workspaces` → 400 sans contact | message avec le mode à choisir |
| Index périmé pour l'impact | `stale_index=true` | bandeau « Réindexer puis relancer » |
| PDF scanné / document disparu | `status no_text|missing` | statut dans la liste Documents |
| Hub > 200 dépendants | non développé, listé dans `hubs` | ligne « utilitaire importé partout » |
| Requête cross-site | 403 `same_origin_required` | — (journal serveur) |

Journal : `~/.develop/logs/develop.log` (RotatingFileHandler 5×5 Mo) + `/api/logs` dans Réglages > Diagnostics.

## 9. Ordre de construction en lots parallélisables

Règle : un fichier appartient à un seul lot ; un lot ne dépend que des **interfaces** de la section 3 des lots précédents (stubs acceptés). Les modules existants non listés ne sont pas touchés.

**Lot 0 — Socle (séquentiel, bloque tout).** `pyproject.toml` (deps, extras, pytest), `errors.py`, `config.py` (champs ajoutés), `storage/schema.sql` v2 + `storage/db.py` (méthodes ajoutées, embeddings par sha1, `write_lock`, `quick_check`) + `tests/test_storage.py` adapté, `llm/offline_guard.py` (strict loopback seul, sink, `follow_redirects=False`) + `llm/socket_guard.py` + `tests/test_offline_guard.py`, `git_tools/repo.py` (env, `diff_hunks`), `code_analysis/model.py` (champs), `code_analysis/base.py`, `tests/conftest.py` (`tmp_home`, `no_network`, `sample_project` git, `fake_llm`), `tests/test_no_network_libs.py`, `cli.py` squelette (`status`, `doctor` vide), `__main__.py`.

**Lot 1 (4 agents en parallèle, après Lot 0).**
- 1A Extracteurs : `code_analysis/extractors/{python,javascript,php,sql,java,dart,openapi}.py`, `code_analysis/resolver.py`, `tests/test_extractors_*.py` + goldens `tests/golden/*.json`, compléments de fixture (section 10).
- 1B Pipeline : `indexing/{pipeline,fileops,events,history,watcher}.py`, `tests/test_indexer_incremental.py` (fonctionne avec le registre existant même sans extracteur : chunks + FTS).
- 1C Recherche et embeddings : `search/{identifiers,vectors,hybrid}.py`, `llm/embeddings.py`, `tests/test_search_hybrid.py`, `tests/test_embeddings.py`.
- 1D Serveur, UI de base, conseiller : `server/{app,state,middleware,sse,deps}.py`, `server/routes/{system,workspaces,repos,index,network,advisor}.py`, `advisor/{catalog.json,recommend.py}`, `web/static/{app,api,ui}.js` (ajouts), `web/static/views/{setup,settings}.js`, `app.css`, `tests/test_api_core.py`, `tests/test_advisor.py`, `tools/fake_llm_server.py`. Dépend de 1B uniquement par `Indexer.run`/`ProgressBus` (stub tant que 1B n'est pas livré).

**Lot 2 (3 agents, après 1A–1D).**
- 2A Liaison et explorateur : `indexing/linker.py`, `rag/cards.py`, `server/routes/explorer.py`, `web/static/views/explorer.js`, `tests/test_linker.py`, `tests/test_cards.py`, `tests/test_api_explorer.py`.
- 2B Chat : `rag/{budget,context,prompts,citations,chat}.py`, `server/routes/{chat,search}.py`, `web/static/views/chat.js`, `cli.py` (`ask`, `search`), `tests/test_rag_context.py`, `tests/test_api_chat.py`, `tests/test_cli.py`.
- 2C Documents et notes : `indexing/{documents,notes}.py`, `server/routes/{documents,notes}.py`, `web/static/views/{documents,notes}.js`, `tests/test_documents_service.py`, `tests/test_notes_service.py`, `tests/test_api_documents_notes.py`.

**Lot 3 (2 agents, après Lot 2).**
- 3A Impact : `impact/{gitdiff,analyzer,report}.py`, `server/routes/impact.py`, `web/static/views/impact.js`, `cli.py` (`impact`), `tests/test_impact.py`.
- 3B Durcissement et finition : `cli.py` (`doctor` complet, `status --verify-offline`, `workspaces`, `notes export`, `net-log`), `server/routes/network.py` (`verify-offline`), watcher branché dans `state.py`, `docs/API.md` mis à jour, `README.md`, `tests/test_failures.py` (base corrompue, disque plein simulé, run interrompu), `tests/test_ui_static.py`, vérification `pip install -e .` dans un venv 3.9 propre.

**v1.5 (après v1, même protocole)** : boucle d'outils bornée (`rag/tools.py`), FormRequest/Resource Laravel, `same_shape` Jaccard complet, impact par champ, passe de dérive batch, `open-in-editor`. **v2** : synthèses hiérarchiques, tree-sitter, OCR.

## 10. Stratégie de tests et fixture `tests/fixtures/sample_project`

**Principes.** Aucun test ne sort sur le réseau (fixture autouse `no_network` : mode strict forcé, `socket.socket.connect` et `socket.getaddrinfo` lèvent hors loopback) ; aucun test n'exige Ollama (`httpx.MockTransport` injecté par `offline_guard.make_async_client(inner=…)` et par l'override `deps.get_client`) ; tests `live` (marqueur, désactivés par défaut) pour un vrai Ollama chez l'utilisateur ; `python3.9 -W error::DeprecationWarning -m pytest -q` < 60 s.

**Fixtures pytest (`conftest.py`).** `tmp_home` (`DEVELOP_HOME` → `tmp_path`) ; `sample_project` (copie en `tmp_path`, `git init` + `add` + `commit` si git disponible, sinon `pytest.skip` des tests git) ; `workspace` (WorkspaceConfig avec les 6 dépôts de la fixture, strict) ; `db` (Database en mémoire migrée) ; `indexed` (run complet, scope module) ; `fake_llm` (handler Ollama `/api/version|tags|show|chat` NDJSON token par token avec `[S1]` dans la réponse, `/api/embed` vecteurs déterministes par hash ; variante OpenAI `/v1/models|chat/completions` SSE `[DONE]`|`embeddings`) ; `fake_embedder` (sac de mots haché, similarité plausible) ; `client` (TestClient avec `AppState` de test).

**Contenu de la fixture** (existant conservé ; ajouts marqués +) :
- `frontend/` React/TS + axios : `src/api/client.ts` (`axios.create({baseURL: import.meta.env.VITE_API_URL ?? …})`, `http.get('/orders')`, template `/orders/${id}/confirm`, `fetch(\`${API_BASE}/customers?q=\`)`, interfaces `Order`/`Customer`), `components/OrderList.tsx`, `CustomerSearch.tsx`, `Notifications.jsx`, `utils/money.ts`, `.env.example` (`VITE_API_URL=http://localhost:8000/api`), `package.json`, `node_modules/leftpad` (doit être exclu), `logo.png` (binaire) ; + `vite.config.js` (proxy `/api`), + `tsconfig.json` (`paths @/*`).
- `backend/` Laravel : `routes/api.php` (`Route::prefix('v1')->group`, `get/post` avec `{order}`, `apiResource('customers')->only([...])`), `app/Models/{Order,Customer,OrderLine}.php` (`$table`, `$fillable`, `$casts`, `belongsTo/hasMany`), `app/Http/Controllers/*`, `app/Services/OrderService.php` (`Http::post('http://localhost:3001/notifications')` = appel client PHP), `database/migrations/*` (`Schema::create` orders/order_lines), `composer.json` ; + `bootstrap/app.php` (`withRouting(api: …)`), + `app/Http/Requests/ConfirmOrderRequest.php` (v1.5).
- `services/notifier/` Express : `src/server.js` (`app.get/post/delete('/notifications/:userId')`), `src/store.js`, `package.json`.
- `database/schema.sql` (`CREATE TABLE` ×3 avec `REFERENCES`).
- `docs/` : `dictionnaire-donnees.md` (tableaux `Colonne|Type|Description` → `data_models(kind='dictionary')`), `specification-commandes.md`, `decisions/2026-09-01-reunion-statuts.md`, `glossaire.json` ; + `spec.pdf` (une page, texte « confirmation des commandes »), + `config.yaml`, + `exports.csv`.
- `tools/report.py` (dataclass, fonctions, classe) ; + `python-api/` (FastAPI `APIRouter(prefix='/v2')` + `@router.get('/orders/{order_id}')`, modèle pydantic).
- + `mobile/` Dart : `lib/models/order.dart` (`fromJson` avec clés `total_amount`), `lib/services/api.dart` (`Dio(BaseOptions(baseUrl))`, `dio.get('/orders/$id')`), `pubspec.yaml`.
- + `java/` Spring : `OrderController.java` (`@RestController @RequestMapping("/api/v2/orders")` + `@GetMapping("/{id}")`), `Order.java` (`@Entity @Table(name="orders")`), `pom.xml`.
- + `pieges/` : `latin1.txt`, `nom-unicodé-é.md`, `bundle.min.js` (ligne de 20 000 caractères), `secret.pem` (jamais chunké) ; fichiers de 3 Mo et symlinks générés dans `tmp_path` par les tests.

**Suites.** `test_extractors_{python,js,php,sql,java,dart,openapi}.py` (goldens JSON : symboles, endpoints avec `normalized_path`, modèles avec champs, imports résolus, lignes exactes) ; `test_linker.py` (table paramétrée appel → route → score et preuve attendus : `/orders/${id}/confirm` ↔ `POST /api/orders/{order}/confirm` 0.85 via `/api`, `Http::post` PHP ↔ Express `POST /notifications` 1.0, méthode contradictoire → aucune arête, override `reject` appliqué) ; `test_indexer_incremental.py` (run 1 → compteurs ; toucher 1 fichier → exactement 1 réindexé d'après `index_runs.stats` ; suppression → cascade symbols/chunks/relations ; renommage ; bump `EXTRACTOR_VERSION` → réextraction du langage seul ; run interrompu simulé → `interrupted` puis reprise ; `node_modules`, `.pem`, binaire, 3 Mo, symlink exclus) ; `test_search_hybrid.py` (« order controller » trouve `OrderController` par `terms` ; identifiants exacts ; RRF avec `fake_embedder` ; BM25 seul sans embedder ; filtres `kind:note since:`) ; `test_embeddings.py` (clé sha1, reprise, changement de modèle, couverture) ; `test_cards.py` (ModelCard `Order` fusionne migration + `$fillable` + dictionnaire + interface TS et signale `confirmed_at` absent côté TS) ; `test_rag_context.py` (budget respecté, numérotation `[S1]`, citations invalides retirées, `num_ctx` plafonné par la RAM) ; `test_llm_clients.py` (existant : NDJSON/SSE, 404, flux coupé, annulation) ; `test_offline_guard.py` (strict bloque `api.openai.com`, `10.0.0.5` et `host.docker.internal` ; `local_network` autorise `192.168.1.10` ; journal ; proxy env ignoré ; garde socket avec serveur démarré ; `verify-offline`) ; `test_api_*.py` (setup → repos → index → search → chat SSE `status, cards, sources, delta, done` ; `search_only` sans LLM ; upload octet-stream ; notes CRUD + révisions + `supersede` ; CSRF 403 sans same-origin ; `activate` 409 pendant un run) ; `test_impact.py` (modification de `OrderService::confirm` dans la copie git → endpoint `POST /api/orders/:param/confirm` certain, appel `client.ts:32` probable, note du 2026-09-01 possible ; `stale_index`) ; `test_documents_service.py` (pdf/docx/csv/html, `no_text`, `missing`, dictionnaire détecté, dérive `exists=false` après suppression d'une table) ; `test_advisor.py` (profil 32 Go sans GPU → 7b recommandé, 70b trop gros, nomic-embed-text toujours proposé, instructions macOS) ; `test_failures.py` ; `test_cli.py` (`main([...])` avec `DEVELOP_HOME` temporaire, `ask --json`, `impact --md`) ; `test_no_network_libs.py` + `test_ui_static.py` (grep).