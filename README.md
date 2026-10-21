# Develop

**Develop** est la mémoire locale d'un projet logiciel : un outil de développeur, entièrement en Python, qui lit vos
dépôts (frontend, backend, base de données, documentation), vos documents de conception et vos notes datées, construit
une carte citable du projet (fichiers, symboles, routes, modèles de données, relations) et vous laisse l'interroger
dans une interface de type chat branchée sur un **LLM local** (Ollama ou tout serveur compatible OpenAI).

Tout fonctionne **hors ligne**. Rien ne sort de votre machine. Vos dépôts ne sont **jamais modifiés**.

## Pourquoi

Quand un projet grossit, on oublie ce qu'un changement va casser, et le frontendeur doit déranger le backendeur pour
connaître les champs de tel modèle. Develop répond à ces questions en citant ses sources :

- « Quels champs a la table `orders` ? » → le dictionnaire de données, la migration et l'interface TypeScript.
- « Qui appelle `POST /orders/{id}/confirm` ? » → la fonction `confirmOrder` du frontend et le composant qui l'utilise.
- « Qu'est-ce que je casse si je modifie `OrderService::confirm` ? » → l'analyse d'impact déterministe, sans LLM.
- « Qu'a-t-on décidé sur le statut *payé* ? » → la note de réunion du 2026-09-01.

## Comment Develop évite d'« avaler » tout le projet

Le modèle ne lit jamais le projet. Develop garde une mémoire persistante (SQLite) et, à chaque question :

1. analyse la question (identifiants, routes, tables, intention, dates) ;
2. cherche sur six canaux : identifiants exacts, routes (vocabulaire de la question rapproché des chemins et des
   contrôleurs : « authentification » → `login`, `AuthController`), recherche lexicale (FTS5), recherche sémantique
   (embeddings locaux, si un modèle est configuré), graphe de relations (un saut), notes par date ;
3. fusionne les résultats (Reciprocal Rank Fusion), déduplique, diversifie ;
4. calcule un **budget de contexte** à partir de la fenêtre réelle du modèle et de la RAM de la machine, puis remplit
   ce budget avec les meilleurs passages, chacun identifié `[S1]`, `[S2]`… ;
5. laisse au modèle au plus quatre tours d'**outils locaux** (rechercher, lire une fenêtre de fichier, appelants d'un
   symbole, détail d'un modèle, notes, impact) pour compléter ce qui manque, dans la limite du budget ;
6. rend une réponse citée ; chaque citation est un lien vers le fichier, la ligne, le document ou la note.

Un projet de 10 000 lignes, 30 documents et 100 notes tient donc dans un modèle de 8k à 32k de contexte, et un modèle
de 128k n'est jamais rempli inutilement (ce qui serait lent sur un portable).

## Prérequis

- Python 3.9 ou plus récent (3.11+ recommandé) ;
- Git (optionnel mais recommandé : respect du `.gitignore`, diff pour l'analyse d'impact) ;
- un moteur de LLM local : [Ollama](https://ollama.com) (recommandé), ou LM Studio, llama.cpp, vLLM, Jan…
  Develop fonctionne sans moteur (indexation, recherche, explorateur, notes, impact) ; seul le chat en a besoin.

Le seul moment qui nécessite Internet est l'installation (dépendances Python, téléchargement d'un modèle).

## Installation

Develop s'installe comme un outil Python, en une commande, sans Docker ni sudo : un environnement dédié dans
`~/.develop/venv` et une commande `develop`. Ses données restent dans `~/.develop`.

**macOS / Linux**

```bash
git clone https://github.com/pamstore/Develop.git && cd Develop && sh install.sh
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/pamstore/Develop.git; cd Develop; powershell -ExecutionPolicy Bypass -File install.ps1
```

Le script vérifie Python ≥ 3.9 (et indique comment l'installer sinon : Homebrew ou python.org, winget sous Windows),
crée l'environnement, installe le paquet, puis place la commande `develop` dans `~/.local/bin` (macOS/Linux, avec la
ligne à ajouter au PATH si besoin) ou dans `%USERPROFILE%\.develop\bin` ajouté au PATH utilisateur (Windows).
Relancer le script met à jour l'installation. Options : `--desktop` (fenêtre native, voir « Lancer »),
`--python CHEMIN`, `--from CHEMIN` (dossier, archive `.tar.gz`/`.zip` ou wheel), `--bin-dir DOSSIER`
(`-Desktop`, `-Python`, `-From`, `-BinDir` sous Windows). Seul pip utilise le réseau.

**Alternative : pipx**, depuis le dépôt cloné :

```bash
pipx install .
```

(`pipx install ".[desktop]"` pour la fenêtre native.)

**Pour développer** (tests, modifications du code) :

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

Désinstaller : supprimer `~/.develop/venv` et `~/.local/bin/develop` (`%USERPROFILE%\.develop\bin` sous Windows).
La mémoire des projets reste dans `~/.develop/workspaces`.

## Lancer

```bash
develop serve
```

Démarre l'API et l'interface sur `http://127.0.0.1:8765` et ouvre le navigateur dès que le serveur répond. Si le port
est occupé, Develop prend le suivant et l'affiche. Options : `--port`, `--no-browser`, `--workspace`.

```bash
develop app
```

La même interface dans une fenêtre native, sans navigateur ; fermer la fenêtre arrête le serveur. Nécessite
[pywebview](https://pywebview.flowrl.com) (extra `desktop` : `sh install.sh --desktop`, `install.ps1 -Desktop` ou
`pip install "develop-local[desktop]"`). Sans lui, `develop app` l'explique et se replie sur le navigateur. Sous
Linux, pywebview s'appuie sur GTK/WebKit (`python3-gi gir1.2-webkit2-4.1`) ou Qt ; sous Windows, sur WebView2
(fourni avec Windows 10/11).

Au premier lancement, l'assistant de configuration vous guide en quatre étapes :

1. **Dépôts** : indiquez le dossier racine (Develop découvre les sous-projets) ou chaque dépôt avec son type.
2. **Moteur d'IA** : Develop détecte votre machine (CPU, RAM, puce Apple ou GPU, disque) et les moteurs installés, puis
   recommande des modèles adaptés avec la commande exacte à copier (`ollama pull …`) et les étapes d'installation pour
   votre système. Un modèle d'embeddings léger est toujours proposé.
3. **Mode d'exécution** : hors ligne strict (défaut) ou réseau local (serveur LLM sur une autre machine du LAN).
4. **Indexation** : première indexation complète, puis incrémentale à chaque ouverture.

En ligne de commande, `develop doctor` affiche le même diagnostic et les mêmes recommandations.

## Pourquoi pas Docker

Develop lit vos dépôts locaux, parle à Ollama sur votre machine et doit rester hors ligne : un conteneur ajouterait
montages, réseau et image lourde pour ce qui n'est qu'un paquet Python. Il s'installe donc comme un outil Python
(script ci-dessus ou pipx). Des exécutables autonomes (PyInstaller, sans Python préinstallé) sont prévus plus tard.

## Utilisation quotidienne

| Écran | Ce qu'on y fait |
|---|---|
| Chat | poser des questions, voir les sources, cliquer les citations, choisir le mode (auto, recherche seule, sans outils) |
| Explorateur | parcourir fichiers, symboles, routes (avec les correspondances frontend ↔ backend), modèles de données |
| Documents | déposer le dictionnaire de données, des spécifications (Markdown, PDF, SQL, CSV, JSON, HTML, DOCX) |
| Journal | notes datées, taguées, avec un statut (valide, remplacée, historique) |
| Impact | à partir des changements Git, d'un fichier ou d'un symbole : ce qui est potentiellement affecté, avec une confiance |
| Réglages | dépôts, moteur, mode réseau, journal des requêtes bloquées, réindexation |

CLI équivalente : `develop index`, `develop search "…"`, `develop ask "…"`, `develop impact --range HEAD~3..HEAD`,
`develop status`, `develop doctor`.

## Réglages mesurés avec un modèle réel

Test du 2026-10-21 sur un MacBook Pro M1 Max 32 Go avec `qwen3.8:27b` (Ollama 0.40, 18 Go, 262k de contexte) :

| Question | Mode | Tours LLM | Durée |
|---|---|---|---|
| « Quels champs a la table orders ? » | auto, réflexion activée | 1 | 243 s |
| « Qui appelle POST /orders/{id}/confirm ? » | auto, réflexion désactivée | 3 (2 outils `callers`) | 374 s |
| « Qu'a-t-on décidé sur le statut payé ? » | sans outils, réflexion désactivée | 1 | 64 s |

Les réponses étaient exactes et entièrement citées (champs, chaîne d'appel frontend → route → contrôleur → service →
notifier, décisions datées), y compris une incohérence détectée entre `schema.sql` et la migration. La lenteur vient du
modèle (≈ 7 tokens/s sur cette machine, prefill compris), pas de la recherche (quelques millisecondes).

Conseils :

- `think` (réflexion interne des modèles Qwen 3.x, DeepSeek-R1, gpt-oss…) est **désactivé par défaut** dans
  `LLMConfig.think` : il triple le temps de réponse pour un gain faible sur des questions sourcées. Activez-le dans
  `workspace.json` (`"llm": {"think": true}`) pour les analyses difficiles.
- Pour une question factuelle, le mode **Sans outils** (`develop ask --no-tools`) répond en un seul tour.
- Pour l'usage quotidien sur un portable, préférez un modèle de 8 à 14 milliards de paramètres (`develop doctor` en
  propose avec leur vitesse estimée) et gardez le 27B pour les analyses d'impact complexes.

### Pourquoi « Recherche dans le projet… » semblait long

Mesure du 2026-10-21 sur un index de 5 910 morceaux : la recherche elle-même (identifiants, lexical, route, sémantique,
graphe, fiches) tient en moins de 100 ms. Le temps perçu venait d'ailleurs :

| Cause | Durée mesurée | Correctif |
|---|---|---|
| Ollama déchargeait le modèle de chat après 5 min d'inactivité, rechargé à la question suivante | 28 s (modèle de 19 Go) | `keep_alive` 30 min sur chaque appel, préchauffage au démarrage de `develop serve` |
| Premier chargement de l'embedder avec une fenêtre de 40k tokens | 2 s et 4,8 Go de mémoire | fenêtre d'embedding limitée à 4 096 tokens (0,8 Go), `keep_alive` 30 min |
| Lecture du contexte par le modèle (préremplissage de 4 à 8k tokens) affichée sous « Recherche… » | 15 à 40 s selon le modèle | états « Chargement du modèle… » et « Le modèle lit N sources (≈ X tokens)… » avec compteur de secondes |

Le préremplissage reste proportionnel à la taille du modèle : un 8 à 14B lit le même contexte deux à trois fois plus
vite qu'un 27B.

## Garanties

- **Lecture seule** : Develop ne crée, ne modifie ni ne supprime aucun fichier dans vos dépôts (même Git est appelé avec
  `GIT_OPTIONAL_LOCKS=0`). Sa mémoire vit dans `~/.develop/workspaces/<espace>/` (`DEVELOP_HOME` pour changer).
- **Hors ligne** : tous les clients HTTP passent par une garde qui refuse, avant toute résolution DNS, les hôtes qui ne
  sont pas la boucle locale (mode strict) ou le réseau privé explicitement autorisé (mode réseau local). Chaque
  tentative bloquée est journalisée et visible dans l'interface.
- **Sans dépendance lourde** : FastAPI, httpx, pydantic, numpy, pypdf. Pas de Node, pas de CDN, pas de tree-sitter
  obligatoire : l'analyse de code repose sur `ast` pour Python et des extracteurs par framework pour JavaScript/TypeScript
  (React, Express, NestJS, Next.js), PHP/Laravel, SQL, Java/Spring et Dart/Flutter.
- **Secrets** : les fichiers `.env`, clés et certificats sont détectés mais jamais lus ni indexés.
- **Moteur interchangeable** : changer de modèle ne détruit pas la mémoire du projet ; changer de modèle d'embeddings
  recalcule seulement les vecteurs.

## Architecture

```
develop/
├── system/            profil machine, détection des moteurs
├── llm/               garde hors ligne, adaptateurs Ollama / OpenAI-compatible, conseiller de modèles, catalogue
├── project_scanner/   parcours des dépôts, langages, détection de stack
├── code_analysis/     extracteurs (symboles, routes, modèles, imports), normalisation des routes
├── document_indexer/  chargeurs (md, pdf, sql, csv, json, html, docx) et découpage
├── storage/           SQLite : fichiers, symboles, routes, modèles, relations, documents, notes, chunks, FTS5, embeddings
├── indexing/          indexation incrémentale et embeddings
├── knowledge_graph/   résolution des imports, appariement appels ↔ routes, liaison des modèles, mentions
├── search/            analyse de la question et recherche hybride multi-canaux
├── chat/              budget de contexte, assembleur, outils locaux, orchestrateur (SSE)
├── impact_analysis/   diff Git → symboles → propagation pondérée dans le graphe
├── api/               FastAPI (docs/API.md)
├── web/               interface statique (HTML/CSS/JS, sans build)
└── cli.py             commandes develop …
```

Les décisions détaillées sont dans [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), le contrat HTTP dans
[docs/API.md](docs/API.md), les règles du conseiller de modèles dans [docs/MODEL_RULES.md](docs/MODEL_RULES.md).

## Tests

```bash
.venv/bin/python -m pytest -q
```

La fixture `tests/fixtures/sample_project` est un mini-monorepo réaliste (React/TypeScript, Laravel, Express, docs, SQL)
utilisé par les tests de bout en bout.

## Licence

MIT.
