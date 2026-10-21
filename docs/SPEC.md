# Develop — spécification d'intention

Condensé des échanges avec l'utilisateur (2026-10-15), qui ont guidé la construction. Les contrats techniques sont
dans `docs/API.md` et `docs/ARCHITECTURE.md`.

## Problème
Quand un projet devient gros, un développeur qui modifie quelque chose oublie les éléments affectés par son
changement. Un frontendeur qui doit créer une interface doit déranger le backendeur pour connaître les données de tel
ou tel modèle.

## Idée
Un tout petit outil local, « Develop », où le développeur dépose le dictionnaire de données, les documents de
conception de tout genre (PDF, Markdown, SQL, specs) et des notes internes datées (journal), et une interface
similaire à ChatGPT où il branche un modèle LLM local (téléchargé via Ollama, Docker ou autre) capable de lire les
documents, et où il pose ses questions.

## Contraintes fortes
- Local-first, fonctionne entièrement HORS LIGNE en usage quotidien. Aucune API cloud.
- Le développeur ouvre Develop, indique le(s) repo(s) locaux (frontend, backend, docs, database…), configure le LLM.
- Develop ne modifie jamais les fichiers des repos (lecture seule).
- Docker ne doit pas être obligatoire.
- Le modèle d'embeddings doit aussi être local ; s'il est absent, la recherche doit quand même fonctionner.
- Contrôle de conformité hors ligne : toute tentative de joindre un service distant est bloquée et signalée.
- Tout en Python ; pas de build Node/React ; UI web statique servie par le moteur Python, sans CDN.

## Parcours au premier lancement
1. Ouvrir Develop : vérifie moteur LLM, modèles disponibles, espace disque.
2. Sélectionner les dépôts locaux (frontend, backend, autres). Analyse sans modifier.
3. Configurer le LLM local (Ollama, service Docker, autre serveur compatible API OpenAI). Vérifier l'accès.
4. Indexer : fichiers, symboles, modèles, contrats API, documents, relations, historique Git.
5. Travailler hors ligne : questions, dépendances, analyse de changements. Tout reste sur la machine.

## Écran de configuration
A. Dépôts du projet (chemins frontend/backend/… ; sélecteur de dossier)
B. Moteur d'IA : Ollama | service LLM dans Docker | autre serveur local compatible OpenAI ; choix du modèle
C. Mode d'exécution : hors ligne strict | local avec services réseau internes

## Multi-dépôts
Vue unifiée : le frontend appelle tel endpoint → tel service → tel modèle → telles règles métier/migrations →
documentées par telles notes. Correspondances via contrats API, URL, types partagés, configs ; ces relations ne sont
pas toujours explicites.

## Trois couches indépendantes
Projet source (lecture seule) · Mémoire Develop (index, relations, notes, analyses, historique) · Moteur IA (LLM local,
remplaçable sans reconstruire la mémoire). Index incrémental.

## Conseiller de modèles selon la machine (ajout du 2026-10-15)
Profil machine (OS, puce, cœurs, RAM, GPU, disque, moteurs présents) × catalogue local de modèles (tag Ollama, taille,
RAM minimale, usage, contexte, licence) → classement recommandé / possible mais lent / trop gros, avec la raison
chiffrée, la commande `ollama pull …` et des étapes d'installation adaptées à l'OS. Accessible dans l'assistant et via
`develop doctor`. Rien ne requiert Internet sauf le téléchargement du modèle.

## Gestion du contexte : « ne pas avaler bêtement tout le projet » (ajout du 2026-10-15)
Règle fondamentale : le LLM ne lit jamais tout le projet ; la fenêtre de contexte est un budget.
1. Budget calculé : fenêtre réelle du modèle (`/api/show`), plafond selon la RAM ; répartition consignes / contexte
   récupéré (adaptatif par type de question) / historique / réserve de réponse ; chaque passage porte un [Sn].
2. Recherche multi-canaux : identifiants exacts, lexical (FTS5), sémantique (embeddings locaux, optionnel), structurel
   (1 saut dans le graphe), temporel (notes par date) ; fusion RRF, déduplication, diversité, remplissage du budget.
3. Boucle d'outils bornée : search, read_file (fenêtre limitée), symbol, callers, endpoint, model, notes, impact,
   list_dir ; outils natifs ou protocole JSON texte ; au plus 4 tours ; budget cumulé ; signalement si épuisé.
4. Synthèses hiérarchiques optionnelles (résumés par fichier/module, invalidés par l'indexation incrémentale).
5. Indexation incrémentale (sha1 + mtime + taille + commit) et détection de dérive doc/code (mention d'un symbole,
   d'une table ou d'un champ qui n'existe plus) ; notes avec statut valide / remplacée / historique.
6. Découpage : documents 300-700 tokens avec chevauchement ; code par fonction/classe ; note = une unité.
