# Intégration continue (GitHub Actions)

Le workflow `.github/workflows/ci.yml` vérifie, à chaque `push` et `pull_request` sur `main` (et à la demande via
*Run workflow*), que Develop s'installe et fonctionne sur **Linux, macOS et Windows**. Le badge sous le titre du
README reflète le dernier résultat sur `main`. Une seule exécution par branche à la fois : un nouveau push annule la
précédente.

## Ce que la CI vérifie

| Job | Où | Quoi |
|---|---|---|
| `tests (os, python)` | ubuntu-latest, macos-latest, windows-latest × Python 3.9 et 3.12 (six combinaisons, aucune n'arrête les autres) | `pip install -e ".[dev]"` puis `pytest tests -o addopts='' -q --durations=10`, avec `PYTHONUTF8=1` et `DEVELOP_HOME` dans le dossier temporaire du runner (rien n'est écrit dans `~/.develop`). Sous Windows, l'étape tourne dans `pwsh`. Node 24 est installé pour les harness web (`tests/web_harness/*.mjs`, qui exigent Node ≥ 22.15 : sans lui, les tests `test_web_*` seraient ignorés). |
| `install-scripts (ubuntu-latest)` et `(macos-latest)` | Linux, macOS | `sh install.sh` avec `HOME` redirigé vers un dossier temporaire, puis `~/.local/bin/develop --version`, `develop doctor --json`, `develop status --json`, et enfin `develop serve --port 8765 --no-browser` en arrière-plan jusqu'à ce que `GET http://127.0.0.1:8765/api/status` réponde 200 (30 s au plus), puis arrêt du serveur. |
| `install-scripts (windows-latest)` | Windows | `powershell -ExecutionPolicy Bypass -File install.ps1` sous **Windows PowerShell 5.1** (celui de tous les Windows 10/11), avec `DEVELOP_HOME` redirigé vers un dossier temporaire ; puis `develop.cmd --version`, `develop doctor --json`, `develop status --json`, `develop --version` depuis le PATH, et `python -m develop serve --port 8765 --no-browser` en arrière-plan avec `Invoke-WebRequest` sur `/api/status` (30 s au plus), puis arrêt du processus. |
| `report-failure` | ubuntu-latest | Si un job ci-dessus échoue : crée ou met à jour l'issue **« CI : échec »** (voir ci-dessous). |
| `report-success` | ubuntu-latest | Si tout est vert : ferme l'issue « CI : échec » si elle est ouverte, avec un commentaire qui cite l'exécution. |

Chaque job `tests` et `install-scripts` écrit sa sortie complète dans un fichier `output.log` (via `tee` /
`Tee-Object`) publié comme artefact de l'exécution, même en cas d'échec : `pytest-<os>-<python>` et `install-<os>`
(conservés 14 jours). Le job Windows y ajoute `serve.out.log` et `serve.err.log` (sortie du serveur) ; les jobs Linux
et macOS, `serve.log` et `status.json`.

Les runners n'ont pas d'Ollama : `develop doctor` et `develop status` rapportent « aucun moteur », ce qui est le
comportement attendu. L'extra `desktop` (pywebview) n'est pas installé en CI.

## Lire un échec : l'issue automatique

Quand au moins un job échoue, le job `report-failure` télécharge tous les artefacts et crée (ou met à jour, si elle est
déjà ouverte) une issue intitulée **« CI : échec »**. Son corps contient :

- le lien vers l'exécution, l'événement, la branche et le commit ;
- pour chaque job en échec : son nom, le lien vers son journal, l'étape qui a échoué et les **150 dernières lignes** de
  son `output.log` (ou une mention si le job a échoué avant de produire l'artefact, par exemple à l'installation des
  dépendances).

Le dépôt étant public, l'issue (comme les journaux) se lit sans jeton : c'est le point d'entrée pour corriger, y compris
pour un assistant qui n'a pas accès à une machine Windows. Le même compte rendu figure dans le résumé de l'exécution
(onglet *Summary*). À chaque nouvel échec, le corps est remplacé et un commentaire cite la nouvelle exécution ; dès que
la CI repasse au vert, `report-success` ferme l'issue.

Si le jeton de l'exécution est en lecture seule (pull request venant d'un fork), l'issue ne peut pas être écrite : un
avertissement l'indique et le compte rendu reste disponible dans le résumé de l'exécution.

Correspondance entre un job et son artefact : `tests (ubuntu-latest, 3.9)` → `pytest-ubuntu-latest-3.9` ;
`install-scripts (windows-latest)` → `install-windows-latest`.

## Lancer les mêmes vérifications localement

Tests (depuis le dépôt, environnement de développement : `pip install -e ".[dev]"`) :

```bash
PYTHONUTF8=1 DEVELOP_HOME="$(mktemp -d)" python -m pytest tests -o addopts='' -q --durations=10
```

```powershell
$env:PYTHONUTF8 = "1"; $env:DEVELOP_HOME = Join-Path $env:TEMP "develop-home"
python -m pytest tests -o addopts='' -q --durations=10
```

Les tests `test_web_*` demandent Node ≥ 22.15 dans le PATH (sinon ils sont ignorés) ; `test_install_scripts.py`
demande `sh` (Git Bash suffit sous Windows) et, pour l'analyse syntaxique d'`install.ps1`, `pwsh` ou `powershell`.

Parcours d'installation sans toucher à votre `~/.develop` (macOS / Linux) :

```bash
export HOME="$(mktemp -d)"          # tout va dans ce dossier jetable (venv, commande, données)
sh install.sh
"$HOME/.local/bin/develop" --version
"$HOME/.local/bin/develop" doctor --json
"$HOME/.local/bin/develop" serve --port 8765 --no-browser &
sleep 5; curl -fsS http://127.0.0.1:8765/api/status; kill %1
```

Sous Windows (PowerShell) :

```powershell
$env:DEVELOP_HOME = Join-Path $env:TEMP "develop-home"   # venv, commande et données dans ce dossier
powershell -ExecutionPolicy Bypass -File install.ps1
& "$env:DEVELOP_HOME\bin\develop.cmd" --version
& "$env:DEVELOP_HOME\bin\develop.cmd" doctor --json
$p = Start-Process "$env:DEVELOP_HOME\venv\Scripts\python.exe" -ArgumentList '-m','develop','serve','--port','8765','--no-browser' -PassThru -NoNewWindow
Start-Sleep 5; Invoke-WebRequest http://127.0.0.1:8765/api/status -UseBasicParsing; Stop-Process $p
```

Note : `install.ps1` ajoute son dossier `bin` au PATH utilisateur (registre `HKCU\Environment`), y compris lors d'un
essai avec `DEVELOP_HOME` redirigé ; retirez l'entrée si vous ne gardez pas cet essai.

Vérifier le fichier du workflow après modification :

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

## Limites

- `install.sh` a été validé sur macOS en local ; `install.ps1` n'est exécuté **que** par la CI (pas de PowerShell sur
  la machine de développement) : l'issue « CI : échec » est la source de vérité pour Windows.
- La CI ne teste ni le chat avec un modèle (pas de LLM sur les runners) ni la fenêtre native `develop app`.
- `macos-latest` est un runner Apple Silicon ; `ubuntu-latest` et `windows-latest` suivent les images courantes de
  GitHub, dont les versions de Python préinstallées peuvent changer (les scripts choisissent le premier Python ≥ 3.9 du
  PATH, comme chez l'utilisateur).
