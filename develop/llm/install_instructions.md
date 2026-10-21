# Installer et configurer un LLM local pour Develop — commandes vérifiées au 15 octobre 2026

Toutes les commandes proviennent des documentations officielles (voir Sources). Après chaque installation, la **vérification** attendue par Develop est la même : un serveur compatible OpenAI répond sur `GET <base_url>/v1/models` et sur `POST <base_url>/v1/chat/completions`.

| Moteur | Base URL par défaut | Port | `/v1/models` | `/v1/embeddings` | Plateformes |
|---|---|---|---|---|---|
| Ollama natif | `http://localhost:11434/v1` | 11434 | oui | oui | macOS 14+ (Apple Silicon ou Intel-CPU), Linux, Windows 10 22H2+ |
| Ollama Docker | `http://localhost:11434/v1` | 11434 | oui | oui | Linux (GPU NVIDIA/AMD), Windows WSL 2 ; **pas de GPU sur macOS** |
| LM Studio | `http://localhost:1234/v1` | 1234 | oui | oui | macOS 12+ (Apple Silicon), Windows, Linux |
| llama.cpp `llama-server` | `http://127.0.0.1:9931/v1` (**8080 avant le 9 oct. 2026** et dans Docker) | 9931 / 8080 | oui (1 modèle) | oui (`--embeddings`) | macOS, Linux, Windows |
| vLLM | `http://localhost:8000/v1` | 8000 | oui | oui (modèle d'embedding seulement) | Linux + GPU NVIDIA/AMD/Intel (pas macOS/Windows grand public) |

---

## 1. Ollama natif (recommandé par Develop)

### 1.1 macOS (14 Sonoma ou plus récent)
```bash
# Option A — application officielle (barre de menus, mises à jour automatiques)
open https://ollama.com/download/Ollama.dmg      # glisser Ollama.app dans /Applications, puis le lancer une fois
# ou script officiel
curl -fsSL https://ollama.com/install.sh | sh

# Option B — Homebrew (serveur seul, démarre à l'ouverture de session)
brew install ollama
brew services start ollama
# Option B' — application graphique via Homebrew
brew install --cask ollama-app
```
Vérification :
```bash
ollama -v
curl -s http://localhost:11434/api/version           # {"version":"0.40.x"}
curl -s http://localhost:11434/v1/models             # liste OpenAI-compatible (vide au début)
```
Variables d'environnement (application) : `launchctl setenv VAR valeur` puis **Quitter Ollama** dans la barre de menus et le relancer.
```bash
launchctl setenv OLLAMA_KEEP_ALIVE "30m"
launchctl setenv OLLAMA_FLASH_ATTENTION "1"
launchctl setenv OLLAMA_KV_CACHE_TYPE "q8_0"
launchctl setenv OLLAMA_MODELS "/Volumes/Data/ollama"   # optionnel : gros disque
```
Logs : `cat ~/.ollama/logs/server.log`. Modèles : `~/.ollama/models`.
Mac Intel : CPU uniquement (pas d'accélération Metal). Relever la limite GPU Metal (non supporté par Apple, à refaire après redémarrage) : `sudo sysctl iogpu.wired_limit_mb=26000` (`=0` pour revenir au défaut, garder 8–16 Go pour macOS).

Désinstallation : `sudo rm -rf /Applications/Ollama.app /usr/local/bin/ollama && rm -rf ~/.ollama` (supprime aussi les modèles).

### 1.2 Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh                    # installe le binaire + service systemd + détecte NVIDIA/AMD
# version précise :  curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION=0.40.4 sh
# mise à jour : relancer le même script
```
Installation manuelle (sans script) :
```bash
sudo rm -rf /usr/lib/ollama
curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | sudo tar x -C /usr
# GPU AMD : paquet ROCm en plus
curl -fsSL https://ollama.com/download/ollama-linux-amd64-rocm.tar.zst | sudo tar x -C /usr
# ARM64 : ollama-linux-arm64.tar.zst
ollama serve                                   # puis, dans un autre terminal : ollama -v
```
Vérification et service :
```bash
systemctl status ollama
curl -s http://localhost:11434/api/version
nvidia-smi                                     # GPU NVIDIA visible ? (driver ≥ 550 ; CUDA)
journalctl -e -u ollama                        # logs ; --follow pour suivre
```
Variables d'environnement (service) :
```bash
sudo systemctl edit ollama.service
# ajouter sous [Service] :
#   Environment="OLLAMA_MODELS=/data/ollama"
#   Environment="OLLAMA_KEEP_ALIVE=30m"
#   Environment="OLLAMA_FLASH_ATTENTION=1"
#   Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
sudo chown -R ollama:ollama /data/ollama        # si OLLAMA_MODELS est déplacé
sudo systemctl daemon-reload && sudo systemctl restart ollama
```
Modèles par défaut : `/usr/share/ollama/.ollama/models` (utilisateur `ollama`).
Désinstallation : `sudo systemctl stop ollama && sudo systemctl disable ollama && sudo rm /etc/systemd/system/ollama.service && sudo rm -r $(which ollama | tr 'bin' 'lib') && sudo rm $(which ollama) && sudo userdel ollama && sudo groupdel ollama && sudo rm -r /usr/share/ollama`.

### 1.3 Windows (10 22H2 / 11)
```powershell
# Option A — installateur officiel (sans droits admin) : https://ollama.com/download/OllamaSetup.exe
#   emplacement personnalisé : OllamaSetup.exe /DIR="D:\Ollama"
# Option B — script officiel PowerShell
irm https://ollama.com/install.ps1 | iex
# Option C — paquet winget communautaire (non officiel)
winget install --id Ollama.Ollama -e
```
Prérequis GPU : pilote NVIDIA 551.61+ ; AMD : ROCm v7/HIP ou pilote Radeon Vulkan.
Vérification :
```powershell
ollama -v
Invoke-RestMethod http://localhost:11434/api/version
Invoke-RestMethod http://localhost:11434/v1/models
nvidia-smi
```
Variables d'environnement : quitter Ollama (icône de la zone de notification) → Paramètres → « Modifier les variables d'environnement pour votre compte » → ajouter `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE` → OK → relancer Ollama depuis le menu Démarrer.
Modèles : `%USERPROFILE%\.ollama\models`. Logs : `%LOCALAPPDATA%\Ollama\server.log`. Debug : `$env:OLLAMA_DEBUG="1"` puis relancer.

### 1.4 Télécharger, lister, tester, supprimer (toutes plateformes)
```bash
ollama pull qwen3.5:9b                 # modèle de chat/code conseillé par Develop
ollama pull qwen3-embedding:0.6b       # embeddings pour l'index du dépôt
ollama list                            # modèles sur disque (alias : ollama ls)
ollama run qwen3.5:9b --verbose        # chat interactif ; --verbose affiche les tok/s
ollama ps                              # modèles chargés : colonne PROCESSOR = 100% GPU / 100% CPU / 48%/52% CPU/GPU, colonne CONTEXT
ollama stop qwen3.5:9b                 # décharge de la mémoire
ollama rm qwen3.5:9b                   # supprime du disque
ollama serve --help                    # liste complète des variables d'environnement
```
Équivalents API (utilisés par Develop) :
```bash
curl http://localhost:11434/api/tags                                   # modèles locaux (name, size, details.parameter_size, quantization_level)
curl http://localhost:11434/api/pull -d '{"model": "qwen3.5:9b"}'       # téléchargement (flux JSON de progression)
curl http://localhost:11434/api/ps                                     # modèles chargés + size_vram
curl -X DELETE http://localhost:11434/api/delete -d '{"model": "qwen3.5:9b"}'
curl http://localhost:11434/api/embed -d '{"model": "qwen3-embedding:0.6b", "input": "def hello(): pass"}'
```
Fixer le contexte (le défaut est 4096 ; Ollama conseille ≥ 64 000 pour les agents de code **si la mémoire le permet**) :
```bash
# par requête (API native) : "options": {"num_ctx": 16384}
curl http://localhost:11434/api/chat -d '{"model": "qwen3.5:9b", "stream": false, "options": {"num_ctx": 16384},
  "messages": [{"role": "user", "content": "Explique cette fonction : def f(x): return x*2"}]}'
# globalement : OLLAMA_CONTEXT_LENGTH=16384 (variable d'environnement du serveur)
# ou par Modelfile :  printf 'FROM qwen3.5:9b\nPARAMETER num_ctx 16384\n' > Modelfile && ollama create qwen3.5-16k -f Modelfile
```
API compatible OpenAI : base `http://localhost:11434/v1/`, clé obligatoire mais ignorée (`api_key="ollama"`). Endpoints : `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/responses` (≥ 0.13.3). Limites : pas de `num_ctx` via l'API OpenAI (utiliser un Modelfile ou `OLLAMA_CONTEXT_LENGTH`), images en base64 seulement, `tool_choice`/`logit_bias`/`n` ignorés.
```bash
curl http://localhost:11434/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "Dis bonjour"}]}'
```

---

## 2. Ollama via Docker

### 2.1 Linux
```bash
# CPU seul
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# GPU NVIDIA : 1) NVIDIA Container Toolkit (Debian/Ubuntu ; RHEL/Fedora : yum install -y nvidia-container-toolkit)
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --gpus all ubuntu nvidia-smi                       # test
# 2) Ollama avec GPU
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# GPU AMD (ROCm)
docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama:rocm
```
Utilisation et vérification :
```bash
docker exec -it ollama ollama pull qwen3.5:9b
docker exec -it ollama ollama list
docker exec -it ollama ollama ps
docker logs ollama
curl -s http://localhost:11434/api/version
```
Le volume nommé `ollama` conserve les modèles entre recréations. Variables d'environnement : `-e OLLAMA_KEEP_ALIVE=30m -e OLLAMA_FLASH_ATTENTION=1 -e OLLAMA_KV_CACHE_TYPE=q8_0` sur la ligne `docker run`. Si le GPU retombe en CPU après une mise à jour, désactiver la gestion cgroup systemd dans `/etc/docker/daemon.json` (doc Ollama) ; AMD : ajouter `--group-add` avec les GID de `ls -lnd /dev/kfd /dev/dri /dev/dri/*`.

### 2.2 Windows
Docker Desktop + WSL 2 + pilote NVIDIA : mêmes commandes qu'en Linux avec `--gpus=all` (seul passthrough GPU supporté par Docker Desktop). Préférer l'installateur natif (§1.3) sur un poste de travail.

### 2.3 macOS
**Pas d'accélération GPU dans Docker Desktop sur Mac** (FAQ Ollama : absence de passthrough Metal). Installer Ollama nativement (§1.1) ; n'utiliser Docker que pour des frontaux (Open WebUI…) qui joignent l'hôte via `http://host.docker.internal:11434` après `launchctl setenv OLLAMA_HOST "0.0.0.0:11434"`.

---

## 3. LM Studio (interface graphique + serveur local)

Installation : https://lmstudio.ai (macOS 12+ Apple Silicon, Windows, Linux) ; macOS : `brew install --cask lm-studio`. **Lancer l'application une fois** avant d'utiliser le CLI `lms` (installé dans `~/.lmstudio/bin/lms`, pas toujours dans le PATH). Headless (serveur seul, « llmster ») : `curl -fsSL https://lmstudio.ai/install.sh | bash` (Linux/macOS) ou `irm https://lmstudio.ai/install.ps1 | iex` (Windows), puis `lms daemon up`.
```bash
# Serveur : onglet Developer → « Start server », ou :
lms server start --port 1234            # --cors si une appli web doit l'appeler ; --bind 0.0.0.0 pour le réseau local
lms server status
# Modèles
lms get qwen2.5-coder-7b-instruct@q4_k_m   # téléchargement depuis le catalogue LM Studio (filtre --gguf / --mlx)
lms ls                                      # modèles sur disque
lms load <identifiant> --gpu=max --context-length=16384
lms ps                                      # modèles chargés
lms unload --all
lms server stop
```
Vérification :
```bash
curl http://localhost:1234/v1/models
curl http://localhost:1234/api/v0/models                        # état loaded/not-loaded, quantization, max_context_length (déprécié en 0.4 : /api/v1/models)
curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "<identifiant affiché par lms ls>", "messages": [{"role": "user", "content": "Dis bonjour"}]}'
curl http://localhost:1234/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model": "<modèle d embedding chargé>", "input": "Bonjour"}'
```
SDK OpenAI : `base_url="http://localhost:1234/v1"`, `api_key="lm-studio"`. Aucune authentification par défaut (activable : Server Settings → Require Authentication → Manage Tokens, en-tête `Authorization: Bearer $LM_API_TOKEN`). Réglages utiles : « Just in Time Model Loading » (charge à la première requête), « Auto Unload Unused JIT Models ». Pour choisir une variante : Discover → chercher le modèle → prendre une quantisation Q4_K_M dont la taille ≤ budget mémoire de Develop.

---

## 4. llama.cpp — `llama-server`

Installation : `brew install llama.cpp` (macOS/Linux) · `winget install llama.cpp` (Windows) · `sudo port install llama.cpp` · `nix profile install nixpkgs#llama-cpp` · ou binaires des releases GitHub (0.7.0 publiée le 15/10/2026).
```bash
# Télécharger + servir depuis Hugging Face (-hf <org>/<repo>[:quant]) ; -ngl all = tout sur GPU/Metal ; -c = contexte
llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M -c 16384 -ngl all --host 127.0.0.1 --port 9931
llama-server -hf ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M -c 16384 -ngl all
# GGUF local
llama-server -m /chemin/modele.gguf -c 16384 -ngl all --alias mon-modele --api-key ma-cle   # Windows : llama-server.exe -m C:\modeles\x.gguf

# Modèle d'embedding (flag --embeddings ou --embedding ; --pooling mean|cls|last si nécessaire)
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --port 9932
```
**Port par défaut : 9931 depuis le 9 oct. 2026 (PR #30159), 8080 avant et dans l'image Docker.** Develop sonde les deux ; détecter localement avec `llama-server --help | grep -- --port` (variable `LLAMA_ARG_PORT`).
Vérification :
```bash
curl http://127.0.0.1:9931/health
curl http://127.0.0.1:9931/v1/models          # un seul élément en mode mono-modèle (id = chemin -m ou --alias)
curl http://127.0.0.1:9931/props              # n'existe que chez llama-server (sert à le distinguer de LM Studio)
curl http://127.0.0.1:9931/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "x", "messages": [{"role": "user", "content": "Dis bonjour"}]}'
curl http://127.0.0.1:9932/v1/embeddings -H "Content-Type: application/json" -d '{"model": "x", "input": "Bonjour"}'
```
`/v1/embeddings` exige un pooling ≠ `none` (sinon utiliser `/embeddings`, non-OpenAI). Mode routeur : `llama-server` sans `-m` expose `POST /models/load` (`{"model": "ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"}`) et `GET /models`.
Docker :
```bash
docker run -p 8080:8080 -v /chemin/modeles:/models ghcr.io/ggml-org/llama.cpp:server -m /models/x.gguf -c 8192 --host 0.0.0.0 --port 8080
docker run --gpus all -p 8080:8080 -v /chemin/modeles:/models ghcr.io/ggml-org/llama.cpp:server-cuda -m /models/x.gguf -c 8192 --host 0.0.0.0 --port 8080 --n-gpu-layers 99
```

---

## 5. vLLM (Linux + GPU, serveur haut débit)

Prérequis : Linux, Python 3.11–3.14, GPU NVIDIA (CUDA) / AMD ROCm / Intel XPU ; builds CPU x86/ARM ; Apple Silicon seulement via le projet séparé vLLM-Metal. Non adapté à un poste macOS/Windows grand public.
```bash
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install vllm --torch-backend=auto
vllm download-kernels

vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --host 127.0.0.1 --port 8000 --max-model-len 16384   # --api-key ma-cle ou VLLM_API_KEY
# Embeddings : servir séparément un modèle d'embedding (runner pooling détecté automatiquement)
vllm serve Qwen/Qwen3-Embedding-0.6B --port 8001
```
Vérification :
```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-Coder-7B-Instruct", "messages": [{"role": "user", "content": "Dis bonjour"}]}'
curl http://localhost:8001/v1/embeddings -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Embedding-0.6B", "input": "Bonjour"}'
```
Docker :
```bash
docker run --runtime nvidia --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model Qwen/Qwen2.5-Coder-7B-Instruct
```
vLLM charge les poids HF en BF16/FP16 par défaut (7B ≈ 15 Go de VRAM) : ajouter `--quantization awq|gptq` avec un dépôt quantisé, ou rester sur Ollama/llama.cpp en dessous de 24 Go de VRAM.

---

## 6. Pièges connus et remèdes

1. **Port déjà occupé** — `listen tcp 127.0.0.1:11434: bind: address already in use` = un second `ollama serve` alors que l'app/le service tourne déjà. Diagnostic : `lsof -i :11434` (macOS/Linux), `sudo ss -lntp | grep 11434` (Linux), `netstat -ano | findstr "11434"` (Windows). Remède : ne pas relancer `ollama serve`, ou `OLLAMA_HOST=127.0.0.1:11435 ollama serve` et pointer Develop dessus. Même logique pour 1234 (LM Studio `--port`), 9931/8080 (llama-server), 8000 (vLLM), 1337 (Jan).
2. **Modèle qui ne tient pas** — `model requires more system memory (X GiB) than is available (Y GiB)` : prendre un tag plus petit, réduire `num_ctx` (coût = contexte × `OLLAMA_NUM_PARALLEL`), `ollama stop <modèle>`, fermer les applications gourmandes. `ollama ps` montre la part CPU/GPU : un modèle partiellement sur CPU est 5 à 20× plus lent (dense).
3. **Ollama n'écoute que sur 127.0.0.1** — pour un conteneur ou une autre machine : `OLLAMA_HOST=0.0.0.0:11434` (macOS `launchctl setenv` + relance de l'app ; Linux `systemctl edit` ; Windows variables utilisateur). Les app récentes ont aussi un interrupteur « Expose Ollama to the network ».
4. **CORS** — Ollama accepte `127.0.0.1`/`0.0.0.0` par défaut ; ajouter les autres origines via `OLLAMA_ORIGINS=*` (403 sinon). LM Studio : CORS désactivé par défaut (`--cors`). Jan : activé par défaut.
5. **Docker Desktop sur Mac = pas de GPU** — installer Ollama/LM Studio/llama.cpp nativement (Metal).
6. **Contexte par défaut 4096** — toujours passer `num_ctx` (API native) ou `OLLAMA_CONTEXT_LENGTH` ; l'API OpenAI d'Ollama ne permet pas de le régler.
7. **llama-server : changement de port 8080 → 9931** (9 oct. 2026) — lire `llama-server --help` ou sonder les deux ports.
8. **Mac Intel** : CPU seulement ; **macOS < 14** et **Windows < 10 22H2** : Ollama non supporté ; **`lms`** exige d'avoir lancé LM Studio une fois.

---

## 7. Appel minimal depuis Develop (Python 3.9, stdlib)
```python
import json, urllib.request

def chat(prompt, model="qwen3.5:9b", base="http://127.0.0.1:11434", num_ctx=16384, timeout=600):
    body = json.dumps({"model": model, "stream": False, "options": {"num_ctx": num_ctx},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base + "/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    tok_s = data["eval_count"] / (data["eval_duration"] / 1e9) if data.get("eval_duration") else None
    return data["message"]["content"], tok_s      # tok_s réel -> boucle de rétrogradation (règles §9)

def embed(texts, model="qwen3-embedding:0.6b", base="http://127.0.0.1:11434", timeout=120):
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(base + "/api/embed", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["embeddings"]
```
Pour LM Studio / llama-server / vLLM, utiliser `POST <base_url>/v1/chat/completions` avec le même corps au format OpenAI (`{"model", "messages"}`) et `POST <base_url>/v1/embeddings` (`{"model", "input"}`).