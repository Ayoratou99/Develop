# Règles de recommandation des modèles (recherche du 2026-10-15)

# Règles de recommandation d'un LLM local selon la machine — prêtes à coder (état au 2026-10-15)

Toutes les tailles sont celles des tags Ollama par défaut (Q4_K_M, Go décimaux). Les formules reproduisent le comportement du scheduler Ollama (llm/memory.go) et les mesures communautaires ; elles sont volontairement conservatrices (mieux vaut proposer un modèle qui tourne qu'un modèle qui swappe).

## 0. Entrées nécessaires
- `probe()` du module de détection : `ram.total`, `apple.apple_silicon`, `apple.chip`, `apple.gpu_wired_limit_mb`, `gpus[].vram_total`, `gpus[].bandwidth_gbs`, `disk.free_gb`, `ollama.*`, `docker.*`, `openai_servers`.
- Entrée du catalogue : `size_gb`, `params_b`, `active_params_b` (MoE), `moe`, `category`, `context_window`, `quality_rank`, `license`.
- Paramètres de l'outil : `num_ctx` souhaité (défaut 16384 pour un assistant de code ; 32768+ pour l'agentique), `num_parallel` (1), `kv_q8` (False).

## 1. Budget mémoire (Go) par classe de machine
```
si apple_silicon :
    mode = "unified"
    budget = iogpu.wired_limit_mb × 1 048 576 / 1e9   si la sysctl est > 0 (limite Metal relevée par l'utilisateur)
    sinon budget = RAM × 0,70  si RAM ≤ 16 Gio   (Metal ne « wire » que ~65–75 % de la RAM unifiée)
                  = RAM × 0,75  sinon
    bande_passante = table APPLE_BANDWIDTH_GBS[puce]  (M1 68 · M1 Pro 200 · M1 Max 400 · M1 Ultra 800 · M2 100 · M2 Pro 200 · M2 Max 400 · M2 Ultra 800 · M3 100 · M3 Pro 150 · M3 Max 400 · M3 Ultra 819 · M4 120 · M4 Pro 273 · M4 Max 546)
sinon si VRAM du plus gros GPU dédié ≥ 4 Go (NVIDIA via nvidia-smi, AMD via sysfs/rocm-smi, Windows via registre) :
    mode = "gpu"
    budget = VRAM − 0,5 Go                      (100 % GPU)
    budget_offload = VRAM + (RAM − 4 Go)        (offload partiel couche par couche vers le CPU)
    bande_passante = table GPU_BANDWIDTH_GBS[nom] (RTX 3060 360 · 4060 Ti 288 · 4070 504 · 4080 717 · 3090 936 · 4090 1008 · 5090 1792 · A6000 768 · A100 1555 · H100 3350 …), None si inconnu
sinon :
    mode = "cpu"
    budget = RAM − max(4 Go, 0,25 × RAM)        (laisser l'OS respirer)
    bande_passante = 38 Go/s (DDR4 dual-channel, hypothèse basse) ; 80 si DDR5 détectée ; 110 si LPDDR5x ; 150–250 pour une station 4–8 canaux
```
Retirer 2 Go du budget si `ram.available` < 25 % de `ram.total` (applications lourdes ouvertes) et le signaler.

## 2. Besoin mémoire d'un modèle
```
poids  = size_gb du catalogue (ou params_b × 0,62 pour un Q4_K_M inconnu ; × 1,07 en Q8_0 ; × 2,0 en FP16)
kv(ctx) = 0,125 × (ctx / 1000) × (params_b / 8) ^ 0,8    Go   (÷ 2 si OLLAMA_KV_CACHE_TYPE=q8_0)
         (valeurs exactes si l'architecture est connue : 7B Qwen 56 Kio/token, 8B Llama 128 Kio, 14B 192 Kio, 27–32B 256 Kio, 70B 320 Kio)
besoin(ctx) = poids × 1,10 + kv(ctx) × num_parallel + 0,75
```
Échelle de contexte essayée dans l'ordre : 131072, 65536, 32768, 16384, 8192, 4096, 2048 (bornée par `num_ctx` demandé et par `context_window` du modèle). `num_ctx_max` = le plus grand contexte tel que `besoin(ctx) ≤ budget`.

## 3. Vitesse de génération estimée (tok/s, borne par la bande passante)
```
poids_vitesse = poids                                    (dense)
              = active_params_b × 0,62 + 1,0             (MoE : seuls les experts actifs sont lus)
facteur = 0,7 (unified, gpu) ou 0,6 (cpu) ; × 0,7 pour un MoE (surcoût de routage)
tok_s = facteur × bande_passante / poids_vitesse         (None si bande passante inconnue)
```
Repères : M1 7B 12–14 tok/s · M4 7B 23 · M1 Max 32B 14 · M4 Max 32B 22 · RTX 3060 14B 18 · RTX 4080 14B 60 · RTX 4090 32B 42 · CPU DDR5 7B 11–16 · CPU DDR4 7B 5–8 · MoE 30B-A3B sur CPU DDR5 ~30. Le prefill (lecture du prompt) est borné par le calcul : les puces de base et le CPU sont lents sur les longs prompts, prévenir l'utilisateur au-delà de 16K.

## 4. Classement (trois classes)
```
embedding : "recommandé" si num_ctx_max existe (ou offload possible en mode gpu), sinon "trop gros".

si num_ctx_max est None :
    mode gpu et besoin(2048) ≤ budget_offload → "possible mais lent"
         (raison : offload partiel ; MoE = acceptable, dense = 5 à 20× plus lent)
    sinon → "trop gros" (ne tient pas même à 2K)
sinon si tok_s < 4            → "trop gros" (inutilisable au quotidien)
sinon si tok_s < 10           → "possible mais lent"
sinon si num_ctx_max < 4096   → "possible mais lent" (contexte trop étroit pour du code)
sinon                         → "recommandé"
```
Mode gpu : `full_gpu = (num_ctx_max is not None)` ; afficher « 100 % GPU » ou « offload partiel ».

## 5. Contexte conseillé à l'installation
- recommandé → `num_ctx = min(num_ctx_max, 32768, context_window)` et jamais moins de 8192 pour un assistant de code ;
- possible mais lent → `num_ctx ≤ 4096` + `OLLAMA_KV_CACHE_TYPE=q8_0` + `OLLAMA_FLASH_ATTENTION=1` ;
- toujours passer `num_ctx` explicitement dans `options` de l'API (défaut Ollama 4096 ; Ollama conseille ≥ 64000 pour les agents de code, uniquement si la mémoire le permet) ;
- recalculer `besoin` à chaque changement de `num_ctx` ou de `num_parallel`.

## 6. Ordre de proposition à l'intérieur d'une classe
1. Trier par `quality_rank` croissant puis `size_gb` décroissant (le plus gros modèle qui tient dans sa classe de qualité).
2. Préférer un MoE (qwen3.6:35b, qwen3-coder:30b, glm-4.7-flash, gpt-oss:20b) à un dense de même taille dès que la bande passante est faible (CPU, puces M de base, offload).
3. Par usage dans Develop :
   - `chat_code` (assistant principal) : toujours proposer un `chat_code` ; si aucun n'est « recommandé », proposer le meilleur `general` ou `small` recommandé.
   - `reasoning` : optionnel, uniquement si ≥ 16 Go ; filtrer les balises `<think>` de deepseek-r1 ; gpt-oss:20b a des tools, deepseek-r1 non.
   - `embedding` : toujours en installer un (indexation du dépôt) : qwen3-embedding:0.6b par défaut ; bge-m3 si documentation en français ; qwen3-embedding:4b si ≥ 16 Go ; all-minilm:33m ou nomic-embed-text si ≤ 4 Go.
   - `small` : proposer en second choix « rapide » même sur une grosse machine (autocomplétion, classification, routage : qwen2.5-coder:1.5b, granite4.2:3b).
4. Préférer les capacités `tools` pour l'agentique (exclure deepseek-r1 des usages agentiques).
5. Licence : afficher la licence ; marquer « usage commercial sous conditions » pour Llama Community, NVIDIA Open Model License, DeepSeek Model License ; « non affichée » pour ornith-1.5 ; jamais proposer codestral (MNPL, non-production) en contexte professionnel.
6. Disque : exiger `disk.free_gb ≥ size_gb × 1,1 + 1`, sinon proposer de déplacer `OLLAMA_MODELS` vers un volume plus grand avant le premier `pull`.

## 7. Grille de décision par palier (tags du catalogue, contexte 4–8K sauf mention)
### A. Apple Silicon (RAM unifiée)
| RAM | Budget | Recommandé | Possible mais lent | Trop gros |
|---|---|---|---|---|
| 8 Go | ~6 Go | qwen3.5:4b (3,3), qwen3:4b (2,5), qwen3.5:2b, nemotron-3-nano:4b, granite4.2:3b + qwen3-embedding:0.6b | gemma4:e2b (4,6) à ctx ≤ 4K ; qwen2.5-coder:7b (4,7) à ctx ≤ 2K (risque de swap) | ≥ 6 Go de poids |
| 16 Go | ~12 Go | qwen3.5:9b (6,6), ornith-1.5:9b, ministral-3:8b, qwen2.5-coder:7b, deepseek-r1:8b, granite4.2:8b, gemma4:e4b + bge-m3 / qwen3-embedding:0.6b | gemma4:12b (8,0), qwen2.5-coder:14b, deepseek-r1:14b (9,0) à ctx ≤ 8K : M4 ~12 tok/s, M1 ~5 ; gpt-oss:20b uniquement avec iogpu.wired_limit_mb relevé | ≥ 14 Go de poids |
| 24 Go | ~19 Go | gemma4:12b, qwen2.5-coder:14b, deepseek-r1:14b (ctx 32K), gpt-oss:20b (14), devstral-small-2:24b (15) à ctx ≤ 4K | qwen3.6:27b-coding / qwen3.8:27b (18) → ne tient pas sans relever la limite Metal | ≥ 19 Go de poids |
| 32 Go | ~26 Go | qwen3.6:27b-coding, qwen3.8:27b (18), qwen3-coder:30b, glm-4.7-flash (19), devstral-small-2, gpt-oss:20b, + qwen3-embedding:4b | puce de base (M1/M2/M3/M4 non Pro) : 27B dense ~4–5 tok/s → possible mais lent ; Pro ~8–11 ; Max ≥ 15 | qwen3.6:35b (24) sauf wired_limit ≥ 29 Go ; ≥ 52 Go |
| 48 Go | ~39 Go | qwen3.6:35b (24, MoE très rapide), tout ce qui précède avec ctx 32K+ | — | qwen3-coder-next (52) |
| 64 Go | ~52 Go | qwen3.6:35b, 27B denses à ctx 64K+ | qwen3-coder-next:latest (52) seulement avec `sudo sysctl iogpu.wired_limit_mb=58000` et ctx ≤ 8K | — |
| ≥ 96 Go | ~72 Go+ | qwen3-coder-next:latest (ctx 32K), gpt-oss:120b (65) | — | — |

### B. CPU seul x86 (pas de GPU dédié)
| RAM | Budget | Recommandé | Possible mais lent | Trop gros |
|---|---|---|---|---|
| ≤ 4 Go | ~0–1 Go | qwen2.5-coder:1.5b (0,99) + all-minilm:33m, uniquement si rien d'autre ne tourne | — | tout le reste |
| 8 Go | ~4 Go | qwen2.5-coder:1.5b, llama3.2:3b (2,0), granite4.2:3b (2,2), qwen3:4b (2,5) + nomic-embed-text | qwen3.5:4b (3,3), nemotron-3-nano:4b (2,8) à ctx ≤ 2K | ≥ 4,5 Go de poids |
| 16 Go | ~12 Go | 3–4B (20–35 tok/s) ; qwen2.5-coder:7b, qwen3.5:9b, ministral-3:8b sur DDR5 (11–16 tok/s) + qwen3-embedding:0.6b | 7–9B sur DDR4 (5–8 tok/s) ; gemma4:12b / 14B (2–7 tok/s) | ≥ 14 Go de poids |
| 32 Go | ~24 Go | gpt-oss:20b (14, MoE, 20–30 tok/s DDR5), qwen3-coder:30b / glm-4.7-flash (19, MoE) ; 7–9B denses | qwen2.5-coder:14b (~6 tok/s DDR5, ~3 DDR4) ; 27B denses (2–3 tok/s) | qwen3.6:35b (24) → juste, ctx ≤ 4K seulement |
| ≥ 64 Go | ~48 Go+ | qwen3.6:35b (MoE 3B actifs), gpt-oss:20b, qwen3-coder:30b | qwen3-coder-next (52, MoE 3B actifs : ~8–12 tok/s DDR5) ; 27B denses | 70B+ denses |

### C. GPU NVIDIA/AMD dédié (RAM ≥ 16 Go supposée pour l'offload)
| VRAM | Recommandé (100 % GPU) | Possible mais lent (offload partiel / ctx réduit) | Trop gros |
|---|---|---|---|
| 4 Go | qwen2.5-coder:1.5b, granite4.2:3b, llama3.2:3b + embeddings ≤ 1,3 Go | qwen3:4b, qwen3.5:4b (ctx 2K) ; 7B en offload majoritaire | ≥ 12B |
| 6 Go | qwen3:4b, qwen3.5:4b, nemotron-3-nano:4b, qwen3.5:2b | qwen2.5-coder:7b, deepseek-r1:8b (4,7–5,2) à ctx ≤ 2–4K | ≥ 12B |
| 8 Go | qwen2.5-coder:7b, deepseek-r1:8b, qwen3:8b, llama3.1:8b, granite4.2:8b à ctx ≤ 8K ; gemma4:e2b | qwen3.5:9b, ministral-3:8b, gemma4:e4b (6–6,6) à ctx ≤ 4K ; 12–14B en offload (30–40 % CPU, 5–10 tok/s) | ≥ 24B |
| 12 Go | 7–9B (ctx 32K) ; gemma4:12b, qwen2.5-coder:14b, deepseek-r1:14b à ctx ≤ 8K (RTX 3060 : ~18 tok/s) | gpt-oss:20b, devstral-small-2 en offload ; MoE 30B (19 Go) en offload (acceptable : 25–30 % CPU) | 27B denses |
| 16 Go | 14B à ctx 16–32K (60–70 tok/s sur 4080) ; gpt-oss:20b (14, ~140 tok/s sur 4080) | devstral-small-2:24b (15) à ctx ≤ 4K ou 18 % CPU (~18 tok/s) ; qwen3-coder:30b / glm-4.7-flash (19) 25–30 % CPU → 50–57 tok/s ; 27B denses ~5–10 tok/s | qwen3.6:35b, qwen3-coder-next |
| 24 Go | qwen3.6:27b-coding, qwen3.8:27b (18) à ctx ≤ 8K (3090 : ~28 tok/s, 4090 : ~42) ; qwen3-coder:30b, glm-4.7-flash (19) à ctx ≤ 4K ; 14B tout contexte | qwen3.6:35b (24) en offload léger (MoE : acceptable) | qwen3-coder-next (52, ~50 % CPU, 2–5 tok/s) |
| 32 Go | qwen3.6:35b (24) à ctx 8K, 27B denses à ctx 32K (5090 : ~75 tok/s) | qwen3-coder-next en offload (~40 % CPU, MoE : 10–15 tok/s) | 120B denses |
| 48 Go (2×24 / A6000) | 27B Q8, qwen3.6:35b ctx 64K+ | qwen3-coder-next (52) en offload léger (A6000 : ~15 tok/s) | — |
| ≥ 80 Go | qwen3-coder-next (52) tout contexte, gpt-oss:120b (65) | — | — |

## 8. Exclusions et avertissements à coder
- Ne jamais proposer : `*-cloud` (glm-5.3, deepseek-v4-pro, kimi-k3, mistral-large-4, minimax-m3 : cloud-only, pas de téléchargement) ; tags 404 ou retirés (deepseek-v4, qwen3.7, mistral-small-4, glm-5, deepseek-v3.2) ; codellama, starcoder2, deepseek-coder v1/v2, codestral (obsolètes 2024, licences restrictives) ; laguna-xs-2.1 (sorties vides signalées sur macOS/Metal).
- Avertir : `apple.rosetta == True` → « Python tourne sous Rosetta, installer un Python arm64 » ; Mac Intel → CPU seul (classe B) ; Docker Desktop sur macOS → aucun GPU, installer Ollama natif ; Windows < 10 22H2 ou macOS < 14 → Ollama non supporté.
- `ram.available` faible → proposer `ollama stop <modèle>` / fermer les applications avant le chargement.
- Multi-GPU : si le modèle tient sur un seul GPU il y est chargé entier ; sinon réparti (`OLLAMA_SCHED_SPREAD=1`). Sommer les VRAM uniquement pour le `budget_offload`.

## 9. Vérification après installation (boucle de rétrogradation)
1. `GET /api/ps` (ou `ollama ps`) après la première requête : si `size_vram < size` en mode gpu (PROCESSOR ≠ « 100% GPU ») ou si le modèle n'apparaît pas → rétrograder d'une classe ou réduire `num_ctx` d'un cran de l'échelle.
2. Mesurer les tok/s réels (`eval_count / eval_duration` de la réponse `/api/chat`) : < 4 → proposer le modèle recommandé suivant ; 4–10 → afficher « lent ».
3. Erreur `model requires more system memory (X GiB) than is available (Y GiB)` → choisir le tag plus petit suivant ou réduire `num_ctx` (coût = contexte × OLLAMA_NUM_PARALLEL).
4. Toujours définir `OLLAMA_MODELS` avant le premier `pull` si le disque système est petit (7B = 5 Go, 14B = 9 Go, 27B = 18 Go, 35B MoE = 24 Go, 80B MoE = 52 Go).

## 10. Variables d'environnement à régler pour Develop
`OLLAMA_KEEP_ALIVE=30m` (éviter le rechargement entre deux requêtes), `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` (divise la KV par 2, quasi sans perte), `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=2` (chat + embedding), `OLLAMA_CONTEXT_LENGTH=<num_ctx_max>` ; `num_thread` = cœurs physiques (P-cores) en mode cpu.

## Sources
- https://ollama.com/library/qwen3.6/tags
- https://ollama.com/library/qwen3.6:27b-coding
- https://ollama.com/library/qwen3.5/tags
- https://ollama.com/library/qwen3.8/tags
- https://ollama.com/library/qwen3-coder-next/tags
- https://ollama.com/library/qwen3-coder/tags
- https://ollama.com/library/devstral-small-2/tags
- https://ollama.com/library/glm-4.7-flash/tags
- https://ollama.com/library/ornith-1.5
- https://ollama.com/library/ornith-1.5/tags
- https://ollama.com/library/qwen2.5-coder/tags
- https://ollama.com/library/gpt-oss/tags
- https://ollama.com/library/deepseek-r1/tags
- https://ollama.com/library/qwen3/tags
- https://ollama.com/library/gemma4/tags
- https://ollama.com/library/granite4.2/tags
- https://ollama.com/library/ministral-3
- https://ollama.com/library/ministral-3/tags
- https://ollama.com/library/llama3.1/tags
- https://ollama.com/library/llama3.2/tags
- https://ollama.com/library/nemotron-3-nano
- https://ollama.com/library/nemotron-3-nano/tags
- https://ollama.com/library/qwen3-embedding/tags
- https://ollama.com/library/embeddinggemma-2/tags
- https://ollama.com/library/embeddinggemma/tags
- https://ollama.com/library/bge-m3/tags
- https://ollama.com/library/nomic-embed-text/tags
- https://ollama.com/library/all-minilm/tags
- https://ollama.com/library?sort=popular
- https://ollama.com/library?sort=newest
- https://huggingface.co/Qwen/Qwen3-Coder-Next
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct
- https://huggingface.co/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3-8B
- https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://huggingface.co/google/gemma-4-12b-it
- https://huggingface.co/zai-org/GLM-4.7-Flash
- https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- https://huggingface.co/BAAI/bge-m3
- https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://huggingface.co/mistralai/Codestral-22B-v0.1
- https://docs.ollama.com/faq
- https://docs.ollama.com/context-length
- https://docs.ollama.com/gpu
- https://docs.ollama.com/modelfile
- https://docs.ollama.com/macos
- https://docs.ollama.com/linux
- https://docs.ollama.com/windows
- https://docs.ollama.com/docker
- https://docs.ollama.com/cli
- https://docs.ollama.com/troubleshooting
- https://docs.ollama.com/api/openai-compatibility
- https://github.com/ollama/ollama/blob/main/docs/api.md
- https://github.com/ollama/ollama/blob/main/envconfig/config.go
- https://raw.githubusercontent.com/ollama/ollama/v0.6.0/llm/memory.go
- https://ollama.com/download
- https://formulae.brew.sh/formula/ollama
- https://formulae.brew.sh/cask/ollama-app
- https://hub.docker.com/r/ollama/ollama
- https://docs.docker.com/desktop/features/gpu/
- https://lmstudio.ai/docs/cli
- https://lmstudio.ai/docs/cli/server-start
- https://lmstudio.ai/docs/cli/get
- https://lmstudio.ai/docs/developer/core/server
- https://lmstudio.ai/docs/developer/core/headless
- https://lmstudio.ai/docs/developer/openai-compat/models
- https://lmstudio.ai/docs/developer/openai-compat/embeddings
- https://lmstudio.ai/docs/app/api/endpoints/rest
- https://formulae.brew.sh/cask/lm-studio
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/install.md
- https://github.com/ggml-org/llama.cpp/pull/30159
- https://github.com/ggml-org/llama.cpp/discussions/4167
- https://docs.vllm.ai/en/latest/getting_started/quickstart/
- https://docs.vllm.ai/en/latest/getting_started/installation/
- https://docs.vllm.ai/en/latest/models/pooling_models/
- https://docs.vllm.ai/en/latest/deployment/docker/
- https://jan.ai/docs/desktop/api-server
- https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
- https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap
- https://developer.apple.com/forums/thread/659846
- https://developer.apple.com/documentation/apple-silicon/about-the-rosetta-translation-environment.md
- https://eclecticlight.co/sysctl-information/
- https://www.kernel.org/doc/html/latest/filesystems/proc.html
- https://docs.kernel.org/6.16/gpu/amdgpu/driver-misc.html
- https://man.archlinux.org/man/nvidia-smi.1.en
- https://learn.microsoft.com/en-us/windows/desktop/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex
- https://learn.microsoft.com/en-ie/Windows/Win32/cimwin32prov/win32-videocontroller
- https://support.microsoft.com/help/5067470
- https://docs.python.org/3.9/library/subprocess.html
- https://docs.python.org/3.9/library/winreg.html
- https://www.glukhov.org/post/2026/01/choosing-best-llm-for-ollama-on-16gb-vram-gpu/
- https://eastondev.com/blog/en/posts/ai/20260528-ollama-hardware-guide/
- https://localaimaster.com/blog/run-llm-cpu-only
- https://www.popularai.org/p/fix-ollama-cpu-offloading-slow-inference
- https://insiderllm.com/guides/m4-max-ultra-local-llms-apple-silicon/
- https://kunalganglani.com/llm-benchmarks