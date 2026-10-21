"""Tests du conseiller de modèles (develop.llm.advisor) : profils construits à la main, aucun réseau."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from develop.llm import advisor
from develop.llm.advisor import (Advice, InstallStep, Recommendation, VERDICT_POSSIBLE_SLOW, VERDICT_RECOMMENDED,
                                 VERDICT_TOO_BIG, advise, describe_tier, estimate_speed, format_advice_text,
                                 load_catalog)
from develop.llm.base import ModelInfo
from develop.system.engines import EngineProbe, EnginesReport
from develop.system.profile import GpuInfo, MachineProfile


# --------------------------------------------------------------------------- profils et moteurs

def _profile(**kw) -> MachineProfile:
    base = dict(os="linux", os_version="", arch="x86_64", cpu_model="Intel Core i7-10700", cpu_cores_physical=8,
                cpu_cores_logical=16, ram_total_gb=16.0, ram_available_gb=10.0, disk_free_gb=200.0, disk_path="/home")
    base.update(kw)
    return MachineProfile(**base)


def m1_max_32() -> MachineProfile:
    return _profile(os="macos", arch="arm64", cpu_model="Apple M1 Max", cpu_cores_physical=10, cpu_cores_logical=10,
                    ram_total_gb=32.0, ram_available_gb=20.0, apple_silicon=True, unified_memory=True,
                    gpus=[GpuInfo("apple", "Apple M1 Max", None, 32)])


def laptop_cpu_8() -> MachineProfile:
    return _profile(os="linux", cpu_model="Intel Core i5-8250U", cpu_cores_physical=4, cpu_cores_logical=8,
                    ram_total_gb=8.0, ram_available_gb=5.0)


def linux_rtx4090() -> MachineProfile:
    return _profile(os="linux", cpu_model="AMD Ryzen 9 7950X", ram_total_gb=64.0, ram_available_gb=48.0,
                    gpus=[GpuInfo("nvidia", "NVIDIA GeForce RTX 4090", 24.0)])


def windows_16_no_gpu() -> MachineProfile:
    return _profile(os="windows", cpu_model="Intel Core i7-10700", ram_total_gb=16.0, ram_available_gb=9.0,
                    gpus=[GpuInfo("intel", "Intel UHD Graphics 630", None)])


def unknown_ram() -> MachineProfile:
    return _profile(os="linux", cpu_model="", ram_total_gb=None, ram_available_gb=None, disk_free_gb=None)


def rtx3070_8() -> MachineProfile:
    return _profile(os="linux", cpu_model="AMD Ryzen 5 5600X", ram_total_gb=16.0, ram_available_gb=10.0,
                    gpus=[GpuInfo("nvidia", "NVIDIA GeForce RTX 3070", 8.0)])


def _ollama_probe(reachable: bool, models=()) -> EngineProbe:
    return EngineProbe(engine="ollama", label="Ollama", base_url="http://127.0.0.1:11434", reachable=reachable,
                       version="0.12.0" if reachable else "", models=[ModelInfo(name=m) for m in models],
                       error="" if reachable else "refused")


def _engines(ollama_binary, reachable, models=(), docker=False, container=None, openai=()) -> EnginesReport:
    servers = [EngineProbe(engine="openai_compat", label=label, base_url=url, reachable=True,
                           models=[ModelInfo(name="m")]) for label, url in openai]
    return EnginesReport(ollama_binary=ollama_binary, ollama_models_dir="~/.ollama/models",
                         ollama=_ollama_probe(reachable, models), openai_servers=servers, docker_available=docker,
                         docker_running=docker, docker_ollama_container=container,
                         recommended_engine="ollama" if reachable else "none",
                         recommended_base_url="http://127.0.0.1:11434" if reachable else "")


def _by_tag(advice: Advice, tag: str) -> Recommendation:
    for r in advice.recommendations + advice.embedding:
        if r.tag == tag:
            return r
    raise AssertionError("tag absent du conseil : %s" % tag)


def _step(advice: Advice, title_prefix: str):
    for s in advice.install_steps:
        if s.title.startswith(title_prefix):
            return s
    return None


# --------------------------------------------------------------------------- catalogue

def test_load_catalog_reads_package_file_and_caches():
    cat = load_catalog()
    assert cat["version"] and len(cat["models"]) >= 30
    tags = {m["tag"] for m in cat["models"]}
    assert {"qwen2.5-coder:7b", "qwen3-coder-next:latest", "nomic-embed-text:v1.5", "all-minilm:33m"} <= tags
    assert cat["models"] is not load_catalog()["models"]  # copie : l'appelant ne peut pas corrompre le cache
    categories = {m["category"] for m in cat["models"]}
    assert categories == {"chat_code", "reasoning", "general", "small", "embedding"}


def test_load_catalog_missing_file_gives_empty_catalog_with_warning(tmp_path: Path):
    with pytest.warns(UserWarning):
        cat = load_catalog(tmp_path / "absent.json")
    assert cat == {"version": "", "models": []}
    advice = advise(m1_max_32(), catalog=cat)  # le conseiller continue sans modèle
    assert advice.recommendations == [] and advice.best_chat is None and advice.best_embedding is None
    assert _step(advice, "Installer Ollama") is not None
    assert format_advice_text(advice)


def test_load_catalog_invalid_json_warns(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.warns(UserWarning):
        assert load_catalog(bad)["models"] == []


def test_catalog_excludes_cloud_and_obsolete_tags():
    cat = {"models": [
        {"tag": "glm-5.3-cloud", "category": "chat_code", "size_gb": 1, "params_b": 1, "quality_rank": 1},
        {"tag": "codellama:7b", "category": "chat_code", "size_gb": 4, "params_b": 7, "quality_rank": 1},
        {"tag": "codestral:22b", "category": "chat_code", "size_gb": 13, "params_b": 22, "quality_rank": 1},
        {"tag": "qwen2.5-coder:7b", "category": "chat_code", "size_gb": 4.7, "params_b": 7.6, "quality_rank": 3,
         "context_window": 32768, "capabilities": ["tools"]},
    ]}
    advice = advise(m1_max_32(), catalog=cat)
    assert [r.tag for r in advice.recommendations] == ["qwen2.5-coder:7b"]


# --------------------------------------------------------------------------- règles mémoire / vitesse

def test_kv_cache_and_memory_need_follow_model_rules():
    # kv(ctx) = 0,125 × (ctx/1000) × (params/8)^0,8 ; besoin = poids × 1,10 + kv + 0,75
    assert advisor.kv_cache_gb(16000, 8.0) == pytest.approx(2.0)
    model = {"size_gb": 18.0, "params_b": 27.0}
    kv = advisor.kv_cache_gb(16384, 27.0)
    assert advisor.memory_needed_gb(model, 16384) == pytest.approx(18.0 * 1.10 + kv + 0.75)
    assert advisor.memory_needed_gb(model, 2048) < advisor.memory_needed_gb(model, 16384)


def test_describe_tier_by_machine_class():
    assert describe_tier(m1_max_32()) == "Apple Silicon 32 Go unifiés"
    assert describe_tier(linux_rtx4090()) == "GPU NVIDIA 24 Go"
    assert describe_tier(windows_16_no_gpu()) == "CPU seul 16 Go"
    assert describe_tier(laptop_cpu_8()) == "CPU seul 8 Go"
    assert describe_tier(unknown_ram()) == "inconnue"
    # un GPU intégré < 4 Go de VRAM ne fait pas une classe GPU
    assert describe_tier(_profile(gpus=[GpuInfo("nvidia", "GeForce MX150", 2.0)])) == "CPU seul 16 Go"


def test_estimate_speed_bandwidth_over_active_bytes():
    cat = load_catalog()
    dense27 = next(m for m in cat["models"] if m["tag"] == "qwen3.6:27b-coding")
    moe35 = next(m for m in cat["models"] if m["tag"] == "qwen3.6:35b")
    s_m1max = estimate_speed(dense27, m1_max_32())
    s_4090 = estimate_speed(dense27, linux_rtx4090())
    s_cpu = estimate_speed(dense27, windows_16_no_gpu())
    assert s_m1max is not None and 10 <= s_m1max <= 20  # repère MODEL_RULES : M1 Max 32B ≈ 14 tok/s
    assert s_4090 is not None and 30 <= s_4090 <= 50  # repère : RTX 4090 32B ≈ 42 tok/s
    assert s_cpu is not None and s_cpu < 4
    assert estimate_speed(dense27, unknown_ram()) is None
    # MoE : seuls les 3B actifs sont lus → bien plus rapide qu'un dense de même poids
    assert estimate_speed(moe35, m1_max_32()) > 3 * s_m1max
    # mode forcé
    assert estimate_speed(dense27, m1_max_32(), mode="cpu") < s_m1max
    # jamais lever sur une entrée vide / un profil vide
    assert estimate_speed({}, {}) is None


# --------------------------------------------------------------------------- scénarios machine

def test_m1_max_32gb_unified():
    advice = advise(m1_max_32())
    assert advice.mode == "unified"
    assert advice.machine_class == "Apple Silicon 32 Go unifiés"
    assert advice.memory_budget_gb == pytest.approx(24.0)  # 75 % de 32 Go
    big_coders = [r for r in advice.recommendations
                  if r.category == "chat_code" and 18 <= r.size_gb <= 19 and r.verdict == VERDICT_RECOMMENDED]
    assert big_coders, "un modèle chat_code de 18-19 Go doit être recommandé sur un M1 Max 32 Go"
    next52 = _by_tag(advice, "qwen3-coder-next:latest")
    assert next52.verdict == VERDICT_TOO_BIG and next52.size_gb == 52
    assert advice.best_chat in {r.tag for r in big_coders}
    assert advice.best_embedding is not None
    assert _by_tag(advice, advice.best_embedding).verdict == VERDICT_RECOMMENDED
    # étapes macOS (Ollama non sondé → installation complète)
    install = _step(advice, "Installer Ollama")
    assert install is not None and "brew install ollama" in install.commands
    assert "brew services start ollama" in install.commands
    assert any("Ollama.dmg" in install.notes for _ in [0])
    verify = _step(advice, "Vérifier le service")
    assert verify is not None and any("127.0.0.1:11434/api/tags" in c for c in verify.commands)
    pull = _step(advice, "Télécharger les modèles")
    assert pull is not None and pull.commands == ["ollama pull %s" % advice.best_chat,
                                                  "ollama pull %s" % advice.best_embedding]
    assert _step(advice, "Vérifier dans Develop") is not None
    assert _step(advice, "Régler le contexte") is not None
    assert advice.offline_note.startswith("Le téléchargement des modèles est la seule étape")
    # la phrase de raison est chiffrée et mentionne la puce
    best = _by_tag(advice, advice.best_chat)
    assert "Go" in best.reason and "M1 Max" in best.reason and best.reason.endswith(".")
    assert best.pull_command == "ollama pull " + best.tag
    assert best.memory_needed_gb > best.size_gb
    assert best.estimated_tokens_per_s and best.estimated_tokens_per_s >= 8


def test_cpu_only_laptop_8gb():
    advice = advise(laptop_cpu_8())
    assert advice.mode == "cpu" and advice.machine_class == "CPU seul 8 Go"
    assert advice.memory_budget_gb == pytest.approx(4.0)  # 8 − max(4, 25 %)
    recommended = [r for r in advice.recommendations if r.verdict == VERDICT_RECOMMENDED]
    assert recommended, "au moins un petit modèle doit être recommandé"
    assert all(r.category == "small" for r in recommended)
    assert advice.best_chat is not None and _by_tag(advice, advice.best_chat).category == "small"
    for tag in ("qwen2.5-coder:7b", "qwen3.6:27b-coding", "qwen3-coder-next:latest"):
        assert _by_tag(advice, tag).verdict != VERDICT_RECOMMENDED
    assert _by_tag(advice, "qwen3-coder-next:latest").verdict == VERDICT_TOO_BIG
    assert advice.best_embedding is not None and _by_tag(advice, advice.best_embedding).size_gb < 1.5
    assert any("RAM" in w for w in advice.warnings)
    assert _step(advice, "Installer Ollama").commands == ["curl -fsSL https://ollama.com/install.sh | sh"]


def test_linux_rtx4090_24gb_vram_64gb_ram():
    advice = advise(linux_rtx4090())
    assert advice.mode == "gpu" and advice.machine_class == "GPU NVIDIA 24 Go"
    assert advice.memory_budget_gb == pytest.approx(23.5)  # VRAM − 0,5 Go
    for tag in ("qwen3.6:27b-coding", "qwen3.8:27b", "qwen3-coder:30b", "glm-4.7-flash:latest"):
        rec = _by_tag(advice, tag)
        assert rec.verdict == VERDICT_RECOMMENDED, tag
        assert "100 %" not in rec.reason or "VRAM" in rec.reason
    # 52 Go : ne tient pas en VRAM, mais VRAM + (RAM − 4) = 84 Go permettent un déchargement partiel
    next52 = _by_tag(advice, "qwen3-coder-next:latest")
    assert next52.verdict == VERDICT_POSSIBLE_SLOW
    assert "déchargement partiel" in next52.reason and "RAM" in next52.reason
    assert "MoE" in next52.reason
    assert next52.estimated_tokens_per_s is not None and next52.estimated_tokens_per_s < _by_tag(advice, "qwen3.6:35b").estimated_tokens_per_s
    best = _by_tag(advice, advice.best_chat)
    assert best.category == "chat_code" and best.quality_rank == 1 and best.estimated_tokens_per_s >= 30
    assert "4090" in best.reason
    assert not any("Aucun GPU" in w for w in advice.warnings)


def test_rtx4090_offload_too_big_without_ram():
    """Sans RAM connue, aucun déchargement : le 52 Go est trop gros même pour un RTX 4090."""
    p = linux_rtx4090()
    p.ram_total_gb = None
    p.ram_available_gb = None
    advice = advise(p)
    assert advice.mode == "gpu"
    assert _by_tag(advice, "qwen3-coder-next:latest").verdict == VERDICT_TOO_BIG
    assert _by_tag(advice, "qwen3.6:27b-coding").verdict == VERDICT_RECOMMENDED
    assert any("RAM totale inconnue" in w for w in advice.warnings)


def test_gpu_12gb_recommends_14b_and_offloads_bigger_models():
    p = _profile(cpu_model="Intel Core i5-12400", ram_total_gb=32.0, ram_available_gb=20.0,
                 gpus=[GpuInfo("nvidia", "NVIDIA GeForce RTX 3060", 12.0)])
    advice = advise(p)
    assert advice.machine_class == "GPU NVIDIA 12 Go"
    assert _by_tag(advice, "qwen2.5-coder:14b").verdict == VERDICT_RECOMMENDED
    # MODEL_RULES §7C (12 Go) : gpt-oss:20b en offload partiel (MoE, acceptable), 27B denses trop gros
    gpt = _by_tag(advice, "gpt-oss:20b")
    assert gpt.verdict == VERDICT_POSSIBLE_SLOW and "déchargement partiel" in gpt.reason
    assert _by_tag(advice, "qwen3.6:27b-coding").verdict == VERDICT_TOO_BIG  # ~2 tok/s en offload
    assert _by_tag(advice, "qwen3-coder-next:latest").verdict == VERDICT_TOO_BIG  # 52 Go > 12 + 28


def test_windows_16gb_without_gpu():
    advice = advise(windows_16_no_gpu())
    assert advice.mode == "cpu" and advice.machine_class == "CPU seul 16 Go"
    assert advice.memory_budget_gb == pytest.approx(12.0)
    assert advice.best_chat is not None
    best = _by_tag(advice, advice.best_chat)
    assert best.verdict == VERDICT_RECOMMENDED and best.params_b <= 14
    # DDR4 supposée (38 Go/s) : les 7B tiennent mais sont lents, les 18 Go sont trop gros
    seven = _by_tag(advice, "qwen2.5-coder:7b")
    assert seven.verdict == VERDICT_POSSIBLE_SLOW and seven.estimated_tokens_per_s < 8
    assert "tok/s" in seven.reason
    assert _by_tag(advice, "qwen3.6:27b-coding").verdict == VERDICT_TOO_BIG
    install = _step(advice, "Installer Ollama")
    assert install is not None and "irm https://ollama.com/install.ps1 | iex" in install.commands
    assert "OllamaSetup.exe" in install.notes
    verify = _step(advice, "Vérifier le service")
    assert any(c.startswith("Invoke-RestMethod") for c in verify.commands)
    assert any("SetEnvironmentVariable" in c for c in _step(advice, "Régler le contexte").commands)


def test_unknown_ram_everything_possible_slow_with_warning():
    advice = advise(unknown_ram())
    assert advice.mode == "unknown" and advice.machine_class == "inconnue"
    assert advice.memory_budget_gb == 0
    assert advice.recommendations and all(r.verdict == VERDICT_POSSIBLE_SLOW for r in advice.recommendations)
    assert all(r.estimated_tokens_per_s is None for r in advice.recommendations)
    assert any("RAM totale inconnue" in w for w in advice.warnings)
    assert advice.best_chat is None
    # un embedding léger reste proposé
    assert advice.best_embedding is not None
    emb = _by_tag(advice, advice.best_embedding)
    assert emb.verdict == VERDICT_RECOMMENDED and emb.size_gb <= 1.0
    json.dumps(advice.to_dict())


def test_small_machine_always_gets_a_light_embedding():
    advice = advise(_profile(ram_total_gb=4.0, ram_available_gb=2.0))
    assert advice.memory_budget_gb == 0
    assert advice.best_embedding in ("all-minilm:33m", "nomic-embed-text:v1.5")
    emb = _by_tag(advice, advice.best_embedding)
    assert emb.verdict == VERDICT_RECOMMENDED
    assert any("RAM" in w for w in advice.warnings)
    assert _step(advice, "Télécharger les modèles") is not None


def test_apple_8gb_base_chip_uses_70_percent_budget():
    p = _profile(os="macos", arch="arm64", cpu_model="Apple M1", ram_total_gb=8.0, ram_available_gb=4.0,
                 apple_silicon=True, unified_memory=True)
    advice = advise(p)
    assert advice.memory_budget_gb == pytest.approx(5.6)
    assert _by_tag(advice, "qwen3.5:4b").verdict == VERDICT_RECOMMENDED
    assert _by_tag(advice, "qwen2.5-coder:7b").verdict != VERDICT_RECOMMENDED
    assert "M1" in _by_tag(advice, advice.best_chat).reason


def test_apple_64gb_recommends_moe_35b_and_offers_wired_limit_for_52gb():
    p = _profile(os="macos", arch="arm64", cpu_model="Apple M4 Max", ram_total_gb=64.0, ram_available_gb=40.0,
                 apple_silicon=True, unified_memory=True)
    advice = advise(p)
    assert advice.memory_budget_gb == pytest.approx(48.0)
    assert _by_tag(advice, "qwen3.6:35b").verdict == VERDICT_RECOMMENDED
    next52 = _by_tag(advice, "qwen3-coder-next:latest")
    assert next52.verdict == VERDICT_POSSIBLE_SLOW and "iogpu.wired_limit_mb" in next52.reason


def test_low_available_ram_reduces_budget_by_2gb_and_warns():
    p = m1_max_32()
    p.ram_available_gb = 5.0  # < 25 % de 32
    advice = advise(p)
    assert advice.memory_budget_gb == pytest.approx(22.0)
    assert any("2 Go" in w and "disponible" in w for w in advice.warnings)


# --------------------------------------------------------------------------- tri, meilleur choix

def test_recommendations_sorted_by_verdict_then_quality_then_size():
    advice = advise(linux_rtx4090())
    order = {VERDICT_RECOMMENDED: 0, VERDICT_POSSIBLE_SLOW: 1, VERDICT_TOO_BIG: 2}
    keys = [(order[r.verdict], r.quality_rank, -r.size_gb) for r in advice.recommendations]
    assert keys == sorted(keys)
    assert all(r.category != "embedding" for r in advice.recommendations)
    assert all(r.category == "embedding" for r in advice.embedding)
    assert len(advice.recommendations) + len(advice.embedding) == len(load_catalog()["models"])


def test_best_chat_falls_back_to_general_then_small():
    cat = {"models": [
        {"tag": "big-coder:70b", "category": "chat_code", "size_gb": 43, "params_b": 70, "quality_rank": 1,
         "context_window": 32768, "capabilities": ["tools"]},
        {"tag": "mid-general:8b", "category": "general", "size_gb": 4.9, "params_b": 8, "quality_rank": 2,
         "context_window": 32768, "capabilities": ["tools"]},
        {"tag": "tiny:1b", "category": "small", "size_gb": 0.8, "params_b": 1, "quality_rank": 3,
         "context_window": 32768, "capabilities": []},
        {"tag": "emb:small", "category": "embedding", "size_gb": 0.3, "params_b": 0.1, "quality_rank": 1,
         "context_window": 2048, "capabilities": ["embedding"]},
    ]}
    advice = advise(m1_max_32(), catalog=cat)
    assert advice.best_chat == "mid-general:8b" and advice.best_embedding == "emb:small"
    advice = advise(laptop_cpu_8(), catalog=cat)
    assert advice.best_chat == "tiny:1b"
    assert any("protocole texte" in w for w in advice.warnings)  # pas de capacité tools


def test_best_embedding_by_quality_then_smallest_under_16gb():
    advice = advise(laptop_cpu_8())
    best = _by_tag(advice, advice.best_embedding)
    assert best.quality_rank == 1
    rank1 = [r for r in advice.embedding if r.quality_rank == 1 and r.verdict == VERDICT_RECOMMENDED]
    assert best.size_gb == min(r.size_gb for r in rank1)
    assert advice.embedding[0].tag == advice.best_embedding


# --------------------------------------------------------------------------- modèles installés

def test_installed_models_mark_installed_and_skip_pull():
    advice = advise(rtx3070_8(), installed_models=["qwen2.5-coder:7b"])
    assert advice.best_chat == "qwen2.5-coder:7b"
    assert _by_tag(advice, "qwen2.5-coder:7b").installed is True
    assert _by_tag(advice, "qwen2.5-coder:14b").installed is False
    pull = _step(advice, "Télécharger les modèles")
    assert pull is not None and pull.commands == ["ollama pull %s" % advice.best_embedding]
    assert all("qwen2.5-coder:7b" not in c for c in pull.commands)
    # tout installé → plus d'étape de téléchargement
    advice = advise(rtx3070_8(), installed_models=["qwen2.5-coder:7b", advice.best_embedding])
    assert _step(advice, "Télécharger les modèles") is None
    assert _step(advice, "Vérifier dans Develop") is not None


def test_installed_tag_matching_ignores_latest_suffix():
    advice = advise(m1_max_32(), installed_models=["glm-4.7-flash", "qwen3-coder-next", "nomic-embed-text:latest"])
    assert _by_tag(advice, "glm-4.7-flash:latest").installed
    assert _by_tag(advice, "qwen3-coder-next:latest").installed
    assert _by_tag(advice, "nomic-embed-text:v1.5").installed is False  # :v1.5 n'est pas :latest
    advice = advise(m1_max_32(), installed_models=["qwen3.6:27b-coding:latest"])
    assert _by_tag(advice, "qwen3.6:27b-coding").installed


def test_installed_models_taken_from_engines_report():
    eng = _engines("/usr/local/bin/ollama", True, models=["qwen2.5-coder:7b", "qwen3-embedding:0.6b"])
    advice = advise(rtx3070_8(), engines=eng)
    assert _by_tag(advice, "qwen2.5-coder:7b").installed and _by_tag(advice, "qwen3-embedding:0.6b").installed
    assert _step(advice, "Télécharger les modèles") is None


# --------------------------------------------------------------------------- moteur et étapes

def test_steps_when_ollama_absent():
    advice = advise(m1_max_32(), engines=_engines(None, False))
    titles = [s.title for s in advice.install_steps]
    assert titles[:3] == ["Installer Ollama", "Vérifier le service", "Télécharger les modèles"]
    assert "Démarrer Ollama" not in titles
    assert advice.engine["ollama_installed"] is False and advice.engine["ollama_running"] is False
    assert not any(s.title.startswith("Alternative : Ollama dans Docker") for s in advice.install_steps)


def test_steps_when_ollama_installed_but_not_running():
    advice = advise(m1_max_32(), engines=_engines("/opt/homebrew/bin/ollama", False))
    titles = [s.title for s in advice.install_steps]
    assert "Installer Ollama" not in titles and titles[0] == "Démarrer Ollama"
    assert "brew services start ollama" in advice.install_steps[0].commands
    assert "Vérifier le service" in titles
    linux = advise(linux_rtx4090(), engines=_engines("/usr/local/bin/ollama", False))
    assert "sudo systemctl start ollama" in _step(linux, "Démarrer Ollama").commands
    win = advise(windows_16_no_gpu(), engines=_engines("C:\\Ollama\\ollama.exe", False))
    assert "ollama serve" in _step(win, "Démarrer Ollama").commands


def test_steps_when_ollama_reachable_only_pull_and_verify():
    eng = _engines("/usr/local/bin/ollama", True)
    advice = advise(m1_max_32(), engines=eng)
    titles = [s.title for s in advice.install_steps]
    assert "Installer Ollama" not in titles and "Démarrer Ollama" not in titles and "Vérifier le service" not in titles
    assert titles[0] == "Télécharger les modèles"
    assert "Vérifier dans Develop" in titles
    assert advice.engine["ollama_running"] is True and advice.engine["version"] == "0.12.0"
    assert advice.engine["ollama_installed"] is True


def test_docker_alternative_step_and_mac_gpu_warning():
    eng = _engines(None, False, docker=True)
    mac = advise(m1_max_32(), engines=eng)
    docker_step = _step(mac, "Alternative : Ollama dans Docker")
    assert docker_step is not None and docker_step.optional
    assert "docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama" in docker_step.commands
    assert not any("--gpus=all" in c for c in docker_step.commands)
    assert "GPU" in docker_step.notes and "natif" in docker_step.notes
    assert any("Docker" in w and "GPU" in w for w in mac.warnings)
    assert mac.engine["docker"]["available"] is True

    linux = advise(linux_rtx4090(), engines=eng)
    docker_step = _step(linux, "Alternative : Ollama dans Docker")
    assert any("--gpus=all" in c for c in docker_step.commands)
    assert not any("Docker" in w and "GPU" in w for w in linux.warnings)


def test_docker_only_ollama_uses_container_commands():
    eng = _engines(None, False, docker=True, container="ollama")
    advice = advise(linux_rtx4090(), engines=eng)
    start = _step(advice, "Démarrer Ollama")
    assert start is not None and start.commands == ["docker start ollama"]
    pull = _step(advice, "Télécharger les modèles")
    assert all(c.startswith("docker exec -it ollama ollama pull ") for c in pull.commands)


def test_openai_compatible_alternative_step():
    eng = _engines(None, False, openai=[("LM Studio", "http://127.0.0.1:1234/v1")])
    advice = advise(m1_max_32(), engines=eng)
    alt = _step(advice, "Alternative : serveur compatible OpenAI")
    assert alt is not None and alt.optional
    assert "http://localhost:1234/v1" in alt.notes and "LM Studio" in alt.notes
    assert "Déjà joignable" in alt.notes
    assert advice.engine["openai_servers"][0]["base_url"] == "http://127.0.0.1:1234/v1"


def test_context_step_mentions_num_ctx_and_memory():
    advice = advise(m1_max_32(), num_ctx=16384)
    ctx = _step(advice, "Régler le contexte")
    assert ctx is not None and str(advice.num_ctx) in ctx.notes and "cache KV" in ctx.notes
    assert advice.num_ctx <= 16384
    assert any("OLLAMA_KV_CACHE_TYPE" in c for c in ctx.commands)
    # contexte demandé plus large : conseillé borné par ce qui tient en mémoire pour le meilleur modèle
    wide = advise(m1_max_32(), num_ctx=131072)
    assert wide.num_ctx <= 131072
    best = _by_tag(wide, wide.best_chat)
    assert best.num_ctx_max is not None and wide.num_ctx <= best.num_ctx_max


# --------------------------------------------------------------------------- avertissements

def test_low_disk_warning():
    p = m1_max_32()
    p.disk_free_gb = 5.0
    advice = advise(p)
    assert any("Disque" in w and "5 Go" in w for w in advice.warnings)
    ok = advise(m1_max_32())
    assert not any(w.startswith("Disque") for w in ok.warnings)


def test_no_gpu_big_dense_model_warning():
    cat = {"models": [
        {"tag": "dense-15b", "category": "chat_code", "size_gb": 9.3, "params_b": 15, "quality_rank": 1,
         "context_window": 32768, "capabilities": ["tools"]},
        {"tag": "emb:small", "category": "embedding", "size_gb": 0.3, "params_b": 0.1, "quality_rank": 1,
         "context_window": 2048, "capabilities": ["embedding"]},
    ]}
    station = _profile(cpu_model="Intel Xeon w7-2495X", ram_total_gb=64.0, ram_available_gb=50.0)
    advice = advise(station, catalog=cat)
    assert advice.mode == "cpu" and advice.best_chat == "dense-15b"
    assert any("Aucun GPU" in w for w in advice.warnings)


def test_license_warning_for_best_chat_with_conditional_license():
    cat = {"models": [
        {"tag": "llama-x:8b", "category": "chat_code", "size_gb": 4.9, "params_b": 8, "quality_rank": 1,
         "context_window": 32768, "capabilities": ["tools"], "license": "Llama 3.1 Community License"},
        {"tag": "emb:small", "category": "embedding", "size_gb": 0.3, "params_b": 0.1, "quality_rank": 1,
         "context_window": 2048, "capabilities": ["embedding"], "license": "Apache 2.0"},
    ]}
    advice = advise(m1_max_32(), catalog=cat)
    assert any("usage commercial sous conditions" in w for w in advice.warnings)
    assert _by_tag(advice, "llama-x:8b").license_warning


def test_profile_warnings_are_propagated_once():
    p = m1_max_32()
    p.warnings = ["Détection partielle: test", "Détection partielle: test"]
    advice = advise(p)
    assert advice.warnings.count("Détection partielle: test") == 1


# --------------------------------------------------------------------------- robustesse

@pytest.mark.parametrize("profile", [
    {}, {"os": "macos"}, {"ram_total_gb": "abc"}, {"ram_total_gb": -5, "gpus": [{"vendor": "nvidia"}]},
    {"os": "linux", "ram_total_gb": 16, "gpus": [{"vendor": "amd", "name": "Radeon RX 7900 XTX", "vram_gb": 24}]},
    {"os": "freebsd", "ram_total_gb": 16, "cpu_model": None},
])
def test_advise_never_raises_on_incomplete_profiles(profile):
    advice = advise(profile)
    assert isinstance(advice, Advice) and advice.best_embedding is not None
    assert advice.install_steps and advice.profile_summary
    json.dumps(advice.to_dict(), ensure_ascii=False)
    assert format_advice_text(advice)


def test_advise_tolerates_bad_num_ctx_and_malformed_catalog_entries():
    cat = {"models": [
        {"tag": "ok:1b", "category": "small", "size_gb": 0.8, "params_b": 1, "quality_rank": 1, "context_window": 8192},
        {"tag": "weird", "category": "small", "size_gb": "n/a", "params_b": None, "context_window": "x"},
        {"no_tag": True}, "not a dict",
        {"tag": "emb", "category": "embedding", "size_gb": 0.3},
    ]}
    advice = advise(m1_max_32(), num_ctx="oops", catalog=cat)
    assert advice.num_ctx > 0 and {r.tag for r in advice.recommendations} == {"ok:1b", "weird"}
    assert advice.best_embedding == "emb"
    advice = advise(m1_max_32(), num_ctx=0, catalog=[])
    assert advice.recommendations == [] and advice.num_ctx > 0


def test_engines_as_dict_is_accepted():
    eng = {"ollama_binary": "/usr/bin/ollama", "ollama": {"reachable": True, "version": "0.13.0",
                                                           "models": ["qwen2.5-coder:7b"]},
           "openai_servers": [], "docker_available": False, "docker_running": False, "docker_ollama_container": None}
    advice = advise(rtx3070_8(), engines=eng)
    assert advice.engine["ollama_running"] and advice.engine["version"] == "0.13.0"
    assert _by_tag(advice, "qwen2.5-coder:7b").installed


# --------------------------------------------------------------------------- sérialisation et rendu

def test_to_dict_is_json_serializable_and_matches_api_doc():
    eng = _engines("/usr/local/bin/ollama", True, models=["qwen2.5-coder:7b"], docker=True,
                   openai=[("LM Studio", "http://127.0.0.1:1234/v1")])
    advice = advise(m1_max_32(), engines=eng)
    data = advice.to_dict()
    text = json.dumps(data, ensure_ascii=False)
    assert json.loads(text) == data
    for key in ("profile_summary", "tier", "machine_class", "mode", "memory_budget_gb", "num_ctx", "recommendations",
                "embedding", "best_chat", "best_embedding", "install_steps", "warnings", "engine", "offline_note"):
        assert key in data, key
    assert data["tier"] == data["machine_class"]
    rec = data["recommendations"][0]
    for key in ("tag", "family", "category", "params_b", "active_params_b", "size_gb", "context_window", "quality_rank",
                "capabilities", "license", "verdict", "reason", "pull_command", "installed", "memory_needed_gb",
                "estimated_tokens_per_s", "notes"):
        assert key in rec, key
    assert rec["verdict"] in ("recommended", "possible_slow", "too_big")
    step = data["install_steps"][0]
    assert set(step) == {"title", "commands", "notes", "optional"} and isinstance(step["commands"], list)
    assert set(data["engine"]) >= {"ollama_installed", "ollama_running", "version", "docker", "openai_servers"}
    assert InstallStep("t", ["c"]).to_dict() == {"title": "t", "commands": ["c"], "notes": "", "optional": False}


@pytest.mark.parametrize("profile_factory", [m1_max_32, laptop_cpu_8, linux_rtx4090, windows_16_no_gpu, unknown_ram])
def test_format_advice_text_is_readable_in_terminal(profile_factory):
    eng = _engines("/usr/local/bin/ollama", False, docker=True, openai=[("LM Studio", "http://127.0.0.1:1234/v1")])
    advice = advise(profile_factory(), engines=eng)
    text = format_advice_text(advice)
    assert text.strip()
    lines = text.splitlines()
    assert all(len(line) <= 120 for line in lines), max(lines, key=len)
    assert all(len(line) <= 100 for line in lines), max(lines, key=len)
    assert "Machine :" in text and "Classe :" in text and "Étapes d'installation" in text
    assert "Embedding conseillé" in text and advice.offline_note.split(" ;")[0] in text.replace("\n", " ")
    for step in advice.install_steps:
        assert step.title in text
    shown = [r for r in advice.recommendations[:5]]
    for r in shown:
        assert r.tag in text and r.verdict_label in text
    if advice.warnings:
        assert "Avertissements" in text


def test_format_advice_text_wraps_long_commands_with_shell_continuation():
    p = _profile(os="linux", cpu_model="AMD Ryzen 7 7800X3D", ram_total_gb=32.0, ram_available_gb=20.0,
                 gpus=[GpuInfo("amd", "AMD Radeon RX 7900 XTX", 24.0)])
    advice = advise(p, engines=_engines(None, False, docker=True))
    docker_step = _step(advice, "Alternative : Ollama dans Docker")
    assert any("ollama/ollama:rocm" in c for c in docker_step.commands)
    text = format_advice_text(advice)
    lines = text.splitlines()
    assert all(len(line) <= 100 for line in lines), max(lines, key=len)
    joined = [line for line in lines if line.rstrip().endswith("\\")]
    assert joined, "la longue commande ROCm doit être coupée avec une continuation shell"
    assert advice.machine_class == "GPU AMD 24 Go"


def test_format_advice_text_marks_installed_and_best_choice():
    advice = advise(rtx3070_8(), installed_models=["qwen2.5-coder:7b"])
    text = format_advice_text(advice)
    assert "meilleur choix" in text and "installé" in text
    assert "Moteur : non sondé" in text
    probed = format_advice_text(advise(rtx3070_8(), engines=_engines("/usr/bin/ollama", True)))
    assert "Ollama joignable (v0.12.0)" in probed


# --------------------------------------------------------------------------- corrections après vérification

def _numbers(text: str):
    """Tous les nombres d'une phrase française (« 2,7 » → 2.7), dans l'ordre."""
    import re
    return [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:,\d+)?", text)]


def test_memory_needed_gb_is_computed_at_the_verdict_context():
    """memory_needed_gb (JSON) = besoin au contexte retenu (memory_ctx), jamais > budget pour un « recommandé »."""
    for profile in (m1_max_32(),
                    _profile(cpu_model="Intel Core i5-12400", ram_total_gb=32.0, ram_available_gb=20.0,
                             gpus=[GpuInfo("nvidia", "NVIDIA GeForce RTX 3060", 12.0)]),
                    _profile(os="macos", arch="arm64", cpu_model="Apple M2 Pro", ram_total_gb=16.0,
                             ram_available_gb=9.0, apple_silicon=True, unified_memory=True)):
        advice = advise(profile)
        data = advice.to_dict()
        for rec, raw in zip(advice.recommendations + advice.embedding, data["recommendations"] + data["embedding"]):
            assert raw["memory_ctx"] == rec.memory_ctx and rec.memory_ctx is not None
            if rec.verdict == VERDICT_RECOMMENDED:
                assert rec.memory_ctx == rec.num_ctx_max
                assert rec.memory_needed_gb <= advice.memory_budget_gb + 0.01, rec.tag
            model = {"size_gb": rec.size_gb, "params_b": rec.params_b}
            assert rec.memory_needed_gb == pytest.approx(advisor.memory_needed_gb(model, rec.memory_ctx), abs=0.01)
    best = _by_tag(advise(m1_max_32()), "qwen3.6:27b-coding")
    assert best.verdict == VERDICT_RECOMMENDED and best.num_ctx_max == 8192
    assert best.memory_needed_gb == pytest.approx(18 * 1.10 + advisor.kv_cache_gb(8192, 27.3) + 0.75, abs=0.01)
    assert 23.2 <= best.memory_needed_gb <= 23.3  # et non 26 Go (besoin au contexte demandé de 16k)
    assert "23,3 Go" in best.reason  # la phrase et le champ disent la même chose


def test_low_ram_warning_depends_on_mode():
    gpu = advise(_profile(ram_total_gb=8.0, ram_available_gb=5.0, cpu_model="AMD Ryzen 5 5600X",
                          gpus=[GpuInfo("nvidia", "NVIDIA GeForce RTX 3060", 12.0)]))
    assert gpu.mode == "gpu" and _by_tag(gpu, gpu.best_chat).params_b > 4  # un 7B+ est recommandé en VRAM
    assert not any("≤ 4B" in w for w in gpu.warnings)
    assert any(w.startswith("RAM de 8 Go seulement") and "déchargement" in w for w in gpu.warnings)
    cpu = advise(laptop_cpu_8())
    assert any("≤ 4B" in w for w in cpu.warnings)
    assert not any(w.startswith("RAM de") for w in advise(unknown_ram()).warnings)


def test_wired_limit_only_proposed_from_16gb_and_leaves_memory_to_macos():
    mac8 = advise(_profile(os="macos", arch="arm64", cpu_model="Apple M1", ram_total_gb=8.0, ram_available_gb=4.0,
                           apple_silicon=True, unified_memory=True))
    e2b = _by_tag(mac8, "gemma4:e2b")
    assert e2b.verdict == VERDICT_POSSIBLE_SLOW and "iogpu.wired_limit_mb=" not in e2b.reason
    assert "déconseillé" in e2b.reason and "déchargement partiel" in e2b.reason
    mac24 = advise(_profile(os="macos", arch="arm64", cpu_model="Apple M4", ram_total_gb=24.0, ram_available_gb=15.0,
                            apple_silicon=True, unified_memory=True))
    r27 = _by_tag(mac24, "qwen3.6:27b-coding")  # 21,2 Go câblés ne laisseraient que 2,8 Go à macOS
    assert r27.verdict == VERDICT_POSSIBLE_SLOW and "iogpu.wired_limit_mb=" not in r27.reason
    mac64 = advise(_profile(os="macos", arch="arm64", cpu_model="Apple M4 Max", ram_total_gb=64.0,
                            ram_available_gb=40.0, apple_silicon=True, unified_memory=True))
    r52 = _by_tag(mac64, "qwen3-coder-next:latest")
    m = __import__("re").search(r"iogpu\.wired_limit_mb=(\d+)", r52.reason)
    assert m, r52.reason
    wired_gb = int(m.group(1)) * 1048576 / 1e9
    assert wired_gb <= 64.0 - max(3.0, 0.05 * 64.0) and "à macOS" in r52.reason
    for ram, chip in ((8.0, "Apple M1"), (12.0, "Apple M2")):
        advice = advise(_profile(os="macos", arch="arm64", cpu_model=chip, ram_total_gb=ram, ram_available_gb=ram / 2,
                                 apple_silicon=True, unified_memory=True))
        assert not any("iogpu.wired_limit_mb=" in r.reason for r in advice.recommendations + advice.embedding)


def test_unknown_or_null_budget_never_says_zero_gb():
    unknown = advise(unknown_ram())
    text = format_advice_text(unknown)
    assert "0 Go" not in text and "budget mémoire inconnu" in text
    alt = _step(unknown, "Alternative : serveur compatible OpenAI")
    assert "0 Go" not in alt.notes and "non détectée" in alt.notes
    tiny = advise(_profile(ram_total_gb=4.0, ram_available_gb=2.0))
    assert tiny.memory_budget_gb == 0
    assert not any("~0 Go" in r.reason for r in tiny.recommendations + tiny.embedding)
    assert "aucun budget mémoire sûr" in format_advice_text(tiny)
    assert "0 Go" not in _step(tiny, "Alternative : serveur compatible OpenAI").notes
    ok = format_advice_text(advise(m1_max_32()))
    assert "budget mémoire ~24 Go" in ok


def test_recommended_reason_numbers_add_up_including_margin():
    for profile in (m1_max_32(), linux_rtx4090(), rtx3070_8(), windows_16_no_gpu()):
        advice = advise(profile)
        checked = 0
        for rec in advice.recommendations:
            if rec.verdict != VERDICT_RECOMMENDED or "de marge" not in rec.reason:
                continue
            head = rec.reason.split("≈")[0]
            total = _numbers(rec.reason.split("≈")[1])[0]
            parts = _numbers(head)
            # « X Go de poids + ~Y Go de cache pour Nk de contexte + ~M Go de marge » → X, Y, (N), M
            weight, kv, margin = parts[0], parts[1], parts[-1]
            assert abs(weight + kv + margin - total) <= 0.2, rec.reason
            assert margin == pytest.approx(rec.size_gb * 0.10 + 0.75, abs=0.06), rec.reason
            assert total == pytest.approx(rec.memory_needed_gb, abs=0.06)
            checked += 1
        assert checked, "au moins une phrase « recommandé » chiffrée attendue"
    best = _by_tag(advise(m1_max_32()), "qwen3.6:27b-coding")
    assert "+ ~2,6 Go de marge" in best.reason and "≈ 23,3 Go" in best.reason  # 18 + 2,7 + 2,6 = 23,3
    for rec in advise(laptop_cpu_8()).recommendations:
        if rec.verdict != VERDICT_RECOMMENDED:
            assert "+ marge" in rec.reason or "la marge" in rec.reason or "de marge" in rec.reason, rec.reason


def test_speed_floor_marks_unusable_models_too_big():
    """MODEL_RULES §4 : tok/s estimés < 4 → « trop gros », 4–8 → « possible mais lent »."""
    gpu12 = advise(_profile(cpu_model="AMD Ryzen 7 5800X", ram_total_gb=32.0, ram_available_gb=20.0,
                            gpus=[GpuInfo("nvidia", "NVIDIA GeForce RTX 3060", 12.0)]))
    for tag in ("qwen3.6:27b-coding", "devstral-small-2:24b"):
        rec = _by_tag(gpu12, tag)
        assert rec.verdict == VERDICT_TOO_BIG and rec.estimated_tokens_per_s < 4, tag
        assert "inutilisable" in rec.reason and "tok/s" in rec.reason
    cpu16 = advise(windows_16_no_gpu())  # DDR4 38 Go/s
    for tag in ("qwen2.5-coder:14b", "deepseek-r1:14b", "gemma4:12b"):
        rec = _by_tag(cpu16, tag)
        assert rec.verdict == VERDICT_TOO_BIG and rec.estimated_tokens_per_s < 4, tag
        assert rec.num_ctx_max is not None  # il tiendrait en mémoire : c'est bien la vitesse qui disqualifie
    seven = _by_tag(cpu16, "qwen2.5-coder:7b")
    assert seven.verdict == VERDICT_POSSIBLE_SLOW and 4 <= seven.estimated_tokens_per_s < 8
    # le plancher s'applique aussi à un modèle qui ne tient que dans RAM − 2 Go (CPU 16 Go : budget 12, RAM − 2 = 14)
    cat = {"models": [{"tag": "dense-17b", "category": "chat_code", "size_gb": 10.5, "params_b": 17, "quality_rank": 1,
                       "context_window": 32768, "capabilities": ["tools"]}]}
    rec = advise(windows_16_no_gpu(), catalog=cat).recommendations[0]
    assert 12 < rec.memory_needed_gb <= 14 and rec.num_ctx_max is None
    assert rec.verdict == VERDICT_TOO_BIG and "inutilisable" in rec.reason and rec.estimated_tokens_per_s < 4
    # un MoE en offload sur un gros GPU reste « possible mais lent » (≥ 4 tok/s)
    assert _by_tag(advise(linux_rtx4090()), "qwen3-coder-next:latest").verdict == VERDICT_POSSIBLE_SLOW
    assert advisor.SPEED_FLOOR_TOK_S == 4.0


def test_best_embedding_prefers_qwen3_embedding_4b_from_16gb():
    for profile in (m1_max_32(), linux_rtx4090(), windows_16_no_gpu(),
                    _profile(os="macos", arch="arm64", cpu_model="Apple M3 Ultra", ram_total_gb=192.0,
                             ram_available_gb=150.0, apple_silicon=True, unified_memory=True)):
        advice = advise(profile)
        assert advice.best_embedding == "qwen3-embedding:4b", advice.machine_class
        assert _by_tag(advice, "qwen3-embedding:4b").verdict == VERDICT_RECOMMENDED
        assert advice.embedding[0].tag == "qwen3-embedding:4b"
    # < 16 Go de RAM : le plus petit de rang 1
    assert advise(laptop_cpu_8()).best_embedding == "qwen3-embedding:0.6b"
    assert advise(_profile(os="macos", arch="arm64", cpu_model="Apple M1", ram_total_gb=8.0, ram_available_gb=4.0,
                           apple_silicon=True, unified_memory=True)).best_embedding == "qwen3-embedding:0.6b"
    # 16 Go de RAM mais 4 Go de VRAM : le 4b ne tient pas en VRAM (déchargement) → pas « recommandé » → 0.6b
    gtx = advise(_profile(ram_total_gb=16.0, ram_available_gb=10.0,
                          gpus=[GpuInfo("nvidia", "NVIDIA GeForce GTX 1650", 4.0)]))
    assert gtx.best_embedding == "qwen3-embedding:0.6b"
    assert _by_tag(gtx, "qwen3-embedding:4b").verdict != VERDICT_RECOMMENDED
    # un embedding de même rang déjà installé reste conseillé (pas de téléchargement ni de réindexation)
    installed = advise(m1_max_32(), installed_models=["qwen3-embedding:0.6b"])
    assert installed.best_embedding == "qwen3-embedding:0.6b"
    assert _step(installed, "Télécharger les modèles").commands == ["ollama pull %s" % installed.best_chat]


def test_offload_reason_without_bandwidth_is_well_formed():
    gtx = advise(_profile(ram_total_gb=16.0, ram_available_gb=10.0,
                          gpus=[GpuInfo("nvidia", "NVIDIA GeForce GTX 1650", 4.0)]))
    rec = _by_tag(gtx, "qwen3-embedding:4b")
    assert rec.estimated_tokens_per_s is None
    assert "vitesse non estimée" in rec.reason and "non estimée attendus" not in rec.reason


def test_docker_step_proposes_a_single_docker_run_with_the_alternative_in_notes():
    eng = _engines(None, False, docker=True)
    linux = advise(linux_rtx4090(), engines=eng)
    step = _step(linux, "Alternative : Ollama dans Docker")
    runs = [c for c in step.commands if c.startswith("docker run")]
    assert len(runs) == 1 and "--gpus=all" in runs[0]
    assert " ou, sans GPU : docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama" in step.notes
    assert step.commands[-1] == "docker exec -it ollama ollama pull %s" % linux.best_chat
    mac = advise(m1_max_32(), engines=eng)
    step = _step(mac, "Alternative : Ollama dans Docker")
    assert [c for c in step.commands if c.startswith("docker run")] == [
        "docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama"]
    assert " ou," not in step.notes  # pas de variante GPU sur Mac : une seule commande, rien à choisir
    # conteneur déjà présent : docker start, jamais un docker run --name qui échouerait
    existing = advise(linux_rtx4090(), engines=_engines(None, False, docker=True, container="ollama-gpu"))
    step = _step(existing, "Alternative : Ollama dans Docker")
    assert not any(c.startswith("docker run") for c in step.commands)
    assert step.commands[0] == "docker start ollama-gpu"
    assert step.commands[-1].startswith("docker exec -it ollama-gpu ollama pull ")
    assert "docker rm -f ollama-gpu" in step.notes and "--gpus=all" in step.notes


def test_advised_num_ctx_is_at_least_8192_when_a_model_allows_it():
    gpu12 = advise(_profile(cpu_model="AMD Ryzen 7 5800X", ram_total_gb=32.0, ram_available_gb=20.0,
                            gpus=[GpuInfo("nvidia", "NVIDIA GeForce RTX 3060", 12.0)]))
    assert gpu12.num_ctx >= 8192
    best = _by_tag(gpu12, gpu12.best_chat)
    assert best.category == "chat_code" and best.num_ctx_max >= 8192
    assert _by_tag(gpu12, "qwen2.5-coder:14b").num_ctx_max == 4096  # tient, mais pas pour un assistant de code
    assert str(gpu12.num_ctx) in _step(gpu12, "Régler le contexte").notes
    for profile in (m1_max_32(), linux_rtx4090(), rtx3070_8(), windows_16_no_gpu(), laptop_cpu_8()):
        advice = advise(profile)
        assert advice.num_ctx >= 8192, advice.machine_class
    # aucun modèle ne permet 8k : le contexte conseillé reste celui que le modèle permet
    cat = {"models": [{"tag": "only-4k", "category": "chat_code", "size_gb": 3.0, "params_b": 4, "quality_rank": 1,
                       "context_window": 4096, "capabilities": ["tools"]}]}
    advice = advise(m1_max_32(), catalog=cat)
    assert advice.best_chat == "only-4k" and advice.num_ctx == 4096
    # sans aucun modèle recommandé : 8k par défaut (MODEL_RULES §5), borné par la demande
    assert advise(_profile(ram_total_gb=4.0, ram_available_gb=2.0), catalog={"models": []}).num_ctx == 8192
    assert advise(m1_max_32(), num_ctx=4096).num_ctx == 4096  # la demande explicite reste respectée
