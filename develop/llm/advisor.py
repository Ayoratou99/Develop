"""Conseiller de modèles : croise le profil machine, l'état des moteurs et le catalogue local.

Tout est hors ligne : aucune requête réseau, aucun téléchargement. Le conseiller reproduit les
règles de ``docs/MODEL_RULES.md`` de façon volontairement conservatrice :

* budget mémoire par classe de machine (Apple unifié ≈ 70–75 % de la RAM ; GPU dédié = VRAM − 0,5 Go
  avec déchargement partiel possible vers la RAM ; CPU seul = RAM − max(4 Go, 25 %)) ;
* besoin d'un modèle = poids × 1,10 + cache KV(contexte) + 0,75 Go de marge ;
* vitesse ≈ facteur × bande passante mémoire ÷ octets lus par token (MoE : experts actifs seulement) ;
* verdict « recommandé » / « possible mais lent » / « trop gros » (dont < 4 tok/s estimés, inutilisable) avec une
  phrase d'explication chiffrée dont les nombres s'additionnent (poids + cache + marge ≈ besoin).

Il ne lève jamais sur un profil incomplet : chaque champ absent est remplacé par une hypothèse prudente.
"""
from __future__ import annotations

import json
import logging
import math
import re
import textwrap
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

log = logging.getLogger(__name__)

CATALOG_PATH = Path(__file__).parent / "catalog.json"

# --------------------------------------------------------------------------- constantes des règles

VERDICT_RECOMMENDED = "recommended"
VERDICT_POSSIBLE_SLOW = "possible_slow"
VERDICT_TOO_BIG = "too_big"
VERDICT_ORDER = {VERDICT_RECOMMENDED: 0, VERDICT_POSSIBLE_SLOW: 1, VERDICT_TOO_BIG: 2}
VERDICT_LABELS = {VERDICT_RECOMMENDED: "recommandé", VERDICT_POSSIBLE_SLOW: "possible mais lent",
                  VERDICT_TOO_BIG: "trop gros"}

MODE_UNIFIED = "unified"
MODE_GPU = "gpu"
MODE_CPU = "cpu"
MODE_UNKNOWN = "unknown"

CATEGORY_LABELS = {"chat_code": "chat/code", "reasoning": "raisonnement", "general": "généraliste",
                   "small": "petit modèle", "embedding": "embedding"}

SPEED_MIN_TOK_S = 8.0  # en dessous : « possible mais lent »
SPEED_FLOOR_TOK_S = 4.0  # en dessous : « trop gros », inutilisable au quotidien (MODEL_RULES §4)
MIN_USEFUL_CTX = 4096  # contexte minimal pour qu'un modèle reste « recommandé »
MIN_CODE_CTX = 8192  # contexte conseillé minimal pour un assistant de code (MODEL_RULES §5)
CTX_LADDER = (131072, 65536, 32768, 16384, 8192, 4096, 2048)
WIRED_LIMIT_MIN_RAM_GB = 16.0  # en dessous, ne jamais proposer de relever iogpu.wired_limit_mb (MODEL_RULES §7A)
WIRED_LIMIT_OS_RESERVE_GB = 3.0  # macOS garde au moins max(3 Go, 5 % de la RAM) hors de la limite Metal
WIRED_LIMIT_OS_RESERVE_RATIO = 0.05
PREFERRED_LARGE_EMBEDDING = "qwen3-embedding:4b"  # meilleur embedding pour le code dès 16 Go (MODEL_RULES §6.3)
LARGE_EMBEDDING_MIN_RAM_GB = 16.0
WEIGHT_OVERHEAD = 1.10  # surcoût de chargement d'Ollama sur le poids
MEMORY_MARGIN_GB = 0.75  # marge fixe (buffers, graphe de calcul)
GB_PER_BILLION_PARAMS_Q4 = 0.62  # octets par paramètre en Q4_K_M, en Go par milliard
MIN_DEDICATED_VRAM_GB = 4.0
OFFLOAD_PENALTY = 0.7  # surcoût du découpage couche par couche GPU/CPU
LOW_RAM_PRESSURE_RATIO = 0.25
LOW_RAM_PENALTY_GB = 2.0

# Bandes passantes mémoire (Go/s) — MODEL_RULES.md §1
APPLE_BANDWIDTH_GBS = {
    "M1": 68, "M1 Pro": 200, "M1 Max": 400, "M1 Ultra": 800,
    "M2": 100, "M2 Pro": 200, "M2 Max": 400, "M2 Ultra": 800,
    "M3": 100, "M3 Pro": 150, "M3 Max": 400, "M3 Ultra": 819,
    "M4": 120, "M4 Pro": 273, "M4 Max": 546,
}
# Repli prudent pour une puce inconnue (génération plus récente) : le plus petit de sa variante.
_APPLE_VARIANT_FALLBACK = {"Ultra": 800, "Max": 400, "Pro": 150, "": 68}
GPU_BANDWIDTH_GBS = [  # (motif regex sur le nom en minuscules, Go/s) — le premier motif qui correspond gagne
    (r"h200", 4800), (r"h100", 3350), (r"a100", 1555), (r"v100", 900), (r"l40", 864), (r"a10\b", 600),
    (r"t4\b", 320), (r"a6000", 768), (r"a5000", 768), (r"a4000", 448), (r"rtx\s*6000", 960),
    (r"5090", 1792), (r"5080", 960), (r"5070\s*ti", 896), (r"5070", 672), (r"5060", 448),
    (r"4090", 1008), (r"4080", 717), (r"4070\s*ti", 504), (r"4070", 504), (r"4060\s*ti", 288), (r"4060", 272),
    (r"3090", 936), (r"3080", 760), (r"3070", 448), (r"3060", 360), (r"3050", 224),
    (r"2080", 448), (r"2070", 448), (r"2060", 336), (r"1080", 320), (r"1070", 256), (r"1060", 192),
    (r"7900\s*xtx", 960), (r"7900\s*xt", 800), (r"7800\s*xt", 624), (r"7700\s*xt", 432), (r"7600", 288),
    (r"6950", 576), (r"6900", 512), (r"6800", 512), (r"6700", 384), (r"6600", 256),
]
CPU_BANDWIDTH_DEFAULT_GBS = 38.0  # DDR4 double canal, hypothèse basse
CPU_BANDWIDTH_DDR5_GBS = 80.0  # plateformes DDR5 / LPDDR5x connues
CPU_BANDWIDTH_WORKSTATION_GBS = 150.0  # 4–8 canaux (Xeon, EPYC, Threadripper)

OLLAMA_URL = "http://127.0.0.1:11434"
OFFLINE_NOTE = ("Le téléchargement des modèles est la seule étape qui nécessite Internet ; "
                "tout le reste fonctionne hors ligne.")
LIGHT_EMBEDDINGS = ("nomic-embed-text:v1.5", "nomic-embed-text", "all-minilm:33m", "all-minilm")
EXCLUDED_TAG_PATTERNS = (r"-cloud$", r"^codellama", r"^starcoder2", r"^deepseek-coder", r"^codestral", r"^laguna-xs")
LICENSE_WARNINGS = (  # (fragment de licence en minuscules, avertissement)
    ("llama", "usage commercial sous conditions (Llama Community License)"),
    ("nvidia", "usage commercial sous conditions (NVIDIA Open Model License)"),
    ("deepseek", "usage commercial sous conditions (DeepSeek Model License)"),
    ("non affichée", "licence non affichée sur Ollama, à confirmer avant usage commercial"),
    ("non vérifiée", "licence à confirmer avant usage commercial"),
)
_VENDOR_LABELS = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple"}
_OS_LABELS = {"macos": "macOS", "linux": "Linux", "windows": "Windows"}


# --------------------------------------------------------------------------- structures publiques

@dataclass
class Recommendation:
    tag: str
    family: str
    category: str
    params_b: Optional[float]
    active_params_b: Optional[float]
    size_gb: float
    context_window: int
    quality_rank: int
    capabilities: List[str]
    license: str
    verdict: str  # recommended | possible_slow | too_big
    reason: str  # une phrase française chiffrée
    pull_command: str
    installed: bool
    memory_needed_gb: float  # besoin (Go) au contexte retenu pour le verdict (memory_ctx), cohérent avec reason
    estimated_tokens_per_s: Optional[float]
    notes: str = ""
    num_ctx_max: Optional[int] = None  # plus grand contexte de l'échelle qui tient dans le budget
    license_warning: str = ""
    memory_ctx: Optional[int] = None  # contexte (tokens) auquel memory_needed_gb est calculé

    @property
    def verdict_label(self) -> str:
        return VERDICT_LABELS.get(self.verdict, self.verdict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verdict_label"] = self.verdict_label
        return d


@dataclass
class InstallStep:
    title: str
    commands: List[str]
    notes: str = ""
    optional: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"title": self.title, "commands": list(self.commands), "notes": self.notes, "optional": self.optional}


@dataclass
class Advice:
    profile_summary: str
    machine_class: str
    mode: str  # unified | gpu | cpu | unknown
    memory_budget_gb: float
    num_ctx: int
    recommendations: List[Recommendation] = field(default_factory=list)
    embedding: List[Recommendation] = field(default_factory=list)
    best_chat: Optional[str] = None
    best_embedding: Optional[str] = None
    install_steps: List[InstallStep] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    engine: Dict[str, Any] = field(default_factory=dict)
    offline_note: str = OFFLINE_NOTE
    catalog_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_summary": self.profile_summary,
            "machine_class": self.machine_class,
            "tier": self.machine_class,  # nom utilisé par docs/API.md
            "mode": self.mode,
            "memory_budget_gb": self.memory_budget_gb,
            "num_ctx": self.num_ctx,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "embedding": [r.to_dict() for r in self.embedding],
            "best_chat": self.best_chat,
            "best_embedding": self.best_embedding,
            "install_steps": [s.to_dict() for s in self.install_steps],
            "warnings": list(self.warnings),
            "engine": dict(self.engine),
            "offline_note": self.offline_note,
            "catalog_version": self.catalog_version,
        }


# --------------------------------------------------------------------------- catalogue

_CATALOG_CACHE: Dict[str, Dict[str, Any]] = {}


def _excluded(tag: str) -> bool:
    low = tag.lower()
    return any(re.search(p, low) for p in EXCLUDED_TAG_PATTERNS)


def load_catalog(path: Optional[Path] = None, refresh: bool = False) -> Dict[str, Any]:
    """Lit ``catalog.json`` embarqué dans le package (mis en cache).

    Un fichier absent ou illisible donne un catalogue vide avec un avertissement : le conseiller
    continue de fonctionner (étapes d'installation, classe de machine) sans proposer de modèle.
    """
    p = Path(path) if path else CATALOG_PATH
    key = str(p)
    if not refresh and key in _CATALOG_CACHE:
        cached = _CATALOG_CACHE[key]
        return dict(cached, models=list(cached["models"]))
    data: Dict[str, Any] = {"version": "", "models": []}
    try:
        with open(p, encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and isinstance(raw.get("models"), list):
            data = dict(raw)
        elif isinstance(raw, list):
            data = {"version": "", "models": raw}
        else:
            warnings.warn("Catalogue de modèles invalide (%s) : aucun modèle proposé." % p)
    except FileNotFoundError:
        warnings.warn("Catalogue de modèles introuvable (%s) : aucun modèle proposé." % p)
        log.warning("catalogue introuvable : %s", p)
    except (OSError, ValueError) as exc:
        warnings.warn("Catalogue de modèles illisible (%s : %s) : aucun modèle proposé." % (p, exc))
        log.warning("catalogue illisible %s : %s", p, exc)
    models = [m for m in data.get("models", []) if isinstance(m, dict) and m.get("tag") and not _excluded(str(m["tag"]))]
    data["models"] = models
    data.setdefault("version", "")
    _CATALOG_CACHE[key] = data
    return dict(data, models=list(models))


# --------------------------------------------------------------------------- utilitaires

def _pget(obj: Any, name: str, default: Any = None) -> Any:
    """Lit un champ d'un objet ou d'un dict, ``default`` si absent ou None."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        value = obj.get(name, default)
    else:
        value = getattr(obj, name, default)
    return default if value is None else value


def _num(value: Any) -> Optional[float]:
    """Nombre strictement positif et fini, sinon None."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f) or f <= 0:
        return None
    return f


def _fr(x: float, decimals: int = 1) -> str:
    """Nombre en notation française : 4,7 ; 18 ; 0,27."""
    s = ("%%.%df" % decimals) % x
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _gb(x: float) -> str:
    return _fr(x, 2 if x < 1 else 1) + " Go"


def _ctx_label(n: int) -> str:
    return "%dk" % int(round(n / 1024.0)) if n >= 1024 else str(n)


def _tok_label(tok_s: Optional[float]) -> str:
    if tok_s is None:
        return "vitesse non estimée"
    if tok_s < 1:
        return "< 1 tok/s"
    return "~%d tok/s" % int(round(tok_s))


def _norm_tag(tag: str) -> str:
    t = str(tag).strip().lower()
    if t.startswith("library/"):
        t = t[len("library/"):]
    if t.endswith(":latest"):
        t = t[: -len(":latest")]
    return t


def _installed_set(installed_models: Optional[Iterable[Any]], engines: Any) -> Set[str]:
    names: Set[str] = set()
    for item in list(installed_models or []):
        name = item if isinstance(item, str) else _pget(item, "name", "")
        if name:
            names.add(_norm_tag(name))
    probe = _pget(engines, "ollama")
    for item in list(_pget(probe, "models", []) or []):
        name = item if isinstance(item, str) else _pget(item, "name", "")
        if name:
            names.add(_norm_tag(name))
    return names


_APPLE_CHIP_RE = re.compile(r"\bM(\d+)(?:\s+(Pro|Max|Ultra))?\b", re.IGNORECASE)
_DDR5_CPU_RE = re.compile(r"core\s*ultra|ryzen\s*ai|ryzen\s+[3579]\s+[789]\d{3}|ryzen\s+[3579]\s+pro\s+[789]\d{3}", re.I)
_WORKSTATION_CPU_RE = re.compile(r"xeon|epyc|threadripper", re.I)


def _apple_chip(cpu_model: str) -> str:
    m = _APPLE_CHIP_RE.search(cpu_model or "")
    if not m:
        return ""
    variant = (m.group(2) or "").title()
    return ("M%s %s" % (m.group(1), variant)).strip()


def _apple_bandwidth(chip: str) -> float:
    if chip in APPLE_BANDWIDTH_GBS:
        return float(APPLE_BANDWIDTH_GBS[chip])
    variant = chip.split(" ", 1)[1] if " " in chip else ""
    return float(_APPLE_VARIANT_FALLBACK.get(variant, 68))


def _gpu_bandwidth(name: str) -> Optional[float]:
    low = (name or "").lower()
    for pattern, bw in GPU_BANDWIDTH_GBS:
        if re.search(pattern, low):
            return float(bw)
    return None


def _cpu_bandwidth(cpu_model: str) -> float:
    model = cpu_model or ""
    if _WORKSTATION_CPU_RE.search(model):
        return CPU_BANDWIDTH_WORKSTATION_GBS
    if _DDR5_CPU_RE.search(model):
        return CPU_BANDWIDTH_DDR5_GBS
    return CPU_BANDWIDTH_DEFAULT_GBS


def _short_gpu_name(name: str) -> str:
    words = [w for w in (name or "").replace("(TM)", "").split()]
    while words and words[0].lower() in ("nvidia", "geforce", "amd", "radeon", "ati", "intel", "arc"):
        words.pop(0)
    return " ".join(words) or (name or "GPU")


# --------------------------------------------------------------------------- classe de machine

@dataclass
class _Machine:
    mode: str
    label: str  # ex. « Apple Silicon 32 Go unifiés »
    hw_name: str  # ex. « votre M1 Max »
    ram_gb: Optional[float]
    ram_available_gb: Optional[float]
    budget_gb: float
    budget_offload_gb: Optional[float]
    bandwidth_gbs: Optional[float]
    cpu_bandwidth_gbs: float
    vram_gb: Optional[float] = None
    gpu_name: str = ""
    gpu_vendor: str = ""
    chip: str = ""
    ram_pressure: bool = False
    warnings: List[str] = field(default_factory=list)


def _classify(profile: Any) -> _Machine:
    ram = _num(_pget(profile, "ram_total_gb"))
    avail = _num(_pget(profile, "ram_available_gb"))
    cpu_model = str(_pget(profile, "cpu_model", "") or "")
    os_name = str(_pget(profile, "os", "") or "").lower()
    apple = bool(_pget(profile, "apple_silicon", False)) or bool(_pget(profile, "unified_memory", False))
    gpus = list(_pget(profile, "gpus", []) or [])
    dedicated = []
    for g in gpus:
        vendor = str(_pget(g, "vendor", "") or "").lower()
        vram = _num(_pget(g, "vram_gb"))
        if vendor in ("nvidia", "amd") and vram and vram >= MIN_DEDICATED_VRAM_GB:
            dedicated.append((vram, vendor, str(_pget(g, "name", "") or "GPU")))
    cpu_bw = _cpu_bandwidth(cpu_model)
    warns: List[str] = []

    if apple and ram:
        chip = _apple_chip(cpu_model)
        budget = ram * (0.70 if ram <= 16 else 0.75)
        machine = _Machine(mode=MODE_UNIFIED, label="Apple Silicon %s Go unifiés" % _fr(ram),
                           hw_name="votre %s" % (chip or "puce Apple"), ram_gb=ram, ram_available_gb=avail,
                           budget_gb=budget, budget_offload_gb=None, bandwidth_gbs=_apple_bandwidth(chip),
                           cpu_bandwidth_gbs=cpu_bw, chip=chip or "Apple Silicon", gpu_vendor="apple")
    elif dedicated:
        dedicated.sort(reverse=True)
        vram, vendor, name = dedicated[0]
        total_vram = sum(v for v, _, _ in dedicated)
        offload = total_vram + (ram - 4.0 if ram and ram > 4.0 else 0.0)
        if ram is None:
            warns.append("RAM totale inconnue : le déchargement partiel vers la RAM n'est pas pris en compte.")
        short = _short_gpu_name(name)
        machine = _Machine(mode=MODE_GPU, label="GPU %s %s Go" % (_VENDOR_LABELS.get(vendor, vendor.upper()), _fr(vram)),
                           hw_name="votre %s" % short, ram_gb=ram, ram_available_gb=avail, budget_gb=vram - 0.5,
                           budget_offload_gb=offload, bandwidth_gbs=_gpu_bandwidth(name), cpu_bandwidth_gbs=cpu_bw,
                           vram_gb=vram, gpu_name=short, gpu_vendor=vendor)
        if machine.bandwidth_gbs is None:
            warns.append("Bande passante de %s inconnue : vitesse non estimée pour les modèles sur GPU." % short)
    elif ram:
        budget = ram - max(4.0, 0.25 * ram)
        machine = _Machine(mode=MODE_CPU, label="CPU seul %s Go" % _fr(ram), hw_name="votre machine sans GPU",
                           ram_gb=ram, ram_available_gb=avail, budget_gb=budget, budget_offload_gb=None,
                           bandwidth_gbs=cpu_bw, cpu_bandwidth_gbs=cpu_bw)
        if os_name == "macos":
            warns.append("Mac Intel : CPU seul, pas d'accélération Metal (Ollama exige macOS 14 ou plus récent).")
    else:
        machine = _Machine(mode=MODE_UNKNOWN, label="inconnue", hw_name="votre machine", ram_gb=None,
                           ram_available_gb=avail, budget_gb=0.0, budget_offload_gb=None, bandwidth_gbs=None,
                           cpu_bandwidth_gbs=cpu_bw)
        warns.append("RAM totale inconnue : classe de machine indéterminée, tous les modèles sont marqués "
                     "« possible mais lent » ; vérifiez la mémoire disponible avant de télécharger.")

    if ram and avail is not None and avail < LOW_RAM_PRESSURE_RATIO * ram:
        machine.ram_pressure = True
        if machine.mode == MODE_GPU and machine.budget_offload_gb is not None:
            machine.budget_offload_gb = max(0.0, machine.budget_offload_gb - LOW_RAM_PENALTY_GB)
        else:
            machine.budget_gb = max(0.0, machine.budget_gb - LOW_RAM_PENALTY_GB)
        warns.append("Seulement %s de RAM disponible sur %s (< 25 %%) : budget réduit de 2 Go ; fermez des "
                     "applications ou lancez `ollama stop <modèle>` avant de charger un modèle." % (_gb(avail), _gb(ram)))
    machine.budget_gb = max(0.0, round(machine.budget_gb, 2))
    if machine.budget_offload_gb is not None:
        machine.budget_offload_gb = round(machine.budget_offload_gb, 2)
    machine.warnings = warns
    return machine


def describe_tier(profile: Any) -> str:
    """Classe de machine en une expression : « Apple Silicon 32 Go unifiés », « GPU NVIDIA 24 Go », « CPU seul 16 Go »."""
    return _classify(profile).label


# --------------------------------------------------------------------------- mémoire et vitesse

def _model_size_gb(model: Dict[str, Any]) -> float:
    size = _num(model.get("size_gb"))
    if size:
        return size
    params = _num(model.get("params_b"))
    return params * GB_PER_BILLION_PARAMS_Q4 if params else 0.5


def _model_params_b(model: Dict[str, Any]) -> float:
    params = _num(model.get("params_b"))
    if params:
        return params
    return max(0.1, _model_size_gb(model) / GB_PER_BILLION_PARAMS_Q4)


def kv_cache_gb(ctx: int, params_b: float) -> float:
    """Cache KV estimé (Go) pour ``ctx`` tokens : 0,125 × (ctx/1000) × (params/8)^0,8."""
    return 0.125 * (max(ctx, 0) / 1000.0) * (max(params_b, 0.1) / 8.0) ** 0.8


def memory_needed_gb(model: Dict[str, Any], ctx: int) -> float:
    """Besoin mémoire (Go) = poids × 1,10 + cache KV(ctx) + 0,75 Go de marge."""
    return _model_size_gb(model) * WEIGHT_OVERHEAD + kv_cache_gb(ctx, _model_params_b(model)) + MEMORY_MARGIN_GB


def _is_moe(model: Dict[str, Any]) -> bool:
    return bool(model.get("moe")) and _num(model.get("active_params_b")) is not None


def _speed(model: Dict[str, Any], machine: _Machine, mode: Optional[str] = None,
           cpu_fraction: float = 0.0) -> Optional[float]:
    mode = mode or machine.mode
    if mode == MODE_UNKNOWN:
        return None
    bw = machine.cpu_bandwidth_gbs if mode == MODE_CPU else machine.bandwidth_gbs
    if not bw:
        return None
    if _is_moe(model):
        read_gb = float(_num(model.get("active_params_b")) or 1.0) * GB_PER_BILLION_PARAMS_Q4 + 1.0
        factor = 0.7
    else:
        read_gb = _model_size_gb(model)
        factor = 1.0
    factor *= 0.6 if mode == MODE_CPU else 0.7
    if cpu_fraction > 0:
        cpu_fraction = min(1.0, cpu_fraction)
        bw = 1.0 / ((1.0 - cpu_fraction) / bw + cpu_fraction / machine.cpu_bandwidth_gbs)
        factor *= OFFLOAD_PENALTY
    return round(factor * bw / max(read_gb, 0.05), 1)


def estimate_speed(model: Dict[str, Any], profile: Any, mode: Optional[str] = None) -> Optional[float]:
    """Tokens/s estimés : facteur × bande passante ÷ Go lus par token (None si la bande passante est inconnue).

    ``mode`` force ``unified`` / ``gpu`` / ``cpu`` ; par défaut celui déduit du profil.
    """
    try:
        return _speed(dict(model or {}), _classify(profile), mode)
    except Exception:  # noqa: BLE001 - jamais lever pour une estimation
        return None


# --------------------------------------------------------------------------- évaluation d'un modèle

def _license_warning(license_text: str) -> str:
    low = (license_text or "").lower()
    for fragment, warning in LICENSE_WARNINGS:
        if fragment in low:
            return warning
    return ""


def _margin_gb(size_gb: float) -> float:
    """Marge comptée dans le besoin : surcoût de chargement (10 % du poids) + marge fixe (buffers)."""
    return size_gb * (WEIGHT_OVERHEAD - 1.0) + MEMORY_MARGIN_GB


def _wired_limit_cap_gb(ram_gb: Optional[float]) -> Optional[float]:
    """Plafond (Go) pour ``iogpu.wired_limit_mb`` : RAM − max(3 Go, 5 %) ; None sous 16 Go (jamais proposé)."""
    if not ram_gb or ram_gb < WIRED_LIMIT_MIN_RAM_GB:
        return None
    return ram_gb - max(WIRED_LIMIT_OS_RESERVE_GB, WIRED_LIMIT_OS_RESERVE_RATIO * ram_gb)


def _budget_phrase(budget_gb: float) -> str:
    """« le budget de ~12 Go », ou une formule honnête quand le budget calculé est nul."""
    if budget_gb <= 0:
        return "le budget sûr (nul sur cette machine)"
    return "le budget de ~%s" % _gb(budget_gb)


def _evaluate(model: Dict[str, Any], machine: _Machine, num_ctx: int, installed: Set[str]) -> Recommendation:
    tag = str(model.get("tag", ""))
    size = _model_size_gb(model)
    params = _model_params_b(model)
    active = _num(model.get("active_params_b"))
    moe = _is_moe(model)
    category = str(model.get("category") or "general")
    ctx_window = int(_num(model.get("context_window")) or 4096)
    caps = [str(c) for c in (model.get("capabilities") or [])]
    ctx_req = max(256, min(int(num_ctx), ctx_window))
    ladder = sorted({c for c in CTX_LADDER if c <= ctx_req} | {ctx_req}, reverse=True)
    ctx_min = ladder[-1]
    budget = machine.budget_gb
    need_req = memory_needed_gb(model, ctx_req)
    need_min = memory_needed_gb(model, ctx_min)
    num_ctx_max: Optional[int] = next((c for c in ladder if memory_needed_gb(model, c) <= budget), None)
    tok_s = _speed(model, machine)
    hw = machine.hw_name
    size_s, budget_s, margin_s = _gb(size), _gb(budget), _gb(_margin_gb(size))
    is_embedding = category == "embedding"
    # Besoin exporté (memory_needed_gb) : toujours celui chiffré dans la phrase, au contexte retenu pour le verdict.
    need_shown, ctx_shown = need_req, ctx_req

    if machine.mode == MODE_UNKNOWN:
        if is_embedding and size <= 1.0:
            verdict = VERDICT_RECOMMENDED
            reason = ("%s de poids seulement (≈ %s avec le cache et la marge) : convient même sans connaître la "
                      "mémoire de la machine." % (size_s, _gb(need_req)))
        else:
            verdict = VERDICT_POSSIBLE_SLOW
            reason = ("Mémoire de la machine inconnue : impossible de vérifier que %s de poids + cache + marge (≈ %s "
                      "à %s de contexte) tiennent, à essayer avec prudence." % (size_s, _gb(need_req), _ctx_label(ctx_req)))
    elif num_ctx_max is not None:
        ctx_fit = num_ctx_max
        need_fit = memory_needed_gb(model, ctx_fit)
        kv_fit = kv_cache_gb(ctx_fit, params)
        need_shown, ctx_shown = need_fit, ctx_fit
        bw_s = int(machine.bandwidth_gbs or machine.cpu_bandwidth_gbs)
        if is_embedding:
            verdict = VERDICT_RECOMMENDED
            reason = ("%s de poids, ≈ %s avec le cache et la marge : négligeable pour %s (~%s utilisables)."
                      % (size_s, _gb(need_fit), hw, budget_s))
        elif tok_s is not None and tok_s < SPEED_FLOOR_TOK_S:
            verdict = VERDICT_TOO_BIG
            reason = ("%s de poids + cache + marge ≈ %s tiendraient dans les ~%s de %s, mais sa bande passante "
                      "mémoire (~%d Go/s) ne donnerait que %s%s : inutilisable au quotidien (< %d tok/s)."
                      % (size_s, _gb(need_fit), budget_s, hw, bw_s, _tok_label(tok_s),
                         " malgré l'architecture MoE" if moe else "", int(SPEED_FLOOR_TOK_S)))
        elif tok_s is not None and tok_s < SPEED_MIN_TOK_S:
            verdict = VERDICT_POSSIBLE_SLOW
            reason = ("%s de poids + ~%s de cache pour %s de contexte + ~%s de marge ≈ %s tiennent dans les ~%s de "
                      "%s, mais sa bande passante mémoire (~%d Go/s) ne donne que %s%s."
                      % (size_s, _gb(kv_fit), _ctx_label(ctx_fit), margin_s, _gb(need_fit), budget_s, hw, bw_s,
                         _tok_label(tok_s), " malgré l'architecture MoE" if moe else ""))
        elif ctx_fit < MIN_USEFUL_CTX:
            verdict = VERDICT_POSSIBLE_SLOW
            reason = ("%s de poids + cache + marge ≈ %s pour ~%s utilisables : ne tient qu'avec un contexte réduit "
                      "à %s, trop étroit pour un assistant de code (%s)."
                      % (size_s, _gb(need_fit), budget_s, _ctx_label(ctx_fit), _tok_label(tok_s)))
        else:
            verdict = VERDICT_RECOMMENDED
            comfort = "confortable" if need_fit <= 0.8 * budget else "ça passe"
            if ctx_fit < ctx_req:
                comfort += " en limitant le contexte à %s" % _ctx_label(ctx_fit)
            reason = ("%s de poids + ~%s de cache pour %s de contexte + ~%s de marge (chargement, buffers) ≈ %s ; "
                      "%s dispose de ~%s utilisables : %s (%s%s)."
                      % (size_s, _gb(kv_fit), _ctx_label(ctx_fit), margin_s, _gb(need_fit), hw, budget_s, comfort,
                         _tok_label(tok_s), " estimés" if tok_s is not None else ""))
    elif machine.mode == MODE_GPU and machine.budget_offload_gb and need_min <= machine.budget_offload_gb:
        ctx_off = next((c for c in ladder if c <= MIN_USEFUL_CTX and memory_needed_gb(model, c) <= machine.budget_offload_gb), ctx_min)
        need_off = memory_needed_gb(model, ctx_off)
        need_shown, ctx_shown = need_off, ctx_off
        cpu_fraction = max(0.0, min(1.0, (need_off - budget) / need_off))
        tok_s = _speed(model, machine, cpu_fraction=cpu_fraction)
        pct = int(round(cpu_fraction * 100))
        if tok_s is not None and tok_s < SPEED_FLOOR_TOK_S and not is_embedding:
            verdict = VERDICT_TOO_BIG
            reason = ("%s de poids + cache + marge ≈ %s à %s de contexte dépassent les ~%s de VRAM de %s : ~%d %% des "
                      "couches iraient en RAM et le déchargement ne donnerait que %s : inutilisable au quotidien "
                      "(< %d tok/s)."
                      % (size_s, _gb(need_off), _ctx_label(ctx_off), budget_s, hw, pct, _tok_label(tok_s),
                         int(SPEED_FLOOR_TOK_S)))
        else:
            verdict = VERDICT_POSSIBLE_SLOW
            speed_s = ("%s attendus" % _tok_label(tok_s)) if tok_s is not None else _tok_label(None)
            reason = ("%s de poids + cache + marge ≈ %s à %s de contexte dépassent les ~%s de VRAM de %s : ~%d %% des "
                      "couches iront en RAM (déchargement partiel), %s%s."
                      % (size_s, _gb(need_off), _ctx_label(ctx_off), budget_s, hw, pct, speed_s,
                         " (MoE : acceptable)" if moe else " (dense : 5 à 20× plus lent qu'en VRAM)"))
    elif machine.mode in (MODE_CPU, MODE_UNIFIED) and machine.ram_gb and need_min <= machine.ram_gb - 2.0:
        need_shown, ctx_shown = need_min, ctx_min
        ram_s = _gb(machine.ram_gb)
        verdict = VERDICT_POSSIBLE_SLOW
        if is_embedding:
            reason = ("%s de poids (≈ %s avec le cache et la marge) dépassent %s mais tiennent dans vos %s de RAM : "
                      "utilisable si rien d'autre ne tourne." % (size_s, _gb(need_min), _budget_phrase(budget), ram_s))
        elif tok_s is not None and tok_s < SPEED_FLOOR_TOK_S:
            verdict = VERDICT_TOO_BIG
            reason = ("%s de poids + cache + marge ≈ %s à %s de contexte dépassent %s, et %s ne donnerait que %s "
                      "même en fermant tout : inutilisable au quotidien (< %d tok/s)."
                      % (size_s, _gb(need_min), _ctx_label(ctx_min), _budget_phrase(budget), hw, _tok_label(tok_s),
                         int(SPEED_FLOOR_TOK_S)))
        elif machine.mode == MODE_UNIFIED:
            head = ("%s de poids + cache + marge ≈ %s à %s de contexte dépassent les ~%s que Metal peut réserver sur "
                    "vos %s unifiés" % (size_s, _gb(need_min), _ctx_label(ctx_min), budget_s, ram_s))
            wired_mb = int(math.ceil(need_min * 1e9 / 1048576 / 500.0) * 500)
            wired_gb = wired_mb * 1048576 / 1e9
            cap = _wired_limit_cap_gb(machine.ram_gb)
            if cap is not None and wired_gb <= cap:
                # Le levier ne laisse pas macOS sans mémoire : au moins max(3 Go, 5 % de la RAM) restent libres.
                reason = ("%s : seulement en relevant la limite Metal (sudo sysctl iogpu.wired_limit_mb=%d, laisse "
                          "~%s à macOS) ou avec un déchargement partiel vers le CPU (%s)."
                          % (head, wired_mb, _gb(machine.ram_gb - wired_gb), _tok_label(tok_s)))
            else:
                reserve = max(WIRED_LIMIT_OS_RESERVE_GB, WIRED_LIMIT_OS_RESERVE_RATIO * machine.ram_gb)
                reason = ("%s : seulement avec un déchargement partiel vers le CPU (%s), en fermant les autres "
                          "applications ; relever iogpu.wired_limit_mb ne laisserait pas les ~%s dont macOS a besoin, "
                          "déconseillé." % (head, _tok_label(tok_s), _gb(reserve)))
        else:
            reason = ("%s de poids + cache + marge ≈ %s à %s de contexte dépassent %s mais tiennent dans vos %s "
                      "de RAM : risque de swap, à n'essayer qu'en fermant les autres applications (%s)."
                      % (size_s, _gb(need_min), _ctx_label(ctx_min), _budget_phrase(budget), ram_s, _tok_label(tok_s)))
    else:
        verdict = VERDICT_TOO_BIG
        need_shown, ctx_shown = need_min, ctx_min
        if machine.mode == MODE_GPU:
            limit, limit_label = (machine.budget_offload_gb or budget), "exploitables (VRAM + RAM) par"
        elif machine.ram_gb:
            limit, limit_label = max(0.0, machine.ram_gb - 2.0), "exploitables sur"
        else:
            limit, limit_label = budget, "utilisables par"
        reason = ("%s de poids + cache + marge ≈ %s même à %s de contexte, au-delà des ~%s %s %s."
                  % (size_s, _gb(need_min), _ctx_label(ctx_min), _gb(limit), limit_label, hw))

    return Recommendation(
        tag=tag, family=str(model.get("family", "")), category=category,
        params_b=_num(model.get("params_b")), active_params_b=active, size_gb=round(size, 3),
        context_window=ctx_window, quality_rank=int(_num(model.get("quality_rank")) or 9),
        capabilities=caps, license=str(model.get("license", "")), verdict=verdict, reason=reason,
        pull_command="ollama pull %s" % tag, installed=_norm_tag(tag) in installed,
        memory_needed_gb=round(need_shown, 2), estimated_tokens_per_s=tok_s, notes=str(model.get("notes", "")),
        num_ctx_max=num_ctx_max, license_warning=_license_warning(str(model.get("license", ""))),
        memory_ctx=int(ctx_shown),
    )


def _sort_key(rec: Recommendation):
    return (VERDICT_ORDER.get(rec.verdict, 9), rec.quality_rank, -rec.size_gb)


def _best_in(recs: Sequence[Recommendation], category: str) -> Optional[Recommendation]:
    pool = [r for r in recs if r.category == category and r.verdict == VERDICT_RECOMMENDED]
    if not pool:
        return None
    # MODEL_RULES §5 : jamais moins de 8k pour un assistant de code → un modèle qui tient à ≥ 8k passe devant
    # un modèle mieux classé qui ne tient qu'à 4k.
    roomy = [r for r in pool if (r.num_ctx_max or 0) >= MIN_CODE_CTX]
    pool = roomy or pool
    pool.sort(key=_sort_key)
    with_tools = [r for r in pool if "tools" in r.capabilities]
    return with_tools[0] if with_tools else pool[0]


# --------------------------------------------------------------------------- moteur et étapes

def _engine_state(engines: Any) -> Dict[str, Any]:
    if engines is None:
        return {"probed": False, "ollama_installed": False, "ollama_running": False, "version": "",
                "base_url": OLLAMA_URL, "ollama_binary": None,
                "docker": {"available": False, "running": False, "ollama_container": None},
                "openai_servers": [], "recommended_engine": "none", "recommended_base_url": ""}
    probe = _pget(engines, "ollama")
    running = bool(_pget(probe, "reachable", False))
    binary = _pget(engines, "ollama_binary")
    container = _pget(engines, "docker_ollama_container")
    servers = []
    for p in list(_pget(engines, "openai_servers", []) or []):
        if _pget(p, "reachable", False):
            names = [n if isinstance(n, str) else _pget(n, "name", "") for n in list(_pget(p, "models", []) or [])]
            servers.append({"label": str(_pget(p, "label", "")), "base_url": str(_pget(p, "base_url", "")),
                            "models": [n for n in names if n]})
    return {
        "probed": True,
        "ollama_installed": bool(binary) or running or bool(container),
        "ollama_running": running,
        "version": str(_pget(probe, "version", "") or ""),
        "base_url": str(_pget(probe, "base_url", OLLAMA_URL) or OLLAMA_URL),
        "ollama_binary": binary,
        "docker": {"available": bool(_pget(engines, "docker_available", False)),
                   "running": bool(_pget(engines, "docker_running", False)),
                   "ollama_container": container},
        "openai_servers": servers,
        "recommended_engine": str(_pget(engines, "recommended_engine", "none") or "none"),
        "recommended_base_url": str(_pget(engines, "recommended_base_url", "") or ""),
    }


def _os_key(profile: Any) -> str:
    os_name = str(_pget(profile, "os", "") or "").lower()
    return os_name if os_name in ("macos", "linux", "windows") else "other"


def _install_steps(profile: Any, machine: _Machine, engine: Dict[str, Any], best_chat: Optional[Recommendation],
                   best_emb: Optional[Recommendation], num_ctx: int) -> List[InstallStep]:
    os_key = _os_key(profile)
    apple = bool(_pget(profile, "apple_silicon", False))
    installed, running = engine["ollama_installed"], engine["ollama_running"]
    container = engine["docker"]["ollama_container"]
    docker_only = bool(container) and not engine.get("ollama_binary")
    has_nvidia = machine.gpu_vendor == "nvidia"
    has_amd = machine.gpu_vendor == "amd"
    steps: List[InstallStep] = []

    if not installed:
        if os_key == "macos":
            notes = ("Sans Homebrew : téléchargez l'application officielle (https://ollama.com/download/Ollama.dmg), "
                     "glissez-la dans /Applications et lancez-la une fois ; ou `brew install --cask ollama-app`. "
                     "macOS 14 (Sonoma) ou plus récent requis.")
            if not apple:
                notes += " Mac Intel : CPU seul, pas d'accélération Metal."
            steps.append(InstallStep("Installer Ollama", ["brew install ollama", "brew services start ollama"], notes))
        elif os_key == "windows":
            steps.append(InstallStep(
                "Installer Ollama", ["irm https://ollama.com/install.ps1 | iex"],
                "Ou l'installateur officiel https://ollama.com/download/OllamaSetup.exe (sans droits administrateur ; "
                "OllamaSetup.exe /DIR=\"D:\\Ollama\" pour choisir l'emplacement). Windows 10 22H2 ou 11 requis ; "
                "pilote NVIDIA 551.61+ ou ROCm/Vulkan pour AMD."))
        else:
            notes = ("Installe le binaire et le service systemd, détecte les GPU NVIDIA (pilote ≥ 550) et AMD (ROCm). "
                     "Vérifier ensuite : systemctl status ollama.")
            if os_key == "other":
                notes = "Système non reconnu : voir https://ollama.com/download. " + notes
            steps.append(InstallStep("Installer Ollama", ["curl -fsSL https://ollama.com/install.sh | sh"], notes))
    elif not running:
        if docker_only:
            steps.append(InstallStep("Démarrer Ollama", ["docker start %s" % container],
                                     "Le conteneur Docker « %s » existe mais ne répond pas sur %s." % (container, engine["base_url"])))
        elif os_key == "macos":
            steps.append(InstallStep("Démarrer Ollama", ["brew services start ollama"],
                                     "Si Ollama a été installé comme application : lancez Ollama.app (icône dans la barre "
                                     "de menus) ; sinon `ollama serve` dans un terminal."))
        elif os_key == "windows":
            steps.append(InstallStep("Démarrer Ollama", ["ollama serve"],
                                     "Ou lancez Ollama depuis le menu Démarrer (icône dans la zone de notification)."))
        else:
            steps.append(InstallStep("Démarrer Ollama", ["sudo systemctl start ollama"],
                                     "Sans systemd : `ollama serve` dans un terminal ; logs : journalctl -e -u ollama."))

    if not running:
        if os_key == "windows":
            cmds = ["Invoke-RestMethod http://127.0.0.1:11434/api/version", "Invoke-RestMethod http://127.0.0.1:11434/api/tags"]
        else:
            cmds = ["curl -s http://127.0.0.1:11434/api/version", "curl -s http://127.0.0.1:11434/api/tags"]
        steps.append(InstallStep("Vérifier le service", cmds,
                                 "Réponse attendue : {\"version\": …} puis la liste des modèles (vide au début). "
                                 "Si « address already in use » : le service tourne déjà, ne relancez pas `ollama serve`."))

    missing = [r for r in (best_chat, best_emb) if r is not None and not r.installed]
    if missing:
        prefix = "docker exec -it %s " % container if docker_only else ""
        cmds = [prefix + r.pull_command for r in missing]
        total = sum(r.size_gb for r in missing)
        notes = "Seule étape nécessitant Internet : ~%s à télécharger." % _gb(total)
        disk = _num(_pget(profile, "disk_free_gb"))
        if disk is not None:
            notes += " Disque : %s libres." % _gb(disk)
        notes += (" Si le disque système est petit, définissez OLLAMA_MODELS vers un autre volume avant le premier pull."
                  " Vérifier ensuite : ollama list.")
        steps.append(InstallStep("Télécharger les modèles", cmds, notes))

    if os_key == "macos":
        ctx_cmds = ['launchctl setenv OLLAMA_FLASH_ATTENTION "1"', 'launchctl setenv OLLAMA_KV_CACHE_TYPE "q8_0"',
                    'launchctl setenv OLLAMA_KEEP_ALIVE "30m"']
        ctx_extra = "Puis quittez Ollama (barre de menus) et relancez-le ; brew : brew services restart ollama."
    elif os_key == "windows":
        ctx_cmds = ['[Environment]::SetEnvironmentVariable("OLLAMA_FLASH_ATTENTION", "1", "User")',
                    '[Environment]::SetEnvironmentVariable("OLLAMA_KV_CACHE_TYPE", "q8_0", "User")',
                    '[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "30m", "User")']
        ctx_extra = "Puis quittez Ollama (zone de notification) et relancez-le depuis le menu Démarrer."
    else:
        ctx_cmds = ["sudo systemctl edit ollama.service",
                    "sudo systemctl daemon-reload && sudo systemctl restart ollama"]
        ctx_extra = ('Dans l\'éditeur, sous [Service] : Environment="OLLAMA_FLASH_ATTENTION=1" '
                     'Environment="OLLAMA_KV_CACHE_TYPE=q8_0" Environment="OLLAMA_KEEP_ALIVE=30m".')
    if best_chat is not None:
        kv = kv_cache_gb(num_ctx, best_chat.params_b or 1.0)
        ctx_notes = ("Contexte conseillé pour %s : %d tokens (~%s de cache KV, %s de poids). Develop le passe dans "
                     "options.num_ctx (défaut Ollama : 4096). " % (best_chat.tag, num_ctx, _gb(kv), _gb(best_chat.size_gb)))
    else:
        ctx_notes = "Contexte conseillé : %d tokens, à passer dans options.num_ctx (défaut Ollama : 4096). " % num_ctx
    ctx_notes += "OLLAMA_KV_CACHE_TYPE=q8_0 et OLLAMA_FLASH_ATTENTION=1 divisent le cache par 2. " + ctx_extra
    steps.append(InstallStep("Régler le contexte", ctx_cmds, ctx_notes))

    present = [r.tag for r in (best_chat, best_emb) if r is not None]
    steps.append(InstallStep(
        "Vérifier dans Develop", ["develop doctor"],
        "Ou, dans l'assistant de configuration (écran « Moteur d'IA »), bouton « Vérifier » : Develop lit "
        "%s/api/tags, confirme la présence de %s, puis teste un appel de chat et un embedding."
        % (engine["base_url"], " et ".join(present) if present else "vos modèles")))

    if engine["docker"]["available"]:
        # Une seule commande `docker run --name ollama` est proposée (deux à la suite : la seconde échouerait,
        # « Conflict. The container name "/ollama" is already in use ») ; la variante alternative est dans la note.
        base = "docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama"
        gpu_cmd: Optional[str] = None
        if has_nvidia and os_key != "macos":
            gpu_cmd = "docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama"
        elif has_amd and os_key == "linux":
            gpu_cmd = ("docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama "
                       "-p 11434:11434 --name ollama ollama/ollama:rocm")
        name = str(container) if container else "ollama"
        cmds: List[str] = []
        if container:
            cmds.append("docker start %s" % name)
            notes = ("Conteneur « %s » déjà présent : docker start suffit (un nouveau docker run --name %s "
                     "échouerait). Pour le recréer%s : docker rm -f %s puis %s."
                     % (name, name, " avec le GPU" if gpu_cmd else "", name, gpu_cmd or base))
        else:
            cmds.append(gpu_cmd or base)
            notes = ("Une seule des deux variantes : avec le GPU (ci-dessus) ou, sans GPU : %s." % base) if gpu_cmd else ""
        if best_chat is not None:
            cmds.append("docker exec -it %s ollama pull %s" % (name, best_chat.tag))
        if os_key == "macos":
            notes += (" Sur Mac, Docker Desktop n'accède pas au GPU (pas de passthrough Metal) : le conteneur tournera "
                      "sur CPU, préférez Ollama natif. Le volume « ollama » conserve les modèles.")
        elif os_key == "windows":
            notes += (" Docker Desktop + WSL 2 + pilote NVIDIA : --gpus=all est le seul passthrough GPU supporté. "
                      "Préférez l'installateur natif sur un poste de travail.")
        else:
            notes += " Le volume nommé « ollama » conserve les modèles entre recréations."
            if has_nvidia:
                notes += (" GPU NVIDIA : installer d'abord le NVIDIA Container Toolkit (sudo apt-get install -y "
                          "nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && "
                          "sudo systemctl restart docker).")
        steps.append(InstallStep("Alternative : Ollama dans Docker", cmds, notes.strip(), optional=True))

    if os_key == "macos":
        alt_cmds = ["brew install --cask lm-studio", "lms server start --port 1234", "brew install llama.cpp",
                    "llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M -c 16384 -ngl all --port 9931"]
    elif os_key == "windows":
        alt_cmds = ["irm https://lmstudio.ai/install.ps1 | iex", "lms daemon up", "lms server start --port 1234",
                    "winget install llama.cpp",
                    "llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M -c 16384 -ngl all --port 9931"]
    else:
        alt_cmds = ["curl -fsSL https://lmstudio.ai/install.sh | bash", "lms daemon up", "lms server start --port 1234",
                    "brew install llama.cpp",
                    "llama-server -hf ggml-org/gemma-3-4b-it-GGUF:Q4_K_M -c 16384 -ngl all --port 9931"]
    if machine.mode == MODE_UNKNOWN:
        budget_hint = "Choisissez une quantisation Q4_K_M adaptée à la mémoire de la machine (non détectée ici)."
    elif machine.budget_gb <= 0:
        budget_hint = "Choisissez la plus petite quantisation disponible : aucun budget mémoire sûr sur cette machine."
    else:
        budget_hint = "Choisissez une quantisation Q4_K_M dont la taille tient dans le budget de ~%s." % _gb(machine.budget_gb)
    alt_notes = ("URL de base par défaut : LM Studio http://localhost:1234/v1, llama-server http://127.0.0.1:9931/v1 "
                 "(8080 avant le 9 oct. 2026), vLLM http://localhost:8000/v1 ; clé API quelconque (ex. « lm-studio »). "
                 + budget_hint)
    if engine["openai_servers"]:
        alt_notes += " Déjà joignable : " + ", ".join("%s (%s)" % (s["label"], s["base_url"]) for s in engine["openai_servers"]) + "."
    steps.append(InstallStep("Alternative : serveur compatible OpenAI (LM Studio, llama.cpp)", alt_cmds, alt_notes, optional=True))
    return steps


# --------------------------------------------------------------------------- avertissements

def _warnings(profile: Any, machine: _Machine, engine: Dict[str, Any], best_chat: Optional[Recommendation],
              best_emb: Optional[Recommendation], num_ctx: int) -> List[str]:
    out: List[str] = list(machine.warnings)
    os_key = _os_key(profile)
    ram = machine.ram_gb
    if ram and ram <= 8:
        # 8 Go inclus : une machine à 8 Go remonte exactement 8,0 et n'a que ~4-6 Go de budget (MODEL_RULES §7).
        if machine.mode == MODE_GPU:
            # Avec un GPU dédié, c'est la VRAM qui porte le modèle : la RAM ne limite que le déchargement partiel.
            out.append("RAM de %s seulement : le déchargement partiel vers la RAM est très limité (VRAM + RAM − 4 Go) ; "
                       "fermez les autres applications pendant l'utilisation." % _gb(ram))
        else:
            out.append("RAM de %s seulement : seuls les petits modèles (≤ 4B) sont envisageables ; fermez les autres "
                       "applications pendant l'utilisation." % _gb(ram))
    if machine.mode == MODE_CPU and best_chat is not None:
        dense_big = (best_chat.params_b or 0) > 14 and best_chat.active_params_b is None
        if dense_big:
            out.append("Aucun GPU dédié : %s (%sB) tournera sur le processeur à %s ; un modèle ≤ 14B ou un MoE "
                       "sera plus agréable." % (best_chat.tag, _fr(best_chat.params_b or 0), _tok_label(best_chat.estimated_tokens_per_s)))
    if os_key == "macos" and engine["docker"]["available"]:
        out.append("Docker Desktop sur Mac n'accède pas au GPU (Metal) : préférez Ollama natif plutôt qu'un conteneur.")
    if best_chat is not None and "tools" not in best_chat.capabilities:
        out.append("%s n'annonce pas d'appels d'outils natifs : le chat utilisera le protocole texte "
                   "(moins fiable pour l'agentique)." % best_chat.tag)
    disk = _num(_pget(profile, "disk_free_gb"))
    if disk is not None:
        to_download = [r for r in (best_chat, best_emb) if r is not None and not r.installed]
        if to_download:
            biggest = max(to_download, key=lambda r: r.size_gb)
            needed = max(biggest.size_gb + 2.0, biggest.size_gb * 1.1 + 1.0) + sum(r.size_gb for r in to_download if r is not biggest)
            if disk < needed:
                out.append("Disque : %s libres sur %s, il en faut au moins %s pour %s (+ marge) : libérez de "
                           "l'espace ou déplacez OLLAMA_MODELS vers un autre volume avant le premier pull."
                           % (_gb(disk), _pget(profile, "disk_path", "") or "le volume des modèles", _gb(needed), biggest.tag))
    slow_prefill = machine.mode == MODE_CPU or (machine.mode == MODE_UNIFIED and (machine.bandwidth_gbs or 0) < 150)
    if num_ctx > 16384 and slow_prefill:
        out.append("Au-delà de 16k de contexte, la lecture du prompt (prefill) devient lente sur cette machine.")
    if best_chat is not None and best_chat.license_warning:
        out.append("%s : %s." % (best_chat.tag, best_chat.license_warning))
    for w in list(_pget(profile, "warnings", []) or []):
        w = str(w)
        if w and w not in out and "RAM totale inconnue" not in w:
            out.append(w)
    deduped: List[str] = []
    for w in out:
        if w not in deduped:
            deduped.append(w)
    return deduped


# --------------------------------------------------------------------------- point d'entrée

def _profile_summary(profile: Any) -> str:
    try:
        from develop.system.profile import describe  # import local : évite tout cycle
        text = describe(profile)
        if text:
            return text
    except Exception:  # noqa: BLE001 - profil partiel ou dict
        pass
    parts = [_OS_LABELS.get(str(_pget(profile, "os", "") or "").lower(), str(_pget(profile, "os", "") or "système inconnu"))]
    cpu = _pget(profile, "cpu_model", "")
    if cpu:
        parts.append(str(cpu))
    ram = _num(_pget(profile, "ram_total_gb"))
    if ram:
        parts.append("%s de %s" % (_gb(ram), "mémoire unifiée" if _pget(profile, "unified_memory", False) else "RAM"))
    return ", ".join(parts)


def advise(profile: Any, engines: Any = None, installed_models: Optional[Sequence[Any]] = None,
           num_ctx: int = 16384, catalog: Optional[Dict[str, Any]] = None) -> Advice:
    """Conseille des modèles et des étapes d'installation pour ``profile`` (MachineProfile ou dict).

    Ne lève jamais sur un profil incomplet ; aucune requête réseau.
    """
    try:
        num_ctx = int(num_ctx) if num_ctx else 16384
    except (TypeError, ValueError):
        num_ctx = 16384
    num_ctx = max(512, num_ctx)
    if catalog is None:
        catalog = load_catalog()
    elif isinstance(catalog, list):
        catalog = {"version": "", "models": list(catalog)}
    models = [m for m in (catalog.get("models") or []) if isinstance(m, dict) and m.get("tag") and not _excluded(str(m["tag"]))]

    machine = _classify(profile)
    installed = _installed_set(installed_models, engines)
    engine = _engine_state(engines)

    recs: List[Recommendation] = []
    for m in models:
        try:
            recs.append(_evaluate(m, machine, num_ctx, installed))
        except Exception as exc:  # noqa: BLE001 - une entrée de catalogue malformée ne bloque pas le reste
            log.warning("modèle %s ignoré : %s", m.get("tag"), exc)
    chat_like = sorted([r for r in recs if r.category != "embedding"], key=_sort_key)
    embeddings = [r for r in recs if r.category == "embedding"]

    best_chat = _best_in(chat_like, "chat_code") or _best_in(chat_like, "general") or _best_in(chat_like, "small")

    emb_ok = [r for r in embeddings if r.verdict == VERDICT_RECOMMENDED]
    emb_ok.sort(key=lambda r: (r.quality_rank, r.size_gb))
    best_emb: Optional[Recommendation] = None
    if emb_ok:
        # Un embedding déjà installé du meilleur rang évite un téléchargement et une réindexation complète.
        top_rank = emb_ok[0].quality_rank
        candidates = [r for r in emb_ok if r.installed and r.quality_rank == top_rank] or emb_ok
        best_emb = candidates[0]
        if machine.ram_gb and machine.ram_gb >= LARGE_EMBEDDING_MIN_RAM_GB:
            # MODEL_RULES §6.3 : qwen3-embedding:4b (meilleur pour le code) dès 16 Go, s'il tient dans le budget.
            large = next((r for r in candidates if _norm_tag(r.tag) == _norm_tag(PREFERRED_LARGE_EMBEDDING)
                          and r.quality_rank <= best_emb.quality_rank), None)
            if large is not None:
                best_emb = large
    if best_emb is None and embeddings:
        limit = (machine.ram_gb - 2.0) if machine.ram_gb else None
        light = [r for r in embeddings if _norm_tag(r.tag) in {_norm_tag(t) for t in LIGHT_EMBEDDINGS}]
        light.sort(key=lambda r: r.size_gb)
        candidates = [r for r in light if limit is None or r.memory_needed_gb <= limit] or light or sorted(embeddings, key=lambda r: r.size_gb)
        best_emb = candidates[0]
        best_emb.verdict = VERDICT_RECOMMENDED
        best_emb.reason = ("Embedding le plus léger du catalogue (%s de poids) : fonctionne même sur une machine "
                           "modeste, indispensable pour indexer le dépôt." % _gb(best_emb.size_gb))
    embeddings.sort(key=lambda r: (VERDICT_ORDER.get(r.verdict, 9), r.quality_rank, r.size_gb))
    if best_emb is not None:
        embeddings = [best_emb] + [r for r in embeddings if r is not best_emb]  # le conseillé toujours en tête

    # Contexte conseillé : ce que le meilleur modèle permet (≥ 8k dès qu'un candidat le permet, cf. _best_in),
    # sinon 8k par défaut pour un assistant de code (MODEL_RULES §5).
    if best_chat is not None:
        advised_ctx = min(num_ctx, best_chat.num_ctx_max or MIN_USEFUL_CTX)
    else:
        advised_ctx = min(num_ctx, MIN_CODE_CTX)

    steps = _install_steps(profile, machine, engine, best_chat, best_emb, advised_ctx)
    warns = _warnings(profile, machine, engine, best_chat, best_emb, advised_ctx)

    return Advice(
        profile_summary=_profile_summary(profile), machine_class=machine.label, mode=machine.mode,
        memory_budget_gb=round(machine.budget_gb, 1), num_ctx=advised_ctx, recommendations=chat_like,
        embedding=embeddings, best_chat=best_chat.tag if best_chat else None,
        best_embedding=best_emb.tag if best_emb else None, install_steps=steps, warnings=warns, engine=engine,
        offline_note=OFFLINE_NOTE, catalog_version=str(catalog.get("version", "") or ""),
    )


# --------------------------------------------------------------------------- rendu texte (CLI)

_WIDTH = 98


def _wrap(text: str, indent: str = "     ", width: int = _WIDTH) -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent,
                         break_long_words=False, break_on_hyphens=False)


def _command_lines(cmd: str, width: int = _WIDTH) -> List[str]:
    """Une commande sur une ligne « $ … », coupée aux espaces avec « \\ » si elle dépasse la largeur."""
    prefix, cont = "     $ ", "         "
    if len(prefix + cmd) <= width:
        return [prefix + cmd]
    parts = textwrap.wrap(cmd, width=width - len(cont) - 2, break_long_words=False, break_on_hyphens=False)
    lines = [(prefix if i == 0 else cont) + part for i, part in enumerate(parts)]
    return [line + " \\" for line in lines[:-1]] + lines[-1:]


def _engine_line(engine: Dict[str, Any]) -> str:
    if not engine.get("probed"):
        return "Moteur : non sondé (lancez `develop doctor` pour détecter Ollama, Docker et les serveurs locaux)"
    if engine["ollama_running"]:
        ollama = "Ollama joignable" + (" (v%s)" % engine["version"] if engine.get("version") else "")
    elif engine["ollama_installed"]:
        ollama = "Ollama installé mais arrêté"
    else:
        ollama = "Ollama absent"
    docker = engine["docker"]
    if docker.get("ollama_container"):
        docker_s = "Docker : conteneur %s" % docker["ollama_container"]
    elif docker.get("available"):
        docker_s = "Docker disponible" + ("" if docker.get("running") else " (arrêté)")
    else:
        docker_s = "Docker absent"
    servers = engine.get("openai_servers") or []
    servers_s = ("serveurs OpenAI : " + ", ".join("%s (%s)" % (s["label"], s["base_url"]) for s in servers)) if servers \
        else "aucun serveur compatible OpenAI"
    return "Moteur : %s ; %s ; %s" % (ollama, docker_s, servers_s)


def format_advice_text(advice: Advice, top: int = 5) -> str:
    """Rendu texte du conseil pour la CLI (``develop doctor``), lisible sur 100 colonnes."""
    lines: List[str] = []
    lines.append(_wrap("Machine : " + advice.profile_summary, indent=""))
    if advice.mode == MODE_UNKNOWN:
        budget_s = "budget mémoire inconnu (RAM non détectée)"
    elif advice.memory_budget_gb <= 0:
        budget_s = "aucun budget mémoire sûr"
    else:
        budget_s = "budget mémoire ~%s" % _gb(advice.memory_budget_gb)
    lines.append(_wrap("Classe : %s — %s, contexte conseillé %d tokens"
                       % (advice.machine_class, budget_s, advice.num_ctx), indent=""))
    lines.append(_wrap(_engine_line(advice.engine), indent="", width=_WIDTH))
    lines.append("")

    shown = list(advice.recommendations[:top])
    best = next((r for r in advice.recommendations if r.tag == advice.best_chat), None)
    if best is not None and best not in shown:  # le meilleur choix est toujours affiché, même mal classé
        shown = shown[: max(0, top - 1)] + [best]
    counts = {v: sum(1 for r in advice.recommendations if r.verdict == v) for v in VERDICT_ORDER}
    lines.append("Modèles (%d meilleurs sur %d — %d recommandés, %d possibles mais lents, %d trop gros) :"
                 % (len(shown), len(advice.recommendations), counts[VERDICT_RECOMMENDED],
                    counts[VERDICT_POSSIBLE_SLOW], counts[VERDICT_TOO_BIG]))
    if not shown:
        lines.append("  (catalogue vide)")
    for i, r in enumerate(shown, 1):
        flags = []
        if r.installed:
            flags.append("installé")
        if r.tag == advice.best_chat:
            flags.append("meilleur choix")
        head = "  %d. %s [%s] %s, %s%s" % (i, r.tag, r.verdict_label, _gb(r.size_gb),
                                           CATEGORY_LABELS.get(r.category, r.category),
                                           (" — " + ", ".join(flags)) if flags else "")
        lines.append(_wrap(head, indent="", width=_WIDTH))
        lines.append(_wrap(r.reason))
    lines.append("")

    emb = next((r for r in advice.embedding if r.tag == advice.best_embedding), None)
    if emb is not None:
        lines.append("Embedding conseillé : %s (%s)%s" % (emb.tag, _gb(emb.size_gb), " — installé" if emb.installed else ""))
        lines.append(_wrap(emb.reason))
    else:
        lines.append("Embedding conseillé : aucun (catalogue vide)")
    lines.append("")

    lines.append("Étapes d'installation :")
    for i, step in enumerate(advice.install_steps, 1):
        lines.append("  %d. %s%s" % (i, step.title, " (optionnel)" if step.optional else ""))
        for cmd in step.commands:
            lines.extend(_command_lines(cmd))
        if step.notes:
            lines.append(_wrap(step.notes))
    lines.append("")

    if advice.warnings:
        lines.append("Avertissements :")
        for w in advice.warnings:
            lines.append(textwrap.fill(w, width=_WIDTH, initial_indent="  - ", subsequent_indent="    ",
                                       break_long_words=False, break_on_hyphens=False))
        lines.append("")
    lines.append(_wrap(advice.offline_note, indent=""))
    return "\n".join(lines)


__all__ = [
    "Advice", "InstallStep", "Recommendation", "advise", "describe_tier", "estimate_speed", "format_advice_text",
    "kv_cache_gb", "load_catalog", "memory_needed_gb", "OFFLINE_NOTE",
    "VERDICT_RECOMMENDED", "VERDICT_POSSIBLE_SLOW", "VERDICT_TOO_BIG", "VERDICT_LABELS",
]
