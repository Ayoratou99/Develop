"""Profil matériel de la machine (stdlib uniquement, psutil optionnel).

Tout est tolérant aux pannes : chaque sonde a un timeout court et renvoie ``None``
plutôt que de lever une exception, pour que l'assistant de configuration puisse
toujours afficher *quelque chose*.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_GB = 1024 ** 3
# Sous Windows, aucune sonde ne doit faire apparaître de console (fenêtre native « develop app »).
_NO_WINDOW: Dict[str, int] = ({"creationflags": subprocess.CREATE_NO_WINDOW}  # type: ignore[attr-defined]
                              if hasattr(subprocess, "CREATE_NO_WINDOW") else {})


@dataclass
class GpuInfo:
    vendor: str  # "apple" | "nvidia" | "amd" | "intel" | "unknown"
    name: str
    vram_gb: Optional[float] = None  # None pour la mémoire unifiée Apple
    cores: Optional[int] = None  # coeurs GPU Apple si connus


@dataclass
class MachineProfile:
    os: str  # "macos" | "linux" | "windows" | autre
    os_version: str
    arch: str  # "arm64" | "x86_64" | ...
    cpu_model: str
    cpu_cores_physical: Optional[int]
    cpu_cores_logical: Optional[int]
    ram_total_gb: Optional[float]
    ram_available_gb: Optional[float]
    gpus: List[GpuInfo] = field(default_factory=list)
    apple_silicon: bool = False
    unified_memory: bool = False
    disk_free_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None
    disk_path: str = ""
    python_version: str = ""
    warnings: List[str] = field(default_factory=list)

    # ----- aides pour le conseiller -----
    @property
    def accelerator_memory_gb(self) -> Optional[float]:
        """Mémoire exploitable par l'accélérateur: VRAM dédiée, ou ~75 % de la RAM unifiée Apple."""
        if self.unified_memory and self.ram_total_gb:
            return round(self.ram_total_gb * 0.75, 1)
        vrams = [g.vram_gb for g in self.gpus if g.vram_gb]
        return max(vrams) if vrams else None

    @property
    def has_gpu(self) -> bool:
        return self.apple_silicon or any(g.vendor in ("nvidia", "amd") and g.vram_gb for g in self.gpus)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["accelerator_memory_gb"] = self.accelerator_memory_gb
        data["has_gpu"] = self.has_gpu
        return data


# --------------------------------------------------------------------------- utilitaires

def _run(cmd: List[str], timeout: float = 3.0) -> Optional[str]:
    """Exécute une commande et renvoie stdout, ou None en cas d'échec/timeout/absence/refus.

    Décodage UTF-8 avec remplacement : la console Windows (cp1252, cp850) ne garantit pas un flux décodable.
    """
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             timeout=timeout, check=False, **_NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _to_gb(n_bytes: Optional[float]) -> Optional[float]:
    if n_bytes is None:
        return None
    return round(n_bytes / _GB, 1)


def _os_name() -> str:
    s = sys.platform
    if s == "darwin":
        return "macos"
    if s.startswith("linux"):
        return "linux"
    if s.startswith("win"):
        return "windows"
    return s


def _arch() -> str:
    m = platform.machine().lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("x86_64", "amd64"):
        return "x86_64"
    return m or "unknown"


# --------------------------------------------------------------------------- macOS

def _macos_sysctl(key: str) -> Optional[str]:
    return _run(["sysctl", "-n", key], timeout=2.0)


def _macos_ram_available() -> Optional[float]:
    """Estime la mémoire disponible via vm_stat (pages libres + inactives + spéculatives)."""
    out = _run(["vm_stat"], timeout=2.0)
    if not out:
        return None
    page_size = 4096
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page_size = int(m.group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(label + r":\s+(\d+)", out)
        if m:
            pages += int(m.group(1))
    return _to_gb(pages * page_size) if pages else None


def _macos_gpus(cpu_model: str, apple_silicon: bool) -> List[GpuInfo]:
    gpus: List[GpuInfo] = []
    # system_profiler peut prendre plusieurs secondes : timeout court, et un repli raisonnable.
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=6.0)
    if out:
        try:
            data = json.loads(out)
            for item in data.get("SPDisplaysDataType", []):
                name = item.get("sppci_model") or item.get("_name") or "GPU"
                cores = None
                raw_cores = item.get("sppci_cores")
                if raw_cores and str(raw_cores).isdigit():
                    cores = int(raw_cores)
                vram = None
                raw_vram = item.get("spdisplays_vram") or item.get("spdisplays_vram_shared")
                if raw_vram:
                    mv = re.search(r"([\d.]+)\s*(GB|MB)", str(raw_vram))
                    if mv:
                        vram = float(mv.group(1)) / (1024 if mv.group(2) == "MB" else 1)
                vendor = "apple" if "apple" in name.lower() else (
                    "nvidia" if "nvidia" in name.lower() else ("amd" if ("amd" in name.lower() or "radeon" in name.lower()) else "intel" if "intel" in name.lower() else "unknown"))
                gpus.append(GpuInfo(vendor=vendor, name=name, vram_gb=vram if vendor != "apple" else None, cores=cores))
        except (ValueError, TypeError):
            pass
    if not gpus and apple_silicon:
        gpus.append(GpuInfo(vendor="apple", name=cpu_model or "Apple GPU", vram_gb=None, cores=None))
    return gpus


def _detect_macos(profile: MachineProfile) -> None:
    profile.cpu_model = _macos_sysctl("machdep.cpu.brand_string") or platform.processor() or ""
    profile.apple_silicon = profile.arch == "arm64" and "apple" in profile.cpu_model.lower() or (
        profile.arch == "arm64" and not profile.cpu_model)
    profile.unified_memory = profile.apple_silicon
    phys = _macos_sysctl("hw.physicalcpu")
    logi = _macos_sysctl("hw.logicalcpu")
    profile.cpu_cores_physical = int(phys) if phys and phys.isdigit() else None
    profile.cpu_cores_logical = int(logi) if logi and logi.isdigit() else os.cpu_count()
    mem = _macos_sysctl("hw.memsize")
    profile.ram_total_gb = _to_gb(int(mem)) if mem and mem.isdigit() else None
    profile.ram_available_gb = _macos_ram_available()
    profile.gpus = _macos_gpus(profile.cpu_model, profile.apple_silicon)
    if profile.apple_silicon and not profile.cpu_model:
        profile.cpu_model = "Apple Silicon"


# --------------------------------------------------------------------------- Linux

def _linux_meminfo() -> Dict[str, int]:
    info: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":") and parts[1].isdigit():
                    info[parts[0][:-1]] = int(parts[1]) * 1024  # kB -> bytes
    except OSError:
        pass
    return info


def _linux_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or ""


def _nvidia_gpus() -> List[GpuInfo]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], timeout=4.0)
    gpus: List[GpuInfo] = []
    if not out:
        return gpus
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1].replace(".", "", 1).isdigit():
            gpus.append(GpuInfo(vendor="nvidia", name=parts[0], vram_gb=round(float(parts[1]) / 1024, 1)))
        elif parts and parts[0]:
            gpus.append(GpuInfo(vendor="nvidia", name=parts[0]))
    return gpus


def _amd_gpus_linux() -> List[GpuInfo]:
    gpus: List[GpuInfo] = []
    drm = Path("/sys/class/drm")
    if not drm.exists():
        return gpus
    for card in sorted(drm.glob("card[0-9]")):
        vendor_file = card / "device" / "vendor"
        try:
            vendor = vendor_file.read_text().strip().lower()
        except OSError:
            continue
        if vendor != "0x1002":  # AMD
            continue
        vram = None
        vram_file = card / "device" / "mem_info_vram_total"
        try:
            vram = _to_gb(int(vram_file.read_text().strip()))
        except (OSError, ValueError):
            pass
        gpus.append(GpuInfo(vendor="amd", name="AMD GPU (%s)" % card.name, vram_gb=vram))
    return gpus


def _detect_linux(profile: MachineProfile) -> None:
    profile.cpu_model = _linux_cpu_model()
    profile.cpu_cores_logical = os.cpu_count()
    try:
        # coeurs physiques : paires (physical id, core id) uniques
        pairs = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip() and phys is not None and core is not None:
                    pairs.add((phys, core))
                    phys = core = None
        profile.cpu_cores_physical = len(pairs) or None
    except OSError:
        profile.cpu_cores_physical = None
    mem = _linux_meminfo()
    profile.ram_total_gb = _to_gb(mem.get("MemTotal"))
    profile.ram_available_gb = _to_gb(mem.get("MemAvailable"))
    profile.gpus = _nvidia_gpus() + _amd_gpus_linux()


# --------------------------------------------------------------------------- Windows

def _windows_memory() -> Dict[str, Optional[float]]:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]  # AttributeError hors Windows
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) or not status.ullTotalPhys:
            return {"total": None, "available": None}  # appel refusé (renvoie 0) : rien d'inventé
        return {"total": _to_gb(status.ullTotalPhys), "available": _to_gb(status.ullAvailPhys)}
    except Exception:  # noqa: BLE001 - toute erreur ctypes
        return {"total": None, "available": None}


_WIN_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
_WIN_DISPLAY_CLASS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
_WIN_ADAPTER_RAM_CAP = 4095 * 1024 * 1024  # Win32_VideoController.AdapterRAM est un uint32 : faux au-delà de 4 Go
_WIN_INVENTORY_SCRIPT = (
    "$ErrorActionPreference = 'SilentlyContinue'; "
    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    "$cpu = @(Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores); "
    "$gpu = @(Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM); "
    "@{ cpu = $cpu; gpu = $gpu } | ConvertTo-Json -Compress -Depth 3"
)


def _gpu_vendor(name: str) -> str:
    low = name.lower()
    if "nvidia" in low:
        return "nvidia"
    if "amd" in low or "radeon" in low:
        return "amd"
    if "intel" in low:
        return "intel"
    return "unknown"


def _windows_registry_value(key: str, name: str):  # type: ignore[no-untyped-def]
    """Valeur ``HKLM\\key\\name`` ou ``None`` (clé absente, droits insuffisants, hors Windows)."""
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            value, _kind = winreg.QueryValueEx(handle, name)
            return value
    except OSError:
        return None


def _windows_registry_subkeys(key: str) -> List[str]:
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return []
    names: List[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            count = winreg.QueryInfoKey(handle)[0]
            for i in range(count):
                try:
                    names.append(winreg.EnumKey(handle, i))
                except OSError:
                    break
    except OSError:
        pass
    return names


def _windows_cpu_model() -> str:
    """Nom commercial du processeur (registre, sans sous-processus) ; ``platform.processor`` en repli."""
    name = _windows_registry_value(_WIN_CPU_KEY, "ProcessorNameString")
    text = " ".join(str(name).split()) if name else ""
    return text or platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")


def _windows_gpus_registry() -> List[GpuInfo]:
    """Cartes graphiques d'après la classe « Display » du registre : nom du pilote et VRAM 64 bits
    (``HardwareInformation.qwMemorySize``), fiable au-delà de 4 Go contrairement à WMI. Sans sous-processus."""
    gpus: List[GpuInfo] = []
    seen: set = set()
    for sub in _windows_registry_subkeys(_WIN_DISPLAY_CLASS_KEY):
        if not sub.isdigit():
            continue
        key = _WIN_DISPLAY_CLASS_KEY + "\\" + sub
        name = _windows_registry_value(key, "DriverDesc")
        label = " ".join(str(name).split()) if name else ""
        if not label or label in seen or "microsoft" in label.lower():  # Basic/Remote Display Adapter : pas un GPU
            continue
        raw = _windows_registry_value(key, "HardwareInformation.qwMemorySize")
        if isinstance(raw, (bytes, bytearray)):
            raw = int.from_bytes(bytes(raw)[:8], "little") if raw else 0
        vram = _to_gb(float(raw)) if isinstance(raw, int) and raw > 0 else None
        seen.add(label)
        gpus.append(GpuInfo(vendor=_gpu_vendor(label), name=label, vram_gb=vram))
    return gpus


def _windows_inventory() -> Tuple[Optional[int], List[GpuInfo]]:
    """Un seul appel PowerShell (``Get-CimInstance``, timeout 10 s) : coeurs physiques et cartes graphiques.

    ``AdapterRAM`` est ignoré au-delà de 4 Go (compteur 32 bits). ``(None, [])`` si PowerShell manque,
    échoue ou dépasse le délai : jamais d'exception.
    """
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        return None, []
    out = _run([shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
                _WIN_INVENTORY_SCRIPT], timeout=10.0)
    if not out:
        return None, []
    try:
        data = json.loads(out)
    except ValueError:
        return None, []
    if not isinstance(data, dict):
        return None, []

    def _items(value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict):
            return [value]
        return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []

    cores = 0
    for item in _items(data.get("cpu")):
        n = item.get("NumberOfCores")
        if isinstance(n, (int, float)) and n > 0:
            cores += int(n)
    gpus: List[GpuInfo] = []
    for item in _items(data.get("gpu")):
        name = " ".join(str(item.get("Name") or "").split())
        if not name:
            continue
        ram = item.get("AdapterRAM")
        vram = _to_gb(float(ram)) if isinstance(ram, (int, float)) and 0 < ram < _WIN_ADAPTER_RAM_CAP else None
        gpus.append(GpuInfo(vendor=_gpu_vendor(name), name=name, vram_gb=vram))
    return (cores or None), gpus


def _detect_windows(profile: MachineProfile) -> None:
    profile.cpu_model = _windows_cpu_model()
    profile.cpu_cores_logical = os.cpu_count()
    profile.cpu_cores_physical = None
    mem = _windows_memory()
    profile.ram_total_gb = mem["total"]
    profile.ram_available_gb = mem["available"]
    gpus = _nvidia_gpus() or _windows_gpus_registry()
    cores, cim_gpus = _windows_inventory()
    profile.cpu_cores_physical = cores
    profile.gpus = gpus or cim_gpus


# --------------------------------------------------------------------------- commun

def _psutil_fill(profile: MachineProfile) -> None:
    """Complète avec psutil s'il est installé (optionnel)."""
    try:
        import psutil  # type: ignore
    except ImportError:
        return
    try:
        vm = psutil.virtual_memory()
        profile.ram_total_gb = profile.ram_total_gb or _to_gb(vm.total)
        profile.ram_available_gb = profile.ram_available_gb or _to_gb(vm.available)
        profile.cpu_cores_physical = profile.cpu_cores_physical or psutil.cpu_count(logical=False)
        profile.cpu_cores_logical = profile.cpu_cores_logical or psutil.cpu_count(logical=True)
    except Exception:  # noqa: BLE001
        pass


def detect_machine(disk_path: Optional[str] = None) -> MachineProfile:
    """Détecte la machine. ``disk_path`` : volume où seront stockés les modèles (défaut: home)."""
    profile = MachineProfile(
        os=_os_name(), os_version=platform.platform(), arch=_arch(), cpu_model="",
        cpu_cores_physical=None, cpu_cores_logical=os.cpu_count(),
        ram_total_gb=None, ram_available_gb=None, python_version=platform.python_version(),
    )
    try:
        if profile.os == "macos":
            _detect_macos(profile)
        elif profile.os == "linux":
            _detect_linux(profile)
        elif profile.os == "windows":
            _detect_windows(profile)
    except Exception as exc:  # noqa: BLE001 - jamais planter l'assistant de configuration
        profile.warnings.append("Détection partielle: %s" % exc)
    _psutil_fill(profile)

    path = disk_path or os.environ.get("OLLAMA_MODELS") or str(Path.home())
    profile.disk_path = path
    try:
        usage = shutil.disk_usage(path)
        profile.disk_free_gb = _to_gb(usage.free)
        profile.disk_total_gb = _to_gb(usage.total)
    except OSError as exc:
        profile.warnings.append("Espace disque inconnu pour %s: %s" % (path, exc))

    if profile.ram_total_gb is None:
        profile.warnings.append("RAM totale inconnue : les recommandations seront prudentes.")
    return profile


def describe(profile: MachineProfile) -> str:
    """Résumé lisible en une ligne, en français."""
    parts = []
    parts.append({"macos": "macOS", "linux": "Linux", "windows": "Windows"}.get(profile.os, profile.os))
    if profile.cpu_model:
        parts.append(profile.cpu_model)
    if profile.cpu_cores_logical:
        parts.append("%d cœurs" % profile.cpu_cores_logical)
    if profile.ram_total_gb:
        parts.append("%s Go de %s" % (("%g" % profile.ram_total_gb), "mémoire unifiée" if profile.unified_memory else "RAM"))
    for g in profile.gpus:
        if g.vendor in ("nvidia", "amd") and g.vram_gb:
            parts.append("%s %g Go VRAM" % (g.name, g.vram_gb))
    if profile.disk_free_gb is not None:
        parts.append("%g Go libres" % profile.disk_free_gb)
    return ", ".join(parts)


if __name__ == "__main__":  # pragma: no cover - usage manuel: python -m develop.system.profile
    p = detect_machine()
    print(describe(p))
    print(json.dumps(p.to_dict(), indent=2, ensure_ascii=False))
