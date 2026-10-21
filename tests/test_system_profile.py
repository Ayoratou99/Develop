from __future__ import annotations

import json

from develop.system import profile as sp


def test_detect_machine_never_raises_and_has_basics():
    p = sp.detect_machine()
    assert p.os in ("macos", "linux", "windows") or isinstance(p.os, str)
    assert p.arch
    assert p.python_version
    assert p.cpu_cores_logical is None or p.cpu_cores_logical >= 1
    # Sérialisable en JSON pour l'API
    json.dumps(p.to_dict())


def test_describe_is_one_readable_line():
    p = sp.detect_machine()
    text = sp.describe(p)
    assert isinstance(text, str) and text and "\n" not in text


def test_accelerator_memory_apple_unified():
    p = sp.MachineProfile(os="macos", os_version="", arch="arm64", cpu_model="Apple M2 Pro",
                          cpu_cores_physical=10, cpu_cores_logical=10, ram_total_gb=32.0,
                          ram_available_gb=10.0, apple_silicon=True, unified_memory=True)
    assert p.has_gpu
    assert p.accelerator_memory_gb == 24.0


def test_accelerator_memory_nvidia():
    p = sp.MachineProfile(os="linux", os_version="", arch="x86_64", cpu_model="AMD Ryzen",
                          cpu_cores_physical=8, cpu_cores_logical=16, ram_total_gb=64.0,
                          ram_available_gb=40.0, gpus=[sp.GpuInfo("nvidia", "RTX 4090", 24.0)])
    assert p.has_gpu
    assert p.accelerator_memory_gb == 24.0


def test_no_gpu_cpu_only():
    p = sp.MachineProfile(os="linux", os_version="", arch="x86_64", cpu_model="Intel",
                          cpu_cores_physical=4, cpu_cores_logical=8, ram_total_gb=16.0, ram_available_gb=8.0)
    assert not p.has_gpu
    assert p.accelerator_memory_gb is None


def test_bad_disk_path_only_warns():
    p = sp.detect_machine(disk_path="/this/path/does/not/exist")
    assert any("disque" in w.lower() for w in p.warnings)
