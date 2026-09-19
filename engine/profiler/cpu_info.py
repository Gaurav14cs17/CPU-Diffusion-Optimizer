"""CPU capability detection."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass
class CPUInfo:
    """Snapshot of host CPU / memory relevant to inference."""

    architecture: str
    processor: str
    physical_cores: int
    logical_cores: int
    total_ram_bytes: int
    available_ram_bytes: int
    avx2: bool = False
    avx512: bool = False
    neon: bool = False
    torch_num_threads: int | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "processor": self.processor,
            "physical_cores": self.physical_cores,
            "logical_cores": self.logical_cores,
            "total_ram_bytes": self.total_ram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "avx2": self.avx2,
            "avx512": self.avx512,
            "neon": self.neon,
            "torch_num_threads": self.torch_num_threads,
            "flags": list(self.flags),
        }


def _read_cpu_flags() -> list[str]:
    flags: list[str] = []
    # Linux /proc/cpuinfo
    cpuinfo = "/proc/cpuinfo"
    if os.path.exists(cpuinfo):
        try:
            with open(cpuinfo, encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.lower().startswith("flags") or line.lower().startswith("features"):
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            flags = parts[1].strip().split()
                        break
        except OSError:
            pass
    return flags


def inspect_cpu() -> CPUInfo:
    """Collect CPU feature flags and memory stats for profiling reports."""
    flags = _read_cpu_flags()
    flag_set = {f.lower() for f in flags}
    arch = platform.machine().lower()

    avx2 = "avx2" in flag_set
    avx512 = any(f.startswith("avx512") for f in flag_set)
    neon = arch in {"aarch64", "arm64", "armv8"} or "neon" in flag_set or "asimd" in flag_set

    torch_threads: int | None = None
    try:
        import torch

        torch_threads = int(torch.get_num_threads())
    except Exception:  # noqa: BLE001
        torch_threads = None

    vm = psutil.virtual_memory()
    return CPUInfo(
        architecture=platform.machine(),
        processor=platform.processor() or platform.machine(),
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count(logical=True) or 1,
        total_ram_bytes=int(vm.total),
        available_ram_bytes=int(vm.available),
        avx2=avx2,
        avx512=avx512,
        neon=neon,
        torch_num_threads=torch_threads,
        flags=flags,
    )
