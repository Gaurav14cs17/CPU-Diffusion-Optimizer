"""Profile report serialization (JSON + human-readable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.core.types import JSONDict
from engine.profiler.cpu_info import CPUInfo


@dataclass
class Hotspot:
    name: str
    latency_ms: float
    share_pct: float

    def to_dict(self) -> JSONDict:
        return {
            "name": self.name,
            "latency_ms": self.latency_ms,
            "share_pct": self.share_pct,
        }


@dataclass
class ProfileReport:
    model_name: str
    resolution: str
    steps: int
    device: str
    total_latency_ms: float
    mean_step_latency_ms: float
    per_step_latency_ms: list[float]
    block_latency_ms: dict[str, float]
    operator_latency_ms: dict[str, float]
    peak_ram_bytes: int
    batch_size: int
    dtype: str
    cpu: CPUInfo
    hotspots: list[Hotspot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "model_name": self.model_name,
            "resolution": self.resolution,
            "steps": self.steps,
            "device": self.device,
            "total_latency_ms": self.total_latency_ms,
            "total_latency_sec": self.total_latency_ms / 1000.0,
            "mean_step_latency_ms": self.mean_step_latency_ms,
            "per_step_latency_ms": list(self.per_step_latency_ms),
            "block_latency_ms": dict(self.block_latency_ms),
            "operator_latency_ms": dict(self.operator_latency_ms),
            "peak_ram_bytes": self.peak_ram_bytes,
            "batch_size": self.batch_size,
            "dtype": self.dtype,
            "cpu": self.cpu.to_dict(),
            "hotspots": [h.to_dict() for h in self.hotspots],
            "metadata": dict(self.metadata),
        }

    def to_human_readable(self) -> str:
        lines = [
            "CPU PROFILE",
            "",
            f"Model: {self.model_name}",
            f"Resolution: {self.resolution}",
            f"Steps: {self.steps}",
            f"Device: {self.device.upper()}",
            f"Batch: {self.batch_size}",
            f"Dtype: {self.dtype}",
            "",
            f"Total: {self.total_latency_ms / 1000.0:.3f} sec",
            f"Mean step: {self.mean_step_latency_ms:.2f} ms",
            f"Peak RAM: {self.peak_ram_bytes / (1024 ** 2):.1f} MiB",
            "",
            "Top hotspots:",
        ]
        if not self.hotspots:
            lines.append("  (none measured)")
        else:
            for hotspot in self.hotspots[:10]:
                lines.append(f"  {hotspot.name:<24} {hotspot.share_pct:5.1f}%")
        lines.extend(
            [
                "",
                "CPU:",
                f"  arch={self.cpu.architecture} cores={self.cpu.logical_cores} "
                f"avx2={self.cpu.avx2} avx512={self.cpu.avx512} neon={self.cpu.neon}",
            ]
        )
        return "\n".join(lines)
