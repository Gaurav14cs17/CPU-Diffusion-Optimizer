"""Benchmark metrics containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.core.types import JSONDict


@dataclass
class BenchmarkMetrics:
    total_latency_ms: float
    latency_per_step_ms: float
    peak_ram_bytes: int
    model_size_bytes: int
    cache_hit_rate: float = 0.0
    cache_memory_bytes: int = 0
    num_steps: int = 0
    batch_size: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_sec(self) -> float:
        return self.total_latency_ms / 1000.0

    def to_dict(self) -> JSONDict:
        return {
            "total_latency_ms": self.total_latency_ms,
            "total_latency_sec": self.total_latency_sec,
            "latency_per_step_ms": self.latency_per_step_ms,
            "peak_ram_bytes": self.peak_ram_bytes,
            "model_size_bytes": self.model_size_bytes,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_memory_bytes": self.cache_memory_bytes,
            "num_steps": self.num_steps,
            "batch_size": self.batch_size,
            "extra": dict(self.extra),
        }
