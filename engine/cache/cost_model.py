"""CPU-aware reuse cost model (Phase 3 fleshes out calibration)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CostEstimate:
    t_reuse_sec: float
    t_compute_sec: float
    tensor_bytes: int
    notes: str = ""


class CostModel(ABC):
    @abstractmethod
    def estimate_compute_time(self, *, block_latency_sec: float) -> float:
        ...

    @abstractmethod
    def estimate_reuse_time(
        self,
        *,
        tensor_bytes: int,
        memory_bandwidth_bytes_per_sec: float,
        decompression_sec: float = 0.0,
        cache_lookup_sec: float = 1e-6,
    ) -> float:
        ...

    def estimate(
        self,
        *,
        block_latency_sec: float,
        tensor_bytes: int,
        memory_bandwidth_bytes_per_sec: float,
    ) -> CostEstimate:
        t_compute = self.estimate_compute_time(block_latency_sec=block_latency_sec)
        t_reuse = self.estimate_reuse_time(
            tensor_bytes=tensor_bytes,
            memory_bandwidth_bytes_per_sec=memory_bandwidth_bytes_per_sec,
        )
        return CostEstimate(
            t_reuse_sec=t_reuse,
            t_compute_sec=t_compute,
            tensor_bytes=tensor_bytes,
        )


class BandwidthCostModel(CostModel):
    """T_reuse ≈ bytes/bandwidth + lookup; T_compute ≈ measured block time."""

    def estimate_compute_time(self, *, block_latency_sec: float) -> float:
        return max(block_latency_sec, 0.0)

    def estimate_reuse_time(
        self,
        *,
        tensor_bytes: int,
        memory_bandwidth_bytes_per_sec: float,
        decompression_sec: float = 0.0,
        cache_lookup_sec: float = 1e-6,
    ) -> float:
        bw = max(memory_bandwidth_bytes_per_sec, 1.0)
        return (tensor_bytes / bw) + decompression_sec + cache_lookup_sec
