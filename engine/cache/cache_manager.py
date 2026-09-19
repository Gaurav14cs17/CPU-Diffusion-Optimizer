"""Coordinates cache backends, policies, and logging (Phase 2 wiring)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from engine.cache.block_cache import BlockCache
from engine.cache.cache_policy import CachePolicy, ReuseDecision, default_policy
from engine.cache.cost_model import BandwidthCostModel, CostModel
from engine.core.config import CacheConfig
from engine.core.types import CacheMode

logger = logging.getLogger(__name__)


@dataclass
class CacheLogEntry:
    block_id: str
    timestep: int
    cache_hit: bool
    cache_miss: bool
    feature_change: float
    feature_size: int
    compute_time: float
    reuse_time: float
    decision: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "timestep": self.timestep,
            "cache_hit": self.cache_hit,
            "cache_miss": self.cache_miss,
            "feature_change": self.feature_change,
            "feature_size": self.feature_size,
            "compute_time": self.compute_time,
            "reuse_time": self.reuse_time,
            "decision": self.decision,
        }


@dataclass
class CacheManager:
    """Orchestrates block feature caching decisions."""

    config: CacheConfig
    block_cache: BlockCache = field(default_factory=BlockCache)
    policy: CachePolicy | None = None
    cost_model: CostModel = field(default_factory=BandwidthCostModel)
    log: list[CacheLogEntry] = field(default_factory=list)
    # Rough default DRAM bandwidth guess for CPU (bytes/s); calibrate later.
    memory_bandwidth_bytes_per_sec: float = 20e9

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = default_policy(self.config.mode, self.config.threshold)

    @property
    def enabled(self) -> bool:
        return self.config.mode != CacheMode.DISABLED

    def record(self, entry: CacheLogEntry) -> None:
        self.log.append(entry)
        logger.debug("cache %s", entry.to_dict())

    def decide(
        self,
        *,
        block_id: str,
        timestep: int,
        feature_change: float,
        feature_bytes: int,
        block_latency_sec: float,
    ) -> ReuseDecision:
        assert self.policy is not None
        estimate = self.cost_model.estimate(
            block_latency_sec=block_latency_sec,
            tensor_bytes=feature_bytes,
            memory_bandwidth_bytes_per_sec=self.memory_bandwidth_bytes_per_sec,
        )
        decision = self.policy.should_reuse(
            block_id=block_id,
            timestep=timestep,
            feature_change=feature_change,
            t_reuse=estimate.t_reuse_sec,
            t_compute=estimate.t_compute_sec,
        )
        self.record(
            CacheLogEntry(
                block_id=block_id,
                timestep=timestep,
                cache_hit=decision.reuse,
                cache_miss=not decision.reuse,
                feature_change=feature_change,
                feature_size=feature_bytes,
                compute_time=estimate.t_compute_sec,
                reuse_time=estimate.t_reuse_sec,
                decision=decision.reason,
            )
        )
        return decision
