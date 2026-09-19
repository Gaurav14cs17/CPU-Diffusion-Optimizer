"""Cache policy interfaces (Phase 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.core.types import CacheMode


@dataclass
class ReuseDecision:
    reuse: bool
    reason: str
    feature_change: float
    t_reuse: float
    t_compute: float


class CachePolicy(ABC):
    @abstractmethod
    def should_reuse(
        self,
        *,
        block_id: str,
        timestep: int,
        feature_change: float,
        t_reuse: float,
        t_compute: float,
    ) -> ReuseDecision:
        ...


class ThresholdPolicy(CachePolicy):
    """Reuse when feature change is below a fixed threshold (no cost model)."""

    def __init__(self, threshold: float = 0.015) -> None:
        self.threshold = threshold

    def should_reuse(
        self,
        *,
        block_id: str,
        timestep: int,
        feature_change: float,
        t_reuse: float,
        t_compute: float,
    ) -> ReuseDecision:
        reuse = feature_change < self.threshold
        return ReuseDecision(
            reuse=reuse,
            reason="below_threshold" if reuse else "above_threshold",
            feature_change=feature_change,
            t_reuse=t_reuse,
            t_compute=t_compute,
        )


class CpuAwarePolicy(CachePolicy):
    """Reuse only when T_reuse < T_compute AND feature is stable."""

    def __init__(self, threshold: float = 0.015) -> None:
        self.threshold = threshold

    def should_reuse(
        self,
        *,
        block_id: str,
        timestep: int,
        feature_change: float,
        t_reuse: float,
        t_compute: float,
    ) -> ReuseDecision:
        stable = feature_change < self.threshold
        cheaper = t_reuse < t_compute
        reuse = stable and cheaper
        if reuse:
            reason = "stable_and_cheaper"
        elif not stable:
            reason = "unstable"
        else:
            reason = "reuse_not_cheaper"
        return ReuseDecision(
            reuse=reuse,
            reason=reason,
            feature_change=feature_change,
            t_reuse=t_reuse,
            t_compute=t_compute,
        )


class AlwaysReuseWhenEligible(CachePolicy):
    """Used by static/timestep schedules — eligibility is decided by the runtime."""

    def should_reuse(
        self,
        *,
        block_id: str,
        timestep: int,
        feature_change: float,
        t_reuse: float,
        t_compute: float,
    ) -> ReuseDecision:
        cheaper = t_reuse < t_compute
        reuse = cheaper
        return ReuseDecision(
            reuse=reuse,
            reason="schedule_reuse" if reuse else "reuse_not_cheaper",
            feature_change=feature_change,
            t_reuse=t_reuse,
            t_compute=t_compute,
        )


def default_policy(mode: CacheMode, threshold: float = 0.015) -> CachePolicy:
    if mode == CacheMode.CPU_AWARE:
        return CpuAwarePolicy(threshold=threshold)
    if mode in {CacheMode.STATIC_BLOCK, CacheMode.TIMESTEP}:
        return AlwaysReuseWhenEligible()
    if mode == CacheMode.ADAPTIVE:
        return ThresholdPolicy(threshold=threshold)
    return ThresholdPolicy(threshold=threshold)
