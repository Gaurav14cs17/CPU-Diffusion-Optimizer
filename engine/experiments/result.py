"""Experiment result records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from engine.benchmark.comparison import ComparisonReport
from engine.benchmark.metrics import BenchmarkMetrics
from engine.core.types import JSONDict, OptimizationDecision
from engine.profiler.report import ProfileReport


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    seed: int
    config: dict[str, Any]
    baseline: BenchmarkMetrics
    optimized: BenchmarkMetrics
    comparison: ComparisonReport
    profile: ProfileReport | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    decision: OptimizationDecision = OptimizationDecision.INCONCLUSIVE

    def to_dict(self) -> JSONDict:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "seed": self.seed,
            "created_at": self.created_at,
            "config": dict(self.config),
            "baseline": self.baseline.to_dict(),
            "optimized": self.optimized.to_dict(),
            "comparison": self.comparison.to_dict(),
            "profile": self.profile.to_dict() if self.profile else None,
            "decision": self.decision.value,
        }
