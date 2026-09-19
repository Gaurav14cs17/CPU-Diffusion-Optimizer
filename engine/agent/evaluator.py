"""Agent evaluator — never accept without measured evidence."""

from __future__ import annotations

from engine.benchmark.comparison import ComparisonReport
from engine.core.types import OptimizationDecision


def evaluate(comparison: ComparisonReport) -> OptimizationDecision:
    return comparison.decision
