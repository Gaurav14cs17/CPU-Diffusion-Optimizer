"""Block skipping with mandatory evidence (never blind skip)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlockSkipRecord:
    block_id: str
    reason: str
    threshold: float
    expected_savings_sec: float
    observed_savings_sec: float | None = None
    quality_impact: float | None = None


class BlockSkipPlanner:
    """Identify low-contribution blocks — decisions must be benchmarked."""

    def propose(self, *args, **kwargs) -> list[BlockSkipRecord]:  # noqa: ANN002
        return []
