"""Experiment definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.core.config import EngineConfig


@dataclass
class Experiment:
    """A single optimization experiment specification."""

    name: str
    config: EngineConfig
    hypothesis: str = ""
    tags: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "tags": list(self.tags or []),
            "config": self.config.model_dump(mode="json"),
        }
