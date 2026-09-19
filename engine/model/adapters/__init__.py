"""Model adapters."""

from engine.model.adapters.base import ModelAdapter
from engine.model.adapters.toy import ToyDiffusionAdapter

__all__ = ["ModelAdapter", "ToyDiffusionAdapter"]
