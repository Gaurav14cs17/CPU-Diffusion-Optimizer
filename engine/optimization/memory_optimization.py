"""Memory optimization helpers (reduce copies, promote channels-last, etc.)."""

from __future__ import annotations

from torch import nn


def optimize_memory_layout(model: nn.Module) -> nn.Module:
    """Phase 4: contiguous params, avoid repeated host copies."""
    return model
