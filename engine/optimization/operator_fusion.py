"""Operator fusion hooks (CPU). Stub for Phase 4."""

from __future__ import annotations

from torch import nn


def fuse_module(model: nn.Module) -> nn.Module:
    """Placeholder — real fusion will target Linear+Activation / Conv patterns."""
    return model
