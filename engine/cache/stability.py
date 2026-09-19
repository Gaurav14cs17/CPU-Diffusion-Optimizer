"""Feature stability measurement across timesteps."""

from __future__ import annotations

import torch


def feature_change(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Normalized L2 feature change D(l,t).

    D = ||F(l,t) - F(l,t-1)|| / (||F(l,t-1)|| + eps)
    """
    if current.shape != previous.shape:
        raise ValueError(
            f"Shape mismatch for feature_change: {tuple(current.shape)} vs {tuple(previous.shape)}"
        )
    if current.dtype != previous.dtype:
        previous = previous.to(dtype=current.dtype)

    delta = torch.linalg.vector_norm((current - previous).float())
    denom = torch.linalg.vector_norm(previous.float()) + epsilon
    return float(delta / denom)
