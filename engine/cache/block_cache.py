"""Block-level cache facade (wraps FeatureCache)."""

from __future__ import annotations

from typing import Any

import torch

from engine.cache.feature_cache import FeatureCache, InMemoryFeatureCache, RegionKey


class BlockCache:
    """Whole-block cache. Spatial tiling can specialize RegionKey later."""

    def __init__(self, backend: FeatureCache | None = None) -> None:
        self.backend = backend or InMemoryFeatureCache()

    def get_block(self, block_id: str, timestep: int) -> torch.Tensor | None:
        return self.backend.get(RegionKey(block_id=block_id, timestep=timestep))

    def put_block(self, block_id: str, timestep: int, value: torch.Tensor) -> None:
        self.backend.put(RegionKey(block_id=block_id, timestep=timestep), value)

    def invalidate(self, block_id: str | None = None) -> None:
        self.backend.invalidate(block_id)

    def stats(self) -> dict[str, Any]:
        return self.backend.stats()
