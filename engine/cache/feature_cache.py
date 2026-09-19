"""Feature cache interfaces (Phase 2 implementation).

Phase 1 provides the public API surface so later work plugs in without
rewriting callers. Spatial / tile caching hooks are reserved via RegionKey.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Hashable

import torch


@dataclass(frozen=True)
class RegionKey:
    """Identifies a cacheable spatial region (whole-block by default).

    Future: tile/patch coordinates without changing FeatureCache callers.
    """

    block_id: str
    timestep: int
    region: tuple[int, int, int, int] | None = None  # (y0, x0, y1, x1) or None = full


class FeatureCache(ABC):
    """Store and retrieve block feature tensors across diffusion timesteps."""

    @abstractmethod
    def get(self, key: RegionKey) -> torch.Tensor | None:
        ...

    @abstractmethod
    def put(self, key: RegionKey, value: torch.Tensor) -> None:
        ...

    @abstractmethod
    def invalidate(self, block_id: str | None = None) -> None:
        ...

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        ...


class InMemoryFeatureCache(FeatureCache):
    """Simple host-memory cache used as the Phase 2 default backend."""

    def __init__(self) -> None:
        self._store: dict[Hashable, torch.Tensor] = {}
        self.hits = 0
        self.misses = 0

    def _hash(self, key: RegionKey) -> Hashable:
        return (key.block_id, key.timestep, key.region)

    def get(self, key: RegionKey) -> torch.Tensor | None:
        value = self._store.get(self._hash(key))
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: RegionKey, value: torch.Tensor) -> None:
        self._store[self._hash(key)] = value.detach()

    def invalidate(self, block_id: str | None = None) -> None:
        if block_id is None:
            self._store.clear()
            return
        drop = [k for k in self._store if isinstance(k, tuple) and k[0] == block_id]
        for k in drop:
            del self._store[k]

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "entries": len(self._store),
            "approx_bytes": sum(t.numel() * t.element_size() for t in self._store.values()),
        }
