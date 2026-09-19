"""Per-operator timing hooks (Phase 1: module-level)."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from torch import nn


@dataclass
class OperatorTiming:
    name: str
    calls: int = 0
    total_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "total_ms": self.total_ms,
            "mean_ms": self.mean_ms,
        }


@dataclass
class OperatorProfiler:
    """Accumulate forward-hook timings for named modules."""

    timings: dict[str, OperatorTiming] = field(default_factory=dict)
    _starts: dict[int, float] = field(default_factory=dict)
    _handles: list[Any] = field(default_factory=list)

    def attach(self, model: nn.Module, prefix: str = "") -> None:
        for name, module in model.named_modules():
            if name == "":
                continue
            full = f"{prefix}{name}" if prefix else name

            def pre_hook(_mod: nn.Module, _inp: tuple[Any, ...], *, _id: int = id(module)) -> None:
                self._starts[_id] = time.perf_counter()

            def post_hook(
                _mod: nn.Module,
                _inp: tuple[Any, ...],
                _out: Any,
                *,
                _id: int = id(module),
                _name: str = full,
            ) -> None:
                start = self._starts.pop(_id, None)
                if start is None:
                    return
                elapsed = (time.perf_counter() - start) * 1000.0
                timing = self.timings.setdefault(_name, OperatorTiming(name=_name))
                timing.calls += 1
                timing.total_ms += elapsed

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._starts.clear()

    def top(self, n: int = 10) -> list[OperatorTiming]:
        return sorted(self.timings.values(), key=lambda t: t.total_ms, reverse=True)[:n]

    def reset(self) -> None:
        self.timings.clear()
        self._starts.clear()

    @contextmanager
    def profile(self, model: nn.Module) -> Iterator["OperatorProfiler"]:
        self.reset()
        self.attach(model)
        try:
            yield self
        finally:
            self.detach()


def merge_by_suffix(timings: dict[str, OperatorTiming], suffix: str) -> float:
    total = 0.0
    for name, timing in timings.items():
        if name.endswith(suffix) or f".{suffix}" in name or name.split(".")[-1] == suffix:
            total += timing.total_ms
    return total


def aggregate_hotspots(timings: dict[str, OperatorTiming]) -> dict[str, float]:
    """Roll operator timings into coarse hotspot buckets."""
    buckets: dict[str, float] = defaultdict(float)
    for name, timing in timings.items():
        leaf = name.split(".")[-1]
        if "attn" in name or leaf == "attn":
            buckets["Attention"] += timing.total_ms
        elif leaf == "mlp" or ".mlp." in name:
            buckets["MLP"] += timing.total_ms
        elif "block_" in name and name.count(".") <= 3:
            buckets[name.split(".")[-1].replace("block_", "Block ")] += timing.total_ms
        elif "time_embed" in name or "timestep" in name:
            buckets["Timestep Embedding"] += timing.total_ms
        elif leaf in {"input_proj", "output_proj"}:
            buckets["Convolution"] += timing.total_ms
    return dict(buckets)
