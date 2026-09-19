"""High-level model metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.core.types import DeviceKind, DTypeName, JSONDict
from engine.model.graph import ModelGraph


@dataclass
class ModelInfo:
    """Summary information about a loaded diffusion model."""

    name: str
    kind: str
    device: DeviceKind = DeviceKind.CPU
    dtype: DTypeName = DTypeName.FP32
    parameter_count: int = 0
    num_blocks: int = 0
    latent_shape: tuple[int, ...] = ()
    components: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    graph: ModelGraph | None = None

    def to_dict(self) -> JSONDict:
        return {
            "name": self.name,
            "kind": self.kind,
            "device": self.device.value,
            "dtype": self.dtype.value,
            "parameter_count": self.parameter_count,
            "num_blocks": self.num_blocks,
            "latent_shape": list(self.latent_shape),
            "components": list(self.components),
            "metadata": dict(self.metadata),
            "graph": self.graph.to_dict() if self.graph is not None else None,
        }
