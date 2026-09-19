"""Graph node representation for diffusion models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.core.types import JSONDict, NodeKind, Shape


@dataclass
class GraphNode:
    """A single node in the normalized computation graph."""

    name: str
    kind: NodeKind
    input_shapes: list[Shape] = field(default_factory=list)
    output_shapes: list[Shape] = field(default_factory=list)
    dtype: str = "fp32"
    parameter_count: int = 0
    estimated_flops: float = 0.0
    measured_latency_ms: float | None = None
    memory_bytes: int | None = None
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def path_segments(self) -> list[str]:
        return [segment for segment in self.name.split(".") if segment]

    def to_dict(self) -> JSONDict:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "input_shapes": [list(s) for s in self.input_shapes],
            "output_shapes": [list(s) for s in self.output_shapes],
            "dtype": self.dtype,
            "parameter_count": self.parameter_count,
            "estimated_flops": self.estimated_flops,
            "measured_latency_ms": self.measured_latency_ms,
            "memory_bytes": self.memory_bytes,
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> GraphNode:
        return cls(
            name=str(data["name"]),
            kind=NodeKind(data["kind"]),
            input_shapes=[tuple(s) for s in data.get("input_shapes", [])],
            output_shapes=[tuple(s) for s in data.get("output_shapes", [])],
            dtype=str(data.get("dtype", "fp32")),
            parameter_count=int(data.get("parameter_count", 0)),
            estimated_flops=float(data.get("estimated_flops", 0.0)),
            measured_latency_ms=data.get("measured_latency_ms"),
            memory_bytes=data.get("memory_bytes"),
            dependencies=list(data.get("dependencies", [])),
            metadata=dict(data.get("metadata", {})),
        )
