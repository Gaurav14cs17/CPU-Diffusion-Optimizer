"""Normalized computation graph with hierarchical traversal."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterator

from engine.core.types import JSONDict, NodeKind
from engine.model.node import GraphNode


@dataclass
class ModelGraph:
    """Directed acyclic view of a diffusion model as named nodes.

    Supports path-style traversal such as::

        Model → DiT → Block 12 → Attention → QKV Linear

    via dotted names, e.g. ``model.dit.block_12.attention.qkv``.
    """

    model_name: str
    root: str = "model"
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.name] = node

    def add_edge(self, src: str, dst: str) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError(f"Cannot add edge {src} → {dst}: missing node")
        self.edges.append((src, dst))
        if src not in self.nodes[dst].dependencies:
            self.nodes[dst].dependencies.append(src)

    def get(self, name: str) -> GraphNode:
        return self.nodes[name]

    def children(self, name: str) -> list[GraphNode]:
        prefix = f"{name}."
        direct: list[GraphNode] = []
        for node_name, node in self.nodes.items():
            if not node_name.startswith(prefix):
                continue
            remainder = node_name[len(prefix) :]
            if "." not in remainder:
                direct.append(node)
        return sorted(direct, key=lambda n: n.name)

    def find_by_kind(self, kind: NodeKind) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.kind == kind]

    def walk_bfs(self, start: str | None = None) -> Iterator[GraphNode]:
        start_name = start or self.root
        if start_name not in self.nodes:
            return
        queue: deque[str] = deque([start_name])
        seen: set[str] = set()
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            yield self.nodes[current]
            for child in self.children(current):
                queue.append(child.name)

    def path(self, dotted: str) -> list[GraphNode]:
        """Resolve a dotted path into ordered nodes that exist in the graph."""
        parts = [p for p in dotted.split(".") if p]
        resolved: list[GraphNode] = []
        cumulative: list[str] = []
        for part in parts:
            cumulative.append(part)
            name = ".".join(cumulative)
            if name in self.nodes:
                resolved.append(self.nodes[name])
        return resolved

    def total_parameters(self) -> int:
        return sum(n.parameter_count for n in self.nodes.values())

    def to_dict(self) -> JSONDict:
        return {
            "model_name": self.model_name,
            "root": self.root,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [{"src": s, "dst": d} for s, d in self.edges],
            "total_parameters": self.total_parameters(),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> ModelGraph:
        graph = cls(model_name=str(data["model_name"]), root=str(data.get("root", "model")))
        for node_data in data.get("nodes", {}).values():
            graph.add_node(GraphNode.from_dict(node_data))
        for edge in data.get("edges", []):
            graph.edges.append((edge["src"], edge["dst"]))
        return graph
