"""analyze command."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.tree import Tree

from cli.commands._common import resolve_config
from engine.model.loader import load_model


def run_analyze(*, model: str, config_path: str | None, console: Console) -> None:
    cfg = resolve_config(model=model, config_path=config_path)
    loaded = load_model(cfg.model)
    graph = loaded.graph

    tree = Tree(f"[bold]{graph.model_name}[/bold] ({loaded.info.parameter_count} params)")
    for node in graph.walk_bfs():
        if node.name == graph.root:
            continue
        depth = node.name.count(".")
        label = f"{node.name} [{node.kind.value}] params={node.parameter_count}"
        # Simple flat listing with indent via tree children of root for depth-1
        if depth == 1:
            tree.add(label)
        else:
            tree.add(("  " * (depth - 1)) + label)

    console.print(tree)
    console.print("\nPath example: model.dit.block_0.attention.qkv")
    path_nodes = graph.path("model.dit.block_0.attention.qkv")
    console.print(" → ".join(n.name for n in path_nodes))

    out = Path(cfg.experiment.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")
    console.print(f"Wrote {out / 'graph.json'}")
