"""optimize command."""

from __future__ import annotations

from rich.console import Console

from cli.commands._common import resolve_config
from engine.model.loader import load_model
from engine.optimization.optimizer import NullOptimizer


def run_optimize(*, model: str, config_path: str | None, console: Console) -> None:
    cfg = resolve_config(model=model, config_path=config_path)
    loaded = load_model(cfg.model)
    plan = NullOptimizer().plan(loaded.graph, cfg)
    console.print("Optimization plan (Phase 1):")
    for action in plan.actions:
        console.print(f"  • {action}")
    console.print(
        "\nNo transforms applied yet. Run `python -m cli experiment` "
        "to measure baseline vs identity optimized path."
    )
