"""cache subcommands."""

from __future__ import annotations

import typer
from rich.console import Console

from cli.commands._common import resolve_config
from engine.cache.stability import feature_change
from engine.core.types import CacheMode
from engine.model.loader import load_model

cache_app = typer.Typer(help="Cache analysis commands (Phase 2 expands this).")
console = Console()


@cache_app.command("analyze")
def cache_analyze(
    model: str = typer.Option("toy", "--model", "-m"),
    config: str | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Summarize cache configuration and demonstrate stability math on a toy run."""
    cfg = resolve_config(model=model, config_path=config)
    loaded = load_model(cfg.model)
    console.print(f"Cache mode: {cfg.cache.mode.value}")
    console.print(f"Threshold: {cfg.cache.threshold}")
    console.print(f"Blocks: {loaded.info.num_blocks}")

    if cfg.cache.mode == CacheMode.DISABLED:
        console.print(
            "[yellow]Cache disabled (Phase 1 default). "
            "Set cache.mode to cpu_aware in configs for Phase 2.[/yellow]"
        )

    # Demonstrate stability helper with two identical tensors (change ≈ 0)
    import torch

    a = torch.randn(2, 4, 4)
    b = a.clone()
    d = feature_change(a, b, epsilon=cfg.cache.epsilon)
    console.print(f"Sanity feature_change(identical) = {d:.6e}")
