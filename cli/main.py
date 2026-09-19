"""CLI entrypoint for CPU Diffusion Optimizer.

Usage:
  python -m cli experiment --config configs/cache.yaml
  python -m cli profile --model toy
  python -m cli analyze --model toy
  python -m cli benchmark --model toy
"""

from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console

from cli.commands.analyze import run_analyze
from cli.commands.benchmark import run_benchmark
from cli.commands.cache import cache_app
from cli.commands.optimize import run_optimize
from cli.commands.profile import run_profile

app = typer.Typer(
    name="cdo",
    help="CPU Diffusion Optimizer — CPU-first diffusion inference optimization engine.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(cache_app, name="cache")
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    _setup_logging(verbose)


@app.command("profile")
def profile_cmd(
    model: str = typer.Option("toy", "--model", "-m", help="Model name or kind"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="YAML config path"),
    steps: int = typer.Option(8, "--steps", help="Diffusion steps"),
) -> None:
    """Profile a model on CPU and print a human-readable report."""
    run_profile(model=model, config_path=config, steps=steps, console=console)


@app.command("analyze")
def analyze_cmd(
    model: str = typer.Option("toy", "--model", "-m"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Build and display the normalized model graph."""
    run_analyze(model=model, config_path=config, console=console)


@app.command("benchmark")
def benchmark_cmd(
    model: str = typer.Option("toy", "--model", "-m"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    steps: int = typer.Option(8, "--steps"),
) -> None:
    """Run a baseline CPU benchmark."""
    run_benchmark(model=model, config_path=config, steps=steps, console=console)


@app.command("optimize")
def optimize_cmd(
    model: str = typer.Option("toy", "--model", "-m"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
) -> None:
    """Plan optimizations (Phase 1: identity plan only)."""
    run_optimize(model=model, config_path=config, console=console)


@app.command("experiment")
def experiment_cmd(
    config: str = typer.Option(
        "configs/cache.yaml",
        "--config",
        "-c",
        help="Experiment YAML config",
    ),
) -> None:
    """Run baseline vs optimized experiment and write results/."""
    from engine.core.config import load_config
    from engine.experiments.runner import ExperimentRunner

    cfg = load_config(config)
    console.print(f"[bold]Experiment:[/bold] {cfg.experiment.name}")
    console.print(f"Config: {config}")
    result = ExperimentRunner(cfg).run()
    console.print(
        f"[green]Done.[/green] decision={result.decision.value} "
        f"speedup={result.comparison.speedup:.3f}x "
        f"→ {cfg.experiment.output_dir}/"
    )


if __name__ == "__main__":
    app()
