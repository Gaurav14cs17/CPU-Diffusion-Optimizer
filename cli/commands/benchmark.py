"""benchmark command."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from cli.commands._common import resolve_config
from engine.benchmark.benchmark import Benchmark
from engine.model.loader import load_model


def run_benchmark(
    *,
    model: str,
    config_path: str | None,
    steps: int,
    console: Console,
) -> None:
    cfg = resolve_config(model=model, config_path=config_path, steps=steps)
    loaded = load_model(cfg.model)
    metrics, _ = Benchmark(cfg.benchmark).run(loaded, seed=cfg.experiment.seed, label="baseline")
    console.print(
        f"Baseline: {metrics.total_latency_sec:.4f} sec/image "
        f"({metrics.latency_per_step_ms:.2f} ms/step) "
        f"peak_ram={metrics.peak_ram_bytes / (1024 ** 2):.1f} MiB"
    )
    out = Path(cfg.experiment.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    console.print(f"Wrote {out / 'baseline.json'}")
