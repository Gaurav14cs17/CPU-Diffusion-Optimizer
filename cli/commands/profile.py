"""profile command."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from cli.commands._common import resolve_config
from engine.model.loader import load_model
from engine.profiler.profiler import Profiler


def run_profile(
    *,
    model: str,
    config_path: str | None,
    steps: int,
    console: Console,
) -> None:
    cfg = resolve_config(model=model, config_path=config_path, steps=steps)
    loaded = load_model(cfg.model)
    report = Profiler(cfg.profiler).profile(loaded, seed=cfg.experiment.seed)
    console.print(report.to_human_readable())
    out = Path(cfg.experiment.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (out / "profile.txt").write_text(report.to_human_readable(), encoding="utf-8")
    console.print(f"Wrote {out / 'profile.json'}")
