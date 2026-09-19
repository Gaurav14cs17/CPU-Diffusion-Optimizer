"""Experiment persistence tests."""

from __future__ import annotations

from pathlib import Path

from engine.core.config import (
    BenchmarkConfig,
    EngineConfig,
    ExperimentConfig,
    ModelConfig,
    ProfilerConfig,
)
from engine.experiments.database import ExperimentDatabase
from engine.experiments.runner import ExperimentRunner


def test_experiment_runner_writes_artifacts(tmp_path: Path) -> None:
    cfg = EngineConfig(
        experiment=ExperimentConfig(name="unit_exp", seed=1, output_dir=str(tmp_path)),
        model=ModelConfig(
            kind="toy_diffusion",
            num_steps=2,
            height=8,
            width=8,
            hidden_dim=16,
            num_blocks=2,
        ),
        profiler=ProfilerConfig(warmup_runs=0, measure_runs=1),
        benchmark=BenchmarkConfig(warmup_runs=0, measure_runs=1),
    )
    result = ExperimentRunner(cfg).run()
    assert (tmp_path / "baseline.json").is_file()
    assert (tmp_path / "optimized.json").is_file()
    assert (tmp_path / "comparison.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "experiments.sqlite").is_file()

    db = ExperimentDatabase(tmp_path / "experiments.sqlite")
    assert result.experiment_id in db.list_ids()
    loaded = db.get(result.experiment_id)
    assert loaded is not None
    assert loaded["name"] == "unit_exp"
