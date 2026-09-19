"""Benchmark and quality tests."""

from __future__ import annotations

import torch

from engine.benchmark.benchmark import Benchmark
from engine.benchmark.comparison import compare
from engine.benchmark.quality import QualityValidator, pixel_mse
from engine.core.config import BenchmarkConfig, ModelConfig, QualityConfig
from engine.core.types import OptimizationDecision
from engine.model.loader import load_model


def test_benchmark_runs() -> None:
    loaded = load_model(
        ModelConfig(kind="toy_diffusion", num_steps=2, height=8, width=8, hidden_dim=16, num_blocks=2)
    )
    metrics, output = Benchmark(BenchmarkConfig(warmup_runs=0, measure_runs=1)).run(
        loaded, seed=1, label="test"
    )
    assert metrics.total_latency_ms > 0
    assert output.ndim == 4


def test_quality_identical_passes() -> None:
    t = torch.randn(1, 4, 8, 8)
    result = QualityValidator(QualityConfig(max_mse=1e-6, min_psnr=20.0)).evaluate(t, t.clone())
    assert result.passed
    assert result.mse == 0.0


def test_quality_large_delta_fails() -> None:
    a = torch.zeros(1, 1, 4, 4)
    b = torch.ones(1, 1, 4, 4)
    assert pixel_mse(a, b) == 1.0
    result = QualityValidator(QualityConfig(metrics=["mse"], max_mse=1e-3)).evaluate(a, b)
    assert result.passed is False


def test_comparison_identity_inconclusive_or_keep() -> None:
    loaded = load_model(
        ModelConfig(kind="toy_diffusion", num_steps=2, height=8, width=8, hidden_dim=16, num_blocks=2)
    )
    bench = Benchmark(BenchmarkConfig(warmup_runs=0, measure_runs=1))
    base_m, base_o = bench.run(loaded, seed=0, label="b")
    opt_m, opt_o = bench.run(loaded, seed=0, label="o")
    quality = QualityValidator().evaluate(base_o, opt_o)
    report = compare(base_m, opt_m, quality)
    assert report.decision in {
        OptimizationDecision.KEEP,
        OptimizationDecision.INCONCLUSIVE,
        OptimizationDecision.REJECT,
    }
    assert report.speedup > 0
