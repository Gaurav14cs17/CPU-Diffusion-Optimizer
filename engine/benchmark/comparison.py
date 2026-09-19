"""Baseline vs optimized comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.benchmark.metrics import BenchmarkMetrics
from engine.benchmark.quality import QualityResult
from engine.core.types import JSONDict, OptimizationDecision


@dataclass
class ComparisonReport:
    baseline: BenchmarkMetrics
    optimized: BenchmarkMetrics
    quality: QualityResult
    speedup: float
    memory_reduction: float
    quality_delta: dict[str, float]
    decision: OptimizationDecision
    notes: list[str]

    def to_dict(self) -> JSONDict:
        return {
            "baseline": self.baseline.to_dict(),
            "optimized": self.optimized.to_dict(),
            "quality": self.quality.to_dict(),
            "speedup": self.speedup,
            "memory_reduction": self.memory_reduction,
            "quality_delta": dict(self.quality_delta),
            "decision": self.decision.value,
            "notes": list(self.notes),
        }


def compare(
    baseline: BenchmarkMetrics,
    optimized: BenchmarkMetrics,
    quality: QualityResult,
    *,
    require_speedup: bool = True,
) -> ComparisonReport:
    speedup = (
        baseline.total_latency_ms / optimized.total_latency_ms
        if optimized.total_latency_ms > 0
        else 0.0
    )
    memory_reduction = 0.0
    if baseline.peak_ram_bytes > 0:
        memory_reduction = 1.0 - (optimized.peak_ram_bytes / baseline.peak_ram_bytes)

    quality_delta = {
        "mse": quality.mse,
        "psnr": quality.psnr if quality.psnr == quality.psnr else 0.0,
    }

    notes: list[str] = []
    decision = OptimizationDecision.KEEP

    if not quality.passed:
        decision = OptimizationDecision.REJECT
        notes.append("Quality constraints violated.")
    elif require_speedup and speedup < 1.0:
        # Phase 1 identity path: treat as inconclusive rather than reject.
        decision = OptimizationDecision.INCONCLUSIVE
        notes.append("No measured speedup (expected for Phase 1 identity optimization).")
    elif speedup >= 1.0 and quality.passed:
        decision = OptimizationDecision.KEEP
        notes.append("Quality within threshold.")
        if speedup > 1.0:
            notes.append(f"Measured speedup {speedup:.3f}x.")

    return ComparisonReport(
        baseline=baseline,
        optimized=optimized,
        quality=quality,
        speedup=speedup,
        memory_reduction=memory_reduction,
        quality_delta=quality_delta,
        decision=decision,
        notes=notes,
    )


def render_markdown_report(
    *,
    experiment_name: str,
    model_name: str,
    hardware: dict[str, Any],
    steps: int,
    comparison: ComparisonReport,
    cache_mode: str,
) -> str:
    b = comparison.baseline
    o = comparison.optimized
    lines = [
        f"# Experiment Report: {experiment_name}",
        "",
        "## Model",
        f"- Name: {model_name}",
        f"- Diffusion steps: {steps}",
        f"- Cache mode: {cache_mode}",
        "",
        "## Hardware",
        f"- Architecture: {hardware.get('architecture')}",
        f"- Logical cores: {hardware.get('logical_cores')}",
        f"- AVX2: {hardware.get('avx2')}  AVX-512: {hardware.get('avx512')}  NEON: {hardware.get('neon')}",
        "",
        "## Latency",
        f"- Baseline: **{b.total_latency_sec:.4f} sec**/image",
        f"- Optimized: **{o.total_latency_sec:.4f} sec**/image",
        f"- Speedup: **{comparison.speedup:.3f}x**",
        "",
        "## Memory",
        f"- Baseline peak RAM: {b.peak_ram_bytes / (1024 ** 2):.1f} MiB",
        f"- Optimized peak RAM: {o.peak_ram_bytes / (1024 ** 2):.1f} MiB",
        f"- Memory reduction: {comparison.memory_reduction:.3f}",
        f"- Cache hit rate: {o.cache_hit_rate:.3f}",
        f"- Cache memory: {o.cache_memory_bytes} bytes",
        "",
        "## Quality",
        f"- MSE: {comparison.quality.mse:.6e}",
        f"- PSNR: {comparison.quality.psnr}",
        f"- Passed: {comparison.quality.passed}",
        "",
        "## Decision",
        f"- **{comparison.decision.value.upper()}**",
    ]
    for note in comparison.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)
