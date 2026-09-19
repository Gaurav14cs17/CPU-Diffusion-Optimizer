"""Benchmark runner."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch
from torch import nn

from engine.benchmark.metrics import BenchmarkMetrics
from engine.core.config import BenchmarkConfig
from engine.core.exceptions import BenchmarkError
from engine.model.loader import LoadedModel
from engine.profiler.memory_profiler import MemoryProfiler


InferenceFn = Callable[[], dict[str, Any]]


def _model_size_bytes(model: nn.Module) -> int:
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buf in model.buffers():
        total += buf.numel() * buf.element_size()
    return total


class Benchmark:
    """Measure latency / memory for a diffusion inference callable."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()

    def run(
        self,
        loaded: LoadedModel,
        *,
        seed: int = 42,
        label: str = "run",
        inference_fn: InferenceFn | None = None,
        cache_hit_rate: float = 0.0,
        cache_memory_bytes: int = 0,
    ) -> tuple[BenchmarkMetrics, torch.Tensor]:
        cfg = loaded.config
        adapter = loaded.adapter
        model = loaded.module
        mem = MemoryProfiler()
        mem.reset_peak()

        def default_fn() -> dict[str, Any]:
            return adapter.run_inference(
                model,
                num_steps=cfg.num_steps,
                batch_size=cfg.batch_size,
                seed=seed,
            )

        fn = inference_fn or default_fn

        try:
            for _ in range(self.config.warmup_runs):
                fn()

            totals: list[float] = []
            step_means: list[float] = []
            hit_rates: list[float] = []
            cache_memories: list[int] = []
            last_output: torch.Tensor | None = None
            last_extra: dict[str, Any] = {}

            for run_idx in range(self.config.measure_runs):
                mem.sample()
                t0 = time.perf_counter()
                # Re-bind seed-aware default if using default path
                if inference_fn is None:

                    def _fn(idx: int = run_idx) -> dict[str, Any]:
                        return adapter.run_inference(
                            model,
                            num_steps=cfg.num_steps,
                            batch_size=cfg.batch_size,
                            seed=seed + idx,
                        )

                    result = _fn()
                else:
                    result = fn()
                totals.append((time.perf_counter() - t0) * 1000.0)
                mem.sample()
                per_step = result.get("per_step_ms") or []
                if per_step:
                    step_means.append(sum(per_step) / len(per_step))
                hit_rates.append(float(result.get("cache_hit_rate", cache_hit_rate) or 0.0))
                cache_memories.append(int(result.get("cache_memory_bytes", cache_memory_bytes) or 0))
                last_extra = {
                    k: result[k]
                    for k in ("cache_stats", "cache_log")
                    if k in result
                }
                out = result["output"]
                assert isinstance(out, torch.Tensor)
                last_output = out.detach().cpu()

            if last_output is None or not totals:
                raise BenchmarkError(f"Benchmark '{label}' produced no output")

            mean_total = sum(totals) / len(totals)
            mean_step = sum(step_means) / len(step_means) if step_means else mean_total / max(cfg.num_steps, 1)
            snap = mem.sample()
            mean_hit = sum(hit_rates) / len(hit_rates) if hit_rates else cache_hit_rate
            mean_cache_mem = int(sum(cache_memories) / len(cache_memories)) if cache_memories else cache_memory_bytes

            metrics = BenchmarkMetrics(
                total_latency_ms=mean_total,
                latency_per_step_ms=mean_step,
                peak_ram_bytes=snap.peak_rss_bytes,
                model_size_bytes=_model_size_bytes(model),
                cache_hit_rate=mean_hit,
                cache_memory_bytes=mean_cache_mem,
                num_steps=cfg.num_steps,
                batch_size=cfg.batch_size,
                extra={
                    "label": label,
                    "measure_runs": self.config.measure_runs,
                    **last_extra,
                },
            )
            return metrics, last_output
        except BenchmarkError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BenchmarkError(f"Benchmark '{label}' failed: {exc}") from exc
