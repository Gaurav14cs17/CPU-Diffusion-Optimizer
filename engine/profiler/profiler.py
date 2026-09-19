"""CPU profiler for diffusion inference."""

from __future__ import annotations

import time
from typing import Any

from engine.core.config import ProfilerConfig
from engine.core.exceptions import ProfileError
from engine.model.loader import LoadedModel
from engine.profiler.cpu_info import inspect_cpu
from engine.profiler.memory_profiler import MemoryProfiler
from engine.profiler.operator_profiler import OperatorProfiler, aggregate_hotspots
from engine.profiler.report import Hotspot, ProfileReport


class Profiler:
    """Profile a loaded diffusion model on CPU."""

    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self.config = config or ProfilerConfig()

    def profile(self, loaded: LoadedModel, *, seed: int = 42) -> ProfileReport:
        cfg = loaded.config
        adapter = loaded.adapter
        model = loaded.module

        mem = MemoryProfiler()
        mem.reset_peak()
        op_profiler = OperatorProfiler()

        try:
            # Warmup
            for _ in range(self.config.warmup_runs):
                adapter.run_inference(
                    model,
                    num_steps=cfg.num_steps,
                    batch_size=cfg.batch_size,
                    seed=seed,
                )

            totals: list[float] = []
            per_step_accum: list[list[float]] = []
            block_accum: dict[str, float] = {}
            operator_ms: dict[str, float] = {}

            for run_idx in range(self.config.measure_runs):
                mem.sample()
                with op_profiler.profile(model):
                    t0 = time.perf_counter()
                    result = adapter.run_inference(
                        model,
                        num_steps=cfg.num_steps,
                        batch_size=cfg.batch_size,
                        seed=seed + run_idx,
                    )
                    totals.append((time.perf_counter() - t0) * 1000.0)
                mem.sample()

                per_step = list(result.get("per_step_ms", []))
                per_step_accum.append(per_step)
                for name, ms in result.get("block_timings_ms", {}).items():
                    block_accum[name] = block_accum.get(name, 0.0) + float(ms)

                for name, timing in op_profiler.timings.items():
                    operator_ms[name] = operator_ms.get(name, 0.0) + timing.total_ms

            if not totals:
                raise ProfileError("No profiling measurements collected")

            n = float(self.config.measure_runs)
            mean_total = sum(totals) / n
            mean_blocks = {k: v / n for k, v in block_accum.items()}
            mean_ops = {k: v / n for k, v in operator_ms.items()}

            # Average per-step across runs (pad-safe)
            max_len = max((len(s) for s in per_step_accum), default=0)
            mean_steps: list[float] = []
            for i in range(max_len):
                vals = [s[i] for s in per_step_accum if i < len(s)]
                mean_steps.append(sum(vals) / len(vals) if vals else 0.0)

            from engine.profiler.operator_profiler import OperatorTiming

            op_timings = {
                k: OperatorTiming(name=k, calls=1, total_ms=v) for k, v in mean_ops.items()
            }
            hotspot_source: dict[str, float] = {
                k.replace("block_", "DiT Block "): v for k, v in mean_blocks.items()
            }
            for name, ms in aggregate_hotspots(op_timings).items():
                if name.startswith("Block "):
                    continue
                hotspot_source[name] = ms

            total_for_share = mean_total if mean_total > 0 else 1.0
            hotspots = [
                Hotspot(name=name, latency_ms=ms, share_pct=100.0 * ms / total_for_share)
                for name, ms in sorted(hotspot_source.items(), key=lambda x: x[1], reverse=True)
            ]

            snap = mem.sample()
            resolution = f"{cfg.height}x{cfg.width}"
            return ProfileReport(
                model_name=cfg.name,
                resolution=resolution,
                steps=cfg.num_steps,
                device="cpu",
                total_latency_ms=mean_total,
                mean_step_latency_ms=(sum(mean_steps) / len(mean_steps)) if mean_steps else 0.0,
                per_step_latency_ms=mean_steps,
                block_latency_ms=mean_blocks,
                operator_latency_ms=mean_ops,
                peak_ram_bytes=snap.peak_rss_bytes if self.config.track_memory else 0,
                batch_size=cfg.batch_size,
                dtype="fp32",
                cpu=inspect_cpu(),
                hotspots=hotspots,
                metadata={
                    "warmup_runs": self.config.warmup_runs,
                    "measure_runs": self.config.measure_runs,
                    "seed": seed,
                    "parameter_count": loaded.info.parameter_count,
                },
            )
        except ProfileError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ProfileError(f"Profiling failed: {exc}") from exc
