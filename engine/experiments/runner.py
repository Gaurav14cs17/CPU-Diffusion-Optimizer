"""Experiment runner — full vertical slice.

Pipeline:
  load model → build graph → profile → baseline benchmark →
  adaptive/static block cache → optimized benchmark → quality →
  compare → save images + persist
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from engine.benchmark.benchmark import Benchmark
from engine.benchmark.comparison import compare, render_markdown_report
from engine.benchmark.quality import QualityValidator
from engine.benchmark.visualize import save_latent_image
from engine.cache.cache_manager import CacheManager
from engine.core import jsonutil
from engine.core.config import EngineConfig
from engine.core.exceptions import ExperimentError
from engine.core.types import CacheMode, OptimizationDecision
from engine.experiments.database import ExperimentDatabase
from engine.experiments.experiment import Experiment
from engine.experiments.result import ExperimentResult
from engine.model.loader import load_model
from engine.profiler.cpu_info import inspect_cpu
from engine.profiler.profiler import Profiler

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Execute a controlled baseline vs optimized experiment."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def run(self) -> ExperimentResult:
        cfg = self.config
        output_dir = Path(cfg.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        cache_enabled = cfg.optimization.enable_cache or cfg.cache.mode != CacheMode.DISABLED
        experiment = Experiment(
            name=cfg.experiment.name,
            config=cfg,
            hypothesis=(
                f"CPU block/feature cache mode={cfg.cache.mode.value} "
                f"threshold={cfg.cache.threshold} improves latency without breaking quality."
                if cache_enabled
                else "Baseline measurement only (cache disabled)."
            ),
            tags=["toy", "cache" if cache_enabled else "phase1"],
        )

        logger.info("Loading model kind=%s name=%s", cfg.model.kind, cfg.model.name)
        loaded = load_model(cfg.model)

        logger.info("Profiling on CPU…")
        profile = Profiler(cfg.profiler).profile(loaded, seed=cfg.experiment.seed)

        logger.info("Running baseline benchmark (cache disabled)…")
        bench = Benchmark(cfg.benchmark)
        baseline_metrics, _baseline_last = bench.run(
            loaded,
            seed=cfg.experiment.seed,
            label="baseline",
        )

        detail: dict = {}

        if cache_enabled:
            if cfg.cache.mode == CacheMode.DISABLED:
                cfg.cache.mode = CacheMode.CPU_AWARE
            logger.info(
                "Running optimized benchmark (mode=%s, threshold=%.4f)…",
                cfg.cache.mode.value,
                cfg.cache.threshold,
            )

            run_counter = {"i": 0}

            def cached_fn() -> dict:
                mgr = CacheManager(config=cfg.cache)
                idx = run_counter["i"]
                run_counter["i"] += 1
                return loaded.adapter.run_inference(
                    loaded.module,
                    num_steps=cfg.model.num_steps,
                    batch_size=cfg.model.batch_size,
                    seed=cfg.experiment.seed + idx,
                    cache_manager=mgr,
                )

            detail_mgr = CacheManager(config=cfg.cache)
            detail = loaded.adapter.run_inference(
                loaded.module,
                num_steps=cfg.model.num_steps,
                batch_size=cfg.model.batch_size,
                seed=cfg.experiment.seed,
                cache_manager=detail_mgr,
            )
            (output_dir / "cache_log.json").write_text(
                jsonutil.dumps(detail.get("cache_log", []), indent=2),
                encoding="utf-8",
            )
            (output_dir / "cache_stats.json").write_text(
                jsonutil.dumps(detail.get("cache_stats", {}), indent=2),
                encoding="utf-8",
            )

            optimized_metrics, _optimized_last = bench.run(
                loaded,
                seed=cfg.experiment.seed,
                label="optimized",
                inference_fn=cached_fn,
            )
            if detail.get("cache_hit_rate") is not None:
                optimized_metrics.cache_hit_rate = float(detail["cache_hit_rate"])
                optimized_metrics.cache_memory_bytes = int(detail.get("cache_memory_bytes", 0))
                optimized_metrics.extra["cache_stats"] = detail.get("cache_stats", {})
        else:
            logger.info("Running optimized benchmark (identity / cache disabled)…")
            optimized_metrics, _optimized_last = bench.run(
                loaded,
                seed=cfg.experiment.seed,
                label="optimized",
            )

        # Paired quality check — identical seed for both paths
        logger.info("Evaluating quality (paired seed=%s)…", cfg.experiment.seed)
        baseline_out = loaded.adapter.run_inference(
            loaded.module,
            num_steps=cfg.model.num_steps,
            batch_size=cfg.model.batch_size,
            seed=cfg.experiment.seed,
        )["output"]
        if cache_enabled:
            q_mgr = CacheManager(config=cfg.cache)
            optimized_out = loaded.adapter.run_inference(
                loaded.module,
                num_steps=cfg.model.num_steps,
                batch_size=cfg.model.batch_size,
                seed=cfg.experiment.seed,
                cache_manager=q_mgr,
            )["output"]
        else:
            optimized_out = loaded.adapter.run_inference(
                loaded.module,
                num_steps=cfg.model.num_steps,
                batch_size=cfg.model.batch_size,
                seed=cfg.experiment.seed,
            )["output"]

        quality = QualityValidator(cfg.benchmark.quality).evaluate(baseline_out, optimized_out)
        comparison = compare(baseline_metrics, optimized_metrics, quality)

        if not cache_enabled and quality.passed:
            comparison.decision = OptimizationDecision.INCONCLUSIVE
            comparison.notes.append(
                "Cache disabled — identity path. Set cache.mode / enable_cache to optimize."
            )
        elif cache_enabled and quality.passed and comparison.speedup >= 1.0:
            comparison.decision = OptimizationDecision.KEEP
            comparison.notes.append(
                f"Cache mode={cfg.cache.mode.value} hit_rate={optimized_metrics.cache_hit_rate:.3f}"
            )
        elif cache_enabled and quality.passed and comparison.speedup < 1.0:
            comparison.decision = OptimizationDecision.INCONCLUSIVE
            comparison.notes.append(
                "Cache ran but no net speedup on this hardware/model size "
                f"(hit_rate={optimized_metrics.cache_hit_rate:.3f})."
            )

        # Save latent visualizations
        logger.info("Saving latent images → %s", images_dir)
        save_latent_image(baseline_out, images_dir / "baseline.png")
        save_latent_image(optimized_out, images_dir / "optimized.png")

        experiment_id = str(uuid.uuid4())
        result = ExperimentResult(
            experiment_id=experiment_id,
            name=experiment.name,
            seed=cfg.experiment.seed,
            config=experiment.to_dict(),
            baseline=baseline_metrics,
            optimized=optimized_metrics,
            comparison=comparison,
            profile=profile,
            decision=comparison.decision,
        )

        self._write_artifacts(output_dir, result, profile.to_human_readable())
        ExperimentDatabase(output_dir / "experiments.sqlite").save(result)

        logger.info(
            "Experiment complete: decision=%s speedup=%.3fx hit_rate=%.3f → %s",
            result.decision.value,
            comparison.speedup,
            optimized_metrics.cache_hit_rate,
            output_dir,
        )
        return result

    def _write_artifacts(
        self,
        output_dir: Path,
        result: ExperimentResult,
        profile_text: str,
    ) -> None:
        try:
            (output_dir / "baseline.json").write_text(
                jsonutil.dumps(result.baseline.to_dict(), indent=2),
                encoding="utf-8",
            )
            (output_dir / "optimized.json").write_text(
                jsonutil.dumps(result.optimized.to_dict(), indent=2),
                encoding="utf-8",
            )
            (output_dir / "comparison.json").write_text(
                jsonutil.dumps(result.comparison.to_dict(), indent=2),
                encoding="utf-8",
            )
            (output_dir / "profile.json").write_text(
                jsonutil.dumps(result.profile.to_dict() if result.profile else {}, indent=2),
                encoding="utf-8",
            )
            (output_dir / "profile.txt").write_text(profile_text, encoding="utf-8")

            hardware = inspect_cpu().to_dict()
            report_md = render_markdown_report(
                experiment_name=result.name,
                model_name=self.config.model.name,
                hardware=hardware,
                steps=self.config.model.num_steps,
                comparison=result.comparison,
                cache_mode=self.config.cache.mode.value,
            )
            report_md += (
                "\n## Images\n"
                "- `images/baseline.png` — baseline latent visualization\n"
                "- `images/optimized.png` — optimized latent visualization\n"
            )
            (output_dir / "report.md").write_text(report_md, encoding="utf-8")
            (output_dir / "experiment.json").write_text(
                jsonutil.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ExperimentError(f"Failed to write results: {exc}") from exc
