"""Callable agent tools that wrap the engine (no UI dependencies)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from engine.core import jsonutil
from engine.core.config import EngineConfig, ModelConfig, load_config
from engine.core.types import CacheMode
from engine.model.loader import load_model
from engine.profiler.cpu_info import inspect_cpu
from engine.profiler.profiler import Profiler

TOOL_NAMES = [
    "help",
    "read_results",
    "compare_results",
    "inspect_cpu",
    "build_model_graph",
    "profile_model",
    "analyze_cache",
    "run_experiment",
    "run_benchmark",
    "list_images",
    "list_txt2img_models",
    "set_txt2img_model",
    "generate_image",
]

ToolFn = Callable[..., dict[str, Any]]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_dir() -> Path:
    return _project_root() / "results"


def _safe_read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    text = (
        text.replace(": Infinity", ": null")
        .replace(": -Infinity", ": null")
        .replace(": NaN", ": null")
    )
    return json.loads(text)


def tool_help() -> dict[str, Any]:
    return {
        "ok": True,
        "message": (
            "I am the CPU Diffusion Optimizer agent. Try:\n"
            "• help — this list\n"
            "• profile — profile the toy model on CPU\n"
            "• experiment / optimize — run baseline vs cache experiment\n"
            "• compare / results — show last measured results\n"
            "• cache — cache hit rate\n"
            "• graph — model graph\n"
            "• hardware — CPU features\n"
            "• images — list saved images\n"
            "• list models — CPU txt2img catalog (sd-turbo, sdxs, bk-sdm-tiny, …)\n"
            "• use model sd-turbo — select active image model\n"
            "• generate a cat image — offline CPU Diffusers\n"
            "• generate a dog with sdxs — one-shot model override"
        ),
        "tools": TOOL_NAMES,
    }


def tool_read_results() -> dict[str, Any]:
    root = _results_dir()
    comparison = _safe_read_json(root / "comparison.json")
    report = (root / "report.md").read_text(encoding="utf-8") if (root / "report.md").is_file() else None
    if comparison is None and report is None:
        return {
            "ok": False,
            "message": "No results yet. Ask me to 'run experiment' first.",
        }
    return {
        "ok": True,
        "comparison": comparison,
        "report": report,
        "message": "Loaded results/ artifacts.",
    }


def tool_compare_results() -> dict[str, Any]:
    data = tool_read_results()
    if not data.get("ok") or not data.get("comparison"):
        return data
    c = data["comparison"]
    msg = (
        f"Baseline: {c['baseline']['total_latency_sec']:.4f} sec\n"
        f"Optimized: {c['optimized']['total_latency_sec']:.4f} sec\n"
        f"Speedup: {c['speedup']:.3f}x\n"
        f"Cache hit rate: {c['optimized']['cache_hit_rate']:.3f}\n"
        f"Quality MSE: {c['quality']['mse']:.6e}  PSNR: {c['quality']['psnr']}\n"
        f"Decision: {c['decision'].upper()}\n"
    )
    notes = c.get("notes") or []
    if notes:
        msg += "Notes:\n" + "\n".join(f"• {n}" for n in notes)
    return {"ok": True, "message": msg, "comparison": c}


def tool_inspect_cpu() -> dict[str, Any]:
    info = inspect_cpu().to_dict()
    msg = (
        f"CPU arch={info['architecture']} cores={info['logical_cores']}\n"
        f"AVX2={info['avx2']} AVX-512={info['avx512']} NEON={info['neon']}\n"
        f"RAM total={info['total_ram_bytes'] / (1024**3):.1f} GiB "
        f"available={info['available_ram_bytes'] / (1024**3):.1f} GiB\n"
        f"torch threads={info.get('torch_num_threads')}"
    )
    return {"ok": True, "message": msg, "cpu": info}


def tool_build_model_graph(config_path: str | None = None) -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    loaded = load_model(cfg.model)
    nodes = sorted(loaded.graph.nodes)
    sample_path = " → ".join(
        n.name for n in loaded.graph.path("model.dit.block_0.attention.qkv")
    )
    msg = (
        f"Model: {loaded.info.name} ({loaded.info.parameter_count} params, "
        f"{loaded.info.num_blocks} blocks)\n"
        f"Nodes: {len(nodes)}\n"
        f"Example path: {sample_path}\n"
        f"Kinds present: "
        + ", ".join(sorted({n.kind.value for n in loaded.graph.nodes.values()}))
    )
    return {
        "ok": True,
        "message": msg,
        "parameter_count": loaded.info.parameter_count,
        "num_blocks": loaded.info.num_blocks,
        "node_count": len(nodes),
    }


def tool_profile_model(config_path: str | None = None) -> dict[str, Any]:
    cfg = _load_cfg(config_path)
    # Keep agent profile snappy
    cfg.profiler.warmup_runs = min(cfg.profiler.warmup_runs, 1)
    cfg.profiler.measure_runs = min(cfg.profiler.measure_runs, 1)
    cfg.model.num_steps = min(cfg.model.num_steps, 8)
    loaded = load_model(cfg.model)
    report = Profiler(cfg.profiler).profile(loaded, seed=cfg.experiment.seed)
    out = _results_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(jsonutil.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (out / "profile.txt").write_text(report.to_human_readable(), encoding="utf-8")
    tops = report.hotspots[:5]
    lines = [
        report.to_human_readable(),
        "",
        "Top hotspots:",
        *[f"  {h.name}: {h.share_pct:.1f}%" for h in tops],
    ]
    return {
        "ok": True,
        "message": "\n".join(lines),
        "total_latency_sec": report.total_latency_ms / 1000.0,
        "hotspots": [h.to_dict() for h in tops],
    }


def tool_analyze_cache() -> dict[str, Any]:
    root = _results_dir()
    stats = _safe_read_json(root / "cache_stats.json")
    comparison = _safe_read_json(root / "comparison.json")
    if stats is None and comparison is None:
        return {
            "ok": False,
            "message": "No cache stats. Ask me to 'run experiment' with cache enabled.",
        }
    hit = None
    if stats:
        hit = stats.get("hit_rate")
    elif comparison:
        hit = comparison.get("optimized", {}).get("cache_hit_rate")
    msg = f"Cache hit rate: {float(hit or 0):.3f}\n"
    if stats:
        msg += f"Hits: {stats.get('hits')}  Misses: {stats.get('misses')}\n"
        if stats.get("last_change"):
            msg += "Last feature changes:\n"
            for k, v in list(stats["last_change"].items())[:8]:
                msg += f"  {k}: {v}\n"
    return {"ok": True, "message": msg.strip(), "cache_stats": stats}


def tool_run_experiment(config_path: str | None = None) -> dict[str, Any]:
    from engine.experiments.runner import ExperimentRunner

    cfg = _load_cfg(config_path)
    # Reasonable defaults for interactive agent runs
    cfg.profiler.warmup_runs = 0
    cfg.profiler.measure_runs = 1
    cfg.benchmark.warmup_runs = 0
    cfg.benchmark.measure_runs = 1
    if cfg.cache.mode == CacheMode.DISABLED and cfg.optimization.enable_cache:
        cfg.cache.mode = CacheMode.CPU_AWARE
    result = ExperimentRunner(cfg).run()
    msg = (
        f"Experiment '{result.name}' finished.\n"
        f"Decision: {result.decision.value.upper()}\n"
        f"Speedup: {result.comparison.speedup:.3f}x\n"
        f"Cache hit rate: {result.optimized.cache_hit_rate:.3f}\n"
        f"Artifacts → {cfg.experiment.output_dir}/ "
        f"(report.md, images/, comparison.json)"
    )
    return {
        "ok": True,
        "message": msg,
        "decision": result.decision.value,
        "speedup": result.comparison.speedup,
        "experiment_id": result.experiment_id,
    }


def tool_run_benchmark(config_path: str | None = None) -> dict[str, Any]:
    from engine.benchmark.benchmark import Benchmark

    cfg = _load_cfg(config_path)
    cfg.benchmark.warmup_runs = 0
    cfg.benchmark.measure_runs = 1
    cfg.model.num_steps = min(cfg.model.num_steps, 8)
    loaded = load_model(cfg.model)
    metrics, _ = Benchmark(cfg.benchmark).run(loaded, seed=cfg.experiment.seed, label="agent")
    msg = (
        f"Benchmark baseline: {metrics.total_latency_sec:.4f} sec/image "
        f"({metrics.latency_per_step_ms:.2f} ms/step)\n"
        f"Peak RAM: {metrics.peak_ram_bytes / (1024**2):.1f} MiB"
    )
    return {"ok": True, "message": msg, "metrics": metrics.to_dict()}


def tool_list_images() -> dict[str, Any]:
    img_dir = _results_dir() / "images"
    if not img_dir.is_dir():
        return {"ok": False, "message": "No results/images directory yet."}
    files = sorted(p.name for p in img_dir.glob("*") if p.is_file())
    urls = [f"/results/images/{name}" for name in files]
    msg = "Images:\n" + "\n".join(f"• {u}" for u in urls) if urls else "No images found."
    return {"ok": True, "message": msg, "images": urls}


def tool_list_txt2img_models() -> dict[str, Any]:
    from engine.agent.txt2img_models import format_model_list, get_active_key, list_models

    models = [
        {
            "key": m.key,
            "name": m.name,
            "path": m.path,
            "steps": m.steps,
            "size": m.size,
            "notes": m.notes,
        }
        for m in list_models()
    ]
    return {
        "ok": True,
        "message": format_model_list(),
        "active": get_active_key(),
        "models": models,
    }


def tool_set_txt2img_model(model_key: str = "sd-turbo") -> dict[str, Any]:
    from engine.agent.txt2img_models import set_active_key

    try:
        model = set_active_key(model_key)
    except KeyError as exc:
        return {"ok": False, "message": str(exc)}
    return {
        "ok": True,
        "message": (
            f"Active txt2img model → **{model.key}** ({model.path or model.kind})\n"
            f"steps={model.steps} size={model.size}\n{model.notes}\n\n"
            "Next: `generate a cat image`"
        ),
        "active": model.key,
        "path": model.path,
    }


def tool_generate_image(
    prompt: str = "a scenic landscape",
    model_key: str | None = None,
) -> dict[str, Any]:
    from engine.agent.image_gen import generate_image

    result = generate_image(prompt, _results_dir() / "images", model_key=model_key)
    if not result.get("ok"):
        return {
            "ok": False,
            "message": str(result.get("message") or "Image generation failed."),
            "images": [],
        }
    url = str(result["url"])
    return {
        "ok": True,
        "message": str(result["message"]),
        "images": [url],
        "prompt": result.get("prompt"),
        "backend": result.get("backend"),
        "model_key": result.get("model_key"),
    }


def _load_cfg(config_path: str | None) -> EngineConfig:
    root = _project_root()
    path = Path(config_path) if config_path else root / "configs" / "cache.yaml"
    if not path.is_absolute():
        path = root / path
    if path.is_file():
        return load_config(path)
    return EngineConfig(model=ModelConfig(kind="toy_diffusion"))


REGISTRY: dict[str, ToolFn] = {
    "help": tool_help,
    "read_results": tool_read_results,
    "compare_results": tool_compare_results,
    "inspect_cpu": tool_inspect_cpu,
    "build_model_graph": tool_build_model_graph,
    "profile_model": tool_profile_model,
    "analyze_cache": tool_analyze_cache,
    "run_experiment": tool_run_experiment,
    "run_benchmark": tool_run_benchmark,
    "list_images": tool_list_images,
    "list_txt2img_models": tool_list_txt2img_models,
    "set_txt2img_model": tool_set_txt2img_model,
    "generate_image": tool_generate_image,
}


def list_tools() -> list[str]:
    return list(TOOL_NAMES)


def call_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    key = name.lower()
    if key not in REGISTRY:
        return {"ok": False, "message": f"Unknown tool '{name}'. Try 'help'."}
    return REGISTRY[key](**kwargs)
