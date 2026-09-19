"""Shared CLI helpers."""

from __future__ import annotations

from pathlib import Path

from engine.core.config import EngineConfig, ModelConfig, load_config


def resolve_config(
    *,
    model: str,
    config_path: str | None,
    steps: int | None = None,
) -> EngineConfig:
    if config_path:
        cfg = load_config(Path(config_path))
    else:
        kind = "toy_diffusion" if model in {"toy", "toy_diffusion"} else model
        cfg = EngineConfig(model=ModelConfig(name=model, kind=kind))
    if steps is not None:
        cfg.model.num_steps = steps
    return cfg
