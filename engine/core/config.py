"""Structured configuration loaded from YAML / dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from engine.core.exceptions import ConfigError
from engine.core.types import CacheMode


class ModelConfig(BaseModel):
    name: str = "toy"
    kind: str = "toy_diffusion"
    path: str | None = None
    channels: int = 4
    height: int = 16
    width: int = 16
    hidden_dim: int = 32
    num_blocks: int = 4
    num_steps: int = 8
    batch_size: int = 1


class ProfilerConfig(BaseModel):
    warmup_runs: int = 1
    measure_runs: int = 3
    track_memory: bool = True


class QualityConfig(BaseModel):
    metrics: list[str] = Field(default_factory=lambda: ["mse", "psnr"])
    max_mse: float = 1e-3
    min_psnr: float = 30.0


class BenchmarkConfig(BaseModel):
    warmup_runs: int = 1
    measure_runs: int = 3
    quality: QualityConfig = Field(default_factory=QualityConfig)


class CacheConfig(BaseModel):
    mode: CacheMode = CacheMode.DISABLED
    threshold: float = 0.015
    epsilon: float = 1e-6
    max_consecutive_reuses: int = 1


class OptimizationConfig(BaseModel):
    enable_cache: bool = False
    enable_int8: bool = False
    enable_block_skip: bool = False


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    seed: int = 42
    output_dir: str = "results"


class EngineConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


def load_config(path: str | Path) -> EngineConfig:
    """Load an :class:`EngineConfig` from a YAML file."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {config_path}")
    try:
        return EngineConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid config {config_path}: {exc}") from exc
