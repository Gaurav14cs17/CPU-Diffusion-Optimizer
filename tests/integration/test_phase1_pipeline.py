"""Integration: config load + analyze path."""

from __future__ import annotations

from pathlib import Path

from engine.core.config import load_config
from engine.model.loader import load_model


def test_load_default_cache_config() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "cache.yaml")
    assert cfg.model.kind == "toy_diffusion"
    assert cfg.cache.mode.value == "cpu_aware"
    assert cfg.optimization.enable_cache is True
    loaded = load_model(cfg.model)
    assert loaded.graph.total_parameters() > 0
