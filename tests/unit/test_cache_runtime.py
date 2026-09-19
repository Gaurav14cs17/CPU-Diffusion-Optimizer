"""Runtime cache integration tests."""

from __future__ import annotations

from engine.cache.cache_manager import CacheManager
from engine.cache.runtime import run_cached_sample
from engine.core.config import CacheConfig, ModelConfig
from engine.core.types import CacheMode
from engine.model.loader import load_model


def test_static_block_cache_gets_hits() -> None:
    loaded = load_model(
        ModelConfig(
            kind="toy_diffusion",
            height=8,
            width=8,
            hidden_dim=16,
            num_blocks=4,
            num_steps=6,
        )
    )
    mgr = CacheManager(CacheConfig(mode=CacheMode.STATIC_BLOCK, threshold=0.5))
    result = run_cached_sample(
        loaded.module,  # type: ignore[arg-type]
        mgr,
        num_steps=6,
        batch_size=1,
        seed=0,
    )
    assert result["cache_hit_rate"] > 0.0
    assert result["output"].shape[0] == 1


def test_disabled_cache_zero_hits() -> None:
    loaded = load_model(
        ModelConfig(kind="toy_diffusion", height=8, width=8, hidden_dim=16, num_blocks=3, num_steps=4)
    )
    mgr = CacheManager(CacheConfig(mode=CacheMode.DISABLED))
    result = run_cached_sample(
        loaded.module,  # type: ignore[arg-type]
        mgr,
        num_steps=4,
        batch_size=1,
        seed=0,
    )
    assert result["cache_hit_rate"] == 0.0
