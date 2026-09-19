"""Cache unit tests (interfaces + stability + policy)."""

from __future__ import annotations

import pytest
import torch

from engine.cache.block_cache import BlockCache
from engine.cache.cache_policy import CpuAwarePolicy, ThresholdPolicy
from engine.cache.cost_model import BandwidthCostModel
from engine.cache.feature_cache import InMemoryFeatureCache, RegionKey
from engine.cache.stability import feature_change


def test_feature_cache_hit_miss() -> None:
    cache = InMemoryFeatureCache()
    key = RegionKey(block_id="block_0", timestep=1)
    assert cache.get(key) is None
    assert cache.misses == 1
    cache.put(key, torch.ones(2, 2))
    got = cache.get(key)
    assert got is not None
    assert torch.equal(got, torch.ones(2, 2))
    assert cache.hits == 1


def test_cache_invalidation() -> None:
    cache = BlockCache()
    cache.put_block("a", 0, torch.zeros(1))
    cache.put_block("b", 0, torch.ones(1))
    cache.invalidate("a")
    assert cache.get_block("a", 0) is None
    assert cache.get_block("b", 0) is not None


def test_feature_change_identical_and_different() -> None:
    a = torch.randn(4, 4)
    assert feature_change(a, a.clone()) < 1e-6
    b = a + 10.0
    assert feature_change(b, a) > 0.5


def test_feature_change_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        feature_change(torch.zeros(2, 2), torch.zeros(3, 3))


def test_threshold_policy() -> None:
    policy = ThresholdPolicy(threshold=0.1)
    reuse = policy.should_reuse(
        block_id="b0", timestep=1, feature_change=0.05, t_reuse=0.01, t_compute=0.1
    )
    miss = policy.should_reuse(
        block_id="b0", timestep=2, feature_change=0.5, t_reuse=0.01, t_compute=0.1
    )
    assert reuse.reuse is True
    assert miss.reuse is False


def test_cpu_aware_requires_cheaper_reuse() -> None:
    policy = CpuAwarePolicy(threshold=0.1)
    # Stable but reuse more expensive than compute
    decision = policy.should_reuse(
        block_id="b0", timestep=1, feature_change=0.01, t_reuse=1.0, t_compute=0.1
    )
    assert decision.reuse is False
    assert decision.reason == "reuse_not_cheaper"


def test_cost_model_bandwidth() -> None:
    model = BandwidthCostModel()
    est = model.estimate(
        block_latency_sec=0.05,
        tensor_bytes=10_000_000,
        memory_bandwidth_bytes_per_sec=1e9,
    )
    assert est.t_compute_sec == 0.05
    assert est.t_reuse_sec == pytest.approx(0.01, rel=1e-3)
