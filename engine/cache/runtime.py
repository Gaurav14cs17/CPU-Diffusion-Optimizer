"""Runtime block-feature caching for diffusion timesteps."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from engine.cache.cache_manager import CacheLogEntry, CacheManager
from engine.cache.stability import feature_change
from engine.core.types import CacheMode
from examples.toy_diffusion.model import ToyDiT, ToyDiffusionModel


@dataclass
class CacheRuntimeStats:
    hits: int = 0
    misses: int = 0
    block_latency_sec: dict[str, float] = field(default_factory=dict)
    last_change: dict[str, float] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "block_latency_sec": dict(self.block_latency_sec),
            "last_change": dict(self.last_change),
        }


class CachedDenoiser:
    """Run a ToyDiT with optional cross-timestep block feature reuse."""

    def __init__(self, model: ToyDiffusionModel, manager: CacheManager) -> None:
        self.model = model
        self.denoiser: ToyDiT = model.denoiser
        self.manager = manager
        self.stats = CacheRuntimeStats()
        self._prev_features: dict[str, torch.Tensor] = {}
        self._consecutive_reuses: dict[str, int] = {}
        self._calibrated = False

    def reset(self) -> None:
        self._prev_features.clear()
        self._consecutive_reuses.clear()
        self.stats = CacheRuntimeStats()
        self.manager.log.clear()
        self.manager.block_cache.invalidate()
        self._calibrated = False
    def _calibrate_block_latency(self, sample_tokens: torch.Tensor) -> None:
        if self._calibrated:
            return
        with torch.inference_mode():
            x = sample_tokens
            for idx, block in enumerate(self.denoiser.blocks):
                block_id = f"block_{idx}"
                t0 = time.perf_counter()
                x = block(x)
                self.stats.block_latency_sec[block_id] = max(time.perf_counter() - t0, 1e-6)
        self._calibrated = True

    def _schedule_allows_reuse(self, timestep: int, block_idx: int) -> bool:
        """Return True when the cache *mode schedule* permits attempting reuse."""
        mode = self.manager.config.mode
        n_blocks = len(self.denoiser.blocks)

        if mode == CacheMode.DISABLED:
            return False
        # Only middle block(s) are cache candidates — preserves residual stream ends.
        mid = n_blocks // 2
        cacheable = {mid} if n_blocks <= 4 else {mid - 1, mid}
        if block_idx not in cacheable:
            return False
        if timestep == 0:
            return False

        if mode == CacheMode.STATIC_BLOCK:
            return timestep % 2 == 0
        if mode == CacheMode.TIMESTEP:
            return timestep % 3 != 0
        # adaptive / cpu_aware: eligible every other step to limit drift
        return timestep % 2 == 0

    def _log(
        self,
        *,
        block_id: str,
        timestep: int,
        reuse: bool,
        feature_change_v: float,
        feature_bytes: int,
        compute_time: float,
        reuse_time: float,
        decision: str,
    ) -> None:
        self.manager.record(
            CacheLogEntry(
                block_id=block_id,
                timestep=timestep,
                cache_hit=reuse,
                cache_miss=not reuse,
                feature_change=feature_change_v,
                feature_size=feature_bytes,
                compute_time=compute_time,
                reuse_time=reuse_time,
                decision=decision,
            )
        )

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, *, step_index: int) -> torch.Tensor:
        dit = self.denoiser
        b, _, h, w = x.shape
        h_map = dit.input_proj(x)
        tokens = h_map.flatten(2).transpose(1, 2)
        t_emb = dit.time_embed(timesteps).unsqueeze(1)
        tokens = tokens + t_emb

        if self.manager.enabled and not self._calibrated:
            self._calibrate_block_latency(tokens.clone())

        for idx, block in enumerate(dit.blocks):
            block_id = f"block_{idx}"
            prev = self._prev_features.get(block_id)
            eligible = (
                self.manager.enabled
                and self._schedule_allows_reuse(step_index, idx)
                and prev is not None
                and prev.shape == tokens.shape
                and self._consecutive_reuses.get(block_id, 0)
                < self.manager.config.max_consecutive_reuses
            )

            reuse = False
            reason = "recompute"
            change = self.stats.last_change.get(block_id, float("inf"))
            feature_bytes = int((prev if prev is not None else tokens).numel() * tokens.element_size())
            latency = self.stats.block_latency_sec.get(block_id, 1e-3)
            estimate = self.manager.cost_model.estimate(
                block_latency_sec=latency,
                tensor_bytes=feature_bytes,
                memory_bandwidth_bytes_per_sec=self.manager.memory_bandwidth_bytes_per_sec,
            )

            if eligible:
                mode = self.manager.config.mode
                # Schedule modes treat features as reusable when eligible
                change_for_policy = 0.0 if mode in {CacheMode.STATIC_BLOCK, CacheMode.TIMESTEP} else change
                decision = self.manager.policy.should_reuse(  # type: ignore[union-attr]
                    block_id=block_id,
                    timestep=step_index,
                    feature_change=change_for_policy,
                    t_reuse=estimate.t_reuse_sec,
                    t_compute=estimate.t_compute_sec,
                )
                reuse = decision.reuse
                reason = decision.reason

            if reuse and prev is not None:
                tokens = prev
                self.stats.hits += 1
                self._consecutive_reuses[block_id] = self._consecutive_reuses.get(block_id, 0) + 1
                self._log(
                    block_id=block_id,
                    timestep=step_index,
                    reuse=True,
                    feature_change_v=change,
                    feature_bytes=feature_bytes,
                    compute_time=estimate.t_compute_sec,
                    reuse_time=estimate.t_reuse_sec,
                    decision=reason,
                )
                self.manager.block_cache.put_block(block_id, step_index, tokens)
                continue

            t0 = time.perf_counter()
            tokens = block(tokens)
            elapsed = time.perf_counter() - t0
            prev_lat = self.stats.block_latency_sec.get(block_id, elapsed)
            self.stats.block_latency_sec[block_id] = 0.8 * prev_lat + 0.2 * elapsed
            self.stats.misses += 1
            self._consecutive_reuses[block_id] = 0

            if prev is not None and prev.shape == tokens.shape:
                change = feature_change(tokens, prev, epsilon=self.manager.config.epsilon)
                self.stats.last_change[block_id] = change
            else:
                change = float("inf")
                self.stats.last_change[block_id] = change

            self._prev_features[block_id] = tokens.detach().clone()
            self.manager.block_cache.put_block(block_id, step_index, tokens)
            self._log(
                block_id=block_id,
                timestep=step_index,
                reuse=False,
                feature_change_v=change,
                feature_bytes=int(tokens.numel() * tokens.element_size()),
                compute_time=estimate.t_compute_sec,
                reuse_time=estimate.t_reuse_sec,
                decision=reason if eligible else "forced_or_cold",
            )

        tokens = dit.norm_out(tokens)
        h_map = tokens.transpose(1, 2).reshape(b, -1, h, w)
        return dit.output_proj(h_map)


def run_cached_sample(
    model: ToyDiffusionModel,
    manager: CacheManager,
    *,
    num_steps: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Full diffusion loop with optional block caching."""
    model = model.to("cpu")
    model.eval()
    runner = CachedDenoiser(model, manager)
    runner.reset()

    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (batch_size, model.spec.channels, model.spec.height, model.spec.width)
    x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

    block_timings_ms: dict[str, float] = {f"block_{i}": 0.0 for i in range(model.num_blocks)}
    per_step_ms: list[float] = []

    t0 = time.perf_counter()
    for step in range(num_steps):
        step_t0 = time.perf_counter()
        t_val = float(num_steps - step) / float(num_steps)
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
        noise_pred = runner.forward(x, t, step_index=step)
        x = x - (1.0 / num_steps) * noise_pred
        per_step_ms.append((time.perf_counter() - step_t0) * 1000.0)
    total_ms = (time.perf_counter() - t0) * 1000.0

    for block_id, sec in runner.stats.block_latency_sec.items():
        block_timings_ms[block_id] = sec * 1000.0 * num_steps

    cache_stats = manager.block_cache.stats()
    return {
        "output": x.detach().cpu(),
        "total_ms": total_ms,
        "per_step_ms": per_step_ms,
        "block_timings_ms": block_timings_ms,
        "cache_hit_rate": runner.stats.hit_rate,
        "cache_memory_bytes": int(cache_stats.get("approx_bytes", 0)),
        "cache_stats": runner.stats.to_dict(),
        "cache_log": [e.to_dict() for e in manager.log],
    }
