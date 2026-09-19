# Diffusion Feature / Block Caching

## Why diffusion has temporal redundancy

Adjacent diffusion timesteps often produce highly similar intermediate activations inside UNet / DiT blocks. That redundancy is the opportunity for **feature/block caching**: reuse `F(l, t-1)` when `F(l, t)` would change little.

## LLM KV cache vs diffusion feature cache

| | LLM KV cache | Diffusion feature cache |
|--|--------------|-------------------------|
| Axis | Sequence length (tokens) | Timestep (denoising steps) |
| What is stored | Keys/values for attention | Block / feature-map outputs |
| Invalidation | New tokens append | Feature change / cost model |
| Risk | Usually exact | Approximate — must quality-check |

## Stability

```
D(l,t) = ||F(l,t) - F(l,t-1)|| / (||F(l,t-1)|| + ε)
```

If `D(l,t) < threshold`, the block is a candidate for reuse.

## CPU-aware decision

Reuse only when:

```
T_reuse < T_compute
```

where `T_reuse` accounts for memory read, optional decompression, and lookup; `T_compute` is measured block latency.

## Cache modes

1. Disabled
2. Static block cache
3. Timestep-based cache
4. Adaptive feature cache
5. CPU-aware adaptive cache (default experimental mode in Phase 2+)

## Spatial caching (future)

`RegionKey` supports optional `(y0, x0, y1, x1)` so whole-block caching can evolve into tile/patch caching without rewriting the cache engine.
