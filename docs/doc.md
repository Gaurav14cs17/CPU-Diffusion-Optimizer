# Technical Discussion — CPU Diffusion Optimizer

This document is for technical discussion of goals, design choices, pipeline behavior, and open questions. It complements the shorter notes in `architecture.md`, `caching.md`, and `cpu_optimization.md`.

---

## 1. Problem statement

Diffusion inference is usually discussed in a **GPU** context (CUDA kernels, VRAM, batch throughput). Many real deployments still need **CPU** paths:

- Edge / laptop / cloud instances without GPUs
- Cost or compliance constraints
- Debugging, CI, and reproducible research harnesses

On CPUs, diffusion is often **memory-bandwidth bound** rather than pure FLOP-bound. Naïve “port the GPU recipe” (same caches, same quantization claims without measured kernels) fails or lies.

**Core question this project asks:**

> Can we systematically *discover*, *apply*, and *accept or reject* CPU-side optimizations for diffusion inference using measured evidence — not vibes?

---

## 2. Goals and non-goals

### Goals

| Goal | Meaning |
|------|---------|
| CPU-first engine | Primary path is x86 (AVX2/AVX-512) and ARM NEON where practical |
| Evidence loop | Every optimization is benchmarked + quality-gated |
| Pluggable frontends | CLI, library API, agent tools, UI — all call the same engine |
| Incremental phases | Toy model first; real UNet/DiT adapters later without rewriting the loop |

### Non-goals (for now)

- Shipping a consumer image-generation product
- Claiming Stable Diffusion / FLUX speedups without adapters + real weights
- Accepting “optimizations” that do not change measured latency or that break quality gates
- Making the engine depend on the UI or agent

---

## 3. System architecture

```
┌─────────────┐     ┌─────────────┐
│     UI      │────►│     API     │──┐
└─────────────┘     └─────────────┘  │
                                     ▼
┌─────────────┐                  ┌─────────────┐
│    Agent    │─────────────────►│   Engine    │
└─────────────┘                  └─────────────┘
┌─────────────┐                         ▲
│     CLI     │─────────────────────────┘
└─────────────┘
```

**Dependency rule:** edges point *into* the engine. The engine is a standalone Python library.

| Layer | Responsibility |
|-------|----------------|
| **Engine** | Load, graph, profile, cache, optimize, benchmark, quality, experiments |
| **CLI** | Headless entry (`python -m cli …`) |
| **API** | HTTP surface for UI / automation |
| **Agent** | Maps natural language → engine tools |
| **UI** | Developer IDE-like console; no optimization logic |

---

## 4. End-to-end pipeline

The vertical slice implemented by `ExperimentRunner`:

```
Config
  → Load model (adapter)
  → Build computation graph
  → CPU profile (latency / memory / hardware)
  → Baseline benchmark (cache OFF)
  → Optimized path (e.g. CPU-aware block cache)
  → Paired quality check (same seed)
  → Compare metrics → KEEP / REJECT / INCONCLUSIVE
  → Persist artifacts under results/
```

### Decision policy (summary)

| Condition | Decision |
|-----------|----------|
| Quality fails gates | **REJECT** |
| Quality OK + measured speedup ≥ 1× | **KEEP** |
| Quality OK + no net speedup | **INCONCLUSIVE** |
| Cache disabled (identity path) | **INCONCLUSIVE** |

No optimization is accepted without **paired** measurement.

### Artifacts

Typical outputs in `results/`:

- `baseline.json`, `optimized.json`, `comparison.json`
- `profile.json` / `profile.txt`
- `cache_stats.json`, `cache_log.json`
- `report.md`
- `images/baseline.png`, `images/optimized.png`
- `experiments.sqlite`

---

## 5. Model choice

### Experiment / optimize path

Default model is **`toy_diffusion`** (`examples/toy_diffusion/model.py`):

- Small DiT-like stack of residual blocks (attention + MLP)
- No external checkpoints
- Deterministic for a fixed seed
- Sized for fast CPU iteration (config example: 32×32 latent, 6 blocks, 24 steps)

**Why toy first?** The product is the *optimization engine*, not a particular foundation model. Toy weights let us validate graph extraction, profiling, cache decisions, and the accept/reject loop before investing in heavy adapters.

### Adapters (extensibility)

Loader resolves `ModelConfig.kind` through a registry:

- `toy_diffusion` — implemented (default experiment path)
- `diffusers` — offline CPU Diffusers (e.g. `stabilityai/sd-turbo`)
- `fastsdcpu` / `openvino` — FastSDCPU-**aligned** baseline (same Turbo/OpenVINO model class as [rupeshs/fastsdcpu](https://github.com/rupeshs/fastsdcpu); does not vendor that app). Install: `pip install -e ".[fastsdcpu]"`. Config: `configs/fastsdcpu.yaml`
- `pytorch` — scaffolding for custom `nn.Module` checkpoints

**Why not FastSDCPU as the default?** That project is a consumer image app (GUI + distilled few-step models + OpenVINO). This repo’s product is the *evidence loop* (profile → optimize → quality-gate). FastSDCPU-class models are a **baseline adapter** for later real-UNet measurements, not a replacement for `toy_diffusion`.

### Image chat path (offline CPU, multi-model)

UI/agent image gen uses a **catalog** (`configs/txt2img_models.yaml`):

| Key | HF / kind | Notes |
|-----|-----------|--------|
| `sd-turbo` | `stabilityai/sd-turbo` | Default, 2-step |
| `sdxl-turbo` | `stabilityai/sdxl-turbo` | Heavier turbo |
| `sdxs` | `IDKiro/sdxs-512-0.9` | 1-step (FastSD-style) |
| `bk-sdm-tiny` | `nota-ai/bk-sdm-tiny` | Small compressed SD |
| `tiny-sd` | `segmind/tiny-sd` | Compact distill |
| `lcm-dreamshaper` | `SimianLuo/LCM_Dreamshaper_v7` | LCM few-step |
| `toy` | builtin | Latent fallback |

Agent / UI:

- `list models` · model picker dropdown · **Auto**
- `use model sdxs`
- `generate a cat image` / **Generate**
- **Upload** an image → type guidance → **Edit** (img2img, strength slider)

Env overrides: `CDO_TXT2IMG_MODEL`, `CDO_TXT2IMG_STEPS`, `CDO_TXT2IMG_SIZE`, `CDO_TXT2IMG_CATALOG`. Install: `pip install -e ".[diffusers]"`.

---

## 6. Caching discussion

### Motivation

Adjacent diffusion timesteps often produce similar intermediate activations. Reusing block outputs when change is small can skip compute — analogous in spirit to LLM KV cache, but along the **timestep** axis and with **approximate** reuse risk.

| | LLM KV cache | Diffusion feature / block cache |
|--|--------------|-----------------------------------|
| Axis | Token sequence | Denoising timestep |
| Stored | K/V | Block / feature maps |
| Exactness | Usually exact | Approximate → must quality-gate |

### Stability metric

\[
D(l,t) = \frac{\|F(l,t) - F(l,t-1)\|}{\|F(l,t-1)\| + \varepsilon}
\]

If \(D(l,t) < \tau\), the block is a **reuse candidate**.

### CPU-aware gate

Even if features are stable, reuse must be cheaper than compute:

\[
T_{\text{reuse}} < T_{\text{compute}}
\]

`BandwidthCostModel` approximates:

- \(T_{\text{compute}}\) ≈ measured block latency
- \(T_{\text{reuse}}\) ≈ `bytes / DRAM_bandwidth + lookup (+ optional decompress)`

Default experimental mode: **`cpu_aware`** (`configs/cache.yaml`).

### Modes

1. Disabled  
2. Static block cache  
3. Timestep-based cache  
4. Adaptive feature cache  
5. CPU-aware adaptive cache  

### Open discussion points

- Calibrating DRAM bandwidth per machine instead of a fixed guess (~20 GB/s)
- Cap on consecutive reuses (`max_consecutive_reuses`) vs quality drift
- Spatial / tile caching (`RegionKey`) for large feature maps
- Interaction with INT8 and fused ops (reuse of quantized tensors)

---

## 7. CPU optimization stance

Designed considerations:

- ISA detection: AVX2 / AVX-512 / NEON
- Peak RSS / RAM pressure
- Tensor copies and layout / cache locality
- Threading (`torch` / OMP)
- Planned: operator fusion, INT8 (real kernels), INT4 later

Quantization path (aspirational):

```
FP32 → FP16/BF16 (where useful on CPU) → INT8 → INT4
```

Fake quantization that does not change kernels is **not** treated as a real win.

---

## 8. Benchmark and quality methodology

Every claim should come from:

1. Warmup runs  
2. Repeated timed measurements  
3. Quality metrics (MSE, PSNR; SSIM optional later)  
4. Accept only if gates pass (`max_mse`, `min_psnr` in config)

Paired seeds ensure baseline vs optimized compare the same stochastic draw.

---

## 9. Agent and UI

Agent tools wrap the engine (`profile_model`, `run_experiment`, `compare_results`, `generate_image`, …). The UI is a Cursor-like shell: explorer, artifact views, terminal, composer. Optimization math stays in the engine.

Discussion topics for later:

- How much planning/reasoning belongs in the agent vs fixed tool scripts  
- Safe auto-patch loops (propose config/code change → re-run experiment → keep/revert)  
- Multi-model experiment matrices in the UI  

---

## 10. Design tradeoffs

| Choice | Benefit | Cost |
|--------|---------|------|
| Toy model first | Fast, deterministic CI | No SD/FLUX numbers yet |
| Evidence-required KEEP | Trustworthy claims | Slow iteration; many INCONCLUSIVE on tiny models |
| CPU-aware cache | Avoids reuse that is slower than compute | Needs good bandwidth / latency estimates |
| Separate image-gen via same pipeline | Offline, consistent with engine | Toy latents ≠ photoreal demos |
| Engine independence | Library-usable | More packaging discipline |

---

## 11. Phase roadmap (discussion)

| Phase | Focus | Notes |
|-------|--------|------|
| 1 | Toy, graph, profiler, benchmark | Done |
| 2 | Feature/block cache + stability | Wired in runner |
| 3 | Cost model calibration, experiments DB | Partial |
| 4 | INT8, memory, block skipping | Stub / early |
| 5 | Agent tools + patch loop | Tools exist; patch loop early |
| 6 | API + UI | Working vertical slice |

---

## 12. Open questions for discussion

1. When is block reuse net-negative on small models (toy) but positive on SD-scale UNets?  
2. Should KEEP require a minimum speedup margin (e.g. ≥ 1.05×) to absorb noise?  
3. How do we report **energy** / thermal throttling on laptops?  
4. What’s the right abstraction for Diffusers pipelines (full pipeline vs UNet-only)?  
5. Can the agent propose cache thresholds from profile hotspots automatically?  
6. How should we version experiment configs so results are comparable across commits?

---

## 13. How to reproduce the discussion baseline

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m cli experiment --config configs/cache.yaml
```

Inspect `results/report.md` and `results/comparison.json` for the measured decision.

---

## 14. Related docs

| Doc | Scope |
|-----|--------|
| `architecture.md` | Layer diagram, pipeline one-liner |
| `caching.md` | Feature/block cache theory |
| `cpu_optimization.md` | CPU-bound notes + methodology |
| `agent.md` | Agent tool surface |
| `api.md` | HTTP API |

---

*This document is intended as a living technical discussion artifact. Update it when pipeline semantics, cache policy, or model adapters change.*
