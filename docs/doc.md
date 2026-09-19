# Technical Discussion — CPU Diffusion Optimizer

Pipeline-first technical reference. Complements `architecture.md`, `caching.md`, and `cpu_optimization.md`.

---

## 1. Problem and product

Diffusion inference is usually a **GPU** story. Many deployments still need **CPU** paths (edge, cost, compliance, CI). On CPUs, diffusion is often **memory-bandwidth bound**; naïve GPU recipes fail or overclaim.

**Core question:**

> Can we systematically *discover*, *apply*, and *accept or reject* CPU-side optimizations using measured evidence?

**Product:** an evidence loop, not an image-generation app. Frontends (CLI, API, agent, UI) only call the engine.

```
UI ──► API ──► Engine
Agent ───────► Engine
CLI ─────────► Engine
```

---

## 2. Pipeline overview

Implemented by `ExperimentRunner` (`engine/experiments/runner.py`):

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Config  │──►│   Load   │──►│  Graph   │──►│ Profile  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                      │
     ┌────────────────────────────────────────────────┘
     ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Baseline │──►│ Optimize │──►│ Quality  │──►│ Decision │
│ bench    │   │ + bench  │   │ (paired) │   │ KEEP/…   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                      │
                                                      ▼
                                               ┌──────────┐
                                               │ Persist  │
                                               │ results/ │
                                               └──────────┘
```

| Stage | Module(s) | Output |
|-------|-----------|--------|
| **Config** | `EngineConfig`, YAML (`configs/*.yaml`) | Hypothesis + knobs |
| **Load** | `load_model`, adapters | `LoadedModel` (module + adapter) |
| **Graph** | `adapter.build_graph` | Normalized `ModelGraph` |
| **Profile** | `Profiler` | Latency / memory / CPU info |
| **Baseline** | `Benchmark` (cache OFF) | `baseline.json` metrics |
| **Optimize** | `CacheManager` + modes | Cached inference path |
| **Optimized bench** | `Benchmark` (cache ON) | `optimized.json`, cache stats |
| **Quality** | `QualityValidator` (same seed) | MSE / PSNR gate |
| **Decision** | `compare` | KEEP / REJECT / INCONCLUSIVE |
| **Persist** | runner + `ExperimentDatabase` | Artifacts under `results/` |

CLI entry for the full pipeline:

```bash
python -m cli experiment --config configs/cache.yaml
```

Partial stages via CLI: `analyze`, `profile`, `benchmark`.

---

## 3. Stage: Config

Inputs that drive every later stage:

| Knob | Role in pipeline |
|------|------------------|
| `model.kind` / `name` | Which adapter loads |
| `model.num_steps`, `batch_size` | Inference shape |
| `cache.mode`, `threshold` | Optimize-stage policy |
| `benchmark.quality` | Quality-stage gates (`max_mse`, `min_psnr`) |
| `experiment.seed`, `output_dir` | Paired runs + artifact path |

Default experiment config: `configs/cache.yaml` (CPU-aware cache mode).

---

## 4. Stage: Load + Graph

```
ModelConfig.kind
  → MODEL_ADAPTERS registry
  → adapter.load → to_cpu → build_graph → info
  → LoadedModel
```

| `kind` | Role in pipeline | Notes |
|--------|------------------|--------|
| `toy_diffusion` | **Default experiment path** | Small DiT-like stack; no checkpoints; fast CPU loop |
| `diffusers` | Real offline CPU Diffusers | e.g. `stabilityai/sd-turbo` |
| `fastsdcpu` / `openvino` | FastSDCPU-aligned baseline | `pip install -e ".[fastsdcpu]"`; `configs/fastsdcpu.yaml` |
| `pytorch` | Custom `nn.Module` scaffolding | Early |

**Why toy first?** The pipeline (graph → profile → cache → gate) is the product. Validate stages before heavy adapters.

Graph is built at load time so profiling and cache hooks attach to named blocks, not opaque pipelines.

---

## 5. Stage: Profile

`Profiler` runs a timed CPU pass on the loaded model:

- Operator / block latency
- Peak memory / RSS pressure
- Hardware snapshot (`inspect_cpu`: AVX2 / AVX-512 / NEON where available)

Feeds hotspot intuition and cost-model inputs for the optimize stage. Artifacts: `profile.json`, `profile.txt`.

CPU-bound stance for this stage and later opts:

- Memory bandwidth and tensor copies matter as much as FLOPs
- Threading (`torch` / OMP), layout, locality
- Planned later in the same pipeline slot: fusion, INT8 (real kernels), INT4 — fake quant that does not change kernels is not a win

---

## 6. Stage: Baseline benchmark

Always runs with **cache disabled**:

```
Benchmark.run(loaded, seed=…, label="baseline")
```

Warmup + repeated timed measurements. Establishes the paired reference for speedup and quality. Artifact: `baseline.json`.

---

## 7. Stage: Optimize (cache path)

Adjacent timesteps often have similar block activations. The optimize stage tries **reuse** along the timestep axis (approximate → must pass quality).

### Stability (candidate)

\[
D(l,t) = \frac{\|F(l,t) - F(l,t-1)\|}{\|F(l,t-1)\| + \varepsilon}
\]

If \(D(l,t) < \tau\), block \(l\) at step \(t\) is a reuse candidate.

### CPU-aware gate (accept reuse)

Even if stable, reuse only when cheaper than compute:

\[
T_{\text{reuse}} < T_{\text{compute}}
\]

`BandwidthCostModel`: \(T_{\text{compute}}\) ≈ measured block latency; \(T_{\text{reuse}}\) ≈ bytes / DRAM bandwidth + lookup (+ optional decompress).

### Modes (pipeline knobs)

| Mode | Behavior in this stage |
|------|------------------------|
| Disabled | Identity path (no optimize) |
| Static block cache | Fixed block reuse |
| Timestep-based | Reuse by step schedule |
| Adaptive feature | Stability threshold only |
| **CPU-aware** (default experimental) | Stability **and** \(T_{\text{reuse}} < T_{\text{compute}}\) |

Runtime: `CacheManager` wraps `adapter.run_inference(..., cache_manager=…)`. Artifacts: `cache_log.json`, `cache_stats.json`, hit rate on optimized metrics.

Open knobs: per-machine DRAM calibration, `max_consecutive_reuses`, spatial/tile keys, interaction with INT8.

---

## 8. Stage: Optimized benchmark

Same `Benchmark` harness as baseline, with cache-enabled `inference_fn` when optimize is on. Artifact: `optimized.json` (+ cache hit / memory fields when available).

If cache is off, this stage is an identity re-run (feeds INCONCLUSIVE later).

---

## 9. Stage: Quality (paired)

Same seed for both paths:

```
baseline_out  = run_inference(…, seed=S)           # no cache
optimized_out = run_inference(…, seed=S, cache=…)  # optimize path
QualityValidator.evaluate(baseline_out, optimized_out)
```

Metrics: MSE, PSNR (SSIM later). Gates from config. Failures force **REJECT** regardless of speedup.

Latent visualizations: `images/baseline.png`, `images/optimized.png`.

---

## 10. Stage: Decision

| Condition | Decision |
|-----------|----------|
| Quality fails gates | **REJECT** |
| Quality OK + speedup ≥ 1× | **KEEP** |
| Quality OK + no net speedup | **INCONCLUSIVE** |
| Cache disabled (identity) | **INCONCLUSIVE** |

No optimization is accepted without **paired** measurement. Open: require a noise margin (e.g. ≥ 1.05×) before KEEP.

---

## 11. Stage: Persist

Typical `results/`:

```
results/
├── baseline.json
├── optimized.json
├── comparison.json
├── profile.json / profile.txt
├── cache_stats.json / cache_log.json
├── report.md
├── experiment.json
├── experiments.sqlite
└── images/
    ├── baseline.png
    └── optimized.png
```

---

## 12. Side pipelines (same engine)

Not the experiment vertical slice, but same load/inference surface:

### Image chat (offline CPU catalog)

`configs/txt2img_models.yaml` → agent/UI generate / edit:

| Key | Notes |
|-----|--------|
| `sd-turbo` | Default, 2-step |
| `sdxl-turbo` | Heavier turbo |
| `sdxs` | 1-step (FastSD-style) |
| `bk-sdm-tiny` / `tiny-sd` | Compact |
| `lcm-dreamshaper` | LCM few-step |
| `toy` | Latent fallback |

Env: `CDO_TXT2IMG_MODEL`, `CDO_TXT2IMG_STEPS`, `CDO_TXT2IMG_SIZE`, `CDO_TXT2IMG_CATALOG`. Install: `pip install -e ".[diffusers]"`.

### Agent / API / UI

Agent tools (`profile_model`, `run_experiment`, `compare_results`, `generate_image`, …) invoke engine stages. UI is a console only — no optimization math. Future: auto-patch loop (propose change → re-run pipeline → keep/revert).

---

## 13. Goals, non-goals, tradeoffs

**Goals:** CPU-first engine · evidence loop · pluggable frontends · incremental adapters without rewriting the pipeline.

**Non-goals (for now):** consumer image product · SD/FLUX speed claims without adapters · accepting unmeasured “wins” · engine depending on UI/agent.

| Choice | Benefit | Cost |
|--------|---------|------|
| Toy default in pipeline | Fast CI, deterministic | No SD-scale numbers yet |
| Evidence-required KEEP | Trustworthy | Many INCONCLUSIVE on tiny models |
| CPU-aware in optimize stage | Avoids slow reuse | Needs good bandwidth estimates |
| Shared engine for image chat | One load/infer path | Toy latents ≠ photoreal demos |

---

## 14. Phase roadmap (by pipeline maturity)

| Phase | Pipeline focus | Status |
|-------|----------------|--------|
| 1 | Load → graph → profile → baseline bench | Done |
| 2 | Optimize (feature/block cache) + quality gate | Wired in runner |
| 3 | Cost-model calibration, experiments DB | Partial |
| 4 | INT8 / memory / block skipping in optimize slot | Stub / early |
| 5 | Agent tools + patch loop around pipeline | Tools exist; patch early |
| 6 | API + UI frontends | Working vertical slice |

---

## 15. Open questions

1. When is block reuse net-negative on toy but positive on SD-scale UNets?  
2. Should KEEP require a minimum speedup margin (e.g. ≥ 1.05×)?  
3. How to report energy / thermal throttling on laptops?  
4. Diffusers abstraction: full pipeline vs UNet-only in load/graph?  
5. Can the agent set cache thresholds from profile hotspots?  
6. How to version configs so `results/` are comparable across commits?

---

## 16. Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
python -m cli experiment --config configs/cache.yaml
```

Inspect `results/report.md` and `results/comparison.json` for the measured decision.

---

## 17. Related docs

| Doc | Scope |
|-----|--------|
| `architecture.md` | Layer diagram, pipeline one-liner |
| `caching.md` | Feature/block cache theory (optimize stage) |
| `cpu_optimization.md` | CPU-bound notes + methodology |
| `agent.md` | Agent tool surface |
| `api.md` | HTTP API |

---

*Living doc. Update when pipeline stages, cache policy, or adapters change.*
