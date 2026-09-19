# Architecture

## Layers

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

Dependency direction is one-way into the engine. The engine must remain usable as a standalone Python library.

## Engine pipeline

```
Model → Loader → Graph → CPU Profiler → Hotspot Detection
      → Optimization Planner → Experiment → Benchmark
      → Quality Check → Accept / Reject
```

No optimization is accepted without measured evidence.

## CPU-first

Primary execution path is CPU (AVX2 / AVX-512 / NEON where available). GPU may be added later as an alternate device backend without restructuring core modules.

## Phase status

| Phase | Focus                                      | Status   |
|-------|--------------------------------------------|----------|
| 1     | Toy model, graph, profiler, benchmark      | Done     |
| 2     | Feature/block cache + stability            | Interfaces ready |
| 3     | CPU cost model, adaptive reuse, experiments| Partial (runner) |
| 4     | INT8, memory, block skipping               | Stub     |
| 5     | Agent tools + patch loop                   | Stub     |
| 6     | API + UI                                   | Stub     |
