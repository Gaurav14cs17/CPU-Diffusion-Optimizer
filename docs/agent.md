# Agent Architecture (Phase 5)

The agent is a reasoning layer over **engine tools**. It never bypasses benchmarks.

## Workflow

```
User request → understand repo → locate model → build graph
→ profile CPU → hypothesize → small reversible patch
→ benchmark → quality test → compare → keep or revert → explain
```

## Tools

- read_file, search_code, find_symbol
- build_model_graph, profile_model, run_benchmark
- inspect_cpu, inspect_memory, analyze_cache_stability
- create_patch, apply_patch, run_tests, compare_results

## Principle

Prefer small, reversible changes. Never rewrite the entire project to apply a cache policy. Never declare success without measured speedup **and** quality within threshold.
