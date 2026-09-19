export type Screen =
  | "project"
  | "graph"
  | "profile"
  | "cache"
  | "optimize"
  | "experiments"
  | "benchmark"
  | "generated"
  | "agent";

export interface ComparisonPayload {
  baseline: {
    total_latency_sec: number;
    peak_ram_bytes: number;
    cache_hit_rate: number;
  };
  optimized: {
    total_latency_sec: number;
    peak_ram_bytes: number;
    cache_hit_rate: number;
    cache_memory_bytes: number;
  };
  quality: {
    mse: number;
    psnr: number;
    passed: boolean;
  };
  speedup: number;
  memory_reduction: number;
  decision: string;
  notes: string[];
}

export interface ProfilePayload {
  model_name: string;
  resolution: string;
  steps: number;
  device: string;
  total_latency_sec: number;
  mean_step_latency_ms: number;
  peak_ram_bytes: number;
  hotspots: { name: string; share_pct: number; latency_ms: number }[];
  cpu: {
    architecture: string;
    logical_cores: number;
    avx2: boolean;
    avx512: boolean;
    neon: boolean;
  };
}

export interface CacheStats {
  hits: number;
  misses: number;
  hit_rate: number;
  last_change?: Record<string, number>;
}
