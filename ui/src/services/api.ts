import type { CacheStats, ComparisonPayload, ProfilePayload } from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function getJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(path);
    if (!res.ok) return null;
    const text = await res.text();
    const safe = text
      .replace(/:\s*Infinity\b/g, ": null")
      .replace(/:\s*-Infinity\b/g, ": null")
      .replace(/:\s*NaN\b/g, ": null");
    return JSON.parse(safe) as T;
  } catch {
    return null;
  }
}

async function getText(path: string): Promise<string | null> {
  try {
    const res = await fetch(path);
    if (!res.ok) return null;
    return await res.text();
  } catch {
    return null;
  }
}

export interface AgentChatResult {
  reply: string;
  tool: string;
  images: string[];
  ok: boolean;
}

/** UI reads artifacts / calls engine API — no optimization logic here. */
export const resultsApi = {
  comparison: () => getJson<ComparisonPayload>("/results/comparison.json"),
  profile: () => getJson<ProfilePayload>("/results/profile.json"),
  cacheStats: () => getJson<CacheStats>("/results/cache_stats.json"),
  report: () => getText("/results/report.md"),
  profileText: () => getText("/results/profile.txt"),
  imageUrls: () => ({
    baseline: "/results/images/baseline.png",
    optimized: "/results/images/optimized.png",
  }),
};

export async function agentChat(
  message: string,
  opts?: { signal?: AbortSignal },
): Promise<AgentChatResult> {
  const res = await fetch(`${API_BASE}/api/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, config_path: "configs/cache.yaml" }),
    signal: opts?.signal,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Agent API ${res.status}: ${detail}`);
  }
  const data = (await res.json()) as AgentChatResult;
  // Bust browser cache so newly generated images always show
  data.images = (data.images || []).map((src) =>
    src.includes("?") ? `${src}&t=${Date.now()}` : `${src}?t=${Date.now()}`,
  );
  return data;
}

export async function apiHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.ok;
  } catch {
    return false;
  }
}
