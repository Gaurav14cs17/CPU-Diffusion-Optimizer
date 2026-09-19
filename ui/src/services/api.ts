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

export interface Txt2ImgModelInfo {
  key: string;
  name: string;
  path: string | null;
  steps: number;
  size: number;
  notes: string;
  kind: string;
}

export interface Txt2ImgModelsResponse {
  active: string;
  default: string;
  models: Txt2ImgModelInfo[];
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
  data.images = (data.images || []).map((src) =>
    src.includes("?") ? `${src}&t=${Date.now()}` : `${src}?t=${Date.now()}`,
  );
  return data;
}

export async function fetchTxt2ImgModels(): Promise<Txt2ImgModelsResponse | null> {
  return getJson<Txt2ImgModelsResponse>("/api/txt2img/models");
}

export async function selectTxt2ImgModel(
  modelKey: string,
): Promise<Txt2ImgModelsResponse> {
  const res = await fetch(`${API_BASE}/api/txt2img/models`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_key: modelKey }),
  });
  if (!res.ok) {
    throw new Error(`Select model failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as Txt2ImgModelsResponse;
}

export interface UploadImageResult {
  ok: boolean;
  url: string;
  path: string;
  filename: string;
}

export async function uploadTxt2ImgSource(file: File): Promise<UploadImageResult> {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch(`${API_BASE}/api/txt2img/upload`, {
    method: "POST",
    body,
  });
  const data = (await res.json().catch(() => ({}))) as UploadImageResult & {
    message?: string;
  };
  if (!res.ok || data.ok === false) {
    throw new Error(data.message || `Upload failed: ${res.status}`);
  }
  return data;
}

export interface EditImageResult {
  ok: boolean;
  message: string;
  url?: string | null;
  images: string[];
  prompt?: string | null;
  backend?: string | null;
  model_key?: string | null;
  strength?: number | null;
}

export async function editTxt2Img(args: {
  prompt: string;
  imagePath: string;
  modelKey?: string;
  strength?: number;
}): Promise<EditImageResult> {
  const res = await fetch(`${API_BASE}/api/txt2img/edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt: args.prompt,
      image_path: args.imagePath,
      model_key: args.modelKey,
      strength: args.strength ?? 0.55,
    }),
  });
  if (!res.ok) {
    throw new Error(`Edit failed: ${res.status} ${await res.text()}`);
  }
  const data = (await res.json()) as EditImageResult;
  data.images = (data.images || []).map((src) =>
    src.includes("?") ? `${src}&t=${Date.now()}` : `${src}?t=${Date.now()}`,
  );
  if (data.url && !data.url.includes("?")) {
    data.url = `${data.url}?t=${Date.now()}`;
  }
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
