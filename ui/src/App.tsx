import {
  CSSProperties,
  FormEvent,
  MouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { agentChat, apiHealth, resultsApi } from "./services/api";
import type {
  CacheStats,
  ComparisonPayload,
  ProfilePayload,
  Screen,
} from "./types";
import "./styles.css";

function looksLikeImageRequest(msg: string): boolean {
  const t = msg.toLowerCase();
  return (
    /\b(generate|genrate|create|draw|make|paint|render)\b/.test(t) ||
    /\b(image|picture|photo|png)\b/.test(t) ||
    /\b(cat|dog|bird|car|sunset|flower)\b/.test(t)
  );
}

type ChatMsg = {
  role: "user" | "assistant";
  text: string;
  images?: string[];
  tool?: string;
};

type EditorTab = {
  id: Screen | "file";
  title: string;
  path?: string;
};

const ACTIVITY: { id: Screen; label: string; icon: string }[] = [
  { id: "project", label: "Explorer", icon: "☰" },
  { id: "graph", label: "Graph", icon: "⬡" },
  { id: "profile", label: "Profile", icon: "◷" },
  { id: "cache", label: "Cache", icon: "▣" },
  { id: "optimize", label: "Optimize", icon: "⚡" },
  { id: "experiments", label: "Experiments", icon: "◎" },
  { id: "benchmark", label: "Benchmark", icon: "▤" },
];

const FILES = [
  { path: "configs/cache.yaml", screen: "project" as Screen },
  { path: "results/report.md", screen: "experiments" as Screen },
  { path: "results/profile.txt", screen: "profile" as Screen },
  { path: "results/comparison.json", screen: "benchmark" as Screen },
  { path: "results/cache_stats.json", screen: "cache" as Screen },
  { path: "engine/cache/feature_cache.py", screen: "optimize" as Screen },
  { path: "examples/toy_diffusion/model.py", screen: "graph" as Screen },
];

function fmtSec(v: number) {
  return `${v.toFixed(4)}s`;
}

function fmtMiB(bytes: number) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

function Badge({ decision }: { decision: string }) {
  return <span className={`badge ${decision.toLowerCase()}`}>{decision}</span>;
}

export default function App() {
  const [screen, setScreen] = useState<Screen>("benchmark");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [agentOpen, setAgentOpen] = useState(true);
  const [bottomTab, setBottomTab] = useState<"terminal" | "output">("terminal");
  const [bottomOpen, setBottomOpen] = useState(true);
  const [comparison, setComparison] = useState<ComparisonPayload | null>(null);
  const [profile, setProfile] = useState<ProfilePayload | null>(null);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [report, setReport] = useState("");
  const [profileText, setProfileText] = useState("");
  const [apiOk, setApiOk] = useState(false);
  const [busy, setBusy] = useState(false);
  const [busyHint, setBusyHint] = useState("");
  const composerEndRef = useRef<HTMLDivElement | null>(null);
  const [terminal, setTerminal] = useState(
    "$ cdo — CPU Diffusion Optimizer\nConnecting to engine API…\n",
  );
  const [agentInput, setAgentInput] = useState("");
  const [agentLog, setAgentLog] = useState<ChatMsg[]>([
    {
      role: "assistant",
      text:
        "Composer ready.\n\nOptimization: `profile` · `run experiment` · `compare` · `cache`\nImages: `generate a cat image` · `draw a dog`\n\nImages appear below the reply in this panel.",
    },
  ]);
  const [openTabs, setOpenTabs] = useState<EditorTab[]>([
    { id: "benchmark", title: "comparison.json" },
  ]);

  const images = useMemo(() => resultsApi.imageUrls(), []);
  const activeTab = openTabs.find((t) => t.id === screen) ?? openTabs[0];

  useEffect(() => {
    composerEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [agentLog, busy]);

  function openScreen(next: Screen, title?: string) {
    setScreen(next);
    setOpenTabs((tabs) => {
      if (tabs.some((t) => t.id === next)) return tabs;
      return [...tabs, { id: next, title: title ?? next }];
    });
  }

  function closeTab(id: Screen | "file", e: MouseEvent) {
    e.stopPropagation();
    setOpenTabs((tabs) => {
      const next = tabs.filter((t) => t.id !== id);
      if (screen === id && next.length) setScreen(next[next.length - 1].id as Screen);
      return next.length ? next : [{ id: "project", title: "Welcome" }];
    });
  }

  async function refreshArtifacts() {
    const [c, p, cs, r, pt] = await Promise.all([
      resultsApi.comparison(),
      resultsApi.profile(),
      resultsApi.cacheStats(),
      resultsApi.report(),
      resultsApi.profileText(),
    ]);
    setComparison(c);
    setProfile(p);
    setCacheStats(cs);
    setReport(r ?? "");
    setProfileText(pt ?? "");
  }

  useEffect(() => {
    void (async () => {
      const ok = await apiHealth();
      setApiOk(ok);
      await refreshArtifacts();
      setTerminal(
        (t) =>
          t +
          (ok ? "✓ Engine API connected (/api)\n" : "✗ Engine API offline\n") +
          "✓ Loaded results/\n",
      );
    })();
  }, []);

  async function onAgentSubmit(e: FormEvent) {
    e.preventDefault();
    const msg = agentInput.trim();
    if (!msg || busy) return;
    setAgentInput("");
    setAgentOpen(true);
    setAgentLog((log) => [...log, { role: "user", text: msg }]);
    setBusy(true);
    const imageReq = looksLikeImageRequest(msg);
    setBusyHint(
      imageReq
        ? "Generating image… this can take 10–60 seconds"
        : "Running engine tool…",
    );
    setBottomTab("terminal");
    setBottomOpen(true);
    setTerminal((t) => t + `\n$ agent ${JSON.stringify(msg)}\n`);
    try {
      const result = await agentChat(msg);
      setAgentLog((log) => [
        ...log,
        {
          role: "assistant",
          text: result.reply,
          images: result.images,
          tool: result.tool,
        },
      ]);
      setTerminal(
        (t) =>
          t +
          `→ tool=${result.tool} ok=${result.ok}` +
          (result.images?.length ? ` images=${result.images.length}` : "") +
          "\n",
      );
      await refreshArtifacts();
      if (result.tool === "generate_image") {
        // stay on current view; image is in composer
      } else if (result.tool === "run_experiment" || result.tool === "compare_results") {
        openScreen("benchmark", "comparison.json");
      } else if (result.tool === "profile_model") {
        openScreen("profile", "profile.txt");
      } else if (result.tool === "analyze_cache") {
        openScreen("cache", "cache_stats.json");
      } else if (result.tool === "build_model_graph") {
        openScreen("graph", "model.graph");
      }
    } catch (err) {
      const text =
        err instanceof Error
          ? `${err.message}\n\nIf generating an image, wait up to 60s and retry.\nStart API: python -m engine.api`
          : "Agent failed.";
      setAgentLog((log) => [...log, { role: "assistant", text }]);
      setTerminal((t) => t + "→ error\n");
      setApiOk(false);
    } finally {
      setBusy(false);
      setBusyHint("");
    }
  }

  return (
    <div
      className="ide"
      style={
        {
          "--sidebar-w": sidebarOpen ? "240px" : "0px",
          "--agent-w": agentOpen ? "360px" : "0px",
          "--bottom-h": bottomOpen ? "180px" : "28px",
        } as CSSProperties
      }
    >
      {/* Title bar */}
      <header className="titlebar">
        <div className="traffic">
          <span /><span /><span />
        </div>
        <div className="titlebar-center">CPU Diffusion Optimizer — workspace</div>
        <div className="titlebar-right">
          <span className={`pill ${apiOk ? "ok" : "bad"}`}>
            {apiOk ? "API · connected" : "API · offline"}
          </span>
        </div>
      </header>

      <div className="ide-body">
        {/* Activity bar */}
        <nav className="activity">
          {ACTIVITY.map((a) => (
            <button
              key={a.id}
              type="button"
              title={a.label}
              className={screen === a.id ? "active" : ""}
              onClick={() => {
                setSidebarOpen(true);
                openScreen(a.id, a.label);
              }}
            >
              <span aria-hidden>{a.icon}</span>
            </button>
          ))}
          <div className="activity-spacer" />
          <button
            type="button"
            title="Toggle AI Agent"
            className={agentOpen ? "active" : ""}
            onClick={() => setAgentOpen((v) => !v)}
          >
            <span aria-hidden>AI</span>
          </button>
        </nav>

        {/* Sidebar explorer */}
        {sidebarOpen && (
          <aside className="sidebar">
            <div className="sidebar-head">
              <span>EXPLORER</span>
              <button type="button" className="icon-btn" onClick={() => setSidebarOpen(false)}>
                ×
              </button>
            </div>
            <div className="sidebar-section">OPEN EDITORS</div>
            {openTabs.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`tree-item ${screen === t.id ? "active" : ""}`}
                onClick={() => setScreen(t.id as Screen)}
              >
                {t.title}
              </button>
            ))}
            <div className="sidebar-section">CPU-DIFFUSION-OPTIMIZER</div>
            <div className="tree-folder">engine/</div>
            <div className="tree-folder">cli/</div>
            <div className="tree-folder">ui/</div>
            {FILES.map((f) => (
              <button
                key={f.path}
                type="button"
                className={`tree-item file ${screen === f.screen ? "active" : ""}`}
                onClick={() => openScreen(f.screen, f.path.split("/").pop())}
              >
                {f.path}
              </button>
            ))}          </aside>
        )}

        {/* Editor + bottom */}
        <div className="editor-col">
          <div className="tabs">
            {openTabs.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`tab ${screen === t.id ? "active" : ""}`}
                onClick={() => setScreen(t.id as Screen)}
              >
                <span>{t.title}</span>
                <span
                  className="tab-close"
                  onClick={(e) => closeTab(t.id, e)}
                  role="presentation"
                >
                  ×
                </span>
              </button>
            ))}
          </div>
          <div className="breadcrumb">
            {activeTab?.title ?? "editor"}
            {comparison && screen === "benchmark" && (
              <>
                {" · "}
                <Badge decision={comparison.decision} />
                {" · "}
                {comparison.speedup.toFixed(3)}x
              </>
            )}
          </div>
          <div className="editor">
            {renderEditor({
              screen,
              comparison,
              profile,
              cacheStats,
              report,
              profileText,
              images,
            })}
          </div>

          {/* Bottom panel */}
          <div className={`bottom ${bottomOpen ? "open" : "collapsed"}`}>
            <div className="bottom-tabs">
              <button
                type="button"
                className={bottomTab === "terminal" ? "active" : ""}
                onClick={() => {
                  setBottomTab("terminal");
                  setBottomOpen(true);
                }}
              >
                TERMINAL
              </button>
              <button
                type="button"
                className={bottomTab === "output" ? "active" : ""}
                onClick={() => {
                  setBottomTab("output");
                  setBottomOpen(true);
                }}
              >
                OUTPUT
              </button>
              <button
                type="button"
                className="icon-btn bottom-toggle"
                onClick={() => setBottomOpen((v) => !v)}
              >
                {bottomOpen ? "▾" : "▴"}
              </button>
            </div>
            {bottomOpen && (
              <pre className="bottom-body">
                {bottomTab === "terminal"
                  ? terminal
                  : profileText || report || "(no output yet — run profile / experiment)"}
              </pre>
            )}
          </div>
        </div>

        {/* AI Agent / Composer */}
        {agentOpen && (
          <aside className="composer">
            <div className="composer-head">
              <div>
                <strong>AI Optimization Agent</strong>
                <div className="composer-sub">Composer · engine tools</div>
              </div>
              <button type="button" className="icon-btn" onClick={() => setAgentOpen(false)}>
                ×
              </button>
            </div>
            <div className="composer-log">
              {agentLog.map((m, i) => (
                <div key={i} className={`bubble ${m.role}`}>
                  <div className="bubble-meta">
                    {m.role === "user" ? "You" : "Agent"}
                    {m.tool ? ` · ${m.tool}` : ""}
                  </div>
                  <div className="bubble-text">{m.text}</div>
                  {m.images && m.images.length > 0 && (
                    <div className="agent-images">
                      {m.images.map((src) => (
                        <a key={src} href={src} target="_blank" rel="noreferrer">
                          <img src={src} alt="generated" />
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {busy && (
                <div className="bubble assistant">
                  <div className="bubble-meta">Agent · working</div>
                  <div className="bubble-text">{busyHint || "Working…"}</div>
                </div>
              )}
              <div ref={composerEndRef} />
            </div>
            <form className="composer-input" onSubmit={onAgentSubmit}>
              <textarea
                value={agentInput}
                onChange={(e) => setAgentInput(e.target.value)}
                placeholder='Try: generate a cat image'
                rows={3}
                disabled={busy}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void onAgentSubmit(e as unknown as FormEvent);
                  }
                }}
              />
              <div className="composer-actions">
                <span className="hint">
                  {busy ? busyHint || "Working…" : "Enter send · Shift+Enter newline"}
                </span>
                <button type="submit" disabled={busy || !agentInput.trim()}>
                  {busy ? "…" : "Send"}
                </button>
              </div>
            </form>
          </aside>
        )}
      </div>

      <footer className="statusbar">
        <span>{apiOk ? "● main" : "● disconnected"}</span>
        <span>CPU · UTF-8</span>
        <span>{comparison ? `speedup ${comparison.speedup.toFixed(2)}x` : "no run"}</span>
        <span className="grow" />
        <button type="button" onClick={() => setAgentOpen(true)}>
          Agent
        </button>
        <button type="button" onClick={() => setBottomOpen(true)}>
          Terminal
        </button>
      </footer>
    </div>
  );
}

function renderEditor(args: {
  screen: Screen;
  comparison: ComparisonPayload | null;
  profile: ProfilePayload | null;
  cacheStats: CacheStats | null;
  report: string;
  profileText: string;
  images: { baseline: string; optimized: string };
}) {
  const { screen, comparison, profile, cacheStats, report, profileText, images } = args;

  if (screen === "project") {
    return (
      <div className="doc">
        <h1>Welcome</h1>
        <p className="muted">
          Cursor-style workspace for CPU diffusion optimization. Use the left explorer, edit/view
          artifacts in the center, chat with the agent on the right, and watch the terminal below.
        </p>
        <div className="code-block">
          <div className="code-label">configs/cache.yaml</div>
          <pre>{`experiment:
  name: toy_cpu_aware_cache
cache:
  mode: cpu_aware
  threshold: 0.05
optimization:
  enable_cache: true`}</pre>
        </div>
      </div>
    );
  }

  if (screen === "graph") {
    return (
      <div className="doc">
        <h1>Model Graph</h1>
        <pre className="code-block graph">{`model
└─ dit
   ├─ timestep_embedding
   ├─ block_0 … block_N
   │   ├─ attention → qkv
   │   ├─ mlp
   │   └─ norm1 / norm2
   └─ latent_op`}</pre>
      </div>
    );
  }

  if (screen === "profile") {
    return (
      <div className="doc">
        <h1>CPU Profile</h1>
        {!profile ? (
          <p className="muted">No profile yet. Ask the agent: profile</p>
        ) : (
          <>
            <div className="metrics">
              <div>
                <span>Total</span>
                <strong>{fmtSec(profile.total_latency_sec)}</strong>
              </div>
              <div>
                <span>Peak RAM</span>
                <strong>{fmtMiB(profile.peak_ram_bytes)}</strong>
              </div>
              <div>
                <span>Steps</span>
                <strong>{profile.steps}</strong>
              </div>
            </div>
            <h2>Hotspots</h2>
            {(profile.hotspots ?? []).slice(0, 8).map((h) => (
              <div key={h.name} className="hotspot">
                <div className="hotspot-label">
                  <span>{h.name}</span>
                  <span>{h.share_pct.toFixed(1)}%</span>
                </div>
                <div className="bar">
                  <span style={{ width: `${Math.min(h.share_pct, 100)}%` }} />
                </div>
              </div>
            ))}
            <pre className="code-block">{profileText}</pre>
          </>
        )}
      </div>
    );
  }

  if (screen === "cache") {
    return (
      <div className="doc">
        <h1>Cache Analysis</h1>
        {!cacheStats ? (
          <p className="muted">No cache stats. Ask: run experiment</p>
        ) : (
          <div className="metrics">
            <div>
              <span>Hit rate</span>
              <strong>{(cacheStats.hit_rate * 100).toFixed(1)}%</strong>
            </div>
            <div>
              <span>Hits</span>
              <strong>{cacheStats.hits}</strong>
            </div>
            <div>
              <span>Misses</span>
              <strong>{cacheStats.misses}</strong>
            </div>
          </div>
        )}
      </div>
    );
  }

  if (screen === "optimize") {
    return (
      <div className="doc">
        <h1>Optimization Suggestions</h1>
        <ul>
          <li>CPU-aware adaptive feature cache (middle DiT blocks)</li>
          <li>INT8 linear layers (Phase 4)</li>
          <li>Reduce host tensor copies / improve layout</li>
        </ul>
        {profile?.hotspots?.[0] && (
          <p>
            Top hotspot: <code>{profile.hotspots[0].name}</code> (
            {profile.hotspots[0].share_pct.toFixed(1)}%)
          </p>
        )}
      </div>
    );
  }

  if (screen === "experiments" || screen === "benchmark" || screen === "agent") {
    return (
      <div className="doc">
        <h1>Benchmark Comparison</h1>
        {!comparison ? (
          <p className="muted">No results. Ask the agent: run experiment</p>
        ) : (
          <>
            <p>
              Decision <Badge decision={comparison.decision} />
            </p>
            <div className="metrics">
              <div>
                <span>Baseline</span>
                <strong>{fmtSec(comparison.baseline.total_latency_sec)}</strong>
              </div>
              <div>
                <span>Optimized</span>
                <strong>{fmtSec(comparison.optimized.total_latency_sec)}</strong>
              </div>
              <div>
                <span>Speedup</span>
                <strong>{comparison.speedup.toFixed(3)}x</strong>
              </div>
              <div>
                <span>MSE</span>
                <strong>{comparison.quality.mse.toExponential(2)}</strong>
              </div>
            </div>
            <div className="images">
              <figure>
                <img src={images.baseline} alt="baseline" />
                <figcaption>baseline.png</figcaption>
              </figure>
              <figure>
                <img src={images.optimized} alt="optimized" />
                <figcaption>optimized.png</figcaption>
              </figure>
            </div>
            <pre className="code-block">{report}</pre>
          </>
        )}
      </div>
    );
  }

  return null;
}
