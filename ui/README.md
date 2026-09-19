# UI — CPU Diffusion Optimizer

Cursor-like developer console. Talks to the **engine API** only.

## Run UI + Agent API

```bash
# Terminal 1 — engine API (required for the agent)
source .venv/bin/activate
python -m engine.api

# Terminal 2 — UI
cd ui
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open http://localhost:5173

Type in the **AI Optimization Agent** panel:

| Text | Action |
|------|--------|
| `help` | List commands |
| `profile` | CPU profile toy model |
| `run experiment` | Full baseline vs cache experiment |
| `compare` / `results` | Show last speedup / decision |
| `cache` | Cache hit stats |
| `graph` | Model graph summary |
| `hardware` | CPU / RAM features |
| `images` | List result images |

Flow: `UI → POST /api/agent/chat → Engine tools` (no optimization logic in the UI).
