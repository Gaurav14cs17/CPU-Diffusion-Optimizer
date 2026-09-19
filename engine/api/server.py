"""FastAPI server — UI talks to the engine only through this API."""

from __future__ import annotations

from pathlib import Path

from engine.api.routes import agent_chat, health
from engine.api.schemas import AgentChatRequest


def create_app():  # type: ignore[no-untyped-def]
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:  # pragma: no cover
        raise ImportError("API extras not installed. Run: pip install -e '.[api]'") from exc

    app = FastAPI(
        title="CPU Diffusion Optimizer",
        version="0.1.0",
        description="Engine API for CPU-first diffusion optimization",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    root = Path(__file__).resolve().parents[2]
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    app.mount("/results", StaticFiles(directory=str(results)), name="results")

    @app.get("/health")
    def _health():  # type: ignore[no-untyped-def]
        return health()

    @app.post("/agent/chat")
    def _agent_chat(body: AgentChatRequest):  # type: ignore[no-untyped-def]
        return agent_chat(body)

    @app.get("/experiments")
    def _experiments():  # type: ignore[no-untyped-def]
        comparison = results / "comparison.json"
        if not comparison.is_file():
            return []
        return [{"id": "latest", "path": str(comparison)}]

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        "engine.api.server:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
