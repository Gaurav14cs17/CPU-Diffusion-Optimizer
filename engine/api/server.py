"""FastAPI server — UI talks to the engine only through this API."""

import hashlib
import time
from pathlib import Path

from engine.api.routes import agent_chat, health, txt2img_edit, txt2img_models, txt2img_select
from engine.api.schemas import AgentChatRequest, Txt2ImgEditRequest, Txt2ImgSelectRequest


def create_app():  # type: ignore[no-untyped-def]
    try:
        from fastapi import FastAPI, File, UploadFile
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
    (results / "uploads").mkdir(parents=True, exist_ok=True)
    (results / "images").mkdir(parents=True, exist_ok=True)
    app.mount("/results", StaticFiles(directory=str(results)), name="results")

    @app.get("/health")
    def _health():  # type: ignore[no-untyped-def]
        return health()

    @app.post("/agent/chat")
    def _agent_chat(body: AgentChatRequest):  # type: ignore[no-untyped-def]
        return agent_chat(body)

    @app.get("/txt2img/models")
    def _txt2img_models():  # type: ignore[no-untyped-def]
        return txt2img_models()

    @app.post("/txt2img/models")
    def _txt2img_select(body: Txt2ImgSelectRequest):  # type: ignore[no-untyped-def]
        return txt2img_select(body)

    @app.post("/txt2img/upload")
    async def _txt2img_upload(file: UploadFile = File(...)):  # type: ignore[no-untyped-def]
        """Accept an image, normalize to RGB PNG under results/uploads/."""
        from io import BytesIO

        from PIL import Image, ImageOps

        uploads = results / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        raw = await file.read()
        if not raw:
            return {"ok": False, "message": "Empty file upload."}
        if len(raw) > 25 * 1024 * 1024:
            return {"ok": False, "message": "Image too large (max 25 MB)."}
        try:
            img = Image.open(BytesIO(raw))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"Could not read image: {exc}"}

        # Cap very large photos for CPU img2img stability
        max_side = 1536
        w, h = img.size
        scale = min(1.0, max_side / float(max(w, h)))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)

        digest = hashlib.sha1(f"{time.time()}-{file.filename}-{len(raw)}".encode()).hexdigest()[:10]
        name = f"upload_{digest}.png"
        dest = uploads / name
        img.save(dest, format="PNG", optimize=True)
        return {
            "ok": True,
            "url": f"/results/uploads/{name}",
            "path": f"uploads/{name}",
            "filename": name,
            "width": img.size[0],
            "height": img.size[1],
        }

    @app.post("/txt2img/edit")
    def _txt2img_edit(body: Txt2ImgEditRequest):  # type: ignore[no-untyped-def]
        return txt2img_edit(body)

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
