"""HTTP route handlers."""

from __future__ import annotations

from engine import __version__
from engine.agent.agent import OptimizationAgent
from engine.agent.txt2img_models import (
    get_active_key,
    get_default_key,
    list_models,
    set_active_key,
)
from engine.agent.image_gen import edit_image
from engine.api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    HealthResponse,
    Txt2ImgEditRequest,
    Txt2ImgEditResponse,
    Txt2ImgModelsResponse,
    Txt2ImgModelInfo,
    Txt2ImgSelectRequest,
)
from pathlib import Path

_agent = OptimizationAgent()


def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


def agent_chat(req: AgentChatRequest) -> AgentChatResponse:
    reply = _agent.chat(req.message, config_path=req.config_path)
    return AgentChatResponse(
        reply=reply.text,
        tool=reply.tool,
        images=reply.images,
        ok=bool(reply.data.get("ok", True)),
    )


def txt2img_models() -> Txt2ImgModelsResponse:
    models = [
        Txt2ImgModelInfo(
            key=m.key,
            name=m.name,
            path=m.path,
            steps=m.steps,
            size=m.size,
            notes=m.notes,
            kind=m.kind,
        )
        for m in list_models()
    ]
    return Txt2ImgModelsResponse(
        active=get_active_key(),
        default=get_default_key(),
        models=models,
    )


def txt2img_select(req: Txt2ImgSelectRequest) -> Txt2ImgModelsResponse:
    key = req.model_key.strip()
    if key.lower() in {"auto", "default"}:
        key = get_default_key()
    set_active_key(key)
    return txt2img_models()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_image_path(image_path: str) -> Path:
    """Resolve an upload/result image path safely under the project."""
    root = _project_root()
    raw = image_path.strip()
    if raw.startswith("/results/"):
        raw = raw[len("/results/") :]
    elif raw.startswith("results/"):
        raw = raw[len("results/") :]
    candidate = (root / "results" / raw).resolve()
    results_root = (root / "results").resolve()
    if not str(candidate).startswith(str(results_root)):
        raise ValueError("image_path must be under results/")
    return candidate


def txt2img_edit(req: Txt2ImgEditRequest) -> Txt2ImgEditResponse:
    try:
        source = resolve_image_path(req.image_path)
    except ValueError as exc:
        return Txt2ImgEditResponse(ok=False, message=str(exc))
    out_dir = _project_root() / "results" / "images"
    result = edit_image(
        req.prompt,
        source,
        out_dir,
        model_key=req.model_key,
        strength=req.strength,
    )
    url = str(result.get("url") or "") if result.get("ok") else None
    return Txt2ImgEditResponse(
        ok=bool(result.get("ok")),
        message=str(result.get("message") or ""),
        url=url,
        images=[url] if url else [],
        prompt=str(result.get("prompt")) if result.get("prompt") else None,
        backend=str(result.get("backend")) if result.get("backend") else None,
        model_key=str(result.get("model_key")) if result.get("model_key") else None,
        strength=float(result["strength"]) if result.get("strength") is not None else None,
    )
