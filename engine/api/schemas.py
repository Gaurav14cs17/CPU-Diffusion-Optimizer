"""API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class AgentChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    config_path: str | None = Field(default="configs/cache.yaml")


class AgentChatResponse(BaseModel):
    reply: str
    tool: str
    images: list[str] = Field(default_factory=list)
    ok: bool = True


class ProjectLoadRequest(BaseModel):
    path: str


class ModelLoadRequest(BaseModel):
    kind: str = "toy_diffusion"
    name: str = "toy"
    path: str | None = None
    num_steps: int = 8
    height: int = 16
    width: int = 16


class ProfileRequest(BaseModel):
    model_id: str
    seed: int = 42


class OptimizeRequest(BaseModel):
    model_id: str
    enable_cache: bool = False
    enable_int8: bool = False


class BenchmarkRequest(BaseModel):
    model_id: str
    seed: int = 42


class PatchRequest(BaseModel):
    path: str
    diff: str = Field(default="")


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    decision: str


class Txt2ImgModelInfo(BaseModel):
    key: str
    name: str
    path: str | None = None
    steps: int = 2
    size: int = 512
    notes: str = ""
    kind: str = "diffusers"


class Txt2ImgModelsResponse(BaseModel):
    active: str
    default: str
    models: list[Txt2ImgModelInfo] = Field(default_factory=list)


class Txt2ImgSelectRequest(BaseModel):
    model_key: str


class Txt2ImgEditRequest(BaseModel):
    prompt: str
    image_path: str = Field(description="Path under results/uploads or results/images")
    model_key: str | None = None
    strength: float = Field(default=0.55, ge=0.05, le=1.0)


class Txt2ImgEditResponse(BaseModel):
    ok: bool
    message: str
    url: str | None = None
    images: list[str] = Field(default_factory=list)
    prompt: str | None = None
    backend: str | None = None
    model_key: str | None = None
    strength: float | None = None
