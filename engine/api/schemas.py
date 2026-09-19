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
