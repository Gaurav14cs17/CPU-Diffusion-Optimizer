"""HTTP route handlers."""

from __future__ import annotations

from engine import __version__
from engine.agent.agent import OptimizationAgent
from engine.api.schemas import AgentChatRequest, AgentChatResponse, HealthResponse

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
