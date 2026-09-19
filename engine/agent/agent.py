"""Optimization agent — text in → engine tools → text (+ images) out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engine.agent.planner import plan_from_text
from engine.agent.tools import call_tool, list_tools


@dataclass
class AgentMessage:
    role: str
    content: str
    images: list[str] = field(default_factory=list)
    tool: str | None = None


@dataclass
class AgentSession:
    messages: list[AgentMessage] = field(default_factory=list)


@dataclass
class AgentReply:
    text: str
    tool: str
    images: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


class OptimizationAgent:
    """Every user message maps to an engine tool (including image generation)."""

    def __init__(self) -> None:
        self.session = AgentSession()

    def available_tools(self) -> list[str]:
        return list_tools()

    def chat(self, user_message: str, *, config_path: str | None = None) -> AgentReply:
        self.session.messages.append(AgentMessage(role="user", content=user_message))
        plan = plan_from_text(user_message)

        kwargs: dict[str, Any] = dict(plan.kwargs)
        if plan.tool in {
            "profile_model",
            "run_experiment",
            "run_benchmark",
            "build_model_graph",
        }:
            kwargs["config_path"] = config_path

        result = call_tool(plan.tool, **kwargs)
        text = str(result.get("message") or "")
        images = list(result.get("images") or [])

        if not result.get("ok", True) and not text:
            text = "Tool failed."

        reply = AgentReply(text=text, tool=plan.tool, images=images, data=result)
        self.session.messages.append(
            AgentMessage(
                role="assistant",
                content=reply.text,
                images=reply.images,
                tool=reply.tool,
            )
        )
        return reply
