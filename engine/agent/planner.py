"""Intent routing from natural language → engine tools."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from engine.agent import tools as toolmod
from engine.agent.image_gen import extract_image_prompt, is_image_generation_request
from engine.agent.txt2img_models import extract_model_hint


@dataclass
class Plan:
    tool: str
    reason: str
    kwargs: dict[str, Any] = field(default_factory=dict)


_USE_MODEL = re.compile(
    r"^(?:use|set|switch(?:\s+to)?)\s+model\s+(.+)$",
    re.IGNORECASE,
)


def plan_from_text(message: str) -> Plan:
    """Map user text to a single engine tool (deterministic, no LLM required)."""
    text = message.lower().strip()

    if not text or text in {"help", "?", "hi", "hello"}:
        return Plan("help", "greeting_or_help")

    if any(k in text for k in ("list models", "show models", "txt2img models", "image models")):
        return Plan("list_txt2img_models", "user_requested_model_list")

    use = _USE_MODEL.match(message.strip())
    if use:
        return Plan("set_txt2img_model", "user_selected_model", {"model_key": use.group(1).strip()})

    if is_image_generation_request(message):
        kwargs: dict[str, Any] = {"prompt": extract_image_prompt(message)}
        hint = extract_model_hint(message)
        if hint:
            kwargs["model_key"] = hint
        return Plan("generate_image", "user_requested_image", kwargs)

    if any(k in text for k in ("experiment", "optimize", "optimisation", "full pipeline", "run cache")):
        return Plan("run_experiment", "user_requested_experiment")

    if any(k in text for k in ("profile", "hotspot", "latency", "slow")):
        return Plan("profile_model", "user_requested_profile")

    if any(k in text for k in ("benchmark", "compare", "speedup", "results", "report", "decision")):
        return Plan("compare_results", "user_requested_comparison")

    if any(k in text for k in ("cache", "hit rate", "feature cache", "block cache")):
        return Plan("analyze_cache", "user_requested_cache")

    if any(k in text for k in ("graph", "unet", "dit", "blocks", "attention")) or (
        "model" in text and "use model" not in text and "list model" not in text
    ):
        return Plan("build_model_graph", "user_requested_graph")

    if any(k in text for k in ("cpu", "hardware", "avx", "neon", "ram", "memory", "cores")):
        return Plan("inspect_cpu", "user_requested_cpu")

    if any(k in text for k in ("list images", "show images", "latent")):
        return Plan("list_images", "user_requested_images")

    if "image" in text or "png" in text:
        return Plan("list_images", "user_requested_images")

    if "test" in text:
        return Plan("run_benchmark", "user_requested_quick_benchmark")

    results = toolmod.tool_read_results()
    if results.get("ok"):
        return Plan("compare_results", "default_show_results")
    return Plan("help", "default_help")
