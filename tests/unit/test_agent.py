"""Agent intent → tool routing tests."""

from __future__ import annotations

from engine.agent.agent import OptimizationAgent
from engine.agent.image_gen import extract_image_prompt, is_image_generation_request
from engine.agent.planner import plan_from_text


def test_plan_help() -> None:
    assert plan_from_text("help").tool == "help"


def test_plan_profile() -> None:
    assert plan_from_text("show cpu profile hotspots").tool == "profile_model"


def test_plan_experiment() -> None:
    assert plan_from_text("run experiment please").tool == "run_experiment"


def test_plan_compare() -> None:
    assert plan_from_text("what was the speedup?").tool == "compare_results"


def test_plan_generate_image() -> None:
    plan = plan_from_text("generate a cat image")
    assert plan.tool == "generate_image"
    assert "cat" in plan.kwargs["prompt"].lower()


def test_extract_prompt() -> None:
    assert "orange cat" in extract_image_prompt("please generate an image of an orange cat").lower()


def test_is_image_request() -> None:
    assert is_image_generation_request("draw a dog")
    assert not is_image_generation_request("run experiment")


def test_agent_help_chat() -> None:
    agent = OptimizationAgent()
    reply = agent.chat("help")
    assert reply.tool == "help"
    assert "profile" in reply.text.lower()
