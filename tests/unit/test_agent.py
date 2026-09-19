"""Agent intent → tool routing tests."""

from __future__ import annotations

from pathlib import Path

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


def test_plan_list_models() -> None:
    assert plan_from_text("list models").tool == "list_txt2img_models"


def test_plan_use_model() -> None:
    plan = plan_from_text("use model sdxs")
    assert plan.tool == "set_txt2img_model"
    assert plan.kwargs["model_key"] == "sdxs"


def test_plan_generate_with_model() -> None:
    plan = plan_from_text("generate a cat with sdxs")
    assert plan.tool == "generate_image"
    assert plan.kwargs.get("model_key") == "sdxs"


def test_catalog_has_many_models() -> None:
    from engine.agent.txt2img_models import list_models

    keys = {m.key for m in list_models()}
    assert "sd-turbo" in keys
    assert "sdxs" in keys
    assert "toy" in keys
    assert len(keys) >= 5


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


def test_generate_image_offline_pipeline(tmp_path) -> None:
    from engine.agent.image_gen import generate_image

    result = generate_image("a red cube", tmp_path)
    assert result["ok"] is True
    assert "backend" in result
    # Diffusers if installed, else toy fallback — both offline
    assert "diffusers:" in str(result["backend"]) or result["backend"] == "toy_diffusion_pipeline"
    path = Path(str(result["path"]))
    assert path.is_file()
    assert path.stat().st_size > 100
