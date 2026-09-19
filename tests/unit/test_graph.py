"""Unit tests for model graph and toy model."""

from __future__ import annotations

from engine.core.config import ModelConfig
from engine.core.types import NodeKind
from engine.model.loader import load_model


def test_toy_model_loads_and_builds_graph() -> None:
    loaded = load_model(ModelConfig(name="toy", kind="toy_diffusion", num_blocks=3, num_steps=4))
    assert loaded.info.parameter_count > 0
    assert loaded.graph.get("model").kind == NodeKind.MODEL
    assert loaded.graph.get("model.dit").kind == NodeKind.DIT
    assert loaded.graph.get("model.dit.block_0.attention.qkv").kind == NodeKind.LINEAR


def test_graph_path_traversal() -> None:
    loaded = load_model(ModelConfig(kind="toy_diffusion", num_blocks=2))
    path = loaded.graph.path("model.dit.block_0.attention.qkv")
    names = [n.name for n in path]
    assert names[0] == "model"
    assert names[-1] == "model.dit.block_0.attention.qkv"


def test_toy_inference_deterministic() -> None:
    loaded = load_model(ModelConfig(kind="toy_diffusion", num_steps=3, height=8, width=8, hidden_dim=16))
    a = loaded.adapter.run_inference(loaded.module, num_steps=3, batch_size=1, seed=7)
    b = loaded.adapter.run_inference(loaded.module, num_steps=3, batch_size=1, seed=7)
    assert a["output"].shape == b["output"].shape
    assert (a["output"] - b["output"]).abs().max().item() == 0.0
