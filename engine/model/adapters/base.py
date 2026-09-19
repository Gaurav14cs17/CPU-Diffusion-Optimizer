"""Model adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from engine.core.config import ModelConfig
from engine.model.graph import ModelGraph
from engine.model.model_info import ModelInfo


class ModelAdapter(ABC):
    """Adapter that wraps a concrete framework model into engine primitives."""

    name: str

    @abstractmethod
    def load(self, config: ModelConfig) -> nn.Module:
        """Load / construct the underlying module (CPU by default)."""

    @abstractmethod
    def build_graph(self, model: nn.Module, config: ModelConfig) -> ModelGraph:
        """Build the normalized computation graph."""

    @abstractmethod
    def info(self, model: nn.Module, config: ModelConfig, graph: ModelGraph) -> ModelInfo:
        """Return high-level model metadata."""

    @abstractmethod
    def run_inference(
        self,
        model: nn.Module,
        *,
        num_steps: int,
        batch_size: int,
        seed: int,
        collect_block_outputs: bool = False,
    ) -> dict[str, Any]:
        """Run a full diffusion sampling loop on CPU.

        Returns a dict containing at least ``output`` (torch.Tensor) and
        optionally ``per_step_ms``, ``block_timings_ms``, etc.
        """

    def to_cpu(self, model: nn.Module) -> nn.Module:
        model = model.to("cpu")
        model.eval()
        return model

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
