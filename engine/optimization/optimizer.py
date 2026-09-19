"""Optimizer orchestrator (Phase 4 implementations plug in here)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from engine.core.config import EngineConfig
from engine.model.graph import ModelGraph


@dataclass
class OptimizationPlan:
    actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Optimizer(ABC):
    @abstractmethod
    def plan(self, graph: ModelGraph, config: EngineConfig) -> OptimizationPlan:
        ...

    @abstractmethod
    def optimize(self, model: nn.Module, plan: OptimizationPlan) -> nn.Module:
        ...


class NullOptimizer(Optimizer):
    """Phase 1 no-op optimizer — returns the model unchanged."""

    def plan(self, graph: ModelGraph, config: EngineConfig) -> OptimizationPlan:
        actions: list[str] = []
        if config.optimization.enable_cache:
            actions.append("adaptive_feature_cache")
        if config.optimization.enable_int8:
            actions.append("int8_linear")
        if config.optimization.enable_block_skip:
            actions.append("block_skip")
        if not actions:
            actions.append("identity")
        return OptimizationPlan(actions=actions, metadata={"phase": 1})

    def optimize(self, model: nn.Module, plan: OptimizationPlan) -> nn.Module:
        return model
