"""Simple component registry for adapters and optimizers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from engine.core.exceptions import EngineError

T = TypeVar("T")


class Registry:
    """Name → factory registry used by model adapters and optimizers."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        key = name.lower()
        if key in self._entries:
            raise EngineError(f"{self.kind} '{name}' is already registered")
        self._entries[key] = factory

    def get(self, name: str) -> Callable[..., Any]:
        key = name.lower()
        if key not in self._entries:
            known = ", ".join(sorted(self._entries)) or "(none)"
            raise EngineError(f"Unknown {self.kind} '{name}'. Known: {known}")
        return self._entries[key]

    def names(self) -> list[str]:
        return sorted(self._entries)


MODEL_ADAPTERS = Registry("model_adapter")
OPTIMIZERS = Registry("optimizer")
