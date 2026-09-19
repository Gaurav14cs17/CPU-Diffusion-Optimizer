"""PyTorch adapter helpers (generic modules)."""

from __future__ import annotations

from engine.core.registry import MODEL_ADAPTERS
from engine.model.adapters.base import ModelAdapter

# Placeholder registration point for generic torch modules (Phase 1+).
# Concrete adapters register themselves on import.


def get_adapter(kind: str) -> type[ModelAdapter]:
    factory = MODEL_ADAPTERS.get(kind)
    if isinstance(factory, type) and issubclass(factory, ModelAdapter):
        return factory
    raise TypeError(f"Adapter factory for '{kind}' is not a ModelAdapter subclass")
