"""Model loading facade."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from engine.core.config import ModelConfig
from engine.core.exceptions import ModelLoadError
from engine.core.registry import MODEL_ADAPTERS
from engine.model.adapters.base import ModelAdapter
from engine.model.graph import ModelGraph
from engine.model.model_info import ModelInfo

# Ensure built-in adapters register themselves.
from engine.model.adapters import toy as _toy  # noqa: F401


@dataclass
class LoadedModel:
    config: ModelConfig
    module: nn.Module
    adapter: ModelAdapter
    graph: ModelGraph
    info: ModelInfo


def load_model(config: ModelConfig) -> LoadedModel:
    """Load a model via its registered adapter and build the graph."""
    try:
        adapter_cls = MODEL_ADAPTERS.get(config.kind)
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(str(exc)) from exc

    adapter: ModelAdapter
    if isinstance(adapter_cls, type):
        adapter = adapter_cls()
    else:
        adapter = adapter_cls()  # type: ignore[operator]

    try:
        module = adapter.load(config)
        module = adapter.to_cpu(module)
        graph = adapter.build_graph(module, config)
        info = adapter.info(module, config, graph)
    except ModelLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ModelLoadError(f"Failed to load model '{config.name}': {exc}") from exc

    return LoadedModel(config=config, module=module, adapter=adapter, graph=graph, info=info)
