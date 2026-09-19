"""FastSDCPU adapter registration (no heavyweight model download in CI)."""

from __future__ import annotations

import pytest

from engine.core.config import ModelConfig
from engine.core.exceptions import ModelLoadError
from engine.core.registry import MODEL_ADAPTERS
from engine.model.loader import load_model


def test_fastsdcpu_adapter_registered() -> None:
    assert MODEL_ADAPTERS.get("fastsdcpu") is not None
    assert MODEL_ADAPTERS.get("openvino") is not None


def test_fastsdcpu_load_fails_cleanly_without_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing OpenVINO/Diffusers extras or bad id should raise ModelLoadError."""
    monkeypatch.setenv("CDO_FASTSDCPU_BACKEND", "pytorch")

    with pytest.raises(ModelLoadError):
        load_model(
            ModelConfig(
                name="missing",
                kind="fastsdcpu",
                path="pytorch:this-model-definitely-does-not-exist-cdo-test/xyz",
                height=512,
                width=512,
                num_steps=1,
            )
        )
