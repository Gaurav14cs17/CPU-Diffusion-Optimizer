"""FastSDCPU-aligned adapter for CPU / OpenVINO text-to-image baselines.

This does **not** vendor [FastSDCPU](https://github.com/rupeshs/fastsdcpu). It loads the
same class of models that project uses (SD Turbo / LCM / OpenVINO exports) so we can
benchmark our optimization loop against a known-fast CPU SD path.

Backend selection (``ModelConfig.path`` or env ``CDO_FASTSDCPU_BACKEND``):

- ``openvino`` (default) — Optimum Intel / OpenVINO pipeline
- ``pytorch`` — Diffusers CPU (same models, slower; useful without OpenVINO)

Default OpenVINO weights: ``rupeshs/sd-turbo-openvino`` (FastSDCPU Turbo path).
Install: ``pip install -e ".[fastsdcpu]"``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Literal

import torch
from torch import nn

from engine.core.config import ModelConfig
from engine.core.exceptions import ModelLoadError
from engine.core.registry import MODEL_ADAPTERS
from engine.core.types import DeviceKind, DTypeName, NodeKind
from engine.model.adapters.base import ModelAdapter, set_seed
from engine.model.graph import ModelGraph
from engine.model.model_info import ModelInfo
from engine.model.node import GraphNode

logger = logging.getLogger(__name__)

# FastSDCPU-aligned defaults (Turbo OpenVINO + few-step sampling).
DEFAULT_OV_MODEL = "rupeshs/sd-turbo-openvino"
DEFAULT_PT_MODEL = "stabilityai/sd-turbo"
DEFAULT_STEPS = 1
DEFAULT_SIZE = 512
BackendName = Literal["openvino", "pytorch"]


def _resolve_backend(config: ModelConfig) -> BackendName:
    raw = (os.environ.get("CDO_FASTSDCPU_BACKEND") or "").strip().lower()
    # Allow ``path`` like ``openvino:rupeshs/...`` or plain HF id.
    path = (config.path or "").strip()
    if path.startswith("pytorch:") or path.startswith("pt:"):
        return "pytorch"
    if path.startswith("openvino:") or path.startswith("ov:"):
        return "openvino"
    if raw in ("pytorch", "pt", "torch"):
        return "pytorch"
    if raw in ("openvino", "ov"):
        return "openvino"
    # Prefer OpenVINO when the extra is importable.
    try:
        import optimum.intel  # noqa: F401

        return "openvino"
    except ImportError:
        return "pytorch"


def _strip_backend_prefix(path: str | None, backend: BackendName) -> str:
    if not path:
        return DEFAULT_OV_MODEL if backend == "openvino" else DEFAULT_PT_MODEL
    for prefix in ("openvino:", "ov:", "pytorch:", "pt:"):
        if path.startswith(prefix):
            rest = path[len(prefix) :].strip()
            return rest or (DEFAULT_OV_MODEL if backend == "openvino" else DEFAULT_PT_MODEL)
    return path


class _FastSDPipeModule(nn.Module):
    """Thin wrapper so the loader can treat an external pipeline as a module."""

    def __init__(self, pipe: Any, model_id: str, backend: BackendName) -> None:
        super().__init__()
        self.pipe = pipe
        self.model_id = model_id
        self.backend = backend

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise RuntimeError("Use FastSDCPUAdapter.run_inference(), not forward().")


class FastSDCPUAdapter(ModelAdapter):
    """Baseline adapter aligned with FastSDCPU model choices (OpenVINO or PyTorch)."""

    name = "fastsdcpu"

    def load(self, config: ModelConfig) -> nn.Module:
        backend = _resolve_backend(config)
        model_id = _strip_backend_prefix(config.path, backend)
        if backend == "openvino":
            return self._load_openvino(model_id)
        return self._load_pytorch(model_id)

    def _load_openvino(self, model_id: str) -> nn.Module:
        try:
            from optimum.intel import OVStableDiffusionPipeline
        except ImportError as exc:
            raise ModelLoadError(
                'OpenVINO backend requires extras. Run: pip install -e ".[fastsdcpu]" '
                '(or set CDO_FASTSDCPU_BACKEND=pytorch / path="pytorch:stabilityai/sd-turbo")'
            ) from exc

        device = (os.environ.get("DEVICE") or os.environ.get("CDO_OV_DEVICE") or "CPU").upper()
        logger.info("Loading FastSDCPU OpenVINO model on %s: %s", device, model_id)
        try:
            pipe = OVStableDiffusionPipeline.from_pretrained(model_id)
            if hasattr(pipe, "to"):
                pipe.to(device)
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load OpenVINO model '{model_id}': {exc}. "
                "Try a FastSDCPU export such as rupeshs/sd-turbo-openvino, or "
                'path="pytorch:stabilityai/sd-turbo".'
            ) from exc
        return _FastSDPipeModule(pipe, model_id, "openvino")

    def _load_pytorch(self, model_id: str) -> nn.Module:
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise ModelLoadError(
                'PyTorch FastSDCPU path needs diffusers. Run: pip install -e ".[diffusers]" '
                'or pip install -e ".[fastsdcpu]"'
            ) from exc

        logger.info("Loading FastSDCPU-aligned Diffusers model on CPU: %s", model_id)
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                dtype=torch.float32,
                safety_checker=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load Diffusers model '{model_id}': {exc}") from exc

        pipe = pipe.to("cpu")
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        pipe.set_progress_bar_config(disable=True)
        return _FastSDPipeModule(pipe, model_id, "pytorch")

    def to_cpu(self, model: nn.Module) -> nn.Module:
        if isinstance(model, _FastSDPipeModule):
            if model.backend == "pytorch" and hasattr(model.pipe, "to"):
                model.pipe = model.pipe.to("cpu")
            if hasattr(model.pipe, "eval"):
                model.pipe.eval()
            return model
        return super().to_cpu(model)

    def build_graph(self, model: nn.Module, config: ModelConfig) -> ModelGraph:
        assert isinstance(model, _FastSDPipeModule)
        graph = ModelGraph(model_name=config.name)
        h = config.height if config.height >= 64 else DEFAULT_SIZE
        w = config.width if config.width >= 64 else DEFAULT_SIZE
        graph.add_node(
            GraphNode(
                name="model",
                kind=NodeKind.MODEL,
                parameter_count=self.count_parameters(model),
                metadata={
                    "hf_model": model.model_id,
                    "backend": model.backend,
                    "op": "fastsdcpu_pipeline",
                    "aligned_with": "https://github.com/rupeshs/fastsdcpu",
                },
            )
        )
        graph.add_node(
            GraphNode(
                name="model.text_encoder",
                kind=NodeKind.TEXT_ENCODER,
                parameter_count=0,
                dependencies=["model"],
                metadata={"op": "text_encoder"},
            )
        )
        graph.add_edge("model", "model.text_encoder")
        graph.add_node(
            GraphNode(
                name="model.unet",
                kind=NodeKind.UNET,
                parameter_count=0,
                dependencies=["model.text_encoder"],
                metadata={"op": "unet"},
            )
        )
        graph.add_edge("model.text_encoder", "model.unet")
        graph.add_node(
            GraphNode(
                name="model.vae",
                kind=NodeKind.VAE,
                output_shapes=[(config.batch_size, 3, h, w)],
                parameter_count=0,
                dependencies=["model.unet"],
                metadata={"op": "vae_decode"},
            )
        )
        graph.add_edge("model.unet", "model.vae")
        return graph

    def info(self, model: nn.Module, config: ModelConfig, graph: ModelGraph) -> ModelInfo:
        assert isinstance(model, _FastSDPipeModule)
        h = config.height if config.height >= 64 else DEFAULT_SIZE
        w = config.width if config.width >= 64 else DEFAULT_SIZE
        dtype = DTypeName.INT8 if model.backend == "openvino" else DTypeName.FP32
        return ModelInfo(
            name=config.name,
            kind=self.name,
            device=DeviceKind.CPU,
            dtype=dtype,
            parameter_count=self.count_parameters(model),
            num_blocks=1,
            latent_shape=(config.batch_size, 4, h // 8, w // 8),
            components=["text_encoder", "unet", "vae"],
            metadata={
                "hf_model": model.model_id,
                "backend": model.backend,
                "aligned_with": "fastsdcpu",
            },
            graph=graph,
        )

    def count_parameters(self, model: nn.Module) -> int:  # type: ignore[override]
        if not isinstance(model, _FastSDPipeModule):
            return super().count_parameters(model)
        total = 0
        pipe = model.pipe
        for attr in ("unet", "vae", "text_encoder", "text_encoder_2"):
            mod = getattr(pipe, attr, None)
            if isinstance(mod, nn.Module):
                total += sum(p.numel() for p in mod.parameters())
        return total

    def run_inference(
        self,
        model: nn.Module,
        *,
        num_steps: int,
        batch_size: int,
        seed: int,
        collect_block_outputs: bool = False,
        cache_manager: Any | None = None,
        prompt: str = "a photo of a cat",
        height: int | None = None,
        width: int | None = None,
        guidance_scale: float | None = None,
    ) -> dict[str, Any]:
        assert isinstance(model, _FastSDPipeModule)
        # OpenVINO / Diffusers pipelines are opaque; block cache hooks land later.
        _ = cache_manager
        set_seed(seed)
        pipe = model.pipe
        steps = max(1, num_steps if num_steps > 0 else DEFAULT_STEPS)
        mid = (model.model_id or "").lower()
        if guidance_scale is None:
            # Turbo / LCM / SDXS / Hyper-SD: FastSDCPU-style low guidance.
            guidance_scale = (
                0.0
                if any(k in mid for k in ("turbo", "lcm", "sdxs", "hyper", "lightning"))
                else 7.5
            )
        h = height or DEFAULT_SIZE
        w = width or DEFAULT_SIZE

        generator = torch.Generator(device="cpu").manual_seed(seed)
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_images_per_prompt": max(1, batch_size),
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "height": h,
            "width": w,
        }
        # OpenVINO pipelines may not accept a torch Generator.
        if model.backend == "pytorch":
            call_kwargs["generator"] = generator

        t0 = time.perf_counter()
        with torch.inference_mode():
            result = pipe(**call_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        pil_image = result.images[0]

        import numpy as np

        arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        out: dict[str, Any] = {
            "output": tensor,
            "pil_image": pil_image,
            "per_step_ms": [elapsed_ms / steps] * steps,
            "block_timings_ms": {"pipeline": elapsed_ms},
            "total_ms": elapsed_ms,
            "prompt": prompt,
            "model_id": model.model_id,
            "backend": model.backend,
        }
        if collect_block_outputs:
            out["block_outputs"] = {}
        return out


MODEL_ADAPTERS.register("fastsdcpu", FastSDCPUAdapter)
MODEL_ADAPTERS.register("openvino", FastSDCPUAdapter)
