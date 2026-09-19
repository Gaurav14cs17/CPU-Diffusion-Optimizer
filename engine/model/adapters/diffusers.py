"""Diffusers adapter for offline CPU text-to-image.

Default model: ``stabilityai/sd-turbo`` (1–4 steps, designed for fast sampling).
Requires optional deps: ``pip install -e ".[diffusers]"``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

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

# Good default for CPU: few-step turbo model (downloaded once, then offline).
DEFAULT_HF_MODEL = "stabilityai/sd-turbo"
DEFAULT_STEPS = 2
DEFAULT_SIZE = 512


class _DiffusersPipeModule(nn.Module):
    """Thin nn.Module wrapper so the engine loader can treat a pipeline as a module."""

    def __init__(self, pipe: Any, model_id: str) -> None:
        super().__init__()
        self.pipe = pipe
        self.model_id = model_id
        self.img2img_pipe: Any | None = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise RuntimeError("Use DiffusersAdapter.run_inference(), not forward().")

    def get_img2img_pipe(self) -> Any:
        if self.img2img_pipe is None:
            from diffusers import AutoPipelineForImage2Image

            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)
            self.img2img_pipe = self.img2img_pipe.to("cpu")
            if hasattr(self.img2img_pipe, "enable_attention_slicing"):
                self.img2img_pipe.enable_attention_slicing()
            if hasattr(self.img2img_pipe, "enable_vae_slicing"):
                self.img2img_pipe.enable_vae_slicing()
            self.img2img_pipe.set_progress_bar_config(disable=True)
        return self.img2img_pipe


class DiffusersAdapter(ModelAdapter):
    """Stable Diffusion / Diffusers pipelines on CPU."""

    name = "diffusers"

    def load(self, config: ModelConfig) -> nn.Module:
        try:
            from diffusers import AutoPipelineForText2Image
        except ImportError as exc:
            raise ModelLoadError(
                'diffusers is not installed. Run: pip install -e ".[diffusers]"'
            ) from exc

        model_id = (config.path or DEFAULT_HF_MODEL).strip()
        logger.info("Loading Diffusers model on CPU: %s", model_id)
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
        return _DiffusersPipeModule(pipe, model_id)

    def to_cpu(self, model: nn.Module) -> nn.Module:
        if isinstance(model, _DiffusersPipeModule):
            model.pipe = model.pipe.to("cpu")
            model.eval()
            return model
        return super().to_cpu(model)

    def build_graph(self, model: nn.Module, config: ModelConfig) -> ModelGraph:
        assert isinstance(model, _DiffusersPipeModule)
        graph = ModelGraph(model_name=config.name)
        h = config.height if config.height >= 64 else DEFAULT_SIZE
        w = config.width if config.width >= 64 else DEFAULT_SIZE
        graph.add_node(
            GraphNode(
                name="model",
                kind=NodeKind.MODEL,
                parameter_count=self.count_parameters(model),
                metadata={"hf_model": model.model_id, "op": "diffusers_pipeline"},
            )
        )
        graph.add_node(
            GraphNode(
                name="model.unet",
                kind=NodeKind.UNET,
                parameter_count=0,
                dependencies=["model"],
                metadata={"op": "unet"},
            )
        )
        graph.add_edge("model", "model.unet")
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
        assert isinstance(model, _DiffusersPipeModule)
        h = config.height if config.height >= 64 else DEFAULT_SIZE
        w = config.width if config.width >= 64 else DEFAULT_SIZE
        return ModelInfo(
            name=config.name,
            kind=self.name,
            device=DeviceKind.CPU,
            dtype=DTypeName.FP32,
            parameter_count=self.count_parameters(model),
            num_blocks=1,
            latent_shape=(config.batch_size, 4, h // 8, w // 8),
            components=["unet", "vae", "text_encoder"],
            metadata={"hf_model": model.model_id},
            graph=graph,
        )

    def count_parameters(self, model: nn.Module) -> int:  # type: ignore[override]
        if not isinstance(model, _DiffusersPipeModule):
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
        assert isinstance(model, _DiffusersPipeModule)
        _ = cache_manager
        set_seed(seed)
        pipe = model.pipe
        steps = max(1, num_steps)
        mid = (model.model_id or "").lower()
        if guidance_scale is None:
            guidance_scale = 0.0 if ("turbo" in mid or "lcm" in mid or "sdxs" in mid) else 7.5
        h = height or DEFAULT_SIZE
        w = width or DEFAULT_SIZE

        generator = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                num_images_per_prompt=max(1, batch_size),
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=h,
                width=w,
                generator=generator,
            )
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
        }
        if collect_block_outputs:
            out["block_outputs"] = {}
        return out

    def run_img2img(
        self,
        model: nn.Module,
        *,
        prompt: str,
        init_image: Any,
        num_steps: int = 2,
        strength: float = 0.55,
        seed: int = 42,
        guidance_scale: float | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> dict[str, Any]:
        """Text-guided image-to-image on CPU."""
        assert isinstance(model, _DiffusersPipeModule)
        from PIL import Image

        set_seed(seed)
        mid = (model.model_id or "").lower()
        if guidance_scale is None:
            guidance_scale = 0.0 if ("turbo" in mid or "lcm" in mid or "sdxs" in mid) else 7.5
        steps = max(1, num_steps)
        strength = float(min(1.0, max(0.05, strength)))

        if not isinstance(init_image, Image.Image):
            raise TypeError("init_image must be a PIL.Image")
        size = height or width or DEFAULT_SIZE
        image = init_image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)

        pipe = model.get_img2img_pipe()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.perf_counter()
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        pil_image = result.images[0]

        import numpy as np

        arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return {
            "output": tensor,
            "pil_image": pil_image,
            "total_ms": elapsed_ms,
            "prompt": prompt,
            "strength": strength,
            "model_id": model.model_id,
        }


MODEL_ADAPTERS.register("diffusers", DiffusersAdapter)
