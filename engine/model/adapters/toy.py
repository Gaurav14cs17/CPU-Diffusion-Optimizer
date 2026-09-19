"""Adapter for the built-in toy diffusion model."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from engine.core.config import ModelConfig
from engine.core.registry import MODEL_ADAPTERS
from engine.core.types import DeviceKind, DTypeName, NodeKind
from engine.model.adapters.base import ModelAdapter, set_seed
from engine.model.graph import ModelGraph
from engine.model.model_info import ModelInfo
from engine.model.node import GraphNode
from examples.toy_diffusion.model import ToyDiffusionModel, build_toy_model


def _estimate_linear_flops(in_f: int, out_f: int, tokens: int) -> float:
    return float(2 * tokens * in_f * out_f)


class ToyDiffusionAdapter(ModelAdapter):
    name = "toy_diffusion"

    def load(self, config: ModelConfig) -> nn.Module:
        set_seed(0)
        return build_toy_model(
            channels=config.channels,
            height=config.height,
            width=config.width,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            seed=0,
        )

    def build_graph(self, model: nn.Module, config: ModelConfig) -> ModelGraph:
        assert isinstance(model, ToyDiffusionModel)
        graph = ModelGraph(model_name=config.name, root="model")
        tokens = config.height * config.width
        hidden = config.hidden_dim

        graph.add_node(
            GraphNode(
                name="model",
                kind=NodeKind.MODEL,
                input_shapes=[(config.batch_size, config.channels, config.height, config.width)],
                output_shapes=[(config.batch_size, config.channels, config.height, config.width)],
                parameter_count=self.count_parameters(model),
                metadata={"num_steps": config.num_steps},
            )
        )
        graph.add_node(
            GraphNode(
                name="model.dit",
                kind=NodeKind.DIT,
                input_shapes=[(config.batch_size, config.channels, config.height, config.width)],
                output_shapes=[(config.batch_size, config.channels, config.height, config.width)],
                parameter_count=self.count_parameters(model.denoiser),
                dependencies=["model"],
            )
        )
        graph.add_edge("model", "model.dit")

        graph.add_node(
            GraphNode(
                name="model.dit.timestep_embedding",
                kind=NodeKind.TIMESTEP_EMBEDDING,
                input_shapes=[(config.batch_size,)],
                output_shapes=[(config.batch_size, hidden)],
                parameter_count=self.count_parameters(model.denoiser.time_embed),
                estimated_flops=float(4 * hidden * hidden),
                dependencies=["model.dit"],
            )
        )
        graph.add_edge("model.dit", "model.dit.timestep_embedding")

        prev = "model.dit"
        for idx, block in enumerate(model.denoiser.blocks):
            block_name = f"model.dit.block_{idx}"
            graph.add_node(
                GraphNode(
                    name=block_name,
                    kind=NodeKind.TRANSFORMER_BLOCK,
                    input_shapes=[(config.batch_size, tokens, hidden)],
                    output_shapes=[(config.batch_size, tokens, hidden)],
                    parameter_count=self.count_parameters(block),
                    estimated_flops=_estimate_linear_flops(hidden, hidden * 3, tokens)
                    + _estimate_linear_flops(hidden, hidden, tokens)
                    + _estimate_linear_flops(hidden, hidden * 4, tokens)
                    + _estimate_linear_flops(hidden * 4, hidden, tokens),
                    dependencies=[prev],
                )
            )
            graph.add_edge(prev, block_name)

            attn_name = f"{block_name}.attention"
            graph.add_node(
                GraphNode(
                    name=attn_name,
                    kind=NodeKind.ATTENTION,
                    input_shapes=[(config.batch_size, tokens, hidden)],
                    output_shapes=[(config.batch_size, tokens, hidden)],
                    parameter_count=self.count_parameters(block.attn),
                    estimated_flops=_estimate_linear_flops(hidden, hidden * 3, tokens)
                    + _estimate_linear_flops(hidden, hidden, tokens),
                    dependencies=[block_name],
                )
            )
            graph.add_edge(block_name, attn_name)

            qkv_name = f"{attn_name}.qkv"
            graph.add_node(
                GraphNode(
                    name=qkv_name,
                    kind=NodeKind.LINEAR,
                    input_shapes=[(config.batch_size, tokens, hidden)],
                    output_shapes=[(config.batch_size, tokens, hidden * 3)],
                    parameter_count=self.count_parameters(block.attn.qkv),
                    estimated_flops=_estimate_linear_flops(hidden, hidden * 3, tokens),
                    dependencies=[attn_name],
                )
            )
            graph.add_edge(attn_name, qkv_name)

            mlp_name = f"{block_name}.mlp"
            graph.add_node(
                GraphNode(
                    name=mlp_name,
                    kind=NodeKind.MLP,
                    input_shapes=[(config.batch_size, tokens, hidden)],
                    output_shapes=[(config.batch_size, tokens, hidden)],
                    parameter_count=self.count_parameters(block.mlp),
                    estimated_flops=_estimate_linear_flops(hidden, hidden * 4, tokens)
                    + _estimate_linear_flops(hidden * 4, hidden, tokens),
                    dependencies=[block_name],
                )
            )
            graph.add_edge(block_name, mlp_name)

            for norm_suffix, module in (("norm1", block.norm1), ("norm2", block.norm2)):
                norm_name = f"{block_name}.{norm_suffix}"
                graph.add_node(
                    GraphNode(
                        name=norm_name,
                        kind=NodeKind.NORM,
                        input_shapes=[(config.batch_size, tokens, hidden)],
                        output_shapes=[(config.batch_size, tokens, hidden)],
                        parameter_count=self.count_parameters(module),
                        dependencies=[block_name],
                    )
                )
                graph.add_edge(block_name, norm_name)

            prev = block_name

        graph.add_node(
            GraphNode(
                name="model.latent_op",
                kind=NodeKind.LATENT_OP,
                input_shapes=[(config.batch_size, config.channels, config.height, config.width)],
                output_shapes=[(config.batch_size, config.channels, config.height, config.width)],
                parameter_count=0,
                dependencies=[prev],
                metadata={"op": "euler_denoise_step"},
            )
        )
        graph.add_edge(prev, "model.latent_op")
        return graph

    def info(self, model: nn.Module, config: ModelConfig, graph: ModelGraph) -> ModelInfo:
        assert isinstance(model, ToyDiffusionModel)
        return ModelInfo(
            name=config.name,
            kind=self.name,
            device=DeviceKind.CPU,
            dtype=DTypeName.FP32,
            parameter_count=self.count_parameters(model),
            num_blocks=model.num_blocks,
            latent_shape=(config.batch_size, config.channels, config.height, config.width),
            components=["dit", "timestep_embedding", "latent_op"],
            metadata={"spec": model.spec.__dict__},
            graph=graph,
        )

    def run_inference(
        self,
        model: nn.Module,
        *,
        num_steps: int,
        batch_size: int,
        seed: int,
        collect_block_outputs: bool = False,
        cache_manager: Any | None = None,
    ) -> dict[str, Any]:
        assert isinstance(model, ToyDiffusionModel)
        model = self.to_cpu(model)

        if cache_manager is not None and getattr(cache_manager, "enabled", False):
            from engine.cache.runtime import run_cached_sample

            result = run_cached_sample(
                model,
                cache_manager,
                num_steps=num_steps,
                batch_size=batch_size,
                seed=seed,
            )
            if collect_block_outputs:
                result["block_outputs"] = {}
            return result

        block_timings_ms: dict[str, float] = {f"block_{i}": 0.0 for i in range(model.num_blocks)}
        per_step_ms: list[float] = []

        handles = []
        for idx, block in enumerate(model.denoiser.blocks):

            def _pre(_module: nn.Module, _inputs: tuple[Any, ...], *, _idx: int = idx) -> None:
                block_timings_ms[f"_start_{_idx}"] = time.perf_counter()

            def _post(
                _module: nn.Module,
                _inputs: tuple[Any, ...],
                _output: Any,
                *,
                _idx: int = idx,
            ) -> None:
                start = block_timings_ms.pop(f"_start_{_idx}", time.perf_counter())
                block_timings_ms[f"block_{_idx}"] += (time.perf_counter() - start) * 1000.0

            handles.append(block.register_forward_pre_hook(_pre))
            handles.append(block.register_forward_hook(_post))

        device = next(model.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        shape = (
            batch_size,
            model.spec.channels,
            model.spec.height,
            model.spec.width,
        )
        x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

        t0 = time.perf_counter()
        with torch.inference_mode():
            for step in range(num_steps):
                step_t0 = time.perf_counter()
                t_val = float(num_steps - step) / float(num_steps)
                t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
                noise_pred = model.predict_noise(x, t)
                x = x - (1.0 / num_steps) * noise_pred
                per_step_ms.append((time.perf_counter() - step_t0) * 1000.0)
        total_ms = (time.perf_counter() - t0) * 1000.0

        for handle in handles:
            handle.remove()

        clean_blocks = {
            k: v for k, v in block_timings_ms.items() if k.startswith("block_") and not k.startswith("_")
        }

        result: dict[str, Any] = {
            "output": x.detach().cpu(),
            "total_ms": total_ms,
            "per_step_ms": per_step_ms,
            "block_timings_ms": clean_blocks,
            "cache_hit_rate": 0.0,
            "cache_memory_bytes": 0,
        }
        if collect_block_outputs:
            result["block_outputs"] = {}
        return result


MODEL_ADAPTERS.register("toy_diffusion", ToyDiffusionAdapter)
MODEL_ADAPTERS.register("toy", ToyDiffusionAdapter)
