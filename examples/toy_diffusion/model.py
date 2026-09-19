"""Tiny deterministic toy diffusion model for CPU tests and Phase 1 demos.

Architecture (intentionally small):

    latent ──► DiT-like stack of residual blocks ──► noise prediction
                 ├─ Attention (cheap linear attention)
                 └─ MLP

No external weights are required. Designed to run quickly on CPU with
reproducible outputs for a fixed seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ToyDiffusionSpec:
    channels: int = 4
    height: int = 16
    width: int = 16
    hidden_dim: int = 32
    num_blocks: int = 4
    num_heads: int = 4


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.linear_1.in_features // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.linear_1.in_features:
            emb = F.pad(emb, (0, self.linear_1.in_features - emb.shape[-1]))
        return self.linear_2(F.silu(self.linear_1(emb)))


class ToyAttention(nn.Module):
    """Simple multi-head self-attention over flattened spatial tokens."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class ToyMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class ToyTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ToyAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ToyMLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ToyDiT(nn.Module):
    def __init__(self, spec: ToyDiffusionSpec) -> None:
        super().__init__()
        self.spec = spec
        self.input_proj = nn.Conv2d(spec.channels, spec.hidden_dim, kernel_size=1)
        self.time_embed = TimestepEmbedding(spec.hidden_dim)
        self.blocks = nn.ModuleList(
            [ToyTransformerBlock(spec.hidden_dim, spec.num_heads) for _ in range(spec.num_blocks)]
        )
        self.norm_out = nn.LayerNorm(spec.hidden_dim)
        self.output_proj = nn.Conv2d(spec.hidden_dim, spec.channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        h_map = self.input_proj(x)
        tokens = h_map.flatten(2).transpose(1, 2)  # (B, N, C)
        t_emb = self.time_embed(timesteps).unsqueeze(1)
        tokens = tokens + t_emb
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm_out(tokens)
        h_map = tokens.transpose(1, 2).reshape(b, -1, h, w)
        return self.output_proj(h_map)


class ToyDiffusionModel(nn.Module):
    """Minimal diffusion sampler with a DiT-like denoiser."""

    def __init__(self, spec: ToyDiffusionSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or ToyDiffusionSpec()
        self.denoiser = ToyDiT(self.spec)

    @property
    def num_blocks(self) -> int:
        return self.spec.num_blocks

    def predict_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.denoiser(x, t)

    @torch.inference_mode()
    def sample(
        self,
        *,
        num_steps: int,
        batch_size: int = 1,
        seed: int = 0,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """DDPM-like iterative denoising on CPU (simplified schedule)."""
        device = next(self.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        shape = (batch_size, self.spec.channels, self.spec.height, self.spec.width)
        x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        intermediates: list[torch.Tensor] = []

        for step in range(num_steps):
            # Timestep decreases from nearly T to 0
            t_val = float(num_steps - step) / float(num_steps)
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
            noise_pred = self.predict_noise(x, t)
            # Simple Euler-like update toward lower noise
            x = x - (1.0 / num_steps) * noise_pred
            if return_intermediates:
                intermediates.append(x.detach().clone())

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {"output": x}
        if return_intermediates:
            result["intermediates"] = intermediates
        return result


def build_toy_model(
    *,
    channels: int = 4,
    height: int = 16,
    width: int = 16,
    hidden_dim: int = 32,
    num_blocks: int = 4,
    seed: int = 0,
) -> ToyDiffusionModel:
    torch.manual_seed(seed)
    spec = ToyDiffusionSpec(
        channels=channels,
        height=height,
        width=width,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=max(1, hidden_dim // 8),
    )
    model = ToyDiffusionModel(spec)
    model.eval()
    return model.to("cpu")
