"""Shared enums and lightweight type aliases for the engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

Shape = tuple[int, ...]
JSONDict = dict[str, Any]


class DeviceKind(str, Enum):
    """Execution device. CPU is the primary path; GPU is reserved for later."""

    CPU = "cpu"
    CUDA = "cuda"  # future
    MPS = "mps"  # future


class DTypeName(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"  # future


class NodeKind(str, Enum):
    """Normalized node categories in the internal model graph."""

    MODEL = "model"
    TEXT_ENCODER = "text_encoder"
    UNET = "unet"
    DIT = "dit"
    TRANSFORMER_BLOCK = "transformer_block"
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "normalization"
    CONV = "convolution"
    LINEAR = "linear"
    VAE = "vae"
    SCHEDULER = "scheduler"
    TIMESTEP_EMBEDDING = "timestep_embedding"
    CONDITIONING = "conditioning"
    LATENT_OP = "latent_op"
    RESIDUAL = "residual"
    OTHER = "other"


class CacheMode(str, Enum):
    DISABLED = "disabled"
    STATIC_BLOCK = "static_block"
    TIMESTEP = "timestep"
    ADAPTIVE = "adaptive"
    CPU_AWARE = "cpu_aware"


class OptimizationDecision(str, Enum):
    KEEP = "keep"
    REJECT = "reject"
    INCONCLUSIVE = "inconclusive"
