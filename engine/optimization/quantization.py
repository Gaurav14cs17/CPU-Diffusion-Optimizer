"""CPU-oriented quantization abstractions (INT8 first; INT4 later)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from torch import nn

from engine.core.types import DTypeName


class QuantScheme(str, Enum):
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


class Quantizer(ABC):
    """Safe quantization interface — no fake quant that silently lies."""

    target: DTypeName

    @abstractmethod
    def supported(self) -> bool:
        ...

    @abstractmethod
    def quantize(self, model: nn.Module) -> nn.Module:
        ...


class Int8Quantizer(Quantizer):
    target = DTypeName.INT8

    def supported(self) -> bool:
        # Phase 4 will probe torch.ao.quantization / IPEX availability.
        return False

    def quantize(self, model: nn.Module) -> nn.Module:
        if not self.supported():
            raise NotImplementedError(
                "INT8 quantization requires Phase 4 backends (torch dynamo / IPEX)."
            )
        return model
