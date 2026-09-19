"""Output quality metrics (CPU-side, no external vision models required)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from engine.core.config import QualityConfig
from engine.core.types import JSONDict


@dataclass
class QualityResult:
    mse: float
    psnr: float
    ssim: float | None
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> JSONDict:
        return {
            "mse": self.mse,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "passed": self.passed,
            "details": dict(self.details),
        }


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def pixel_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for MSE: {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = _to_numpy(a) - _to_numpy(b)
    return float(np.mean(diff**2))


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: float | None = None) -> float:
    mse = pixel_mse(a, b)
    if mse <= 0.0:
        return float("inf")
    if data_range is None:
        data_range = float(max(np.max(np.abs(_to_numpy(a))), np.max(np.abs(_to_numpy(b))), 1e-8))
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim_simple(a: torch.Tensor, b: torch.Tensor) -> float:
    """Lightweight structural similarity on flattened tensors (approx)."""
    x = _to_numpy(a).astype(np.float64).ravel()
    y = _to_numpy(b).astype(np.float64).ravel()
    c1 = 1e-4
    c2 = 9e-4
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(num / den) if den != 0 else 0.0


class QualityValidator:
    """Compare baseline vs optimized tensors against configured thresholds."""

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()

    def evaluate(self, baseline: torch.Tensor, optimized: torch.Tensor) -> QualityResult:
        metrics_wanted = {m.lower() for m in self.config.metrics}
        mse = pixel_mse(baseline, optimized)
        psnr_v = psnr(baseline, optimized) if "psnr" in metrics_wanted else float("nan")
        ssim_v = ssim_simple(baseline, optimized) if "ssim" in metrics_wanted else None

        passed = True
        details: dict[str, Any] = {
            "max_mse": self.config.max_mse,
            "min_psnr": self.config.min_psnr,
        }
        if "mse" in metrics_wanted and mse > self.config.max_mse:
            passed = False
            details["mse_violation"] = True
        if "psnr" in metrics_wanted and np.isfinite(psnr_v) and psnr_v < self.config.min_psnr:
            passed = False
            details["psnr_violation"] = True

        return QualityResult(
            mse=mse,
            psnr=psnr_v,
            ssim=ssim_v,
            passed=passed,
            details=details,
        )
