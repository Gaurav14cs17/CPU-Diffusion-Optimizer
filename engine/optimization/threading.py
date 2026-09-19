"""Threading configuration for CPU inference."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class ThreadingConfig:
    intra_op_threads: int | None = None
    inter_op_threads: int | None = None
    omp_num_threads: int | None = None


def apply_threading(config: ThreadingConfig) -> dict[str, int]:
    applied: dict[str, int] = {}
    if config.intra_op_threads is not None:
        torch.set_num_threads(config.intra_op_threads)
        applied["intra_op_threads"] = config.intra_op_threads
    if config.inter_op_threads is not None:
        torch.set_num_interop_threads(config.inter_op_threads)
        applied["inter_op_threads"] = config.inter_op_threads
    if config.omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(config.omp_num_threads)
        applied["omp_num_threads"] = config.omp_num_threads
    return applied
