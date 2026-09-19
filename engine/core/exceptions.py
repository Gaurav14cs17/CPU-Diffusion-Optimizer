"""Engine exceptions."""

from __future__ import annotations


class EngineError(Exception):
    """Base error for the CPU Diffusion Optimizer engine."""


class ModelLoadError(EngineError):
    """Raised when a model cannot be loaded or adapted."""


class ProfileError(EngineError):
    """Raised when profiling fails."""


class BenchmarkError(EngineError):
    """Raised when a benchmark run fails."""


class ExperimentError(EngineError):
    """Raised when an experiment cannot be completed."""


class OptimizationRejected(EngineError):
    """Raised when an optimization fails quality or speed gates."""


class ConfigError(EngineError):
    """Raised when configuration is invalid."""
