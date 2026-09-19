"""Diffusers adapter stub (Phase 1: not yet wired for full SD).

Importing this module is safe without ``diffusers`` installed.
"""

from __future__ import annotations

from engine.core.exceptions import ModelLoadError


class DiffusersAdapter:
    """Reserved for Stable Diffusion / Diffusers pipelines (later phase)."""

    name = "diffusers"

    def load(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise ModelLoadError(
            "Diffusers adapter is not implemented in Phase 1. "
            "Use kind='toy_diffusion' or install Phase 4+ dependencies."
        )
