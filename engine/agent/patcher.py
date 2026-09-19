"""Reversible patch apply / revert (Phase 5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Patch:
    path: str
    diff: str


def apply_patch(patch: Patch) -> None:
    raise NotImplementedError("apply_patch lands in Phase 5")


def revert_patch(patch: Patch) -> None:
    raise NotImplementedError("revert_patch lands in Phase 5")
