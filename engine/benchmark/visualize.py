"""Save latent / tensor visualizations as PNG without extra dependencies."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import torch


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png_rgb(path: Path, rgb: np.ndarray) -> None:
    """Write HxWx3 uint8 array as a PNG file."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be HxWx3 uint8")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    png += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def latent_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Map a (C,H,W) or (1,C,H,W) latent to an RGB uint8 image for inspection."""
    t = tensor.detach().float().cpu()
    if t.ndim == 4:
        t = t[0]
    # Use first 3 channels, or broadcast if fewer
    if t.shape[0] >= 3:
        img = t[:3]
    else:
        img = t[:1].repeat(3, 1, 1)
    arr = img.permute(1, 2, 0).numpy()
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - lo) / (hi - lo)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def save_latent_image(tensor: torch.Tensor, path: str | Path) -> Path:
    out = Path(path)
    write_png_rgb(out, latent_to_rgb(tensor))
    return out
