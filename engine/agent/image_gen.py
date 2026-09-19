"""Text → image generation for the agent chat.

Primary: HTTP image API (no local GPU required).
Fallback: toy diffusion latent visualization seeded from the prompt.
"""

from __future__ import annotations

import hashlib
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_image_prompt(message: str) -> str:
    """Pull a usable prompt out of casual chat text."""
    text = message.strip()
    # Common prefixes
    patterns = [
        r"^(please\s+)?(can you\s+)?(generate|genrate|generat|create|draw|make|paint|render|show)\s+(me\s+)?(an?\s+)?(image|picture|photo|png)?\s*(of\s+|for\s+|about\s+)?",
        r"^(i want\s+(you to\s+)?(generate|genrate|create|draw|make)\s+(an?\s+)?(image|picture)?\s*(of\s+)?)?",
        r"^(an?\s+)?(image|picture|photo)\s+(of\s+)?",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^(an?|the)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip(" .,!?:;\"'")
    cleaned = re.sub(r"\s+(image|picture|photo|png)$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or text


def is_image_generation_request(message: str) -> bool:
    text = message.lower().strip()
    verbs = (
        "generate",
        "genrate",  # common typo
        "generat",
        "create",
        "draw",
        "make",
        "paint",
        "render",
        "imagine",
        "txt2img",
        "text to image",
    )
    nouns = ("image", "picture", "photo", "png", "artwork", "illustration")
    subjects = (
        "dog",
        "cat",
        "animal",
        "bird",
        "car",
        "landscape",
        "person",
        "robot",
        "flower",
        "tree",
        "house",
        "sunset",
        "mountain",
    )
    has_verb = any(v in text for v in verbs)
    has_noun = any(n in text for n in nouns)
    has_subject = any(s in text for s in subjects)
    if has_verb and (has_noun or has_subject):
        return True
    if has_noun and has_subject:
        return True
    # "generate a cat" / "draw dog"
    if has_verb and len(text.split()) <= 12:
        return True
    return False


def _slug(prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    words = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40]
    return f"{words or 'image'}_{digest}"


def _generate_via_pollinations(prompt: str, dest: Path, *, width: int = 512, height: int = 512) -> bool:
    """Fetch a generated image from the public Pollinations endpoint."""
    quoted = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{quoted}"
        f"?width={width}&height={height}&nologo=true&enhance=true"
    )
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "cpu-diffusion-optimizer/0.1"},
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
        if len(data) < 1000:
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Pollinations image gen failed: %s", exc)
        return False


def _generate_via_toy_diffusion(prompt: str, dest: Path) -> bool:
    """Deterministic toy-latent fallback when network image gen is unavailable."""
    try:
        from engine.benchmark.visualize import save_latent_image
        from examples.toy_diffusion.model import build_toy_model
    except Exception as exc:  # noqa: BLE001
        logger.warning("Toy fallback import failed: %s", exc)
        return False

    seed = int(hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8], 16) % (2**31)
    model = build_toy_model(height=32, width=32, hidden_dim=32, num_blocks=4, seed=seed)
    out = model.sample(num_steps=12, batch_size=1, seed=seed)
    tensor = out["output"]
    assert hasattr(tensor, "shape")
    save_latent_image(tensor, dest)  # type: ignore[arg-type]
    return True


def generate_image(prompt: str, output_dir: Path) -> dict[str, str | bool]:
    """Generate an image for ``prompt`` and write it under ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = extract_image_prompt(prompt)
    filename = f"gen_{_slug(clean)}.png"
    dest = output_dir / filename

    backend = "pollinations"
    ok = _generate_via_pollinations(clean, dest)
    if not ok:
        backend = "toy_diffusion_latent"
        ok = _generate_via_toy_diffusion(clean, dest)

    if not ok:
        return {
            "ok": False,
            "prompt": clean,
            "message": "Image generation failed (network and local fallback).",
        }

    rel = f"/results/images/{filename}"
    note = (
        f'Generated image for prompt: "{clean}"\n'
        f"Saved: results/images/{filename}\n"
        f"Backend: {backend}"
    )
    if backend == "toy_diffusion_latent":
        note += (
            "\n(Network image API unavailable — showing toy diffusion latent "
            "visualization seeded from your prompt.)"
        )
    return {
        "ok": True,
        "prompt": clean,
        "path": str(dest),
        "url": rel,
        "backend": backend,
        "message": note,
    }
