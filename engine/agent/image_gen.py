"""Text → image generation via offline CPU models (multi-model catalog).

Primary: Diffusers entries from ``configs/txt2img_models.yaml``.
Fallback: built-in ``toy_diffusion``.
No third-party online image APIs.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

from engine.agent.txt2img_models import (
    Txt2ImgModel,
    extract_model_hint,
    get_model,
    strip_model_hint,
)

logger = logging.getLogger(__name__)

# Cached loaded Diffusers pipelines (key = hf_id|size).
_DIFFUSERS_CACHE: dict[str, Any] = {}


def extract_image_prompt(message: str) -> str:
    """Pull a usable prompt out of casual chat text."""
    text = message.strip()
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
    cleaned = strip_model_hint(cleaned)
    return cleaned or text


def is_image_generation_request(message: str) -> bool:
    text = message.lower().strip()
    verbs = (
        "generate",
        "genrate",
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
    if has_verb and len(text.split()) <= 12:
        return True
    return False


def _slug(prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]
    words = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")[:40]
    return f"{words or 'image'}_{digest}"


def _prompt_seed(prompt: str) -> int:
    return int(hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8], 16) % (2**31)


def _resolve_model(model_key: str | None) -> Txt2ImgModel:
    return get_model(model_key)


def _get_loaded_diffusers(model: Txt2ImgModel) -> Any:
    from engine.core.config import ModelConfig
    from engine.model.loader import load_model

    model_id = model.hf_id
    size = max(64, model.size)
    cache_key = f"{model_id}|{size}"
    if cache_key not in _DIFFUSERS_CACHE:
        config = ModelConfig(
            name=model.key,
            kind="diffusers",
            path=model_id,
            height=size,
            width=size,
            num_steps=max(1, model.steps),
            batch_size=1,
        )
        logger.info("Loading CPU Diffusers model %s (first call may download)…", model_id)
        _DIFFUSERS_CACHE[cache_key] = load_model(config)
    return _DIFFUSERS_CACHE[cache_key]


def _generate_via_diffusers(prompt: str, dest: Path, model: Txt2ImgModel) -> tuple[bool, str]:
    try:
        loaded = _get_loaded_diffusers(model)
        model_id = model.hf_id
        steps = max(1, model.steps)
        size = max(64, model.size)
        seed = _prompt_seed(prompt)
        result = loaded.adapter.run_inference(
            loaded.module,
            num_steps=steps,
            batch_size=1,
            seed=seed,
            prompt=prompt,
            height=size,
            width=size,
            guidance_scale=model.guidance_scale,
        )
        pil = result.get("pil_image")
        if pil is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            pil.save(dest, format="PNG")
        else:
            from engine.benchmark.visualize import save_latent_image

            save_latent_image(result["output"], dest)
        return True, model_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("Diffusers CPU image gen failed (%s): %s", model.hf_id, exc)
        return False, model.hf_id


def _generate_via_toy(prompt: str, dest: Path, model: Txt2ImgModel | None = None) -> bool:
    try:
        from engine.benchmark.visualize import save_latent_image
        from engine.core.config import ModelConfig
        from engine.model.loader import load_model
    except Exception as exc:  # noqa: BLE001
        logger.warning("Toy pipeline import failed: %s", exc)
        return False

    seed = _prompt_seed(prompt)
    steps = model.steps if model and model.kind == "toy_diffusion" else 24
    config = ModelConfig(
        name="toy",
        kind="toy_diffusion",
        channels=4,
        height=32,
        width=32,
        hidden_dim=64,
        num_blocks=6,
        num_steps=steps,
        batch_size=1,
    )
    try:
        loaded = load_model(config)
        result = loaded.adapter.run_inference(
            loaded.module,
            num_steps=config.num_steps,
            batch_size=config.batch_size,
            seed=seed,
        )
        save_latent_image(result["output"], dest)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Toy pipeline image gen failed: %s", exc)
        return False


def generate_image(
    prompt: str,
    output_dir: Path,
    *,
    model_key: str | None = None,
) -> dict[str, str | bool]:
    """Generate an image for ``prompt`` offline on CPU using the selected model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hint = extract_model_hint(prompt)
    clean = extract_image_prompt(prompt)
    try:
        model = _resolve_model(model_key or hint)
    except KeyError as exc:
        return {"ok": False, "prompt": clean, "message": str(exc)}

    filename = f"gen_{_slug(clean)}_{model.key.replace('/', '_')}.png"
    dest = output_dir / filename

    if model.kind == "toy_diffusion":
        ok = _generate_via_toy(clean, dest, model)
        if ok:
            return {
                "ok": True,
                "prompt": clean,
                "path": str(dest),
                "url": f"/results/images/{filename}",
                "backend": "toy_diffusion_pipeline",
                "model_key": model.key,
                "message": (
                    f'Generated image for prompt: "{clean}"\n'
                    f"Saved: results/images/{filename}\n"
                    f"Backend: toy_diffusion (offline CPU)\n"
                    f"Model key: {model.key}"
                ),
            }
        return {"ok": False, "prompt": clean, "message": "Toy diffusion generation failed."}

    ok, model_id = _generate_via_diffusers(clean, dest, model)
    if ok:
        backend = f"diffusers:{model_id}"
        return {
            "ok": True,
            "prompt": clean,
            "path": str(dest),
            "url": f"/results/images/{filename}",
            "backend": backend,
            "model_key": model.key,
            "message": (
                f'Generated image for prompt: "{clean}"\n'
                f"Saved: results/images/{filename}\n"
                f"Backend: {backend} (offline CPU)\n"
                f"Catalog key: {model.key} · steps={model.steps} · size={model.size}\n"
                f"{model.notes}"
            ),
        }

    # Fall back to toy if Diffusers fails
    ok = _generate_via_toy(clean, dest)
    if ok:
        return {
            "ok": True,
            "prompt": clean,
            "path": str(dest),
            "url": f"/results/images/{filename}",
            "backend": "toy_diffusion_pipeline",
            "model_key": "toy",
            "message": (
                f'Generated image for prompt: "{clean}"\n'
                f"Saved: results/images/{filename}\n"
                f"Backend: toy_diffusion (fallback after {model.key} failed)\n"
                'Install/check: pip install -e ".[diffusers]" · `list models`'
            ),
        }

    return {
        "ok": False,
        "prompt": clean,
        "message": f"Image generation failed for model '{model.key}'.",
    }


def edit_image(
    prompt: str,
    source_path: Path,
    output_dir: Path,
    *,
    model_key: str | None = None,
    strength: float = 0.55,
) -> dict[str, str | bool | float]:
    """Text-guided img2img edit of an uploaded/source image (offline CPU)."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    clean = extract_image_prompt(prompt)
    if not source_path.is_file():
        return {"ok": False, "prompt": clean, "message": f"Source image not found: {source_path}"}

    try:
        model = _resolve_model(model_key)
    except KeyError as exc:
        return {"ok": False, "prompt": clean, "message": str(exc)}

    if model.kind == "toy_diffusion":
        return {
            "ok": False,
            "prompt": clean,
            "message": "Img2img needs a Diffusers model. Use `use model sd-turbo` (not toy).",
        }

    filename = f"edit_{_slug(clean)}_{model.key.replace('/', '_')}.png"
    dest = output_dir / filename
    try:
        loaded = _get_loaded_diffusers(model)
        init = Image.open(source_path).convert("RGB")
        # Prefer adapter method when available
        adapter = loaded.adapter
        if not hasattr(adapter, "run_img2img"):
            return {"ok": False, "prompt": clean, "message": "Adapter has no img2img support."}
        result = adapter.run_img2img(
            loaded.module,
            prompt=clean,
            init_image=init,
            num_steps=max(1, model.steps),
            strength=strength,
            seed=_prompt_seed(clean + str(source_path)),
            guidance_scale=model.guidance_scale,
            height=max(64, model.size),
            width=max(64, model.size),
        )
        pil = result["pil_image"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        pil.save(dest, format="PNG")
        backend = f"diffusers-img2img:{model.hf_id}"
        return {
            "ok": True,
            "prompt": clean,
            "path": str(dest),
            "url": f"/results/images/{filename}",
            "backend": backend,
            "model_key": model.key,
            "strength": float(strength),
            "source": str(source_path),
            "message": (
                f'Text-guided edit: "{clean}"\n'
                f"Source: {source_path.name}\n"
                f"Saved: results/images/{filename}\n"
                f"Backend: {backend}\n"
                f"strength={strength:.2f} · steps={model.steps} · model={model.key}"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Img2img failed: %s", exc)
        return {"ok": False, "prompt": clean, "message": f"Img2img failed: {exc}"}
