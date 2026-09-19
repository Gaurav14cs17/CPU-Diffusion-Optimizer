"""Catalog of CPU text-to-image models + active selection."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Txt2ImgModel:
    key: str
    name: str
    kind: str
    path: str | None
    steps: int
    size: int
    guidance_scale: float | None
    notes: str = ""

    @property
    def hf_id(self) -> str:
        return self.path or self.key


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _catalog_path() -> Path:
    env = os.environ.get("CDO_TXT2IMG_CATALOG")
    if env:
        return Path(env)
    return _project_root() / "configs" / "txt2img_models.yaml"


def _state_path() -> Path:
    return _project_root() / "results" / "txt2img_active.json"


_CATALOG: dict[str, Txt2ImgModel] | None = None
_DEFAULT_KEY: str = "sd-turbo"
_ACTIVE_KEY: str | None = None


def _load_catalog() -> dict[str, Txt2ImgModel]:
    global _CATALOG, _DEFAULT_KEY
    if _CATALOG is not None:
        return _CATALOG

    path = _catalog_path()
    raw: dict[str, Any] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    models_raw = raw.get("models") or {}
    catalog: dict[str, Txt2ImgModel] = {}
    for key, meta in models_raw.items():
        if not isinstance(meta, dict):
            continue
        catalog[str(key).lower()] = Txt2ImgModel(
            key=str(key).lower(),
            name=str(meta.get("name") or key),
            kind=str(meta.get("kind") or "diffusers"),
            path=(str(meta["path"]) if meta.get("path") else None),
            steps=int(meta.get("steps") or 2),
            size=int(meta.get("size") or 512),
            guidance_scale=(
                float(meta["guidance_scale"])
                if meta.get("guidance_scale") is not None
                else None
            ),
            notes=str(meta.get("notes") or ""),
        )
    if not catalog:
        catalog = {
            "sd-turbo": Txt2ImgModel(
                key="sd-turbo",
                name="SD Turbo",
                kind="diffusers",
                path="stabilityai/sd-turbo",
                steps=2,
                size=512,
                guidance_scale=0.0,
                notes="Built-in fallback catalog entry.",
            ),
            "toy": Txt2ImgModel(
                key="toy",
                name="Toy diffusion",
                kind="toy_diffusion",
                path=None,
                steps=24,
                size=32,
                guidance_scale=None,
                notes="Built-in offline latent model.",
            ),
        }
    _DEFAULT_KEY = str(raw.get("default") or next(iter(catalog))).lower()
    if _DEFAULT_KEY not in catalog:
        _DEFAULT_KEY = next(iter(catalog))
    _CATALOG = catalog
    return catalog


def list_models() -> list[Txt2ImgModel]:
    return list(_load_catalog().values())


def get_model(key: str | None = None) -> Txt2ImgModel:
    catalog = _load_catalog()
    k = (key or get_active_key()).lower().strip()
    # Allow raw HF ids not in catalog
    if k in catalog:
        return catalog[k]
    if "/" in k:
        steps = int(os.environ.get("CDO_TXT2IMG_STEPS", "2"))
        size = int(os.environ.get("CDO_TXT2IMG_SIZE", "512"))
        return Txt2ImgModel(
            key=k,
            name=k,
            kind="diffusers",
            path=k,
            steps=steps,
            size=size,
            guidance_scale=0.0,
            notes="Custom Hugging Face model id.",
        )
    # fuzzy: match name substring
    for m in catalog.values():
        if k in m.key or k in m.name.lower() or (m.path and k in m.path.lower()):
            return m
    raise KeyError(f"Unknown txt2img model '{key}'. Try: {', '.join(sorted(catalog))}")


def get_default_key() -> str:
    _load_catalog()
    return _DEFAULT_KEY


def get_active_key() -> str:
    global _ACTIVE_KEY
    env = os.environ.get("CDO_TXT2IMG_MODEL", "").strip()
    if env:
        return env.lower()
    if _ACTIVE_KEY:
        return _ACTIVE_KEY
    state = _state_path()
    if state.is_file():
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
            key = str(data.get("active") or "").lower()
            if key:
                _ACTIVE_KEY = key
                return key
        except (OSError, json.JSONDecodeError):
            pass
    return _load_catalog() and _DEFAULT_KEY


def set_active_key(key: str) -> Txt2ImgModel:
    global _ACTIVE_KEY
    model = get_model(key)
    # Store catalog key when possible, else hf id
    catalog = _load_catalog()
    store = model.key if model.key in catalog else (model.path or model.key)
    _ACTIVE_KEY = store
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"active": store}, indent=2), encoding="utf-8")
    logger.info("Active txt2img model → %s (%s)", store, model.path or model.kind)
    return model


def format_model_list(active: str | None = None) -> str:
    active = (active or get_active_key()).lower()
    lines = ["CPU text-to-image models:", ""]
    for m in list_models():
        mark = "→" if m.key == active or (m.path and m.path.lower() == active) else " "
        lines.append(
            f"{mark} **{m.key}** — {m.name}\n"
            f"    path={m.path or '(builtin)'}  steps={m.steps}  size={m.size}\n"
            f"    {m.notes}"
        )
    lines.append("")
    lines.append(f"Active: **{active}**")
    lines.append('Switch: `use model sd-turbo` · Generate: `generate a cat with sdxs`')
    return "\n".join(lines)


_MODEL_HINT = re.compile(
    r"\b(?:with|using|via|on)\s+(?:model\s+)?([a-z0-9._\-]+(?:/[a-z0-9._\-]+)?)\b",
    re.IGNORECASE,
)


def extract_model_hint(message: str) -> str | None:
    """Parse optional model from 'generate a cat with sdxs' / 'using sd-turbo'."""
    catalog = _load_catalog()
    text = message.strip()
    # Prefer explicit "with/using …"
    m = _MODEL_HINT.search(text)
    if m:
        token = m.group(1).lower()
        if token in catalog or "/" in token:
            return token
        for key, model in catalog.items():
            if token in key or token in model.name.lower().replace(" ", ""):
                return key
    # Trailing bare catalog key: "generate a cat sdxs"
    parts = re.split(r"\s+", text.lower())
    if parts and parts[-1] in catalog:
        return parts[-1]
    return None


def strip_model_hint(prompt: str) -> str:
    """Remove 'with sdxs' / 'using sd-turbo' clutter from the image prompt."""
    cleaned = _MODEL_HINT.sub("", prompt).strip()
    catalog = _load_catalog()
    parts = cleaned.split()
    if parts and parts[-1].lower() in catalog:
        parts = parts[:-1]
    return " ".join(parts).strip(" ,.-") or prompt
