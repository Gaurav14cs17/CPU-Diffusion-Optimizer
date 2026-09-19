"""JSON helpers that never emit invalid Infinity/NaN tokens."""

from __future__ import annotations

import json
import math
from typing import Any


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(_sanitize(obj), **kwargs)
