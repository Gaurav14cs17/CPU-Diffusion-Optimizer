"""Process memory sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass
class MemorySnapshot:
    rss_bytes: int
    peak_rss_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {"rss_bytes": self.rss_bytes, "peak_rss_bytes": self.peak_rss_bytes}


class MemoryProfiler:
    """Track resident set size around inference runs."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._peak = 0

    def sample(self) -> MemorySnapshot:
        rss = int(self._process.memory_info().rss)
        self._peak = max(self._peak, rss)
        return MemorySnapshot(rss_bytes=rss, peak_rss_bytes=self._peak)

    def reset_peak(self) -> None:
        self._peak = int(self._process.memory_info().rss)
