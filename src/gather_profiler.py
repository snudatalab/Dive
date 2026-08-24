"""Lightweight CUDA-event profiler for gather-decode bottleneck localization.

Default OFF. Enable by setting environment variable KVZIP_PROFILE_GATHER=1
*before* importing this module's consumers, OR by calling enable() at runtime.

Design goals:
- Zero overhead when disabled (constant-time bool check).
- No host<->device sync inside hot path (we use cudaEventRecord only).
- Sync once at report() time.
- No mutation of tensors / no logic change.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Tuple

import torch

_ENABLED: bool = os.environ.get("KVZIP_PROFILE_GATHER", "0") == "1"
# event_pairs: bucket -> list of (start_event, end_event)
_EVENTS: "dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]]" = defaultdict(list)
_COUNTS: "dict[str, int]" = defaultdict(int)


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def disable() -> None:
    global _ENABLED
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def reset() -> None:
    _EVENTS.clear()
    _COUNTS.clear()


class _Region:
    __slots__ = ("name", "start", "end")

    def __init__(self, name: str):
        self.name = name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end.record()
        _EVENTS[self.name].append((self.start, self.end))
        _COUNTS[self.name] += 1
        return False


class _NullRegion:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_NULL = _NullRegion()


def region(name: str):
    if not _ENABLED:
        return _NULL
    return _Region(name)


def report() -> str:
    if not _EVENTS:
        return "[gather_profiler] no data (disabled or not invoked)"
    torch.cuda.synchronize()
    rows = []
    for name, pairs in _EVENTS.items():
        total_ms = 0.0
        for s, e in pairs:
            total_ms += s.elapsed_time(e)
        n = _COUNTS[name]
        rows.append((name, total_ms, n, total_ms / max(n, 1)))
    rows.sort(key=lambda r: -r[1])
    lines = ["[gather_profiler] bucket totals (ms)  total / calls / mean_ms"]
    for name, total_ms, n, mean_ms in rows:
        lines.append(f"  {name:<24} {total_ms:10.3f}  {n:8d}  {mean_ms:8.4f}")
    return "\n".join(lines)
