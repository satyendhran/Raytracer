"""Lightweight host-side debug timing helpers."""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

_ENABLED = False
_START = perf_counter()


def set_debug(enabled: bool) -> None:
    global _ENABLED, _START
    _ENABLED = bool(enabled)
    _START = perf_counter()


def enabled() -> bool:
    return _ENABLED


def now() -> float:
    return perf_counter()


def log(message: str) -> None:
    if _ENABLED:
        print(f"[debug +{perf_counter() - _START:8.3f}s] {message}", flush=True)


def completed(message: str) -> None:
    if _ENABLED:
        print(f"[debug +{perf_counter() - _START:8.3f}s] completed {message}", flush=True)


@contextmanager
def span(label: str):
    if not _ENABLED:
        yield
        return
    t0 = perf_counter()
    log(f"begin {label}")
    try:
        yield
    finally:
        completed(f"{label} ({perf_counter() - t0:.3f}s)")
