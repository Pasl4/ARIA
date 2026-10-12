"""Open benchmark registry.

register_benchmark is the extension point for your own adapters. Nothing is
built until a run asks for the name, so a factory that imports its heavy
dependencies (dataset clients, harness packages, docker) inside itself keeps
them from blocking other benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from .base import BenchmarkAdapter

# name -> "module:ClassName" (imported on first build) or a factory callable
# added through register_benchmark. No adapters ship with the framework, so
# both start empty.
_LAZY: dict[str, str] = {}
_FACTORIES: dict[str, Callable[..., BenchmarkAdapter]] = {}


def register_benchmark(name: str, factory: Callable[..., BenchmarkAdapter]) -> None:
    """Register a benchmark factory under `name`. Names are unique."""
    if name in _FACTORIES or name in _LAZY:
        raise ValueError(f"benchmark {name!r} is already registered")
    _FACTORIES[name] = factory


def available_benchmarks() -> tuple[str, ...]:
    return tuple(sorted({*_LAZY, *_FACTORIES}))


def build_benchmark(name: str, **options: Any) -> BenchmarkAdapter:
    """Instantiate the adapter registered as `name`, passing `options` to its factory."""
    if name in _FACTORIES:
        return _FACTORIES[name](**options)
    try:
        target = _LAZY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown benchmark {name!r}; expected one of {available_benchmarks()}"
        ) from exc
    module_name, class_name = target.split(":")
    cls = getattr(import_module(module_name), class_name)
    return cls(**options)
