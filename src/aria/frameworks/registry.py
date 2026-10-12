"""Open, lazy framework registry (mirrors the benchmark registry).

The experiment layer selects the runner via `ExperimentConfig.framework`;
adopters plug custom frameworks in with `register_framework` without editing
aria.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Any

from .base import FrameworkRunner

# name -> "module:ClassName" (lazy) or a registered factory.
_LAZY: dict[str, str] = {
    "autogen": "aria.frameworks.autogen.runner:AutogenRunner",
}
_FACTORIES: dict[str, Callable[..., FrameworkRunner]] = {}


def register_framework(name: str, factory: Callable[..., FrameworkRunner]) -> None:
    if name in _FACTORIES or name in _LAZY:
        raise ValueError(f"framework {name!r} is already registered")
    _FACTORIES[name] = factory


def available_frameworks() -> tuple[str, ...]:
    return tuple(sorted({*_LAZY, *_FACTORIES}))


def build_framework_runner(name: str, **kwargs: Any) -> FrameworkRunner:
    """Build a runner. kwargs are runner-specific (autogen: executor_client)."""
    if name in _FACTORIES:
        return _FACTORIES[name](**kwargs)
    try:
        target = _LAZY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown framework {name!r}; expected one of {available_frameworks()}"
        ) from exc
    module_name, class_name = target.split(":")
    cls = getattr(import_module(module_name), class_name)
    return cls(**kwargs)
