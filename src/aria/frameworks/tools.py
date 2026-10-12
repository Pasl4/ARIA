"""Tool registry: resolves symbolic tool names to framework-native tools.

AgentSpec.tools carries symbolic names ("python_exec", "file_browser",
"consult_advisor"), so the team spec stays framework-neutral. Each framework
package registers one factory per symbol when it is imported. A factory
receives a ToolContext with the episode-specific objects (workspace, advisor,
meter, status provider) and returns the framework-native tool.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# (symbol, framework) -> factory(ToolContext) -> native tool
_REGISTRY: dict[tuple[str, str], Callable[[ToolContext], Any]] = {}

# Long enough for a scoped test run or a pip install, short enough that a
# runaway command cannot consume the episode's wall clock. Override per
# benchmark via `runtime.exec_timeout_s`.
DEFAULT_EXEC_TIMEOUT_S = 600


@dataclass
class ToolContext:
    """Everything a tool factory may need; frameworks fill what they have."""

    framework: str
    workspace: Path | None = None
    advisor: Any | None = None  # AdvisorLike (organic consult tool)
    meter: Any | None = None  # TokenMeter
    status_provider: Callable[[], str] | None = None  # controller.latest_status
    # The organic consult tool gets the same pinned task header and
    # per-episode consult cap as policy-driven consults.
    header_provider: Callable[[], str] | None = None  # controller task header
    consult_budget: int | None = None  # per-episode consult cap (None = uncapped)
    cost_multiplier: str = "10x"
    # Per-code-block execution timeout in seconds. The AutoGen default (60 s)
    # is too short for test suites, and a timed-out run looks like a failing
    # one in the execution evidence.
    exec_timeout_s: int = DEFAULT_EXEC_TIMEOUT_S
    extras: dict[str, Any] = field(default_factory=dict)
    # Objects with an async `stop()` to shut down at episode end, e.g. a
    # container-backed executor. Factories append what they create; the
    # runner closes them in its `finally`.
    closeables: list[Any] = field(default_factory=list)


def register_tool(
    symbol: str, framework: str, factory: Callable[[ToolContext], Any]
) -> None:
    key = (symbol, framework)
    if key in _REGISTRY:
        raise ValueError(f"tool {symbol!r} is already registered for {framework!r}")
    _REGISTRY[key] = factory


def resolve_tool(symbol: str, framework: str, context: ToolContext) -> Any:
    try:
        factory = _REGISTRY[(symbol, framework)]
    except KeyError as exc:
        available = sorted(s for s, fw in _REGISTRY if fw == framework)
        raise ValueError(
            f"tool {symbol!r} is not registered for framework {framework!r}; "
            f"available there: {available}"
        ) from exc
    return factory(context)


def available_tools(framework: str) -> tuple[str, ...]:
    return tuple(sorted(s for s, fw in _REGISTRY if fw == framework))
