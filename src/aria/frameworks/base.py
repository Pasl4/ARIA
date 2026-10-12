"""FrameworkRunner: the interface that keeps the episode loop framework-agnostic.

One runner per framework owns everything framework-native (team construction,
the episode loop with the InterventionController hooked in, guidance injection,
final-answer capture, stop-reason classification); the layers above are shared.
`supported_contact_points` lists the ContactPointKinds the framework produces;
the run manifest records it so decision granularity is comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..benchmarks.base import BenchmarkAdapter, Task
from ..core.types import ContactPointKind
from ..detectors.base import StepRecord
from ..orchestration.controller import InterventionController
from ..telemetry.token_meter import TokenMeter
from .spec import TeamSpec

STOP_COMPLETED = "completed"
STOP_MAX_ROUNDS = "max_rounds"
STOP_CRASHED = "crashed"


@dataclass(frozen=True)
class EpisodeOutcome:
    final_answer: str | None
    stop_reason: str  # STOP_COMPLETED | STOP_MAX_ROUNDS | STOP_CRASHED
    records: list[StepRecord]  # final execution log, judged by finalize()
    # The agent's workspace changes (git diff against the setup commit),
    # extracted before the workspace is released; None if there is no
    # workspace or extraction failed. `EpisodeRunConfig.delivery` decides
    # whether this or the typed final_answer is scored.
    workspace_patch: str | None = None


@dataclass(frozen=True)
class EpisodeRunConfig:
    condition: str = "aria"
    seed: int = 0
    organic_consult: bool = False
    cost_multiplier: str = "10x"  # derived from pricing; quoted by the organic consult tool
    # Which artifact is scored: "workspace_patch" (git diff of the agent's
    # changes) or "typed" (the model's typed final answer). Both are logged.
    delivery: str = "workspace_patch"
    # TeamSpec.runtime.for_framework(runner.name) plus `gate_revision_rounds`:
    # the runner's native knobs and cross-framework budgets such as `max_steps`.
    runtime: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FrameworkRunner(Protocol):
    name: str
    supported_contact_points: frozenset[ContactPointKind]

    async def run_episode(
        self,
        task: Task,
        spec: TeamSpec,
        controller: InterventionController,
        meter: TokenMeter,
        config: EpisodeRunConfig,
        benchmark: BenchmarkAdapter,
    ) -> EpisodeOutcome:
        """Run one episode; must not raise (crashes map to STOP_CRASHED)."""
        ...
