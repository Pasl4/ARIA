"""Framework-free benchmark adapter interface.

An adapter holds everything benchmark-specific: task loading and the fixed
train/test split, the contact-point detector, scoring (prefer execution-based
ground-truth checks over LLM judges), prompt variables and the workspace. It
imports no agent framework; the FrameworkRunner builds the team from the
TeamSpec. Only load_tasks, build_detector and score are abstract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..detectors.base import ContactPointDetector


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str  # full task statement given to the team
    domain: str = ""
    gold: dict[str, Any] = field(default_factory=dict)  # ground truth, hidden from executor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskScore:
    score: float  # normalized to [0, 1]
    delivered: bool
    detail: dict[str, Any] = field(default_factory=dict)


class BenchmarkAdapter(ABC):
    name: str

    @abstractmethod
    def load_tasks(self, split: str, limit: int | None = None) -> list[Task]:
        """Load the tasks of the 'train' or 'test' split.

        Splits must be disjoint and fixed: commit them as a manifest (see
        split_manifest) instead of sampling at runtime.
        """

    def prompt_vars(self, task: Task) -> dict[str, str]:
        """Template variables for AgentSpec.render_prompt (e.g. domain_knowledge).

        Information parity: nothing here may exceed what the executor team's
        task prompt provides.
        """
        return {}

    async def prepare_workspace(self, task: Task) -> Path | None:
        """Materialize the per-episode workspace (e.g. data files or a repo checkout).

        Returns None when no workspace is needed. Runners bind their native
        code-execution tool to it and call `release_workspace` in a finally
        block. It must be private to the episode: the agent writes here, and a
        shared directory would let one episode inherit another's work.
        """
        return None

    async def release_workspace(self, workspace: Path | None) -> None:
        """Tear down what prepare_workspace built. Default: nothing to do."""
        return None

    async def workspace_patch(self, workspace: Path | None) -> str | None:
        """Return the agent's workspace changes as a scoreable artifact.

        Used by the `workspace_patch` delivery channel (the config default).
        None means the benchmark has no workspace notion, and the runner scores
        the typed final answer instead. Runners call it before release_workspace.
        """
        return None

    @abstractmethod
    def build_detector(self) -> ContactPointDetector:
        """Build the deterministic contact-point detector for this benchmark.

        The phase schema comes from <teams_root>/<benchmark>/phases.yaml
        (default teams_root: configs/teams). Freeze it before training: it is
        part of state_signature, so a policy trained on another schema fails
        verification on load.
        """

    @abstractmethod
    async def score(self, task: Task, final_answer: str | None) -> TaskScore:
        """Score the delivered artifact against ground truth.

        `final_answer` is the workspace patch under the `workspace_patch`
        delivery channel, otherwise the typed final answer. None means nothing
        was delivered (e.g. a crash): return delivered=False. Whether max-rounds
        answers count as delivered is decided by the runner's stop_reason.
        """

    def execution_image(self, task: Task) -> str | None:
        """Docker image for the executor's own code blocks, or None to use the host.

        Separate from scoring, which a benchmark harness may run in its own
        containers. Return an image when the repository's tests cannot import on
        the host, else the evidence level can never reach VERIFIED. Passed to tool
        factories as `ToolContext.extras["exec_image"]`; the built-in python_exec
        runs on the host and ignores it, so a custom tool must honour it.
        """
        return None

    def execution_setup(self, task: Task) -> str | None:
        """One-time shell setup in the execution image before the first block runs.

        E.g. the repository's own editable install; None means no setup. Passed
        to tool factories as `ToolContext.extras["exec_setup"]`.
        """
        return None

    def advisor_inspector(self, workspace: "Path | None", task: Task) -> object | None:
        """Optional backend for the advisor's read-only inspection; None uses the host.

        On the host, commands run via subprocess in `workspace` and gate_diff_injection
        appends `git diff` at the pre-delivery gate. When graded state lives elsewhere
        (e.g. a per-task container), return an object with `exec(argv, cap) -> str`,
        `gate_snapshot(cap) -> (heading, body)` and optionally `prompt_hint() -> str`
        for the advisor prompt. The advisor still enforces its command whitelist.
        """
        return None

    def advisor_domain_context(self, task: Task) -> str:
        """Optional benchmark-specific context for the advisor's system prompt.

        Never the gold answer, and never information the executor does not
        also have (information parity, see prompt_vars).
        """
        return ""

    def task_timeout_s(self, task: Task) -> float | None:
        """Agent-phase wall-clock budget the benchmark declares for this task, or None.

        None (the default) applies the configured `episode_timeout_s`; a
        declared value is capped at it. Container benchmarks that ship a
        per-task agent timeout return it so episodes stay under the limit their
        own harness enforces.
        """
        return None
