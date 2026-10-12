"""Deterministic contact-point detection (no LLM call). Every detector follows:

* An empty history means the plan was just drafted: FRAME at phase start.
* Records with ``is_plan=True`` classify as FRAME regardless of their text, since
  planning prompts contain task vocabulary that would match phase markers.
* With ``upcoming_instruction``, the phase describes the step the decision gates,
  not the completed one; otherwise a "consult before X" rule fires one step late.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

from ..core.types import ContactPoint, Phase


@dataclass
class StepRecord:
    """One completed executor step, as built by the framework translator."""

    index: int
    description: str  # what the executor set out to do
    output: str  # full step transcript (model text and executor output)
    # Bytes a real process wrote, carried out of band so the model cannot write
    # into them. Detection reads this field, not the transcript, because a model
    # can emit fabricated console blocks that would otherwise reach VERIFIED.
    # Empty means the step executed nothing and yields no verification evidence.
    executed_output: str = ""
    # Tool-execution channel, out of band like executed_output: whether a
    # FunctionTool ran and what the harness returned. Filled by the framework
    # translator from ToolCallExecutionEvents and read by heuristic.obs_status
    # (outcome3); False/"" on pure-console steps.
    tool_ran: bool = False
    tool_output: str = ""
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None  # raw error text, None if the step succeeded
    # Failed-then-retried internal segments (see count_internal_repairs).
    # Adds to the failure streak in evidence_state only when the step fails.
    n_internal_repairs: int = 0
    # Evidence predicates from the benchmark's marker tables (see
    # MarkerTables.derive), filled by the framework translator; neutral when
    # no tables are configured.
    is_state_changing: bool = False  # invalidates prior verification
    verification: str = "none"  # none | pass | fail (see MarkerTables.derive)
    # Annotation-only error identity (see error_fingerprint); never gates.
    error_target: str = ""
    is_plan: bool = False  # a (re)planning event, classifies as FRAME
    is_final: bool = False  # step delivered the final answer


class ContactPointDetector(ABC):
    """Builds the ContactPoint before each decision, deterministically."""

    @abstractmethod
    def detect(
        self,
        plan: str,
        history: list[StepRecord],
        upcoming_instruction: str | None = None,
        phase_hint: Phase | None = None,
    ) -> ContactPoint:
        """Return the contact point for the decision after ``history``.

        Implementations follow the module-level contract. ``phase_hint`` is an
        optional pre-computed phase, e.g. from an async PhaseClassifier run by a
        custom controller (the built-in controller passes none). It replaces
        only the text classification; the structural FRAME rules and the
        position/evidence logic stay with the detector.
        """

    def reset(self) -> None:
        """Clear any per-episode state before a new episode."""


class PhaseClassifier(Protocol):
    """Extension slot for a pluggable, possibly LLM-backed phase classifier.

    Not used by default: detection is deterministic and phases.yaml accepts
    only ``classifier: heuristic``. A custom controller can call it before the
    detector and pass the result as ``phase_hint``.
    """

    async def classify_phase(self, step_text: str, plan: str) -> Phase | None:
        """Return the phase of ``step_text``, or None to use the detector's own."""
        ...
