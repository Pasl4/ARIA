"""Generic deterministic detector, the default implementation.

Phase: whole-word markers over the upcoming instruction (else the last completed
step); the per-benchmark markers are hashed into the state signature, so validate
them on hand labels before training. Position: phase start iff the phase differs
from the previous contact point's. Evidence: the precedence in evidence_state().
Benchmarks with a fixed programmatic workflow can subclass ContactPointDetector.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

from ..core.types import (
    Commitment,
    ContactPoint,
    Evidence,
    ExecStatus,
    ObsStatus,
    Phase,
    PhasePosition,
)
from .base import ContactPointDetector, StepRecord

# Fallback markers for a HeuristicDetector built without any; build_detector
# always passes the phases.yaml markers, where an unlisted phase gets none.
DEFAULT_PHASE_MARKERS: dict[Phase, tuple[str, ...]] = {
    Phase.FRAME: ("plan", "understand the task", "requirements", "outline"),
    Phase.EXPLORE: ("load", "inspect", "search", "browse", "explore", "locate", "list"),
    Phase.REASON: ("analyze", "implement", "compute", "test", "fix", "derive", "write"),
    Phase.DELIVER: ("final", "answer", "submit", "conclude", "summarize", "deliver"),
}

# Failing steps (each plus its internal repairs) since the last passing
# verification that turn a mechanical error into conceptual evidence.
CONCEPTUAL_STREAK = 2


def commitment_state(records: list[StepRecord]) -> Commitment:
    """Return the commitment of the current plan segment.

    Committed from the first completed state-changing step after the most
    recent plan record (episode start or replan), so a replan resets it. Phase
    detours within a segment do not reset it, since the work done still exists.
    Logged covariate, not part of the default state index.
    """
    for step in reversed(records):
        if step.is_plan:
            break
        if step.is_state_changing:
            return Commitment.COMMITTED
    return Commitment.UNCOMMITTED


def exec_status(records: list[StepRecord]) -> ExecStatus:
    """Return the exec3 verdict from the completed-step log.

    NONE until a record carries executed console bytes; otherwise the last
    executing record decides. Its ``error`` is derived from the executed bytes,
    never the transcript, so model prose cannot set the verdict. A step that
    ran code but printed nothing is not visible here.
    """
    last = None
    for record in records:
        if record.executed_output:
            last = record
    if last is None:
        return ExecStatus.NONE
    return ExecStatus.ERROR if last.error is not None else ExecStatus.CLEAN


# Frames a tool bridge can write into its tool return. The ERROR frame marks an
# environment-reported failure inside a normal return (the tool does not
# raise, so autogen's is_error stays False). The noise frame is an
# end-of-episode notice: termination, not a failure.
OBS_FAIL_FRAMES = ("=== TOOL ERROR ===",)
OBS_NOISE_FRAMES = ("The conversation is already over",)
# Transcript fallback for runners that pass only TaskResult.messages to the
# translator, so no ToolCallExecutionEvents fill tool_ran/tool_output. Only
# tool-bridge banners count, not "Execution result of code block", so outcome3
# stays identical to exec3 on execution-based benchmarks.
OBS_ACT_FRAMES = ("=== TOOL RESULT ===", "=== TOOL ERROR ===")


def _obs_acting(record: StepRecord) -> bool:
    """Return True if this step acted against the environment.

    ARIA_OBS_FALLBACK=0 (default 1) disables only the transcript fallback, not the
    machine channels (executed_output, tool_ran). Use it to evaluate a policy
    trained without the fallback and record it with the run: nothing logs it and
    the checkpoint state_version (layout descriptor) is the same either way.
    A banner forged by the model can shift the state cell, never the score.
    """
    if record.executed_output:
        return True
    text = record.tool_output or record.output or ""
    if any(frame in text for frame in OBS_NOISE_FRAMES):
        return False  # termination is not an action
    if record.tool_ran:
        return True
    if os.environ.get("ARIA_OBS_FALLBACK", "1") == "0":
        return False
    return any(frame in text for frame in OBS_ACT_FRAMES)


def obs_status(records: list[StepRecord]) -> ObsStatus:
    """Return the outcome3 verdict, the tool-aware sibling of exec_status.

    NONE until a record acted against the environment (executed console bytes
    or a harness-stamped tool result). Otherwise the last acting record
    decides: failed on a machine-flagged error or an error frame in the tool
    return, ok otherwise. An end-of-episode notice does not count as acting.
    Where executed bytes exist this equals exec_status by construction.
    """
    last = None
    for record in records:
        if _obs_acting(record):
            last = record
    if last is None:
        return ObsStatus.NONE
    if last.error is not None:
        return ObsStatus.FAILED
    text = last.tool_output or last.output or ""
    if any(frame in text for frame in OBS_FAIL_FRAMES):
        return ObsStatus.FAILED
    return ObsStatus.OK


def evidence_state(
    records: list[StepRecord], *, conceptual_streak: int = CONCEPTUAL_STREAK
) -> Evidence:
    """Return the evidence level of a step log; shared with the MAST annotator.

    * A failing step (error and/or failed verification, counted once) adds
      1 + n_internal_repairs to a target-blind streak; only a pass clears it.
    * CONCEPTUAL is sticky (streak >= threshold) until a pass; MECHANICAL means
      the last completed step failed; VERIFIED needs a pass with no state
      change or failure since.
    * The controller folds stalls into CONCEPTUAL; this sees only the log.
    """
    # note_final synthesizes the is_final record for the answer; no executor
    # ran, so counting it would mask the last real step's MECHANICAL evidence.
    completed = [r for r in records if not r.is_plan and not r.is_final]
    verified_since_change = False
    streak = 0
    for step in completed:
        if step.is_state_changing:
            verified_since_change = False  # any edit invalidates the check
        if step.verification == "pass":
            verified_since_change = True
            streak = 0
        elif step.error is not None or step.verification == "fail":
            verified_since_change = False
            streak += 1 + step.n_internal_repairs
    if streak >= conceptual_streak:
        return Evidence.CONCEPTUAL
    if completed and (
        completed[-1].error is not None or completed[-1].verification == "fail"
    ):
        return Evidence.MECHANICAL
    if verified_since_change:
        return Evidence.VERIFIED
    return Evidence.UNCHECKED


@lru_cache(maxsize=512)
def _marker_pattern(marker: str) -> re.Pattern[str]:
    """Whole-word match for a marker phrase (multi-word markers allowed)."""
    return re.compile(r"(?<!\w)" + re.escape(marker.lower()) + r"(?!\w)")


def _matches_marker(marker: str, text: str) -> bool:
    return _marker_pattern(marker).search(text) is not None


class HeuristicDetector(ContactPointDetector):
    def __init__(
        self,
        phase_markers: dict[Phase, tuple[str, ...]] | None = None,
        conceptual_streak: int = CONCEPTUAL_STREAK,
    ) -> None:
        self._markers = phase_markers or DEFAULT_PHASE_MARKERS
        self._conceptual_streak = conceptual_streak
        self._prev_phase: Phase | None = None

    def reset(self) -> None:
        self._prev_phase = None

    def detect(
        self,
        plan: str,
        history: list[StepRecord],
        upcoming_instruction: str | None = None,
        phase_hint: Phase | None = None,
    ) -> ContactPoint:
        if not history or history[-1].is_plan:
            # Structural contract: a (re)drafted plan is FRAME at phase start,
            # regardless of any text.
            self._prev_phase = Phase.FRAME
            return ContactPoint(
                phase=Phase.FRAME,
                position=PhasePosition.START,
                error=Evidence.UNCHECKED,
                commitment=commitment_state(history),
                step_index=history[-1].index + 1 if history else 0,
                step_summary="plan drafted",
                executor_status=self._status_snapshot(plan, history, upcoming_instruction),
                exec_status=exec_status(history),
                obs_status=obs_status(history),
            )

        # A classifier hint replaces the marker classification; structural
        # rules above and position/error logic below are never delegated.
        phase = (
            phase_hint
            if phase_hint is not None
            else self._classify_phase(history[-1], upcoming_instruction)
        )
        position = PhasePosition.START if phase != self._prev_phase else PhasePosition.REST
        self._prev_phase = phase
        error = evidence_state(history, conceptual_streak=self._conceptual_streak)

        return ContactPoint(
            phase=phase,
            position=position,
            error=error,
            commitment=commitment_state(history),
            step_index=history[-1].index + 1,
            step_summary=upcoming_instruction or history[-1].description,
            executor_status=self._status_snapshot(plan, history, upcoming_instruction),
            exec_status=exec_status(history),
            obs_status=obs_status(history),
        )

    # ------------------------------------------------------------------

    def _classify_phase(self, last_step: StepRecord, upcoming: str | None) -> Phase:
        # The decision gates the upcoming step; classify that when known.
        if upcoming is not None:
            return self._phase_of_text(upcoming)
        return self._phase_of_text(last_step.description)

    def _phase_of_text(self, text: str) -> Phase:
        text = text.lower()
        # Whole-word matching: with a substring test a DELIVER marker "diff"
        # would match "different" and misclassify ordinary instructions.
        for phase in (Phase.DELIVER, Phase.REASON, Phase.EXPLORE, Phase.FRAME):
            if any(_matches_marker(marker, text) for marker in self._markers[phase]):
                return phase
        # No marker matched: mid-workflow steps default to the reasoning phase.
        return Phase.REASON

    def _status_snapshot(
        self, plan: str, history: list[StepRecord], upcoming: str | None
    ) -> str:
        """Return the complete trajectory, forwarded to the advisor on CONSULT.

        The snapshot is untruncated (information parity with the executor); the
        advisor's budget (``AdvisorConfig.max_context_chars``) owns trimming and
        marks it visibly. Built only from the team's StepRecords, so no
        grader-side data; executed output is listed separately because it comes
        from a real process.
        """
        lines = [f"Plan:\n{plan}", "", "Steps so far:"]
        for step in history:
            status = "FAILED" if step.error else "ok"
            lines.append(f"\n  {step.index + 1}. [{status}] {step.description}")
            if step.output:
                lines.append(f"  --- transcript ---\n{step.output}")
            if step.executed_output:
                lines.append(
                    f"  --- executed output (real process bytes) ---\n{step.executed_output}"
                )
            if step.error:
                lines.append(f"  --- error ---\n{step.error}")
        if upcoming:
            lines += ["", f"Next planned step:\n{upcoming}"]
        return "\n".join(lines)
