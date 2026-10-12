"""MAST Tier-A annotators: deterministic failure-mode labels over traces.

MAST (Cemri et al., 2025, arXiv:2503.13657) defines 14 failure modes in three
clusters; Tier A covers those derivable from a persisted trace without an LLM.
A label is True, False or None (not determinable, dropped from the denominator);
labels are analysis-only and never feed state or reward. FM-3.2 shares
`evidence_state` with the contact-point detector; `ModeLabel.evidence` records what fired.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from ..core.types import Evidence
from ..detectors.base import StepRecord
from ..detectors.heuristic import evidence_state

# --- the taxonomy, verbatim (arXiv:2503.13657) -----------------------------

MODE_NAMES: dict[str, str] = {
    "FM-1.1": "Disobey task specification",
    "FM-1.2": "Disobey role specification",
    "FM-1.3": "Step repetition",
    "FM-1.4": "Loss of conversation history",
    "FM-1.5": "Unaware of termination conditions",
    "FM-2.1": "Conversation reset",
    "FM-2.2": "Fail to ask for clarification",
    "FM-2.3": "Task derailment",
    "FM-2.4": "Information withholding",
    "FM-2.5": "Ignored other agent's input",
    "FM-2.6": "Reasoning-action mismatch",
    "FM-3.1": "Premature termination",
    "FM-3.2": "No or incomplete verification",
    "FM-3.3": "Incorrect verification",
}

CLUSTERS: dict[str, tuple[str, ...]] = {
    "FC1": ("FM-1.1", "FM-1.2", "FM-1.3", "FM-1.4", "FM-1.5"),
    "FC2": ("FM-2.1", "FM-2.2", "FM-2.3", "FM-2.4", "FM-2.5", "FM-2.6"),
    "FC3": ("FM-3.1", "FM-3.2", "FM-3.3"),
}
CLUSTER_NAMES: dict[str, str] = {
    "FC1": "System design issues",
    "FC2": "Inter-agent misalignment",
    "FC3": "Task verification",
}

# Modes Tier A can decide. The others need judgement and are absent from the
# annotation rather than reported as False.
TIER_A_MODES: tuple[str, ...] = (
    "FM-1.3",
    "FM-1.5",
    "FM-2.1",
    "FM-2.5",
    "FM-3.1",
    "FM-3.2",
    "FM-3.3",
)

# Two step descriptions at least this similar are the same instruction
# reissued. High because agents phrase related steps alike, and a false
# FM-1.3 inflates the FC1 incidence.
REPETITION_THRESHOLD = 0.90
_WORD = re.compile(r"[a-z0-9_]+")
# Content words only: an overlap test against "the", "and", "file" would
# find guidance in every follow-up step.
_STOPWORDS = frozenset(
    "a an the and or but if then this that these those is are was were be been "
    "to of in on at by for with from as it its you your we our i they them "
    "do does did done can could should would will shall may might must not no "
    "so than there here what which who when where how all any both each more "
    "most other some such only own same too very s t just now".split()
)


@dataclass(frozen=True)
class ModeLabel:
    """One failure-mode verdict for one episode.

    `fired is None` means not determinable from this trace, not "did not
    occur". Aggregation drops such labels from the denominator.
    """

    mode: str
    fired: bool | None
    evidence: str = ""

    @property
    def name(self) -> str:
        return MODE_NAMES[self.mode]


def _records(trace: Mapping[str, Any]) -> list[StepRecord]:
    """Rebuild StepRecords from a persisted trace.

    Unknown keys are ignored and missing keys take their defaults.
    """
    fields = set(StepRecord.__dataclass_fields__)
    out: list[StepRecord] = []
    for raw in trace.get("records", []):
        out.append(StepRecord(**{k: v for k, v in raw.items() if k in fields}))
    return out


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


# --- individual annotators -------------------------------------------------


def step_repetition(records: Sequence[StepRecord]) -> ModeLabel:
    """FM-1.3: "unnecessary reiteration of previously completed steps".

    Compares each executed step's description with all earlier ones and
    counts a repeat only when no step since the earlier one (the later step
    included) is state-changing: re-running tests after a new patch is the
    normal coding loop, not reiteration. Proxy scope: catches reissued
    instructions, not equivalent work described differently (under-counts).
    """
    # Plans and the synthesised final-answer record are not executed steps;
    # the final record's fixed description would trigger false repeats.
    executed = [r for r in records if not r.is_plan and not r.is_final]
    for pos, step in enumerate(executed):
        text = _normalize(step.description)
        if not text:
            continue
        for earlier_pos, earlier in enumerate(executed[:pos]):
            if any(s.is_state_changing for s in executed[earlier_pos + 1 : pos + 1]):
                continue  # the workspace moved, so redoing the check is warranted
            if SequenceMatcher(None, _normalize(earlier.description), text).ratio() >= (
                REPETITION_THRESHOLD
            ):
                return ModeLabel(
                    "FM-1.3",
                    True,
                    f"step {step.index} repeats step {earlier.index} with no change between",
                )
    return ModeLabel("FM-1.3", False)


def unaware_of_termination(trace: Mapping[str, Any]) -> ModeLabel:
    """FM-1.5: the team never recognized it should stop and ran out of steps.

    `stop_reason == "max_rounds"` is exactly this: the budget ended the
    episode, not the team. Not to be confused with FM-3.1 (premature
    termination), the opposite failure.
    """
    reason = str(trace.get("stop_reason", ""))
    if reason == "max_rounds":
        return ModeLabel("FM-1.5", True, "episode ended on the step budget")
    return ModeLabel("FM-1.5", False, f"stop_reason={reason}")


def conversation_reset(records: Sequence[StepRecord]) -> ModeLabel:
    """FM-2.1: the orchestrator restarted planning mid-episode.

    In MagenticOne a replan rewrites the task ledger and clears the message
    thread, which is a conversation reset in MAST's sense. The opening plan
    is not one, so only replans after the first count.
    """
    plans = [r.index for r in records if r.is_plan]
    if len(plans) > 1:
        return ModeLabel("FM-2.1", True, f"{len(plans) - 1} replan(s) at steps {plans[1:]}")
    return ModeLabel("FM-2.1", False)


def guidance_ignored(trace: Mapping[str, Any], records: Sequence[StepRecord]) -> ModeLabel:
    """FM-2.5: advisor guidance arrived and the next step ignored it.

    Proxy: content-word overlap between the guidance and the description of
    the following step. Zero overlap is weak evidence (the executor may act
    on advice while wording the step differently), so treat it as a corpus-level
    rate to spot-check against manual labels, not as a per-episode verdict.
    `None` when no guidance was followed by a step.
    """
    by_index = {r.index: r for r in records if not r.is_plan}
    considered = ignored = 0
    for consult in trace.get("consults", []):
        guidance = str(consult.get("guidance", ""))
        if consult.get("advisor_failed") or not guidance.strip():
            continue
        following = [i for i in sorted(by_index) if i > int(consult.get("step_index", -1))]
        if not following:
            continue  # guidance landed on the final answer; nothing follows
        considered += 1
        if not (_words(guidance) & _words(by_index[following[0]].description)):
            ignored += 1
    if considered == 0:
        return ModeLabel("FM-2.5", None, "no guidance with a following step")
    return ModeLabel(
        "FM-2.5", ignored > 0, f"{ignored}/{considered} guided steps share no content"
    )


def premature_termination(
    trace: Mapping[str, Any], records: Sequence[StepRecord]
) -> ModeLabel:
    """FM-3.1: the episode delivered while its own evidence was still broken.

    "Ending before all necessary information has been exchanged": the team
    handed in an answer although the last thing its log shows is an
    unresolved failure (MECHANICAL) or a persistent block (CONCEPTUAL).
    Not determinable when nothing was delivered: a non-delivery is a
    different failure, not an early one.
    """
    if not trace.get("delivered"):
        return ModeLabel("FM-3.1", None, "nothing delivered")
    evidence = evidence_state(list(records))
    if evidence in (Evidence.MECHANICAL, Evidence.CONCEPTUAL):
        return ModeLabel("FM-3.1", True, f"delivered on {evidence.name} evidence")
    return ModeLabel("FM-3.1", False, f"delivered on {evidence.name} evidence")


def incomplete_verification(records: Sequence[StepRecord]) -> ModeLabel:
    """FM-3.2: the outcome was never checked, or the check was invalidated.

    Uses the same `evidence_state` as the contact-point detector, so
    "verified" has one definition. Any terminal evidence other than VERIFIED
    means the work is not backed by a passing check that survived the last edit.
    """
    evidence = evidence_state(list(records))
    return ModeLabel(
        "FM-3.2", evidence is not Evidence.VERIFIED, f"terminal evidence {evidence.name}"
    )


def incorrect_verification(
    trace: Mapping[str, Any], records: Sequence[StepRecord], *, score_is_verified: bool
) -> ModeLabel:
    """FM-3.3: the trace claims green, the ground truth disagrees.

    Requires a trustworthy score: a placeholder scorer that gives every
    episode 0.0 would make this fire on every verified delivery. Such runs
    pass `score_is_verified=False` and are labelled `None`.
    """
    if not score_is_verified:
        return ModeLabel("FM-3.3", None, "score not execution-verified")
    if not trace.get("delivered"):
        return ModeLabel("FM-3.3", None, "nothing delivered")
    if evidence_state(list(records)) is not Evidence.VERIFIED:
        return ModeLabel("FM-3.3", False, "trace never claimed a passing check")
    resolved = float(trace.get("benchmark_score", 0.0)) > 0.0
    return ModeLabel(
        "FM-3.3", not resolved, "claimed a pass but the harness scored 0" if not resolved else ""
    )


# --- episode + corpus level ------------------------------------------------


def annotate(trace: Mapping[str, Any], *, score_is_verified: bool = True) -> dict[str, ModeLabel]:
    """All Tier-A labels for one persisted episode trace.

    Pass `score_is_verified=False` for runs scored by anything other than the
    benchmark's execution-based harness; see `incorrect_verification`.
    """
    records = _records(trace)
    labels = [
        step_repetition(records),
        unaware_of_termination(trace),
        conversation_reset(records),
        guidance_ignored(trace, records),
        premature_termination(trace, records),
        incomplete_verification(records),
        incorrect_verification(trace, records, score_is_verified=score_is_verified),
    ]
    return {label.mode: label for label in labels}


@dataclass(frozen=True)
class ModeIncidence:
    """Incidence of one mode over a corpus of traces."""

    mode: str  # a mode id ("FM-3.2") or a cluster id ("FC3")
    fired: int
    determinable: int  # the denominator: traces where the label was not None

    @property
    def name(self) -> str:
        return MODE_NAMES.get(self.mode) or CLUSTER_NAMES[self.mode]

    @property
    def rate(self) -> float | None:
        """Fired over determinable, or None when nothing was determinable.

        An empty denominator is missing data, not a rate of zero.
        """
        return self.fired / self.determinable if self.determinable else None


def mode_incidence(
    annotations: Iterable[Mapping[str, ModeLabel]],
) -> dict[str, ModeIncidence]:
    """Per-mode incidence over annotated episodes.

    Each mode carries its own denominator, because determinability differs
    per mode and per run. Reporting all modes against the episode count
    would understate any mode that is often not determinable.
    """
    fired = dict.fromkeys(TIER_A_MODES, 0)
    determinable = dict.fromkeys(TIER_A_MODES, 0)
    for annotation in annotations:
        for mode, label in annotation.items():
            if label.fired is None:
                continue
            determinable[mode] += 1
            fired[mode] += int(label.fired)
    return {
        mode: ModeIncidence(mode, fired[mode], determinable[mode]) for mode in TIER_A_MODES
    }


def cluster_incidence(
    annotations: Iterable[Mapping[str, ModeLabel]],
) -> dict[str, ModeIncidence]:
    """Per-cluster incidence: an episode counts once if any of the cluster's modes fired.

    An episode is determinable for a cluster only if at least one of its
    modes was determinable; otherwise it is left out of that cluster's
    numerator and denominator.
    """
    fired = dict.fromkeys(CLUSTERS, 0)
    determinable = dict.fromkeys(CLUSTERS, 0)
    for annotation in annotations:
        for cluster, modes in CLUSTERS.items():
            labels = [annotation[m].fired for m in modes if m in annotation]
            known = [v for v in labels if v is not None]
            if not known:
                continue
            determinable[cluster] += 1
            fired[cluster] += int(any(known))
    return {
        cluster: ModeIncidence(cluster, fired[cluster], determinable[cluster])
        for cluster in CLUSTERS
    }
