"""InterventionController: ARIA's decision logic at orchestration contact points.

Framework-agnostic: an integration maps its log to StepRecords and calls
`start_episode`, `on_contact_point` at each decision moment (ledger round,
node or task boundary) and `finalize`. The controller alone encodes state,
computes rewards, closes transitions and enforces the consult budget, so every
condition (learned or baseline) is logged under the same reward model.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast, runtime_checkable

from ..advisor.advisor import AdvisorReply
from ..core.reward import RewardModel
from ..core.state import DEFAULT_LAYOUT, StateLayout
from ..core.types import (
    Action,
    ContactPointKind,
    Evidence,
    Phase,
    Transition,
)
from ..detectors.base import ContactPointDetector, StepRecord
from ..detectors.markers import MarkerTables
from ..policies.base import InterventionPolicy
from ..telemetry.token_meter import TokenMeter, call_purpose
from .seeding import SeedingConfig, SeedingState

logger = logging.getLogger(__name__)

# Single definition of how advisor guidance is framed to the executor. The
# wording follows Anthropic's advisor tool guidance and is recorded in the run
# manifest.
GUIDANCE_MARKER = "[Expert advisor guidance]"
GUIDANCE_TEMPLATE = (
    f"{GUIDANCE_MARKER} Give this advice serious weight; deviate only if a "
    "concrete observation contradicts it: {guidance}"
)


def format_guidance(guidance: str) -> str:
    return GUIDANCE_TEMPLATE.format(guidance=guidance)


def _failed_cycles_segment(records: list[StepRecord]) -> int:
    """Failed verification runs since the last replan (logged annotation)."""
    n = 0
    for step in reversed(records):
        if step.is_plan:
            break
        if step.verification == "fail":
            n += 1
    return n


def _steps_since_last_pass(records: list[StepRecord]) -> int:
    """Completed steps since the last passing verification (logged annotation)."""
    n = 0
    for step in reversed(records):
        if step.is_plan:
            continue
        if step.verification == "pass":
            break
        n += 1
    return n


@runtime_checkable
class AdvisorLike(Protocol):
    """Advisor interface the controller depends on; test stubs implement it."""

    workspace: Any  # per-episode workspace, set by the framework runner

    async def consult(
        self, executor_status: str, pinned_header: str = "", is_gate: bool = False
    ) -> AdvisorReply: ...


@dataclass
class _Pending:
    state: int
    action: Action
    kind: ContactPointKind = ContactPointKind.PRE_STEP
    # Log position of the decision, so an offline replay can map a SILENT
    # contact point back to the trace prefix records[:n_records].
    step_index: int = -1
    n_records: int = 0
    consult_cost_usd: float = 0.0  # metered USD of this decision's consult
    assessment: str | None = None  # advisor's self-assessment (covariate only)
    advisor_failed: bool = False
    budget_clamped: bool = False
    stall_folded: bool = False  # stall evidence fold applied at this contact point
    # Decision-time annotations merged into Transition.info; logged for
    # analysis and alternative layouts, never read by the learner.
    annotations: dict[str, object] = field(default_factory=dict)


@dataclass
class InterventionController:
    detector: ContactPointDetector
    policy: InterventionPolicy
    advisor: AdvisorLike
    reward_model: RewardModel
    eval_mode: bool = False
    # Wall-clock deadline (time.monotonic seconds), set per episode by the
    # EpisodeRunner and checked at every contact point. Synchronous calls in
    # the framework can block asyncio cancellation, so past the deadline the
    # controller raises and the episode is recorded as crashed.
    wall_clock_deadline: float | None = None
    # Uniform per-episode consult cap, enforced here so it applies
    # identically to every condition (None disables it).
    consult_budget: int | None = None
    # Optional TokenMeter reference: lets the controller stamp each consult's
    # metered USD onto its transition. None keeps costs at 0.0 (tests).
    meter: TokenMeter | None = None
    # Per-benchmark evidence marker tables, from which the framework
    # translator derives per-step predicates (is_state_changing,
    # verification). None keeps the predicates neutral.
    markers: MarkerTables | None = None
    # The active state layout; alternative layouts are config entries
    # (`state.dims`) resolved by the experiment.
    layout: StateLayout = DEFAULT_LAYOUT
    # Deficit-targeted consult seeding (see seeding.py); disabled by default.
    seeding: SeedingConfig = field(default_factory=SeedingConfig)

    # Per-episode bookkeeping (read by the host / episode runner).
    n_consults: int = 0
    n_contact_points: int = 0
    transitions: list[Transition] = field(default_factory=list)
    latest_status: str = ""  # last status snapshot (used by the organic tool)
    latest_records: list[StepRecord] = field(default_factory=list)
    final_answer: str | None = None  # set by integrations that capture it
    stop_reason: str = ""  # completed | max_rounds | crashed | "" (unknown)
    # Episode-trace accumulation: raw consult events and framework ledger
    # snapshots. Telemetry only, never read by detection or policy.
    consults: list[dict[str, object]] = field(default_factory=list)
    ledgers: list[dict[str, object]] = field(default_factory=list)
    _pending: _Pending | None = None
    _stalls_since_consult: int = 0  # stall grace counter (see on_contact_point)
    _cp_index_phase: int = 0  # contact-point counter within the phase
    _cp_prev_phase: Phase | None = None
    # Transitions awaiting a verdict; `policy.learn` runs only on a scored
    # finalize. Otherwise a crashed episode's consult costs would be learned
    # while its censored benefit is not, biasing the learner against consults.
    _learn_buffer: list[Transition] = field(default_factory=list)
    _finalized: bool = False
    _seed_state: SeedingState = field(default_factory=SeedingState)

    def start_episode(self, episode: int | None = None, task_header: str = "") -> None:
        self.detector.reset()
        self.n_consults = 0
        self.n_contact_points = 0
        self.transitions = []
        self.latest_status = ""
        self.latest_records = []
        self.final_answer = None
        self.stop_reason = ""
        self.consults = []
        self.ledgers = []
        self._pending = None
        self.wall_clock_deadline = None
        self._stalls_since_consult = 0
        self._cp_index_phase = 0
        self._cp_prev_phase = None
        self._learn_buffer = []
        self._finalized = False
        # Task header (the task text); the advisor prepends it after the context
        # front-trim, so the trim never removes it.
        self._task_header = task_header
        # Seeding plan for this episode: designated cells ordered by deficit;
        # the 1-based episode number decides whether the seed fires.
        consult_visits = getattr(self.policy, "consult_visits", None)
        if (
            episode is not None
            and self.seeding.enabled
            and callable(consult_visits)
        ):
            visits_of = cast("Callable[[int], float]", consult_visits)
            deficits = []
            for spec in self.seeding.cells:
                states = [
                    s
                    for s in range(self.layout.n_states)
                    if self.layout.label(s).startswith(spec)
                ]
                visits = sum(visits_of(s) for s in states)
                deficits.append((self.seeding.deficit_target - visits, spec))
            self._seed_state.begin(self.seeding, episode + 1, deficits)
        else:
            self._seed_state = SeedingState()

    # ------------------------------------------------------------------
    # Called by the framework integration at each contact point
    # ------------------------------------------------------------------

    async def on_contact_point(
        self,
        plan: str,
        records: list[StepRecord],
        upcoming_instruction: str | None = None,
        kind: ContactPointKind = ContactPointKind.PRE_STEP,
    ) -> str | None:
        """Run one decision; return formatted advisor guidance on CONSULT.

        Returns None on SILENT or advisor failure, and raises TimeoutError
        past `wall_clock_deadline`. `records` is the execution log so far
        (empty means the plan was just drafted). `upcoming_instruction` is the
        step this decision gates, if known. `kind` is the structural kind of
        the decision moment, stamped onto the ContactPoint and logged.
        """
        if (
            self.wall_clock_deadline is not None
            and time.monotonic() > self.wall_clock_deadline
        ):
            raise TimeoutError(
                "episode wall-clock budget exceeded at contact point "
                f"{self.n_contact_points}"
            )
        cp = self.detector.detect(plan, records, upcoming_instruction)
        # The controller stamps consulted_before and cp_index (0-based) itself: decision
        # history is not log content, and the executor cannot forge either.
        cp = replace(
            cp,
            consulted_before=self.n_consults > 0,
            kind=kind,
            cp_index=self.n_contact_points,
        )
        stall_folded = False
        if kind is ContactPointKind.STALL:
            # A stall signals trouble even without a traceback, so raise the
            # evidence to CONCEPTUAL (the ordinal maximum). The first stall
            # after a consult is not folded, so a consult cannot itself
            # produce the state that triggers further consults.
            grace = self.n_consults > 0 and self._stalls_since_consult == 0
            self._stalls_since_consult += 1
            if not grace:
                stall_folded = True
                cp = replace(cp, error=max(cp.error, Evidence.CONCEPTUAL))
        state = self.layout.encode(cp)
        self.latest_status = cp.executor_status
        self.latest_records = list(records)
        self.n_contact_points += 1
        if cp.phase is not self._cp_prev_phase:
            self._cp_prev_phase = cp.phase
            self._cp_index_phase = 0
        self._cp_index_phase += 1
        # Decision-time annotations: logged with the transition and never read
        # by the learner unless promoted to a layout dimension.
        annotations: dict[str, object] = {
            "cp_index_episode": self.n_contact_points,
            "cp_index_phase": self._cp_index_phase,
            "n_failed_cycles_segment": _failed_cycles_segment(records),
            "steps_since_last_pass": _steps_since_last_pass(records),
            "n_internal_repairs_cum": sum(r.n_internal_repairs for r in records),
            "n_consults": self.n_consults,
            "commitment": cp.commitment.name.lower(),
            # Raw covariates, so alternative layouts can be re-encoded offline
            # from the transition log.
            "phase_marker": cp.phase.name,
            "evidence_marker": cp.error.name,
            "exec_status": cp.exec_status.name,
        }

        self.resolve_pending(records, next_state=state)

        action = self.policy.decide(cp)
        decision_info_fn = getattr(self.policy, "decision_info", None)
        decision_meta = (
            cast("dict[str, Any]", decision_info_fn())
            if callable(decision_info_fn)
            else {}
        )
        budget_clamped = False
        if (
            action == Action.CONSULT
            and self.consult_budget is not None
            and self.n_consults >= self.consult_budget
        ):
            action = Action.SILENT
            budget_clamped = True

        # Seeding may force a consult the policy declined: at the first entry
        # into the designated cell, or at an already-consulted gate. It never
        # overrides the consult budget.
        seeded_cell = ""
        budget_left = self.consult_budget is None or self.n_consults < self.consult_budget
        if action is Action.SILENT and not budget_clamped and budget_left:
            is_gate = kind is ContactPointKind.PRE_DELIVERY
            spec = self._seed_state.wants_consult(self.layout.label(state), is_gate)
            if spec is not None:
                action = Action.CONSULT
                self._seed_state.used = True
                seeded_cell = spec
            elif self._seed_state.second_gate and is_gate and cp.consulted_before:
                action = Action.CONSULT
                seeded_cell = "second_consult_gate"

        # p_consult is the policy's own consult propensity, for off-policy
        # estimation. Seed-forced consults log 1.0 and cap-forced SILENT logs
        # 0.0; both are flagged so estimators can exclude them.
        if seeded_cell:
            annotations["p_consult"] = 1.0
        elif budget_clamped:
            annotations["p_consult"] = 0.0
        elif "p_consult" in decision_meta:
            annotations["p_consult"] = decision_meta["p_consult"]
        annotations["exploratory"] = bool(decision_meta.get("exploratory", False))
        annotations["seeded"] = bool(seeded_cell)
        annotations["seeded_cell"] = seeded_cell
        annotations["cap_forced_silent"] = budget_clamped

        if action == Action.CONSULT:
            self.n_consults += 1
            self._stalls_since_consult = 0  # the stall grace window restarts here
            # Explicit purpose tag: overrides any outer tag (e.g. "planning")
            # so consult tokens are always attributed to consulting.
            calls_before = len(self.meter.calls) if self.meter is not None else 0
            with call_purpose("consult"):
                reply = await self.advisor.consult(
                    cp.executor_status,
                    pinned_header=self._task_header,
                    is_gate=kind == ContactPointKind.PRE_DELIVERY,
                )
            # Metered USD of this consult (meter delta across the call). Cost
            # is in dollars, not call counts, since later consults carry more
            # context.
            consult_cost = (
                sum(c.cost_usd for c in self.meter.calls[calls_before:])
                if self.meter is not None
                else 0.0
            )
            self.consults.append(
                {
                    "step_index": cp.step_index,
                    "kind": kind.name.lower(),
                    "state_label": self.layout.label(state),
                    "assessment": reply.assessment,
                    "guidance": reply.guidance,
                    "advisor_failed": reply.failed,
                    "cost_usd": consult_cost,
                    # The exact advisor input, persisted so each consult can
                    # be replayed.
                    "executor_status": cp.executor_status,
                    "task_header": self._task_header[:3000],
                    "tool_log": getattr(reply, "tool_log", ""),
                    "seeded": bool(seeded_cell),
                }
            )
            self._pending = _Pending(
                state=state,
                action=action,
                kind=kind,
                step_index=cp.step_index,
                n_records=len(records),
                consult_cost_usd=consult_cost,
                assessment=reply.assessment,
                advisor_failed=reply.failed,
                stall_folded=stall_folded,
                annotations=annotations,
            )
            logger.info(
                "contact point %d (%s): CONSULT (%s)",
                cp.step_index,
                self.layout.label(state),
                reply.assessment,
            )
            return format_guidance(reply.guidance) if not reply.failed else None

        self._pending = _Pending(
            state=state,
            action=action,
            kind=kind,
            step_index=cp.step_index,
            n_records=len(records),
            budget_clamped=budget_clamped,
            stall_folded=stall_folded,
            annotations=annotations,
        )
        return None

    def resolve_pending(
        self,
        records: list[StepRecord],
        next_state: int = -1,
        terminal: bool = False,
        terminal_reward: float = 0.0,
        learn: bool = True,
    ) -> Transition | None:
        """Resolve the previous decision against the observed outcome.

        Public so integrations can resolve on paths that bypass the next contact
        point, e.g. a replan that clears the log (call it before the clear so the
        failing evidence is still judged). With `learn=True` the transition is
        buffered and reaches `policy.learn`, in order, only on a scored
        `finalize`; `learn=False` records it without learning (no verdict).
        """
        if self._pending is None:
            return None
        pending, self._pending = self._pending, None

        next_step_success = bool(records) and records[-1].error is None
        reward = self.reward_model.step_reward(
            pending.action, consult_cost_usd=pending.consult_cost_usd
        )
        if terminal:
            reward += terminal_reward

        transition = Transition(
            state=pending.state,
            action=pending.action,
            reward=reward,
            next_state=-1 if terminal else next_state,
            terminal=terminal,
            info={
                "state_label": self.layout.label(pending.state),
                "contact_kind": pending.kind.name.lower(),
                "step_index": pending.step_index,
                "n_records": pending.n_records,
                "consult_cost_usd": pending.consult_cost_usd,
                "next_step_success": next_step_success,
                "assessment": pending.assessment,
                "advisor_failed": pending.advisor_failed,
                "budget_clamped": pending.budget_clamped,
                "stall_folded": pending.stall_folded,
                **pending.annotations,
            },
        )
        if learn:
            # Flushed in finalize only for a scored episode; order is
            # preserved, so the Q-updates match those of live learning.
            self._learn_buffer.append(transition)
        self.transitions.append(transition)
        return transition

    def note_ledger(self, snapshot: dict[str, object]) -> None:
        """Append a framework ledger or replan snapshot to the episode trace.

        Telemetry only; never read by detection or policy.
        """
        self.ledgers.append(dict(snapshot))

    def note_final(self, answer: str, stop_reason: str) -> None:
        """Record the integration's final answer and append it as a final step.

        Integrations call this so the host does not infer the answer from
        message positions.
        """
        self.final_answer = answer
        self.stop_reason = stop_reason
        self.latest_records = [
            *self.latest_records,
            StepRecord(
                index=len(self.latest_records),
                description="Synthesize and state the final answer",
                output=answer,
                is_final=True,
            ),
        ]

    def _mc_adjusted(self, buffer: list[Transition]) -> list[Transition]:
        """Under `rl.backup == "mc"`, replace each reward with its realized return.

        The episode is complete at flush time, so G_t = sum_k gamma^(k-t) r_k is exact.
        Each transition becomes terminal with reward G_t, turning `TabularQPolicy.update`
        into an every-visit Monte-Carlo update. With gate revision the terminal reward
        lands on a later (state, action) than the consult that earned it; a 1-step
        backup would credit the silent post-revision gate instead.
        """
        cfg = getattr(self.policy, "rl_config", None)
        if getattr(cfg, "backup", "onestep") != "mc":
            return buffer
        gamma = float(getattr(cfg, "gamma", 0.99))
        out: list[Transition] = []
        g = 0.0
        for t in reversed(buffer):
            g = t.reward + gamma * g
            out.append(replace(t, reward=g, terminal=True, next_state=-1))
        out.reverse()
        return out

    def finalize(
        self,
        score: float,
        delivered: bool,
        final_records: list[StepRecord] | None = None,
        scored: bool = True,
    ) -> float:
        """Close the last transition with the terminal reward and return it.

        Delivery enters the return once, through the terminal reward. `scored=False`
        means no verdict exists (crash or unjudged episode): the transition is recorded
        but not learned, so infrastructure failures do not become terminal Q-targets.
        `final_records` becomes the authoritative log, so steps after the last contact
        point (e.g. a final test run) reach the persisted trace that MAST annotation reads.
        """
        # Idempotent: if the runner already finalized (e.g. the wall-clock
        # timeout fired between finalize and return), a second call neither
        # decays epsilon again nor re-learns the episode.
        if self._finalized:
            return 0.0
        if final_records:
            # Keep the synthesized final-answer record note_final appended;
            # it is not part of any framework's executor log.
            tail = [r for r in self.latest_records if r.is_final]
            self.latest_records = [
                *final_records,
                *(replace(r, index=len(final_records) + i) for i, r in enumerate(tail)),
            ]
        terminal_reward = self.reward_model.terminal_reward(score, delivered)
        self.resolve_pending(
            final_records or [],
            terminal=True,
            terminal_reward=terminal_reward,
            learn=scored,
        )
        if scored:
            for transition in self._mc_adjusted(self._learn_buffer):
                self.policy.learn(transition)
        self._learn_buffer = []
        self._finalized = True
        self.policy.end_episode(eval_mode=self.eval_mode)
        return terminal_reward

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)

    def transitions_as_dicts(self) -> list[dict[str, object]]:
        return [t.as_dict() for t in self.transitions]
