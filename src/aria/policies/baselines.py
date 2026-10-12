"""Baseline consultation policies, by condition name (see registry.py).

- autonomous, organic: AlwaysSilentPolicy (organic enables the consult_advisor tool)
- always_consult: AlwaysConsultPolicy; its episode cost sets reward.cost_anchor_high_usd
- random_advisory: RandomAdvisoryPolicy, budget-matched to a learned intervention rate
- plan_advisory, early_once, gate_once, early_gate_2x: fixed structural schedules
- anthropic_heuristic: AnthropicAdvisorPolicy, Anthropic's advisor guidance as a rule
"""

from __future__ import annotations

import numpy as np

from ..core.types import (
    Action,
    ContactPoint,
    ContactPointKind,
    Evidence,
    Phase,
    PhasePosition,
)
from .base import StaticPolicyBase


class AlwaysSilentPolicy(StaticPolicyBase):
    """Never consults: fully autonomous execution."""

    def decide(self, contact_point: ContactPoint) -> Action:
        return Action.SILENT


class AlwaysConsultPolicy(StaticPolicyBase):
    """Consults at every contact point."""

    def decide(self, contact_point: ContactPoint) -> Action:
        return Action.CONSULT


class RandomAdvisoryPolicy(StaticPolicyBase):
    """Consults at uniformly random contact points with fixed probability.

    For a budget-matched comparison, set consult_probability to the learned
    policy's eval-time intervention rate, so the two conditions differ in
    when they consult but not in how often.
    """

    def __init__(self, consult_probability: float, seed: int | None = None) -> None:
        if not 0.0 <= consult_probability <= 1.0:
            raise ValueError("consult_probability must be in [0, 1]")
        self.consult_probability = consult_probability
        self._rng = np.random.default_rng(seed)

    def decide(self, contact_point: ContactPoint) -> Action:
        return (
            Action.CONSULT
            if self._rng.random() < self.consult_probability
            else Action.SILENT
        )


class ConsultAtPlanPolicy(StaticPolicyBase):
    """Consults exactly once, at the first FRAME contact point.

    Agent-agent analog of Magentic-UI's cooperative planning: the advisor
    reviews the initial plan, then the executor runs autonomously. Relies on
    the detector contract that the first contact point of every episode (and
    every replan) is FRAME.
    """

    def __init__(self) -> None:
        self._consulted = False

    def decide(self, contact_point: ContactPoint) -> Action:
        if not self._consulted and contact_point.phase == Phase.FRAME:
            self._consulted = True
            return Action.CONSULT
        return Action.SILENT

    def end_episode(self, eval_mode: bool = False) -> None:
        self._consulted = False


class EarlyOncePolicy(StaticPolicyBase):
    """Consults exactly once, at the first interior decision after the plan.

    The advisor sees at least one executed step before a wrong path is
    committed. Structural trigger (PRE_STEP with cp_index >= 1), no keywords.
    """

    def __init__(self) -> None:
        self._consulted = False

    def decide(self, contact_point: ContactPoint) -> Action:
        if (
            not self._consulted
            and contact_point.kind is ContactPointKind.PRE_STEP
            and contact_point.cp_index >= 1
        ):
            self._consulted = True
            return Action.CONSULT
        return Action.SILENT

    def end_episode(self, eval_mode: bool = False) -> None:
        self._consulted = False


class GateOncePolicy(StaticPolicyBase):
    """Consults exactly once, at the pre-delivery gate.

    Analog of cascade routing (FrugalGPT, AutoMix), which escalates once a
    candidate answer exists.
    """

    def __init__(self) -> None:
        self._consulted = False

    def decide(self, contact_point: ContactPoint) -> Action:
        if not self._consulted and contact_point.kind is ContactPointKind.PRE_DELIVERY:
            self._consulted = True
            return Action.CONSULT
        return Action.SILENT

    def end_episode(self, eval_mode: bool = False) -> None:
        self._consulted = False


class EarlyGateTwicePolicy(StaticPolicyBase):
    """Consults twice: once early, once at the pre-delivery gate.

    The union of EarlyOncePolicy and GateOncePolicy (condition
    ``early_gate_2x``): the first PRE_STEP with cp_index >= 1 shows the
    advisor one executed step, the PRE_DELIVERY gate shows it the finished
    attempt. Structural triggers only, no keywords.
    """

    def __init__(self) -> None:
        self._early_done = False
        self._gate_done = False

    def decide(self, contact_point: ContactPoint) -> Action:
        if not self._gate_done and contact_point.kind is ContactPointKind.PRE_DELIVERY:
            self._gate_done = True
            return Action.CONSULT
        if (
            not self._early_done
            and contact_point.kind is ContactPointKind.PRE_STEP
            and contact_point.cp_index >= 1
        ):
            self._early_done = True
            return Action.CONSULT
        return Action.SILENT

    def end_episode(self, eval_mode: bool = False) -> None:
        self._early_done = False
        self._gate_done = False


class AnthropicAdvisorPolicy(StaticPolicyBase):
    """Anthropic's advisor-tool executor guidance as a deterministic rule.

    - "before committing to an approach": CONSULT at reason-phase start if not yet consulted
    - "before declaring done": CONSULT at deliver-phase start, once per episode
      (every time with ``deliver_every_time=True``)
    - "when stuck": CONSULT on a CONCEPTUAL error; MECHANICAL errors mostly self-repair
      and trigger a consult only with ``consult_on_mechanical=True``. SILENT otherwise.
    """

    def __init__(
        self,
        consult_on_mechanical: bool = False,
        deliver_every_time: bool = False,
    ) -> None:
        self._consult_on_mechanical = consult_on_mechanical
        self._deliver_every_time = deliver_every_time
        self._deliver_consulted = False

    def decide(self, contact_point: ContactPoint) -> Action:
        cp = contact_point
        if cp.error == Evidence.CONCEPTUAL:
            return Action.CONSULT
        if self._consult_on_mechanical and cp.error == Evidence.MECHANICAL:
            return Action.CONSULT
        if (
            cp.phase == Phase.REASON
            and cp.position == PhasePosition.START
            and not cp.consulted_before
        ):
            return Action.CONSULT
        if cp.phase == Phase.DELIVER and cp.position == PhasePosition.START:
            if self._deliver_every_time or not self._deliver_consulted:
                self._deliver_consulted = True
                return Action.CONSULT
        return Action.SILENT

    def end_episode(self, eval_mode: bool = False) -> None:
        self._deliver_consulted = False
