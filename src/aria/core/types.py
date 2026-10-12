"""Core MDP types shared across the framework.

The state index is built from `state.dims` (see `StateLayout` in
core/state.py); the default layout has 24 cells. Actions: SILENT | CONSULT.
Phase, Evidence and Commitment are detector outputs outside the default state
index, logged per decision and read by some baseline policies. The consult bit
is decision history, stamped by the controller (detectors never see decisions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Phase(IntEnum):
    """Workflow phase, detected deterministically from the execution log.

    The names match the phase marker lists in the benchmark's phases.yaml.
    """

    FRAME = 0
    EXPLORE = 1
    REASON = 2
    DELIVER = 3


class PhasePosition(IntEnum):
    """Whether a step is the first of its phase or a later one.

    Phase entry is the point where advice can still shape the approach.
    """

    START = 0
    REST = 1


class Evidence(IntEnum):
    """Quality of the current execution evidence, ordinal from best to worst.

    The controller's stall fold computes max(error, CONCEPTUAL), so CONCEPTUAL
    must stay the maximum. MECHANICAL: errors that usually self-repair (syntax,
    transient tool failures); CONCEPTUAL: a real block (recurring errors, no
    convergence). VERIFIED needs a passing check from the benchmark's marker
    tables (detectors/markers.py); without tables, clean steps read UNCHECKED.
    """

    VERIFIED = 0
    UNCHECKED = 1
    MECHANICAL = 2
    CONCEPTUAL = 3


class Commitment(IntEnum):
    """Whether a state-changing step has executed in the current plan segment.

    Set by the first completed state-changing step and reset at episode start
    and at every replan. Absorbing within a plan segment, since a detour back
    to exploration leaves the work done in place. Logged covariate only, not
    part of the default state index.
    """

    UNCOMMITTED = 0
    COMMITTED = 1


class ExecStatus(IntEnum):
    """Verdict of the last executed action (exec3 dim).

    Read only from `executed_output` bytes, never from the transcript. NONE
    means nothing has produced process output yet. An agent that prints a fake
    "clean" can only shift which cell the policy sees, never the score.
    """

    NONE = 0
    CLEAN = 1
    ERROR = 2


class ObsStatus(IntEnum):
    """Verdict of the last record that acted on the environment (outcome3 dim).

    Generalizes ExecStatus to tool-calling environments: the verdict comes from
    executed console bytes or a harness-stamped tool result. Where every acting
    record carries executed bytes, outcome3 equals exec3; in a pure tool-calling
    environment exec3 stays NONE and outcome3 carries the verdict. An
    environment's end-of-episode refusal counts as not acting, never as a failure.
    """

    NONE = 0
    OK = 1
    FAILED = 2


class Action(IntEnum):
    SILENT = 0
    CONSULT = 1


class ContactPointKind(IntEnum):
    """Structural kind of a decision moment.

    PRE_STEP      before the next workflow step is dispatched (every framework)
    PRE_DELIVERY  before the final answer is synthesized
    STALL         a stall/replan moment (stall-detecting frameworks, e.g. MagenticOne)

    Each FrameworkRunner declares its kinds in `supported_contact_points`; manifests record them.
    """

    PRE_STEP = 0
    PRE_DELIVERY = 1
    STALL = 2


@dataclass(frozen=True)
class ContactPoint:
    """A decision point between two executor steps.

    phase/position/error come from the benchmark's deterministic detector over
    the execution log, with no LLM call. consulted_before is stamped by the
    controller (decision history).
    """

    phase: Phase
    position: PhasePosition
    error: Evidence
    step_index: int
    # Detector-derived logged covariate, outside the default state index.
    commitment: Commitment = Commitment.UNCOMMITTED
    consulted_before: bool = False
    kind: ContactPointKind = ContactPointKind.PRE_STEP
    step_summary: str = ""
    # Full status snapshot forwarded to the advisor on CONSULT.
    executor_status: str = ""
    # 0-based ordinal of this decision in the episode (a controller counter the
    # executor cannot forge) and the exec3 verdict, stamped from executed bytes.
    cp_index: int = 0
    exec_status: ExecStatus = ExecStatus.NONE
    # Environment-observation verdict (outcome3 dim); see ObsStatus.
    obs_status: ObsStatus = ObsStatus.NONE


@dataclass
class Transition:
    state: int
    action: Action
    reward: float
    next_state: int  # -1 for terminal
    terminal: bool
    # Annotations for analysis (advisor assessment, state label, clamps, ...).
    # Never consumed by the learner.
    info: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """JSON-safe representation for episode logs."""
        return {
            "state": self.state,
            "action": int(self.action),
            "reward": self.reward,
            "next_state": self.next_state,
            "terminal": self.terminal,
            "info": dict(self.info),
        }


@dataclass
class RLConfig:
    """Hyperparameters of the tabular learner.

    n_states derives from the active state layout: `ExperimentConfig.rl_config()`
    sets it from `state.dims`, and AriaPolicy checks it against its layout. The
    default (24) matches the default layout, for direct construction in tests.
    """

    # gamma = 1: the consult cost is charged at the consult itself, so a
    # discount < 1 would shrink only the terminal benefit and bias the policy
    # toward late consults.
    gamma: float = 1.0
    # Epsilon anneals linearly from epsilon_start to epsilon_end over
    # epsilon_anneal_episodes, then holds (<= 0 holds at start). The 0.10 floor
    # keeps each action's propensity >= 0.05, so inverse-propensity weights for
    # off-policy re-weighting (e.g. a lambda sweep) stay <= 20.
    epsilon_start: float = 0.4
    epsilon_end: float = 0.10
    epsilon_anneal_episodes: int = 110
    # Learning rate alpha = alpha_init / n_visits(state, action); with
    # alpha_init = 1 each cell holds the exact mean of its observed returns.
    alpha_init: float = 1.0
    # Episode-flush backup. "mc" (default) gives each decision the realized
    # episode return; the coarse layout aliases many underlying states, so
    # bootstrapped 1-step targets are biased and credit late rescues to the
    # wrong cell. "onestep" keeps the 1-step bootstrapped targets.
    backup: str = "mc"  # mc | onestep
    # Literal default because state.py imports this module; `rl_config()`
    # overwrites it from the layout and AriaPolicy raises on a mismatch.
    n_states: int = 24
    n_actions: int = 2
