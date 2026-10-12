"""Common interface for all intervention policies (learned and baselines).

Only the policy object differs between conditions, except that "organic" also
enables the consult tool. The InterventionController computes rewards with one
RewardModel for all conditions and assembles transitions; a policy chooses
SILENT or CONSULT, consumes finished Transitions via ``learn`` if it learns,
and gets logging and checkpoints by implementing ``telemetry_snapshot`` and
``state_dict``.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..core.types import Action, ContactPoint, Transition


class InterventionPolicy(Protocol):
    def decide(self, contact_point: ContactPoint) -> Action:
        """Choose SILENT or CONSULT at a contact point."""
        ...

    def learn(self, transition: Transition) -> float | None:
        """Consume a finished transition.

        Returns the TD error for learned policies, None for static baselines.
        """
        ...

    def end_episode(self, eval_mode: bool = False) -> None:
        """Reset per-episode state and, unless evaluating, decay exploration."""
        ...

    def telemetry_snapshot(self) -> dict[str, Any] | None:
        """Learner telemetry for the episode log (None for static baselines)."""
        ...

    def state_dict(self) -> dict[str, Any] | None:
        """Checkpointable state (None for stateless policies)."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from a state_dict() checkpoint."""
        ...


class StaticPolicyBase:
    """Shared no-op learning/telemetry interface for non-learning baselines."""

    def learn(self, transition: Transition) -> float | None:
        return None

    def end_episode(self, eval_mode: bool = False) -> None:
        return None

    def telemetry_snapshot(self) -> dict[str, Any] | None:
        return None

    def state_dict(self) -> dict[str, Any] | None:
        return None

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise ValueError(f"{type(self).__name__} has no checkpointable state")
