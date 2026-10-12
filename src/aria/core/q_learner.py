"""Tabular value table with epsilon-greedy action selection.

Each cell holds the running mean of its observed returns (alpha = 1/n). Under
the default "mc" backup the controller's episode flush passes realized
returns as terminal rewards, so `update` is an every-visit Monte Carlo mean.
The state space is not fixed here; its size comes from the active
`StateLayout` via `RLConfig.n_states`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

import numpy as np

from .types import Action, RLConfig

logger = logging.getLogger(__name__)


class TabularQPolicy:
    def __init__(self, config: RLConfig, seed: int | None = None) -> None:
        self._cfg = config
        self.q = np.zeros((config.n_states, config.n_actions), dtype=np.float64)
        self.visit_counts = np.zeros((config.n_states, config.n_actions), dtype=np.float64)
        self.epsilon = config.epsilon_start
        self._rng = np.random.default_rng(seed)
        self._frozen = False
        # Consult propensity of the last select_action call, logged for
        # inverse-propensity weighting. Computed before the draw, so it
        # describes the sampling distribution, not the sampled outcome.
        self.last_p_consult: float = 0.0
        self.last_exploratory: bool = False

    @property
    def config(self) -> RLConfig:
        return self._cfg

    def select_action(self, state: int) -> Action:
        greedy = Action(int(np.argmax(self.q[state])))
        self.last_p_consult = self.epsilon / self._cfg.n_actions + (
            (1.0 - self.epsilon) if greedy is Action.CONSULT else 0.0
        )
        if self._rng.random() < self.epsilon:
            self.last_exploratory = True
            return Action(int(self._rng.integers(0, self._cfg.n_actions)))
        self.last_exploratory = False
        return greedy

    def update(
        self,
        state: int,
        action: Action,
        reward: float,
        next_state: int,
        terminal: bool,
    ) -> float:
        """Move Q[state, action] toward the target and return the TD error.

        No-op when frozen. alpha = alpha_init / n_visits, so with
        alpha_init = 1 the cell is the exact sample mean of its targets.
        """
        if self._frozen:
            return 0.0

        self.visit_counts[state, action] += 1
        alpha = self._cfg.alpha_init / max(self.visit_counts[state, action], 1.0)

        if terminal or next_state < 0:
            td_target = reward
        else:
            td_target = reward + self._cfg.gamma * float(np.max(self.q[next_state]))

        td_error = td_target - self.q[state, action]
        self.q[state, action] += alpha * td_error
        return td_error

    def set_epsilon(self, epsilon: float) -> None:
        """Manually set exploration rate (clamped to [0, 1])."""
        self.epsilon = float(max(0.0, min(1.0, epsilon)))

    def freeze(self) -> None:
        """Freeze policy: disable updates and exploration (greedy only)."""
        self._frozen = True
        self.epsilon = 0.0

    @property
    def frozen(self) -> bool:
        return self._frozen

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.q.tolist(),
            "visit_counts": self.visit_counts.tolist(),
            "epsilon": self.epsilon,
            "frozen": self._frozen,
            "config": asdict(self._cfg),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], seed: int | None = None) -> TabularQPolicy:
        # Drop config keys RLConfig does not define; they configure options
        # this learner does not implement.
        known = set(RLConfig.__dataclass_fields__)
        config = RLConfig(**{k: v for k, v in data["config"].items() if k in known})
        policy = cls(config, seed=seed)
        q = np.array(data["q"], dtype=np.float64)
        visits = np.array(data["visit_counts"], dtype=np.float64)
        expected = (config.n_states, config.n_actions)
        # Reject checkpoints from a different state layout instead of loading
        # a table whose indices mean something else.
        if q.shape != expected or visits.shape != expected:
            raise ValueError(
                f"checkpoint shape mismatch: q={q.shape}, visits={visits.shape}, "
                f"config expects {expected}; refusing to load a checkpoint from "
                f"a different state-space version"
            )
        policy.q = q
        policy.visit_counts = visits
        policy.epsilon = float(data.get("epsilon", config.epsilon_start))
        policy._frozen = bool(data.get("frozen", False))
        return policy

