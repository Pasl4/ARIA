"""ARIA, the learned tabular policy.

Owns the value table, the exploration schedule and its own telemetry, exposed
through the InterventionPolicy protocol plus optional hooks the controller
probes (`decision_info`, `consult_visits`, `rl_config`), so nothing reaches into internals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.q_learner import TabularQPolicy
from ..core.state import DEFAULT_LAYOUT, StateLayout
from ..core.types import Action, ContactPoint, RLConfig, Transition


class AriaPolicy:
    def __init__(
        self,
        rl_config: RLConfig | None = None,
        seed: int | None = None,
        checkpoint: str | Path | None = None,
        frozen: bool = False,
        layout: StateLayout | None = None,
    ) -> None:
        cfg = rl_config or RLConfig()
        self._layout = layout or DEFAULT_LAYOUT
        self._seed = seed  # kept so a checkpoint load stays reproducible
        if cfg.n_states != self._layout.n_states:
            raise ValueError(
                f"RLConfig.n_states={cfg.n_states} does not match the state layout "
                f"{self._layout.descriptor()!r} (n_states={self._layout.n_states}); "
                "the experiment derives n_states from the layout"
            )
        self._config = cfg
        self._q = TabularQPolicy(cfg, seed=seed)
        self._td_errors_episode: list[float] = []
        self._td_errors_last_episode: list[float] = []
        self._prev_greedy: tuple[int, ...] | None = None
        self._policy_changes_last_episode = 0
        # Training episodes completed. Drives the linear epsilon schedule and
        # must survive checkpoints, or a resume restarts the anneal.
        self._episodes_seen = 0
        if checkpoint is not None:
            self.load(checkpoint)
        if frozen:
            self.freeze()

    # ------------------------------------------------------------------
    # InterventionPolicy protocol
    # ------------------------------------------------------------------

    def decide(self, contact_point: ContactPoint) -> Action:
        return self._q.select_action(self._layout.encode(contact_point))

    def decision_info(self) -> dict[str, Any]:
        """Propensity and exploration flag of the last decide().

        Logged per decision for inverse-propensity weighting.
        """
        return {
            "p_consult": self._q.last_p_consult,
            "exploratory": self._q.last_exploratory,
        }

    @property
    def rl_config(self) -> RLConfig:
        """The learner's config; the controller reads `backup` from it."""
        return self._config

    def consult_visits(self, state: int) -> float:
        """Number of CONSULT visits of a state.

        The controller's optional seeding uses it to target the least-visited
        configured cells.
        """
        return float(self._q.visit_counts[state, Action.CONSULT])

    def learn(self, transition: Transition) -> float | None:
        td = self._q.update(
            transition.state,
            transition.action,
            transition.reward,
            transition.next_state,
            transition.terminal,
        )
        self._td_errors_episode.append(td)
        return td

    def end_episode(self, eval_mode: bool = False) -> None:
        if not eval_mode:
            self._episodes_seen += 1
            if self._config.epsilon_anneal_episodes > 0:
                start, end = self._config.epsilon_start, self._config.epsilon_end
                frac = min(1.0, self._episodes_seen / self._config.epsilon_anneal_episodes)
                self._q.set_epsilon(start - (start - end) * frac)
        greedy = self._greedy_snapshot()
        self._policy_changes_last_episode = (
            sum(a != b for a, b in zip(self._prev_greedy, greedy, strict=True))
            if self._prev_greedy is not None
            else 0
        )
        self._prev_greedy = greedy
        # The runner reads the telemetry snapshot after finalize(), when the
        # live buffer is already empty; keep a copy so td_error_mean and
        # td_error_max describe this episode.
        self._td_errors_last_episode = list(self._td_errors_episode)
        self._td_errors_episode = []

    def telemetry_snapshot(self) -> dict[str, Any]:
        # After end_episode the live buffer is empty; the snapshot describes
        # the episode that just finished.
        td = self._td_errors_episode or self._td_errors_last_episode
        return {
            "epsilon": self._q.epsilon,
            "gamma": self._config.gamma,
            "td_error_mean": (sum(abs(e) for e in td) / len(td)) if td else 0.0,
            "td_error_max": max((abs(e) for e in td), default=0.0),
            "q_max_mean": float(np.mean(np.max(self._q.q, axis=1))),
            "total_visits": int(np.sum(self._q.visit_counts)),
            "policy_changes": self._policy_changes_last_episode,
            "frozen": self._q.frozen,
        }

    def state_dict(self) -> dict[str, Any]:
        # The layout descriptor (not a bare count) is the checkpoint's state
        # identity, so an equal-sized but differently arranged layout fails
        # on load instead of mis-indexing the Q-table.
        return {
            "policy": self._q.to_dict(),
            "state_version": self._layout.descriptor(),
            "episodes_seen": self._episodes_seen,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = state.get("state_version")
        if version is not None and version != self._layout.descriptor():
            raise ValueError(
                f"checkpoint state_version={version!r} != current layout "
                f"{self._layout.descriptor()!r}"
            )
        # Pass the seed: otherwise the exploration RNG is re-seeded from the
        # OS on every resume and a seeded run is not reproducible.
        self._q = TabularQPolicy.from_dict(state["policy"], seed=self._seed)
        self._episodes_seen = int(state.get("episodes_seen", 0))

    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Greedy evaluation mode: no updates, epsilon = 0."""
        self._q.freeze()

    @property
    def epsilon(self) -> float:
        return self._q.epsilon

    @property
    def gamma(self) -> float:
        return float(self._config.gamma)

    @property
    def q_table(self) -> np.ndarray:
        """The live Q-table array for analysis and plots (not part of the protocol)."""
        return self._q.q

    def _greedy_snapshot(self) -> tuple[int, ...]:
        return tuple(int(a) for a in np.argmax(self._q.q, axis=1).tolist())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    def load(self, path: str | Path) -> None:
        """Load from a save()/checkpoint JSON file."""
        p = Path(path)
        payload = json.loads(p.read_text())
        if "policy" in payload:
            self.load_state_dict(payload)
        elif "config" in payload:  # bare TabularQPolicy.to_dict() payload
            self._q = TabularQPolicy.from_dict(payload, seed=self._seed)
        else:
            raise ValueError(f"cannot interpret checkpoint at {p}")
        # Bare payloads carry no state_version; guard the shape against the
        # active layout so a foreign table cannot mis-index silently.
        if self._q.q.shape[0] != self._layout.n_states:
            raise ValueError(
                f"checkpoint has {self._q.q.shape[0]} states, layout "
                f"{self._layout.descriptor()!r} needs {self._layout.n_states}"
            )
