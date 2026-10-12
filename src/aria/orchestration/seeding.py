"""Deficit-targeted consult seeding.

When enabled, every Nth episode forces one consult at the first entry into the
designated cell with the largest visit deficit, so rarely consulted cells
(e.g. an interior state after an error) become estimable; blanket forcing
would inflate consult Q-values everywhere. Seeded decisions enter Q-updates
and are flagged (`seeded`, `seeded_cell`) so off-policy (IPW) estimates can exclude them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# These prefixes match no label of the default layout, so with that layout the
# cell-targeted seed fires only if `seeding.cells` lists prefixes of its labels
# (see `StateLayout.label`). The second gate consult does not depend on cells.
DEFAULT_CELLS: tuple[str, ...] = (
    "INTERIOR/ERROR",
    "GATE/CLEAN/YES",
    "GATE/ERROR",
)


@dataclass(frozen=True)
class SeedingConfig:
    enabled: bool = False
    # Fire in episodes where (1-based episode number) % every == 0.
    every: int = 5
    # Deficit = target - consult-visits summed over a cell spec's states.
    deficit_target: int = 12
    # Label prefixes of the active layout; a spec without the consult part
    # matches both consult values.
    cells: tuple[str, ...] = DEFAULT_CELLS
    # Seeded episodes also force a second consult at the gate when the
    # episode arrives there already-consulted and under the cap, a decision
    # that is otherwise rarely observed.
    second_consult_gate: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> SeedingConfig:
        if not raw:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown seeding keys {sorted(unknown)}; known: {sorted(known)}")
        cleaned = dict(raw)
        if "cells" in cleaned:
            cleaned["cells"] = tuple(cleaned["cells"])
        return cls(**cleaned)


@dataclass
class SeedingState:
    """Per-episode seeding bookkeeping, reset by the controller."""

    active: bool = False
    second_gate: bool = False
    used: bool = False
    designated: tuple[str, ...] = field(default_factory=tuple)  # deficit-ordered

    def begin(
        self, cfg: SeedingConfig, episode_number: int, deficits: list[tuple[float, str]]
    ) -> None:
        self.active = cfg.enabled and cfg.every > 0 and episode_number % cfg.every == 0
        # With second_consult_gate, every seeded episode gets the gate follow-up:
        # a gate cell with consult=YES is reachable only after a prior consult.
        self.second_gate = self.active and cfg.second_consult_gate
        self.used = False
        # Highest deficit first; ties keep the configured cell order.
        self.designated = tuple(spec for _, spec in sorted(deficits, key=lambda d: -d[0]))

    def wants_consult(self, label: str, is_gate: bool) -> str | None:
        """Return the cell spec to credit if this decision is seed-forced, else None.

        Fires at the first entry into the top-deficit cell. At the gate (the
        episode's last decision) any other designated cell is accepted as a
        fallback, so the seed is not lost when the top cell never occurs.
        """
        if not self.active or self.used or not self.designated:
            return None
        top = self.designated[0]
        if label.startswith(top):
            return top
        if is_gate:
            for spec in self.designated[1:]:
                if label.startswith(spec):
                    return spec
        return None
