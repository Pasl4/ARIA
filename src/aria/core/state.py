"""State layout: an ordered list of registry dims, encoded as one index.

The layout comes from `state.dims`; its descriptor is part of `state_signature`
and `n_states` derives from it. Dims: position4 (`_position4`), exec3
(`ExecStatus`), outcome3 (`ObsStatus`), consult (consulted earlier this episode);
default position4 x exec3 x consult (24 cells). No dim uses a keyword table; the
default dims read no model text, outcome3 can (see ARIA_OBS_FALLBACK).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .types import ContactPoint, ContactPointKind, ExecStatus, ObsStatus

_POSITION4_NAMES = ("FRAME", "EXPLORE", "REASON", "DELIVER")


def _position4(cp: ContactPoint) -> int:
    """Position of the decision within the episode (position4 dim).

    Built from the controller's decision counter and the pre-delivery event,
    which the executor cannot forge. Values follow `_POSITION4_NAMES`: first
    decision, 2nd-3rd, 4th and later, pre-delivery gate. The names mirror Phase
    but are purely positional; no keyword markers are involved.
    """
    if cp.kind is ContactPointKind.PRE_DELIVERY:
        return 3
    if cp.cp_index == 0:
        return 0
    return 1 if cp.cp_index <= 2 else 2


@dataclass(frozen=True)
class _Dim:
    size: int
    extract: Callable[[ContactPoint], int]
    value_name: Callable[[int], str]


DIM_REGISTRY: dict[str, _Dim] = {
    "position4": _Dim(4, _position4, lambda v: _POSITION4_NAMES[v]),
    "exec3": _Dim(
        len(ExecStatus), lambda cp: int(cp.exec_status), lambda v: ExecStatus(v).name
    ),
    "outcome3": _Dim(
        len(ObsStatus), lambda cp: int(cp.obs_status), lambda v: ObsStatus(v).name
    ),
    "consult": _Dim(
        2, lambda cp: int(cp.consulted_before), lambda v: "YES" if v else "NO"
    ),
}


@dataclass(frozen=True)
class StateLayout:
    """An ordered dimension list; everything else derives from it."""

    dims: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = [d for d in self.dims if d not in DIM_REGISTRY]
        if unknown:
            raise ValueError(
                f"unknown state dims {unknown}; known: {sorted(DIM_REGISTRY)}"
            )
        if len(set(self.dims)) != len(self.dims):
            raise ValueError(f"duplicate state dims in {self.dims}")
        if not self.dims:
            raise ValueError("state layout needs at least one dim")

    @property
    def n_states(self) -> int:
        n = 1
        for d in self.dims:
            n *= DIM_REGISTRY[d].size
        return n

    def descriptor(self) -> str:
        """Layout identity for signatures and checkpoints.

        Each dim name is followed by its size, e.g.
        'position44.exec33.consult2', so layouts with equal n_states but
        different dims stay distinct.
        """
        return ".".join(f"{d}{DIM_REGISTRY[d].size}" for d in self.dims)

    def encode(self, cp: ContactPoint) -> int:
        state = 0
        for d in self.dims:
            dim = DIM_REGISTRY[d]
            state = state * dim.size + dim.extract(cp)
        return state

    def decode(self, state: int) -> dict[str, int]:
        if not 0 <= state < self.n_states:
            raise ValueError(f"state index out of range: {state}")
        values: dict[str, int] = {}
        for d in reversed(self.dims):
            size = DIM_REGISTRY[d].size
            values[d] = state % size
            state //= size
        return values

    def label(self, state: int) -> str:
        """Human-readable label for plots, e.g. 'DELIVER/ERROR/YES'."""
        values = self.decode(state)
        return "/".join(DIM_REGISTRY[d].value_name(values[d]) for d in self.dims)


DEFAULT_DIMS: tuple[str, ...] = ("position4", "exec3", "consult")
DEFAULT_LAYOUT = StateLayout(DEFAULT_DIMS)
N_STATES = DEFAULT_LAYOUT.n_states  # 24, the default layout's size
