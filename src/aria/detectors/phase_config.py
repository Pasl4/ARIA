"""Phase schema loaded from ``<teams_root>/<benchmark>/phases.yaml``.

The file sets the phase markers, the conceptual-error streak, the classifier (only
``heuristic``) and the evidence marker tables (schema: docs/configuration.md). Run
manifests and promoted policy cards carry ``state_signature`` (layout descriptor
plus phase_signature()), so loading a promoted policy trained under a different
schema or layout fails. A raw checkpoint is checked only against the layout.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ..core.state import DEFAULT_LAYOUT, StateLayout
from ..core.types import Phase
from .base import ContactPointDetector
from .heuristic import CONCEPTUAL_STREAK, HeuristicDetector
from .markers import MarkerTables

DEFAULT_TEAMS_ROOT = Path("configs/teams")
# Detection is deterministic and "heuristic" is the only classifier. The field
# stays in the schema and the signature so a config requesting another one
# fails instead of being silently ignored.
CLASSIFIERS = ("heuristic",)
# Bump when predicate semantics change. Version 2 judges evidence_of_run and
# pass_requires per execution segment.
PREDICATE_VERSION = 2


@dataclass(frozen=True)
class PhasesConfig:
    benchmark: str
    markers: dict[Phase, tuple[str, ...]]
    conceptual_streak: int = CONCEPTUAL_STREAK
    classifier: str = "heuristic"
    # Evidence marker tables, frozen and hashed like the phase markers.
    state_changing: tuple[str, ...] = ()
    verification_evidence: tuple[str, ...] = ()
    verification_pass: tuple[str, ...] = ()

    def canonical(self) -> dict[str, object]:
        """Return a deterministic, JSON-safe representation for the signature.

        The classifier and the evidence marker tables are included because
        states produced under different ones are not comparable.
        """
        return {
            "benchmark": self.benchmark,
            "classifier": self.classifier,
            "conceptual_streak": self.conceptual_streak,
            "markers": {p.name: sorted(m) for p, m in sorted(self.markers.items())},
            "state_changing": sorted(self.state_changing),
            "verification": {
                "evidence_of_run": sorted(self.verification_evidence),
                "pass_requires": sorted(self.verification_pass),
            },
            # Hashes the predicate semantics as well as the regex tables, so a
            # semantics change cannot reuse an old state_signature.
            "predicate_version": PREDICATE_VERSION,
        }

    def marker_tables(self) -> MarkerTables:
        """Return the frozen tables in the form the translator consumes."""
        return MarkerTables(
            state_changing=self.state_changing,
            verification_evidence=self.verification_evidence,
            verification_pass=self.verification_pass,
        )


def load_phases_config(
    benchmark: str, teams_root: str | Path = DEFAULT_TEAMS_ROOT
) -> PhasesConfig:
    path = Path(teams_root) / benchmark / "phases.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    # Strict schema: a silently ignored key would bypass the signature freeze.
    allowed = {"classifier", "conceptual_streak", "markers", "state_changing", "verification"}
    unknown_keys = set(raw) - allowed
    if unknown_keys:
        raise ValueError(f"{path}: unknown top-level keys {sorted(unknown_keys)}")
    classifier = str(raw.get("classifier", "heuristic"))
    if classifier not in CLASSIFIERS:
        raise ValueError(f"{path}: unknown classifier {classifier!r}; expected {CLASSIFIERS}")
    raw_markers: dict[str, list[str]] = raw.get("markers") or {}
    unknown = set(raw_markers) - {p.name.lower() for p in Phase}
    if unknown:
        raise ValueError(f"{path}: unknown phases {sorted(unknown)}")
    markers = {
        phase: tuple(raw_markers.get(phase.name.lower(), ())) for phase in Phase
    }
    state_changing = raw.get("state_changing") or {}
    verification = raw.get("verification") or {}
    return PhasesConfig(
        benchmark=benchmark,
        markers=markers,
        conceptual_streak=int(raw.get("conceptual_streak", CONCEPTUAL_STREAK)),
        classifier=classifier,
        state_changing=tuple(state_changing.get("patterns", ())),
        verification_evidence=tuple(verification.get("evidence_of_run", ())),
        verification_pass=tuple(verification.get("pass_requires", ())),
    )


def phase_signature(config: PhasesConfig) -> str:
    """sha256 over the canonical phase schema (stable across dict ordering)."""
    payload = json.dumps(config.canonical(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def state_signature(config: PhasesConfig, layout: StateLayout = DEFAULT_LAYOUT) -> str:
    """Return the state-space identity a trained policy is bound to.

    Layout descriptor (dims and sizes; a bare state count cannot tell two
    layouts of equal size apart) plus the phase-schema hash.
    """
    return f"{layout.descriptor()}:{phase_signature(config)}"


def build_detector(config: PhasesConfig) -> ContactPointDetector:
    """Build the deterministic marker detector for a phase config."""
    return HeuristicDetector(
        phase_markers=config.markers,
        conceptual_streak=config.conceptual_streak,
    )
