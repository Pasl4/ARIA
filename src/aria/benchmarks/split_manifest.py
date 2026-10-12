"""Committed train/test split manifests, shared by all adapters.

One implementation keeps adapters from drifting apart, which matters because a
contaminated split still produces plausible results. `read` checks that the
manifest exists (a split is committed, never sampled at runtime, so a missing
file is an error), that train and test do not overlap, and that the recorded
counts match the id lists, which catches hand-edits.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write(
    path: str | Path,
    dataset: str,
    rule: str,
    train: Iterable[str],
    test: Iterable[str],
    **extra: Any,
) -> dict[str, Any]:
    """Write a manifest, refusing an overlapping split."""
    train_ids, test_ids = sorted(set(train)), sorted(set(test))
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"train/test overlap: {sorted(overlap)[:5]}")
    manifest: dict[str, Any] = {
        "dataset": dataset,
        "rule": rule,
        "counts": {"train": len(train_ids), "test": len(test_ids)},
        **extra,
        "train": train_ids,
        "test": test_ids,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read(path: str | Path, generator: str) -> dict[str, Any]:
    """Load and validate a committed manifest.

    `generator` names the function that would create it, so the error tells a
    reader what to run rather than only what is missing.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(
            f"no committed split at {file}; generate it once with {generator} and commit it"
        )
    manifest = json.loads(file.read_text())
    if set(manifest["train"]) & set(manifest["test"]):
        raise ValueError(f"split manifest {file} has overlapping train/test ids")
    counts = manifest.get("counts")
    if counts and (
        counts.get("train") != len(manifest["train"])
        or counts.get("test") != len(manifest["test"])
    ):
        raise ValueError(
            f"split manifest {file}: counts {counts} disagree with the id lists "
            f"({len(manifest['train'])}/{len(manifest['test'])}) (hand-edited?)"
        )
    return manifest


def select(manifest: Mapping[str, Any], split: str) -> set[str]:
    """The id set for one split, with the split name validated."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be train|test, got {split!r}")
    return set(manifest[split])
