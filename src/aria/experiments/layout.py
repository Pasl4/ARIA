"""Results layout: the single definition of the on-disk path scheme.

Paths are built only via `run_dir` and `analysis_dir`; RunLogger names the files:
    results/<benchmark>/<condition>/<framework>/<mode>/[<executor_model>/]seed_<n>/
        run_manifest.json  episodes.jsonl  calls.jsonl  cost_summary.json
        traces/episode_*.json  checkpoints/checkpoint_ep*.json
    results/<benchmark>/analysis/
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_RESULTS_ROOT = Path("results")


def run_dir(
    results_root: str | Path,
    benchmark: str,
    condition: str,
    framework: str,
    mode: str,
    seed: int,
    executor_model: str = "",
) -> Path:
    """Return the directory that holds one run's episodes.

    `executor_model` is part of the path so runs that differ only in
    `models.executor.model` get separate directories and resume never mixes
    two models in one episodes.jsonl. "/" in the model id becomes "__"; an
    empty string omits this level.
    """
    base = Path(results_root) / benchmark / condition / framework / mode
    if executor_model:
        base = base / executor_model.replace("/", "__")
    return base / f"seed_{seed}"


def analysis_dir(results_root: str | Path, benchmark: str) -> Path:
    return Path(results_root) / benchmark / "analysis"


def is_run_complete(directory: Path) -> bool:
    """Return True when the run's manifest and cost summary both exist.

    The experiment loop writes the cost summary last.
    """
    return (directory / "run_manifest.json").exists() and (
        directory / "cost_summary.json"
    ).exists()
