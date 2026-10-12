"""aria-eval: evaluate a condition (test split unless `run.split` is set), one row per seed.

    aria-eval <experiment.yaml> [--policy <policy_id>]
    aria-eval <experiment.yaml> --checkpoint <run_dir>/checkpoints/checkpoint_ep<NNN>.json

--policy loads a promoted artifact only if its state_signature matches the
current state layout and phase schema, so a policy for another state space fails.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..artifacts.store import PolicyStore
from ..config import load_config
from ..detectors import load_phases_config, state_signature
from ..evaluation.metrics import aggregate_run
from ..experiments.experiment import run_experiment
from ..telemetry.run_logger import RunLogger

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Argument(..., help="Experiment YAML"),
    checkpoint: Path | None = typer.Option(None, help="Raw checkpoint JSON (aria runs)"),
    policy: str | None = typer.Option(
        None, "--policy", help="Promoted policy_id from the artifact store"
    ),
    store_root: Path = typer.Option(Path("artifacts/policies"), help="Artifact store root"),
    secrets: Path | None = typer.Option(None, help="Secrets YAML (default: aria_secrets.yaml)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    cfg = load_config(config, secrets)
    if checkpoint is not None and policy is not None:
        raise typer.BadParameter("--checkpoint and --policy are mutually exclusive")
    if policy is not None:
        store = PolicyStore(store_root)
        expected = state_signature(
            load_phases_config(cfg.benchmark, cfg.teams_root), cfg.state_layout()
        )
        store.verify(policy, expected)
        cfg.condition_options["checkpoint"] = str(store.resolve(policy))
    elif checkpoint is not None:
        cfg.condition_options["checkpoint"] = str(checkpoint)
    run_dirs = asyncio.run(run_experiment(cfg, mode="eval"))

    table = Table(title=f"{cfg.benchmark} / {cfg.condition} / {cfg.framework}")
    for col in (
        "seed dir", "score", "delivery", "tokens", "adv-cost frac", "interv. rate"
    ):
        table.add_column(col)
    for d in run_dirs:
        m = aggregate_run(RunLogger.read_episodes(d))
        table.add_row(
            d.name,
            f"{m.benchmark_score_mean:.4f} +/- {m.benchmark_score_std:.4f}",
            f"{m.delivery_rate:.2%}",
            f"{m.total_tokens_mean:,.0f}",
            f"{m.advisory_cost_fraction_mean:.2%}",
            f"{m.intervention_rate_mean:.2%}",
        )
    Console().print(table)


if __name__ == "__main__":
    app()
