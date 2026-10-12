"""aria-train: train ARIA (or run a learning-relevant condition) on a benchmark.

    aria-train <experiment.yaml>
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from ..config import load_config
from ..experiments.experiment import run_experiment

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Argument(..., help="Experiment YAML"),
    secrets: Path | None = typer.Option(None, help="Secrets YAML (default: aria_secrets.yaml)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    cfg = load_config(config, secrets)
    run_dirs = asyncio.run(run_experiment(cfg, mode="train"))
    for d in run_dirs:
        typer.echo(f"run written to {d}")


if __name__ == "__main__":
    app()
