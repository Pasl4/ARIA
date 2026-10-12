"""aria-run: execute an experiment matrix (benchmark x condition x framework).

    aria-run <matrix.yaml> --dry-run
    aria-run <matrix.yaml> --filter benchmark=<name>
    aria-run <matrix.yaml>            # resumable: complete cells are skipped

--dry-run prints each cell's overrides with the merge layer that set each value.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer

from ..experiments.experiment import run_experiment
from ..experiments.matrix import RunSpec, describe_cell, expand, load_matrix

app = typer.Typer(add_completion=False)


def _matches(spec: RunSpec, filters: dict[str, str]) -> bool:
    values = {
        "benchmark": spec.benchmark,
        "condition": spec.condition,
        "framework": spec.framework,
    }
    return all(values.get(key) == value for key, value in filters.items())


@app.command()
def main(
    matrix_path: Path = typer.Argument(..., help="Matrix YAML (configs/experiments/)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the expansion, run nothing"),
    filters: list[str] = typer.Option(
        [], "--filter", help="Restrict cells, e.g. --filter benchmark=<name>"
    ),
    secrets: Path | None = typer.Option(None, help="Secrets YAML (default: aria_secrets.yaml)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)
    parsed_filters = dict(f.split("=", 1) for f in filters)
    unknown = set(parsed_filters) - {"benchmark", "condition", "framework"}
    if unknown:
        raise typer.BadParameter(f"unknown filter keys: {sorted(unknown)}")

    matrix = load_matrix(matrix_path)
    specs = [s for s in expand(matrix, secrets) if _matches(s, parsed_filters)]
    typer.echo(f"matrix {matrix.name!r}: {len(specs)} cells x {len(matrix.seeds)} seeds")

    for spec in specs:
        typer.echo(describe_cell(spec))
        if spec.is_complete():
            typer.echo("    -> complete, skipping")

    if dry_run:
        return

    for spec in specs:
        if spec.is_complete():
            continue
        typer.echo(f"running {spec.cell} ...")
        run_dirs = asyncio.run(run_experiment(spec.config, mode=spec.mode))
        for d in run_dirs:
            typer.echo(f"    written {d}")


if __name__ == "__main__":
    app()
