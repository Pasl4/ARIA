"""aria-policy: manage trained policy artifacts in the PolicyStore.

    aria-policy promote results/<benchmark>/aria/autogen/train/<executor_model>/seed_0
    aria-policy list
    aria-policy show <policy_id>

Promotion writes a card (source run, config hash, state_signature) that aria-eval verifies.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..artifacts.store import PolicyStore

app = typer.Typer(add_completion=False)


@app.command()
def promote(
    run_dir: Path = typer.Argument(..., help="A completed training run directory"),
    policy_id: str | None = typer.Option(None, "--id", help="Override the derived policy id"),
    store_root: Path = typer.Option(Path("artifacts/policies"), help="Artifact store root"),
) -> None:
    card = PolicyStore(store_root).promote(run_dir, policy_id)
    typer.echo(f"promoted {card.policy_id} -> {store_root}/{card.family}/{card.benchmark}/")


@app.command("list")
def list_policies(
    store_root: Path = typer.Option(Path("artifacts/policies"), help="Artifact store root"),
) -> None:
    cards = PolicyStore(store_root).list_cards()
    table = Table(title=f"{len(cards)} policy artifacts")
    for col in ("policy_id", "family", "benchmark", "framework", "gamma", "seed", "episodes"):
        table.add_column(col)
    for card in cards:
        table.add_row(
            card.policy_id, card.family, card.benchmark, card.framework,
            str(card.gamma), str(card.seed), str(card.episodes_trained),
        )
    Console().print(table)


@app.command()
def show(
    policy_id: str,
    store_root: Path = typer.Option(Path("artifacts/policies"), help="Artifact store root"),
) -> None:
    card = PolicyStore(store_root).get_card(policy_id)
    for key, value in card.__dict__.items():
        typer.echo(f"{key:>26}: {value}")


if __name__ == "__main__":
    app()
