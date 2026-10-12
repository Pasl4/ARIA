"""aria-preflight: one short real call per model role to check credentials before a run.

    aria-preflight [<config.yaml>] [--role executor|advisor|all] [--secrets <secrets.yaml>]

Every role goes through `build_model_client` against its `base_url`, so a wrong
`base_url`, a key without model access or a model id AutoGen does not know (set
`model_info`) fails here. Runs build the advisor with OpenAICompatibleClient, so
an advisor model without `model_info` can fail here but not in a run.
`aria-run` does not call this.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import typer
import yaml
from autogen_core.models import UserMessage
from rich.console import Console
from rich.table import Table

from ..config import merge_secrets
from ..frameworks.autogen.clients import build_model_client

app = typer.Typer(add_completion=False)
console = Console()

PROMPT = "Reply with the single word: ok"


async def check_role(role: str, model_cfg: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"role": role, "model": model_cfg.get("model", "?")}
    if not model_cfg.get("api_key"):
        result["status"] = "NO KEY"
        result["detail"] = "no api_key; is aria_secrets.yaml present and filled?"
        return result
    try:
        client = build_model_client(model_cfg)
    except Exception as exc:  # model_info missing, bad base_url, ...
        result["status"] = "CONFIG"
        result["detail"] = str(exc)[:160]
        return result

    started = time.monotonic()
    try:
        response = await client.create([UserMessage(content=PROMPT, source="user")])
    except Exception as exc:
        result["status"] = "FAIL"
        result["detail"] = f"{type(exc).__name__}: {str(exc)[:140]}"
        return result
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    usage = response.usage
    result["status"] = "OK"
    result["latency_s"] = f"{time.monotonic() - started:.2f}"
    result["tokens"] = (
        f"{int(getattr(usage, 'prompt_tokens', 0) or 0)}"
        f"/{int(getattr(usage, 'completion_tokens', 0) or 0)}"
    )
    result["detail"] = str(response.content)[:60].replace("\n", " ")
    return result


@app.command()
def main(
    config: Path = typer.Argument(
        Path("configs/base/default.yaml"), help="YAML with a models block"
    ),
    role: str = typer.Option("all", help="executor | advisor | all"),
    secrets: Path | None = typer.Option(None, help="Secrets YAML (default: aria_secrets.yaml)"),
) -> None:
    raw = yaml.safe_load(config.read_text()) or {}
    merge_secrets(raw, secrets)  # same key resolution a real run performs
    models: dict[str, Any] = raw.get("models", {})
    if not models:
        raise typer.BadParameter(f"{config} has no `models:` block")
    roles = list(models) if role == "all" else [role]
    unknown = [r for r in roles if r not in models]
    if unknown:
        raise typer.BadParameter(f"unknown role(s) {unknown}; {config} has {list(models)}")

    results = [asyncio.run(check_role(r, models[r])) for r in roles]

    table = Table("role", "model", "status", "latency", "tok in/out", "detail")
    for r in results:
        table.add_row(
            r["role"],
            r["model"],
            f"[green]{r['status']}[/green]" if r["status"] == "OK" else f"[red]{r['status']}[/red]",
            r.get("latency_s", "-"),
            r.get("tokens", "-"),
            r.get("detail", ""),
        )
    console.print(table)

    # Pricing is checked too: with a non-empty pricing block a run refuses to
    # start on a missing or `tbd` price (an empty block prices every call at 0).
    pricing = raw.get("pricing", {})
    unpriced = [
        m["model"] for m in models.values() if str(pricing.get(m.get("model"), "tbd")) == "tbd"
    ]
    if unpriced:
        console.print(f"[yellow]pricing missing or `tbd` for: {', '.join(unpriced)}[/yellow]")

    if any(r["status"] != "OK" for r in results):
        raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()
