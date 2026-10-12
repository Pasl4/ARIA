"""Framework-neutral team specification, shared by every framework runner.

Agents (roster, prompts, tools, params) live under a teams root, by default
`configs/teams/` in the working directory. `_defaults/agents.yaml` and
`_defaults/prompts/` are overridden per benchmark by `<benchmark>/team.yaml`
and `<benchmark>/prompts/`. Placeholders are filled strictly from the adapter's
`prompt_vars(task)` and runner vars such as {date_today}; a missing value raises.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TEAMS_ROOT = Path("configs/teams")
DEFAULTS_DIR = "_defaults"

AGENT_KINDS = ("coder", "file_surfer", "assistant", "code_executor")


class _StrictFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):  # type: ignore[override]
        if isinstance(key, str) and key not in kwargs:
            raise KeyError(
                f"prompt template needs variable {key!r} but it was not provided"
            )
        return super().get_value(key, args, kwargs)


_FORMATTER = _StrictFormatter()


@dataclass(frozen=True)
class AgentSpec:
    name: str
    kind: str  # one of AGENT_KINDS; runners map kind -> native agent type
    description: str = ""  # used by orchestrators for speaker selection
    prompt_template: str = ""  # raw template text (already resolved from files)
    model_role: str = "executor"  # key into ExperimentConfig.models; the AutoGen runner ignores it
    tools: tuple[str, ...] = ()  # symbolic names, resolved via aria.frameworks.tools
    params: Mapping[str, Any] = field(default_factory=dict)

    def render_prompt(self, variables: Mapping[str, Any]) -> str:
        """Render the prompt template; every placeholder must have a value."""
        return _FORMATTER.vformat(self.prompt_template, (), dict(variables))


@dataclass(frozen=True)
class RuntimeKnobs:
    """Cross-framework runtime knobs plus per-framework overrides."""

    common: Mapping[str, Any] = field(default_factory=dict)
    per_framework: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def for_framework(self, name: str) -> dict[str, Any]:
        merged = dict(self.common)
        merged.update(self.per_framework.get(name, {}))
        return merged


@dataclass(frozen=True)
class TeamSpec:
    benchmark: str
    agents: tuple[AgentSpec, ...]
    runtime: RuntimeKnobs

    def agent(self, name: str) -> AgentSpec:
        for a in self.agents:
            if a.name == name:
                return a
        raise KeyError(f"no agent {name!r} in team for {self.benchmark}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _resolve_prompt(name: str, entry: dict[str, Any], bench_dir: Path, root: Path) -> str:
    """Return the prompt text for one agent.

    Benchmark `prompts/` overrides `_defaults/prompts/`. An explicit `prompt:`
    path is tried relative to the benchmark directory, then to `_defaults/`.
    """
    explicit = entry.get("prompt")
    candidates = (
        [bench_dir / explicit, root / DEFAULTS_DIR / explicit]
        if explicit
        else [
            bench_dir / "prompts" / f"{name}.md",
            root / DEFAULTS_DIR / "prompts" / f"{name}.md",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text()
    if entry.get("kind") == "code_executor":  # executors have no prompt
        return ""
    raise FileNotFoundError(
        f"no prompt found for agent {name!r}; looked at {[str(c) for c in candidates]}"
    )


def load_team_spec(
    benchmark: str, teams_root: str | Path = DEFAULT_TEAMS_ROOT
) -> TeamSpec:
    root = Path(teams_root)
    bench_dir = root / benchmark
    defaults = _read_yaml(root / DEFAULTS_DIR / "agents.yaml").get("agents", {})
    team_raw = _read_yaml(bench_dir / "team.yaml")

    if team_raw.get("benchmark", benchmark) != benchmark:
        raise ValueError(
            f"{bench_dir / 'team.yaml'} declares benchmark "
            f"{team_raw.get('benchmark')!r}, expected {benchmark!r}"
        )

    roster: list[str] = list(team_raw.get("agents", []))
    if not roster:
        raise ValueError(f"{bench_dir / 'team.yaml'} defines no agents")
    overrides: dict[str, dict[str, Any]] = team_raw.get("overrides", {}) or {}
    unknown_overrides = set(overrides) - set(roster)
    if unknown_overrides:
        raise ValueError(
            f"{bench_dir / 'team.yaml'} overrides unknown agents {sorted(unknown_overrides)}"
        )

    agents: list[AgentSpec] = []
    for name in roster:
        entry: dict[str, Any] = dict(defaults.get(name, {}))
        entry.update(overrides.get(name, {}))
        if "kind" not in entry:
            raise ValueError(
                f"agent {name!r} has no `kind` (not in _defaults/agents.yaml and "
                f"not overridden in {bench_dir / 'team.yaml'})"
            )
        if entry["kind"] not in AGENT_KINDS:
            raise ValueError(f"agent {name!r}: unknown kind {entry['kind']!r}")
        agents.append(
            AgentSpec(
                name=name,
                kind=entry["kind"],
                description=str(entry.get("description", "")).strip(),
                prompt_template=_resolve_prompt(name, entry, bench_dir, root),
                model_role=str(entry.get("model_role", "executor")),
                tools=tuple(entry.get("tools", ())),
                params=dict(entry.get("params", {})),
            )
        )

    runtime_raw: dict[str, Any] = team_raw.get("runtime", {}) or {}
    runtime = RuntimeKnobs(
        common=dict(runtime_raw.get("common", {})),
        per_framework={
            k: dict(v) for k, v in runtime_raw.items() if k != "common"
        },
    )
    return TeamSpec(benchmark=benchmark, agents=tuple(agents), runtime=runtime)
