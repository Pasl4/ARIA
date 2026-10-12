"""Run logging: per-episode JSONL, traces, call records, checkpoints and a manifest.

A run directory (see experiments/layout.py) holds run_manifest.json (settings,
git commit, config without API keys), episodes.jsonl (one EpisodeStats per
line, append-only), episodes.dropped.jsonl (rows removed on resume),
traces/episode_*.json, calls.jsonl (per-call token records),
cost_summary.json and checkpoints/*.json (learned-policy state_dicts).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class EpisodeStats:
    episode: int
    task_id: str
    condition: str
    benchmark: str
    framework: str
    seed: int
    # Outcome
    benchmark_score: float
    delivered: bool
    stop_reason: str  # completed | max_rounds | crashed
    terminal_reward: float
    total_reward: float
    final_answer: str
    # Decision behaviour
    steps_taken: int
    n_consults: int  # controller-decided consults
    n_organic_consults: int  # executor-decided consults (condition "organic")
    intervention_rate: float  # (n_consults + n_organic) / steps
    # Cost totals; the per-role/per-purpose breakdown
    # (TokenMeter.episode_summary()) is in `usage`, per-call records in calls.jsonl.
    total_tokens: int
    cost_usd: float
    advisory_cost_fraction: float
    advisor_failures: int
    # Wall-clock episode duration in seconds.
    duration_s: float = 0.0
    # Wall-clock budget of the agent phase: the run ceiling, or the benchmark's
    # per-task timeout capped at it. 0.0 means the budget was not recorded.
    agent_timeout_s: float = 0.0
    # The workspace git diff is logged next to the typed answer; delivery_channel
    # names the one the harness scored ("workspace_patch" | "typed").
    workspace_patch: str = ""
    delivery_channel: str = "typed"
    # Relative path of this episode's trace file, "" if not written.
    trace_file: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    # False when the harness could not judge or the episode crashed: missing
    # data, excluded from score aggregates rather than averaged in as a zero.
    scored: bool = True
    # Learner telemetry from policy.telemetry_snapshot(), None for static baselines.
    # A dict keeps the JSONL schema stable across policy types and state spaces.
    learner: dict[str, Any] | None = None
    # Per-decision transitions (with info annotations) for offline analysis
    transitions: list[dict[str, Any]] = field(default_factory=list)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


class RunLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episodes_path = self.output_dir / "episodes.jsonl"

    def write_manifest(self, **manifest: Any) -> Path:
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "git_commit": _git_commit(),
            **manifest,
        }
        path = self.output_dir / "run_manifest.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def log_episode(self, stats: EpisodeStats) -> None:
        with self._episodes_path.open("a") as f:
            f.write(json.dumps(dataclasses.asdict(stats), default=str) + "\n")

    def log_calls(self, calls: list[Any]) -> None:
        """Append per-call token records (CallRecord) to calls.jsonl."""
        if not calls:
            return
        with (self.output_dir / "calls.jsonl").open("a") as f:
            for call in calls:
                f.write(json.dumps(call.as_dict(), default=str) + "\n")

    def write_cost_summary(self, summary: dict[str, Any]) -> Path:
        """Write the whole-run token and cost roll-up (TokenMeter.run_summary())."""
        path = self.output_dir / "cost_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        return path

    def write_trace(self, episode: int, trace: dict[str, Any]) -> Path:
        """Write one full episode trace to traces/episode_<n>.json."""
        trace_dir = self.output_dir / "traces"
        trace_dir.mkdir(exist_ok=True)
        path = trace_dir / f"episode_{episode:03d}.json"
        path.write_text(json.dumps(trace, indent=2, default=str))
        return path

    def save_checkpoint(self, *, episode: int, policy_state: dict[str, Any]) -> Path:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        path = checkpoint_dir / f"checkpoint_ep{episode:03d}.json"
        path.write_text(json.dumps({"episode": episode, **policy_state}, indent=2))
        return path

    @staticmethod
    def keep_episodes(run_dir: str | Path, keep: set[int]) -> None:
        """Rewrite episodes.jsonl to exactly the episodes in `keep`, for resume.

        Rows not in `keep` are about to be re-run (episodes after the resumed
        checkpoint, or crashed episodes it does not cover); keeping them would
        count retried episodes twice. The last row per index wins.
        """
        path = Path(run_dir) / "episodes.jsonl"
        if not path.exists():
            return
        rows: dict[int, str] = {}
        dropped: list[str] = []
        for line in path.read_text().splitlines():
            if not line:
                continue
            episode = json.loads(line).get("episode")
            if episode in keep:
                rows[int(episode)] = line
            else:
                dropped.append(line)
        # Append dropped rows to episodes.dropped.jsonl so a resume never
        # discards logged data.
        if dropped:
            with (Path(run_dir) / "episodes.dropped.jsonl").open("a") as fh:
                fh.write("".join(f"{line}\n" for line in dropped))
        path.write_text("".join(f"{rows[i]}\n" for i in sorted(rows)))

    @staticmethod
    def read_episodes(run_dir: str | Path) -> list[dict[str, Any]]:
        path = Path(run_dir) / "episodes.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]
