"""Bundle a finished run into run_log.jsonl and run_report.json.

Reads the manifest, episodes.jsonl and per-episode traces after the run and
joins them by episode number. run_log.jsonl has one sorted line per episode
(stats plus the trace's records and ledgers; without a trace, null records and
`trace_missing: true`). run_report.json has the manifest, a per-task table and
aggregates (solve rate, evidence, state-label counts, cost, events).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bundle_run(run_dir: str | Path, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Bundle `run_dir` and return the paths of run_log.jsonl and run_report.json.

    Both files go into `out_dir` when given, else into `run_dir`.
    """
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    episodes = _read_jsonl(run_dir / "episodes.jsonl")
    calls = _read_jsonl(run_dir / "calls.jsonl")

    calls_by_episode: dict[int | None, list[dict[str, Any]]] = {}
    for call in calls:
        calls_by_episode.setdefault(call.get("episode"), []).append(call)

    # ---- run_log.jsonl: one complete line per episode -------------------
    log_path = out / "run_log.jsonl"
    per_task: list[dict[str, Any]] = []
    evidence_totals: Counter[str] = Counter()
    label_totals: Counter[str] = Counter()
    events_totals: Counter[str] = Counter()
    with log_path.open("w") as f:
        for ep in sorted(episodes, key=lambda e: e["episode"]):
            n = ep["episode"]
            trace_path = run_dir / "traces" / f"episode_{n:03d}.json"
            trace = json.loads(trace_path.read_text()) if trace_path.exists() else None
            line = {
                **ep,
                # Records and ledgers come from the trace; its transitions
                # duplicate the episode line's and are not carried twice.
                "records": (trace or {}).get("records"),
                "ledgers": (trace or {}).get("ledgers"),
                "trace_missing": trace is None,
            }
            f.write(json.dumps(line) + "\n")

            labels = [t["info"]["state_label"] for t in ep.get("transitions", [])]
            evidence = Counter(label.split("/")[1] for label in labels)
            evidence_totals.update(evidence)
            label_totals.update(labels)
            usage = ep.get("usage", {}) or {}
            events_totals.update(usage.get("events", {}) or {})
            per_task.append(
                {
                    "episode": n,
                    "task_id": ep["task_id"],
                    "score": ep["benchmark_score"],
                    "delivered": ep["delivered"],
                    "stop_reason": ep["stop_reason"],
                    "steps": ep["steps_taken"],
                    "duration_s": ep["duration_s"],
                    "cost_usd": ep["cost_usd"],
                    "total_tokens": ep.get("total_tokens", 0),
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                    "failed_calls": usage.get("failed_calls", 0),
                    "n_consults": ep.get("n_consults", 0),
                    "evidence": dict(evidence),
                    "events": dict(usage.get("events", {}) or {}),
                    "trace_missing": trace is None,
                }
            )

    # ---- run_report.json: manifest + aggregates -------------------------
    n_eps = len(per_task)
    solved = [row["task_id"] for row in per_task if row["score"] == 1.0]
    crashed = [row["task_id"] for row in per_task if row["stop_reason"] == "crashed"]
    report = {
        "manifest": manifest,
        "totals": {
            "episodes": n_eps,
            "solved": len(solved),
            "solve_rate": (len(solved) / n_eps) if n_eps else 0.0,
            "delivered": sum(1 for row in per_task if row["delivered"]),
            "crashed": len(crashed),
            "duration_s": sum(row["duration_s"] for row in per_task),
            "cost_usd": sum(row["cost_usd"] for row in per_task),
            "total_tokens": sum(row["total_tokens"] for row in per_task),
            "reasoning_tokens": sum(row["reasoning_tokens"] for row in per_task),
            "failed_calls": sum(row["failed_calls"] for row in per_task),
            "traces_missing": sum(1 for row in per_task if row["trace_missing"]),
        },
        "solved_task_ids": sorted(solved),
        "crashed_task_ids": sorted(crashed),
        "evidence_totals": dict(evidence_totals),
        "state_label_totals": dict(label_totals),
        "events_totals": dict(events_totals),
        "per_task": per_task,
    }
    report_path = out / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return log_path, report_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="run directory (contains episodes.jsonl)")
    parser.add_argument("--out", default=None, help="output directory (default: run_dir)")
    args = parser.parse_args()
    log_path, report_path = bundle_run(args.run_dir, args.out)
    print(log_path)
    print(report_path)


if __name__ == "__main__":
    main()
