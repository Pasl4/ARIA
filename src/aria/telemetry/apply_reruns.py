"""Replace crashed episodes in a bundled run with their re-runs, visibly.

A crashed episode has no verdict, so a paired comparison of two runs would
otherwise be conditioned on tasks that crashed in neither. Replacements are
marked in run_log.jsonl (`rerun`, `rerun_of` with the superseded outcome) and
run_report.json (`reruns`, `per_task`). Only `crashed` episodes are eligible
and a re-run is kept whatever its outcome, so this cannot act as resampling.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

REPLACEABLE = "crashed"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def apply_reruns(
    log_path: str | Path,
    report_path: str | Path,
    rerun_dirs: list[str | Path],
    note: str = "",
) -> dict[str, Any]:
    """Replace crashed episodes in a bundled run with their re-runs.

    `rerun_dirs` are bundled run directories, each holding run_log.jsonl.
    Rewrites run_log.jsonl and run_report.json in place and returns a summary.
    """
    log_path, report_path = Path(log_path), Path(report_path)
    rows = _read_jsonl(log_path)
    report = json.loads(report_path.read_text())

    replacements: dict[str, dict[str, Any]] = {}
    for d in rerun_dirs:
        for row in _read_jsonl(Path(d) / "run_log.jsonl"):
            task = row["task_id"]
            if task in replacements:
                raise ValueError(f"{task} appears in more than one re-run directory")
            replacements[task] = row

    by_task = {r["task_id"]: r for r in rows}
    unknown = sorted(set(replacements) - set(by_task))
    if unknown:
        raise ValueError(f"re-run tasks absent from the original run: {unknown}")
    # Only episodes without a verdict are eligible; replacing a scored episode
    # would turn re-running into resampling.
    ineligible = sorted(t for t in replacements if by_task[t]["stop_reason"] != REPLACEABLE)
    if ineligible:
        raise ValueError(
            f"refusing to replace episodes that already have a verdict: {ineligible}"
        )

    applied: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    for row in rows:
        task = row["task_id"]
        new = replacements.get(task)
        if new is None:
            merged.append(row)
            continue
        # Keep the original episode index so the file order and any earlier
        # reference stay valid; carry the superseded outcome alongside.
        line = {
            **new,
            "episode": row["episode"],
            "rerun": True,
            "rerun_of": {
                "stop_reason": row["stop_reason"],
                "duration_s": row.get("duration_s"),
                "cost_usd": row.get("cost_usd"),
                "scored": row.get("scored"),
            },
        }
        merged.append(line)
        applied.append(
            {
                "task_id": task,
                "original_stop_reason": row["stop_reason"],
                "new_stop_reason": new["stop_reason"],
                "new_score": new["benchmark_score"],
                "new_duration_s": new.get("duration_s"),
                "new_cost_usd": new.get("cost_usd"),
            }
        )

    with log_path.open("w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")

    # Re-derive the aggregates from the merged set rather than patching the
    # old numbers, so totals cannot drift from the lines they describe.
    evidence: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    events: Counter[str] = Counter()
    per_task: list[dict[str, Any]] = []
    for row in merged:
        usage = row.get("usage", {}) or {}
        state_labels = [t["info"]["state_label"] for t in row.get("transitions", [])]
        ev = Counter(x.split("/")[1] for x in state_labels)
        evidence.update(ev)
        labels.update(state_labels)
        events.update(usage.get("events", {}) or {})
        per_task.append(
            {
                "episode": row["episode"],
                "task_id": row["task_id"],
                "score": row["benchmark_score"],
                "delivered": row["delivered"],
                "stop_reason": row["stop_reason"],
                "steps": row["steps_taken"],
                "duration_s": row["duration_s"],
                "cost_usd": row["cost_usd"],
                "total_tokens": row.get("total_tokens", 0),
                "reasoning_tokens": usage.get("reasoning_tokens", 0),
                "failed_calls": usage.get("failed_calls", 0),
                "evidence": dict(ev),
                "events": dict(usage.get("events", {}) or {}),
                "rerun": bool(row.get("rerun")),
            }
        )

    scored = [r for r in per_task if r["stop_reason"] != REPLACEABLE]
    solved = [r["task_id"] for r in per_task if r["score"] == 1.0]
    crashed = [r["task_id"] for r in per_task if r["stop_reason"] == REPLACEABLE]
    report["reruns"] = {
        # The default note describes the protocol enforced here; pass `note` to override.
        "note": note
        or (
            "Crashed episodes (no verdict) are replaced by one re-run each, "
            "selected via run.task_ids. Only crashed episodes are eligible, and "
            "a re-run is kept regardless of its outcome."
        ),
        "n_applied": len(applied),
        "applied": applied,
        "rerun_dirs": [str(d) for d in rerun_dirs],
    }
    report["totals"].update(
        {
            "episodes": len(per_task),
            "scored_episodes": len(scored),
            "solved": len(solved),
            "solve_rate_all_tasks": len(solved) / len(per_task) if per_task else 0.0,
            "solve_rate_scored": len(solved) / len(scored) if scored else 0.0,
            "crashed": len(crashed),
            "duration_s": sum(r["duration_s"] for r in per_task),
            "cost_usd": sum(r["cost_usd"] for r in per_task),
            "total_tokens": sum(r["total_tokens"] for r in per_task),
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in per_task),
            "failed_calls": sum(r["failed_calls"] for r in per_task),
            "rerun_episodes": len(applied),
        }
    )
    report["solved_task_ids"] = sorted(solved)
    report["crashed_task_ids"] = sorted(crashed)
    report["evidence_totals"] = dict(evidence)
    report["state_label_totals"] = dict(labels)
    report["events_totals"] = dict(events)
    report["per_task"] = per_task
    report_path.write_text(json.dumps(report, indent=2))
    return report["reruns"] | {"totals": report["totals"]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--note", default="")
    parser.add_argument("rerun_dirs", nargs="+")
    args = parser.parse_args()
    summary = apply_reruns(args.log, args.report, args.rerun_dirs, args.note)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
