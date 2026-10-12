"""Merge the slices of one benchmark pass back into one run.

A pass may run as parallel slices (`run.task_offset` + `task_limit`). Each
slice numbers its episodes from 0, so slices are joined on `task_id`, and the
report is re-derived from the merged episodes rather than summed from slice
reports. Slice manifests must agree on executor model, reasoning effort,
condition, benchmark, framework and split, else ValueError is raised.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

# Fields that define the run and must agree across slices. Others may differ
# (task_offset, task_limit, name, results_root, git commit, timestamps).
_MUST_AGREE = ("model", "condition", "benchmark", "framework", "split", "reasoning_effort")


def _dig(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            found = _dig(v, key)
            if found is not None:
                return found
    return None


def _fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    models = _dig(manifest, "models") or {}
    executor = models.get("executor", {}) if isinstance(models, dict) else {}
    return {
        "model": executor.get("model"),
        "reasoning_effort": executor.get("reasoning_effort"),
        "condition": _dig(manifest, "condition"),
        "benchmark": _dig(manifest, "benchmark"),
        "framework": _dig(manifest, "framework"),
        "split": _dig(manifest, "split"),
    }


def merge_slices(
    slice_dirs: list[str | Path], out_dir: str | Path, label: str
) -> tuple[Path, Path]:
    """Join bundled slices into `<label>.run_log.jsonl` and `<label>.run_report.json`.

    Each slice directory must already be bundled (run_log.jsonl and
    run_manifest.json present). Returns the two written paths.
    """
    dirs = [Path(d) for d in slice_dirs]
    manifests = [json.loads((d / "run_manifest.json").read_text()) for d in dirs]
    prints = [_fingerprint(m) for m in manifests]
    for field in _MUST_AGREE:
        values = {p[field] for p in prints}
        if len(values) > 1:
            raise ValueError(f"slices disagree on {field}: {sorted(map(str, values))}")

    episodes: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for d in dirs:
        for line in (d / "run_log.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task = row["task_id"]
            if task in seen:
                raise ValueError(f"{task} appears in {seen[task]} and {d} (overlapping slices)")
            seen[task] = d
            row["slice"] = d.parents[4].name
            episodes.append(row)
    episodes.sort(key=lambda r: r["task_id"])
    # Renumber so the merged file reads like a single pass; the original
    # per-slice index stays available as `slice_episode`.
    for i, row in enumerate(episodes):
        row["slice_episode"] = row["episode"]
        row["episode"] = i

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{label}.run_log.jsonl"
    with log_path.open("w") as f:
        for row in episodes:
            f.write(json.dumps(row) + "\n")

    evidence: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    events: Counter[str] = Counter()
    per_task: list[dict[str, Any]] = []
    for row in episodes:
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
                "slice": row["slice"],
            }
        )

    scored = [r for r in per_task if r["stop_reason"] != "crashed"]
    solved = [r["task_id"] for r in per_task if r["score"] == 1.0]
    crashed = [r["task_id"] for r in per_task if r["stop_reason"] == "crashed"]
    report = {
        "label": label,
        "manifest": manifests[0],
        "slices": [
            {
                "dir": str(d),
                "episodes": sum(
                    1 for r in per_task if r["slice"] == d.parents[4].name
                ),
            }
            for d in dirs
        ],
        "totals": {
            "episodes": len(per_task),
            "scored_episodes": len(scored),
            "solved": len(solved),
            # Two rates: over all tasks (crashes count as unsolved) and over the
            # episodes the harness judged (crashes excluded as missing data).
            "solve_rate_all_tasks": len(solved) / len(per_task) if per_task else 0.0,
            "solve_rate_scored": len(solved) / len(scored) if scored else 0.0,
            "crashed": len(crashed),
            "duration_s": sum(r["duration_s"] for r in per_task),
            "cost_usd": sum(r["cost_usd"] for r in per_task),
            "total_tokens": sum(r["total_tokens"] for r in per_task),
            "reasoning_tokens": sum(r["reasoning_tokens"] for r in per_task),
            "failed_calls": sum(r["failed_calls"] for r in per_task),
        },
        "solved_task_ids": sorted(solved),
        "crashed_task_ids": sorted(crashed),
        "evidence_totals": dict(evidence),
        "state_label_totals": dict(labels),
        "events_totals": dict(events),
        "per_task": per_task,
    }
    report_path = out / f"{label}.run_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    return log_path, report_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("slices", nargs="+")
    args = parser.parse_args()
    for path in merge_slices(args.slices, args.out, args.label):
        print(path)


if __name__ == "__main__":
    main()
