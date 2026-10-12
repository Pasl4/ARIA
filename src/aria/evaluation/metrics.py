"""The six evaluation metrics, computed from episode logs.

Task performance: benchmark score (primary, per benchmark), delivery rate.
Cost: total tokens, advisory-cost fraction. Efficiency: intervention rate,
value of advisor. Conditions are compared with paired per-task deltas, since
every condition runs the same per-seed task list; within-run std mixes task
difficulty with stochasticity and is descriptive only.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Any


def _consults(e: dict[str, Any]) -> int:
    return int(e["n_consults"]) + int(e.get("n_organic_consults", 0))


@dataclass(frozen=True)
class RunMetrics:
    condition: str
    benchmark: str
    n_episodes: int
    # Episodes with a trustworthy score. n_scored < n_episodes means some
    # episodes were excluded from the score mean as missing data; report it
    # alongside the mean.
    n_scored: int
    # Task performance
    benchmark_score_mean: float
    benchmark_score_std: float
    delivery_rate: float
    # Cost
    total_tokens_mean: float
    cost_usd_mean: float
    advisory_cost_fraction_mean: float
    # Efficiency
    intervention_rate_mean: float
    consults_per_task_mean: float
    advisor_failure_rate: float


def aggregate_run(episodes: list[dict[str, Any]]) -> RunMetrics:
    """Aggregate one run's episodes.jsonl records into a metric row."""
    if not episodes:
        raise ValueError("no episodes to aggregate")
    # Unscored episodes (harness could not judge, or a crash) are missing
    # data, not zeros: averaging them in would let an infrastructure failure
    # lower the score. n_scored makes the exclusion visible.
    judged = [e for e in episodes if e.get("scored", True)]
    scores = [float(e["benchmark_score"]) for e in judged] or [0.0]
    return RunMetrics(
        condition=str(episodes[0]["condition"]),
        benchmark=str(episodes[0]["benchmark"]),
        n_episodes=len(episodes),
        n_scored=len(judged),
        benchmark_score_mean=statistics.fmean(scores),
        benchmark_score_std=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        delivery_rate=statistics.fmean(float(bool(e["delivered"])) for e in episodes),
        total_tokens_mean=statistics.fmean(float(e["total_tokens"]) for e in episodes),
        cost_usd_mean=statistics.fmean(float(e["cost_usd"]) for e in episodes),
        advisory_cost_fraction_mean=statistics.fmean(
            float(e["advisory_cost_fraction"]) for e in episodes
        ),
        intervention_rate_mean=statistics.fmean(
            float(e["intervention_rate"]) for e in episodes
        ),
        consults_per_task_mean=statistics.fmean(float(_consults(e)) for e in episodes),
        advisor_failure_rate=statistics.fmean(
            float(e.get("advisor_failures", 0) > 0) for e in episodes
        ),
    )


def aggregate_seeds(runs: list[RunMetrics]) -> dict[str, tuple[float, float]]:
    """Cross-seed mean and std of the main per-run metrics."""
    if not runs:
        raise ValueError("no runs to aggregate")
    out: dict[str, tuple[float, float]] = {}
    for name in (
        "benchmark_score_mean",
        "delivery_rate",
        "total_tokens_mean",
        "cost_usd_mean",
        "advisory_cost_fraction_mean",
        "intervention_rate_mean",
        "consults_per_task_mean",
    ):
        values = [getattr(r, name) for r in runs]
        out[name] = (
            statistics.fmean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
    return out


def value_of_advisor(condition: RunMetrics, autonomous: RunMetrics) -> float:
    """Average task-performance gain per initiative transfer.

    (score_condition - score_autonomous) / consults_per_task, i.e. how much
    benchmark score one consultation buys on average, measured against the
    fully autonomous executor on the same benchmark, split and seeds.
    Returns 0.0 when the condition never consults.
    """
    if condition.benchmark != autonomous.benchmark:
        raise ValueError("conditions must be on the same benchmark")
    if condition.consults_per_task_mean == 0:
        return 0.0
    gain = condition.benchmark_score_mean - autonomous.benchmark_score_mean
    return gain / condition.consults_per_task_mean


def _mean_score_by_seed_and_task(
    episodes: list[dict[str, Any]],
) -> dict[int, dict[str, float]]:
    """Map seed -> task_id -> mean score.

    Repeats of a task within a seed are averaged: a training run walks the
    task list several times, so a task_id alone is not one observation.
    """
    buckets: dict[int, dict[str, list[float]]] = {}
    for e in episodes:
        by_task = buckets.setdefault(int(e.get("seed", 0)), {})
        by_task.setdefault(str(e["task_id"]), []).append(float(e["benchmark_score"]))
    return {
        s: {t: statistics.fmean(v) for t, v in by_task.items()}
        for s, by_task in buckets.items()
    }


def paired_score_delta(
    episodes_a: list[dict[str, Any]],
    episodes_b: list[dict[str, Any]],
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """Paired per-task score delta (a - b) with a hierarchical bootstrap CI.

    Episodes pair on (seed, task_id), which removes task difficulty and
    seed-to-seed variation. The 95% CI resamples seeds, then tasks within
    each drawn seed, because one task's deltas across seeds are not iid.
    The point estimate is the mean over seeds of each seed's mean task delta,
    which stays correct when seeds cover different numbers of tasks.
    """
    a_by_seed = _mean_score_by_seed_and_task(episodes_a)
    b_by_seed = _mean_score_by_seed_and_task(episodes_b)

    # seed -> [per-task deltas], keeping only tasks both conditions ran
    per_seed: dict[int, list[float]] = {}
    for s, a_tasks in a_by_seed.items():
        b_tasks = b_by_seed.get(s)
        if not b_tasks:
            continue
        deltas = [a - b_tasks[t] for t, a in a_tasks.items() if t in b_tasks]
        if deltas:
            per_seed[s] = deltas
    if not per_seed:
        raise ValueError("no overlapping (seed, task_id) pairs between the two runs")

    seeds = sorted(per_seed)
    point = statistics.fmean(statistics.fmean(per_seed[s]) for s in seeds)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        drawn: list[float] = []
        for s in rng.choices(seeds, k=len(seeds)):  # level 1: seeds
            tasks = per_seed[s]
            drawn.append(statistics.fmean(rng.choices(tasks, k=len(tasks))))  # level 2
        means.append(statistics.fmean(drawn))
    means.sort()
    lo = means[int(0.025 * n_bootstrap)]
    hi = means[int(0.975 * n_bootstrap)]
    return point, (lo, hi)
