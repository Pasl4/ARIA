"""Framework-agnostic episode runner: one FrameworkRunner call over one task.

Conditions differ only in the controller's policy (plus the consult tool for
the organic condition) and frameworks only in the FrameworkRunner, so scoring,
finalize and stats assembly are shared. Only stop_reason "completed" can count as
delivered; "max_rounds" (loop cut off, an answer may exist) and "crashed" (an
exception escaped the team) never do.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..benchmarks.base import BenchmarkAdapter, Task, TaskScore
from ..frameworks.base import STOP_COMPLETED, STOP_CRASHED, EpisodeRunConfig, FrameworkRunner
from ..frameworks.spec import TeamSpec
from ..orchestration.controller import InterventionController
from ..telemetry.run_logger import EpisodeStats
from ..telemetry.token_meter import TokenMeter

logger = logging.getLogger(__name__)


@dataclass
class EpisodeRunner:
    benchmark: BenchmarkAdapter
    framework: FrameworkRunner
    spec: TeamSpec
    controller: InterventionController
    meter: TokenMeter
    config: EpisodeRunConfig = field(default_factory=EpisodeRunConfig)
    # Wall-clock cap for the agent phase only (framework.run_episode); None
    # means no cap. Scoring has its own timeouts and stays outside, so a slow
    # test script cannot turn a gradable episode into a crash without a verdict.
    agent_timeout_s: float | None = None

    async def run(self, task: Task, episode: int) -> EpisodeStats:
        started = time.monotonic()
        self.meter.start_episode(episode)
        self.controller.start_episode(
            episode, task_header=f"Task:\n{task.prompt}"
        )
        if self.agent_timeout_s:
            # Backup for the asyncio timeout below: AutoGen can absorb the
            # cancellation (synchronous docker/subprocess calls block the
            # loop), so the controller also checks this deadline at every
            # contact point and raises. The episode then ends as crashed.
            self.controller.wall_clock_deadline = started + self.agent_timeout_s

        if self.agent_timeout_s:
            async with asyncio.timeout(self.agent_timeout_s):
                outcome = await self.framework.run_episode(
                    task, self.spec, self.controller, self.meter, self.config, self.benchmark
                )
        else:
            outcome = await self.framework.run_episode(
                task, self.spec, self.controller, self.meter, self.config, self.benchmark
            )

        # A crashed episode is missing data, not a zero (see EpisodeStats.scored).
        # The framework runner returns stop_reason=crashed instead of raising,
        # so `scored` comes from stop_reason; otherwise `finalize` would turn a
        # placeholder 0.0 into a terminal Q-target.
        crashed = outcome.stop_reason == STOP_CRASHED
        scored = not crashed
        # Score the agent's workspace diff when configured and available;
        # benchmarks without a workspace return None and fall back to the
        # typed answer, which is logged either way.
        use_workspace = (
            self.config.delivery == "workspace_patch"
            and outcome.workspace_patch is not None
        )
        scored_artifact = outcome.workspace_patch if use_workspace else outcome.final_answer
        try:
            task_score = await self.benchmark.score(task, scored_artifact)
        except Exception:
            # A scoring failure keeps the episode and its trace. Harness errors
            # under concurrent container load are usually transient, and an
            # unscored episode's transitions never reach the Q-table, so retry
            # once after a pause.
            logger.warning("scoring failed for %s; retrying once in 90s", task.id)
            await asyncio.sleep(90)
            try:
                task_score = await self.benchmark.score(task, scored_artifact)
                logger.info("scoring retry succeeded for %s", task.id)
            except Exception:
                logger.exception("scoring failed twice for %s; kept as unscored", task.id)
                task_score = TaskScore(score=0.0, delivered=False)
                scored = False
        if crashed:
            # No verdict exists; keep the episode row (a hole in episodes.jsonl
            # would silently shrink the sample) but mark it unscored.
            task_score = TaskScore(score=0.0, delivered=False)

        # The harness verdict alone drives the reward, so a resolved episode is
        # not penalized for overrunning its round budget. Clean stopping is
        # reported separately through `delivered`.
        delivered = task_score.delivered and outcome.stop_reason == STOP_COMPLETED
        terminal_reward = self.controller.finalize(
            task_score.score,
            delivered=task_score.delivered,
            final_records=outcome.records,
            scored=scored,
        )

        usage = self.meter.episode_summary()
        n_organic = int(usage["events"].get("organic_consult", 0))
        steps_taken = max(1, self.controller.n_contact_points)
        return EpisodeStats(
            episode=episode,
            task_id=task.id,
            condition=self.config.condition,
            benchmark=self.benchmark.name,
            framework=self.framework.name,
            seed=self.config.seed,
            benchmark_score=task_score.score,
            scored=scored,
            delivered=delivered,
            stop_reason=outcome.stop_reason,
            terminal_reward=terminal_reward,
            total_reward=self.controller.total_reward,
            final_answer=outcome.final_answer or "",
            workspace_patch=outcome.workspace_patch or "",
            delivery_channel="workspace_patch" if use_workspace else "typed",
            steps_taken=steps_taken,
            n_consults=self.controller.n_consults,
            n_organic_consults=n_organic,
            intervention_rate=(self.controller.n_consults + n_organic) / steps_taken,
            total_tokens=int(usage["total_tokens"]),
            cost_usd=float(usage["cost_usd"]),
            advisory_cost_fraction=float(usage["advisory_cost_fraction"]),
            advisor_failures=int(usage["events"].get("advisor_failure", 0)),
            duration_s=time.monotonic() - started,
            usage=usage,
            learner=self.controller.policy.telemetry_snapshot(),
            transitions=self.controller.transitions_as_dicts(),
        )

    def build_trace(self, stats: EpisodeStats) -> dict[str, Any]:
        """Assemble the episode trace from the controller's accumulated state.

        Call before the next start_episode(), which clears that state; the
        experiment loop writes the result via RunLogger.write_trace.
        """
        c = self.controller
        return {
            "episode": stats.episode,
            "task_id": stats.task_id,
            "condition": stats.condition,
            "benchmark": stats.benchmark,
            "framework": stats.framework,
            "seed": stats.seed,
            "benchmark_score": stats.benchmark_score,
            "delivered": stats.delivered,
            "stop_reason": stats.stop_reason,
            "final_answer": stats.final_answer,
            "workspace_patch": stats.workspace_patch,
            "delivery_channel": stats.delivery_channel,
            "duration_s": stats.duration_s,
            "records": [asdict(r) for r in c.latest_records],
            "consults": list(c.consults),
            "ledgers": list(c.ledgers),
            "transitions": stats.transitions,
        }
