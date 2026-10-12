"""Experiment orchestration: config -> N seeds x M episodes -> run directories.

`mode="train"` learns on the train split; `mode="eval"` runs the frozen policy
(or a static baseline) on the test split. The framework comes from the
framework registry. The run manifest records the runner's contact points and
the benchmark's state_signature, so each result traces to the decision
granularity and state-space version it was produced under.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from ..advisor import Advisor, MeteredCompletionClient, OpenAICompatibleClient
from ..benchmarks.registry import build_benchmark
from ..config import ExperimentConfig, ModelConfig
from ..core.reward import RewardModel
from ..detectors import load_phases_config, state_signature
from ..frameworks.autogen.clients import build_model_client
from ..frameworks.autogen.consult_tool import CONSULT_TOOL_DESCRIPTION_TEMPLATE
from ..frameworks.autogen.metering import MeteredChatCompletionClient
from ..frameworks.autogen.retry import RetryOnEmptyClient
from ..frameworks.base import STOP_CRASHED, EpisodeRunConfig
from ..frameworks.registry import build_framework_runner
from ..frameworks.spec import load_team_spec
from ..orchestration.controller import GUIDANCE_TEMPLATE, InterventionController
from ..orchestration.seeding import SeedingConfig
from ..policies.registry import build_policy
from ..telemetry.run_logger import EpisodeStats, RunLogger
from ..telemetry.token_meter import TokenMeter, TokenPricing
from . import layout
from .episode_runner import EpisodeRunner

logger = logging.getLogger(__name__)


def _openai_client(model_cfg: ModelConfig, max_tokens: int | None = None):
    return OpenAICompatibleClient(
        model=model_cfg.model,
        api_key=model_cfg.api_key,
        base_url=model_cfg.base_url,
        temperature=model_cfg.temperature,
        max_tokens=max_tokens or model_cfg.max_tokens,
        max_retries=model_cfg.max_retries or 3,
        timeout=model_cfg.timeout,
        reasoning_effort=model_cfg.reasoning_effort,
    )


def _episode_budget_s(benchmark, task, cap_s: float) -> float:
    """Wall-clock budget for one episode's agent phase.

    A per-task budget declared via `benchmark.task_timeout_s` is capped at
    `cap_s`; without one the flat `cap_s` applies.
    """
    declared = benchmark.task_timeout_s(task)
    return cap_s if declared is None else min(float(declared), cap_s)


def _crashed_stats(
    cfg: ExperimentConfig,
    episode: int,
    task_id: str,
    seed: int,
    exc: BaseException,
    meter: TokenMeter,
    n_consults: int = 0,
) -> EpisodeStats:
    """Build the stats row for an episode that died outside the framework guard.

    The episode is logged rather than dropped, since a missing row would shrink
    the sample. Its spend comes from the meter: the calls happened and are in
    calls.jsonl, so zeros would understate cost and efficiency metrics.
    """
    usage = meter.episode_summary()
    return EpisodeStats(
        episode=episode,
        task_id=task_id,
        condition=cfg.condition,
        benchmark=cfg.benchmark,
        framework=cfg.framework,
        seed=seed,
        benchmark_score=0.0,
        # A crashed or unjudged episode carries no evidence about the result,
        # so it is excluded from score averages instead of counting as 0.
        scored=False,
        delivered=False,
        stop_reason=STOP_CRASHED,
        terminal_reward=0.0,
        total_reward=0.0,
        final_answer="",
        steps_taken=0,
        # Taken from the controller, not from metered advisor calls: a JSON
        # repair call shares the consult's purpose tag and would be counted
        # twice. random_advisory is budget-matched against this count.
        n_consults=n_consults,
        n_organic_consults=int(usage["events"].get("organic_consult", 0)),
        intervention_rate=0.0,
        total_tokens=int(usage["total_tokens"]),
        cost_usd=float(usage["cost_usd"]),
        advisory_cost_fraction=float(usage["advisory_cost_fraction"]),
        advisor_failures=int(usage["events"].get("advisor_failure", 0)),
        usage={**usage, "crash": f"{type(exc).__name__}: {exc}"[:500]},
    )


def _resume_point(run_dir: Path, learned: bool) -> tuple[set[int], Path | None]:
    """Return the episode indices to skip and the checkpoint to resume from.

    For a learned policy the log and the Q-table must agree: episodes logged
    after the newest checkpoint carry learning it does not contain, so they
    are re-run and their rows dropped. Static policies carry no state and
    skip every logged episode that did not crash.
    """
    rows = [r for r in RunLogger.read_episodes(run_dir) if r.get("episode") is not None]
    all_logged = {int(r["episode"]) for r in rows}
    # Crashed episodes have no verdict and are retried, unless the resumed
    # checkpoint already contains them (a retry would apply their Q-updates twice).
    logged = {
        int(r["episode"]) for r in rows if r.get("stop_reason") != STOP_CRASHED
    }
    checkpoint: Path | None = None
    if learned and logged:
        checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_ep*.json"))
        if checkpoints:
            # checkpoint_epNNN.json holds the policy after episode index NNN-1.
            n = int(checkpoints[-1].stem.removeprefix("checkpoint_ep"))
            # Every episode before the checkpoint is already in it, crashed or not.
            logged = {e for e in all_logged if e < n}
            checkpoint = checkpoints[-1]
        else:
            logged = set()  # nothing to continue from: redo the run
    # Trim the log to the kept episodes so retried ones do not appear twice.
    RunLogger.keep_episodes(run_dir, logged)
    return logged, checkpoint


def _benchmark_options(cfg: ExperimentConfig) -> dict:
    """Return a copy of `cfg.benchmark_options` as benchmark constructor kwargs."""
    return dict(cfg.benchmark_options)


async def run_experiment(cfg: ExperimentConfig, mode: str = "train") -> list[Path]:
    """Run all seeds of one condition; returns the run directories."""
    eval_mode = mode == "eval"
    benchmark = build_benchmark(cfg.benchmark, **_benchmark_options(cfg))
    spec = load_team_spec(cfg.benchmark, cfg.teams_root)
    phases_cfg = load_phases_config(cfg.benchmark, cfg.teams_root)
    # Active state layout from config. Not named `layout`, which would shadow
    # the results-layout module imported above.
    state_layout = cfg.state_layout()
    # `mode` picks the default split; an explicit `run.split` wins, so an eval
    # on the train split (e.g. for model selection) stays off the held-out
    # test set. Selecting by test score would bias later test numbers.
    split = cfg.run.split if cfg.run.split_is_explicit else ("test" if eval_mode else "train")
    if eval_mode and split != "test":
        logger.warning(
            "eval mode on split=%r (explicitly configured). Results on this "
            "split are not held-out numbers.",
            split,
        )
    runtime = {
        **spec.runtime.for_framework(cfg.framework),
        "gate_revision_rounds": cfg.advisor.gate_revision_rounds,
    }
    # Hard ceiling per episode, since a hang is invisible to an exception
    # guard. The fallback is the longest legitimate episode (every round uses
    # its full command timeout); team specs usually set a tighter value.
    episode_timeout_s = float(
        runtime.get(
            "episode_timeout_s",
            int(runtime.get("max_steps", 20)) * int(runtime.get("exec_timeout_s", 600)),
        )
    )
    # No `limit=` here: an adapter applies it before the per-seed shuffle, so
    # every seed would get the same prefix. The cap is applied after the
    # shuffle below, which makes it a per-seed sample.
    all_tasks = benchmark.load_tasks(split)
    if not all_tasks:
        raise ValueError(f"no tasks loaded for benchmark={cfg.benchmark} split={split}")
    reward_cfg = cfg.reward_config()
    # Validate at startup rather than at the first priced consult. This runs
    # for every condition: validate() skips only the anchor and price checks
    # when lambda_cost == 0, and the cost-rule check applies to `autonomous` too.
    reward_cfg.validate()
    reward_model = RewardModel(reward_cfg)
    # Same for prices: the meter refuses missing or `tbd` entries, but only at
    # the first call. An empty pricing table skips this check (offline tests).
    if cfg.pricing:
        for role, model in cfg.models.items():
            price = cfg.pricing.get(model.model, "missing")
            if price is None or price == "missing":
                raise ValueError(
                    f"pricing for {role} model {model.model!r} is "
                    f"{'tbd' if price is None else 'missing'}; fill in the real "
                    "USD/1M rates in the pricing block before any run"
                )

    run_dirs: list[Path] = []
    for seed in cfg.run.seeds:
        tasks = list(all_tasks)
        if cfg.run.task_ids:
            # An explicit id list (RunConfig.task_ids) overrides shuffle, offset
            # and limit; a limit on top would drop tasks the caller named.
            by_id = {t.id: t for t in tasks}
            missing = [i for i in cfg.run.task_ids if i not in by_id]
            if missing:
                raise ValueError(f"run.task_ids not in the {cfg.run.split} split: {missing}")
            tasks = [by_id[i] for i in cfg.run.task_ids]
        else:
            if cfg.run.shuffle_tasks:
                random.Random(seed).shuffle(tasks)
            if cfg.run.task_offset:
                tasks = tasks[cfg.run.task_offset :]
            if cfg.run.task_limit is not None:
                tasks = tasks[: cfg.run.task_limit]

        run_dir = layout.run_dir(
            cfg.run.results_root,
            cfg.benchmark,
            cfg.condition,
            cfg.framework,
            mode,
            seed,
            executor_model=cfg.models["executor"].model,
        )
        run_logger = RunLogger(run_dir)

        options = dict(cfg.condition_options)
        if eval_mode and cfg.condition == "aria":
            options.setdefault("frozen", True)
        policy = build_policy(
            cfg.condition,
            seed=seed,
            rl_config=cfg.rl_config(),
            options=options,
            layout=state_layout,
        )
        # Resume: skip logged episodes so the log and the policy describe the
        # same run. Checkpoint alignment applies only to stateful policies in
        # train mode; eval writes no checkpoints, so a frozen policy treated as
        # learned would find none and re-run the whole split.
        done, checkpoint = _resume_point(
            run_dir, learned=policy.state_dict() is not None and not eval_mode
        )
        if checkpoint is not None:
            policy.load_state_dict(json.loads(checkpoint.read_text()))
        if done:
            logger.warning(
                "[seed %d] resuming at episode %d (%s)",
                seed,
                max(done) + 1,
                checkpoint.name if checkpoint is not None else "stateless policy",
            )

        # One TokenMeter per run; metering happens exactly once, at each
        # client boundary.
        meter = TokenMeter(
            pricing={
                model_id: (
                    TokenPricing(
                        prompt=p.prompt,
                        completion=p.completion,
                        cached_prompt=p.cached_prompt,
                    )
                    if p
                    else None
                )
                for model_id, p in cfg.pricing.items()
            }
        )
        # The retry wraps the meter so retried calls are still billed: a
        # filtered response costs input tokens even though it returns nothing.
        executor_client = RetryOnEmptyClient(
            MeteredChatCompletionClient(
                build_model_client(cfg.models["executor"].model_dump()),
                meter,
                role="executor",
                model_label=cfg.models["executor"].model,
            ),
            meter=meter,
        )
        advisor_completion_client = MeteredCompletionClient(
            _openai_client(cfg.models["advisor"], max_tokens=cfg.advisor.max_advice_tokens),
            meter,
            role="advisor",
            model_label=cfg.models["advisor"].model,
        )
        framework = build_framework_runner(cfg.framework, executor_client=executor_client)

        run_logger.write_manifest(
            mode=mode,
            seed=seed,
            split=split,
            n_tasks=len(tasks),
            task_order=[t.id for t in tasks],
            framework=cfg.framework,
            supported_contact_points=sorted(
                k.name.lower() for k in framework.supported_contact_points
            ),
            state_signature=state_signature(phases_cfg, state_layout),
            phase_classifier=phases_cfg.classifier,
            runtime=runtime,
            guidance_template=GUIDANCE_TEMPLATE,
            consult_tool_description=CONSULT_TOOL_DESCRIPTION_TEMPLATE,
            config=cfg.manifest_view(),
        )

        for episode in range(cfg.run.episodes):
            if episode in done:
                continue
            task = tasks[episode % len(tasks)]
            # Per task where the benchmark declares one, the flat ceiling
            # otherwise. See _episode_budget_s.
            budget_s = _episode_budget_s(benchmark, task, episode_timeout_s)
            advisor = Advisor(
                advisor_completion_client,
                style=cfg.advisor.style,  # type: ignore[arg-type]
                meter=meter,
                domain_context=(
                    benchmark.advisor_domain_context(task) + cfg.advisor.extra_context
                ),
                max_guidance_words=cfg.advisor.max_guidance_words,
                fc3_nudge=cfg.advisor.fc3_nudge,
                max_context_chars=cfg.advisor.max_context_chars,
                max_header_chars=cfg.advisor.max_header_chars,
                repo_access=cfg.advisor.repo_access,
                max_tool_calls=cfg.advisor.max_tool_calls,
                gate_diff_injection=cfg.advisor.gate_diff_injection,
            )
            controller = InterventionController(
                detector=benchmark.build_detector(),
                policy=policy,
                advisor=advisor,
                reward_model=reward_model,
                eval_mode=eval_mode,
                consult_budget=cfg.consult_budget,
                meter=meter,
                markers=phases_cfg.marker_tables(),
                layout=state_layout,
                seeding=SeedingConfig.from_dict(cfg.seeding),
            )
            runner = EpisodeRunner(
                benchmark=benchmark,
                framework=framework,
                spec=spec,
                controller=controller,
                meter=meter,
                agent_timeout_s=budget_s,
                config=EpisodeRunConfig(
                    condition=cfg.condition,
                    seed=seed,
                    organic_consult=cfg.organic_consult(),
                    cost_multiplier=cfg.cost_multiplier(),
                    runtime=runtime,
                    delivery=cfg.delivery,
                ),
            )
            # Run-level guard so one bad episode does not end the run: the
            # framework guards only the episode body, not scoring, finalize or
            # trace writing.
            try:
                # The wall-clock cap is applied inside the runner, around the
                # agent phase only (EpisodeRunner.agent_timeout_s).
                stats = await runner.run(task, episode)
            except Exception as exc:  # noqa: BLE001  # deliberate run-level guard
                logger.exception(
                    "[seed %d] ep %d crashed (task=%s); continuing", seed, episode, task.id
                )
                # Mirror the framework crash path (agent timeouts also land here) so epsilon
                # decays once per episode and TD telemetry rotates; finalize is idempotent if
                # the runner already called it. scored=False keeps buffered transitions out
                # of the Q-table, since without a verdict the learner would see only the costs.
                try:
                    controller.finalize(0.0, delivered=False, scored=False)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[seed %d] ep %d: crash-path finalize failed", seed, episode
                    )
                stats = _crashed_stats(
                    cfg, episode, task.id, seed, exc, meter, controller.n_consults
                )
            else:
                # A failed trace write keeps the episode: the result stands and
                # only its trace reference is missing.
                try:
                    trace_path = run_logger.write_trace(episode, runner.build_trace(stats))
                    stats.trace_file = str(trace_path.relative_to(run_logger.output_dir))
                except Exception:  # noqa: BLE001
                    logger.exception("[seed %d] ep %d: trace write failed", seed, episode)
            # Set after the try/except/else so the crash path records it too:
            # a row produced by a timeout then shows the budget that caused it.
            stats.agent_timeout_s = budget_s
            run_logger.log_episode(stats)
            run_logger.log_calls(meter.episode_calls)
            logger.info(
                "[seed %d] ep %d/%d task=%s score=%.3f consults=%d+%d stop=%s budget=%.0fs",
                seed, episode + 1, cfg.run.episodes, task.id,
                stats.benchmark_score, stats.n_consults, stats.n_organic_consults,
                stats.stop_reason, budget_s,
            )

            state = policy.state_dict()
            if (
                state is not None
                and not eval_mode
                and cfg.run.checkpoint_every > 0
                and (episode + 1) % cfg.run.checkpoint_every == 0
            ):
                run_logger.save_checkpoint(episode=episode + 1, policy_state=state)

        state = policy.state_dict()
        if state is not None and not eval_mode:
            run_logger.save_checkpoint(episode=cfg.run.episodes, policy_state=state)
        # After a resume the meter has seen only this process's episodes, so
        # the summary is flagged as partial; calls.jsonl (append-only) holds
        # the complete per-call record.
        run_logger.write_cost_summary(
            {**meter.run_summary(), "partial_resumed_run": bool(done)}
        )
        run_dirs.append(run_dir)
    return run_dirs
