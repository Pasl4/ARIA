"""AutogenRunner: the reference runner, wrapping AriaGroupChat.

Supports all three contact-point kinds (ledger round = PRE_STEP with the
upcoming instruction, completion = PRE_DELIVERY, stall/replan = STALL),
declared below and recorded in every run manifest.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from autogen_core.models import ChatCompletionClient

from ...benchmarks.base import BenchmarkAdapter, Task
from ...core.types import ContactPointKind
from ...orchestration.controller import InterventionController
from ...telemetry.token_meter import TokenMeter
from ..base import (
    STOP_COMPLETED,
    STOP_CRASHED,
    EpisodeOutcome,
    EpisodeRunConfig,
    FrameworkRunner,
)
from ..spec import TeamSpec
from ..tools import DEFAULT_EXEC_TIMEOUT_S, ToolContext
from .team import AriaGroupChat, extract_final_answer
from .teams import build_participants

logger = logging.getLogger(__name__)


class AutogenRunner(FrameworkRunner):
    name = "autogen"
    supported_contact_points = frozenset(
        {ContactPointKind.PRE_STEP, ContactPointKind.PRE_DELIVERY, ContactPointKind.STALL}
    )

    def __init__(self, executor_client: ChatCompletionClient) -> None:
        # Must already be metered (MeteredChatCompletionClient); the runner
        # never records tokens itself.
        self._executor_client = executor_client

    async def run_episode(
        self,
        task: Task,
        spec: TeamSpec,
        controller: InterventionController,
        meter: TokenMeter,
        config: EpisodeRunConfig,
        benchmark: BenchmarkAdapter,
    ) -> EpisodeOutcome:
        workspace: Path | None = None
        closeables: list[Any] = []
        try:
            workspace = await benchmark.prepare_workspace(task)
            # With repo_access the advisor's read-only inspections run in this
            # episode's workspace; the next episode overwrites it.
            if getattr(controller.advisor, "_repo_access", False):
                controller.advisor.workspace = workspace
            # Optional inspection backend (None by default) for benchmarks whose
            # graded state lives in a container. Set even without repo_access,
            # because the gate diff injection uses it.
            controller.advisor.inspector = benchmark.advisor_inspector(workspace, task)  # pyright: ignore[reportAttributeAccessIssue]
            prompt_vars = {
                "date_today": datetime.now(UTC).strftime("%Y-%m-%d"),
                **benchmark.prompt_vars(task),
            }
            tool_context = ToolContext(
                framework=self.name,
                workspace=workspace,
                advisor=controller.advisor,
                meter=meter,
                status_provider=lambda: controller.latest_status,
                # Same pinned task header and per-episode consult cap as
                # policy-path consults.
                header_provider=lambda: getattr(controller, "_task_header", ""),
                consult_budget=controller.consult_budget,
                cost_multiplier=config.cost_multiplier,
                exec_timeout_s=int(
                    config.runtime.get("exec_timeout_s", DEFAULT_EXEC_TIMEOUT_S)
                ),
                # Adapter hints for container-backed execution (see
                # BenchmarkAdapter.execution_image). The stock python_exec runs
                # on the host and ignores them; custom tool factories may not.
                extras={
                    "exec_image": benchmark.execution_image(task),
                    "exec_setup": benchmark.execution_setup(task),
                },
                closeables=closeables,
            )
            participants = build_participants(
                spec,
                self._executor_client,
                prompt_vars,
                tool_context,
                include_consult_tool=config.organic_consult,
            )
            team = AriaGroupChat(
                participants,
                model_client=self._executor_client,
                max_turns=int(config.runtime.get("max_steps", 20)),
                max_stalls=int(config.runtime.get("max_stalls", 3)),
                controller=controller,
                gate_revision_rounds=int(config.runtime.get("gate_revision_rounds", 0)),
                # Enabled only for the "organic" condition, like the consult tool.
                organic_liaison=config.organic_consult,
            )
            result = await team.run(task=task.prompt)
            final_answer = controller.final_answer or extract_final_answer(result.messages)
            stop_reason = controller.stop_reason or STOP_COMPLETED
        except Exception:
            logger.exception("autogen episode crashed (task=%s)", task.id)
            final_answer = None
            stop_reason = STOP_CRASHED
        finally:
            # Read the workspace diff before teardown, so crashed episodes keep
            # their work in the trace and the workspace_patch delivery channel
            # does not depend on the model's typed diff.
            workspace_patch: str | None = None
            try:
                workspace_patch = await benchmark.workspace_patch(workspace)
            except Exception:  # noqa: BLE001
                logger.exception("workspace patch extraction failed (task=%s)", task.id)
            # Stop everything tool factories opened for this episode (e.g. a
            # container-backed executor); one failed stop does not block the rest.
            for closeable in closeables:
                try:
                    await closeable.stop()
                except Exception:  # noqa: BLE001
                    logger.warning("episode closeable failed to stop", exc_info=True)
            # Release the workspace, also after a crash, so the next episode
            # does not inherit this agent's edits.
            await benchmark.release_workspace(workspace)
        return EpisodeOutcome(
            final_answer=final_answer,
            stop_reason=stop_reason,
            records=list(controller.latest_records),
            workspace_patch=workspace_patch,
        )
