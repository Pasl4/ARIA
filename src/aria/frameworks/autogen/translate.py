"""Deterministic translation of an AutoGen group-chat thread into StepRecords.

A step is one orchestrator instruction plus the addressed agents' responses
up to the next instruction. Task-ledger messages (matched by fixed prefix)
become ``is_plan`` records with a canonical description, and advisor guidance
after GUIDANCE_MARKER is stripped from instructions, so neither the planning
prompt's vocabulary nor the advisor's wording reaches phase classification.
"""

from __future__ import annotations

from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    TextMessage,
    ToolCallExecutionEvent,
)

from ...detectors.base import StepRecord
from ...detectors.errors import count_internal_repairs, error_fingerprint, extract_error
from ...detectors.markers import MarkerTables
from ...orchestration.controller import GUIDANCE_MARKER
from .agents.coder import EXECUTED_OUTPUT_KEY

# Fixed prefix of ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT in autogen-agentchat.
LEDGER_PREFIX = "We are working to address the following user request"
PLAN_RECORD_DESCRIPTION = "Draft or revise the plan for the task"


def _clean_instruction(text: str) -> str:
    """Drop injected advisor guidance from an instruction's record text."""
    idx = text.find(GUIDANCE_MARKER)
    return text[:idx].rstrip() if idx != -1 else text


def thread_to_records(
    thread: list[BaseAgentEvent | BaseChatMessage],
    orchestrator_name: str,
    markers: MarkerTables | None = None,
) -> list[StepRecord]:
    records: list[StepRecord] = []
    instruction: str | None = None
    is_plan = False
    outputs: list[str] = []
    executed: list[str | None] = []
    tool_error: str | None = None
    tool_error_name: str = ""
    tool_ran = False
    tool_out: list[str] = []
    prev_finding: str = ""

    def close_step() -> None:
        nonlocal instruction, is_plan, outputs, executed, tool_error, tool_error_name
        nonlocal prev_finding, tool_ran, tool_out
        if instruction is None:
            return
        output = "\n".join(outputs)
        executed_output = "\n".join(x for x in executed if x)
        # None means the agent reports no executor output (file_surfer and library
        # agents, CodeExecutorAgent included); "" means its executor ran nothing.
        ran_code = any(x is not None for x in executed)
        # Errors come from executor output if the agent reports any, else from
        # the transcript; a false error there can only lower the evidence.
        observed = executed_output if ran_code else output
        error = None if is_plan else (extract_error(observed) or tool_error)
        is_state_changing, verification = False, "none"
        if markers is not None and not is_plan:
            # `is_state_changing` matches commands the agent wrote (git apply,
            # sed -i, ...), so it reads the transcript; a forged match can only
            # invalidate an earlier check.
            is_state_changing, _, prev_finding = markers.derive(output, prev_finding)
            # `verification` can only raise the evidence state, so it reads
            # executed output, which the model cannot forge, with no
            # transcript fallback (agents reporting no executor output get "none").
            _, verification, _ = markers.derive(executed_output, prev_finding)
        records.append(
            StepRecord(
                index=len(records),
                description=instruction,
                output=output,
                executed_output=executed_output,
                error=error,
                n_internal_repairs=0 if is_plan else count_internal_repairs(observed),
                is_state_changing=is_state_changing,
                verification=verification,
                error_target=(
                    error_fingerprint(observed, tool_name=tool_error_name) if error else ""
                ),
                is_plan=is_plan,
                tool_ran=tool_ran,
                tool_output="\n".join(tool_out),
            )
        )
        instruction, is_plan, outputs, executed = None, False, [], []
        tool_error, tool_error_name = None, ""
        tool_ran, tool_out = False, []

    for msg in thread:
        if isinstance(msg, ToolCallExecutionEvent):
            # Tool failures of library agents are read from the is_error flag,
            # since autogen's bare "Error: {e}" text does not match the error
            # patterns. The last event wins, like the coder's last segment.
            if instruction is not None:
                # Tool returns (capped per result) feed the outcome3 state dim
                # out of band, so the model cannot write into this channel.
                tool_ran = True
                tool_out.extend((r.content or "")[:4000] for r in msg.content)
                failed = [r for r in msg.content if r.is_error]
                if failed:
                    tool_error = failed[-1].content[:1000]
                    tool_error_name = failed[-1].name or ""
                else:
                    tool_error, tool_error_name = None, ""
            continue
        if not isinstance(msg, BaseChatMessage):
            continue
        if msg.source == orchestrator_name and isinstance(msg, TextMessage):
            close_step()
            if msg.content.lstrip().startswith(LEDGER_PREFIX):
                instruction, is_plan = PLAN_RECORD_DESCRIPTION, True
            else:
                instruction, is_plan = _clean_instruction(msg.content), False
        elif instruction is not None:
            outputs.append(msg.to_model_text())
            # No default: a missing key gives None (no executor output reported),
            # while "" means the executor ran nothing.
            executed.append((msg.metadata or {}).get(EXECUTED_OUTPUT_KEY))
    close_step()
    return records
