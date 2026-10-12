"""Consult tool for organic routing (condition "organic").

The executor decides when to consult, steered only by its system prompt and the
tool description (modeled on Anthropic's advisor-tool guidance; cost multiplier
from pricing). Consults share the status snapshot, pinned header and per-episode
cap of policy-path consults. Calls under the cap count as organic_consult before
the advisor replies; calls over it return a notice instead (organic_consult_capped).
"""

from __future__ import annotations

from collections.abc import Callable

from autogen_core.tools import FunctionTool

from ...advisor import Advisor
from ...telemetry.token_meter import TokenMeter, call_purpose

CONSULT_TOOL_DESCRIPTION_TEMPLATE = (
    "Consult a senior expert advisor model that sees your full progress so far. "
    "This is EXPENSIVE — roughly {cost_multiplier} your own cost per call. "
    "Call it BEFORE substantive work (before committing to an interpretation or "
    "approach — orientation like reading and locating things is not substantive "
    "work), when you are stuck (errors recurring, approach not converging), and "
    "once before declaring the task done. The advisor adds most of its value on "
    "the first call, before the approach crystallizes; do not call it for steps "
    "whose next action is dictated by output you just read. Describe your "
    "situation and question precisely."
)


def make_consult_advisor_tool(
    advisor: Advisor,
    meter: TokenMeter | None = None,
    status_provider: Callable[[], str] | None = None,
    cost_multiplier: str = "10x",
    header_provider: Callable[[], str] | None = None,
    consult_budget: int | None = None,
) -> FunctionTool:
    """Build the consult_advisor tool for one episode.

    status_provider returns the controller's latest status snapshot (typically
    `lambda: controller.latest_status`); header_provider returns the episode's
    pinned task header, the only way the advisor sees the task text.
    consult_budget is the per-episode cap (None: uncapped). The call counter
    lives in the tool, so build a new tool per episode.
    """
    consults_used = 0

    async def consult_advisor(question: str) -> str:
        nonlocal consults_used
        if consult_budget is not None and consults_used >= consult_budget:
            if meter is not None:
                meter.count_event("organic_consult_capped")
            return (
                "[consult_advisor unavailable: the consult budget for this "
                "task is exhausted. Continue with your own judgment.]"
            )
        consults_used += 1
        if meter is not None:
            meter.count_event("organic_consult")
        status = status_provider() if status_provider is not None else ""
        payload = f"{status}\n\nExecutor's question:\n{question}" if status else question
        header = header_provider() if header_provider is not None else ""
        with call_purpose("organic_consult"):
            reply = await advisor.consult(payload, pinned_header=header)
        return reply.guidance

    return FunctionTool(
        consult_advisor,
        name="consult_advisor",
        description=CONSULT_TOOL_DESCRIPTION_TEMPLATE.format(cost_multiplier=cost_multiplier),
    )
