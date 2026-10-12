"""RetryOnEmptyClient: retry completions that a provider filtered to nothing.

Some providers discard a whole response (for example Gemini's
"function_call_filter: MALFORMED_FUNCTION_CALL" or its recitation filter),
which autogen returns as an empty string. The coder then ends its turn and
MagenticOne continues with an empty facts sheet and plan, also missing from
the advisor's status snapshot. Retrying in the client covers autogen's calls too.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema

logger = logging.getLogger(__name__)

EMPTY_RESPONSE_EVENT = "model_empty_response"

# Appended to the context on each retry. Recitation filters discard replies
# that quote repository files verbatim, and autogen reports the finish_reason
# as "unknown", so the nudge names the likely cause itself.
RETRY_NUDGE = (
    "Note: your previous reply was discarded by the provider's content "
    "filter — most likely because it reproduced existing file contents "
    "verbatim (recitation). Answer again, paraphrasing or summarizing file "
    "contents instead of quoting them at length; show only the minimal "
    "changed lines."
)


def is_empty(result: CreateResult) -> bool:
    """True when the provider returned nothing usable.

    A non-str `content` is a tool-call answer: not empty, and handled by the
    caller. `thought` counts as content because autogen parks the model's text
    there when it classifies the answer as a tool call.
    """
    content = result.content
    if not isinstance(content, str):
        return False
    return not content.strip() and not (result.thought or "").strip()


class RetryOnEmptyClient(ChatCompletionClient):
    """Wrap a client so a filtered or empty completion is retried and counted.

    Retries up to `max_retries` times with RETRY_NUDGE appended; if every
    attempt is empty, the empty result is returned so the caller's own
    handling applies. Each empty result is counted on the meter (if given) as
    `model_empty_response`, so its rate can be reported per model.
    """

    def __init__(
        self,
        inner: ChatCompletionClient,
        max_retries: int = 3,
        meter: Any | None = None,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._meter = meter

    async def create(self, *args: Any, **kwargs: Any) -> CreateResult:
        result = await self._inner.create(*args, **kwargs)
        for attempt in range(self._max_retries):
            if not is_empty(result):
                return result
            if self._meter is not None:
                self._meter.count_event(EMPTY_RESPONSE_EVENT)
            logger.warning(
                "empty completion (finish_reason=%s, thought=%r); retry %d/%d "
                "with recitation nudge",
                result.finish_reason,
                (result.thought or "")[:80],
                attempt + 1,
                self._max_retries,
            )
            retry_args = args
            if args and isinstance(args[0], (list, tuple)):
                nudged = [*args[0], UserMessage(content=RETRY_NUDGE, source="system")]
                retry_args = (nudged, *args[1:])
            result = await self._inner.create(*retry_args, **kwargs)
        if is_empty(result):
            if self._meter is not None:
                self._meter.count_event(EMPTY_RESPONSE_EVENT)
            logger.error(
                "model returned nothing %d times; passing the empty result through",
                self._max_retries + 1,
            )
        return result

    # --- pure delegation ---------------------------------------------------

    def create_stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[str | CreateResult, None]:
        return self._inner.create_stream(*args, **kwargs)

    async def close(self) -> None:
        await self._inner.close()

    def actual_usage(self) -> RequestUsage:
        return self._inner.actual_usage()

    def total_usage(self) -> RequestUsage:
        return self._inner.total_usage()

    def count_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []
    ) -> int:
        return self._inner.count_tokens(messages, tools=tools)

    def remaining_tokens(
        self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []
    ) -> int:
        return self._inner.remaining_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> Any:
        return self._inner.capabilities

    @property
    def model_info(self) -> ModelInfo:
        return self._inner.model_info
