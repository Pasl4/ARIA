"""MeteredChatCompletionClient: the executor-side metering point.

Wraps an AutoGen ChatCompletionClient and records every `create` and
`create_stream` call in the TokenMeter under the purpose tag active at call
time (see telemetry/token_meter.py); everything else is delegated. Recording
per call, rather than diffing `total_usage()`, attributes each call exactly
once even when several components share one client.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema

from ...telemetry.token_meter import TokenMeter

logger = logging.getLogger(__name__)

# Client attribute where the probe accumulates cached prompt tokens until
# _record drains them. Not a ContextVar, because autogen runs the SDK request
# in its own asyncio context. A running total keeps the sum exact even if two
# calls on one client interleave.
_CACHED_ATTR = "_aria_cached_prompt_tokens"

# Same accumulator for hidden reasoning tokens (e.g. Gemini thinking tokens).
# Some OpenAI-compatible layers bill them as output but expose them only as
# total_tokens - prompt_tokens - completion_tokens, which autogen's
# RequestUsage drops; without this, metered cost would omit them.
_REASONING_ATTR = "_aria_reasoning_tokens"


def _take_int(inner: object, attr: str) -> int:
    """Read and reset one of the probe's accumulators."""
    value = int(getattr(inner, attr, 0) or 0)
    if value:
        setattr(inner, attr, 0)
    return value


def _take_cached_tokens(inner: object) -> int:
    return _take_int(inner, _CACHED_ATTR)


def _install_cache_probe(inner: ChatCompletionClient) -> bool:
    """Patch the SDK's `create` to capture cached and reasoning token counts.

    autogen's `RequestUsage` drops both; the probe recovers them so executor
    cost applies cache discounts like the advisor client and includes hidden
    reasoning tokens. Best effort: if the SDK internals change, metering still
    works but both counts are lost, so executor cost is no longer exact.
    Streaming responses are not covered. Returns False if nothing was patched.
    """
    completions = getattr(getattr(inner, "_client", None), "chat", None)
    completions = getattr(completions, "completions", None)
    original = getattr(completions, "create", None)
    if completions is None or original is None or getattr(original, "_aria_probe", False):
        return False

    async def create(*args: Any, **kwargs: Any) -> Any:
        response = await original(*args, **kwargs)
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        hit = int(getattr(details, "cached_tokens", 0) or 0)
        if hit:
            setattr(inner, _CACHED_ATTR, getattr(inner, _CACHED_ATTR, 0) + hit)
        # Reasoning tokens: prefer the explicit field, else use the residual
        # total - prompt - completion, clamped at 0 when total_tokens is
        # missing or smaller than prompt + completion.
        out_details = getattr(usage, "completion_tokens_details", None)
        thinking = int(getattr(out_details, "reasoning_tokens", 0) or 0)
        if not thinking and usage is not None:
            thinking = max(
                0,
                int(getattr(usage, "total_tokens", 0) or 0)
                - int(getattr(usage, "prompt_tokens", 0) or 0)
                - int(getattr(usage, "completion_tokens", 0) or 0),
            )
        if thinking:
            setattr(inner, _REASONING_ATTR, getattr(inner, _REASONING_ATTR, 0) + thinking)
        return response

    create._aria_probe = True  # type: ignore[attr-defined]
    completions.create = create
    return True


class MeteredChatCompletionClient(ChatCompletionClient):
    def __init__(
        self,
        inner: ChatCompletionClient,
        meter: TokenMeter,
        role: str = "executor",
        model_label: str = "",
        default_purpose: str = "agent_step",
    ) -> None:
        self._inner = inner
        self._meter = meter
        self._role = role
        self._model_label = model_label
        self._default_purpose = default_purpose
        if not _install_cache_probe(inner):
            logger.debug(
                "cached-token probe not installed for %s; executor cost stays "
                "an upper bound",
                model_label or type(inner).__name__,
            )

    def _record(self, result: CreateResult, latency_s: float = 0.0) -> None:
        usage = result.usage
        # Count answers cut off at max_tokens, like `advisor_truncated`. A
        # truncated call can end a turn mid-patch, and it means the
        # completion-token distribution is censored at the cap.
        if result.finish_reason == "length":
            self._meter.count_event(f"{self._role}_truncated")
        # Drain the accumulators so no value carries over to the next call.
        cached = _take_cached_tokens(self._inner)
        reasoning = _take_int(self._inner, _REASONING_ATTR)
        self._meter.record_call(
            role=self._role,
            model=self._model_label,
            prompt_tokens=int(usage.prompt_tokens or 0),
            completion_tokens=int(usage.completion_tokens or 0),
            reasoning_tokens=reasoning,
            cached_prompt_tokens=min(cached, int(usage.prompt_tokens or 0)),
            latency_s=latency_s,
            cache_hit=bool(result.cached),
            default_purpose=self._default_purpose,
        )

    # ------------------------------------------------------------------
    # ChatCompletionClient interface (metered)
    # ------------------------------------------------------------------

    def _record_failure(self, exc: BaseException, latency_s: float) -> None:
        """Record a call that raised: zero tokens, its latency and error type.

        Keeps failed calls (for example read timeouts) and the wall-clock time
        they consumed visible in calls.jsonl.
        """
        # Drain both accumulators so they do not carry over to the next call.
        _take_cached_tokens(self._inner)
        _take_int(self._inner, _REASONING_ATTR)
        self._meter.record_call(
            role=self._role,
            model=self._model_label,
            prompt_tokens=0,
            completion_tokens=0,
            latency_s=latency_s,
            failed=type(exc).__name__,
            default_purpose=self._default_purpose,
        )

    async def create(self, *args: Any, **kwargs: Any) -> CreateResult:
        started = time.monotonic()
        try:
            result = await self._inner.create(*args, **kwargs)
        except BaseException as exc:
            self._record_failure(exc, time.monotonic() - started)
            raise
        self._record(result, time.monotonic() - started)
        return result

    def create_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[str | CreateResult, None]:
        inner_stream = self._inner.create_stream(*args, **kwargs)

        async def metered() -> AsyncGenerator[str | CreateResult, None]:
            started = time.monotonic()
            try:
                async for chunk in inner_stream:
                    if isinstance(chunk, CreateResult):
                        self._record(chunk, time.monotonic() - started)
                    yield chunk
            except BaseException as exc:
                self._record_failure(exc, time.monotonic() - started)
                raise

        return metered()

    # ------------------------------------------------------------------
    # Pure delegation
    # ------------------------------------------------------------------

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
