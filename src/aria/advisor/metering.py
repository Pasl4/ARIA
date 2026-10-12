"""MeteredCompletionClient: the advisor-side metering point.

Wraps any CompletionClient and records every call in a TokenMeter. It is
framework-free, so LangGraph or CrewAI users get the same per-call accounting
as the AutoGen integration. Calls default to purpose "consult"; the organic
consult tool overrides it with `call_purpose("organic_consult")`.
"""

from __future__ import annotations

import time

from ..telemetry.token_meter import TokenMeter
from .base import CompletionClient, CompletionResult


class MeteredCompletionClient(CompletionClient):
    def __init__(
        self,
        inner: CompletionClient,
        meter: TokenMeter,
        role: str = "advisor",
        model_label: str = "",
        default_purpose: str = "consult",
    ) -> None:
        self._inner = inner
        self._meter = meter
        self._role = role
        self._model_label = model_label
        self._default_purpose = default_purpose

    async def complete(self, system: str, user: str) -> CompletionResult:
        # Timed like executor calls, so consult latency is recorded.
        started = time.monotonic()
        try:
            result = await self._inner.complete(system, user)
        except BaseException as exc:
            # A call that raises is recorded with its latency, zero tokens and
            # the exception name, then re-raised.
            self._meter.record_call(
                role=self._role,
                model=self._model_label,
                prompt_tokens=0,
                completion_tokens=0,
                latency_s=time.monotonic() - started,
                failed=type(exc).__name__,
                default_purpose=self._default_purpose,
            )
            raise
        self._meter.record_call(
            role=self._role,
            model=self._model_label,
            latency_s=time.monotonic() - started,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            # Thinking tokens are billed at the completion rate; omitting them
            # understates consult cost for reasoning models.
            reasoning_tokens=result.usage.reasoning_tokens,
            cached_prompt_tokens=result.usage.cached_prompt_tokens,
            default_purpose=self._default_purpose,
        )
        return result
