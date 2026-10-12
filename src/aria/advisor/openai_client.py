"""OpenAI-compatible completion client (OpenAI, Gemini, vLLM, Ollama, ...).

Covers users outside AutoGen (LangGraph, CrewAI, plain scripts) without
pulling in any agent framework. The `openai` package is imported lazily so
the core stays importable without it.
"""

from __future__ import annotations

from typing import Any

from .base import CompletionClient, CompletionResult, CompletionUsage


class OpenAICompatibleClient(CompletionClient):
    """CompletionClient for any endpoint that speaks the OpenAI chat API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 3,
        timeout: float | None = 120.0,
        reasoning_effort: str | None = None,
        # Passed through as the provider's structured-output switch (e.g.
        # {"type": "json_object"}), for callers such as an LLM judge that must
        # reproduce a reference evaluator's request exactly.
        response_format: dict | None = None,
    ) -> None:
        from openai import AsyncOpenAI  # lazy: optional at import time

        self._response_format = response_format

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Sent with every request when set, so the reasoning effort the provider
        # applies matches the one recorded in the run manifest.
        self._reasoning_effort = reasoning_effort
        # `timeout` applies per request; with retries the worst case is about
        # timeout * (max_retries + 1), plus backoff.
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, max_retries=max_retries, timeout=timeout
        )

    async def complete(self, system: str, user: str) -> CompletionResult:
        kwargs: dict = {}
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        if self._reasoning_effort is not None:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if self._response_format is not None:
            kwargs["response_format"] = self._response_format
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        usage = response.usage
        details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        choice = response.choices[0]
        return CompletionResult(
            text=choice.message.content or "",
            # finish_reason "length" means max_tokens cut the reply off. The flag
            # separates a cap set too low from an outage; truncated JSON would
            # otherwise surface only as a generic `advisor_failure`.
            truncated=getattr(choice, "finish_reason", None) == "length",
            usage=CompletionUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cached_prompt_tokens=getattr(details, "cached_tokens", 0) or 0,
                reasoning_tokens=_reasoning_tokens(usage, completion_details),
            ),
        )


def _reasoning_tokens(usage: Any, completion_details: Any) -> int:
    """Return thinking tokens from the explicit field or the arithmetic residual.

    Some OpenAI-compatible endpoints never populate
    `completion_tokens_details.reasoning_tokens` and report thinking only in
    `total_tokens`, as the residual over prompt and completion tokens. The
    explicit field wins when present; the residual is the fallback.
    """
    explicit = int(getattr(completion_details, "reasoning_tokens", 0) or 0)
    if explicit:
        return explicit
    return max(
        0,
        int(getattr(usage, "total_tokens", 0) or 0)
        - int(getattr(usage, "prompt_tokens", 0) or 0)
        - int(getattr(usage, "completion_tokens", 0) or 0),
    )
