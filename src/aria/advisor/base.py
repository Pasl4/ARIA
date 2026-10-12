"""Minimal completion-client protocol the advisor speaks.

One system and one user message in, text and usage out, so any framework's
model client can be adapted in a few lines:

* AutoGen: aria.frameworks.autogen.AutogenCompletionAdapter
* LangGraph, CrewAI or plain SDKs: aria.advisor.OpenAICompatibleClient
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CompletionUsage:
    """Token counts for one completion, as reported by the provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Cached prompt tokens (billed at a discount), when the API reports them
    # (OpenAI prompt_tokens_details.cached_tokens).
    cached_prompt_tokens: int = 0
    # Thinking tokens, when reported. They are billed as completion tokens and
    # count against max_tokens, so a reasoning model can exhaust the output cap
    # before producing visible text.
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class CompletionResult:
    """Reply text, token usage and truncation flag of one completion."""

    text: str
    usage: CompletionUsage = CompletionUsage()
    # True when the provider stopped because the output cap (max_tokens) was hit.
    truncated: bool = False


@runtime_checkable
class CompletionClient(Protocol):
    """Anything that turns a system and a user message into a CompletionResult."""

    async def complete(self, system: str, user: str) -> CompletionResult: ...
