"""AutoGen model-client helpers.

`build_model_client` turns a `models.*` config block into an AutoGen client
(OpenAI-compatible: OpenAI, Gemini, vLLM, ...). `AutogenCompletionAdapter`
exposes such a client through the advisor's minimal CompletionClient
protocol.
"""

from __future__ import annotations

from typing import Any

from autogen_core.models import ChatCompletionClient, ModelInfo, SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ...advisor.base import CompletionClient, CompletionResult, CompletionUsage


def build_model_client(cfg: dict[str, Any]) -> ChatCompletionClient:
    """Build an OpenAI-compatible AutoGen client from a `models.*` config block.

    Uses model (required) and, when set, api_key, base_url, max_retries,
    temperature, timeout, max_tokens and reasoning_effort; non-OpenAI models
    also need model_info.
    """
    kwargs: dict[str, Any] = {"model": cfg["model"]}
    # Forward max_tokens so an output cap set in the config takes effect.
    for key in ("api_key", "base_url", "max_retries", "temperature", "timeout",
                "max_tokens", "reasoning_effort"):
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    if cfg.get("model_info"):
        kwargs["model_info"] = ModelInfo(**cfg["model_info"])
    return OpenAIChatCompletionClient(**kwargs)


class AutogenCompletionAdapter(CompletionClient):
    """Adapt an AutoGen ChatCompletionClient to the advisor protocol.

    max_tokens, if given, caps the advisor's output per call.
    """

    def __init__(self, client: ChatCompletionClient, max_tokens: int | None = None) -> None:
        self._client = client
        self._extra: dict[str, Any] = {}
        if max_tokens is not None:
            self._extra["max_tokens"] = max_tokens

    async def complete(self, system: str, user: str) -> CompletionResult:
        result = await self._client.create(
            [
                SystemMessage(content=system),
                UserMessage(content=user, source="executor"),
            ],
            extra_create_args=self._extra,
        )
        assert isinstance(result.content, str)
        usage = result.usage
        return CompletionResult(
            text=result.content,
            usage=CompletionUsage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            ),
        )
