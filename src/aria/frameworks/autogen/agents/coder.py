"""CoderAgent, vendored from Magentic-UI (magentic_ui.agents._coder) and trimmed.

Keeps the write/execute/debug loop (per-block execution, error framing,
`max_debug_rounds` default 3, combined transcript). Omits approval guards,
pause/resume, Docker bootstrap, output summarization, image handling and the
token-limited context. Executor, system prompt and description come from the
tool registry and TeamSpec.
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator, Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

logger = logging.getLogger(__name__)

# Message-metadata key carrying executor-only output (see on_messages_stream).
EXECUTED_OUTPUT_KEY = "aria_executed_output"

_CODE_BLOCK_PATTERN = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")

# Only fences with these tags are executed (the vendored agent runs every
# fence). Other fences such as diff, patch, console or untagged text are
# content, not actions; executing them fails with "unknown language" and
# would add a spurious error to the state's error streak.
_EXECUTABLE_LANGUAGES = frozenset({"python", "py", "sh", "bash", "shell"})


def extract_markdown_code_blocks(markdown_text: str) -> list[CodeBlock]:
    """Return all fenced code blocks, as Magentic-UI's _extract_markdown_code_blocks does."""
    blocks: list[CodeBlock] = []
    for language, code in _CODE_BLOCK_PATTERN.findall(markdown_text):
        blocks.append(CodeBlock(code=code, language=(language or "").strip()))
    return blocks


def executable_code_blocks(markdown_text: str) -> list[CodeBlock]:
    """Return the fenced blocks whose language the executor runs."""
    return [
        block
        for block in extract_markdown_code_blocks(markdown_text)
        if block.language.lower() in _EXECUTABLE_LANGUAGES
    ]


def _thread_to_context(
    thread: Sequence[BaseChatMessage], agent_name: str
) -> list[LLMMessage]:
    context: list[LLMMessage] = []
    for message in thread:
        if not isinstance(message, TextMessage):
            continue
        if message.source.startswith(agent_name):
            context.append(AssistantMessage(content=message.content, source=message.source))
        else:
            context.append(UserMessage(content=message.content, source=message.source))
    return context


class CoderAgent(BaseChatAgent):
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        code_executor: CodeExecutor,
        system_prompt: str,
        description: str,
        max_debug_rounds: int = 3,
    ) -> None:
        super().__init__(name, description)
        self._model_client = model_client
        self._code_executor = code_executor
        self._system_prompt = system_prompt
        self._max_debug_rounds = max_debug_rounds
        self._chat_history: list[BaseChatMessage] = []
        self._started = False

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)

    async def _lazy_start(self) -> None:
        if not self._started:
            start = getattr(self._code_executor, "start", None)
            if start is not None:
                await start()
            self._started = True

    async def on_messages(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        response: Response | None = None
        async for item in self.on_messages_stream(messages, cancellation_token):
            if isinstance(item, Response):
                response = item
        assert response is not None
        return response

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        await self._lazy_start()
        self._chat_history.extend(messages)
        inner: list[BaseChatMessage] = []

        for _round in range(self._max_debug_rounds):
            context = [SystemMessage(content=self._system_prompt)] + _thread_to_context(
                [*self._chat_history, *inner], self.name
            )
            # Empty or filtered answers are retried by the wrapping
            # RetryOnEmptyClient; a second retry loop here would multiply calls.
            create_result = await self._model_client.create(
                messages=context, cancellation_token=cancellation_token
            )
            # A model may answer with tool calls although this agent declares
            # none; `content` is then a list of FunctionCall and autogen moves
            # the text, which holds the code blocks, into `thought`. Recover it
            # instead of failing; no text at all ends the round.
            content = create_result.content
            if not isinstance(content, str):
                recovered = create_result.thought or ""
                logger.warning(
                    "%s: model answered with %s instead of text; "
                    "recovered %d chars from `thought`",
                    self.name,
                    type(content).__name__,
                    len(recovered),
                )
                content = recovered
            code_msg = TextMessage(source=f"{self.name}-llm", content=content)
            inner.append(code_msg)

            code_blocks = executable_code_blocks(content)
            if not code_blocks:
                break

            all_ok = True
            for i, block in enumerate(code_blocks):
                try:
                    result = await self._code_executor.execute_code_blocks(
                        [block], cancellation_token
                    )
                    exit_code, output = result.exit_code or 0, result.output
                except Exception as exc:  # framing identical to Magentic-UI
                    exit_code, output = 1, (
                        f"An exception occurred while executing the code block: {exc}"
                    )
                if output.strip() == "" and exit_code == 0:
                    output = (
                        "The script ran but produced no output to console. The POSIX "
                        f"exit code was: {exit_code}. If you were expecting output, "
                        "consider revising the script to ensure content is printed "
                        "to stdout."
                    )
                elif exit_code != 0:
                    all_ok = False
                    output = (
                        f"The script ran, then exited with an error (POSIX exit code: "
                        f"{exit_code})\nIts output was:\n{output}"
                    )
                inner.append(
                    TextMessage(
                        source=f"{self.name}-executor",
                        content=(
                            f"Execution result of code block {i + 1}:\n"
                            f"```console\n{output}\n```"
                        ),
                    )
                )
            if all_ok:
                break

        self._chat_history.extend(inner)
        combined = "\n".join(m.content for m in inner if isinstance(m, TextMessage))
        # Executor output travels in `metadata`, which to_model_text() excludes,
        # so the model cannot write into it. Detection reads the metadata; a
        # model imitating the "Execution result of code block N:" framing in
        # its own message therefore cannot forge a passing test run.
        executed = "\n".join(
            m.content
            for m in inner
            if isinstance(m, TextMessage) and m.source == f"{self.name}-executor"
        )
        yield Response(
            chat_message=TextMessage(
                source=self.name,
                content=combined or "No output.",
                metadata={EXECUTED_OUTPUT_KEY: executed},
            ),
            inner_messages=list(inner),
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._chat_history = []
