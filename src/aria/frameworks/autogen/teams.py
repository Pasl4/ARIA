"""Build AutoGen participants from a TeamSpec.

- coder: vendored Magentic-UI CoderAgent (write/execute/debug loop)
- file_surfer: AssistantAgent with the registry's read-only file tools
- assistant: AssistantAgent with its declared tools except python_exec
- code_executor: CodeExecutorAgent over the workspace executor
- advisor_liaison (condition "organic"): holds the consult tool; the team picks when to consult
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.base import ChatAgent
from autogen_core.models import ChatCompletionClient

from ..spec import AgentSpec, TeamSpec
from ..tools import ToolContext, resolve_tool
from .agents import CoderAgent

# Name of the consult liaison, which exists only in teams of condition
# "organic" (config.organic_consult). team.py keys its organic-only handling
# on this name, so it is defined once here.
LIAISON_NAME = "advisor_liaison"


def build_participants(
    spec: TeamSpec,
    model_client: ChatCompletionClient,
    prompt_vars: Mapping[str, Any],
    tool_context: ToolContext,
    include_consult_tool: bool = False,
) -> list[ChatAgent]:
    """Build one native agent per AgentSpec, plus the liaison if requested."""
    participants: list[ChatAgent] = []
    consult_tool = (
        resolve_tool("consult_advisor", "autogen", tool_context)
        if include_consult_tool
        else None
    )

    for agent in spec.agents:
        participants.append(
            _build_agent(agent, model_client, prompt_vars, tool_context)
        )

    if consult_tool is not None:
        # The tool lives on a dedicated liaison, not on a roster agent: the
        # MagenticOne orchestrator selects speakers by agent description, so a
        # tool attached to another agent is invisible at routing time.
        participants.append(
            AssistantAgent(
                LIAISON_NAME,
                model_client=model_client,
                description=(
                    "An agent that can consult a senior expert advisor for "
                    "strategic guidance when the team is stuck or about to "
                    "commit to an approach, and once before declaring the "
                    "task done. Consulting is expensive; it adds most of its "
                    "value before the approach crystallizes."
                ),
                system_message=(
                    "You relay the team's situation to an expert advisor via the "
                    "consult_advisor tool and report the guidance back verbatim."
                ),
                tools=[consult_tool],
            )
        )

    return participants


def _build_agent(
    agent: AgentSpec,
    model_client: ChatCompletionClient,
    prompt_vars: Mapping[str, Any],
    tool_context: ToolContext,
    extra_tools: list[Any] | None = None,
) -> ChatAgent:
    prompt = agent.render_prompt(prompt_vars) if agent.prompt_template else ""

    if agent.kind == "coder":
        executor = resolve_tool("python_exec", "autogen", tool_context)
        return CoderAgent(
            agent.name,
            model_client=model_client,
            code_executor=executor,
            system_prompt=prompt,
            description=agent.description,
            max_debug_rounds=int(agent.params.get("max_debug_rounds", 3)),
        )

    if agent.kind == "file_surfer":
        tools = list(resolve_tool("file_browser", "autogen", tool_context))
        tools += extra_tools or []
        return AssistantAgent(
            agent.name,
            model_client=model_client,
            description=agent.description,
            system_message=prompt,
            tools=tools,
        )

    if agent.kind == "assistant":
        tools = [
            resolve_tool(symbol, "autogen", tool_context)
            for symbol in agent.tools
            if symbol != "python_exec"
        ]
        tools += extra_tools or []
        return AssistantAgent(
            agent.name,
            model_client=model_client,
            description=agent.description,
            system_message=prompt,
            tools=tools or None,
        )

    if agent.kind == "code_executor":
        executor = resolve_tool("python_exec", "autogen", tool_context)
        return CodeExecutorAgent(
            agent.name, code_executor=executor, description=agent.description or None
        )

    raise ValueError(f"unknown agent kind {agent.kind!r}")
