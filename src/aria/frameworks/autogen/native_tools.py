"""AutoGen tool factories for the tool registry, registered at package import.

- python_exec: LocalCommandLineCodeExecutor bound to the workspace, used as
  the CoderAgent's executor.
- file_browser: read-only file FunctionTools bound to the workspace; a small
  stand-in for Magentic-UI's MarkdownFileBrowser covering what the file_surfer
  prompt describes.
- consult_advisor: the organic-routing tool (condition "organic").
"""

from __future__ import annotations

import logging
from pathlib import Path

from autogen_core.code_executor import CodeExecutor
from autogen_core.tools import FunctionTool
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

from ..tools import ToolContext, register_tool
from .consult_tool import make_consult_advisor_tool

logger = logging.getLogger(__name__)

_MAX_READ_CHARS = 20_000


def _python_exec(context: ToolContext) -> CodeExecutor:
    work_dir = context.workspace or Path.cwd()
    # Use the configured timeout, not the vendor default (60 s): a test suite
    # killed by the timeout looks like a failing one.
    return LocalCommandLineCodeExecutor(
        work_dir=work_dir, timeout=context.exec_timeout_s
    )


def _file_browser(context: ToolContext) -> list[FunctionTool]:
    root = Path(context.workspace or Path.cwd()).resolve()

    def _resolve(rel: str) -> Path:
        candidate = (root / rel).resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError(f"path {rel!r} escapes the task workspace")
        return candidate

    async def open_path(path: str) -> str:
        """Open a file (returns its text content) or list a directory."""
        target = _resolve(path)
        if target.is_dir():
            return "\n".join(sorted(p.name for p in target.iterdir()))
        return target.read_text(errors="replace")[:_MAX_READ_CHARS]

    async def list_current_directory() -> str:
        """List the files in the task working directory."""
        return "\n".join(sorted(p.name for p in root.iterdir()))

    async def find_file(query: str) -> str:
        """Find files whose name contains the query."""
        matches = [str(p.relative_to(root)) for p in root.rglob(f"*{query}*")]
        return "\n".join(sorted(matches)[:50]) or "no matches"

    return [
        FunctionTool(open_path, name="open_path", description=open_path.__doc__ or ""),
        FunctionTool(
            list_current_directory,
            name="list_current_directory",
            description=list_current_directory.__doc__ or "",
        ),
        FunctionTool(find_file, name="find_file", description=find_file.__doc__ or ""),
    ]


def _consult_advisor(context: ToolContext) -> FunctionTool:
    assert context.advisor is not None, "organic consult tool needs an advisor"
    return make_consult_advisor_tool(
        context.advisor,
        context.meter,
        status_provider=context.status_provider,
        cost_multiplier=context.cost_multiplier,
        header_provider=context.header_provider,
        consult_budget=context.consult_budget,
    )


_registered = False


def register() -> None:
    """Register the autogen tool factories; idempotent, called at package import."""
    global _registered
    if _registered:
        return
    register_tool("python_exec", "autogen", _python_exec)
    register_tool("file_browser", "autogen", _file_browser)
    register_tool("consult_advisor", "autogen", _consult_advisor)
    _registered = True
