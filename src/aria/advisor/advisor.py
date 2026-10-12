"""The advisor: a stronger, more expensive model consulted at chosen moments.

Framework-agnostic (any CompletionClient). On CONSULT it receives the executor
status (front-trimmed to `max_context_chars`) and returns guidance under a word
budget. A failed call or unparseable reply gets one repair retry, then a
`failed` sentinel reply. The assessment never feeds the reward (core/reward.py),
which would be circular; an off_track verdict can trigger a gate revision.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .base import CompletionClient

if TYPE_CHECKING:
    from ..telemetry.token_meter import TokenMeter

logger = logging.getLogger(__name__)

AdvisorStyle = Literal["strategic", "redirect", "full_plan"]

ASSESSMENT_ON_TRACK = "on_track"
ASSESSMENT_OFF_TRACK = "off_track"
ASSESSMENT_FAILED = "failed"

_STYLE_INSTRUCTIONS: dict[AdvisorStyle, str] = {
    "strategic": (
        "Give strategic guidance only: assess the current approach and point out "
        "risks or the single most important course correction. Do not solve the "
        "task yourself and do not write a new plan."
    ),
    "redirect": (
        "If the executor is off track, give one concrete redirect: name the exact "
        "step to change and what to do instead. If on track, confirm briefly."
    ),
    "full_plan": (
        "Rewrite the remaining plan from the current state: a short numbered list "
        "of the steps the executor should take from here."
    ),
}

# Verification nudge for MAST FC3 task-verification failures (arXiv:2503.13657),
# toggled by `advisor.fc3_nudge` and recorded in the run manifest. It lives in the
# advisor prompt, not the executor prompts, so all consulting conditions share the
# advice content and differ only in consult timing.
_FC3_NUDGE = (
    "Multi-agent runs most often fail by stopping early, by never checking the "
    "result, or by trusting a check that could not have failed. Weigh those "
    "against what the executor has actually shown you."
)

_SYSTEM_TEMPLATE = (
    "You are a senior expert advisor to a smaller AI agent (the executor) working "
    "through a multi-step task. You are consulted rarely; make it count.\n\n"
    "{style_instruction}\n\n"
    "Keep your guidance under {word_budget} words — the executor needs a focused "
    "course input, not a comprehensive plan.\n\n"
    "{fc3_nudge}"
    "Respond with ONLY a JSON object (no markdown fences):\n"
    '{{"assessment": "on_track|off_track", "guidance": "<your advice, concise>"}}'
)

# ---- Repository inspection (opt-in via `advisor.repo_access`)
_TOOL_INSTRUCTIONS = (
    "\n\nYou may inspect the repository before answering. To do so, respond "
    "with ONLY {{\"tool_command\": \"<command>\"}} — allowed commands: cat, grep, ls, "
    "head, tail, sed -n, find, rg, wc, and read-only git (diff/status/log/show); "
    "relative paths only. You have up to {max_tools} inspections. At a "
    "pre-delivery consult, VERIFY the actual workspace changes (git diff) "
    "against the task before assessing — do not endorse work you have not "
    "checked. In your final guidance, name the single most important "
    "divergence concretely (file and symptom) and the next concrete action."
)

_TOOL_WHITELIST = {"cat", "grep", "ls", "head", "tail", "sed", "find", "rg", "wc", "git"}
_GIT_READONLY = {"diff", "status", "log", "show"}


def _run_repo_command(workspace, cmd: str, cap: int = 8000, executor=None) -> str:
    """Run one whitelisted inspection command inside the workspace.

    The check covers program names, sed -i, git subcommands and, on the host,
    absolute or parent paths; write-capable options such as find -delete/-exec
    or sed w are not blocked. `executor` is an optional backend that runs the
    validated argv elsewhere, e.g. in the episode container of a benchmark
    whose graded state is not in the host workspace. Validation stays here.
    """
    import shlex
    import subprocess

    try:
        argv = shlex.split(cmd)
    except ValueError:
        return "[tool error: unparseable command]"
    if not argv or argv[0] not in _TOOL_WHITELIST:
        return f"[tool error: only {sorted(_TOOL_WHITELIST)} are allowed]"
    if argv[0] == "git" and (len(argv) < 2 or argv[1] not in _GIT_READONLY):
        return "[tool error: only read-only git subcommands (diff/status/log/show)]"
    if argv[0] == "sed" and any(a.startswith("-i") for a in argv[1:]):
        return "[tool error: sed -i (in-place write) is not allowed]"
    # Path containment applies only to the host backend, where an absolute or
    # parent path would leave the workspace. A container backend targets an
    # ephemeral task container whose graded state often sits at an absolute
    # path named in the task (e.g. /app), so such paths are allowed there.
    if executor is None:
        for a in argv[1:]:
            if a.startswith("/") or ".." in a:
                return "[tool error: absolute or parent-directory paths are not allowed]"
    if executor is not None:
        return executor(argv, cap)
    try:
        r = subprocess.run(
            argv, cwd=str(workspace), capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        return "[tool error: timeout]"
    except Exception as exc:  # noqa: BLE001
        return f"[tool error: {type(exc).__name__}]"
    out = (r.stdout + ("\n" + r.stderr if r.stderr else "")).strip()
    return out[:cap] if out else f"[exit {r.returncode}, no output]"


_REPAIR_PROMPT = (
    "Your previous reply was not valid JSON. Respond again with ONLY the JSON "
    'object {"assessment": "on_track|off_track", "guidance": "..."} and nothing else.'
)


@dataclass(frozen=True)
class AdvisorReply:
    """Outcome of one consult: the advisor's assessment and guidance text."""

    assessment: str  # on_track | off_track | failed
    guidance: str
    # repo_access only: transcript of the repository inspections made during
    # this consult, persisted so the advice can be audited.
    tool_log: str = ""

    @property
    def failed(self) -> bool:
        return self.assessment == ASSESSMENT_FAILED


def _reply_from(parsed: dict[str, object], tool_log: str) -> AdvisorReply:
    assessment = str(parsed.get("assessment", ASSESSMENT_ON_TRACK)).lower()
    if assessment not in (ASSESSMENT_ON_TRACK, ASSESSMENT_OFF_TRACK):
        assessment = ASSESSMENT_ON_TRACK
    return AdvisorReply(
        assessment=assessment,
        guidance=str(parsed.get("guidance", "")),
        tool_log=tool_log,
    )


def _parse_json_object(raw: str) -> dict[str, object]:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    return json.loads(raw)


class Advisor:
    """Builds the advisor prompt, runs optional inspection rounds and parses the reply."""

    def __init__(
        self,
        client: CompletionClient,
        style: AdvisorStyle = "strategic",
        meter: TokenMeter | None = None,
        domain_context: str = "",
        max_guidance_words: int = 80,
        fc3_nudge: bool = True,
        max_context_chars: int = 60_000,
        max_header_chars: int = 3_000,
        repo_access: bool = False,
        max_tool_calls: int = 3,
        gate_diff_injection: bool = False,
    ) -> None:
        # Token recording happens at the client layer (wrap `client` in a
        # MeteredCompletionClient); the meter here is only for event counts.
        self._client = client
        self._style: AdvisorStyle = style
        self._meter = meter
        self._domain_context = domain_context
        self._max_guidance_words = max_guidance_words
        self._fc3_nudge = fc3_nudge
        self._max_context_chars = max_context_chars
        self._max_header_chars = max_header_chars
        self._repo_access = repo_access
        self._max_tool_calls = max_tool_calls
        self._gate_diff_injection = gate_diff_injection
        # Set per episode by the framework runner when repo_access is on; with
        # neither a workspace nor an inspector, the consult runs without inspection.
        self.workspace = None
        # Optional per-episode inspection backend from
        # BenchmarkAdapter.advisor_inspector; None runs commands on the host in
        # `workspace`, the default unless an adapter keeps its state elsewhere.
        self.inspector = None

    def _fit_context(self, status: str) -> str:
        """Apply the declared context budget, visibly.

        Trims from the front: a trajectory is chronological and the work under
        review is at the end. The trim is marked in the prompt, logged and
        counted as `advisor_context_trimmed`.
        """
        budget = self._max_context_chars
        if budget <= 0 or len(status) <= budget:
            return status
        dropped = len(status) - budget
        if self._meter is not None:
            self._meter.count_event("advisor_context_trimmed")
        logger.warning("advisor context trimmed by %d characters", dropped)
        return (
            f"[NOTE: the first {dropped} characters of this trajectory were "
            f"omitted to fit the advisor's declared context budget of {budget} "
            f"characters. What follows is the most recent work.]\n\n"
            f"{status[-budget:]}"
        )

    async def consult(
        self, executor_status: str, pinned_header: str = "", is_gate: bool = False
    ) -> AdvisorReply:
        """Consult the advisor; a failed call or unparseable reply yields a `failed` reply.

        `pinned_header` (the task text) is prepended after the front-trim, so
        it is never trimmed away. It holds only text already in the executor's
        context and is capped at `max_header_chars`. With `is_gate` and
        `gate_diff_injection`, the workspace diff (or the inspector's snapshot) is
        appended.
        """
        system = _SYSTEM_TEMPLATE.format(
            style_instruction=_STYLE_INSTRUCTIONS[self._style],
            word_budget=self._max_guidance_words,
            fc3_nudge=f"{_FC3_NUDGE}\n\n" if self._fc3_nudge else "",
        )
        if self._domain_context:
            system += f"\n\nDomain context:\n{self._domain_context}"
        inspect_enabled = self._repo_access and (
            self.workspace is not None or self.inspector is not None
        )
        if inspect_enabled:
            system += _TOOL_INSTRUCTIONS.format(max_tools=self._max_tool_calls)
            hint = getattr(self.inspector, "prompt_hint", None)
            if callable(hint):
                system += hint()  # pyright: ignore[reportOperatorIssue]

        executor_status = self._fit_context(executor_status)
        if pinned_header:
            executor_status = f"{pinned_header[:self._max_header_chars]}\n\n{executor_status}"
        # Gate verification: append the actual workspace changes so the verdict
        # does not depend on the advisor choosing to inspect; tool rounds still
        # add depth. Appended after _fit_context so the front-trim never drops it.
        if is_gate and self._gate_diff_injection and (
            self.workspace is not None or self.inspector is not None
        ):
            if self._meter is not None:
                self._meter.count_event("advisor_auto_diff")
        if is_gate and self._gate_diff_injection and self.inspector is not None:
            # A custom backend has no git tree, so the git wording below does not
            # apply; the backend supplies its own heading and body.
            heading, body = self.inspector.gate_snapshot(20_000)  # pyright: ignore[reportGeneralTypeIssues]
            executor_status += f"\n\n{heading}{body}"
        elif is_gate and self._gate_diff_injection and self.workspace is not None:
            diff = _run_repo_command(self.workspace, "git diff", cap=20_000)
            executor_status += (
                "\n\n[PRE-DELIVERY CONSULT] The executor believes the task is "
                "complete and is about to finalize. Below is the ACTUAL "
                "workspace diff — this is the work you are endorsing. An "
                "empty diff means no file was ever changed, whatever the "
                "transcript claims.\n$ git diff\n"
                f"{diff}"
            )
        # Tool rounds: the advisor may inspect the repo before answering.
        tool_log = ""
        if inspect_enabled:
            for _ in range(self._max_tool_calls):
                try:
                    result = await self._client.complete(system, executor_status)
                    parsed = _parse_json_object(result.text)
                except Exception:  # noqa: BLE001 - fall through to answer loop
                    break
                cmd = parsed.get("tool_command")
                if not isinstance(cmd, str) or not cmd.strip():
                    # A reply without tool_command is usually the final answer;
                    # return it instead of paying for the identical prompt again.
                    if "assessment" in parsed or "guidance" in parsed:
                        if self._meter is not None:
                            self._meter.count_event("advisor_answer_in_tool_loop")
                        return _reply_from(parsed, tool_log)
                    break  # parsed, but neither tool nor answer -> repair path
                output = _run_repo_command(
                    self.workspace, cmd,
                    executor=getattr(self.inspector, "exec", None),
                )
                if self._meter is not None:
                    self._meter.count_event("advisor_tool_call")
                tool_log += f"$ {cmd}\n{output}\n"
                executor_status += (
                    f"\n\n[Repository inspection] $ {cmd}\n{output}\n"
                    "[Continue: inspect further or give your final JSON answer.]"
                )

        # The client is stateless, so the repair turn re-sends the full status
        # with the repair instruction appended.
        repair_user = f"{executor_status}\n\n{_REPAIR_PROMPT}"
        for attempt, user in enumerate((executor_status, repair_user)):
            try:
                result = await self._client.complete(system, user)
                if result.truncated and self._meter is not None:
                    # Counted apart from advisor_failure: hitting the output cap
                    # means max_advice_tokens is too low for this model, not an outage.
                    self._meter.count_event("advisor_truncated")
                    logger.warning("advisor output hit the max_tokens cap")
                parsed = _parse_json_object(result.text)
            except Exception:
                logger.warning("advisor call failed (attempt %d)", attempt + 1, exc_info=True)
                continue
            return _reply_from(parsed, tool_log)

        if self._meter is not None:
            self._meter.count_event("advisor_failure")
        return AdvisorReply(assessment=ASSESSMENT_FAILED, guidance="", tool_log=tool_log)
