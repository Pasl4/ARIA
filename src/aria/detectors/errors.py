"""Deterministic error extraction from raw step output.

Framework-agnostic: every integration passes its native log text through this
module, so the error signal is derived the same way everywhere. Most of the
framing it recognizes comes from the coder agent. The signal feeds the
exec3/outcome3 dims and the evidence covariate, so the patterns anchor to
actual failure syntax and ignore prose. Validate them on real transcripts.
"""

from __future__ import annotations

import re

_ERROR_PATTERNS = re.compile(
    r"(Traceback \(most recent call last\)"  # python traceback
    r"|^\s*(?:[A-Za-z_][A-Za-z0-9_.]*Error|Exception):"  # exception line
    r"|^\s*AssertionError\b"
    r"|\bexitcode:\s*[1-9]\d*\b"  # code-executor exit status
    r"|\bexit code:?\s+[1-9]\d*\b"  # incl. the coder's parenthesized exit-code framing
    r"|\bnon-zero exit\b"
    r"|^FAILED\b"  # pytest per-test line, anchored to line start
    r"|^CHECK FAILED\b"  # framing of a task-provided check() helper (code-emitted)
    r"|\b[1-9]\d* failed\b)",  # pytest summary ("3 failed"), nonzero only
    re.MULTILINE,
)


_ERROR_REGION_CAP = 1000

# The coder joins its internal write -> execute -> debug rounds into one message
# and frames each execution as "Execution result of code block N:" (see
# frameworks/autogen/agents/coder.py). tests/frameworks/test_coder_agent.py
# guards this coupling, so this module needs no import from frameworks/.
_EXECUTION_SEGMENT_MARKER = re.compile(
    r"^Execution result of code block \d+:\s*$", re.MULTILINE
)


def last_execution_segment(output: str) -> str:
    """Return the output after the last execution-framing marker.

    This is how the step ended. End-state judgments scope to it so that a stale
    signal from an earlier round cannot override the final outcome.
    """
    segments = list(_EXECUTION_SEGMENT_MARKER.finditer(output))
    return output[segments[-1].end() :] if segments else output


def execution_segments(output: str) -> list[str]:
    """Return every execution segment in order, or [] without execution framing.

    A segment is the text after one framing marker up to the next. One step can
    aggregate many code blocks, so judging only the last segment would discard
    green test runs earlier in the step; MarkerTables.derive applies the
    anti-stale rules on top.
    """
    markers = list(_EXECUTION_SEGMENT_MARKER.finditer(output))
    if not markers:
        return []
    out = []
    for cur, nxt in zip(markers, [*markers[1:], None], strict=False):
        end = nxt.start() if nxt is not None else len(output)
        out.append(output[cur.end() : end])
    return out


def extract_error(output: str) -> str | None:
    """Return the error region of the output, or None if clean.

    Output with the coder's execution framing is scanned only from the last
    ``Execution result of code block N:`` marker on, so a step that failed
    internally and then self-repaired reads clean. Output without framing is
    scanned in full. The region starts at the first error match and is capped
    at ``_ERROR_REGION_CAP`` characters to keep StepRecords bounded.
    """
    region = last_execution_segment(output)
    match = _ERROR_PATTERNS.search(region)
    if match is None:
        return None
    return region[match.start() : match.start() + _ERROR_REGION_CAP]


_TRACEBACK_FRAME = re.compile(r'File "([^"]+)", line \d+, in (\w+)')
_EXCEPTION_CLASS = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b", re.MULTILINE
)
_EXIT_CODE = re.compile(r"\bexit ?code:?\s+([1-9]\d*)\b")


def error_fingerprint(output: str, tool_name: str = "") -> str:
    """Return an annotation-only error identity, logged per failing step.

    It does not gate escalation: the failure streak in evidence_state is
    target-blind. Priority: exception class ("Class:module:function" with the
    deepest traceback frame, else the bare class), else "tool:<name>" for tool
    failures, else "exit:<code>", else "". Scans the same last segment as extract_error.
    """
    region = last_execution_segment(output)
    cls_matches = _EXCEPTION_CLASS.findall(region)
    if cls_matches:
        frames = _TRACEBACK_FRAME.findall(region)
        if frames:
            path, func = frames[-1]  # deepest frame = the raise site
            module = path.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".py")
            return f"{cls_matches[-1]}:{module}:{func}"
        return cls_matches[-1]
    if tool_name:
        return f"tool:{tool_name}"
    exit_match = _EXIT_CODE.search(region)
    if exit_match:
        return f"exit:{exit_match.group(1)}"
    return ""


# The coder runs at most max_debug_rounds (default 3) rounds per turn, so at
# most 2 can fail before the final one. Round boundaries are not marked in the
# joined output and one round can emit many blocks, so an uncapped count of
# failing blocks would push a single failing step straight to CONCEPTUAL.
_MAX_INTERNAL_REPAIRS = 2


def count_internal_repairs(output: str) -> int:
    """Count failed execution segments before the final one, capped.

    These are the failures the coder retried internally. The final failure is
    not a repair, and output without execution framing counts 0. A failing step
    adds 1 + this count to the streak in evidence_state.
    """
    markers = list(_EXECUTION_SEGMENT_MARKER.finditer(output))
    if len(markers) < 2:
        return 0
    failing = sum(
        1
        for cur, nxt in zip(markers[:-1], markers[1:], strict=True)
        if _ERROR_PATTERNS.search(output[cur.end() : nxt.start()])
    )
    return min(failing, _MAX_INTERNAL_REPAIRS)
