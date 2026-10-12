"""Per-benchmark evidence marker tables: one definition, three consumers.

``MarkerTables.derive()`` sets ``is_state_changing`` and ``verification`` (none,
pass or fail) from step text alone (code blocks, console) for evidence_state,
commitment_state and the MAST annotators. The tables live in phases.yaml and are
hashed into state_signature, so editing them makes promoted policies fail
verification; validate them on hand labels. A ``FINDING:`` line changes state only
when it differs from the previous one, so a repeated finding does not invalidate
verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import execution_segments, extract_error

_FINDING_LINE = re.compile(r"^FINDING:\s*(.+?)\s*$", re.MULTILINE)

# The code executor appends a bare "Timeout" line when it kills a run. A
# truncated check has no outcome, so it reads as no verification, not a failed
# one; the step still carries its exit error. Only the last run segment is
# checked, so a Timeout line elsewhere cannot erase a pass or mask a failure.
_EXECUTION_TIMEOUT = re.compile(r"^Timeout\s*$", re.MULTILINE)


@dataclass(frozen=True)
class MarkerTables:
    """Frozen per-benchmark regex patterns, matched in multiline mode.

    ``verification_evidence`` matches the test runner's own result lines, not
    the command the agent typed. Invocations vary across runners and phrasing,
    and a stated intent without a run ("I will now run pytest") would
    otherwise read as a failed verification.
    """

    state_changing: tuple[str, ...] = ()
    verification_evidence: tuple[str, ...] = ()
    verification_pass: tuple[str, ...] = ()

    def derive(self, output: str, prev_finding: str = "") -> tuple[bool, str, str]:
        """Return (is_state_changing, verification, finding) for one step.

        A pass requires that a check ran, reported success, and that the step
        ends without an error. A run without a success token reads "fail"; a
        timed-out run reads "none" because it produced no outcome.
        """
        findings = _FINDING_LINE.findall(output)
        finding = findings[-1] if findings else prev_finding
        finding_changed = bool(finding) and finding != prev_finding

        state_changing = finding_changed or any(
            re.search(p, output, re.MULTILINE) for p in self.state_changing
        )

        # Judge the last segment that ran a check: whole-output matching would
        # let a stale early pass survive a later failing round, and last-segment
        # scoping would miss a green run followed by a harmless print. Output
        # without execution framing is treated as one segment.
        segments = execution_segments(output) or [output]

        def _has(patterns: tuple[str, ...], text: str) -> bool:
            return any(re.search(p, text, re.MULTILINE) for p in patterns)

        run_indices = [i for i, s in enumerate(segments) if _has(self.verification_evidence, s)]
        if not run_indices:
            verification = "none"
        else:
            last_run = segments[run_indices[-1]]
            # From the last run onward: segment text carries the source of the
            # next executed block, so a write after the check is visible here
            # even when its console only reports the write.
            after_run = "".join(segments[run_indices[-1] :])
            if _EXECUTION_TIMEOUT.search(last_run):
                verification = "none"  # the check never completed, no outcome
            else:
                green = (
                    _has(self.verification_pass, last_run)
                    and extract_error(last_run) is None
                    and extract_error(output) is None
                )
                if green and _has(self.state_changing, after_run):
                    # The check passed but the artifact changed afterwards:
                    # the end state is unverified, yet nothing failed.
                    verification = "none"
                else:
                    verification = "pass" if green else "fail"

        return state_changing, verification, finding
