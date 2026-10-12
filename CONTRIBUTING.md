# Contributing

## Setup

Python 3.11 to 3.13.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Keep the environment at `.venv`. pyright is configured to read it from there.

## Checks

CI runs these on pushes to main and on every pull request, plus a gitleaks
secret scan and a package build.
Run them locally before opening a PR:

```bash
.venv/bin/ruff check src
.venv/bin/pyright
```

## Rules

- Prompt text, log messages and regexes are behaviour too. Changing one is a
  behaviour change, not a cleanup.
- The AutoGen packages are pinned exactly because
  `src/aria/frameworks/autogen/team.py` mirrors private orchestrator methods
  from autogen-agentchat 0.7.5. A version bump means diffing those methods
  against the new release first.
- Never commit credentials. Keys go in `aria_secrets.yaml`, which is
  gitignored; `aria_secrets.example.yaml` shows the format.

## Style

- No formatter runs on this codebase, so keep diffs to the lines you mean to
  change and do not reformat surrounding code.
- ruff settings live in `pyproject.toml`. Fix new findings rather than adding
  ignores; a justified per-line `# noqa` is fine.
- Docs and comments in English.

## Comments and docstrings

- Say why, not what, in the present tense. Skip comments that restate the
  code.
- Keep them short: a line comment at most 2 lines, a comment block at most
  4 lines. A docstring is one summary line plus at most one short paragraph
  or bullet list.
- No project history, dates, names, ticket ids or numbers from past runs.
  Describe the code as it is, not how it got there.
- ASCII only: no em dashes, arrows or other unicode symbols.
- No emphasis in capitals; acronyms and identifiers are fine.

ruff flags commented-out code and TODO-style markers (rules `ERA` and `FIX`).
