# Adding a benchmark

ARIA ships no benchmarks. A benchmark is plugged in as an adapter: a
`BenchmarkAdapter` subclass, a team directory under the teams root and a few
lines of Python that register the adapter before a run starts. This page
covers the interface, the episode lifecycle that calls it, the config files
next to it and a complete toy example that runs.

## What an adapter owns

Everything benchmark-specific, and nothing framework-specific:

- task loading and the fixed train/test split
- the contact-point detector (usually built from `phases.yaml`)
- scoring against ground truth
- prompt variables and an optional advisor context
- the per-episode workspace and the artifact that gets scored
- optional hints for container execution, advisor inspection and per-task
  time limits

`aria.benchmarks.base` imports no agent framework. The executor team itself
is built by the framework runner from the team spec; the adapter only
parameterises it.

## The interface

```python
from aria.benchmarks import BenchmarkAdapter, Task, TaskScore
```

`Task(id, prompt, domain="", gold={}, metadata={})` is one task. `prompt` is
the full task statement the team receives. `gold` holds the ground truth and
never reaches the executor or the advisor. `metadata` is free-form for the
adapter's own use.

`TaskScore(score, delivered, detail={})` is one verdict. `score` is in
[0, 1].

Set the class attribute `name` to the name you register the adapter under.
It is written into every episode row.

### Required

| Hook | Semantics |
|---|---|
| `load_tasks(split, limit=None) -> list[Task]` | Tasks for `"train"` or `"test"`. Splits must be disjoint and fixed: commit them as a manifest (`aria.benchmarks.split_manifest`) rather than sampling at runtime. The experiment calls it once per run without `limit` and applies shuffle, offset and limit per seed itself, so every seed gets its own sample. |
| `build_detector() -> ContactPointDetector` | A fresh deterministic detector, called once per episode. The normal implementation is `build_detector(load_phases_config(name, teams_root))` from `aria.detectors`. |
| `async score(task, final_answer) -> TaskScore` | The verdict for the scored artifact (see delivery below). `final_answer=None` means nothing was delivered: return `delivered=False`. Raise when your harness produced no verdict at all; the score call is then retried once after 90 s with the same artifact, and if it fails again the episode is kept as unscored, which keeps it out of the learner and out of the score mean instead of counting as a 0. |

### Optional (all default to no-ops)

| Hook | Default | Semantics |
|---|---|---|
| `prompt_vars(task) -> dict[str, str]` | `{}` | Values for `{placeholders}` in the agent prompts, merged over the runner's `{date_today}`. Rendering is strict: a placeholder without a value raises. Nothing here may exceed what the task prompt already gives the team. |
| `async prepare_workspace(task) -> Path \| None` | `None` | Creates the episode's working directory (data files, a repository checkout). The built-in code executor and file tools are bound to it; with `None` they fall back to the current working directory. It must be private to the episode, otherwise one episode inherits another's edits and the comparison across conditions breaks. The coder runs model-written code on the host in this directory, so make it disposable and run untrusted work in a VM or container (see the README's Security section). |
| `async workspace_patch(workspace) -> str \| None` | `None` | The agent's actual changes as a scoreable string, for example `git diff` against the setup commit. Called before teardown, also after a crash. `None` means no workspace notion; the typed final answer is scored instead. |
| `async release_workspace(workspace)` | nothing | Tears down what `prepare_workspace` built. Runs in a `finally`, crash path included. |
| `execution_image(task) -> str \| None` | `None` | A container image the executor's code should run in. Passed to tool factories as `ToolContext.extras["exec_image"]`. The built-in `python_exec` runs on the host and ignores it, see [Custom tools](#custom-tools). |
| `execution_setup(task) -> str \| None` | `None` | One-time shell setup inside that image. Passed as `ToolContext.extras["exec_setup"]`. |
| `advisor_inspector(workspace, task) -> object \| None` | `None` | A backend for the advisor's read-only inspection when the graded state does not live in the host workspace (for example inside a task container). See below. |
| `advisor_domain_context(task) -> str` | `""` | Extra text for the advisor's system prompt, followed by `advisor.extra_context` from the config. Never the gold answer, and nothing the executor does not also have. |
| `task_timeout_s(task) -> float \| None` | `None` | A per-task wall-clock budget the benchmark itself declares for the agent phase. It is capped at the team's `episode_timeout_s`; `None` uses that flat value. |

### The advisor inspector

With `advisor.repo_access: true` the advisor may run up to
`advisor.max_tool_calls` whitelisted inspection commands (`cat`, `grep`,
`ls`, `head`, `tail`, `sed` without `-i`, `find`, `rg`, `wc`, `git diff`,
`status`, `log`, `show`) before it answers. The whitelist checks the command
name, not every flag (`find -exec`, for example, is not blocked), so it
guards against accidents, not against a hostile model. With
`advisor.gate_diff_injection: true` every pre-delivery consult also gets the
real `git diff` appended. By default both run on the host in the episode
workspace, which then has to be a git checkout, and on the host the diff is
only appended when `repo_access` is on as well.

If the graded state lives elsewhere, return an object from
`advisor_inspector` with:

- `exec(argv: list[str], cap: int) -> str`: runs an already validated command
  wherever the state lives. The whitelist is still enforced by the advisor
  before `exec` is called. Absolute paths are allowed with an inspector,
  since there is no host workspace to escape from.
- `gate_snapshot(cap: int) -> tuple[str, str]`: heading and body that replace
  the `git diff` block at the pre-delivery gate.
- optionally `prompt_hint() -> str`, appended to the advisor's system prompt
  when inspection is on.

The runner sets the inspector for every episode, so it is used by
`gate_diff_injection` even when `repo_access` is off.

### Writing a detector

Most benchmarks use the stock `HeuristicDetector` through
`aria.detectors.build_detector(load_phases_config(...))`, configured entirely
by `phases.yaml` (schema in [configuration.md](configuration.md#phasesyaml)).

Read `phases.yaml` from the same teams root as the experiment config. The
experiment hashes `<teams_root>/<benchmark>/phases.yaml` into the
`state_signature` of the run and uses its evidence tables; if
`build_detector()` read a different file, the manifest would describe a
detector that did not run. The usual way is a `teams_root` constructor
argument set through `benchmark_options`, as in the example below.

A benchmark with an explicit programmatic workflow can subclass
`ContactPointDetector` instead. The contract (from `aria/detectors/base.py`):
an empty history returns a FRAME/START contact point, records with
`is_plan=True` classify as FRAME, and when `upcoming_instruction` is given
the phase describes that next step, not the completed one.

## The episode lifecycle

What calls which hook, in order. Useful when a hook has side effects.

Once per run (`aria.experiments.experiment.run_experiment`):

1. `build_benchmark(cfg.benchmark, **cfg.benchmark_options)` builds the
   adapter.
2. The team spec and `phases.yaml` are loaded from `cfg.teams_root`.
3. `load_tasks(split)` runs once. Per seed the task list is shuffled with
   the seed (`run.shuffle_tasks`), then cut by `run.task_offset` and
   `run.task_limit`, or replaced by `run.task_ids`. Episode `i` runs task
   `i % len(tasks)`, so `run.episodes` larger than the task list means
   repeated passes.

Per episode:

1. `task_timeout_s(task)` sets the wall-clock budget of the agent phase.
2. `advisor_domain_context(task)` builds the advisor's system prompt.
3. `build_detector()` creates the detector for the controller.
4. The AutoGen runner calls `prepare_workspace(task)`, then
   `advisor_inspector(workspace, task)`, `prompt_vars(task)`,
   `execution_image(task)` and `execution_setup(task)`, builds the team and
   runs it on `task.prompt`.
5. In its `finally`: `workspace_patch(workspace)`, then the `stop()` of every
   closeable a tool factory registered, then `release_workspace(workspace)`.
6. `score(task, artifact)`, outside the time budget. The artifact is the
   workspace patch when `delivery: workspace_patch` and the patch is not
   `None`, otherwise the typed final answer.

The terminal reward is `score` when `TaskScore.delivered` is true and 0
otherwise. Crashed episodes (an exception, or the time budget) are logged
with `scored: false` and never reach the learner. The reported `delivered`
metric is stricter: it also needs the team to have stopped on its own rather
than at `max_steps`.

## Registering the adapter

The registry starts empty, and `aria-train`, `aria-eval` and `aria-run`
import no user code. Before any of them can build your benchmark, a few lines
of Python have to call `register_benchmark`:

```python
from aria.benchmarks import register_benchmark

register_benchmark("toy", lambda **options: ToyBenchmark(**options))
```

The factory receives `benchmark_options` from the config as keyword
arguments. Names are unique; registering the same name twice raises.

Two ways to start a run after that:

- Hand over to the stock CLIs from a small driver script, so all their flags
  keep working (shown in the example below).
- Call the Python API directly:

  ```python
  import asyncio

  import toy_bench
  from aria.config import load_config
  from aria.experiments.experiment import run_experiment

  toy_bench.register()
  cfg = load_config("configs/experiments/toy_eval.yaml")
  asyncio.run(run_experiment(cfg, mode="eval"))
  ```

  For a matrix, `aria.experiments.matrix.load_matrix` and `expand` return
  one `RunSpec` per cell; run each with
  `run_experiment(spec.config, mode=spec.mode)`.

`aria-run --dry-run` is the exception: expansion never builds a benchmark,
so it works without registration.

## Helpers for adapters

`aria.benchmarks.split_manifest` writes and reads the committed train/test
split. `write(path, dataset, rule, train, test, **extra)` refuses an
overlapping split; `read(path, generator)` refuses a missing file (naming
`generator` as the thing to run), overlapping ids and counts that disagree
with the id lists; `select(manifest, split)` returns the id set for `train`
or `test`.

## Team directory

Each benchmark has a directory under the teams root (default
`configs/teams`, relative to the working directory):

```
configs/teams/
|-- _defaults/
|   |-- agents.yaml          # roster entries shared by all benchmarks
|   `-- prompts/<agent>.md   # default system prompts
`-- <benchmark>/
    |-- team.yaml            # roster, per-benchmark overrides, runtime knobs
    |-- phases.yaml          # phase markers and evidence tables
    `-- prompts/<agent>.md   # optional per-benchmark prompt overrides
```

The directory name must equal the benchmark name. A per-benchmark prompt
overrides the default one. Field reference:
[configuration.md](configuration.md#team-spec).

## Custom tools

Agents name tools by symbol. The AutoGen package registers three:
`python_exec` (the coder's host executor, bound to the workspace),
`file_browser` (read-only file tools) and `consult_advisor` (attached
by the framework itself). Add your own with
`register_tool` before the run:

```python
from autogen_core.tools import FunctionTool

from aria.frameworks.tools import ToolContext, register_tool


def calculator_tool(context: ToolContext) -> FunctionTool:
    async def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return str(eval(expression, {"__builtins__": {}}))  # toy only

    return FunctionTool(calculate, name="calculate", description="Evaluate an arithmetic expression.")


register_tool("calculator", "autogen", calculator_tool)
```

Then give the symbol to an `assistant` agent in `team.yaml`
(`tools: [calculator]`). With the stock AutoGen runner a tool result enters
the state only through `outcome3`, and only if the return contains
`=== TOOL RESULT ===` or `=== TOOL ERROR ===`. The factory receives a
`ToolContext` with the workspace, the adapter's `exec_image`/`exec_setup` in
`extras`, the `exec_timeout_s` from the team runtime and a `closeables` list:
append any object with an async `stop()` that must be shut down when the
episode ends (a container, a server).

Two limits worth knowing. A symbol can be registered only once, so
`python_exec` cannot be replaced, and the `coder` kind always uses it. Code
that has to run inside a container therefore needs either an `assistant`
agent with its own execution tool, or a custom framework runner
(`aria.frameworks.registry.register_framework`).

## Example: a toy benchmark

A complete project that runs. The task is to compute an arithmetic
expression and leave the result in `answer.txt`; the workspace file is the
scored artifact. Layout, relative to the working directory you run from:

```
toy_bench.py
make_split.py
aria_toy.py
aria_secrets.yaml
configs/
|-- base/default.yaml
|-- base/conditions.yaml
|-- experiments/toy.yaml
|-- experiments/toy_eval.yaml
|-- families.yaml
|-- splits/toy.json              # written by make_split.py
`-- teams/
    |-- _defaults/agents.yaml
    |-- _defaults/prompts/coder.md
    |-- toy/team.yaml
    `-- toy/phases.yaml
```

`toy_bench.py`:

```python
"""Toy benchmark: evaluate an arithmetic expression, leave the result in answer.txt."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from aria.benchmarks import BenchmarkAdapter, Task, TaskScore, register_benchmark
from aria.benchmarks import split_manifest
from aria.detectors import ContactPointDetector, build_detector, load_phases_config

# id -> (expression, expected answer). A real adapter loads its dataset here.
TASKS = {
    "add-1": ("2 + 3", "5"),
    "mul-1": ("6 * 7", "42"),
    "add-2": ("40 + 2", "42"),
    "sub-1": ("10 - 4", "6"),
}


class ToyBenchmark(BenchmarkAdapter):
    name = "toy"

    def __init__(
        self,
        split_file: str = "configs/splits/toy.json",
        teams_root: str = "configs/teams",
    ) -> None:
        self.split_file = split_file
        self.teams_root = teams_root

    def load_tasks(self, split: str, limit: int | None = None) -> list[Task]:
        manifest = split_manifest.read(self.split_file, generator="make_split.py")
        ids = sorted(split_manifest.select(manifest, split))
        tasks = [
            Task(
                id=task_id,
                prompt=f"Compute {TASKS[task_id][0]}. Write only the result "
                "to answer.txt in the working directory.",
                gold={"answer": TASKS[task_id][1]},
            )
            for task_id in ids
        ]
        return tasks if limit is None else tasks[:limit]

    def build_detector(self) -> ContactPointDetector:
        # Same teams_root as the experiment config, so the detector and the
        # state_signature in the manifest describe the same phases.yaml.
        return build_detector(load_phases_config(self.name, self.teams_root))

    async def prepare_workspace(self, task: Task) -> Path | None:
        return Path(tempfile.mkdtemp(prefix=f"toy_{task.id}_"))

    async def workspace_patch(self, workspace: Path | None) -> str | None:
        if workspace is None:
            return None
        answer = workspace / "answer.txt"
        return answer.read_text() if answer.exists() else ""

    async def release_workspace(self, workspace: Path | None) -> None:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)

    async def score(self, task: Task, final_answer: str | None) -> TaskScore:
        if final_answer is None:
            return TaskScore(score=0.0, delivered=False)
        solved = final_answer.strip() == task.gold["answer"]
        return TaskScore(score=float(solved), delivered=True)


def register() -> None:
    register_benchmark("toy", lambda **options: ToyBenchmark(**options))
```

`make_split.py`, run once; commit the manifest it writes:

```python
from aria.benchmarks import split_manifest

split_manifest.write(
    "configs/splits/toy.json",
    dataset="toy arithmetic",
    rule="fixed by hand",
    train=["add-1", "mul-1"],
    test=["add-2", "sub-1"],
)
```

`aria_toy.py`, the driver that registers the adapter and hands over to the
CLIs:

```python
"""The aria CLIs with the toy benchmark registered.

    python aria_toy.py run <matrix.yaml> [--dry-run] [--filter key=value]
    python aria_toy.py train <experiment.yaml>
    python aria_toy.py eval <experiment.yaml> [--policy ID | --checkpoint PATH]
"""

import logging
import sys

import toy_bench
from aria.cli import evaluate, run, train

toy_bench.register()
logging.getLogger("autogen_core").setLevel(logging.WARNING)  # one line per message otherwise

apps = {"run": run.app, "train": train.app, "eval": evaluate.app}
apps[sys.argv.pop(1)]()
```

`configs/teams/_defaults/agents.yaml`:

```yaml
agents:
  coder:
    kind: coder
    description: >-
      Writes py or sh code blocks, runs them in the task workspace and
      debugs them on errors.
    tools: [python_exec]
    params:
      max_debug_rounds: 3
```

`configs/teams/_defaults/prompts/coder.md`:

```text
You are a coding assistant. Write py or sh code blocks; they are executed for
you in the task workspace and the console output comes back in the next message.
The date today is: {date_today}
```

`configs/teams/toy/team.yaml`:

```yaml
benchmark: toy
agents: [coder]
runtime:
  common:
    max_steps: 10
    exec_timeout_s: 60
    episode_timeout_s: 600
  autogen:
    max_stalls: 3
```

`configs/teams/toy/phases.yaml`:

```yaml
classifier: heuristic
conceptual_streak: 2
markers:
  frame: [plan, understand]
  explore: [ls, cat, inspect]
  reason: [compute, write, fix]
  deliver: [final, done]
state_changing:
  patterns:
    - '>\s*\S+'
verification:
  evidence_of_run:
    - '^=+ test session starts'
  pass_requires:
    - '\b[1-9]\d* passed\b'
```

`configs/base/default.yaml`, the base layer of the matrix. Point the two
models at any OpenAI-compatible endpoint; `model_info` is needed for model
ids AutoGen does not know. `aria-preflight` checks both roles through
AutoGen, so the advisor gets one too. The reward anchors and the consult
price here are placeholders: measure them for your own model pair before
training (see [configuration.md](configuration.md#reward)).

```yaml
framework: autogen
teams_root: configs/teams
consult_budget: 4
models:
  executor:
    model: small-model
    base_url: http://127.0.0.1:8765/v1
    reasoning_effort: low
    model_info: { vision: false, function_calling: true, json_output: true, family: unknown, structured_output: false }
  advisor:
    model: large-model
    base_url: http://127.0.0.1:8765/v1
    reasoning_effort: high
    model_info: { vision: false, function_calling: true, json_output: true, family: unknown, structured_output: false }
pricing:
  small-model: { prompt: 0.50, completion: 3.00, cached_prompt: 0.05 }
  large-model: { prompt: 2.00, completion: 12.00, cached_prompt: 0.20 }
reward:
  lambda_cost: 0.10
  flat_cost_usd: 0.05
  cost_anchor_low_usd: 0.0
  cost_anchor_high_usd: 0.20
run:
  episodes: 4
  checkpoint_every: 2
```

`configs/base/conditions.yaml`:

```yaml
conditions:
  aria: {}
  autonomous: {}
  always_consult: {}
```

`configs/experiments/toy.yaml`, the training matrix:

```yaml
name: toy
base: configs/base/default.yaml
condition_defaults: configs/base/conditions.yaml
benchmarks: [toy]
conditions: [autonomous, always_consult, aria]
frameworks: [autogen]
seeds: [0]
mode: train
overrides:
  benchmark:toy:
    benchmark_options: { split_file: configs/splits/toy.json, teams_root: configs/teams }
```

`configs/experiments/toy_eval.yaml`, a single-run config for `aria-eval`.
Single-run configs are complete on their own; there is no base layer.

```yaml
name: toy_eval
benchmark: toy
benchmark_options: { split_file: configs/splits/toy.json, teams_root: configs/teams }
condition: aria
teams_root: configs/teams
consult_budget: 4
models:
  executor:
    model: small-model
    base_url: http://127.0.0.1:8765/v1
    reasoning_effort: low
    model_info: { vision: false, function_calling: true, json_output: true, family: unknown, structured_output: false }
  advisor:
    model: large-model
    base_url: http://127.0.0.1:8765/v1
    reasoning_effort: high
    model_info: { vision: false, function_calling: true, json_output: true, family: unknown, structured_output: false }
pricing:
  small-model: { prompt: 0.50, completion: 3.00, cached_prompt: 0.05 }
  large-model: { prompt: 2.00, completion: 12.00, cached_prompt: 0.20 }
reward: { lambda_cost: 0.10, flat_cost_usd: 0.05, cost_anchor_high_usd: 0.20 }
run: { episodes: 2, seeds: [0] }
```

`configs/families.yaml`, read by `aria-policy` and `aria-eval --policy`:

```yaml
families:
  toy: arithmetic
```

`aria_secrets.yaml` holds the key (`api_key: "..."`), see the README.

Run it:

```bash
python make_split.py

# expansion only, nothing is called
python aria_toy.py run configs/experiments/toy.yaml --dry-run

# train the three cells; finished cells are skipped on a rerun
python aria_toy.py run configs/experiments/toy.yaml

# turn the aria training run into a policy artifact, then evaluate it
# frozen on the test split
aria-policy promote results/toy/aria/autogen/train/small-model/seed_0
aria-policy list
python aria_toy.py eval configs/experiments/toy_eval.yaml --policy <policy_id>
```

`aria-eval --policy` checks the artifact's `state_signature` against the
current `phases.yaml` and state layout before loading, so editing the phase
markers after training makes the evaluation refuse to start.

The example needs only an endpoint that speaks the OpenAI chat-completions
format; a scripted local stub is enough to exercise the whole loop. With real
models the loop is the same; only `base_url`, the model ids, `model_info`, the
prices and the key change.

## Before the first real training run

- Commit the split manifest. Never sample splits at runtime.
- Freeze `phases.yaml`. It is part of the `state_signature`, so a promoted
  policy trained under one version refuses to load under another. A raw
  checkpoint is checked only against the state layout.
- Measure the reward anchors and the consult price for your model pair:
  an `autonomous` run gives the low anchor, an `always_consult` run with
  `reward.lambda_cost: 0` gives the high anchor and the mean consult price.
  The defaults in the code belong to one reference setup and do not
  transfer.
- Run `aria-preflight <config>` to check keys, endpoints and prices with one
  short call per role.
