# Experiment setup

How to set up an ARIA experiment on your own benchmark, with setup patterns
for three common kinds of benchmarks. The adapter mechanics are in
[adapters.md](adapters.md), every config field is in
[configuration.md](configuration.md).

How well ARIA performs always depends on the workflow and the
executor/advisor model pair, so train and evaluate the policy on the pair and
team you actually plan to use.

## Concepts

- **Executor and advisor.** The executor team runs every step on the cheaper
  model. The advisor is only called when a consult happens and never acts on
  the task itself. Both are set under `models`.
- **Contact points and state.** The orchestrator hook raises a decision at
  fixed structural moments (before each step, before delivery, before a
  stall replan), and a deterministic detector describes each one. The state
  is a small discrete layout set in `state.dims`: `exec3` reads the outcome
  of the last code block the `coder` executed and suits coding teams;
  `outcome3` also counts tool results (in the stock AutoGen runner, tool
  returns that contain the `=== TOOL RESULT ===` or `=== TOOL ERROR ===`
  banner), so it suits tool-calling teams that produce no process output
  (where process output exists the two are identical).
- **Price of asking.** The reward is the task score minus `lambda_cost` times
  the consult cost, normalised by two measured cost anchors. A higher
  `lambda_cost` makes the policy consult less.
- **Budget.** `consult_budget` caps consults per episode.
- **Frozen evaluation.** The policy learns on the train split, is then frozen
  and runs on the held-out test split.

## Setting it up

1. **Adapter and team.** Write and register the adapter, and write the team
   spec, `phases.yaml` and the prompts, as described in
   [adapters.md](adapters.md).
2. **Splits.** Fix disjoint train and test splits and commit them with
   `aria.benchmarks.split_manifest`.
3. **Reference runs.** Run `autonomous` with the executor model
   (executor-only) and with the advisor model as executor (advisor-only).
   The gap between the two is the headroom available to a consult policy.
   Tasks the executor fails and the advisor solves are the most informative
   for training.
4. **Price and anchors.** Measure `cost_anchor_low_usd` from the
   executor-only run, and `cost_anchor_high_usd` plus the mean price per
   consult (`flat_cost_usd`) from an `always_consult` run with
   `lambda_cost: 0` under the same `consult_budget` as training. The defaults
   in the code belong to one reference setup and do not transfer.
5. **Training.** Run condition `aria` in train mode over several seeds.
   Consult seeding (the `seeding` block, with `cells` set explicitly) helps
   when some CONSULT cells are rarely visited.
6. **Evaluation.** Promote the trained policy with `aria-policy promote` and
   run it in eval mode on the test split with the same `consult_budget` as in
   training. Keep `seeding` off in evaluation configs.

Run `aria-preflight <config>` before the first paid run.

## Patterns for common benchmark kinds

The adapters for these kinds of benchmarks are not part of this repository;
these notes describe what an adapter for each kind has to provide.

Two of the patterns run the agent's code inside a container. The stock
`coder` always executes on the host through `python_exec`, so these need a
custom framework runner (`aria.frameworks.registry.register_framework`) whose
coder executes in the container. A tool on an `assistant` agent is no
substitute here, since `exec3` only reads the `coder`'s blocks.

### Repository tasks (SWE-bench style)

- Workspace: a private git checkout at the task's base commit.
- Delivery: `workspace_patch` returns the `git diff`; the official harness
  scores it in its own container.
- Execution: the agent's tests only work where the repository's dependencies
  are installed, so the coder's blocks have to run in the task's environment
  image. The adapter declares it with `execution_image` and
  `execution_setup`; the runner has to honour them (see above).
- Team: `coder` and `file_surfer`. State: `exec3`.
- Advisor: `repo_access` and `gate_diff_injection` on, so the advisor can read
  the code and sees the real diff before delivery.

### Terminal tasks in a container (Terminal-Bench style)

- One container per episode; every code block runs in it, so files and
  installs persist across steps (custom runner, see above).
- Delivery: `workspace_patch` runs the benchmark's verifier in the live
  container (it is called before teardown) and returns the verdict, which
  `score` turns into the result.
- With `repo_access` on, `advisor_inspector` lets the advisor inspect the
  container instead of the host workspace, and with `gate_diff_injection`
  its `gate_snapshot` replaces the git diff at the gate. `task_timeout_s`
  passes the task's own time limit (capped at `episode_timeout_s`).
- Team: `coder` only, with a step budget large enough for long shell
  sessions. State: `exec3`.

### Tool-calling conversations (tau2-bench style)

- `prepare_workspace` starts a private environment process per episode with
  the benchmark's user simulator; the team acts only through tools that talk
  to it, registered as custom tools (see [adapters.md](adapters.md)).
- Delivery: end the simulation and save the trajectory in
  `workspace_patch`, before `release_workspace` stops the environment, and
  let `score` run the official evaluator on it.
- Team: a single `assistant` agent with those tools. State: `outcome3`. The
  stock AutoGen runner sees a tool call only through its return text, so each
  tool includes `=== TOOL RESULT ===` in its return (`=== TOOL ERROR ===` for
  an environment-reported failure).
- Advisor: `repo_access` and `gate_diff_injection` off, since the workspace
  holds no task state.
