# Configuration

All configuration is YAML. Nothing ships with the package; paths below are
the conventions the code uses, relative to the working directory.

| File | Read by | Default path |
|---|---|---|
| Experiment config (one run) | `aria-train`, `aria-eval`, `aria.config.load_config` | given on the command line |
| Matrix (many runs) | `aria-run`, `aria.experiments.matrix.load_matrix` | given on the command line, conventionally `configs/experiments/` |
| Base layer and condition defaults | the matrix, via `base:` and `condition_defaults:` | conventionally `configs/base/default.yaml`, `configs/base/conditions.yaml` |
| Team spec and phase schema | every run | `<teams_root>/<benchmark>/`, `teams_root` defaults to `configs/teams` |
| Task families | `aria-policy` and `aria-eval --policy` | `configs/families.yaml` |
| Model config for preflight | `aria-preflight` | `configs/base/default.yaml` |
| Secrets | `aria-train`, `aria-eval`, `aria-run`, `aria-preflight` | `aria_secrets.yaml`, or `--secrets PATH` |

Unknown keys raise in the blocks where a typo would silently change a
result: `advisor`, `state` and `run` when the config is loaded, `rl`,
`reward` and `seeding` when the run starts (before the first model call),
and top-level keys of `phases.yaml`. Top-level keys of the experiment config
and of a matrix, and fields inside `models` and `pricing` entries, are not
checked, so a misspelt one there is ignored. `aria-run --dry-run` and the
manifest show what was actually set.

## Experiment config

One YAML file describes one run of one condition on one benchmark, over all
its seeds. It maps onto `aria.config.ExperimentConfig`.

| Field | Default | Meaning |
|---|---|---|
| `name` | `experiment` | Label, recorded in the manifest. |
| `benchmark` | required | Registered benchmark name. |
| `benchmark_options` | `{}` | Keyword arguments for the adapter's factory. |
| `condition` | `aria` | Registered policy that makes the CONSULT/SILENT decisions (`aria.policies.registry.register_policy` adds more). Built in: `aria` (learned table); `autonomous` and `always_consult` (also the calibration runs for the reward anchors); `organic` (the executor calls the advisor as a tool); `plan_advisory`, `early_once`, `gate_once`, `early_gate_2x` (fixed schedules); `anthropic_heuristic` (a fixed rule on phase and error evidence); `random_advisory` (needs `condition_options.consult_probability`). |
| `condition_options` | `{}` | Options for the condition's policy, e.g. `checkpoint`, `frozen`. |
| `framework` | `autogen` | Framework runner. |
| `teams_root` | `configs/teams` | Where the team spec and `phases.yaml` live. |
| `delivery` | `workspace_patch` | Scored artifact: the adapter's `workspace_patch`, or `typed` for the model's final answer. Without a patch the typed answer is scored. |
| `consult_budget` | `null` | Per-episode cap on consults. `null` is uncapped. |
| `models` | required | Exactly two roles, `executor` and `advisor`. |
| `pricing` | `{}` | USD per 1M tokens, keyed by model id. |
| `advisor` | see below | Advisor prompt and inspection settings. |
| `state` | `dims: [position4, exec3, consult]` | State layout. |
| `rl` | `{}` | Learner hyperparameters. |
| `reward` | `{}` | Reward components. |
| `seeding` | `{}` | Deficit-targeted consult seeding during training. Off when empty. |
| `run` | see below | Episodes, seeds, split, task selection, output root. |

A complete example:

```yaml
name: example_train
benchmark: toy
benchmark_options: { teams_root: configs/teams }
condition: aria
teams_root: configs/teams
consult_budget: 4
models:
  executor:
    model: small-model
    base_url: https://llm.example.com/v1
    reasoning_effort: low
    timeout: 240
    model_info: { vision: false, function_calling: true, json_output: true, family: unknown, structured_output: false }
  advisor:
    model: large-model
    base_url: https://llm.example.com/v1
    reasoning_effort: high
    timeout: 240
    model_info: { vision: false, function_calling: true, json_output: true, family: unknown, structured_output: false }
pricing:
  small-model: { prompt: 0.50, completion: 3.00, cached_prompt: 0.05 }
  large-model: { prompt: 2.00, completion: 12.00, cached_prompt: 0.20 }
advisor:
  style: strategic
  max_guidance_words: 250
state:
  dims: [position4, exec3, consult]
rl:
  epsilon_start: 0.4
  epsilon_end: 0.10
  epsilon_anneal_episodes: 110
reward:
  lambda_cost: 0.10
  flat_cost_usd: 0.05
  cost_anchor_low_usd: 0.0
  cost_anchor_high_usd: 0.20
run:
  episodes: 150
  seeds: [1, 2, 3]
  checkpoint_every: 10
```

### models

Both roles go through an OpenAI-compatible endpoint. Fields of
`ModelConfig`:

| Field | Default | Meaning |
|---|---|---|
| `model` | required | Model id as the endpoint knows it. |
| `api_key` | `null` | Normally merged in from `aria_secrets.yaml`, never written to manifests. |
| `base_url` | `null` | Endpoint; `null` uses the OpenAI default. |
| `temperature` | `null` | Not sent when `null`; the provider default applies. |
| `max_tokens` | `null` | Per-call output cap. For the advisor, `advisor.max_advice_tokens` is used instead. |
| `max_retries` | `3` | SDK retries for rate limits and transient errors. They multiply the worst case of `timeout`. |
| `timeout` | `120.0` | Per-call wall clock in seconds. |
| `reasoning_effort` | `null` | Sent as is. Set it explicitly per role: left to the provider, thinking time and cost become an unmeasured lever. |
| `model_info` | `null` | AutoGen `ModelInfo` for model ids AutoGen does not know (`vision`, `function_calling`, `json_output`, `family`, `structured_output`, ...). A run reads it for the executor only; `aria-preflight` checks both roles through AutoGen, so set it for the advisor too. |

### pricing

USD per 1M tokens per model id: `prompt`, `completion`, and optionally
`cached_prompt` for prefix-cache hits. Set `cached_prompt` where the provider
discounts cache hits; the orchestrator re-sends a growing prefix every round,
so most prompt tokens are repeats.

Prices are keyed by model id, never by role, so an advisor-tier model run as
executor is billed at its own rate. A price may be written as `tbd` while the
model choice is open. The run then refuses to start: when the table is
non-empty, every configured model needs a real entry. An empty table means
no billing at all, which is only meant for offline tests.

`cost_multiplier` (the "roughly Nx your own cost" quoted to the executor in
the `organic` condition, where it calls the advisor as a tool) is the
advisor/executor completion-price ratio, or `10x` while prices are missing.

### advisor

| Field | Default | Meaning |
|---|---|---|
| `style` | `strategic` | `strategic` (assess, name the key correction, no plan), `redirect` (one concrete redirect) or `full_plan` (rewrite the remaining plan). |
| `max_guidance_words` | `80` | Word budget requested in the advisor prompt. |
| `max_advice_tokens` | `8192` | Advisor output cap per call, thinking included. |
| `fc3_nudge` | `true` | A short passage in the advisor prompt about premature stops and missing or weak verification. Keep it identical between training and evaluation. |
| `max_context_chars` | `60000` | How much of the trajectory the advisor sees. Trims the oldest material first and says so in the prompt. |
| `max_header_chars` | `3000` | How much of the pinned task header (the task text) survives; it is never front-trimmed. Raise it for long task statements. |
| `extra_context` | `""` | Static text appended to the advisor system prompt, e.g. the team's tool roster. It may only describe what the team itself has. |
| `repo_access` | `false` | Lets the advisor run whitelisted inspection commands (`cat`, `grep`, `ls`, `head`, `tail`, `sed`, `find`, `rg`, `wc`, `git diff/status/log/show`) in the workspace or through the adapter's inspector before answering. The whitelist checks the command name, not every flag (`find -exec`, for example, is not blocked), so it guards against accidents, not against a hostile model. Use it only on a disposable per-episode workspace. |
| `max_tool_calls` | `3` | Inspection commands per consult. |
| `gate_revision_rounds` | `0` | Above 0: if the advisor says `off_track` at the pre-delivery gate, the team keeps working instead of finalising, once per episode, still bounded by `max_steps`. |
| `gate_diff_injection` | `false` | Append the real workspace diff to every pre-delivery consult. On the host this needs `repo_access: true` (and a git workspace); with an adapter inspector, its `gate_snapshot` is used even when `repo_access` is off. |

### state

`state.dims` is an ordered list of dimension names from
`aria.core.state.DIM_REGISTRY`. The state index is their mixed-radix
product, and `rl.n_states` is always derived from it.

| Dim | Size | Values | Source |
|---|---|---|---|
| `position4` | 4 | FRAME, EXPLORE, REASON, DELIVER | Structural: first decision, 2nd and 3rd, 4th and later, pre-delivery gate. Stamped by the controller. |
| `exec3` | 3 | NONE, CLEAN, ERROR | Verdict of the last executed action, read only from process output, never from model text. |
| `outcome3` | 3 | NONE, OK, FAILED | Like `exec3`, but a tool result also counts as an action. In the stock AutoGen runner that is a tool return containing `=== TOOL RESULT ===` or `=== TOOL ERROR ===`; the error banner or an error in the step reads FAILED, anything else OK. For tool-calling environments; identical to `exec3` where only process output exists. |
| `consult` | 2 | NO, YES | Whether the advisor was already consulted this episode. |

The default `[position4, exec3, consult]` has 24 states. Any ordered subset
without duplicates is valid:

```yaml
state:
  dims: [position4, outcome3, consult]
```

The layout descriptor (e.g. `position44.outcome33.consult2`) plus a hash of
`phases.yaml` form the `state_signature` in every manifest and policy card.
`aria-eval --policy` refuses to load a promoted policy under a different
signature. A raw checkpoint (`--checkpoint` or
`condition_options.checkpoint`) is checked only against the state layout,
not against `phases.yaml`.

### rl

Fields of `aria.core.types.RLConfig`:

| Field | Default | Meaning |
|---|---|---|
| `gamma` | `1.0` | Discount. The consult cost is charged at the consult itself, so a discount below 1 would shrink only the terminal benefit and bias the policy towards late consults. |
| `epsilon_start` | `0.4` | Exploration rate at the start. |
| `epsilon_end` | `0.10` | Floor after the anneal. At 0.10 the minority-action propensity stays at 0.05, so inverse-propensity weights for off-policy re-weighting (for example a `lambda_cost` sweep) stay at or below 20. |
| `epsilon_anneal_episodes` | `110` | Linear anneal length in training episodes; `<= 0` holds `epsilon_start`. |
| `alpha_init` | `1.0` | Step size `alpha_init / n_visits`; at 1 every cell is the exact mean of its observed returns. |
| `backup` | `mc` | `mc` gives each decision its realised return from that decision to episode end (every-visit Monte Carlo). `onestep` uses one-step bootstrapped targets; the coarse state is not Markov, which biases them, so `mc` is the default. |

`n_states` is derived from the state layout and overwrites any configured
value; `n_actions` is fixed at 2 (SILENT, CONSULT). Leave both out of the
`rl` block. In eval mode the `aria` condition is frozen automatically:
greedy, no exploration, no updates.

### reward

`R = r_T - lambda * C`. `r_T` is the adapter's score in [0, 1] at episode
end (0 when not delivered). The cost term is charged at each CONSULT.
Fields of `aria.core.reward.RewardConfig`:

| Field | Default | Meaning |
|---|---|---|
| `lambda_cost` | `0.10` | Weight on the normalised cost: the reward, in solved tasks, charged per anchor span of advisor spend. |
| `cost_anchor_low_usd` | `0.0` | Measured USD per episode of an `autonomous` run. |
| `cost_anchor_high_usd` | `0.1663` | Measured USD per episode of an `always_consult` run. |
| `cost_term_rule` | `flat` | `flat`: every consult costs `flat_cost_usd`. `metered`: the consult's metered USD. `fixed`: the normalised constant `fixed_cost_per_consult`. |
| `flat_cost_usd` | `0.049` | Declared mean advisor price per consult. |
| `fixed_cost_per_consult` | `0.19` | Used only by `fixed`, must be in (0, 1]. |

Dollars are normalised by the anchor span, so with `flat` one consult costs
`lambda_cost * flat_cost_usd / (high - low)` in reward and a solved task is
worth `(high - low) / lambda_cost` dollars. With the defaults that is 0.0295
per consult and $1.66 per solve. The anchor and price defaults are measured
values for one reference setup; measure your own (an `autonomous` run for
the low anchor, an `always_consult` run with `lambda_cost: 0` for the high
anchor and the mean consult price).

The run refuses to start on an unknown `cost_term_rule`, when the high
anchor does not exceed the low one, or when `flat` has a non-positive price.
`lambda_cost: 0` skips the anchor and `flat` price checks, which is what lets
the anchor-measuring run start. `fixed` skips the anchor check and requires
`fixed_cost_per_consult` in (0, 1].

Telemetry always records metered dollars, whatever the rule.

### seeding

Deficit-targeted consult seeding for training. Every `every`-th episode
forces one consult at the first entry into the configured cell with the
largest visit deficit, so cells the natural policy rarely reaches become
estimable. Seeded decisions are flagged in the transition log.

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Off unless set. |
| `every` | `5` | Fire in episodes whose 1-based number is a multiple of this. |
| `deficit_target` | `12` | Deficit = target minus CONSULT visits over the cell's states. |
| `cells` | `[INTERIOR/ERROR, GATE/CLEAN/YES, GATE/ERROR]` | Label prefixes over the active layout. The defaults match no label of the default layout, so set them explicitly. |
| `second_consult_gate` | `true` | In seeded episodes, also consult at a gate reached already consulted, within the budget. |

```yaml
seeding:
  enabled: true
  every: 3
  deficit_target: 30
  cells: [EXPLORE/ERROR, REASON/ERROR, DELIVER/CLEAN/YES, DELIVER/ERROR]
```

Seeding applies only to the learned `aria` policy and never exceeds
`consult_budget`. Eval mode does not switch it off, so set
`enabled: false` (or leave the block empty) in evaluation configs.

### run

| Field | Default | Meaning |
|---|---|---|
| `episodes` | `100` | Episodes per seed. Episode `i` runs task `i % n_tasks`, so more episodes than tasks means repeated passes. |
| `seeds` | `[0, 1, 2, 3, 4]` | One run directory per seed. |
| `split` | `train` | Only used when set explicitly; otherwise train mode uses `train` and eval mode `test`. An eval on a non-test split logs a warning. |
| `task_limit` | `null` | Keep the first N tasks after shuffling. |
| `task_offset` | `0` | Skip the first N tasks after shuffling. For splitting an evaluation pass into parallel slices. |
| `task_ids` | `null` | Run exactly these ids in this order; overrides shuffle, offset and limit. |
| `shuffle_tasks` | `true` | Shuffle the task list per seed, seeded. |
| `checkpoint_every` | `20` | Training checkpoint interval in episodes; `<= 0` writes only the final one. |
| `results_root` | `results` | Output root. |

Runs are resumable: an existing run directory continues where its log ends
(a learned policy continues from its newest checkpoint).

## Matrix

A matrix expands into benchmark x condition x framework cells, each a full
experiment config. Schema (`aria.experiments.matrix.MatrixConfig`):

| Field | Default | Meaning |
|---|---|---|
| `name` | required | Becomes `name` of every cell. |
| `base` | required | Path to the base layer, a full or partial experiment config. |
| `condition_defaults` | `null` | Path to a file with a `conditions:` mapping, one entry per condition. |
| `benchmarks` | required | List of benchmark names. |
| `conditions` | required | List of condition names. |
| `frameworks` | `[autogen]` | List of framework names. |
| `seeds` | `[0, 1, 2, 3, 4]` | Replaces `run.seeds` in every cell. |
| `mode` | `train` | `train` or `eval` for every cell. |
| `overrides` | `{}` | Scoped partial configs, keys below. |

Override scopes are `benchmark:<name>`, `condition:<name>`,
`framework:<name>` and `cell:<benchmark>/<condition>`. Any other key raises.
Layers merge deep, later wins:

```
base < condition-defaults < benchmark: < condition: < framework: < cell:
```

`name`, `benchmark`, `condition`, `framework` and `run.seeds` are then set
from the matrix itself.

```yaml
name: main
base: configs/base/default.yaml
condition_defaults: configs/base/conditions.yaml
benchmarks: [toy]
conditions: [autonomous, always_consult, aria]
frameworks: [autogen]
seeds: [1, 2, 3]
mode: train
overrides:
  benchmark:toy:
    run: { episodes: 150 }
    consult_budget: 4
    benchmark_options: { teams_root: configs/teams }
  cell:toy/aria:
    seeding: { enabled: true, every: 3, cells: [REASON/ERROR, DELIVER/CLEAN/YES] }
```

`configs/base/conditions.yaml`:

```yaml
conditions:
  aria: {}
  autonomous: {}
  always_consult: {}
```

Every merged value remembers the layer that set it.
`aria-run <matrix> --dry-run` prints, per cell, each key set by a layer
above the base together with that layer, so an override that clobbers another
is visible before anything runs. `--filter benchmark=...`, `condition=...` or
`framework=...` restricts the cells. A cell counts as complete once every
seed directory has both `run_manifest.json` and `cost_summary.json`;
complete cells are skipped.

## Team spec

Loaded by `aria.frameworks.spec.load_team_spec(benchmark, teams_root)`.

`_defaults/agents.yaml` holds roster entries by agent name:

```yaml
agents:
  coder:
    kind: coder
    description: >-
      Writes and runs py or sh code blocks in the task workspace and
      debugs them on errors.
    tools: [python_exec]
    params:
      max_debug_rounds: 3
  file_surfer:
    kind: file_surfer
    description: >-
      Reads files in the task workspace. Cannot create or modify files.
    tools: [file_browser]
```

`<benchmark>/team.yaml` picks the roster, overrides entries and sets the
runtime knobs:

```yaml
benchmark: toy
agents: [coder, file_surfer]
overrides:
  coder:
    prompt: prompts/coder.md
runtime:
  common:
    max_steps: 25
    exec_timeout_s: 600
    episode_timeout_s: 3600
  autogen:
    max_stalls: 3
```

Agent entry fields (defaults entry, then override, merged key by key):

| Field | Meaning |
|---|---|
| `kind` | `coder` (Magentic-UI coder: writes and runs code blocks through `python_exec`), `file_surfer` (assistant with the read-only `file_browser` tools), `assistant` (plain assistant with the tools listed in `tools`), `code_executor` (runs code blocks, no model, no prompt). |
| `description` | What the orchestrator reads to choose the next speaker. Write it for that purpose. |
| `prompt` | Optional explicit prompt path, relative to the benchmark directory, then to `_defaults/`. Without it: `<benchmark>/prompts/<agent>.md`, then `_defaults/prompts/<agent>.md`. Every agent except `code_executor` needs a prompt file in one of these places; a missing file raises at load time. |
| `tools` | Tool symbols, resolved through the tool registry by `assistant` agents, which skip `python_exec`. `coder` always uses `python_exec` and `file_surfer` always uses `file_browser`, whatever this lists. |
| `params` | Kind-specific; the coder reads `max_debug_rounds` (default 3). |
| `model_role` | Recorded in the spec. The AutoGen runner builds every agent on the executor client. |

`team.yaml` must list at least one agent, may only override agents in its
roster, and if it names `benchmark:` it must match the directory. Prompts
are format strings rendered strictly from `{date_today}` plus the adapter's
`prompt_vars`; an unknown placeholder raises. Literal braces in a prompt are
written `{{` and `}}`.

Runtime knobs: `runtime.common` merged with `runtime.<framework>`, recorded
in every manifest. The AutoGen runner reads:

| Knob | Default | Meaning |
|---|---|---|
| `max_steps` | `20` | Orchestrator rounds before the episode stops with `max_rounds`. |
| `max_stalls` | `3` | Stall count that triggers a replan (a STALL contact point). |
| `exec_timeout_s` | `600` | Per code-block timeout. Too short and a cut-off test run reads like a failing one. |
| `episode_timeout_s` | `max_steps * exec_timeout_s` | Wall-clock cap on the agent phase. The adapter's `task_timeout_s` can lower it per task. |

Other keys are kept and recorded but not read.

## phases.yaml

`<teams_root>/<benchmark>/phases.yaml` configures the deterministic
detector. Loaded by `aria.detectors.load_phases_config`.

```yaml
classifier: heuristic
conceptual_streak: 2
markers:
  frame: [read the issue, understand, plan]
  explore: [grep, find, locate, inspect]
  reason: [edit, patch, fix, implement, debug]
  deliver: [final, diff, submit]
state_changing:
  patterns:
    - '\bsed -i\b'
    - '\.write_text\('
verification:
  evidence_of_run:
    - '^=+ test session starts'
    - '\bRan \d+ tests? in '
  pass_requires:
    - '\b[1-9]\d* passed\b'
```

| Key | Default | Meaning |
|---|---|---|
| `classifier` | `heuristic` | The only accepted value. |
| `conceptual_streak` | `2` | Failure count (each failing step plus its internal repairs) since the last passing check at which a mechanical error becomes conceptual. |
| `markers.<phase>` | empty | Keywords per phase (`frame`, `explore`, `reason`, `deliver`), matched against the upcoming instruction. |
| `state_changing.patterns` | empty | Regexes (multiline) for steps that change the artifact. A change invalidates an earlier passing check. |
| `verification.evidence_of_run` | empty | Regexes that show a check actually ran. Match the runner's own output, not the command the agent typed. |
| `verification.pass_requires` | empty | Regexes that show it passed. A pass also needs the step to end without an error. |

Phase, evidence and commitment are logged covariates; the default state
layout reads only the structural position, the executed-output verdict and
the consult bit. The parsed content of `phases.yaml` (classifier, streak,
markers and evidence tables) is hashed into the `state_signature`, so freeze
the file before training.

## families.yaml

Maps benchmarks to task families for the policy store layout
(`artifacts/policies/<family>/<benchmark>/<policy_id>/`):

```yaml
families:
  toy: arithmetic
```

A benchmark missing from the map lands under `unassigned`. `aria-policy`
and `aria-eval --policy` need the file to exist.

## Secrets

`aria_secrets.yaml` in the working directory, or `--secrets PATH`:

```yaml
api_key: "..."
# executor_api_key: "..."
# advisor_api_key: "..."
```

`api_key` applies to both roles, `<role>_api_key` overrides it. A key set
directly in a config wins over both. Keys never reach manifests.

## Environment variables

| Variable | Effect |
|---|---|
| `ARIA_OBS_FALLBACK` | `0` turns off the transcript fallback of `outcome3` (the `=== TOOL RESULT ===` and `=== TOOL ERROR ===` banners a tool bridge echoes into its return). The machine channels stay on. Default on. Use it to evaluate a policy under the exact encoding it was trained with, and record the setting with the run. |
