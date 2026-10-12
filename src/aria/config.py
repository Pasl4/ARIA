"""Experiment configuration: one YAML file fully describes a run.

API keys live in `aria_secrets.yaml` in the working directory (keep it out of
version control) and are merged in at load time, so experiment configs stay
committable. Loading checks that the models block defines exactly the executor
and advisor roles; unknown keys in `rl:` / `reward:` raise when the run builds
them (rl_config(), reward_config()).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .core.reward import RewardConfig
from .core.state import DIM_REGISTRY, StateLayout
from .core.types import RLConfig


def _dataclass_from_dict(cls: type, values: dict[str, Any], section: str) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"unknown {section} config keys: {sorted(unknown)}; known: {sorted(known)}"
        )
    return cls(**values)


class ModelConfig(BaseModel):
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None  # per-call output cap
    # Retries cover rate limits and transient 5xx errors, so keep them above
    # zero. Each retry can wait a full `timeout`, so the worst case per call is
    # timeout x (retries + 1).
    max_retries: int | None = 3
    # Per-call wall-clock cap in seconds. Without it a hung request blocks the
    # episode until `episode_timeout_s`.
    timeout: float | None = 120.0
    # Thinking effort per role. Set it explicitly and hold it equal across
    # conditions: provider defaults vary and can change with a model update,
    # long thinking produces silent reads that trip the timeout, and some layers
    # bill thinking as output without reporting it (see frameworks/autogen/metering.py).
    reasoning_effort: str | None = None
    model_info: dict[str, Any] | None = None

    def sanitized(self) -> dict[str, Any]:
        """Manifest-safe dump of the model config without the API key."""
        return self.model_dump(exclude={"api_key"})


class PricingConfig(BaseModel):
    """USD per 1M tokens.

    `cached_prompt` is the provider's discounted rate for prefix-cache hits.
    An orchestrator loop re-sends a growing prefix every round, so most prompt
    tokens are repeats eligible for the prefix cache; leaving it None prices
    them at the full rate, an upper bound that differs across conditions.
    """

    prompt: float = 0.0
    completion: float = 0.0
    cached_prompt: float | None = None


class StateConfig(BaseModel):
    """State-layout selection: a layout is a list of dim names, not code.

    The dims list defines the state space: index arithmetic, n_states, the
    checkpoint and policy-card signature, and the manifest record.
    """

    model_config = ConfigDict(extra="forbid")

    dims: list[str] = Field(default_factory=lambda: ["position4", "exec3", "consult"])

    @field_validator("dims")
    @classmethod
    def _known_dims(cls, v: list[str]) -> list[str]:
        unknown = [d for d in v if d not in DIM_REGISTRY]
        if unknown:
            raise ValueError(
                f"unknown state dims {unknown}; known: {sorted(DIM_REGISTRY)}"
            )
        return v


class AdvisorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    style: str = "strategic"  # strategic | redirect | full_plan
    max_guidance_words: int = 80  # word budget requested in the advisor prompt
    max_advice_tokens: int = 8192  # output cap: headroom for thinking + answer
    # Verification nudge in the advisor system prompt (MAST task-verification
    # category FC3, see advisor.py). It changes what the advisor says, so it is
    # recorded in the manifest and held identical for every consulting
    # condition, which leaves consult timing as the only difference.
    fc3_nudge: bool = True
    # Trajectory characters shown to the advisor. When the limit binds, the
    # oldest material is trimmed, the prompt says so, and
    # `advisor_context_trimmed` is counted.
    max_context_chars: int = 60_000
    # Characters kept of the pinned task header (the task statement). Raise it
    # where task statements exceed the limit; otherwise the tail of the header
    # (often the completion contract) is cut.
    max_header_chars: int = 3_000
    # Static text appended to the advisor system prompt, e.g. the team's tool
    # roster, so the advisor does not steer the team away from tools it assumes
    # are missing. It may only describe what the team itself has, so the
    # advisor gets no information the team lacks.
    extra_context: str = ""
    # Read-only repository inspection for the advisor (whitelisted commands
    # incl. git diff), up to max_tool_calls per consult. Independently,
    # gate_revision_rounds > 0 lets an off_track verdict at the pre-delivery
    # gate re-open the group chat once per episode instead of finalizing.
    repo_access: bool = False
    max_tool_calls: int = 3
    gate_revision_rounds: int = 0
    # Append the actual workspace changes to every PRE_DELIVERY consult: the
    # adapter's advisor_inspector snapshot when it provides one, else the host
    # `git diff` (needs repo_access, which binds the workspace).
    gate_diff_injection: bool = False


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes: int = 100
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    # An explicit split wins over the mode default, so an eval-mode run that
    # targets train (e.g. for model selection) stays on train.
    split: str = "train"
    task_limit: int | None = None
    # Start of the slice of the (optionally shuffled) task list, applied before
    # task_limit, e.g. to run one pass as parallel shards. Valid only for
    # non-learning conditions, whose episodes share no state. Recorded in the
    # manifest.
    task_offset: int = 0
    # Run exactly these task ids in the given order, ignoring shuffle, offset
    # and limit (e.g. to re-run episodes that produced no verdict). An unknown
    # id fails at startup. Recorded in the manifest, so a re-run stays
    # distinguishable from a fresh pass.
    task_ids: list[str] | None = None
    shuffle_tasks: bool = True  # per-seed task order (seeded)
    checkpoint_every: int = 20  # <= 0: final checkpoint only
    results_root: str = "results"  # layout: results/<benchmark>/<condition>/...

    @property
    def split_is_explicit(self) -> bool:
        return "split" in self.model_fields_set


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    benchmark: str
    benchmark_options: dict[str, Any] = Field(default_factory=dict)
    condition: str = "aria"
    condition_options: dict[str, Any] = Field(default_factory=dict)
    framework: str = "autogen"  # which FrameworkRunner drives the episodes
    teams_root: str = "configs/teams"  # TeamSpec + phase-schema location
    # Scored artifact: "workspace_patch" (the agent's git diff, default) or
    # "typed" (the model-typed final answer). A benchmark without a workspace
    # yields no patch and falls back to typed.
    delivery: str = "workspace_patch"
    advisor: AdvisorConfig = Field(default_factory=AdvisorConfig)
    # Consult cap per episode (constrained-MDP budget), applied to all
    # conditions alike; None = off.
    consult_budget: int | None = None
    models: dict[str, ModelConfig]  # exactly: executor, advisor
    # Keyed by model id, not by role. `tbd` in the YAML loads as None; metering
    # raises before computing any cost from it.
    pricing: dict[str, PricingConfig | None] = Field(default_factory=dict)
    rl: dict[str, Any] = Field(default_factory=dict)
    reward: dict[str, Any] = Field(default_factory=dict)
    # Deficit-targeted consult seeding (orchestration/seeding.py); {} = off.
    seeding: dict[str, Any] = Field(default_factory=dict)
    state: StateConfig = Field(default_factory=StateConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @field_validator("models")
    @classmethod
    def _require_roles(cls, v: dict[str, ModelConfig]) -> dict[str, ModelConfig]:
        if set(v) != {"executor", "advisor"}:
            raise ValueError(f"models must define exactly executor and advisor, got {sorted(v)}")
        return v

    @field_validator("pricing", mode="before")
    @classmethod
    def _tbd_prices(cls, v: Any) -> Any:
        """Map `tbd` price entries to None (price not yet known)."""
        if isinstance(v, dict):
            return {
                k: (None if isinstance(p, str) and p.lower() == "tbd" else p)
                for k, p in v.items()
            }
        return v

    def rl_config(self) -> RLConfig:
        cfg = _dataclass_from_dict(RLConfig, self.rl, "rl")
        # n_states always derives from the layout, never from the rl block.
        cfg.n_states = self.state_layout().n_states
        return cfg

    def state_layout(self) -> StateLayout:
        return StateLayout(tuple(self.state.dims))

    def reward_config(self) -> RewardConfig:
        return _dataclass_from_dict(RewardConfig, self.reward, "reward")

    def organic_consult(self) -> bool:
        return self.condition == "organic"

    def cost_multiplier(self) -> str:
        """Advisor/executor completion-price ratio as a string, e.g. "10x".

        Quoted in the consult tool description ("roughly Nx your own cost").
        Pricing is keyed by model id; the roles resolve through `models`.
        Falls back to "10x" while prices are tbd.
        """
        adv = self.pricing.get(self.models["advisor"].model)
        exe = self.pricing.get(self.models["executor"].model)
        if not adv or not exe or exe.completion <= 0:
            return "10x"
        return f"{max(1, round(adv.completion / exe.completion))}x"

    def manifest_view(self) -> dict[str, Any]:
        """Full config for run manifests, with model identities but no API keys."""
        data = self.model_dump(exclude={"models"})
        data["models"] = {role: m.sanitized() for role, m in self.models.items()}
        return data


def merge_secrets(
    raw: dict[str, Any], secrets_path: str | Path | None = None
) -> dict[str, Any]:
    """Inject API keys from aria_secrets.yaml into a raw config mapping.

    Secrets are a flat mapping: `api_key` as the default plus optional per-role
    overrides (`executor_api_key`, `advisor_api_key`). A key already set in the
    config is kept; committed configs should not carry keys themselves.
    """
    secrets_file = Path(secrets_path) if secrets_path else Path("aria_secrets.yaml")
    if secrets_file.exists():
        secrets = yaml.safe_load(secrets_file.read_text()) or {}
        default_key = secrets.get("api_key")
        for role, model_cfg in raw.get("models", {}).items():
            model_cfg.setdefault("api_key", secrets.get(f"{role}_api_key", default_key))
    return raw


def load_config(path: str | Path, secrets_path: str | Path | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return ExperimentConfig(**merge_secrets(raw, secrets_path))
