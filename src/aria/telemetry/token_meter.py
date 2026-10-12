"""TokenMeter: per-call token and cost measurement for benchmark logging.

Tokens are recorded only at the model-client boundary, by
MeteredChatCompletionClient (executor) and MeteredCompletionClient (advisor),
so no call is counted twice. Each call becomes a `CallRecord` tagged with the
purpose set by `call_purpose(...)`. Consult cost grows with transcript length,
so consult spend is recorded in dollars rather than inferred from consult counts.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

_PURPOSE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "aria_call_purpose", default=None
)


@contextmanager
def call_purpose(purpose: str) -> Iterator[None]:
    """Tag every model call made in this task while the context is active.

    The AutoGen integration uses: planning, ledger and final_answer
    (orchestrator), agent_step (executor default), consult (advisor default,
    controller-triggered) and organic_consult (executor-triggered advisor calls).
    """
    token = _PURPOSE.set(purpose)
    try:
        yield
    finally:
        _PURPOSE.reset(token)


def current_purpose() -> str | None:
    return _PURPOSE.get()


@dataclass(frozen=True)
class TokenPricing:
    """Prices for one model in USD per 1M tokens.

    `cached_prompt` is the rate for provider-cached prompt tokens; None falls
    back to the full prompt rate.
    """

    prompt: float = 0.0
    completion: float = 0.0
    cached_prompt: float | None = None

    def cost(
        self, prompt_tokens: int, completion_tokens: int, cached_prompt_tokens: int = 0
    ) -> float:
        cached_rate = self.cached_prompt if self.cached_prompt is not None else self.prompt
        fresh_prompt = max(0, prompt_tokens - cached_prompt_tokens)
        return (
            fresh_prompt * self.prompt
            + cached_prompt_tokens * cached_rate
            + completion_tokens * self.completion
        ) / 1_000_000


@dataclass(frozen=True)
class CallRecord:
    """One model call. The atomic unit of all cost/token reporting."""

    index: int  # global call index within the run
    episode: int | None
    role: str  # executor | advisor
    purpose: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    cache_hit: bool  # whole response served from a client-side cache (cost 0)
    cost_usd: float
    # Thinking tokens, billed at the completion rate but reported separately. Taken from
    # completion_tokens_details, else total - prompt - completion (frameworks/autogen/metering.py).
    reasoning_tokens: int = 0
    # Wall-clock duration of the call in seconds.
    latency_s: float = 0.0
    # Exception class name if the call raised, else "". A failed call books zero
    # tokens and cost (the provider may still bill it); its latency stays logged.
    failed: str = ""

    @property
    def total_tokens(self) -> int:
        # Thinking tokens are generated and billed, so they count toward the total.
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenMeter:
    # Keyed by model id, not role, so a strong model running as executor is
    # billed at its own rate. None marks a price left as tbd in the config.
    pricing: dict[str, TokenPricing | None] = field(default_factory=dict)

    calls: list[CallRecord] = field(default_factory=list)  # full run
    episode_calls: list[CallRecord] = field(default_factory=list)
    events: dict[str, int] = field(default_factory=dict)  # episode scope
    run_events: dict[str, int] = field(default_factory=dict)
    _episode: int | None = None

    # ------------------------------------------------------------------
    # Recording (called only by the metered client wrappers, plus count_event)
    # ------------------------------------------------------------------

    def record_call(
        self,
        *,
        role: str,
        model: str = "",
        prompt_tokens: int,
        completion_tokens: int,
        # Billed at the completion rate; see CallRecord.reasoning_tokens.
        reasoning_tokens: int = 0,
        cached_prompt_tokens: int = 0,
        cache_hit: bool = False,
        # Seconds; see CallRecord.latency_s.
        latency_s: float = 0.0,
        # Exception class name when the call raised; see CallRecord.failed.
        failed: str = "",
        purpose: str | None = None,
        default_purpose: str = "other",
    ) -> CallRecord:
        purpose = purpose or current_purpose() or default_purpose
        # Cache hits and failed calls cost 0.0 (a failed call has no usage and is
        # not estimated); every other call needs a configured per-model price.
        cost = (
            0.0
            if cache_hit or failed
            else self._price_for(model).cost(
                prompt_tokens, completion_tokens + reasoning_tokens, cached_prompt_tokens
            )
        )
        record = CallRecord(
            index=len(self.calls),
            episode=self._episode,
            role=role,
            purpose=purpose,
            model=model,
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            reasoning_tokens=int(reasoning_tokens),
            cached_prompt_tokens=int(cached_prompt_tokens),
            cache_hit=cache_hit,
            latency_s=latency_s,
            failed=failed,
            cost_usd=cost,
        )
        self.calls.append(record)
        self.episode_calls.append(record)
        return record

    def _price_for(self, model: str) -> TokenPricing:
        """Return the price for a model id.

        An empty pricing table means cost-free offline use (tests). Otherwise a
        missing model or a tbd entry (None) raises ValueError, so no call is billed
        at a guessed rate.
        """
        if not self.pricing:
            return TokenPricing()
        if model not in self.pricing:
            raise ValueError(
                f"no price configured for model {model!r}; add it to the "
                "pricing block (keyed by model id) in configs/base/default.yaml"
            )
        entry = self.pricing[model]
        if entry is None:
            raise ValueError(
                f"price for model {model!r} is tbd; fill in the real rates "
                "in configs/base/default.yaml before any run"
            )
        return entry

    def count_event(self, name: str) -> None:
        self.events[name] = self.events.get(name, 0) + 1
        self.run_events[name] = self.run_events.get(name, 0) + 1

    # ------------------------------------------------------------------
    # Scoping
    # ------------------------------------------------------------------

    def start_episode(self, episode: int) -> None:
        """Open a new episode scope; run-level calls and events persist."""
        self._episode = episode
        self.episode_calls = []
        self.events = {}

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize(calls: list[CallRecord], events: dict[str, int]) -> dict[str, Any]:
        by_role: dict[str, dict[str, float]] = {}
        by_purpose: dict[str, dict[str, float]] = {}
        for call in calls:
            for bucket, key in ((by_role, call.role), (by_purpose, call.purpose)):
                entry = bucket.setdefault(
                    key,
                    {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "reasoning_tokens": 0,
                        "failed_calls": 0,
                        "cost_usd": 0.0,
                    },
                )
                entry["calls"] += 1
                entry["prompt_tokens"] += call.prompt_tokens
                entry["completion_tokens"] += call.completion_tokens
                entry["reasoning_tokens"] += call.reasoning_tokens
                entry["failed_calls"] += 1 if call.failed else 0
                entry["cost_usd"] += call.cost_usd
        total_cost = sum(c.cost_usd for c in calls)
        advisor_cost = sum(c.cost_usd for c in calls if c.role == "advisor")
        return {
            "calls": len(calls),
            "total_tokens": sum(c.total_tokens for c in calls),
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "reasoning_tokens": sum(c.reasoning_tokens for c in calls),
            "cached_prompt_tokens": sum(c.cached_prompt_tokens for c in calls),
            # Failed calls book no tokens or cost; without these two fields an
            # episode lost to timed-out requests looks like a cheap, quick one.
            "failed_calls": sum(1 for c in calls if c.failed),
            "failed_latency_s": sum(c.latency_s for c in calls if c.failed),
            "cost_usd": total_cost,
            "advisory_cost_usd": advisor_cost,
            "advisory_cost_fraction": (advisor_cost / total_cost) if total_cost > 0 else 0.0,
            "by_role": by_role,
            "by_purpose": by_purpose,
            "events": dict(events),
        }

    def episode_summary(self) -> dict[str, Any]:
        return self._summarize(self.episode_calls, self.events)

    def run_summary(self) -> dict[str, Any]:
        return self._summarize(self.calls, self.run_events)

    def consult_costs(self) -> list[float]:
        """Return the USD cost of every advisor consult in the run.

        Basis for comparing consult spend between conditions in dollars.
        """
        return [
            c.cost_usd
            for c in self.calls
            if c.role == "advisor" and c.purpose in ("consult", "organic_consult")
        ]
