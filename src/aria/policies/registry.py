"""Policy registry: maps condition names to policy factories.

This is the single switch point between conditions; ``register_policy`` adds
new ones without editing aria itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.state import StateLayout
from ..core.types import RLConfig
from .base import InterventionPolicy
from .baselines import (
    AlwaysConsultPolicy,
    AlwaysSilentPolicy,
    AnthropicAdvisorPolicy,
    ConsultAtPlanPolicy,
    EarlyGateTwicePolicy,
    EarlyOncePolicy,
    GateOncePolicy,
    RandomAdvisoryPolicy,
)
from .learned import AriaPolicy

PolicyFactory = Callable[..., InterventionPolicy]
# Factories receive keyword args: seed, rl_config, options (dict), layout
# (StateLayout | None; only index-based policies need it).
_REGISTRY: dict[str, PolicyFactory] = {}


def register_policy(condition: str, factory: PolicyFactory) -> None:
    """Register a condition name -> policy factory (adopter extension point)."""
    if condition in _REGISTRY:
        raise ValueError(f"condition {condition!r} is already registered")
    _REGISTRY[condition] = factory


def available_conditions() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_policy(
    condition: str,
    *,
    seed: int | None = None,
    rl_config: RLConfig | None = None,
    options: dict[str, Any] | None = None,
    layout: StateLayout | None = None,
) -> InterventionPolicy:
    try:
        factory = _REGISTRY[condition]
    except KeyError as exc:
        raise ValueError(
            f"Unknown condition {condition!r}; expected one of {available_conditions()}"
        ) from exc
    return factory(seed=seed, rl_config=rl_config, options=options or {}, layout=layout)


# ---------------------------------------------------------------------------
# Built-in conditions
# ---------------------------------------------------------------------------

register_policy(
    "aria",
    lambda *, seed=None, rl_config=None, options=None, layout=None: AriaPolicy(
        rl_config=rl_config,
        seed=seed,
        checkpoint=(options or {}).get("checkpoint"),
        frozen=bool((options or {}).get("frozen", False)),
        layout=layout,
    ),
)
# No advisor: the executor works alone.
register_policy(
    "autonomous",
    lambda *, seed=None, rl_config=None, options=None, layout=None: AlwaysSilentPolicy(),
)
# Executor self-routes: the runner enables the consult_advisor tool for this
# condition and the external policy stays silent.
register_policy(
    "organic",
    lambda *, seed=None, rl_config=None, options=None, layout=None: AlwaysSilentPolicy(),
)
register_policy(
    "plan_advisory",
    lambda *, seed=None, rl_config=None, options=None, layout=None: ConsultAtPlanPolicy(),
)
register_policy(
    "random_advisory",
    lambda *, seed=None, rl_config=None, options=None, layout=None: RandomAdvisoryPolicy(
        consult_probability=float((options or {})["consult_probability"]), seed=seed
    ),
)
register_policy(
    "anthropic_heuristic",
    lambda *, seed=None, rl_config=None, options=None, layout=None: AnthropicAdvisorPolicy(
        consult_on_mechanical=bool((options or {}).get("consult_on_mechanical", False)),
        deliver_every_time=bool((options or {}).get("deliver_every_time", False)),
    ),
)
register_policy(
    "early_gate_2x",
    lambda *, seed=None, rl_config=None, options=None, layout=None: EarlyGateTwicePolicy(),
)
register_policy(
    "early_once",
    lambda *, seed=None, rl_config=None, options=None, layout=None: EarlyOncePolicy(),
)
register_policy(
    "gate_once",
    lambda *, seed=None, rl_config=None, options=None, layout=None: GateOncePolicy(),
)
register_policy(
    "always_consult",
    lambda *, seed=None, rl_config=None, options=None, layout=None: AlwaysConsultPolicy(),
)
