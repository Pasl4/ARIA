"""Reward model: benchmark quality minus a declared price per consult.

R = r_T - lambda * C, with `r_T` in [0, 1] the official score at episode end
(0 if undelivered) and C the normalized cost charged at each consult. The
default "flat" rule prices every consult at `flat_cost_usd`, the constant shadow
price of a per-episode consult count; a metered price would charge late,
long-context consults such as the pre-delivery gate more and vary with retries
and thinking tokens the policy cannot observe. Returns may go negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Action


@dataclass
class RewardConfig:
    """Every reward component in one place.

    Manifests record the `reward:` block as written, so unset fields mean these
    defaults. The anchor and price defaults belong to one reference setup;
    other benchmarks and model pairs need their own measured anchors and price.
    """

    # Weight on the normalized cost C, in solved tasks per anchor span: a
    # solve is worth anchor_span / lambda dollars.
    lambda_cost: float = 0.10
    # Normalization anchors in USD per episode, both measured: low = an
    # autonomous episode on the executor model, high = an always_consult
    # episode (the most the action space can spend).
    cost_anchor_low_usd: float = 0.0
    cost_anchor_high_usd: float = 0.1663
    # "flat": declared mean price per consult action (default). "metered":
    # C = metered USD / anchor span. "fixed": hand-set normalized constant.
    # The rule sets only the reward; telemetry meters real dollars under every rule.
    cost_term_rule: str = "flat"
    # Declared mean advisor price per consult in USD, valid for the default
    # 60k-char advisor context limit.
    flat_cost_usd: float = 0.049
    fixed_cost_per_consult: float = 0.19
    version: str = "v4"

    def validate(self) -> None:
        """Raise on an unknown rule or on invalid anchors and prices.

        `lambda_cost = 0` skips the anchor and flat-price checks: the cost term is
        then identically zero, which lets the always_consult run that measures
        the high anchor start before that anchor exists.
        """
        if self.cost_term_rule not in ("flat", "metered", "fixed"):
            raise ValueError(
                f"unknown reward.cost_term_rule {self.cost_term_rule!r}; "
                "expected flat | metered | fixed"
            )
        if self.cost_term_rule == "fixed":
            if not 0.0 < self.fixed_cost_per_consult <= 1.0:
                raise ValueError("reward.fixed_cost_per_consult must be in (0, 1]")
            return
        if self.lambda_cost == 0.0:
            return
        if self.cost_anchor_high_usd <= self.cost_anchor_low_usd:
            raise ValueError(
                "reward.cost_anchor_high_usd must exceed cost_anchor_low_usd; "
                "both are measured from autonomous and always_consult runs "
                "of the benchmark being run and have no meaningful "
                "universal default"
            )
        if self.cost_term_rule == "flat" and self.flat_cost_usd <= 0.0:
            raise ValueError(
                "reward.flat_cost_usd must be the measured mean consult price "
                "in USD (> 0) under cost_term_rule \'flat\'"
            )


@dataclass
class RewardModel:
    """Pure functions over measured quantities; holds no episode state.

    Per-task baselines belong in offline evaluation only: in the live update
    they would change the argmax the run explores under.
    """

    config: RewardConfig = field(default_factory=RewardConfig)

    # --- layer 1: quality -------------------------------------------------

    def terminal_reward(self, score: float, delivered: bool) -> float:
        """The benchmark score, clamped to [0, 1]. Undelivered scores 0.

        A wrong answer and no answer both score 0, as the benchmark scores
        them; delivery rate is reported as a separate metric.
        """
        if not delivered:
            return 0.0
        return max(0.0, min(1.0, score))

    # --- layer 2: cost ----------------------------------------------------

    def normalized_cost(self, consult_cost_usd: float) -> float:
        """USD on the [0, 1] anchor scale.

        Not clamped above 1, so spending beyond the always_consult anchor is
        penalized in full and shows up as C > 1 in the logs.
        """
        cfg = self.config
        span = cfg.cost_anchor_high_usd - cfg.cost_anchor_low_usd
        if span <= 0:
            raise ValueError("cost anchors unmeasured; call RewardConfig.validate()")
        return max(0.0, consult_cost_usd / span)

    def cost_term(self, consult_cost_usd: float) -> float:
        """The `- lambda * C` contribution of one CONSULT decision.

        Charged at the consult itself, so the action that spent the money
        carries the penalty. "flat" and "fixed" price the action regardless of
        the metered spend, including truncated or failed advisor replies, so
        the policy gains nothing from degraded consults. "metered" prices the
        measured spend.
        """
        cfg = self.config
        if cfg.lambda_cost == 0.0:
            return 0.0
        if cfg.cost_term_rule == "flat":
            return -cfg.lambda_cost * self.normalized_cost(cfg.flat_cost_usd)
        if cfg.cost_term_rule == "fixed":
            return -cfg.lambda_cost * cfg.fixed_cost_per_consult
        if consult_cost_usd <= 0.0:
            return 0.0
        return -cfg.lambda_cost * self.normalized_cost(consult_cost_usd)

    # --- composition ------------------------------------------------------

    def step_reward(self, action: Action, consult_cost_usd: float = 0.0) -> float:
        """Reward of one decision (the cost term for CONSULT, else 0).

        The caller adds the terminal quality term at the terminal transition,
        since it belongs to the episode rather than a single decision.
        """
        if action == Action.CONSULT:
            return self.cost_term(consult_cost_usd)
        return 0.0
