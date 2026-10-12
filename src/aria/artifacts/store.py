"""PolicyStore: trained policies as citable artifacts, kept apart from results.

`promote` copies a run's final checkpoint into the store with a provenance
card. Lookups go through `index.jsonl` (`resolve(policy_id)`), never through
hardcoded paths. Cards record `state_signature` and `framework`, because a
Q-table is bound to one state space and to one framework's contact points.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

DEFAULT_STORE_ROOT = Path("artifacts/policies")
DEFAULT_FAMILIES = Path("configs/families.yaml")


@dataclass(frozen=True)
class PolicyCard:
    """Provenance of a promoted policy, stored as card.json and in the index."""

    policy_id: str
    family: str
    benchmark: str
    framework: str
    condition: str
    seed: int
    gamma: float | None
    episodes_trained: int
    state_signature: str
    supported_contact_points: list[str]
    source_run: str
    git_commit: str | None
    config_hash: str
    created_at: str


def load_families(path: str | Path = DEFAULT_FAMILIES) -> dict[str, str]:
    """Return the benchmark-to-family map from the `families` key of a YAML file."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return dict(raw.get("families", {}))


class PolicyStore:
    """Policy artifacts under `root`, grouped by task family and benchmark.

    The family level (map read from `families_path`) lets one policy be reused
    across the benchmarks of a family. Layout: `index.jsonl` (one card per
    line) and `<family>/<benchmark>/<policy_id>/` holding policy.json
    (AriaPolicy.state_dict()) and card.json.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_STORE_ROOT,
        families_path: str | Path = DEFAULT_FAMILIES,
    ) -> None:
        self.root = Path(root)
        self._families = load_families(families_path)

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote(self, run_dir: str | Path, policy_id: str | None = None) -> PolicyCard:
        """Copy the final checkpoint of `run_dir` into the store and index its card."""
        run_dir = Path(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_ep*.json"))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")
        final = checkpoints[-1]
        payload = json.loads(final.read_text())

        config = manifest.get("config", {})
        benchmark = str(config.get("benchmark", "unknown"))
        family = self._families.get(benchmark, "unassigned")
        condition = str(config.get("condition", "unknown"))
        framework = str(manifest.get("framework", config.get("framework", "unknown")))
        seed = int(manifest.get("seed", -1))
        gamma = config.get("rl", {}).get("gamma")
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        if policy_id is None:
            gamma_tag = f"g{gamma}" if gamma is not None else "gNA"
            policy_id = f"{benchmark}_{condition}_{gamma_tag}_seed{seed}_{config_hash[:7]}"

        card = PolicyCard(
            policy_id=policy_id,
            family=family,
            benchmark=benchmark,
            framework=framework,
            condition=condition,
            seed=seed,
            gamma=gamma,
            episodes_trained=int(payload.get("episode", -1)),
            state_signature=str(manifest.get("state_signature", "")),
            supported_contact_points=list(manifest.get("supported_contact_points", [])),
            source_run=str(run_dir),
            git_commit=manifest.get("git_commit"),
            config_hash=config_hash,
            created_at=datetime.now(UTC).isoformat(),
        )

        target = self.root / family / benchmark / policy_id
        if target.exists():
            raise FileExistsError(f"policy {policy_id!r} already exists at {target}")
        target.mkdir(parents=True)
        shutil.copyfile(final, target / "policy.json")
        (target / "card.json").write_text(json.dumps(asdict(card), indent=2))
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "index.jsonl").open("a") as f:
            f.write(json.dumps(asdict(card)) + "\n")
        return card

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def list_cards(self) -> list[PolicyCard]:
        index = self.root / "index.jsonl"
        if not index.exists():
            return []
        return [
            PolicyCard(**json.loads(line))
            for line in index.read_text().splitlines()
            if line.strip()
        ]

    def get_card(self, policy_id: str) -> PolicyCard:
        for card in self.list_cards():
            if card.policy_id == policy_id:
                return card
        raise KeyError(f"no policy {policy_id!r} in {self.root / 'index.jsonl'}")

    def resolve(self, policy_id: str) -> Path:
        """Return the path to the artifact's policy.json (loadable by AriaPolicy.load)."""
        card = self.get_card(policy_id)
        path = self.root / card.family / card.benchmark / card.policy_id / "policy.json"
        if not path.exists():
            raise FileNotFoundError(f"index lists {policy_id!r} but {path} is missing")
        return path

    def verify(self, policy_id: str, expected_state_signature: str) -> PolicyCard:
        """Return the card if its state_signature matches the current state space.

        The signature combines the state layout and the phase-schema hash; call
        this before an eval loads the policy. Raises ValueError on a mismatch;
        cards without a recorded signature pass unchecked.
        """
        card = self.get_card(policy_id)
        if card.state_signature and card.state_signature != expected_state_signature:
            raise ValueError(
                f"policy {policy_id!r} was trained under state_signature "
                f"{card.state_signature!r}, current schema is "
                f"{expected_state_signature!r}; refusing to load across "
                f"state-space versions"
            )
        return card
