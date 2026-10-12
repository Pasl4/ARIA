"""Experiment matrix: one YAML expands to benchmark x condition x framework.

Merge precedence, later layers win:
    base < condition-defaults < benchmark:<b> < condition:<c> < framework:<f> < cell:<b>/<c>
Each value records the layer that set it, so `aria-run --dry-run` prints effective
configs with origins and an override that clobbers another layer is visible. Paths
are relative to the working directory; see tests/fixtures/configs/experiments/example_main.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from ..config import ExperimentConfig, merge_secrets
from . import layout


class MatrixConfig(BaseModel):
    name: str
    base: str
    condition_defaults: str | None = None
    benchmarks: list[str]
    conditions: list[str]
    frameworks: list[str] = Field(default_factory=lambda: ["autogen"])
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    mode: str = "train"
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    benchmark: str
    condition: str
    framework: str
    mode: str
    config: ExperimentConfig
    # dotted key path -> layer that set the value (e.g. "rl.gamma" -> "condition:aria")
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def cell(self) -> str:
        return f"{self.benchmark}/{self.condition}/{self.framework}"

    def seed_dirs(self) -> list[Path]:
        return [
            layout.run_dir(
                self.config.run.results_root,
                self.benchmark,
                self.condition,
                self.framework,
                self.mode,
                seed,
                executor_model=self.config.models["executor"].model,
            )
            for seed in self.config.run.seeds
        ]

    def is_complete(self) -> bool:
        return all(layout.is_run_complete(d) for d in self.seed_dirs())


def _deep_merge(
    dst: dict[str, Any],
    src: dict[str, Any],
    layer: str,
    provenance: dict[str, str],
    prefix: str = "",
) -> None:
    for key, value in src.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            if not isinstance(dst.get(key), dict):
                dst[key] = {}
            _deep_merge(dst[key], value, layer, provenance, prefix=f"{path}.")
        else:
            dst[key] = value
            provenance[path] = layer


def load_matrix(path: str | Path) -> MatrixConfig:
    return MatrixConfig(**(yaml.safe_load(Path(path).read_text()) or {}))


def expand(
    matrix: MatrixConfig, secrets_path: str | Path | None = None
) -> list[RunSpec]:
    base_raw = yaml.safe_load(Path(matrix.base).read_text()) or {}
    condition_defaults: dict[str, dict[str, Any]] = {}
    if matrix.condition_defaults:
        condition_defaults = (
            yaml.safe_load(Path(matrix.condition_defaults).read_text()) or {}
        ).get("conditions", {})

    unknown = {
        key
        for key in matrix.overrides
        if not key.startswith(("benchmark:", "condition:", "framework:", "cell:"))
    }
    if unknown:
        raise ValueError(f"unknown override scopes: {sorted(unknown)}")

    specs: list[RunSpec] = []
    for benchmark in matrix.benchmarks:
        for condition in matrix.conditions:
            for framework in matrix.frameworks:
                merged: dict[str, Any] = {}
                provenance: dict[str, str] = {}
                layers = [
                    ("base", base_raw),
                    (f"condition-defaults:{condition}", condition_defaults.get(condition, {})),
                    (f"benchmark:{benchmark}", matrix.overrides.get(f"benchmark:{benchmark}", {})),
                    (f"condition:{condition}", matrix.overrides.get(f"condition:{condition}", {})),
                    (f"framework:{framework}", matrix.overrides.get(f"framework:{framework}", {})),
                    (
                        f"cell:{benchmark}/{condition}",
                        matrix.overrides.get(f"cell:{benchmark}/{condition}", {}),
                    ),
                ]
                for layer_name, payload in layers:
                    _deep_merge(merged, dict(payload), layer_name, provenance)

                matrix_fields = {
                    "name": matrix.name,
                    "benchmark": benchmark,
                    "condition": condition,
                    "framework": framework,
                }
                for key, value in matrix_fields.items():
                    merged[key] = value
                    provenance[key] = "matrix"
                merged.setdefault("run", {})
                merged["run"]["seeds"] = list(matrix.seeds)
                provenance["run.seeds"] = "matrix"

                merged = merge_secrets(merged, secrets_path)
                specs.append(
                    RunSpec(
                        benchmark=benchmark,
                        condition=condition,
                        framework=framework,
                        mode=matrix.mode,
                        config=ExperimentConfig(**merged),
                        provenance=provenance,
                    )
                )
    return specs


def describe_cell(spec: RunSpec) -> str:
    """Human-readable dry-run block: effective non-base values with origins."""
    lines = [f"[{spec.cell}] mode={spec.mode} seeds={spec.config.run.seeds}"]
    axis_keys = {"name", "benchmark", "condition", "framework", "mode"}
    interesting = {
        path: layer
        for path, layer in sorted(spec.provenance.items())
        if layer != "base" and path.split(".", 1)[0] not in axis_keys
    }
    for path, layer in interesting.items():
        lines.append(f"    {path:<40} <- {layer}")
    return "\n".join(lines)
