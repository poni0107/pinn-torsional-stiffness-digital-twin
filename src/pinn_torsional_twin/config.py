"""Experiment configuration and TOML loading."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import tomllib


@dataclass
class ExperimentConfig:
    """Numerical defaults retained from the validated legacy implementation."""

    mat_file: str
    Jm: float = 6.20e-4
    Jl: float = 2.20e-4
    bv: float = 4.00e-3
    k0: float = 350.0
    k_final: float = 245.0
    center_fraction: float = 0.55
    width_fraction: float = 0.055
    kappa_min: float = 0.60
    kappa_max: float = 1.05
    collocation_points: int = 1501
    measurements: int = 121
    noise: float = 0.003
    pretrain_epochs: int = 2000
    epochs: int = 6000
    finetune_epochs: int = 2000
    lr_delta: float = 1e-4
    lr_stiffness: float = 1e-5
    seed: int = 2026
    min_pretrain_r2: float = 0.95
    max_pretrain_relative_rmse: float = 0.10


def load_toml(path: str | Path) -> dict:
    """Load one experiment file without introducing a YAML dependency."""

    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    return document.get("experiment", document)


def experiment_config_from_toml(path: str | Path, mat_file: str | Path) -> ExperimentConfig:
    """Build the typed numerical configuration from a TOML experiment file."""

    raw = load_toml(path)
    accepted = {item.name for item in fields(ExperimentConfig)} - {"mat_file"}
    unknown = set(raw) - accepted - {
        "name",
        "stiffness_profile",
        "relative_formulation",
        "first_order_physics",
        "sigmoid_robustness",
        "outdir",
        "sigmoid_seeds",
        "sigmoid_lr",
        "online_strides",
        "online_adam_steps",
        "noise_seed",
        "online_benchmark",
        "relative_state_checkpoint",
        "free_baseline_csv",
        "allow_poor_pretrain",
        "first_order_pretrain_only",
    }
    if unknown:
        raise ValueError(f"Unknown experiment configuration keys: {sorted(unknown)}")
    values = {name: raw[name] for name in accepted if name in raw}
    return ExperimentConfig(mat_file=str(mat_file), **values)
