"""Cross-platform command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .config import ExperimentConfig, load_toml
from .data import load_input
from .physics.two_inertia import simulate_two_inertia, sigmoid_stiffness_reference


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ENTRY_POINT = REPOSITORY_ROOT / "src" / "mvm_pinn_jera1.py"


def build_legacy_command(
    config_path: str | Path,
    *,
    mat_file: str | Path | None = None,
    outdir: str | Path | None = None,
) -> list[str]:
    """Translate a portable TOML config into the validated script's CLI."""

    raw = load_toml(config_path)
    dataset = Path(mat_file) if mat_file else REPOSITORY_ROOT / "data" / "jera1.mat"
    output = Path(outdir) if outdir else REPOSITORY_ROOT / raw.get("outdir", "outputs/run")
    command = [
        sys.executable,
        str(LEGACY_ENTRY_POINT),
        "--mat",
        str(dataset),
        "--outdir",
        str(output),
    ]
    scalar_options = {
        "pretrain_epochs": "--pretrain-epochs",
        "epochs": "--epochs",
        "finetune_epochs": "--finetune-epochs",
        "noise": "--noise",
        "measurements": "--measurements",
        "stiffness_profile": "--stiffness-profile",
        "relative_formulation": "--relative-formulation",
        "first_order_physics": "--first-order-physics",
        "sigmoid_robustness": "--sigmoid-robustness",
        "sigmoid_lr": "--sigmoid-lr",
        "noise_seed": "--noise-seed",
        "relative_state_checkpoint": "--relative-state-checkpoint",
        "free_baseline_csv": "--free-baseline-csv",
    }
    for key, option in scalar_options.items():
        if key in raw:
            value = raw[key]
            if key in {"relative_state_checkpoint", "free_baseline_csv"}:
                value = REPOSITORY_ROOT / value
            command.extend((option, str(value)))
    list_options = {
        "sigmoid_seeds": "--sigmoid-seeds",
        "online_strides": "--online-strides",
        "online_adam_steps": "--online-adam-steps",
    }
    for key, option in list_options.items():
        if key in raw:
            command.append(option)
            command.extend(str(value) for value in raw[key])
    for key, option in {
        "online_benchmark": "--online-benchmark",
        "allow_poor_pretrain": "--allow-poor-pretrain",
        "first_order_pretrain_only": "--first-order-pretrain-only",
    }.items():
        if raw.get(key, False):
            command.append(option)
    return command


def _check_data(path: Path) -> int:
    time, torque = load_input(path)
    report = {
        "path": str(path.resolve()),
        "samples": int(len(time)),
        "duration_seconds": float(time[-1]),
        "torque_min": float(np.min(torque)),
        "torque_max": float(np.max(torque)),
        "channels_used": ["t", "Mem"],
    }
    print(json.dumps(report, indent=2))
    return 0


def _generate_reference(path: Path, output: Path) -> int:
    time, torque = load_input(path)
    config = ExperimentConfig(mat_file=str(path))
    reference = simulate_two_inertia(
        time,
        torque,
        config,
        lambda current: sigmoid_stiffness_reference(current, float(time[-1]), config),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **reference)
    print(f"Synthetic reference saved to {output}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pinn-torsional-twin")
    subcommands = result.add_subparsers(dest="command", required=True)
    check = subcommands.add_parser("check-data", help="validate t and Mem in a MAT file")
    check.add_argument("--mat", type=Path, default=REPOSITORY_ROOT / "data" / "jera1.mat")
    generate = subcommands.add_parser("generate-reference", help="simulate synthetic encoder states")
    generate.add_argument("--mat", type=Path, default=REPOSITORY_ROOT / "data" / "jera1.mat")
    generate.add_argument("--output", type=Path, required=True)
    run = subcommands.add_parser("run", help="run one TOML-defined validated experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--mat", type=Path)
    run.add_argument("--outdir", type=Path)
    run.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.command == "check-data":
        return _check_data(arguments.mat)
    if arguments.command == "generate-reference":
        return _generate_reference(arguments.mat, arguments.output)
    command = build_legacy_command(
        arguments.config, mat_file=arguments.mat, outdir=arguments.outdir
    )
    if arguments.dry_run:
        print(subprocess.list2cmdline(command))
        return 0
    return subprocess.run(command, check=False).returncode

