"""Generate synthetic ODE encoder responses driven by the measured Mem profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mvm_pinn_jera1 import Config, load_input, simulate, true_k
from utils import DEFAULT_MAT, OUTPUTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "reference_data.npz")
    args = parser.parse_args()
    t, mem = load_input(args.mat)
    config = Config(str(args.mat), noise=0.0, measurements=len(t))
    duration = float(t[-1])
    reference = simulate(t, mem, config, lambda x: true_k(x, duration, config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **reference)
    print(f"Saved synthetic ODE sensor/reference data to {args.output}")
    print("Only t and Mem originate from jera1.mat; encoder responses are simulated.")


if __name__ == "__main__":
    main()
