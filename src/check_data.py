"""Validate the local jera1.mat interface and print sampling information."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils import DEFAULT_MAT, load_jera1, sampling_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=DEFAULT_MAT)
    args = parser.parse_args()
    data = load_jera1(args.mat)
    report = sampling_metrics(data["t"])
    report["variables"] = {name: list(values.shape) for name, values in data.items()}
    report["THref_used_by_PINN"] = False
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
