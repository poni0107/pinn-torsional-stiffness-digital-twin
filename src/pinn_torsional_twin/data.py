"""MAT input and synthetic encoder-measurement preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def load_input(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load, sort, and deduplicate the measured time and torque channels.

    THref is deliberately ignored because it is not an input to the PINN loss.
    """

    data = loadmat(Path(path))
    if "t" not in data or "Mem" not in data:
        raise KeyError("jera1.mat must contain t and Mem")
    time = np.asarray(data["t"], dtype=float).squeeze()
    torque = np.asarray(data["Mem"], dtype=float).squeeze()
    valid = np.isfinite(time) & np.isfinite(torque)
    time, torque = time[valid], torque[valid]
    order = np.argsort(time)
    time, torque = time[order], torque[order]
    keep = np.r_[True, np.diff(time) > 0]
    time, torque = time[keep], torque[keep]
    if len(time) < 20:
        raise ValueError("At least 20 valid MAT samples are required")
    return time - time[0], torque


def sample_encoder_measurements(
    reference: dict[str, Any],
    config: Any,
    seed: int | None = None,
    relative_noise: bool = False,
    uniform_times: bool = False,
) -> dict[str, Any]:
    """Create sparse/noisy encoder channels from a simulated reference response."""

    count = min(max(2, config.measurements), len(reference["t"]))
    rng = np.random.default_rng(config.seed if seed is None else int(seed))
    if uniform_times:
        measurement_time = np.linspace(
            float(reference["t"][0]), float(reference["t"][-1]), count
        )
        indices = np.searchsorted(reference["t"], measurement_time).clip(
            0, len(reference["t"]) - 1
        )
    else:
        indices = np.unique(
            np.linspace(0, len(reference["t"]) - 1, count, dtype=int)
        )
        measurement_time = reference["t"][indices]

    result: dict[str, Any] = {
        "t": measurement_time,
        "idx": indices,
        "sampling_scheme": (
            "exact uniform time grid"
            if uniform_times
            else "selected source-grid indices"
        ),
    }
    if relative_noise:
        motor = np.interp(measurement_time, reference["t"], reference["omega_m"])
        load = np.interp(measurement_time, reference["t"], reference["omega_l"])
        relative = motor - load
        epsilon = rng.normal(
            0, config.noise * max(np.std(relative), 1e-12), relative.shape
        )
        result["omega_m"] = motor + 0.5 * epsilon
        result["omega_l"] = load - 0.5 * epsilon
        result["noise_model"] = (
            "seeded differential encoder noise: std(epsilon)=noise*std(omega_m-omega_l), "
            "split +epsilon/2 and -epsilon/2"
        )
    else:
        for name in ("omega_m", "omega_l"):
            clean = np.interp(measurement_time, reference["t"], reference[name])
            result[name] = clean + rng.normal(
                0, config.noise * max(np.std(clean), 1e-12), clean.shape
            )
        result["noise_model"] = (
            "independent per-encoder Gaussian noise scaled by each encoder standard deviation"
        )
    result["delta_dot"] = result["omega_m"] - result["omega_l"]
    return result


# Compatibility name used by the validated legacy script.
measurements = sample_encoder_measurements

