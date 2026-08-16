"""Reference two-inertia motor-load dynamics."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def sigmoid_stiffness_reference(time: np.ndarray | float, duration: float, config: Any):
    """Known smooth 350-to-245 N m/rad profile used only to simulate/evaluate."""

    values = np.asarray(time, dtype=float)
    center = config.center_fraction * duration
    width = config.width_fraction * duration
    sigmoid = 1 / (1 + np.exp(-(values - center) / width))
    start = 1 / (1 + np.exp(center / width))
    finish = 1 / (1 + np.exp(-(duration - center) / width))
    progress = np.clip((sigmoid - start) / (finish - start), 0, 1)
    return config.k0 + (config.k_final - config.k0) * progress


def simulate_two_inertia(
    time: np.ndarray,
    torque: np.ndarray,
    config: Any,
    stiffness_profile: Callable[[np.ndarray | float], np.ndarray | float],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step_divisor: int = 2000,
) -> dict[str, np.ndarray]:
    """Generate synthetic encoder/state channels from measured torque input."""

    input_torque = interp1d(
        time,
        torque,
        kind="linear",
        bounds_error=False,
        fill_value=(torque[0], torque[-1]),
    )
    duration = float(time[-1])

    def right_hand_side(current_time: float, state: np.ndarray):
        theta_m, omega_m, theta_l, omega_l = state
        stiffness = float(stiffness_profile(current_time))
        shaft_torque = stiffness * (theta_m - theta_l) + config.bv * (
            omega_m - omega_l
        )
        return (
            omega_m,
            (float(input_torque(current_time)) - shaft_torque) / config.Jm,
            omega_l,
            shaft_torque / config.Jl,
        )

    solution = solve_ivp(
        right_hand_side,
        (0, duration),
        np.zeros(4),
        t_eval=time,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=duration / max_step_divisor,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return {
        "t": time,
        "Mem": torque,
        "theta_m": solution.y[0],
        "omega_m": solution.y[1],
        "theta_l": solution.y[2],
        "omega_l": solution.y[3],
        "delta": solution.y[0] - solution.y[2],
        "delta_dot": solution.y[1] - solution.y[3],
        "k_true": np.asarray(stiffness_profile(time), dtype=float),
    }


# Compatibility names used by the validated legacy script.
true_k = sigmoid_stiffness_reference
simulate = simulate_two_inertia

