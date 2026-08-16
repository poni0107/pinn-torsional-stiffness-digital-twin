"""First-order weak-form residual construction."""

from __future__ import annotations

from typing import Any

import torch


def build_constant_weak_terms(
    time: torch.Tensor,
    delta: torch.Tensor,
    relative_speed: torch.Tensor,
    torque: torch.Tensor,
    config: Any,
    *,
    window_length: int = 101,
    stride: int | None = None,
    delta_scale: float = 1.0,
) -> dict:
    """Build A, B and kinematic terms for constant-stiffness identification."""

    if stride is None:
        stride = max(1, window_length // 4)
    inverse_inertia_sum = 1 / config.Jm + 1 / config.Jl
    a_terms, b_terms, kinematic_terms = [], [], []
    for start in range(0, len(time) - window_length + 1, stride):
        stop = start + window_length
        current_time = time[start:stop]
        integral_delta = torch.trapz(delta[start:stop, 0], current_time)
        integral_speed = torch.trapz(relative_speed[start:stop, 0], current_time)
        integral_torque = torch.trapz(torque[start:stop, 0], current_time)
        a_terms.append(inverse_inertia_sum * integral_delta)
        b_terms.append(
            relative_speed[stop - 1, 0]
            - relative_speed[start, 0]
            + config.bv * inverse_inertia_sum * integral_speed
            - integral_torque / config.Jm
        )
        kinematic_terms.append(
            delta[stop - 1, 0] - delta[start, 0] - integral_speed
        )
    a = torch.stack(a_terms)
    b = torch.stack(b_terms)
    kinematic = torch.stack(kinematic_terms)
    return {
        "A": a,
        "B": b,
        "Rkin": kinematic,
        "window_length": window_length,
        "stride": stride,
        "window_count": len(a),
        "dynamic_scale": torch.sqrt(torch.mean(b**2)).clamp_min(1e-12),
        "kinematic_scale": max(float(delta_scale), 1e-12),
    }


def build_sigmoid_weak_terms(
    time: torch.Tensor,
    delta: torch.Tensor,
    relative_speed: torch.Tensor,
    torque: torch.Tensor,
    config: Any,
    *,
    duration: float,
    window_length: int = 101,
    stride: int = 25,
    delta_scale: float = 1.0,
) -> dict:
    """Build frozen windows for time-varying weak stiffness identification."""

    inverse_inertia_sum = 1 / config.Jm + 1 / config.Jl
    time_windows, delta_windows, b_terms, kinematic_terms = [], [], [], []
    for start in range(0, len(time) - window_length + 1, stride):
        stop = start + window_length
        current_time = time[start:stop]
        current_delta = delta[start:stop, 0]
        current_speed = relative_speed[start:stop, 0]
        integral_speed = torch.trapz(current_speed, current_time)
        integral_torque = torch.trapz(torque[start:stop, 0], current_time)
        time_windows.append(current_time)
        delta_windows.append(current_delta)
        b_terms.append(
            relative_speed[stop - 1, 0]
            - relative_speed[start, 0]
            + config.bv * inverse_inertia_sum * integral_speed
            - integral_torque / config.Jm
        )
        kinematic_terms.append(
            delta[stop - 1, 0] - delta[start, 0] - integral_speed
        )
    b = torch.stack(b_terms)
    kinematic = torch.stack(kinematic_terms)
    return {
        "t_windows": torch.stack(time_windows),
        "delta_windows": torch.stack(delta_windows),
        "B": b,
        "Rkin": kinematic,
        "invsum": inverse_inertia_sum,
        "duration": float(duration),
        "window_length": window_length,
        "stride": stride,
        "window_count": len(b),
        "weak_grid_points": int(len(time)),
        "weak_grid_step_seconds": float(time[1] - time[0]),
        "window_duration_seconds": float(time[window_length - 1] - time[0]),
        "stride_duration_seconds": float(stride * (time[1] - time[0])),
        "window_selection": "all",
        "dynamic_scale": torch.sqrt(torch.mean(b**2)).clamp_min(1e-12),
        "kinematic_scale": max(float(delta_scale), 1e-12),
    }


def weak_sigmoid_losses(model: torch.nn.Module, terms: dict):
    """Return dynamic, kinematic and total weak losses with autograd intact."""

    stiffness_integrals = torch.trapz(
        model(terms["t_windows"]) * terms["delta_windows"],
        terms["t_windows"],
        dim=1,
    )
    weak_dynamic = terms["B"] + terms["invsum"] * stiffness_integrals
    dynamic = torch.mean((weak_dynamic / terms["dynamic_scale"]) ** 2)
    kinematic = torch.mean((terms["Rkin"] / terms["kinematic_scale"]) ** 2)
    return dynamic, kinematic, dynamic + kinematic

