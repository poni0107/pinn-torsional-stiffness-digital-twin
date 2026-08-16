"""Neural state models and physically bounded stiffness models."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def init_linear(module: torch.nn.Module) -> None:
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_normal_(module.weight)
        torch.nn.init.zeros_(module.bias)


class DeltaNet(torch.nn.Module):
    """Legacy second-order relative-angle network."""

    def __init__(self):
        super().__init__()
        self.register_buffer("freq", torch.arange(1.0, 256.0, 2.0).reshape(1, -1))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(257, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )
        self.apply(init_linear)
        torch.nn.init.normal_(self.net[-1].weight, std=1e-3)

    def forward(self, tau):
        phase = 2 * math.pi * tau * self.freq
        features = torch.cat((tau, torch.sin(phase), torch.cos(phase)), dim=1)
        return tau * self.net(features)


class StiffnessNet(torch.nn.Module):
    """Legacy bounded free-profile stiffness network."""

    def __init__(self, config: Any):
        super().__init__()
        self.c = config
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )
        self.net.apply(init_linear)
        last = self.net[-1]
        torch.nn.init.zeros_(last.weight)
        fraction = (1 - config.kappa_min) / (config.kappa_max - config.kappa_min)
        torch.nn.init.constant_(last.bias, math.log(fraction / (1 - fraction)))

    def forward(self, tau):
        raw = self.c.kappa_min + (
            self.c.kappa_max - self.c.kappa_min
        ) * torch.sigmoid(self.net(tau))
        return 1.0 + tau * (raw - 1.0)


class ConstantStiffness(torch.nn.Module):
    """One bounded trainable scalar, independent of time."""

    def __init__(self, config: Any, initial: float = 288.75):
        super().__init__()
        self.c = config
        self.initial_k_const = float(initial)
        low = config.kappa_min * config.k0
        high = config.kappa_max * config.k0
        fraction = np.clip((initial - low) / (high - low), 1e-6, 1 - 1e-6)
        self.raw = torch.nn.Parameter(torch.tensor(math.log(fraction / (1 - fraction))))

    def value(self):
        low = self.c.kappa_min * self.c.k0
        high = self.c.kappa_max * self.c.k0
        return low + (high - low) * torch.sigmoid(self.raw)

    def forward(self, tau):
        return (self.value() / self.c.k0) * torch.ones_like(tau)


class WeakSigmoidStiffness(torch.nn.Module):
    """Four bounded parameters defining a monotonically decreasing k(t)."""

    def __init__(
        self,
        duration,
        k_high_init=330.0,
        k_low_init=270.0,
        center_fraction_init=0.50,
        width_fraction_init=0.10,
        k_min=210.0,
        k_max=367.5,
    ):
        super().__init__()
        self.duration = float(duration)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.initial_values = {
            "k_high": float(k_high_init),
            "k_low": float(k_low_init),
            "t_center": float(center_fraction_init * self.duration),
            "width": float(width_fraction_init * self.duration),
        }

        def logit(probability):
            probability = float(np.clip(probability, 1e-9, 1 - 1e-9))
            return math.log(probability / (1 - probability))

        low_fraction = (k_low_init - k_min) / (k_max - k_min)
        high_fraction = (k_high_init - k_low_init) / (k_max - k_low_init)
        width_fraction = (width_fraction_init - 0.005) / (0.25 - 0.005)
        self.raw_low = torch.nn.Parameter(torch.tensor(logit(low_fraction)))
        self.raw_high_conditional = torch.nn.Parameter(torch.tensor(logit(high_fraction)))
        self.raw_center = torch.nn.Parameter(torch.tensor(logit(center_fraction_init)))
        self.raw_width = torch.nn.Parameter(torch.tensor(logit(width_fraction)))
        values = self.physical_parameters()
        for name, target in self.initial_values.items():
            if not math.isclose(
                float(values[name].detach()), target, rel_tol=0, abs_tol=1e-9
            ):
                raise AssertionError(f"Weak sigmoid initialization failed for {name}")

    def physical_parameters(self):
        k_low = self.k_min + (self.k_max - self.k_min) * torch.sigmoid(self.raw_low)
        k_high = k_low + (self.k_max - k_low) * torch.sigmoid(
            self.raw_high_conditional
        )
        center = self.duration * torch.sigmoid(self.raw_center)
        width = self.duration * (
            0.005 + (0.25 - 0.005) * torch.sigmoid(self.raw_width)
        )
        return {"k_high": k_high, "k_low": k_low, "t_center": center, "width": width}

    def forward(self, time):
        parameters = self.physical_parameters()
        return parameters["k_low"] + (
            parameters["k_high"] - parameters["k_low"]
        ) * torch.sigmoid(
            (parameters["t_center"] - time) / parameters["width"]
        )


class RelativeStateNet(torch.nn.Module):
    """First-order state model with physical delta and v_delta outputs."""

    def __init__(self, delta_scale, v_scale):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.v_scale = float(v_scale)
        self.register_buffer("freq", torch.arange(1.0, 256.0, 2.0).reshape(1, -1))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(257, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 2),
        )
        self.apply(init_linear)
        torch.nn.init.normal_(self.net[-1].weight, std=1e-3)

    def forward(self, tau):
        phase = 2 * math.pi * tau * self.freq
        features = torch.cat((tau, torch.sin(phase), torch.cos(phase)), dim=1)
        raw = tau * self.net(features)
        return torch.cat(
            (raw[:, :1] * self.delta_scale, raw[:, 1:] * self.v_scale), dim=1
        )

