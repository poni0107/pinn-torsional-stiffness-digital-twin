from __future__ import annotations

import unittest

import torch

from pinn_torsional_twin.config import ExperimentConfig
from pinn_torsional_twin.models import WeakSigmoidStiffness
from pinn_torsional_twin.physics.weak import (
    build_constant_weak_terms,
    build_sigmoid_weak_terms,
    weak_sigmoid_losses,
)


class WeakResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.config = ExperimentConfig("unused.mat")
        self.time = torch.linspace(0.0, 0.75, 301)
        self.delta = (0.01 * torch.sin(17 * self.time)).reshape(-1, 1)
        self.speed = (0.17 * torch.cos(17 * self.time)).reshape(-1, 1)
        self.torque = (0.5 + 0.1 * torch.sin(5 * self.time)).reshape(-1, 1)

    def test_constant_terms_match_inline_legacy_equations(self):
        length, stride = 51, 12
        terms = build_constant_weak_terms(
            self.time, self.delta, self.speed, self.torque, self.config,
            window_length=length, stride=stride, delta_scale=0.01
        )
        inverse_sum = 1 / self.config.Jm + 1 / self.config.Jl
        expected_a, expected_b, expected_kinematic = [], [], []
        for start in range(0, len(self.time) - length + 1, stride):
            stop = start + length
            current_time = self.time[start:stop]
            integral_delta = torch.trapz(self.delta[start:stop, 0], current_time)
            integral_speed = torch.trapz(self.speed[start:stop, 0], current_time)
            integral_torque = torch.trapz(self.torque[start:stop, 0], current_time)
            expected_a.append(inverse_sum * integral_delta)
            expected_b.append(
                self.speed[stop - 1, 0] - self.speed[start, 0]
                + self.config.bv * inverse_sum * integral_speed
                - integral_torque / self.config.Jm
            )
            expected_kinematic.append(
                self.delta[stop - 1, 0] - self.delta[start, 0] - integral_speed
            )
        torch.testing.assert_close(terms["A"], torch.stack(expected_a), rtol=0, atol=0)
        torch.testing.assert_close(terms["B"], torch.stack(expected_b), rtol=0, atol=0)
        torch.testing.assert_close(
            terms["Rkin"], torch.stack(expected_kinematic), rtol=0, atol=0
        )

    def test_sigmoid_loss_matches_inline_legacy_equations_and_gradient(self):
        terms = build_sigmoid_weak_terms(
            self.time, self.delta, self.speed, self.torque, self.config,
            duration=0.75, window_length=51, stride=12, delta_scale=0.01
        )
        model = WeakSigmoidStiffness(0.75)
        dynamic, kinematic, total = weak_sigmoid_losses(model, terms)
        integrals = torch.trapz(
            model(terms["t_windows"]) * terms["delta_windows"],
            terms["t_windows"], dim=1
        )
        residual = terms["B"] + terms["invsum"] * integrals
        expected_dynamic = torch.mean((residual / terms["dynamic_scale"]) ** 2)
        expected_kinematic = torch.mean((terms["Rkin"] / terms["kinematic_scale"]) ** 2)
        torch.testing.assert_close(dynamic, expected_dynamic, rtol=0, atol=0)
        torch.testing.assert_close(kinematic, expected_kinematic, rtol=0, atol=0)
        torch.testing.assert_close(total, expected_dynamic + expected_kinematic, rtol=0, atol=0)
        total.backward()
        self.assertIsNotNone(model.raw_low.grad)
        self.assertTrue(torch.isfinite(model.raw_low.grad))


if __name__ == "__main__":
    unittest.main()

