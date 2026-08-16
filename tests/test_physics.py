from __future__ import annotations

import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from pinn_torsional_twin.config import ExperimentConfig
from pinn_torsional_twin.physics.two_inertia import (
    sigmoid_stiffness_reference,
    simulate_two_inertia,
)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.config = ExperimentConfig("unused.mat")

    def test_reference_profile_matches_legacy_formula(self):
        duration = 0.75
        time = np.linspace(0.0, duration, 1501)
        center = self.config.center_fraction * duration
        width = self.config.width_fraction * duration
        sigmoid = 1 / (1 + np.exp(-(time - center) / width))
        start = 1 / (1 + np.exp(center / width))
        finish = 1 / (1 + np.exp(-(duration - center) / width))
        progress = np.clip((sigmoid - start) / (finish - start), 0, 1)
        legacy = self.config.k0 + (self.config.k_final - self.config.k0) * progress
        modular = sigmoid_stiffness_reference(time, duration, self.config)
        np.testing.assert_array_equal(modular, legacy)
        self.assertEqual(float(modular[0]), 350.0)
        self.assertEqual(float(modular[-1]), 245.0)

    def test_simulator_matches_independent_legacy_rhs(self):
        time = np.linspace(0.0, 0.05, 51)
        torque = 0.3 + 0.1 * np.sin(2 * np.pi * 7 * time)
        stiffness = lambda value: np.zeros_like(np.asarray(value, float)) + 300.0
        modular = simulate_two_inertia(
            time, torque, self.config, stiffness, max_step_divisor=100
        )
        interpolation = interp1d(
            time, torque, kind="linear", bounds_error=False,
            fill_value=(torque[0], torque[-1])
        )

        def rhs(current_time, state):
            theta_m, omega_m, theta_l, omega_l = state
            shaft = 300.0 * (theta_m - theta_l) + self.config.bv * (omega_m - omega_l)
            return (
                omega_m,
                (float(interpolation(current_time)) - shaft) / self.config.Jm,
                omega_l,
                shaft / self.config.Jl,
            )

        legacy = solve_ivp(
            rhs, (0.0, float(time[-1])), np.zeros(4), t_eval=time,
            method="DOP853", rtol=1e-9, atol=1e-11,
            max_step=float(time[-1]) / 100,
        )
        np.testing.assert_array_equal(modular["theta_m"], legacy.y[0])
        np.testing.assert_array_equal(modular["omega_m"], legacy.y[1])
        np.testing.assert_array_equal(modular["theta_l"], legacy.y[2])
        np.testing.assert_array_equal(modular["omega_l"], legacy.y[3])


if __name__ == "__main__":
    unittest.main()

