from __future__ import annotations

import unittest

import numpy as np
import torch

from pinn_torsional_twin.config import ExperimentConfig
from pinn_torsional_twin.models import (
    ConstantStiffness,
    DeltaNet,
    RelativeStateNet,
    StiffnessNet,
    WeakSigmoidStiffness,
)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(2026)
        self.config = ExperimentConfig("unused.mat")

    def test_known_initial_conditions_are_structural(self):
        zero = torch.zeros((1, 1))
        self.assertEqual(float(DeltaNet()(zero)), 0.0)
        relative = RelativeStateNet(0.02, 3.0)(zero)
        torch.testing.assert_close(relative, torch.zeros_like(relative), rtol=0, atol=0)
        self.assertEqual(float(StiffnessNet(self.config)(zero)), 1.0)

    def test_constant_stiffness_has_neutral_initialization_and_bounds(self):
        model = ConstantStiffness(self.config)
        self.assertAlmostEqual(float(model.value()), 288.75, places=12)
        with torch.no_grad():
            model.raw.fill_(-100.0)
            self.assertGreaterEqual(float(model.value()), 210.0)
            model.raw.fill_(100.0)
            self.assertLessEqual(float(model.value()), 367.5)

    def test_sigmoid_parameterization_is_bounded_and_monotone(self):
        model = WeakSigmoidStiffness(0.75)
        parameters = {name: float(value) for name, value in model.physical_parameters().items()}
        self.assertAlmostEqual(parameters["k_high"], 330.0, places=10)
        self.assertAlmostEqual(parameters["k_low"], 270.0, places=10)
        self.assertAlmostEqual(parameters["t_center"], 0.375, places=10)
        self.assertAlmostEqual(parameters["width"], 0.075, places=10)
        time = torch.linspace(0.0, 0.75, 501)
        stiffness = model(time).detach().numpy()
        self.assertTrue(np.all(np.diff(stiffness) <= 0))
        self.assertGreaterEqual(float(np.min(stiffness)), 210.0)
        self.assertLessEqual(float(np.max(stiffness)), 367.5)


if __name__ == "__main__":
    unittest.main()

