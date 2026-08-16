from __future__ import annotations

import importlib.util
import unittest

import torch

from pinn_torsional_twin.cli import LEGACY_ENTRY_POINT
from pinn_torsional_twin.config import ExperimentConfig
from pinn_torsional_twin.models import DeltaNet, RelativeStateNet, WeakSigmoidStiffness


class LegacyEntrypointTests(unittest.TestCase):
    @staticmethod
    def load_entrypoint():
        specification = importlib.util.spec_from_file_location(
            "mvm_pinn_phase1_compatibility", LEGACY_ENTRY_POINT
        )
        module = importlib.util.module_from_spec(specification)
        assert specification.loader is not None
        specification.loader.exec_module(module)
        return module

    def test_entrypoint_uses_modular_runtime_definitions(self):
        module = self.load_entrypoint()
        self.assertIs(module.Config, ExperimentConfig)
        self.assertIs(module.RelativeStateNet, RelativeStateNet)
        self.assertIs(module.WeakSigmoidStiffness, WeakSigmoidStiffness)

    def test_modular_networks_are_bitwise_equal_to_compatibility_definitions(self):
        module = self.load_entrypoint()
        torch.set_default_dtype(torch.float64)
        tau = torch.linspace(0.0, 1.0, 17).reshape(-1, 1)

        torch.manual_seed(2026)
        legacy_delta = module._LegacyDeltaNet()
        torch.manual_seed(2026)
        modular_delta = DeltaNet()
        torch.testing.assert_close(
            modular_delta(tau), legacy_delta(tau), rtol=0, atol=0
        )

        torch.manual_seed(3030)
        legacy_relative = module._LegacyRelativeStateNet(0.02, 3.0)
        torch.manual_seed(3030)
        modular_relative = RelativeStateNet(0.02, 3.0)
        torch.testing.assert_close(
            modular_relative(tau), legacy_relative(tau), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
