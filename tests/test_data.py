from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from pinn_torsional_twin.config import ExperimentConfig
from pinn_torsional_twin.data import load_input, sample_encoder_measurements


class DataTests(unittest.TestCase):
    def test_load_input_sorts_deduplicates_and_ignores_thref(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.mat"
            time = np.r_[np.linspace(0.0, 0.75, 25), 0.25][::-1]
            torque = (2.0 * time + 1.0)
            savemat(path, {"t": time, "Mem": torque, "THref": np.full_like(time, 999.0)})
            loaded_time, loaded_torque = load_input(path)
        self.assertEqual(len(loaded_time), 25)
        self.assertAlmostEqual(float(loaded_time[0]), 0.0)
        self.assertTrue(np.all(np.diff(loaded_time) > 0))
        np.testing.assert_allclose(loaded_torque, 2.0 * loaded_time + 1.0)

    def test_measurement_sampling_is_seed_deterministic(self):
        time = np.linspace(0.0, 0.75, 41)
        reference = {
            "t": time,
            "omega_m": np.sin(time),
            "omega_l": np.cos(time),
        }
        config = ExperimentConfig("unused.mat", measurements=17, noise=0.003)
        first = sample_encoder_measurements(reference, config, seed=7, uniform_times=True)
        second = sample_encoder_measurements(reference, config, seed=7, uniform_times=True)
        np.testing.assert_array_equal(first["omega_m"], second["omega_m"])
        np.testing.assert_array_equal(first["omega_l"], second["omega_l"])
        self.assertEqual(len(first["t"]), 17)


if __name__ == "__main__":
    unittest.main()

