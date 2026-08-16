from __future__ import annotations

import unittest
from pathlib import Path

from pinn_torsional_twin.cli import REPOSITORY_ROOT, build_legacy_command
from pinn_torsional_twin.config import experiment_config_from_toml, load_toml


class ConfigCliTests(unittest.TestCase):
    def test_main_config_loads_and_builds_portable_command(self):
        config_path = REPOSITORY_ROOT / "configs" / "main_clean.toml"
        raw = load_toml(config_path)
        typed = experiment_config_from_toml(config_path, "input.mat")
        command = build_legacy_command(config_path, mat_file="input.mat", outdir="result")
        self.assertEqual(raw["measurements"], 1501)
        self.assertEqual(typed.collocation_points, 1501)
        self.assertIn("--stiffness-profile", command)
        self.assertIn("first_order", command)
        self.assertIn(str(Path("input.mat")), command)
        self.assertTrue(Path(command[1]).is_absolute())

    def test_every_shipped_config_is_typed_and_cli_translatable(self):
        for config_path in sorted((REPOSITORY_ROOT / "configs").glob("*.toml")):
            with self.subTest(config=config_path.name):
                typed = experiment_config_from_toml(config_path, "input.mat")
                command = build_legacy_command(config_path, mat_file="input.mat")
                self.assertEqual(typed.mat_file, "input.mat")
                self.assertEqual(command[1], str(REPOSITORY_ROOT / "src" / "mvm_pinn_jera1.py"))


if __name__ == "__main__":
    unittest.main()
