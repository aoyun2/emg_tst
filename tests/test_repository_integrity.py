from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryIntegrityTests(unittest.TestCase):
    def test_required_experiment_protocol_is_present(self) -> None:
        protocol = ROOT / "docs" / "EXPERIMENT_PROTOCOL.md"
        self.assertTrue(protocol.is_file(), f"Missing required protocol: {protocol}")
        text = protocol.read_text(encoding="utf-8")
        self.assertIn("later-trial, participant-calibrated design", text)
        self.assertIn("knee-angle RMSE ranked candidates", text)

    def test_readme_does_not_advertise_removed_submission_directory(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("final_submission/", readme)

    def test_figure_dependencies_are_isolated_from_prediction_dependencies(self) -> None:
        # The figure and manuscript tooling lives outside this repository, so
        # the only requirement here is that the prediction environment stays
        # free of plotting dependencies.
        base = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        figures = (ROOT / "requirements-figures.txt").read_text(encoding="utf-8")
        self.assertNotIn("matplotlib", base)
        self.assertIn("matplotlib", figures)
        self.assertIn("numpy>=2.2,<2.3", figures)


if __name__ == "__main__":
    unittest.main()
