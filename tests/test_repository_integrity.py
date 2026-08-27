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

    def test_current_figure_generator_replaces_obsolete_version(self) -> None:
        current = ROOT / "analysis" / "gait120_submission_figures.py"
        obsolete = ROOT / "analysis" / "gait120_submission_figures_original_style.py"
        self.assertTrue(current.is_file(), f"Missing figure generator: {current}")
        self.assertFalse(obsolete.exists(), f"Obsolete figure generator remains: {obsolete}")

    def test_figure_dependencies_are_isolated_from_prediction_dependencies(self) -> None:
        base = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        figures = (ROOT / "requirements-figures.txt").read_text(encoding="utf-8")
        self.assertNotIn("opencv-python", base)
        self.assertIn("opencv-python", figures)
        self.assertIn("numpy>=2.2,<2.3", figures)


if __name__ == "__main__":
    unittest.main()
