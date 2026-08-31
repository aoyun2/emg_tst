"""The documented reproduction commands must work on a fresh extraction.

The checkpoint analysis previously required a tree of per-rollout summary.json
files that the supplement does not ship, so the command in the README failed on
an extracted archive. These tests run it against the shipped table instead and
check that the published values come back.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "manuscript" / "Additional_file_reproducibility.zip"

PUBLISHED = {
    "breakpoint_rmse_deg": 12.9689,
    "overall_spearman_rho": 0.32747,
    "above_slope": 0.01304,
    "below_slope": -0.00707,
}


@unittest.skipUnless(ARCHIVE.is_file(), "archive not built")
class SupplementEntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_per_window_table_is_shipped(self) -> None:
        csv = self.root / "results" / "analysis" / "per_window_rollouts.csv"
        self.assertTrue(csv.is_file(), "supplement ships no per-window table")
        self.assertEqual(len(csv.read_text(encoding="utf-8").splitlines()) - 1, 1120)

    def test_checkpoint_analysis_runs_from_the_shipped_table(self) -> None:
        out = self.root / "out"
        result = subprocess.run(
            [sys.executable, "-m", "analysis.gait120_checkpoint_correlation",
             "--per-window-csv",
             str(self.root / "results" / "analysis" / "per_window_rollouts.csv"),
             "--out-dir", str(out)],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-800:])

        produced = json.loads(
            (out / "checkpoint_correlation.json").read_text(encoding="utf-8")
        )["accuracy_level"]
        self.assertAlmostEqual(
            produced["breakpoint_rmse_deg"], PUBLISHED["breakpoint_rmse_deg"], places=3
        )
        self.assertAlmostEqual(
            produced["overall_spearman_rho"], PUBLISHED["overall_spearman_rho"], places=4
        )
        self.assertAlmostEqual(
            produced["above_breakpoint"]["slope_per_degree"],
            PUBLISHED["above_slope"], places=5,
        )
        self.assertAlmostEqual(
            produced["below_breakpoint"]["slope_per_degree"],
            PUBLISHED["below_slope"], places=5,
        )

    def test_gain_sweep_is_shipped_and_matches_the_reported_count(self) -> None:
        sweep = json.loads(
            (self.root / "results" / "physics" / "gain_sweep.json").read_text(
                encoding="utf-8")
        )
        self.assertEqual(len(sweep), 5)
        self.assertIn({"kp": 400, "kd": 20, "force": 160},
                      [{k: r[k] for k in ("kp", "kd", "force")} for r in sweep])


if __name__ == "__main__":
    unittest.main()
