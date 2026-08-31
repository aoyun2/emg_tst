"""Regression tests for record hygiene in the checkpoint analysis.

A non-finite row must drop out of every array together, including the participant
labels the cluster bootstrap resamples. Before the mask was applied to the
labels, one such row misaligned them against the data or raised IndexError.
"""

from __future__ import annotations

import json
import math
import unittest
import zipfile
from pathlib import Path

import numpy as np

from analysis.gait120_checkpoint_correlation import WindowRow, _checkpoint_result

ROOT = Path(__file__).resolve().parents[1]


def _row(query_id: str, subject: str, rmse: float, excess: float) -> WindowRow:
    return WindowRow(
        checkpoint="ck",
        query_id=query_id,
        subject=subject,
        panel_index=int(query_id[1:]),
        prediction_rmse_deg=rmse,
        excess_instability_auc=excess,
        reference_auc=0.5,
        fused_auc=0.5 + excess,
        match_knee_rmse_deg=4.0 + 0.1 * rmse,
        match_thigh_rms_deg=3.0 + 0.05 * rmse,
        recorded_steps=34.0,
        expected_steps=34.0,
    )


class NonFiniteRowTests(unittest.TestCase):
    def _rows(self, n: int = 24) -> list[WindowRow]:
        rng = np.random.default_rng(11)
        return [
            _row(f"w{i:03d}", f"S{i % 8:03d}", 5.0 + i * 0.4, float(rng.normal()))
            for i in range(n)
        ]

    def test_nonfinite_outcome_does_not_break_the_cluster_bootstrap(self) -> None:
        rows = self._rows()
        rows[7] = _row("w007", "S007", 7.8, float("nan"))
        result = _checkpoint_result(
            "ck", rows, np.random.default_rng(3), bootstrap_draws=200,
            permutation_draws=0,
        )
        self.assertEqual(result["n_windows"], len(rows) - 1)

    def test_nonfinite_predictor_is_dropped_too(self) -> None:
        rows = self._rows()
        rows[3] = _row("w003", "S003", float("nan"), 0.2)
        result = _checkpoint_result(
            "ck", rows, np.random.default_rng(3), bootstrap_draws=200,
            permutation_draws=0,
        )
        self.assertEqual(result["n_windows"], len(rows) - 1)

    def test_all_finite_rows_are_kept(self) -> None:
        rows = self._rows()
        result = _checkpoint_result(
            "ck", rows, np.random.default_rng(3), bootstrap_draws=200,
            permutation_draws=0,
        )
        self.assertEqual(result["n_windows"], len(rows))


class ArchiveJsonTests(unittest.TestCase):
    """Every shipped record must parse under a strict JSON reader."""

    ARCHIVE = ROOT / "manuscript" / "Additional_file_reproducibility.zip"

    def test_no_bare_nan_tokens_in_the_archive(self) -> None:
        if not self.ARCHIVE.is_file():
            self.skipTest("archive not built")

        def reject(constant: str) -> float:
            raise ValueError(f"non-JSON constant {constant!r}")

        with zipfile.ZipFile(self.ARCHIVE) as z:
            names = [n for n in z.namelist() if n.endswith(".json")]
            self.assertTrue(names, "archive ships no JSON records")
            for name in names:
                with self.subTest(record=name):
                    json.loads(z.read(name).decode("utf-8"), parse_constant=reject)


class FiniteOnlyTests(unittest.TestCase):
    def test_writer_helper_maps_non_finite_to_none(self) -> None:
        from mocap_phys_eval.run_gait120_residual_fusion import _finite_only

        cleaned = _finite_only(
            {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2.5}}
        )
        self.assertIsNone(cleaned["a"])
        self.assertIsNone(cleaned["b"][1])
        self.assertEqual(cleaned["c"]["d"], 2.5)
        self.assertTrue(math.isfinite(cleaned["b"][0]))
        json.dumps(cleaned, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
