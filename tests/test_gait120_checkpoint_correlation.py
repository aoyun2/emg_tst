from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import gait120_checkpoint_correlation as cc


def _write_panel(
    root: Path,
    *,
    checkpoints: list[tuple[str, float]],
    n_windows: int = 60,
    coupling: float = 1.0,
    seed: int = 11,
) -> None:
    """Write simulation summaries where instability tracks prediction error.

    ``coupling`` scales how strongly excess instability follows RMSE, and each
    checkpoint's RMSE is centred on its own accuracy level.  Match-quality
    controls are correlated with both so that residualizing them actually
    matters, as it does on real panels.
    """
    rng = np.random.default_rng(seed)
    match_knee = rng.uniform(3.0, 12.0, size=n_windows)
    match_thigh = rng.uniform(2.0, 15.0, size=n_windows)
    window_effect = rng.normal(0.0, 1.0, size=n_windows)

    for label, level in checkpoints:
        rmse = level * np.exp(0.15 * window_effect) + 0.05 * match_knee
        excess = (
            coupling * 0.01 * rmse
            + 0.002 * match_knee
            + rng.normal(0.0, 0.01, size=n_windows)
        )
        for index in range(n_windows):
            out = root / label / f"q{index:03d}"
            out.mkdir(parents=True, exist_ok=True)
            reference_auc = 0.8 + 0.01 * float(window_effect[index])
            (out / "summary.json").write_text(
                json.dumps(
                    {
                        "query_id": f"q{index:03d}",
                        "panel_index": index,
                        "subject": f"S{31 + index % 60:03d}",
                        "checkpoint": label,
                        "prediction_rmse_deg": {
                            "fused": float(rmse[index]),
                            "no_emg": float(rmse[index]) + 0.3,
                        },
                        "match": {
                            "knee_rmse_deg": float(match_knee[index]),
                            "thigh_rms_deg": float(match_thigh[index]),
                            "length": 33,
                        },
                        "simulation": {
                            "reference": {"risk_auc": reference_auc},
                            "fused": {
                                "risk_auc": reference_auc + float(excess[index]),
                                "recorded_steps": 33 if index % 5 else 12,
                            },
                            "dt": 0.03,
                        },
                    }
                ),
                encoding="utf-8",
            )


def _fast(rows, rng, *, label: str = "c0"):
    """Same statistics with a resampling budget sized for a test, not a paper."""
    return cc._checkpoint_result(
        label, rows, rng, bootstrap_draws=400, permutation_draws=400
    )


class CheckpointCorrelationTests(unittest.TestCase):
    def test_recovers_a_planted_association_and_its_absence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_panel(root / "coupled", checkpoints=[("c0", 8.0)], coupling=1.0)
            _write_panel(root / "null", checkpoints=[("c0", 8.0)], coupling=0.0)

            rng = np.random.default_rng(0)
            coupled = _fast(cc._load_rows(root / "coupled"), rng)
            null = _fast(cc._load_rows(root / "null"), rng)

        self.assertGreater(coupled["partial_spearman_rho"], 0.3)
        self.assertTrue(coupled["interval_excludes_zero"])
        self.assertLess(abs(null["partial_spearman_rho"]), 0.3)

    def test_excess_instability_is_measured_against_the_paired_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_panel(root, checkpoints=[("c0", 8.0)], n_windows=5)
            rows = cc._load_rows(root)
        for row in rows:
            self.assertAlmostEqual(
                row.excess_instability_auc, row.fused_auc - row.reference_auc, places=12
            )

    def test_dropoff_is_located_where_the_association_disappears(self) -> None:
        """A ladder that is coupled above 4 degrees and null below should say so."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, level in enumerate([1.0, 2.0, 3.0, 4.0]):
                _write_panel(
                    root,
                    checkpoints=[(f"c{index:02d}", level)],
                    coupling=0.0,
                    seed=100 + index,
                )
            for index, level in enumerate([6.0, 8.0, 10.0, 12.0], start=4):
                _write_panel(
                    root,
                    checkpoints=[(f"c{index:02d}", level)],
                    coupling=6.0,
                    seed=100 + index,
                )

            rows = cc._load_rows(root)
            rng = np.random.default_rng(0)
            by_checkpoint: dict[str, list[cc.WindowRow]] = {}
            for row in rows:
                by_checkpoint.setdefault(row.checkpoint, []).append(row)
            per_checkpoint = [
                _fast(by_checkpoint[label], rng, label=label)
                for label in sorted(by_checkpoint)
            ]
            per_checkpoint.sort(key=lambda r: r["mean_prediction_rmse_deg"])
            dropoff = cc._dropoff(per_checkpoint)

        self.assertTrue(dropoff["estimated"])
        self.assertGreater(dropoff["breakpoint_mean_rmse_deg"], 4.0)
        self.assertLess(dropoff["breakpoint_mean_rmse_deg"], 7.0)
        self.assertGreater(
            dropoff["high_error_segment_mean_rho"],
            dropoff["low_error_segment_mean_rho"],
        )

    def test_spread_is_reported_so_a_null_can_be_interpreted(self) -> None:
        """A near-zero rho on a near-zero spread is not evidence of no effect."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_panel(root, checkpoints=[("c0", 4.7)])
            rng = np.random.default_rng(0)
            result = _fast(cc._load_rows(root), rng)
        for key in (
            "prediction_rmse_sd_deg",
            "prediction_rmse_iqr_deg",
            "excess_instability_sd",
        ):
            self.assertIn(key, result)
            self.assertTrue(np.isfinite(result[key]))

    def test_early_terminated_rollouts_are_counted_not_hidden(self) -> None:
        """A fallen rollout integrates a shorter trace, shrinking its excess area.

        Both conditions stop together so the paired difference stays like-for-like,
        but the magnitude is attenuated for exactly the windows that destabilised.
        The analysis must surface how often that happened.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_panel(root, checkpoints=[("c0", 8.0)])
            rng = np.random.default_rng(0)
            result = _fast(cc._load_rows(root), rng)
        self.assertGreater(result["truncated_rollouts"], 0)
        self.assertGreater(result["truncated_fraction"], 0.0)
        self.assertLess(result["mean_recorded_steps"], result["mean_expected_steps"])
        self.assertTrue(np.isfinite(result["partial_spearman_rho_per_step"]))

    def test_missing_panel_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                cc._load_rows(Path(tmp))


if __name__ == "__main__":
    unittest.main()
