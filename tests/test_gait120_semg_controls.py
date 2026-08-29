from __future__ import annotations

import unittest

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.run_gait120_residual_fusion as fusion
import emg_tst.run_gait120_semg_controls as controls
from tests.test_gait120_training_path import _synthetic_examples


class SemgSurrogateTests(unittest.TestCase):
    """A surrogate is only a valid control if it keeps the signal but breaks the link."""

    def setUp(self) -> None:
        gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
        self.examples = _synthetic_examples(n_subjects=4, n_frames=24, seed=5)
        _, self.emg = fusion._features(self.examples)
        self.rng = np.random.default_rng(0)

    def test_identity_surrogate_is_the_untouched_signal(self) -> None:
        built = controls._BUILDERS["identity"](self.emg, self.examples, self.rng)
        np.testing.assert_allclose(built, self.emg)

    def test_circular_shift_preserves_every_value_within_a_block(self) -> None:
        """Rotation must change alignment only, never amplitude distribution."""
        built = controls._circular_shift(self.emg, self.examples, self.rng)
        for rows in controls._blocks(self.examples):
            np.testing.assert_allclose(
                np.sort(built[rows], axis=0), np.sort(self.emg[rows], axis=0)
            )
            self.assertFalse(np.allclose(built[rows], self.emg[rows]))

    def test_participant_swap_gives_every_participant_someone_elses_signal(self) -> None:
        built = controls._participant_swap(self.emg, self.examples, self.rng)
        subject = np.asarray(self.examples.subject_number).reshape(-1)
        for value in np.unique(subject).tolist():
            rows = np.flatnonzero(subject == value)
            self.assertFalse(np.allclose(built[rows], self.emg[rows]))
        # No sEMG is invented: every surrogate row is some real recorded row.
        original = {tuple(np.round(row, 9)) for row in self.emg}
        for row in built:
            self.assertIn(tuple(np.round(row, 9)), original)

    def test_phase_randomization_preserves_the_power_spectrum(self) -> None:
        built = controls._phase_randomized(self.emg, self.examples, self.rng)
        for rows in controls._blocks(self.examples):
            if rows.size < 8:
                continue
            np.testing.assert_allclose(
                np.abs(np.fft.rfft(built[rows], axis=0)),
                np.abs(np.fft.rfft(self.emg[rows], axis=0)),
                rtol=1.0e-6,
                atol=1.0e-8,
            )

    def test_surrogates_do_not_disturb_the_kinematic_stage(self) -> None:
        """Only sEMG changes, so any difference is attributable to sEMG content."""
        kinematic, _ = fusion._features(self.examples)
        before = kinematic.copy()
        for name in controls.SURROGATES:
            controls._BUILDERS[name](self.emg, self.examples, np.random.default_rng(0))
        np.testing.assert_array_equal(fusion._features(self.examples)[0], before)

    def test_surrogates_are_deterministic_for_a_fixed_seed(self) -> None:
        for name in controls.SURROGATES:
            first = controls._BUILDERS[name](
                self.emg, self.examples, np.random.default_rng(controls.SEED)
            )
            second = controls._BUILDERS[name](
                self.emg, self.examples, np.random.default_rng(controls.SEED)
            )
            np.testing.assert_allclose(first, second, err_msg=name)

    def test_summarize_effects_recovers_a_known_paired_effect(self) -> None:
        """The recorded-vs-surrogate contrast rests on this summary."""
        rng = np.random.default_rng(4)
        effects = rng.normal(0.30, 0.25, size=90)
        summary = controls._summarize_effects(effects)
        self.assertAlmostEqual(summary["mean_deg"], float(np.mean(effects)), places=12)
        low, _high = summary["bootstrap_95pct_ci_deg"]
        self.assertGreater(low, 0.0)
        self.assertLess(summary["two_sided_randomization_p"], 0.01)
        self.assertEqual(summary["participant_count"], 90)

    def test_summarize_effects_reports_a_null_as_null(self) -> None:
        # Centre the sample so the null holds by construction. A single random
        # draw is significant about one time in twenty, which would make this
        # test flaky rather than informative.
        rng = np.random.default_rng(5)
        effects = rng.normal(0.0, 0.25, size=90)
        effects = effects - float(np.mean(effects))
        summary = controls._summarize_effects(effects)
        low, high = summary["bootstrap_95pct_ci_deg"]
        self.assertLess(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertGreater(summary["two_sided_randomization_p"], 0.05)

    def test_noncontiguous_participant_rows_fail_closed(self) -> None:
        """Surrogates assume recorded order; a shuffled set must not pass silently."""
        shuffled = gait.ExampleSet(
            **{
                field: np.asarray(getattr(self.examples, field))[::-1]
                for field in gait.ExampleSet.__dataclass_fields__
            }
        )
        scrambled = gait.ExampleSet(
            **{
                **{
                    field: getattr(shuffled, field)
                    for field in gait.ExampleSet.__dataclass_fields__
                },
                "subject_number": np.tile(
                    np.asarray([1, 2], dtype=np.int16), len(shuffled) // 2
                ),
                "trial_index": np.ones(len(shuffled), dtype=np.int16),
            }
        )
        with self.assertRaises(RuntimeError):
            controls._blocks(scrambled)


if __name__ == "__main__":
    unittest.main()
