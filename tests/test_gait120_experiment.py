import tempfile
import unittest
from pathlib import Path

import numpy as np

import emg_tst.gait120_experiment as gait


class Gait120ExperimentTests(unittest.TestCase):
    def test_example_builder_preserves_native_frames_and_trial_boundaries(self) -> None:
        frames_per_trial = 75
        trials = np.repeat(np.arange(1, 6, dtype=np.int16), frames_per_trial)
        frames = np.tile(np.arange(frames_per_trial, dtype=np.int16), 5)
        knee = np.concatenate(
            [np.arange(frames_per_trial, dtype=np.float32) + 100 * trial for trial in range(5)]
        )
        emg = np.column_stack(
            [knee + channel for channel in range(gait.N_EMG)]
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            np.savez_compressed(
                cache_dir / "S001.npz",
                cache_version=np.asarray(gait.CACHE_VERSION),
                motion_hz=np.asarray(100),
                subject=np.asarray("S001"),
                emg_frame_native_value=emg,
                knee_flexion_deg=knee,
                trial_index=trials,
                frame_index=frames,
            )
            original_frames = gait.INPUT_FRAMES
            try:
                gait.INPUT_FRAMES = 60
                train, validation, test, _ = gait._build_examples(cache_dir, [1])
            finally:
                gait.INPUT_FRAMES = original_frames

        self.assertEqual(len(train), 18)
        self.assertEqual(len(validation), 6)
        self.assertEqual(len(test), 6)
        np.testing.assert_array_equal(test.input_end_frame, np.arange(59, 65))
        np.testing.assert_array_equal(test.target_frame, np.arange(69, 75))
        self.assertTrue(np.all(test.trial_index == 5))
        self.assertTrue(np.all(test.subject_number == 1))

    def test_participant_metrics_are_participant_level(self) -> None:
        examples = gait.ExampleSet(
            x=np.zeros((4, 1, gait.N_EMG + 1), dtype=np.float32),
            y_standardized=np.zeros(4, dtype=np.float32),
            y_deg=np.asarray([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
            target_mean_deg=np.zeros(4, dtype=np.float32),
            target_std_deg=np.ones(4, dtype=np.float32),
            subject_number=np.asarray([1, 1, 2, 2], dtype=np.int16),
            trial_index=np.full(4, 5, dtype=np.int16),
            input_end_frame=np.arange(4, dtype=np.int16),
            target_frame=np.arange(10, 14, dtype=np.int16),
        )
        result = gait._participant_metrics(
            examples, np.asarray([1.0, -1.0, 12.0, 8.0], dtype=np.float32)
        )

        self.assertEqual([row["subject"] for row in result["participants"]], ["S001", "S002"])
        self.assertAlmostEqual(result["participants"][0]["rmse_deg"], 1.0)
        self.assertAlmostEqual(result["participants"][1]["rmse_deg"], 2.0)
        self.assertAlmostEqual(result["mean_participant_rmse_deg"], 1.5)


if __name__ == "__main__":
    unittest.main()
