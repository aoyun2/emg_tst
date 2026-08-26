from __future__ import annotations

import unittest

import numpy as np

from mocap_phys_eval.reference_centered import (
    CONTROL_HZ,
    exact_control_frames,
    inject_prediction_error,
)


class ReferenceCenteredTests(unittest.TestCase):
    def test_exact_control_frames_selects_every_third_recording(self) -> None:
        values = np.arange(100, dtype=np.float32)
        selected = exact_control_frames(values, control_hz=CONTROL_HZ)
        np.testing.assert_array_equal(selected, values[::3])
        self.assertEqual(selected.shape, (34,))

    def test_multicondition_time_axis_is_last(self) -> None:
        values = np.arange(200, dtype=np.float32).reshape(2, 100)
        selected = exact_control_frames(values, control_hz=CONTROL_HZ)
        np.testing.assert_array_equal(selected, values[:, ::3])

    def test_wrong_rate_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            exact_control_frames(np.arange(10), control_hz=50.0)

    def test_error_injection_preserves_exact_frame_rmse(self) -> None:
        reference = np.linspace(20.0, 50.0, 34, dtype=np.float32)
        measured = np.linspace(0.0, 20.0, 100, dtype=np.float32)
        prediction = measured + np.linspace(-4.0, 6.0, 100, dtype=np.float32)
        target, error = inject_prediction_error(
            reference,
            prediction,
            measured,
            knee_sign=-1.0,
            control_hz=CONTROL_HZ,
        )
        np.testing.assert_array_equal(error, (prediction - measured)[::3])
        np.testing.assert_allclose(target - reference, -error, rtol=0.0, atol=2.0e-6)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.square(target - reference)))),
            float(np.sqrt(np.mean(np.square(error)))),
            places=5,
        )

    def test_error_injection_does_not_clip_targets(self) -> None:
        target, _ = inject_prediction_error(
            np.zeros(34, dtype=np.float32),
            np.full(100, 200.0, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            knee_sign=1.0,
            control_hz=CONTROL_HZ,
        )
        self.assertTrue(np.all(target == 200.0))

    def test_error_injection_rejects_nonphysical_sign(self) -> None:
        with self.assertRaises(ValueError):
            inject_prediction_error(
                np.zeros(34),
                np.zeros(100),
                np.zeros(100),
                knee_sign=0.5,
                control_hz=CONTROL_HZ,
            )


if __name__ == "__main__":
    unittest.main()
