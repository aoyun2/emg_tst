from __future__ import annotations

import unittest

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.run_gait120_kinematic_input_check as check
import emg_tst.run_gait120_residual_fusion as fusion


class KinematicInputSelectionTests(unittest.TestCase):
    """The check is only meaningful if the comparison arm truly excludes the knee."""

    def setUp(self) -> None:
        gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
        rng = np.random.default_rng(7)
        n = 200
        self.data = {
            "knee_flexion_deg": rng.normal(30.0, 12.0, size=n).astype(np.float32),
            "thigh_pitch_deg": rng.normal(10.0, 8.0, size=n).astype(np.float32),
            "thigh_quat_wxyz": np.tile(
                np.asarray([1.0, 0.0, 0.0, 0.0]), (n, 1)
            ).astype(np.float32),
        }

    def test_surrounding_body_arm_contains_no_knee_channel(self) -> None:
        body = check._kinematic_channels(self.data, "surrounding_body")
        knee = np.asarray(self.data["knee_flexion_deg"], dtype=np.float64).reshape(-1)
        for column in range(body.shape[1]):
            values = np.asarray(body[:, column], dtype=np.float64)
            if np.std(values) < 1.0e-9:
                continue
            self.assertLess(
                abs(float(np.corrcoef(values, knee)[0, 1])),
                0.999,
                f"column {column} reproduces the knee signal",
            )
        self.assertFalse(np.any(np.all(np.isclose(body, knee[:, None]), axis=0)))

    def test_knee_arm_is_exactly_the_knee(self) -> None:
        knee = check._kinematic_channels(self.data, "knee_history")
        self.assertEqual(knee.shape[1], 1)
        np.testing.assert_allclose(
            knee.reshape(-1), self.data["knee_flexion_deg"], rtol=1e-6
        )

    def test_surrounding_body_arm_has_four_channels(self) -> None:
        self.assertEqual(
            check._kinematic_channels(self.data, "surrounding_body").shape[1], 4
        )

    def test_unknown_input_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            check._kinematic_channels(self.data, "hip_only")

    def test_identity_quaternion_maps_to_zero_orientation(self) -> None:
        euler = check.thigh_euler_deg(np.asarray([[1.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(euler, np.zeros((1, 3)), atol=1e-9)

    def test_quarter_turn_about_x_is_ninety_degrees_of_roll(self) -> None:
        half = np.sqrt(0.5)
        euler = check.thigh_euler_deg(np.asarray([[half, half, 0.0, 0.0]]))
        self.assertAlmostEqual(float(euler[0, 0]), 90.0, places=6)
        self.assertAlmostEqual(float(euler[0, 1]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
