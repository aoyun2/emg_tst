from __future__ import annotations

import numpy as np
import unittest

from mocap_phys_eval.recording import _moving_target_pd_commands


class MovingTargetPdTests(unittest.TestCase):
    def test_constant_target_does_not_add_feedforward(self) -> None:
        target = np.asarray([0.4, 0.4, 0.4], dtype=np.float64)
        command, velocity = _moving_target_pd_commands(target, dt=0.03, kp=800.0, kd=40.0)
        np.testing.assert_allclose(velocity, 0.0)
        np.testing.assert_allclose(command, target)

    def test_ramp_command_reproduces_moving_target_pd_identity(self) -> None:
        target = np.asarray([0.2, 0.23, 0.29], dtype=np.float64)
        command, velocity = _moving_target_pd_commands(target, dt=0.03, kp=800.0, kd=40.0)
        expected_velocity = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
        np.testing.assert_allclose(velocity, expected_velocity)
        np.testing.assert_allclose(command, target + 0.05 * expected_velocity)

        q = np.asarray([0.21, 0.22, 0.25], dtype=np.float64)
        qdot = np.asarray([0.1, 0.7, 1.2], dtype=np.float64)
        servo_torque = 800.0 * (command - q) - 40.0 * qdot
        moving_target_pd_torque = 800.0 * (target - q) + 40.0 * (velocity - qdot)
        np.testing.assert_allclose(servo_torque, moving_target_pd_torque)

    def test_invalid_parameters_fail_closed(self) -> None:
        for dt, kp, kd in (
            (0.0, 800.0, 40.0),
            (0.03, 0.0, 40.0),
            (0.03, 800.0, -1.0),
        ):
            with self.subTest(dt=dt, kp=kp, kd=kd):
                with self.assertRaises(ValueError):
                    _moving_target_pd_commands(
                        np.asarray([0.1, 0.2]), dt=dt, kp=kp, kd=kd
                    )


if __name__ == "__main__":
    unittest.main()
