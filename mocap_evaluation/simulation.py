from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class SimulationResult:
    qpos: np.ndarray
    ctrl: np.ndarray
    rmse_knee_deg: float


LEG_XML = """
<mujoco model="sagittal_leg">
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="hip" pos="0 0 1.0">
      <joint name="thigh_hinge" type="hinge" axis="0 1 0" range="-120 120" damping="1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" density="600"/>
      <body name="knee" pos="0 0 -0.4">
        <joint name="knee_hinge" type="hinge" axis="0 1 0" range="-150 20" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.025" density="500"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_thigh" joint="thigh_hinge" kp="40" ctrlrange="-2.5 2.5"/>
    <position name="act_knee" joint="knee_hinge" kp="50" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>
"""


def _deg_to_rad(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(np.asarray(x, dtype=np.float32))


def run_leg_simulation(
    thigh_angle_deg: np.ndarray,
    knee_angle_deg: np.ndarray,
    *,
    sample_hz: float = 200.0,
) -> SimulationResult:
    model = mujoco.MjModel.from_xml_string(LEG_XML)
    data = mujoco.MjData(model)

    thigh = _deg_to_rad(thigh_angle_deg)
    knee = _deg_to_rad(knee_angle_deg)
    n = min(len(thigh), len(knee))

    steps_per_sample = max(1, int(round((1.0 / sample_hz) / model.opt.timestep)))
    qpos_hist = []
    ctrl_hist = []

    for i in range(n):
        # override with query batch angles
        data.ctrl[0] = float(thigh[i])
        data.ctrl[1] = float(knee[i])
        for _ in range(steps_per_sample):
            mujoco.mj_step(model, data)
        qpos_hist.append(np.copy(data.qpos[:2]))
        ctrl_hist.append(np.copy(data.ctrl[:2]))

    qpos = np.asarray(qpos_hist, dtype=np.float32)
    ctrl = np.asarray(ctrl_hist, dtype=np.float32)
    rmse = float(np.sqrt(np.mean((qpos[:, 1] - knee[: len(qpos)]) ** 2)))

    return SimulationResult(qpos=qpos, ctrl=ctrl, rmse_knee_deg=float(np.rad2deg(rmse)))
