"""Prosthetic gait simulation/evaluation.

Supports:
- MuJoCo physics backend
- Deterministic kinematic evaluator (dependency fallback)

Public APIs accept included-angle convention (degrees):
- 180 = straight / neutral
- smaller values = increased flexion magnitude

Internally we convert to flexion convention for the physics backend.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from tqdm import tqdm

try:
    import mujoco
    import mujoco.viewer
    _MUJOCO_AVAILABLE = True
except Exception:  # pragma: no cover
    _MUJOCO_AVAILABLE = False


SIM_FPS_DEFAULT = 200.0
HUMANOID_INIT_POS_Z = 1.05
FALL_HEIGHT_THRESHOLD = 0.55


def _rad(deg: float) -> float:
    return math.radians(float(deg))


def _included_to_flexion(angles_deg: np.ndarray) -> np.ndarray:
    """Convert included-angle convention (180=straight) to flexion degrees."""
    return 180.0 - np.asarray(angles_deg, dtype=np.float64)


def _safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def _safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0


def _contact_events(frames: List[int], min_gap: int = 20) -> int:
    if not frames:
        return 0
    out, prev = 1, frames[0]
    for fr in frames[1:]:
        if fr - prev > min_gap:
            out += 1
        prev = fr
    return out


def _gait_symmetry(right_frames: List[int], left_frames: List[int]) -> float:
    def intervals(frames: List[int]) -> np.ndarray:
        if len(frames) < 2:
            return np.array([], dtype=np.float64)
        u = sorted(set(frames))
        vals = []
        prev = u[0]
        for f in u[1:]:
            if f - prev > 5:
                vals.append(f - prev)
            prev = f
        return np.asarray(vals, dtype=np.float64)

    r = intervals(right_frames)
    l = intervals(left_frames)
    if r.size == 0 or l.size == 0:
        return 0.5
    mr, ml = float(np.mean(r)), float(np.mean(l))
    den = mr + ml
    return 0.0 if den < 1e-9 else float(abs(mr - ml) / den)


@dataclass
class EvalMetrics:
    com_height: List[float]
    pred_knee: List[float]
    ref_knee: List[float]
    right_contact_frames: List[int]
    left_contact_frames: List[int]
    fall_detected: bool
    fall_frame: int

    @classmethod
    def empty(cls) -> "EvalMetrics":
        return cls([], [], [], [], [], False, -1)

    def to_dict(self) -> dict:
        pred = np.asarray(self.pred_knee, dtype=np.float64)
        ref = np.asarray(self.ref_knee, dtype=np.float64)
        err = pred - ref
        com = np.asarray(self.com_height, dtype=np.float64)

        rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0
        mae = _safe_mean(np.abs(err))
        com_mean = _safe_mean(com)
        com_std = _safe_std(com)

        sr = _contact_events(self.right_contact_frames)
        sl = _contact_events(self.left_contact_frames)
        sym = _gait_symmetry(self.right_contact_frames, self.left_contact_frames)

        stability = 1.0 - min(com_std / 0.25, 1.0) * 0.55 - sym * 0.25 - (0.35 if self.fall_detected else 0.0)
        stability = max(0.0, min(1.0, stability))
        return {
            "com_height_mean": com_mean,
            "com_height_std": com_std,
            "fall_detected": bool(self.fall_detected),
            "fall_frame": int(self.fall_frame),
            "knee_rmse_deg": rmse,
            "knee_mae_deg": mae,
            "step_count": int(sr + sl),
            "gait_symmetry": float(sym),
            "stability_score": float(stability),
            # Raw time series for visualization
            "com_height_series": com.tolist(),
            "pred_knee_series": pred.tolist(),
            "ref_knee_series": ref.tolist(),
            "right_contact_frames": list(self.right_contact_frames),
            "left_contact_frames": list(self.left_contact_frames),
        }


def run_kinematic_evaluation(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
    sample_thigh_right: Optional[np.ndarray] = None,
) -> dict:
    ref = _included_to_flexion(np.asarray(mocap_segment["knee_right"], dtype=np.float64))
    pred = _included_to_flexion(np.asarray(predicted_knee, dtype=np.float64))
    T = min(len(ref), len(pred))
    if sample_thigh_right is not None:
        T = min(T, len(sample_thigh_right))
    if T <= 0:
        return {
            "com_height_mean": HUMANOID_INIT_POS_Z,
            "com_height_std": 0.0,
            "fall_detected": False,
            "fall_frame": -1,
            "knee_rmse_deg": 0.0,
            "knee_mae_deg": 0.0,
            "step_count": 0,
            "gait_symmetry": 0.0,
            "stability_score": 0.0,
            "mode": "kinematic",
        }

    ref = ref[:T]
    pred = pred[:T]
    if sample_thigh_right is not None:
        hip = _included_to_flexion(np.asarray(sample_thigh_right, dtype=np.float64)[:T])
    else:
        hip = _included_to_flexion(np.asarray(mocap_segment.get("hip_right", np.full(T, 180.0)), dtype=np.float64)[:T])

    dev = np.zeros(T, dtype=np.float64)
    L1 = L2 = 0.45
    for i in range(T):
        h = _rad(hip[i])
        kr = _rad(ref[i])
        kp = _rad(pred[i])
        xr = L1 * math.sin(h) + L2 * math.sin(h - kr)
        zr = -L1 * math.cos(h) - L2 * math.cos(h - kr)
        xp = L1 * math.sin(h) + L2 * math.sin(h - kp)
        zp = -L1 * math.cos(h) - L2 * math.cos(h - kp)
        dev[i] = math.hypot(xp - xr, zp - zr)

    fall = bool(float(np.max(dev)) > 0.18)
    return {
        "com_height_mean": float(HUMANOID_INIT_POS_Z - np.mean(0.5 * dev)),
        "com_height_std": float(np.std(0.5 * dev)),
        "fall_detected": fall,
        "fall_frame": int(np.argmax(dev)) if fall else -1,
        "knee_rmse_deg": float(np.sqrt(np.mean((pred - ref) ** 2))),
        "knee_mae_deg": float(np.mean(np.abs(pred - ref))),
        "step_count": -1,
        "gait_symmetry": 0.0,
        "stability_score": float(max(0.0, 1.0 - np.max(dev) / 0.30) * (0.6 if fall else 1.0)),
        "mode": "kinematic",
    }


_MJCF = """
<mujoco model="prosthetic_eval_humanoid">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <compiler angle="degree"/>
  <worldbody>
    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.8 0.9 0.8 1"/>
    <light pos="0 0 4"/>
    <body name="torso" pos="0 0 1.0">
      <freejoint/>
      <geom type="capsule" fromto="0 0 -0.2 0 0 0.2" size="0.09" rgba="0.6 0.6 0.65 1"/>
      <body name="right_thigh" pos="0 -0.10 -0.10">
        <joint name="right_hip" type="hinge" axis="0 1 0" range="-70 70" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05"/>
        <body name="right_shank" pos="0 0 -0.42">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="0 130" damping="2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.045" rgba="1 0.5 0.1 1"/>
          <body name="right_foot" pos="0 0 -0.42">
            <joint name="right_ankle" type="hinge" axis="0 1 0" range="-40 50" damping="2"/>
            <geom name="right_foot_geom" type="box" size="0.11 0.05 0.03" pos="0.07 0 -0.02" rgba="1 0.5 0.1 1"/>
          </body>
        </body>
      </body>
      <body name="left_thigh" pos="0 0.10 -0.10">
        <joint name="left_hip" type="hinge" axis="0 1 0" range="-70 70" damping="2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05"/>
        <body name="left_shank" pos="0 0 -0.42">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="0 130" damping="2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.045"/>
          <body name="left_foot" pos="0 0 -0.42">
            <joint name="left_ankle" type="hinge" axis="0 1 0" range="-40 50" damping="2"/>
            <geom name="left_foot_geom" type="box" size="0.11 0.05 0.03" pos="0.07 0 -0.02"/>
          </body>
        </body>
      </body>
      <body name="right_arm" pos="0 -0.22 0.15">
        <joint name="right_shoulder" type="hinge" axis="0 1 0" range="-90 90" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03"/>
      </body>
      <body name="left_arm" pos="0 0.22 0.15">
        <joint name="left_shoulder" type="hinge" axis="0 1 0" range="-90 90" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="right_hip" kp="150"/>
    <position joint="left_hip" kp="150"/>
    <position joint="right_knee" kp="180"/>
    <position joint="left_knee" kp="180"/>
    <position joint="right_ankle" kp="120"/>
    <position joint="left_ankle" kp="120"/>
    <position joint="right_shoulder" kp="80"/>
    <position joint="left_shoulder" kp="80"/>
  </actuator>
</mujoco>
"""


class _MuJoCoRunner:
    def __init__(self, use_gui: bool, fps: float):
        self.use_gui = use_gui
        self.fps = float(fps)
        self.dt = 1.0 / max(self.fps, 1.0)

    def run(
        self,
        mocap_segment: dict,
        predicted_knee: np.ndarray,
        fall_threshold: float,
        sample_thigh_right: Optional[np.ndarray] = None,
    ) -> dict:
        model = mujoco.MjModel.from_xml_string(_MJCF)
        model.opt.timestep = self.dt
        data = mujoco.MjData(model)

        T = int(min(len(predicted_knee), len(mocap_segment["knee_right"])))
        if sample_thigh_right is not None:
            T = int(min(T, len(sample_thigh_right)))
        if T <= 0:
            out = run_kinematic_evaluation(mocap_segment, predicted_knee, sample_thigh_right=sample_thigh_right)
            out["mode"] = "mujoco_physics_empty"
            return out

        pred = _included_to_flexion(np.asarray(predicted_knee, dtype=np.float64)[:T])
        ref = _included_to_flexion(np.asarray(mocap_segment["knee_right"], dtype=np.float64)[:T])
        if sample_thigh_right is not None:
            hip_r = _included_to_flexion(np.asarray(sample_thigh_right, dtype=np.float64)[:T])
        else:
            hip_r = _included_to_flexion(np.asarray(mocap_segment.get("hip_right", np.full(T, 180.0)), dtype=np.float64)[:T])
        hip_l = _included_to_flexion(np.asarray(mocap_segment.get("hip_left", np.full(T, 180.0)), dtype=np.float64)[:T])
        knee_l = _included_to_flexion(np.asarray(mocap_segment.get("knee_left", np.full(T, 180.0)), dtype=np.float64)[:T])
        ankle_r = _included_to_flexion(np.asarray(mocap_segment.get("ankle_right", np.full(T, 180.0)), dtype=np.float64)[:T])
        ankle_l = _included_to_flexion(np.asarray(mocap_segment.get("ankle_left", np.full(T, 180.0)), dtype=np.float64)[:T])

        ridx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
        lidx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")

        metrics = EvalMetrics.empty()

        def _step_loop(viewer_obj=None):
            for t in tqdm(range(T), desc="Simulating", unit="step", leave=False):
                # Actuator order matches MJCF definition.
                data.ctrl[0] = _rad(hip_r[t])
                data.ctrl[1] = _rad(hip_l[t])
                data.ctrl[2] = _rad(pred[t])
                data.ctrl[3] = _rad(knee_l[t])
                data.ctrl[4] = _rad(ankle_r[t])
                data.ctrl[5] = _rad(ankle_l[t])
                # Simple anti-phase arm swing from hip motion.
                data.ctrl[6] = _rad(-0.6 * hip_r[t])
                data.ctrl[7] = _rad(-0.6 * hip_l[t])

                mujoco.mj_step(model, data)
                if viewer_obj is not None:
                    viewer_obj.sync()
                    time.sleep(self.dt)

                com_h = float(data.subtree_com[0, 2])
                rc = False
                lc = False
                for c in range(data.ncon):
                    con = data.contact[c]
                    g1, g2 = int(con.geom1), int(con.geom2)
                    if ridx in (g1, g2):
                        rc = True
                    if lidx in (g1, g2):
                        lc = True

                metrics.com_height.append(com_h)
                metrics.pred_knee.append(float(pred[t]))
                metrics.ref_knee.append(float(ref[t]))
                if rc:
                    metrics.right_contact_frames.append(t)
                if lc:
                    metrics.left_contact_frames.append(t)
                if (not metrics.fall_detected) and com_h < fall_threshold:
                    metrics.fall_detected = True
                    metrics.fall_frame = t

        if self.use_gui:
            with mujoco.viewer.launch_passive(model, data) as viewer_obj:
                _step_loop(viewer_obj)
                # Keep the viewer window open so the user can inspect the
                # final pose and replay camera angles.  The window stays
                # until the user closes it manually.
                print("[MuJoCo] Simulation complete — close the viewer window to continue.")
                while viewer_obj.is_running():
                    time.sleep(0.05)
        else:
            _step_loop(None)

        out = metrics.to_dict()
        out["mode"] = "mujoco_physics" + ("+gui" if self.use_gui else "")
        return out


def simulate_prosthetic_walking(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
    use_gui: bool = False,
    fps: float = SIM_FPS_DEFAULT,
    sample_thigh_right: Optional[np.ndarray] = None,
) -> dict:
    """Run prosthetic gait simulation via MuJoCo physics.

    Parameters
    ----------
    sample_thigh_right:
        If provided, the right hip/thigh actuator is driven by this signal
        (in included-angle degrees, same convention as all other angles)
        instead of the matched mocap segment's hip_right channel.  Use this
        to keep the right-leg inputs (thigh + knee) anchored to the sample
        being evaluated rather than the mocap reference.
    """
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError(
            "MuJoCo is required but not installed.\n"
            "Install it with:  pip install mujoco"
        )
    return _MuJoCoRunner(use_gui=use_gui, fps=fps).run(
        mocap_segment, predicted_knee, FALL_HEIGHT_THRESHOLD,
        sample_thigh_right=sample_thigh_right,
    )


def run_visual_demo(use_full_db: bool = False):
    from mocap_evaluation.mocap_loader import load_aggregated_database
    from mocap_evaluation.motion_matching import find_best_match

    db = load_aggregated_database()
    T = int(min(600, len(db["knee_right"])))
    qk = db["knee_right"][:T]
    qt = db["hip_right"][:T]
    _, _, seg = find_best_match(qk, qt, db)

    out = simulate_prosthetic_walking(seg, qk, use_gui=True)
    print("Demo metrics:", out)


if __name__ == "__main__":
    run_visual_demo()
