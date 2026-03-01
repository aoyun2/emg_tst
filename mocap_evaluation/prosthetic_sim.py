"""Prosthetic gait simulation / evaluation.

Drives a full humanoid skeleton from motion-capture data via MuJoCo physics.
All joints track their mocap reference angles *except* the right knee, which
is driven by the model's predicted knee angle, and the right hip, which is
driven by the sample's thigh_angle -- allowing evaluation of how well the
prediction maintains stable, natural gait.

The torso root uses 6 separate joints (3 slide + 3 hinge), ALL unactuated.
The humanoid's position, height, and orientation are determined entirely by
physics — ground contact, gravity, and the forces produced by the actuated
body joints.  If the joint angles produce a realistic walking gait, the
humanoid walks naturally.  If the predicted knee angle is bad, the body
falls, veers off course, or fails to advance.

Multi-DOF joints use separate hinge joints per axis (MuJoCo has no ball
joint with position actuators).  Hips have flex + abd + rot (3 DOF),
shoulders have flex + abd + rot (3 DOF), spine segments have flex + lateral
+ rot (3 DOF).  All BVH rotation channels are fully represented.

Supports:
- MuJoCo physics backend with full-body mocap-driven humanoid
- Optional reference humanoid (semi-transparent) showing ground-truth mocap
- Mocap segment visualization (stick-figure kinematic rendering for verification)
- Fall prediction for short segments via CoM trend analysis
- Trajectory recording for replay and GIF rendering
- Deterministic kinematic evaluator (dependency fallback)

Public APIs accept included-angle convention (degrees):
- 180 = straight / neutral
- smaller values = increased flexion magnitude

Internally we convert to flexion convention for the physics backend.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except Exception:  # pragma: no cover
    _MUJOCO_AVAILABLE = False

try:
    import mujoco.viewer  # noqa: F811
    _VIEWER_AVAILABLE = True
except Exception:  # pragma: no cover — headless environments
    _VIEWER_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────────────

SIM_FPS_DEFAULT = 200.0
HUMANOID_INIT_POS = np.array([0.0, 0.0, 1.05])
HUMANOID_INIT_POS_Z = 1.05      # legacy value used by kinematic evaluator
FALL_HEIGHT_THRESHOLD = 0.55
REF_Y_OFFSET = 1.5              # lateral offset for reference humanoid

# GL rendering may not be available in headless environments.  After the first
# failure to create a MuJoCo Renderer we set this flag so subsequent calls skip
# the attempt entirely (avoids C++ recursive-init abort).
_GL_RENDERER_AVAILABLE: Optional[bool] = None  # None = not yet tested


def _gl_available() -> bool:
    """Test whether MuJoCo's GL renderer can be initialised."""
    global _GL_RENDERER_AVAILABLE
    if _GL_RENDERER_AVAILABLE is not None:
        return _GL_RENDERER_AVAILABLE
    if not _MUJOCO_AVAILABLE:
        _GL_RENDERER_AVAILABLE = False
        return False
    try:
        _test_model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><geom size='0.1'/></body></worldbody></mujoco>"
        )
        _test_r = mujoco.Renderer(_test_model, height=8, width=8)
        _test_r.close()
        _GL_RENDERER_AVAILABLE = True
    except Exception:
        _GL_RENDERER_AVAILABLE = False
    return _GL_RENDERER_AVAILABLE


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _pad_or_trim(arr: np.ndarray, T: int, default: float = 180.0) -> np.ndarray:
    """Ensure *arr* has exactly *T* elements, padding with *default* if short."""
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) >= T:
        return arr[:T]
    return np.concatenate([arr, np.full(T - len(arr), default)])


# ── Data classes ─────────────────────────────────────────────────────────────

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
            # Raw time series for visualization (included-angle convention: 180=straight)
            "com_height_series": com.tolist(),
            "pred_knee_series": (180.0 - pred).tolist(),
            "ref_knee_series": (180.0 - ref).tolist(),
            "right_contact_frames": list(self.right_contact_frames),
            "left_contact_frames": list(self.left_contact_frames),
        }


@dataclass
class SimTrajectory:
    """Recorded simulation trajectory for replay and GIF rendering."""
    qpos_history: np.ndarray   # (T, nq)
    fps: float
    mjcf: str                  # MJCF XML used to create the model


# ── MJCF humanoid model ─────────────────────────────────────────────────────
#
# Full-body humanoid matching the CMU cgspeed BVH skeleton hierarchy.
# Root uses 6 separate joints (3 slide + 3 hinge), all unactuated and
# undamped — position and orientation are determined entirely by physics.
#
# Multi-DOF joints (hips, shoulders, spine) use separate hinge joints per
# axis to fully represent all BVH rotation channels:
#   - Hips: flex (Xrot) + abd (Zrot) + rot (Yrot) = 3 DOF each
#   - Shoulders: flex (Xrot) + abd (Zrot) + rot (Yrot) = 3 DOF each
#   - Spine segments: flex (Xrot) + lateral (Zrot) + rotation (Yrot) = 3 DOF each
#   - All other joints: 1-DOF hinge (flex/ext only)
#
# Body hierarchy:
#   Pelvis (root: slide X/Y/Z + hinge yaw/pitch/roll)
#   ├── Right leg: hip(3DOF) → knee → ankle → toe
#   ├── Left leg:  hip(3DOF) → knee → ankle → toe
#   └── Spine chain: LowerBack(3DOF) → Spine(3DOF) → Spine1(3DOF)
#       ├── Neck → Head
#       ├── Right arm: Clavicle → Shoulder(3DOF) → Elbow → Wrist → Fingers/Thumb
#       └── Left arm:  Clavicle → Shoulder(3DOF) → Elbow → Wrist → Fingers/Thumb
#
# Collision: body geoms collide with floor but NOT with each other
# (contype=1/conaffinity=2 for body, contype=2/conaffinity=1 for floor).

# Number of actuators per humanoid (prediction or reference).
# Multi-DOF breakdown:
#   Legs: 2*3(hip) + 2*1(knee) + 2*1(ankle) + 2*1(toe) = 12
#   Spine: 3*3(lower_back, spine, spine1) = 9
#   Head:  1(neck) + 1(head) = 2
#   Arms:  2*1(clav) + 2*3(shoulder) + 2*1(elbow) + 2*1(wrist) = 12
#   Fingers: 6 (3 per hand)
#   Total: 12 + 9 + 2 + 12 + 6 = 41
# Root (XY, Z, orientation) has NO actuators — all physics-driven.
N_ACT_PER_HUMANOID = 41

_MJCF = """
<mujoco model="prosthetic_eval_humanoid">
  <compiler angle="degree"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicit"/>

  <default>
    <joint damping="120" armature="0.02"/>
    <geom condim="3" friction="1 0.5 0.5" contype="1" conaffinity="2"/>
  </default>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1"
          rgba="0.8 0.9 0.8 1" contype="2" conaffinity="1"/>
    <light pos="0 0 5" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="3 3 4" dir="-0.5 -0.5 -1" diffuse="0.4 0.4 0.4"/>

    <body name="pelvis" pos="0 0 1.05">
      <joint name="root_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="root_y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="0"/>
      <joint name="root_yaw"   type="hinge" axis="0 0 1" damping="0"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" damping="0"/>
      <joint name="root_roll"  type="hinge" axis="1 0 0" damping="0"/>
      <geom type="capsule" fromto="0 0 -0.12 0 0 0" size="0.085"
            rgba="0.6 0.6 0.65 1"/>

      <!-- ── Right leg (hip: 3-DOF flex/abd/rot) ── -->
      <body name="right_thigh" pos="0 -0.10 -0.12">
        <joint name="right_hip" type="hinge" axis="0 -1 0"
               range="-70 70" damping="200"/>
        <joint name="right_hip_abd" type="hinge" axis="-1 0 0"
               range="-45 45" damping="100"/>
        <joint name="right_hip_rot" type="hinge" axis="0 0 1"
               range="-45 45" damping="80"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05"/>
        <body name="right_shank" pos="0 0 -0.42">
          <joint name="right_knee" type="hinge" axis="0 1 0"
                 range="0 130" damping="200"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.045"
                rgba="1 0.5 0.1 1"/>
          <body name="right_foot" pos="0 0 -0.42">
            <joint name="right_ankle" type="hinge" axis="0 -1 0"
                   range="-50 40" damping="120"/>
            <geom name="right_foot_geom" type="box"
                  size="0.11 0.05 0.03" pos="0.07 0 -0.02"
                  rgba="1 0.5 0.1 1"/>
            <body name="right_toe" pos="0.18 0 -0.02">
              <joint name="right_toe" type="hinge" axis="0 -1 0"
                     range="-30 45" damping="80"/>
              <geom type="box" size="0.04 0.04 0.02"
                    rgba="1 0.5 0.1 1"/>
            </body>
          </body>
        </body>
      </body>

      <!-- ── Left leg (hip: 3-DOF flex/abd/rot) ── -->
      <body name="left_thigh" pos="0 0.10 -0.12">
        <joint name="left_hip" type="hinge" axis="0 -1 0"
               range="-70 70" damping="200"/>
        <joint name="left_hip_abd" type="hinge" axis="-1 0 0"
               range="-45 45" damping="100"/>
        <joint name="left_hip_rot" type="hinge" axis="0 0 1"
               range="-45 45" damping="80"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05"/>
        <body name="left_shank" pos="0 0 -0.42">
          <joint name="left_knee" type="hinge" axis="0 1 0"
                 range="0 130" damping="200"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.045"/>
          <body name="left_foot" pos="0 0 -0.42">
            <joint name="left_ankle" type="hinge" axis="0 -1 0"
                   range="-50 40" damping="120"/>
            <geom name="left_foot_geom" type="box"
                  size="0.11 0.05 0.03" pos="0.07 0 -0.02"/>
            <body name="left_toe" pos="0.18 0 -0.02">
              <joint name="left_toe" type="hinge" axis="0 -1 0"
                     range="-30 45" damping="80"/>
              <geom type="box" size="0.04 0.04 0.02"/>
            </body>
          </body>
        </body>
      </body>

      <!-- ── Spine chain (each segment: 3-DOF flex/lateral/rot) ── -->
      <body name="lower_back" pos="0 0 0">
        <joint name="lower_back" type="hinge" axis="0 1 0"
               range="-30 30" damping="150"/>
        <joint name="lower_back_lat" type="hinge" axis="-1 0 0"
               range="-20 20" damping="80"/>
        <joint name="lower_back_rot" type="hinge" axis="0 0 1"
               range="-30 30" damping="80"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.07" size="0.075"
              rgba="0.6 0.6 0.65 1"/>
        <body name="spine" pos="0 0 0.07">
          <joint name="spine_jnt" type="hinge" axis="0 1 0"
                 range="-30 30" damping="150"/>
          <joint name="spine_lat" type="hinge" axis="-1 0 0"
                 range="-20 20" damping="80"/>
          <joint name="spine_rot" type="hinge" axis="0 0 1"
                 range="-30 30" damping="80"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.07"
                rgba="0.6 0.6 0.65 1"/>
          <body name="spine1" pos="0 0 0.06">
            <joint name="spine1_jnt" type="hinge" axis="0 1 0"
                   range="-30 30" damping="150"/>
            <joint name="spine1_lat" type="hinge" axis="-1 0 0"
                   range="-20 20" damping="80"/>
            <joint name="spine1_rot" type="hinge" axis="0 0 1"
                   range="-30 30" damping="80"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.065"
                  rgba="0.6 0.6 0.65 1"/>

            <!-- Neck → Head -->
            <body name="neck_body" pos="0 0 0.08">
              <joint name="neck" type="hinge" axis="0 1 0"
                     range="-30 30" damping="60"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.035"
                    rgba="0.85 0.75 0.65 1"/>
              <body name="head" pos="0 0 0.06">
                <joint name="head_jnt" type="hinge" axis="0 1 0"
                       range="-30 30" damping="40"/>
                <geom type="sphere" size="0.09" rgba="0.85 0.75 0.65 1"/>
              </body>
            </body>

            <!-- ── Right arm (shoulder: 3-DOF flex/abd/rot) ── -->
            <body name="right_clavicle" pos="0 -0.12 0.03">
              <joint name="right_clav" type="hinge" axis="0 -1 0"
                     range="-20 20" damping="25"/>
              <geom type="capsule" fromto="0 0 0 0 -0.08 0" size="0.025"/>
              <body name="right_upper_arm" pos="0 -0.10 0">
                <joint name="right_shoulder" type="hinge" axis="0 -1 0"
                       range="-90 90" damping="50"/>
                <joint name="right_shoulder_abd" type="hinge" axis="-1 0 0"
                       range="-90 90" damping="35"/>
                <joint name="right_shoulder_rot" type="hinge" axis="0 0 1"
                       range="-90 90" damping="30"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.28" size="0.03"/>
                <body name="right_forearm" pos="0 0 -0.28">
                  <joint name="right_elbow" type="hinge" axis="0 -1 0"
                         range="0 130" damping="35"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025"
                        rgba="0.85 0.75 0.65 1"/>
                  <body name="right_hand" pos="0 0 -0.25">
                    <joint name="right_wrist" type="hinge" axis="0 -1 0"
                           range="-60 60" damping="20"/>
                    <geom type="box" size="0.035 0.015 0.05" pos="0 0 -0.05"
                          rgba="0.85 0.75 0.65 1"/>
                    <body name="right_fing_base" pos="0 0 -0.08">
                      <joint name="right_finger" type="hinge" axis="0 -1 0"
                             range="0 90" damping="8"/>
                      <geom type="capsule" fromto="0 0 0 0 0 -0.025"
                            size="0.008" rgba="0.85 0.75 0.65 1"/>
                      <body name="right_fing_tip" pos="0 0 -0.025">
                        <joint name="right_fing_idx" type="hinge" axis="0 -1 0"
                               range="0 90" damping="8"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.015"
                              size="0.006" rgba="0.85 0.75 0.65 1"/>
                      </body>
                    </body>
                    <body name="right_thumb_body" pos="0.01 0.015 -0.03">
                      <joint name="right_thumb" type="hinge" axis="0 0 1"
                             range="-30 60" damping="8"/>
                      <geom type="capsule" fromto="0 0 0 0.015 0.01 -0.02"
                            size="0.007" rgba="0.85 0.75 0.65 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>

            <!-- ── Left arm (shoulder: 3-DOF flex/abd/rot) ── -->
            <body name="left_clavicle" pos="0 0.12 0.03">
              <joint name="left_clav" type="hinge" axis="0 -1 0"
                     range="-20 20" damping="25"/>
              <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.025"/>
              <body name="left_upper_arm" pos="0 0.10 0">
                <joint name="left_shoulder" type="hinge" axis="0 -1 0"
                       range="-90 90" damping="50"/>
                <joint name="left_shoulder_abd" type="hinge" axis="-1 0 0"
                       range="-90 90" damping="35"/>
                <joint name="left_shoulder_rot" type="hinge" axis="0 0 1"
                       range="-90 90" damping="30"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.28" size="0.03"/>
                <body name="left_forearm" pos="0 0 -0.28">
                  <joint name="left_elbow" type="hinge" axis="0 -1 0"
                         range="0 130" damping="35"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025"
                        rgba="0.85 0.75 0.65 1"/>
                  <body name="left_hand" pos="0 0 -0.25">
                    <joint name="left_wrist" type="hinge" axis="0 -1 0"
                           range="-60 60" damping="20"/>
                    <geom type="box" size="0.035 0.015 0.05" pos="0 0 -0.05"
                          rgba="0.85 0.75 0.65 1"/>
                    <body name="left_fing_base" pos="0 0 -0.08">
                      <joint name="left_finger" type="hinge" axis="0 -1 0"
                             range="0 90" damping="8"/>
                      <geom type="capsule" fromto="0 0 0 0 0 -0.025"
                            size="0.008" rgba="0.85 0.75 0.65 1"/>
                      <body name="left_fing_tip" pos="0 0 -0.025">
                        <joint name="left_fing_idx" type="hinge" axis="0 -1 0"
                               range="0 90" damping="8"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.015"
                              size="0.006" rgba="0.85 0.75 0.65 1"/>
                      </body>
                    </body>
                    <body name="left_thumb_body" pos="0.01 -0.015 -0.03">
                      <joint name="left_thumb" type="hinge" axis="0 0 -1"
                             range="-30 60" damping="8"/>
                      <geom type="capsule" fromto="0 0 0 0.015 -0.01 -0.02"
                            size="0.007" rgba="0.85 0.75 0.65 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- 41 actuators: body joints only. Root is fully physics-driven. -->
    <!-- ctrl[0..11]: legs (hips 3-DOF each + knee + ankle + toe) -->
    <position joint="right_hip"          kp="2000"/>
    <position joint="right_hip_abd"      kp="1000"/>
    <position joint="right_hip_rot"      kp="800"/>
    <position joint="left_hip"           kp="2000"/>
    <position joint="left_hip_abd"       kp="1000"/>
    <position joint="left_hip_rot"       kp="800"/>
    <position joint="right_knee"         kp="2000"/>
    <position joint="left_knee"          kp="2000"/>
    <position joint="right_ankle"        kp="1500"/>
    <position joint="left_ankle"         kp="1500"/>
    <position joint="right_toe"          kp="800"/>
    <position joint="left_toe"           kp="800"/>
    <!-- ctrl[12..20]: spine chain (3-DOF each) + neck + head -->
    <position joint="lower_back"         kp="1500"/>
    <position joint="lower_back_lat"     kp="800"/>
    <position joint="lower_back_rot"     kp="800"/>
    <position joint="spine_jnt"          kp="1500"/>
    <position joint="spine_lat"          kp="800"/>
    <position joint="spine_rot"          kp="800"/>
    <position joint="spine1_jnt"         kp="1500"/>
    <position joint="spine1_lat"         kp="800"/>
    <position joint="spine1_rot"         kp="800"/>
    <position joint="neck"               kp="800"/>
    <position joint="head_jnt"           kp="500"/>
    <!-- ctrl[23..24]: clavicles -->
    <position joint="right_clav"         kp="300"/>
    <position joint="left_clav"          kp="300"/>
    <!-- ctrl[25..30]: shoulders (3-DOF each) + elbows -->
    <position joint="right_shoulder"     kp="500"/>
    <position joint="right_shoulder_abd" kp="400"/>
    <position joint="right_shoulder_rot" kp="300"/>
    <position joint="left_shoulder"      kp="500"/>
    <position joint="left_shoulder_abd"  kp="400"/>
    <position joint="left_shoulder_rot"  kp="300"/>
    <position joint="right_elbow"        kp="400"/>
    <position joint="left_elbow"         kp="400"/>
    <!-- ctrl[33..34]: wrists -->
    <position joint="right_wrist"        kp="250"/>
    <position joint="left_wrist"         kp="250"/>
    <!-- ctrl[35..40]: fingers + thumbs -->
    <position joint="right_finger"       kp="100"/>
    <position joint="right_fing_idx"     kp="100"/>
    <position joint="right_thumb"        kp="100"/>
    <position joint="left_finger"        kp="100"/>
    <position joint="left_fing_idx"      kp="100"/>
    <position joint="left_thumb"         kp="100"/>
    <!-- Root: NO actuators. XY, Z, and orientation are all physics-driven.
         The humanoid must walk via ground contact, not position tracking. -->
  </actuator>
</mujoco>
"""


def _build_dual_mjcf() -> str:
    """Build MJCF with both prediction humanoid and a reference humanoid.

    The reference model has the same skeleton as the prediction model,
    semi-transparent blue, offset laterally by REF_Y_OFFSET.
    """
    import re
    ref_y = REF_Y_OFFSET
    rb = "0.3 0.5 0.8 0.5"  # reference blue (semi-transparent)
    rg = "0.2 0.7 0.3 0.5"  # reference green (prosthetic side)

    # Extract the <body name="pelvis" ...> ... </body> block from _MJCF
    body_start = _MJCF.index('<body name="pelvis"')
    wb_end = _MJCF.index("</worldbody>")
    body_xml = _MJCF[body_start:wb_end].rstrip()
    while body_xml.endswith("\n") or body_xml.endswith(" "):
        body_xml = body_xml.rstrip()

    # Prefix all name="..." and joint="..." values with ref_
    def _prefix_attr(match):
        attr = match.group(1)
        val = match.group(2)
        return f'{attr}="ref_{val}"'
    ref_xml = re.sub(r'(name|joint)="([^"]+)"', _prefix_attr, body_xml)

    # Change the root position to offset laterally
    ref_xml = ref_xml.replace(
        'name="ref_pelvis" pos="0 0 1.05"',
        f'name="ref_pelvis" pos="0 {ref_y} 1.05"',
    )

    # Replace all colours with semi-transparent blue/green
    ref_xml = re.sub(r'rgba="1 0\.5 0\.1 1"', f'rgba="{rg}"', ref_xml)
    ref_xml = re.sub(r'rgba="0\.6 0\.6 0\.65 1"', f'rgba="{rb}"', ref_xml)
    ref_xml = re.sub(r'rgba="0\.85 0\.75 0\.65 1"', f'rgba="{rb}"', ref_xml)

    ref_body_block = (
        "\n    <!-- ══ Reference humanoid (ground-truth mocap) "
        "═══════════════════════ -->\n    "
        + ref_xml + "\n"
    )

    # Reference actuator block (same 46 actuators, prefixed names)
    ref_actuators = """    <!-- Reference model actuators (46 actuators) -->
    <position joint="ref_right_hip"          kp="2000"/>
    <position joint="ref_right_hip_abd"      kp="1000"/>
    <position joint="ref_right_hip_rot"      kp="800"/>
    <position joint="ref_left_hip"           kp="2000"/>
    <position joint="ref_left_hip_abd"       kp="1000"/>
    <position joint="ref_left_hip_rot"       kp="800"/>
    <position joint="ref_right_knee"         kp="2000"/>
    <position joint="ref_left_knee"          kp="2000"/>
    <position joint="ref_right_ankle"        kp="1500"/>
    <position joint="ref_left_ankle"         kp="1500"/>
    <position joint="ref_right_toe"          kp="800"/>
    <position joint="ref_left_toe"           kp="800"/>
    <position joint="ref_lower_back"         kp="1500"/>
    <position joint="ref_lower_back_lat"     kp="800"/>
    <position joint="ref_lower_back_rot"     kp="800"/>
    <position joint="ref_spine_jnt"          kp="1500"/>
    <position joint="ref_spine_lat"          kp="800"/>
    <position joint="ref_spine_rot"          kp="800"/>
    <position joint="ref_spine1_jnt"         kp="1500"/>
    <position joint="ref_spine1_lat"         kp="800"/>
    <position joint="ref_spine1_rot"         kp="800"/>
    <position joint="ref_neck"               kp="800"/>
    <position joint="ref_head_jnt"           kp="500"/>
    <position joint="ref_right_clav"         kp="300"/>
    <position joint="ref_left_clav"          kp="300"/>
    <position joint="ref_right_shoulder"     kp="500"/>
    <position joint="ref_right_shoulder_abd" kp="400"/>
    <position joint="ref_right_shoulder_rot" kp="300"/>
    <position joint="ref_left_shoulder"      kp="500"/>
    <position joint="ref_left_shoulder_abd"  kp="400"/>
    <position joint="ref_left_shoulder_rot"  kp="300"/>
    <position joint="ref_right_elbow"        kp="400"/>
    <position joint="ref_left_elbow"         kp="400"/>
    <position joint="ref_right_wrist"        kp="250"/>
    <position joint="ref_left_wrist"         kp="250"/>
    <position joint="ref_right_finger"       kp="100"/>
    <position joint="ref_right_fing_idx"     kp="100"/>
    <position joint="ref_right_thumb"        kp="100"/>
    <position joint="ref_left_finger"        kp="100"/>
    <position joint="ref_left_fing_idx"      kp="100"/>
    <position joint="ref_left_thumb"         kp="100"/>"""

    base = _MJCF
    base = base.replace(
        "  </worldbody>",
        ref_body_block + "  </worldbody>",
    )
    base = base.replace(
        "  </actuator>",
        ref_actuators + "\n  </actuator>",
    )
    return base


# ── MuJoCo physics runner ───────────────────────────────────────────────────

class _MuJoCoRunner:
    """Run a full-body MuJoCo simulation driven by actuators + predicted knee."""

    def __init__(self, use_gui: bool, fps: float, show_reference: bool = False):
        self.use_gui = use_gui
        self.fps = float(fps)
        self.dt = 1.0 / max(self.fps, 1.0)
        self.show_reference = show_reference

    def run(
        self,
        mocap_segment: dict,
        predicted_knee: np.ndarray,
        fall_threshold: float,
        sample_thigh_right: Optional[np.ndarray] = None,
        reference_knee: Optional[np.ndarray] = None,
    ) -> dict:
        mjcf_str = _build_dual_mjcf() if self.show_reference else _MJCF
        model = mujoco.MjModel.from_xml_string(mjcf_str)
        data = mujoco.MjData(model)

        # ── Determine frame count ────────────────────────────────────────
        T = int(min(len(predicted_knee), len(mocap_segment["knee_right"])))
        if sample_thigh_right is not None:
            T = int(min(T, len(sample_thigh_right)))
        if T <= 0:
            out = run_kinematic_evaluation(
                mocap_segment, predicted_knee,
                sample_thigh_right=sample_thigh_right,
            )
            out["mode"] = "mujoco_physics_empty"
            return out

        # ── Helper: get a flexion signal from the mocap segment ──────────
        def _flex(key, default=180.0):
            return _included_to_flexion(_pad_or_trim(
                mocap_segment.get(key, np.full(T, default)), T, default))

        # ── Convert joint angles to flexion (degrees) ────────────────────
        pred = _included_to_flexion(np.asarray(predicted_knee, dtype=np.float64)[:T])
        ref = _included_to_flexion(np.asarray(mocap_segment["knee_right"], dtype=np.float64)[:T])

        if sample_thigh_right is not None:
            hip_r = _included_to_flexion(np.asarray(sample_thigh_right, dtype=np.float64)[:T])
        else:
            hip_r = _flex("hip_right")

        # ── Pre-compute controls (radians) ────────────────────────────────
        # 41 actuators per humanoid: body joints only. Root is physics-driven.
        NA = N_ACT_PER_HUMANOID
        n_act = NA * 2 if self.show_reference else NA
        controls = np.zeros((T, n_act), dtype=np.float64)

        # ── Helper: get an extra rotation channel (raw BVH degrees, 0=neutral) ─
        def _extra(key: str) -> np.ndarray:
            """Get an extra rotation channel (abd/rot/lateral) as radians."""
            raw = _pad_or_trim(
                mocap_segment.get(key, np.zeros(T)), T, default=0.0)
            return np.radians(raw)

        def _fill_humanoid(off, knee_signal):
            """Fill control columns [off : off+NA] for one humanoid."""
            # ── Legs (hips: 3-DOF each) ────────────────────────────
            controls[:, off + 0] = np.radians(hip_r)                     # right_hip flex
            controls[:, off + 1] = _extra("hip_right_abd")               # right_hip abd
            controls[:, off + 2] = _extra("hip_right_rot")               # right_hip rot
            controls[:, off + 3] = np.radians(_flex("hip_left"))         # left_hip flex
            controls[:, off + 4] = _extra("hip_left_abd")                # left_hip abd
            controls[:, off + 5] = _extra("hip_left_rot")                # left_hip rot
            controls[:, off + 6] = np.radians(knee_signal)               # right_knee
            controls[:, off + 7] = np.radians(_flex("knee_left"))        # left_knee
            controls[:, off + 8] = np.radians(_flex("ankle_right"))      # right_ankle
            controls[:, off + 9] = np.radians(_flex("ankle_left"))       # left_ankle
            controls[:, off + 10] = np.radians(_flex("toe_right"))       # right_toe
            controls[:, off + 11] = np.radians(_flex("toe_left"))        # left_toe
            # ── Spine chain (3-DOF each) + neck + head ─────────────
            controls[:, off + 12] = np.radians(_flex("pelvis_tilt"))     # lower_back flex
            controls[:, off + 13] = _extra("pelvis_lateral")             # lower_back lateral
            controls[:, off + 14] = _extra("pelvis_rotation")            # lower_back rot
            controls[:, off + 15] = np.radians(_flex("trunk_lean"))      # spine flex
            controls[:, off + 16] = _extra("trunk_lateral")              # spine lateral
            controls[:, off + 17] = _extra("trunk_rotation")             # spine rot
            controls[:, off + 18] = np.radians(_flex("upper_trunk"))     # spine1 flex
            controls[:, off + 19] = _extra("upper_trunk_lateral")        # spine1 lateral
            controls[:, off + 20] = _extra("upper_trunk_rotation")       # spine1 rot
            controls[:, off + 21] = np.radians(_flex("neck"))            # neck
            controls[:, off + 22] = np.radians(_flex("head"))            # head
            # ── Clavicles ──────────────────────────────────────────
            controls[:, off + 23] = np.radians(_flex("clavicle_right"))
            controls[:, off + 24] = np.radians(_flex("clavicle_left"))
            # ── Shoulders (3-DOF each) + elbows ────────────────────
            controls[:, off + 25] = np.radians(_flex("shoulder_right"))  # right shoulder flex
            controls[:, off + 26] = _extra("shoulder_right_abd")         # right shoulder abd
            controls[:, off + 27] = _extra("shoulder_right_rot")         # right shoulder rot
            controls[:, off + 28] = np.radians(_flex("shoulder_left"))   # left shoulder flex
            controls[:, off + 29] = _extra("shoulder_left_abd")          # left shoulder abd
            controls[:, off + 30] = _extra("shoulder_left_rot")          # left shoulder rot
            controls[:, off + 31] = np.radians(_flex("elbow_right"))     # right elbow
            controls[:, off + 32] = np.radians(_flex("elbow_left"))      # left elbow
            # ── Wrists ─────────────────────────────────────────────
            controls[:, off + 33] = np.radians(_flex("wrist_right"))
            controls[:, off + 34] = np.radians(_flex("wrist_left"))
            # ── Fingers + thumbs ───────────────────────────────────
            controls[:, off + 35] = np.radians(_flex("finger_right"))
            controls[:, off + 36] = np.radians(_flex("finger_index_right"))
            controls[:, off + 37] = np.radians(_flex("thumb_right"))
            controls[:, off + 38] = np.radians(_flex("finger_left"))
            controls[:, off + 39] = np.radians(_flex("finger_index_left"))
            controls[:, off + 40] = np.radians(_flex("thumb_left"))

        _fill_humanoid(0, pred)  # prediction model: right knee = PREDICTION

        if self.show_reference:
            if reference_knee is not None:
                ref_knee_visual = _included_to_flexion(
                    np.asarray(reference_knee, dtype=np.float64)[:T])
            else:
                ref_knee_visual = ref
            _fill_humanoid(NA, ref_knee_visual)  # reference: right knee = GT

        # ── Physics substeps per data frame ──────────────────────────────
        steps_per_frame = max(1, round(self.dt / model.opt.timestep))

        # ── Geom IDs for foot contact detection ──────────────────────────
        ridx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
        lidx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")

        # ── Body ID for CoM metrics ───────────────────────────────────────
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        metrics = EvalMetrics.empty()
        qpos_history: List[np.ndarray] = []

        # ── Warmup: initialise pose from frame-0 data, then settle ────
        # All root joints start at qpos=0 (body pos sets initial position).
        # Root orientation starts at zero (upright) — frame-0 orientation
        # is normalised to zero, so the model spawns upright.
        # We set body joint angles from frame-0 controls, then let
        # physics settle to resolve floor contact.
        data.ctrl[:] = controls[0]
        for i in range(model.nu):
            jnt_id = model.actuator_trnid[i, 0]
            data.qpos[model.jnt_qposadr[jnt_id]] = controls[0, i]
        for _ in range(500):
            mujoco.mj_step(model, data)
        data.qvel[:] = 0
        for _ in range(300):
            mujoco.mj_step(model, data)
        data.qvel[:] = 0

        # Reset metrics after warmup
        metrics = EvalMetrics.empty()

        def _step_loop(viewer_obj=None):
            for t in tqdm(range(T), desc="Simulating", unit="step", leave=False):
                if viewer_obj is not None and not viewer_obj.is_running():
                    break

                # Drive all joint actuators
                data.ctrl[:] = controls[t]

                # Physics substeps
                for _ in range(steps_per_frame):
                    mujoco.mj_step(model, data)

                if viewer_obj is not None:
                    viewer_obj.sync()
                    time.sleep(self.dt)

                # Record trajectory
                qpos_history.append(data.qpos.copy())

                # ── Collect metrics ──────────────────────────────────────
                com_h = float(data.subtree_com[torso_id, 2])
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

        def _replay_loop(viewer_obj, qpos_hist):
            """Loop the recorded trajectory until the viewer is closed."""
            paused = [False]
            speed = [1.0]

            def _key_cb(key):
                if key == 32:             # Space
                    paused[0] = not paused[0]
                elif key == 93:           # ]
                    speed[0] = min(speed[0] * 2.0, 16.0)
                    print(f"[replay] speed {speed[0]:.1f}×")
                elif key == 91:           # [
                    speed[0] = max(speed[0] * 0.5, 0.125)
                    print(f"[replay] speed {speed[0]:.1f}×")

            print("[MuJoCo] Replay controls: SPACE=pause  ]/[=speed up/down")

            while viewer_obj.is_running():
                for t in range(len(qpos_hist)):
                    if not viewer_obj.is_running():
                        return
                    while paused[0] and viewer_obj.is_running():
                        viewer_obj.sync()
                        time.sleep(0.05)
                    if not viewer_obj.is_running():
                        return
                    data.qpos[:] = qpos_hist[t]
                    mujoco.mj_forward(model, data)
                    viewer_obj.sync()
                    time.sleep(self.dt / max(speed[0], 0.01))

        # ── Run simulation (GUI or headless) ─────────────────────────────
        gui_worked = False
        if self.use_gui and _VIEWER_AVAILABLE:
            try:
                ctx = mujoco.viewer.launch_passive(
                    model, data,
                    show_left_ui=True,
                    show_right_ui=True,
                )
                with ctx as viewer_obj:
                    # Camera: side-tracking view
                    viewer_obj.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer_obj.cam.trackbodyid = torso_id
                    viewer_obj.cam.distance = 4.5 if self.show_reference else 3.5
                    viewer_obj.cam.elevation = -15.0
                    viewer_obj.cam.azimuth = 90.0

                    try:
                        viewer_obj.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                        viewer_obj.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
                    except Exception:
                        pass

                    if self.show_reference:
                        print("[MuJoCo] Reference model (blue) = ground-truth mocap")
                        print("[MuJoCo] Prediction model (grey/orange) = model output")

                    _step_loop(viewer_obj)
                    print("[MuJoCo] Simulation complete -- looping replay. "
                          "Close the viewer window to continue.")
                    _replay_loop(viewer_obj, qpos_history)
                gui_worked = True
            except Exception as exc:
                print(f"[MuJoCo] Viewer failed ({exc}), falling back to headless.")
                # Reset state for headless re-run
                data = mujoco.MjData(model)
                data.ctrl[:] = controls[0]
                for i in range(model.nu):
                    jnt_id = model.actuator_trnid[i, 0]
                    data.qpos[model.jnt_qposadr[jnt_id]] = controls[0, i]
                for _ in range(500):
                    mujoco.mj_step(model, data)
                data.qvel[:] = 0
                for _ in range(300):
                    mujoco.mj_step(model, data)
                data.qvel[:] = 0
                metrics = EvalMetrics.empty()
                qpos_history.clear()
                _step_loop(None)
        else:
            _step_loop(None)

        # ── Build result ─────────────────────────────────────────────────
        out = metrics.to_dict()
        out["mode"] = "mujoco_physics" + ("+gui" if gui_worked else "")

        # Fall prediction (useful for short segments that may not complete a fall)
        com_arr = np.asarray(metrics.com_height, dtype=np.float64)
        if com_arr.size > 0:
            out["fall_prediction"] = predict_fall(
                com_arr, fps=self.fps, fall_threshold=fall_threshold)
        else:
            out["fall_prediction"] = {
                "predicted_fall": False, "time_to_fall_s": float("inf"),
                "confidence": 0.0, "com_velocity": 0.0,
                "com_acceleration": 0.0, "min_com": 1.0, "trend_slope": 0.0,
            }

        # Attach trajectory for caller to save/render if desired
        out["_trajectory"] = SimTrajectory(
            qpos_history=np.array(qpos_history),
            fps=self.fps,
            mjcf=mjcf_str,
        )
        return out


# ── Kinematic evaluator (fallback when MuJoCo unavailable) ───────────────────

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
        hip = _included_to_flexion(np.asarray(
            mocap_segment.get("hip_right", np.full(T, 180.0)), dtype=np.float64)[:T])

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


# ── Public API ───────────────────────────────────────────────────────────────

def simulate_prosthetic_walking(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
    use_gui: bool = True,
    fps: float = SIM_FPS_DEFAULT,
    sample_thigh_right: Optional[np.ndarray] = None,
    save_trajectory: Optional[str | Path] = None,
    render_gif: Optional[str | Path] = None,
    show_reference: bool = False,
    reference_knee: Optional[np.ndarray] = None,
) -> dict:
    """Run prosthetic gait simulation via MuJoCo physics.

    Parameters
    ----------
    mocap_segment :
        Dict of joint angle arrays (included-angle, degrees) from motion
        matching, plus ``root_pos`` (T, 3) in MuJoCo Z-up metres.
    predicted_knee :
        Model-predicted right knee angle (included-angle, degrees).
    use_gui :
        Launch interactive MuJoCo viewer (requires display).
    fps :
        Data frame rate (default 200 Hz).
    sample_thigh_right :
        If provided, drives the right hip from this signal instead of
        the matched mocap hip_right.
    save_trajectory :
        Path to save the recorded trajectory (.npz) for later replay.
    render_gif :
        Path to save an animated GIF of the simulation.
    show_reference :
        When True, show a semi-transparent reference humanoid alongside
        the prediction model.
    reference_knee :
        Right knee signal (included-angle, degrees) for the reference
        humanoid.  Only used when *show_reference* is True.

    Returns
    -------
    dict
        Evaluation metrics, plus ``"trajectory_path"`` and/or
        ``"gif_path"`` if those outputs were requested.
    """
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError(
            "MuJoCo is required but not installed.\n"
            "Install it with:  pip install mujoco"
        )

    result = _MuJoCoRunner(
        use_gui=use_gui, fps=fps, show_reference=show_reference,
    ).run(
        mocap_segment, predicted_knee, FALL_HEIGHT_THRESHOLD,
        sample_thigh_right=sample_thigh_right,
        reference_knee=reference_knee,
    )

    # Extract trajectory (not JSON-serialisable, handled separately)
    trajectory = result.pop("_trajectory", None)

    if trajectory is not None and save_trajectory is not None:
        p = Path(save_trajectory)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_trajectory(p, trajectory)
        result["trajectory_path"] = str(p)
        print(f"[sim] Trajectory saved -> {p}")

    if trajectory is not None and render_gif is not None:
        p = Path(render_gif)
        p.parent.mkdir(parents=True, exist_ok=True)
        _render_gif(trajectory, p)
        if p.exists():
            result["gif_path"] = str(p)

    return result


# ── Trajectory I/O ───────────────────────────────────────────────────────────

def _save_trajectory(path: Path, traj: SimTrajectory) -> None:
    np.savez_compressed(
        path,
        qpos_history=traj.qpos_history,
        fps=np.array([traj.fps]),
        mjcf=np.array([traj.mjcf]),
    )


def save_trajectory(path: str | Path, traj: SimTrajectory) -> None:
    """Save a SimTrajectory to a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_trajectory(path, traj)


def load_trajectory(path: str | Path) -> SimTrajectory:
    """Load a SimTrajectory from a .npz file."""
    d = np.load(path, allow_pickle=True)
    return SimTrajectory(
        qpos_history=d["qpos_history"],
        fps=float(d["fps"][0]),
        mjcf=str(d["mjcf"][0]),
    )


# ── Replay ───────────────────────────────────────────────────────────────────

def replay_trajectory(
    trajectory_or_path: SimTrajectory | str | Path,
    speed: float = 1.0,
) -> None:
    """Replay a recorded simulation trajectory in the MuJoCo viewer.

    Loops continuously until the viewer window is closed.
    """
    if not _MUJOCO_AVAILABLE or not _VIEWER_AVAILABLE:
        raise RuntimeError("MuJoCo with viewer is required for replay.")

    if isinstance(trajectory_or_path, (str, Path)):
        traj = load_trajectory(trajectory_or_path)
    else:
        traj = trajectory_or_path

    model = mujoco.MjModel.from_xml_string(traj.mjcf)
    data = mujoco.MjData(model)
    dt = 1.0 / max(traj.fps, 1.0) / max(speed, 0.01)
    T = len(traj.qpos_history)

    print(f"[replay] {T} frames @ {traj.fps:.0f} Hz "
          f"(speed {speed:.1f}x, duration {T / traj.fps:.1f}s)")
    print("[replay] Controls: SPACE=pause  ]/[=speed up/down  "
          "Close the viewer window to stop.")

    paused = [False]
    cur_speed = [speed]

    def _key_cb(key):
        if key == 32:             # Space
            paused[0] = not paused[0]
        elif key == 93:           # ]
            cur_speed[0] = min(cur_speed[0] * 2.0, 16.0)
            print(f"[replay] speed {cur_speed[0]:.1f}×")
        elif key == 91:           # [
            cur_speed[0] = max(cur_speed[0] * 0.5, 0.125)
            print(f"[replay] speed {cur_speed[0]:.1f}×")

    with mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=_key_cb,
    ) as viewer:
        # Side-tracking camera
        torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = torso_id
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 90.0

        try:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
        except Exception:
            pass

        while viewer.is_running():
            for t in range(T):
                if not viewer.is_running():
                    return
                while paused[0] and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                if not viewer.is_running():
                    return
                data.qpos[:] = traj.qpos_history[t]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt / max(cur_speed[0] / speed, 0.01))


# ── GIF rendering ────────────────────────────────────────────────────────────

def _render_gif(
    traj: SimTrajectory,
    output_path: Path,
    width: int = 640,
    height: int = 480,
    gif_fps: int = 30,
) -> None:
    """Render a SimTrajectory to an animated GIF (headless-safe)."""
    if not _gl_available():
        print("[gif] GL renderer not available (headless?), skipping GIF.")
        return

    try:
        from PIL import Image
    except ImportError:
        print("[gif] Pillow not installed, skipping GIF render.")
        return

    model = mujoco.MjModel.from_xml_string(traj.mjcf)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    T = len(traj.qpos_history)
    skip = max(1, round(traj.fps / gif_fps))
    frame_indices = list(range(0, T, skip))

    # Camera: side-tracking
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    camera.distance = 4.5
    camera.elevation = -15.0
    camera.azimuth = 90.0

    frames: List = []
    for t in tqdm(frame_indices, desc="Rendering GIF", unit="frame", leave=False):
        data.qpos[:] = traj.qpos_history[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        frames.append(Image.fromarray(pixels))

    if not frames:
        return

    duration_ms = int(1000 * skip / traj.fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    dur_s = T / traj.fps
    print(f"[gif] Saved {len(frames)}-frame GIF ({dur_s:.1f}s) -> {output_path}")


def render_simulation_gif(
    trajectory_or_path: SimTrajectory | str | Path,
    output_path: str | Path,
    width: int = 640,
    height: int = 480,
    gif_fps: int = 30,
) -> None:
    """Render a saved or in-memory trajectory to an animated GIF."""
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo is required for GIF rendering.")

    if isinstance(trajectory_or_path, (str, Path)):
        traj = load_trajectory(trajectory_or_path)
    else:
        traj = trajectory_or_path

    _render_gif(traj, Path(output_path), width=width, height=height, gif_fps=gif_fps)


# ── Kinematic mocap playback (no physics) ────────────────────────────────────


def render_mocap_kinematic(
    mocap_segment: dict,
    fps: float = SIM_FPS_DEFAULT,
    save_trajectory: Optional[str | Path] = None,
    render_gif: Optional[str | Path] = None,
    use_gui: bool = False,
    gif_fps: int = 30,
) -> Optional[SimTrajectory]:
    """Pose the humanoid kinematically from the mocap segment (NO physics).

    This drives every joint directly from the BVH data — including
    right knee — so you see exactly what the loaded mocap segment
    describes.  No gravity, no contacts, no simulation.  Pure data
    playback.

    Use this to visually verify the loaded segment before comparing
    with physics simulation output.

    Parameters
    ----------
    mocap_segment : dict from motion matching (all joint keys + root_pos)
    fps : data frame rate
    save_trajectory : save .npz for later ``replay_trajectory()``
    render_gif : save an animated GIF of the kinematic playback
    use_gui : if True, open MuJoCo viewer for interactive viewing
    gif_fps : frame rate for GIF output

    Returns
    -------
    SimTrajectory or None
    """
    if not _MUJOCO_AVAILABLE:
        print("[kinematic] MuJoCo not available, cannot render.")
        return None

    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)

    T = len(mocap_segment.get("knee_right", []))
    if T == 0:
        return None

    dt = 1.0 / max(fps, 1.0)

    # ── Build control signals (same logic as _fill_humanoid) ─────────
    def _flex(key, default=180.0):
        return _included_to_flexion(_pad_or_trim(
            mocap_segment.get(key, np.full(T, default)), T, default))

    def _extra(key):
        raw = _pad_or_trim(
            mocap_segment.get(key, np.zeros(T)), T, default=0.0)
        return np.radians(raw)

    # Root XY
    root_pos = mocap_segment.get("root_pos", np.zeros((T, 3)))
    if np.ndim(root_pos) == 2:
        root_pos = root_pos[:T]
    else:
        root_pos = np.zeros((T, 3))
    root_xy = root_pos[:, :2].copy()
    root_xy -= root_xy[0]

    # Root position (BVH→MuJoCo Z-up, metres), normalised to start at origin.
    root_pos = mocap_segment.get("root_pos", np.zeros((T, 3)))
    if np.ndim(root_pos) == 2:
        root_pos = root_pos[:T]
    else:
        root_pos = np.zeros((T, 3))
    root_xy = root_pos[:, :2].copy()
    root_xy -= root_xy[0]

    # Root orientation (BVH Y-up → MuJoCo Z-up), normalised to start at 0.
    root_ori = np.zeros((T, 3), dtype=np.float64)
    root_ori[:, 0] = np.radians(_pad_or_trim(
        mocap_segment.get("root_yaw", np.zeros(T)), T, default=0.0))
    root_ori[:, 1] = np.radians(-_pad_or_trim(
        mocap_segment.get("root_pitch", np.zeros(T)), T, default=0.0))
    root_ori[:, 2] = np.radians(_pad_or_trim(
        mocap_segment.get("root_roll", np.zeros(T)), T, default=0.0))
    root_ori -= root_ori[0]

    # Use mocap hip for right side (kinematic = pure BVH)
    hip_r = _flex("hip_right")

    NA = N_ACT_PER_HUMANOID
    controls = np.zeros((T, NA), dtype=np.float64)

    # Legs
    controls[:, 0] = np.radians(hip_r)
    controls[:, 1] = _extra("hip_right_abd")
    controls[:, 2] = _extra("hip_right_rot")
    controls[:, 3] = np.radians(_flex("hip_left"))
    controls[:, 4] = _extra("hip_left_abd")
    controls[:, 5] = _extra("hip_left_rot")
    controls[:, 6] = np.radians(_flex("knee_right"))  # MOCAP knee (not prediction)
    controls[:, 7] = np.radians(_flex("knee_left"))
    controls[:, 8] = np.radians(_flex("ankle_right"))
    controls[:, 9] = np.radians(_flex("ankle_left"))
    controls[:, 10] = np.radians(_flex("toe_right"))
    controls[:, 11] = np.radians(_flex("toe_left"))
    # Spine
    controls[:, 12] = np.radians(_flex("pelvis_tilt"))
    controls[:, 13] = _extra("pelvis_lateral")
    controls[:, 14] = _extra("pelvis_rotation")
    controls[:, 15] = np.radians(_flex("trunk_lean"))
    controls[:, 16] = _extra("trunk_lateral")
    controls[:, 17] = _extra("trunk_rotation")
    controls[:, 18] = np.radians(_flex("upper_trunk"))
    controls[:, 19] = _extra("upper_trunk_lateral")
    controls[:, 20] = _extra("upper_trunk_rotation")
    controls[:, 21] = np.radians(_flex("neck"))
    controls[:, 22] = np.radians(_flex("head"))
    # Clavicles
    controls[:, 23] = np.radians(_flex("clavicle_right"))
    controls[:, 24] = np.radians(_flex("clavicle_left"))
    # Shoulders + elbows
    controls[:, 25] = np.radians(_flex("shoulder_right"))
    controls[:, 26] = _extra("shoulder_right_abd")
    controls[:, 27] = _extra("shoulder_right_rot")
    controls[:, 28] = np.radians(_flex("shoulder_left"))
    controls[:, 29] = _extra("shoulder_left_abd")
    controls[:, 30] = _extra("shoulder_left_rot")
    controls[:, 31] = np.radians(_flex("elbow_right"))
    controls[:, 32] = np.radians(_flex("elbow_left"))
    # Wrists
    controls[:, 33] = np.radians(_flex("wrist_right"))
    controls[:, 34] = np.radians(_flex("wrist_left"))
    # Fingers
    controls[:, 35] = np.radians(_flex("finger_right"))
    controls[:, 36] = np.radians(_flex("finger_index_right"))
    controls[:, 37] = np.radians(_flex("thumb_right"))
    controls[:, 38] = np.radians(_flex("finger_left"))
    controls[:, 39] = np.radians(_flex("finger_index_left"))
    controls[:, 40] = np.radians(_flex("thumb_left"))

    # ── Root joint qpos indices (not actuated, set directly) ──────────
    root_jnt_ids = {}
    for jname in ("root_x", "root_y", "root_z",
                   "root_yaw", "root_pitch", "root_roll"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        root_jnt_ids[jname] = model.jnt_qposadr[jid]

    # ── Kinematic playback: set qpos directly (no physics) ───────────
    qpos_history: List[np.ndarray] = []

    for t_idx in range(T):
        # Set each actuated body joint directly to its target value
        for i in range(model.nu):
            jnt_id = model.actuator_trnid[i, 0]
            data.qpos[model.jnt_qposadr[jnt_id]] = controls[t_idx, i]
        # Set root position and orientation directly (no actuators)
        data.qpos[root_jnt_ids["root_x"]] = root_xy[t_idx, 0]
        data.qpos[root_jnt_ids["root_y"]] = root_xy[t_idx, 1]
        # root_z: leave at initial height (kinematic, no gravity)
        data.qpos[root_jnt_ids["root_yaw"]]   = root_ori[t_idx, 0]
        data.qpos[root_jnt_ids["root_pitch"]] = root_ori[t_idx, 1]
        data.qpos[root_jnt_ids["root_roll"]]  = root_ori[t_idx, 2]
        # Forward kinematics only (no dynamics)
        mujoco.mj_forward(model, data)
        qpos_history.append(data.qpos.copy())

    traj = SimTrajectory(
        qpos_history=np.array(qpos_history),
        fps=fps,
        mjcf=_MJCF,
    )

    # ── Save outputs ─────────────────────────────────────────────────
    if save_trajectory is not None:
        p = Path(save_trajectory)
        p.parent.mkdir(parents=True, exist_ok=True)
        _save_trajectory(p, traj)
        print(f"[kinematic] Trajectory saved -> {p}")

    if render_gif is not None:
        p = Path(render_gif)
        p.parent.mkdir(parents=True, exist_ok=True)
        _render_gif(traj, p, gif_fps=gif_fps)

    if use_gui and _VIEWER_AVAILABLE:
        print("[kinematic] Playing mocap segment (kinematic, no physics).")
        print("[kinematic] This is the RAW BVH data — what the mocap says.")
        replay_trajectory(traj, speed=1.0)

    return traj


def render_mocap_side_by_side_gif(
    mocap_segment: dict,
    sim_trajectory: SimTrajectory | str | Path,
    output_path: str | Path,
    fps: float = SIM_FPS_DEFAULT,
    gif_fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> None:
    """Render a side-by-side GIF: left = kinematic mocap, right = physics sim.

    This is the key visualization for verifying that the physics simulation
    lines up with what the mocap data actually describes.
    """
    if not _MUJOCO_AVAILABLE:
        print("[side-by-side] MuJoCo not available.")
        return

    if not _gl_available():
        print("[side-by-side] GL renderer not available (headless?), skipping.")
        return

    try:
        from PIL import Image
    except ImportError:
        print("[side-by-side] Pillow not installed, skipping.")
        return

    # Build kinematic trajectory
    kinematic_traj = render_mocap_kinematic(mocap_segment, fps=fps)
    if kinematic_traj is None:
        return

    # Load sim trajectory if path
    if isinstance(sim_trajectory, (str, Path)):
        sim_traj = load_trajectory(sim_trajectory)
    else:
        sim_traj = sim_trajectory

    T = min(len(kinematic_traj.qpos_history), len(sim_traj.qpos_history))
    skip = max(1, round(fps / gif_fps))
    frame_indices = list(range(0, T, skip))

    # Set up two renderers
    kin_model = mujoco.MjModel.from_xml_string(kinematic_traj.mjcf)
    kin_data = mujoco.MjData(kin_model)

    sim_model = mujoco.MjModel.from_xml_string(sim_traj.mjcf)
    sim_data = mujoco.MjData(sim_model)

    try:
        kin_renderer = mujoco.Renderer(kin_model, height=height, width=width)
        sim_renderer = mujoco.Renderer(sim_model, height=height, width=width)
    except Exception as exc:
        print(f"[side-by-side] Renderer init failed ({exc}), skipping.")
        return

    # Cameras
    def _make_camera(model_obj):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(
            model_obj, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance = 3.5
        cam.elevation = -15.0
        cam.azimuth = 90.0
        return cam

    kin_cam = _make_camera(kin_model)
    sim_cam = _make_camera(sim_model)

    frames: List = []
    for t_idx in tqdm(frame_indices, desc="Rendering side-by-side", unit="frame",
                      leave=False):
        # Kinematic frame
        kin_data.qpos[:] = kinematic_traj.qpos_history[t_idx]
        mujoco.mj_forward(kin_model, kin_data)
        kin_renderer.update_scene(kin_data, camera=kin_cam)
        kin_pixels = kin_renderer.render()

        # Sim frame
        sim_data.qpos[:len(sim_traj.qpos_history[t_idx])] = sim_traj.qpos_history[t_idx]
        mujoco.mj_forward(sim_model, sim_data)
        sim_renderer.update_scene(sim_data, camera=sim_cam)
        sim_pixels = sim_renderer.render()

        # Concatenate horizontally with a label bar
        combined = np.concatenate([kin_pixels, sim_pixels], axis=1)
        frames.append(Image.fromarray(combined))

    if not frames:
        return

    duration_ms = int(1000 * skip / fps)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    dur_s = T / fps
    print(f"[side-by-side] Saved {len(frames)}-frame GIF ({dur_s:.1f}s) -> {output_path}")
    print(f"[side-by-side] Left = kinematic mocap (raw BVH), Right = physics sim")


# ── Mocap segment visualization (plots) ──────────────────────────────────────


def visualize_mocap_segment(
    mocap_segment: dict,
    out_path: str | Path,
    fps: float = SIM_FPS_DEFAULT,
    title: str = "Loaded mocap segment",
) -> None:
    """Render a multi-panel plot showing ALL joint angles in the loaded segment.

    This lets you visually verify what the simulation is being driven by,
    before comparing with physics output.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_frames = len(mocap_segment.get("knee_right", []))
    if T_frames == 0:
        return
    t = np.arange(T_frames) / fps

    # Group joints for organized display
    groups = [
        ("Legs (included angle, 180=straight)", [
            ("knee_right", "Knee R"), ("knee_left", "Knee L"),
            ("hip_right", "Hip R"), ("hip_left", "Hip L"),
            ("ankle_right", "Ankle R"), ("ankle_left", "Ankle L"),
            ("toe_right", "Toe R"), ("toe_left", "Toe L"),
        ]),
        ("Legs: Hip abd/rot (raw BVH deg, 0=neutral)", [
            ("hip_right_abd", "Hip R abd"), ("hip_left_abd", "Hip L abd"),
            ("hip_right_rot", "Hip R rot"), ("hip_left_rot", "Hip L rot"),
        ]),
        ("Spine chain (included angle)", [
            ("pelvis_tilt", "Pelvis tilt"), ("trunk_lean", "Trunk lean"),
            ("upper_trunk", "Upper trunk"),
            ("neck", "Neck"), ("head", "Head"),
        ]),
        ("Spine: lateral/rotation (raw BVH deg)", [
            ("pelvis_lateral", "Pelvis lat"), ("pelvis_rotation", "Pelvis rot"),
            ("trunk_lateral", "Trunk lat"), ("trunk_rotation", "Trunk rot"),
            ("upper_trunk_lateral", "UpperTrunk lat"),
            ("upper_trunk_rotation", "UpperTrunk rot"),
        ]),
        ("Arms (included angle)", [
            ("shoulder_right", "Shoulder R"), ("shoulder_left", "Shoulder L"),
            ("elbow_right", "Elbow R"), ("elbow_left", "Elbow L"),
            ("clavicle_right", "Clav R"), ("clavicle_left", "Clav L"),
            ("wrist_right", "Wrist R"), ("wrist_left", "Wrist L"),
        ]),
        ("Arms: shoulder abd/rot (raw BVH deg)", [
            ("shoulder_right_abd", "Shoulder R abd"),
            ("shoulder_left_abd", "Shoulder L abd"),
            ("shoulder_right_rot", "Shoulder R rot"),
            ("shoulder_left_rot", "Shoulder L rot"),
        ]),
        ("Root position (metres, MuJoCo Z-up)", []),  # special handling
    ]

    n_panels = len(groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 * n_panels),
                             sharex=True)

    for i, (group_title, keys) in enumerate(groups):
        ax = axes[i]
        if group_title.startswith("Root position"):
            # Special: plot root_pos XYZ
            rp = mocap_segment.get("root_pos", np.zeros((T_frames, 3)))
            if rp.ndim == 2 and rp.shape[1] >= 3:
                ax.plot(t, rp[:T_frames, 0], label="X (forward)", lw=1.2)
                ax.plot(t, rp[:T_frames, 1], label="Y (lateral)", lw=1.2)
                ax.plot(t, rp[:T_frames, 2], label="Z (height)", lw=1.2)
            # Also overlay root orientation
            for ch, lbl in [("root_pitch", "Pitch"), ("root_yaw", "Yaw"),
                            ("root_roll", "Roll")]:
                sig = mocap_segment.get(ch)
                if sig is not None:
                    ax2 = ax.twinx()
                    ax2.plot(t, sig[:T_frames], label=lbl, lw=0.8,
                             ls="--", alpha=0.5)
                    ax2.set_ylabel("Root orient (deg)", fontsize=7)
                    ax2.legend(loc="upper right", fontsize=6)
                    break  # only one twinx
        else:
            for key, label in keys:
                sig = mocap_segment.get(key)
                if sig is not None and np.any(sig != 0):
                    ax.plot(t, sig[:T_frames], label=label, lw=1.0)
        ax.set_ylabel("degrees", fontsize=8)
        ax.set_title(group_title, fontsize=9)
        ax.legend(fontsize=6, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{title}  ({T_frames} frames, {T_frames/fps:.2f}s @ {fps:.0f} Hz)",
                 fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[viz] Mocap segment plot saved -> {out_path}")


def visualize_sim_vs_mocap(
    mocap_segment: dict,
    sim_metrics: dict,
    out_path: str | Path,
    fps: float = SIM_FPS_DEFAULT,
    title: str = "Physics sim vs Mocap reference",
) -> None:
    """Side-by-side comparison of simulation output vs the kinematic mocap input.

    Shows:
    - Knee angle: mocap reference vs what the sim actually achieved
    - CoM height trajectory with fall threshold
    - Root position (X forward) comparison if available
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_knee = np.asarray(sim_metrics.get("pred_knee_series", []))
    ref_knee = np.asarray(sim_metrics.get("ref_knee_series", []))
    com_h = np.asarray(sim_metrics.get("com_height_series", []))

    if pred_knee.size == 0:
        return

    T = len(pred_knee)
    t = np.arange(T) / fps

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Knee angles
    ax = axes[0]
    ax.plot(t, ref_knee[:T], label="Mocap reference knee", lw=1.5,
            color="green", alpha=0.8)
    ax.plot(t, pred_knee[:T], label="Sim actual knee (prediction-driven)",
            lw=1.5, color="orange")
    if sim_metrics.get("fall_detected") and sim_metrics["fall_frame"] >= 0:
        ff = sim_metrics["fall_frame"] / fps
        ax.axvline(ff, color="red", ls="--", alpha=0.6,
                   label=f"Fall @ {ff:.2f}s")
    ax.set_ylabel("Knee (deg, 180=straight)")
    ax.set_title(f"{title}  |  RMSE={sim_metrics.get('knee_rmse_deg', 0):.2f}°")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: CoM height + fall prediction
    ax = axes[1]
    if com_h.size > 0:
        ax.plot(t[:len(com_h)], com_h[:T], lw=1.2, color="steelblue",
                label="CoM height")
        ax.axhline(FALL_HEIGHT_THRESHOLD, color="red", ls="--", alpha=0.5,
                   label=f"Fall threshold ({FALL_HEIGHT_THRESHOLD}m)")

        # Fall prediction annotation
        fp = sim_metrics.get("fall_prediction", {})
        if fp.get("predicted_fall"):
            tta = fp.get("time_to_fall_s", float("inf"))
            conf = fp.get("confidence", 0)
            ax.annotate(
                f"FALL PREDICTED\nETA: {tta:.2f}s  conf: {conf:.0%}",
                xy=(t[-1], com_h[-1] if com_h.size > 0 else 0.8),
                fontsize=9, color="red", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8),
            )

    ax.set_ylabel("CoM height (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Joint angle overview (hip, ankle for context)
    ax = axes[2]
    for key, label, color in [
        ("hip_right", "Hip R (mocap)", "tab:blue"),
        ("hip_left", "Hip L (mocap)", "tab:cyan"),
        ("ankle_right", "Ankle R (mocap)", "tab:purple"),
        ("ankle_left", "Ankle L (mocap)", "tab:pink"),
    ]:
        sig = mocap_segment.get(key)
        if sig is not None:
            ax.plot(t, sig[:T], label=label, lw=0.8, color=color, alpha=0.7)
    ax.set_ylabel("Angle (deg, 180=straight)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Mocap context: other joint angles driving the simulation")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[viz] Sim vs mocap plot saved -> {out_path}")


# ── Fall prediction for short segments ───────────────────────────────────────


def predict_fall(
    com_height_series: np.ndarray,
    fps: float = SIM_FPS_DEFAULT,
    fall_threshold: float = FALL_HEIGHT_THRESHOLD,
    lookback_s: float = 0.5,
) -> dict:
    """Predict whether a fall is imminent based on CoM height trend.

    Even if the segment is too short for the body to actually reach the
    ground, we can detect if:
    1. CoM is dropping (negative velocity)
    2. CoM is accelerating downward (negative acceleration)
    3. Linear extrapolation predicts crossing the fall threshold

    Parameters
    ----------
    com_height_series : (T,) CoM height in metres
    fps : frame rate
    fall_threshold : height below which a fall is detected
    lookback_s : seconds to use for trend estimation

    Returns
    -------
    dict with keys:
        predicted_fall : bool
        time_to_fall_s : float (estimated seconds until fall, inf if not falling)
        confidence : float 0-1
        com_velocity : float (m/s, negative = dropping)
        com_acceleration : float (m/s², negative = accelerating down)
        min_com : float (minimum CoM height observed)
        trend_slope : float (linear fit slope in m/s)
    """
    com = np.asarray(com_height_series, dtype=np.float64)
    T = len(com)

    if T < 10:
        return {
            "predicted_fall": False,
            "time_to_fall_s": float("inf"),
            "confidence": 0.0,
            "com_velocity": 0.0,
            "com_acceleration": 0.0,
            "min_com": float(com.min()) if T > 0 else 1.0,
            "trend_slope": 0.0,
        }

    # Use the last lookback_s seconds for trend
    lookback_frames = min(T, max(10, int(lookback_s * fps)))
    tail = com[-lookback_frames:]
    t_tail = np.arange(lookback_frames) / fps

    # Linear fit for slope (m/s)
    if lookback_frames > 1:
        coeffs = np.polyfit(t_tail, tail, 1)
        slope = coeffs[0]  # m/s
    else:
        slope = 0.0

    # Instantaneous velocity and acceleration (finite differences)
    if T >= 3:
        dt = 1.0 / fps
        vel = np.diff(com) / dt
        acc = np.diff(vel) / dt
        cur_vel = float(vel[-1])
        cur_acc = float(np.mean(acc[-min(20, len(acc)):]))
    else:
        cur_vel = 0.0
        cur_acc = 0.0

    min_com = float(com.min())
    cur_com = float(com[-1])

    # Already fell?
    already_fell = cur_com < fall_threshold

    # Time to fall estimate via linear extrapolation
    if slope < -0.001:
        time_to_fall = (fall_threshold - cur_com) / slope
        if time_to_fall < 0:
            time_to_fall = 0.0  # already below
    else:
        time_to_fall = float("inf")

    # Quadratic extrapolation (accounts for acceleration)
    # h(t) = h0 + v*t + 0.5*a*t^2 = threshold
    # Solve: 0.5*a*t^2 + v*t + (h0 - threshold) = 0
    time_to_fall_quad = float("inf")
    if cur_acc < -0.1 or cur_vel < -0.05:
        a = 0.5 * cur_acc
        b = cur_vel
        c = cur_com - fall_threshold
        if abs(a) > 1e-6:
            disc = b * b - 4 * a * c
            if disc >= 0:
                t1 = (-b - math.sqrt(disc)) / (2 * a)
                t2 = (-b + math.sqrt(disc)) / (2 * a)
                positive_roots = [r for r in (t1, t2) if r > 0]
                if positive_roots:
                    time_to_fall_quad = min(positive_roots)
        elif b < -0.01:
            time_to_fall_quad = -c / b

    best_ttf = min(time_to_fall, time_to_fall_quad)

    # Confidence scoring
    confidence = 0.0
    if already_fell:
        confidence = 1.0
    else:
        # Factor 1: How negative is the velocity?
        vel_factor = min(1.0, max(0.0, -cur_vel / 0.5))
        # Factor 2: How negative is the acceleration?
        acc_factor = min(1.0, max(0.0, -cur_acc / 5.0))
        # Factor 3: How close to threshold?
        proximity_factor = max(0.0, 1.0 - (cur_com - fall_threshold) / 0.5)
        # Factor 4: Is the trend consistently downward?
        trend_factor = min(1.0, max(0.0, -slope / 0.3))

        confidence = min(1.0, (
            vel_factor * 0.3 +
            acc_factor * 0.2 +
            proximity_factor * 0.25 +
            trend_factor * 0.25
        ))

    predicted = already_fell or (confidence > 0.4 and best_ttf < 5.0)

    return {
        "predicted_fall": bool(predicted),
        "time_to_fall_s": float(best_ttf),
        "confidence": float(confidence),
        "com_velocity": float(cur_vel),
        "com_acceleration": float(cur_acc),
        "min_com": float(min_com),
        "trend_slope": float(slope),
    }
