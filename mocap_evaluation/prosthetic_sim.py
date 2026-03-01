"""Prosthetic gait simulation / evaluation.

Drives a MuJoCo humanoid from motion-capture joint angles via stiff PD
position actuators.  All joints track their mocap reference *except* the
right knee, which is driven by the model's predicted signal.

Root translation (XYZ) and orientation (yaw/pitch/roll) are fully
unactuated — the humanoid's position and balance emerge entirely from
physics (gravity, ground contact, and joint-produced torques).  If joint
angles describe a valid walking gait the body walks forward naturally.
If the predicted knee angle is poor the body staggers, veers, or falls.

Public API uses included-angle convention (degrees, 180 = straight).
Internally angles are converted to flexion radians for MuJoCo actuators.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except Exception:
    _MUJOCO_AVAILABLE = False

try:
    import mujoco.viewer
    _VIEWER_AVAILABLE = True
except Exception:
    _VIEWER_AVAILABLE = False


# ── Constants ────────────────────────────────────────────────────────────────

SIM_FPS_DEFAULT = 200.0
FALL_HEIGHT_THRESHOLD = 0.55
_INIT_HEIGHT = 1.05
REF_Y_OFFSET = 1.5  # kept for API compat, unused in new code

_GL_RENDERER_AVAILABLE: Optional[bool] = None


def _gl_available() -> bool:
    """Test whether MuJoCo's offscreen GL renderer can be initialised."""
    global _GL_RENDERER_AVAILABLE
    if _GL_RENDERER_AVAILABLE is not None:
        return _GL_RENDERER_AVAILABLE
    if not _MUJOCO_AVAILABLE:
        _GL_RENDERER_AVAILABLE = False
        return False
    try:
        m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><geom size='0.1'/></body></worldbody></mujoco>"
        )
        r = mujoco.Renderer(m, height=8, width=8)
        r.close()
        _GL_RENDERER_AVAILABLE = True
    except Exception:
        _GL_RENDERER_AVAILABLE = False
    return _GL_RENDERER_AVAILABLE


# ── Helpers ──────────────────────────────────────────────────────────────────

def _included_to_flexion(deg: np.ndarray) -> np.ndarray:
    """Convert included-angle (180=straight) to flexion degrees (0=straight)."""
    return 180.0 - np.asarray(deg, dtype=np.float64)


def _pad_or_trim(arr: np.ndarray, T: int, default: float = 180.0) -> np.ndarray:
    """Ensure *arr* has exactly *T* elements."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if len(arr) >= T:
        return arr[:T]
    return np.concatenate([arr, np.full(T - len(arr), default)])


def _contact_events(frames: List[int], min_gap: int = 20) -> int:
    """Count distinct foot-strike events from a list of contact frame indices."""
    if not frames:
        return 0
    count, prev = 1, frames[0]
    for f in frames[1:]:
        if f - prev > min_gap:
            count += 1
        prev = f
    return count


def _gait_symmetry(right_frames: List[int], left_frames: List[int]) -> float:
    """0 = perfectly symmetric, 1 = maximally asymmetric."""
    def _intervals(frames):
        if len(frames) < 2:
            return np.array([], dtype=np.float64)
        u = sorted(set(frames))
        return np.array(
            [u[i + 1] - u[i] for i in range(len(u) - 1) if u[i + 1] - u[i] > 5],
            dtype=np.float64,
        )

    r, l = _intervals(right_frames), _intervals(left_frames)
    if r.size == 0 or l.size == 0:
        return 0.5
    mr, ml = float(r.mean()), float(l.mean())
    return abs(mr - ml) / max(mr + ml, 1e-9)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    """Collects per-frame simulation data for post-hoc analysis."""
    com_height: List[float]
    pred_knee: List[float]
    ref_knee: List[float]
    right_contact_frames: List[int]
    left_contact_frames: List[int]
    fall_detected: bool = False
    fall_frame: int = -1

    @classmethod
    def empty(cls) -> "EvalMetrics":
        return cls([], [], [], [], [])

    def to_dict(self) -> dict:
        pred = np.asarray(self.pred_knee, dtype=np.float64)
        ref = np.asarray(self.ref_knee, dtype=np.float64)
        com = np.asarray(self.com_height, dtype=np.float64)
        err = pred - ref

        rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0
        mae = float(np.mean(np.abs(err))) if err.size else 0.0
        com_mean = float(com.mean()) if com.size else 0.0
        com_std = float(com.std()) if com.size else 0.0

        sr = _contact_events(self.right_contact_frames)
        sl = _contact_events(self.left_contact_frames)
        sym = _gait_symmetry(self.right_contact_frames, self.left_contact_frames)

        stab = 1.0 - min(com_std / 0.25, 1.0) * 0.55 - sym * 0.25
        if self.fall_detected:
            stab -= 0.35
        stab = max(0.0, min(1.0, stab))

        return {
            "com_height_mean": com_mean,
            "com_height_std": com_std,
            "fall_detected": bool(self.fall_detected),
            "fall_frame": int(self.fall_frame),
            "knee_rmse_deg": rmse,
            "knee_mae_deg": mae,
            "step_count": int(sr + sl),
            "gait_symmetry": float(sym),
            "stability_score": float(stab),
            # Time-series for plotting (included-angle convention: 180=straight)
            "com_height_series": com.tolist(),
            "pred_knee_series": (180.0 - pred).tolist(),
            "ref_knee_series": (180.0 - ref).tolist(),
            "right_contact_frames": list(self.right_contact_frames),
            "left_contact_frames": list(self.left_contact_frames),
        }


@dataclass
class SimTrajectory:
    """Recorded simulation trajectory for replay / GIF rendering."""
    qpos_history: np.ndarray  # (T, nq)
    fps: float
    mjcf: str


# ── MJCF humanoid model ─────────────────────────────────────────────────────
#
# Full-body humanoid matching the CMU cgspeed BVH skeleton hierarchy.
# Root: 6 unactuated joints (3 slide + 3 hinge) — fully physics-driven.
# Body joints: position actuators (PD controllers) tracking mocap targets.
#
# Multi-DOF joints (hips, shoulders, spine) use separate hinges per axis.
# Collision: body geoms collide with floor only (not with each other).

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

      <!-- ── Right leg (hip: 3-DOF) ── -->
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

      <!-- ── Left leg (hip: 3-DOF) ── -->
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

      <!-- ── Spine chain (each segment: 3-DOF) ── -->
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

            <!-- Neck + Head -->
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

            <!-- ── Right arm (shoulder: 3-DOF) ── -->
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

            <!-- ── Left arm (shoulder: 3-DOF) ── -->
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
    <!-- 41 position actuators: body joints only.  Root is physics-driven. -->
    <!-- ctrl[0..11]: legs -->
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
    <!-- ctrl[12..22]: spine + neck + head -->
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
    <!-- ctrl[25..32]: shoulders + elbows -->
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
  </actuator>
</mujoco>
"""


# ── Joint mapping ────────────────────────────────────────────────────────────
#
# Maps each actuator (in MJCF declaration order) to a segment dict key and
# a conversion type:
#   "flex"  → included-angle → flexion → radians:  rad(180 - value)
#   "extra" → raw BVH degrees → radians:           rad(value)

_JOINT_MAP = [
    # (actuator_joint_name,      segment_key,                conv)
    # ── Legs ──
    ("right_hip",                "hip_right",                "flex"),
    ("right_hip_abd",            "hip_right_abd",            "extra"),
    ("right_hip_rot",            "hip_right_rot",            "extra"),
    ("left_hip",                 "hip_left",                 "flex"),
    ("left_hip_abd",             "hip_left_abd",             "extra"),
    ("left_hip_rot",             "hip_left_rot",             "extra"),
    ("right_knee",               "knee_right",               "flex"),
    ("left_knee",                "knee_left",                "flex"),
    ("right_ankle",              "ankle_right",              "flex"),
    ("left_ankle",               "ankle_left",               "flex"),
    ("right_toe",                "toe_right",                "flex"),
    ("left_toe",                 "toe_left",                 "flex"),
    # ── Spine chain ──
    ("lower_back",               "pelvis_tilt",              "flex"),
    ("lower_back_lat",           "pelvis_lateral",           "extra"),
    ("lower_back_rot",           "pelvis_rotation",          "extra"),
    ("spine_jnt",                "trunk_lean",               "flex"),
    ("spine_lat",                "trunk_lateral",            "extra"),
    ("spine_rot",                "trunk_rotation",           "extra"),
    ("spine1_jnt",               "upper_trunk",              "flex"),
    ("spine1_lat",               "upper_trunk_lateral",      "extra"),
    ("spine1_rot",               "upper_trunk_rotation",     "extra"),
    ("neck",                     "neck",                     "flex"),
    ("head_jnt",                 "head",                     "flex"),
    # ── Clavicles ──
    ("right_clav",               "clavicle_right",           "flex"),
    ("left_clav",                "clavicle_left",            "flex"),
    # ── Shoulders + elbows ──
    ("right_shoulder",           "shoulder_right",           "flex"),
    ("right_shoulder_abd",       "shoulder_right_abd",       "extra"),
    ("right_shoulder_rot",       "shoulder_right_rot",       "extra"),
    ("left_shoulder",            "shoulder_left",            "flex"),
    ("left_shoulder_abd",        "shoulder_left_abd",        "extra"),
    ("left_shoulder_rot",        "shoulder_left_rot",        "extra"),
    ("right_elbow",              "elbow_right",              "flex"),
    ("left_elbow",               "elbow_left",              "flex"),
    # ── Wrists ──
    ("right_wrist",              "wrist_right",              "flex"),
    ("left_wrist",               "wrist_left",               "flex"),
    # ── Fingers + thumbs ──
    ("right_finger",             "finger_right",             "flex"),
    ("right_fing_idx",           "finger_index_right",       "flex"),
    ("right_thumb",              "thumb_right",              "flex"),
    ("left_finger",              "finger_left",              "flex"),
    ("left_fing_idx",            "finger_index_left",        "flex"),
    ("left_thumb",               "thumb_left",               "flex"),
]

assert len(_JOINT_MAP) == N_ACT_PER_HUMANOID


def _build_controls(
    segment: dict,
    T: int,
    predicted_knee: np.ndarray,
    sample_thigh_right: Optional[np.ndarray],
) -> np.ndarray:
    """Build (T, N_ACT_PER_HUMANOID) control matrix in radians.

    *predicted_knee* overrides the right knee channel.
    *sample_thigh_right* (if given) overrides the right hip channel.
    Both are in included-angle degrees.
    """
    NA = N_ACT_PER_HUMANOID
    controls = np.zeros((T, NA), dtype=np.float64)

    pred_flex = _included_to_flexion(
        np.asarray(predicted_knee, dtype=np.float64)[:T])

    for i, (jnt_name, seg_key, conv) in enumerate(_JOINT_MAP):
        if jnt_name == "right_knee":
            # Right knee: driven by model prediction
            controls[:, i] = np.radians(_pad_or_trim(pred_flex, T, default=0.0))

        elif jnt_name == "right_hip" and sample_thigh_right is not None:
            # Right hip: optionally overridden by sample thigh angle
            hip_flex = _included_to_flexion(
                np.asarray(sample_thigh_right, dtype=np.float64)[:T])
            controls[:, i] = np.radians(_pad_or_trim(hip_flex, T, default=0.0))

        elif conv == "flex":
            raw = _pad_or_trim(
                segment.get(seg_key, np.full(T, 180.0)), T, default=180.0)
            controls[:, i] = np.radians(_included_to_flexion(raw))

        else:  # "extra"
            raw = _pad_or_trim(
                segment.get(seg_key, np.zeros(T)), T, default=0.0)
            controls[:, i] = np.radians(raw)

    return controls


# ── Core MuJoCo physics simulation ──────────────────────────────────────────

def _warmup(model, data, controls_frame0: np.ndarray) -> None:
    """Set initial pose from frame-0 controls and let physics settle."""
    # Set actuator targets
    data.ctrl[:N_ACT_PER_HUMANOID] = controls_frame0

    # Set body joint qpos directly to match targets (skip root joints)
    for i in range(min(model.nu, N_ACT_PER_HUMANOID)):
        jnt_id = model.actuator_trnid[i, 0]
        data.qpos[model.jnt_qposadr[jnt_id]] = controls_frame0[i]

    # Forward kinematics to update body positions
    mujoco.mj_forward(model, data)

    # Settle: let the body find ground contact
    for _ in range(500):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0

    # Brief second settle with zero velocity
    for _ in range(200):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0


def _run_sim(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
    fps: float,
    sample_thigh_right: Optional[np.ndarray],
    use_gui: bool,
) -> tuple:
    """Run MuJoCo physics. Returns (metrics, qpos_history, mjcf_str)."""
    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)

    # Frame count
    T = int(min(len(predicted_knee), len(mocap_segment["knee_right"])))
    if sample_thigh_right is not None:
        T = int(min(T, len(sample_thigh_right)))
    if T <= 0:
        return EvalMetrics.empty(), [], _MJCF

    # Build per-frame controls
    controls = _build_controls(
        mocap_segment, T, predicted_knee, sample_thigh_right)

    # Reference knee for metrics (flexion degrees)
    ref_flex = _included_to_flexion(
        np.asarray(mocap_segment["knee_right"], dtype=np.float64)[:T])
    pred_flex = _included_to_flexion(
        np.asarray(predicted_knee, dtype=np.float64)[:T])

    # Look up geom/body IDs
    ridx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_geom")
    lidx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_geom")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    dt = 1.0 / max(fps, 1.0)
    steps_per_frame = max(1, round(dt / model.opt.timestep))

    # ── Warmup ──
    _warmup(model, data, controls[0])

    # ── Step loop ──
    metrics = EvalMetrics.empty()
    qpos_history: List[np.ndarray] = []

    def _step_loop(viewer_obj=None):
        for t in tqdm(range(T), desc="Simulating", unit="fr", leave=False):
            if viewer_obj is not None and not viewer_obj.is_running():
                break

            data.ctrl[:N_ACT_PER_HUMANOID] = controls[t]

            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

            if viewer_obj is not None:
                viewer_obj.sync()
                time.sleep(dt)

            qpos_history.append(data.qpos.copy())

            # Metrics
            com_h = float(data.subtree_com[pelvis_id, 2])
            rc = lc = False
            for c in range(data.ncon):
                g1, g2 = int(data.contact[c].geom1), int(data.contact[c].geom2)
                if ridx in (g1, g2):
                    rc = True
                if lidx in (g1, g2):
                    lc = True

            metrics.com_height.append(com_h)
            metrics.pred_knee.append(float(pred_flex[t]))
            metrics.ref_knee.append(float(ref_flex[t]))
            if rc:
                metrics.right_contact_frames.append(t)
            if lc:
                metrics.left_contact_frames.append(t)
            if not metrics.fall_detected and com_h < FALL_HEIGHT_THRESHOLD:
                metrics.fall_detected = True
                metrics.fall_frame = t

    # Run with GUI or headless
    if use_gui and _VIEWER_AVAILABLE:
        try:
            with mujoco.viewer.launch_passive(
                model, data, show_left_ui=True, show_right_ui=True,
            ) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = pelvis_id
                viewer.cam.distance = 3.5
                viewer.cam.elevation = -15.0
                viewer.cam.azimuth = 90.0
                try:
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
                except Exception:
                    pass
                _step_loop(viewer)
        except Exception as exc:
            print(f"[sim] Viewer failed ({exc}), falling back to headless.")
            # Full reset for headless re-run
            data = mujoco.MjData(model)
            _warmup(model, data, controls[0])
            metrics = EvalMetrics.empty()
            qpos_history.clear()
            _step_loop(None)
    else:
        _step_loop(None)

    return metrics, qpos_history, _MJCF


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
        Accepted for API compatibility.  Reference comparison is done
        through the returned metrics / plots rather than a second
        in-scene humanoid.
    reference_knee :
        Accepted for API compatibility.

    Returns
    -------
    dict
        Evaluation metrics including time-series for plotting.
    """
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError(
            "MuJoCo is required but not installed.\n"
            "Install it with:  pip install mujoco"
        )

    metrics, qpos_history, mjcf_str = _run_sim(
        mocap_segment, predicted_knee, fps,
        sample_thigh_right=sample_thigh_right,
        use_gui=use_gui,
    )

    # If simulation produced no frames, fall back to kinematic eval
    if not metrics.com_height:
        result = run_kinematic_evaluation(
            mocap_segment, predicted_knee,
            sample_thigh_right=sample_thigh_right,
        )
        result["mode"] = "mujoco_physics_empty"
        return result

    result = metrics.to_dict()
    result["mode"] = "mujoco_physics"

    # Fall prediction
    com_arr = np.asarray(metrics.com_height, dtype=np.float64)
    result["fall_prediction"] = predict_fall(
        com_arr, fps=fps, fall_threshold=FALL_HEIGHT_THRESHOLD)

    # Save trajectory
    trajectory = None
    if qpos_history:
        trajectory = SimTrajectory(
            qpos_history=np.array(qpos_history),
            fps=fps,
            mjcf=mjcf_str,
        )

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


# ── Kinematic evaluator (fallback) ───────────────────────────────────────────

def run_kinematic_evaluation(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
    sample_thigh_right: Optional[np.ndarray] = None,
) -> dict:
    """Lightweight evaluation without physics — foot endpoint deviation."""
    ref = _included_to_flexion(np.asarray(mocap_segment["knee_right"], dtype=np.float64))
    pred = _included_to_flexion(np.asarray(predicted_knee, dtype=np.float64))
    T = min(len(ref), len(pred))
    if sample_thigh_right is not None:
        T = min(T, len(sample_thigh_right))
    if T <= 0:
        return {
            "com_height_mean": _INIT_HEIGHT,
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

    ref, pred = ref[:T], pred[:T]
    if sample_thigh_right is not None:
        hip = _included_to_flexion(np.asarray(sample_thigh_right, dtype=np.float64)[:T])
    else:
        hip = _included_to_flexion(np.asarray(
            mocap_segment.get("hip_right", np.full(T, 180.0)), dtype=np.float64)[:T])

    L1 = L2 = 0.45
    dev = np.zeros(T, dtype=np.float64)
    for i in range(T):
        h = math.radians(float(hip[i]))
        kr = math.radians(float(ref[i]))
        kp = math.radians(float(pred[i]))
        xr = L1 * math.sin(h) + L2 * math.sin(h - kr)
        zr = -L1 * math.cos(h) - L2 * math.cos(h - kr)
        xp = L1 * math.sin(h) + L2 * math.sin(h - kp)
        zp = -L1 * math.cos(h) - L2 * math.cos(h - kp)
        dev[i] = math.hypot(xp - xr, zp - zr)

    fall = bool(float(np.max(dev)) > 0.18)
    return {
        "com_height_mean": float(_INIT_HEIGHT - np.mean(0.5 * dev)),
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
    """Replay a recorded trajectory in the MuJoCo viewer (loops until closed)."""
    if not _MUJOCO_AVAILABLE or not _VIEWER_AVAILABLE:
        raise RuntimeError("MuJoCo with viewer is required for replay.")

    traj = (load_trajectory(trajectory_or_path)
            if isinstance(trajectory_or_path, (str, Path))
            else trajectory_or_path)

    model = mujoco.MjModel.from_xml_string(traj.mjcf)
    data = mujoco.MjData(model)
    dt = 1.0 / max(traj.fps, 1.0) / max(speed, 0.01)
    T = len(traj.qpos_history)

    print(f"[replay] {T} frames @ {traj.fps:.0f} Hz "
          f"(speed {speed:.1f}x, duration {T / traj.fps:.1f}s)")

    cur_speed = [speed]
    paused = [False]

    def _key_cb(key):
        if key == 32:         # Space
            paused[0] = not paused[0]
        elif key == 93:       # ]
            cur_speed[0] = min(cur_speed[0] * 2.0, 16.0)
        elif key == 91:       # [
            cur_speed[0] = max(cur_speed[0] * 0.5, 0.125)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=True,
        key_callback=_key_cb,
    ) as viewer:
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 90.0

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

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    camera.distance = 3.5
    camera.elevation = -15.0
    camera.azimuth = 90.0

    frames: List = []
    for t in tqdm(frame_indices, desc="Rendering GIF", unit="frame", leave=False):
        data.qpos[:] = traj.qpos_history[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(Image.fromarray(renderer.render()))

    renderer.close()

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
    print(f"[gif] Saved {len(frames)}-frame GIF ({T / traj.fps:.1f}s) -> {output_path}")


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

    traj = (load_trajectory(trajectory_or_path)
            if isinstance(trajectory_or_path, (str, Path))
            else trajectory_or_path)

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

    Drives every joint directly from BVH data — including right knee.
    No gravity, no contacts, no simulation.  Pure data playback.
    """
    if not _MUJOCO_AVAILABLE:
        print("[kinematic] MuJoCo not available.")
        return None

    model = mujoco.MjModel.from_xml_string(_MJCF)
    data = mujoco.MjData(model)

    T = len(mocap_segment.get("knee_right", []))
    if T == 0:
        return None

    # Build controls using mocap knee (not a prediction)
    controls = _build_controls(
        mocap_segment, T,
        predicted_knee=mocap_segment["knee_right"],  # use mocap knee
        sample_thigh_right=None,  # use mocap hip
    )

    # Root position and orientation from segment
    root_pos = mocap_segment.get("root_pos", np.zeros((T, 3)))
    if np.ndim(root_pos) == 2:
        root_pos = root_pos[:T]
    else:
        root_pos = np.zeros((T, 3))
    root_xy = root_pos[:, :2].copy()
    root_xy -= root_xy[0]

    root_ori = np.zeros((T, 3), dtype=np.float64)
    root_ori[:, 0] = np.radians(_pad_or_trim(
        mocap_segment.get("root_yaw", np.zeros(T)), T, default=0.0))
    root_ori[:, 1] = np.radians(-_pad_or_trim(
        mocap_segment.get("root_pitch", np.zeros(T)), T, default=0.0))
    root_ori[:, 2] = np.radians(_pad_or_trim(
        mocap_segment.get("root_roll", np.zeros(T)), T, default=0.0))
    root_ori -= root_ori[0]

    # Root joint qpos indices
    root_jnt_ids = {}
    for jname in ("root_x", "root_y", "root_z",
                   "root_yaw", "root_pitch", "root_roll"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        root_jnt_ids[jname] = model.jnt_qposadr[jid]

    # Kinematic playback
    qpos_history: List[np.ndarray] = []
    for t_idx in range(T):
        for i in range(model.nu):
            jnt_id = model.actuator_trnid[i, 0]
            data.qpos[model.jnt_qposadr[jnt_id]] = controls[t_idx, i]

        data.qpos[root_jnt_ids["root_x"]] = root_xy[t_idx, 0]
        data.qpos[root_jnt_ids["root_y"]] = root_xy[t_idx, 1]
        data.qpos[root_jnt_ids["root_yaw"]]   = root_ori[t_idx, 0]
        data.qpos[root_jnt_ids["root_pitch"]] = root_ori[t_idx, 1]
        data.qpos[root_jnt_ids["root_roll"]]  = root_ori[t_idx, 2]

        mujoco.mj_forward(model, data)
        qpos_history.append(data.qpos.copy())

    traj = SimTrajectory(
        qpos_history=np.array(qpos_history),
        fps=fps,
        mjcf=_MJCF,
    )

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
    """Render side-by-side GIF: left = kinematic mocap, right = physics sim."""
    if not _MUJOCO_AVAILABLE or not _gl_available():
        print("[side-by-side] Renderer not available, skipping.")
        return

    try:
        from PIL import Image
    except ImportError:
        print("[side-by-side] Pillow not installed, skipping.")
        return

    kinematic_traj = render_mocap_kinematic(mocap_segment, fps=fps)
    if kinematic_traj is None:
        return

    sim_traj = (load_trajectory(sim_trajectory)
                if isinstance(sim_trajectory, (str, Path))
                else sim_trajectory)

    T = min(len(kinematic_traj.qpos_history), len(sim_traj.qpos_history))
    skip = max(1, round(fps / gif_fps))
    frame_indices = list(range(0, T, skip))

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
    for t_idx in tqdm(frame_indices, desc="Rendering side-by-side",
                      unit="frame", leave=False):
        kin_data.qpos[:] = kinematic_traj.qpos_history[t_idx]
        mujoco.mj_forward(kin_model, kin_data)
        kin_renderer.update_scene(kin_data, camera=kin_cam)
        kin_pixels = kin_renderer.render()

        sim_data.qpos[:] = sim_traj.qpos_history[t_idx]
        mujoco.mj_forward(sim_model, sim_data)
        sim_renderer.update_scene(sim_data, camera=sim_cam)
        sim_pixels = sim_renderer.render()

        combined = np.concatenate([kin_pixels, sim_pixels], axis=1)
        frames.append(Image.fromarray(combined))

    kin_renderer.close()
    sim_renderer.close()

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
    print(f"[side-by-side] Saved {len(frames)}-frame GIF ({T / fps:.1f}s) -> {output_path}")


# ── Visualization ────────────────────────────────────────────────────────────

def visualize_mocap_segment(
    mocap_segment: dict,
    out_path: str | Path,
    fps: float = SIM_FPS_DEFAULT,
    title: str = "Loaded mocap segment",
) -> None:
    """Multi-panel plot showing ALL joint angles in the loaded segment."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T_frames = len(mocap_segment.get("knee_right", []))
    if T_frames == 0:
        return
    t = np.arange(T_frames) / fps

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
        ("Root position (metres, MuJoCo Z-up)", []),
    ]

    n_panels = len(groups)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3 * n_panels), sharex=True)

    for i, (group_title, keys) in enumerate(groups):
        ax = axes[i]
        if group_title.startswith("Root position"):
            rp = mocap_segment.get("root_pos", np.zeros((T_frames, 3)))
            if rp.ndim == 2 and rp.shape[1] >= 3:
                ax.plot(t, rp[:T_frames, 0], label="X (forward)", lw=1.2)
                ax.plot(t, rp[:T_frames, 1], label="Y (lateral)", lw=1.2)
                ax.plot(t, rp[:T_frames, 2], label="Z (height)", lw=1.2)
            for ch, lbl in [("root_pitch", "Pitch"), ("root_yaw", "Yaw"),
                            ("root_roll", "Roll")]:
                sig = mocap_segment.get(ch)
                if sig is not None:
                    ax2 = ax.twinx()
                    ax2.plot(t, sig[:T_frames], label=lbl, lw=0.8,
                             ls="--", alpha=0.5)
                    ax2.set_ylabel("Root orient (deg)", fontsize=7)
                    ax2.legend(loc="upper right", fontsize=6)
                    break
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
    fig.suptitle(f"{title}  ({T_frames} frames, {T_frames / fps:.2f}s @ {fps:.0f} Hz)",
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
    """Side-by-side comparison of simulation output vs kinematic mocap input."""
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
    ax.set_title(f"{title}  |  RMSE={sim_metrics.get('knee_rmse_deg', 0):.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: CoM height
    ax = axes[1]
    if com_h.size > 0:
        ax.plot(t[:len(com_h)], com_h[:T], lw=1.2, color="steelblue",
                label="CoM height")
        ax.axhline(FALL_HEIGHT_THRESHOLD, color="red", ls="--", alpha=0.5,
                   label=f"Fall threshold ({FALL_HEIGHT_THRESHOLD}m)")
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

    # Panel 3: Other joint angles for context
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


# ── Fall prediction ──────────────────────────────────────────────────────────

def predict_fall(
    com_height_series: np.ndarray,
    fps: float = SIM_FPS_DEFAULT,
    fall_threshold: float = FALL_HEIGHT_THRESHOLD,
    lookback_s: float = 0.5,
) -> dict:
    """Predict whether a fall is imminent based on CoM height trend."""
    com = np.asarray(com_height_series, dtype=np.float64)
    T = len(com)

    _default = {
        "predicted_fall": False,
        "time_to_fall_s": float("inf"),
        "confidence": 0.0,
        "com_velocity": 0.0,
        "com_acceleration": 0.0,
        "min_com": float(com.min()) if T > 0 else 1.0,
        "trend_slope": 0.0,
    }
    if T < 10:
        return _default

    # Trend from last lookback_s seconds
    lb = min(T, max(10, int(lookback_s * fps)))
    tail = com[-lb:]
    t_tail = np.arange(lb) / fps

    slope = float(np.polyfit(t_tail, tail, 1)[0]) if lb > 1 else 0.0

    # Instantaneous velocity and acceleration
    if T >= 3:
        dt = 1.0 / fps
        vel = np.diff(com) / dt
        acc = np.diff(vel) / dt
        cur_vel = float(vel[-1])
        cur_acc = float(np.mean(acc[-min(20, len(acc)):]))
    else:
        cur_vel = cur_acc = 0.0

    min_com = float(com.min())
    cur_com = float(com[-1])
    already_fell = cur_com < fall_threshold

    # Time-to-fall via linear extrapolation
    if slope < -0.001:
        ttf_linear = (fall_threshold - cur_com) / slope
        if ttf_linear < 0:
            ttf_linear = 0.0
    else:
        ttf_linear = float("inf")

    # Quadratic extrapolation
    ttf_quad = float("inf")
    if cur_acc < -0.1 or cur_vel < -0.05:
        a = 0.5 * cur_acc
        b = cur_vel
        c = cur_com - fall_threshold
        if abs(a) > 1e-6:
            disc = b * b - 4 * a * c
            if disc >= 0:
                roots = [(-b - math.sqrt(disc)) / (2 * a),
                         (-b + math.sqrt(disc)) / (2 * a)]
                pos = [r for r in roots if r > 0]
                if pos:
                    ttf_quad = min(pos)
        elif b < -0.01:
            ttf_quad = -c / b

    best_ttf = min(ttf_linear, ttf_quad)

    # Confidence
    if already_fell:
        confidence = 1.0
    else:
        vel_factor = min(1.0, max(0.0, -cur_vel / 0.5))
        acc_factor = min(1.0, max(0.0, -cur_acc / 5.0))
        prox_factor = max(0.0, 1.0 - (cur_com - fall_threshold) / 0.5)
        trend_factor = min(1.0, max(0.0, -slope / 0.3))
        confidence = min(1.0, (
            vel_factor * 0.3 + acc_factor * 0.2 +
            prox_factor * 0.25 + trend_factor * 0.25
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
