"""Prosthetic gait simulation / evaluation.

Drives a full humanoid skeleton from motion-capture data via MuJoCo physics.
All joints track their mocap reference angles *except* the right knee, which
is driven by the model's predicted knee angle, and the right hip, which is
driven by the sample's thigh_angle -- allowing evaluation of how well the
prediction maintains stable, natural gait.

The torso root uses slide joints for translation (X, Y, Z) and three hinge
joints for orientation (yaw, pitch, roll).  X and Y are driven by stiff
position actuators (kp=2000, damping=200) from BVH root trajectories.
Yaw, pitch, and roll are driven by position actuators (kp=500, damping=50)
from BVH root orientation.  The Z slide joint has **no actuator** and
minimal damping (2), so the body moves vertically under gravity alone and
can fall naturally when gait is unstable.

All joint actuators use near-critically-damped PD control (high kp with
matched joint damping) so the skeleton tracks the mocap reference smoothly
without oscillation.

Supports:
- MuJoCo physics backend with full-body mocap-driven humanoid
- Optional reference humanoid (semi-transparent) showing ground-truth mocap
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
# ALL BVH joints are represented and driven from mocap data, with the sole
# exception of the right knee (model prediction) and optionally the right
# hip (sample thigh_angle).
#
# Body hierarchy:
#   Pelvis (root: slide X/Y/Z + hinge yaw/pitch/roll)
#   ├── Right leg: hip → knee → ankle → toe
#   ├── Left leg:  hip → knee → ankle → toe
#   └── Spine chain: LowerBack → Spine → Spine1
#       ├── Neck → Head
#       ├── Right arm: Clavicle → Shoulder → Elbow → Wrist → Fingers/Thumb
#       └── Left arm:  Clavicle → Shoulder → Elbow → Wrist → Fingers/Thumb
#
# Collision: body geoms collide with floor but NOT with each other
# (contype=1/conaffinity=2 for body, contype=2/conaffinity=1 for floor).

# Number of actuators per humanoid (prediction or reference)
N_ACT_PER_HUMANOID = 32

_MJCF = """
<mujoco model="prosthetic_eval_humanoid">
  <compiler angle="degree"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicit"/>

  <default>
    <joint damping="50" armature="0.02"/>
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
      <joint name="root_x" type="slide" axis="1 0 0" damping="200"/>
      <joint name="root_y" type="slide" axis="0 1 0" damping="200"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="2"/>
      <joint name="root_yaw"   type="hinge" axis="0 0 1" damping="50"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" damping="50"/>
      <joint name="root_roll"  type="hinge" axis="1 0 0" damping="50"/>
      <geom type="capsule" fromto="0 0 -0.12 0 0 0" size="0.085"
            rgba="0.6 0.6 0.65 1"/>

      <!-- ── Right leg ── -->
      <body name="right_thigh" pos="0 -0.10 -0.12">
        <joint name="right_hip" type="hinge" axis="0 -1 0" range="-70 70" damping="80"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05"/>
        <body name="right_shank" pos="0 0 -0.42">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="0 130" damping="70"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.045"
                rgba="1 0.5 0.1 1"/>
          <body name="right_foot" pos="0 0 -0.42">
            <joint name="right_ankle" type="hinge" axis="0 -1 0"
                   range="-50 40" damping="40"/>
            <geom name="right_foot_geom" type="box"
                  size="0.11 0.05 0.03" pos="0.07 0 -0.02"
                  rgba="1 0.5 0.1 1"/>
            <body name="right_toe" pos="0.18 0 -0.02">
              <joint name="right_toe" type="hinge" axis="0 -1 0"
                     range="-30 45" damping="20"/>
              <geom type="box" size="0.04 0.04 0.02"
                    rgba="1 0.5 0.1 1"/>
            </body>
          </body>
        </body>
      </body>

      <!-- ── Left leg ── -->
      <body name="left_thigh" pos="0 0.10 -0.12">
        <joint name="left_hip" type="hinge" axis="0 -1 0" range="-70 70" damping="80"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.05"/>
        <body name="left_shank" pos="0 0 -0.42">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="0 130" damping="70"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.42" size="0.045"/>
          <body name="left_foot" pos="0 0 -0.42">
            <joint name="left_ankle" type="hinge" axis="0 -1 0"
                   range="-50 40" damping="40"/>
            <geom name="left_foot_geom" type="box"
                  size="0.11 0.05 0.03" pos="0.07 0 -0.02"/>
            <body name="left_toe" pos="0.18 0 -0.02">
              <joint name="left_toe" type="hinge" axis="0 -1 0"
                     range="-30 45" damping="20"/>
              <geom type="box" size="0.04 0.04 0.02"/>
            </body>
          </body>
        </body>
      </body>

      <!-- ── Spine chain (LowerBack → Spine → Spine1) ── -->
      <body name="lower_back" pos="0 0 0">
        <joint name="lower_back" type="hinge" axis="0 1 0"
               range="-30 30"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.07" size="0.075"
              rgba="0.6 0.6 0.65 1"/>
        <body name="spine" pos="0 0 0.07">
          <joint name="spine_jnt" type="hinge" axis="0 1 0"
                 range="-30 30"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.07"
                rgba="0.6 0.6 0.65 1"/>
          <body name="spine1" pos="0 0 0.06">
            <joint name="spine1_jnt" type="hinge" axis="0 1 0"
                   range="-30 30"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.06" size="0.065"
                  rgba="0.6 0.6 0.65 1"/>

            <!-- Neck → Head -->
            <body name="neck_body" pos="0 0 0.08">
              <joint name="neck" type="hinge" axis="0 1 0"
                     range="-30 30" damping="10"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.04" size="0.035"
                    rgba="0.85 0.75 0.65 1"/>
              <body name="head" pos="0 0 0.06">
                <joint name="head_jnt" type="hinge" axis="0 1 0"
                       range="-30 30" damping="5"/>
                <geom type="sphere" size="0.09" rgba="0.85 0.75 0.65 1"/>
              </body>
            </body>

            <!-- ── Right arm (clavicle → shoulder → elbow → wrist → fingers) ── -->
            <body name="right_clavicle" pos="0 -0.12 0.03">
              <joint name="right_clav" type="hinge" axis="0 -1 0"
                     range="-20 20" damping="2"/>
              <geom type="capsule" fromto="0 0 0 0 -0.08 0" size="0.025"/>
              <body name="right_upper_arm" pos="0 -0.10 0">
                <joint name="right_shoulder" type="hinge" axis="0 -1 0"
                       range="-90 90" damping="2"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.28" size="0.03"/>
                <body name="right_forearm" pos="0 0 -0.28">
                  <joint name="right_elbow" type="hinge" axis="0 -1 0"
                         range="0 130" damping="1"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025"
                        rgba="0.85 0.75 0.65 1"/>
                  <body name="right_hand" pos="0 0 -0.25">
                    <joint name="right_wrist" type="hinge" axis="0 -1 0"
                           range="-60 60" damping="1"/>
                    <geom type="box" size="0.035 0.015 0.05" pos="0 0 -0.05"
                          rgba="0.85 0.75 0.65 1"/>
                    <body name="right_fing_base" pos="0 0 -0.08">
                      <joint name="right_finger" type="hinge" axis="0 -1 0"
                             range="0 90" damping="0.5"/>
                      <geom type="capsule" fromto="0 0 0 0 0 -0.025"
                            size="0.008" rgba="0.85 0.75 0.65 1"/>
                      <body name="right_fing_tip" pos="0 0 -0.025">
                        <joint name="right_fing_idx" type="hinge" axis="0 -1 0"
                               range="0 90" damping="0.5"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.015"
                              size="0.006" rgba="0.85 0.75 0.65 1"/>
                      </body>
                    </body>
                    <body name="right_thumb_body" pos="0.01 0.015 -0.03">
                      <joint name="right_thumb" type="hinge" axis="0 0 1"
                             range="-30 60" damping="0.5"/>
                      <geom type="capsule" fromto="0 0 0 0.015 0.01 -0.02"
                            size="0.007" rgba="0.85 0.75 0.65 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>

            <!-- ── Left arm (clavicle → shoulder → elbow → wrist → fingers) ── -->
            <body name="left_clavicle" pos="0 0.12 0.03">
              <joint name="left_clav" type="hinge" axis="0 -1 0"
                     range="-20 20" damping="2"/>
              <geom type="capsule" fromto="0 0 0 0 0.08 0" size="0.025"/>
              <body name="left_upper_arm" pos="0 0.10 0">
                <joint name="left_shoulder" type="hinge" axis="0 -1 0"
                       range="-90 90" damping="2"/>
                <geom type="capsule" fromto="0 0 0 0 0 -0.28" size="0.03"/>
                <body name="left_forearm" pos="0 0 -0.28">
                  <joint name="left_elbow" type="hinge" axis="0 -1 0"
                         range="0 130" damping="1"/>
                  <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.025"
                        rgba="0.85 0.75 0.65 1"/>
                  <body name="left_hand" pos="0 0 -0.25">
                    <joint name="left_wrist" type="hinge" axis="0 -1 0"
                           range="-60 60" damping="1"/>
                    <geom type="box" size="0.035 0.015 0.05" pos="0 0 -0.05"
                          rgba="0.85 0.75 0.65 1"/>
                    <body name="left_fing_base" pos="0 0 -0.08">
                      <joint name="left_finger" type="hinge" axis="0 -1 0"
                             range="0 90" damping="0.5"/>
                      <geom type="capsule" fromto="0 0 0 0 0 -0.025"
                            size="0.008" rgba="0.85 0.75 0.65 1"/>
                      <body name="left_fing_tip" pos="0 0 -0.025">
                        <joint name="left_fing_idx" type="hinge" axis="0 -1 0"
                               range="0 90" damping="0.5"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.015"
                              size="0.006" rgba="0.85 0.75 0.65 1"/>
                      </body>
                    </body>
                    <body name="left_thumb_body" pos="0.01 -0.015 -0.03">
                      <joint name="left_thumb" type="hinge" axis="0 0 -1"
                             range="-30 60" damping="0.5"/>
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
    <!-- ctrl[0..7]: legs + toes -->
    <position joint="right_hip"      kp="400"/>
    <position joint="left_hip"       kp="400"/>
    <position joint="right_knee"     kp="400"/>
    <position joint="left_knee"      kp="400"/>
    <position joint="right_ankle"    kp="300"/>
    <position joint="left_ankle"     kp="300"/>
    <position joint="right_toe"      kp="150"/>
    <position joint="left_toe"       kp="150"/>
    <!-- ctrl[8..10]: spine chain -->
    <position joint="lower_back"     kp="300"/>
    <position joint="spine_jnt"      kp="300"/>
    <position joint="spine1_jnt"     kp="300"/>
    <!-- ctrl[11..12]: neck + head -->
    <position joint="neck"           kp="150"/>
    <position joint="head_jnt"       kp="100"/>
    <!-- ctrl[13..14]: clavicles -->
    <position joint="right_clav"     kp="100"/>
    <position joint="left_clav"      kp="100"/>
    <!-- ctrl[15..18]: shoulders + elbows -->
    <position joint="right_shoulder" kp="150"/>
    <position joint="left_shoulder"  kp="150"/>
    <position joint="right_elbow"    kp="100"/>
    <position joint="left_elbow"     kp="100"/>
    <!-- ctrl[19..20]: wrists -->
    <position joint="right_wrist"    kp="80"/>
    <position joint="left_wrist"     kp="80"/>
    <!-- ctrl[21..26]: fingers + thumbs -->
    <position joint="right_finger"   kp="30"/>
    <position joint="right_fing_idx" kp="30"/>
    <position joint="right_thumb"    kp="30"/>
    <position joint="left_finger"    kp="30"/>
    <position joint="left_fing_idx"  kp="30"/>
    <position joint="left_thumb"     kp="30"/>
    <!-- ctrl[27..28]: root XY tracking (Z has no actuator — free for gravity) -->
    <position joint="root_x" kp="2000"/>
    <position joint="root_y" kp="2000"/>
    <!-- ctrl[29..31]: root orientation from mocap -->
    <position joint="root_yaw"   kp="500"/>
    <position joint="root_pitch" kp="500"/>
    <position joint="root_roll"  kp="500"/>
  </actuator>
</mujoco>
"""


def _build_dual_mjcf() -> str:
    """Build MJCF with both prediction humanoid and a reference humanoid.

    The reference model has the same full-body skeleton as the prediction
    model, semi-transparent blue, offset laterally by REF_Y_OFFSET.
    It uses a text-replace approach: prefix all names with ``ref_``,
    change colours, and add the reference actuator block.
    """
    import re
    ref_y = REF_Y_OFFSET
    rb = "0.3 0.5 0.8 0.5"  # reference blue (semi-transparent)
    rg = "0.2 0.7 0.3 0.5"  # reference green (prosthetic side)

    # Extract just the <body name="pelvis" ...> ... </body> block from _MJCF
    # (everything between the second <body and its matching </body> before </worldbody>)
    body_start = _MJCF.index('<body name="pelvis"')
    # Find the </body> that closes the pelvis — it's the last </body> before </worldbody>
    wb_end = _MJCF.index("</worldbody>")
    # Count nesting to find the matching close tag
    body_xml = _MJCF[body_start:wb_end].rstrip()
    # Remove trailing whitespace/newlines after the last </body>
    while body_xml.endswith("\n") or body_xml.endswith(" "):
        body_xml = body_xml.rstrip()

    # Prefix all name="..." and joint="..." values with ref_
    def _prefix_attr(match):
        attr = match.group(1)  # name or joint
        val = match.group(2)
        return f'{attr}="ref_{val}"'
    ref_xml = re.sub(r'(name|joint)="([^"]+)"', _prefix_attr, body_xml)

    # Change the root position to offset laterally
    ref_xml = ref_xml.replace(
        'name="ref_pelvis" pos="0 0 1.05"',
        f'name="ref_pelvis" pos="0 {ref_y} 1.05"',
    )

    # Replace all colours with semi-transparent blue/green
    ref_xml = re.sub(r'rgba="1 0\.5 0\.1 1"', f'rgba="{rg}"', ref_xml)  # orange → green
    ref_xml = re.sub(r'rgba="0\.6 0\.6 0\.65 1"', f'rgba="{rb}"', ref_xml)  # grey → blue
    ref_xml = re.sub(r'rgba="0\.85 0\.75 0\.65 1"', f'rgba="{rb}"', ref_xml)  # skin → blue

    ref_body_block = (
        "\n    <!-- ══ Reference humanoid (ground-truth mocap) "
        "═══════════════════════ -->\n    "
        + ref_xml + "\n"
    )

    # Build reference actuator block (same joint order, prefixed names)
    ref_actuators = """    <!-- Reference model actuators -->
    <position joint="ref_right_hip"      kp="400"/>
    <position joint="ref_left_hip"       kp="400"/>
    <position joint="ref_right_knee"     kp="400"/>
    <position joint="ref_left_knee"      kp="400"/>
    <position joint="ref_right_ankle"    kp="300"/>
    <position joint="ref_left_ankle"     kp="300"/>
    <position joint="ref_right_toe"      kp="150"/>
    <position joint="ref_left_toe"       kp="150"/>
    <position joint="ref_lower_back"     kp="300"/>
    <position joint="ref_spine_jnt"      kp="300"/>
    <position joint="ref_spine1_jnt"     kp="300"/>
    <position joint="ref_neck"           kp="150"/>
    <position joint="ref_head_jnt"       kp="100"/>
    <position joint="ref_right_clav"     kp="100"/>
    <position joint="ref_left_clav"      kp="100"/>
    <position joint="ref_right_shoulder" kp="150"/>
    <position joint="ref_left_shoulder"  kp="150"/>
    <position joint="ref_right_elbow"    kp="100"/>
    <position joint="ref_left_elbow"     kp="100"/>
    <position joint="ref_right_wrist"    kp="80"/>
    <position joint="ref_left_wrist"     kp="80"/>
    <position joint="ref_right_finger"   kp="30"/>
    <position joint="ref_right_fing_idx" kp="30"/>
    <position joint="ref_right_thumb"    kp="30"/>
    <position joint="ref_left_finger"    kp="30"/>
    <position joint="ref_left_fing_idx"  kp="30"/>
    <position joint="ref_left_thumb"     kp="30"/>
    <position joint="ref_root_x" kp="2000"/>
    <position joint="ref_root_y" kp="2000"/>
    <position joint="ref_root_yaw"   kp="500"/>
    <position joint="ref_root_pitch" kp="500"/>
    <position joint="ref_root_roll"  kp="500"/>"""

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

        # Root orientation from BVH (raw degrees — NOT included-angle)
        root_pitch = _pad_or_trim(
            mocap_segment.get("root_pitch", np.zeros(T)), T, default=0.0)
        root_yaw = _pad_or_trim(
            mocap_segment.get("root_yaw", np.zeros(T)), T, default=0.0)
        root_roll = _pad_or_trim(
            mocap_segment.get("root_roll", np.zeros(T)), T, default=0.0)
        # Normalise root yaw so frame-0 starts at 0 (no absolute heading)
        root_yaw = root_yaw - root_yaw[0]

        # ── Root position from BVH (MuJoCo Z-up coords, metres) ─────────
        raw_root = mocap_segment.get("root_pos", np.zeros((T, 3)))
        if raw_root.shape[0] < T:
            pad_rows = np.tile(raw_root[-1:], (T - raw_root.shape[0], 1))
            raw_root = np.vstack([raw_root, pad_rows])
        raw_root = raw_root[:T].astype(np.float64)

        has_root_motion = np.any(np.abs(raw_root - raw_root[0]) > 0.001)
        if has_root_motion:
            root_pos = raw_root - raw_root[0] + HUMANOID_INIT_POS
        else:
            root_pos = np.tile(HUMANOID_INIT_POS, (T, 1))

        # ── Pre-compute controls (radians) ────────────────────────────────
        # Actuator order (N_ACT_PER_HUMANOID = 32 per humanoid):
        #  0: right_hip      1: left_hip       2: right_knee(PRED)  3: left_knee
        #  4: right_ankle    5: left_ankle     6: right_toe         7: left_toe
        #  8: lower_back     9: spine         10: spine1
        # 11: neck          12: head
        # 13: right_clav    14: left_clav
        # 15: right_shoulder 16: left_shoulder 17: right_elbow      18: left_elbow
        # 19: right_wrist   20: left_wrist
        # 21: right_finger  22: right_fing_idx 23: right_thumb
        # 24: left_finger   25: left_fing_idx  26: left_thumb
        # 27: root_x        28: root_y
        # 29: root_yaw      30: root_pitch     31: root_roll
        NA = N_ACT_PER_HUMANOID
        n_act = NA * 2 if self.show_reference else NA
        controls = np.zeros((T, n_act), dtype=np.float64)

        def _fill_humanoid(off, knee_signal):
            """Fill control columns [off : off+NA] for one humanoid."""
            controls[:, off + 0] = np.radians(hip_r)
            controls[:, off + 1] = np.radians(_flex("hip_left"))
            controls[:, off + 2] = np.radians(knee_signal)
            controls[:, off + 3] = np.radians(_flex("knee_left"))
            controls[:, off + 4] = np.radians(_flex("ankle_right"))
            controls[:, off + 5] = np.radians(_flex("ankle_left"))
            controls[:, off + 6] = np.radians(_flex("toe_right"))
            controls[:, off + 7] = np.radians(_flex("toe_left"))
            controls[:, off + 8] = np.radians(_flex("pelvis_tilt"))
            controls[:, off + 9] = np.radians(_flex("trunk_lean"))
            controls[:, off + 10] = np.radians(_flex("upper_trunk"))
            controls[:, off + 11] = np.radians(_flex("neck"))
            controls[:, off + 12] = np.radians(_flex("head"))
            controls[:, off + 13] = np.radians(_flex("clavicle_right"))
            controls[:, off + 14] = np.radians(_flex("clavicle_left"))
            controls[:, off + 15] = np.radians(_flex("shoulder_right"))
            controls[:, off + 16] = np.radians(_flex("shoulder_left"))
            controls[:, off + 17] = np.radians(_flex("elbow_right"))
            controls[:, off + 18] = np.radians(_flex("elbow_left"))
            controls[:, off + 19] = np.radians(_flex("wrist_right"))
            controls[:, off + 20] = np.radians(_flex("wrist_left"))
            controls[:, off + 21] = np.radians(_flex("finger_right"))
            controls[:, off + 22] = np.radians(_flex("finger_index_right"))
            controls[:, off + 23] = np.radians(_flex("thumb_right"))
            controls[:, off + 24] = np.radians(_flex("finger_left"))
            controls[:, off + 25] = np.radians(_flex("finger_index_left"))
            controls[:, off + 26] = np.radians(_flex("thumb_left"))
            controls[:, off + 27] = root_pos[:, 0]   # metres, not radians
            controls[:, off + 28] = root_pos[:, 1]
            controls[:, off + 29] = np.radians(root_yaw)
            controls[:, off + 30] = np.radians(root_pitch)
            controls[:, off + 31] = np.radians(root_roll)

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

        # ── Warmup: let actuators settle to frame-0 pose ────────────────
        # 1000 steps (1.0 s) to fully settle joints + body height.
        data.ctrl[:] = controls[0]
        for _ in range(1000):
            mujoco.mj_step(model, data)

        # Reset metrics after warmup
        metrics = EvalMetrics.empty()

        def _step_loop(viewer_obj=None):
            for t in tqdm(range(T), desc="Simulating", unit="step", leave=False):
                if viewer_obj is not None and not viewer_obj.is_running():
                    break

                # Drive all actuators (joints + root XY)
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
                # Use prediction model's subtree CoM (not world-body
                # subtree which would include the reference humanoid).
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
            """Loop the recorded trajectory until the viewer is closed.

            Keyboard controls (active during replay):
              Space  — pause / resume
              ]      — speed up (2×)
              [      — slow down (0.5×)
              R      — reset to frame 0
            """
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
                    torso_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                    viewer_obj.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer_obj.cam.trackbodyid = torso_id
                    viewer_obj.cam.distance = 4.5 if self.show_reference else 3.5
                    viewer_obj.cam.elevation = -15.0
                    viewer_obj.cam.azimuth = 90.0

                    # Enable useful visualisation flags so rendering
                    # options in the viewer are active, not greyed out.
                    try:
                        viewer_obj.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                        viewer_obj.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
                    except Exception:
                        pass  # older MuJoCo versions may not have all flags

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
                for _ in range(1000):
                    mujoco.mj_step(model, data)
                metrics = EvalMetrics.empty()
                qpos_history.clear()
                _step_loop(None)
        else:
            _step_loop(None)

        # ── Build result ─────────────────────────────────────────────────
        out = metrics.to_dict()
        out["mode"] = "mujoco_physics" + ("+gui" if gui_worked else "")

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
        the matched mocap hip_right.  Keeps both right-leg inputs
        (thigh + knee) anchored to the evaluated sample.
    save_trajectory :
        Path to save the recorded trajectory (.npz) for later replay.
    render_gif :
        Path to save an animated GIF of the simulation.
    show_reference :
        When True, show a semi-transparent reference humanoid alongside
        the prediction model.  The reference follows all mocap joints
        with its right knee driven by *reference_knee* (or the matched
        mocap knee if *reference_knee* is None).  Works in both GUI and
        headless modes (GIF renders and trajectory files will include
        both humanoids).
    reference_knee :
        Right knee signal (included-angle, degrees) for the reference
        humanoid.  Only used when *show_reference* is True.  Typically
        set to the ground-truth label so the viewer can compare GT vs
        prediction side by side.

    Returns
    -------
    dict
        Evaluation metrics (same keys as before), plus ``"trajectory_path"``
        and/or ``"gif_path"`` if those outputs were requested.
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

    Parameters
    ----------
    trajectory_or_path :
        A SimTrajectory object or path to a saved .npz file.
    speed :
        Playback speed multiplier (1.0 = real-time).
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

        # Enable useful visualisation options
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
    try:
        from PIL import Image
    except ImportError:
        print("[gif] Pillow not installed, skipping GIF render.")
        return

    try:
        model = mujoco.MjModel.from_xml_string(traj.mjcf)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)
    except Exception as exc:
        print(f"[gif] Renderer init failed ({exc}), skipping GIF.")
        return

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
    """Render a saved or in-memory trajectory to an animated GIF.

    Parameters
    ----------
    trajectory_or_path :
        A SimTrajectory object or path to a saved .npz file.
    output_path :
        Destination for the .gif file.
    width, height :
        Resolution of each rendered frame.
    gif_fps :
        Target frame rate for the GIF (frames are subsampled from the
        full simulation rate).
    """
    if not _MUJOCO_AVAILABLE:
        raise RuntimeError("MuJoCo is required for GIF rendering.")

    if isinstance(trajectory_or_path, (str, Path)):
        traj = load_trajectory(trajectory_or_path)
    else:
        traj = trajectory_or_path

    _render_gif(traj, Path(output_path), width=width, height=height,
                gif_fps=gif_fps)


# ── Demo ─────────────────────────────────────────────────────────────────────

def run_visual_demo(use_full_db: bool = False):
    from mocap_evaluation.mocap_loader import load_aggregated_database
    from mocap_evaluation.motion_matching import find_best_match

    db = load_aggregated_database()
    T = int(min(600, len(db["knee_right"])))
    qk = db["knee_right"][:T]
    qt = db["hip_right"][:T]
    _, _, seg = find_best_match(qk, qt, db)

    out = simulate_prosthetic_walking(
        seg, qk, use_gui=True,
        render_gif="demo_simulation.gif",
        show_reference=True,
    )
    print("Demo metrics:", {k: v for k, v in out.items()
                           if not k.endswith("_series") and not k.endswith("_frames")})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prosthetic simulation tools")
    sub = ap.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run a visual demo with mocap data")

    rp = sub.add_parser("replay", help="Replay a saved trajectory")
    rp.add_argument("path", help="Path to .npz trajectory file")
    rp.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed (default: 1.0)")

    gp = sub.add_parser("gif", help="Render a trajectory to GIF")
    gp.add_argument("path", help="Path to .npz trajectory file")
    gp.add_argument("--out", default=None, help="Output GIF path")
    gp.add_argument("--width", type=int, default=640)
    gp.add_argument("--height", type=int, default=480)
    gp.add_argument("--fps", type=int, default=30, help="GIF frame rate")

    args = ap.parse_args()

    if args.command == "demo" or args.command is None:
        run_visual_demo()
    elif args.command == "replay":
        replay_trajectory(args.path, speed=args.speed)
    elif args.command == "gif":
        out = args.out or Path(args.path).with_suffix(".gif")
        render_simulation_gif(args.path, out, width=args.width,
                              height=args.height, gif_fps=args.fps)
