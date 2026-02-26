"""
PyBullet prosthetic walking simulator.

Design
------
Each robot is simulated **individually** (one robot per GIF).

The predicted-knee robot uses PD position control on all lower-body joints:
the right knee target comes from the model's prediction while every other
joint target comes from the motion-captured reference.  Because the PD gains
are finite and the base is free under gravity, the humanoid will maintain
balance when predictions are accurate and fall when they deviate
significantly — giving a physically grounded evaluation.

The ground-truth robot uses kinematic playback (all joints from mocap) so it
always walks correctly and serves as a visual reference.

Joint layout (pybullet_data humanoid/humanoid.urdf)
---------------------------------------------------
The URDF is Y-up; we apply a 90-degree rotation around X at load time so the
humanoid stands upright in PyBullet's Z-up world.

  Revolute  : right_knee, left_knee, right_elbow, left_elbow
  Spherical : chest, neck, right_hip, left_hip, right_ankle, left_ankle,
              right_shoulder, left_shoulder

Knee joints have axis (0,0,1) and limits [-pi, 0]: biomechanical flexion
(positive) maps to **negative** URDF angles.

Stability metrics
-----------------
  com_height_mean  : average CoM height (m)
  com_height_std   : standard deviation — higher = more bobbing/instability
  fall_detected    : True if CoM height drops below fall_threshold
  fall_frame       : frame index of first fall (-1 if no fall)
  knee_rmse_deg    : RMSE between predicted and mocap right-knee angle (degrees)
  knee_mae_deg     : MAE  between predicted and mocap right-knee angle
  step_count       : number of completed right-foot ground contacts
  gait_symmetry    : |mean_right_step_time - mean_left_step_time| / mean_step_time
                     0 = perfect symmetry, 1 = completely asymmetric
  stability_score  : composite 0-1 (higher = more stable)
"""
from __future__ import annotations

import math
import os
import time as _time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pybullet as p
    import pybullet_data
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False

try:
    from scipy.spatial.transform import Rotation
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

# The humanoid URDF is Y-up.  Rotate 90° around X to stand upright in Z-up.
_BASE_QUAT = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]

# Base Z position so the humanoid's feet sit on the ground plane (Z=0).
HUMANOID_INIT_HEIGHT = 3.5

# Centre-of-mass is ~3.5 m at rest; a fall brings it below this threshold.
FALL_HEIGHT_THRESHOLD = 2.5

SIM_TIMESTEP = 1.0 / 200.0   # seconds per physics step (match 200 Hz)

# PD gains for position control.
KP_REVOLUTE   = 0.4
KD_REVOLUTE   = 4.0
MAX_FORCE_REV = 1000.0

KP_SPHERICAL  = 0.4
KD_SPHERICAL  = 4.0
MAX_FORCE_SPH = 1000.0

# External force spring to guide the base horizontally along the trajectory.
_BASE_GUIDE_KP = 300.0
_BASE_GUIDE_KD = 100.0

# Lower-body joint name substrings we actively control
_LOWER_BODY_KEYWORDS = (
    "hip", "knee", "ankle", "foot"
)

# GIF frame capture: record every Nth simulation frame
_CAPTURE_EVERY = 8   # 200 Hz / 8 = 25 fps effective in the saved video


# ── Helper: Euler → quaternion ────────────────────────────────────────────────


def _euler_to_quat_zxy(z_deg: float, x_deg: float, y_deg: float) -> List[float]:
    """
    Convert CMU BVH-style ZXY Euler angles (degrees) to quaternion [x,y,z,w].
    Falls back to identity if scipy unavailable.
    """
    if _SCIPY_AVAILABLE:
        r = Rotation.from_euler("ZXY", [z_deg, x_deg, y_deg], degrees=True)
        q = r.as_quat()   # [x, y, z, w]
        return q.tolist()
    # Fallback: approximate as single-axis rotation around X (sagittal plane)
    half = math.radians(x_deg) * 0.5
    return [math.sin(half), 0.0, 0.0, math.cos(half)]


def _sagittal_quat(angle_deg: float) -> List[float]:
    """
    Quaternion for sagittal flexion of a spherical hip/ankle joint.

    Produces a rotation about the joint's local Z-axis, which — after the
    90-degree base rotation — corresponds to the lateral (mediolateral) axis
    in the world frame.  Positive angle = hip flexion (thigh forward) or
    ankle dorsiflexion (toes up).

    Format: [x, y, z, w] (PyBullet quaternion convention).
    """
    half = math.radians(angle_deg) * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


# ── Joint discovery ───────────────────────────────────────────────────────────


def _discover_joints(body_id: int, client: int) -> Dict[str, dict]:
    """
    Return {joint_name_lower: {'index': int, 'type': int, 'name': str}}.
    """
    n  = p.getNumJoints(body_id, physicsClientId=client)
    mp = {}
    for i in range(n):
        info  = p.getJointInfo(body_id, i, physicsClientId=client)
        name  = info[1].decode("utf-8")
        jtype = info[2]
        mp[name.lower()] = {"index": i, "type": jtype, "name": name}
    return mp


def _find_joint(joint_map: Dict[str, dict], keyword: str) -> Optional[dict]:
    """
    Find first joint whose name contains `keyword` (case-insensitive).
    """
    kw = keyword.lower().replace("_", "")
    for name, info in joint_map.items():
        if kw in name.replace("_", ""):
            return info
    return None


# ── Metrics accumulator ───────────────────────────────────────────────────────


class _MetricsAccum:
    def __init__(self, fall_threshold: float = FALL_HEIGHT_THRESHOLD):
        self.fall_threshold = fall_threshold
        self.com_heights:         List[float] = []
        self.knee_pred_deg:       List[float] = []
        self.knee_mocap_deg:      List[float] = []
        self.right_contact_frames: List[int]  = []
        self.left_contact_frames:  List[int]  = []
        self.fall_detected = False
        self.fall_frame    = -1

    def record(self, frame: int, com_height: float,
               knee_pred: float, knee_mocap: float,
               right_contact: bool, left_contact: bool):
        self.com_heights.append(com_height)
        self.knee_pred_deg.append(knee_pred)
        self.knee_mocap_deg.append(knee_mocap)
        if right_contact:
            self.right_contact_frames.append(frame)
        if left_contact:
            self.left_contact_frames.append(frame)
        if not self.fall_detected and com_height < self.fall_threshold:
            self.fall_detected = True
            self.fall_frame    = frame

    def summarise(self) -> dict:
        heights = np.array(self.com_heights)
        pred    = np.array(self.knee_pred_deg)
        mocap   = np.array(self.knee_mocap_deg)

        err     = pred - mocap
        knee_rmse = float(np.sqrt(np.mean(err ** 2)))
        knee_mae  = float(np.mean(np.abs(err)))

        com_mean = float(np.mean(heights)) if len(heights) else 0.0
        com_std  = float(np.std(heights))  if len(heights) else 0.0

        step_right = _count_contacts(self.right_contact_frames, min_gap=20)
        step_left  = _count_contacts(self.left_contact_frames,  min_gap=20)
        step_count = step_right + step_left

        sym = _gait_symmetry(self.right_contact_frames, self.left_contact_frames)

        stable_base = 1.0 - min(com_std / 0.30, 1.0)
        fall_penalty = 0.5 if self.fall_detected else 0.0
        sym_penalty  = 0.2 * sym
        stability_score = max(0.0, stable_base - fall_penalty - sym_penalty)

        return {
            "com_height_mean":  com_mean,
            "com_height_std":   com_std,
            "fall_detected":    self.fall_detected,
            "fall_frame":       self.fall_frame,
            "knee_rmse_deg":    knee_rmse,
            "knee_mae_deg":     knee_mae,
            "step_count":       step_count,
            "gait_symmetry":    float(sym),
            "stability_score":  float(stability_score),
        }


def _count_contacts(frames: List[int], min_gap: int = 20) -> int:
    """Count distinct contact events (rising edges) from a list of active frames."""
    if not frames:
        return 0
    count   = 1
    last    = frames[0]
    for f in frames[1:]:
        if f - last > min_gap:
            count += 1
        last = f
    return count


def _gait_symmetry(right: List[int], left: List[int]) -> float:
    """Gait symmetry index [0 = perfect, 1 = maximally asymmetric]."""
    def intervals(frames):
        if len(frames) < 2:
            return []
        fs = sorted(set(frames))
        ints = []
        prev = fs[0]
        for f in fs[1:]:
            if f - prev > 5:
                ints.append(f - prev)
            prev = f
        return ints

    ri = intervals(right)
    li = intervals(left)
    if not ri or not li:
        return 0.5   # unknown
    mr = np.mean(ri)
    ml = np.mean(li)
    total = mr + ml
    if total < 1e-6:
        return 0.0
    return float(abs(mr - ml) / total)


# ── GIF helper ───────────────────────────────────────────────────────────────


def _save_gif(frames: list, path: str, capture_fps: float = 25.0) -> None:
    """Save a list of (H, W, 3) uint8 numpy arrays as an animated GIF."""
    if not frames:
        return
    duration_ms = max(1, int(1000.0 / capture_fps))
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        print(f"  [sim] Visualization saved -> {path}"
              f"  ({len(frames)} frames @ {capture_fps:.0f} fps)")
    except ImportError:
        npy_path = path.rsplit(".", 1)[0] + "_frames.npy"
        np.save(npy_path, np.array(frames, dtype=np.uint8))
        print(f"  [sim] Frames saved -> {npy_path}"
              "  (install Pillow for GIF: pip install pillow)")
    except Exception as exc:
        warnings.warn(f"Could not save GIF to {path}: {exc}", RuntimeWarning)


# ── Root trajectory builder ──────────────────────────────────────────────────


def _build_root_trajectory(
    mocap_segment: dict,
    T: int,
    fps: float,
) -> np.ndarray:
    """
    Build the root position trajectory (T, 3) in PyBullet Z-up coordinates.

    Uses real BVH root positions when available; falls back to a constant-speed
    forward walk at ~1.35 m/s.  The Z column (height) is normalised so that the
    minimum root position sits at HUMANOID_INIT_HEIGHT.
    """
    root_pos_all = mocap_segment["root_pos"][:T]
    root_z_range = root_pos_all[:, 2].max() - root_pos_all[:, 2].min()
    has_real_root = root_z_range > 0.02 and root_pos_all[:, 2].mean() > 0.3

    if has_real_root:
        z_min     = root_pos_all[:, 2].min()
        z_offset  = HUMANOID_INIT_HEIGHT - z_min
        root_traj = root_pos_all.copy()
        root_traj[:, 2] += z_offset
    else:
        t_s       = np.arange(T, dtype=np.float32) / fps
        root_traj = np.zeros((T, 3), dtype=np.float32)
        root_traj[:, 0] = 1.35 * t_s
        root_traj[:, 2] = HUMANOID_INIT_HEIGHT

    return root_traj


# ── Colour helpers ───────────────────────────────────────────────────────────


def _color_robot_prosthetic(robot: int, jmap: Dict[str, dict], client: int):
    """Grey body + orange right-leg chain (prosthetic)."""
    n = p.getNumJoints(robot, physicsClientId=client)
    p.changeVisualShape(robot, -1, rgbaColor=[0.6, 0.6, 0.6, 1.0],
                        physicsClientId=client)
    for i in range(n):
        info = p.getJointInfo(robot, i, physicsClientId=client)
        name = info[1].decode("utf-8").lower()
        if "right" in name and ("knee" in name or "leg" in name
                                or "foot" in name or "ankle" in name):
            p.changeVisualShape(robot, i, rgbaColor=[1.0, 0.4, 0.1, 1.0],
                                physicsClientId=client)
        else:
            p.changeVisualShape(robot, i, rgbaColor=[0.6, 0.6, 0.6, 1.0],
                                physicsClientId=client)


def _color_robot_ghost(robot: int, client: int):
    """Semi-transparent blue for ground-truth reference."""
    n = p.getNumJoints(robot, physicsClientId=client)
    blue = [0.2, 0.5, 0.9, 0.7]
    p.changeVisualShape(robot, -1, rgbaColor=blue, physicsClientId=client)
    for i in range(n):
        p.changeVisualShape(robot, i, rgbaColor=blue, physicsClientId=client)


# ── Simulator class ───────────────────────────────────────────────────────────


class ProstheticSimulator:
    """
    Run a PyBullet physics simulation replacing the right knee with model
    predictions and driving all other joints from motion-capture data.

    Each robot is simulated **independently** — no two robots appear in the
    same scene, so every GIF shows exactly one humanoid.
    """

    def __init__(
        self,
        use_gui: bool = False,
        fall_threshold: float = FALL_HEIGHT_THRESHOLD,
        physics_steps_per_frame: int = 4,
    ):
        if not _PYBULLET_AVAILABLE:
            raise RuntimeError(
                "PyBullet is not installed. Run: pip install pybullet"
            )
        self.use_gui  = use_gui
        self.fall_threshold = fall_threshold
        self.sub_steps      = physics_steps_per_frame
        self._client: Optional[int] = None

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self):
        mode = p.GUI if self.use_gui else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(SIM_TIMESTEP / self.sub_steps,
                      physicsClientId=self._client)
        self._plane = p.loadURDF("plane.urdf",
                                 physicsClientId=self._client)
        if self.use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self._client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                       physicsClientId=self._client)
            p.setRealTimeSimulation(0, physicsClientId=self._client)
        return self

    def __exit__(self, *_):
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None

    # ── robot helpers ─────────────────────────────────────────────────────

    def _load_robot(self, start_pos: Optional[List[float]] = None) -> int:
        if start_pos is None:
            start_pos = [0.0, 0.0, HUMANOID_INIT_HEIGHT]
        robot = p.loadURDF(
            "humanoid/humanoid.urdf",
            start_pos,
            _BASE_QUAT,
            useFixedBase=False,
            physicsClientId=self._client,
        )
        return robot

    def _disable_motors(self, robot: int):
        """Disable default velocity motors on revolute/prismatic joints."""
        n = p.getNumJoints(robot, physicsClientId=self._client)
        for i in range(n):
            info  = p.getJointInfo(robot, i, physicsClientId=self._client)
            jtype = info[2]
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                p.setJointMotorControl2(
                    robot, i,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0, force=0,
                    physicsClientId=self._client,
                )

    # ── PD control helpers ────────────────────────────────────────────────

    def _set_revolute_pd(self, robot: int, idx: int, angle_rad: float):
        """Drive a revolute joint toward angle_rad via PD position control."""
        p.setJointMotorControl2(
            robot, idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle_rad,
            positionGain=KP_REVOLUTE,
            velocityGain=KD_REVOLUTE,
            force=MAX_FORCE_REV,
            physicsClientId=self._client,
        )

    def _set_spherical_pd(self, robot: int, idx: int, quat: List[float]):
        """Drive a spherical joint toward orientation quat [x,y,z,w]."""
        p.setJointMotorControlMultiDof(
            robot, idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=quat,
            positionGain=KP_SPHERICAL,
            velocityGain=KD_SPHERICAL,
            force=[MAX_FORCE_SPH, MAX_FORCE_SPH, MAX_FORCE_SPH],
            physicsClientId=self._client,
        )

    # ── kinematic setters (for GT playback) ───────────────────────────────

    def _set_joint_kinematic(self, robot: int, jmap: dict,
                             kw: str, angle_deg: float,
                             negate_for_knee: bool = False):
        """Kinematically set a joint by keyword."""
        info = _find_joint(jmap, kw)
        if info is None:
            return
        deg = -angle_deg if negate_for_knee else angle_deg
        if info["type"] == p.JOINT_REVOLUTE:
            p.resetJointState(robot, info["index"], math.radians(deg),
                              physicsClientId=self._client)
        elif info["type"] == p.JOINT_SPHERICAL:
            p.resetJointStateMultiDof(
                robot, info["index"], _sagittal_quat(deg),
                physicsClientId=self._client,
            )

    # ── CoM helper ────────────────────────────────────────────────────────

    def _compute_com_height(self, robot: int) -> float:
        """Compute whole-body centre-of-mass Z coordinate."""
        total_mass = 0.0
        com_z      = 0.0
        base_pos, _ = p.getBasePositionAndOrientation(
            robot, physicsClientId=self._client)
        base_mass   = p.getDynamicsInfo(
            robot, -1, physicsClientId=self._client)[0]
        com_z      += base_mass * base_pos[2]
        total_mass += base_mass

        for i in range(p.getNumJoints(robot, physicsClientId=self._client)):
            link_state = p.getLinkState(robot, i,
                                        physicsClientId=self._client)
            link_mass  = p.getDynamicsInfo(robot, i,
                                           physicsClientId=self._client)[0]
            com_z     += link_mass * link_state[0][2]
            total_mass += link_mass
        return com_z / max(total_mass, 1e-9)

    # ── foot contact detection ────────────────────────────────────────────

    def _check_foot_contacts(self, robot: int) -> Tuple[bool, bool]:
        """Return (right_foot_contact, left_foot_contact)."""
        right = False
        left  = False
        for contact in p.getContactPoints(robot, self._plane,
                                          physicsClientId=self._client):
            link_idx = contact[3]
            info     = p.getJointInfo(robot, link_idx,
                                      physicsClientId=self._client)
            name     = info[1].decode("utf-8").lower()
            if "right" in name and ("foot" in name or "ankle" in name):
                right = True
            if "left" in name and ("foot" in name or "ankle" in name):
                left = True
        return right, left

    # ── main entry point ──────────────────────────────────────────────────

    def run(
        self,
        mocap_segment: dict,
        predicted_knee: np.ndarray,
        fps: float = 200.0,
        gif_output_pred: Optional[str] = None,
        gif_output_gt: Optional[str] = None,
    ) -> dict:
        """
        Simulate the prosthetic knee and optionally produce GIFs.

        Runs two **separate** simulations — one for the predicted robot
        (physics-based, right knee = model prediction) and one for the
        ground-truth robot (kinematic playback, all joints from mocap).
        Each GIF shows exactly one robot.

        Returns metrics from the predicted simulation.
        """
        T = len(predicted_knee)
        assert len(mocap_segment["knee_right"]) >= T

        root_traj = _build_root_trajectory(mocap_segment, T, fps)

        # ── Phase 1: predicted robot (PD physics) ────────────────────────
        want_pred = gif_output_pred is not None
        pred_metrics, pred_frames = self._run_predicted(
            mocap_segment, predicted_knee, root_traj, T, fps,
            capture_frames=want_pred,
        )

        # ── Phase 2: ground-truth robot (kinematic) ──────────────────────
        gt_frames = None
        if gif_output_gt is not None:
            gt_frames = self._run_ground_truth(
                mocap_segment, root_traj, T, fps,
            )

        # ── Save GIFs ────────────────────────────────────────────────────
        capture_fps = fps / _CAPTURE_EVERY
        if pred_frames and gif_output_pred:
            _save_gif(pred_frames, gif_output_pred, capture_fps=capture_fps)
        if gt_frames and gif_output_gt:
            _save_gif(gt_frames, gif_output_gt, capture_fps=capture_fps)

        return pred_metrics

    # ── predicted-robot simulation (PD control + gravity) ─────────────────

    def _run_predicted(
        self,
        mocap_segment: dict,
        predicted_knee: np.ndarray,
        root_traj: np.ndarray,
        T: int,
        fps: float,
        capture_frames: bool = True,
    ) -> Tuple[dict, Optional[List]]:
        """
        Simulate the prosthetic robot with PD control.

        All lower-body joints use PD position control; the right knee target
        is the model prediction.  The base floats freely under gravity,
        guided horizontally by spring forces.  If the knee prediction is
        wrong the robot loses balance and falls — this is detected via CoM
        height.
        """
        robot = self._load_robot(
            [float(root_traj[0, 0]), 0.0, float(root_traj[0, 2])]
        )
        jmap = _discover_joints(robot, self._client)
        self._disable_motors(robot)
        _color_robot_prosthetic(robot, jmap, self._client)

        # Set initial pose kinematically, then switch to PD
        for kw, ang in [
            ("righthip",   float(mocap_segment["hip_right"][0])),
            ("lefthip",    float(mocap_segment["hip_left"][0])),
            ("rightankle", float(mocap_segment["ankle_right"][0])),
            ("leftankle",  float(mocap_segment["ankle_left"][0])),
        ]:
            self._set_joint_kinematic(robot, jmap, kw, ang)
        for kw, ang in [
            ("rightknee",  float(predicted_knee[0])),
            ("leftknee",   float(mocap_segment["knee_left"][0])),
        ]:
            self._set_joint_kinematic(robot, jmap, kw, ang, negate_for_knee=True)

        # Settle for a few steps
        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)

        accum  = _MetricsAccum(fall_threshold=self.fall_threshold)
        frames: Optional[List] = [] if capture_frames else None

        _proj = p.computeProjectionMatrixFOV(
            fov=45, aspect=640.0 / 480.0,
            nearVal=0.1, farVal=50.0,
            physicsClientId=self._client,
        )

        for t in range(T):
            fr = {
                "hip_right":   float(mocap_segment["hip_right"][t]),
                "hip_left":    float(mocap_segment["hip_left"][t]),
                "knee_right":  float(mocap_segment["knee_right"][t]),
                "knee_left":   float(mocap_segment["knee_left"][t]),
                "ankle_right": float(mocap_segment["ankle_right"][t]),
                "ankle_left":  float(mocap_segment["ankle_left"][t]),
            }
            pred_knee_deg = float(predicted_knee[t])

            # PD targets — right knee from prediction, rest from mocap
            rk_info = _find_joint(jmap, "rightknee")
            if rk_info:
                self._set_revolute_pd(
                    robot, rk_info["index"],
                    math.radians(-pred_knee_deg),  # negate for URDF convention
                )
            lk_info = _find_joint(jmap, "leftknee")
            if lk_info:
                self._set_revolute_pd(
                    robot, lk_info["index"],
                    math.radians(-fr["knee_left"]),
                )
            for kw, ang in [
                ("righthip",   fr["hip_right"]),
                ("lefthip",    fr["hip_left"]),
                ("rightankle", fr["ankle_right"]),
                ("leftankle",  fr["ankle_left"]),
            ]:
                info = _find_joint(jmap, kw)
                if info:
                    self._set_spherical_pd(
                        robot, info["index"], _sagittal_quat(ang),
                    )

            # Guide base horizontally toward trajectory
            bp, _ = p.getBasePositionAndOrientation(
                robot, physicsClientId=self._client)
            bv, _ = p.getBaseVelocity(
                robot, physicsClientId=self._client)
            target_x = float(root_traj[t, 0])
            fx = _BASE_GUIDE_KP * (target_x - bp[0]) - _BASE_GUIDE_KD * bv[0]
            fy = _BASE_GUIDE_KP * (0.0 - bp[1]) - _BASE_GUIDE_KD * bv[1]
            p.applyExternalForce(
                robot, -1, [fx, fy, 0],
                [bp[0], bp[1], bp[2]], p.WORLD_FRAME,
                physicsClientId=self._client,
            )

            # Sub-step physics
            for _ in range(self.sub_steps):
                p.stepSimulation(physicsClientId=self._client)

            # Metrics
            com_h = self._compute_com_height(robot)
            right_c, left_c = self._check_foot_contacts(robot)
            accum.record(t, com_h, pred_knee_deg, fr["knee_right"],
                         right_c, left_c)

            # Frame capture
            if frames is not None and t % _CAPTURE_EVERY == 0:
                bp_now, _ = p.getBasePositionAndOrientation(
                    robot, physicsClientId=self._client)
                rx = float(bp_now[0])
                rz = float(bp_now[2])
                _view = p.computeViewMatrix(
                    cameraEyePosition=[rx, -12.0, rz + 1.5],
                    cameraTargetPosition=[rx, 0.0, rz + 0.5],
                    cameraUpVector=[0, 0, 1],
                    physicsClientId=self._client,
                )
                _, _, _rgba, _, _ = p.getCameraImage(
                    640, 480, _view, _proj,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=self._client,
                )
                frames.append(
                    np.array(_rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
                )

        p.removeBody(robot, physicsClientId=self._client)
        return accum.summarise(), frames

    # ── ground-truth simulation (kinematic playback) ──────────────────────

    def _run_ground_truth(
        self,
        mocap_segment: dict,
        root_traj: np.ndarray,
        T: int,
        fps: float,
    ) -> List:
        """
        Kinematically play back all mocap joints for a clean reference GIF.
        """
        robot = self._load_robot(
            [float(root_traj[0, 0]), 0.0, float(root_traj[0, 2])]
        )
        jmap = _discover_joints(robot, self._client)
        _color_robot_ghost(robot, self._client)

        frames: List = []
        _proj = p.computeProjectionMatrixFOV(
            fov=45, aspect=640.0 / 480.0,
            nearVal=0.1, farVal=50.0,
            physicsClientId=self._client,
        )

        for t in range(T):
            fr = {
                "hip_right":   float(mocap_segment["hip_right"][t]),
                "hip_left":    float(mocap_segment["hip_left"][t]),
                "knee_right":  float(mocap_segment["knee_right"][t]),
                "knee_left":   float(mocap_segment["knee_left"][t]),
                "ankle_right": float(mocap_segment["ankle_right"][t]),
                "ankle_left":  float(mocap_segment["ankle_left"][t]),
            }

            # Teleport base
            base_pos = [float(root_traj[t, 0]), 0.0, float(root_traj[t, 2])]
            p.resetBasePositionAndOrientation(
                robot, base_pos, _BASE_QUAT,
                physicsClientId=self._client,
            )

            # Set all joints kinematically
            for kw, ang in [
                ("righthip",   fr["hip_right"]),
                ("lefthip",    fr["hip_left"]),
                ("rightankle", fr["ankle_right"]),
                ("leftankle",  fr["ankle_left"]),
            ]:
                self._set_joint_kinematic(robot, jmap, kw, ang)
            for kw, ang in [
                ("rightknee",  fr["knee_right"]),
                ("leftknee",   fr["knee_left"]),
            ]:
                self._set_joint_kinematic(robot, jmap, kw, ang,
                                          negate_for_knee=True)

            p.stepSimulation(physicsClientId=self._client)

            # Frame capture
            if t % _CAPTURE_EVERY == 0:
                rx = float(root_traj[t, 0])
                rz = float(root_traj[t, 2])
                _view = p.computeViewMatrix(
                    cameraEyePosition=[rx, -12.0, rz + 1.5],
                    cameraTargetPosition=[rx, 0.0, rz + 0.5],
                    cameraUpVector=[0, 0, 1],
                    physicsClientId=self._client,
                )
                _, _, _rgba, _, _ = p.getCameraImage(
                    640, 480, _view, _proj,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=self._client,
                )
                frames.append(
                    np.array(_rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
                )

        p.removeBody(robot, physicsClientId=self._client)
        return frames


# ── Kinematic reference (no physics, pure FK) ─────────────────────────────────


def run_kinematic_evaluation(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
) -> dict:
    """
    Lightweight kinematic evaluation that does NOT require PyBullet.
    """
    T = len(predicted_knee)
    assert T > 0

    L_THIGH = 0.40
    L_SHANK = 0.38

    def foot_pos_sagittal(hip_ang_deg: float, knee_ang_deg: float) -> np.ndarray:
        hip_rad   = math.radians(hip_ang_deg)
        knee_rad  = math.radians(knee_ang_deg)
        thigh_x = -L_THIGH * math.sin(hip_rad)
        thigh_z = -L_THIGH * math.cos(hip_rad)
        shank_ang = hip_rad - knee_rad
        shank_x = thigh_x - L_SHANK * math.sin(shank_ang)
        shank_z = thigh_z - L_SHANK * math.cos(shank_ang)
        return np.array([shank_x, shank_z])

    hip_r = mocap_segment.get("hip_right",   np.zeros(T))
    knee_r_mocap = mocap_segment.get("knee_right", np.zeros(T))

    foot_dev = np.zeros(T)
    knee_err = predicted_knee[:T] - knee_r_mocap[:T]

    for t in range(T):
        fp_mocap = foot_pos_sagittal(float(hip_r[t]), float(knee_r_mocap[t]))
        fp_pred  = foot_pos_sagittal(float(hip_r[t]), float(predicted_knee[t]))
        foot_dev[t] = float(np.linalg.norm(fp_pred - fp_mocap))

    com_dev = foot_dev / 2.0

    knee_rmse = float(np.sqrt(np.mean(knee_err ** 2)))
    knee_mae  = float(np.mean(np.abs(knee_err)))

    max_dev = float(np.max(foot_dev))
    fall_detected = max_dev > 0.15

    stable = max(0.0, 1.0 - max_dev / 0.30)
    if fall_detected:
        stable *= 0.5

    return {
        "com_height_mean":     float(HUMANOID_INIT_HEIGHT - np.mean(com_dev)),
        "com_height_std":      float(np.std(com_dev)),
        "fall_detected":       fall_detected,
        "fall_frame":          int(np.argmax(foot_dev)) if fall_detected else -1,
        "knee_rmse_deg":       knee_rmse,
        "knee_mae_deg":        knee_mae,
        "foot_deviation_mean": float(np.mean(foot_dev)),
        "foot_deviation_max":  max_dev,
        "step_count":          -1,
        "gait_symmetry":       0.0,
        "stability_score":     stable,
        "mode":                "kinematic",
    }


# ── Public convenience function ───────────────────────────────────────────────


def simulate_prosthetic_walking(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
    use_physics: bool = True,
    use_gui: bool = False,
    fps: float = 200.0,
    gif_output_pred: Optional[str] = None,
    gif_output_gt: Optional[str] = None,
) -> dict:
    """
    High-level entry point for prosthetic walking evaluation.

    Runs a PyBullet physics simulation for the predicted robot (right knee =
    model prediction, PD-controlled) and a kinematic playback for the
    ground-truth robot.  Each robot is simulated individually so the GIFs
    show exactly one humanoid each.

    Falls back to kinematic-only evaluation if PyBullet is unavailable.
    """
    if use_physics and _PYBULLET_AVAILABLE:
        effective_gui = use_gui
        if use_gui and not os.environ.get("DISPLAY"):
            warnings.warn(
                "No DISPLAY found -- running headless PyBullet."
                + (" Saving visualisation to GIF(s) instead."
                   if (gif_output_pred or gif_output_gt) else ""),
                RuntimeWarning,
                stacklevel=2,
            )
            effective_gui = False
        try:
            with ProstheticSimulator(use_gui=effective_gui) as sim:
                metrics = sim.run(
                    mocap_segment, predicted_knee, fps=fps,
                    gif_output_pred=gif_output_pred,
                    gif_output_gt=gif_output_gt,
                )
            metrics["mode"] = "physics" + ("+gui" if effective_gui else "")
            return metrics
        except Exception as exc:
            warnings.warn(
                f"PyBullet physics simulation failed ({exc}). "
                "Falling back to kinematic evaluation.",
                RuntimeWarning,
                stacklevel=2,
            )

    metrics = run_kinematic_evaluation(mocap_segment, predicted_knee)
    return metrics


# ── Standalone visualisation demo ────────────────────────────────────────────


def run_visual_demo(use_full_db: bool = False):
    """
    Launch a PyBullet GUI window showing a walking simulation demo.
    """
    if not _PYBULLET_AVAILABLE:
        print("ERROR: PyBullet is required.  Install: pip install pybullet",
              file=__import__("sys").stderr)
        return

    from mocap_evaluation.mocap_loader import (
        _interp_gait_curve,
        _KNEE_R,
        _HIP_R,
        TARGET_FPS,
        load_full_cmu_database,
        load_or_generate_mocap_database,
    )
    from mocap_evaluation.motion_matching import find_best_match

    print("=" * 60)
    print("PYBULLET VISUALISATION DEMO")
    print("=" * 60)

    fps      = TARGET_FPS
    CADENCE  = 110.0
    cycle_s  = 60.0 / (CADENCE / 2.0)
    spc      = int(round(cycle_s * fps))
    N_CYCLES = 3
    T        = spc * N_CYCLES

    rng = np.random.default_rng(42)

    knee_true  = np.tile(_interp_gait_curve(_KNEE_R, spc), N_CYCLES).astype(np.float32)
    thigh_true = np.tile(_interp_gait_curve(_HIP_R,  spc), N_CYCLES).astype(np.float32)
    knee_imu   = knee_true  + rng.normal(0, 2.0, T).astype(np.float32)
    thigh_imu  = thigh_true + rng.normal(0, 1.5, T).astype(np.float32)

    predicted = knee_true + rng.normal(0, 5.0, T).astype(np.float32)

    if use_full_db:
        db = load_full_cmu_database()
    else:
        db = load_or_generate_mocap_database()

    db_dur = len(db["knee_right"]) / fps
    print(f"  Database: {db_dur:.1f}s  source={db['source']}")
    print(f"  Query   : {T} frames ({T/fps:.2f}s)")

    start, dist, segment = find_best_match(knee_imu, thigh_imu, db)
    cat = segment.get("category", "unknown")
    print(f"  Match   : start={start}, DTW={dist:.4f}, category={cat}")

    print(f"\n  Launching PyBullet window -- {T/fps:.1f}s playback ...")
    print("  Close the window to exit.\n")

    with ProstheticSimulator(use_gui=True) as sim:
        metrics = sim.run(segment, predicted, fps=float(fps))

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")


def main():
    """CLI entry point for ``python -m mocap_evaluation.prosthetic_sim``."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Visualise prosthetic walking simulation in PyBullet"
    )
    ap.add_argument("--demo", action="store_true",
                    help="Run visual demo")
    ap.add_argument("--full-db", action="store_true",
                    help="Use full CMU mocap database (downloads if needed)")
    args = ap.parse_args()

    if args.demo:
        run_visual_demo(use_full_db=args.full_db)
    else:
        print("Usage:  python -m mocap_evaluation.prosthetic_sim --demo")
        print("        python -m mocap_evaluation.prosthetic_sim --demo --full-db")


if __name__ == "__main__":
    main()
