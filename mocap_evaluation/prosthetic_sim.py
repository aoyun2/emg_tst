"""
PyBullet prosthetic walking simulator.

Design
------
All body joints are driven towards motion-capture target angles via high-gain
PD position control.  The right knee is replaced by the model's predicted
angle.  Because the PD gains are high but finite, the humanoid will maintain
balance when predictions are accurate and may stumble or fall when predictions
deviate significantly — giving a physically grounded evaluation.

Joint layout (pybullet_data humanoid/humanoid.urdf)
---------------------------------------------------
Discovered at runtime via getJointInfo; the semantic names we look for are
the substrings below.  The parser is tolerant of capitalisation differences.

  Revolute  : chest, neck, rightKnee, rightElbow, leftKnee, leftElbow
  Spherical : rightHip, rightAnkle, rightShoulder, leftHip, leftAnkle, leftShoulder

We control only the lower-body joints relevant to walking; upper-body joints
are locked at zero (resting pose).

Stability metrics
-----------------
  com_height_mean  : average pelvis/CoM height (m)
  com_height_std   : standard deviation — higher = more bobbing/instability
  fall_detected    : True if CoM height drops below fall_threshold (default 0.55 m)
  fall_frame       : frame index of first fall (-1 if no fall)
  knee_rmse_deg    : RMSE between predicted and mocap right-knee angle (degrees)
  knee_mae_deg     : MAE  between predicted and mocap right-knee angle
  step_count       : number of completed right-foot ground contacts
  gait_symmetry    : |mean_right_step_time – mean_left_step_time| / mean_step_time
                     0 = perfect symmetry, 1 = completely asymmetric
  stability_score  : composite 0–1 (higher = more stable)
"""
from __future__ import annotations

import math
import os
import time as _time
import warnings
from contextlib import contextmanager
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

FALL_HEIGHT_THRESHOLD = 1.8    # metres; if CoM drops below this → fall
HUMANOID_INIT_HEIGHT  = 3.47   # metres; pelvis height for upright humanoid.urdf
# Quaternion that rotates the Y-up humanoid.urdf so it stands upright in PyBullet's
# Z-up world: 90° rotation around world X axis maps URDF +Y (spine) → world +Z (up).
HUMANOID_INIT_QUAT    = [0.7071068, 0.0, 0.0, 0.7071068]
SIM_TIMESTEP          = 1.0 / 200.0   # seconds per physics step (match 200 Hz)

# PD gains for position control — tuned so the humanoid tracks mocap well
# without being so stiff that contact forces become unrealistic.
KP_REVOLUTE   = 300.0          # position gain, revolute joints
KD_REVOLUTE   = 30.0           # velocity gain, revolute joints
MAX_FORCE_REV = 600.0          # N·m

KP_SPHERICAL  = 300.0
KD_SPHERICAL  = 30.0
MAX_FORCE_SPH = 600.0          # per-axis force for setJointMotorControlMultiDof

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

    The humanoid.urdf (pybullet_data) defines its revolute knee joints with
    ``axis xyz="0 0 1"`` (local Z-axis).  For consistency, we treat spherical
    hip and ankle joints the same way: the primary sagittal degree-of-freedom
    is a rotation about the local Z-axis.  This matches the URDF frame
    orientations and produces anatomically correct flexion/extension in
    PyBullet's Z-up world.

    Format: [x, y, z, w] (PyBullet quaternion convention).
    """
    half = math.radians(angle_deg) * 0.5
    return [0.0, 0.0, math.sin(half), math.cos(half)]


# ── Joint discovery ───────────────────────────────────────────────────────────


def _discover_joints(body_id: int) -> Dict[str, dict]:
    """
    Return {joint_name_lower: {'index': int, 'type': int, 'name': str}}.
    type: 0=REVOLUTE, 1=PRISMATIC, 2=SPHERICAL, 3=PLANAR, 4=FIXED.
    In PyBullet: JOINT_REVOLUTE=0, JOINT_SPHERICAL=2 (sometimes 4 in docs).
    """
    n  = p.getNumJoints(body_id)
    mp = {}
    for i in range(n):
        info  = p.getJointInfo(body_id, i)
        name  = info[1].decode("utf-8")
        jtype = info[2]
        mp[name.lower()] = {"index": i, "type": jtype, "name": name}
    return mp


def _find_joint(joint_map: Dict[str, dict], keyword: str) -> Optional[dict]:
    """
    Find first joint whose name contains `keyword` (case-insensitive).
    Underscores are stripped from both sides before comparison so that
    'righthip' matches 'right_hip', 'rightknee' matches 'right_knee', etc.
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

        # Step count: count rising edges in contact arrays
        step_right = _count_contacts(self.right_contact_frames, min_gap=20)
        step_left  = _count_contacts(self.left_contact_frames,  min_gap=20)
        step_count = step_right + step_left

        # Gait symmetry from inter-contact intervals
        sym = _gait_symmetry(self.right_contact_frames, self.left_contact_frames)

        # Stability score: penalise falls, high CoM variance, asymmetry
        stable_base = 1.0 - min(com_std / 0.10, 1.0)   # 0.10 m std = 0
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
    """
    Compute gait symmetry index [0 = perfect, 1 = maximally asymmetric]
    based on mean step interval.
    """
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
    """
    Save a list of (H, W, 3) uint8 numpy arrays as an animated GIF.

    Requires Pillow.  Falls back to saving a .npy frame dump if Pillow is
    not installed so the caller can at least retrieve the raw frames.
    """
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
        print(f"  [sim] Visualization saved → {path}"
              f"  ({len(frames)} frames @ {capture_fps:.0f} fps)")
    except ImportError:
        npy_path = path.rsplit(".", 1)[0] + "_frames.npy"
        np.save(npy_path, np.array(frames, dtype=np.uint8))
        print(f"  [sim] Frames saved → {npy_path}"
              "  (install Pillow for GIF: pip install pillow)")
    except Exception as exc:
        warnings.warn(f"Could not save GIF to {path}: {exc}", RuntimeWarning)

# ── Simulator class ───────────────────────────────────────────────────────────


class ProstheticSimulator:
    """
    Run a PyBullet physics simulation replacing the right knee with model
    predictions and driving all other joints from motion-capture data.

    Parameters
    ----------
    use_gui          : show the PyBullet GUI (requires a display)
    fall_threshold   : CoM height (m) below which a fall is declared
    physics_steps_per_frame : sub-steps per mocap frame (improves stability)
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
        self._plane:  Optional[int] = None
        self._robot:  Optional[int] = None
        self._joint_map: Dict[str, dict] = {}

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self):
        mode = p.GUI if self.use_gui else p.DIRECT
        self._client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(SIM_TIMESTEP / self.sub_steps, physicsClientId=self._client)
        self._plane = p.loadURDF("plane.urdf", physicsClientId=self._client)
        if self.use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0,
                                       physicsClientId=self._client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1,
                                       physicsClientId=self._client)
            # Disable real-time so we control stepping
            p.setRealTimeSimulation(0, physicsClientId=self._client)
        return self

    def __exit__(self, *_):
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None

    # ── robot setup ───────────────────────────────────────────────────────

    def _load_robot(self, start_pos: Optional[List[float]] = None,
                    start_quat: Optional[List[float]] = None) -> int:
        if start_pos is None:
            start_pos = [0.0, 0.0, HUMANOID_INIT_HEIGHT]
        if start_quat is None:
            start_quat = HUMANOID_INIT_QUAT
        robot = p.loadURDF(
            "humanoid/humanoid.urdf",
            start_pos,
            start_quat,
            useFixedBase=False,
            physicsClientId=self._client,
        )
        return robot

    def _disable_motors(self, robot: int):
        """Disable default velocity motors so we can apply PD control freely.
        Note: spherical joints do NOT support VELOCITY_CONTROL mode in PyBullet,
        so only revolute/prismatic motors are disabled here."""
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
            # Spherical joints: skip — they don't have a default motor to disable

    # ── joint control helpers ─────────────────────────────────────────────

    def _set_revolute(self, robot: int, idx: int, angle_rad: float):
        # Negate: with HUMANOID_INIT_QUAT (R_x+90°), positive revolute angle =
        # extension; our convention is positive = flexion, so negate here.
        p.setJointMotorControl2(
            robot, idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=-angle_rad,
            positionGain=KP_REVOLUTE,
            velocityGain=KD_REVOLUTE,
            force=MAX_FORCE_REV,
            physicsClientId=self._client,
        )

    def _set_spherical(self, robot: int, idx: int, quat: List[float]):
        """quat = [x, y, z, w]."""
        p.setJointMotorControlMultiDof(
            robot, idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=quat,
            positionGain=KP_SPHERICAL,
            velocityGain=KD_SPHERICAL,
            force=[MAX_FORCE_SPH, MAX_FORCE_SPH, MAX_FORCE_SPH],
            physicsClientId=self._client,
        )

    def _init_pose(self, robot: int, frame0: dict):
        """Set the robot to the first-frame pose so it doesn't fall from rest."""
        jmap = _discover_joints(robot)
        for kw, angle_deg in [
            ("righthip",   frame0.get("hip_right",   0.0)),
            ("rightknee",  frame0.get("knee_right",  0.0)),
            ("rightankle", frame0.get("ankle_right", 0.0)),
            ("lefthip",    frame0.get("hip_left",    0.0)),
            ("leftknee",   frame0.get("knee_left",   0.0)),
            ("leftankle",  frame0.get("ankle_left",  0.0)),
        ]:
            info = _find_joint(jmap, kw)
            if info is None:
                continue
            angle_rad = math.radians(angle_deg)
            if info["type"] == p.JOINT_REVOLUTE:
                p.resetJointState(robot, info["index"], -angle_rad,
                                  physicsClientId=self._client)
            elif info["type"] == p.JOINT_SPHERICAL:
                q = _sagittal_quat(angle_deg)
                p.resetJointStateMultiDof(robot, info["index"], q,
                                          physicsClientId=self._client)

    # ── CoM helper ────────────────────────────────────────────────────────

    def _compute_com_height(self, robot: int) -> float:
        """Compute whole-body centre-of-mass Z coordinate."""
        total_mass = 0.0
        com_z      = 0.0

        # Base link
        base_pos, _ = p.getBasePositionAndOrientation(robot, physicsClientId=self._client)
        base_mass   = p.getDynamicsInfo(robot, -1, physicsClientId=self._client)[0]
        com_z      += base_mass * base_pos[2]
        total_mass += base_mass

        for i in range(p.getNumJoints(robot, physicsClientId=self._client)):
            link_state = p.getLinkState(robot, i, physicsClientId=self._client)
            link_com   = link_state[0]
            link_mass  = p.getDynamicsInfo(robot, i, physicsClientId=self._client)[0]
            com_z     += link_mass * link_com[2]
            total_mass += link_mass

        return com_z / max(total_mass, 1e-9)

    # ── foot contact detection ────────────────────────────────────────────

    def _check_foot_contacts(self, robot: int) -> Tuple[bool, bool]:
        """Return (right_foot_contact, left_foot_contact) with the ground plane."""
        right = False
        left  = False
        for contact in p.getContactPoints(robot, self._plane,
                                          physicsClientId=self._client):
            link_idx = contact[3]
            info     = p.getJointInfo(robot, link_idx, physicsClientId=self._client)
            name     = info[1].decode("utf-8").lower()
            if "right" in name and ("foot" in name or "ankle" in name):
                right = True
            if "left" in name and ("foot" in name or "ankle" in name):
                left = True
        return right, left

    # ── Kinematic joint setters ───────────────────────────────────────────

    def _set_joint_kinematic(self, robot: int, jmap: dict, kw: str, angle_deg: float):
        """Kinematically set a joint by keyword (no forces, direct position)."""
        info = _find_joint(jmap, kw)
        if info is None:
            return
        angle_rad = math.radians(angle_deg)
        if info["type"] == p.JOINT_REVOLUTE:
            p.resetJointState(robot, info["index"], -angle_rad,
                              physicsClientId=self._client)
        elif info["type"] == p.JOINT_SPHERICAL:
            p.resetJointStateMultiDof(robot, info["index"], _sagittal_quat(angle_deg),
                                      physicsClientId=self._client)

    def _set_joint_pd(self, robot: int, jmap: dict, kw: str, angle_deg: float):
        """Apply PD position control to a joint by keyword."""
        info = _find_joint(jmap, kw)
        if info is None:
            return
        if info["type"] == p.JOINT_REVOLUTE:
            self._set_revolute(robot, info["index"], math.radians(angle_deg))
        elif info["type"] == p.JOINT_SPHERICAL:
            self._set_spherical(robot, info["index"], _sagittal_quat(angle_deg))

    # ── main simulation loop ──────────────────────────────────────────────

    def run(
        self,
        mocap_segment: dict,
        predicted_knee: np.ndarray,
        fps: float = 200.0,
        gif_output: Optional[str] = None,          # legacy: single combined GIF
        gif_output_pred: Optional[str] = None,     # GIF: prosthetic robot only
        gif_output_gt: Optional[str] = None,       # GIF: ground-truth robot only
    ) -> dict:
        """
        Drive the humanoid with mocap joint angles (kinematic playback),
        replacing the right knee with `predicted_knee` (degrees).

        Two robots are simulated side-by-side (separated 1.5 m in Y):
          • Predicted (Y=0): grey body, **orange right leg** (prosthetic joint)
          • Ground truth (Y=1.5): all joints from mocap (semi-transparent blue)

        Each robot is captured from its own dedicated side-view camera, producing
        two independent GIF files so the viewer can compare them without overlap.

        Parameters
        ----------
        mocap_segment    : dict of (T,) arrays from motion_matching
        predicted_knee   : (T,) right-knee flexion in degrees (model output)
        fps              : playback rate (Hz)
        gif_output       : legacy path — saves a combined wide-angle GIF showing
                           both robots; use gif_output_pred/gt instead
        gif_output_pred  : path for the prosthetic (predicted) robot GIF
        gif_output_gt    : path for the ground-truth robot GIF

        Returns
        -------
        dict of evaluation metrics (see module docstring)
        """
        T = len(predicted_knee)
        assert len(mocap_segment["knee_right"]) >= T

        # ── Root trajectory ────────────────────────────────────────────────
        # The GT robot follows this trajectory kinematically.
        # The predicted robot is initialised here and then released to physics.
        t_s = np.arange(T, dtype=np.float32) / fps
        root_traj = np.zeros((T, 3), dtype=np.float32)
        root_traj[:, 0] = 1.35 * t_s           # constant forward walk speed
        root_traj[:, 2] = HUMANOID_INIT_HEIGHT  # constant pelvis height

        # ── Load both robots ───────────────────────────────────────────────
        # Predicted robot at Y=0 (physics-driven after init).
        # Ground-truth robot at Y=50 — far enough that neither camera captures
        # the other robot (55° FOV, cameras 6 m away → ±4.2 m reach at Y=0/50).
        _GT_Y = 50.0
        init_pos_pred = [float(root_traj[0, 0]), 0.0,   float(root_traj[0, 2])]
        init_pos_gt   = [float(root_traj[0, 0]), _GT_Y, float(root_traj[0, 2])]

        self._robot = self._load_robot(init_pos_pred, HUMANOID_INIT_QUAT)
        robot = self._robot
        jmap  = _discover_joints(robot)

        self._robot_gt = self._load_robot(init_pos_gt, HUMANOID_INIT_QUAT)
        robot_gt = self._robot_gt
        jmap_gt  = _discover_joints(robot_gt)

        # ── Colour coding ──────────────────────────────────────────────────
        _color_robot_prosthetic(robot, jmap, self._client)
        _color_robot_ghost(robot_gt, self._client)
        if self.use_gui:
            self._debug_ids: List[int] = []

        # ── Initialise predicted robot pose, then release to physics ──────
        frame0 = {
            "hip_right":   float(mocap_segment["hip_right"][0]),
            "knee_right":  float(mocap_segment["knee_right"][0]),
            "ankle_right": float(mocap_segment["ankle_right"][0]),
            "hip_left":    float(mocap_segment["hip_left"][0]),
            "knee_left":   float(mocap_segment["knee_left"][0]),
            "ankle_left":  float(mocap_segment["ankle_left"][0]),
        }
        self._init_pose(robot, frame0)
        self._disable_motors(robot)

        accum          = _MetricsAccum(fall_threshold=self.fall_threshold)
        com_heights_gt: List[float] = []
        dt             = 1.0 / fps

        # ── GIF frame buffers ─────────────────────────────────────────────
        want_pred = gif_output_pred is not None
        want_gt   = gif_output_gt   is not None
        want_both = gif_output       is not None
        _frames_pred: Optional[List] = [] if want_pred else None
        _frames_gt:   Optional[List] = [] if want_gt   else None
        _frames_both: Optional[List] = [] if want_both else None

        # Full-body side view: camera 6 m away, centred at mid-body Z=3.0.
        # Visible Z range: 3.0 ± 6*tan(27.5°) = 3.0 ± 3.1 → [−0.1, 6.1].
        _proj = p.computeProjectionMatrixFOV(
            fov=55, aspect=640.0 / 480.0,
            nearVal=0.1, farVal=100.0,
            physicsClientId=self._client,
        )
        _CAM_DIST = 6.0
        _CAM_Z    = 3.0   # mid-body target height

        for t in range(T):
            frame_start = _time.time()

            fr = {
                "hip_right":   float(mocap_segment["hip_right"][t]),
                "hip_left":    float(mocap_segment["hip_left"][t]),
                "knee_right":  float(mocap_segment["knee_right"][t]),
                "knee_left":   float(mocap_segment["knee_left"][t]),
                "ankle_right": float(mocap_segment["ankle_right"][t]),
                "ankle_left":  float(mocap_segment["ankle_left"][t]),
            }
            pred_knee_deg = float(predicted_knee[t])

            # ── Predicted robot: PD control, root free (physics) ─────────
            for kw, ang in [
                ("righthip",   fr["hip_right"]),
                ("lefthip",    fr["hip_left"]),
                ("leftknee",   fr["knee_left"]),
                ("rightankle", fr["ankle_right"]),
                ("leftankle",  fr["ankle_left"]),
            ]:
                self._set_joint_pd(robot, jmap, kw, ang)
            # ★ RIGHT KNEE = model prediction (prosthetic substitution) ★
            self._set_joint_pd(robot, jmap, "rightknee", pred_knee_deg)

            # ── Ground-truth robot: kinematic playback ────────────────────
            gt_pos_t = [float(root_traj[t, 0]), _GT_Y, float(root_traj[t, 2])]
            p.resetBasePositionAndOrientation(
                robot_gt, gt_pos_t, HUMANOID_INIT_QUAT,
                physicsClientId=self._client,
            )
            for kw, ang in [
                ("righthip",   fr["hip_right"]),
                ("rightknee",  fr["knee_right"]),
                ("lefthip",    fr["hip_left"]),
                ("leftknee",   fr["knee_left"]),
                ("rightankle", fr["ankle_right"]),
                ("leftankle",  fr["ankle_left"]),
            ]:
                self._set_joint_kinematic(robot_gt, jmap_gt, kw, ang)

            # Physics sub-steps (advances simulation by SIM_TIMESTEP total)
            for _ in range(self.sub_steps):
                p.stepSimulation(physicsClientId=self._client)

            # ── Frame capture (every _CAPTURE_EVERY steps) ────────────────
            if t % _CAPTURE_EVERY == 0:
                # Track predicted robot's current X position
                rx = float(p.getBasePositionAndOrientation(
                    robot, physicsClientId=self._client)[0][0])

                if _frames_pred is not None:
                    _view = p.computeViewMatrix(
                        cameraEyePosition=[rx, -_CAM_DIST, _CAM_Z],
                        cameraTargetPosition=[rx, 0.0, _CAM_Z],
                        cameraUpVector=[0, 0, 1],
                        physicsClientId=self._client,
                    )
                    _, _, _rgba, _, _ = p.getCameraImage(
                        640, 480, _view, _proj,
                        renderer=p.ER_TINY_RENDERER,
                        physicsClientId=self._client,
                    )
                    _frames_pred.append(
                        np.array(_rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
                    )

                if _frames_gt is not None:
                    _view = p.computeViewMatrix(
                        cameraEyePosition=[rx, _GT_Y - _CAM_DIST, _CAM_Z],
                        cameraTargetPosition=[rx, _GT_Y, _CAM_Z],
                        cameraUpVector=[0, 0, 1],
                        physicsClientId=self._client,
                    )
                    _, _, _rgba, _, _ = p.getCameraImage(
                        640, 480, _view, _proj,
                        renderer=p.ER_TINY_RENDERER,
                        physicsClientId=self._client,
                    )
                    _frames_gt.append(
                        np.array(_rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
                    )

                if _frames_both is not None:
                    _view = p.computeViewMatrix(
                        cameraEyePosition=[rx, -_CAM_DIST, _CAM_Z],
                        cameraTargetPosition=[rx, 0.0, _CAM_Z],
                        cameraUpVector=[0, 0, 1],
                        physicsClientId=self._client,
                    )
                    _, _, _rgba, _, _ = p.getCameraImage(
                        640, 480, _view, _proj,
                        renderer=p.ER_TINY_RENDERER,
                        physicsClientId=self._client,
                    )
                    _frames_both.append(
                        np.array(_rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]
                    )

            # ── Metrics ───────────────────────────────────────────────────
            com_h    = self._compute_com_height(robot)
            com_h_gt = self._compute_com_height(robot_gt)
            com_heights_gt.append(com_h_gt)
            right_c, left_c = self._check_foot_contacts(robot)
            accum.record(t, com_h, pred_knee_deg, fr["knee_right"], right_c, left_c)

            # ── GUI: camera tracking ───────────────────────────────────────
            if self.use_gui:
                _update_gui_overlay(
                    self._client, robot, robot_gt, root_traj, t, T,
                    fps, pred_knee_deg, fr["knee_right"], com_h,
                    accum.fall_detected, self._debug_ids,
                )
                elapsed = _time.time() - frame_start
                sleep   = dt - elapsed
                if sleep > 0:
                    _time.sleep(sleep)

        p.removeBody(robot,    physicsClientId=self._client)
        p.removeBody(robot_gt, physicsClientId=self._client)
        self._robot    = None
        self._robot_gt = None

        # ── Save GIFs ─────────────────────────────────────────────────────
        capture_fps = fps / _CAPTURE_EVERY
        if _frames_pred:
            _save_gif(_frames_pred, gif_output_pred, capture_fps=capture_fps)
        if _frames_gt:
            _save_gif(_frames_gt,   gif_output_gt,   capture_fps=capture_fps)
        if _frames_both:
            _save_gif(_frames_both, gif_output,       capture_fps=capture_fps)

        metrics = accum.summarise()

        gt_arr   = np.array(com_heights_gt)
        pred_arr = np.array(accum.com_heights)
        metrics["com_deviation_mean"] = float(np.mean(np.abs(pred_arr - gt_arr)))
        metrics["com_deviation_max"]  = float(np.max(np.abs(pred_arr - gt_arr)))

        return metrics


# ── Kinematic reference (no physics, pure FK) ─────────────────────────────────


def run_kinematic_evaluation(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
) -> dict:
    """
    Lightweight kinematic evaluation that does NOT require PyBullet.

    Computes the kinematic deviation caused by substituting the predicted
    knee angle for the mocap knee angle.  Uses a simplified sagittal-plane
    2-segment model (thigh + shank) to estimate the resulting foot position
    and CoM shift.

    This is useful as a fallback when pybullet is not installed, or as a fast
    complement to the full physics simulation.

    Parameters
    ----------
    mocap_segment  : dict from motion_matching (same format as ProstheticSimulator.run)
    predicted_knee : (T,) predicted right-knee flexion in degrees

    Returns
    -------
    dict of evaluation metrics
    """
    T = len(predicted_knee)
    assert T > 0

    # Segment lengths (scaled to 1.0; real values proportional to height)
    L_THIGH = 0.40   # metres (≈ thigh length for 1.7 m person)
    L_SHANK = 0.38

    def foot_pos_sagittal(hip_ang_deg: float, knee_ang_deg: float) -> np.ndarray:
        """
        2-D foot position (x,z) relative to hip, sagittal plane.
        Angles positive = flexion (knee flexion bends leg backwards).
        """
        hip_rad   = math.radians(hip_ang_deg)
        knee_rad  = math.radians(knee_ang_deg)
        # Thigh segment (hip to knee) — hip angle rotates thigh forward
        thigh_x = -L_THIGH * math.sin(hip_rad)
        thigh_z = -L_THIGH * math.cos(hip_rad)
        # Shank segment (knee to ankle) — knee angle folds shank backward
        shank_ang = hip_rad - knee_rad   # global angle of shank
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

    # CoM approximation: right-side thigh midpoint deviation
    com_dev = foot_dev / 2.0   # rough: error propagates about halfway to CoM

    knee_rmse = float(np.sqrt(np.mean(knee_err ** 2)))
    knee_mae  = float(np.mean(np.abs(knee_err)))

    # Heuristic fall proxy: maximum foot deviation > 0.15 m suggests instability
    max_dev = float(np.max(foot_dev))
    fall_detected = max_dev > 0.15

    # Stability score
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
        "step_count":          -1,   # not computable without physics
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
    gif_output: Optional[str] = None,         # legacy: combined wide-angle GIF
    gif_output_pred: Optional[str] = None,    # prosthetic robot GIF
    gif_output_gt: Optional[str] = None,      # ground-truth robot GIF
) -> dict:
    """
    High-level entry point.

    Attempts PyBullet physics simulation; falls back to kinematic evaluation
    if PyBullet is unavailable or fails.

    GIF outputs
    -----------
    gif_output_pred  : animated GIF of the prosthetic robot (right knee =
                       model prediction, highlighted in orange).
    gif_output_gt    : animated GIF of the ground-truth robot (all joints from
                       mocap, semi-transparent blue).
    gif_output       : legacy combined wide-angle view (both robots).

    Frame capture uses PyBullet's built-in software renderer — works headless.

    Parameters
    ----------
    mocap_segment   : matched mocap segment dict (from motion_matching)
    predicted_knee  : (T,) right-knee angle predictions in degrees
    use_physics     : if False, use lightweight kinematic evaluation only
    use_gui         : show PyBullet GUI (requires a display)
    fps             : playback rate (Hz)

    Returns
    -------
    dict of evaluation metrics
    """
    if use_physics and _PYBULLET_AVAILABLE:
        effective_gui = use_gui
        if use_gui and not os.environ.get("DISPLAY"):
            warnings.warn(
                "No DISPLAY found — running headless PyBullet."
                + (" Saving visualisation to GIF(s) instead."
                   if (gif_output or gif_output_pred or gif_output_gt) else ""),
                RuntimeWarning,
                stacklevel=2,
            )
            effective_gui = False
        try:
            with ProstheticSimulator(use_gui=effective_gui) as sim:
                metrics = sim.run(
                    mocap_segment, predicted_knee, fps=fps,
                    gif_output=gif_output,
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


# ── GUI visualisation helpers ────────────────────────────────────────────────


def _color_robot_prosthetic(robot: int, jmap: Dict[str, dict], client: int):
    """
    Color the predicted robot:
      - right knee/leg links orange (prosthetic)
      - body neutral grey
    """
    n = p.getNumJoints(robot, physicsClientId=client)
    # Base link
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
    """Color the ground-truth robot semi-transparent blue."""
    n = p.getNumJoints(robot, physicsClientId=client)
    blue = [0.2, 0.5, 0.9, 0.5]
    p.changeVisualShape(robot, -1, rgbaColor=blue, physicsClientId=client)
    for i in range(n):
        p.changeVisualShape(robot, i, rgbaColor=blue, physicsClientId=client)


def _update_gui_overlay(
    client: int,
    robot_pred: int,
    robot_gt: int,
    root_traj: np.ndarray,
    t: int,
    T: int,
    fps: float,
    pred_knee: float,
    gt_knee: float,
    com_h: float,
    fall: bool,
    debug_ids: List[int],
):
    """Update camera, HUD text, and labels each frame."""
    # Remove previous debug text
    for did in debug_ids:
        p.removeUserDebugItem(did, physicsClientId=client)
    debug_ids.clear()

    rx = float(root_traj[t, 0])
    rz = float(root_traj[t, 2])

    # Camera: side view tracking the humanoid
    p.resetDebugVisualizerCamera(
        cameraDistance=2.8,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[rx, 0.75, rz],
        physicsClientId=client,
    )

    # Labels above each robot
    d1 = p.addUserDebugText(
        "PREDICTED",
        [rx, -0.3, rz + 0.7],
        textColorRGB=[1.0, 0.4, 0.1],
        textSize=1.5,
        lifeTime=0,
        physicsClientId=client,
    )
    d2 = p.addUserDebugText(
        "GROUND TRUTH",
        [rx, 1.8, rz + 0.7],
        textColorRGB=[0.2, 0.5, 0.9],
        textSize=1.5,
        lifeTime=0,
        physicsClientId=client,
    )
    debug_ids.extend([d1, d2])

    # HUD: time, knee error, CoM height
    time_s = t / fps
    knee_err = abs(pred_knee - gt_knee)
    status = "FALL!" if fall else "OK"
    hud = (
        f"t={time_s:5.2f}s  [{t}/{T}]\n"
        f"R-Knee pred={pred_knee:5.1f}  gt={gt_knee:5.1f}  err={knee_err:4.1f}\n"
        f"CoM height={com_h:.3f}m   {status}"
    )
    d3 = p.addUserDebugText(
        hud,
        [rx - 1.5, -0.5, rz + 1.2],
        textColorRGB=[1, 1, 1],
        textSize=1.2,
        lifeTime=0,
        physicsClientId=client,
    )
    debug_ids.append(d3)


# ── Standalone visualisation demo ────────────────────────────────────────────


def run_visual_demo(use_full_db: bool = False):
    """
    Launch a PyBullet GUI window showing a walking simulation demo.

    The query is generated from Winter (2009) biomechanical norms with
    simulated IMU sensor noise — matching what rigtest.py would record.
    Pass use_full_db=True to load real CMU mocap data (downloads if needed).
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

    fps      = TARGET_FPS   # 200 Hz
    CADENCE  = 110.0        # steps/min — natural comfortable walking
    cycle_s  = 60.0 / (CADENCE / 2.0)
    spc      = int(round(cycle_s * fps))   # samples per gait cycle (~218)
    N_CYCLES = 3
    T        = spc * N_CYCLES             # ~654 frames (~3.3 s)

    rng = np.random.default_rng(42)

    # Realistic query: Winter 2009 kinematics + IMU sensor noise
    knee_true  = np.tile(_interp_gait_curve(_KNEE_R, spc), N_CYCLES).astype(np.float32)
    thigh_true = np.tile(_interp_gait_curve(_HIP_R,  spc), N_CYCLES).astype(np.float32)
    knee_imu   = knee_true  + rng.normal(0, 2.0, T).astype(np.float32)
    thigh_imu  = thigh_true + rng.normal(0, 1.5, T).astype(np.float32)

    # Simulated model prediction: true knee + ~5° RMS error
    predicted = knee_true + rng.normal(0, 5.0, T).astype(np.float32)

    # Load database
    if use_full_db:
        db = load_full_cmu_database()
    else:
        db = load_or_generate_mocap_database()

    db_dur = len(db["knee_right"]) / fps
    print(f"  Database: {db_dur:.1f}s  source={db['source']}")
    print(f"  Query   : {T} frames ({T/fps:.2f}s), Winter 2009 kinematics + IMU noise")

    # Motion match
    start, dist, segment = find_best_match(knee_imu, thigh_imu, db)
    cat = segment.get("category", "unknown")
    print(f"  Match   : start={start}, DTW={dist:.4f}, category={cat}")

    # Run with GUI
    print(f"\n  Launching PyBullet window — {T/fps:.1f}s playback …")
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
                    help="Run visual demo with synthetic data")
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
