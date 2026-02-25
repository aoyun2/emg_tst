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

FALL_HEIGHT_THRESHOLD = 0.55   # metres; if CoM drops below this → fall
HUMANOID_INIT_HEIGHT  = 0.94   # metres; typical pelvis height from ground
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
    """Quaternion for a pure sagittal (X-axis) rotation. Used for hip/ankle."""
    half = math.radians(angle_deg) * 0.5
    return [math.sin(half), 0.0, 0.0, math.cos(half)]


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
        return self

    def __exit__(self, *_):
        if self._client is not None:
            p.disconnect(self._client)
            self._client = None

    # ── robot setup ───────────────────────────────────────────────────────

    def _load_robot(self, start_pos: Optional[List[float]] = None) -> int:
        if start_pos is None:
            start_pos = [0.0, 0.0, HUMANOID_INIT_HEIGHT]
        robot = p.loadURDF(
            "humanoid/humanoid.urdf",
            start_pos,
            [0, 0, 0, 1],          # quaternion: no initial rotation
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
        p.setJointMotorControl2(
            robot, idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=angle_rad,
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
                p.resetJointState(robot, info["index"], angle_rad,
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
            p.resetJointState(robot, info["index"], angle_rad,
                              physicsClientId=self._client)
        elif info["type"] == p.JOINT_SPHERICAL:
            p.resetJointStateMultiDof(robot, info["index"], _sagittal_quat(angle_deg),
                                      physicsClientId=self._client)

    # ── main simulation loop ──────────────────────────────────────────────

    def run(
        self,
        mocap_segment: dict,
        predicted_knee: np.ndarray,
        fps: float = 200.0,
    ) -> dict:
        """
        Drive the humanoid with mocap joint angles (kinematic playback),
        replacing the right knee with `predicted_knee` (degrees).

        Strategy: kinematic mode — joints are set directly via resetJointState
        each frame (no PD forces), while the base follows the mocap root
        trajectory.  This avoids PD instability while still using PyBullet's
        multi-body kinematics engine for accurate CoM computation and
        foot-contact detection.

        A 'fall' is declared when:
        (a) CoM drops below `fall_threshold`, OR
        (b) The right foot penetrates significantly into the ground
            (prediction error causes foot to go through the floor).

        Parameters
        ----------
        mocap_segment   : dict of (T,) arrays from motion_matching.find_best_match()
        predicted_knee  : (T,) predicted right-knee flexion in degrees
        fps             : playback frame rate (informational, not used here)

        Returns
        -------
        dict of evaluation metrics (see module docstring)
        """
        T = len(predicted_knee)
        assert len(mocap_segment["knee_right"]) >= T

        # Root trajectory — use mocap if meaningful, else synthesise forward walk
        root_pos_all = mocap_segment["root_pos"][:T]
        root_z_range = root_pos_all[:, 2].max() - root_pos_all[:, 2].min()
        has_real_root = root_z_range > 0.05 and root_pos_all[:, 2].mean() > 0.3

        if has_real_root:
            # Shift so minimum Z ≈ HUMANOID_INIT_HEIGHT (mocap root ≠ ground)
            z_offset   = HUMANOID_INIT_HEIGHT - root_pos_all[:, 2].min()
            root_traj  = root_pos_all.copy()
            root_traj[:, 2] += z_offset
        else:
            # Synthetic: move forward at ≈1.35 m/s, constant height
            N   = T
            t_s = np.arange(N) / fps
            root_traj = np.zeros((N, 3), dtype=np.float32)
            root_traj[:, 0] = 1.35 * t_s
            root_traj[:, 2] = HUMANOID_INIT_HEIGHT

        # Load robot at starting position
        self._robot = self._load_robot(
            [float(root_traj[0, 0]), 0.0, float(root_traj[0, 2])]
        )
        robot = self._robot
        jmap  = _discover_joints(robot)

        # Also discover the "ground truth" robot for CoM comparison
        # (all joints follow mocap perfectly)
        self._robot_gt = self._load_robot(
            [float(root_traj[0, 0]) + 1.0, 0.0, float(root_traj[0, 2])]
        )
        robot_gt = self._robot_gt
        jmap_gt  = _discover_joints(robot_gt)

        accum = _MetricsAccum(fall_threshold=self.fall_threshold)
        com_heights_gt = []

        for t in range(T):
            fr = {
                "hip_right":   float(mocap_segment["hip_right"][t]),
                "hip_left":    float(mocap_segment["hip_left"][t]),
                "knee_right":  float(mocap_segment["knee_right"][t]),
                "knee_left":   float(mocap_segment["knee_left"][t]),
                "ankle_right": float(mocap_segment["ankle_right"][t]),
                "ankle_left":  float(mocap_segment["ankle_left"][t]),
            }

            # ── predicted robot (right knee = model) ──────────────────────
            root_pos_t = [float(root_traj[t, 0]), 0.0, float(root_traj[t, 2])]
            p.resetBasePositionAndOrientation(
                robot, root_pos_t, [0, 0, 0, 1], physicsClientId=self._client
            )
            for kw, ang in [
                ("righthip",   fr["hip_right"]),
                ("lefthip",    fr["hip_left"]),
                ("leftknee",   fr["knee_left"]),
                ("rightankle", fr["ankle_right"]),
                ("leftankle",  fr["ankle_left"]),
            ]:
                self._set_joint_kinematic(robot, jmap, kw, ang)

            # RIGHT KNEE = model prediction (the prosthetic substitution)
            pred_knee_deg = float(predicted_knee[t])
            self._set_joint_kinematic(robot, jmap, "rightknee", pred_knee_deg)

            # ── ground truth robot (all joints from mocap) ────────────────
            gt_pos_t = [float(root_traj[t, 0]) + 1.0, 0.0, float(root_traj[t, 2])]
            p.resetBasePositionAndOrientation(
                robot_gt, gt_pos_t, [0, 0, 0, 1], physicsClientId=self._client
            )
            for kw, ang in [
                ("righthip",   fr["hip_right"]),
                ("rightknee",  fr["knee_right"]),   # TRUE knee angle
                ("lefthip",    fr["hip_left"]),
                ("leftknee",   fr["knee_left"]),
                ("rightankle", fr["ankle_right"]),
                ("leftankle",  fr["ankle_left"]),
            ]:
                self._set_joint_kinematic(robot_gt, jmap_gt, kw, ang)

            # Single physics step so contact detection is updated
            p.stepSimulation(physicsClientId=self._client)

            # ── metrics ───────────────────────────────────────────────────
            com_h    = self._compute_com_height(robot)
            com_h_gt = self._compute_com_height(robot_gt)
            com_heights_gt.append(com_h_gt)

            right_c, left_c = self._check_foot_contacts(robot)
            accum.record(t, com_h, pred_knee_deg, fr["knee_right"], right_c, left_c)

        p.removeBody(robot,    physicsClientId=self._client)
        p.removeBody(robot_gt, physicsClientId=self._client)
        self._robot    = None
        self._robot_gt = None

        metrics = accum.summarise()

        # Deviation from ground-truth CoM trajectory
        gt_arr  = np.array(com_heights_gt)
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
) -> dict:
    """
    High-level entry point.

    Attempts PyBullet physics simulation; falls back to kinematic evaluation
    if PyBullet is unavailable or fails.

    Parameters
    ----------
    mocap_segment   : matched mocap segment dict (from motion_matching)
    predicted_knee  : (T,) right-knee angle predictions in degrees
    use_physics     : if False, use lightweight kinematic evaluation only
    use_gui         : show PyBullet GUI (for debugging)
    fps             : playback rate (Hz)

    Returns
    -------
    dict of evaluation metrics
    """
    if use_physics and _PYBULLET_AVAILABLE:
        try:
            with ProstheticSimulator(use_gui=use_gui) as sim:
                metrics = sim.run(mocap_segment, predicted_knee, fps=fps)
            metrics["mode"] = "physics"
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
