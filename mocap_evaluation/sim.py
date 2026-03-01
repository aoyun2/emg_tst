"""Physics simulation: kinematic replay of a MoCap Act clip with knee override.

Architecture
============
1. Load the matched clip's full reference trajectory from the MoCap Act HDF5.
2. For each frame:
   a. Set ALL joints from the reference (so the whole body walks correctly).
   b. Override the RIGHT KNEE with the model's predicted angle.
   c. Call mj_forward (propagate kinematics — no dynamics / policy needed).
3. Render live via ``mujoco.viewer.launch_passive()`` — NON-BLOCKING, never freezes.
4. Collect heuristics: CoM height, foot positions, fall detection, step count.

No MoCap Act policy is used — this is pure kinematic replay, which is
simpler, deterministic, and guaranteed not to hang.

Viewer behaviour
----------------
* ``use_viewer=True``  : opens an interactive MuJoCo window; animation plays
  once at real-time speed, then the window stays open until the user closes it.
* ``use_viewer=False`` : headless run, no window opened; metrics collected as
  usual.  Add ``--no-gui`` to ``run_sim.py`` to trigger this mode.

Fall detection
--------------
The torso CoM height is tracked every frame.  A fall is detected when it drops
below ``FALL_HEIGHT_M`` (default 0.5 m).

Gait symmetry
-------------
Right / left step intervals are estimated from sign changes in the
right/left foot z-velocity.  Ratio close to 1.0 = symmetric gait.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from mocap_evaluation.db import (
    KNEE_QPOS_IDX,
    HIP_QPOS_IDX,
    _ROOT_DOFS,
    _NATIVE_FPS,
    _KNEE_JOINT,
    _HIP_JOINT,
    get_h5_path,
    included_to_knee_rad,
    included_to_hip_rad,
    knee_rad_to_included,
    hip_rad_to_included,
    _get_mocap_joint_indices,
)

# ── Physics constants ─────────────────────────────────────────────────────────

FALL_HEIGHT_M = 0.50   # CoM below this → fall detected
SIM_FPS       = 30.0   # native MoCap Act rate for playback

# Root qpos layout for CMU humanoid: pos(3) + quat(4) + joints(56) = 63 total
_CMU_N_JOINTS = 56
_CMU_QPOS_DIM = _ROOT_DOFS + _CMU_N_JOINTS   # 63


# ── Reference trajectory loading ──────────────────────────────────────────────

def _load_full_trajectory(clip_id: str) -> Optional[Tuple[np.ndarray, float]]:
    """Load a clip's full qpos trajectory (root + all joints) from HDF5.

    Returns (qpos_array, fps) where qpos_array has shape (T, 63),
    or None if the clip cannot be loaded.

    The assembly follows the MuJoCo CMU humanoid qpos layout:
      [0:3]   root position (x, y, z)
      [3:7]   root quaternion (w, x, y, z)  — MuJoCo convention
      [7:63]  hinge joints in mocap_joints order

    If root position/quaternion are absent in the HDF5, the humanoid is
    placed at a fixed stand-up pose (no forward locomotion, but joint
    angles still animate correctly for heuristic evaluation).
    """
    h5_path = get_h5_path()
    if h5_path is None:
        return None

    try:
        import h5py
        knee_mi, hip_mi = _get_mocap_joint_indices(h5_path)

        with h5py.File(str(h5_path), "r") as f:
            if clip_id not in f:
                warnings.warn(f"[sim] Clip {clip_id!r} not found in HDF5.")
                return None
            grp = f[clip_id]

            # ── Hinge joints (56 DOFs) ──────────────────────────────────────
            joints_data = None
            for key in ("walkers/walker_0/joints", "walkers/0/joints", "joints"):
                if key not in grp:
                    continue
                arr = np.asarray(grp[key], dtype=np.float64)
                # Stored as (n_joints, T) → transpose to (T, n_joints)
                if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                if arr.ndim == 2 and arr.shape[1] >= _CMU_N_JOINTS:
                    joints_data = arr[:, :_CMU_N_JOINTS]
                    break
            if joints_data is None:
                return None

            T = len(joints_data)

            # ── Root position (3) ────────────────────────────────────────────
            pos = None
            for key in ("walkers/walker_0/position", "walkers/0/position", "position"):
                if key not in grp:
                    continue
                arr = np.asarray(grp[key], dtype=np.float64)
                if arr.shape == (3, T):
                    arr = arr.T
                if arr.shape == (T, 3):
                    pos = arr
                    break

            # ── Root quaternion (4) in HDF5 convention (x,y,z,w) ────────────
            quat = None
            for key in ("walkers/walker_0/quaternion", "walkers/0/quaternion", "quaternion"):
                if key not in grp:
                    continue
                arr = np.asarray(grp[key], dtype=np.float64)
                if arr.shape == (4, T):
                    arr = arr.T
                if arr.shape == (T, 4):
                    quat = arr
                    break

            # ── Fallback: fixed neutral pose ─────────────────────────────────
            if pos is None:
                pos = np.zeros((T, 3), dtype=np.float64)
                pos[:, 2] = 1.2   # stand ~1.2 m above ground
            if quat is None:
                # identity quaternion (w=1, x=y=z=0)
                quat = np.zeros((T, 4), dtype=np.float64)
                quat[:, 0] = 1.0
            else:
                # HDF5 stores (x,y,z,w); MuJoCo expects (w,x,y,z) — reorder
                quat = np.concatenate([quat[:, 3:4], quat[:, :3]], axis=1)

            qpos = np.concatenate([pos, quat, joints_data], axis=1)  # (T, 63)

            dt  = float(grp.attrs.get("dt", 1.0 / _NATIVE_FPS))
            fps = 1.0 / dt if dt > 0 else _NATIVE_FPS

            return qpos, fps

    except Exception as exc:
        warnings.warn(f"[sim] Failed to load trajectory for {clip_id!r}: {exc}")
        return None


def _load_via_trajectory_loader(clip_id: str) -> Optional[Tuple[np.ndarray, float]]:
    """Fallback: use dm_control HDF5TrajectoryLoader to get qpos."""
    try:
        h5_path = get_h5_path()
        if h5_path is None:
            return None
        from dm_control.locomotion.mocap import loader as mocap_loader
        if not hasattr(mocap_loader, "HDF5TrajectoryLoader"):
            return None
        tl   = mocap_loader.HDF5TrajectoryLoader(str(h5_path))
        traj = tl.get_trajectory(clip_id) if hasattr(tl, "get_trajectory") else tl.load(clip_id)
        fps  = float(getattr(traj, "fps", _NATIVE_FPS))
        if fps <= 0:
            fps = 1.0 / float(getattr(traj, "dt", 1.0 / _NATIVE_FPS))
        try:
            d = traj.as_dict()
            if "qpos" in d:
                qpos = np.asarray(d["qpos"], dtype=np.float64)
                if qpos.ndim == 2 and qpos.shape[1] >= _CMU_QPOS_DIM:
                    return qpos[:, :_CMU_QPOS_DIM], fps
        except Exception:
            pass
        arr = getattr(traj, "qpos", None)
        if arr is not None:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= _CMU_QPOS_DIM:
                return arr[:, :_CMU_QPOS_DIM], fps
    except Exception:
        pass
    return None


def load_clip_trajectory(clip_id: str) -> Tuple[np.ndarray, float]:
    """Load full qpos trajectory for clip_id.

    Returns (qpos, fps).  qpos has shape (T, 63).
    Tries h5py direct read first, then HDF5TrajectoryLoader.
    """
    result = _load_full_trajectory(clip_id)
    if result is not None:
        return result
    result = _load_via_trajectory_loader(clip_id)
    if result is not None:
        return result
    raise RuntimeError(
        f"Cannot load trajectory for clip {clip_id!r}. "
        "Check that the MoCap Act HDF5 file is accessible."
    )


# ── MuJoCo model creation ─────────────────────────────────────────────────────

_PHYSICS_CACHE: Optional[object] = None   # cached physics to avoid repeated init


def _make_physics():
    """Create a dm_control physics instance for the CMU humanoid with a floor."""
    global _PHYSICS_CACHE
    if _PHYSICS_CACHE is not None:
        return _PHYSICS_CACHE

    from dm_control.locomotion.walkers import cmu_humanoid
    from dm_control.locomotion import arenas
    from dm_control import mjcf

    walker = cmu_humanoid.CMUHumanoid()
    arena  = arenas.Floor()
    arena.attach(walker)

    from dm_control import mujoco as dm_mujoco
    physics = dm_mujoco.Physics.from_mjcf_model(arena.mjcf_model)

    _PHYSICS_CACHE = physics
    return physics


def _find_joint_qpos_indices(physics) -> Tuple[int, int]:
    """Return (knee_qpos_idx, hip_qpos_idx) in the combined arena+walker model."""
    knee_idx = hip_idx = None
    for jnt_i in range(physics.model.njnt):
        name = physics.model.id2name(jnt_i, "joint")
        addr = int(physics.model.jnt_qposadr[jnt_i])
        if _KNEE_JOINT in name:
            knee_idx = addr
        elif _HIP_JOINT in name:
            hip_idx = addr
        if knee_idx is not None and hip_idx is not None:
            break
    if knee_idx is None:
        knee_idx = KNEE_QPOS_IDX
        warnings.warn(f"[sim] {_KNEE_JOINT} not found; using fallback idx {KNEE_QPOS_IDX}")
    if hip_idx is None:
        hip_idx = HIP_QPOS_IDX
        warnings.warn(f"[sim] {_HIP_JOINT} not found; using fallback idx {HIP_QPOS_IDX}")
    return knee_idx, hip_idx


def _find_body_id(physics, name_fragment: str) -> Optional[int]:
    """Return first body id whose name contains name_fragment, or None."""
    for i in range(physics.model.nbody):
        name = physics.model.id2name(i, "body")
        if name_fragment.lower() in name.lower():
            return i
    return None


# ── Simulation kernel ─────────────────────────────────────────────────────────

def _run_frames(
    physics,
    ref_qpos: np.ndarray,
    knee_pred_rad: np.ndarray,
    knee_qpos_idx: int,
    hip_pred_rad: Optional[np.ndarray],
    hip_qpos_idx: int,
    ref_knee_rad: Optional[np.ndarray],
    fps: float,
    viewer=None,
) -> dict:
    """Inner loop: advance kinematic replay, collect metrics.

    Parameters
    ----------
    physics       : dm_control Physics
    ref_qpos      : (T, n_qpos) reference qpos for all joints
    knee_pred_rad : (T,) predicted knee angle in MuJoCo radians (negative = flex)
    knee_qpos_idx : qpos index for the right knee joint
    hip_pred_rad  : (T,) predicted thigh/hip angle in MuJoCo radians, or None
    hip_qpos_idx  : qpos index for the right hip joint
    ref_knee_rad  : (T,) reference knee in radians (for RMSE computation), or None
    fps           : playback frame rate (controls time.sleep duration)
    viewer        : mujoco.viewer passive handle, or None for headless
    """
    import mujoco

    n_qpos    = physics.model.nq
    n_frames  = min(len(ref_qpos), len(knee_pred_rad))
    if hip_pred_rad is not None:
        n_frames = min(n_frames, len(hip_pred_rad))
    dt        = 1.0 / fps

    com_heights:   List[float] = []
    rfoot_z:       List[float] = []
    lfoot_z:       List[float] = []
    knee_pred_inc: List[float] = []
    knee_ref_inc:  List[float] = []

    # Body IDs for CoM and foot tracking
    torso_id  = _find_body_id(physics, "thorax") or _find_body_id(physics, "root") or 0
    rfoot_id  = _find_body_id(physics, "rfoot")
    lfoot_id  = _find_body_id(physics, "lfoot")

    t_start = time.perf_counter()

    for frame in range(n_frames):
        if viewer is not None and not viewer.is_running():
            break

        # ── Set reference qpos (clip trajectory) ─────────────────────────────
        n_copy = min(n_qpos, ref_qpos.shape[1])
        physics.data.qpos[:n_copy] = ref_qpos[frame, :n_copy]

        # ── Override right knee + right thigh with query window values ───────
        if 0 <= knee_qpos_idx < n_qpos:
            physics.data.qpos[knee_qpos_idx] = knee_pred_rad[frame]
        if hip_pred_rad is not None and 0 <= hip_qpos_idx < n_qpos:
            physics.data.qpos[hip_qpos_idx] = hip_pred_rad[frame]

        # ── Forward kinematics ────────────────────────────────────────────────
        mujoco.mj_forward(physics.model._model, physics.data._data)

        # ── Collect metrics ───────────────────────────────────────────────────
        torso_z = float(physics.data.xpos[torso_id, 2]) if torso_id is not None else 1.0
        com_heights.append(torso_z)

        if rfoot_id is not None:
            rfoot_z.append(float(physics.data.xpos[rfoot_id, 2]))
        if lfoot_id is not None:
            lfoot_z.append(float(physics.data.xpos[lfoot_id, 2]))

        pred_inc = float(knee_rad_to_included(np.array([knee_pred_rad[frame]]))[0])
        knee_pred_inc.append(pred_inc)
        if ref_knee_rad is not None and frame < len(ref_knee_rad):
            ref_inc = float(knee_rad_to_included(np.array([ref_knee_rad[frame]]))[0])
            knee_ref_inc.append(ref_inc)

        # ── Render ────────────────────────────────────────────────────────────
        if viewer is not None and viewer.is_running():
            viewer.sync()

        # ── Real-time pacing ──────────────────────────────────────────────────
        elapsed   = time.perf_counter() - t_start
        target    = (frame + 1) * dt
        remaining = target - elapsed
        if remaining > 0.002:
            time.sleep(remaining)

    return _compute_metrics(
        com_heights, rfoot_z, lfoot_z, knee_pred_inc, knee_ref_inc, fps
    )


def _compute_metrics(
    com_heights: List[float],
    rfoot_z: List[float],
    lfoot_z: List[float],
    knee_pred_inc: List[float],
    knee_ref_inc: List[float],
    fps: float,
) -> dict:
    """Compute gait heuristics from per-frame measurements."""
    com = np.asarray(com_heights, dtype=np.float32)

    # Fall detection
    fall_mask   = com < FALL_HEIGHT_M
    fall_frame  = int(np.argmax(fall_mask)) if fall_mask.any() else -1
    fall_detect = bool(fall_mask.any())

    # Step count from foot contact (foot z below threshold)
    CONTACT_THRESH = 0.05  # m — foot is on ground
    def _count_steps(z_arr: list) -> Tuple[int, List[int]]:
        if not z_arr:
            return 0, []
        z   = np.asarray(z_arr)
        on  = z < CONTACT_THRESH
        # Count rising edges (lift-offs) as steps
        edges = np.diff(on.astype(int))
        frames_liftoff = list(np.where(edges == -1)[0])
        return len(frames_liftoff), frames_liftoff

    r_steps, r_frames = _count_steps(rfoot_z)
    l_steps, l_frames = _count_steps(lfoot_z)
    total_steps = r_steps + l_steps

    # Gait symmetry: ratio of right/left step intervals
    gait_sym = 1.0
    if r_steps > 1 and l_steps > 1:
        r_intervals = np.diff(r_frames).mean() if len(r_frames) > 1 else 1.0
        l_intervals = np.diff(l_frames).mean() if len(l_frames) > 1 else 1.0
        gait_sym = float(min(r_intervals, l_intervals) / max(r_intervals, l_intervals + 1e-9))

    # Stability score: fraction of frames without fall × gait symmetry
    non_fall_frac  = float(1.0 - fall_mask.mean()) if len(com) else 0.0
    stability_score = non_fall_frac * gait_sym

    # Knee angle RMSE vs reference
    knee_pred = np.asarray(knee_pred_inc)
    if knee_ref_inc:
        knee_ref = np.asarray(knee_ref_inc[:len(knee_pred)])
        n = min(len(knee_pred), len(knee_ref))
        knee_rmse = float(np.sqrt(np.mean((knee_pred[:n] - knee_ref[:n]) ** 2)))
        knee_mae  = float(np.mean(np.abs(knee_pred[:n] - knee_ref[:n])))
    else:
        knee_rmse = knee_mae = float("nan")

    return {
        "fall_detected":      fall_detect,
        "fall_frame":         fall_frame,
        "step_count":         total_steps,
        "right_steps":        r_steps,
        "left_steps":         l_steps,
        "right_step_frames":  r_frames,
        "left_step_frames":   l_frames,
        "gait_symmetry":      gait_sym,
        "stability_score":    stability_score,
        "com_height_mean":    float(com.mean()) if len(com) else 0.0,
        "com_height_std":     float(com.std())  if len(com) else 0.0,
        "com_height_min":     float(com.min())  if len(com) else 0.0,
        "com_heights":        com.tolist(),
        "knee_rmse_deg":      knee_rmse,
        "knee_mae_deg":       knee_mae,
        "knee_pred_deg":      knee_pred.tolist(),
        "n_frames":           len(com),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_simulation(
    clip_id: str,
    clip_start_frame: int,
    n_frames: int,
    knee_pred_included_deg: np.ndarray,
    thigh_pred_included_deg: Optional[np.ndarray] = None,
    use_viewer: bool = True,
    label: str = "",
) -> dict:
    """Run kinematic replay simulation for one scenario.

    The right knee AND right thigh (hip) are both overridden each frame
    using the angles from the query window that was used for motion matching.
    All other joints follow the matched clip reference trajectory.

    Parameters
    ----------
    clip_id                 : MoCap Act clip ID (e.g. ``"CMU_012_03"``)
    clip_start_frame        : Frame offset within the clip's trajectory
                              (from ``matching.clip_info_for_start``).
    n_frames                : Number of frames to simulate (= query length).
    knee_pred_included_deg  : (≥n_frames,) predicted knee angle (included-angle °)
    thigh_pred_included_deg : (≥n_frames,) thigh/hip angle from the query window
                              (included-angle °).  Overrides rfemurry each frame.
    use_viewer              : Open an interactive MuJoCo window.
    label                   : Scenario name for console output.

    Returns
    -------
    dict of heuristic metrics (see ``_compute_metrics``).
    """
    if label:
        print(f"\n  [{label}]  Loading clip trajectory …")

    # ── Load reference trajectory ─────────────────────────────────────────────
    try:
        ref_qpos_full, clip_fps = load_clip_trajectory(clip_id)
    except Exception as exc:
        warnings.warn(f"[sim] {exc}  — running joint-angle-only fallback.")
        return _run_fallback(
            n_frames, knee_pred_included_deg, thigh_pred_included_deg, use_viewer, label
        )

    # Slice to the matched segment + requested length
    start    = min(clip_start_frame, max(0, len(ref_qpos_full) - n_frames))
    ref_qpos = ref_qpos_full[start : start + n_frames]

    # Pad with last frame if the clip is shorter than n_frames
    if len(ref_qpos) < n_frames:
        pad      = n_frames - len(ref_qpos)
        ref_qpos = np.concatenate([ref_qpos, np.tile(ref_qpos[-1:], (pad, 1))])

    from mocap_evaluation.db import _resample, TARGET_FPS

    def _to_clip_fps(sig_200hz: np.ndarray) -> np.ndarray:
        """Resample a 200-Hz signal to clip_fps, then pad/trim to n_frames."""
        if abs(clip_fps - TARGET_FPS) > 0.5:
            sig = _resample(sig_200hz.astype(np.float32), TARGET_FPS, clip_fps)
        else:
            sig = sig_200hz.astype(np.float32)
        sig = sig[:n_frames]
        if len(sig) < n_frames:
            sig = np.concatenate([sig, np.full(n_frames - len(sig), sig[-1])])
        return sig

    knee_at_clip_fps = _to_clip_fps(np.asarray(knee_pred_included_deg))
    knee_pred_rad    = included_to_knee_rad(knee_at_clip_fps).astype(np.float64)

    hip_pred_rad: Optional[np.ndarray] = None
    if thigh_pred_included_deg is not None:
        thigh_at_clip_fps = _to_clip_fps(np.asarray(thigh_pred_included_deg))
        hip_pred_rad      = included_to_hip_rad(thigh_at_clip_fps).astype(np.float64)

    # Reference knee for RMSE (from the ref_qpos itself)
    ref_knee_rad = ref_qpos[:, KNEE_QPOS_IDX].copy() if ref_qpos.shape[1] > KNEE_QPOS_IDX else None

    # ── Build physics ─────────────────────────────────────────────────────────
    try:
        physics = _make_physics()
    except Exception as exc:
        warnings.warn(f"[sim] Cannot create dm_control physics: {exc}")
        return _run_fallback(
            n_frames, knee_pred_included_deg, thigh_pred_included_deg, use_viewer, label
        )

    knee_qpos_idx, hip_qpos_idx = _find_joint_qpos_indices(physics)

    if label:
        thigh_note = f", thigh override={'yes' if hip_pred_rad is not None else 'no'}"
        print(f"  [{label}]  Simulating {n_frames} frames @ {clip_fps:.0f} Hz "
              f"(knee+thigh overrides from query window{thigh_note}) …")

    def _call_run_frames(viewer=None):
        return _run_frames(
            physics, ref_qpos, knee_pred_rad, knee_qpos_idx,
            hip_pred_rad, hip_qpos_idx,
            ref_knee_rad, clip_fps, viewer,
        )

    # ── Run simulation loop ───────────────────────────────────────────────────
    if use_viewer:
        try:
            import mujoco
            import mujoco.viewer
            with mujoco.viewer.launch_passive(
                physics.model._model,
                physics.data._data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                viewer.cam.distance  = 5.0
                viewer.cam.azimuth   = 135.0
                viewer.cam.elevation = -20.0
                metrics = _call_run_frames(viewer)
                if viewer.is_running():
                    print(f"  [{label}]  Animation done — close viewer to continue.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
        except Exception as exc:
            print(f"  [{label}]  Viewer unavailable ({exc}); running headless.")
            metrics = _call_run_frames()
    else:
        metrics = _call_run_frames()

    return metrics


def _run_fallback(
    n_frames: int,
    knee_pred_included_deg: np.ndarray,
    thigh_ref_included_deg: Optional[np.ndarray],
    use_viewer: bool,
    label: str,
) -> dict:
    """Minimal metrics when physics cannot be initialised."""
    warnings.warn(f"[sim] [{label}] Physics init failed — returning angle-only metrics.")
    knee_pred = np.asarray(knee_pred_included_deg[:n_frames], dtype=np.float32)
    # Stable CoM placeholder
    com_heights = [1.0] * n_frames
    knee_ref_inc = (
        list(thigh_ref_included_deg[:n_frames])
        if thigh_ref_included_deg is not None else []
    )
    return _compute_metrics(com_heights, [], [], list(knee_pred), knee_ref_inc, SIM_FPS)


# ── Summary plot ──────────────────────────────────────────────────────────────

def plot_sim_results(
    scenarios: Dict[str, dict],
    scenario_labels: Dict[str, str],
    knee_signals:   Dict[str, np.ndarray],   # scenario_key → included-angle °
    fps: float,
    out_path: str = "sim_results.png",
) -> None:
    """Save a 3-panel summary of simulation results across scenarios.

    Panels: (1) Knee angle trajectories, (2) CoM height, (3) metrics table.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"gt": "steelblue", "good": "darkorange", "bad": "firebrick"}
    n_sc   = len(scenarios)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle("Simulation results — kinematic replay with knee override", fontsize=11)

    # ── Panel 1: Knee angles ──────────────────────────────────────────────────
    ax = axes[0]
    for key, sig in knee_signals.items():
        t = np.arange(len(sig)) / fps
        c = colors.get(key, "gray")
        ax.plot(t, sig, lw=1.8, color=c, label=scenario_labels.get(key, key))
    ax.set_ylabel("Knee included-angle (°)")
    ax.set_title("Knee angle — query vs scenarios")
    ax.set_ylim(0, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: CoM height ───────────────────────────────────────────────────
    ax = axes[1]
    for key, res in scenarios.items():
        com = np.asarray(res["com_heights"])
        t   = np.arange(len(com)) / fps
        c   = colors.get(key, "gray")
        stab = res["stability_score"]
        ax.plot(t, com, lw=1.8, color=c,
                label=f"{scenario_labels.get(key, key)}  stab={stab:.2f}")
    ax.axhline(FALL_HEIGHT_M, color="red", ls="--", lw=1, label=f"Fall threshold ({FALL_HEIGHT_M} m)")
    ax.set_ylabel("Torso height (m)")
    ax.set_title("Centre of Mass height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: metrics table ────────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    col_labels = ["Scenario", "Fall?", "Steps", "Gait sym", "CoM mean", "Knee RMSE"]
    rows = []
    for key, res in scenarios.items():
        rows.append([
            scenario_labels.get(key, key),
            "YES" if res["fall_detected"] else "no",
            str(res["step_count"]),
            f"{res['gait_symmetry']:.3f}",
            f"{res['com_height_mean']:.3f} m",
            f"{res['knee_rmse_deg']:.1f}°" if not np.isnan(res["knee_rmse_deg"]) else "—",
        ])
    tbl = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    ax.set_title("Heuristic metrics", pad=10)

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Simulation plot saved → {out_path}")
