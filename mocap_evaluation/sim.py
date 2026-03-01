"""Physics simulation: kinematic replay of a MoCap Act clip with knee+thigh override.

Architecture
============
1. Load clip trajectory from HDF5: joints (mocap_joints order), root pos, root quat.
2. Build a raw ``mujoco.MjModel`` from the CMU humanoid MJCF + a floor plane.
3. Map HDF5 mocap_joints ordering → MuJoCo qpos addresses using joint names.
4. For each frame:
   a. Set root pos/quat from HDF5 reference.
   b. Set all hinge joints by name (correct ordering regardless of model version).
   c. Override right knee (rtibiarx) and right thigh (rfemurry) from query window.
   d. Call mj_forward → pure forward kinematics, no policy, no dynamics instability.
5. Render live via ``mujoco.viewer.launch_passive(model, data)`` — raw mujoco objects,
   guaranteed non-blocking.
6. Collect heuristics: torso height, foot z-positions, fall detection, step count.

Why raw mujoco instead of dm_control Physics wrapper
------------------------------------------------------
``mujoco.viewer.launch_passive()`` requires a ``mujoco.MjModel`` and ``mujoco.MjData``.
The dm_control Physics wrapper exposes these via private attributes (``._model``,
``._data``, ``.ptr``, …) that differ between dm_control versions.  Using raw mujoco
objects directly is always correct and has no version dependency.

Why joint-name-based setting instead of index-based
-----------------------------------------------------
The HDF5 stores joints in ``mocap_joints`` order (biomechanical grouping),
while the MuJoCo qpos array uses model-definition order.  These orderings differ.
Setting each joint by name via ``name_to_qposadr[joint_name]`` is correct in all
cases and does not require a pre-computed permutation.
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from mocap_evaluation.db import (
    _KNEE_JOINT,
    _HIP_JOINT,
    _NATIVE_FPS,
    get_h5_path,
    included_to_knee_rad,
    included_to_hip_rad,
    knee_rad_to_included,
    TARGET_FPS,
    _resample,
)

# ── Physics constants ─────────────────────────────────────────────────────────

FALL_HEIGHT_M = 0.50   # torso CoM below this → fall detected

# ── MuJoCo model creation ─────────────────────────────────────────────────────

_MODEL_CACHE: Optional[Tuple] = None   # (model, data, name_to_qposadr, mocap_names)


def _make_mujoco_model() -> Tuple:
    """Build a raw mujoco.MjModel for the CMU humanoid with a floor plane.

    Returns
    -------
    (model, data, name_to_qposadr, mocap_names)
        model            : mujoco.MjModel
        data             : mujoco.MjData
        name_to_qposadr  : dict[str, int] — joint name → qpos array index
        mocap_names      : list[str] — joint names in mocap_joints order
                           (= HDF5 walkers/walker_0/joints column order)
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        import mujoco
        model, _, name_to_qposadr, mocap_names = _MODEL_CACHE
        data = mujoco.MjData(model)   # fresh data each call
        return model, data, name_to_qposadr, mocap_names

    import mujoco
    from dm_control.locomotion.walkers import cmu_humanoid

    walker = cmu_humanoid.CMUHumanoid()

    # Add a visible floor plane and decent lighting to the humanoid's worldbody
    wb = walker.mjcf_model.worldbody
    wb.add("geom", type="plane", size=[15, 15, 0.1],
           rgba=[0.8, 0.8, 0.8, 1], name="floor")
    walker.mjcf_model.visual.headlight.set_attributes(
        ambient=[0.45, 0.45, 0.45], diffuse=[0.6, 0.6, 0.6]
    )

    xml = walker.mjcf_model.to_xml_string()
    model = mujoco.MjModel.from_xml_string(xml)
    data  = mujoco.MjData(model)

    # Build joint-name → qpos-address mapping (hinge joints only)
    name_to_qposadr: Dict[str, int] = {}
    for i in range(model.njnt):
        jtype = model.jnt_type[i]
        if jtype == mujoco.mjtJoint.mjJNT_HINGE or jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jname:
                name_to_qposadr[jname] = int(model.jnt_qposadr[i])

    # mocap_joints order = column order of HDF5 walkers/walker_0/joints
    mocap_names = [j.name for j in walker.mocap_joints]

    _MODEL_CACHE = (model, data, name_to_qposadr, mocap_names)
    return model, data, name_to_qposadr, mocap_names


def _body_id(model, name_fragment: str) -> int:
    """Return first body id containing name_fragment (case-insensitive), or 0."""
    import mujoco
    for i in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if bname and name_fragment.lower() in bname.lower():
            return i
    return 0


# ── Trajectory loading ────────────────────────────────────────────────────────

def _load_trajectory_raw(
    clip_id: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float]]:
    """Load raw trajectory data for a clip from HDF5.

    Returns
    -------
    (joints, pos, quat_wxyz, mocap_names, fps)
        joints      : (T, 56) float32 — hinge angles in mocap_joints order (radians)
        pos         : (T, 3)  float32 — root CoM position (x, y, z)
        quat_wxyz   : (T, 4)  float32 — root quaternion in MuJoCo (w, x, y, z)
        mocap_names : list[str] length 56
        fps         : float
    Returns None on failure.
    """
    h5_path = get_h5_path()
    if h5_path is None:
        return None

    from dm_control.locomotion.walkers import cmu_humanoid
    walker = cmu_humanoid.CMUHumanoid()
    mocap_names = [j.name for j in walker.mocap_joints]

    try:
        import h5py
        with h5py.File(str(h5_path), "r") as f:
            if clip_id not in f:
                warnings.warn(f"[sim] Clip {clip_id!r} not in HDF5.")
                return None
            grp = f[clip_id]

            # ── Hinge joints (stored as (n_joints, T), mocap_joints order) ──
            joints = None
            for key in ("walkers/walker_0/joints", "walkers/0/joints", "joints"):
                if key not in grp:
                    continue
                arr = np.asarray(grp[key], dtype=np.float32)
                # (n_joints, T) → (T, n_joints)
                if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                if arr.ndim == 2:
                    joints = arr
                    break
            if joints is None:
                return None

            T = len(joints)

            # ── Root position (3, T) → (T, 3) ───────────────────────────────
            pos = np.zeros((T, 3), dtype=np.float32)
            pos[:, 2] = 0.9  # default stand height
            for key in ("walkers/walker_0/position", "walkers/0/position", "position"):
                if key not in grp:
                    continue
                arr = np.asarray(grp[key], dtype=np.float32)
                if arr.shape == (3, T):
                    arr = arr.T
                if arr.shape == (T, 3):
                    pos = arr
                    # Ensure z is at least 0.5 m (walker stands above floor)
                    if float(pos[:, 2].mean()) < 0.1:
                        pos[:, 2] += 0.9
                break

            # ── Root quaternion (4, T) → (T, 4), HDF5=(x,y,z,w)→MuJoCo=(w,x,y,z) ──
            quat_wxyz = np.zeros((T, 4), dtype=np.float32)
            quat_wxyz[:, 0] = 1.0  # default identity
            for key in ("walkers/walker_0/quaternion", "walkers/0/quaternion", "quaternion"):
                if key not in grp:
                    continue
                arr = np.asarray(grp[key], dtype=np.float32)
                if arr.shape == (4, T):
                    arr = arr.T
                if arr.shape == (T, 4):
                    # HDF5 convention is (x, y, z, w); MuJoCo expects (w, x, y, z)
                    quat_wxyz = np.concatenate([arr[:, 3:4], arr[:, :3]], axis=1)
                break

            dt  = float(grp.attrs.get("dt", 1.0 / _NATIVE_FPS))
            fps = 1.0 / dt if dt > 0 else _NATIVE_FPS

            return joints, pos, quat_wxyz, mocap_names, fps

    except Exception as exc:
        warnings.warn(f"[sim] HDF5 load failed for {clip_id!r}: {exc}")
        return None


# ── Simulation kernel ─────────────────────────────────────────────────────────

def _run_frames(
    model,
    data,
    name_to_qposadr: Dict[str, int],
    joints: np.ndarray,       # (T_clip, 56) mocap_joints order, radians
    pos: np.ndarray,          # (T_clip, 3)
    quat_wxyz: np.ndarray,    # (T_clip, 4)
    mocap_names: List[str],
    start_frame: int,
    knee_pred_rad: np.ndarray,  # (n_frames,) at clip_fps
    hip_pred_rad: Optional[np.ndarray],  # (n_frames,) at clip_fps or None
    ref_knee_col: int,          # column index in joints for rtibiarx (for RMSE)
    fps: float,
    viewer=None,
) -> dict:
    import mujoco

    n_frames = len(knee_pred_rad)
    if hip_pred_rad is not None:
        n_frames = min(n_frames, len(hip_pred_rad))

    T_clip = len(joints)
    dt     = 1.0 / fps

    com_heights:   List[float] = []
    rfoot_z:       List[float] = []
    lfoot_z:       List[float] = []
    knee_pred_inc: List[float] = []
    knee_ref_inc:  List[float] = []

    # Body IDs for metric collection
    torso_id = _body_id(model, "thorax")
    rfoot_id = _body_id(model, "rfoot")
    lfoot_id = _body_id(model, "lfoot")

    # knee qpos index for override + RMSE ref
    knee_qposadr = name_to_qposadr.get(_KNEE_JOINT, -1)
    hip_qposadr  = name_to_qposadr.get(_HIP_JOINT,  -1)

    t_start = time.perf_counter()

    for frame in range(n_frames):
        if viewer is not None and not viewer.is_running():
            break

        clip_idx = min(start_frame + frame, T_clip - 1)

        # ── Set root ─────────────────────────────────────────────────────────
        data.qpos[0:3] = pos[clip_idx]
        data.qpos[3:7] = quat_wxyz[clip_idx]

        # ── Set all hinge joints by name (correct ordering) ──────────────────
        row = joints[clip_idx]
        for mi, mname in enumerate(mocap_names):
            if mi < len(row) and mname in name_to_qposadr:
                data.qpos[name_to_qposadr[mname]] = float(row[mi])

        # ── Override right knee + right thigh from query window ───────────────
        if knee_qposadr >= 0:
            data.qpos[knee_qposadr] = knee_pred_rad[frame]
        if hip_pred_rad is not None and hip_qposadr >= 0:
            data.qpos[hip_qposadr] = hip_pred_rad[frame]

        # ── Forward kinematics ────────────────────────────────────────────────
        mujoco.mj_forward(model, data)

        # ── Collect metrics ───────────────────────────────────────────────────
        com_heights.append(float(data.xpos[torso_id, 2]))
        rfoot_z.append(float(data.xpos[rfoot_id, 2]))
        lfoot_z.append(float(data.xpos[lfoot_id, 2]))

        pred_inc = float(knee_rad_to_included(np.array([knee_pred_rad[frame]]))[0])
        knee_pred_inc.append(pred_inc)
        if ref_knee_col >= 0 and ref_knee_col < joints.shape[1]:
            ref_k_rad = float(joints[clip_idx, ref_knee_col])
            knee_ref_inc.append(float(knee_rad_to_included(np.array([ref_k_rad]))[0]))

        # ── Render ────────────────────────────────────────────────────────────
        if viewer is not None and viewer.is_running():
            viewer.sync()

        # ── Real-time pacing ──────────────────────────────────────────────────
        target = (frame + 1) * dt
        gap    = target - (time.perf_counter() - t_start)
        if gap > 0.001:
            time.sleep(gap)

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
    com = np.asarray(com_heights, dtype=np.float32)

    # Fall detection
    fall_mask   = com < FALL_HEIGHT_M
    fall_frame  = int(np.argmax(fall_mask)) if fall_mask.any() else -1
    fall_detect = bool(fall_mask.any())

    # Step count from foot lift-offs
    CONTACT_THRESH = 0.06  # m
    def _steps(z_list):
        if not z_list:
            return 0, []
        z   = np.asarray(z_list)
        on  = z < CONTACT_THRESH
        liftoffs = list(np.where(np.diff(on.astype(int)) == -1)[0])
        return len(liftoffs), liftoffs

    r_steps, r_frames = _steps(rfoot_z)
    l_steps, l_frames = _steps(lfoot_z)

    gait_sym = 1.0
    if len(r_frames) > 1 and len(l_frames) > 1:
        ri = float(np.diff(r_frames).mean())
        li = float(np.diff(l_frames).mean())
        gait_sym = float(min(ri, li) / max(ri, li, 1e-9))

    non_fall_frac   = float(1.0 - fall_mask.mean()) if len(com) else 0.0
    stability_score = non_fall_frac * gait_sym

    knee_pred = np.asarray(knee_pred_inc)
    if knee_ref_inc:
        kr = np.asarray(knee_ref_inc[:len(knee_pred)])
        n  = min(len(knee_pred), len(kr))
        knee_rmse = float(np.sqrt(np.mean((knee_pred[:n] - kr[:n]) ** 2)))
        knee_mae  = float(np.mean(np.abs(knee_pred[:n] - kr[:n])))
    else:
        knee_rmse = knee_mae = float("nan")

    return {
        "fall_detected":     fall_detect,
        "fall_frame":        fall_frame,
        "step_count":        r_steps + l_steps,
        "right_steps":       r_steps,
        "left_steps":        l_steps,
        "right_step_frames": r_frames,
        "left_step_frames":  l_frames,
        "gait_symmetry":     gait_sym,
        "stability_score":   stability_score,
        "com_height_mean":   float(com.mean()) if len(com) else 0.0,
        "com_height_std":    float(com.std())  if len(com) else 0.0,
        "com_height_min":    float(com.min())  if len(com) else 0.0,
        "com_heights":       com.tolist(),
        "knee_rmse_deg":     knee_rmse,
        "knee_mae_deg":      knee_mae,
        "knee_pred_deg":     knee_pred.tolist(),
        "n_frames":          len(com),
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
    """Kinematic replay of matched clip with right knee + thigh overridden.

    Parameters
    ----------
    clip_id                 : MoCap Act clip ID (e.g. ``"CMU_012_03"``)
    clip_start_frame        : Start frame within the clip (from matching).
    n_frames                : Frames to simulate (= query length at 200 Hz).
    knee_pred_included_deg  : (≥n_frames,) at 200 Hz — predicted knee angle.
    thigh_pred_included_deg : (≥n_frames,) at 200 Hz — IMU thigh angle.
    use_viewer              : Open interactive MuJoCo window.
    label                   : Scenario name for console output.
    """
    if label:
        print(f"\n  [{label}]  Building physics model …", flush=True)

    # ── Build raw mujoco model ────────────────────────────────────────────────
    try:
        model, data, name_to_qposadr, mocap_names = _make_mujoco_model()
    except Exception as exc:
        warnings.warn(f"[sim] Cannot build mujoco model: {exc}")
        return _fallback_metrics(n_frames)

    if label:
        print(f"  [{label}]  Loading HDF5 trajectory for {clip_id} …", flush=True)

    # ── Load trajectory ───────────────────────────────────────────────────────
    raw = _load_trajectory_raw(clip_id)
    if raw is None:
        warnings.warn(f"[sim] Cannot load trajectory for {clip_id!r}.")
        return _fallback_metrics(n_frames)

    joints, pos, quat_wxyz, _, clip_fps = raw

    # ── Resample 200-Hz query signals to clip_fps ─────────────────────────────
    def _to_clip_fps(sig: np.ndarray) -> np.ndarray:
        s = np.asarray(sig, dtype=np.float32)
        if abs(clip_fps - TARGET_FPS) > 0.5:
            s = _resample(s, TARGET_FPS, clip_fps)
        s = s[:n_frames]
        if len(s) < n_frames:
            s = np.concatenate([s, np.full(n_frames - len(s), s[-1])])
        return s

    knee_cf = _to_clip_fps(np.asarray(knee_pred_included_deg))
    knee_pred_rad = included_to_knee_rad(knee_cf).astype(np.float64)

    hip_pred_rad: Optional[np.ndarray] = None
    if thigh_pred_included_deg is not None:
        thigh_cf = _to_clip_fps(np.asarray(thigh_pred_included_deg))
        hip_pred_rad = included_to_hip_rad(thigh_cf).astype(np.float64)

    # Find the column in joints for the knee (for RMSE vs reference)
    ref_knee_col = -1
    try:
        ref_knee_col = mocap_names.index(_KNEE_JOINT)
    except ValueError:
        pass

    if label:
        override = "knee+thigh" if hip_pred_rad is not None else "knee only"
        print(f"  [{label}]  Simulating {n_frames} frames @ {clip_fps:.0f} Hz "
              f"({override} overridden from query window) …", flush=True)

    def _call(viewer=None):
        return _run_frames(
            model, data, name_to_qposadr,
            joints, pos, quat_wxyz, mocap_names,
            clip_start_frame,
            knee_pred_rad, hip_pred_rad, ref_knee_col,
            clip_fps, viewer,
        )

    # ── Run with viewer ───────────────────────────────────────────────────────
    if use_viewer:
        try:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(
                model, data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                viewer.cam.distance  = 4.5
                viewer.cam.azimuth   = 150.0
                viewer.cam.elevation = -18.0
                metrics = _call(viewer)
                if viewer.is_running():
                    print(f"  [{label}]  Done — close viewer window to continue.",
                          flush=True)
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
        except Exception as exc:
            print(f"  [{label}]  Viewer failed ({exc}); running headless.",
                  flush=True)
            metrics = _call()
    else:
        metrics = _call()

    return metrics


def _fallback_metrics(n_frames: int) -> dict:
    return _compute_metrics(
        [1.0] * n_frames, [], [], [], [], _NATIVE_FPS
    )


# ── Summary plot ──────────────────────────────────────────────────────────────

def plot_sim_results(
    scenarios: Dict[str, dict],
    scenario_labels: Dict[str, str],
    knee_signals: Dict[str, np.ndarray],
    fps: float,
    out_path: str = "sim_results.png",
) -> None:
    """Save + display a 3-panel simulation summary."""
    import matplotlib
    import matplotlib.pyplot as plt

    colors = {"gt": "steelblue", "good": "darkorange", "bad": "firebrick"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    fig.suptitle("Simulation results — kinematic replay, knee+thigh override",
                 fontsize=11)

    ax = axes[0]
    for key, sig in knee_signals.items():
        t = np.arange(len(sig)) / fps
        ax.plot(t, sig, lw=1.8, color=colors.get(key, "gray"),
                label=scenario_labels.get(key, key))
    ax.set_ylabel("Knee included-angle (°)")
    ax.set_title("Knee angle — scenarios")
    ax.set_ylim(0, 200)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for key, res in scenarios.items():
        com = np.asarray(res["com_heights"])
        t   = np.arange(len(com)) / fps
        ax.plot(t, com, lw=1.8, color=colors.get(key, "gray"),
                label=f"{scenario_labels.get(key, key)}  stab={res['stability_score']:.2f}")
    ax.axhline(FALL_HEIGHT_M, color="red", ls="--", lw=1,
               label=f"Fall threshold ({FALL_HEIGHT_M} m)")
    ax.set_ylabel("Torso height (m)")
    ax.set_title("Centre of Mass height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    ax.set_title("Heuristic metrics", pad=10)

    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Sim results plot saved → {out_path}")

    # Try to display interactively
    try:
        plt.show(block=False)
        plt.pause(0.5)
    except Exception:
        pass
    plt.close(fig)
