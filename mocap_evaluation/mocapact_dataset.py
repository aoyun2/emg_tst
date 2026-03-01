"""Motion-matching database built from MocapAct (dm_control) reference trajectories.

Replaces the CMU BVH-based database for motion matching.  Uses the same
reference clip data that the MocapAct physics simulation uses internally,
giving physics-consistent joint angle trajectories without needing the CMU
BVH archive.

The returned database dict is drop-in compatible with
``motion_matching.find_best_match`` and ``find_top_k_matches``.

Public API
----------
load_mocapact_database(cache_path, use_cache, target_fps) -> dict
    Keys: knee_right, hip_right, file_boundaries, categories, root_pos, fps, source
"""
from __future__ import annotations

import warnings
from math import gcd
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ── constants ─────────────────────────────────────────────────────────────────

TARGET_FPS: float = 200.0   # match EMG/IMU pipeline rate
_NATIVE_FPS: float = 30.0   # dm_control default control frequency

# Joint names in the dm_control CMU humanoid MuJoCo model
_KNEE_JOINT = "rtibiarx"   # right tibia around x-axis (knee flexion)
_HIP_JOINT  = "rfemury"    # right femur around y-axis (hip flex/ext)

_CACHE_FILE = ".cache_mocapact.npz"

# Fallback joint qpos addresses if physics model query fails.
# These are the known values for the dm_control CMU humanoid (V2020).
# qpos layout: 7 free-joint DOFs (pos+quat) then hinge joints alphabetically.
_KNEE_QPOS_FALLBACK = 47   # rtibiarx qpos index
_HIP_QPOS_FALLBACK  = 40   # rfemury qpos index


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, from_fps: float, to_fps: float) -> np.ndarray:
    """Polyphase resample a 1-D array from from_fps to to_fps."""
    if abs(from_fps - to_fps) < 0.5:
        return arr.astype(np.float32)
    from scipy.signal import resample_poly
    up = int(round(to_fps))
    down = int(round(from_fps))
    g = gcd(up, down)
    return resample_poly(arr.astype(np.float64), up // g, down // g).astype(np.float32)


def _rad_knee_to_included_deg(rad: np.ndarray) -> np.ndarray:
    """MuJoCo rtibiarx (rad, 0=straight, +flex) → included-angle degrees."""
    deg = np.degrees(np.abs(np.asarray(rad, dtype=np.float64)))
    return np.clip(180.0 - deg, 0.0, 180.0).astype(np.float32)


def _rad_hip_to_included_deg(rad: np.ndarray) -> np.ndarray:
    """MuJoCo rfemury (rad, 0=neutral, +flex/-ext) → included-angle degrees."""
    deg = np.degrees(np.asarray(rad, dtype=np.float64))
    return (180.0 - deg).astype(np.float32)


# ── Clip ID helpers ───────────────────────────────────────────────────────────

def _clip_id_to_subject_trial(clip_id: str) -> Tuple[int, int]:
    """``"CMU_009_12"`` → (9, 12)."""
    parts = clip_id.split("_")
    return int(parts[1]), int(parts[2])


def get_locomotion_clip_ids() -> List[str]:
    """Return list of clip IDs from dm_control's LOCOMOTION_SMALL subset."""
    from dm_control.locomotion.tasks.reference_pose import cmu_subsets
    clips = cmu_subsets.LOCOMOTION_SMALL
    # Different dm_control versions expose the collection differently
    if hasattr(clips, "ids"):
        return list(clips.ids)
    if hasattr(clips, "_clips"):
        return [c.id if hasattr(c, "id") else str(c) for c in clips._clips]
    # Fallback: iterate
    try:
        return [c.id if hasattr(c, "id") else str(c) for c in clips]
    except TypeError:
        pass
    raise RuntimeError(
        "Cannot enumerate clips from dm_control LOCOMOTION_SMALL.  "
        "Check your dm_control version."
    )


# ── Joint-index discovery ─────────────────────────────────────────────────────

def _get_qpos_addresses(env) -> Tuple[int, int]:
    """Return (knee_qpos_addr, hip_qpos_addr) from the physics model."""
    try:
        from mocap_evaluation.mocapact_sim import _get_physics
        physics = _get_physics(env)
        knee_addr = hip_addr = None
        for jnt_i in range(physics.model.njnt):
            name = physics.model.id2name(jnt_i, "joint")
            addr = int(physics.model.jnt_qposadr[jnt_i])
            if name == _KNEE_JOINT:
                knee_addr = addr
            elif name == _HIP_JOINT:
                hip_addr = addr
            if knee_addr is not None and hip_addr is not None:
                break
        if knee_addr is None or hip_addr is None:
            raise ValueError("Joint not found in model")
        return knee_addr, hip_addr
    except Exception as exc:
        warnings.warn(
            f"[mocapact_dataset] Joint-address discovery failed ({exc}); "
            f"using fallback indices {_KNEE_QPOS_FALLBACK}, {_HIP_QPOS_FALLBACK}."
        )
        return _KNEE_QPOS_FALLBACK, _HIP_QPOS_FALLBACK


# ── Reference data loading ────────────────────────────────────────────────────

def _load_via_hdf5_loader(clip_id: str, knee_addr: int, hip_addr: int) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Try loading via dm_control's HDF5TrajectoryLoader.

    Returns (knee_rad, hip_rad, fps) or None.
    """
    try:
        import dm_control
        from dm_control.locomotion.mocap import loader as mocap_loader
        assets_dir = Path(dm_control.__file__).parent / "locomotion" / "mocap" / "assets"

        if not assets_dir.is_dir():
            return None

        if hasattr(mocap_loader, "HDF5TrajectoryLoader"):
            tl = mocap_loader.HDF5TrajectoryLoader(str(assets_dir))
            traj = tl.load(clip_id)
            # Different versions use different attribute names
            for attr in ("qpos", "joints_pos", "joint_pos"):
                data = getattr(traj, attr, None)
                if data is not None:
                    data = np.asarray(data, dtype=np.float32)
                    if data.ndim == 2 and data.shape[1] > max(knee_addr, hip_addr):
                        fps = float(getattr(traj, "fps", _NATIVE_FPS))
                        return data[:, knee_addr], data[:, hip_addr], fps
    except Exception:
        pass
    return None


def _load_via_hdf5_direct(clip_id: str, knee_addr: int, hip_addr: int) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Read dm_control HDF5 asset files directly with h5py.

    Returns (knee_rad, hip_rad, fps) or None.
    """
    try:
        import h5py
        import dm_control
        assets_dir = Path(dm_control.__file__).parent / "locomotion" / "mocap" / "assets"

        candidates = [
            assets_dir / f"{clip_id}.hdf5",
            assets_dir / f"{clip_id.lower()}.hdf5",
        ]
        for p in candidates:
            if not p.exists():
                continue
            with h5py.File(str(p), "r") as f:
                # Try known dataset names
                for key in ("qpos", "joints_pos", "joint_pos", "walker/joints_pos"):
                    if key not in f:
                        continue
                    data = np.asarray(f[key], dtype=np.float32)
                    if data.ndim != 2:
                        continue
                    if data.shape[1] <= max(knee_addr, hip_addr):
                        continue
                    # Read fps from attributes (try dt → fps, or fps directly)
                    dt = f.attrs.get("dt", None)
                    fps_attr = f.attrs.get("fps", None)
                    if dt is not None and float(dt) > 0:
                        fps = 1.0 / float(dt)
                    elif fps_attr is not None:
                        fps = float(fps_attr)
                    else:
                        fps = _NATIVE_FPS
                    return data[:, knee_addr], data[:, hip_addr], fps
    except Exception:
        pass
    return None


def _load_via_env_stepping(clip_id: str) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Extract reference trajectory by resetting and stepping the env.

    This is the most reliable fallback: the env is reset to the specific clip
    and stepped with zero actions.  The task internally tracks the reference
    state.  We collect the reference joint observations at each step.

    Returns (knee_included_deg, hip_included_deg, fps) directly (no radians).
    """
    try:
        from mocap_evaluation.mocapact_sim import create_walking_env, _get_physics
        import numpy as np

        env = create_walking_env(clip_id=clip_id)
        obs = env.reset()
        physics = _get_physics(env)
        task = env.unwrapped._env.task

        knee_vals: List[float] = []
        hip_vals: List[float] = []

        max_steps = int(20.0 * _NATIVE_FPS)  # up to 20 s
        done = False

        for _ in range(max_steps):
            if done:
                break
            # Try to access the reference state from the task's internals
            try:
                # dm_control stores ref walker state; step-0 is current frame
                ref = task._reference_observations_at_current_step(physics)
                qpos = ref.get("joints_pos", None)
                if qpos is not None and len(qpos) > max(_KNEE_QPOS_FALLBACK, _HIP_QPOS_FALLBACK):
                    k = float(qpos[_KNEE_QPOS_FALLBACK])
                    h = float(qpos[_HIP_QPOS_FALLBACK])
                    knee_vals.append(k)
                    hip_vals.append(h)
                    obs, _, done, _ = env.step(np.zeros(env.action_space.shape))
                    continue
            except Exception:
                pass

            # Fallback: read the physics qpos directly (after reset the agent
            # starts exactly at the reference pose)
            try:
                k = float(physics.named.data.qpos[_KNEE_JOINT])
                h = float(physics.named.data.qpos[_HIP_JOINT])
            except Exception:
                k = float(physics.data.qpos[_KNEE_QPOS_FALLBACK])
                h = float(physics.data.qpos[_HIP_QPOS_FALLBACK])

            knee_vals.append(k)
            hip_vals.append(h)
            obs, _, done, _ = env.step(np.zeros(env.action_space.shape))

        env.close()

        if not knee_vals:
            return None

        knee_inc = _rad_knee_to_included_deg(np.array(knee_vals, dtype=np.float32))
        hip_inc  = _rad_hip_to_included_deg(np.array(hip_vals,  dtype=np.float32))
        return knee_inc, hip_inc, _NATIVE_FPS

    except Exception:
        return None


def _extract_clip_angles(
    clip_id: str,
    knee_addr: int,
    hip_addr: int,
    target_fps: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load knee/hip angle arrays (included-angle degrees) for one clip.

    Tries HDF5TrajectoryLoader → direct HDF5 → env-stepping.
    Returns (knee_deg, hip_deg) resampled to target_fps, or None.
    """
    # Method 1: dm_control HDF5TrajectoryLoader
    result = _load_via_hdf5_loader(clip_id, knee_addr, hip_addr)
    if result is not None:
        knee_rad, hip_rad, fps = result
        knee = _resample(_rad_knee_to_included_deg(knee_rad), fps, target_fps)
        hip  = _resample(_rad_hip_to_included_deg(hip_rad),   fps, target_fps)
        return knee, hip

    # Method 2: Direct HDF5 read
    result = _load_via_hdf5_direct(clip_id, knee_addr, hip_addr)
    if result is not None:
        knee_rad, hip_rad, fps = result
        knee = _resample(_rad_knee_to_included_deg(knee_rad), fps, target_fps)
        hip  = _resample(_rad_hip_to_included_deg(hip_rad),   fps, target_fps)
        return knee, hip

    # Method 3: Env-stepping (already returns included-angle degrees)
    result = _load_via_env_stepping(clip_id)
    if result is not None:
        knee_inc, hip_inc, fps = result
        knee = _resample(knee_inc, fps, target_fps)
        hip  = _resample(hip_inc,  fps, target_fps)
        return knee, hip

    return None


# ── Joint-index bootstrap ─────────────────────────────────────────────────────

_CACHED_JOINT_ADDRS: Optional[Tuple[int, int]] = None


def _get_joint_addresses_once() -> Tuple[int, int]:
    """Return (knee_qpos_addr, hip_qpos_addr), creating a temp env once."""
    global _CACHED_JOINT_ADDRS
    if _CACHED_JOINT_ADDRS is not None:
        return _CACHED_JOINT_ADDRS

    try:
        from mocap_evaluation.mocapact_sim import create_walking_env
        # Use a specific clip so the env is deterministic
        env = create_walking_env()
        env.reset()
        addrs = _get_qpos_addresses(env)
        env.close()
    except Exception:
        addrs = (_KNEE_QPOS_FALLBACK, _HIP_QPOS_FALLBACK)

    _CACHED_JOINT_ADDRS = addrs
    return addrs


# ── Main public API ───────────────────────────────────────────────────────────

def load_mocapact_database(
    cache_path: str = _CACHE_FILE,
    use_cache: bool = True,
    target_fps: float = TARGET_FPS,
    clip_ids: Optional[List[str]] = None,
    quiet: bool = False,
) -> dict:
    """Build (or load from cache) a motion-matching database from MocapAct clips.

    The database is compatible with ``motion_matching.find_best_match`` and
    ``find_top_k_matches``.  All angles are in **included-angle degrees**
    (180 = straight / neutral).

    Parameters
    ----------
    cache_path :
        Path for the ``.npz`` cache file (speeds up subsequent calls).
    use_cache :
        If True and cache exists, load from cache.
    target_fps :
        Resample all clips to this frame rate (default 200 Hz to match EMG pipeline).
    clip_ids :
        Explicit list of MocapAct clip IDs (e.g. ``["CMU_009_12"]``).
        If None, uses ``dm_control``'s ``LOCOMOTION_SMALL`` subset.
    quiet :
        Suppress progress bars.

    Returns
    -------
    dict
        ``knee_right``     : (N,) float32 – included-angle degrees
        ``hip_right``      : (N,) float32 – included-angle degrees
        ``file_boundaries``: list of (start, end, clip_id, category) tuples
        ``categories``     : (N,) object array of category strings
        ``root_pos``       : (N, 3) float32 zeros (placeholder)
        ``fps``            : float
        ``source``         : str
    """
    cache = Path(cache_path)

    # ── Try loading from cache ─────────────────────────────────────────────────
    if use_cache and cache.exists():
        try:
            raw = np.load(str(cache), allow_pickle=True)
            db = {k: raw[k] for k in raw.files}
            # file_boundaries is stored as object array; restore as list of tuples
            if "file_boundaries" in db:
                db["file_boundaries"] = [tuple(b) for b in db["file_boundaries"].tolist()]
            if not quiet:
                n = len(db.get("knee_right", []))
                print(f"[mocapact_dataset] Loaded {n} frames from cache {cache}")
            return db
        except Exception as exc:
            warnings.warn(f"[mocapact_dataset] Cache load failed ({exc}); rebuilding.")

    # ── Enumerate clips ────────────────────────────────────────────────────────
    if clip_ids is None:
        clip_ids = get_locomotion_clip_ids()
    if not quiet:
        print(f"[mocapact_dataset] Building database from {len(clip_ids)} clips …")

    # ── Discover joint indices from physics model ──────────────────────────────
    knee_addr, hip_addr = _get_joint_addresses_once()

    # ── Iterate over clips ─────────────────────────────────────────────────────
    all_knee: List[np.ndarray] = []
    all_hip:  List[np.ndarray] = []
    boundaries: List[tuple] = []
    skipped = 0
    cursor = 0

    clip_iter = clip_ids if quiet else tqdm(clip_ids, desc="MocapAct clips", unit="clip")
    for clip_id in clip_iter:
        result = _extract_clip_angles(clip_id, knee_addr, hip_addr, target_fps)
        if result is None:
            skipped += 1
            continue

        knee, hip = result
        n = len(knee)
        if n < 2:
            skipped += 1
            continue

        all_knee.append(knee)
        all_hip.append(hip)
        # Store clip_id directly as fname; resolve_clip_from_match handles it.
        boundaries.append((cursor, cursor + n, clip_id, "locomotion"))
        cursor += n

    if not all_knee:
        raise RuntimeError(
            "No MocapAct clips could be loaded.  "
            "Ensure dm_control is installed and its HDF5 assets are accessible.\n"
            "  pip install dm_control mocapact stable-baselines3"
        )

    if not quiet:
        print(
            f"[mocapact_dataset] Loaded {len(all_knee)} clips "
            f"({skipped} skipped), {cursor} frames total at {target_fps:.0f} Hz."
        )

    knee_arr = np.concatenate(all_knee)
    hip_arr  = np.concatenate(all_hip)
    cats     = np.array(["locomotion"] * cursor, dtype=object)
    root_pos = np.zeros((cursor, 3), dtype=np.float32)

    db = {
        "knee_right":      knee_arr,
        "hip_right":       hip_arr,
        "file_boundaries": boundaries,
        "categories":      cats,
        "root_pos":        root_pos,
        "fps":             np.float32(target_fps),
        "source":          np.array("mocapact_reference"),
    }

    # ── Save cache ─────────────────────────────────────────────────────────────
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # boundaries is a list of tuples — save as object array
        save_dict = dict(db)
        save_dict["file_boundaries"] = np.array(boundaries, dtype=object)
        np.savez(str(cache), **save_dict)
        if not quiet:
            print(f"[mocapact_dataset] Cached to {cache}")
    except Exception as exc:
        warnings.warn(f"[mocapact_dataset] Cache save failed: {exc}")

    return db
