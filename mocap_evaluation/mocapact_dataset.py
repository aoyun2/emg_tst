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

# Joint names in the dm_control CMU humanoid MuJoCo model (V2020).
# rfemurry = right femur Y-axis rotation (sagittal hip flex/ext).
# Note: the name has a double-r — "rfemur" body + "ry" axis suffix.
_KNEE_JOINT = "rtibiarx"   # right tibia around x-axis (knee flexion)
_HIP_JOINT  = "rfemurry"   # right femur around y-axis (hip flex/ext)

_CACHE_FILE = ".cache_mocapact.npz"

# Fallback joint qpos addresses if physics model query fails.
# These are the known values for the dm_control CMU humanoid (V2020).
# qpos layout: 7 free-joint DOFs (pos+quat) then hinge joints in model order.
_KNEE_QPOS_FALLBACK = 47   # rtibiarx qpos index
_HIP_QPOS_FALLBACK  = 41   # rfemurry qpos index (was rfemury — corrected)

# Number of root DOFs in the CMU humanoid free joint (3 position + 4 quaternion).
# HDF5 `walker/joints_pos` stores only hinge DOFs (no root), so the hinge-only
# index = qpos_addr - _CMU_ROOT_DOFS.
_CMU_ROOT_DOFS = 7

# Fallback HDF5 joint indices for the walkers/walker_0/joints array.
# This array uses walker.mocap_joints ordering (biomechanical leg/spine/arm grouping):
#   0-6:  left leg   (lfemurrz, lfemurry, lfemurrx, ltibiarx, lfootrz, lfootrx, ltoesrx)
#   7-13: right leg  (rfemurrz, rfemurry, rfemurrx, rtibiarx, rfootrz, rfootrx, rtoesrx)
#   14-55: spine/head/arms
#   rfemurry = mocap_joints index  8  (right leg, Y-axis hip rotation)
#   rtibiarx = mocap_joints index 10  (right leg, X-axis tibia/knee rotation)
_KNEE_HDF5_FALLBACK = 10   # rtibiarx position in mocap_joints
_HIP_HDF5_FALLBACK  =  8   # rfemurry position in mocap_joints


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
    """MuJoCo rfemurry (rad, 0=neutral, +flex/-ext) → included-angle degrees."""
    deg = np.degrees(np.asarray(rad, dtype=np.float64))
    return (180.0 - deg).astype(np.float32)


# ── Clip ID helpers ───────────────────────────────────────────────────────────

def _clip_id_to_subject_trial(clip_id: str) -> Tuple[int, int]:
    """``"CMU_009_12"`` → (9, 12)."""
    parts = clip_id.split("_")
    return int(parts[1]), int(parts[2])


_SUBSET_MAP = {
    "all":              "ALL",
    "locomotion_small": "LOCOMOTION_SMALL",
    "locomotion":       "LOCOMOTION_SMALL",
    "walk_tiny":        "WALK_TINY",
    "run_jump_tiny":    "RUN_JUMP_TINY",
}


def get_locomotion_clip_ids(subset: str = "all") -> List[str]:
    """Return clip IDs for the requested dm_control CMU subset.

    Parameters
    ----------
    subset : str
        One of ``"all"`` (1,144 clips, ~3.5 hrs — default),
        ``"locomotion_small"`` (243 clips, ~40 min walking/running),
        ``"walk_tiny"`` (35 clips), or ``"run_jump_tiny"`` (50 clips).

    Returns
    -------
    List[str]
        Clip IDs like ``["CMU_001_01", ...]``.
    """
    from dm_control.locomotion.tasks.reference_pose import cmu_subsets

    attr = _SUBSET_MAP.get(subset.lower(), subset.upper())
    if not hasattr(cmu_subsets, attr):
        raise ValueError(
            f"Unknown subset {subset!r}.  "
            f"Available: {list(_SUBSET_MAP)}"
        )
    clips = getattr(cmu_subsets, attr)

    # Different dm_control versions expose the collection differently
    if hasattr(clips, "ids"):
        return list(clips.ids)
    if hasattr(clips, "_clips"):
        return [c.id if hasattr(c, "id") else str(c) for c in clips._clips]
    try:
        return [c.id if hasattr(c, "id") else str(c) for c in clips]
    except TypeError:
        pass
    raise RuntimeError(
        f"Cannot enumerate clips from dm_control {attr}.  "
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

def _get_cmu_mocap_path() -> Optional[Path]:
    """Return path to the consolidated CMU mocap H5 file, downloading if needed.

    Search order:
    1. ``MOCAPACT_H5_PATH`` environment variable (explicit override).
    2. dm_control's built-in search: package dir, then ``~/.dm_control/``.
       On first access the ~454 MB CMU dataset is auto-downloaded to
       ``~/.dm_control/cmu_2020_*.h5``.

    Set ``MOCAPACT_H5_PATH`` to point at the file if it lives somewhere
    other than the two default locations (common on Windows).
    """
    import os

    # 1. Explicit override via environment variable
    env_path = os.environ.get("MOCAPACT_H5_PATH", "").strip()
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        warnings.warn(
            f"[mocapact_dataset] MOCAPACT_H5_PATH={env_path!r} does not exist; "
            "falling back to dm_control default search."
        )

    # 2. dm_control default search (package dir → ~/.dm_control/ → download)
    try:
        from dm_control.locomotion.mocap import cmu_mocap_data
        p = Path(cmu_mocap_data.get_path_for_cmu())
        return p if p.exists() else None
    except Exception:
        return None


def _load_via_hdf5_loader(clip_id: str, knee_addr: int, hip_addr: int) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Try loading via dm_control's HDF5TrajectoryLoader.

    Returns (knee_rad, hip_rad, fps) or None.
    """
    try:
        from dm_control.locomotion.mocap import loader as mocap_loader

        mocap_path = _get_cmu_mocap_path()
        if mocap_path is None:
            return None

        if not hasattr(mocap_loader, "HDF5TrajectoryLoader"):
            return None

        tl = mocap_loader.HDF5TrajectoryLoader(str(mocap_path))
        # API: get_trajectory() in dm_control ≥1.0; older versions used load()
        traj = tl.get_trajectory(clip_id) if hasattr(tl, "get_trajectory") else tl.load(clip_id)

        # as_dict() is the canonical API in newer dm_control versions.
        # "walker/joints_pos" / "walker/joints" are hinge-only (no root DOFs);
        # "qpos" is the full generalized position including root DOFs.
        try:
            d = traj.as_dict()
            # (key, is_full_qpos)
            for key, is_full_qpos in (
                ("walker/joints_pos", False),
                ("walker/joints",     False),
                ("joints",            False),
                ("qpos",              True),
            ):
                if key not in d:
                    continue
                data = np.asarray(d[key], dtype=np.float32)
                if data.ndim != 2:
                    continue
                k_idx = knee_addr if is_full_qpos else knee_addr - _CMU_ROOT_DOFS
                h_idx = hip_addr  if is_full_qpos else hip_addr  - _CMU_ROOT_DOFS
                if k_idx < 0 or h_idx < 0 or data.shape[1] <= max(k_idx, h_idx):
                    continue
                fps = float(getattr(traj, "fps", _NATIVE_FPS))
                return data[:, k_idx], data[:, h_idx], fps
        except Exception:
            pass

        # Fallback: direct attribute access (older dm_control versions).
        # "qpos" contains root DOFs; others are hinge-only.
        for attr, is_full_qpos in (
            ("qpos",       True),
            ("joints_pos", False),
            ("joint_pos",  False),
            ("joints",     False),
        ):
            data = getattr(traj, attr, None)
            if data is not None:
                data = np.asarray(data, dtype=np.float32)
                k_idx = knee_addr if is_full_qpos else knee_addr - _CMU_ROOT_DOFS
                h_idx = hip_addr  if is_full_qpos else hip_addr  - _CMU_ROOT_DOFS
                if data.ndim == 2 and k_idx >= 0 and h_idx >= 0 and data.shape[1] > max(k_idx, h_idx):
                    fps = float(getattr(traj, "fps", _NATIVE_FPS))
                    return data[:, k_idx], data[:, h_idx], fps
    except Exception:
        pass
    return None


def _get_hdf5_joint_indices() -> Tuple[int, int]:
    """Return (knee_hdf5_idx, hip_hdf5_idx) in mocap_joints ordering.

    The HDF5 ``walkers/walker_0/joints`` array uses the walker's
    ``mocap_joints`` ordering (biomechanical leg/spine/arm grouping), NOT the
    alphabetical ``observable_joints`` order and NOT the MuJoCo qpos order.
    We look up each target joint's position in ``mocap_joints`` so that HDF5
    indexing is always correct regardless of qpos layout.
    """
    try:
        from dm_control.locomotion.walkers import cmu_humanoid
        walker = cmu_humanoid.CMUHumanoid()
        names = [j.name for j in walker.mocap_joints]
        return names.index(_KNEE_JOINT), names.index(_HIP_JOINT)
    except Exception:
        return _KNEE_HDF5_FALLBACK, _HIP_HDF5_FALLBACK


def _load_via_hdf5_direct(clip_id: str, knee_addr: int, hip_addr: int) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Read joint data directly from the consolidated CMU mocap H5 file with h5py.

    The actual dm_control HDF5 structure is:
        clip_id / walkers / walker_0 / joints   (shape [56, T] — transposed!)

    The ``walkers/walker_0/joints`` array is indexed by the walker's
    ``observable_joints`` list (alphabetical order), NOT by qpos address.
    Use ``_get_hdf5_joint_indices()`` to obtain the correct column indices.

    Returns (knee_rad, hip_rad, fps) or None.
    """
    try:
        import h5py

        mocap_path = _get_cmu_mocap_path()
        if mocap_path is None:
            return None

        # Indices for the walkers/walker_0/joints array (alphabetical observable_joints order)
        obs_knee_idx, obs_hip_idx = _get_hdf5_joint_indices()
        # Indices for any hypothetical walker/joints_pos key (qpos-minus-root order)
        h_knee = knee_addr - _CMU_ROOT_DOFS
        h_hip  = hip_addr  - _CMU_ROOT_DOFS

        # (key, knee_idx, hip_idx, needs_transpose)
        # needs_transpose: data stored (joints, time) → must be transposed to (time, joints).
        joint_key_candidates = [
            # Actual dm_control HDF5: shape (56, T), alphabetical observable_joints order
            ("walkers/walker_0/joints", obs_knee_idx, obs_hip_idx, True),
            ("walkers/0/joints",        obs_knee_idx, obs_hip_idx, True),
            # Hypothetical consolidated keys: shape (T, 56), qpos-minus-root order
            ("walker/joints_pos",       h_knee,       h_hip,       False),
            ("walker/joints",           h_knee,       h_hip,       False),
            ("joints",                  h_knee,       h_hip,       False),
            # Full qpos (T, n_qpos): uses raw qpos addresses
            ("qpos",                    knee_addr,    hip_addr,    False),
            ("joints_pos",              h_knee,       h_hip,       False),
        ]

        with h5py.File(str(mocap_path), "r") as f:
            if clip_id not in f:
                return None
            clip_grp = f[clip_id]

            for key, k_idx, h_idx, needs_transpose in joint_key_candidates:
                if key not in clip_grp:
                    continue
                data = np.asarray(clip_grp[key], dtype=np.float32)
                if needs_transpose and data.ndim == 2:
                    data = data.T  # (joints, time) → (time, joints)
                if data.ndim != 2:
                    continue
                if k_idx < 0 or h_idx < 0 or data.shape[1] <= max(k_idx, h_idx):
                    continue
                dt = clip_grp.attrs.get("dt", None) or f.attrs.get("dt", None)
                fps_attr = clip_grp.attrs.get("fps", None) or f.attrs.get("fps", None)
                if dt is not None and float(dt) > 0:
                    fps = 1.0 / float(dt)
                elif fps_attr is not None:
                    fps = float(fps_attr)
                else:
                    fps = _NATIVE_FPS
                return data[:, k_idx], data[:, h_idx], fps
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
                h = float(physics.named.data.qpos[_HIP_JOINT])  # rfemurry
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
    cache_path: Optional[str] = None,
    use_cache: bool = True,
    target_fps: float = TARGET_FPS,
    subset: str = "all",
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
        Path for the ``.npz`` cache file.  Defaults to
        ``.cache_mocapact_{subset}.npz``.
    use_cache :
        If True and cache exists, load from cache.
    target_fps :
        Resample all clips to this frame rate (default 200 Hz to match EMG pipeline).
    subset :
        Which dm_control CMU clip collection to use.  One of:
        ``"all"`` (1,144 clips, ~3.5 hrs — **default**),
        ``"locomotion_small"`` (243 clips, ~40 min walking/running),
        ``"walk_tiny"`` (35 clips), or ``"run_jump_tiny"`` (50 clips).
        Ignored when *clip_ids* is provided explicitly.
    clip_ids :
        Explicit list of clip IDs (overrides *subset*).
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
    if cache_path is None:
        cache_path = f".cache_mocapact_{subset}.npz"
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
        clip_ids = get_locomotion_clip_ids(subset=subset)
    if not quiet:
        print(f"[mocapact_dataset] Building database from {len(clip_ids)} clips "
              f"(subset={subset!r}) …")

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
        import os
        # Build a helpful message showing what paths were checked
        h5_hint = ""
        try:
            from dm_control.locomotion.mocap import cmu_mocap_data
            paths = [os.path.expanduser(p) for p in cmu_mocap_data.H5_PATHS["2020"]]
            h5_hint = (
                "\nSearched for HDF5 file at:\n"
                + "".join(f"  {p}\n" for p in paths)
                + "If the file is elsewhere, set: "
                "MOCAPACT_H5_PATH=/path/to/cmu_2020_dfe3e9e0.h5"
            )
        except Exception:
            pass
        raise RuntimeError(
            "No MocapAct clips could be loaded.  "
            "Ensure dm_control and h5py are installed and the HDF5 assets are accessible.\n"
            "  pip install dm_control h5py"
            + h5_hint
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
        "source":          np.array(f"mocapact_reference/{subset}"),
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
