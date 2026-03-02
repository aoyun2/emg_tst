"""MoCap Act motion-matching database.

Loads knee and hip angles (included-angle degrees, 180=straight) from either:

* **dm_control CMU HDF5** (subset ``"locomotion_small"``, ``"walk_tiny"``,
  ``"run_jump_tiny"``, ``"all"``): single ``cmu_2020_*.h5`` file,
  ~837 clips, auto-downloaded by dm_control.
* **Microsoft MoCapAct HDF5** (subset ``"mocapact"``): per-clip
  ``CMU_*.hdf5`` files, ~2 589 snippets.  Requires the dataset to be
  downloaded separately (see ``mocapact_ms.download_instructions()``).

Cache: ``.cache_mocapact_{subset}.npz``  (auto-built on first run).

Angle conventions
-----------------
  included_angle = 180 - |flexion_deg|
  Full extension  → 180°
  Deep flexion    →  ~60°
"""
from __future__ import annotations

import warnings
from math import gcd
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ── Joint constants (CMU humanoid V2020) ─────────────────────────────────────

TARGET_FPS   = 200.0   # resample all clips to match EMG pipeline
_NATIVE_FPS  = 30.0    # dm_control default control frequency

# MuJoCo joint names
_KNEE_JOINT = "rtibiarx"   # right tibia x-axis (knee flexion)
_HIP_JOINT  = "rfemurry"   # right femur y-axis (hip flex/ext)

# Full qpos indices (root has 7 DOFs: pos(3) + quat(4))
KNEE_QPOS_IDX = 47
HIP_QPOS_IDX  = 41
_ROOT_DOFS    = 7

# mocap_joints ordering for HDF5 walkers/walker_0/joints:
#   0-6:  left leg   (lfemurrz, lfemurry, lfemurrx, ltibiarx, lfootrz, lfootrx, ltoesrx)
#   7-13: right leg  (rfemurrz, rfemurry, rfemurrx, rtibiarx, rfootrz, rfootrx, rtoesrx)
_KNEE_MOCAP_IDX = 10   # rtibiarx in mocap_joints
_HIP_MOCAP_IDX  =  8   # rfemurry in mocap_joints

_SUBSET_MAP = {
    "all":              "ALL",
    "locomotion_small": "LOCOMOTION_SMALL",
    "locomotion":       "LOCOMOTION_SMALL",
    "walk_tiny":        "WALK_TINY",
    "run_jump_tiny":    "RUN_JUMP_TINY",
}


# ── Angle conversion ──────────────────────────────────────────────────────────

def knee_rad_to_included(rad: np.ndarray) -> np.ndarray:
    """MuJoCo rtibiarx (rad, 0=straight, negative=flexion) → included-angle °."""
    deg = np.degrees(np.abs(np.asarray(rad, dtype=np.float64)))
    return np.clip(180.0 - deg, 0.0, 180.0).astype(np.float32)


def hip_rad_to_included(rad: np.ndarray) -> np.ndarray:
    """MuJoCo rfemurry (rad, 0=neutral, positive=flexion) → included-angle °."""
    deg = np.degrees(np.asarray(rad, dtype=np.float64))
    return (180.0 - deg).astype(np.float32)


def included_to_knee_rad(inc_deg: np.ndarray) -> np.ndarray:
    """Included-angle ° → MuJoCo rtibiarx radians (negative = flexion)."""
    return -np.radians(180.0 - np.asarray(inc_deg, dtype=np.float64)).astype(np.float64)


def included_to_hip_rad(inc_deg: np.ndarray) -> np.ndarray:
    """Included-angle ° → MuJoCo rfemurry radians (positive = flexion)."""
    return np.radians(180.0 - np.asarray(inc_deg, dtype=np.float64)).astype(np.float64)


# ── Polyphase resample ────────────────────────────────────────────────────────

def _resample(arr: np.ndarray, from_fps: float, to_fps: float) -> np.ndarray:
    if abs(from_fps - to_fps) < 0.5:
        return arr.astype(np.float32)
    from scipy.signal import resample_poly
    up   = int(round(to_fps))
    down = int(round(from_fps))
    g    = gcd(up, down)
    return resample_poly(arr.astype(np.float64), up // g, down // g).astype(np.float32)


# ── HDF5 path discovery ───────────────────────────────────────────────────────

def get_h5_path() -> Optional[Path]:
    """Return path to the CMU MoCap Act HDF5 file (auto-downloaded if absent)."""
    import os
    env_override = os.environ.get("MOCAPACT_H5_PATH", "").strip()
    if env_override:
        p = Path(env_override)
        if p.exists():
            return p
        warnings.warn(f"MOCAPACT_H5_PATH={env_override!r} not found; trying dm_control default.")
    try:
        from dm_control.locomotion.mocap import cmu_mocap_data
        p = Path(cmu_mocap_data.get_path_for_cmu())
        return p if p.exists() else None
    except Exception:
        return None


# ── Per-clip angle extraction ─────────────────────────────────────────────────

def _extract_via_loader(clip_id: str, h5_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Try dm_control HDF5TrajectoryLoader → (knee_rad, hip_rad, fps)."""
    try:
        from dm_control.locomotion.mocap import loader as mocap_loader
        if not hasattr(mocap_loader, "HDF5TrajectoryLoader"):
            return None
        tl   = mocap_loader.HDF5TrajectoryLoader(str(h5_path))
        traj = tl.get_trajectory(clip_id) if hasattr(tl, "get_trajectory") else tl.load(clip_id)
        fps  = float(getattr(traj, "fps", _NATIVE_FPS))
        if fps <= 0:
            fps = 1.0 / float(getattr(traj, "dt", 1.0 / _NATIVE_FPS))

        # Try as_dict() first
        try:
            d = traj.as_dict()
            for key, is_full in (
                ("walker/joints_pos", False),
                ("walker/joints",     False),
                ("joints_pos",        False),
                ("joints",            False),
                ("qpos",              True),
            ):
                if key not in d:
                    continue
                data = np.asarray(d[key], dtype=np.float32)
                if data.ndim != 2:
                    continue
                k_idx = KNEE_QPOS_IDX if is_full else KNEE_QPOS_IDX - _ROOT_DOFS
                h_idx = HIP_QPOS_IDX  if is_full else HIP_QPOS_IDX  - _ROOT_DOFS
                if k_idx < 0 or h_idx < 0 or data.shape[1] <= max(k_idx, h_idx):
                    continue
                return data[:, k_idx], data[:, h_idx], fps
        except Exception:
            pass

        # Try direct attributes
        for attr, is_full in (
            ("qpos",       True),
            ("joints_pos", False),
            ("joint_pos",  False),
            ("joints",     False),
        ):
            data = getattr(traj, attr, None)
            if data is None:
                continue
            data = np.asarray(data, dtype=np.float32)
            k_idx = KNEE_QPOS_IDX if is_full else KNEE_QPOS_IDX - _ROOT_DOFS
            h_idx = HIP_QPOS_IDX  if is_full else HIP_QPOS_IDX  - _ROOT_DOFS
            if data.ndim == 2 and k_idx >= 0 and h_idx >= 0 and data.shape[1] > max(k_idx, h_idx):
                return data[:, k_idx], data[:, h_idx], fps
    except Exception:
        pass
    return None


def _get_mocap_joint_indices(h5_path: Path) -> Tuple[int, int]:
    """Return (knee_idx, hip_idx) in the HDF5 walkers/walker_0/joints array."""
    try:
        from dm_control.locomotion.walkers import cmu_humanoid
        walker = cmu_humanoid.CMUHumanoid()
        names  = [j.name for j in walker.mocap_joints]
        return names.index(_KNEE_JOINT), names.index(_HIP_JOINT)
    except Exception:
        return _KNEE_MOCAP_IDX, _HIP_MOCAP_IDX


def _extract_via_h5py(clip_id: str, h5_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Direct h5py read → (knee_rad, hip_rad, fps)."""
    try:
        import h5py
        knee_mi, hip_mi = _get_mocap_joint_indices(h5_path)
        k_hinge = KNEE_QPOS_IDX - _ROOT_DOFS
        h_hinge = HIP_QPOS_IDX  - _ROOT_DOFS

        with h5py.File(str(h5_path), "r") as f:
            if clip_id not in f:
                return None
            grp = f[clip_id]

            candidates = [
                # (key, knee_col, hip_col, needs_transpose)
                ("walkers/walker_0/joints", knee_mi,  hip_mi,  True),
                ("walkers/0/joints",        knee_mi,  hip_mi,  True),
                ("walker/joints_pos",       k_hinge,  h_hinge, False),
                ("walker/joints",           k_hinge,  h_hinge, False),
                ("joints",                  k_hinge,  h_hinge, False),
                ("qpos",                    KNEE_QPOS_IDX, HIP_QPOS_IDX, False),
            ]
            for key, k_col, h_col, transpose in candidates:
                if key not in grp:
                    continue
                data = np.asarray(grp[key], dtype=np.float32)
                if transpose and data.ndim == 2:
                    data = data.T
                if data.ndim != 2:
                    continue
                if k_col < 0 or h_col < 0 or data.shape[1] <= max(k_col, h_col):
                    continue
                dt = (grp.attrs.get("dt") or f.attrs.get("dt") or (1.0 / _NATIVE_FPS))
                fps = (grp.attrs.get("fps") or f.attrs.get("fps") or (1.0 / float(dt)))
                return data[:, k_col], data[:, h_col], float(fps)
    except Exception:
        pass
    return None


def _extract_clip(clip_id: str, h5_path: Path, target_fps: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract (knee_included_deg, hip_included_deg) at target_fps for one clip."""
    for fn in (_extract_via_loader, _extract_via_h5py):
        result = fn(clip_id, h5_path)
        if result is None:
            continue
        knee_rad, hip_rad, fps = result
        knee = _resample(knee_rad_to_included(knee_rad), fps, target_fps)
        hip  = _resample(hip_rad_to_included(hip_rad),   fps, target_fps)
        return knee, hip
    return None


# ── Clip ID enumeration ───────────────────────────────────────────────────────

def get_clip_ids(subset: str = "all") -> List[str]:
    """Return clip IDs for the requested subset.

    For ``subset="all"`` the clip list is read directly from the HDF5 file's
    top-level keys, which yields all ~2589 MoCapAct clips.  dm_control's own
    ``cmu_subsets.ALL`` only covers its smaller curated set (~836 clips), so
    we bypass it here to get the full dataset.

    For all other subsets (``locomotion_small``, ``walk_tiny``,
    ``run_jump_tiny``) the dm_control ``cmu_subsets`` constants are used as
    before.
    """
    if subset.lower() == "all":
        h5_path = get_h5_path()
        if h5_path is not None:
            try:
                import h5py
                with h5py.File(str(h5_path), "r") as f:
                    ids = sorted(f.keys())
                if ids:
                    return ids
            except Exception as exc:
                warnings.warn(
                    f"[db] Could not enumerate HDF5 keys for 'all' subset: {exc}. "
                    "Falling back to dm_control cmu_subsets.ALL."
                )

    from dm_control.locomotion.tasks.reference_pose import cmu_subsets
    attr  = _SUBSET_MAP.get(subset.lower(), subset.upper())
    clips = getattr(cmu_subsets, attr)
    if hasattr(clips, "ids"):
        return list(clips.ids)
    if hasattr(clips, "_clips"):
        return [c.id if hasattr(c, "id") else str(c) for c in clips._clips]
    return [c.id if hasattr(c, "id") else str(c) for c in clips]


# ── Main public API ───────────────────────────────────────────────────────────

def load_database(
    subset: str = "locomotion_small",
    use_cache: bool = True,
    target_fps: float = TARGET_FPS,
) -> dict:
    """Build (or load cached) motion-matching database from MoCap Act clips.

    Parameters
    ----------
    subset : str
        ``"mocapact"``        Microsoft MoCapAct (~2 589 snippets, recommended).
                              Requires MOCAPACT_MS_DIR to be set; see
                              ``mocap_evaluation.mocapact_ms.download_instructions()``.
        ``"all"``             dm_control CMU HDF5 (all top-level keys, ~837 clips).
        ``"locomotion_small"`` dm_control CMU locomotion subset (~316 clips).
        ``"walk_tiny"``       dm_control CMU walk-only subset (36 clips).
        ``"run_jump_tiny"``   dm_control CMU run+jump subset (50 clips).
    use_cache : bool
        Load from ``.cache_mocapact_{subset}.npz`` if it exists.
    target_fps : float
        Resample all clips to this rate (200 Hz = EMG pipeline default).

    Returns
    -------
    dict with keys:
        ``knee_right``      (N,) float32 included-angle degrees
        ``hip_right``       (N,) float32 included-angle degrees
        ``file_boundaries`` list of (start, end, clip_id, category) tuples
        ``fps``             float
        ``source``          str
    """
    # ── Microsoft MoCapAct path ───────────────────────────────────────────────
    if subset.lower() == "mocapact":
        from mocap_evaluation.mocapact_ms import load_database_ms
        cache_path = Path(".cache_mocapact_ms.npz")
        if use_cache and cache_path.exists():
            try:
                raw = np.load(str(cache_path), allow_pickle=True)
                db  = {k: raw[k] for k in raw.files}
                if "file_boundaries" in db:
                    db["file_boundaries"] = [tuple(b) for b in db["file_boundaries"].tolist()]
                n     = len(db.get("knee_right", []))
                clips = len(db.get("file_boundaries", []))
                print(f"[db] Loaded {n} frames ({clips} snippets) from cache {cache_path}")
                return db
            except Exception as exc:
                warnings.warn(f"[db] Cache load failed ({exc}); rebuilding.")

        db = load_database_ms(target_fps=target_fps)
        try:
            save = dict(db)
            save["file_boundaries"] = np.array(db["file_boundaries"], dtype=object)
            np.savez(str(cache_path), **save)
            print(f"[db] Cached to {cache_path}")
        except Exception as exc:
            warnings.warn(f"[db] Cache save failed: {exc}")
        return db

    # ── dm_control CMU HDF5 path ──────────────────────────────────────────────
    cache_path = Path(f".cache_mocapact_{subset}.npz")

    if use_cache and cache_path.exists():
        try:
            raw = np.load(str(cache_path), allow_pickle=True)
            db  = {k: raw[k] for k in raw.files}
            if "file_boundaries" in db:
                db["file_boundaries"] = [tuple(b) for b in db["file_boundaries"].tolist()]
            n     = len(db.get("knee_right", []))
            clips = len(db.get("file_boundaries", []))
            # Warn if an "all" cache looks like it was built with the old
            # dm_control clip list (~836) rather than the full HDF5 (~2589).
            if subset.lower() == "all" and clips < 1000:
                warnings.warn(
                    f"[db] Cache {cache_path} has only {clips} clips — it was "
                    "probably built with dm_control's cmu_subsets.ALL (836 clips) "
                    "rather than the full MoCapAct HDF5 (2589 clips). "
                    f"Delete {cache_path} and re-run to rebuild with all clips."
                )
            print(f"[db] Loaded {n} frames ({clips} clips) from cache {cache_path}")
            return db
        except Exception as exc:
            warnings.warn(f"[db] Cache load failed ({exc}); rebuilding.")

    h5_path = get_h5_path()
    if h5_path is None:
        raise RuntimeError(
            "MoCap Act HDF5 file not found.  Install dm_control and h5py; "
            "the file is auto-downloaded on first use (~454 MB).\n"
            "  pip install dm_control h5py\n"
            "Or set MOCAPACT_H5_PATH=/path/to/cmu_2020_*.h5"
        )

    clip_ids = get_clip_ids(subset)
    print(f"[db] Building database: {len(clip_ids)} clips (subset={subset!r}) …")

    from tqdm import tqdm
    all_knee: List[np.ndarray] = []
    all_hip:  List[np.ndarray] = []
    boundaries: List[tuple]    = []
    skipped = 0
    cursor  = 0

    for clip_id in tqdm(clip_ids, desc="Loading clips", unit="clip"):
        result = _extract_clip(clip_id, h5_path, target_fps)
        if result is None or len(result[0]) < 2:
            skipped += 1
            continue
        knee, hip = result
        n = len(knee)
        all_knee.append(knee)
        all_hip.append(hip)
        boundaries.append((cursor, cursor + n, clip_id, "locomotion"))
        cursor += n

    if not all_knee:
        raise RuntimeError(
            "No clips could be loaded.  Check that the HDF5 file is accessible."
        )

    print(f"[db] {len(all_knee)} clips loaded ({skipped} skipped), "
          f"{cursor} frames @ {target_fps:.0f} Hz.")

    db = {
        "knee_right":      np.concatenate(all_knee),
        "hip_right":       np.concatenate(all_hip),
        "file_boundaries": boundaries,
        "fps":             np.float32(target_fps),
        "source":          np.array(f"mocapact/{subset}"),
    }

    try:
        save = dict(db)
        save["file_boundaries"] = np.array(boundaries, dtype=object)
        np.savez(str(cache_path), **save)
        print(f"[db] Cached to {cache_path}")
    except Exception as exc:
        warnings.warn(f"[db] Cache save failed: {exc}")

    return db
