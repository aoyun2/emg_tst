"""
Mocap data loader.

Source : CMU Graphics Lab Motion Capture Database — BVH format with
         per-file category metadata.

Data is stored under a common root (default ``mocap_data/``):

    mocap_data/
        cmu/         — CMU Graphics Lab

Download:
    python -m mocap_evaluation.cmu_downloader

Returned database dict (all angles in **degrees**, all arrays at TARGET_FPS):

    Legs (included angle: 180 = straight/neutral):
        hip_right, hip_left, knee_right, knee_left,
        ankle_right, ankle_left, toe_right, toe_left

    Spine chain (included angle):
        pelvis_tilt, trunk_lean, upper_trunk

    Head (included angle):
        neck, head

    Arms (included angle):
        clavicle_right, clavicle_left,
        shoulder_right, shoulder_left,
        elbow_right, elbow_left,
        wrist_right, wrist_left

    Fingers (included angle):
        finger_right, finger_index_right, thumb_right,
        finger_left, finger_index_left, thumb_left

    Root orientation (raw BVH degrees):
        root_pitch, root_yaw, root_roll

    Other:
        root_pos        : (N, 3) pelvis position in metres (MuJoCo Z-up)
        fps             : float  always TARGET_FPS after resampling
        source          : str    e.g. "cmu_bvh+aggregated"
        categories      : (N,) str array — per-frame category (optional)
        file_boundaries : list of (start, end, filename, category) tuples
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
import urllib.error
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from mocap_evaluation.bvh_parser import BVHParser

# ── constants ─────────────────────────────────────────────────────────────────

TARGET_FPS: int = 200          # match our IMU / EMG pipeline rate
MOCAP_DATA_DIR = Path("mocap_data")

# Mapping from CMU BVH joint names → our semantic names.
# Covers ALL non-root joints in the CMU cgspeed skeleton hierarchy.
_CMU_JOINT_MAP = {
    # Legs
    "RightUpLeg": "hip_right",
    "LeftUpLeg":  "hip_left",
    "RightLeg":   "knee_right",
    "LeftLeg":    "knee_left",
    "RightFoot":  "ankle_right",
    "LeftFoot":   "ankle_left",
    "RightToeBase": "toe_right",
    "LeftToeBase":  "toe_left",
    # Spine chain
    "LowerBack":  "pelvis_tilt",
    "Spine":      "trunk_lean",
    "Spine1":     "upper_trunk",
    # Head chain
    "Neck":       "neck",
    "Head":       "head",
    # Shoulder girdle (clavicle) + arms
    "RightShoulder": "clavicle_right",
    "LeftShoulder":  "clavicle_left",
    "RightArm":      "shoulder_right",
    "LeftArm":       "shoulder_left",
    "RightForeArm":  "elbow_right",
    "LeftForeArm":   "elbow_left",
    # Hands
    "RightHand":  "wrist_right",
    "LeftHand":   "wrist_left",
    # Fingers
    "RightFingerBase":  "finger_right",
    "RightHandIndex1":  "finger_index_right",
    "RThumb":           "thumb_right",
    "LeftFingerBase":   "finger_left",
    "LeftHandIndex1":   "finger_index_left",
    "LThumb":           "thumb_left",
}

# All semantic joint keys (excluding root_pos and root_pitch/yaw/roll)
_ALL_JOINT_KEYS = list(_CMU_JOINT_MAP.values())

# ── Normal gait kinematics from Winter (2009) ─────────────────────────────────
# Each dict maps gait-cycle percentage [0, 100] → angle in degrees.
# Right side only; left side is phase-shifted by 50%.

_KNEE_R: Dict[int, float] = {
    0: 3, 5: 18, 15: 5, 38: 3, 50: 38, 60: 60, 73: 28, 88: 5, 100: 3
}
_HIP_R: Dict[int, float] = {
    0: 20, 12: 26, 31: 2, 48: -10, 62: -8, 78: 18, 100: 20
}
_ANKLE_R: Dict[int, float] = {
    0: 0, 10: -4, 40: 10, 55: -18, 65: 5, 85: 0, 100: 0
}
_PELVIS_TILT: Dict[int, float] = {
    0: 4, 25: 6, 50: 4, 75: 6, 100: 4
}
_TRUNK_LEAN: Dict[int, float] = {
    0: 3, 50: 4, 100: 3
}

# ── helpers ───────────────────────────────────────────────────────────────────


def _interp_gait_curve(kv: Dict[int, float], n_samples: int) -> np.ndarray:
    """Interpolate a gait-cycle keyframe dict to n_samples."""
    xp = sorted(kv.keys())
    fp = [kv[x] for x in xp]
    t  = np.linspace(0, 100, n_samples)
    return np.interp(t, xp, fp).astype(np.float32)


def _resample(signal: np.ndarray, src_fps: float, tgt_fps: int = TARGET_FPS) -> np.ndarray:
    """Polyphase resample a 1-D signal from src_fps to tgt_fps."""
    from scipy.signal import resample_poly
    if abs(src_fps - tgt_fps) < 0.5:
        return signal.astype(np.float32)
    g   = gcd(int(round(src_fps)), tgt_fps)
    up  = tgt_fps // g
    dn  = int(round(src_fps)) // g
    return resample_poly(signal, up, dn).astype(np.float32)


def _resample_2d(arr: np.ndarray, src_fps: float, tgt_fps: int) -> np.ndarray:
    """Resample each column of a 2-D array independently."""
    return np.stack([_resample(arr[:, c], src_fps, tgt_fps) for c in range(arr.shape[1])], axis=1)


# ── BVH angle extraction ────────────────────────────────────────────────────


def _extract_angles_from_bvh(parser: BVHParser) -> Optional[dict]:
    """
    Extract walking-relevant joint angles from a parsed BVH file.

    Supports the CMU cgspeed skeleton (joint names "RightLeg", etc.).

    Returns the database dict at TARGET_FPS, or None if key joints are missing.
    """
    src_fps = parser.fps
    joint_map       = _CMU_JOINT_MAP
    root_joint_name = "Hips"
    source_tag      = "cmu_bvh"

    angles: dict = {}
    for bvh_joint, key in joint_map.items():
        arr = parser.get_flexion(bvh_joint)
        if arr is None:
            continue
        angles[key] = _resample(arr, src_fps, TARGET_FPS)

    # We need at least the knee signals
    if "knee_right" not in angles or "knee_left" not in angles:
        return None

    # Ensure positive = flexion first (CMU cgspeed stores flexion as negative
    # Xrotation; Bandai Namco uses the same ZXY convention so the same fix
    # applies), then convert to included-angle convention:
    #   180° = straight, decreasing with flexion magnitude.
    for k in ("knee_right", "knee_left", "hip_right", "hip_left"):
        if k in angles and float(angles[k].mean()) < -5.0:
            angles[k] = -angles[k]

    # Knees & elbows: flexion is always positive, abs() is safe
    for k in ("knee_right", "knee_left", "elbow_right", "elbow_left"):
        if k in angles:
            angles[k] = np.clip(180.0 - np.abs(angles[k]), 0.0, 180.0).astype(np.float32)

    # Hips, ankles, toes: MUST preserve sign for extension/plantarflexion.
    for k in ("hip_right", "hip_left", "ankle_right", "ankle_left",
              "toe_right", "toe_left"):
        if k in angles:
            angles[k] = (180.0 - angles[k]).astype(np.float32)

    # Spine chain, neck, head: typically small symmetric values
    for k in ("pelvis_tilt", "trunk_lean", "upper_trunk", "neck", "head"):
        if k in angles:
            angles[k] = np.clip(180.0 - np.abs(angles[k]), 0.0, 180.0).astype(np.float32)

    # Shoulders & clavicles: sign-flip heuristic, then included-angle
    for k in ("shoulder_right", "shoulder_left",
              "clavicle_right", "clavicle_left"):
        if k in angles and float(angles[k].mean()) < -5.0:
            angles[k] = -angles[k]
        if k in angles:
            angles[k] = (180.0 - angles[k]).astype(np.float32)

    # Wrists: preserve sign
    for k in ("wrist_right", "wrist_left"):
        if k in angles:
            angles[k] = (180.0 - angles[k]).astype(np.float32)

    # Fingers & thumbs: flexion, abs() is safe
    for k in ("finger_right", "finger_index_right", "thumb_right",
              "finger_left", "finger_index_left", "thumb_left"):
        if k in angles:
            angles[k] = np.clip(180.0 - np.abs(angles[k]), 0.0, 180.0).astype(np.float32)

    # Fill missing signals with neutral (180 = straight/neutral)
    N = len(angles["knee_right"])
    for k in _ALL_JOINT_KEYS:
        if k not in angles:
            angles[k] = np.full(N, 180.0, dtype=np.float32)

    # Root position (metres; BVH positions are typically in cm)
    # BVH uses Y-up: Xposition=lateral, Yposition=height, Zposition=forward.
    # MuJoCo uses Z-up: index 0=forward(X), 1=lateral(Y), 2=height(Z).
    root_pos_raw = parser.get_positions(root_joint_name)
    if root_pos_raw is not None and root_pos_raw.shape[1] >= 3:
        rp = _resample_2d(root_pos_raw[:, :3], src_fps, TARGET_FPS) / 100.0
        rp_zup = np.empty_like(rp)
        rp_zup[:, 0] = rp[:, 2]   # forward: BVH Z → MuJoCo X
        rp_zup[:, 1] = rp[:, 0]   # lateral: BVH X → MuJoCo Y
        rp_zup[:, 2] = rp[:, 1]   # height:  BVH Y → MuJoCo Z
        angles["root_pos"] = rp_zup[:N]
    else:
        angles["root_pos"] = np.zeros((N, 3), dtype=np.float32)

    # Root orientation from Hips rotation channels (degrees).
    # BVH Y-up convention:
    #   Xrotation = pitch (forward/backward lean, sagittal plane)
    #   Yrotation = yaw   (turning, transverse plane)
    #   Zrotation = roll  (lateral lean, frontal plane)
    # Stored as raw BVH degrees; the simulation converts to MuJoCo Z-up.
    for ch_name, key in [("Xrotation", "root_pitch"),
                         ("Yrotation", "root_yaw"),
                         ("Zrotation", "root_roll")]:
        ch = parser.get_channel(root_joint_name, ch_name)
        if ch is not None:
            angles[key] = _resample(ch, src_fps, TARGET_FPS)[:N]
        else:
            angles[key] = np.zeros(N, dtype=np.float32)

    angles["fps"]    = float(TARGET_FPS)
    angles["source"] = source_tag
    return angles


def load_cmu_bvh(bvh_path: str | Path) -> Optional[dict]:
    """Parse a local BVH file and return the database dict."""
    bvh_path = Path(bvh_path)
    if not bvh_path.exists():
        return None
    try:
        parser = BVHParser().parse(bvh_path)
    except (ValueError, IndexError) as exc:
        print(f"  [warn] Skipping {bvh_path.name}: {exc}")
        return None
    return _extract_angles_from_bvh(parser)


# Alias kept for backward compat — works for any skeleton, not just CMU.
load_bvh = load_cmu_bvh


def _load_local_bvh_segments(bvh_dir: Path) -> tuple[list, list]:
    """Load all local .bvh files in a directory and return (segments, meta)."""
    bvh_files = sorted(bvh_dir.glob("*.bvh")) if bvh_dir.exists() else []
    segments = []
    meta = []
    for bf in tqdm(bvh_files, desc=f"Loading {bvh_dir.name}", unit="file", disable=len(bvh_files) == 0):
        cat = _category_for_file(bf.name)
        db = load_cmu_bvh(bf)
        if db is not None:
            segments.append(db)
            meta.append((bf.name, cat))
    return segments, meta


# ── Category detection ────────────────────────────────────────────────────────


def _category_for_file(filename: str) -> str:
    """
    Look up the motion category for a BVH filename.

    Recognises CMU catalog format (``07_01.bvh``).
    """
    try:
        from mocap_evaluation.cmu_catalog import CATALOG
    except ImportError:
        return "unknown"

    for trial in CATALOG:
        if trial.filename == filename:
            return trial.category
    return "unknown"


# ── Auto-download helpers ─────────────────────────────────────────────────────


def _auto_download_cmu(dest_dir: Path) -> None:
    existing = sorted(dest_dir.glob("*.bvh"))
    if not existing:
        # No files at all — full download
        try:
            from mocap_evaluation.cmu_downloader import download_all
            print(f"[mocap_loader] Downloading CMU data to {dest_dir} …")
            download_all(dest_dir=dest_dir)
        except Exception as exc:
            print(f"  CMU download failed: {exc}")
    else:
        # Files exist — verify completeness and download missing
        try:
            from mocap_evaluation.cmu_downloader import verify_and_download_missing
            verify_and_download_missing(dest_dir=dest_dir)
        except Exception as exc:
            print(f"  CMU verification failed: {exc}")




# ── Concatenation ─────────────────────────────────────────────────────────────


def _concatenate_databases(dbs: list, meta: Optional[list] = None) -> dict:
    """
    Concatenate multiple database dicts along the time axis.

    Parameters
    ----------
    dbs  : list of database dicts (each has 1-D joint arrays + root_pos)
    meta : optional parallel list of (filename, category) tuples — one per db.
           When provided, per-frame ``categories`` array and
           ``file_boundaries`` list are added to the result.
    """
    keys_1d = _ALL_JOINT_KEYS + ["root_pitch", "root_yaw", "root_roll"]
    result: dict = {}
    for k in keys_1d:
        result[k] = np.concatenate([d[k] for d in dbs])
    result["root_pos"] = np.concatenate([d["root_pos"] for d in dbs], axis=0)
    result["fps"]    = float(TARGET_FPS)
    result["source"] = dbs[0]["source"]

    # ── per-frame metadata ────────────────────────────────────────────────
    if meta is not None:
        boundaries: List[tuple] = []
        cats: List[np.ndarray] = []
        offset = 0
        for db, (fname, cat) in zip(dbs, meta):
            n = len(db["knee_right"])
            boundaries.append((offset, offset + n, fname, cat))
            cats.append(np.full(n, cat, dtype=object))
            offset += n
        result["categories"] = np.concatenate(cats)
        result["file_boundaries"] = boundaries

    return result


# ── Cache helpers ─────────────────────────────────────────────────────────────

# Keys that are per-frame 1-D arrays in the database dict.
_CACHE_KEYS_1D = _ALL_JOINT_KEYS + ["root_pitch", "root_yaw", "root_roll"]

# Bump this when angle extraction or resampling logic changes so that stale
# caches are automatically invalidated (the fingerprint includes this string).
_CACHE_VERSION = "v4"


def _cache_fingerprint(bvh_dir: Path) -> str:
    """Fast fingerprint of a BVH directory: sorted filenames + sizes + code version."""
    if not bvh_dir.exists():
        return ""
    entries = [f"_version={_CACHE_VERSION}"]
    for f in sorted(bvh_dir.glob("*.bvh")):
        entries.append(f"{f.name}:{f.stat().st_size}")
    return hashlib.md5("|".join(entries).encode()).hexdigest()


def _cache_path(mocap_root: Path, datasets: Sequence[str]) -> Path:
    """Return the path for the cached .npz for a given set of datasets."""
    tag = "_".join(sorted(datasets))
    return mocap_root / f".cache_{tag}.npz"


def _save_cache(path: Path, db: dict) -> None:
    """Persist the aggregated database to a .npz file."""
    save_dict: dict = {}
    for k in _CACHE_KEYS_1D:
        save_dict[k] = db[k]
    save_dict["root_pos"] = db["root_pos"]
    save_dict["fps"] = np.float64(db["fps"])
    save_dict["source"] = np.array(db["source"])
    if "categories" in db:
        save_dict["categories"] = db["categories"]
    if "file_boundaries" in db:
        save_dict["file_boundaries"] = np.array(db["file_boundaries"], dtype=object)
    # Store the fingerprints so we can detect stale caches.
    if "_fingerprints" in db:
        save_dict["_fingerprints"] = np.array(db["_fingerprints"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save_dict)


def _load_cache(path: Path) -> Optional[dict]:
    """Load a cached .npz, returning the database dict or None."""
    if not path.exists():
        return None
    try:
        raw = np.load(path, allow_pickle=True)
    except Exception:
        return None
    db: dict = {}
    for k in _CACHE_KEYS_1D:
        if k not in raw:
            return None
        db[k] = raw[k]
    db["root_pos"] = raw["root_pos"]
    db["fps"] = float(raw["fps"])
    db["source"] = str(raw["source"])
    if "categories" in raw:
        db["categories"] = raw["categories"]
    if "file_boundaries" in raw:
        # Rows are stored as numpy object arrays; convert back to tuples
        # so downstream code gets consistent (start, end, filename, cat).
        db["file_boundaries"] = [tuple(row) for row in raw["file_boundaries"]]
    if "_fingerprints" in raw:
        db["_fingerprints"] = str(raw["_fingerprints"])
    return db


# ── Dataset registry ─────────────────────────────────────────────────────────

# Maps short dataset names to (sub-folder, auto-download function).
_DATASET_REGISTRY: Dict[str, tuple] = {}


def _register_datasets():
    """Populate the dataset registry (called once at import time)."""
    _DATASET_REGISTRY["cmu"] = ("cmu", _auto_download_cmu)


ALL_DATASETS = ("cmu",)


# ── Main entry point — always aggregated ──────────────────────────────────────


def load_aggregated_database(
    mocap_root: str | Path = MOCAP_DATA_DIR,
    try_download: bool = True,
    datasets: Optional[Sequence[str]] = None,
    use_cache: bool = True,
) -> dict:
    """Load an aggregated mocap database from dataset sub-folders.

    Parameters
    ----------
    mocap_root   : root directory containing per-dataset sub-folders
    try_download : auto-download missing datasets
    datasets     : which datasets to include, e.g. ``["cmu"]``.
                   ``None`` defaults to ``["cmu"]``.
    use_cache    : if True, cache the parsed database to a ``.npz`` file
                   inside *mocap_root* and reload from cache on subsequent
                   calls.  The cache is invalidated automatically when the
                   BVH file listing changes (new/removed files).

    Sub-folders searched (relative to *mocap_root*):
        cmu/         — CMU Graphics Lab
    """
    _register_datasets()
    mocap_root = Path(mocap_root)

    if datasets is None:
        active = ["cmu"]
    else:
        active = [d.lower() for d in datasets]
        unknown = set(active) - set(ALL_DATASETS)
        if unknown:
            raise ValueError(
                f"Unknown dataset(s): {unknown}. "
                f"Available: {', '.join(ALL_DATASETS)}"
            )

    # ── Auto-download requested datasets ─────────────────────────────────
    if try_download:
        for name in active:
            subfolder, dl_fn = _DATASET_REGISTRY[name]
            dl_fn(mocap_root / subfolder)

    # ── Try loading from cache ───────────────────────────────────────────
    if use_cache:
        fingerprints = {
            name: _cache_fingerprint(mocap_root / _DATASET_REGISTRY[name][0])
            for name in active
        }
        fp_str = "|".join(f"{k}={v}" for k, v in sorted(fingerprints.items()))
        cp = _cache_path(mocap_root, active)
        cached = _load_cache(cp)
        if cached is not None and cached.get("_fingerprints") == fp_str:
            total_dur = len(cached["knee_right"]) / TARGET_FPS
            n_files = len(cached.get("file_boundaries", []))
            print(f"[mocap_loader] Loaded from cache: {cp.name} "
                  f"({total_dur:.1f}s, {n_files} files)")
            cached.pop("_fingerprints", None)
            return cached

    # ── Parse BVH files ──────────────────────────────────────────────────
    all_segments: list = []
    all_meta: list = []
    for name in active:
        subfolder, _ = _DATASET_REGISTRY[name]
        d = mocap_root / subfolder
        segs, meta = _load_local_bvh_segments(d)
        if segs:
            print(f"[mocap_loader] {name}: {len(segs)} files")
        all_segments.extend(segs)
        all_meta.extend(meta)

    if not all_segments:
        ds_list = ", ".join(active)
        raise RuntimeError(
            f"No BVH mocap data available for datasets: {ds_list}.\n"
            f"Expected files in sub-folders of: {mocap_root}\n"
            "Download with:\n"
            "  python -m mocap_evaluation.cmu_downloader"
        )

    combined = _concatenate_databases(all_segments, meta=all_meta)
    total_dur = len(combined["knee_right"]) / TARGET_FPS
    n_cats = len({m[1] for m in all_meta})
    sources = {d.get("source", "unknown") for d in all_segments}
    combined["source"] = "+".join(sorted(sources)) + "+aggregated"
    print(f"[mocap_loader] Aggregated {len(all_segments)} files, {total_dur:.1f}s total, "
          f"{n_cats} categories (sources: {combined['source']})")

    # ── Save cache ───────────────────────────────────────────────────────
    if use_cache:
        combined["_fingerprints"] = fp_str
        _save_cache(cp, combined)
        combined.pop("_fingerprints", None)
        print(f"[mocap_loader] Saved cache -> {cp.name}")

    return combined


# ── Backward-compatible aliases ───────────────────────────────────────────────


def load_or_generate_mocap_database(
    bvh_dir: str | Path = MOCAP_DATA_DIR,
    try_download: bool = True,
    **kwargs,
) -> dict:
    """Backward-compatible entry point — delegates to :func:`load_aggregated_database`."""
    return load_aggregated_database(mocap_root=bvh_dir, try_download=try_download, **kwargs)


def load_full_cmu_database(
    bvh_dir: str | Path = MOCAP_DATA_DIR,
    try_download: bool = True,
    **kwargs,
) -> dict:
    """Backward-compatible entry point — delegates to :func:`load_aggregated_database`."""
    return load_aggregated_database(mocap_root=bvh_dir, try_download=try_download, **kwargs)
