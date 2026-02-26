"""
Mocap data loader.

Source : CMU Graphics Lab Motion Capture Database (BVH format).
         Supports the full CMU catalog (140+ subjects, 2600+ trials)
         with per-file category metadata for motion-type-aware matching.

Real CMU BVH data is required.  Download with:
    python -m mocap_evaluation.cmu_downloader

Returned database dict (all angles in **degrees**, all arrays at TARGET_FPS):
    knee_right   : (N,)   right knee flexion (+= flexion)
    knee_left    : (N,)   left  knee flexion
    hip_right    : (N,)   right hip flexion/extension from vertical (+= flex)
    hip_left     : (N,)   left  hip flexion/extension
    ankle_right  : (N,)   right ankle dorsiflexion (+= dorsiflexion)
    ankle_left   : (N,)   left  ankle dorsiflexion
    pelvis_tilt  : (N,)   anterior pelvic tilt (+= anterior)
    trunk_lean   : (N,)   trunk forward lean (+= forward)
    root_pos     : (N, 3) pelvis position in metres
    fps          : float  always TARGET_FPS after resampling
    source       : str    "cmu_bvh"
    categories   : (N,) str array — per-frame motion category label (optional)
    file_boundaries : list of (start, end, filename, category) tuples
"""
from __future__ import annotations

import os
import urllib.request
import urllib.error
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mocap_evaluation.bvh_parser import BVHParser

# ── constants ─────────────────────────────────────────────────────────────────

TARGET_FPS: int = 200          # match our IMU / EMG pipeline rate
MOCAP_DATA_DIR = Path("mocap_data")

# CMU mocap: subject 07 is a standard adult walking subject.
# Trials 01-12 are different walking speeds / conditions.
_CMU_BVH_URLS = [
    # cgspeed BVH conversion (commonly mirrored on academic CDNs)
    "https://mocap.cs.cmu.edu/subjects/07/07_01.bvh",
    "http://mocap.cs.cmu.edu/subjects/07/07_01.bvh",
]

# Mapping from CMU BVH joint names → our semantic names
_CMU_JOINT_MAP = {
    "RightLeg":   "knee_right",   # right knee
    "LeftLeg":    "knee_left",
    "RightUpLeg": "hip_right",    # right thigh / hip
    "LeftUpLeg":  "hip_left",
    "RightFoot":  "ankle_right",
    "LeftFoot":   "ankle_left",
    "LowerBack":  "pelvis_tilt",
    "Spine":      "trunk_lean",
}

# Mapping for Bandai Namco Research Motiondataset-2 BVH files.
# Joint names differ from CMU; root position is on "joint_Root" not "Hips".
_BANDAI_JOINT_MAP = {
    "LowerLeg_R": "knee_right",
    "LowerLeg_L": "knee_left",
    "UpperLeg_R": "hip_right",
    "UpperLeg_L": "hip_left",
    "Foot_R":     "ankle_right",
    "Foot_L":     "ankle_left",
    "Spine":      "trunk_lean",
    "Hips":       "pelvis_tilt",
}
_BANDAI_ROOT_JOINT = "joint_Root"

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


# ── CMU BVH loader ────────────────────────────────────────────────────────────


def _download_cmu_bvh(dest: Path) -> bool:
    """Try to download one CMU BVH walking file. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in _CMU_BVH_URLS:
        try:
            print(f"  Trying {url} …")
            urllib.request.urlretrieve(url, dest)
            print(f"  Downloaded → {dest}")
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            print(f"  Failed ({exc})")
    return False


def _extract_angles_from_bvh(parser: BVHParser) -> Optional[dict]:
    """
    Extract walking-relevant joint angles from a parsed BVH file.

    Supports both the CMU cgspeed skeleton (joint names "RightLeg", etc.) and
    the Bandai Namco Motiondataset-2 skeleton ("LowerLeg_R", etc.).  The
    correct joint map is selected automatically based on joint names present
    in the parsed file.

    Returns the database dict at TARGET_FPS, or None if key joints are missing.
    """
    src_fps = parser.fps

    # ── Auto-detect skeleton ───────────────────────────────────────────────
    if "LowerLeg_R" in parser.joints and "LowerLeg_L" in parser.joints:
        joint_map       = _BANDAI_JOINT_MAP
        root_joint_name = _BANDAI_ROOT_JOINT
        source_tag      = "bandai_bvh"
    else:
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

    # Ensure positive = flexion (CMU cgspeed stores flexion as negative Xrotation;
    # Bandai Namco uses the same ZXY convention so the same fix applies).
    for k in ("knee_right", "knee_left", "hip_right", "hip_left"):
        if k in angles and float(angles[k].mean()) < -5.0:
            angles[k] = -angles[k]

    # Fill missing signals with zeros
    N = len(angles["knee_right"])
    for k in ("hip_right", "hip_left", "ankle_right", "ankle_left",
              "pelvis_tilt", "trunk_lean"):
        if k not in angles:
            angles[k] = np.zeros(N, dtype=np.float32)

    # Root position (metres; BVH positions are typically in cm)
    # BVH uses Y-up: Xposition=lateral, Yposition=height, Zposition=forward.
    # PyBullet uses Z-up: index 0=forward(X), 1=lateral(Y), 2=height(Z).
    root_pos_raw = parser.get_positions(root_joint_name)
    if root_pos_raw is not None and root_pos_raw.shape[1] >= 3:
        rp = _resample_2d(root_pos_raw[:, :3], src_fps, TARGET_FPS) / 100.0
        rp_pb = np.empty_like(rp)
        rp_pb[:, 0] = rp[:, 2]   # forward: BVH Z → PyBullet X
        rp_pb[:, 1] = rp[:, 0]   # lateral: BVH X → PyBullet Y
        rp_pb[:, 2] = rp[:, 1]   # height:  BVH Y → PyBullet Z
        angles["root_pos"] = rp_pb[:N]
    else:
        angles["root_pos"] = np.zeros((N, 3), dtype=np.float32)

    angles["fps"]    = float(TARGET_FPS)
    angles["source"] = source_tag
    return angles


def _resample_2d(arr: np.ndarray, src_fps: float, tgt_fps: int) -> np.ndarray:
    """Resample each column of a 2-D array independently."""
    return np.stack([_resample(arr[:, c], src_fps, tgt_fps) for c in range(arr.shape[1])], axis=1)


def load_cmu_bvh(bvh_path: str | Path) -> Optional[dict]:
    """Parse a local CMU BVH file and return the database dict."""
    bvh_path = Path(bvh_path)
    if not bvh_path.exists():
        return None
    parser = BVHParser().parse(bvh_path)
    return _extract_angles_from_bvh(parser)


# ── main entry point ──────────────────────────────────────────────────────────


def load_or_generate_mocap_database(
    bvh_dir: str | Path = MOCAP_DATA_DIR,
    try_download: bool = True,
) -> dict:
    """
    Load mocap database from BVH files found in `bvh_dir`.
    If none found, optionally download CMU BVH data.

    Raises RuntimeError if no BVH data can be loaded.

    Parameters
    ----------
    bvh_dir      : directory to search (and download) BVH files
    try_download : if True, attempt to download CMU BVH if directory empty

    Returns
    -------
    database dict (see module docstring)
    """
    bvh_dir = Path(bvh_dir)
    bvh_files = sorted(bvh_dir.glob("*.bvh")) if bvh_dir.exists() else []

    # ── try local files (delegate to load_full_cmu_database for category metadata) ──
    if bvh_files:
        try:
            return load_full_cmu_database(bvh_dir=bvh_dir, try_download=False)
        except Exception:
            pass

    # ── try download ───────────────────────────────────────────────────────
    if try_download:
        print("[mocap_loader] No local BVH files. Downloading full CMU dataset …")
        try:
            from mocap_evaluation.cmu_downloader import download_all
            downloaded = download_all(dest_dir=bvh_dir)
            if downloaded:
                return load_full_cmu_database(bvh_dir=bvh_dir, try_download=False)
        except Exception:
            pass

    raise RuntimeError(
        "No BVH mocap data available. Download CMU data first:\n"
        "  python -m mocap_evaluation.cmu_downloader\n"
        "Or download Bandai Namco walking data:\n"
        "  python -m mocap_evaluation.bandai_namco_downloader"
    )


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
    keys_1d = ["knee_right", "knee_left", "hip_right", "hip_left",
               "ankle_right", "ankle_left", "pelvis_tilt", "trunk_lean"]
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


# ── Full CMU dataset loader ──────────────────────────────────────────────────


def _category_for_file(filename: str) -> str:
    """
    Look up the motion category for a BVH filename.

    Recognises both the CMU catalog format (``07_01.bvh``) and Bandai Namco
    filenames (``dataset-2_{motion}_{style}_{id}.bvh``).
    """
    # ── Bandai Namco filenames ─────────────────────────────────────────────
    if filename.startswith("dataset-"):
        if "_walk" in filename:
            return "walk"
        if "_run" in filename:
            return "run"
        return "misc"

    # ── CMU catalog lookup ─────────────────────────────────────────────────
    try:
        from mocap_evaluation.cmu_catalog import CATALOG
    except ImportError:
        return "unknown"

    for trial in CATALOG:
        if trial.filename == filename:
            return trial.category
    return "unknown"


def load_full_cmu_database(
    bvh_dir: str | Path = MOCAP_DATA_DIR,
    try_download: bool = True,
) -> dict:
    """
    Load the full CMU mocap database with category metadata.

    This is the recommended entry point for motion matching across
    diverse motion types.  All available BVH files are loaded — no
    category filtering.

    Raises RuntimeError if no BVH data can be loaded.

    Parameters
    ----------
    bvh_dir      : directory containing (or for downloading) BVH files
    try_download : if True and directory is empty, download from CMU

    Returns
    -------
    database dict (see module docstring) with additional keys:
        categories      : (N,) object array of per-frame category strings
        file_boundaries : list of (start_idx, end_idx, filename, category)
    """
    bvh_dir = Path(bvh_dir)

    # ── try download if directory empty ───────────────────────────────────
    bvh_files = sorted(bvh_dir.glob("*.bvh")) if bvh_dir.exists() else []
    if not bvh_files and try_download:
        print("[mocap_loader] No local BVH files. Downloading full CMU dataset …")
        try:
            from mocap_evaluation.cmu_downloader import download_all
            downloaded = download_all(dest_dir=bvh_dir)
            if downloaded:
                bvh_files = sorted(bvh_dir.glob("*.bvh"))
        except Exception as exc:
            print(f"  Download failed: {exc}")

    # ── load all local files with metadata ────────────────────────────────
    if bvh_files:
        segments = []
        meta = []

        for bf in bvh_files:
            cat = _category_for_file(bf.name)
            db = load_cmu_bvh(bf)
            if db is not None:
                dur = len(db["knee_right"]) / TARGET_FPS
                print(f"  Loaded {bf.name}: {dur:.1f}s [{cat}]")
                segments.append(db)
                meta.append((bf.name, cat))

        if segments:
            combined = _concatenate_databases(segments, meta=meta)
            total_dur = len(combined["knee_right"]) / TARGET_FPS
            n_cats = len({m[1] for m in meta})
            sources  = {d.get("source", "unknown") for d in segments}
            combined["source"] = "+".join(sorted(sources))
            print(f"[mocap_loader] Loaded {len(segments)} files, "
                  f"{total_dur:.1f}s total, {n_cats} categories "
                  f"(sources: {combined['source']})")
            return combined

    raise RuntimeError(
        "No BVH mocap data available. Download CMU data first:\n"
        "  python -m mocap_evaluation.cmu_downloader\n"
        "Or download Bandai Namco walking data:\n"
        "  python -m mocap_evaluation.bandai_namco_downloader"
    )
