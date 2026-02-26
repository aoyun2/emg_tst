"""
Mocap data loader.

Source : CMU Graphics Lab Motion Capture Database (BVH format).
         Supports the full CMU catalog (140+ subjects, 2600+ trials)
         with per-file category metadata for motion-type-aware matching.

Falls back to built-in synthetic Winter 2009 gait if no BVH files are found.
Download real CMU data for more diverse motion matching:
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
from typing import Dict, List, Optional, Sequence

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


# ── synthetic generator ───────────────────────────────────────────────────────


def generate_synthetic_gait(
    n_cycles: int = 20,
    cadence_steps_per_min: float = 110.0,
    fps: int = TARGET_FPS,
) -> dict:
    """
    Generate synthetic normal gait kinematics.

    One gait cycle = two steps (right heel-strike to right heel-strike).
    Cadence here is in steps/min (one step = one swing phase).
    At 110 steps/min ≈ 55 cycles/min → cycle period ≈ 1.09 s.

    Returns the mocap database dict (see module docstring).
    """
    cycle_period_s  = 60.0 / (cadence_steps_per_min / 2.0)   # seconds per full cycle
    samples_per_cyc = int(round(cycle_period_s * fps))

    # ── single cycle curves ────────────────────────────────────────────────
    kr = _interp_gait_curve(_KNEE_R,       samples_per_cyc)
    hr = _interp_gait_curve(_HIP_R,        samples_per_cyc)
    ar = _interp_gait_curve(_ANKLE_R,      samples_per_cyc)
    pt = _interp_gait_curve(_PELVIS_TILT,  samples_per_cyc)
    tl = _interp_gait_curve(_TRUNK_LEAN,   samples_per_cyc)

    # Left side: 50% phase shift
    half = samples_per_cyc // 2
    kl = np.roll(kr, half)
    hl = np.roll(hr, half)
    al = np.roll(ar, half)

    # Tile to n_cycles
    def tile(x):
        return np.tile(x, n_cycles).astype(np.float32)

    N = samples_per_cyc * n_cycles

    # Root position: constant forward speed, height oscillates slightly
    # Typical comfortable walking speed ≈ 1.35 m/s
    speed_mps  = 1.35
    t          = np.arange(N, dtype=np.float32) / fps
    root_x     = (speed_mps * t).astype(np.float32)
    # Vertical CoM oscillates ≈ ±2.5 cm at cadence * 2 Hz (each step)
    step_freq  = cadence_steps_per_min / 60.0
    root_z     = (0.90 + 0.025 * np.sin(2 * np.pi * step_freq * 2 * t)).astype(np.float32)
    root_pos   = np.stack([root_x, np.zeros(N, np.float32), root_z], axis=1)

    categories = np.full(N, "walk", dtype=object)
    return {
        "knee_right":     tile(kr),
        "knee_left":      tile(kl),
        "hip_right":      tile(hr),
        "hip_left":       tile(hl),
        "ankle_right":    tile(ar),
        "ankle_left":     tile(al),
        "pelvis_tilt":    tile(pt),
        "trunk_lean":     tile(tl),
        "root_pos":       root_pos,
        "fps":            float(fps),
        "source":         "synthetic",
        "categories":     categories,
        "file_boundaries": [(0, N, "synthetic_gait", "walk")],
    }


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

    Returns the database dict at TARGET_FPS, or None if key joints are missing.
    """
    src_fps = parser.fps
    angles  = {}

    for bvh_joint, key in _CMU_JOINT_MAP.items():
        arr = parser.get_flexion(bvh_joint)
        if arr is None:
            continue
        angles[key] = _resample(arr, src_fps, TARGET_FPS)

    # We need at least the knee signals
    if "knee_right" not in angles or "knee_left" not in angles:
        return None

    # CMU BVH knee flexion convention varies: some conversions store flexion as
    # negative Xrotation.  Ensure positive = flexion (walking range 0–70°).
    for k in ("knee_right", "knee_left"):
        if k in angles and float(angles[k].mean()) < -5.0:
            angles[k] = -angles[k]

    # Fill missing signals with zeros
    N = len(angles["knee_right"])
    for k in ("hip_right", "hip_left", "ankle_right", "ankle_left",
              "pelvis_tilt", "trunk_lean"):
        if k not in angles:
            angles[k] = np.zeros(N, dtype=np.float32)

    # Root position (metres; CMU BVH positions are in cm)
    root_pos_raw = parser.get_positions("Hips")
    if root_pos_raw is not None:
        rp = _resample_2d(root_pos_raw, src_fps, TARGET_FPS) / 100.0  # cm → m
        angles["root_pos"] = rp[:N]
    else:
        angles["root_pos"] = np.zeros((N, 3), dtype=np.float32)

    angles["fps"]    = float(TARGET_FPS)
    angles["source"] = "cmu_bvh"
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

    # No BVH data — fall back to built-in synthetic gait (always works)
    print("[mocap_loader] No BVH data available — using synthetic Winter 2009 gait kinematics")
    return generate_synthetic_gait(n_cycles=20)


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
    """Look up the motion category for a BVH filename via the catalog."""
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
            print(f"[mocap_loader] Loaded {len(segments)} files, "
                  f"{total_dur:.1f}s total, {n_cats} categories")
            return combined

    raise RuntimeError(
        "No CMU BVH mocap data available. Download it first:\n"
        "  python -m mocap_evaluation.cmu_downloader"
    )
