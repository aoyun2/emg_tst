"""
Mocap data loader.

Source : Bandai Namco Research Motiondataset-2, CMU Graphics Lab Motion
         Capture Database, LAFAN1 (Ubisoft La Forge), and SFU Motion Capture
         Database — all in BVH format with per-file category metadata.

All sources are stored in per-dataset sub-folders under a common root
(default ``mocap_data/``):

    mocap_data/
        bandai/   — Bandai Namco Motiondataset-2
        cmu/      — CMU Graphics Lab
        lafan1/   — Ubisoft LAFAN1
        sfu/      — SFU (Simon Fraser University)

Download all sources at once:
    python -m mocap_evaluation.download_all

Or individually:
    python -m mocap_evaluation.bandai_namco_downloader
    python -m mocap_evaluation.cmu_downloader
    python -m mocap_evaluation.lafan1_downloader
    python -m mocap_evaluation.sfu_downloader

Returned database dict (all angles in **degrees**, all arrays at TARGET_FPS):
    knee_right   : (N,)   right knee included angle (180 = straight)
    knee_left    : (N,)   left  knee included angle (180 = straight)
    hip_right    : (N,)   right hip included angle (180 = neutral/straight)
    hip_left     : (N,)   left  hip included angle (180 = neutral/straight)
    ankle_right  : (N,)   right ankle included angle (180 = neutral/straight)
    ankle_left   : (N,)   left  ankle included angle (180 = neutral/straight)
    pelvis_tilt  : (N,)   anterior pelvic tilt (+= anterior)
    trunk_lean   : (N,)   trunk forward lean (+= forward)
    root_pos     : (N, 3) pelvis position in metres
    fps          : float  always TARGET_FPS after resampling
    source       : str    e.g. "bandai_bvh+cmu_bvh+lafan1_bvh+sfu_bvh+aggregated"
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

# Mapping from CMU BVH joint names → our semantic names.
# Also applies to LAFAN1 and SFU skeletons which use the same naming.
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


def _resample_2d(arr: np.ndarray, src_fps: float, tgt_fps: int) -> np.ndarray:
    """Resample each column of a 2-D array independently."""
    return np.stack([_resample(arr[:, c], src_fps, tgt_fps) for c in range(arr.shape[1])], axis=1)


# ── BVH angle extraction ────────────────────────────────────────────────────


def _extract_angles_from_bvh(parser: BVHParser) -> Optional[dict]:
    """
    Extract walking-relevant joint angles from a parsed BVH file.

    Supports the CMU cgspeed skeleton (joint names "RightLeg", etc.), the
    Bandai Namco Motiondataset-2 skeleton ("LowerLeg_R", etc.), and any
    skeleton using the MotionBuilder naming convention (LAFAN1, SFU, etc.).
    The correct joint map is selected automatically based on joint names
    present in the parsed file.

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

    # Ensure positive = flexion first (CMU cgspeed stores flexion as negative
    # Xrotation; Bandai Namco uses the same ZXY convention so the same fix
    # applies), then convert to included-angle convention:
    #   180° = straight, decreasing with flexion magnitude.
    for k in ("knee_right", "knee_left", "hip_right", "hip_left"):
        if k in angles and float(angles[k].mean()) < -5.0:
            angles[k] = -angles[k]

    for k in ("knee_right", "knee_left", "hip_right", "hip_left",
              "ankle_right", "ankle_left", "pelvis_tilt", "trunk_lean"):
        if k in angles:
            angles[k] = np.clip(180.0 - np.abs(angles[k]), 0.0, 180.0).astype(np.float32)

    # Fill missing signals with zeros
    N = len(angles["knee_right"])
    for k in ("hip_right", "hip_left", "ankle_right", "ankle_left",
              "pelvis_tilt", "trunk_lean"):
        if k not in angles:
            angles[k] = np.zeros(N, dtype=np.float32)

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

    angles["fps"]    = float(TARGET_FPS)
    angles["source"] = source_tag
    return angles


def load_cmu_bvh(bvh_path: str | Path) -> Optional[dict]:
    """Parse a local BVH file and return the database dict."""
    bvh_path = Path(bvh_path)
    if not bvh_path.exists():
        return None
    parser = BVHParser().parse(bvh_path)
    return _extract_angles_from_bvh(parser)


# Alias kept for backward compat — works for any skeleton, not just CMU.
load_bvh = load_cmu_bvh


def _load_local_bvh_segments(bvh_dir: Path) -> tuple[list, list]:
    """Load all local .bvh files in a directory and return (segments, meta)."""
    bvh_files = sorted(bvh_dir.glob("*.bvh")) if bvh_dir.exists() else []
    segments = []
    meta = []
    for bf in bvh_files:
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

    Recognises Bandai Namco filenames (``dataset-2_{motion}_{style}_{id}.bvh``),
    LAFAN1 filenames (``{action}{N}_subject{S}.bvh``),
    SFU filenames (``{subject}_{trial}.bvh``),
    and CMU catalog format (``07_01.bvh``).
    """
    # ── Bandai Namco filenames ─────────────────────────────────────────────
    if filename.startswith("dataset-"):
        if "_walk" in filename:
            return "walk"
        if "_run" in filename:
            return "run"
        return "misc"

    # ── LAFAN1 filenames (e.g. "aiming1_subject1.bvh", "walk1_subject2.bvh") ──
    lower = filename.lower()
    if "_subject" in lower:
        action = lower.split("_subject")[0]
        # Strip trailing digits to get base action name
        action_base = action.rstrip("0123456789")
        if action_base in ("walk", "walking"):
            return "walk"
        if action_base in ("run", "running", "sprint"):
            return "run"
        if action_base in ("jump", "jumping"):
            return "jump"
        if action_base in ("dance", "dancing"):
            return "dance"
        if action_base in ("fight", "fighting", "punch", "kick"):
            return "sport"
        if action_base in ("aim", "aiming"):
            return "misc"
        if action_base in ("ground", "obstacle", "push", "fallandgetup"):
            return "misc"
        return action_base if action_base else "misc"

    # ── SFU filenames (e.g. "0005_Walking001.bvh", "0007_Running002.bvh") ──
    # Pattern: 4-digit subject ID + underscore + MotionNameNNN.bvh
    if len(lower) > 5 and lower[:4].isdigit() and lower[4] == "_":
        motion = lower[5:].split(".")[0].rstrip("0123456789")
        if motion in ("walking",):
            return "walk"
        if motion in ("running", "jogging"):
            return "run"
        if motion in ("jumping",):
            return "jump"
        if motion in ("kicking", "boxing", "punching"):
            return "sport"
        return motion if motion else "misc"

    # ── CMU catalog lookup ─────────────────────────────────────────────────
    try:
        from mocap_evaluation.cmu_catalog import CATALOG
    except ImportError:
        return "unknown"

    for trial in CATALOG:
        if trial.filename == filename:
            return trial.category
    return "unknown"


# ── Auto-download helpers ─────────────────────────────────────────────────────


def _auto_download_bandai(dest_dir: Path) -> None:
    if sorted(dest_dir.glob("*.bvh")):
        return
    try:
        from mocap_evaluation.bandai_namco_downloader import download_locomotion
        print(f"[mocap_loader] Downloading Bandai locomotion to {dest_dir} …")
        download_locomotion(dest_dir=dest_dir, verbose=True)
    except Exception as exc:
        print(f"  Bandai download failed: {exc}")


def _auto_download_cmu(dest_dir: Path) -> None:
    if sorted(dest_dir.glob("*.bvh")):
        return
    try:
        from mocap_evaluation.cmu_downloader import download_all
        print(f"[mocap_loader] Downloading CMU data to {dest_dir} …")
        download_all(dest_dir=dest_dir)
    except Exception as exc:
        print(f"  CMU download failed: {exc}")


def _auto_download_lafan1(dest_dir: Path) -> None:
    if sorted(dest_dir.glob("*.bvh")):
        return
    try:
        from mocap_evaluation.lafan1_downloader import download_all
        print(f"[mocap_loader] Downloading LAFAN1 data to {dest_dir} …")
        download_all(dest_dir=dest_dir)
    except Exception as exc:
        print(f"  LAFAN1 download failed: {exc}")


def _auto_download_sfu(dest_dir: Path) -> None:
    if sorted(dest_dir.glob("*.bvh")):
        return
    try:
        from mocap_evaluation.sfu_downloader import download_all
        print(f"[mocap_loader] Downloading SFU data to {dest_dir} …")
        download_all(dest_dir=dest_dir)
    except Exception as exc:
        print(f"  SFU download failed: {exc}")


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


# ── Main entry point — always aggregated ──────────────────────────────────────


def load_aggregated_database(
    mocap_root: str | Path = MOCAP_DATA_DIR,
    try_download: bool = True,
) -> dict:
    """Load an aggregated mocap database from **all** dataset sub-folders.

    Sub-folders searched (relative to *mocap_root*):
        bandai/  — Bandai Namco Motiondataset-2
        cmu/     — CMU Graphics Lab
        lafan1/  — Ubisoft LAFAN1
        sfu/     — SFU Motion Capture Database

    If a sub-folder is empty and ``try_download`` is True the corresponding
    downloader is invoked automatically.
    """
    mocap_root = Path(mocap_root)
    bandai_dir = mocap_root / "bandai"
    cmu_dir    = mocap_root / "cmu"
    lafan1_dir = mocap_root / "lafan1"
    sfu_dir    = mocap_root / "sfu"

    if try_download:
        _auto_download_bandai(bandai_dir)
        _auto_download_cmu(cmu_dir)
        _auto_download_lafan1(lafan1_dir)
        _auto_download_sfu(sfu_dir)

    all_segments: list = []
    all_meta: list = []
    for label, d in [("bandai", bandai_dir), ("cmu", cmu_dir),
                     ("lafan1", lafan1_dir), ("sfu", sfu_dir)]:
        segs, meta = _load_local_bvh_segments(d)
        if segs:
            print(f"[mocap_loader] {label}: {len(segs)} files")
        all_segments.extend(segs)
        all_meta.extend(meta)

    if not all_segments:
        raise RuntimeError(
            "No BVH mocap data available.\n"
            f"Expected files in sub-folders of: {mocap_root}\n"
            "Download everything at once:\n"
            "  python -m mocap_evaluation.download_all\n"
            "Or individually:\n"
            "  python -m mocap_evaluation.bandai_namco_downloader\n"
            "  python -m mocap_evaluation.cmu_downloader\n"
            "  python -m mocap_evaluation.lafan1_downloader\n"
            "  python -m mocap_evaluation.sfu_downloader"
        )

    combined = _concatenate_databases(all_segments, meta=all_meta)
    total_dur = len(combined["knee_right"]) / TARGET_FPS
    n_cats = len({m[1] for m in all_meta})
    sources = {d.get("source", "unknown") for d in all_segments}
    combined["source"] = "+".join(sorted(sources)) + "+aggregated"
    print(f"[mocap_loader] Aggregated {len(all_segments)} files, {total_dur:.1f}s total, "
          f"{n_cats} categories (sources: {combined['source']})")
    return combined


# ── Backward-compatible aliases ───────────────────────────────────────────────
# All three old entry points now delegate to the single aggregated loader so
# that callers always get the full multi-source database regardless of which
# function they call.

load_aggregated_bandai_cmu_database = load_aggregated_database


def load_or_generate_mocap_database(
    bvh_dir: str | Path = MOCAP_DATA_DIR,
    try_download: bool = True,
) -> dict:
    """Backward-compatible entry point — delegates to :func:`load_aggregated_database`."""
    return load_aggregated_database(mocap_root=bvh_dir, try_download=try_download)


def load_full_cmu_database(
    bvh_dir: str | Path = MOCAP_DATA_DIR,
    try_download: bool = True,
) -> dict:
    """Backward-compatible entry point — delegates to :func:`load_aggregated_database`."""
    return load_aggregated_database(mocap_root=bvh_dir, try_download=try_download)
