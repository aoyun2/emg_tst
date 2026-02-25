"""
Motion matching: find the mocap segment that best matches our IMU recording.

Pipeline
--------
1. Query  = z-score-normalised (knee_angle, thigh_angle) from our device at 200 Hz.
2. Database = (knee_right, hip_right) from mocap at 200 Hz.
3. Slide over database with step `stride`, compute DTW distance to query.
4. Return best-matching start index → extract ALL joint angles for that window.

DTW implementation uses a Sakoe-Chiba band for O(n · W) cost instead of O(n²).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


# ── Normalisation ─────────────────────────────────────────────────────────────


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score a 1-D or 2-D array column-wise."""
    if x.ndim == 1:
        m, s = x.mean(), x.std()
        return (x - m) / max(s, eps)
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    s = np.where(s < eps, eps, s)
    return (x - m) / s


# ── DTW ───────────────────────────────────────────────────────────────────────


def dtw_distance(
    x: np.ndarray,
    y: np.ndarray,
    band: int = 40,
) -> float:
    """
    DTW distance between two 1-D or 2-D sequences of equal length.

    Parameters
    ----------
    x, y  : (T,) or (T, D)  – must have the same length T
    band  : Sakoe-Chiba half-width (in samples); None = full (O(T²))

    Returns
    -------
    Normalised DTW distance (divided by T to allow cross-length comparison).
    """
    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]

    T = len(x)
    assert len(y) == T, "x and y must have the same number of frames for this DTW variant"

    INF  = 1e18
    w    = T if band is None else min(band, T)
    prev = np.full(T + 1, INF, dtype=np.float64)
    curr = np.full(T + 1, INF, dtype=np.float64)
    prev[0] = 0.0

    for i in range(1, T + 1):
        j_lo = max(1, i - w)
        j_hi = min(T + 1, i + w + 1)
        curr[:] = INF
        for j in range(j_lo, j_hi):
            cost = float(np.sum((x[i - 1] - y[j - 1]) ** 2))
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[T] / T


# ── Vectorised cross-correlation pre-filter ───────────────────────────────────


def _xcorr_distances(
    query: np.ndarray,
    database: np.ndarray,
    window: int,
    stride: int,
) -> np.ndarray:
    """
    Fast L2 sliding-window distance for pre-filtering (no DTW).

    Returns (n_candidates,) array of L2 distances.
    Each candidate is a contiguous window of length `window` in database.
    """
    q_norm = query - query.mean(axis=0, keepdims=True)
    starts = np.arange(0, len(database) - window + 1, stride)
    dists  = np.empty(len(starts), dtype=np.float64)
    for k, s in enumerate(starts):
        seg = database[s : s + window]
        seg_norm = seg - seg.mean(axis=0, keepdims=True)
        dists[k] = float(np.mean((q_norm - seg_norm) ** 2))
    return dists


# ── Main matching function ────────────────────────────────────────────────────


def find_best_match(
    imu_knee: np.ndarray,
    imu_thigh: np.ndarray,
    mocap_db: dict,
    stride: int = 5,
    dtw_band: int = 40,
    top_k_prefilter: int = 20,
) -> Tuple[int, float, dict]:
    """
    Find the mocap segment that best matches (imu_knee, imu_thigh).

    Parameters
    ----------
    imu_knee   : (T,) knee angle labels from IMU recording (degrees)
    imu_thigh  : (T,) thigh angle from IMU recording (degrees)
    mocap_db   : database dict from mocap_loader.load_or_generate_mocap_database()
    stride     : step size for sliding window pre-filter (samples)
    dtw_band   : Sakoe-Chiba half-bandwidth for DTW
    top_k_prefilter : number of candidates to score with full DTW

    Returns
    -------
    best_start  : int   start index in mocap_db arrays
    best_dist   : float normalised DTW distance
    matched     : dict  {joint_name: (T,) array} for the best window,
                        plus 'root_pos': (T, 3)
    """
    T = len(imu_knee)
    assert len(imu_thigh) == T

    db_len = len(mocap_db["knee_right"])
    if db_len < T:
        raise ValueError(
            f"Mocap database ({db_len} frames) is shorter than query ({T} frames). "
            "Generate more synthetic cycles or use a longer BVH recording."
        )

    # ── build normalised query matrix (T, 2): [knee, thigh] ───────────────
    query_raw = np.stack([imu_knee, imu_thigh], axis=1).astype(np.float64)
    query_n   = _zscore(query_raw)

    # ── build normalised database matrix: [knee_right, hip_right] ─────────
    db_raw = np.stack(
        [mocap_db["knee_right"], mocap_db["hip_right"]], axis=1
    ).astype(np.float64)
    db_n   = _zscore(db_raw)

    # ── stage 1: fast L2 pre-filter ───────────────────────────────────────
    starts     = np.arange(0, db_len - T + 1, stride)
    l2_dists   = _xcorr_distances(query_n, db_n, T, stride)

    # Sort and keep top-K candidates
    k = min(top_k_prefilter, len(l2_dists))
    top_idx   = np.argpartition(l2_dists, k - 1)[:k]
    top_starts = starts[top_idx]

    # ── stage 2: DTW on top-K ─────────────────────────────────────────────
    best_dist  = float("inf")
    best_start = int(top_starts[0])

    for s in top_starts:
        seg = db_n[s : s + T]
        d   = dtw_distance(query_n, seg, band=dtw_band)
        if d < best_dist:
            best_dist  = d
            best_start = int(s)

    # ── extract all joints for best segment ───────────────────────────────
    keys_1d = ["knee_right", "knee_left", "hip_right", "hip_left",
               "ankle_right", "ankle_left", "pelvis_tilt", "trunk_lean"]
    matched: dict = {}
    for k in keys_1d:
        if k in mocap_db:
            matched[k] = mocap_db[k][best_start : best_start + T].copy()
        else:
            matched[k] = np.zeros(T, dtype=np.float32)

    matched["root_pos"] = mocap_db["root_pos"][best_start : best_start + T].copy()

    return best_start, best_dist, matched


# ── Convenience: build a continuous predicted sequence ───────────────────────


def build_predicted_sequence(
    imu_knee_full: np.ndarray,
    imu_thigh_full: np.ndarray,
    mocap_db: dict,
    window: int = 400,
    hop: int = 200,
    **match_kwargs,
) -> Tuple[np.ndarray, dict]:
    """
    Stitch together motion-matched mocap segments to cover a long recording.

    Divides `imu_knee_full` / `imu_thigh_full` into overlapping `window`-length
    chunks (hop=`hop`), finds the best mocap match per chunk, and concatenates
    the matched segments.

    Returns
    -------
    match_distances : (n_chunks,) DTW distances (diagnostic)
    stitched_mocap  : dict of joint angle arrays covering the full recording length,
                      keys same as find_best_match's `matched` output
    """
    N = len(imu_knee_full)
    keys_1d = ["knee_right", "knee_left", "hip_right", "hip_left",
               "ankle_right", "ankle_left", "pelvis_tilt", "trunk_lean"]

    chunks = []
    dists  = []

    for start in range(0, N, hop):
        end = min(start + window, N)
        # Pad last chunk if needed
        knee_q  = imu_knee_full[start:end]
        thigh_q = imu_thigh_full[start:end]
        if len(knee_q) < window:
            pad = window - len(knee_q)
            knee_q  = np.concatenate([knee_q,  np.full(pad, knee_q[-1])])
            thigh_q = np.concatenate([thigh_q, np.full(pad, thigh_q[-1])])

        _, d, seg = find_best_match(knee_q, thigh_q, mocap_db, **match_kwargs)
        # Keep only the non-padded portion
        actual_len = end - start
        trimmed = {k: seg[k][:actual_len] for k in keys_1d}
        trimmed["root_pos"] = seg["root_pos"][:actual_len]
        chunks.append(trimmed)
        dists.append(d)

    # Concatenate chunks
    stitched: dict = {}
    for k in keys_1d:
        stitched[k] = np.concatenate([c[k] for c in chunks])
    stitched["root_pos"] = np.concatenate([c["root_pos"] for c in chunks], axis=0)

    return np.array(dists, dtype=np.float32), stitched
