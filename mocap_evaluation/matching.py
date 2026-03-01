"""DTW motion matching: find the MoCap Act clip that best matches a query.

Query  = (thigh_angle, knee_angle) from IMU / model prediction @ 200 Hz
Database = (hip_right, knee_right) from MoCap Act HDF5 @ 200 Hz

Two-stage matching
------------------
1. Fast L2 sliding-window pre-filter → top-K candidates
2. Full DTW (Sakoe-Chiba band) on each candidate → best match

Match quality is printed to the console and saved as a PNG plot so you can
see immediately how well the found clip aligns with your query signal.

All angles use the **included-angle convention** (180° = full extension).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Feature scaling ───────────────────────────────────────────────────────────

_KNEE_SCALE     = 70.0    # approximate range of knee included-angle (deg)
_HIP_SCALE      = 40.0    # approximate range of hip included-angle (deg)
_KNEE_VEL_SCALE = 140.0   # deg/s at 200 Hz finite-difference
_HIP_VEL_SCALE  = 90.0


def _build_features(knee: np.ndarray, hip: np.ndarray) -> np.ndarray:
    """Return scaled [knee, hip, Δknee, Δhip] feature matrix (T, 4)."""
    knee = knee.astype(np.float64)
    hip  = hip.astype(np.float64)
    dknee = np.concatenate([[0.0], np.diff(knee)])
    dhip  = np.concatenate([[0.0], np.diff(hip)])
    feats  = np.stack([knee, hip, dknee, dhip], axis=1)
    scales = np.array([[_KNEE_SCALE, _HIP_SCALE, _KNEE_VEL_SCALE, _HIP_VEL_SCALE]])
    return feats / scales


# ── DTW (Sakoe-Chiba band) ────────────────────────────────────────────────────

def _dtw(x: np.ndarray, y: np.ndarray, band: int = 40) -> float:
    """Normalised DTW distance between two equal-length sequences.

    Parameters
    ----------
    x, y  : (T,) or (T, D) arrays – must have the same number of rows.
    band  : Sakoe-Chiba half-bandwidth (samples); None = unconstrained O(T²).

    Returns
    -------
    DTW cost divided by T (scale-independent).
    """
    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]
    T  = len(x)
    w  = T if band is None else min(band, T)
    INF = 1e18
    prev = np.full(T + 1, INF)
    curr = np.full(T + 1, INF)
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


# ── L2 pre-filter ─────────────────────────────────────────────────────────────

def _l2_prefilter(query: np.ndarray, db: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Sliding-window mean-squared L2 distance between query and every db window."""
    starts = np.arange(0, len(db) - window + 1, stride)
    dists  = np.empty(len(starts), dtype=np.float64)
    for k, s in enumerate(starts):
        dists[k] = float(np.mean((query - db[s : s + window]) ** 2))
    return dists


# ── Main matching function ────────────────────────────────────────────────────

def find_match(
    knee_query:  np.ndarray,
    thigh_query: np.ndarray,
    db: dict,
    stride: int = 1,
    dtw_band: int = 40,
    prefilter_k: int = 100,
) -> Tuple[int, float, str, Dict[str, np.ndarray]]:
    """Find the database segment that best matches the query.

    Parameters
    ----------
    knee_query  : (T,) included-angle degrees from model prediction / IMU
    thigh_query : (T,) included-angle degrees from IMU
    db          : database dict from ``db.load_database()``
    stride      : sliding-window step (1 = exhaustive)
    dtw_band    : Sakoe-Chiba half-bandwidth
    prefilter_k : candidates passed to full DTW after the L2 pre-filter

    Returns
    -------
    best_start : int   start index in the flat database arrays
    dtw_dist   : float normalised DTW distance (lower = better match)
    clip_id    : str   matched clip ID (e.g. ``"CMU_012_03"``)
    segment    : dict  ``{knee_right, hip_right}`` arrays of length T
    """
    T      = len(knee_query)
    db_len = len(db["knee_right"])
    assert len(thigh_query) == T, "knee_query and thigh_query must be the same length"

    if db_len < T:
        raise ValueError(
            f"Database ({db_len} frames) is shorter than query ({T} frames). "
            "Use a shorter query or a larger database subset."
        )

    query_feat = _build_features(knee_query, thigh_query)
    db_feat    = _build_features(db["knee_right"], db["hip_right"])
    starts     = np.arange(0, db_len - T + 1, stride)

    print(f"  L2 pre-filter: {len(starts):,} windows …", end="", flush=True)
    l2 = _l2_prefilter(query_feat, db_feat, T, stride)
    k  = min(prefilter_k, len(starts))
    top_idx    = np.argpartition(l2, k - 1)[:k]
    top_starts = starts[top_idx]
    print(f" → {k} candidates")

    print(f"  DTW scoring {k} candidates …", end="", flush=True)
    best_dist  = float("inf")
    best_start = int(top_starts[0])
    for s in top_starts:
        d = _dtw(query_feat, db_feat[s : s + T], band=dtw_band)
        if d < best_dist:
            best_dist  = d
            best_start = int(s)
    print(f" done  DTW={best_dist:.4f}")

    # Identify which clip this start index belongs to
    clip_id = "?"
    for start, end, cid, _ in db["file_boundaries"]:
        if int(start) <= best_start < int(end):
            clip_id = str(cid)
            break

    segment = {
        "knee_right": db["knee_right"][best_start : best_start + T].copy(),
        "hip_right":  db["hip_right"][best_start : best_start + T].copy(),
    }

    match_rmse = float(np.sqrt(np.mean((segment["knee_right"] - knee_query) ** 2)))
    print(f"  Best match : {clip_id}  (db frame {best_start})  "
          f"RMSE={match_rmse:.2f}°")

    return best_start, best_dist, clip_id, segment


# ── Match quality plot ────────────────────────────────────────────────────────

def plot_match_quality(
    knee_query:  np.ndarray,
    thigh_query: np.ndarray,
    matched_knee: np.ndarray,
    matched_hip:  np.ndarray,
    dtw_dist: float,
    clip_id: str,
    fps: float,
    out_path: str = "match_quality.png",
) -> None:
    """Save a 2-panel plot showing how well the matched clip aligns with the query.

    Panel 1 – Knee angle: query vs matched clip (lower RMSE = better)
    Panel 2 – Hip / Thigh angle: query vs matched clip
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T   = min(len(knee_query), len(matched_knee))
    t   = np.arange(T) / fps
    rmse_knee = float(np.sqrt(np.mean((matched_knee[:T] - knee_query[:T]) ** 2)))
    rmse_hip  = float(np.sqrt(np.mean((matched_hip[:T]  - thigh_query[:T]) ** 2)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax1.plot(t, knee_query[:T],   lw=2,   color="steelblue", label="Query (prediction/IMU)")
    ax1.plot(t, matched_knee[:T], lw=1.5, color="darkorange", ls="--", label=f"Matched clip  RMSE={rmse_knee:.1f}°")
    ax1.set_ylabel("Knee included-angle (°)")
    ax1.set_title(
        f"Match quality — clip: {clip_id}   DTW={dtw_dist:.4f}   "
        f"Knee RMSE={rmse_knee:.1f}°",
        fontsize=10,
    )
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 200)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, thigh_query[:T],  lw=2,   color="steelblue", label="Query thigh (IMU)")
    ax2.plot(t, matched_hip[:T],  lw=1.5, color="darkorange", ls="--", label=f"Matched hip  RMSE={rmse_hip:.1f}°")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Hip / Thigh included-angle (°)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 200)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Match plot saved → {out_path}")


# ── Clip boundary lookup ──────────────────────────────────────────────────────

def clip_info_for_start(best_start: int, query_len: int, db: dict) -> dict:
    """Return clip metadata for a given best_start index.

    Returns
    -------
    dict with keys: clip_id, db_start, db_end, frame_offset,
                    time_offset_s, time_duration_s, category
    """
    fps = float(db["fps"])
    for seg_start, seg_end, clip_id, category in db["file_boundaries"]:
        seg_start, seg_end = int(seg_start), int(seg_end)
        if seg_start <= best_start < seg_end:
            frame_offset    = best_start - seg_start
            time_offset_s   = frame_offset / fps
            time_duration_s = query_len   / fps
            return {
                "clip_id":         str(clip_id),
                "db_start":        seg_start,
                "db_end":          seg_end,
                "frame_offset":    frame_offset,
                "time_offset_s":   time_offset_s,
                "time_duration_s": time_duration_s,
                "category":        str(category),
            }
    return {
        "clip_id":         "?",
        "db_start":        0,
        "db_end":          0,
        "frame_offset":    0,
        "time_offset_s":   0.0,
        "time_duration_s": query_len / fps,
        "category":        "unknown",
    }
