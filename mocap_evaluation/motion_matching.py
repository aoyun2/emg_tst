"""
Motion matching: find the mocap segment that best matches our IMU recording.

Pipeline
--------
1. Query  = z-score-normalised (knee_angle, thigh_angle) from our device at 200 Hz.
2. Database = (knee_right, hip_right) from mocap at 200 Hz.
3. Slide over database with step `stride`, compute DTW distance to query.
4. Return best-matching start index → extract ALL joint angles for that window.

DTW implementation uses a Sakoe-Chiba band for O(n · W) cost instead of O(n²).

Category-aware matching
-----------------------
When the mocap database contains ``file_boundaries`` metadata (produced by
:func:`mocap_loader.load_full_cmu_database`), matching can be restricted to
specific motion categories (e.g. only "walk" or only "run") and the returned
result includes the matched category.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── Normalisation ─────────────────────────────────────────────────────────────

# Fixed channel scales for matching.  These are typical physiological ranges
# (degrees) used to make the knee and hip channels comparable without removing
# absolute angle information (which z-scoring would do).
_KNEE_SCALE = 70.0   # typical walking knee flexion range (0–70°)
_HIP_SCALE  = 40.0   # typical walking hip flexion range (-10° to 30°)


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score a 1-D or 2-D array column-wise."""
    if x.ndim == 1:
        m, s = x.mean(), x.std()
        return (x - m) / max(s, eps)
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    s = np.where(s < eps, eps, s)
    return (x - m) / s


def _range_scale(x: np.ndarray) -> np.ndarray:
    """
    Scale a (N, 2) matrix where col 0 = knee and col 1 = hip by fixed
    physiological ranges.  Unlike z-scoring, this preserves absolute angle
    values — segments with the right amplitude match better.
    """
    scales = np.array([[_KNEE_SCALE, _HIP_SCALE]])
    return x / scales


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

    Uses direct L2 on the (already range-scaled) input — no per-window
    mean subtraction, so candidates with the right absolute angle values
    are preferred over shape-only matches.
    """
    starts = np.arange(0, len(database) - window + 1, stride)
    dists  = np.empty(len(starts), dtype=np.float64)
    for k, s in enumerate(starts):
        seg = database[s : s + window]
        dists[k] = float(np.mean((query - seg) ** 2))
    return dists


# ── Main matching function ────────────────────────────────────────────────────


def _valid_start_mask(
    db_len: int,
    window: int,
    stride: int,
    mocap_db: dict,
    categories: Optional[Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (starts, mask) where mask[i] is True if starts[i] falls entirely
    within a segment of one of the requested categories.
    """
    starts = np.arange(0, db_len - window + 1, stride)
    if categories is None or "file_boundaries" not in mocap_db:
        return starts, np.ones(len(starts), dtype=bool)

    cat_set = set(categories)
    boundaries = mocap_db["file_boundaries"]
    mask = np.zeros(len(starts), dtype=bool)

    for seg_start, seg_end, _fname, cat in boundaries:
        if cat not in cat_set:
            continue
        # A candidate window [s, s+window) is valid if it fits in this segment
        lo = max(0, np.searchsorted(starts, seg_start))
        hi = np.searchsorted(starts, seg_end - window, side="right")
        mask[lo:hi] = True

    return starts, mask


def find_best_match(
    imu_knee: np.ndarray,
    imu_thigh: np.ndarray,
    mocap_db: dict,
    stride: int = 1,
    dtw_band: int = 40,
    top_k_prefilter: int = 50,
    categories: Optional[Sequence[str]] = None,
) -> Tuple[int, float, dict]:
    """
    Find the mocap segment that best matches (imu_knee, imu_thigh).

    Parameters
    ----------
    imu_knee   : (T,) knee angle labels from IMU recording (degrees)
    imu_thigh  : (T,) thigh angle from IMU recording (degrees)
    mocap_db   : database dict from mocap_loader
    stride     : step size for sliding window pre-filter (samples)
    dtw_band   : Sakoe-Chiba half-bandwidth for DTW
    top_k_prefilter : number of candidates to score with full DTW
    categories : restrict matching to these motion categories (None = all).
                 Only effective when mocap_db has ``file_boundaries`` metadata.

    Returns
    -------
    best_start  : int   start index in mocap_db arrays
    best_dist   : float normalised DTW distance
    matched     : dict  {joint_name: (T,) array} for the best window,
                        plus 'root_pos': (T, 3) and 'category': str
    """
    T = len(imu_knee)
    assert len(imu_thigh) == T

    db_len = len(mocap_db["knee_right"])
    if db_len < T:
        raise ValueError(
            f"Mocap database ({db_len} frames) is shorter than query ({T} frames). "
            "Generate more synthetic cycles or use a longer BVH recording."
        )

    # ── build scaled query matrix (T, 2): [knee, thigh] ────────────────────
    # Range-scaling (not z-score) so absolute angle values are preserved —
    # walking segments match walking queries better than sport/dance.
    query_raw = np.stack([imu_knee, imu_thigh], axis=1).astype(np.float64)
    query_n   = _range_scale(query_raw)

    # ── build scaled database matrix: [knee_right, hip_right] ─────────────
    db_raw = np.stack(
        [mocap_db["knee_right"], mocap_db["hip_right"]], axis=1
    ).astype(np.float64)
    db_n   = _range_scale(db_raw)

    # ── category-aware candidate generation ───────────────────────────────
    starts, valid = _valid_start_mask(db_len, T, stride, mocap_db, categories)
    valid_starts = starts[valid]

    if len(valid_starts) == 0:
        # Fall back to all starts if no valid ones for the requested category
        valid_starts = starts
        valid = np.ones(len(starts), dtype=bool)

    # ── stage 1: fast L2 pre-filter ───────────────────────────────────────
    l2_dists = _xcorr_distances(query_n, db_n, T, stride)
    # Mask out invalid starts with inf
    l2_masked = l2_dists.copy()
    l2_masked[~valid] = float("inf")

    # Sort and keep top-K candidates
    k = min(top_k_prefilter, int(valid.sum()))
    if k == 0:
        k = min(top_k_prefilter, len(l2_dists))
        l2_masked = l2_dists  # fall back to unmasked

    top_idx    = np.argpartition(l2_masked, k - 1)[:k]
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

    # ── attach matched category ───────────────────────────────────────────
    if "categories" in mocap_db:
        matched["category"] = str(mocap_db["categories"][best_start])
    else:
        matched["category"] = "unknown"

    return best_start, best_dist, matched


# ── Top-K matching ───────────────────────────────────────────────────────────


def find_top_k_matches(
    imu_knee: np.ndarray,
    imu_thigh: np.ndarray,
    mocap_db: dict,
    k: int = 3,
    min_separation: Optional[int] = None,
    stride: int = 1,
    dtw_band: int = 40,
    top_k_prefilter: int = 150,
    categories: Optional[Sequence[str]] = None,
) -> List[Tuple[int, float, dict]]:
    """
    Find the top-K distinct mocap segments that best match (imu_knee, imu_thigh).

    Unlike ``find_best_match``, this returns multiple candidates so that the
    caller can run a separate simulation for each.  The query is **not**
    restricted to any motion category by default — any segment in the database
    is eligible.

    Parameters
    ----------
    imu_knee        : (T,) knee angle labels from IMU recording (degrees)
    imu_thigh       : (T,) thigh angle from IMU recording (degrees)
    mocap_db        : database dict from mocap_loader
    k               : number of results to return
    min_separation  : minimum gap between start indices of any two returned
                      matches (prevents near-duplicate windows).
                      Defaults to ``len(imu_knee) // 2``.
    stride          : step size for sliding-window pre-filter
    dtw_band        : Sakoe-Chiba half-bandwidth for DTW
    top_k_prefilter : candidates to score with full DTW (set higher than k)
    categories      : restrict to specific motion categories (None = all)

    Returns
    -------
    List of (start_index, dtw_distance, matched_dict) tuples, sorted by
    ascending DTW distance (best match first).
    """
    T = len(imu_knee)
    assert len(imu_thigh) == T

    if min_separation is None:
        min_separation = max(1, T // 2)

    db_len = len(mocap_db["knee_right"])
    if db_len < T:
        raise ValueError(
            f"Mocap database ({db_len} frames) shorter than query ({T} frames)."
        )

    # Build scaled query and database matrices
    query_raw = np.stack([imu_knee, imu_thigh], axis=1).astype(np.float64)
    query_n   = _range_scale(query_raw)

    db_raw = np.stack(
        [mocap_db["knee_right"], mocap_db["hip_right"]], axis=1
    ).astype(np.float64)
    db_n = _range_scale(db_raw)

    # Category-aware candidate generation
    starts, valid = _valid_start_mask(db_len, T, stride, mocap_db, categories)

    l2_dists = _xcorr_distances(query_n, db_n, T, stride)
    l2_masked = l2_dists.copy()
    l2_masked[~valid] = float("inf")

    # Pre-filter: keep enough candidates (at least k × 4 so we can deduplicate)
    n_prefilter = min(max(top_k_prefilter, k * 4), int(valid.sum()) or len(starts))
    top_idx = np.argpartition(l2_masked, min(n_prefilter - 1, len(l2_masked) - 1))[
        :n_prefilter
    ]
    top_starts = starts[top_idx]

    # Full DTW on all pre-filtered candidates
    dtw_scores: List[Tuple[float, int]] = []
    for s in top_starts:
        seg = db_n[s : s + T]
        d   = dtw_distance(query_n, seg, band=dtw_band)
        dtw_scores.append((d, int(s)))

    # Sort by DTW distance (best first) and de-duplicate by min_separation
    dtw_scores.sort(key=lambda x: x[0])

    selected: List[Tuple[int, float, dict]] = []
    chosen_starts: List[int] = []

    keys_1d = ["knee_right", "knee_left", "hip_right", "hip_left",
               "ankle_right", "ankle_left", "pelvis_tilt", "trunk_lean"]

    for dist, s in dtw_scores:
        if len(selected) >= k:
            break
        # Reject if too close to an already-chosen window
        if any(abs(s - cs) < min_separation for cs in chosen_starts):
            continue
        chosen_starts.append(s)

        matched: dict = {}
        for key in keys_1d:
            if key in mocap_db:
                matched[key] = mocap_db[key][s : s + T].copy()
            else:
                matched[key] = np.zeros(T, dtype=np.float32)
        matched["root_pos"] = mocap_db["root_pos"][s : s + T].copy()
        if "categories" in mocap_db:
            matched["category"] = str(mocap_db["categories"][s])
        else:
            matched["category"] = "unknown"

        selected.append((s, dist, matched))

    return selected


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
