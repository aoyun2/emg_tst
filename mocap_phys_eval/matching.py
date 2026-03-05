from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .utils import (
    quat_align_constant_offset_wxyz,
    quat_conj_wxyz,
    quat_geodesic_deg_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
)


@dataclass(frozen=True)
class MatchCandidate:
    snippet_id: str
    clip_id: str
    start_step: int
    end_step: int
    coarse_score: float
    score: float
    rmse_thigh_deg: float
    rmse_knee_deg: float
    thigh_quat_offset_wxyz: tuple[float, float, float, float]
    thigh_sign: float
    knee_sign: float
    thigh_offset_deg: float
    knee_offset_deg: float


def _deriv(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return np.zeros_like(x)
    return np.diff(x, prepend=x[0])


def _sliding_sse_1d(c: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Return SSE for all valid windows: sum((c[i:i+L]-q)^2)."""
    c = np.asarray(c, dtype=np.float64).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    L = int(q.size)
    if c.size < L or L < 1:
        return np.zeros((0,), dtype=np.float64)
    ones = np.ones((L,), dtype=np.float64)
    q_rev = q[::-1]
    q_sq_sum = float(np.sum(q * q))
    # np.convolve is O(T*L) and implemented in C; L is small (~34) so it's fine.
    sum_c_sq = np.convolve(c * c, ones, mode="valid")
    sum_cq = np.convolve(c, q_rev, mode="valid")
    return (sum_c_sq - 2.0 * sum_cq + q_sq_sum).astype(np.float64)


def _quat_step_angles_deg(q: np.ndarray) -> np.ndarray:
    """Per-step rotation angle magnitude (deg) from a quaternion timeseries.

    This is invariant to left-multiplication by a constant quaternion (global frame
    offset), which is exactly what we need for coarse motion matching with IMUs.

    Returns an array of shape (T-1,) with angles in [0, 180].
    """
    q = quat_normalize_wxyz(np.asarray(q, dtype=np.float64).reshape(-1, 4))
    if int(q.shape[0]) < 2:
        return np.zeros((0,), dtype=np.float64)
    dq = quat_mul_wxyz(quat_conj_wxyz(q[:-1]), q[1:])
    w = np.clip(np.abs(dq[:, 0]), 0.0, 1.0)
    ang = 2.0 * np.arccos(w)
    return np.degrees(ang).astype(np.float64)


def _coarse_match_candidates(
    *,
    bank_thigh_pitch: np.ndarray,
    bank_knee: np.ndarray,
    query_thigh: np.ndarray,
    query_knee: np.ndarray,
    top_k: int,
    per_clip: int = 3,
) -> list[tuple[float, int, int]]:
    """Return list of (coarse_score, clip_i, start_step) candidates."""
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    qd_th = _deriv(query_thigh)
    qd_kn = _deriv(query_knee)

    heap: list[tuple[float, int, int]] = []
    want = int(max(1, top_k))
    per_clip = int(max(1, per_clip))

    clip_range = range(int(bank_thigh_pitch.shape[0]))
    if tqdm is not None:
        clip_range = tqdm(clip_range, desc="Motion match (coarse scan)", unit="clip", leave=False)
    for clip_i in clip_range:
        th = np.asarray(bank_thigh_pitch[clip_i], dtype=np.float64).reshape(-1)
        kn = np.asarray(bank_knee[clip_i], dtype=np.float64).reshape(-1)
        if th.size < qd_th.size or kn.size < qd_kn.size:
            continue
        sse = _sliding_sse_1d(_deriv(th), qd_th) + _sliding_sse_1d(_deriv(kn), qd_kn)
        if sse.size < 1:
            continue
        k = min(per_clip, int(sse.size))
        # Get indices of the k smallest SSEs in this clip.
        idxs = np.argpartition(sse, kth=(k - 1))[:k]
        for start in idxs.tolist():
            score = float(sse[int(start)])
            heapq.heappush(heap, (score, int(clip_i), int(start)))

    # Sort globally, keep top_k.
    heap.sort(key=lambda t: t[0])
    return heap[:want]


def _coarse_match_candidates_quat(
    *,
    bank_thigh_quat: np.ndarray,
    bank_knee: np.ndarray,
    query_thigh_quat_wxyz: np.ndarray,
    query_knee: np.ndarray,
    top_k: int,
    per_clip: int = 3,
) -> list[tuple[float, int, int]]:
    """Coarse scan using quaternion-derived thigh step angles + knee derivative."""
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    q_th = _quat_step_angles_deg(query_thigh_quat_wxyz)
    q_kn = np.asarray(query_knee, dtype=np.float64).reshape(-1)
    if q_th.size < 2 or q_kn.size < 2:
        return []

    qd_th = q_th  # already per-step
    qd_kn = np.diff(q_kn).astype(np.float64)
    Ld = int(min(qd_th.size, qd_kn.size))
    qd_th = qd_th[:Ld]
    qd_kn = qd_kn[:Ld]

    heap: list[tuple[float, int, int]] = []
    want = int(max(1, top_k))
    per_clip = int(max(1, per_clip))

    clip_range = range(int(bank_knee.shape[0]))
    if tqdm is not None:
        clip_range = tqdm(clip_range, desc="Motion match (coarse scan)", unit="clip", leave=False)

    for clip_i in clip_range:
        thq_full = np.asarray(bank_thigh_quat[int(clip_i)], dtype=np.float64).reshape(-1, 4)
        kn_full = np.asarray(bank_knee[int(clip_i)], dtype=np.float64).reshape(-1)
        if thq_full.shape[0] < (Ld + 1) or kn_full.size < (Ld + 1):
            continue
        th_step = _quat_step_angles_deg(thq_full)  # (T-1,)
        kn_step = np.diff(kn_full).astype(np.float64)  # (T-1,)
        Td = int(min(th_step.size, kn_step.size))
        if Td < Ld:
            continue
        th_step = th_step[:Td]
        kn_step = kn_step[:Td]
        sse = _sliding_sse_1d(th_step, qd_th) + _sliding_sse_1d(kn_step, qd_kn)
        if sse.size < 1:
            continue
        k = min(per_clip, int(sse.size))
        idxs = np.argpartition(sse, kth=(k - 1))[:k]
        for start in idxs.tolist():
            score = float(sse[int(start)])
            heapq.heappush(heap, (score, int(clip_i), int(start)))

    heap.sort(key=lambda t: t[0])
    return heap[:want]


def motion_match_one_window(
    *,
    bank,
    query_thigh_deg: np.ndarray,
    query_thigh_quat_wxyz: np.ndarray | None,
    query_knee_deg: np.ndarray,
    top_k: int = 12,
    feature_mode: str = "thigh_knee_d",
) -> list[MatchCandidate]:
    """Motion-match a single query window to the CMU2020 bank.

    Args:
      bank: ClipBank (from reference_bank.py)
      query_thigh_deg: (L,) query thigh segment pitch (world) in degrees
      query_thigh_quat_wxyz: (L,4) query thigh orientation in wxyz (root-relative); used for 3D thigh error metric
      query_knee_deg: (L,) query knee flexion in degrees
      feature_mode: currently only supports 'thigh_knee_d' (derivative coarse search)

    Notes:
      - When query_thigh_quat_wxyz is provided and bank.thigh_quat_wxyz exists, the
        reported "rmse_thigh_deg" is the RMS of the quaternion geodesic angle (deg),
        not an RMSE of a single scalar thigh angle.
    """
    feature_mode = str(feature_mode).strip().lower()
    if feature_mode not in {"thigh_knee_d", "quat_knee_d"}:
        raise ValueError("feature_mode must be 'thigh_knee_d' or 'quat_knee_d'.")

    q_th = np.asarray(query_thigh_deg, dtype=np.float64).reshape(-1)
    q_thq = None
    if query_thigh_quat_wxyz is not None:
        q_thq = quat_normalize_wxyz(np.asarray(query_thigh_quat_wxyz, dtype=np.float64))
    q_kn = np.asarray(query_knee_deg, dtype=np.float64).reshape(-1)
    if q_th.size < 2 or q_kn.size < 2:
        raise RuntimeError("Query window too short for motion matching.")
    L = int(min(q_th.size, q_kn.size))
    q_th = q_th[:L]
    if q_thq is not None:
        q_thq = q_thq[:L]
    q_kn = q_kn[:L]

    if feature_mode == "quat_knee_d":
        if q_thq is None:
            raise RuntimeError("feature_mode='quat_knee_d' requires query_thigh_quat_wxyz.")
        # Prefer anatomical thigh quaternions if available.
        bank_thq = getattr(bank, "thigh_anat_quat_wxyz", None)
        if bank_thq is None:
            bank_thq = getattr(bank, "thigh_quat_wxyz", None)
        if bank_thq is None:
            raise RuntimeError("Bank does not provide thigh quaternions for quaternion motion matching.")
        coarse = _coarse_match_candidates_quat(
            bank_thigh_quat=bank_thq,
            bank_knee=bank.knee_deg,
            query_thigh_quat_wxyz=q_thq,
            query_knee=q_kn,
            top_k=int(top_k),
            per_clip=3,
        )
    else:
        coarse = _coarse_match_candidates(
            bank_thigh_pitch=bank.thigh_pitch_deg,
            bank_knee=bank.knee_deg,
            query_thigh=q_th,
            query_knee=q_kn,
            top_k=int(top_k),
            per_clip=3,
        )

    refined: list[MatchCandidate] = []
    for coarse_score, clip_i, start in coarse:
        th_full = np.asarray(bank.thigh_pitch_deg[int(clip_i)], dtype=np.float64).reshape(-1)
        kn_full = np.asarray(bank.knee_deg[int(clip_i)], dtype=np.float64).reshape(-1)
        if start < 0 or (start + L) > th_full.size or (start + L) > kn_full.size:
            continue

        ref_th = th_full[start : start + L]
        ref_kn = kn_full[start : start + L]

        # Knee offset (degrees). Knee sign should not differ.
        knee_sign = 1.0
        knee_off = float(np.mean(ref_kn - knee_sign * q_kn))
        q_kn_al = knee_sign * q_kn + knee_off
        rmse_kn = float(np.sqrt(float(np.mean((q_kn_al - ref_kn) ** 2))))

        # Thigh pitch alignment is still computed for plotting/debugging, but the
        # *score* uses 3D thigh orientation error when quaternions are available.
        best_pitch = None
        for thigh_sign in (+1.0, -1.0):
            thigh_off = float(np.mean(ref_th - thigh_sign * q_th))
            q_th_al = thigh_sign * q_th + thigh_off
            rmse_th_pitch = float(np.sqrt(float(np.mean((q_th_al - ref_th) ** 2))))
            cand = (rmse_th_pitch, thigh_sign, thigh_off)
            if best_pitch is None or cand[0] < best_pitch[0]:
                best_pitch = cand
        if best_pitch is None:
            continue
        _, thigh_sign, thigh_off = best_pitch

        qoff = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        rmse_th = float("nan")
        if q_thq is not None:
            ref_bank_quat = None
            if hasattr(bank, "thigh_anat_quat_wxyz"):
                ref_bank_quat = getattr(bank, "thigh_anat_quat_wxyz")
            elif hasattr(bank, "thigh_quat_wxyz"):
                ref_bank_quat = getattr(bank, "thigh_quat_wxyz")
            if ref_bank_quat is not None:
                ref_thq_full = np.asarray(ref_bank_quat[int(clip_i)], dtype=np.float64)
            if ref_thq_full.ndim == 2 and ref_thq_full.shape[1] == 4 and (start + L) <= int(ref_thq_full.shape[0]):
                ref_thq = quat_normalize_wxyz(ref_thq_full[start : start + L])
                qoff, q_thq_al = quat_align_constant_offset_wxyz(ref_thq, q_thq)
                err_deg = quat_geodesic_deg_wxyz(ref_thq, q_thq_al)
                rmse_th = float(np.sqrt(float(np.mean(err_deg**2))))

        # Fallback: if quaternion path unavailable, use pitch RMSE.
        if not np.isfinite(rmse_th):
            q_th_al = float(thigh_sign) * q_th + float(thigh_off)
            rmse_th = float(np.sqrt(float(np.mean((q_th_al - ref_th) ** 2))))

        score = rmse_th + rmse_kn

        clip_id = str(bank.clip_id[int(clip_i)])
        start_step = int(start)
        end_step = int(start_step + L - 1)
        try:
            snippet_id = str(bank.snippet_id[int(clip_i)])
        except Exception:
            snippet_id = f"{clip_id}-0-{int(th_full.size) - 1}"

        refined.append(
            MatchCandidate(
                snippet_id=snippet_id,
                clip_id=clip_id,
                start_step=int(start_step),
                end_step=int(end_step),
                coarse_score=float(coarse_score),
                score=float(score),
                rmse_thigh_deg=float(rmse_th),
                rmse_knee_deg=float(rmse_kn),
                thigh_quat_offset_wxyz=(float(qoff[0]), float(qoff[1]), float(qoff[2]), float(qoff[3])),
                thigh_sign=float(thigh_sign),
                knee_sign=float(knee_sign),
                thigh_offset_deg=float(thigh_off),
                knee_offset_deg=float(knee_off),
            )
        )

    refined.sort(key=lambda c: c.score)
    return refined[: int(max(1, top_k))]
