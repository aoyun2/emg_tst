from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, resample


@dataclass
class MatchResult:
    snippet_id: str
    clip_id: str
    score: float
    start_idx: int
    end_idx: int


def _resample_to(x: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    if abs(src_hz - dst_hz) < 1e-6:
        return x
    n_new = max(2, int(round(len(x) * dst_hz / src_hz)))
    return resample(x, n_new).astype(np.float32)


def build_feature(
    thigh_deg: np.ndarray,
    knee_deg: np.ndarray | None,
    *,
    mode: str = "thigh_knee",
) -> np.ndarray:
    """Build per-timestep features used for matching.

    mode:
      - "thigh": uses [thigh, d_thigh]
      - "thigh_knee": uses [thigh, knee, d_thigh, d_knee]
    """
    thigh = np.asarray(thigh_deg, dtype=np.float32).reshape(-1)
    d_thigh = np.gradient(thigh)

    m = mode.lower().strip()
    if m == "thigh":
        return np.stack([thigh, d_thigh], axis=1)
    if m == "thigh_knee":
        if knee_deg is None:
            raise ValueError("knee_deg is required for mode='thigh_knee'")
        knee = np.asarray(knee_deg, dtype=np.float32).reshape(-1)
        d_knee = np.gradient(knee)
        return np.stack([thigh, knee, d_thigh, d_knee], axis=1)
    raise ValueError("mode must be 'thigh' or 'thigh_knee'")


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = 40) -> float:
    na, nb = len(a), len(b)
    inf = float("inf")
    dp = np.full((na + 1, nb + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, na + 1):
        j0 = max(1, i - band)
        j1 = min(nb, i + band)
        for j in range(j0, j1 + 1):
            c = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = c + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[na, nb] / (na + nb))


def _sliding_sse_1d(signal: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute SSE for all windows of `signal` against `query` (same length), using FFT dot products."""
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    L = int(q.shape[0])
    if L < 1 or s.shape[0] < L:
        return np.asarray([], dtype=np.float64)

    # sum((s_w - q)^2) = sum(s_w^2) + sum(q^2) - 2*dot(s_w, q)
    dot = fftconvolve(s, q[::-1], mode="valid")
    dot = np.real(dot)

    s2 = s * s
    cs = np.cumsum(np.concatenate(([0.0], s2)))
    sum_s2 = cs[L:] - cs[:-L]
    sum_q2 = float(np.sum(q * q))
    return sum_s2 + sum_q2 - 2.0 * dot


def _sliding_sse(signal_feat: np.ndarray, query_feat: np.ndarray) -> np.ndarray:
    s = np.asarray(signal_feat, dtype=np.float64)
    q = np.asarray(query_feat, dtype=np.float64)
    if s.ndim != 2 or q.ndim != 2 or s.shape[1] != q.shape[1]:
        raise ValueError("signal_feat and query_feat must be 2D with the same feature dimension.")

    scores: np.ndarray | None = None
    for d in range(int(s.shape[1])):
        sse_d = _sliding_sse_1d(s[:, d], q[:, d])
        if scores is None:
            scores = sse_d
        else:
            scores = scores + sse_d
    return np.asarray(scores if scores is not None else [], dtype=np.float64)


def match_batch_to_snippets(
    query_thigh: np.ndarray,
    query_hz: float,
    snippet_bank: list,
    *,
    query_knee_pred: np.ndarray | None = None,
    feature_mode: str = "thigh_knee",
    top_k: int = 5,
    dtw_band: int = 40,
) -> list[MatchResult]:
    q_t = np.asarray(query_thigh, dtype=np.float32)
    q_k = None if query_knee_pred is None else np.asarray(query_knee_pred, dtype=np.float32)
    matches: list[MatchResult] = []

    # Resample query into each snippet's rate (typically ~33Hz) to reduce compute.
    for snip in snippet_bank:
        s_thigh = np.asarray(snip.thigh_angle_deg, dtype=np.float32).reshape(-1)
        s_knee = np.asarray(snip.knee_angle_deg, dtype=np.float32).reshape(-1)

        q_thigh_rs = _resample_to(q_t, query_hz, float(snip.sample_hz))
        q_knee_rs = None if q_k is None else _resample_to(q_k, query_hz, float(snip.sample_hz))

        Lq = len(q_thigh_rs)
        if len(s_thigh) < Lq or (q_knee_rs is not None and len(s_knee) < Lq):
            continue

        qf = build_feature(q_thigh_rs, q_knee_rs, mode=feature_mode)
        sf_full = build_feature(s_thigh, s_knee if q_knee_rs is not None else None, mode=feature_mode)

        # Stage 1: fast L2 window search (no warping) to propose a small set of candidates.
        sse = _sliding_sse(sf_full, qf)
        if not sse.size:
            continue

        n_candidates = int(min(8, sse.size))
        cand = np.argpartition(sse, n_candidates - 1)[:n_candidates]
        cand = cand[np.argsort(sse[cand])]

        # Stage 2: DTW refine around those candidates.
        best_score = float("inf")
        best_window = (0, Lq)
        for start in cand.tolist():
            start_i = int(start)
            end_i = start_i + Lq
            sf = sf_full[start_i:end_i]
            score = dtw_distance(qf, sf, band=dtw_band)
            if score < best_score:
                best_score = score
                best_window = (start_i, end_i)

        matches.append(
            MatchResult(
                snippet_id=snip.snippet_id,
                clip_id=getattr(snip, "clip_id", ""),
                score=float(best_score),
                start_idx=int(best_window[0]),
                end_idx=int(best_window[1]),
            )
        )

    matches.sort(key=lambda m: m.score)
    return matches[:top_k]
