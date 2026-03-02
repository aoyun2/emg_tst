from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import resample


@dataclass
class MatchResult:
    snippet_id: str
    score: float
    start_idx: int
    end_idx: int


def _resample_to(x: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    if abs(src_hz - dst_hz) < 1e-6:
        return x
    n_new = max(2, int(round(len(x) * dst_hz / src_hz)))
    return resample(x, n_new).astype(np.float32)


def build_feature(thigh_deg: np.ndarray, knee_deg: np.ndarray) -> np.ndarray:
    thigh = np.asarray(thigh_deg, dtype=np.float32).reshape(-1)
    knee = np.asarray(knee_deg, dtype=np.float32).reshape(-1)
    d_thigh = np.gradient(thigh)
    d_knee = np.gradient(knee)
    return np.stack([thigh, knee, d_thigh, d_knee], axis=1)


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


def match_batch_to_snippets(
    query_thigh: np.ndarray,
    query_knee_pred: np.ndarray,
    query_hz: float,
    snippet_bank: list,
    *,
    top_k: int = 5,
    dtw_band: int = 40,
) -> list[MatchResult]:
    q_t = np.asarray(query_thigh, dtype=np.float32)
    q_k = np.asarray(query_knee_pred, dtype=np.float32)
    matches: list[MatchResult] = []

    for snip in snippet_bank:
        s_thigh = _resample_to(np.asarray(snip.thigh_angle_deg), snip.sample_hz, query_hz)
        s_knee = _resample_to(np.asarray(snip.knee_angle_deg), snip.sample_hz, query_hz)

        Lq = len(q_t)
        if len(s_thigh) < Lq:
            continue

        qf = build_feature(q_t, q_k)

        best_score = float("inf")
        best_window = (0, Lq)
        # coarse sliding window for speed
        step = max(1, Lq // 8)
        for start in range(0, len(s_thigh) - Lq + 1, step):
            end = start + Lq
            sf = build_feature(s_thigh[start:end], s_knee[start:end])
            score = dtw_distance(qf, sf, band=dtw_band)
            if score < best_score:
                best_score = score
                best_window = (start, end)

        matches.append(
            MatchResult(
                snippet_id=snip.snippet_id,
                score=float(best_score),
                start_idx=int(best_window[0]),
                end_idx=int(best_window[1]),
            )
        )

    matches.sort(key=lambda m: m.score)
    return matches[:top_k]
