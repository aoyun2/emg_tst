from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --- Raw EMG input extraction ---
# RAW_WINDOW is in *raw EMG samples* (uMyo raw stream is ~400Hz).
# Each 200 Hz timestep sees the last RAW_WINDOW raw samples from each sensor.
RAW_WINDOW = 32
# Legacy-only: older datasets used 5 time-domain features + 8 FFT bands per sensor.
N_FFT_BANDS = 8
ENGINEERED_EMG_FEATURE_NAMES = ["RMS", "MAV", "WL", "ZC", "SSC"] + [f"FFT{i}" for i in range(N_FFT_BANDS)]
RAW_SNIPPET_FEATURE_MODE = "raw_snippets"
EMG_ENVELOPE_FEATURE_MODE = "causal_envelope"

# Recordings are not perfectly uniform in time (Bluetooth + serial jitter). For
# reproducible training/evaluation (and consistent window length in seconds),
# we resample each recording to a fixed uniform rate using its timestamps.
TARGET_HZ = 200.0
GT_PAPER_HZ = 100.0


def _ensure_strictly_increasing(t: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Return a strictly increasing copy of timestamps (seconds)."""
    tt = np.asarray(t, dtype=np.float64).reshape(-1).copy()
    if tt.size < 2:
        return tt
    # rigtest can emit repeated timestamps when draining multiple packets at once;
    # `np.interp` expects a strictly increasing xp array.
    if not np.isfinite(tt[0]):
        tt[0] = 0.0
    for i in range(1, int(tt.size)):
        if not np.isfinite(tt[i]):
            tt[i] = tt[i - 1] + float(eps)
        if tt[i] <= tt[i - 1]:
            tt[i] = tt[i - 1] + float(eps)
    return tt


def _resample_linear_by_timestamps(x: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear resample for 1D or 2D arrays along axis=0."""
    xs = np.asarray(x, dtype=np.float64)
    ts = np.asarray(t_src, dtype=np.float64).reshape(-1)
    td = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    if xs.ndim == 1:
        return np.interp(td, ts, xs).astype(np.float32)
    if xs.ndim != 2:
        raise ValueError(f"expected 1D/2D array, got shape={xs.shape}")
    T, F = int(xs.shape[0]), int(xs.shape[1])
    if T != int(ts.size):
        raise ValueError(f"timestamp length mismatch: x has T={T}, t_src has {ts.size}")
    out = np.empty((int(td.size), F), dtype=np.float32)
    for j in range(F):
        out[:, j] = np.interp(td, ts, xs[:, j])
    return out


def _quat_fix_sign_continuity_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    bad = n.reshape(-1) < 1e-8
    if np.any(bad):
        q[bad] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(n, 1e-8, None)
    for i in range(1, int(q.shape[0])):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    return q


def _quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float64).reshape(-1, 4).copy()
    qq[:, 1:] *= -1.0
    return qq


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    bb = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if aa.shape[0] != bb.shape[0]:
        raise ValueError(f"Quaternion length mismatch: {aa.shape} vs {bb.shape}")
    aw, ax, ay, az = aa[:, 0], aa[:, 1], aa[:, 2], aa[:, 3]
    bw, bx, by, bz = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
    out = np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )
    return _quat_fix_sign_continuity_wxyz(out)


def _quat_relative_to_first_wxyz(q: np.ndarray) -> np.ndarray:
    qq = _quat_fix_sign_continuity_wxyz(q)
    q0 = qq[:1]
    ref = np.repeat(_quat_conj_wxyz(q0), int(qq.shape[0]), axis=0)
    return _quat_fix_sign_continuity_wxyz(_quat_mul_wxyz(ref, qq)).astype(np.float32)


def _resample_quat_slerp_wxyz_by_timestamps(q: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Quaternion resample by SLERP using explicit timestamps (wxyz)."""
    qq = _quat_fix_sign_continuity_wxyz(q)
    ts = np.asarray(t_src, dtype=np.float64).reshape(-1)
    td = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    if qq.shape[0] != int(ts.size):
        raise ValueError(f"timestamp length mismatch: q has T={qq.shape[0]}, t_src has {ts.size}")
    if qq.shape[0] < 2 or td.size < 1:
        return qq.astype(np.float32)

    idx_f = np.clip(np.searchsorted(ts, td, side="right") - 1, 0, qq.shape[0] - 2).astype(np.int64)
    idx_n = idx_f + 1
    t0 = ts[idx_f]
    t1 = ts[idx_n]
    denom = np.where((t1 - t0) > 1e-12, (t1 - t0), 1.0)
    u = (td - t0) / denom

    q0 = qq[idx_f]
    q1 = qq[idx_n].copy()
    dot = np.sum(q0 * q1, axis=1)
    flip = dot < 0.0
    if np.any(flip):
        q1[flip] *= -1.0
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    close = dot > 0.9995
    out = np.empty((int(td.size), 4), dtype=np.float64)
    if np.any(close):
        qc = (1.0 - u[close, None]) * q0[close] + u[close, None] * q1[close]
        out[close] = _quat_fix_sign_continuity_wxyz(qc)
    if np.any(~close):
        theta = np.arccos(dot[~close])
        sin_theta = np.sin(theta)
        a = np.sin((1.0 - u[~close]) * theta) / sin_theta
        b = np.sin(u[~close] * theta) / sin_theta
        out[~close] = a[:, None] * q0[~close] + b[:, None] * q1[~close]
        out[~close] = _quat_fix_sign_continuity_wxyz(out[~close])

    # Normalize and enforce a consistent sign trajectory across the whole output.
    out = _quat_fix_sign_continuity_wxyz(out)
    return out.astype(np.float32)


def _aligned_raw_indices(
    raw_len: int,
    T_imu: int,
    *,
    raw_times: np.ndarray | None = None,
    imu_times: np.ndarray | None = None,
) -> np.ndarray:
    if raw_len <= 0 or T_imu <= 0:
        return np.zeros((max(int(T_imu), 0),), dtype=np.int64)

    if (raw_times is not None and imu_times is not None
            and len(raw_times) == raw_len and len(imu_times) == T_imu):
        raw_ts = _ensure_strictly_increasing(np.asarray(raw_times, dtype=np.float64).reshape(-1))
        imu_ts = np.asarray(imu_times, dtype=np.float64).reshape(-1)
        raw_idx = np.searchsorted(raw_ts, imu_ts, side="right") - 1
        return np.clip(raw_idx, 0, raw_len - 1).astype(np.int64)

    return np.linspace(0, raw_len - 1, int(T_imu)).astype(np.int64)


def _extract_raw_snippets_for_sensor(
    raw: np.ndarray,
    T_imu: int,
    window: int = RAW_WINDOW,
    raw_times: np.ndarray | None = None,
    imu_times: np.ndarray | None = None,
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    T_imu = int(T_imu)
    window = int(window)
    out = np.zeros((max(T_imu, 0), max(window, 0)), dtype=np.float32)
    if raw.size < 1 or T_imu < 1 or window < 1:
        return out

    raw_idx = _aligned_raw_indices(
        int(raw.size),
        T_imu,
        raw_times=raw_times,
        imu_times=imu_times,
    )
    padded = np.concatenate(
        [
            np.full((window - 1,), float(raw[0]), dtype=np.float64),
            raw,
        ],
        axis=0,
    )
    starts = raw_idx.astype(np.int64)
    offsets = np.arange(window, dtype=np.int64)
    return padded[starts[:, None] + offsets[None, :]].astype(np.float32)


def _extract_causal_envelope_for_sensor(
    raw: np.ndarray,
    T_imu: int,
    window: int = RAW_WINDOW,
    raw_times: np.ndarray | None = None,
    imu_times: np.ndarray | None = None,
) -> np.ndarray:
    snippets = _extract_raw_snippets_for_sensor(
        raw,
        T_imu,
        window=window,
        raw_times=raw_times,
        imu_times=imu_times,
    )
    return np.mean(np.abs(snippets), axis=1, keepdims=True).astype(np.float32)


def emg_feature_layout_from_meta(meta: dict) -> dict:
    mode = str(meta.get("emg_feature_mode", "")).strip().lower()
    has_raw = bool(meta.get("has_raw_emg", False))
    n_raw_total = int(meta.get("n_raw_features", 0))
    n_ch = int(meta.get("n_channels", 16))
    per_sensor = int(meta.get("n_emg_features_per_sensor", 0))

    if mode == RAW_SNIPPET_FEATURE_MODE:
        per_sensor = max(per_sensor, RAW_WINDOW)
        return {
            "mode": mode,
            "per_sensor": per_sensor,
            "names": [f"lag{per_sensor - 1 - i}" for i in range(per_sensor)],
        }

    if mode == EMG_ENVELOPE_FEATURE_MODE:
        return {
            "mode": mode,
            "per_sensor": 1,
            "names": ["env"],
        }

    if mode == "legacy_spectr":
        per_sensor = max(per_sensor, n_ch)
        return {
            "mode": mode,
            "per_sensor": per_sensor,
            "names": [f"spectr{i}" for i in range(per_sensor)],
        }

    if has_raw and n_raw_total > 0:
        per_sensor = max(1, n_raw_total // 3)
        if per_sensor == RAW_WINDOW:
            return {
                "mode": RAW_SNIPPET_FEATURE_MODE,
                "per_sensor": per_sensor,
                "names": [f"lag{per_sensor - 1 - i}" for i in range(per_sensor)],
            }
        if per_sensor == len(ENGINEERED_EMG_FEATURE_NAMES):
            return {
                "mode": "engineered_features",
                "per_sensor": per_sensor,
                "names": list(ENGINEERED_EMG_FEATURE_NAMES),
            }
        return {
            "mode": "emg_features",
            "per_sensor": per_sensor,
            "names": [f"emg{i}" for i in range(per_sensor)],
        }

    per_sensor = max(per_sensor, n_ch)
    return {
        "mode": "legacy_spectr",
        "per_sensor": per_sensor,
        "names": [f"spectr{i}" for i in range(per_sensor)],
    }


def load_recording(path: str | Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Loads a recording saved by rigtest.py.

    Features per timestep:
      - N device_spectr channels x 3 sensors (legacy)
      - 1 causal EMG envelope value x 3 sensors (if raw data available)
      - right thigh orientation from uMyo sensor 2:
          - required: `thigh_quat_wxyz` (wxyz quaternion, 4 dims)
    Label: knee included angle (degrees): 0=bent, 180=straight.

    For consistent window duration across recordings (and consistent evaluation),
    (X, y) are resampled to a uniform TARGET_HZ using the `timestamps` recorded
    by rigtest.py.

    Returns:
      X: (T, F) float32
      y: (T,) float32
      meta: dict
    """
    d = np.load(Path(path), allow_pickle=True).item()

    if "raw_emg_channels" in d and "thigh_imu" in d:
        def _load_gt_angle_signals_from_source() -> dict[str, np.ndarray] | None:
            src = str(d.get("source_angle_csv", "")).strip()
            if not src:
                return None
            p = Path(src)
            if not p.exists():
                return None
            try:
                from emg_tst.gt_dataset import (
                    load_marker_based_thigh_orientation,
                    load_processed_angle_signals,
                )
            except Exception:
                return None
            try:
                out = load_processed_angle_signals(p)
                try:
                    marker_orient = load_marker_based_thigh_orientation(p)
                except Exception:
                    marker_orient = None
                if marker_orient is not None and "thigh_quat_wxyz" in marker_orient:
                    out["thigh_quat_wxyz"] = np.asarray(marker_orient["thigh_quat_wxyz"], dtype=np.float32)
                return out
            except Exception:
                return None

        try:
            from scipy.signal import butter, filtfilt  # type: ignore
        except Exception:
            butter = None
            filtfilt = None

        def _paper_preprocess_emg(raw: np.ndarray, raw_t: np.ndarray | None, t_out: np.ndarray) -> np.ndarray:
            x = np.asarray(raw, dtype=np.float64).reshape(-1)
            if x.size < 4:
                return np.zeros((int(t_out.size),), dtype=np.float32)
            if raw_t is not None and raw_t.size >= 2:
                dt = np.diff(np.asarray(raw_t, dtype=np.float64).reshape(-1))
                fs = float(1.0 / max(np.median(dt), 1e-6))
            else:
                fs = 2000.0
            if butter is not None and filtfilt is not None and fs > 50.0:
                try:
                    b_hp, a_hp = butter(2, 20.0 / (0.5 * fs), btype="highpass")
                    x = filtfilt(b_hp, a_hp, x)
                    x = np.abs(x)
                    b_lp, a_lp = butter(2, 5.0 / (0.5 * fs), btype="lowpass")
                    x = filtfilt(b_lp, a_lp, x)
                except Exception:
                    x = np.abs(x)
            else:
                x = np.abs(x)
            if raw_t is None:
                raw_t = np.arange(int(x.size), dtype=np.float64) / float(fs)
            return np.interp(np.asarray(t_out, dtype=np.float64), np.asarray(raw_t, dtype=np.float64), x).astype(np.float32)

        y0 = np.asarray(d["knee_included_deg"], dtype=np.float32).reshape(-1)
        thigh_imu0 = np.asarray(d["thigh_imu"], dtype=np.float32)
        if thigh_imu0.ndim != 2:
            raise ValueError(f"GT recording {str(path)!r} has invalid thigh_imu shape {thigh_imu0.shape}")
        T0 = min(int(y0.shape[0]), int(thigh_imu0.shape[0]))
        y = y0[:T0].astype(np.float32)
        thigh_imu = thigh_imu0[:T0].astype(np.float32)

        gt_angles = _load_gt_angle_signals_from_source()
        thigh_pitch = None
        thigh_quat = None
        if gt_angles is not None:
            thigh_pitch = np.asarray(gt_angles["thigh_pitch_deg"], dtype=np.float32).reshape(-1)
            thigh_quat = np.asarray(gt_angles["thigh_quat_wxyz"], dtype=np.float32).reshape(-1, 4)
        if thigh_pitch is None:
            thigh_pitch = np.asarray(d.get("thigh_pitch_deg", np.zeros((T0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        thigh_pitch = thigh_pitch[:T0].astype(np.float32)

        if thigh_quat is None and "thigh_quat_wxyz" in d:
            qq = np.asarray(d["thigh_quat_wxyz"], dtype=np.float32).reshape(-1, 4)
            if qq.shape[0] >= T0:
                thigh_quat = qq[:T0].astype(np.float32)
        elif thigh_quat is not None and int(thigh_quat.shape[0]) >= T0:
            thigh_quat = thigh_quat[:T0].astype(np.float32)
        else:
            thigh_quat = None

        did_resample = False
        t_src = None
        t_dst = None
        orig_hz = float(d.get("effective_hz", 200.0))
        t_offset = 0.0
        if "timestamps" in d:
            t_src = np.asarray(d["timestamps"], dtype=np.float64).reshape(-1)[:T0]
        if t_src is not None and t_src.size >= 2 and T0 >= 2:
            t_src = _ensure_strictly_increasing(t_src)
            t_offset = float(t_src[0])
            t_src = t_src - t_offset
            dur = float(t_src[-1])
            if np.isfinite(dur) and dur > 1e-6:
                orig_hz = float((T0 - 1) / dur)
                n_dst = int(round(dur * float(GT_PAPER_HZ))) + 1
                n_dst = max(2, n_dst)
                t_dst = (np.arange(n_dst, dtype=np.float64) / float(GT_PAPER_HZ)).astype(np.float64)
                y = _resample_linear_by_timestamps(y, t_src, t_dst)
                thigh_imu = _resample_linear_by_timestamps(thigh_imu, t_src, t_dst)
                thigh_pitch = _resample_linear_by_timestamps(thigh_pitch, t_src, t_dst)
                if thigh_quat is not None:
                    thigh_quat = _resample_quat_slerp_wxyz_by_timestamps(thigh_quat, t_src, t_dst)
                did_resample = True

        T = int(y.shape[0])
        effective_hz = float(GT_PAPER_HZ) if bool(did_resample) else float(orig_hz)
        imu_times = t_dst if t_dst is not None else t_src

        raw_emg = np.asarray(d["raw_emg_channels"], dtype=np.float64)
        if raw_emg.ndim != 2:
            raise ValueError(f"GT recording {str(path)!r} has invalid raw_emg_channels shape {raw_emg.shape}")
        raw_ts = None
        if "raw_emg_times" in d:
            raw_ts = np.asarray(d["raw_emg_times"], dtype=np.float64).reshape(-1) - float(t_offset)
        emg_parts = [_paper_preprocess_emg(raw_emg[ch], raw_ts, np.asarray(imu_times, dtype=np.float64))[:, None] for ch in range(int(raw_emg.shape[0]))]
        emg = np.concatenate(emg_parts, axis=1).astype(np.float32)
        X = np.concatenate([emg, thigh_imu], axis=1).astype(np.float32)
        meta = {
            "n_channels": int(raw_emg.shape[0]),
            "n_raw_features": int(emg.shape[1]),
            "n_features": int(X.shape[1]),
            "n_samples": int(T),
            "effective_hz": float(effective_hz),
            "orig_hz": float(orig_hz),
            "target_hz": float(GT_PAPER_HZ),
            "resampled": bool(did_resample),
            "thigh_mode": "imu6",
            "thigh_n_features": int(thigh_imu.shape[1]),
            "has_thigh_quat": bool(thigh_quat is not None),
            "has_raw_emg": True,
            "emg_feature_mode": "gt_paper_preprocessed",
            "n_emg_features_per_sensor": 1,
            "raw_window_samples": 0,
            "n_angular_velocity_features": 0,
            "thigh_pitch_deg_series": thigh_pitch.astype(np.float32),
            "thigh_quat_series": None if thigh_quat is None else thigh_quat.astype(np.float32),
            "source_dataset": str(d.get("source_dataset", "")),
        }
        return X, y, meta

    # Prefer explicit key; fall back to legacy 'imu'.
    if "knee_included_deg" in d:
        y0 = np.asarray(d["knee_included_deg"], dtype=np.float32)
    else:
        y0 = np.asarray(d["imu"], dtype=np.float32)

    s1 = np.asarray(d["emg_sensor1"], dtype=np.float32)  # (N, T)
    s2 = np.asarray(d["emg_sensor2"], dtype=np.float32)
    s3 = np.asarray(d["emg_sensor3"], dtype=np.float32)

    n_ch = s1.shape[0]
    T0 = min(y0.shape[0], s1.shape[1], s2.shape[1], s3.shape[1])

    # Thigh feature: require quaternion (no fallback to scalar thigh_angle).
    if "thigh_quat_wxyz" not in d:
        raise KeyError(
            f"Recording {str(path)!r} is missing required key 'thigh_quat_wxyz'. "
            "Re-record with the updated uMyo_python_tools/rigtest.py."
        )
    q = np.asarray(d["thigh_quat_wxyz"], dtype=np.float32).reshape(-1, 4)
    T0 = min(T0, q.shape[0])
    q = q[:T0].astype(np.float32)
    # Normalize and enforce sign continuity (q and -q are equivalent).
    n = np.linalg.norm(q, axis=1, keepdims=True)
    bad = n.reshape(-1) < 1e-6
    if np.any(bad):
        q[bad, :] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        n = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.clip(n, 1e-6, None)
    for i in range(1, q.shape[0]):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    thigh = q.astype(np.float32)
    thigh_mode = "quat"

    y = y0[:T0]

    # Resample labels + thigh orientation first so the raw EMG snippets can be
    # aligned directly to the final 200 Hz grid instead of interpolating an
    # already-engineered feature vector.
    did_resample = False
    t_src = None
    t_dst = None
    if "timestamps" in d:
        t_src = np.asarray(d["timestamps"], dtype=np.float64).reshape(-1)
    orig_hz = float(d.get("effective_hz", 0.0))
    t_offset = 0.0  # absolute time offset removed from all timestamps
    if t_src is not None and t_src.size >= 2 and int(T0) >= 2:
        t_src = t_src[: int(T0)]
        t_src = _ensure_strictly_increasing(t_src)
        t_offset = float(t_src[0])
        t_src = t_src - t_offset
        dur = float(t_src[-1])
        if np.isfinite(dur) and dur > 1e-6:
            orig_hz = float((T0 - 1) / dur)
            n_dst = int(round(dur * float(TARGET_HZ))) + 1
            n_dst = max(2, n_dst)
            t_dst = (np.arange(n_dst, dtype=np.float64) / float(TARGET_HZ)).astype(np.float64)
            y = _resample_linear_by_timestamps(y, t_src, t_dst)
            thigh = _resample_quat_slerp_wxyz_by_timestamps(thigh, t_src, t_dst)
            did_resample = True

    T = int(y.shape[0])
    effective_hz = float(TARGET_HZ) if bool(did_resample) else float(orig_hz)
    imu_times = t_dst if t_dst is not None else t_src

    # Causal EMG envelope (preferred) or legacy device_spectr fallback.
    has_raw = "raw_emg_sensor1" in d
    if has_raw:
        raw1 = np.asarray(d["raw_emg_sensor1"], dtype=np.float64)
        raw2 = np.asarray(d["raw_emg_sensor2"], dtype=np.float64)
        raw3 = np.asarray(d["raw_emg_sensor3"], dtype=np.float64)
        # Raw EMG timestamps must use the same zero-reference as imu_times.
        # imu_times starts at 0 (t_offset subtracted above); apply the same
        # offset to raw timestamps so searchsorted alignment is correct.
        def _zero_raw_ts(key: str) -> "np.ndarray | None":
            if key not in d:
                return None
            ts = np.asarray(d[key], dtype=np.float64)
            return ts - t_offset
        raw_ts1 = _zero_raw_ts("raw_emg_times1")
        raw_ts2 = _zero_raw_ts("raw_emg_times2")
        raw_ts3 = _zero_raw_ts("raw_emg_times3")

        feat1 = _extract_causal_envelope_for_sensor(raw1, T, RAW_WINDOW, raw_times=raw_ts1, imu_times=imu_times)
        feat2 = _extract_causal_envelope_for_sensor(raw2, T, RAW_WINDOW, raw_times=raw_ts2, imu_times=imu_times)
        feat3 = _extract_causal_envelope_for_sensor(raw3, T, RAW_WINDOW, raw_times=raw_ts3, imu_times=imu_times)
        emg = np.concatenate([feat1, feat2, feat3], axis=1).astype(np.float32)
        n_raw_feat = int(emg.shape[1])
        emg_feature_mode = EMG_ENVELOPE_FEATURE_MODE
        n_emg_features_per_sensor = 1
    else:
        spectr = np.concatenate([s1[:, :T0].T, s2[:, :T0].T, s3[:, :T0].T], axis=1).astype(np.float32)
        if did_resample and t_src is not None and t_dst is not None and spectr.shape[0] == int(t_src.size):
            spectr = _resample_linear_by_timestamps(spectr, t_src, t_dst)
        else:
            spectr = spectr[:T].astype(np.float32)
        emg = spectr.astype(np.float32)
        n_raw_feat = 0
        emg_feature_mode = "legacy_spectr"
        n_emg_features_per_sensor = int(n_ch)

    X = np.concatenate([emg, thigh], axis=1).astype(np.float32)

    meta = {
        "n_channels": int(n_ch),
        "n_raw_features": int(n_raw_feat),
        "n_features": int(X.shape[1]),
        "n_samples": int(T),
        "effective_hz": float(effective_hz),
        "orig_hz": float(orig_hz),
        "target_hz": float(TARGET_HZ),
        "resampled": bool(did_resample),
        "thigh_mode": str(thigh_mode),
        "thigh_n_features": int(thigh.shape[1]),
        "has_thigh_quat": "thigh_quat_wxyz" in d,
        "has_raw_emg": has_raw,
        "emg_feature_mode": str(emg_feature_mode),
        "n_emg_features_per_sensor": int(n_emg_features_per_sensor),
        "raw_window_samples": int(RAW_WINDOW),
        "n_angular_velocity_features": 0,
    }
    return X, y, meta


@dataclass
class StandardScaler:
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray], eps: float = 1e-6) -> "StandardScaler":
        xs = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        mean = xs.mean(axis=0).astype(np.float32)
        std = xs.std(axis=0).astype(np.float32)
        std = np.maximum(std, eps).astype(np.float32)
        return cls(mean_=mean, std_=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_


class WindowDataset(Dataset):
    """
    Deterministic sliding windows over recordings.
    For 'real-time' mode use predict_last_only=True and label is y[end-1].
    """
    def __init__(
        self,
        X_list: list[np.ndarray],
        y_list: Optional[list[np.ndarray]],
        *,
        window: int,
        stride: int,
        predict_last_only: bool = True,
        label_shift: int = 0,
    ):
        self.window = int(window)
        self.stride = int(stride)
        self.predict_last_only = bool(predict_last_only)
        self.label_shift = int(label_shift)

        self.Xw: list[np.ndarray] = []
        self.yw: list[np.ndarray] = []
        self.has_labels = y_list is not None

        for i, X in enumerate(X_list):
            y = None if y_list is None else y_list[i]
            n = X.shape[0]
            for start in range(0, n - self.window + 1, self.stride):
                end = start + self.window
                self.Xw.append(X[start:end])
                if y is not None:
                    idx = min(end - 1 + self.label_shift, len(y) - 1)
                    if self.predict_last_only:
                        self.yw.append(np.array([y[idx]], dtype=np.float32))
                    else:
                        ys = y[start:end].astype(np.float32)
                        if self.label_shift != 0:
                            ys2 = np.empty_like(ys)
                            for t in range(self.window):
                                src = min(start + t + self.label_shift, len(y) - 1)
                                ys2[t] = y[src]
                            ys = ys2
                        self.yw.append(ys)

    def __len__(self) -> int:
        return len(self.Xw)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.Xw[idx]).float()
        if self.has_labels:
            y = torch.from_numpy(self.yw[idx]).float()
            return x, y
        return x


class SamplesNPZDataset(Dataset):
    """
    Dataset backed by the output of split_to_samples.py.
    """
    def __init__(self, npz_path: str | Path, indices: np.ndarray):
        data = np.load(Path(npz_path))
        self.X = data["X"].astype(np.float32)
        self.y = data["y"].astype(np.float32)
        self.file_id = data["file_id"].astype(np.int32)
        self.start = data["start"].astype(np.int32)
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        return torch.from_numpy(self.X[idx]).float(), torch.tensor([self.y[idx]], dtype=torch.float32)


def indices_by_file(npz_path: str | Path, test_file_id: int):
    data = np.load(Path(npz_path))
    file_id = data["file_id"].astype(np.int32)
    all_idx = np.arange(file_id.shape[0], dtype=np.int64)
    test_idx = all_idx[file_id == test_file_id]
    train_idx = all_idx[file_id != test_file_id]
    return train_idx, test_idx
