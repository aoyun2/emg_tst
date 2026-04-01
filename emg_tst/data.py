from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --- Raw EMG feature extraction ---
# RAW_WINDOW is in *raw EMG samples* (uMyo raw stream is ~400Hz).
RAW_WINDOW = 32   # causal rolling window (~80ms at 400Hz)
N_FFT_BANDS = 8   # frequency bands for spectral features

# Recordings are not perfectly uniform in time (Bluetooth + serial jitter). For
# reproducible training/evaluation (and consistent window length in seconds),
# we resample each recording to a fixed uniform rate using its timestamps.
TARGET_HZ = 200.0


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


def _extract_raw_features_for_sensor(
    raw: np.ndarray,
    T_imu: int,
    window: int = RAW_WINDOW,
    raw_times: np.ndarray | None = None,
    imu_times: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute per-IMU-timestep features from native-rate raw EMG.

    For each of T_imu timesteps, takes a causal rolling window of raw samples
    and computes:
      - RMS (root mean square)
      - MAV (mean absolute value)
      - WL  (waveform length = sum of abs differences)
      - ZC  (zero crossing count, normalized)
      - SSC (slope sign changes, normalized)
      - Spectral power in N_FFT_BANDS frequency bands

    Args:
      raw:       (R,) flat raw EMG at native rate (~400 Hz)
      T_imu:     number of IMU-rate timesteps to produce
      window:    rolling window size (samples)
      raw_times: (R,) timestamps of each raw EMG sample in seconds (optional).
                 When provided, used for accurate time-based alignment instead
                 of the uniform linspace approximation.
      imu_times: (T_imu,) timestamps of IMU timesteps in seconds (optional).
                 Required when raw_times is provided.

    Returns:
      features: (T_imu, 5 + N_FFT_BANDS) float32
    """
    R = len(raw)
    n_feat = 5 + N_FFT_BANDS
    out = np.zeros((T_imu, n_feat), dtype=np.float32)

    if R < window:
        return out  # not enough raw data

    # Map IMU timestep -> raw index.
    # Prefer timestamp-based alignment (new recordings from rigtest_gui.py) over
    # the uniform-linspace approximation (old recordings from rigtest.py).
    # The linspace approximation assumes raw samples are uniformly distributed,
    # but they actually arrive in bursts of 8, which misaligns features by up to
    # ~20 ms – significant noise at 200 Hz.
    if (raw_times is not None and imu_times is not None
            and len(raw_times) == R and len(imu_times) == T_imu):
        raw_ts = np.asarray(raw_times, dtype=np.float64)
        imu_ts = np.asarray(imu_times, dtype=np.float64)
        # For each IMU timestep, find the last raw sample that occurred at or
        # before that time (causal).
        raw_idx = np.searchsorted(raw_ts, imu_ts, side="right") - 1
        raw_idx = np.clip(raw_idx, 0, R - 1).astype(np.int64)
    else:
        # Fallback: uniform distribution assumption
        raw_idx = np.linspace(0, R - 1, T_imu).astype(np.int64)

    for t in range(T_imu):
        end = int(raw_idx[t]) + 1
        start = int(end - window)

        # Causal fixed-length window: pad on the left for the first few timesteps.
        if start < 0:
            pad_n = int(-start)
            pad_val = float(raw[0])
            seg = np.concatenate(
                [
                    np.full((pad_n,), pad_val, dtype=np.float64),
                    raw[0:end].astype(np.float64),
                ],
                axis=0,
            )
        else:
            seg = raw[start:end].astype(np.float64)

        # Defensive: ensure we always compute on a fixed window length.
        if int(seg.size) != int(window):
            if int(seg.size) < 2:
                continue
            seg = seg[-int(window) :].astype(np.float64, copy=False)
        n = int(seg.size)

        # Remove DC offset for frequency features
        seg_centered = seg - seg.mean()

        # RMS — log1p: EMG amplitude is right-skewed / roughly log-normal;
        # log1p maps the heavy tail into a Gaussian-like range that StandardScaler handles well.
        out[t, 0] = np.log1p(np.sqrt(np.mean(seg_centered ** 2)))

        # MAV — same reasoning
        out[t, 1] = np.log1p(np.mean(np.abs(seg_centered)))

        # Waveform length — same reasoning
        out[t, 2] = np.log1p(np.sum(np.abs(np.diff(seg))))

        # Zero crossings (on centered signal)
        signs = np.sign(seg_centered)
        sign_changes = np.diff(signs)
        out[t, 3] = np.count_nonzero(sign_changes) / n

        # Slope sign changes
        d = np.diff(seg)
        if len(d) >= 2:
            out[t, 4] = np.count_nonzero(np.diff(np.sign(d))) / n

        # Spectral power in N_FFT_BANDS bands
        fft_vals = np.fft.rfft(seg_centered)
        power = np.abs(fft_vals) ** 2
        n_bins = len(power)
        # Split into N_FFT_BANDS equal bands
        band_edges = np.linspace(0, n_bins, N_FFT_BANDS + 1, dtype=int)
        for b in range(N_FFT_BANDS):
            lo, hi = band_edges[b], band_edges[b + 1]
            if hi > lo:
                # log1p: spectral power is non-negative and extremely right-skewed
                # (std ~10^7–10^8 raw). log1p compresses the range to ~[0,20],
                # making StandardScaler effective instead of near-zero everywhere.
                out[t, 5 + b] = np.log1p(np.mean(power[lo:hi]))

    return out


def load_recording(path: str | Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Loads a recording saved by rigtest.py.

    Features per timestep:
      - N device_spectr channels x 3 sensors (legacy)
      - (5 + N_FFT_BANDS) raw EMG features x 3 sensors (if raw data available)
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
    # Prefer explicit key; fall back to legacy 'imu'.
    if "knee_included_deg" in d:
        y = np.asarray(d["knee_included_deg"], dtype=np.float32)
    else:
        y = np.asarray(d["imu"], dtype=np.float32)

    s1 = np.asarray(d["emg_sensor1"], dtype=np.float32)  # (N, T)
    s2 = np.asarray(d["emg_sensor2"], dtype=np.float32)
    s3 = np.asarray(d["emg_sensor3"], dtype=np.float32)

    n_ch = s1.shape[0]
    T = min(y.shape[0], s1.shape[1], s2.shape[1], s3.shape[1])

    # Thigh feature: require quaternion (no fallback to scalar thigh_angle).
    if "thigh_quat_wxyz" not in d:
        raise KeyError(
            f"Recording {str(path)!r} is missing required key 'thigh_quat_wxyz'. "
            "Re-record with the updated uMyo_python_tools/rigtest.py."
        )
    q = np.asarray(d["thigh_quat_wxyz"], dtype=np.float32).reshape(-1, 4)
    T = min(T, q.shape[0])
    q = q[:T].astype(np.float32)
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

    y = y[:T]
    # device_spectr is legacy; prefer raw EMG features when available.
    # spectr = np.concatenate([s1[:, :T].T, s2[:, :T].T, s3[:, :T].T], axis=1).astype(np.float32)

    # Raw EMG features (if available)
    has_raw = "raw_emg_sensor1" in d
    if has_raw:
        raw1 = np.asarray(d["raw_emg_sensor1"], dtype=np.float64)
        raw2 = np.asarray(d["raw_emg_sensor2"], dtype=np.float64)
        raw3 = np.asarray(d["raw_emg_sensor3"], dtype=np.float64)

        # Use per-sample timestamps (from rigtest_gui.py) when available.
        # These allow causal timestamp-based alignment instead of the uniform
        # linspace approximation, eliminating the burst-timing misalignment.
        imu_ts = None
        if "timestamps" in d:
            imu_ts = _ensure_strictly_increasing(
                np.asarray(d["timestamps"], dtype=np.float64).reshape(-1)[:T]
            )
        raw_ts1 = np.asarray(d["raw_emg_times1"], dtype=np.float64) if "raw_emg_times1" in d else None
        raw_ts2 = np.asarray(d["raw_emg_times2"], dtype=np.float64) if "raw_emg_times2" in d else None
        raw_ts3 = np.asarray(d["raw_emg_times3"], dtype=np.float64) if "raw_emg_times3" in d else None

        feat1 = _extract_raw_features_for_sensor(raw1, T, raw_times=raw_ts1, imu_times=imu_ts)
        feat2 = _extract_raw_features_for_sensor(raw2, T, raw_times=raw_ts2, imu_times=imu_ts)
        feat3 = _extract_raw_features_for_sensor(raw3, T, raw_times=raw_ts3, imu_times=imu_ts)

        raw_feats = np.concatenate([feat1, feat2, feat3], axis=1)  # (T, 3*(5+N_FFT_BANDS))
        n_raw_feat = raw_feats.shape[1]
        X = np.concatenate([raw_feats, thigh], axis=1)
    else:
        # Fallback for legacy files without raw EMG: use device_spectr
        n_raw_feat = 0
        spectr = np.concatenate([s1[:, :T].T, s2[:, :T].T, s3[:, :T].T], axis=1).astype(np.float32)
        X = np.concatenate([spectr, thigh], axis=1)

    # Resample to a uniform rate for reproducible training/evaluation.
    did_resample = False
    t = None
    if "timestamps" in d:
        t = np.asarray(d["timestamps"], dtype=np.float64).reshape(-1)
    orig_hz = float(d.get("effective_hz", 0.0))
    if t is not None and t.size >= 2 and int(X.shape[0]) >= 2:
        t = t[: int(X.shape[0])]
        t = _ensure_strictly_increasing(t)
        t = t - float(t[0])
        dur = float(t[-1])
        if np.isfinite(dur) and dur > 1e-6:
            T0 = int(X.shape[0])
            orig_hz = float((T0 - 1) / dur)
            n_dst = int(round(dur * float(TARGET_HZ))) + 1
            n_dst = max(2, n_dst)
            t_dst = (np.arange(n_dst, dtype=np.float64) / float(TARGET_HZ)).astype(np.float64)

            if int(X.shape[1]) < 4:
                raise RuntimeError(f"Expected last 4 features to be thigh quaternion, got F={int(X.shape[1])}")
            X_scalar = X[:, : int(X.shape[1]) - 4]
            X_quat = X[:, int(X.shape[1]) - 4 :]
            if X_scalar.size > 0:
                Xs = _resample_linear_by_timestamps(X_scalar, t, t_dst)
            else:
                Xs = np.zeros((int(t_dst.size), 0), dtype=np.float32)
            Xq = _resample_quat_slerp_wxyz_by_timestamps(X_quat, t, t_dst)
            yr = _resample_linear_by_timestamps(y, t, t_dst)

            X = np.concatenate([Xs, Xq], axis=1).astype(np.float32)
            y = np.asarray(yr, dtype=np.float32).reshape(-1)
            T = int(X.shape[0])
            did_resample = True

    effective_hz = float(TARGET_HZ) if bool(did_resample) else float(orig_hz)
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
