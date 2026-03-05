from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ─── Raw EMG feature extraction ──────────────────────────────────────
RAW_WINDOW = 32   # rolling window size for raw features (~80ms at 400Hz)
N_FFT_BANDS = 8   # frequency bands for spectral features

def _extract_raw_features_for_sensor(raw: np.ndarray, T_imu: int, window: int = RAW_WINDOW) -> np.ndarray:
    """
    Compute per-IMU-timestep features from native-rate raw EMG.

    For each of T_imu timesteps, takes a rolling window of raw samples and computes:
      - RMS (root mean square)
      - MAV (mean absolute value)
      - WL  (waveform length = sum of abs differences)
      - ZC  (zero crossing count, normalized)
      - SSC (slope sign changes, normalized)
      - Spectral power in N_FFT_BANDS frequency bands

    Args:
      raw:   (R,) flat raw EMG at native rate (~400Hz)
      T_imu: number of IMU-rate timesteps to produce
      window: rolling window size

    Returns:
      features: (T_imu, 5 + N_FFT_BANDS) float32
    """
    R = len(raw)
    n_feat = 5 + N_FFT_BANDS
    out = np.zeros((T_imu, n_feat), dtype=np.float32)

    if R < window:
        return out  # not enough raw data

    # Map IMU timestep -> raw index (linear interpolation)
    raw_idx = np.linspace(0, R - 1, T_imu).astype(np.int64)

    for t in range(T_imu):
        end = raw_idx[t] + 1
        start = max(0, end - window)
        seg = raw[start:end].astype(np.float64)
        n = len(seg)
        if n < 2:
            continue

        # Remove DC offset for frequency features
        seg_centered = seg - seg.mean()

        # RMS
        out[t, 0] = np.sqrt(np.mean(seg_centered ** 2))

        # MAV
        out[t, 1] = np.mean(np.abs(seg_centered))

        # Waveform length
        out[t, 2] = np.sum(np.abs(np.diff(seg)))

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
                out[t, 5 + b] = np.mean(power[lo:hi])

    return out


def load_recording(path: str | Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Loads a recording saved by rigtest.py.

    Features per timestep:
      - N device_spectr channels × 3 sensors
      - (5 + N_FFT_BANDS) raw EMG features × 3 sensors (if raw data available)
      - right thigh orientation from uMyo sensor 2:
          - preferred: `thigh_quat_wxyz` (wxyz quaternion, 4 dims)
          - legacy fallback: `thigh_angle` (1 dim)
    Label: knee angle from IMU-uMyo diff.

    Returns:
      X: (T, F) float32
      y: (T,) float32
      meta: dict
    """
    d = np.load(Path(path), allow_pickle=True).item()
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
    # device_spectr commented out — using raw EMG features instead
    # spectr = np.concatenate([s1[:, :T].T, s2[:, :T].T, s3[:, :T].T], axis=1).astype(np.float32)

    # Raw EMG features (if available)
    has_raw = "raw_emg_sensor1" in d
    if has_raw:
        raw1 = np.asarray(d["raw_emg_sensor1"], dtype=np.float64)
        raw2 = np.asarray(d["raw_emg_sensor2"], dtype=np.float64)
        raw3 = np.asarray(d["raw_emg_sensor3"], dtype=np.float64)

        feat1 = _extract_raw_features_for_sensor(raw1, T)
        feat2 = _extract_raw_features_for_sensor(raw2, T)
        feat3 = _extract_raw_features_for_sensor(raw3, T)

        raw_feats = np.concatenate([feat1, feat2, feat3], axis=1)  # (T, 3*(5+N_FFT_BANDS))
        n_raw_feat = raw_feats.shape[1]
        X = np.concatenate([raw_feats, thigh], axis=1)
    else:
        # Fallback for legacy files without raw EMG: use device_spectr
        n_raw_feat = 0
        spectr = np.concatenate([s1[:, :T].T, s2[:, :T].T, s3[:, :T].T], axis=1).astype(np.float32)
        X = np.concatenate([spectr, thigh], axis=1)

    meta = {
        "n_channels": int(n_ch),
        "n_raw_features": int(n_raw_feat),
        "n_features": int(X.shape[1]),
        "n_samples": int(T),
        "effective_hz": float(d.get("effective_hz", 0)),
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
