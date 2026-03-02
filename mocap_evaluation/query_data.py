from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class QueryBatch:
    batch_id: str
    thigh_angle_deg: np.ndarray
    knee_angle_pred_deg: np.ndarray
    sample_hz: float


def load_opensim_csv(path: str | Path, thigh_col: str = "thigh_angle", knee_col: str = "knee_angle") -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if thigh_col not in data.dtype.names or knee_col not in data.dtype.names:
        raise KeyError(f"CSV must include columns: {thigh_col}, {knee_col}")
    return np.asarray(data[thigh_col], dtype=np.float32), np.asarray(data[knee_col], dtype=np.float32)


def load_rigtest_npy(path: str | Path, pred_key: str = "knee_pred") -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True).item()
    if "thigh_angle" not in d:
        raise KeyError("rigtest data must contain 'thigh_angle'")
    if pred_key not in d:
        raise KeyError(f"rigtest data must contain prediction key '{pred_key}'")
    return np.asarray(d["thigh_angle"], dtype=np.float32), np.asarray(d[pred_key], dtype=np.float32)


def make_contiguous_batches(
    thigh: np.ndarray,
    knee_pred: np.ndarray,
    *,
    batch_size: int,
    stride: int | None = None,
    sample_hz: float = 200.0,
    prefix: str = "batch",
) -> list[QueryBatch]:
    if stride is None:
        stride = batch_size
    n = min(len(thigh), len(knee_pred))
    out: list[QueryBatch] = []
    for start in range(0, n - batch_size + 1, stride):
        end = start + batch_size
        out.append(
            QueryBatch(
                batch_id=f"{prefix}_{start:06d}",
                thigh_angle_deg=thigh[start:end].astype(np.float32),
                knee_angle_pred_deg=knee_pred[start:end].astype(np.float32),
                sample_hz=float(sample_hz),
            )
        )
    return out
