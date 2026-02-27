"""Utilities for exporting real walking knee/thigh curves from mocap data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mocap_evaluation.mocap_loader import (
    TARGET_FPS,
    load_aggregated_database,
)


@dataclass
class SampleCurves:
    """A compact real-data query containing only thigh pitch and knee included angle."""

    knee_label_included_deg: np.ndarray
    thigh_angle_deg: np.ndarray
    predicted_knee_included_deg: np.ndarray
    fps: int
    category: str
    source_file: str


def _select_walk_boundary(db: dict) -> tuple[int, int, str, str]:
    if "file_boundaries" not in db:
        n = len(db["knee_right"])
        return 0, n, "aggregated", "unknown"

    start, end, fname, cat = db["file_boundaries"][0]
    return int(start), int(end), str(fname), str(cat)


def extract_real_sample_curves(
    mocap_dir: str | Path = "mocap_data",
    seconds: float = 4.0,
    full_database: bool = True,
    seed: int = 13,
    pred_noise_std: float = 2.0,
) -> SampleCurves:
    """Extract one contiguous real walking segment and build eval curves.

    The predicted curve is derived from the label curve + light noise so users can
    exercise the matching/simulation pipeline without synthetic Winter templates.
    """
    db = load_aggregated_database(mocap_root=mocap_dir)

    start, end, source_file, cat = _select_walk_boundary(db)
    seg_len = end - start

    target_len = max(1, int(round(seconds * TARGET_FPS)))
    if seg_len < target_len:
        raise ValueError(
            f"Selected segment too short ({seg_len} frames) for requested "
            f"{target_len}-frame sample."
        )

    rng = np.random.default_rng(seed)
    local_start = int(rng.integers(0, seg_len - target_len + 1))
    s = start + local_start
    e = s + target_len

    knee = db["knee_right"][s:e].astype(np.float32)
    thigh = db["hip_right"][s:e].astype(np.float32)
    pred = (knee + rng.normal(0.0, pred_noise_std, target_len)).astype(np.float32)

    return SampleCurves(
        knee_label_included_deg=np.clip(knee, 0.0, 180.0),
        thigh_angle_deg=thigh,
        predicted_knee_included_deg=np.clip(pred, 0.0, 180.0),
        fps=TARGET_FPS,
        category=cat,
        source_file=source_file,
    )


def save_sample_curves(path: str | Path, curves: SampleCurves) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        knee_label_included_deg=curves.knee_label_included_deg,
        thigh_angle_deg=curves.thigh_angle_deg,
        predicted_knee_included_deg=curves.predicted_knee_included_deg,
        fps=np.array([curves.fps], dtype=np.int32),
        category=np.array([curves.category]),
        source_file=np.array([curves.source_file]),
    )
