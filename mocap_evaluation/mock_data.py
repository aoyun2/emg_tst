"""Mock curve generation for quickly exercising the mocap matching pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mocap_evaluation.mocap_loader import _interp_gait_curve, _KNEE_R, _HIP_R, TARGET_FPS


@dataclass
class MockCurves:
    knee_label_included_deg: np.ndarray
    thigh_angle_deg: np.ndarray
    predicted_knee_included_deg: np.ndarray
    fps: int = TARGET_FPS


def generate_mock_curves(
    length_s: float = 4.0,
    fps: int = TARGET_FPS,
    seed: int = 7,
    pred_noise_std: float = 4.0,
    imu_noise_std: float = 2.0,
) -> MockCurves:
    """Generate synthetic thigh/knee curves aligned with rigtest conventions.

    Knee labels follow rigtest's included-angle convention:
      * straight leg ~= 180 deg
      * flexion increases as angle decreases toward 0 deg
    """
    rng = np.random.default_rng(seed)
    T = max(1, int(round(length_s * fps)))

    cadence = 110.0
    cycle_s = 60.0 / (cadence / 2.0)
    spc = max(8, int(round(cycle_s * fps)))

    knee_flex_cycle = _interp_gait_curve(_KNEE_R, spc)
    hip_cycle = _interp_gait_curve(_HIP_R, spc)

    reps = int(np.ceil(T / spc))
    knee_flex = np.tile(knee_flex_cycle, reps)[:T].astype(np.float32)
    thigh = np.tile(hip_cycle, reps)[:T].astype(np.float32)

    knee_flex_noisy = knee_flex + rng.normal(0.0, imu_noise_std, T).astype(np.float32)
    thigh_noisy = thigh + rng.normal(0.0, 1.2, T).astype(np.float32)

    knee_included = (180.0 - knee_flex_noisy).astype(np.float32)
    pred_included = (knee_included + rng.normal(0.0, pred_noise_std, T)).astype(np.float32)

    return MockCurves(
        knee_label_included_deg=np.clip(knee_included, 0.0, 180.0),
        thigh_angle_deg=thigh_noisy,
        predicted_knee_included_deg=np.clip(pred_included, 0.0, 180.0),
        fps=fps,
    )


def save_mock_curves(path: str, curves: MockCurves) -> None:
    np.savez(
        path,
        knee_label_included_deg=curves.knee_label_included_deg,
        thigh_angle_deg=curves.thigh_angle_deg,
        predicted_knee_included_deg=curves.predicted_knee_included_deg,
        fps=np.array([curves.fps], dtype=np.int32),
    )


def load_mock_curves(path: str) -> MockCurves:
    data = np.load(path)
    return MockCurves(
        knee_label_included_deg=data["knee_label_included_deg"].astype(np.float32),
        thigh_angle_deg=data["thigh_angle_deg"].astype(np.float32),
        predicted_knee_included_deg=data["predicted_knee_included_deg"].astype(np.float32),
        fps=int(data["fps"][0]),
    )
