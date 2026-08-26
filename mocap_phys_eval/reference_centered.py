from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .recording import CompareRecordingPaths, record_compare_rollout
from .sim import OverrideSpec


SOURCE_HZ = 100.0
DECIMATION = 3
CONTROL_HZ = SOURCE_HZ / DECIMATION


def exact_control_frames(values: np.ndarray, *, control_hz: float) -> np.ndarray:
    """Select recorded 100-Hz frames for the 33.333-Hz control timeline.

    This is direct deterministic decimation (indices 0, 3, 6, ...), not
    interpolation.  Fail closed if a different rate relationship is supplied.
    """
    rate = float(control_hz)
    if not np.isfinite(rate) or abs(rate - CONTROL_HZ) > 1.0e-3:
        raise ValueError(
            f"Expected MoCapAct control rate {CONTROL_HZ:.9f} Hz, received {rate:.9f}"
        )
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] < 1:
        raise ValueError("values must have a nonempty final time dimension")
    indices = np.arange(0, array.shape[-1], DECIMATION, dtype=np.int64)
    return np.take(array, indices, axis=-1)


def inject_prediction_error(
    reference_deg: np.ndarray,
    prediction_deg: np.ndarray,
    measured_deg: np.ndarray,
    *,
    knee_sign: float,
    control_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place recorded prediction error in a stable realized reference.

    Prediction and measurement remain on their native recorded 100-Hz target
    frames until direct frame selection. No interpolation, scaling, clipping,
    or offset is applied.
    """
    reference = np.asarray(reference_deg, dtype=np.float32).reshape(-1)
    prediction = np.asarray(prediction_deg, dtype=np.float32).reshape(-1)
    measured = np.asarray(measured_deg, dtype=np.float32).reshape(-1)
    sign = float(knee_sign)
    if sign not in (-1.0, 1.0):
        raise ValueError(f"knee_sign must be +1 or -1, received {sign}")
    if prediction.shape != measured.shape or prediction.size < 1:
        raise ValueError("Prediction and measurement must have the same nonempty shape")
    if not all(np.all(np.isfinite(values)) for values in (reference, prediction, measured)):
        raise ValueError("Reference, prediction, and measurement must be finite")
    error = np.asarray(
        exact_control_frames(prediction - measured, control_hz=control_hz),
        dtype=np.float32,
    ).reshape(-1)
    if reference.shape != error.shape:
        raise ValueError(
            f"Reference has {reference.size} steps but selected prediction error has "
            f"{error.size} steps"
        )
    target = np.asarray(reference + np.float32(sign) * error, dtype=np.float32)
    mapped_rmse = float(np.sqrt(np.mean(np.square(target.astype(np.float64) - reference))))
    error_rmse = float(np.sqrt(np.mean(np.square(error.astype(np.float64)))))
    if not np.isclose(mapped_rmse, error_rmse, rtol=1.0e-6, atol=1.0e-6):
        raise RuntimeError("Reference-centered mapping did not preserve simulation-frame RMSE")
    return target, error


def capture_unmodified_reference(
    *,
    out_npz_path: Path,
    clip_id: str,
    start_step: int,
    end_step: int,
    primary_steps: int,
    warmup_steps: int,
    policy: Any,
    nominal_reference_deg: np.ndarray,
    override: OverrideSpec,
    width: int,
    height: int,
    camera_id: int,
    render_media: bool = True,
    prebuilt_envs: tuple[Any, Any, Any | None] | None = None,
) -> CompareRecordingPaths:
    """Record two deterministic unmodified expert replays for a baseline."""
    return record_compare_rollout(
        out_npz_path=out_npz_path,
        clip_id=clip_id,
        start_step=start_step,
        end_step=end_step,
        primary_steps=primary_steps,
        warmup_steps=warmup_steps,
        policy=policy,
        override=override,
        knee_good_query_deg=nominal_reference_deg,
        knee_bad_query_deg=nominal_reference_deg,
        width=width,
        height=height,
        camera_id=camera_id,
        deterministic_policy=True,
        seed=0,
        run_bad=False,
        panel_labels=("Reference capture", "Deterministic replay", "Unused"),
        moving_target_pd=False,
        apply_good_override=False,
        render_media=render_media,
        prebuilt_envs=prebuilt_envs,
    )


def load_reference_baseline(path: Path, *, required_steps: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as stored:
        if bool(np.asarray(stored["apply_good_override"]).reshape(())):
            raise RuntimeError("Reference baseline was produced with an active override")
        reference = np.asarray(stored["knee_ref_actual_deg"], dtype=np.float32)
        replay = np.asarray(stored["knee_good_actual_deg"], dtype=np.float32)
    if reference.shape != (required_steps,) or replay.shape != reference.shape:
        raise RuntimeError("Reference capture did not complete the required window")
    if not np.array_equal(reference, replay):
        error = float(np.sqrt(np.mean(np.square(reference - replay))))
        raise RuntimeError(f"Deterministic reference replay changed (RMSE={error:.9g} deg)")
    return reference
