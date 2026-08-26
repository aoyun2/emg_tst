from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


CACHE_VERSION = "GAIT120_CAUSAL_RAW_LEVEL_WALKING_V2"
N_EMG = 12
INPUT_FRAMES = 10
HORIZON_FRAMES = 10
TRAIN_TRIALS = (1, 2, 3)
VALIDATION_TRIALS = (4,)
TEST_TRIALS = (5,)
BOOTSTRAP_DRAWS = 100_000
RANDOMIZATION_DRAWS = 1_000_000


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ExampleSet:
    x: np.ndarray
    y_standardized: np.ndarray
    y_deg: np.ndarray
    target_mean_deg: np.ndarray
    target_std_deg: np.ndarray
    subject_number: np.ndarray
    trial_index: np.ndarray
    input_end_frame: np.ndarray
    target_frame: np.ndarray

    def __len__(self) -> int:
        return int(self.y_deg.size)


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        data = {key: np.asarray(stored[key]) for key in stored.files}
    if str(data["cache_version"].reshape(())) != CACHE_VERSION:
        raise RuntimeError(f"Unexpected Gait120 cache version in {path}")
    if int(data["motion_hz"].reshape(())) != 100:
        raise RuntimeError(f"Unexpected motion rate in {path}")
    emg = np.asarray(data["emg_frame_native_value"], dtype=np.float32)
    knee = np.asarray(data["knee_flexion_deg"], dtype=np.float32).reshape(-1)
    trial = np.asarray(data["trial_index"], dtype=np.int16).reshape(-1)
    frame = np.asarray(data["frame_index"], dtype=np.int16).reshape(-1)
    if emg.shape != (knee.size, N_EMG) or trial.size != knee.size or frame.size != knee.size:
        raise RuntimeError(f"Malformed Gait120 cache arrays in {path}")
    if not np.all(np.isfinite(emg)) or not np.all(np.isfinite(knee)):
        raise RuntimeError(f"Non-finite cached values in {path}")
    return data


def _training_scaler(data: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    trial = np.asarray(data["trial_index"], dtype=np.int16).reshape(-1)
    rows = np.flatnonzero(np.isin(trial, TRAIN_TRIALS))
    if rows.size < 1:
        raise RuntimeError("No participant calibration rows")
    emg = np.asarray(data["emg_frame_native_value"], dtype=np.float64)[rows]
    knee = np.asarray(data["knee_flexion_deg"], dtype=np.float64).reshape(-1)[rows]
    emg_mean = np.mean(emg, axis=0)
    emg_std = np.maximum(np.std(emg, axis=0), 1.0e-8)
    knee_mean = float(np.mean(knee))
    knee_std = max(float(np.std(knee)), 1.0e-8)
    return {
        "emg_mean": emg_mean.astype(np.float32),
        "emg_std": emg_std.astype(np.float32),
        "knee_mean": knee_mean,
        "knee_std": knee_std,
    }


def _examples_for_trials(
    *,
    data: dict[str, np.ndarray],
    subject_number: int,
    trials: tuple[int, ...],
    scaler: dict[str, np.ndarray | float],
) -> ExampleSet:
    emg = np.asarray(data["emg_frame_native_value"], dtype=np.float32)
    knee = np.asarray(data["knee_flexion_deg"], dtype=np.float32).reshape(-1)
    trial_all = np.asarray(data["trial_index"], dtype=np.int16).reshape(-1)
    frame_all = np.asarray(data["frame_index"], dtype=np.int16).reshape(-1)
    emg_mean = np.asarray(scaler["emg_mean"], dtype=np.float32)
    emg_std = np.asarray(scaler["emg_std"], dtype=np.float32)
    knee_mean = float(scaler["knee_mean"])
    knee_std = float(scaler["knee_std"])

    xs: list[np.ndarray] = []
    ys: list[float] = []
    y_deg: list[float] = []
    trial_out: list[int] = []
    input_end: list[int] = []
    target_frame: list[int] = []
    for trial_index in trials:
        rows = np.flatnonzero(trial_all == trial_index)
        if rows.size < INPUT_FRAMES + HORIZON_FRAMES + 1:
            raise RuntimeError(f"S{subject_number:03d} trial {trial_index} is too short")
        if not np.array_equal(frame_all[rows], np.arange(rows.size)):
            raise RuntimeError("Cached trial frame indices are not consecutive")
        for local_end in range(INPUT_FRAMES - 1, rows.size - HORIZON_FRAMES):
            window_rows = rows[local_end - INPUT_FRAMES + 1 : local_end + 1]
            target_row = rows[local_end + HORIZON_FRAMES]
            emg_window = (emg[window_rows] - emg_mean[None, :]) / emg_std[None, :]
            knee_window = (knee[window_rows] - knee_mean) / knee_std
            xs.append(
                np.concatenate([emg_window, knee_window[:, None]], axis=1).astype(np.float32)
            )
            target = float(knee[target_row])
            ys.append((target - knee_mean) / knee_std)
            y_deg.append(target)
            trial_out.append(int(trial_index))
            input_end.append(int(local_end))
            target_frame.append(int(local_end + HORIZON_FRAMES))
    n = len(xs)
    return ExampleSet(
        x=np.stack(xs).astype(np.float32),
        y_standardized=np.asarray(ys, dtype=np.float32),
        y_deg=np.asarray(y_deg, dtype=np.float32),
        target_mean_deg=np.full(n, knee_mean, dtype=np.float32),
        target_std_deg=np.full(n, knee_std, dtype=np.float32),
        subject_number=np.full(n, subject_number, dtype=np.int16),
        trial_index=np.asarray(trial_out, dtype=np.int16),
        input_end_frame=np.asarray(input_end, dtype=np.int16),
        target_frame=np.asarray(target_frame, dtype=np.int16),
    )


def _combine(sets: list[ExampleSet]) -> ExampleSet:
    if not sets:
        raise ValueError("No example sets to combine")
    return ExampleSet(
        **{
            field: np.concatenate([getattr(item, field) for item in sets], axis=0)
            for field in ExampleSet.__dataclass_fields__
        }
    )


def _build_examples(
    cache_dir: Path, subject_numbers: list[int]
) -> tuple[ExampleSet, ExampleSet, ExampleSet, dict[str, dict[str, Any]]]:
    train: list[ExampleSet] = []
    validation: list[ExampleSet] = []
    test: list[ExampleSet] = []
    scalers: dict[str, dict[str, Any]] = {}
    for number in subject_numbers:
        subject = f"S{number:03d}"
        data = _load_cache(cache_dir / f"{subject}.npz")
        stored_subject = str(np.asarray(data["subject"]).reshape(()))
        if stored_subject != subject:
            raise RuntimeError(f"Cache subject mismatch: expected {subject}, got {stored_subject}")
        available_trials = set(
            np.asarray(data["trial_index"], dtype=np.int16).reshape(-1).tolist()
        )
        required = set(TRAIN_TRIALS + VALIDATION_TRIALS + TEST_TRIALS)
        if not required.issubset(available_trials):
            raise RuntimeError(f"{subject} lacks required trials {sorted(required - available_trials)}")
        scaler = _training_scaler(data)
        scalers[subject] = {
            "emg_mean": np.asarray(scaler["emg_mean"]).tolist(),
            "emg_std": np.asarray(scaler["emg_std"]).tolist(),
            "knee_mean": float(scaler["knee_mean"]),
            "knee_std": float(scaler["knee_std"]),
        }
        train.append(
            _examples_for_trials(
                data=data, subject_number=number, trials=TRAIN_TRIALS, scaler=scaler
            )
        )
        validation.append(
            _examples_for_trials(
                data=data,
                subject_number=number,
                trials=VALIDATION_TRIALS,
                scaler=scaler,
            )
        )
        test.append(
            _examples_for_trials(
                data=data, subject_number=number, trials=TEST_TRIALS, scaler=scaler
            )
        )
    return _combine(train), _combine(validation), _combine(test), scalers


def _participant_metrics(examples: ExampleSet, prediction_deg: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for subject_number in np.unique(examples.subject_number).tolist():
        keep = examples.subject_number == subject_number
        residual = np.asarray(prediction_deg[keep], dtype=np.float64) - np.asarray(
            examples.y_deg[keep], dtype=np.float64
        )
        rows.append(
            {
                "subject": f"S{int(subject_number):03d}",
                "rmse_deg": float(np.sqrt(np.mean(np.square(residual)))),
                "mae_deg": float(np.mean(np.abs(residual))),
                "n_examples": int(np.sum(keep)),
            }
        )
    all_residual = np.asarray(prediction_deg, dtype=np.float64) - np.asarray(
        examples.y_deg, dtype=np.float64
    )
    return {
        "participants": rows,
        "mean_participant_rmse_deg": float(np.mean([row["rmse_deg"] for row in rows])),
        "pooled_rmse_deg": float(np.sqrt(np.mean(np.square(all_residual)))),
        "pooled_mae_deg": float(np.mean(np.abs(all_residual))),
        "n_examples": len(examples),
    }


def _paired_statistics(
    fused: dict[str, Any], no_emg: dict[str, Any], *, require_gate: bool
) -> dict[str, Any]:
    fused_rows = {row["subject"]: row for row in fused["test"]["participants"]}
    no_emg_rows = {row["subject"]: row for row in no_emg["test"]["participants"]}
    if set(fused_rows) != set(no_emg_rows):
        raise RuntimeError("Paired conditions have different test participants")
    subjects = sorted(fused_rows)
    effects = np.asarray(
        [no_emg_rows[s]["rmse_deg"] - fused_rows[s]["rmse_deg"] for s in subjects],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260824)
    bootstrap_means: list[np.ndarray] = []
    for first in range(0, BOOTSTRAP_DRAWS, 2_000):
        draws = min(2_000, BOOTSTRAP_DRAWS - first)
        index = rng.integers(0, effects.size, size=(draws, effects.size))
        bootstrap_means.append(np.mean(effects[index], axis=1))
    boot = np.concatenate(bootstrap_means)
    lower, upper = np.quantile(boot, [0.025, 0.975]).tolist()

    observed = abs(float(np.mean(effects)))
    exceed = 0
    completed = 0
    for _ in range(0, RANDOMIZATION_DRAWS, 10_000):
        draws = min(10_000, RANDOMIZATION_DRAWS - completed)
        signs = rng.integers(0, 2, size=(draws, effects.size), dtype=np.int8) * 2 - 1
        permuted = np.abs(np.mean(signs * effects[None, :], axis=1))
        exceed += int(np.sum(permuted >= observed - 1.0e-15))
        completed += draws
    p_value = (exceed + 1.0) / (completed + 1.0)
    positive = int(np.sum(effects > 0.0))
    gate = bool(
        float(np.mean(effects)) > 0.0
        and float(lower) > 0.0
        and p_value <= 0.05
        and positive > effects.size / 2
    )
    return {
        "definition": "no-sEMG participant test RMSE minus fused participant test RMSE",
        "subjects": subjects,
        "improvement_deg": effects,
        "mean_improvement_deg": float(np.mean(effects)),
        "median_improvement_deg": float(np.median(effects)),
        "bootstrap_95pct_ci_deg": [float(lower), float(upper)],
        "two_sided_randomization_p": float(p_value),
        "randomization_draws": completed,
        "positive_participants": positive,
        "participant_count": int(effects.size),
        "fused_mean_participant_rmse_deg": float(
            fused["test"]["mean_participant_rmse_deg"]
        ),
        "no_emg_mean_participant_rmse_deg": float(
            no_emg["test"]["mean_participant_rmse_deg"]
        ),
        "gate_applied": bool(require_gate),
        "passed": gate if require_gate else None,
    }
