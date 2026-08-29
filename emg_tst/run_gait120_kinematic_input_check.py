"""Check whether the result depends on the knee predicting its own future.

The reported kinematic stage forecasts knee angle from 60 frames of knee-angle
history.  The obvious objection is that this is autoregression on the target:
a knee angle 100 ms ahead is close to where the knee is now, so a low RMSE might
reflect trajectory smoothness rather than anything reconstructed from the body.
For a transfemoral application the objection sharpens, because the quantity of
interest is whether the knee can be rebuilt from what remains around it.

This check answers that directly.  It refits the same model with the kinematic
stage reading only the surrounding body -- thigh pitch and full thigh
orientation, with nothing measuring the knee -- and compares held-out accuracy.
Everything else is held fixed: same windows, same 100 ms horizon, same
participant-balanced ridge pair, same penalties, same trials 1-3 fit and trial 5
test.

If the two inputs perform comparably, the knee channel is a convenient carrier of
information that is present in the surrounding kinematics anyway, and the
reported result does not rest on the knee predicting itself.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.run_gait120_residual_fusion as fusion


VERSION = "GAIT120_KINEMATIC_INPUT_CHECK_V1"

KINEMATIC_INPUTS = ("knee_history", "surrounding_body")


def thigh_euler_deg(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Thigh orientation as roll/pitch/yaw in degrees."""
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees(np.column_stack([roll, pitch, yaw]))


def _kinematic_channels(data: dict[str, np.ndarray], kinematic_input: str) -> np.ndarray:
    if kinematic_input == "knee_history":
        return np.asarray(data["knee_flexion_deg"], dtype=np.float32).reshape(-1, 1)
    if kinematic_input != "surrounding_body":
        raise ValueError(f"Unknown kinematic input {kinematic_input!r}")
    # Thigh pitch plus full thigh orientation.  No channel here observes the knee.
    return np.column_stack(
        [
            np.asarray(data["thigh_pitch_deg"], dtype=np.float32).reshape(-1),
            thigh_euler_deg(np.asarray(data["thigh_quat_wxyz"], dtype=np.float64)),
        ]
    ).astype(np.float32)


def build_examples(
    cache_dir: Path, subjects: list[int], kinematic_input: str
) -> dict[str, np.ndarray]:
    """Windowed examples matching the confirmation protocol, with one input swapped."""
    windows: list[np.ndarray] = []
    y_standardized: list[float] = []
    y_deg: list[float] = []
    subject_number: list[int] = []
    trial_index: list[int] = []
    target_mean: list[float] = []
    target_std: list[float] = []

    for number in subjects:
        data = dict(np.load(Path(cache_dir) / f"S{number:03d}.npz", allow_pickle=False))
        trial = np.asarray(data["trial_index"], dtype=np.int16).reshape(-1)
        frame = np.asarray(data["frame_index"], dtype=np.int16).reshape(-1)
        emg = np.asarray(data["emg_frame_native_value"], dtype=np.float32)
        knee = np.asarray(data["knee_flexion_deg"], dtype=np.float32).reshape(-1)
        kinematic = _kinematic_channels(data, kinematic_input)

        # Scaling values come from the calibration trials only, as in the
        # confirmation run; later trials never inform their own normalisation.
        calibration = np.flatnonzero(np.isin(trial, gait.TRAIN_TRIALS))
        if calibration.size < 1:
            raise RuntimeError(f"S{number:03d} has no calibration rows")
        emg_mean = emg[calibration].mean(axis=0)
        emg_std = np.maximum(emg[calibration].std(axis=0), 1.0e-8)
        knee_mean = float(knee[calibration].mean())
        knee_std = max(float(knee[calibration].std()), 1.0e-8)
        kinematic_mean = kinematic[calibration].mean(axis=0)
        kinematic_std = np.maximum(kinematic[calibration].std(axis=0), 1.0e-8)

        for trial_number in gait.TRAIN_TRIALS + gait.TEST_TRIALS:
            rows = np.flatnonzero(trial == trial_number)
            if rows.size < fusion.KINEMATIC_FRAMES + gait.HORIZON_FRAMES + 1:
                raise RuntimeError(f"S{number:03d} trial {trial_number} is too short")
            if not np.array_equal(frame[rows], np.arange(rows.size)):
                raise RuntimeError("Cached trial frame indices are not consecutive")
            for end in range(
                fusion.KINEMATIC_FRAMES - 1, rows.size - gait.HORIZON_FRAMES
            ):
                window = rows[end - fusion.KINEMATIC_FRAMES + 1 : end + 1]
                target = float(knee[rows[end + gait.HORIZON_FRAMES]])
                windows.append(
                    np.concatenate(
                        [
                            (emg[window] - emg_mean) / emg_std,
                            (kinematic[window] - kinematic_mean) / kinematic_std,
                        ],
                        axis=1,
                    ).astype(np.float32)
                )
                y_standardized.append((target - knee_mean) / knee_std)
                y_deg.append(target)
                subject_number.append(number)
                trial_index.append(trial_number)
                target_mean.append(knee_mean)
                target_std.append(knee_std)

    return {
        "x": np.stack(windows),
        "y_standardized": np.asarray(y_standardized, dtype=np.float64),
        "y_deg": np.asarray(y_deg, dtype=np.float64),
        "subject_number": np.asarray(subject_number, dtype=np.int64),
        "trial_index": np.asarray(trial_index, dtype=np.int64),
        "target_mean_deg": np.asarray(target_mean, dtype=np.float64),
        "target_std_deg": np.asarray(target_std, dtype=np.float64),
    }


def _features(examples: dict[str, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = examples["x"][rows]
    kinematic = x[:, :, gait.N_EMG :].reshape(rows.size, -1).astype(np.float64)
    emg = x[:, -fusion.EMG_FRAMES :, : gait.N_EMG].reshape(rows.size, -1).astype(np.float64)
    return kinematic, emg


def _participant_rmse(
    examples: dict[str, np.ndarray], rows: np.ndarray, standardized: np.ndarray
) -> dict[str, float]:
    prediction = (
        standardized * examples["target_std_deg"][rows] + examples["target_mean_deg"][rows]
    )
    truth = examples["y_deg"][rows]
    subjects = examples["subject_number"][rows]
    return {
        f"S{int(s):03d}": float(
            np.sqrt(np.mean(np.square(prediction[subjects == s] - truth[subjects == s])))
        )
        for s in np.unique(subjects)
    }


def evaluate(
    cache_dir: Path,
    subjects: list[int],
    kinematic_input: str,
    *,
    kinematic_alpha: float,
    residual_alpha: float,
    gamma: float,
) -> dict[str, Any]:
    examples = build_examples(cache_dir, subjects, kinematic_input)
    fit_rows = np.flatnonzero(np.isin(examples["trial_index"], gait.TRAIN_TRIALS))
    test_rows = np.flatnonzero(np.isin(examples["trial_index"], gait.TEST_TRIALS))

    kinematic_fit, emg_fit = _features(examples, fit_rows)
    kinematic_test, emg_test = _features(examples, test_rows)
    y_fit = examples["y_standardized"][fit_rows]
    subjects_fit = examples["subject_number"][fit_rows]

    kinematic_model = fusion._fit_ridge(
        kinematic_fit, y_fit, subjects_fit, alpha=kinematic_alpha
    )
    base_fit = fusion._predict_standardized(kinematic_model, kinematic_fit)
    residual_model = fusion._fit_ridge(
        emg_fit, y_fit - base_fit, subjects_fit, alpha=residual_alpha
    )

    residual = y_fit - base_fit
    caps = {
        int(s): max(float(np.quantile(np.abs(residual[subjects_fit == s]), 0.95)), 1.0e-6)
        for s in np.unique(subjects_fit)
    }
    cap_test = np.asarray(
        [caps[int(s)] for s in examples["subject_number"][test_rows]], dtype=np.float64
    )

    base_test = fusion._predict_standardized(kinematic_model, kinematic_test)
    raw = fusion._predict_standardized(residual_model, emg_test)
    fused = base_test + float(gamma) * (cap_test * np.tanh(raw / cap_test))

    no_emg = _participant_rmse(examples, test_rows, base_test)
    with_emg = _participant_rmse(examples, test_rows, fused)
    improvement = np.asarray(
        [no_emg[s] - with_emg[s] for s in sorted(no_emg)], dtype=np.float64
    )
    no_emg_mean = float(np.mean(list(no_emg.values())))
    return {
        "kinematic_input": kinematic_input,
        "kinematic_channels": int(examples["x"].shape[2] - gait.N_EMG),
        "observes_the_knee": kinematic_input == "knee_history",
        "no_emg_mean_participant_rmse_deg": no_emg_mean,
        "fused_mean_participant_rmse_deg": float(np.mean(list(with_emg.values()))),
        "mean_semg_improvement_deg": float(np.mean(improvement)),
        "semg_share_of_baseline": float(np.mean(improvement) / no_emg_mean),
        "improved_participants": int(np.sum(improvement > 0.0)),
        "participant_count": int(improvement.size),
        "participants": {"no_emg": no_emg, "fused": with_emg},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--confirmation-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--first-subject", type=int, default=31)
    parser.add_argument("--last-subject", type=int, default=120)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")

    protocol_path = args.confirmation_run_dir.resolve() / "protocol.json"
    chosen = (
        json.loads(protocol_path.read_text(encoding="utf-8")).get("model") or {}
    ).get("confirmation_hyperparameters")
    if not isinstance(chosen, dict):
        raise RuntimeError(f"{protocol_path} does not record confirmation_hyperparameters")

    gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
    subjects = list(range(int(args.first_subject), int(args.last_subject) + 1))
    results = {
        name: evaluate(
            args.cache_dir.resolve(),
            subjects,
            name,
            kinematic_alpha=float(chosen["kinematic_alpha"]),
            residual_alpha=float(chosen["residual_alpha"]),
            gamma=float(chosen["gamma"]),
        )
        for name in KINEMATIC_INPUTS
    }

    knee = results["knee_history"]
    body = results["surrounding_body"]
    penalty = (
        body["no_emg_mean_participant_rmse_deg"]
        - knee["no_emg_mean_participant_rmse_deg"]
    )
    verdict = {
        "kinematic_penalty_for_dropping_knee_deg": float(penalty),
        # If removing every knee-derived input costs little, the reported result
        # is not resting on the knee predicting its own future.
        "result_depends_on_knee_autoregression": bool(penalty > 0.5),
        "note": (
            "Penalty is the held-out RMSE cost of a kinematic stage that observes "
            "only the surrounding body instead of knee-angle history, with the "
            "sEMG correction switched off in both."
        ),
    }

    run_dir.mkdir(parents=True, exist_ok=False)
    gait._atomic_json(
        run_dir / "kinematic_input_check.json",
        {
            "version": VERSION,
            "subjects": subjects,
            "hyperparameters": chosen,
            "verdict": verdict,
            "conditions": results,
            "software": {"python": platform.python_version(), "numpy": np.__version__},
        },
    )

    print(f"{'kinematic input':<24}{'no sEMG':>10}{'with sEMG':>11}{'sEMG gain':>11}{'share':>8}")
    for name in KINEMATIC_INPUTS:
        row = results[name]
        print(
            f"{name:<24}{row['no_emg_mean_participant_rmse_deg']:10.3f}"
            f"{row['fused_mean_participant_rmse_deg']:11.3f}"
            f"{row['mean_semg_improvement_deg']:+11.4f}"
            f"{row['semg_share_of_baseline']:8.1%}"
        )
    print(
        f"\nDropping every knee-derived input costs {penalty:+.3f} deg. "
        f"Depends on knee autoregression: "
        f"{'YES' if verdict['result_depends_on_knee_autoregression'] else 'no'}"
    )
    print(f"Wrote {run_dir / 'kinematic_input_check.json'}")


if __name__ == "__main__":
    main()
