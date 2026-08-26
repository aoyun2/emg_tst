from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import emg_tst.gait120_experiment as gait


VERSION = "GAIT120_BOUNDED_RIDGE_RESIDUAL_FUSION_100MS_V2"
KINEMATIC_FRAMES = 60
EMG_FRAMES = 15
ALPHA_GRID = np.asarray(
    [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0],
    dtype=np.float64,
)
GAMMA_GRID = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(type(value).__name__)


@dataclass(frozen=True)
class RidgeModel:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    coefficient: np.ndarray
    intercept: float
    alpha: float


def _features(examples: gait.ExampleSet) -> tuple[np.ndarray, np.ndarray]:
    if examples.x.shape[1:] != (KINEMATIC_FRAMES, gait.N_EMG + 1):
        raise RuntimeError(f"Unexpected example shape {examples.x.shape}")
    kinematic = np.asarray(examples.x[:, :, gait.N_EMG], dtype=np.float64)
    emg = np.asarray(
        examples.x[:, -EMG_FRAMES:, : gait.N_EMG], dtype=np.float64
    ).reshape(len(examples), EMG_FRAMES * gait.N_EMG)
    if not np.all(np.isfinite(kinematic)) or not np.all(np.isfinite(emg)):
        raise RuntimeError("Non-finite residual-fusion features")
    return kinematic, emg


def _participant_balanced_weights(subject_number: np.ndarray) -> np.ndarray:
    subjects, counts = np.unique(subject_number, return_counts=True)
    count_by_subject = {int(subject): int(count) for subject, count in zip(subjects, counts)}
    n = int(subject_number.size)
    weights = np.asarray(
        [n / (subjects.size * count_by_subject[int(subject)]) for subject in subject_number],
        dtype=np.float64,
    )
    if not np.isclose(np.mean(weights), 1.0, atol=1.0e-12):
        raise RuntimeError("Participant-balanced weights are malformed")
    return weights


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    subject_number: np.ndarray,
    *,
    alpha: float,
) -> RidgeModel:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    weights = _participant_balanced_weights(subject_number)
    weight_sum = float(np.sum(weights))
    mean = np.sum(weights[:, None] * x, axis=0) / weight_sum
    centered = x - mean[None, :]
    variance = np.sum(weights[:, None] * np.square(centered), axis=0) / weight_sum
    std = np.sqrt(np.maximum(variance, 0.0))
    std[std < 1.0e-8] = 1.0
    z = centered / std[None, :]
    y_mean = float(np.sum(weights * y) / weight_sum)
    y_centered = y - y_mean
    root_weight = np.sqrt(weights)
    zw = z * root_weight[:, None]
    yw = y_centered * root_weight
    gram = zw.T @ zw
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    coefficient = np.linalg.solve(gram, zw.T @ yw)
    return RidgeModel(
        feature_mean=mean,
        feature_std=std,
        coefficient=coefficient,
        intercept=y_mean,
        alpha=float(alpha),
    )


def _predict_standardized(model: RidgeModel, x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - model.feature_mean[None, :]) / model.feature_std[
        None, :
    ]
    return np.asarray(model.intercept + z @ model.coefficient, dtype=np.float64)


def _training_correction_caps(
    train: gait.ExampleSet,
    kinematic_prediction: np.ndarray,
) -> dict[int, float]:
    residual = np.asarray(train.y_standardized, dtype=np.float64) - np.asarray(
        kinematic_prediction, dtype=np.float64
    )
    caps: dict[int, float] = {}
    for subject in np.unique(train.subject_number).tolist():
        keep = train.subject_number == subject
        cap = float(np.quantile(np.abs(residual[keep]), 0.95))
        caps[int(subject)] = max(cap, 1.0e-6)
    return caps


def _bounded_correction(
    raw_correction: np.ndarray,
    examples: gait.ExampleSet,
    caps: dict[int, float],
) -> np.ndarray:
    cap = np.asarray(
        [caps[int(subject)] for subject in examples.subject_number], dtype=np.float64
    )
    return cap * np.tanh(np.asarray(raw_correction, dtype=np.float64) / cap)


def _to_degrees(examples: gait.ExampleSet, standardized: np.ndarray) -> np.ndarray:
    return np.asarray(
        standardized * examples.target_std_deg + examples.target_mean_deg,
        dtype=np.float32,
    )


def _score(examples: gait.ExampleSet, standardized: np.ndarray) -> tuple[float, dict[str, Any]]:
    metrics = gait._participant_metrics(examples, _to_degrees(examples, standardized))
    return float(metrics["mean_participant_rmse_deg"]), metrics


def _save_model(path: Path, model: RidgeModel) -> None:
    gait._atomic_npz(
        path,
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
        coefficient=model.coefficient,
        intercept=np.asarray(model.intercept, dtype=np.float64),
        alpha=np.asarray(model.alpha, dtype=np.float64),
    )


def _select_development(
    train: gait.ExampleSet,
    validation: gait.ExampleSet,
    *,
    status,
) -> tuple[RidgeModel, RidgeModel, float, dict[int, float], dict[str, Any]]:
    train_k, train_e = _features(train)
    validation_k, validation_e = _features(validation)
    kinematic_rows: list[dict[str, Any]] = []
    kinematic_candidates: list[tuple[float, float, RidgeModel, np.ndarray]] = []
    for alpha in ALPHA_GRID.tolist():
        model = _fit_ridge(
            train_k, train.y_standardized, train.subject_number, alpha=alpha
        )
        prediction = _predict_standardized(model, validation_k)
        score, _ = _score(validation, prediction)
        kinematic_rows.append({"alpha": alpha, "validation_rmse_deg": score})
        kinematic_candidates.append((score, -alpha, model, prediction))
        status("selecting_kinematic_penalty", alpha=alpha, validation_rmse_deg=score)
    _, _, kinematic_model, validation_base = min(
        kinematic_candidates, key=lambda item: (item[0], item[1])
    )

    training_base = _predict_standardized(kinematic_model, train_k)
    training_residual = np.asarray(train.y_standardized, dtype=np.float64) - training_base
    correction_caps = _training_correction_caps(train, training_base)
    residual_rows: list[dict[str, Any]] = []
    residual_candidates: list[tuple[float, float, float, RidgeModel, np.ndarray]] = []
    for alpha in ALPHA_GRID.tolist():
        model = _fit_ridge(
            train_e, training_residual, train.subject_number, alpha=alpha
        )
        correction = _bounded_correction(
            _predict_standardized(model, validation_e), validation, correction_caps
        )
        for gamma in GAMMA_GRID.tolist():
            prediction = validation_base + gamma * correction
            score, _ = _score(validation, prediction)
            residual_rows.append(
                {
                    "alpha": alpha,
                    "gamma": gamma,
                    "validation_rmse_deg": score,
                }
            )
            residual_candidates.append(
                (score, gamma, -alpha, model, prediction)
            )
        status("selecting_residual_penalty", alpha=alpha)
    _, gamma, _, residual_model, validation_fused = min(
        residual_candidates, key=lambda item: (item[0], item[1], item[2])
    )
    _, validation_no_emg_metrics = _score(validation, validation_base)
    _, validation_fused_metrics = _score(validation, validation_fused)
    selection = {
        "kinematic_alpha": kinematic_model.alpha,
        "residual_alpha": residual_model.alpha,
        "gamma": float(gamma),
        "kinematic_grid": kinematic_rows,
        "residual_grid": residual_rows,
        "validation_no_emg": validation_no_emg_metrics,
        "validation_fused": validation_fused_metrics,
        "training_residual_cap_quantile": 0.95,
        "training_residual_caps_standardized": {
            f"S{subject:03d}": cap for subject, cap in sorted(correction_caps.items())
        },
    }
    return kinematic_model, residual_model, float(gamma), correction_caps, selection


def _fit_fixed(
    train: gait.ExampleSet,
    *,
    kinematic_alpha: float,
    residual_alpha: float,
) -> tuple[RidgeModel, RidgeModel, dict[int, float]]:
    train_k, train_e = _features(train)
    kinematic = _fit_ridge(
        train_k,
        train.y_standardized,
        train.subject_number,
        alpha=kinematic_alpha,
    )
    base = _predict_standardized(kinematic, train_k)
    residual = np.asarray(train.y_standardized, dtype=np.float64) - base
    correction_caps = _training_correction_caps(train, base)
    emg = _fit_ridge(
        train_e, residual, train.subject_number, alpha=residual_alpha
    )
    return kinematic, emg, correction_caps


def _evaluate(
    examples: gait.ExampleSet,
    *,
    kinematic: RidgeModel,
    residual: RidgeModel,
    gamma: float,
    correction_caps: dict[int, float],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    k, e = _features(examples)
    no_emg_standardized = _predict_standardized(kinematic, k)
    correction = _bounded_correction(
        _predict_standardized(residual, e), examples, correction_caps
    )
    fused_standardized = no_emg_standardized + float(gamma) * correction
    no_emg_prediction = _to_degrees(examples, no_emg_standardized)
    fused_prediction = _to_degrees(examples, fused_standardized)
    no_emg = gait._participant_metrics(examples, no_emg_prediction)
    fused = gait._participant_metrics(examples, fused_prediction)
    return fused, no_emg, fused_prediction, no_emg_prediction


def _protocol(
    *,
    phase: str,
    cache_dir: Path,
    subjects: list[int],
    root: Path,
    selected: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        "version": VERSION,
        "phase": phase,
        "dataset": {
            "name": "Gait120",
            "activity": "LevelWalking only",
            "subjects": [f"S{number:03d}" for number in subjects],
            "train_trials": [1, 2, 3],
            "validation_trials": [4],
            "test_trials": [5],
            "cache_dir": str(cache_dir),
            "cache_version": gait.CACHE_VERSION,
            "interpolation": "none",
        },
        "prediction": {
            "forecast_ms": 100,
            "forecast_frames": 10,
            "kinematic_history_ms": 600,
            "kinematic_history_frames": KINEMATIC_FRAMES,
            "emg_history_ms": 150,
            "emg_history_frames": EMG_FRAMES,
        },
        "model": {
            "name": "participant-balanced ridge residual fusion",
            "kinematic_stage": "ridge autoregression on recorded knee-angle history",
            "residual_stage": "ridge prediction of kinematic training residual from causal sEMG envelopes",
            "fusion": "kinematic prediction plus gamma times sEMG residual correction",
            "residual_safety_bound": "participant trial-1-to-3 absolute kinematic residual 95th percentile, applied with tanh to model output",
            "alpha_grid": ALPHA_GRID,
            "gamma_grid": GAMMA_GRID,
            "sample_weighting": "equal total weight per participant",
            "development_selection": "mean participant trial-4 RMSE",
            "confirmation_hyperparameters": selected,
        },
        "gate": {
            "unit": "participant",
            "effect": "no-sEMG trial-5 RMSE minus fused trial-5 RMSE",
            "mean_strictly_positive": True,
            "bootstrap_95pct_lower_strictly_positive": True,
            "two_sided_randomization_p_max": 0.05,
            "strict_majority_positive": True,
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "code_sha256": {
            "runner": gait._sha256(Path(__file__).resolve()),
            "native_loader": gait._sha256(root / "emg_tst" / "gait120_data.py"),
            "cache_export": gait._sha256(root / "emg_tst" / "preprocess_gait120.py"),
            "example_builder": gait._sha256(root / "emg_tst" / "gait120_experiment.py"),
            "experiment_protocol": gait._sha256(root / "docs" / "EXPERIMENT_PROTOCOL.md"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("smoke", "development", "confirmation"), required=True
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--development-run-dir", type=Path)
    parser.add_argument("--first-subject", type=int)
    parser.add_argument("--last-subject", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir.resolve()
    run_dir = args.run_dir.resolve()
    selected: dict[str, float] | None = None
    if args.phase == "development":
        subjects = list(range(1, 31))
    elif args.phase == "confirmation":
        subjects = list(range(31, 121))
        if args.development_run_dir is None:
            raise ValueError("Confirmation requires --development-run-dir")
        development_run_dir = args.development_run_dir.resolve()
        summary_path = development_run_dir / "ablation_summary.json"
        selection_path = development_run_dir / "selected_hyperparameters.json"
        if not summary_path.exists() or not selection_path.exists():
            raise RuntimeError("Passing development artifacts are missing")
        if not bool(json.loads(summary_path.read_text(encoding="utf-8")).get("passed")):
            raise RuntimeError("Development did not pass; confirmation is locked")
        stored = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = {
            "kinematic_alpha": float(stored["kinematic_alpha"]),
            "residual_alpha": float(stored["residual_alpha"]),
            "gamma": float(stored["gamma"]),
        }
    else:
        first = 1 if args.first_subject is None else int(args.first_subject)
        last = first if args.last_subject is None else int(args.last_subject)
        subjects = list(range(first, last + 1))

    gait.INPUT_FRAMES = KINEMATIC_FRAMES
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(
        phase=args.phase,
        cache_dir=cache_dir,
        subjects=subjects,
        root=root,
        selected=selected,
    )
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected = json.loads(json.dumps(protocol, default=_json_default))
        if existing != expected:
            raise RuntimeError("Existing run protocol differs from executable protocol")
    else:
        gait._atomic_json(protocol_path, protocol)
    status_path = run_dir / "pipeline_status.json"

    def status(stage: str, **extra: Any) -> None:
        gait._atomic_json(
            status_path,
            {
                "version": VERSION,
                "phase": args.phase,
                "stage": stage,
                "updated_unix": time.time(),
                **extra,
            },
        )

    status("loading_cached_native_data")
    train, validation, test, scalers = gait._build_examples(cache_dir, subjects)
    gait._atomic_json(
        run_dir / "data_audit.json",
        {
            "subjects": [f"S{number:03d}" for number in subjects],
            "training_examples": len(train),
            "validation_examples": len(validation),
            "test_examples": len(test),
            "example_shape": list(train.x.shape[1:]),
            "kinematic_features": KINEMATIC_FRAMES,
            "emg_features": EMG_FRAMES * gait.N_EMG,
            "scalers": scalers,
            "trial_leakage": False,
        },
    )

    if args.phase in ("development", "smoke"):
        kinematic, residual, gamma, correction_caps, selection = _select_development(
            train, validation, status=status
        )
        selected = {
            "kinematic_alpha": float(kinematic.alpha),
            "residual_alpha": float(residual.alpha),
            "gamma": float(gamma),
        }
        selection.update(selected)
        gait._atomic_json(run_dir / "selection_audit.json", selection)
        gait._atomic_json(run_dir / "selected_hyperparameters.json", selected)
    else:
        assert selected is not None
        kinematic, residual, correction_caps = _fit_fixed(
            train,
            kinematic_alpha=selected["kinematic_alpha"],
            residual_alpha=selected["residual_alpha"],
        )
        gamma = selected["gamma"]

    gait._atomic_json(
        run_dir / "training_residual_caps.json",
        {
            "quantile": 0.95,
            "standardized_caps": {
                f"S{subject:03d}": cap
                for subject, cap in sorted(correction_caps.items())
            },
        },
    )

    _save_model(run_dir / "kinematic_model.npz", kinematic)
    _save_model(run_dir / "emg_residual_model.npz", residual)
    validation_fused, validation_no_emg, _, _ = _evaluate(
        validation,
        kinematic=kinematic,
        residual=residual,
        gamma=gamma,
        correction_caps=correction_caps,
    )
    fused_metrics, no_emg_metrics, fused_prediction, no_emg_prediction = _evaluate(
        test,
        kinematic=kinematic,
        residual=residual,
        gamma=gamma,
        correction_caps=correction_caps,
    )
    fused_result = {
        "condition": "fused",
        "validation": validation_fused,
        "test": fused_metrics,
    }
    no_emg_result = {
        "condition": "no_emg",
        "validation": validation_no_emg,
        "test": no_emg_metrics,
    }
    gait._atomic_json(run_dir / "fused" / "result.json", fused_result)
    gait._atomic_json(run_dir / "no_emg" / "result.json", no_emg_result)
    for condition, prediction in (
        ("fused", fused_prediction),
        ("no_emg", no_emg_prediction),
    ):
        gait._atomic_npz(
            run_dir / condition / "test_predictions.npz",
            prediction_deg=prediction,
            target_deg=test.y_deg,
            subject_number=test.subject_number,
            trial_index=test.trial_index,
            input_end_frame=test.input_end_frame,
            target_frame=test.target_frame,
        )

    require_gate = args.phase in ("development", "confirmation")
    summary = gait._paired_statistics(
        fused_result, no_emg_result, require_gate=require_gate
    )
    summary["phase"] = args.phase
    summary["selected_hyperparameters"] = selected
    gait._atomic_json(run_dir / "ablation_summary.json", summary)
    status(
        "complete",
        passed=summary["passed"],
        mean_improvement_deg=summary["mean_improvement_deg"],
        bootstrap_95pct_ci_deg=summary["bootstrap_95pct_ci_deg"],
        two_sided_randomization_p=summary["two_sided_randomization_p"],
        positive_participants=summary["positive_participants"],
        participant_count=summary["participant_count"],
    )
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
