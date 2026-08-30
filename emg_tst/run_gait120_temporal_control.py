from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.run_gait120_residual_fusion as fusion


VERSION = "GAIT120_RESIDUAL_FUSION_TEMPORAL_CONTROL_500MS_V1"
LAG_FRAMES = 50


def _load_model(path: Path) -> fusion.RidgeModel:
    with np.load(path, allow_pickle=False) as stored:
        return fusion.RidgeModel(
            feature_mean=np.asarray(stored["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(stored["feature_std"], dtype=np.float64),
            coefficient=np.asarray(stored["coefficient"], dtype=np.float64),
            intercept=float(np.asarray(stored["intercept"]).reshape(())),
            alpha=float(np.asarray(stored["alpha"]).reshape(())),
        )


def _subset(examples: gait.ExampleSet, index: np.ndarray) -> gait.ExampleSet:
    return gait.ExampleSet(
        **{
            field: np.asarray(getattr(examples, field))[index]
            for field in gait.ExampleSet.__dataclass_fields__
        }
    )


def _lagged_emg_pair(
    examples: gait.ExampleSet,
) -> tuple[gait.ExampleSet, gait.ExampleSet]:
    keys = [
        (int(subject), int(trial), int(frame))
        for subject, trial, frame in zip(
            examples.subject_number,
            examples.trial_index,
            examples.input_end_frame,
        )
    ]
    row_by_key = {key: row for row, key in enumerate(keys)}
    if len(row_by_key) != len(keys):
        raise RuntimeError("Example identifiers are not unique")

    # An example ending at frame e spans [e - INPUT_FRAMES + 1, e], and the
    # lagged history is the EMG_FRAMES frames ending at e - LAG_FRAMES. Those
    # frames are the leading EMG_FRAMES of the example ending this far back:
    source_offset = LAG_FRAMES - (gait.INPUT_FRAMES - fusion.EMG_FRAMES)
    if source_offset < 0:
        raise RuntimeError("The lag is shorter than the unused part of the window")

    current_rows: list[int] = []
    source_rows: list[int] = []
    for row, (subject, trial, frame) in enumerate(keys):
        source = row_by_key.get((subject, trial, frame - source_offset))
        if source is not None:
            current_rows.append(row)
            source_rows.append(source)
    if not current_rows:
        raise RuntimeError("No examples have a complete 500-ms earlier sEMG history")

    current_index = np.asarray(current_rows, dtype=np.int64)
    source_index = np.asarray(source_rows, dtype=np.int64)
    current = _subset(examples, current_index)
    lagged_x = np.asarray(current.x, dtype=np.float32).copy()
    # Only the trailing EMG_FRAMES are read as features, so only those are
    # replaced, with the leading frames of the source window.
    lagged_x[:, -fusion.EMG_FRAMES :, : gait.N_EMG] = np.asarray(
        examples.x[source_index, : fusion.EMG_FRAMES, : gait.N_EMG], dtype=np.float32
    )
    lagged = gait.ExampleSet(
        x=lagged_x,
        y_standardized=current.y_standardized,
        y_deg=current.y_deg,
        target_mean_deg=current.target_mean_deg,
        target_std_deg=current.target_std_deg,
        subject_number=current.subject_number,
        trial_index=current.trial_index,
        input_end_frame=current.input_end_frame,
        target_frame=current.target_frame,
    )
    # Check the frame the history actually ends on, not the offset between the
    # two example indices. The lagged features are the leading EMG_FRAMES of the
    # source window, so they end at source_end - INPUT_FRAMES + EMG_FRAMES.
    source_end = np.asarray(examples.input_end_frame, dtype=np.int64)[source_index]
    lagged_last_frame = source_end - gait.INPUT_FRAMES + fusion.EMG_FRAMES
    observed_lag = np.asarray(current.input_end_frame, dtype=np.int64) - lagged_last_frame
    if not np.all(observed_lag == LAG_FRAMES):
        raise RuntimeError("Temporal-control lag is not exactly 500 ms")
    return current, lagged


def _condition_result(
    name: str,
    validation: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, Any]:
    return {"condition": name, "validation": validation, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--confirmation-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir.resolve()
    confirmation_dir = args.confirmation_run_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise RuntimeError(f"Temporal-control run directory already exists: {run_dir}")

    primary_summary_path = confirmation_dir / "ablation_summary.json"
    primary_protocol_path = confirmation_dir / "protocol.json"
    primary_summary = json.loads(primary_summary_path.read_text(encoding="utf-8"))
    if not bool(primary_summary.get("passed")):
        raise RuntimeError("Primary confirmation did not pass; temporal control is locked")
    selected = primary_summary["selected_hyperparameters"]
    gamma = float(selected["gamma"])
    residual_alpha = float(selected["residual_alpha"])

    protocol = {
        "version": VERSION,
        "purpose": "secondary temporal falsification; cannot rescue primary confirmation",
        "subjects": [f"S{number:03d}" for number in range(31, 121)],
        "trials": {"train": [1, 2, 3], "validation": [4], "test": [5]},
        "lag": {
            "frames": LAG_FRAMES,
            "milliseconds": 10 * LAG_FRAMES,
            "direction": "sEMG history ends earlier than the aligned history",
            "scope": "within participant and continuous recorded trial",
            "wraparound": False,
            "interpolation": False,
            "missing_history": "discard and rescore aligned model on identical common-support rows",
        },
        "model": {
            "kinematic_stage": "exact fitted primary-confirmation model",
            "residual_stage": "refit on lagged trial-1-to-3 sEMG with fixed primary residual penalty",
            "kinematic_alpha": float(selected["kinematic_alpha"]),
            "residual_alpha": residual_alpha,
            "gamma": gamma,
            "residual_safety_bound": "exact primary participant trial-1-to-3 caps",
        },
        "interpretation": {
            "reproduces_primary_ablation": "lagged condition passes the same four participant-level criteria on common-support test rows",
            "direct_contrast": "lagged participant RMSE minus aligned participant RMSE",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "sha256": {
            "control_runner": gait._sha256(Path(__file__).resolve()),
            "fusion_runner": gait._sha256(root / "emg_tst" / "run_gait120_residual_fusion.py"),
            "example_builder": gait._sha256(root / "emg_tst" / "gait120_experiment.py"),
            "experiment_protocol": gait._sha256(root / "docs" / "EXPERIMENT_PROTOCOL.md"),
            "primary_summary": gait._sha256(primary_summary_path),
            "primary_protocol": gait._sha256(primary_protocol_path),
            "primary_kinematic_model": gait._sha256(confirmation_dir / "kinematic_model.npz"),
            "primary_residual_model": gait._sha256(confirmation_dir / "emg_residual_model.npz"),
        },
    }
    gait._atomic_json(run_dir / "protocol.json", protocol)

    gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
    train, validation, test, _ = gait._build_examples(cache_dir, list(range(31, 121)))
    train_current, train_lagged = _lagged_emg_pair(train)
    validation_current, validation_lagged = _lagged_emg_pair(validation)
    test_current, test_lagged = _lagged_emg_pair(test)

    kinematic = _load_model(confirmation_dir / "kinematic_model.npz")
    aligned_residual = _load_model(confirmation_dir / "emg_residual_model.npz")
    caps_payload = json.loads(
        (confirmation_dir / "training_residual_caps.json").read_text(encoding="utf-8")
    )
    correction_caps = {
        int(subject[1:]): float(cap)
        for subject, cap in caps_payload["standardized_caps"].items()
    }

    train_k, train_lagged_e = fusion._features(train_lagged)
    train_base = fusion._predict_standardized(kinematic, train_k)
    train_residual = np.asarray(train_current.y_standardized, dtype=np.float64) - train_base
    lagged_residual = fusion._fit_ridge(
        train_lagged_e,
        train_residual,
        train_current.subject_number,
        alpha=residual_alpha,
    )
    fusion._save_model(run_dir / "lagged_emg_residual_model.npz", lagged_residual)

    # The lagged model is fitted on the rows that have a 500 ms earlier history.
    # Refit the aligned model on the same rows so the contrast between them is a
    # contrast in input timing and not in how much data each one saw.
    _, train_current_e = fusion._features(train_current)
    aligned_residual = fusion._fit_ridge(
        train_current_e,
        train_residual,
        train_current.subject_number,
        alpha=residual_alpha,
    )
    fusion._save_model(run_dir / "aligned_emg_residual_model.npz", aligned_residual)

    aligned_validation, no_emg_validation, _, _ = fusion._evaluate(
        validation_current,
        kinematic=kinematic,
        residual=aligned_residual,
        gamma=gamma,
        correction_caps=correction_caps,
    )
    lagged_validation, lagged_validation_no_emg, _, _ = fusion._evaluate(
        validation_lagged,
        kinematic=kinematic,
        residual=lagged_residual,
        gamma=gamma,
        correction_caps=correction_caps,
    )
    if lagged_validation_no_emg != no_emg_validation:
        raise RuntimeError("Common-support no-sEMG validation metrics changed")

    aligned_test, no_emg_test, aligned_prediction, no_emg_prediction = fusion._evaluate(
        test_current,
        kinematic=kinematic,
        residual=aligned_residual,
        gamma=gamma,
        correction_caps=correction_caps,
    )
    lagged_test, lagged_test_no_emg, lagged_prediction, lagged_no_emg_prediction = (
        fusion._evaluate(
            test_lagged,
            kinematic=kinematic,
            residual=lagged_residual,
            gamma=gamma,
            correction_caps=correction_caps,
        )
    )
    if not np.array_equal(no_emg_prediction, lagged_no_emg_prediction):
        raise RuntimeError("Common-support no-sEMG test predictions changed")
    if lagged_test_no_emg != no_emg_test:
        raise RuntimeError("Common-support no-sEMG test metrics changed")

    aligned_result = _condition_result("aligned_sEMG", aligned_validation, aligned_test)
    lagged_result = _condition_result("lagged_sEMG_500ms", lagged_validation, lagged_test)
    no_emg_result = _condition_result("no_sEMG", no_emg_validation, no_emg_test)
    gait._atomic_json(run_dir / "aligned_result.json", aligned_result)
    gait._atomic_json(run_dir / "lagged_result.json", lagged_result)
    gait._atomic_json(run_dir / "no_emg_result.json", no_emg_result)

    aligned_summary = gait._paired_statistics(
        aligned_result, no_emg_result, require_gate=True
    )
    aligned_summary["definition"] = (
        "common-support no-sEMG participant RMSE minus aligned-sEMG participant RMSE"
    )
    lagged_summary = gait._paired_statistics(
        lagged_result, no_emg_result, require_gate=True
    )
    lagged_summary["definition"] = (
        "common-support no-sEMG participant RMSE minus 500-ms-lagged-sEMG participant RMSE"
    )
    timing_contrast = gait._paired_statistics(
        aligned_result, lagged_result, require_gate=False
    )
    timing_contrast["definition"] = (
        "500-ms-lagged-sEMG participant RMSE minus aligned-sEMG participant RMSE"
    )
    outcome = {
        "aligned_common_support": aligned_summary,
        "lagged_control": lagged_summary,
        "aligned_vs_lagged": timing_contrast,
        "lagged_reproduces_primary_ablation": bool(lagged_summary["passed"]),
        "common_support": {
            "training_examples": len(train_current),
            "validation_examples": len(validation_current),
            "test_examples": len(test_current),
            "participant_count": len(np.unique(test_current.subject_number)),
        },
    }
    gait._atomic_json(run_dir / "temporal_control_summary.json", outcome)
    gait._atomic_npz(
        run_dir / "test_predictions.npz",
        aligned_prediction_deg=aligned_prediction,
        lagged_prediction_deg=lagged_prediction,
        no_emg_prediction_deg=no_emg_prediction,
        target_deg=test_current.y_deg,
        subject_number=test_current.subject_number,
        trial_index=test_current.trial_index,
        input_end_frame=test_current.input_end_frame,
        target_frame=test_current.target_frame,
    )
    print(json.dumps(outcome, indent=2, default=gait._json_default))


if __name__ == "__main__":
    main()
