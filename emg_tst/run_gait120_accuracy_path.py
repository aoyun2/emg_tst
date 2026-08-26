from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.run_gait120_residual_fusion as fusion


VERSION = "GAIT120_RESIDUAL_FUSION_DATA_AVAILABILITY_PATH_V1"
TRAINING_FRACTIONS = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)


def _subset(examples: gait.ExampleSet, index: np.ndarray) -> gait.ExampleSet:
    return gait.ExampleSet(
        **{
            field: np.asarray(getattr(examples, field))[index]
            for field in gait.ExampleSet.__dataclass_fields__
        }
    )


def _participant_prefix(examples: gait.ExampleSet, fraction: float) -> gait.ExampleSet:
    selected: list[np.ndarray] = []
    for subject in np.unique(examples.subject_number).tolist():
        rows = np.flatnonzero(examples.subject_number == subject)
        count = max(1, int(math.ceil(float(fraction) * rows.size)))
        selected.append(rows[:count])
    index = np.concatenate(selected).astype(np.int64)
    if np.any(np.diff(index) <= 0):
        raise RuntimeError("Participant-prefix rows do not preserve recorded order")
    return _subset(examples, index)


def _condition(name: str, validation: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
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
        raise RuntimeError(f"Accuracy-path run directory already exists: {run_dir}")

    primary_summary_path = confirmation_dir / "ablation_summary.json"
    primary_summary = json.loads(primary_summary_path.read_text(encoding="utf-8"))
    if not bool(primary_summary.get("passed")):
        raise RuntimeError("Primary confirmation did not pass; accuracy path is locked")
    selected = primary_summary["selected_hyperparameters"]
    kinematic_alpha = float(selected["kinematic_alpha"])
    residual_alpha = float(selected["residual_alpha"])
    gamma = float(selected["gamma"])

    protocol = {
        "version": VERSION,
        "purpose": "prespecified secondary RMSE-versus-physics accuracy path",
        "subjects": [f"S{number:03d}" for number in range(31, 121)],
        "trials": {"train": [1, 2, 3], "validation": [4], "test": [5]},
        "training_fractions": list(TRAINING_FRACTIONS),
        "fraction_definition": (
            "first ceil(fraction*n) eligible trial-1-to-3 examples within every participant, "
            "preserving recorded order; no test-based checkpoint selection"
        ),
        "model": {
            "kinematic_alpha": kinematic_alpha,
            "residual_alpha": residual_alpha,
            "gamma": gamma,
            "participant_weighting": "equal total weight among participants at every fraction",
            "standardization": "unchanged participant trial-1-to-3 input standardization",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "sha256": {
            "runner": gait._sha256(Path(__file__).resolve()),
            "fusion_runner": gait._sha256(root / "emg_tst" / "run_gait120_residual_fusion.py"),
            "example_builder": gait._sha256(root / "emg_tst" / "gait120_experiment.py"),
            "experiment_protocol": gait._sha256(root / "docs" / "EXPERIMENT_PROTOCOL.md"),
            "primary_summary": gait._sha256(primary_summary_path),
        },
    }
    gait._atomic_json(run_dir / "protocol.json", protocol)

    gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES
    train, validation, test, _ = gait._build_examples(
        cache_dir, list(range(31, 121))
    )
    checkpoints: list[dict[str, Any]] = []
    for fraction in TRAINING_FRACTIONS:
        label = f"fraction_{int(round(100 * fraction)):03d}pct"
        checkpoint_dir = run_dir / "checkpoints" / label
        fraction_train = _participant_prefix(train, fraction)
        kinematic, residual, correction_caps = fusion._fit_fixed(
            fraction_train,
            kinematic_alpha=kinematic_alpha,
            residual_alpha=residual_alpha,
        )
        validation_fused, validation_no_emg, _, _ = fusion._evaluate(
            validation,
            kinematic=kinematic,
            residual=residual,
            gamma=gamma,
            correction_caps=correction_caps,
        )
        test_fused, test_no_emg, fused_prediction, no_emg_prediction = fusion._evaluate(
            test,
            kinematic=kinematic,
            residual=residual,
            gamma=gamma,
            correction_caps=correction_caps,
        )
        fused_result = _condition(label + "_fused", validation_fused, test_fused)
        no_emg_result = _condition(label + "_no_emg", validation_no_emg, test_no_emg)
        ablation = gait._paired_statistics(
            fused_result, no_emg_result, require_gate=False
        )
        ablation["definition"] = "no-sEMG participant RMSE minus fused participant RMSE"

        fusion._save_model(checkpoint_dir / "kinematic_model.npz", kinematic)
        fusion._save_model(checkpoint_dir / "emg_residual_model.npz", residual)
        gait._atomic_json(
            checkpoint_dir / "training_residual_caps.json",
            {
                "quantile": 0.95,
                "standardized_caps": {
                    f"S{subject:03d}": cap
                    for subject, cap in sorted(correction_caps.items())
                },
            },
        )
        gait._atomic_json(
            checkpoint_dir / "result.json",
            {
                "label": label,
                "training_fraction": fraction,
                "training_examples": len(fraction_train),
                "fused": fused_result,
                "no_emg": no_emg_result,
                "ablation": ablation,
            },
        )
        gait._atomic_npz(
            checkpoint_dir / "test_predictions.npz",
            fused_prediction_deg=fused_prediction,
            no_emg_prediction_deg=no_emg_prediction,
            target_deg=test.y_deg,
            subject_number=test.subject_number,
            trial_index=test.trial_index,
            input_end_frame=test.input_end_frame,
            target_frame=test.target_frame,
        )
        checkpoints.append(
            {
                "label": label,
                "training_fraction": fraction,
                "training_examples": len(fraction_train),
                "fused_mean_participant_rmse_deg": test_fused[
                    "mean_participant_rmse_deg"
                ],
                "no_emg_mean_participant_rmse_deg": test_no_emg[
                    "mean_participant_rmse_deg"
                ],
                "mean_ablation_improvement_deg": ablation["mean_improvement_deg"],
                "test_predictions": str(checkpoint_dir / "test_predictions.npz"),
            }
        )

    full_path = run_dir / "checkpoints" / "fraction_100pct" / "test_predictions.npz"
    with np.load(full_path, allow_pickle=False) as generated, np.load(
        confirmation_dir / "fused" / "test_predictions.npz", allow_pickle=False
    ) as primary_fused, np.load(
        confirmation_dir / "no_emg" / "test_predictions.npz", allow_pickle=False
    ) as primary_no_emg:
        if not np.array_equal(
            generated["fused_prediction_deg"], primary_fused["prediction_deg"]
        ):
            raise RuntimeError("100% fused checkpoint does not reproduce primary confirmation")
        if not np.array_equal(
            generated["no_emg_prediction_deg"], primary_no_emg["prediction_deg"]
        ):
            raise RuntimeError("100% no-sEMG checkpoint does not reproduce primary confirmation")

    summary = {
        "version": VERSION,
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
        "full_checkpoint_reproduces_confirmation_exactly": True,
    }
    gait._atomic_json(run_dir / "accuracy_path_summary.json", summary)
    print(json.dumps(summary, indent=2, default=gait._json_default))


if __name__ == "__main__":
    main()
