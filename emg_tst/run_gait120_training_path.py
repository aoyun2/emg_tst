"""Fit the residual-fusion training path and export per-checkpoint predictions.

This replaces the earlier data-availability path, which refit the closed-form
model on nested fractions of the calibration data.  That design produced almost
no accuracy spread: a linear model fitted to 5% of 90 participants' level
walking is already close to its converged error, so every checkpoint landed in
the same narrow RMSE band and the correlation analysis had nothing to resolve.

Descending from a zero initialization instead traverses the full range between
predicting the participant mean and the converged fit, which is the spread the
checkpoint correlation analysis needs.

Outputs are written in the layout ``prepare_gait120_physics_panel`` already
reads::

    <run-dir>/checkpoints/<label>/test_predictions.npz
    <run-dir>/checkpoints/<label>/summary.json
    <run-dir>/checkpoints/manifest.json
    <run-dir>/training_path.npz
    <run-dir>/protocol.json
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

import emg_tst.gait120_experiment as gait
import emg_tst.gait120_training_path as tpath
import emg_tst.run_gait120_residual_fusion as fusion


VERSION = "GAIT120_RESIDUAL_FUSION_TRAINING_PATH_RUN_V1"

# Above this relative distance, the last descent rung sits far enough from the
# closed-form terminal rung that the ladder has a visible gap at its converged
# end.  Reported as a warning rather than a failure: the terminal rung is exact
# either way, so the run is still usable.
MAX_CLOSED_FORM_GAP = 1.0e-3


def _selected_hyperparameters(confirmation_dir: Path) -> tuple[float, float, float]:
    """Read the penalties the confirmation run was locked to.

    The path must reuse the confirmation model's hyperparameters exactly.
    Reselecting them here would let each checkpoint tune itself, which would
    confound accuracy level with model selection.
    """
    confirmation_dir = Path(confirmation_dir)
    summary_path = confirmation_dir / "ablation_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Confirmation summary not found: {summary_path}")
    if not bool(json.loads(summary_path.read_text(encoding="utf-8")).get("passed")):
        raise RuntimeError(
            f"Confirmation run at {confirmation_dir} did not pass its sEMG gate; "
            "the training path is only defined for a passing model."
        )

    protocol_path = confirmation_dir / "protocol.json"
    if not protocol_path.exists():
        raise RuntimeError(f"Confirmation protocol not found: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    chosen = (protocol.get("model") or {}).get("confirmation_hyperparameters")
    if not isinstance(chosen, dict):
        raise RuntimeError(
            f"{protocol_path} does not record confirmation_hyperparameters; "
            "re-run emg_tst.run_gait120_residual_fusion --phase confirmation."
        )
    try:
        return (
            float(chosen["kinematic_alpha"]),
            float(chosen["residual_alpha"]),
            float(chosen["gamma"]),
        )
    except KeyError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Confirmation hyperparameters are missing {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--confirmation-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-mode",
        choices=tpath.CHECKPOINT_MODES,
        default="rmse_ladder",
        help="How to space checkpoints along the descent (default: rmse_ladder).",
    )
    parser.add_argument("--checkpoints", type=int, default=tpath.N_CHECKPOINTS)
    parser.add_argument("--max-steps", type=int, default=tpath.MAX_STEPS)
    parser.add_argument("--first-subject", type=int, default=31)
    parser.add_argument("--last-subject", type=int, default=120)
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise RuntimeError(f"Training-path run directory already exists: {run_dir}")

    kinematic_alpha, residual_alpha, gamma = _selected_hyperparameters(
        args.confirmation_run_dir.resolve()
    )
    gait.INPUT_FRAMES = fusion.KINEMATIC_FRAMES

    subjects = list(range(int(args.first_subject), int(args.last_subject) + 1))
    train, _validation, test, _scalers = gait._build_examples(cache_dir, subjects)
    print(
        f"[training-path] examples: train={len(train)} test={len(test)} "
        f"subjects={len(subjects)}"
    )

    path = tpath.fit_training_path(
        train,
        kinematic_alpha=kinematic_alpha,
        residual_alpha=residual_alpha,
        max_steps=int(args.max_steps),
        checkpoint_mode=str(args.checkpoint_mode),
        n_checkpoints=int(args.checkpoints),
    )
    print(
        f"[training-path] descent ran {path.converged_step} steps "
        f"(train RMSE {path.train_rmse_deg[0]:.2f} -> {path.train_rmse_deg[-1]:.2f} deg, "
        f"final-rung gap to closed form {path.closed_form_gap:.2e})"
    )
    if path.closed_form_gap > MAX_CLOSED_FORM_GAP:
        print(
            f"[training-path] WARNING: the last descent rung is {path.closed_form_gap:.2e} "
            "from the closed-form terminal rung, so the ladder has a gap at its "
            "converged end. Raise --max-steps to close it."
        )

    picks = path.checkpoint_indices
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_root = run_dir / "checkpoints"

    labels: list[str] = []
    rows: list[dict[str, Any]] = []
    for order, recorded_index in enumerate(picks.tolist()):
        step = int(path.steps[recorded_index])
        label = tpath.checkpoint_label(order, step)
        labels.append(label)

        kinematic, residual, caps = tpath.materialize_checkpoint(path, order, train)
        fused_metrics, no_emg_metrics, fused_deg, no_emg_deg = fusion._evaluate(
            test,
            kinematic=kinematic,
            residual=residual,
            gamma=gamma,
            correction_caps=caps,
        )

        out_dir = checkpoint_root / label
        gait._atomic_npz(
            out_dir / "test_predictions.npz",
            target_deg=np.asarray(test.y_deg, dtype=np.float32),
            subject_number=np.asarray(test.subject_number, dtype=np.int16),
            trial_index=np.asarray(test.trial_index, dtype=np.int16),
            input_end_frame=np.asarray(test.input_end_frame, dtype=np.int32),
            target_frame=np.asarray(test.target_frame, dtype=np.int32),
            fused_prediction_deg=np.asarray(fused_deg, dtype=np.float32),
            no_emg_prediction_deg=np.asarray(no_emg_deg, dtype=np.float32),
        )

        participant_rmse = np.asarray(
            [row["rmse_deg"] for row in fused_metrics["participants"]], dtype=np.float64
        )
        row = {
            "label": label,
            "order": int(order),
            "step": step,
            "recorded_index": int(recorded_index),
            "train_rmse_deg": float(path.train_rmse_deg[recorded_index]),
            "fused_mean_participant_rmse_deg": float(
                fused_metrics["mean_participant_rmse_deg"]
            ),
            "no_emg_mean_participant_rmse_deg": float(
                no_emg_metrics["mean_participant_rmse_deg"]
            ),
            # Spread diagnostics.  A null association at a converged checkpoint
            # means something different when the participants barely differ in
            # accuracy, so the spread is recorded alongside the level.
            "fused_participant_rmse_sd_deg": float(np.std(participant_rmse, ddof=1)),
            "fused_participant_rmse_iqr_deg": float(
                np.subtract(*np.percentile(participant_rmse, [75, 25]))
            ),
            "fused_participant_rmse_range_deg": [
                float(np.min(participant_rmse)),
                float(np.max(participant_rmse)),
            ],
        }
        rows.append(row)
        gait._atomic_json(
            out_dir / "summary.json",
            {"checkpoint": row, "fused": fused_metrics, "no_emg": no_emg_metrics},
        )
        print(
            f"[training-path] {label}: train={row['train_rmse_deg']:.2f} "
            f"test={row['fused_mean_participant_rmse_deg']:.3f} deg "
            f"(sd {row['fused_participant_rmse_sd_deg']:.3f})"
        )

    gait._atomic_json(
        checkpoint_root / "manifest.json",
        {
            "version": VERSION,
            "training_path_version": tpath.VERSION,
            "checkpoint_mode": str(args.checkpoint_mode),
            "labels": labels,
            # The converged end of the path is the confirmation model, so it is
            # the checkpoint the headline simulation result is read from.
            "primary_checkpoint": labels[-1],
            "checkpoints": rows,
        },
    )
    gait._atomic_npz(
        run_dir / "training_path.npz",
        steps=path.steps,
        train_rmse_deg=path.train_rmse_deg,
    )
    gait._atomic_json(
        run_dir / "protocol.json",
        {
            "version": VERSION,
            "cache_dir": str(cache_dir),
            "confirmation_run_dir": str(args.confirmation_run_dir.resolve()),
            "subjects": subjects,
            "kinematic_alpha": kinematic_alpha,
            "residual_alpha": residual_alpha,
            "gamma": gamma,
            "max_steps": int(args.max_steps),
            "descent_steps": int(path.converged_step),
            "checkpoint_steps": [int(path.steps[i]) for i in picks.tolist()],
            "final_rung_gap_to_closed_form": float(path.closed_form_gap),
            "train_rmse_first_deg": float(path.train_rmse_deg[0]),
            "train_rmse_last_deg": float(path.train_rmse_deg[-1]),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    )
    print(f"[training-path] wrote {len(labels)} checkpoints to {checkpoint_root}")


if __name__ == "__main__":
    main()
