from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import emg_tst.gait120_experiment as gait


VERSION = "GAIT120_MODEL_BLIND_PHYSICS_PANEL_V1"
SEED = 42
WINDOW_FRAMES = 100
WINDOW_HZ = 100
WINDOW_STRIDE_FRAMES = 10
TARGET_WINDOWS = 80
CHECKPOINT_LABELS = (
    "fraction_005pct",
    "fraction_010pct",
    "fraction_020pct",
    "fraction_040pct",
    "fraction_060pct",
    "fraction_080pct",
    "fraction_100pct",
)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _checkpoint_predictions(accuracy_dir: Path, label: str) -> dict[str, np.ndarray]:
    path = accuracy_dir / "checkpoints" / label / "test_predictions.npz"
    with np.load(path, allow_pickle=False) as stored:
        return {key: np.asarray(stored[key]) for key in stored.files}


def _subject_rows(data: dict[str, np.ndarray], subject: int) -> np.ndarray:
    rows = np.flatnonzero(
        (np.asarray(data["subject_number"]).reshape(-1) == subject)
        & (np.asarray(data["trial_index"]).reshape(-1) == 5)
    )
    if rows.size < 1:
        raise RuntimeError(f"Checkpoint predictions lack S{subject:03d} trial 5")
    return rows


def _select_panel(candidates: dict[int, list[int]]) -> list[tuple[int, int]]:
    rng = np.random.default_rng(SEED)
    participants = np.asarray(sorted(candidates), dtype=np.int64)
    participant_order = rng.permutation(participants).tolist()
    shuffled: dict[int, list[int]] = {}
    for subject in participant_order:
        starts = np.asarray(candidates[int(subject)], dtype=np.int64)
        shuffled[int(subject)] = rng.permutation(starts).astype(int).tolist()

    selected: list[tuple[int, int]] = []
    round_index = 0
    while len(selected) < TARGET_WINDOWS:
        added = 0
        for subject in participant_order:
            starts = shuffled[int(subject)]
            if round_index < len(starts):
                selected.append((int(subject), int(starts[round_index])))
                added += 1
                if len(selected) == TARGET_WINDOWS:
                    break
        if added == 0:
            break
        round_index += 1
    if len(selected) != TARGET_WINDOWS:
        raise RuntimeError(
            f"Model-blind trial-5 pool provides {len(selected)}/{TARGET_WINDOWS} windows"
        )
    first_round_subjects = {subject for subject, _ in selected[: len(participants)]}
    if len(first_round_subjects) != len(participants):
        raise RuntimeError("Participant-balanced first selection round is malformed")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--accuracy-path-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cache_dir = args.cache_dir.resolve()
    accuracy_dir = args.accuracy_path_dir.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise RuntimeError(f"Physics run directory already exists: {run_dir}")

    predictions = {
        label: _checkpoint_predictions(accuracy_dir, label)
        for label in CHECKPOINT_LABELS
    }
    reference = predictions[CHECKPOINT_LABELS[-1]]
    metadata_fields = (
        "target_deg",
        "subject_number",
        "trial_index",
        "input_end_frame",
        "target_frame",
    )
    for label, data in predictions.items():
        for field in metadata_fields:
            if not np.array_equal(data[field], reference[field]):
                raise RuntimeError(f"{label} has different {field} metadata")

    candidates: dict[int, list[int]] = {}
    subject_prediction_rows: dict[int, dict[int, int]] = {}
    for subject in range(31, 121):
        rows = _subject_rows(reference, subject)
        frames = np.asarray(reference["target_frame"], dtype=np.int64)[rows]
        if not np.array_equal(frames, np.arange(frames[0], frames[-1] + 1)):
            raise RuntimeError(f"S{subject:03d} trial-5 prediction frames are not consecutive")
        row_by_frame = {int(frame): int(row) for frame, row in zip(frames, rows)}
        subject_prediction_rows[subject] = row_by_frame
        starts = [
            int(start)
            for start in range(
                int(frames[0]),
                int(frames[-1]) - WINDOW_FRAMES + 2,
                WINDOW_STRIDE_FRAMES,
            )
            if all((start + offset) in row_by_frame for offset in range(WINDOW_FRAMES))
        ]
        if starts:
            candidates[subject] = starts

    selected = _select_panel(candidates)
    run_dir.mkdir(parents=True, exist_ok=False)
    query_dir = run_dir / "queries"
    rows_out: list[dict[str, Any]] = []
    for panel_index, (subject, start) in enumerate(selected):
        subject_name = f"S{subject:03d}"
        cache_path = cache_dir / f"{subject_name}.npz"
        with np.load(cache_path, allow_pickle=False) as stored:
            trial = np.asarray(stored["trial_index"], dtype=np.int16)
            frame = np.asarray(stored["frame_index"], dtype=np.int16)
            cache_rows = np.flatnonzero(
                (trial == 5) & (frame >= start) & (frame < start + WINDOW_FRAMES)
            )
            if cache_rows.size != WINDOW_FRAMES:
                raise RuntimeError(f"{subject_name} query {start} lacks recorded frames")
            if not np.array_equal(frame[cache_rows], np.arange(start, start + WINDOW_FRAMES)):
                raise RuntimeError("Recorded physics-query frames are not consecutive")
            measured_flexion = np.asarray(
                stored["knee_flexion_deg"], dtype=np.float32
            )[cache_rows]
            thigh_pitch = np.asarray(stored["thigh_pitch_deg"], dtype=np.float32)[
                cache_rows
            ]
            thigh_quat = np.asarray(
                stored["thigh_quat_wxyz"], dtype=np.float32
            )[cache_rows]
            motion_time = np.asarray(stored["motion_time_s"], dtype=np.float64)[cache_rows]

        prediction_rows = np.asarray(
            [subject_prediction_rows[subject][frame] for frame in range(start, start + WINDOW_FRAMES)],
            dtype=np.int64,
        )
        if not np.allclose(
            np.asarray(reference["target_deg"], dtype=np.float32)[prediction_rows],
            measured_flexion,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError("Model targets do not reproduce cached recorded knee angles")
        fused = np.stack(
            [
                np.asarray(predictions[label]["fused_prediction_deg"], dtype=np.float32)[
                    prediction_rows
                ]
                for label in CHECKPOINT_LABELS
            ]
        )
        no_emg = np.stack(
            [
                np.asarray(predictions[label]["no_emg_prediction_deg"], dtype=np.float32)[
                    prediction_rows
                ]
                for label in CHECKPOINT_LABELS
            ]
        )
        query_id = f"{subject_name}_trial05_start{start:04d}"
        path = query_dir / f"{query_id}.npz"
        _atomic_npz(
            path,
            version=np.asarray(VERSION),
            query_id=np.asarray(query_id),
            panel_index=np.asarray(panel_index, dtype=np.int16),
            subject_number=np.asarray(subject, dtype=np.int16),
            trial_index=np.asarray(5, dtype=np.int16),
            start_frame=np.asarray(start, dtype=np.int16),
            sample_hz=np.asarray(WINDOW_HZ, dtype=np.int16),
            frame_index=np.arange(start, start + WINDOW_FRAMES, dtype=np.int16),
            motion_time_s=motion_time,
            knee_flexion_deg=measured_flexion,
            knee_included_deg=(180.0 - measured_flexion).astype(np.float32),
            thigh_pitch_deg=thigh_pitch,
            thigh_quat_wxyz=thigh_quat,
            checkpoint_labels=np.asarray(CHECKPOINT_LABELS),
            fused_prediction_deg=fused,
            no_emg_prediction_deg=no_emg,
        )
        rows_out.append(
            {
                "panel_index": panel_index,
                "query_id": query_id,
                "subject": subject_name,
                "trial": 5,
                "start_frame": start,
                "window_frames": WINDOW_FRAMES,
                "query_npz": str(path.resolve()),
            }
        )

    protocol = {
        "version": VERSION,
        "selection": {
            "dataset": "Gait120",
            "activity": "LevelWalking only",
            "participant_group": "confirmation S031-S120",
            "trial": 5,
            "seed": SEED,
            "window_hz": WINDOW_HZ,
            "window_frames": WINDOW_FRAMES,
            "window_seconds": WINDOW_FRAMES / WINDOW_HZ,
            "candidate_stride_frames": WINDOW_STRIDE_FRAMES,
            "target_windows": TARGET_WINDOWS,
            "eligible_participants": len(candidates),
            "candidate_windows": int(sum(len(value) for value in candidates.values())),
            "algorithm": (
                "seeded participant-balanced rounds; one eligible window per participant "
                "before any participant contributes a second"
            ),
            "model_outputs_used_for_selection": False,
            "motion_match_or_simulation_used_for_selection": False,
        },
        "checkpoint_labels": list(CHECKPOINT_LABELS),
        "alignment": {
            "forecast_assignment": "prediction assigned to its recorded target_frame",
            "predictor_interpolation": False,
            "recorded_frames_synthesized": False,
        },
        "sha256": {
            "panel_runner": gait._sha256(Path(__file__).resolve()),
            "accuracy_path_summary": gait._sha256(
                accuracy_dir / "accuracy_path_summary.json"
            ),
            "experiment_protocol": gait._sha256(root / "docs" / "EXPERIMENT_PROTOCOL.md"),
        },
    }
    gait._atomic_json(run_dir / "panel_protocol.json", protocol)
    gait._atomic_json(
        run_dir / "panel_manifest.json",
        {"protocol": protocol, "windows": rows_out},
    )
    print(json.dumps({"protocol": protocol, "selected_windows": rows_out}, indent=2))


if __name__ == "__main__":
    main()
