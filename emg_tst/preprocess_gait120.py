from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from emg_tst.gait120_data import (
    EMG_CHANNELS,
    EMG_SAMPLES_PER_FRAME,
    MOTION_HZ,
    audit_native_alignment,
    load_level_walking_subject_causal_raw,
)


CACHE_VERSION = "GAIT120_CAUSAL_RAW_LEVEL_WALKING_V2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def convert_subject(subject_dir: Path, cache_dir: Path) -> dict[str, Any]:
    subject_dir = Path(subject_dir).resolve()
    cache_dir = Path(cache_dir).resolve()
    subject = subject_dir.name
    output_path = cache_dir / f"{subject}.npz"
    audit_path = cache_dir / "audits" / f"{subject}.json"
    if output_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if bool(audit.get("passed")) and audit.get("cache_version") == CACHE_VERSION:
            return audit

    alignment = audit_native_alignment(subject_dir)
    trials = load_level_walking_subject_causal_raw(subject_dir)
    if len(trials) < 5:
        raise RuntimeError(f"{subject} has only {len(trials)} usable level-walking trials")

    trial_index: list[np.ndarray] = []
    frame_index: list[np.ndarray] = []
    time_s: list[np.ndarray] = []
    knee_flexion: list[np.ndarray] = []
    knee_included: list[np.ndarray] = []
    thigh_pitch: list[np.ndarray] = []
    thigh_quat: list[np.ndarray] = []
    emg_at_frame: list[np.ndarray] = []
    for trial in trials:
        n = trial.n_frames
        native = np.asarray(trial.emg_native, dtype=np.float32).reshape(
            n, EMG_SAMPLES_PER_FRAME, len(EMG_CHANNELS)
        )
        # Select the final causally processed value in each exact synchronized
        # 20-sample frame block. No interpolation or generated time point is
        # introduced.
        decimated = native[:, -1, :]
        trial_index.append(np.full(n, trial.trial, dtype=np.int16))
        frame_index.append(np.arange(n, dtype=np.int16))
        time_s.append(trial.motion_time_s.astype(np.float64))
        knee_flexion.append(trial.knee_flexion_deg.astype(np.float32))
        knee_included.append(trial.knee_included_deg.astype(np.float32))
        thigh_pitch.append(trial.thigh_pitch_deg.astype(np.float32))
        thigh_quat.append(trial.thigh_quat_wxyz.astype(np.float32))
        emg_at_frame.append(decimated.astype(np.float32))

    arrays = {
        "cache_version": np.asarray(CACHE_VERSION),
        "subject": np.asarray(subject),
        "motion_hz": np.asarray(MOTION_HZ, dtype=np.int16),
        "emg_source_hz": np.asarray(2_000, dtype=np.int16),
        "emg_names": np.asarray(EMG_CHANNELS),
        "trial_index": np.concatenate(trial_index),
        "frame_index": np.concatenate(frame_index),
        "motion_time_s": np.concatenate(time_s),
        "knee_flexion_deg": np.concatenate(knee_flexion),
        "knee_included_deg": np.concatenate(knee_included),
        "thigh_pitch_deg": np.concatenate(thigh_pitch),
        "thigh_quat_wxyz": np.concatenate(thigh_quat, axis=0),
        "emg_frame_native_value": np.concatenate(emg_at_frame, axis=0),
    }
    if not np.all(np.isfinite(arrays["emg_frame_native_value"])):
        raise RuntimeError(f"{subject} contains non-finite native sEMG")
    if not np.all(np.isfinite(arrays["knee_flexion_deg"])):
        raise RuntimeError(f"{subject} contains non-finite knee angles")
    _atomic_npz(output_path, **arrays)

    audit = {
        **alignment,
        "cache_version": CACHE_VERSION,
        "cache_path": str(output_path),
        "cache_sha256": _sha256(output_path),
        "source_raw_sha256": _sha256(subject_dir / "EMG" / "RawData.mat"),
        "source_processed_sha256": _sha256(subject_dir / "EMG" / "ProcessedData.mat"),
        "level_walking_trials": [int(trial.trial) for trial in trials],
        "prediction_emg_value": (
            "final value in each exact 20-sample block after causal 20-500 Hz "
            "Butterworth filtering, rectification, and trailing 250-sample RMS"
        ),
    }
    _atomic_json(audit_path, audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--first-subject", type=int, default=1)
    parser.add_argument("--last-subject", type=int, default=120)
    args = parser.parse_args()

    audits: list[dict[str, Any]] = []
    for subject_number in range(args.first_subject, args.last_subject + 1):
        subject = f"S{subject_number:03d}"
        subject_dir = args.data_root / subject
        if not subject_dir.exists():
            raise FileNotFoundError(f"Missing extracted Gait120 subject: {subject_dir}")
        audit = convert_subject(subject_dir, args.cache_dir)
        audits.append(audit)
        print(
            json.dumps(
                {
                    "subject": subject,
                    "passed": bool(audit["passed"]),
                    "cache_path": audit["cache_path"],
                }
            ),
            flush=True,
        )
    _atomic_json(
        args.cache_dir / "cache_manifest.json",
        {
            "version": CACHE_VERSION,
            "first_subject": args.first_subject,
            "last_subject": args.last_subject,
            "subjects": audits,
        },
    )


if __name__ == "__main__":
    main()
