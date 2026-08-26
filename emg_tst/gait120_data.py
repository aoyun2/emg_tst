from __future__ import annotations

import argparse
import io
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
import scipy.signal

from emg_tst.gt_dataset import hip_angles_to_quat_wxyz


MCOS_MAGIC = 0xDD000000
MOTION_HZ = 100
EMG_HZ = 2_000
EMG_SAMPLES_PER_FRAME = EMG_HZ // MOTION_HZ
EMG_CHANNELS = (
    "VastusLateralis",
    "RectusFemoris",
    "VastusMedialis",
    "TibialisAnterior",
    "BicepsFemoris",
    "Semitendinosus",
    "GastrocnemuisMedialis",  # Spelling in the released MATLAB table.
    "GastrocnemiusLateralis",
    "SoleusMedialis",
    "SoleusLateralis",
    "PeroneusLongus",
    "PeroneusBrevis",
)


def _matlab_opaque_metadata(value: Any) -> np.ndarray:
    """Return the uint32 metadata stored in a SciPy MatlabOpaque value."""

    try:
        names = tuple(value.dtype.names or ())
        if "_ObjectMetadata" in names:
            payload = value["_ObjectMetadata"].item()
        else:
            item = value.item()
            payload = item[3] if len(item) >= 4 else item[2]
        metadata = np.asarray(payload, dtype=np.uint32).reshape(-1)
    except Exception as exc:  # pragma: no cover - malformed external input
        raise ValueError("Expected a scalar MATLAB MCOS opaque object") from exc
    if metadata.size < 6 or int(metadata[0]) != MCOS_MAGIC:
        raise ValueError("Invalid MATLAB MCOS object metadata")
    if int(metadata[1]) != 2 or tuple(metadata[2:4].tolist()) != (1, 1):
        raise ValueError(f"Expected a scalar MATLAB table, received metadata {metadata.tolist()}")
    return metadata


def _workspace_payload(mat_bytes: bytes, short_workspace: np.ndarray) -> np.ndarray:
    """Load the first FileWrapper matrix hidden by SciPy's duplicate-name handling.

    MATLAB writes two inner variables named ``__function_workspace__``. SciPy
    exposes only the short second variable. The long first variable contains the
    MCOS table data. This function reads exactly that first matrix and does not
    execute MATLAB or unsigned native extensions.
    """

    wrapper = np.asarray(short_workspace, dtype=np.uint8).tobytes()
    if len(wrapper) < 16:
        raise ValueError("MATLAB function workspace is truncated")
    matrix_type, matrix_bytes = struct.unpack_from("<II", wrapper, 8)
    if matrix_type != 14 or matrix_bytes <= 0 or 16 + matrix_bytes > len(wrapper):
        raise ValueError("Unexpected MATLAB function-workspace layout")

    header = bytearray(mat_bytes[:128])
    if len(header) != 128:
        raise ValueError("MATLAB file header is truncated")
    # The reconstructed stream contains one ordinary matrix rather than a
    # subsystem at a later file offset.
    header[116:124] = bytes(8)
    stream = io.BytesIO(bytes(header) + wrapper[8 : 16 + matrix_bytes])
    loaded = scipy.io.loadmat(stream, squeeze_me=False, struct_as_record=False)
    outer = loaded["__function_workspace__"].item()
    mcos = outer.MCOS
    names = tuple(mcos.dtype.names or ())
    if "arr" in names:
        return np.asarray(mcos["arr"].item(), dtype=object)
    # SciPy 1.17 exposes the same FileWrapper payload through the standard
    # MatlabOpaque field name instead of the historical synthetic ``arr``
    # field. Both branches return the identical MATLAB object-metadata cell
    # array; no signal values or indices are transformed here.
    if "_ObjectMetadata" in names:
        return np.asarray(mcos["_ObjectMetadata"].item(), dtype=object)
    raise ValueError(f"Unsupported SciPy MATLAB opaque layout: {names}")


class _MatlabTableStore:
    """Minimal, pure-Python reader for the table objects used by Gait120."""

    def __init__(self, mat_path: Path):
        self.path = Path(mat_path)
        raw = self.path.read_bytes()
        self.top = scipy.io.loadmat(
            io.BytesIO(raw), squeeze_me=True, struct_as_record=False
        )
        self._filewrapper = _workspace_payload(raw, self.top["__function_workspace__"])
        self._load_metadata()

    def _load_metadata(self) -> None:
        metadata = np.asarray(self._filewrapper[0, 0], dtype=np.uint8).reshape(-1).tobytes()
        if len(metadata) < 40:
            raise ValueError(f"FileWrapper metadata is truncated in {self.path}")
        integers = np.frombuffer(metadata, dtype="<i4")
        version = int(integers[0])
        if version not in (2, 3, 4):
            raise ValueError(f"Unsupported MATLAB FileWrapper version {version}")
        offsets = integers[2:10].astype(np.int64)
        if offsets.size != 8 or np.any(np.diff(offsets) < 0):
            raise ValueError("Invalid MATLAB FileWrapper region offsets")
        self.version = version
        self.names = [
            part.decode("ascii")
            for part in metadata[40 : int(offsets[0])].split(b"\x00")
            if part
        ]

        def region(first: int, second: int) -> np.ndarray:
            start = int(offsets[first])
            stop = int(offsets[second])
            return np.frombuffer(
                metadata, dtype="<i4", count=(stop - start) // 4, offset=start
            )

        self.class_metadata = region(0, 1)
        self.saveobj_metadata = region(1, 2)
        self.object_metadata = region(2, 3)
        self.normal_metadata = region(3, 4)
        if version == 2:
            self.saved_values = self._filewrapper[2:-1, 0]
        elif version == 3:
            self.saved_values = self._filewrapper[2:-2, 0]
        else:
            self.saved_values = self._filewrapper[2:-3, 0]

    @staticmethod
    def _property_triples(metadata: np.ndarray, object_type_id: int) -> np.ndarray:
        offset = int(metadata[0])
        for _ in range(int(object_type_id)):
            count = int(metadata[offset])
            offset += 1 + 3 * count
            offset += offset % 2
        count = int(metadata[offset])
        return metadata[offset + 1 : offset + 1 + 3 * count].reshape(count, 3)

    def table(self, opaque: Any) -> tuple[tuple[str, ...], np.ndarray]:
        metadata = _matlab_opaque_metadata(opaque)
        object_id = int(metadata[4])
        first = object_id * 6
        record = self.object_metadata[first : first + 6]
        if record.size != 6:
            raise ValueError(f"Unknown MATLAB object id {object_id}")
        class_id, unknown1, unknown2, save_id, normal_id, _dependency_id = (
            int(value) for value in record
        )
        if unknown1 != 0 or unknown2 != 0:
            raise ValueError("Unsupported MATLAB table object metadata")
        class_name_index = int(self.class_metadata[class_id * 4 + 1])
        class_name = self.names[class_name_index - 1]
        if class_name != "table":
            raise ValueError(f"Expected MATLAB table, received {class_name!r}")

        property_metadata = self.saveobj_metadata if save_id else self.normal_metadata
        property_type_id = save_id or normal_id
        properties: dict[str, Any] = {}
        for name_index, property_type, value_index in self._property_triples(
            property_metadata, property_type_id
        ):
            # All Gait120 table fields are stored as ordinary values. Refuse to
            # guess if a future release changes the encoding.
            if int(property_type) != 1:
                raise ValueError("Unsupported non-value MATLAB table property")
            name = self.names[int(name_index) - 1]
            properties[name] = self.saved_values[int(value_index)]

        if "data" not in properties or "varnames" not in properties:
            raise ValueError("MATLAB table is missing data or variable names")
        names = tuple(
            str(np.asarray(value).reshape(-1)[0])
            for value in np.asarray(properties["varnames"], dtype=object).reshape(-1)
        )
        columns = [
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in np.asarray(properties["data"], dtype=object).reshape(-1)
        ]
        if len(names) != len(columns) or not columns:
            raise ValueError("MATLAB table column metadata is inconsistent")
        lengths = {int(column.size) for column in columns}
        if len(lengths) != 1:
            raise ValueError("MATLAB table columns have unequal lengths")
        values = np.column_stack(columns).astype(np.float64, copy=False)
        return names, values


def _read_opensim_mot(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    try:
        end_header = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exc:
        raise ValueError(f"OpenSim file is missing endheader: {path}") from exc
    names = tuple(lines[end_header + 1].split())
    rows = [
        [float(value) for value in line.split()]
        for line in lines[end_header + 2 :]
        if line.strip()
    ]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"Malformed OpenSim table: {path}")
    return names, values


@dataclass(frozen=True)
class Gait120Trial:
    subject: str
    trial: int
    motion_time_s: np.ndarray
    knee_flexion_deg: np.ndarray
    knee_included_deg: np.ndarray
    thigh_pitch_deg: np.ndarray
    thigh_quat_wxyz: np.ndarray
    emg_native: np.ndarray
    emg_names: tuple[str, ...]
    step_frame_counts: tuple[int, ...]

    @property
    def n_frames(self) -> int:
        return int(self.motion_time_s.size)


def load_level_walking_subject(
    subject_dir: Path,
    *,
    emg_field: str = "EMGs_norm",
) -> list[Gait120Trial]:
    """Load native, synchronized level-walking trials without interpolation.

    ``EMGs_norm`` is the authors' released MVC-normalized native-rate signal.
    ``EMGs_interpolated`` is deliberately rejected because it time-normalizes
    every step.
    """

    subject_dir = Path(subject_dir)
    subject = subject_dir.name
    if emg_field == "EMGs_interpolated":
        raise ValueError("Time-normalized/interpolated Gait120 EMG is not permitted")
    processed = _MatlabTableStore(subject_dir / "EMG" / "ProcessedData.mat")
    level = processed.top["LevelWalking"]
    available = np.asarray(level.AvailableTrialIdx, dtype=np.int64).reshape(-1)
    trials: list[Gait120Trial] = []

    for trial_index in available.tolist():
        trial_struct = getattr(level, f"Trial{trial_index:02d}")
        n_steps = int(trial_struct.nSteps)
        if n_steps < 1:
            continue
        motion_parts: list[np.ndarray] = []
        emg_parts: list[np.ndarray] = []
        motion_names: tuple[str, ...] | None = None
        emg_names: tuple[str, ...] | None = None
        step_counts: list[int] = []

        for step_index in range(1, n_steps + 1):
            step_struct = getattr(trial_struct, f"Step{step_index:02d}")
            names, emg = processed.table(getattr(step_struct, emg_field))
            mot_path = (
                subject_dir
                / "JointAngle"
                / "LevelWalking"
                / f"Trial{trial_index:02d}"
                / f"step{step_index:02d}.mot"
            )
            columns, motion = _read_opensim_mot(mot_path)
            if emg.shape[0] != motion.shape[0] * EMG_SAMPLES_PER_FRAME:
                raise ValueError(
                    f"Native EMG/motion length mismatch for {subject} trial {trial_index} "
                    f"step {step_index}: {emg.shape[0]} versus "
                    f"{motion.shape[0]}*{EMG_SAMPLES_PER_FRAME}"
                )
            if names != EMG_CHANNELS:
                raise ValueError(f"Unexpected Gait120 EMG channels for {subject}: {names}")
            if motion_names is not None and columns != motion_names:
                raise ValueError("OpenSim columns change between steps")
            if emg_names is not None and names != emg_names:
                raise ValueError("EMG columns change between steps")

            step_counts.append(int(motion.shape[0]))
            if motion_parts:
                time_col = columns.index("time")
                if not np.isclose(motion_parts[-1][-1, time_col], motion[0, time_col], atol=1e-8):
                    raise ValueError("Consecutive Gait120 steps do not share their boundary time")
                # Consecutive released steps repeat one synchronized boundary
                # frame and the corresponding 20 native EMG samples. Remove
                # that duplicate observation; no values are interpolated.
                motion = motion[1:]
                emg = emg[EMG_SAMPLES_PER_FRAME:]
            motion_parts.append(motion)
            emg_parts.append(emg)
            motion_names = columns
            emg_names = names

        motion_all = np.concatenate(motion_parts, axis=0)
        emg_all = np.concatenate(emg_parts, axis=0)
        assert motion_names is not None and emg_names is not None
        if emg_all.shape[0] != motion_all.shape[0] * EMG_SAMPLES_PER_FRAME:
            raise AssertionError("Combined Gait120 trial lost native synchronization")
        time = motion_all[:, motion_names.index("time")]
        if time.size > 1 and not np.allclose(np.diff(time), 1.0 / MOTION_HZ, atol=1e-8):
            raise ValueError(f"Nonuniform native motion timestamps in {subject} trial {trial_index}")

        knee_angle_r = motion_all[:, motion_names.index("knee_angle_r")]
        hip_flexion = motion_all[:, motion_names.index("hip_flexion_r")]
        hip_adduction = motion_all[:, motion_names.index("hip_adduction_r")]
        hip_rotation = motion_all[:, motion_names.index("hip_rotation_r")]
        trials.append(
            Gait120Trial(
                subject=subject,
                trial=int(trial_index),
                motion_time_s=time.astype(np.float64),
                knee_flexion_deg=(-knee_angle_r).astype(np.float32),
                knee_included_deg=(180.0 + knee_angle_r).astype(np.float32),
                thigh_pitch_deg=hip_flexion.astype(np.float32),
                thigh_quat_wxyz=hip_angles_to_quat_wxyz(
                    hip_flexion, hip_adduction, hip_rotation
                ).astype(np.float32),
                emg_native=emg_all.astype(np.float32),
                emg_names=emg_names,
                step_frame_counts=tuple(step_counts),
            )
        )
    return trials


def _causal_emg_envelope(raw_emg: np.ndarray) -> np.ndarray:
    """Apply the Gait120 bandpass/RMS concept using past samples only."""

    raw = np.asarray(raw_emg, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(EMG_CHANNELS):
        raise ValueError(f"Expected raw sEMG [samples,{len(EMG_CHANNELS)}], got {raw.shape}")
    # The release uses a second-order 20--500 Hz Butterworth bandpass followed
    # by rectification and a 250-sample RMS. Its provided implementation uses
    # filtfilt and centered movmean, which include future samples. The causal
    # version below retains the specified filter/window but permits a genuine
    # ahead-of-time forecast.
    sos = scipy.signal.butter(
        2, (20.0, 500.0), btype="bandpass", fs=float(EMG_HZ), output="sos"
    )
    filtered = scipy.signal.sosfilt(sos, raw, axis=0)
    squared = np.square(np.abs(filtered), dtype=np.float64)
    prefix = np.concatenate(
        [np.zeros((1, squared.shape[1]), dtype=np.float64), np.cumsum(squared, axis=0)],
        axis=0,
    )
    index = np.arange(1, squared.shape[0] + 1, dtype=np.int64)
    start = np.maximum(0, index - 250)
    count = (index - start).astype(np.float64)[:, None]
    mean_square = (prefix[index] - prefix[start]) / count
    return np.sqrt(np.maximum(mean_square, 0.0)).astype(np.float32)


def load_level_walking_subject_causal_raw(subject_dir: Path) -> list[Gait120Trial]:
    """Load level walking from raw sEMG with strictly causal preprocessing."""

    subject_dir = Path(subject_dir)
    subject = subject_dir.name
    raw_store = _MatlabTableStore(subject_dir / "EMG" / "RawData.mat")
    level = raw_store.top["LevelWalking"]
    available = np.asarray(level.AvailableTrialIdx, dtype=np.int64).reshape(-1)
    trials: list[Gait120Trial] = []
    for trial_index in available.tolist():
        trial_struct = getattr(level, f"Trial{trial_index:02d}")
        emg_names, raw_emg = raw_store.table(trial_struct.EMGs_raw)
        if emg_names != EMG_CHANNELS:
            raise ValueError(f"Unexpected Gait120 EMG channels for {subject}: {emg_names}")
        total_frame = np.asarray(trial_struct.TotalFrame, dtype=np.int64).reshape(-1)
        expected_samples = int(total_frame[1] - total_frame[0] + 1) * EMG_SAMPLES_PER_FRAME
        if raw_emg.shape[0] != expected_samples:
            raise ValueError(f"Raw sEMG/frame mismatch for {subject} trial {trial_index}")
        envelope = _causal_emg_envelope(raw_emg)

        motion_parts: list[np.ndarray] = []
        emg_parts: list[np.ndarray] = []
        motion_names: tuple[str, ...] | None = None
        step_counts: list[int] = []
        for step_index in range(1, int(trial_struct.nSteps) + 1):
            target = np.asarray(
                getattr(trial_struct, f"Step{step_index:02d}").TargetFrame,
                dtype=np.int64,
            ).reshape(-1)
            sample_start = int(target[0] - total_frame[0]) * EMG_SAMPLES_PER_FRAME
            sample_stop = int(target[1] - total_frame[0] + 1) * EMG_SAMPLES_PER_FRAME
            step_emg = envelope[sample_start:sample_stop]
            mot_path = (
                subject_dir
                / "JointAngle"
                / "LevelWalking"
                / f"Trial{trial_index:02d}"
                / f"step{step_index:02d}.mot"
            )
            columns, motion = _read_opensim_mot(mot_path)
            if step_emg.shape[0] != motion.shape[0] * EMG_SAMPLES_PER_FRAME:
                raise ValueError(
                    f"Causal raw sEMG/motion mismatch for {subject} trial {trial_index} "
                    f"step {step_index}"
                )
            if motion_names is not None and columns != motion_names:
                raise ValueError("OpenSim columns change between steps")
            step_counts.append(int(motion.shape[0]))
            if motion_parts:
                time_col = columns.index("time")
                if not np.isclose(motion_parts[-1][-1, time_col], motion[0, time_col], atol=1e-8):
                    raise ValueError("Consecutive Gait120 steps do not share their boundary time")
                motion = motion[1:]
                step_emg = step_emg[EMG_SAMPLES_PER_FRAME:]
            motion_parts.append(motion)
            emg_parts.append(step_emg)
            motion_names = columns

        motion_all = np.concatenate(motion_parts, axis=0)
        emg_all = np.concatenate(emg_parts, axis=0)
        assert motion_names is not None
        if emg_all.shape[0] != motion_all.shape[0] * EMG_SAMPLES_PER_FRAME:
            raise AssertionError("Combined causal Gait120 trial lost synchronization")
        time = motion_all[:, motion_names.index("time")]
        if time.size > 1 and not np.allclose(np.diff(time), 1.0 / MOTION_HZ, atol=1e-8):
            raise ValueError(f"Nonuniform native motion timestamps in {subject} trial {trial_index}")
        knee_angle_r = motion_all[:, motion_names.index("knee_angle_r")]
        hip_flexion = motion_all[:, motion_names.index("hip_flexion_r")]
        hip_adduction = motion_all[:, motion_names.index("hip_adduction_r")]
        hip_rotation = motion_all[:, motion_names.index("hip_rotation_r")]
        trials.append(
            Gait120Trial(
                subject=subject,
                trial=int(trial_index),
                motion_time_s=time.astype(np.float64),
                knee_flexion_deg=(-knee_angle_r).astype(np.float32),
                knee_included_deg=(180.0 + knee_angle_r).astype(np.float32),
                thigh_pitch_deg=hip_flexion.astype(np.float32),
                thigh_quat_wxyz=hip_angles_to_quat_wxyz(
                    hip_flexion, hip_adduction, hip_rotation
                ).astype(np.float32),
                emg_native=emg_all.astype(np.float32),
                emg_names=emg_names,
                step_frame_counts=tuple(step_counts),
            )
        )
    return trials


def audit_native_alignment(subject_dir: Path) -> dict[str, Any]:
    """Cross-check raw frame indices, native processed EMG, and OpenSim rows."""

    subject_dir = Path(subject_dir)
    raw = _MatlabTableStore(subject_dir / "EMG" / "RawData.mat")
    loaded_trials = {trial.trial: trial for trial in load_level_walking_subject(subject_dir)}
    level = raw.top["LevelWalking"]
    rows: list[dict[str, Any]] = []
    for trial_index in np.asarray(level.AvailableTrialIdx, dtype=np.int64).reshape(-1).tolist():
        trial_struct = getattr(level, f"Trial{trial_index:02d}")
        names, raw_emg = raw.table(trial_struct.EMGs_raw)
        total_frame = np.asarray(trial_struct.TotalFrame, dtype=np.int64).reshape(-1)
        expected_raw = int(total_frame[1] - total_frame[0] + 1) * EMG_SAMPLES_PER_FRAME
        if names != EMG_CHANNELS or raw_emg.shape[0] != expected_raw:
            raise ValueError(f"Raw trial alignment failed for {subject_dir.name} trial {trial_index}")
        step_frames: list[int] = []
        for step_index in range(1, int(trial_struct.nSteps) + 1):
            target = np.asarray(
                getattr(trial_struct, f"Step{step_index:02d}").TargetFrame,
                dtype=np.int64,
            ).reshape(-1)
            if target.size != 2 or target[0] < total_frame[0] or target[1] > total_frame[1]:
                raise ValueError("Step frame range lies outside its synchronized raw trial")
            step_frames.append(int(target[1] - target[0] + 1))
        loaded = loaded_trials[int(trial_index)]
        if tuple(step_frames) != loaded.step_frame_counts:
            raise ValueError("Raw frame indices disagree with released OpenSim step rows")
        expected_combined = sum(step_frames) - (len(step_frames) - 1)
        if loaded.n_frames != expected_combined:
            raise ValueError("Boundary de-duplication produced an unexpected trial length")
        rows.append(
            {
                "trial": int(trial_index),
                "raw_frames": int(total_frame[1] - total_frame[0] + 1),
                "raw_emg_samples": int(raw_emg.shape[0]),
                "step_frame_counts": step_frames,
                "combined_motion_frames": loaded.n_frames,
                "combined_native_emg_samples": int(loaded.emg_native.shape[0]),
            }
        )
    return {
        "subject": subject_dir.name,
        "motion_hz": MOTION_HZ,
        "emg_hz": EMG_HZ,
        "interpolation_used": False,
        "duplicate_step_boundary_policy": (
            "drop the repeated first frame and its identical 20 native EMG samples "
            "from each subsequent step"
        ),
        "trials": rows,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_native_alignment(args.subject_dir), indent=2))


if __name__ == "__main__":
    main()
