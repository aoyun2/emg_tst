from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

EXPECTED_SNIPPETS = 2589


@dataclass
class MocapSnippet:
    snippet_id: str
    file_path: Path
    thigh_angle_deg: np.ndarray
    knee_angle_deg: np.ndarray
    sample_hz: float


def _first_existing(h5: h5py.File, candidates: Iterable[str]) -> np.ndarray:
    for key in candidates:
        if key in h5:
            return np.asarray(h5[key])
    raise KeyError(f"None of keys found: {list(candidates)}")


def _to_angle_deg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    # Heuristic: if values look like radians, convert
    if np.nanmax(np.abs(x)) <= 6.5:
        return np.rad2deg(x).astype(np.float32)
    return x.astype(np.float32)


def load_snippet_angles(path: str | Path) -> MocapSnippet:
    p = Path(path)
    if p.suffix.lower() in {".npz"}:
        d = np.load(p)
        thigh = _to_angle_deg(d["thigh_angle"])
        knee = _to_angle_deg(d["knee_angle"])
        hz = float(d.get("sample_hz", 30.0))
        return MocapSnippet(p.stem, p, thigh, knee, hz)

    with h5py.File(p, "r") as h5:
        # Common key patterns seen across mocap datasets
        thigh = _first_existing(
            h5,
            [
                "angles/thigh_angle",
                "thigh_angle",
                "observations/thigh_angle",
                "qpos/thigh",
            ],
        )
        knee = _first_existing(
            h5,
            [
                "angles/knee_angle",
                "knee_angle",
                "observations/knee_angle",
                "qpos/knee",
            ],
        )
        hz = float(h5.attrs.get("sample_rate_hz", h5.attrs.get("fps", 30.0)))

    return MocapSnippet(p.stem, p, _to_angle_deg(thigh), _to_angle_deg(knee), hz)


def validate_snippet_count(local_dir: str | Path, expected: int = EXPECTED_SNIPPETS) -> tuple[int, bool]:
    p = Path(local_dir)
    files = sorted(
        list(p.rglob("*.h5"))
        + list(p.rglob("*.hdf5"))
        + list(p.rglob("*.npz"))
        + list(p.rglob("*.npy"))
    )
    return len(files), len(files) == expected
