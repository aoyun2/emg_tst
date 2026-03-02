from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

HF_REPO_ID = "microsoft/mocapact-data"
HF_REPO_TYPE = "dataset"
EXPECTED_SNIPPETS = 2589


@dataclass
class MocapSnippet:
    snippet_id: str
    file_path: Path
    thigh_angle_deg: np.ndarray
    knee_angle_deg: np.ndarray
    sample_hz: float


def list_snippet_files(repo_id: str = HF_REPO_ID) -> list[str]:
    files = list_repo_files(repo_id=repo_id, repo_type=HF_REPO_TYPE)
    candidates = [
        f for f in files if f.lower().endswith((".h5", ".hdf5", ".npz")) and "snippet" in f.lower()
    ]
    candidates.sort()
    return candidates


def download_all_snippets(dest_dir: str | Path, repo_id: str = HF_REPO_ID) -> list[Path]:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    files = list_snippet_files(repo_id=repo_id)
    if not files:
        raise RuntimeError("No snippet files found in MoCapAct HF dataset.")

    local_paths: list[Path] = []
    for f in tqdm(files, desc="Downloading MoCapAct snippets"):
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type=HF_REPO_TYPE,
            filename=f,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        local_paths.append(Path(local))

    index = {
        "repo": repo_id,
        "count": len(local_paths),
        "expected": EXPECTED_SNIPPETS,
        "paths": [str(p) for p in local_paths],
    }
    (dest / "snippet_index.json").write_text(json.dumps(index, indent=2))
    return local_paths


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
    files = sorted(list(p.rglob("*.h5")) + list(p.rglob("*.hdf5")) + list(p.rglob("*.npz")))
    return len(files), len(files) == expected
