from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .cmu_mocap import CmuMocap
from .mocapact_dataset import ExpertPolicy, EXPECTED_SNIPPETS, load_expert_policy_map


@dataclass(frozen=True)
class SnippetTrajectory:
    snippet_id: str
    clip_id: str
    start_step: int
    end_step: int
    sample_hz: float
    thigh_angle_deg: np.ndarray
    knee_angle_deg: np.ndarray


@dataclass(frozen=True)
class SnippetIndex:
    """A motion-matching index over MoCapAct snippet reference joint trajectories."""

    snippet_id: np.ndarray  # (N,) dtype=object[str]
    clip_id: np.ndarray  # (N,) dtype=object[str]
    start_step: np.ndarray  # (N,) int32
    end_step: np.ndarray  # (N,) int32
    sample_hz: np.ndarray  # (N,) float32
    thigh_deg: np.ndarray  # (N,) dtype=object[np.ndarray]
    knee_deg: np.ndarray  # (N,) dtype=object[np.ndarray]

    def __len__(self) -> int:
        return int(self.snippet_id.shape[0])

    def iter_snippets(self) -> Iterator[SnippetTrajectory]:
        for i in range(len(self)):
            yield SnippetTrajectory(
                snippet_id=str(self.snippet_id[i]),
                clip_id=str(self.clip_id[i]),
                start_step=int(self.start_step[i]),
                end_step=int(self.end_step[i]),
                sample_hz=float(self.sample_hz[i]),
                thigh_angle_deg=np.asarray(self.thigh_deg[i], dtype=np.float32),
                knee_angle_deg=np.asarray(self.knee_deg[i], dtype=np.float32),
            )

    def save_npz(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p,
            snippet_id=self.snippet_id,
            clip_id=self.clip_id,
            start_step=self.start_step,
            end_step=self.end_step,
            sample_hz=self.sample_hz,
            thigh_deg=self.thigh_deg,
            knee_deg=self.knee_deg,
        )
        return p

    @classmethod
    def load_npz(cls, path: str | Path) -> "SnippetIndex":
        p = Path(path)
        d = np.load(p, allow_pickle=True)
        return cls(
            snippet_id=d["snippet_id"],
            clip_id=d["clip_id"],
            start_step=d["start_step"],
            end_step=d["end_step"],
            sample_hz=d["sample_hz"],
            thigh_deg=d["thigh_deg"],
            knee_deg=d["knee_deg"],
        )


def default_joint_names_for_side(side: str) -> tuple[str, str]:
    s = side.lower().strip()
    if s in {"left", "l"}:
        return "lfemurrx", "ltibiarx"
    if s in {"right", "r"}:
        return "rfemurrx", "rtibiarx"
    raise ValueError("--side must be 'left' or 'right'")


def build_snippet_index(
    *,
    experts_root: str | Path,
    cmu_h5_path: str | Path,
    side: str = "left",
    thigh_joint: str | None = None,
    knee_joint: str | None = None,
    expected_snippets: int = EXPECTED_SNIPPETS,
    limit: int | None = None,
) -> tuple[SnippetIndex, dict[str, ExpertPolicy]]:
    """Build a snippet index (and expert-policy map) from extracted experts + CMU HDF5.

    Returns:
      (index, expert_map)
    """
    expert_map = load_expert_policy_map(experts_root)
    if expected_snippets is not None and len(expert_map) != expected_snippets and limit is None:
        raise RuntimeError(
            f"Expected {expected_snippets} expert policies, found {len(expert_map)} under {experts_root}."
        )

    if thigh_joint is None or knee_joint is None:
        tj, kj = default_joint_names_for_side(side)
        thigh_joint = thigh_joint or tj
        knee_joint = knee_joint or kj

    cmu = CmuMocap(cmu_h5_path)

    ids: list[str] = []
    clip_ids: list[str] = []
    start_steps: list[int] = []
    end_steps: list[int] = []
    sample_hzs: list[float] = []
    thigh_list: list[np.ndarray] = []
    knee_list: list[np.ndarray] = []

    items = list(expert_map.items())
    items.sort(key=lambda kv: kv[0])
    if limit is not None:
        items = items[: int(limit)]

    for snippet_id, pol in items:
        thigh_rad, hz = cmu.load_joint_series(
            clip_id=pol.clip_id,
            joint_name=thigh_joint,
            start_step=pol.start_step,
            end_step=pol.end_step,
        )
        knee_rad, hz2 = cmu.load_joint_series(
            clip_id=pol.clip_id,
            joint_name=knee_joint,
            start_step=pol.start_step,
            end_step=pol.end_step,
        )
        if abs(hz - hz2) > 1e-6:
            raise RuntimeError(f"Inconsistent sample_hz for clip {pol.clip_id}: {hz} vs {hz2}")

        ids.append(snippet_id)
        clip_ids.append(pol.clip_id)
        start_steps.append(pol.start_step)
        end_steps.append(pol.end_step)
        sample_hzs.append(hz)
        thigh_list.append(np.rad2deg(thigh_rad).astype(np.float32))
        knee_list.append(np.rad2deg(knee_rad).astype(np.float32))

    idx = SnippetIndex(
        snippet_id=np.asarray(ids, dtype=object),
        clip_id=np.asarray(clip_ids, dtype=object),
        start_step=np.asarray(start_steps, dtype=np.int32),
        end_step=np.asarray(end_steps, dtype=np.int32),
        sample_hz=np.asarray(sample_hzs, dtype=np.float32),
        thigh_deg=np.asarray(thigh_list, dtype=object),
        knee_deg=np.asarray(knee_list, dtype=object),
    )
    return idx, expert_map


def build_cmu_clip_index(
    *,
    cmu_h5_path: str | Path,
    side: str = "left",
    thigh_joint: str | None = None,
    knee_joint: str | None = None,
    limit: int | None = None,
) -> SnippetIndex:
    """Build a motion-matching index over full CMU clips (no MoCapAct experts needed)."""
    if thigh_joint is None or knee_joint is None:
        tj, kj = default_joint_names_for_side(side)
        thigh_joint = thigh_joint or tj
        knee_joint = knee_joint or kj

    cmu = CmuMocap(cmu_h5_path)
    clip_ids_all = cmu.list_clips(prefix="CMU_")
    if limit is not None:
        clip_ids_all = clip_ids_all[: int(limit)]

    ids: list[str] = []
    clip_ids: list[str] = []
    start_steps: list[int] = []
    end_steps: list[int] = []
    sample_hzs: list[float] = []
    thigh_list: list[np.ndarray] = []
    knee_list: list[np.ndarray] = []

    for clip_id in clip_ids_all:
        meta = cmu.clip_meta(clip_id)
        if meta.num_steps < 2:
            continue
        start_step = 0
        end_step = int(meta.num_steps) - 1

        thigh_rad, hz = cmu.load_joint_series(
            clip_id=clip_id,
            joint_name=thigh_joint,
            start_step=start_step,
            end_step=end_step,
        )
        knee_rad, hz2 = cmu.load_joint_series(
            clip_id=clip_id,
            joint_name=knee_joint,
            start_step=start_step,
            end_step=end_step,
        )
        if abs(hz - hz2) > 1e-6:
            raise RuntimeError(f"Inconsistent sample_hz for clip {clip_id}: {hz} vs {hz2}")

        snippet_id = f"{clip_id}-{start_step}-{end_step}"

        ids.append(snippet_id)
        clip_ids.append(clip_id)
        start_steps.append(start_step)
        end_steps.append(end_step)
        sample_hzs.append(hz)
        thigh_list.append(np.rad2deg(thigh_rad).astype(np.float32))
        knee_list.append(np.rad2deg(knee_rad).astype(np.float32))

    return SnippetIndex(
        snippet_id=np.asarray(ids, dtype=object),
        clip_id=np.asarray(clip_ids, dtype=object),
        start_step=np.asarray(start_steps, dtype=np.int32),
        end_step=np.asarray(end_steps, dtype=np.int32),
        sample_hz=np.asarray(sample_hzs, dtype=np.float32),
        thigh_deg=np.asarray(thigh_list, dtype=object),
        knee_deg=np.asarray(knee_list, dtype=object),
    )


def write_index_manifest(
    path: str | Path,
    *,
    index: SnippetIndex,
    index_kind: str,
    expert_map: dict[str, ExpertPolicy] | None = None,
) -> Path:
    """Write a small JSON next to the index for human debugging."""
    p = Path(path)
    manifest = {
        "index_kind": str(index_kind),
        "n_snippets": len(index),
        "n_experts": (len(expert_map) if expert_map is not None else None),
        "first_snippet": str(index.snippet_id[0]) if len(index) else None,
        "expected_snippets": (EXPECTED_SNIPPETS if expert_map is not None else None),
    }
    out = p.with_suffix(p.suffix + ".manifest.json")
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
