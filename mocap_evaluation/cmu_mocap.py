from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Canonical "CMU mocap joint order" used by dm_control's CMU humanoid.
# Source: dm_control.locomotion.walkers.cmu_humanoid._CMU_MOCAP_JOINTS
CMU_MOCAP_JOINTS: tuple[str, ...] = (
    "lfemurrz",
    "lfemurry",
    "lfemurrx",
    "ltibiarx",
    "lfootrz",
    "lfootrx",
    "ltoesrx",
    "rfemurrz",
    "rfemurry",
    "rfemurrx",
    "rtibiarx",
    "rfootrz",
    "rfootrx",
    "rtoesrx",
    "lowerbackrz",
    "lowerbackry",
    "lowerbackrx",
    "upperbackrz",
    "upperbackry",
    "upperbackrx",
    "thoraxrz",
    "thoraxry",
    "thoraxrx",
    "lowerneckrz",
    "lowerneckry",
    "lowerneckrx",
    "upperneckrz",
    "upperneckry",
    "upperneckrx",
    "headrz",
    "headry",
    "headrx",
    "lclaviclerz",
    "lclaviclery",
    "lhumerusrz",
    "lhumerusry",
    "lhumerusrx",
    "lradiusrx",
    "lwristry",
    "lhandrz",
    "lhandrx",
    "lfingersrx",
    "lthumbrz",
    "lthumbrx",
    "rclaviclerz",
    "rclaviclery",
    "rhumerusrz",
    "rhumerusry",
    "rhumerusrx",
    "rradiusrx",
    "rwristry",
    "rhandrz",
    "rhandrx",
    "rfingersrx",
    "rthumbrz",
    "rthumbrx",
)


def resolve_default_cmu_h5_path() -> Path:
    """Best-effort discovery of dm_control's preprocessed CMU HDF5 file.

    This tries to use dm_control's own helper. If dm_control is not installed,
    callers must provide a path explicitly.
    """
    try:
        from dm_control.locomotion.mocap import cmu_mocap_data  # type: ignore

        return Path(cmu_mocap_data.get_path_for_cmu(version="2020"))
    except Exception as e:  # pragma: no cover - depends on external install
        raise RuntimeError(
            "Could not auto-locate dm_control CMU mocap file. "
            "Install dm_control or pass --cmu-h5 explicitly."
        ) from e


@dataclass(frozen=True)
class CmuClip:
    clip_id: str
    dt: float
    num_steps: int


class CmuMocap:
    """Small HDF5 reader for dm_control's preprocessed CMU mocap dataset."""

    def __init__(self, h5_path: str | Path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"CMU HDF5 not found: {self.h5_path}")

        self._joint_to_idx = {n: i for i, n in enumerate(CMU_MOCAP_JOINTS)}

    def joint_index(self, joint_name: str) -> int:
        try:
            return int(self._joint_to_idx[joint_name])
        except KeyError as e:
            raise KeyError(
                f"Unknown CMU joint {joint_name!r}. Expected one of: {list(self._joint_to_idx)[:10]}..."
            ) from e

    def clip_meta(self, clip_id: str) -> CmuClip:
        import h5py

        with h5py.File(self.h5_path, "r") as f:
            g = f[clip_id]
            dt = float(g.attrs["dt"])
            num_steps = int(g.attrs["num_steps"])
        return CmuClip(clip_id=clip_id, dt=dt, num_steps=num_steps)

    def list_clips(self, *, prefix: str = "CMU_") -> list[str]:
        """Return all clip ids in the HDF5 (usually 'CMU_###_##')."""
        import h5py

        with h5py.File(self.h5_path, "r") as f:
            keys = list(f.keys())
        if prefix:
            keys = [k for k in keys if str(k).startswith(prefix)]
        keys.sort()
        return [str(k) for k in keys]

    def load_joint_series(
        self,
        *,
        clip_id: str,
        joint_name: str,
        start_step: int,
        end_step: int,
    ) -> tuple[np.ndarray, float]:
        """Load a joint angle series (radians) and return (values, sample_hz)."""
        import h5py

        j_idx = self.joint_index(joint_name)
        start = int(start_step)
        end = int(end_step)
        if end < start:
            raise ValueError(f"end_step < start_step for {clip_id}: {start_step}..{end_step}")

        with h5py.File(self.h5_path, "r") as f:
            g = f[clip_id]
            dt = float(g.attrs["dt"])
            sample_hz = 1.0 / dt
            joints = g["walkers"]["walker_0"]["joints"]
            # Stored as (n_joints, T)
            if start < 0 or end >= joints.shape[1]:
                raise IndexError(
                    f"Requested steps out of range for {clip_id}: {start}..{end}, clip has {joints.shape[1]} steps"
                )
            out = np.asarray(joints[j_idx, start : end + 1], dtype=np.float32)
        return out, float(sample_hz)
