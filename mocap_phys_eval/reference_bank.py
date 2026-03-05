from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .experts import ExpertSnippet, discover_expert_snippets
from .utils import quat_conj_wxyz, quat_mul_wxyz, quat_normalize_wxyz, rotmat_to_quat_wxyz


# Canonical joint order used by dm_control's CMU humanoid.
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


def _cmu_h5_path() -> Path:
    try:
        from dm_control.locomotion.mocap import cmu_mocap_data  # type: ignore

        return Path(cmu_mocap_data.get_path_for_cmu(version="2020"))
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not locate dm_control CMU2020 HDF5. Ensure dm_control is installed and has downloaded CMU data."
        ) from e


def _joint_index(name: str) -> int:
    try:
        return int(CMU_MOCAP_JOINTS.index(str(name)))
    except ValueError as e:
        raise KeyError(f"Unknown joint: {name!r}") from e


@dataclass(frozen=True)
class ClipBank:
    """Reference bank of full CMU2020 clips (one entry per clip)."""

    snippet_id: np.ndarray  # (N,) object[str]
    clip_id: np.ndarray  # (N,) object[str]
    start_step: np.ndarray  # (N,) int32 (always 0)
    end_step: np.ndarray  # (N,) int32
    sample_hz: np.ndarray  # (N,) float32
    # Hip pitch joint (rfemurrx/lfemurrx) in degrees.
    hip_deg: np.ndarray  # (N,) object[np.ndarray]
    # Thigh segment pitch (world) in degrees, computed from body positions.
    thigh_pitch_deg: np.ndarray  # (N,) object[np.ndarray]
    knee_deg: np.ndarray  # (N,) object[np.ndarray]

    def __len__(self) -> int:
        return int(self.snippet_id.shape[0])

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
            hip_deg=self.hip_deg,
            thigh_pitch_deg=self.thigh_pitch_deg,
            knee_deg=self.knee_deg,
        )
        return p

    @classmethod
    def load_npz(cls, path: str | Path) -> "ClipBank":
        p = Path(path)
        d = np.load(p, allow_pickle=True)
        return cls(
            snippet_id=d["snippet_id"],
            clip_id=d["clip_id"],
            start_step=d["start_step"],
            end_step=d["end_step"],
            sample_hz=d["sample_hz"],
            hip_deg=d["hip_deg"],
            thigh_pitch_deg=d["thigh_pitch_deg"],
            knee_deg=d["knee_deg"],
        )


@dataclass(frozen=True)
class ExpertSnippetBank:
    """Reference bank aligned to MoCapAct's clip-snippet experts (~2589 snippets)."""

    snippet_id: np.ndarray  # (N,) object[str]  e.g. "CMU_083_33-0-194"
    clip_id: np.ndarray  # (N,) object[str]  e.g. "CMU_083_33"
    start_step: np.ndarray  # (N,) int32 absolute start step in the CMU clip
    end_step: np.ndarray  # (N,) int32 absolute end step (inclusive)
    sample_hz: np.ndarray  # (N,) float32
    hip_deg: np.ndarray  # (N,) object[np.ndarray] hip pitch joint degrees for the snippet segment
    thigh_pitch_deg: np.ndarray  # (N,) object[np.ndarray] thigh segment pitch degrees for the snippet segment
    # Thigh orientation (root-relative) as (T,4) wxyz. Historically this was the raw MuJoCo
    # body quaternion (rfemur body frame). We now also store an "anatomical" thigh frame
    # aligned to the user's IMU mounting convention (+Y distal, +Z posterior).
    thigh_quat_wxyz: np.ndarray  # (N,) object[np.ndarray]
    thigh_anat_quat_wxyz: np.ndarray  # (N,) object[np.ndarray]
    # Same anatomical frame as thigh_anat_quat_wxyz, but expressed in *world* coordinates
    # (thigh->world). This matches how IMUs typically report orientation (sensor->world)
    # and is the default representation for motion matching to rigtest.py recordings.
    thigh_anat_quat_world_wxyz: np.ndarray  # (N,) object[np.ndarray]
    knee_deg: np.ndarray  # (N,) object[np.ndarray] knee flexion joint degrees for the snippet segment
    expert_model_path: np.ndarray  # (N,) object[str] path to eval_rsi/model directory

    def __len__(self) -> int:
        return int(self.snippet_id.shape[0])

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
            hip_deg=self.hip_deg,
            thigh_pitch_deg=self.thigh_pitch_deg,
            thigh_quat_wxyz=self.thigh_quat_wxyz,
            thigh_anat_quat_wxyz=self.thigh_anat_quat_wxyz,
            thigh_anat_quat_world_wxyz=self.thigh_anat_quat_world_wxyz,
            knee_deg=self.knee_deg,
            expert_model_path=self.expert_model_path,
        )
        return p

    @classmethod
    def load_npz(cls, path: str | Path) -> "ExpertSnippetBank":
        p = Path(path)
        d = np.load(p, allow_pickle=True)
        # Backwards-compat: old banks won't have thigh_anat_quat_wxyz. In that case,
        # fall back to the stored thigh_quat_wxyz (still root-relative).
        thigh_anat = d["thigh_anat_quat_wxyz"] if "thigh_anat_quat_wxyz" in d.files else d["thigh_quat_wxyz"]
        # Backwards-compat: older banks won't have world-frame anatomical quats.
        thigh_anat_world = d["thigh_anat_quat_world_wxyz"] if "thigh_anat_quat_world_wxyz" in d.files else thigh_anat
        return cls(
            snippet_id=d["snippet_id"],
            clip_id=d["clip_id"],
            start_step=d["start_step"],
            end_step=d["end_step"],
            sample_hz=d["sample_hz"],
            hip_deg=d["hip_deg"],
            thigh_pitch_deg=d["thigh_pitch_deg"],
            thigh_quat_wxyz=d["thigh_quat_wxyz"],
            thigh_anat_quat_wxyz=thigh_anat,
            thigh_anat_quat_world_wxyz=thigh_anat_world,
            knee_deg=d["knee_deg"],
            expert_model_path=d["expert_model_path"],
        )


def build_expert_snippet_bank(*, experts_root: str | Path, side: str = "right") -> ExpertSnippetBank:
    """Build a reference bank for all extracted MoCapAct expert snippets.

    This reads dm_control's CMU2020 fitted mocap HDF5 for kinematics, and uses the
    extracted expert model directory tree to enumerate snippet boundaries and
    locate each snippet's expert policy checkpoint directory.
    """
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    side = str(side).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    # Discover all snippet expert model dirs.
    snippets: list[ExpertSnippet] = discover_expert_snippets(experts_root)
    if not snippets:
        raise RuntimeError(
            f"No expert models found under {str(Path(experts_root).resolve())}. "
            "Download/extract the MoCapAct expert model zoo first."
        )

    thigh_joint = "lfemurrx" if side == "left" else "rfemurrx"
    knee_joint = "ltibiarx" if side == "left" else "rtibiarx"
    j_th = _joint_index(thigh_joint)
    j_kn = _joint_index(knee_joint)

    # Indices in walker.mocap_tracking_bodies order (dm_control CMU humanoid).
    if side == "left":
        femur_i = 1
        tibia_i = 2
    else:
        femur_i = 6
        tibia_i = 7

    h5_path = _cmu_h5_path()
    import h5py

    # Group snippet specs by clip to avoid reloading full clip arrays repeatedly.
    by_clip: dict[str, list[ExpertSnippet]] = {}
    for sn in snippets:
        by_clip.setdefault(str(sn.clip_id), []).append(sn)
    for k in by_clip:
        by_clip[k].sort(key=lambda s: (int(s.start_step), int(s.end_step)))

    out_snip: list[str] = []
    out_clip: list[str] = []
    out_start: list[int] = []
    out_end: list[int] = []
    out_hz: list[float] = []
    out_hip: list[np.ndarray] = []
    out_th: list[np.ndarray] = []
    out_thq: list[np.ndarray] = []
    out_thq_anat: list[np.ndarray] = []
    out_thq_anat_world: list[np.ndarray] = []
    out_kn: list[np.ndarray] = []
    out_model: list[str] = []

    with h5py.File(h5_path, "r") as f:
        clip_ids = list(by_clip.keys())
        clip_ids.sort()
        it = clip_ids
        if tqdm is not None:
            it = tqdm(it, desc="Building expert snippet bank", unit="clip")
        for clip_id in it:
            if clip_id not in f:
                continue
            g = f[clip_id]
            dt = float(g.attrs["dt"])
            hz = float(1.0 / dt)

            joints = g["walkers"]["walker_0"]["joints"]  # (n_joints, T)
            T = int(joints.shape[1])
            if T < 2:
                continue
            hip_full = np.rad2deg(np.asarray(joints[j_th, :], dtype=np.float32)).astype(np.float32)
            kn_full = np.rad2deg(np.asarray(joints[j_kn, :], dtype=np.float32)).astype(np.float32)

            root_pos = np.asarray(g["walkers"]["walker_0"]["position"], dtype=np.float64).T  # (T,3)
            body_pos = np.asarray(g["walkers"]["walker_0"]["body_positions"], dtype=np.float64).T  # (T, 3*nb)
            root_q = np.asarray(g["walkers"]["walker_0"]["quaternion"], dtype=np.float64).T  # (T,4) wxyz
            body_q = np.asarray(g["walkers"]["walker_0"]["body_quaternions"], dtype=np.float64).T  # (T, 4*nb) wxyz
            nb = int(body_pos.shape[1] // 3)
            if nb * 3 != int(body_pos.shape[1]) or nb < max(femur_i, tibia_i) + 1:
                raise RuntimeError(f"Unexpected body_positions shape: {body_pos.shape}")
            fem_pos = body_pos[:, 3 * femur_i : 3 * femur_i + 3]
            tib_pos = body_pos[:, 3 * tibia_i : 3 * tibia_i + 3]
            th_full = _thigh_pitch_from_body_positions(root_pos=root_pos, femur_pos=fem_pos, tibia_pos=tib_pos)
            if int(body_q.shape[1]) != int(nb * 4):
                raise RuntimeError(f"Unexpected body_quaternions shape: {body_q.shape} for nb={nb}")
            fem_q_world = body_q[:, 4 * femur_i : 4 * femur_i + 4]
            thq_full = quat_mul_wxyz(
                quat_conj_wxyz(quat_normalize_wxyz(root_q)),
                quat_normalize_wxyz(fem_q_world),
            ).astype(np.float32)
            thq_anat_world_full = _thigh_anat_quat_world_from_body_positions(
                root_pos=root_pos, femur_pos=fem_pos, tibia_pos=tib_pos
            ).astype(np.float32)
            thq_anat_full = quat_mul_wxyz(
                quat_conj_wxyz(quat_normalize_wxyz(root_q)),
                quat_normalize_wxyz(thq_anat_world_full),
            ).astype(np.float32)

            for sn in by_clip.get(clip_id, []):
                s0 = int(sn.start_step)
                s1 = int(sn.end_step)
                if s0 < 0 or s1 < s0 or s1 >= T:
                    continue
                seg = slice(s0, s1 + 1)
                out_snip.append(str(sn.snippet_id))
                out_clip.append(str(sn.clip_id))
                out_start.append(int(s0))
                out_end.append(int(s1))
                out_hz.append(float(hz))
                out_hip.append(np.asarray(hip_full[seg], dtype=np.float32))
                out_th.append(np.asarray(th_full[seg], dtype=np.float32))
                out_thq.append(np.asarray(thq_full[seg], dtype=np.float32))
                out_thq_anat.append(np.asarray(thq_anat_full[seg], dtype=np.float32))
                out_thq_anat_world.append(np.asarray(thq_anat_world_full[seg], dtype=np.float32))
                out_kn.append(np.asarray(kn_full[seg], dtype=np.float32))
                # Store model path relative to repo root if possible.
                mp = Path(sn.model_dir)
                try:
                    mp = mp.resolve().relative_to(Path.cwd().resolve())
                except Exception:
                    mp = mp.resolve()
                out_model.append(str(mp))

    if not out_snip:
        raise RuntimeError("Expert snippet bank is empty (no valid snippets found).")

    return ExpertSnippetBank(
        snippet_id=np.asarray(out_snip, dtype=object),
        clip_id=np.asarray(out_clip, dtype=object),
        start_step=np.asarray(out_start, dtype=np.int32),
        end_step=np.asarray(out_end, dtype=np.int32),
        sample_hz=np.asarray(out_hz, dtype=np.float32),
        hip_deg=np.asarray(out_hip, dtype=object),
        thigh_pitch_deg=np.asarray(out_th, dtype=object),
        thigh_quat_wxyz=np.asarray(out_thq, dtype=object),
        thigh_anat_quat_wxyz=np.asarray(out_thq_anat, dtype=object),
        thigh_anat_quat_world_wxyz=np.asarray(out_thq_anat_world, dtype=object),
        knee_deg=np.asarray(out_kn, dtype=object),
        expert_model_path=np.asarray(out_model, dtype=object),
    )


def _thigh_pitch_from_body_positions(
    *,
    root_pos: np.ndarray,
    femur_pos: np.ndarray,
    tibia_pos: np.ndarray,
) -> np.ndarray:
    """Compute a signed thigh segment pitch angle (deg) from world positions.

    - Uses MuJoCo world axes: +Z up, X/Y horizontal.
    - Forward direction is inferred from overall root displacement in the XY plane.
    """
    root = np.asarray(root_pos, dtype=np.float64)
    fem = np.asarray(femur_pos, dtype=np.float64)
    tib = np.asarray(tibia_pos, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_pos must be (T,3), got {root.shape}")
    if fem.shape != root.shape or tib.shape != root.shape:
        raise ValueError(f"femur/tibia pos must match root shape (T,3), got {fem.shape} and {tib.shape}")

    disp = (root[-1, :2] - root[0, :2]).astype(np.float64)
    dn = float(np.linalg.norm(disp))
    if dn < 1e-6:
        fwd = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        fwd = disp / dn

    v = tib - fem  # femur -> tibia
    down = -v[:, 2]
    fwd_comp = v[:, 0] * float(fwd[0]) + v[:, 1] * float(fwd[1])
    pitch = np.degrees(np.arctan2(fwd_comp, down)).astype(np.float32)
    return pitch


def _thigh_anat_quat_world_from_body_positions(
    *,
    root_pos: np.ndarray,
    femur_pos: np.ndarray,
    tibia_pos: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute an IMU-like *world-frame* thigh orientation quaternion from body positions.

    This defines an "IMU-like" anatomical thigh frame:
    - +Y: distal (femur -> tibia)
    - +Z: posterior/back (opposite the clip forward direction)
    - +X: completes a right-handed frame (+X points roughly to the subject's left)

    This is designed to be compatible with the user's IMU mounting description
    (+X left, +Y distal, +Z posterior).
    """
    root = np.asarray(root_pos, dtype=np.float64)
    fem = np.asarray(femur_pos, dtype=np.float64)
    tib = np.asarray(tibia_pos, dtype=np.float64)
    if root.ndim != 2 or root.shape[1] != 3:
        raise ValueError(f"root_pos must be (T,3), got {root.shape}")
    if fem.shape != root.shape or tib.shape != root.shape:
        raise ValueError(f"femur/tibia pos must match root shape (T,3), got {fem.shape} and {tib.shape}")

    # Forward in the horizontal plane from overall root displacement.
    disp = (root[-1, :2] - root[0, :2]).astype(np.float64)
    dn = float(np.linalg.norm(disp))
    if dn < 1e-9:
        fwd2 = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        fwd2 = disp / dn
    fwd3 = np.asarray([float(fwd2[0]), float(fwd2[1]), 0.0], dtype=np.float64)
    fwd3n = float(np.linalg.norm(fwd3))
    if fwd3n < 1e-9:
        fwd3 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        fwd3 = fwd3 / fwd3n
    back = -fwd3  # posterior direction

    # +Y: distal along segment.
    y = tib - fem
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    y = y / np.clip(yn, float(eps), None)

    # +Z: back direction, projected to be orthogonal to Y.
    backv = back.reshape(1, 3)
    z = backv - np.sum(backv * y, axis=1, keepdims=True) * y
    zn = np.linalg.norm(z, axis=1, keepdims=True)
    # Fallback if projection degenerates (should be rare).
    bad = (zn.reshape(-1) < float(eps)) | ~np.isfinite(zn.reshape(-1))
    if bool(np.any(bad)):
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64).reshape(1, 3)
        z_alt = np.cross(y[bad], up)
        z[bad] = z_alt
        zn = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.clip(zn, float(eps), None)

    # +X: right-handed completion.
    x = np.cross(y, z)
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(xn, float(eps), None)
    # Re-orthogonalize Z for numerical stability.
    z = np.cross(x, y)
    zn2 = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.clip(zn2, float(eps), None)

    R = np.stack([x, y, z], axis=2)  # (T,3,3), columns are local axes in world coords
    q_th_world = rotmat_to_quat_wxyz(R)  # thigh->world (wxyz)
    return quat_normalize_wxyz(q_th_world).astype(np.float32)


def build_cmu2020_clip_bank(*, side: str = "right", limit: int | None = None) -> ClipBank:
    """Build a reference bank from dm_control's CMU2020 fitted mocap HDF5.

    This does NOT require MoCapAct expert policies or rollout downloads.
    """
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    side = str(side).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    thigh_joint = "lfemurrx" if side == "left" else "rfemurrx"
    knee_joint = "ltibiarx" if side == "left" else "rtibiarx"
    j_th = _joint_index(thigh_joint)
    j_kn = _joint_index(knee_joint)

    # Indices in walker.mocap_tracking_bodies order (dm_control CMU humanoid).
    # Confirmed via `env._env.task._walker.mocap_tracking_bodies`:
    #  left:  lfemur=1, ltibia=2
    #  right: rfemur=6, rtibia=7
    if side == "left":
        femur_i = 1
        tibia_i = 2
    else:
        femur_i = 6
        tibia_i = 7

    h5_path = _cmu_h5_path()
    import h5py

    ids: list[str] = []
    clip_ids: list[str] = []
    start_steps: list[int] = []
    end_steps: list[int] = []
    sample_hzs: list[float] = []
    hips: list[np.ndarray] = []
    thigh_pitches: list[np.ndarray] = []
    knees: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as f:
        keys = [str(k) for k in f.keys() if str(k).startswith("CMU_")]
        keys.sort()
        if limit is not None:
            keys = keys[: int(limit)]

        it = keys
        if tqdm is not None:
            it = tqdm(keys, desc="Building CMU2020 reference bank", unit="clip")
        for clip_id in it:
            g = f[clip_id]
            dt = float(g.attrs["dt"])
            hz = float(1.0 / dt)
            joints = g["walkers"]["walker_0"]["joints"]  # (n_joints, T)
            T = int(joints.shape[1])
            if T < 2:
                continue
            start = 0
            end = T - 1
            hip = np.rad2deg(np.asarray(joints[j_th, :], dtype=np.float32)).astype(np.float32)
            kn = np.rad2deg(np.asarray(joints[j_kn, :], dtype=np.float32)).astype(np.float32)

            # Thigh segment pitch from body positions (world).
            root_pos = np.asarray(g["walkers"]["walker_0"]["position"], dtype=np.float64).T  # (T,3)
            body_pos = np.asarray(g["walkers"]["walker_0"]["body_positions"], dtype=np.float64).T  # (T, 3*nb)
            nb = int(body_pos.shape[1] // 3)
            if nb * 3 != int(body_pos.shape[1]) or nb < max(femur_i, tibia_i) + 1:
                raise RuntimeError(f"Unexpected body_positions shape: {body_pos.shape}")
            fem_pos = body_pos[:, 3 * femur_i : 3 * femur_i + 3]
            tib_pos = body_pos[:, 3 * tibia_i : 3 * tibia_i + 3]
            th_pitch = _thigh_pitch_from_body_positions(root_pos=root_pos, femur_pos=fem_pos, tibia_pos=tib_pos)

            snippet_id = f"{clip_id}-{start}-{end}"
            ids.append(snippet_id)
            clip_ids.append(clip_id)
            start_steps.append(start)
            end_steps.append(end)
            sample_hzs.append(hz)
            hips.append(hip)
            thigh_pitches.append(th_pitch)
            knees.append(kn)

    if not ids:
        raise RuntimeError("CMU2020 clip bank is empty.")

    return ClipBank(
        snippet_id=np.asarray(ids, dtype=object),
        clip_id=np.asarray(clip_ids, dtype=object),
        start_step=np.asarray(start_steps, dtype=np.int32),
        end_step=np.asarray(end_steps, dtype=np.int32),
        sample_hz=np.asarray(sample_hzs, dtype=np.float32),
        hip_deg=np.asarray(hips, dtype=object),
        thigh_pitch_deg=np.asarray(thigh_pitches, dtype=object),
        knee_deg=np.asarray(knees, dtype=object),
    )
