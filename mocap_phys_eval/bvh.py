from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .utils import rotmat_to_quat_wxyz


@dataclass
class BvhJoint:
    name: str
    parent: int | None
    children: list[int]
    offset: np.ndarray  # (3,)
    channels: list[str]
    channel_start: int
    channel_count: int


@dataclass
class BvhData:
    joints: list[BvhJoint]
    joint_index: dict[str, int]
    frame_time_s: float
    motion: np.ndarray  # (F, C) float32

    @property
    def n_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.motion.shape[1])


def _rot_x(a_rad: float) -> np.ndarray:
    c = float(np.cos(a_rad))
    s = float(np.sin(a_rad))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_y(a_rad: float) -> np.ndarray:
    c = float(np.cos(a_rad))
    s = float(np.sin(a_rad))
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(a_rad: float) -> np.ndarray:
    c = float(np.cos(a_rad))
    s = float(np.sin(a_rad))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot(axis: str, angle_deg: float) -> np.ndarray:
    a = float(np.deg2rad(float(angle_deg)))
    if axis == "X":
        return _rot_x(a)
    if axis == "Y":
        return _rot_y(a)
    if axis == "Z":
        return _rot_z(a)
    raise ValueError(f"Unknown rotation axis: {axis!r}")


def load_bvh(path: str | Path) -> BvhData:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or "HIERARCHY" not in lines[0].upper():
        raise RuntimeError(f"Not a BVH file (missing HIERARCHY): {p}")

    joints: list[BvhJoint] = []
    joint_index: dict[str, int] = {}
    stack: list[int] = []
    current: int | None = None
    channel_cursor = 0

    i = 0
    # Parse hierarchy.
    while i < len(lines):
        ln = lines[i]
        up = ln.upper()
        if up == "MOTION":
            break
        if up.startswith("ROOT ") or up.startswith("JOINT "):
            name = ln.split(maxsplit=1)[1].strip()
            parent = stack[-1] if stack else None
            idx = len(joints)
            joints.append(
                BvhJoint(
                    name=name,
                    parent=parent,
                    children=[],
                    offset=np.zeros(3, dtype=np.float64),
                    channels=[],
                    channel_start=channel_cursor,
                    channel_count=0,
                )
            )
            joint_index[name] = idx
            if parent is not None:
                joints[parent].children.append(idx)
            current = idx
            i += 1
            continue
        if up.startswith("END SITE"):
            parent = stack[-1] if stack else current
            if parent is None:
                raise RuntimeError("BVH End Site without a parent joint.")
            name = f"{joints[parent].name}__end"
            idx = len(joints)
            joints.append(
                BvhJoint(
                    name=name,
                    parent=parent,
                    children=[],
                    offset=np.zeros(3, dtype=np.float64),
                    channels=[],
                    channel_start=channel_cursor,
                    channel_count=0,
                )
            )
            if parent is not None:
                joints[parent].children.append(idx)
            current = idx
            i += 1
            continue
        if ln == "{":
            if current is None:
                raise RuntimeError("BVH '{' without an active joint.")
            stack.append(current)
            i += 1
            continue
        if ln == "}":
            if stack:
                stack.pop()
            current = stack[-1] if stack else None
            i += 1
            continue
        if up.startswith("OFFSET "):
            if current is None:
                raise RuntimeError("BVH OFFSET without an active joint.")
            parts = ln.split()
            if len(parts) != 4:
                raise RuntimeError(f"Bad OFFSET line: {ln!r}")
            joints[current].offset = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            i += 1
            continue
        if up.startswith("CHANNELS "):
            if current is None:
                raise RuntimeError("BVH CHANNELS without an active joint.")
            parts = ln.split()
            if len(parts) < 3:
                raise RuntimeError(f"Bad CHANNELS line: {ln!r}")
            n = int(parts[1])
            chans = [c.strip() for c in parts[2 : 2 + n]]
            if len(chans) != n:
                raise RuntimeError(f"CHANNELS count mismatch: {ln!r}")
            joints[current].channels = chans
            joints[current].channel_start = channel_cursor
            joints[current].channel_count = n
            channel_cursor += n
            i += 1
            continue

        # Ignore unknown lines in hierarchy (rare).
        i += 1

    if i >= len(lines) or lines[i].upper() != "MOTION":
        raise RuntimeError(f"BVH missing MOTION section: {p}")
    i += 1

    # Parse motion header.
    n_frames = None
    frame_time = None
    while i < len(lines):
        ln = lines[i]
        up = ln.upper()
        if up.startswith("FRAMES:"):
            n_frames = int(ln.split(":")[1].strip())
            i += 1
            continue
        if up.startswith("FRAME TIME:"):
            frame_time = float(ln.split(":")[1].strip())
            i += 1
            break
        i += 1

    if n_frames is None or frame_time is None:
        raise RuntimeError(f"BVH missing Frames/Frame Time header: {p}")

    total_channels = int(channel_cursor)
    if total_channels <= 0:
        raise RuntimeError(f"BVH has no motion channels: {p}")

    # Parse motion data: tolerate line-wrapping by token accumulation.
    motion = np.zeros((int(n_frames), int(total_channels)), dtype=np.float32)
    for f in range(int(n_frames)):
        vals: list[float] = []
        while len(vals) < total_channels:
            if i >= len(lines):
                raise RuntimeError(f"BVH ended early while reading motion frame {f}/{n_frames}.")
            vals.extend([float(x) for x in lines[i].split()])
            i += 1
        motion[f, :] = np.asarray(vals[:total_channels], dtype=np.float32)

    return BvhData(joints=joints, joint_index=joint_index, frame_time_s=float(frame_time), motion=motion)


def _axis_index(axis: str) -> int:
    if axis == "X":
        return 0
    if axis == "Y":
        return 1
    if axis == "Z":
        return 2
    raise ValueError(f"Unknown axis: {axis!r}")


def _joint_channel_slices(j: BvhJoint) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Return (pos_axes, rot_axes) where entries are (axis, channel_index)."""
    pos: list[tuple[str, int]] = []
    rot: list[tuple[str, int]] = []
    for k, ch in enumerate(j.channels):
        idx = int(j.channel_start + k)
        if ch.endswith("position"):
            pos.append((ch[0].upper(), idx))
        elif ch.endswith("rotation"):
            rot.append((ch[0].upper(), idx))
    return pos, rot


def joint_world_positions(bvh: BvhData, *, joint_names: Iterable[str]) -> tuple[np.ndarray, float]:
    """Compute world positions for selected joints. Returns (pos, sample_hz).

    pos has shape (F, J, 3) in BVH units (typically cm).
    """
    names = [str(n) for n in joint_names]
    ids = []
    for n in names:
        if n not in bvh.joint_index:
            raise KeyError(f"BVH missing joint {n!r}. Available examples: {list(bvh.joint_index)[:10]}...")
        ids.append(int(bvh.joint_index[n]))

    # Precompute per-joint channel index mappings.
    parents = [j.parent for j in bvh.joints]
    offsets = [np.asarray(j.offset, dtype=np.float64).reshape(3) for j in bvh.joints]
    pos_axes: list[list[tuple[str, int]]] = []
    rot_axes: list[list[tuple[str, int]]] = []
    for j in bvh.joints:
        p, r = _joint_channel_slices(j)
        pos_axes.append(p)
        rot_axes.append(r)

    F = bvh.n_frames
    out = np.zeros((int(F), int(len(ids)), 3), dtype=np.float64)

    # We still need all parent rotations to propagate, but we only store selected positions.
    world_pos = np.zeros((len(bvh.joints), 3), dtype=np.float64)
    world_rot = np.zeros((len(bvh.joints), 3, 3), dtype=np.float64)

    for f in range(int(F)):
        row = np.asarray(bvh.motion[f], dtype=np.float64).reshape(-1)
        for j_idx, j in enumerate(bvh.joints):
            # Local translation.
            lp = offsets[j_idx].copy()
            for ax, ch_i in pos_axes[j_idx]:
                lp[_axis_index(ax)] += float(row[int(ch_i)])

            # Local rotation (intrinsic, BVH channel order).
            lr = np.eye(3, dtype=np.float64)
            for ax, ch_i in rot_axes[j_idx]:
                lr = lr @ _rot(ax, float(row[int(ch_i)]))

            parent = parents[j_idx]
            if parent is None:
                world_pos[j_idx] = lp
                world_rot[j_idx] = lr
            else:
                world_pos[j_idx] = world_pos[parent] + world_rot[parent] @ lp
                world_rot[j_idx] = world_rot[parent] @ lr

        for k, jid in enumerate(ids):
            out[f, k, :] = world_pos[jid]

    hz = float(1.0 / float(bvh.frame_time_s))
    return out.astype(np.float32), hz


def joint_world_transforms(bvh: BvhData, *, joint_names: Iterable[str]) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute world positions and rotations for selected joints.

    Returns (pos, rot, sample_hz) where:
    - pos has shape (F, J, 3)
    - rot has shape (F, J, 3, 3)
    """
    names = [str(n) for n in joint_names]
    ids: list[int] = []
    for n in names:
        if n not in bvh.joint_index:
            raise KeyError(f"BVH missing joint {n!r}. Available examples: {list(bvh.joint_index)[:10]}...")
        ids.append(int(bvh.joint_index[n]))

    parents = [j.parent for j in bvh.joints]
    offsets = [np.asarray(j.offset, dtype=np.float64).reshape(3) for j in bvh.joints]
    pos_axes: list[list[tuple[str, int]]] = []
    rot_axes: list[list[tuple[str, int]]] = []
    for j in bvh.joints:
        p, r = _joint_channel_slices(j)
        pos_axes.append(p)
        rot_axes.append(r)

    F = bvh.n_frames
    out_pos = np.zeros((int(F), int(len(ids)), 3), dtype=np.float64)
    out_rot = np.zeros((int(F), int(len(ids)), 3, 3), dtype=np.float64)

    world_pos = np.zeros((len(bvh.joints), 3), dtype=np.float64)
    world_rot = np.zeros((len(bvh.joints), 3, 3), dtype=np.float64)

    for f in range(int(F)):
        row = np.asarray(bvh.motion[f], dtype=np.float64).reshape(-1)
        for j_idx, j in enumerate(bvh.joints):
            lp = offsets[j_idx].copy()
            for ax, ch_i in pos_axes[j_idx]:
                lp[_axis_index(ax)] += float(row[int(ch_i)])

            lr = np.eye(3, dtype=np.float64)
            for ax, ch_i in rot_axes[j_idx]:
                lr = lr @ _rot(ax, float(row[int(ch_i)]))

            parent = parents[j_idx]
            if parent is None:
                world_pos[j_idx] = lp
                world_rot[j_idx] = lr
            else:
                world_pos[j_idx] = world_pos[parent] + world_rot[parent] @ lp
                world_rot[j_idx] = world_rot[parent] @ lr

        for k, jid in enumerate(ids):
            out_pos[f, k, :] = world_pos[jid]
            out_rot[f, k, :, :] = world_rot[jid]

    hz = float(1.0 / float(bvh.frame_time_s))
    return out_pos.astype(np.float32), out_rot.astype(np.float32), hz


def _bvh_to_mujoco_basis() -> np.ndarray:
    """Rotation mapping BVH basis (X right, Y up, Z forward) -> MuJoCo-like (X fwd, Y right, Z up)."""
    # v_mj = B @ v_bvh
    return np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)


def extract_right_leg_thigh_quat_and_knee_included_deg(
    bvh: BvhData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (thigh_pitch_deg, thigh_quat_wxyz, knee_included_deg, sample_hz).

    Conventions:
    - BVH is assumed to be in a Y-up world. We map it into a Z-up, MuJoCo-like basis
      before extracting the thigh pitch (for motion matching consistency).
    - Thigh pitch is a *signed* pitch angle in the sagittal plane:
        pitch = atan2(forward_component, down_component)
      where down_component = -thigh_vec_z (after basis mapping).
    - Thigh quaternion is an IMU-like *segment frame* orientation in world coordinates
      (segment->world), expressed in the same MuJoCo-like basis, as wxyz. The local axes are:
        +Y distal (hip -> knee), +Z posterior/back (opposite clip forward), +X completes
        a right-handed frame (points roughly to subject's left).
    - Knee included angle is the included angle at the knee:
        180 = straight, 0 = fully folded.
    """
    def _pick_right_leg_names() -> tuple[str, str, str, str]:
        """Select (root, hip, knee, ankle) joint names for the right leg.

        We support multiple BVH naming conventions so demo clips can come from
        non-CMU datasets (e.g., Mixamo/Bandai).
        """
        names = bvh.joint_index
        root = bvh.joints[0].name

        triples: tuple[tuple[str, str, str], ...] = (
            # CMU-style
            ("RightUpLeg", "RightLeg", "RightFoot"),
            # Mixamo-style / Bandai Namco Research dataset
            ("UpperLeg_R", "LowerLeg_R", "Foot_R"),
        )
        for hip, knee, ankle in triples:
            if hip in names and knee in names and ankle in names:
                return root, hip, knee, ankle

        # Fallback: try a few common alternatives.
        hip_cands = (
            "RightUpLeg",
            "UpperLeg_R",
            "UpLeg_R",
            "Thigh_R",
            "RightThigh",
        )
        knee_cands = (
            "RightLeg",
            "LowerLeg_R",
            "Leg_R",
            "Shin_R",
            "RightShin",
        )
        ankle_cands = (
            "RightFoot",
            "Foot_R",
            "Ankle_R",
            "RightAnkle",
        )

        hip = next((n for n in hip_cands if n in names), None)
        knee = next((n for n in knee_cands if n in names), None)
        ankle = next((n for n in ankle_cands if n in names), None)
        if hip and knee and ankle:
            return root, hip, knee, ankle

        # If we got here, provide a useful error message.
        available = sorted(list(names.keys()))
        preview = ", ".join(available[:40]) + (" ..." if len(available) > 40 else "")
        raise KeyError(
            "Could not find right-leg joints in BVH. "
            "Tried CMU-style (RightUpLeg/RightLeg/RightFoot) and Mixamo-style "
            "(UpperLeg_R/LowerLeg_R/Foot_R). "
            f"Available joints (preview): {preview}"
        )

    root_name, hip_name, knee_name, ankle_name = _pick_right_leg_names()

    pos_bvh, rot_bvh, hz = joint_world_transforms(bvh, joint_names=[root_name, hip_name, knee_name, ankle_name])

    B = _bvh_to_mujoco_basis()
    # Positions: treat last axis as row-vector; pos_mj = pos_bvh @ B^T.
    pos = (np.asarray(pos_bvh, dtype=np.float64) @ B.T).astype(np.float64)
    root = pos[:, 0, :]
    hip = pos[:, 1, :]
    knee = pos[:, 2, :]
    ankle = pos[:, 3, :]

    # Forward direction inferred from overall root displacement in the XY plane (MuJoCo Z-up).
    disp = (root[-1, :2] - root[0, :2]).astype(np.float64)
    dn = float(np.linalg.norm(disp))
    if dn < 1e-6:
        fwd2 = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        fwd2 = disp / dn

    thigh_vec = knee - hip  # hip -> knee
    down = -thigh_vec[:, 2]
    fwd_comp = thigh_vec[:, 0] * float(fwd2[0]) + thigh_vec[:, 1] * float(fwd2[1])
    thigh_pitch = np.degrees(np.arctan2(fwd_comp, down)).astype(np.float32)

    # Knee included angle: vectors emanating from knee.
    thigh_up = hip - knee
    shank_down = ankle - knee
    tu = thigh_up / np.clip(np.linalg.norm(thigh_up, axis=1, keepdims=True), 1e-6, None)
    su = shank_down / np.clip(np.linalg.norm(shank_down, axis=1, keepdims=True), 1e-6, None)
    dot = np.sum(tu * su, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    knee_included = np.degrees(np.arccos(dot)).astype(np.float32)

    # Thigh "IMU-like" segment quaternion in root frame.
    # We derive a segment frame from positions to keep the axis convention consistent
    # with the evaluation reference bank (and the user's IMU mount description).
    fwd3 = np.asarray([float(fwd2[0]), float(fwd2[1]), 0.0], dtype=np.float64)
    fwd3n = float(np.linalg.norm(fwd3))
    if fwd3n < 1e-9:
        fwd3 = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        fwd3 = fwd3 / fwd3n
    back = -fwd3  # posterior

    y = knee - hip  # distal axis
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    y = y / np.clip(yn, 1e-8, None)

    backv = back.reshape(1, 3)
    z = backv - np.sum(backv * y, axis=1, keepdims=True) * y
    zn = np.linalg.norm(z, axis=1, keepdims=True)
    bad = (zn.reshape(-1) < 1e-8) | ~np.isfinite(zn.reshape(-1))
    if bool(np.any(bad)):
        up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64).reshape(1, 3)
        z[bad] = np.cross(y[bad], up)
        zn = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.clip(zn, 1e-8, None)

    x = np.cross(y, z)
    xn = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(xn, 1e-8, None)
    z = np.cross(x, y)
    zn2 = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.clip(zn2, 1e-8, None)

    R_th = np.stack([x, y, z], axis=2)  # thigh->world
    q_th_world = rotmat_to_quat_wxyz(R_th).astype(np.float32)

    # For evaluation, we return thigh->world (IMU-style), not root-relative.
    return thigh_pitch, q_th_world, knee_included, float(hz)
