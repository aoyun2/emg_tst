"""
BVH (Biovision Hierarchy) motion capture file parser.

Handles standard BVH files including the CMU Graphics Lab Motion Capture
Database format. Joints store Euler rotation channels; root additionally
has position channels.

Typical CMU BVH channel order per joint: Zrotation Xrotation Yrotation
Root has: Xposition Yposition Zposition Zrotation Xrotation Yrotation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BVHJoint:
    name: str
    parent: Optional["BVHJoint"]
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    channels: List[str] = field(default_factory=list)  # e.g. ["Xrotation","Yrotation","Zrotation"]
    channel_start: int = 0                               # global index into motion data columns
    children: List["BVHJoint"] = field(default_factory=list)

    @property
    def rotation_channel_indices(self) -> List[Tuple[str, int]]:
        """(channel_name, global_col_index) for rotation channels only."""
        return [
            (ch, self.channel_start + k)
            for k, ch in enumerate(self.channels)
            if "rotation" in ch.lower()
        ]

    @property
    def position_channel_indices(self) -> List[Tuple[str, int]]:
        """(channel_name, global_col_index) for position channels only."""
        return [
            (ch, self.channel_start + k)
            for k, ch in enumerate(self.channels)
            if "position" in ch.lower()
        ]


class BVHParser:
    """
    Parse a BVH file into joint hierarchy + motion data.

    Usage::

        parser = BVHParser().parse("walk.bvh")
        knee_flex = parser.get_flexion("RightLeg")   # (n_frames,) degrees
        hip_flex  = parser.get_flexion("RightUpLeg")
        root_pos  = parser.get_positions("Hips")     # (n_frames, 3)
    """

    def __init__(self):
        self.root: Optional[BVHJoint] = None
        self.joints: Dict[str, BVHJoint] = {}
        self.n_frames: int = 0
        self.frame_time: float = 0.0
        self.data: Optional[np.ndarray] = None   # (n_frames, n_total_channels)
        self._n_channels: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, path: str | Path) -> "BVHParser":
        # utf-8-sig strips BOM automatically (common on Windows-edited files)
        lines = Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines()
        i = 0
        while i < len(lines) and lines[i].strip().upper() != "HIERARCHY":
            i += 1
        if i >= len(lines):
            raise ValueError(f"BVH parse error in {path}: HIERARCHY section not found")
        i += 1  # skip 'HIERARCHY'

        self.root, i = self._parse_node(lines, i)

        # Locate MOTION section
        while i < len(lines) and lines[i].strip().upper() != "MOTION":
            i += 1
        if i >= len(lines):
            raise ValueError(f"BVH parse error in {path}: MOTION section not found")
        i += 1

        # Find "Frames:" line (skip any blanks or comments)
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.lower().startswith("frames") and ":" in stripped:
                break
            if stripped and not stripped.startswith("#"):
                raise ValueError(
                    f"BVH parse error in {path}: expected 'Frames:' "
                    f"but got {stripped!r} at line {i+1}"
                )
            i += 1
        if i >= len(lines):
            raise ValueError(f"BVH parse error in {path}: 'Frames:' line not found")
        self.n_frames = int(lines[i].split(":", 1)[1].strip())
        i += 1

        # Find "Frame Time:" line
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.lower().startswith("frame") and ":" in stripped:
                break
            if stripped and not stripped.startswith("#"):
                raise ValueError(
                    f"BVH parse error in {path}: expected 'Frame Time:' "
                    f"but got {stripped!r} at line {i+1}"
                )
            i += 1
        if i >= len(lines):
            raise ValueError(f"BVH parse error in {path}: 'Frame Time:' line not found")
        self.frame_time = float(lines[i].split(":", 1)[1].strip())
        i += 1

        rows = []
        while i < len(lines):
            s = lines[i].strip()
            if s:
                try:
                    rows.append(list(map(float, s.split())))
                except ValueError:
                    pass
            i += 1

        self.data = np.array(rows[: self.n_frames], dtype=np.float32)
        return self

    def get_rotations(self, joint_name: str) -> Optional[np.ndarray]:
        """
        Return rotation angles for all rotation channels of a joint.
        Shape: (n_frames, n_rot_channels) in degrees.
        """
        if joint_name not in self.joints or self.data is None:
            return None
        items = self.joints[joint_name].rotation_channel_indices
        if not items:
            return None
        cols = [idx for _, idx in items]
        return self.data[:, cols]

    def get_positions(self, joint_name: str) -> Optional[np.ndarray]:
        """
        Return position channels (usually only root/Hips).
        Shape: (n_frames, n_pos_channels).
        """
        if joint_name not in self.joints or self.data is None:
            return None
        items = self.joints[joint_name].position_channel_indices
        if not items:
            return None
        cols = [idx for _, idx in items]
        return self.data[:, cols]

    def get_channel(self, joint_name: str, channel: str) -> Optional[np.ndarray]:
        """Return a single named channel (e.g. 'Xrotation') as (n_frames,)."""
        if joint_name not in self.joints or self.data is None:
            return None
        j = self.joints[joint_name]
        for k, ch in enumerate(j.channels):
            if ch.lower() == channel.lower():
                return self.data[:, j.channel_start + k]
        return None

    def get_flexion(self, joint_name: str, prefer_channel: str = "Xrotation") -> Optional[np.ndarray]:
        """
        Heuristic: return the most likely flexion/extension angle (n_frames,) in degrees.

        Priority order:
          1. Channel named `prefer_channel` if it exists
          2. The rotation channel with the highest variance (most movement)

        For CMU BVH walking data, knee and hip flexion/extension is typically
        the Xrotation channel (sagittal plane).
        """
        if joint_name not in self.joints or self.data is None:
            return None
        rots = self.get_rotations(joint_name)
        if rots is None or rots.shape[1] == 0:
            return None

        # Try preferred channel first
        ch = self.get_channel(joint_name, prefer_channel)
        if ch is not None:
            return ch

        # Fall back to highest-variance channel
        variances = rots.var(axis=0)
        return rots[:, int(np.argmax(variances))]

    def list_joints(self) -> List[str]:
        return list(self.joints.keys())

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time if self.frame_time > 0 else 120.0

    @property
    def duration(self) -> float:
        return self.n_frames * self.frame_time

    # ------------------------------------------------------------------
    # Internal parsing helpers
    # ------------------------------------------------------------------

    def _parse_node(self, lines: List[str], i: int) -> Tuple[Optional[BVHJoint], int]:
        # Skip blank lines
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            return None, i

        tok = lines[i].strip().split()
        if tok[0] not in ("ROOT", "JOINT"):
            return None, i + 1

        name = tok[1]
        joint = BVHJoint(name=name, parent=None)
        self.joints[name] = joint

        # Handle '{' on the same line as ROOT/JOINT: "ROOT Hips {"
        if "{" in tok:
            i += 1  # skip this line, already consumed the '{'
        else:
            i += 1  # skip 'ROOT/JOINT name' line
            # Find opening {
            while i < len(lines) and lines[i].strip() != "{":
                i += 1
            i += 1  # skip '{'

        while i < len(lines):
            tok_str = lines[i].strip()
            if not tok_str:
                i += 1
                continue

            tok = tok_str.split()
            kw = tok[0]

            if kw == "OFFSET":
                joint.offset = np.array([float(tok[1]), float(tok[2]), float(tok[3])])
                i += 1

            elif kw == "CHANNELS":
                n = int(tok[1])
                joint.channel_start = self._n_channels
                joint.channels = tok[2 : 2 + n]
                self._n_channels += n
                i += 1

            elif kw == "JOINT":
                child, i = self._parse_node(lines, i)
                if child is not None:
                    child.parent = joint
                    joint.children.append(child)

            elif kw == "End":  # 'End Site' or 'End Site {'
                # Check if '{' is on the same line
                brace_on_line = "{" in tok_str
                i += 1  # skip 'End Site' line
                depth = 1 if brace_on_line else 0
                while i < len(lines):
                    t = lines[i].strip()
                    if "{" in t:
                        depth += 1
                    if "}" in t:
                        depth -= 1
                        if depth <= 0:
                            i += 1
                            break
                    i += 1

            elif kw == "}":
                i += 1
                break

            else:
                i += 1

        return joint, i
