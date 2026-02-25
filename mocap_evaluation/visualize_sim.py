"""
Visualize the prosthetic simulation as a 2D stick-figure GIF.

Runs the same synthetic smoke-test scenario used by run_evaluation --smoke-test,
then animates:
  • Blue stick figure  = ground-truth motion (all joints from mocap)
  • Red  stick figure  = prosthetic motion   (right knee = model prediction)
  • Right panels       = knee angle time series + CoM height comparison

Usage
-----
    python -m mocap_evaluation.visualize_sim                    # saves example_sim.gif
    python -m mocap_evaluation.visualize_sim --out walk.gif --n-frames 300 --fps 25
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # headless — works without a display
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.animation import FuncAnimation, PillowWriter
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    import pybullet as p
    import pybullet_data
    _PB_OK = True
except ImportError:
    _PB_OK = False

from mocap_evaluation.mocap_loader import generate_synthetic_gait, TARGET_FPS
from mocap_evaluation.motion_matching import find_best_match
from mocap_evaluation.prosthetic_sim import (
    _discover_joints, _find_joint, _sagittal_quat,
    HUMANOID_INIT_HEIGHT,
)

# ── Humanoid joint indices we want to draw ────────────────────────────────────
# (link index → label used in SEGMENTS below)
DRAW_LINKS: Dict[str, int] = {
    "pelvis":   -1,   # base link
    "chest":     1,
    "r_hip":     9,
    "r_knee":   10,
    "r_ankle":  11,
    "l_hip":    12,
    "l_knee":   13,
    "l_ankle":  14,
}

# Connectivity — (from, to)
SEGMENTS_SHARED = [            # same in both figures
    ("pelvis", "chest"),
    ("pelvis", "l_hip"),
    ("l_hip",  "l_knee"),
    ("l_knee", "l_ankle"),
]
SEGMENTS_RIGHT = [             # differ between predicted and GT
    ("pelvis", "r_hip"),
    ("r_hip",  "r_knee"),
    ("r_knee", "r_ankle"),
]


def _get_link_positions_xz(robot: int, client: int) -> Dict[str, np.ndarray]:
    """Return {label: [x, z]} for each drawable link."""
    pos = {}
    base_p, _ = p.getBasePositionAndOrientation(robot, physicsClientId=client)
    pos["pelvis"] = np.array([base_p[0], base_p[2]])

    for label, idx in DRAW_LINKS.items():
        if idx < 0:
            continue
        ls = p.getLinkState(robot, idx, physicsClientId=client)
        pos[label] = np.array([ls[0][0], ls[0][2]])   # X and Z (sagittal)

    return pos


def record_frames(
    mocap_segment: dict,
    predicted_knee: np.ndarray,
    fps: float = float(TARGET_FPS),
    subsample: int = 5,
) -> dict:
    """
    Run the simulation and record per-frame link positions for both the
    predicted robot and the ground-truth robot.

    Parameters
    ----------
    subsample : keep every Nth frame (reduce GIF size)

    Returns
    -------
    dict with keys:
      pred_frames  : list of {label: [x,z]} dicts  (predicted robot)
      gt_frames    : list of {label: [x,z]} dicts  (ground-truth robot)
      knee_pred    : (T,)  predicted knee angles
      knee_gt      : (T,)  mocap knee angles
      com_pred     : (T,)  CoM height of predicted robot
      com_gt       : (T,)  CoM height of ground-truth robot
      t_axis       : (T,)  time in seconds
    """
    T = len(predicted_knee)

    # Build root trajectory (synthetic: forward at 1.35 m/s)
    t_s       = np.arange(T) / fps
    root_traj = np.zeros((T, 3), dtype=np.float32)
    root_traj[:, 0] = 1.35 * t_s
    root_traj[:, 2] = HUMANOID_INIT_HEIGHT

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.loadURDF("plane.urdf", physicsClientId=client)

    # Load both robots
    robot = p.loadURDF("humanoid/humanoid.urdf",
                       [float(root_traj[0, 0]), 0.0, float(root_traj[0, 2])],
                       [0, 0, 0, 1], useFixedBase=False, physicsClientId=client)
    robot_gt = p.loadURDF("humanoid/humanoid.urdf",
                          [float(root_traj[0, 0]) + 3.0, 0.0, float(root_traj[0, 2])],
                          [0, 0, 0, 1], useFixedBase=False, physicsClientId=client)

    jmap    = _discover_joints(robot)
    jmap_gt = _discover_joints(robot_gt)

    def _set_kin(rb, jm, kw, ang_deg):
        info = _find_joint(jm, kw)
        if info is None:
            return
        rad = math.radians(ang_deg)
        if info["type"] == p.JOINT_REVOLUTE:
            p.resetJointState(rb, info["index"], rad, physicsClientId=client)
        elif info["type"] == p.JOINT_SPHERICAL:
            p.resetJointStateMultiDof(rb, info["index"], _sagittal_quat(ang_deg),
                                      physicsClientId=client)

    pred_frames: List[Dict[str, np.ndarray]] = []
    gt_frames:   List[Dict[str, np.ndarray]] = []
    com_pred_arr: List[float] = []
    com_gt_arr:   List[float] = []

    for t in range(T):
        fr = {
            "hip_right":   float(mocap_segment["hip_right"][t]),
            "hip_left":    float(mocap_segment["hip_left"][t]),
            "knee_right":  float(mocap_segment["knee_right"][t]),
            "knee_left":   float(mocap_segment["knee_left"][t]),
            "ankle_right": float(mocap_segment["ankle_right"][t]),
            "ankle_left":  float(mocap_segment["ankle_left"][t]),
        }

        # Set base positions
        rp = [float(root_traj[t, 0]), 0.0, float(root_traj[t, 2])]
        gp = [float(root_traj[t, 0]) + 3.0, 0.0, float(root_traj[t, 2])]
        p.resetBasePositionAndOrientation(robot,    rp, [0,0,0,1], physicsClientId=client)
        p.resetBasePositionAndOrientation(robot_gt, gp, [0,0,0,1], physicsClientId=client)

        # Set joints — predicted robot
        for kw, ang in [("righthip", fr["hip_right"]), ("lefthip", fr["hip_left"]),
                        ("leftknee", fr["knee_left"]), ("rightankle", fr["ankle_right"]),
                        ("leftankle", fr["ankle_left"])]:
            _set_kin(robot, jmap, kw, ang)
        _set_kin(robot, jmap, "rightknee", float(predicted_knee[t]))

        # Set joints — ground truth robot
        for kw, ang in [("righthip", fr["hip_right"]), ("rightknee", fr["knee_right"]),
                        ("lefthip", fr["hip_left"]),   ("leftknee", fr["knee_left"]),
                        ("rightankle", fr["ankle_right"]), ("leftankle", fr["ankle_left"])]:
            _set_kin(robot_gt, jmap_gt, kw, ang)

        p.stepSimulation(physicsClientId=client)

        # Record positions (subsampled)
        if t % subsample == 0:
            # Offset GT positions so both figures appear at same X
            raw_pred = _get_link_positions_xz(robot,    client)
            raw_gt   = _get_link_positions_xz(robot_gt, client)
            # Centre both at the same X
            pred_x0 = raw_pred["pelvis"][0]
            gt_x0   = raw_gt["pelvis"][0]
            pred_pos = {k: v - np.array([pred_x0, 0]) for k, v in raw_pred.items()}
            gt_pos   = {k: v - np.array([gt_x0,   0]) for k, v in raw_gt.items()}
            pred_frames.append(pred_pos)
            gt_frames.append(gt_pos)

        # Always record time series
        # Simplified CoM: just use pelvis height (good enough for visualisation)
        base_pred, _ = p.getBasePositionAndOrientation(robot,    physicsClientId=client)
        base_gt,   _ = p.getBasePositionAndOrientation(robot_gt, physicsClientId=client)
        com_pred_arr.append(base_pred[2])
        com_gt_arr.append(base_gt[2])

    p.disconnect(client)

    return {
        "pred_frames": pred_frames,
        "gt_frames":   gt_frames,
        "knee_pred":   predicted_knee[:T],
        "knee_gt":     mocap_segment["knee_right"][:T],
        "com_pred":    np.array(com_pred_arr, dtype=np.float32),
        "com_gt":      np.array(com_gt_arr,   dtype=np.float32),
        "t_axis":      t_s,
    }


def _draw_stick(ax, pos: Dict[str, np.ndarray], color: str, alpha: float = 1.0,
                lw: float = 3.0, label: Optional[str] = None):
    """Draw one stick figure on axes `ax`."""
    def seg(a, b, **kw):
        if a in pos and b in pos:
            xs = [pos[a][0], pos[b][0]]
            zs = [pos[a][1], pos[b][1]]
            ax.plot(xs, zs, color=color, linewidth=lw, alpha=alpha,
                    solid_capstyle="round", **kw)

    first = True
    for (a, b) in SEGMENTS_SHARED:
        lbl = label if first else None
        seg(a, b, label=lbl)
        first = False
    for (a, b) in SEGMENTS_RIGHT:
        seg(a, b)

    # Draw joint dots
    for k, v in pos.items():
        ax.plot(v[0], v[1], "o", color=color, markersize=5, alpha=alpha)

    # Head
    if "chest" in pos:
        cx, cz = pos["chest"]
        head = plt.Circle((cx, cz + 0.12), 0.07, color=color, alpha=alpha, fill=False, lw=2)
        ax.add_patch(head)


def generate_animation(
    data: dict,
    out_path: str | Path = "example_sim.gif",
    anim_fps: int = 20,
    total_frames: Optional[int] = None,
) -> Path:
    """
    Generate a GIF animation from recorded simulation frames.

    Parameters
    ----------
    data       : output of record_frames()
    out_path   : destination GIF path
    anim_fps   : frames per second in the output GIF
    total_frames : if set, only animate this many frames (for quick previews)
    """
    out_path = Path(out_path)
    pred_frames = data["pred_frames"]
    gt_frames   = data["gt_frames"]

    if total_frames is not None:
        pred_frames = pred_frames[:total_frames]
        gt_frames   = gt_frames[:total_frames]

    N = len(pred_frames)
    t_axis   = data["t_axis"]
    knee_pred = data["knee_pred"]
    knee_gt   = data["knee_gt"]
    com_pred  = data["com_pred"]
    com_gt    = data["com_gt"]

    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.45, wspace=0.35)

    ax_stick  = fig.add_subplot(gs[:, 0])   # stick figure (full left column)
    ax_knee   = fig.add_subplot(gs[0, 1])   # top-right: knee angle
    ax_com    = fig.add_subplot(gs[1, 1])   # bottom-right: CoM height

    # ── Static time-series panels ─────────────────────────────────────────
    ax_knee.plot(t_axis, knee_gt,   color="royalblue", lw=1.5,
                 label="Mocap (ground truth)")
    ax_knee.plot(t_axis, knee_pred, color="tomato",    lw=1.5,
                 label="Model prediction")
    ax_knee.set_xlabel("Time (s)"); ax_knee.set_ylabel("Knee angle (°)")
    ax_knee.set_title("Right Knee Flexion"); ax_knee.legend(fontsize=7)
    ax_knee.set_ylim(-5, 75)

    ax_com.plot(t_axis, com_gt,   color="royalblue", lw=1.5)
    ax_com.plot(t_axis, com_pred, color="tomato",    lw=1.5)
    ax_com.axhline(0.55, color="black", ls="--", lw=1, label="Fall threshold")
    ax_com.set_xlabel("Time (s)"); ax_com.set_ylabel("Pelvis height (m)")
    ax_com.set_title("Pelvis / CoM Height"); ax_com.legend(fontsize=7)
    ax_com.set_ylim(0.4, 1.1)

    # Vertical time cursors (will be updated each frame)
    vl_knee = ax_knee.axvline(0, color="gray", lw=1, ls=":")
    vl_com  = ax_com.axvline(0, color="gray", lw=1, ls=":")

    # ── Stick figure panel ────────────────────────────────────────────────
    ax_stick.set_xlim(-0.75, 0.75)
    ax_stick.set_ylim(-0.15, 1.55)
    ax_stick.set_aspect("equal")
    ax_stick.set_xlabel("Lateral offset (m)")
    ax_stick.set_ylabel("Height (m)")
    ax_stick.set_title("Humanoid Stick Figure (sagittal view)")
    ax_stick.axhline(0, color="sienna", lw=2)   # ground line

    legend_handles = [
        mpatches.Patch(color="royalblue", label="Ground truth (mocap)"),
        mpatches.Patch(color="tomato",    label="Prosthetic (model prediction)"),
    ]
    ax_stick.legend(handles=legend_handles, fontsize=8, loc="upper right")

    # Artists we'll redraw each frame
    stick_artists: List = []

    def init_func():
        return []

    def animate(i):
        nonlocal stick_artists
        # Clear previous stick figures
        for art in stick_artists:
            art.remove()
        stick_artists = []

        # Compute fraction through simulation for time cursor
        frac     = i / max(N - 1, 1)
        t_cursor = frac * t_axis[-1]
        vl_knee.set_xdata([t_cursor, t_cursor])
        vl_com.set_xdata([t_cursor, t_cursor])

        # Draw GT first (behind), then predicted (in front)
        gt_pos   = gt_frames[i]
        pred_pos = pred_frames[i]

        def seg_patch(a, b, pos, color, alpha=1.0):
            if a in pos and b in pos:
                line, = ax_stick.plot(
                    [pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=color, lw=3, alpha=alpha, solid_capstyle="round"
                )
                stick_artists.append(line)
            for k in (a, b):
                if k in pos:
                    dot, = ax_stick.plot(pos[k][0], pos[k][1], "o",
                                         color=color, markersize=5, alpha=alpha)
                    stick_artists.append(dot)

        def head_patch(pos, color, alpha=1.0):
            if "chest" in pos:
                cx, cz = pos["chest"]
                circle = plt.Circle((cx, cz + 0.12), 0.07,
                                    color=color, fill=False, lw=2, alpha=alpha)
                ax_stick.add_patch(circle)
                stick_artists.append(circle)

        # Ground-truth — blue, slightly transparent
        for (a, b) in SEGMENTS_SHARED + SEGMENTS_RIGHT:
            seg_patch(a, b, gt_pos,   "royalblue", alpha=0.7)
        head_patch(gt_pos, "royalblue", alpha=0.7)

        # Predicted — red, full opacity
        for (a, b) in SEGMENTS_SHARED:
            seg_patch(a, b, pred_pos, "tomato", alpha=0.9)
        for (a, b) in SEGMENTS_RIGHT:
            seg_patch(a, b, pred_pos, "tomato", alpha=0.9)
        head_patch(pred_pos, "tomato", alpha=0.9)

        return stick_artists + [vl_knee, vl_com]

    ani = FuncAnimation(
        fig, animate, frames=N,
        init_func=init_func,
        interval=1000 // anim_fps,
        blit=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(str(out_path), writer=PillowWriter(fps=anim_fps))
    plt.close(fig)
    print(f"Saved animation → {out_path}  ({N} frames @ {anim_fps} fps)")
    return out_path


def run_example(
    out: str = "example_sim.gif",
    n_frames: int = 200,
    anim_fps: int = 20,
    noise_std: float = 8.0,
) -> Path:
    """
    Full end-to-end example using synthetic signals.

    Parameters
    ----------
    out        : output GIF path
    n_frames   : number of simulation frames to animate (200 ≈ 1 second)
    anim_fps   : GIF playback speed
    noise_std  : std of random knee prediction error in degrees
                 (higher = more visible difference vs ground truth)
    """
    fps = TARGET_FPS   # 200 Hz
    T   = max(n_frames * 5, 400)   # simulate more than we animate

    print(f"Generating synthetic gait database …")
    db = generate_synthetic_gait(n_cycles=20)

    # Synthetic "recorded" walking signal
    t     = np.linspace(0, 2 * math.pi * (T / fps), T)
    knee_imu  = (30 + 28 * np.sin(t)).astype(np.float32)
    thigh_imu = (13 * np.sin(t + 0.4)).astype(np.float32)

    # Synthetic model prediction = ground truth + structured error
    # (simulates a model that has ~noise_std° RMSE, with some phase shift too)
    rng = np.random.default_rng(42)
    phase_err = 0.15   # radians — slight timing offset
    pred_knee = (
        30 + 28 * np.sin(t + phase_err)
        + rng.normal(0, noise_std, T)
    ).clip(0, 80).astype(np.float32)

    print(f"Running DTW motion matching …")
    _, dtw_dist, segment = find_best_match(knee_imu, thigh_imu, db)
    print(f"  DTW distance = {dtw_dist:.4f}")

    print(f"Recording simulation frames (T={T}) …")
    data = record_frames(segment, pred_knee, fps=float(fps), subsample=5)

    pred_rmse = float(np.sqrt(np.mean((pred_knee - segment["knee_right"][:T]) ** 2)))
    print(f"  Knee prediction RMSE = {pred_rmse:.2f}°")

    print(f"Generating animation …")
    out_path = generate_animation(data, out_path=out, anim_fps=anim_fps,
                                  total_frames=n_frames // 5)

    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate a stick-figure GIF of the prosthetic simulation"
    )
    ap.add_argument("--out",      default="example_sim.gif")
    ap.add_argument("--n-frames", type=int, default=200,
                    help="Simulation frames to include in GIF (200 ≈ 1 s)")
    ap.add_argument("--fps",      type=int, default=20,
                    help="GIF playback FPS")
    ap.add_argument("--noise",    type=float, default=8.0,
                    help="Std of synthetic prediction error (degrees)")
    args = ap.parse_args()

    if not _MPL_OK:
        raise RuntimeError("matplotlib is required: pip install matplotlib")
    if not _PB_OK:
        raise RuntimeError("pybullet is required:  pip install pybullet")

    out = run_example(out=args.out, n_frames=args.n_frames,
                      anim_fps=args.fps, noise_std=args.noise)
    print(f"\nDone.  Open {out} to view the animation.")


if __name__ == "__main__":
    main()
