from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_motion_match(
    *,
    out_path: str | Path,
    sample_hz: float,
    ref_thigh_deg: np.ndarray | None,
    ref_knee_deg: np.ndarray,
    query_thigh_aligned_deg: np.ndarray | None,
    query_knee_aligned_deg: np.ndarray,
    rmse_thigh_deg: float,
    rmse_knee_deg: float,
    thigh_ori_err_deg: np.ndarray | None = None,
    title: str,
) -> Path:
    has_thigh = ref_thigh_deg is not None and query_thigh_aligned_deg is not None
    if has_thigh:
        n_base = int(min(len(ref_thigh_deg), len(query_thigh_aligned_deg), len(ref_knee_deg), len(query_knee_aligned_deg)))  # type: ignore[arg-type]
    else:
        n_base = int(min(len(ref_knee_deg), len(query_knee_aligned_deg)))
    t = np.arange(int(n_base), dtype=np.float32) / float(sample_hz)

    # Layout:
    # - Thigh pitch subplot only if provided (BVH demo)
    # - Knee subplot always
    # - Thigh orientation error subplot if provided
    nrows = 1  # knee
    if has_thigh:
        nrows += 1
    if thigh_ori_err_deg is not None:
        nrows += 1

    fig, axes = plt.subplots(nrows, 1, figsize=(10.5, 3.0 + 2.3 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]  # type: ignore[assignment]

    ax_i = 0
    if has_thigh:
        axes[ax_i].plot(t, np.asarray(ref_thigh_deg, dtype=np.float64)[: t.size], color="0.2", lw=1.8, label="ref (CMU)")  # type: ignore[arg-type]
        axes[ax_i].plot(t, np.asarray(query_thigh_aligned_deg, dtype=np.float64)[: t.size], color="#d55e00", lw=1.6, label="query aligned")  # type: ignore[arg-type]
        axes[ax_i].set_ylabel("Thigh pitch (deg)")
        axes[ax_i].grid(True, alpha=0.25)
        axes[ax_i].legend(loc="upper right")
        ax_i += 1

    axes[ax_i].plot(t, ref_knee_deg[: t.size], color="0.2", lw=1.8, label="ref (CMU)")
    axes[ax_i].plot(t, query_knee_aligned_deg[: t.size], color="#d55e00", lw=1.6, label="query aligned")
    axes[ax_i].set_ylabel("Knee flexion (deg)")
    axes[ax_i].grid(True, alpha=0.25)
    axes[ax_i].legend(loc="upper right")
    ax_i += 1

    if thigh_ori_err_deg is not None:
        e = np.asarray(thigh_ori_err_deg, dtype=np.float64).reshape(-1)
        n = int(min(int(t.size), int(e.size)))
        axes[ax_i].plot(t[:n], e[:n], color="#0072b2", lw=1.6, label="thigh ori err (deg)")
        axes[ax_i].axhline(
            float(np.sqrt(float(np.mean(e[:n] ** 2)))) if n > 0 else 0.0, color="0.6", lw=1.0, ls="--"
        )
        axes[ax_i].set_ylabel("Thigh ori err (deg)")
        axes[ax_i].set_xlabel("Time (s)")
        axes[ax_i].grid(True, alpha=0.25)
        axes[ax_i].legend(loc="upper right")
    else:
        axes[ax_i - 1].set_xlabel("Time (s)")

    fig.suptitle(f"{title}\nRMS thigh ori err={rmse_thigh_deg:.2f} deg, RMSE knee={rmse_knee_deg:.2f} deg")
    return _save(fig, out_path)


def plot_simulation_knee(
    *,
    out_path: str | Path,
    sample_hz: float,
    ref_target_knee_deg: np.ndarray,
    good_target_knee_deg: np.ndarray,
    bad_target_knee_deg: np.ndarray,
    ref_actual_knee_deg: np.ndarray,
    good_actual_knee_deg: np.ndarray,
    bad_actual_knee_deg: np.ndarray,
    title: str,
) -> Path:
    n = int(
        min(
            len(ref_target_knee_deg),
            len(good_target_knee_deg),
            len(bad_target_knee_deg),
            len(ref_actual_knee_deg),
            len(good_actual_knee_deg),
            len(bad_actual_knee_deg),
        )
    )
    t = np.arange(n, dtype=np.float32) / float(sample_hz)
    fig, ax = plt.subplots(1, 1, figsize=(11.5, 4.3), sharex=True)

    ax.plot(t, ref_target_knee_deg[:n], color="0.6", lw=1.2, ls="--", label="REF target")
    ax.plot(t, ref_actual_knee_deg[:n], color="0.2", lw=1.8, label="REF actual")
    ax.plot(t, good_target_knee_deg[:n], color="#f0ad4e", lw=1.2, ls="--", label="PRED target")
    ax.plot(t, good_actual_knee_deg[:n], color="#d55e00", lw=1.8, label="PRED actual")
    ax.plot(t, bad_target_knee_deg[:n], color="#9ecae1", lw=1.2, ls="--", label="BAD target")
    ax.plot(t, bad_actual_knee_deg[:n], color="#0072b2", lw=1.8, label="BAD actual")

    ax.set_ylabel("Right knee flexion (deg)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=3)

    fig.suptitle(title)
    return _save(fig, out_path)


def plot_thigh_quat_match(
    *,
    out_path: str | Path,
    sample_hz: float,
    ref_thigh_quat_wxyz: np.ndarray,
    query_thigh_quat_aligned_wxyz: np.ndarray,
    thigh_ori_err_deg: np.ndarray,
    title: str,
) -> Path:
    """Debug plot: reference vs aligned query thigh quaternion components + geodesic error."""
    ref = np.asarray(ref_thigh_quat_wxyz, dtype=np.float64).reshape(-1, 4)
    q = np.asarray(query_thigh_quat_aligned_wxyz, dtype=np.float64).reshape(-1, 4)
    e = np.asarray(thigh_ori_err_deg, dtype=np.float64).reshape(-1)
    n = int(min(ref.shape[0], q.shape[0], e.size))
    t = np.arange(n, dtype=np.float32) / float(sample_hz)

    fig, axes = plt.subplots(5, 1, figsize=(11.0, 8.8), sharex=True)
    labs = ("w", "x", "y", "z")
    for i, lab in enumerate(labs):
        axes[i].plot(t, ref[:n, i], color="0.2", lw=1.6, label="ref")
        axes[i].plot(t, q[:n, i], color="#d55e00", lw=1.3, label="query aligned")
        axes[i].set_ylabel(f"q[{lab}]")
        axes[i].grid(True, alpha=0.25)
        if i == 0:
            axes[i].legend(loc="upper right")

    axes[4].plot(t, e[:n], color="#0072b2", lw=1.6, label="geodesic err (deg)")
    axes[4].axhline(float(np.sqrt(float(np.mean(e[:n] ** 2)))) if n > 0 else 0.0, color="0.6", lw=1.0, ls="--")
    axes[4].set_ylabel("Err (deg)")
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper right")

    fig.suptitle(title)
    return _save(fig, out_path)


def plot_simulation_angles(
    *,
    out_path: str | Path,
    sample_hz: float,
    ref_target_thigh_deg: np.ndarray,
    ref_target_knee_deg: np.ndarray,
    good_target_thigh_deg: np.ndarray,
    good_target_knee_deg: np.ndarray,
    bad_target_thigh_deg: np.ndarray,
    bad_target_knee_deg: np.ndarray,
    ref_actual_thigh_deg: np.ndarray,
    ref_actual_knee_deg: np.ndarray,
    good_actual_thigh_deg: np.ndarray,
    good_actual_knee_deg: np.ndarray,
    bad_actual_thigh_deg: np.ndarray,
    bad_actual_knee_deg: np.ndarray,
    title: str,
) -> Path:
    n = int(
        min(
            len(ref_actual_thigh_deg),
            len(ref_actual_knee_deg),
            len(good_actual_thigh_deg),
            len(good_actual_knee_deg),
            len(bad_actual_thigh_deg),
            len(bad_actual_knee_deg),
        )
    )
    t = np.arange(n, dtype=np.float32) / float(sample_hz)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True)

    # Thigh
    axes[0].plot(t, ref_target_thigh_deg[:n], color="0.6", lw=1.4, ls="--", label="REF target")
    axes[0].plot(t, ref_actual_thigh_deg[:n], color="0.2", lw=1.8, label="REF actual")
    axes[0].plot(t, good_target_thigh_deg[:n], color="#f0ad4e", lw=1.2, ls="--", label="PRED target")
    axes[0].plot(t, good_actual_thigh_deg[:n], color="#d55e00", lw=1.8, label="PRED actual")
    axes[0].plot(t, bad_target_thigh_deg[:n], color="#9ecae1", lw=1.2, ls="--", label="BAD target")
    axes[0].plot(t, bad_actual_thigh_deg[:n], color="#0072b2", lw=1.8, label="BAD actual")
    axes[0].set_ylabel("Hip pitch / thigh (deg)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3)

    # Knee
    axes[1].plot(t, ref_target_knee_deg[:n], color="0.6", lw=1.4, ls="--", label="REF target")
    axes[1].plot(t, ref_actual_knee_deg[:n], color="0.2", lw=1.8, label="REF actual")
    axes[1].plot(t, good_target_knee_deg[:n], color="#f0ad4e", lw=1.2, ls="--", label="PRED target")
    axes[1].plot(t, good_actual_knee_deg[:n], color="#d55e00", lw=1.8, label="PRED actual")
    axes[1].plot(t, bad_target_knee_deg[:n], color="#9ecae1", lw=1.2, ls="--", label="BAD target")
    axes[1].plot(t, bad_actual_knee_deg[:n], color="#0072b2", lw=1.8, label="BAD actual")
    axes[1].set_ylabel("Knee flexion (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", ncol=3)

    fig.suptitle(title)
    return _save(fig, out_path)


def plot_balance_traces(
    *,
    out_path: str | Path,
    sample_hz: float,
    xcom_margin_ref_m: np.ndarray,
    xcom_margin_good_m: np.ndarray,
    xcom_margin_bad_m: np.ndarray,
    upright_ref: np.ndarray,
    upright_good: np.ndarray,
    upright_bad: np.ndarray,
    risk_trace_ref: np.ndarray,
    risk_trace_good: np.ndarray,
    risk_trace_bad: np.ndarray,
    balance_loss_step_ref: int,
    balance_loss_step_good: int,
    balance_loss_step_bad: int,
    risk_ref: float,
    risk_good: float,
    risk_bad: float,
    title: str,
) -> Path:
    n = int(
        min(
            xcom_margin_ref_m.size,
            xcom_margin_good_m.size,
            xcom_margin_bad_m.size,
            upright_ref.size,
            upright_good.size,
            upright_bad.size,
            risk_trace_ref.size,
            risk_trace_good.size,
            risk_trace_bad.size,
        )
    )
    t = np.arange(n, dtype=np.float32) / float(sample_hz)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True)

    axes[0].plot(t, xcom_margin_ref_m[:n], color="0.2", lw=1.8, label=f"REF (risk={risk_ref:.2f})")
    axes[0].plot(t, xcom_margin_good_m[:n], color="#d55e00", lw=1.8, label=f"PRED (risk={risk_good:.2f})")
    axes[0].plot(t, xcom_margin_bad_m[:n], color="#0072b2", lw=1.8, label=f"BAD (risk={risk_bad:.2f})")
    axes[0].axhline(0.0, color="0.6", lw=1.0, ls="--")
    axes[0].set_ylabel("XCoM margin (m)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, upright_ref[:n], color="0.2", lw=1.8, label=f"REF (bal_loss_step={balance_loss_step_ref})")
    axes[1].plot(t, upright_good[:n], color="#d55e00", lw=1.8, label=f"PRED (bal_loss_step={balance_loss_step_good})")
    axes[1].plot(t, upright_bad[:n], color="#0072b2", lw=1.8, label=f"BAD (bal_loss_step={balance_loss_step_bad})")
    axes[1].axhline(0.40, color="0.6", lw=1.0, ls="--")
    axes[1].set_ylabel("Uprightness (cos tilt)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, risk_trace_ref[:n], color="0.2", lw=1.8, label="REF risk_now")
    axes[2].plot(t, risk_trace_good[:n], color="#d55e00", lw=1.8, label="PRED risk_now")
    axes[2].plot(t, risk_trace_bad[:n], color="#0072b2", lw=1.8, label="BAD risk_now")
    axes[2].axhline(0.70, color="0.6", lw=1.0, ls="--")
    axes[2].axhline(0.90, color="0.6", lw=1.0, ls=":")
    axes[2].set_ylabel("Predicted risk")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")

    fig.suptitle(title)
    return _save(fig, out_path)
