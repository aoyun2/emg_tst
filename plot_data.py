"""
plot_data.py — Publication-quality dataset visualization.

Outputs (figures/):
  overview.pdf/png      — 60 s segment: knee angle, EMG heatmaps, thigh quaternion
  episode.pdf/png       — single flex/extend cycle: angle + raw EMG envelopes (compelling)
  distribution.pdf/png  — ridgeline KDE per recording
  correlation.pdf/png   — Pearson r heatmap: training features vs knee angle
  summary.pdf/png       — dataset-level summary bars

Usage:
  python plot_data.py
  python plot_data.py --overview data4
"""

import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.titleweight":   "bold",
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.framealpha":  0.85,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.1,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

C_KNEE   = "#0072B2"
C_FLAT   = "#E69F00"
C_SPIKE  = "#D55E00"
C_FILL   = "#AED6F1"
C_QUAT   = ["#009E73", "#56B4E9", "#CC79A7", "#F0E442"]
EMG_CMAP = "inferno"
FILE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

# Per-sensor training feature names (must match _extract_raw_features_for_sensor)
_FEAT_NAMES = ["RMS", "MAV", "WL", "ZC", "SSC",
               "FFT0", "FFT1", "FFT2", "FFT3", "FFT4", "FFT5", "FFT6", "FFT7"]
_N_FEAT = len(_FEAT_NAMES)

FIGURES_DIR = Path("figures")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, stem):
    FIGURES_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{stem}.{ext}")
        print(f"  saved -> figures/{stem}.{ext}")


def _load(path):
    return np.load(path, allow_pickle=True).item()


def _norm_rows(mat):
    """Normalize each row of (C, T) to [0, 1]."""
    lo  = mat.min(axis=1, keepdims=True)
    hi  = mat.max(axis=1, keepdims=True)
    return (mat - lo) / np.where(hi > lo, hi - lo, 1.0)


def _thin(arr, n=4000):
    if len(arr) <= n:
        return np.arange(len(arr)), arr
    idx = np.linspace(0, len(arr) - 1, n, dtype=int)
    return idx, arr[idx]


def _busiest_segment(ts, angles, seg_sec):
    """Return (start_idx, end_idx) of the 60 s window with highest angle std."""
    hz  = (len(ts) - 1) / (ts[-1] - ts[0])
    win = int(hz * seg_sec)
    best_std, best_i = -1, 0
    step = max(1, win // 20)
    for i in range(0, len(angles) - win, step):
        s = angles[i:i + win].std()
        if s > best_std:
            best_std, best_i = s, i
    return best_i, best_i + win


# ---------------------------------------------------------------------------
# Figure 1 — Overview
# ---------------------------------------------------------------------------

def fig_overview(d, label, seg_sec=60.0):
    ts     = np.asarray(d["timestamps"],       dtype=np.float64); ts -= ts[0]
    angles = np.asarray(d["knee_included_deg"], dtype=np.float32)
    flags  = np.asarray(d["quality_flags"],    dtype=np.int32)
    thq    = np.asarray(d["thigh_quat_wxyz"],  dtype=np.float32)
    hz     = float(d.get("effective_hz", 200.0))
    n_ch   = int(d["n_channels"])

    s0, s1 = _busiest_segment(ts, angles, seg_sec)
    m      = slice(s0, s1)
    t_seg  = ts[m] - ts[s0]
    a_seg  = angles[m]
    f_seg  = flags[m]
    thq_s  = thq[m]

    fig = plt.figure(figsize=(11, 9.5))
    gs  = gridspec.GridSpec(5, 1, figure=fig,
                            height_ratios=[2.8, 1.1, 1.1, 1.1, 1.8],
                            hspace=0.10, top=0.92, bottom=0.08,
                            left=0.10, right=0.90)
    ax_ang  = fig.add_subplot(gs[0])
    ax_emg  = [fig.add_subplot(gs[i + 1], sharex=ax_ang) for i in range(3)]
    ax_quat = fig.add_subplot(gs[4], sharex=ax_ang)

    # --- knee angle ---
    # Very subtle shading only — don't let it dominate
    in_flat, t0 = False, None
    for i, (tv, fv) in enumerate(zip(t_seg, f_seg)):
        if fv == 2 and not in_flat:
            in_flat, t0 = True, tv
        elif fv != 2 and in_flat:
            in_flat = False
            ax_ang.axvspan(t0, tv, color=C_FLAT, alpha=0.08, linewidth=0)
    if in_flat:
        ax_ang.axvspan(t0, t_seg[-1], color=C_FLAT, alpha=0.08, linewidth=0)

    ax_ang.fill_between(t_seg, 0, a_seg, alpha=0.10, color=C_FILL, zorder=1)
    idx, tp = _thin(t_seg)
    ax_ang.plot(tp, a_seg[idx], color=C_KNEE, lw=1.0, zorder=3, label="Knee angle")

    spk = np.where(f_seg == 1)[0]
    if len(spk):
        ax_ang.scatter(t_seg[spk], a_seg[spk], color=C_SPIKE, s=22, zorder=5,
                       label="Spike replaced", marker="x", linewidths=1.2)

    ax_ang.set_ylabel("Knee angle (deg)")
    ax_ang.set_ylim(-5, 195)
    ax_ang.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax_ang.axhline(180, color="gray", lw=0.5, ls="--", zorder=1)
    ax_ang.legend(loc="lower right", ncol=3, fontsize=7.5)
    ax_ang.set_title(f"{label}  |  {ts[s0]:.0f}-{ts[s1]:.0f} s  |  {hz:.0f} Hz", pad=5)
    plt.setp(ax_ang.get_xticklabels(), visible=False)

    # --- EMG heatmaps (show only active channels 0-3) ---
    active_ch = min(4, n_ch)   # first 4 channels carry real signal
    step  = max(1, (s1 - s0) // 2000)
    t_im  = t_seg[::step]
    s_lbl = ["Sensor 1 (vastus medialis)", "Sensor 2 (semimembranosus)", "Sensor 3 (biceps femoris)"]
    for s, ax_e in enumerate(ax_emg):
        raw = np.asarray(d[f"emg_sensor{s+1}"], dtype=np.float32)[:active_ch, m][:, ::step]
        img = _norm_rows(raw)
        ax_e.imshow(img, aspect="auto", origin="lower", cmap=EMG_CMAP,
                    vmin=0, vmax=1,
                    extent=[t_im[0], t_im[-1], -0.5, active_ch - 0.5],
                    interpolation="nearest")
        ax_e.set_ylabel("Ch", labelpad=2)
        ax_e.set_yticks(range(active_ch))
        ax_e.set_yticklabels(range(active_ch), fontsize=6)
        ax_e.set_title(f"EMG  {s_lbl[s]}  (ch 0-{active_ch-1})", pad=2)
        plt.setp(ax_e.get_xticklabels(), visible=False)

    sm = mpl.cm.ScalarMappable(cmap=EMG_CMAP, norm=mpl.colors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_emg, fraction=0.012, pad=0.01, aspect=30)
    cbar.set_label("norm.", fontsize=7, labelpad=2)
    cbar.set_ticks([0, 1])

    # --- thigh quaternion ---
    for c, (col, lbl) in enumerate(zip(C_QUAT, ["w", "x", "y", "z"])):
        idx2, qp = _thin(thq_s[:, c])
        ax_quat.plot(t_seg[idx2], qp, color=col, lw=0.9, label=lbl)
    ax_quat.set_ylabel("Quaternion")
    ax_quat.set_ylim(-1.1, 1.1)
    ax_quat.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_quat.axhline(0, color="gray", lw=0.4, ls=":", zorder=1)
    ax_quat.set_xlabel("Time within segment (s)")
    ax_quat.legend(loc="lower right", ncol=4, title="Thigh quat.", fontsize=7.5)
    ax_quat.set_title("Thigh quaternion (wxyz)", pad=2)

    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Movement episode with raw EMG envelopes
# ---------------------------------------------------------------------------

def fig_episode(d, label, start_idx, n_samples):
    """
    Movement episode: knee angle + angular velocity + raw EMG envelopes.

    EMG reflects muscle *force/activation*, not joint position directly.
    Correlation is most visible against angular velocity (dAngle/dt):
      - Vastus medialis (extensor): bursts when angle increases (extending)
      - Semimembranosus / Biceps femoris (flexors): bursts when angle decreases
    """
    ts_orig = np.asarray(d["timestamps"],        dtype=np.float64)  # unshifted
    angles  = np.asarray(d["knee_included_deg"],  dtype=np.float32)
    hz      = float(d.get("effective_hz", 200.0))

    end_idx = min(start_idx + n_samples, len(angles))
    t_imu   = ts_orig[start_idx:end_idx] - ts_orig[start_idx]
    a       = angles[start_idx:end_idx]
    a_sm    = uniform_filter1d(a.astype(np.float64), size=7).astype(np.float32)

    # Angular velocity: central difference, smoothed, in deg/s
    dt = float(t_imu[-1] - t_imu[0]) / max(len(t_imu) - 1, 1)
    vel = np.gradient(a_sm.astype(np.float64), dt).astype(np.float32)
    vel = uniform_filter1d(vel, size=15).astype(np.float32)  # extra smoothing

    t0_abs = ts_orig[start_idx]
    t1_abs = ts_orig[end_idx - 1]
    env_smooth_ms = 60

    def _envelope(sensor_idx):
        key_raw = f"raw_emg_sensor{sensor_idx+1}"
        key_ts  = f"raw_emg_times{sensor_idx+1}"
        if key_raw not in d or key_ts not in d:
            return None, None
        raw = np.asarray(d[key_raw], dtype=np.float64)
        rts = np.asarray(d[key_ts],  dtype=np.float64)
        m   = (rts >= t0_abs) & (rts <= t1_abs)
        if m.sum() < 10:
            return None, None
        t_r = rts[m] - t0_abs
        raw_seg = raw[m]
        raw_hz  = len(raw_seg) / (t_r[-1] - t_r[0] + 1e-9)
        win     = max(3, int(raw_hz * env_smooth_ms / 1000))
        env     = uniform_filter1d(np.abs(raw_seg), size=win)
        lo, hi = env.min(), env.max()
        if hi > lo:
            env = (env - lo) / (hi - lo)
        return t_r, env

    s_lbl = ["Sensor 1 (vastus medialis)", "Sensor 2 (semimembranosus)", "Sensor 3 (biceps femoris)"]
    s_col = [FILE_COLORS[0], FILE_COLORS[2], FILE_COLORS[3]]

    # 5 rows: angle, velocity, sensor1, sensor2, sensor3
    fig = plt.figure(figsize=(11, 9.0))
    gs  = gridspec.GridSpec(5, 1, figure=fig,
                            height_ratios=[2.0, 1.2, 1.0, 1.0, 1.0],
                            hspace=0.38, top=0.91, bottom=0.08,
                            left=0.14, right=0.93)

    ax_ang = fig.add_subplot(gs[0])
    ax_vel = fig.add_subplot(gs[1], sharex=ax_ang)
    ax_env = [fig.add_subplot(gs[i + 2], sharex=ax_ang) for i in range(3)]

    # --- knee angle ---
    ax_ang.fill_between(t_imu, 0, a,    alpha=0.12, color=C_FILL, zorder=1)
    ax_ang.fill_between(t_imu, 0, a_sm, alpha=0.25, color=C_KNEE, zorder=2)
    ax_ang.plot(t_imu, a_sm, color=C_KNEE, lw=1.5, zorder=3, label="Knee angle (smoothed)")
    ax_ang.plot(t_imu, a,    color=C_KNEE, lw=0.5, alpha=0.35, zorder=2)
    ax_ang.set_ylabel("Angle (deg)")
    ax_ang.set_ylim(-5, 195)
    ax_ang.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax_ang.axhline(180, color="gray", lw=0.5, ls="--")
    min_i = int(np.argmin(a_sm))
    ax_ang.annotate(
        f"  {a_sm[min_i]:.0f} deg",
        xy=(t_imu[min_i], a_sm[min_i]),
        xytext=(t_imu[min_i], a_sm[min_i] + 28),
        arrowprops=dict(arrowstyle="->", color=C_KNEE, lw=0.9),
        fontsize=8, color=C_KNEE, ha="center",
    )
    ax_ang.set_title(
        f"Movement Episode  |  {label}  |  t = {ts_orig[start_idx]:.1f} s", pad=5)
    ax_ang.legend(loc="upper right", fontsize=8)
    plt.setp(ax_ang.get_xticklabels(), visible=False)

    # --- angular velocity (shows when muscles are active) ---
    v_pos = np.clip(vel, 0, None)   # extension (angle increasing)
    v_neg = np.clip(vel, None, 0)   # flexion (angle decreasing)
    ax_vel.fill_between(t_imu, 0, v_pos, alpha=0.6, color="#2ECC71", label="Extension (+)")
    ax_vel.fill_between(t_imu, 0, v_neg, alpha=0.6, color="#E74C3C", label="Flexion (−)")
    ax_vel.plot(t_imu, vel, color="black", lw=0.6, alpha=0.5)
    ax_vel.axhline(0, color="gray", lw=0.5)
    ax_vel.set_ylabel("dAngle/dt\n(deg/s)")
    ax_vel.legend(loc="upper right", fontsize=7, ncol=2)
    ax_vel.set_title("Angular velocity", pad=3)
    plt.setp(ax_vel.get_xticklabels(), visible=False)

    # --- raw EMG envelopes ---
    for s, (ax_e, col) in enumerate(zip(ax_env, s_col)):
        t_r, env = _envelope(s)
        if t_r is not None:
            idx, t_p = _thin(t_r, n=5000)
            env_p = env[idx]
            ax_e.fill_between(t_p, 0, env_p, alpha=0.55, color=col, zorder=2)
            ax_e.plot(t_p, env_p, color=col, lw=0.6, zorder=3)
            ax_e.set_ylabel(f"EMG\n(norm.)", labelpad=2)
            ax_e.set_ylim(-0.05, 1.15)
            ax_e.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        else:
            ax_e.text(0.5, 0.5, "raw EMG not available",
                      transform=ax_e.transAxes, ha="center", va="center",
                      color="gray", fontsize=8)
        ax_e.set_title(s_lbl[s], pad=3, fontsize=9)
        if s < 2:
            plt.setp(ax_e.get_xticklabels(), visible=False)

    ax_env[-1].set_xlabel("Time (s)")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Ridgeline distribution
# ---------------------------------------------------------------------------

def fig_distribution(files):
    n       = len(files)
    row_h   = 1.0
    overlap = 0.50
    x_grid  = np.linspace(0, 180, 400)

    fig_h = max(4.5, n * row_h * (1 - overlap) + row_h + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.93, bottom=0.10)

    ax.set_xlim(-2, 185)
    ax.set_xlabel("Knee angle (deg)", fontsize=10)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Knee Angle Distributions by Recording", fontweight="bold", pad=8)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.axvline(0,   color="lightgray", lw=0.5, zorder=0)
    ax.axvline(180, color="lightgray", lw=0.5, zorder=0)

    y_bases = []
    for i, fpath in enumerate(files):
        d   = _load(fpath)
        a   = np.asarray(d["knee_included_deg"], dtype=np.float32)
        qf  = np.asarray(d["quality_flags"],     dtype=np.int32)
        a_c = a[qf == 0]
        col = FILE_COLORS[i % len(FILE_COLORS)]
        y_base = i * row_h * (1 - overlap)
        y_bases.append(y_base)

        if len(a_c) > 10:
            kde  = gaussian_kde(a_c, bw_method=0.06)
            dens = kde(x_grid)
            scale = row_h / dens.max()
            dens *= scale

            ax.fill_between(x_grid, y_base, y_base + dens,
                            color=col, alpha=0.55, zorder=i + 1)
            ax.plot(x_grid, y_base + dens,
                    color=col, lw=1.5, zorder=i + 2)

            # mean tick: a short vertical line segment at mean, height ~ 0.4*row_h
            mu = float(a_c.mean())
            mu_dens = float(kde(mu)[0]) * scale   # KDE height at mean
            ax.plot([mu, mu], [y_base, y_base + mu_dens * 0.9],
                    color=col, lw=1.5, ls="--", alpha=0.85, zorder=i + 3)
            ax.text(mu, y_base - 0.06, f"{mu:.0f}",
                    ha="center", va="top", fontsize=7, color=col)

        ax.axhline(y_base, color="lightgray", lw=0.5, zorder=i)
        name = Path(fpath).stem
        ax.text(-3, y_base + row_h * 0.38, name,
                ha="right", va="center", fontsize=9,
                color=col, fontweight="bold")

    total_h = (n - 1) * row_h * (1 - overlap) + row_h
    ax.set_ylim(-0.20, total_h + 0.15)
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Correlation heatmap (uses samples_dataset.npy)
# ---------------------------------------------------------------------------

def fig_correlation(samples_path="samples_dataset.npy"):
    ds      = np.load(Path(samples_path), allow_pickle=True).item()
    X       = ds["X"].astype(np.float32)
    y       = ds["y"].astype(np.float32)
    fids    = ds["file_id"]
    fnames_raw  = ds.get("file_names", np.array([]))
    unique_fids = np.unique(fids)
    n_files     = len(unique_fids)
    fnames = [Path(str(fn)).stem for fn in fnames_raw] if len(fnames_raw) \
             else [f"file_{fid}" for fid in unique_fids]

    X_win = X.mean(axis=1)  # (N, F) — time-averaged feature per window

    corr = np.full((3, n_files, _N_FEAT), np.nan)
    for fi, fid in enumerate(unique_fids):
        mask = fids == fid
        Xf, yf = X_win[mask], y[mask].astype(np.float64)
        if yf.std() < 1e-6:
            continue
        for s in range(3):
            c0 = s * _N_FEAT
            for c in range(_N_FEAT):
                f = Xf[:, c0 + c].astype(np.float64)
                fs = f.std()
                if fs < 1e-12:
                    corr[s, fi, c] = 0.0
                else:
                    corr[s, fi, c] = float(
                        np.mean((f - f.mean()) * (yf - yf.mean())) / (fs * yf.std())
                    )

    vmax = max(0.35, float(np.nanmax(np.abs(corr))))
    s_titles = ["Sensor 1 (vastus medialis)", "Sensor 2 (semimembranosus)", "Sensor 3 (biceps femoris)"]

    row_h = 0.55
    fig_h = max(3.5, n_files * row_h + 2.8)
    fig, axes = plt.subplots(1, 3, figsize=(13, fig_h))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.25,
                        wspace=0.35)
    fig.suptitle(
        "Pearson r: EMG Training Features vs Knee Angle (per recording, window mean)",
        fontweight="bold", fontsize=10, y=0.96,
    )
    fig.text(0.5, 0.88,
             "Near-zero r is expected: EMG-to-angle is nonlinear; "
             "the model exploits sequence context that linear correlation cannot capture.",
             ha="center", fontsize=8, color="gray")

    for s, ax in enumerate(axes):
        mat = corr[s]
        im  = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, origin="upper")

        # x-axis: feature names rotated, grouped
        ax.set_xticks(np.arange(_N_FEAT))
        ax.set_xticklabels(_FEAT_NAMES, fontsize=7.5, rotation=45, ha="right")

        # vertical divider between time-domain and spectral
        ax.axvline(4.5, color="white", lw=2, zorder=5)

        ax.set_yticks(np.arange(n_files))
        ax.set_yticklabels(fnames if s == 0 else [""] * n_files, fontsize=8)
        ax.set_title(s_titles[s], pad=6)

        # group labels below the x-axis (outside the plot)
        ax.annotate("Time-domain", xy=(2, -0.5), xycoords=("data", "axes fraction"),
                    xytext=(2, -0.38), textcoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color="gray")
        ax.annotate("Spectral", xy=(9, -0.5), xycoords=("data", "axes fraction"),
                    xytext=(9, -0.38), textcoords=("data", "axes fraction"),
                    ha="center", fontsize=7, color="gray")

        for fi in range(n_files):
            for c in range(_N_FEAT):
                r = corr[s, fi, c]
                if not np.isnan(r):
                    tc = "white" if abs(r) > vmax * 0.55 else "black"
                    ax.text(c, fi, f"{r:.2f}", ha="center", va="center",
                            fontsize=5.5, color=tc)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                            shrink=0.85)
        cbar.set_label("Pearson r", fontsize=8)
        cbar.set_ticks([-vmax, 0, vmax])
        cbar.set_ticklabels([f"{-vmax:.2f}", "0", f"{vmax:.2f}"])

    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Summary
# ---------------------------------------------------------------------------

def fig_summary(files):
    n = len(files)
    names = [Path(f).stem for f in files]
    dur   = np.zeros(n); ang_lo = np.zeros(n); ang_hi = np.zeros(n)
    means = np.zeros(n); n_samp = np.zeros(n, dtype=int)
    n_ok  = np.zeros(n, dtype=int)
    n_sp  = np.zeros(n, dtype=int)
    n_fl  = np.zeros(n, dtype=int)

    for i, fpath in enumerate(files):
        d  = _load(fpath)
        ts = np.asarray(d["timestamps"], dtype=np.float64)
        a  = np.asarray(d["knee_included_deg"], dtype=np.float32)
        qf = np.asarray(d["quality_flags"],     dtype=np.int32)
        dur[i]    = ts[-1] - ts[0]
        ang_lo[i] = float(a.min())
        ang_hi[i] = float(a.max())
        means[i]  = float(a[qf == 0].mean()) if (qf == 0).any() else float(a.mean())
        n_samp[i] = len(a)
        n_ok[i]   = int((qf == 0).sum())
        n_sp[i]   = int((qf == 1).sum())
        n_fl[i]   = int((qf == 2).sum())

    x, w = np.arange(n), 0.55
    cols  = FILE_COLORS[:n]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18,
                        wspace=0.38)
    fig.suptitle("Dataset Summary", fontweight="bold", fontsize=12, y=0.97)

    def _bars(ax, values, ylabel, title, pct=False):
        for i, col in enumerate(cols):
            ax.bar(x[i], values[i], width=w, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.18)
        if pct:
            ax.set_ylim(0, 100)

    # (a) Duration
    _bars(axes[0], dur / 60, "Duration (min)", "(a) Duration")

    # (b) Angle range
    ax = axes[1]
    for i, col in enumerate(cols):
        ax.bar(x[i] - w/4, ang_lo[i], width=w/2, color="#AED6F1", edgecolor="none")
        ax.bar(x[i] + w/4, ang_hi[i], width=w/2, color=col,       edgecolor="none")
    ax.scatter(x, means, color="white", edgecolors="black", s=32, zorder=5, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Angle (deg)"); ax.set_ylim(0, 205)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.set_title("(b) Angle range")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(color="#AED6F1", label="Min"), Patch(color="gray", label="Max"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="w",
               markeredgecolor="k", markersize=6, label="Mean"),
    ], fontsize=7, loc="upper left", ncol=3)

    # (c) Sample count
    _bars(axes[2], n_samp / 1000, "Samples (x1000)", "(c) Sample count")

    # (d) Quality flags stacked
    ax = axes[3]
    p_ok = n_ok / n_samp * 100
    p_fl = n_fl / n_samp * 100
    p_sp = n_sp / n_samp * 100
    ax.bar(x, p_ok, width=w, color="#2ECC71", label="Normal")
    ax.bar(x, p_fl, width=w, bottom=p_ok,         color=C_FLAT,  label="Flat")
    ax.bar(x, p_sp, width=w, bottom=p_ok + p_fl,  color=C_SPIKE, label="Spike")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Fraction (%)"); ax.set_ylim(0, 108)
    ax.set_title("(d) Quality flags")
    ax.legend(loc="lower right", fontsize=7)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _best_episode_per_file(files, ep_sec=12.0):
    """Return list of (fpath, start_idx, win) — best episode for each file."""
    results = []
    for fpath in files:
        d  = _load(fpath)
        a  = np.asarray(d["knee_included_deg"], dtype=np.float32)
        hz = float(d.get("effective_hz", 200.0))
        win = int(hz * ep_sec)
        best_r, best_i = -1, 0
        for i in range(0, max(1, len(a) - win), max(1, int(hz * 3))):
            r = float(a[i:i+win].max()) - float(a[i:i+win].min())
            if r > best_r:
                best_r, best_i = r, i
        results.append((fpath, best_i, win))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob",     default="data*.npy")
    parser.add_argument("--overview", default=None)
    parser.add_argument("--seg",      type=float, default=60.0)
    args = parser.parse_args()

    files = sorted(f for f in glob.glob(args.glob)
                   if "samples" not in Path(f).name)
    if not files:
        raise SystemExit(f"No recordings found matching '{args.glob}'")
    print(f"Found {len(files)} recording(s)")

    if args.overview:
        stem  = args.overview if args.overview.endswith(".npy") else args.overview + ".npy"
        cands = [f for f in files if Path(f).name == stem]
        ov    = cands[0] if cands else files[0]
    else:
        ov = max(files, key=lambda f: float(
            np.asarray(_load(f)["knee_included_deg"], dtype=np.float32).std()
        ))

    print(f"Overview: {Path(ov).name}")
    print("-" * 40)

    print("Rendering overview...")
    f1 = fig_overview(_load(ov), Path(ov).stem, seg_sec=args.seg)
    _save(f1, "overview"); plt.close(f1)

    print("Rendering episodes (one per recording)...")
    for ep_file, ep_start, ep_win in _best_episode_per_file(files):
        stem = Path(ep_file).stem
        print(f"  {stem}...")
        f2 = fig_episode(_load(ep_file), stem, ep_start, ep_win)
        _save(f2, f"episode_{stem}"); plt.close(f2)

    print("Rendering distribution...")
    f3 = fig_distribution(files)
    _save(f3, "distribution"); plt.close(f3)

    print("Rendering correlation...")
    sp = Path("samples_dataset.npy")
    if sp.exists():
        f4 = fig_correlation(sp)
        _save(f4, "correlation"); plt.close(f4)
    else:
        print("  (skipped - run split_to_samples.py first)")

    print("Rendering summary...")
    f5 = fig_summary(files)
    _save(f5, "summary"); plt.close(f5)

    print("\nDone. All figures written to figures/")


if __name__ == "__main__":
    main()
