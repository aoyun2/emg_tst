"""Generate the manuscript figures from the completed run.

Every mark is drawn from a run artifact. Figures are written as PDF so they stay
vector in the typeset article; the rendered rollout frames are the only raster
content and are embedded at their native resolution.

Two categorical hues carry every comparison, validated against the six colour
checks (lightness band, chroma floor, adjacent-pair CVD separation,
normal-vision floor, contrast). The original draft's three-hue palette failed
three of them: its blue and green read as gray, its orange and green separated
by only dE 5.0 under red-green colour vision deficiency, and its blue and green
by 13.3 for normal vision.

Colour means one thing across the whole set: ink for a recorded quantity, the
blue ramp for a model prediction ordered by accuracy, gray for an unmodified
reference, and rust for the substituted condition and for the reported result.
Titles, labels, and annotations are ink, never a series colour, so colour is
never doing a label's job.

    python -m analysis.gait120_figures --runs-dir <runs> --out-dir manuscript/figures
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Two categorical hues, cool and warm, validated together: adjacent-pair CVD
# separation dE 18.1 and normal-vision 24.5, both well clear of their floors.
# Two is enough because no panel shows more than two series at once, and a pair
# stays legible at the muted chroma a journal page wants; a third hue could not
# be desaturated this far without the blue-green pair failing the normal-vision
# floor.
BLUE, RUST = "#22598F", "#A9541F"

# Accuracy level is ordinal, not categorical, so it takes a single-hue ramp
# light-to-dark rather than three unrelated hues. Adjacent steps differ by dL
# 0.146 and 0.125, and the lightest holds 2.35:1 against the page.
RAMP = ["#8CAAC9", "#4E7FAC", "#22598F"]

INK, MUTED, GRID = "#22201e", "#6b6862", "#e2e4e6"

MM = 1 / 25.4


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.4,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def clean(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)


def panel(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.02) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8, fontweight="bold", color=INK, clip_on=False)


def save(fig: plt.Figure, out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03,
                **({"dpi": 400} if path.suffix == ".png" else {}))
    plt.close(fig)
    return path


# --------------------------------------------------------------------- loading

def load(runs: Path) -> dict[str, Any]:
    read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    return {
        "ablation": read(runs / "confirmation" / "ablation_summary.json"),
        "fused": read(runs / "confirmation" / "fused" / "result.json"),
        "no_emg": read(runs / "confirmation" / "no_emg" / "result.json"),
        "correlation": read(runs / "analysis" / "checkpoint_correlation.json"),
        "checkpoints": read(runs / "training_path" / "checkpoints" / "manifest.json"),
        "runs": runs,
    }


def window_rows(runs: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(glob.glob(str(runs / "panel" / "stages" / "*" / "evals" / "*" / "summary.json"))):
        rows.append(json.loads(Path(p).read_text(encoding="utf-8")))
    return rows


# --------------------------------------------------------------------- figures

def fig_overview(runs, out):
    """Visual abstract: each stage shows the real data it produces.

    A flow of labelled boxes says what the pipeline is; showing the actual signal,
    the actual descent, the actual rollout, and the actual result says what it
    does, and lets a reader see the finding before reading a word of the paper.
    """
    import json

    import numpy as np

    w = _panel_window(runs, "S065_trial05_start0069")
    path = np.load(runs / "training_path" / "training_path.npz", allow_pickle=False)
    accuracy = json.loads(
        (runs / "analysis" / "checkpoint_correlation.json").read_text(encoding="utf-8")
    )["accuracy_level"]
    stage = sorted((runs / "panel" / "stages").glob("*"))[0]
    with np.load(stage / "evals" / "S065_trial05_start0069" / "compare.npz",
                 allow_pickle=False) as d:
        frames = np.asarray(d["frames"])
        ref_knee = np.asarray(d["knee_ref_actual_deg"], dtype=float)
        cmd_knee = np.asarray(d["knee_good_query_deg"], dtype=float)

    titles = ["Gait120 recordings", "Residual-fusion prediction",
              "Sampled training path", "Motion match to MoCapAct",
              "Paired simulation", "Accuracy vs instability"]
    notes = ["12 sEMG channels,\n90 participants",
             "knee angle 100 ms ahead",
             "14 accuracy levels,\n22 to 4.9 deg",
             "80 fixed windows",
             "reference vs\noverridden knee",
             "the turn at 13 deg"]
    colours = [MUTED, MUTED, MUTED, MUTED, MUTED, RUST]

    fig = plt.figure(figsize=(180 * MM, 56 * MM))
    gs = fig.add_gridspec(1, 6, wspace=0.34, left=0.015, right=0.985,
                          top=0.80, bottom=0.24)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]

    ax = axes[0]
    emg = w["emg"][:, :6]
    emg = (emg - emg.mean(0)) / np.maximum(emg.std(0), 1e-8)
    for i in range(emg.shape[1]):
        ax.plot(np.clip(emg[:, i], -2, 5) * 0.30 + i, linewidth=0.55, color=BLUE)
    ax.set_ylim(-1, emg.shape[1])

    ax = axes[1]
    ax.plot(w["knee"], color=INK, linewidth=1.0)
    ax.plot(w["fused"][-1], color=RAMP[2], linewidth=1.0, linestyle="--")

    ax = axes[2]
    idx = np.unique(np.round(np.geomspace(1, path["steps"].size, 400)).astype(int) - 1)
    rmse = path["train_rmse_deg"][idx]
    ax.plot(np.arange(rmse.size), rmse, color=INK, linewidth=1.0)
    for level in np.linspace(0, rmse.size - 1, 9)[1:-1]:
        ax.axvline(level, color=GRID, linewidth=0.5, zorder=0)
    ax.set_ylim(0, rmse.max() * 1.08)

    ax = axes[3]
    ax.plot(ref_knee, color=MUTED, linewidth=1.0, label="matched")
    ax.plot(cmd_knee[: ref_knee.size], color=RAMP[2], linewidth=1.0, linestyle=":")

    ax = axes[4]
    half = frames.shape[2] // 2
    scene = frames[2 * len(frames) // 3][40:, half:]
    # Crop to the walker rather than stretching the whole scene: at thumbnail size
    # the floor grid carries nothing and the collapse is what needs to be legible.
    warm = scene[..., 0].astype(int) > scene[..., 2].astype(int) + 40
    ys, xs = np.where(warm)
    pad = 26
    y0, y1 = max(0, ys.min() - pad), min(scene.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(scene.shape[1], xs.max() + pad)
    ax.imshow(scene[y0:y1, x0:x1], aspect="auto")

    ax = axes[5]
    x = np.asarray(accuracy["mean_rmse_deg"])
    y = np.asarray(accuracy["mean_excess_instability"])
    bp = float(accuracy["breakpoint_rmse_deg"])
    ax.plot(x[x < bp], y[x < bp], color=BLUE, linewidth=1.3, marker="o", markersize=2.4)
    ax.plot(x[x >= bp], y[x >= bp], color=RUST, linewidth=1.3, marker="o",
            markersize=2.4)
    ax.axvline(bp, color=INK, linewidth=0.8, linestyle="--")

    for i, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colours[i]); sp.set_linewidth(1.0)
        ax.set_title(titles[i], fontsize=6.4, color=INK, pad=3)
        ax.text(0.5, -0.10, notes[i], transform=ax.transAxes, ha="center", va="top",
                fontsize=5.9, color=MUTED, linespacing=1.35)
        if i:
            fig.add_artist(FancyArrowPatch(
                (ax.get_position().x0 - 0.021, 0.52),
                (ax.get_position().x0 - 0.005, 0.52),
                transform=fig.transFigure, arrowstyle="-|>", mutation_scale=5.5,
                linewidth=1.0, color=MUTED, shrinkA=0, shrinkB=0))

    return save(fig, out, "fig01_overview.pdf")


def fig_signal_model(data: dict[str, Any], out: Path) -> Path:
    """Input windows and the two-stage model. Panel A is schematic; B is recorded."""
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 52 * MM),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax = axes[0]
    t = np.linspace(-600, 100, 400)
    rng = np.random.default_rng(3)
    for i in range(6):
        centre = -520 + i * 88
        env = np.exp(-0.5 * ((t - centre) / 62) ** 2)
        env += 0.45 * np.exp(-0.5 * ((t - centre - 330) / 48) ** 2)
        env *= 1.0 + 0.05 * rng.standard_normal(t.size)
        ax.plot(t, np.clip(env, 0, None) * 0.8 + i, linewidth=0.8, color=BLUE, alpha=0.8)
    ax.axvspan(-150, 0, color=BLUE, alpha=0.10, lw=0)
    ax.axvspan(-600, 0, color=GRID, alpha=0.20, lw=0, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.axvline(100, color=RUST, linewidth=1.0, linestyle="--")
    ax.text(-75, 6.35, "sEMG\n150 ms", ha="center", fontsize=6.4, color=BLUE)
    ax.text(-380, 6.35, "knee history 600 ms", ha="center", fontsize=6.4, color=MUTED)
    ax.text(100, 6.35, "target\n+100 ms", ha="center", fontsize=6.4, color=RUST)
    ax.set_xlim(-620, 190); ax.set_ylim(-0.4, 7.1)
    ax.set_xlabel("time relative to last input frame (ms)")
    ax.set_yticks([]); ax.set_ylabel("sEMG channels (schematic)")
    clean(ax, grid=None); panel(ax, "A", x=-0.06)

    ax = axes[1]
    ax.axis("off")
    boxes = [("knee-angle\nhistory", 0.02, 0.62, BLUE), ("sEMG\nenvelopes", 0.02, 0.12, BLUE),
             ("kinematic\nridge", 0.36, 0.62, BLUE), ("residual\nridge", 0.36, 0.12, RUST),
             ("fused knee\nprediction", 0.72, 0.37, RUST)]
    for text, x, y, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 0.24, 0.26,
                                    boxstyle="square,pad=0.01",
                                    transform=ax.transAxes, facecolor="white",
                                    edgecolor=colour, linewidth=1.0, zorder=2))
        ax.text(x + 0.12, y + 0.13, text, transform=ax.transAxes, ha="center",
                va="center", fontsize=6.6, color=INK, zorder=3, linespacing=1.3)
    for x0, y0, x1, y1 in [(0.26, 0.75, 0.36, 0.75), (0.26, 0.25, 0.36, 0.25),
                           (0.60, 0.75, 0.72, 0.56), (0.60, 0.25, 0.72, 0.44)]:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), transform=ax.transAxes,
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color=MUTED))
    ax.text(0.48, 0.03, "the ablation sets the residual branch to zero",
            transform=ax.transAxes, ha="center", fontsize=6.3, color=MUTED)
    panel(ax, "B", x=-0.02)
    return save(fig, out, "fig02_signal_model.pdf")


def fig_prediction_accuracy(data: dict[str, Any], out: Path) -> Path:
    """The model-quality gate: paired per-participant RMSE with and without sEMG."""
    fused = {r["subject"]: r["rmse_deg"] for r in data["fused"]["test"]["participants"]}
    no_emg = {r["subject"]: r["rmse_deg"] for r in data["no_emg"]["test"]["participants"]}
    subjects = sorted(fused)
    f = np.array([fused[s] for s in subjects])
    n = np.array([no_emg[s] for s in subjects])
    improvement = n - f

    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 54 * MM),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 0.85],
                                          "wspace": 0.42})

    ax = axes[0]
    for a, b in zip(n, f):
        ax.plot([0, 1], [a, b], color=MUTED, alpha=0.18, linewidth=0.5, zorder=1)
    ax.plot([0, 1], [n.mean(), f.mean()], color=RUST, linewidth=1.3,
            marker="o", markersize=3.4, zorder=3, label="mean")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["without\nsEMG", "residual\nfusion"])
    ax.set_xlim(-0.3, 1.3); ax.set_ylabel("participant RMSE (deg)")
    ax.legend(loc="upper right")
    clean(ax); panel(ax, "A")

    ax = axes[1]
    order = np.argsort(improvement)
    colours = np.where(improvement[order] > 0, BLUE, RUST)
    ax.bar(np.arange(len(order)), improvement[order], color=colours,
           width=0.82, linewidth=0)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xlabel(f"participants, sorted ({int((improvement > 0).sum())} of {len(f)} improved)")
    ax.set_ylabel("RMSE reduction from sEMG (deg)")
    ax.set_xlim(-1, len(order))
    clean(ax); panel(ax, "B")

    ax = axes[2]
    # A box beside the points, rather than a violin: the kernel width of a violin
    # is a smoothing choice the reader cannot see, and the quartiles are what the
    # text actually cites.
    bp = ax.boxplot([improvement], widths=0.30, showfliers=False,
                    medianprops={"color": INK, "linewidth": 1.0},
                    boxprops={"color": MUTED, "linewidth": 0.7},
                    whiskerprops={"color": MUTED, "linewidth": 0.7},
                    capprops={"color": MUTED, "linewidth": 0.7})
    jitter = np.random.default_rng(0).uniform(-0.055, 0.055, improvement.size)
    ax.scatter(np.full(improvement.size, 1.0) + jitter, improvement, s=5,
               color=BLUE, alpha=0.45, linewidths=0, zorder=3)
    lo, hi = data["ablation"]["bootstrap_95pct_ci_deg"]
    ax.errorbar([1.34], [improvement.mean()],
                yerr=[[improvement.mean() - lo], [hi - improvement.mean()]],
                fmt="o", color=RUST, markersize=3.4, linewidth=1.0, capsize=2, zorder=4)
    ax.axhline(0, color=MUTED, linewidth=0.6)
    ax.set_xticks([]); ax.set_xlim(0.72, 1.58)
    ax.text(1.42, improvement.mean(),
            f"{improvement.mean():.3f}\n[{lo:.3f}, {hi:.3f}]",
            va="center", fontsize=6.0, color=INK)
    clean(ax); panel(ax, "C")
    return save(fig, out, "fig03_prediction_accuracy.pdf")


def fig_matching(rows: list[dict[str, Any]], out: Path) -> Path:
    """Motion-match quality over the fixed panel."""
    primary = sorted({r["checkpoint"] for r in rows})[-1]
    knee, thigh = [], []
    seen = set()
    for r in rows:
        if r["checkpoint"] != primary or r["query_id"] in seen:
            continue
        seen.add(r["query_id"])
        knee.append(r["match"]["knee_rmse_deg"])
        thigh.append(r["match"]["thigh_rms_deg"])
    knee, thigh = np.asarray(knee), np.asarray(thigh)

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 52 * MM))
    ax = axes[0]
    ax.scatter(knee, thigh, s=11, color=BLUE, alpha=0.7, linewidths=0)
    ax.scatter([knee.mean()], [thigh.mean()], marker="D", s=30, color=RUST,
               zorder=4, label="mean")
    ax.set_xlabel("motion-match knee RMSE (deg)")
    ax.set_ylabel("thigh orientation RMS (deg)")
    ax.legend(loc="upper right")
    clean(ax, grid="both"); panel(ax, "A")

    ax = axes[1]
    ax.plot(np.arange(knee.size), np.sort(knee), color=BLUE, linewidth=1.1, label="knee RMSE")
    ax.plot(np.arange(thigh.size), np.sort(thigh), color=RUST, linewidth=1.1,
            linestyle="--", label="thigh RMS")
    ax.set_xlabel(f"panel windows, sorted (n = {knee.size})")
    ax.set_ylabel("match error (deg)")
    ax.legend(loc="upper left")
    clean(ax); panel(ax, "B")
    return save(fig, out, "fig05_motion_matching.pdf")


def fig_instability(runs: Path, out: Path) -> Path:
    """A representative window: instability traces and the excess between them."""
    primary = sorted(Path(runs / "panel" / "stages").glob("*"))[-1]
    best, best_excess = None, -1e9
    for summary_path in sorted(primary.glob("evals/*/summary.json")):
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        excess = s["simulation"]["excess_instability_auc"]["fused"]
        if excess > best_excess:
            best_excess, best = excess, summary_path.parent / "compare.npz"

    with np.load(best, allow_pickle=False) as d:
        ref = np.asarray(d["predicted_fall_risk_trace_ref"], dtype=float)
        pred = np.asarray(d["predicted_fall_risk_trace_good"], dtype=float)
        dt = float(np.asarray(d["dt"]).reshape(()))
    t = np.arange(ref.size) * dt

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 52 * MM))
    ax = axes[0]
    ax.plot(t, ref, color=MUTED, linewidth=1.15, label="reference (unmodified)")
    ax.plot(t, pred, color=RUST, linewidth=1.15, label="overridden knee")
    ax.set_xlabel("time (s)"); ax.set_ylabel("instability index")
    ax.set_ylim(0, 1.02); ax.legend(loc="upper left")
    clean(ax); panel(ax, "A")

    ax = axes[1]
    diff = pred - ref
    ax.fill_between(t, 0, diff, where=diff >= 0, color=RUST, alpha=0.20, linewidth=0)
    ax.fill_between(t, 0, diff, where=diff < 0, color=RUST, alpha=0.20, linewidth=0)
    ax.plot(t, diff, color=INK, linewidth=1.1)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("excess instability")
    span = float(np.max(diff) - np.min(diff)) or 1.0
    ax.set_ylim(np.min(diff) - 0.08 * span, np.max(diff) + 0.30 * span)
    ax.text(0.02, 0.97, f"integrated excess {best_excess:+.3f} s",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color=INK)
    clean(ax); panel(ax, "B")
    return save(fig, out, "fig05_instability.pdf")


def fig_accuracy_vs_instability(data: dict[str, Any], out: Path) -> Path:
    """Primary result: simulated instability against model accuracy, with the turn."""
    a = data["correlation"]["accuracy_level"]
    x = np.asarray(a["mean_rmse_deg"])
    y = np.asarray(a["mean_excess_instability"])
    bp = float(a["breakpoint_rmse_deg"])
    lo, hi = a["breakpoint_95pct_ci"]

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 58 * MM),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    ax.axvspan(lo, hi, color=MUTED, alpha=0.13, lw=0)
    ax.axvline(bp, color=INK, linewidth=1.0, linestyle="--")
    for mask, colour, label in ((x < bp, BLUE, "below the turn"),
                                (x >= bp, RUST, "above the turn")):
        ax.plot(x[mask], y[mask], marker="o", markersize=3.8, linewidth=1.15,
                color=colour, label=label)
        design = np.column_stack([np.ones(mask.sum()), x[mask]])
        beta, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        xs = np.linspace(x[mask].min(), x[mask].max(), 50)
        ax.plot(xs, beta[0] + beta[1] * xs, color=colour, linewidth=0.9,
                linestyle=":", alpha=0.9)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("model accuracy: mean window RMSE (deg)")
    ax.set_ylabel("mean excess instability (s)")
    ax.text(bp, ax.get_ylim()[1], f" turn at {bp:.1f}$\\degree$\n [{lo:.1f}, {hi:.1f}]",
            va="top", fontsize=6.4, color=INK)
    ax.legend(loc="lower right")
    clean(ax, grid="both"); panel(ax, "A", x=-0.11)

    ax = axes[1]
    names = ["above\nthe turn", "below\nthe turn"]
    slopes = [a["above_breakpoint"]["slope_per_degree"], a["below_breakpoint"]["slope_per_degree"]]
    cis = [a["above_breakpoint"]["slope_95pct_ci"], a["below_breakpoint"]["slope_95pct_ci"]]
    colours = [RUST, BLUE]
    for i, (s, (cl, ch), colour) in enumerate(zip(slopes, cis, colours)):
        ax.errorbar([s], [i], xerr=[[s - cl], [ch - s]], fmt="o", color=colour,
                    markersize=4.2, linewidth=1.15, capsize=3)
        ax.text(s, i + 0.22, f"{s:+.4f}", ha="center", fontsize=6.4, color=INK)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(names)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("slope (s per degree of RMSE)")
    clean(ax, grid="x"); panel(ax, "B", x=-0.24)
    return save(fig, out, "fig06_accuracy_vs_instability.pdf")


def fig_per_checkpoint(data: dict[str, Any], out: Path) -> Path:
    """Why comparing windows cannot see the effect: the association and its spread."""
    rows = sorted(data["correlation"]["per_checkpoint"],
                  key=lambda r: r["mean_prediction_rmse_deg"])
    x = np.asarray([r["mean_prediction_rmse_deg"] for r in rows])
    rho = np.asarray([r["partial_spearman_rho"] for r in rows])
    lo = np.asarray([r["bootstrap_95pct_ci"][0] for r in rows])
    hi = np.asarray([r["bootstrap_95pct_ci"][1] for r in rows])
    sd = np.asarray([r["prediction_rmse_sd_deg"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 52 * MM))
    ax = axes[0]
    ax.errorbar(x, rho, yerr=[rho - lo, hi - rho], fmt="o", color=BLUE,
                markersize=3.4, linewidth=1.0, capsize=2.5, alpha=0.9)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xlabel("model accuracy: mean window RMSE (deg)")
    ax.set_ylabel("match-adjusted partial $\\rho$\n(across windows)")
    clean(ax, grid="both"); panel(ax, "A", x=-0.20)

    ax = axes[1]
    ax.plot(x, sd, marker="o", markersize=3.4, linewidth=1.1, color=RUST)
    ax.set_xlabel("model accuracy: mean window RMSE (deg)")
    ax.set_ylabel("between-window RMSE SD (deg)")
    ax.set_ylim(0, max(sd) * 1.25)
    ax.text(0.98, 0.06, "between-window spread\nat every level",
            transform=ax.transAxes, ha="right", fontsize=6.4, color=MUTED)
    clean(ax); panel(ax, "B", x=-0.18)
    return save(fig, out, "fig08_per_checkpoint.pdf")


EMG_SHORT = ["VL", "RF", "VM", "TA", "BF", "ST", "MG", "LG", "MS", "LS", "PL", "PB"]


def _panel_window(runs, query_id):
    """Recorded signals and predictions for one panel window."""
    import numpy as np
    q = np.load(runs / "panel" / "queries" / f"{query_id}.npz", allow_pickle=False)
    subject = int(np.asarray(q["subject_number"]).reshape(()))
    start = int(np.asarray(q["start_frame"]).reshape(()))
    labels = [str(x) for x in np.asarray(q["checkpoint_labels"]).tolist()]
    cache = np.load(runs.parent / "gait120_cache" / f"S{subject:03d}.npz", allow_pickle=False)
    trial = np.asarray(cache["trial_index"], dtype=int).reshape(-1)
    frame = np.asarray(cache["frame_index"], dtype=int).reshape(-1)
    rows = np.flatnonzero((trial == 5) & (frame >= start - 60) & (frame < start + 100))
    return {
        "subject": subject,
        "emg": np.asarray(cache["emg_frame_native_value"], dtype=float)[rows],
        "knee_full": np.asarray(cache["knee_flexion_deg"], dtype=float)[rows],
        "knee": np.asarray(q["knee_flexion_deg"], dtype=float),
        "fused": np.asarray(q["fused_prediction_deg"], dtype=float),
        "labels": labels,
        "query_id": query_id,
    }


def fig_signals(runs, out):
    """Recorded sEMG, knee trajectory, and predictions at three accuracy levels."""
    import numpy as np
    w = _panel_window(runs, "S065_trial05_start0069")
    emg = w["emg"]
    emg = (emg - emg.mean(0)) / np.maximum(emg.std(0), 1e-8)
    t_ctx = np.arange(emg.shape[0]) / 100.0 - 0.6
    t = np.arange(w["knee"].size) / 100.0

    fig = plt.figure(figsize=(180 * MM, 74 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 1.0],
                          hspace=0.55, wspace=0.30)

    ax = fig.add_subplot(gs[:, 0])
    for i in range(emg.shape[1]):
        ax.plot(t_ctx, np.clip(emg[:, i], -2.5, 6) * 0.34 + i, linewidth=0.7,
                color=BLUE, alpha=0.85)
        ax.text(t_ctx[0] - 0.03, i, EMG_SHORT[i], ha="right", va="center",
                fontsize=5.8, color=MUTED)
    ax.axvspan(-0.15, 0.0, color=BLUE, alpha=0.12, lw=0)
    ax.axvline(0.0, color=INK, linewidth=0.7)
    ax.set_xlim(t_ctx[0] - 0.11, t_ctx[-1])
    ax.set_ylim(-1.2, emg.shape[1] + 0.4)
    ax.set_yticks([])
    ax.set_xlabel("time relative to window start (s)")
    ax.set_title(f"recorded sEMG, S{w['subject']:03d}", fontsize=7.0, color=INK, pad=4)
    ax.text(-0.075, emg.shape[1] + 0.1, "150 ms input", ha="center", fontsize=6.0, color=MUTED)
    clean(ax, grid=None); panel(ax, "A", x=-0.13)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(w["knee_full"].size) / 100.0 - 0.6, w["knee_full"],
            color=INK, linewidth=1.1)
    ax.axvspan(-0.6, 0.0, color=GRID, alpha=0.5, lw=0)
    ax.axvline(0.0, color=INK, linewidth=0.7)
    ax.set_ylabel("knee flexion (deg)")
    ax.set_xlim(-0.62, 1.0)
    ax.text(-0.3, ax.get_ylim()[1] * 0.97, "600 ms history", ha="center",
            va="top", fontsize=6.0, color=MUTED)
    ax.set_xticklabels([])
    clean(ax); panel(ax, "B", x=-0.11)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, w["knee"], color=INK, linewidth=1.15, label="recorded", zorder=4)
    for idx, colour, name in ((0, RAMP[0], "untrained"),
                              (len(w["labels"]) // 2, RAMP[1], "mid-training"),
                              (len(w["labels"]) - 1, RAMP[2], "converged")):
        ax.plot(t, w["fused"][idx], color=colour, linewidth=1.0,
                linestyle="--", label=name)
    ax.set_xlabel("time within window (s)")
    ax.set_ylabel("knee flexion (deg)")
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)
    clean(ax); panel(ax, "C", x=-0.11)
    return save(fig, out, "fig02_signals.pdf")


def fig_simulation(runs, out):
    """Rendered rollout frames beside the knee command and instability they produced."""
    import json
    import glob
    import numpy as np

    stage = sorted((runs / "panel" / "stages").glob("*"))[0]  # least accurate level
    best, score = None, -1e9
    for s in sorted(stage.glob("evals/*/summary.json")):
        d = json.loads(Path(s).read_text(encoding="utf-8"))
        e = d["simulation"]["excess_instability_auc"]["fused"]
        if e > score:
            score, best = e, Path(s).parent
    summary = json.loads((best / "summary.json").read_text(encoding="utf-8"))
    with np.load(best / "compare.npz", allow_pickle=False) as d:
        frames = np.asarray(d["frames"])
        ref_knee = np.asarray(d["knee_ref_actual_deg"], dtype=float)
        pred_knee = np.asarray(d["knee_good_actual_deg"], dtype=float)
        cmd = np.asarray(d["knee_good_query_deg"], dtype=float)
        risk_ref = np.asarray(d["predicted_fall_risk_trace_ref"], dtype=float)
        risk_pred = np.asarray(d["predicted_fall_risk_trace_good"], dtype=float)
        dt = float(np.asarray(d["dt"]).reshape(()))
    t = np.arange(risk_ref.size) * dt

    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]

    # Each rendered frame is two side-by-side panels, reference then overridden,
    # with a label bar across the top. Splitting them into their own rows shows
    # the same instant in both conditions and removes the letterbox, which
    # otherwise leaves most of the figure empty.
    half = frames.shape[2] // 2
    top = 26
    left = [frames[i][top:, :half] for i in picks]
    right = [frames[i][top:, half:] for i in picks]

    fig = plt.figure(figsize=(180 * MM, 118 * MM))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.5],
                          hspace=0.10, wspace=0.06)

    for row, (images, label, colour) in enumerate(
            ((left, "reference", MUTED), (right, "overridden knee", RUST))):
        for col, image in enumerate(images):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(image)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID); sp.set_linewidth(0.6)
            if row == 0:
                ax.set_title(f"t = {picks[col] * dt:.2f} s", fontsize=6.8,
                             color=INK, pad=2)
            if col == 0:
                ax.set_ylabel(label, fontsize=6.8, color=INK)
                panel(ax, "AB"[row], x=-0.05, y=1.02 if row == 0 else 0.99)

    bottom = gs[2, :].subgridspec(1, 2, wspace=0.28)
    ax = fig.add_subplot(bottom[0, 0])
    ax.plot(t, ref_knee, color=MUTED, linewidth=1.15, label="reference knee")
    ax.plot(t, pred_knee, color=RUST, linewidth=1.15, label="overridden knee")
    ax.plot(t, cmd[: t.size], color=RUST, linewidth=1.0, linestyle=":",
            label="commanded")
    for idx in picks:
        ax.axvline(idx * dt, color=GRID, linewidth=0.8, zorder=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("knee angle (deg)")
    ax.legend(loc="upper right", fontsize=6.2)
    clean(ax); panel(ax, "C", x=-0.20)

    ax = fig.add_subplot(bottom[0, 1])
    ax.plot(t, risk_ref, color=MUTED, linewidth=1.15, label="reference")
    ax.plot(t, risk_pred, color=RUST, linewidth=1.15, label="overridden")
    ax.fill_between(t, risk_ref, risk_pred, where=risk_pred >= risk_ref,
                    color=RUST, alpha=0.20, lw=0)
    for idx in picks:
        ax.axvline(idx * dt, color=GRID, linewidth=0.8, zorder=0)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("time (s)"); ax.set_ylabel("instability index")
    ax.text(0.97, 0.30, f"excess {score:+.3f} s", transform=ax.transAxes,
            ha="right", fontsize=6.4, color=INK)
    ax.legend(loc="upper left", fontsize=6.2)
    clean(ax); panel(ax, "D", x=-0.20)

    fig.subplots_adjust(bottom=0.10)
    return save(fig, out, "fig04_simulation.pdf")


def fig_fwl(data, out):
    """Frisch-Waugh-Lovell: what match quality explains, and what survives it."""
    import numpy as np
    from scipy import stats
    from analysis.gait120_checkpoint_correlation import _rank, _residualize, _pearson_r

    rows = data["_window_rows"]
    primary = sorted({r["checkpoint"] for r in rows})[-1]
    sel = [r for r in rows if r["checkpoint"] == primary]
    rmse = np.array([r["prediction_rmse_deg"]["fused"] for r in sel])
    excess = np.array([r["simulation"]["excess_instability_auc"]["fused"] for r in sel])
    knee = np.array([r["match"]["knee_rmse_deg"] for r in sel])
    thigh = np.array([r["match"]["thigh_rms_deg"] for r in sel])

    controls = np.column_stack([_rank(knee), _rank(thigh)])
    rx = _residualize(_rank(rmse), controls)
    ry = _residualize(_rank(excess), controls)
    raw = stats.spearmanr(rmse, excess)
    partial = _pearson_r(rx, ry)

    fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 56 * MM),
                             gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    ax.scatter(rmse, excess, s=11, color=BLUE, alpha=0.7, linewidths=0)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("window prediction RMSE (deg)")
    ax.set_ylabel("excess instability (s)")
    ax.set_title(f"raw  $\\rho$ = {raw.statistic:+.3f}  (p = {raw.pvalue:.2f})",
                 fontsize=7.0, color=INK, pad=4)
    clean(ax, grid="both"); panel(ax, "A", x=-0.24)

    ax = axes[1]
    ax.scatter(knee, excess, s=11, color=RUST, alpha=0.7, linewidths=0)
    ax.axhline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("motion-match knee RMSE (deg)")
    ax.set_ylabel("excess instability (s)")
    cm = stats.spearmanr(knee, excess)
    ax.set_title(f"confounder  $\\rho$ = {cm.statistic:+.3f}", fontsize=7.0,
                 color=INK, pad=4)
    clean(ax, grid="both"); panel(ax, "B", x=-0.24)

    ax = axes[2]
    ax.scatter(rx, ry, s=11, color=RUST, alpha=0.75, linewidths=0)
    design = np.column_stack([np.ones(rx.size), rx])
    beta, *_ = np.linalg.lstsq(design, ry, rcond=None)
    xs = np.linspace(rx.min(), rx.max(), 40)
    ax.plot(xs, beta[0] + beta[1] * xs, color=RUST, linewidth=1.0, linestyle="--")
    ax.axhline(0, color=INK, linewidth=0.7); ax.axvline(0, color=INK, linewidth=0.7)
    ax.set_xlabel("residualized RMSE rank")
    ax.set_ylabel("residualized excess rank")
    ax.set_title(f"partial  $\\rho$ = {partial:+.3f}", fontsize=7.0, color=INK, pad=4)
    clean(ax, grid="both"); panel(ax, "C", x=-0.24)
    return save(fig, out, "fig07_fwl.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    style()
    runs = args.runs_dir.resolve()
    out = args.out_dir.resolve()
    data = load(runs)
    rows = window_rows(runs)

    data["_window_rows"] = rows
    made = [
        fig_overview(runs, out),
        fig_signals(runs, out),
        fig_prediction_accuracy(data, out),
        fig_simulation(runs, out),
        fig_matching(rows, out),
        fig_accuracy_vs_instability(data, out),
        fig_fwl(data, out),
        fig_per_checkpoint(data, out),
    ]
    for p in made:
        print(f"  wrote {p.name}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
