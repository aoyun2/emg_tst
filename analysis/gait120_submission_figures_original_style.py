from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats


EVIDENCE: Path
RECORDINGS: Path
HIGHRES: Path

# Palette and line hierarchy follow the approved manuscript figures.
INK = "#202533"
GRAY = "#5B5B5B"
MID_GRAY = "#8D929A"
LIGHT_GRAY = "#D7DBE0"
BLUE = "#4F86B4"
RED = "#C64E3A"
ORANGE = "#E58A2B"
BEIGE = "#F4EFE7"
GREEN = "#557A48"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#A9AFB8",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": "#D5DAE1",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / name, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def rounded(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    edge: str,
    face: str = "white",
    lw: float = 1.2,
    radius: float = 0.018,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        edgecolor=edge,
        facecolor=face,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 1.3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            transform=ax.transAxes,
            shrinkA=1.5,
            shrinkB=1.5,
            clip_on=False,
        )
    )


def waveform(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0: float,
    amplitude: float,
    values: np.ndarray,
    *,
    color: str = BLUE,
    lw: float = 1.0,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)
    scale = np.max(np.abs(values)) or 1.0
    x = np.linspace(x0, x1, values.size)
    y = y0 + amplitude * values / scale
    ax.plot(x, y, color=color, lw=lw, transform=ax.transAxes, clip_on=False)


def read_video(path: Path, index: int) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    index = max(0, min(index, count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), index / fps


def source_data_figure(out: Path) -> None:
    setup = cv2.cvtColor(
        cv2.imread(str(HIGHRES / "fig1_lw1200.png")), cv2.COLOR_BGR2RGB
    )
    sensors = cv2.cvtColor(
        cv2.imread(str(HIGHRES / "fig2.png")), cv2.COLOR_BGR2RGB
    )
    # Only the level-walking panel is retained from the multi-activity setup.
    setup_crop = setup[55:430, 0:510]

    fig = plt.figure(figsize=(10.8, 3.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.82, 2.35], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(setup_crop)
    ax.axis("off")
    ax.text(
        0.02,
        0.97,
        "A",
        transform=ax.transAxes,
        va="top",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(sensors)
    ax.axis("off")
    ax.text(
        0.01,
        0.97,
        "B",
        transform=ax.transAxes,
        va="top",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, out, "gait120_data.png")


def pipeline_figure(out: Path) -> None:
    rng = np.random.default_rng(12)
    t = np.linspace(0, 1, 180)
    knee = 0.65 * np.sin(2 * np.pi * (t - 0.08)) + 0.14 * np.sin(4 * np.pi * t)
    emg = [
        np.maximum(0, np.sin(2 * np.pi * (t + p)))
        * (0.54 + 0.46 * rng.random(t.size))
        for p in (0.02, 0.17, 0.36, 0.58)
    ]
    recording = sorted(RECORDINGS.glob("panel_33_*.mp4"))[0]
    frame, _ = read_video(recording, 18)

    fig = plt.figure(figsize=(11.2, 4.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.065, 0.92, "Gait120 input", fontsize=12, fontweight="bold", color=INK)
    ax.plot([0.02, 0.205], [0.88, 0.88], color=INK, lw=1.1)
    rounded(ax, 0.03, 0.51, 0.16, 0.26, edge=BLUE, face="white")
    ax.text(0.05, 0.72, "Knee-angle history", fontsize=9.3, fontweight="bold")
    waveform(ax, 0.05, 0.17, 0.59, 0.045, knee, color=BLUE, lw=1.25)
    rounded(ax, 0.03, 0.14, 0.16, 0.27, edge=BLUE, face="white")
    ax.text(0.05, 0.36, "12 sEMG envelopes", fontsize=9.3, fontweight="bold")
    for yy, values in zip((0.30, 0.25, 0.20, 0.15), emg):
        waveform(ax, 0.05, 0.17, yy, 0.012, values, color=BLUE, lw=0.9)

    ax.text(0.385, 0.92, "Residual-fusion model", fontsize=12, fontweight="bold", color=INK)
    ax.plot([0.245, 0.62], [0.88, 0.88], color=INK, lw=1.1)
    rounded(ax, 0.26, 0.18, 0.34, 0.57, edge="#8A775D", face=BEIGE, lw=1.1, radius=0.035)
    rounded(ax, 0.30, 0.54, 0.25, 0.12, edge=GREEN, face="white")
    ax.text(0.425, 0.605, "Kinematic ridge forecast", ha="center", va="center", fontsize=9.4, fontweight="bold")
    rounded(ax, 0.30, 0.31, 0.25, 0.12, edge=RED, face="white")
    ax.text(0.425, 0.37, "sEMG residual correction", ha="center", va="center", fontsize=9.4, fontweight="bold")
    ax.text(0.425, 0.235, "100-ms knee-angle prediction", ha="center", fontsize=9.3, color=INK)

    arrow(ax, (0.19, 0.64), (0.255, 0.64))
    arrow(ax, (0.19, 0.28), (0.255, 0.37))
    arrow(ax, (0.555, 0.60), (0.62, 0.60), color=GREEN)
    arrow(ax, (0.555, 0.37), (0.62, 0.37), color=RED)

    ax.text(0.735, 0.92, "Motion matching and simulation", fontsize=12, fontweight="bold", color=INK)
    ax.plot([0.67, 0.98], [0.88, 0.88], color=INK, lw=1.1)
    match_ax = fig.add_axes([0.69, 0.53, 0.27, 0.23])
    query = 30 + 18 * knee
    reference = query + 2.2 * np.sin(6 * np.pi * t + 0.3)
    match_ax.plot(t, reference, color=GRAY, lw=1.5, label="MoCapAct reference")
    match_ax.plot(t, query, color=RED, lw=1.5, label="Gait120 query")
    match_ax.set_xticks([])
    match_ax.set_yticks([])
    match_ax.grid(False)
    match_ax.legend(fontsize=7.4, loc="upper right")
    match_ax.set_title("Motion match", loc="left", fontsize=9.4, fontweight="bold")
    image_ax = fig.add_axes([0.69, 0.16, 0.27, 0.27])
    image_ax.imshow(frame)
    image_ax.axis("off")
    image_ax.set_title("Paired MuJoCo comparison", loc="left", fontsize=9.4, fontweight="bold", pad=3)
    arrow(ax, (0.61, 0.49), (0.66, 0.49))
    save(fig, out, "pipeline_overview.png")


def architecture_figure(out: Path) -> None:
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 180)
    knee = 0.62 * np.sin(2 * np.pi * (t - 0.08)) + 0.13 * np.sin(4 * np.pi * t)
    emg = [
        np.maximum(0, np.sin(2 * np.pi * (t + p)))
        * (0.55 + 0.45 * rng.random(t.size))
        for p in (0.02, 0.17, 0.36, 0.58)
    ]
    fig, ax = plt.subplots(figsize=(11.0, 4.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    headings = [
        (0.02, 0.22, "Recorded histories"),
        (0.29, 0.22, "Baseline forecast"),
        (0.55, 0.19, "sEMG correction"),
        (0.78, 0.20, "Paired outputs"),
    ]
    for x, w, title in headings:
        ax.text(x + w / 2, 0.94, title, ha="center", fontsize=12.2, fontweight="bold", color=INK)
        ax.plot([x, x + w], [0.90, 0.90], color=INK, lw=1.1)

    rounded(ax, 0.025, 0.56, 0.20, 0.25, edge=BLUE)
    ax.text(0.045, 0.765, "Knee angle", fontsize=10.2, fontweight="bold")
    ax.text(0.045, 0.71, "60 frames (600 ms)", fontsize=8.8, color=GRAY)
    waveform(ax, 0.045, 0.205, 0.625, 0.045, knee, color=BLUE, lw=1.2)

    rounded(ax, 0.025, 0.13, 0.20, 0.31, edge=BLUE)
    ax.text(0.045, 0.39, "12 sEMG envelopes", fontsize=10.2, fontweight="bold")
    ax.text(0.045, 0.335, "15 frames (150 ms)", fontsize=8.8, color=GRAY)
    for yy, values in zip((0.28, 0.235, 0.19, 0.145), emg):
        waveform(ax, 0.045, 0.205, yy, 0.013, values, color=BLUE, lw=0.85)

    rounded(ax, 0.30, 0.55, 0.20, 0.23, edge=GREEN, face="#F4F7F1")
    ax.text(0.40, 0.69, "Kinematic ridge", ha="center", fontsize=10.6, fontweight="bold")
    ax.text(0.40, 0.625, "future knee angle", ha="center", fontsize=9.0, color=GRAY)

    rounded(ax, 0.55, 0.13, 0.18, 0.31, edge=RED, face="#FBF4F2")
    ax.text(0.64, 0.34, "Residual ridge", ha="center", fontsize=10.6, fontweight="bold")
    ax.text(0.64, 0.255, "sEMG correction", ha="center", fontsize=9.0, color=GRAY)

    rounded(ax, 0.79, 0.58, 0.18, 0.18, edge=GRAY, face="#F6F6F6")
    ax.text(0.88, 0.69, "Without sEMG", ha="center", fontsize=10.4, fontweight="bold")
    ax.text(0.88, 0.63, "baseline only", ha="center", fontsize=8.8, color=GRAY)
    rounded(ax, 0.79, 0.20, 0.18, 0.20, edge=RED, face="#FBF4F2")
    ax.text(0.88, 0.325, "Residual fusion", ha="center", fontsize=10.4, fontweight="bold")
    ax.text(0.88, 0.255, "baseline + correction", ha="center", fontsize=8.8, color=GRAY)

    arrow(ax, (0.225, 0.68), (0.292, 0.68))
    arrow(ax, (0.225, 0.285), (0.542, 0.285))
    arrow(ax, (0.50, 0.68), (0.782, 0.68), color=GREEN)
    ax.plot([0.51, 0.755], [0.64, 0.64], color=GREEN, lw=1.3, transform=ax.transAxes)
    ax.plot([0.755, 0.755], [0.64, 0.30], color=GREEN, lw=1.3, transform=ax.transAxes)
    arrow(ax, (0.73, 0.30), (0.782, 0.30), color=RED)
    ax.text(0.755, 0.30, "+", ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
    save(fig, out, "model_architecture.png")


def ablation_figure(out: Path) -> None:
    summary = json.loads(
        (EVIDENCE / "prediction_confirmation" / "ablation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    values = np.asarray(summary["improvement_deg"], dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = sd / np.sqrt(values.size)
    ci = stats.t.interval(0.95, values.size - 1, loc=mean, scale=se)
    t_result = stats.ttest_1samp(values, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.85), gridspec_kw={"width_ratios": [0.9, 1.55]})
    ax = axes[0]
    rng = np.random.default_rng(20260826)
    jitter = rng.normal(0, 0.055, size=values.size)
    ax.scatter(jitter, values, s=24, color=np.where(values > 0, RED, MID_GRAY), alpha=0.72, edgecolor="white", linewidth=0.35)
    ax.errorbar(0.22, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt="o", color=INK, markersize=5, capsize=4, lw=1.4)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xlim(-0.22, 0.38)
    ax.set_xticks([0, 0.22], ["Participants", "Mean\n(95% CI)"])
    ax.set_ylabel("RMSE reduction with sEMG (degrees)")
    ax.set_title("A  Participant effects", loc="left", fontweight="bold")
    ax.text(0.04, 0.96, f"mean = {mean:.3f}°\n95% CI {ci[0]:.3f} to {ci[1]:.3f}°", transform=ax.transAxes, va="top", fontsize=8.8)

    ax = axes[1]
    ordered = np.sort(values)
    colors = np.where(ordered > 0, RED, MID_GRAY)
    ax.bar(np.arange(1, values.size + 1), ordered, color=colors, width=0.84)
    ax.axhline(0, color=INK, lw=1.0)
    ax.axhline(mean, color=INK, ls="--", lw=1.2)
    ax.set_xlabel("Confirmation participant (ordered)")
    ax.set_ylabel("RMSE reduction with sEMG (degrees)")
    ax.set_title("B  Direction and magnitude", loc="left", fontweight="bold")
    ax.text(0.03, 0.96, f"t(89) = {t_result.statistic:.2f}\np = {t_result.pvalue:.2e}\n70 of 90 improved", transform=ax.transAxes, va="top", fontsize=8.8)
    save(fig, out, "prediction_ablation.png")


def simulation_sequence(out: Path) -> None:
    recording = sorted(RECORDINGS.glob("panel_33_*.mp4"))[0]
    frames = [read_video(recording, index) for index in (0, 28)]
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.85))
    for label, ax, (frame, time) in zip(("A", "B"), axes, frames):
        ax.imshow(frame)
        ax.axis("off")
        ax.text(0.006, 0.94, label, transform=ax.transAxes, va="top", color="white", fontsize=13, fontweight="bold")
        ax.text(0.012, 0.08, f"t = {time:.2f} s", transform=ax.transAxes, va="bottom", color="white", fontsize=10, bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "none", "pad": 2.5})
    fig.subplots_adjust(hspace=0.04, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, out, "simulation_sequence.png")


def matching_figure(out: Path) -> None:
    summary = json.loads(
        (EVIDENCE / "physics" / "matching_summary.json").read_text(encoding="utf-8")
    )
    data = pd.DataFrame(summary["windows"])
    ranks = data["candidate_rank"].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.6), gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    colors = np.where(ranks == 0, BLUE, RED)
    ax.scatter(data["knee_rmse_deg"], data["thigh_rms_deg"], c=colors, s=35, alpha=0.8, edgecolor="white", linewidth=0.45)
    ax.axvline(10, color=MID_GRAY, ls="--", lw=1.0)
    ax.axhline(15, color=MID_GRAY, ls="--", lw=1.0)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 15.7)
    ax.set_xlabel("Knee matching RMSE (degrees)")
    ax.set_ylabel("Right-thigh pitch RMSE (degrees)")
    ax.set_title("A  Matching error", loc="left", fontweight="bold")
    ax.text(0.04, 0.95, "mean knee = 4.32°\nmean thigh = 3.63°", transform=ax.transAxes, va="top", fontsize=8.8)

    ax = axes[1]
    counts = pd.Series(ranks + 1).value_counts().sort_index()
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=np.where(counts.index == 1, BLUE, RED), height=0.62)
    ax.set_yticks(y, [f"Candidate {int(v)}" for v in counts.index])
    ax.invert_yaxis()
    ax.set_xlabel("Windows")
    ax.set_title("B  Stable reference selected", loc="left", fontweight="bold")
    for yy, value in zip(y, counts.values):
        ax.text(value + 0.8, yy, str(int(value)), va="center", fontsize=9.0)
    ax.set_xlim(0, max(counts.values) * 1.15)
    save(fig, out, "motion_matching_quality.png")


def representative_simulations(out: Path) -> None:
    files = [
        sorted(RECORDINGS.glob("panel_00_*.mp4"))[0],
        sorted(RECORDINGS.glob("panel_33_*.mp4"))[0],
        sorted(RECORDINGS.glob("panel_77_*.mp4"))[0],
    ]
    frames = [read_video(path, 18)[0] for path in files]
    fig, axes = plt.subplots(3, 1, figsize=(8.7, 5.6))
    for label, ax, frame, path in zip(("A", "B", "C"), axes, frames, files):
        ax.imshow(frame)
        ax.axis("off")
        subject = path.stem.split("_", 2)[2].replace("_trial05_", ", trial 5, ").replace("start", "start frame ")
        ax.text(0.006, 0.94, f"{label}  {subject}", transform=ax.transAxes, va="top", color="white", fontsize=8.6, bbox={"facecolor": "black", "alpha": 0.70, "edgecolor": "none", "pad": 2.4})
    fig.subplots_adjust(hspace=0.035, left=0.01, right=0.99, top=0.99, bottom=0.01)
    save(fig, out, "simulation_representatives.png")


def training_figure(out: Path) -> None:
    checkpoint = pd.read_csv(EVIDENCE / "analysis" / "checkpoint_summary.csv")
    participant = pd.read_csv(EVIDENCE / "analysis" / "participant_checkpoint.csv")
    order = checkpoint["checkpoint"].tolist()
    fractions = checkpoint["fraction"].to_numpy(float)
    means = checkpoint["mean_rmse_deg"].to_numpy(float)
    lo = checkpoint["rmse_ci_low"].to_numpy(float)
    hi = checkpoint["rmse_ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    for _, group in participant.groupby("subject"):
        values = group.set_index("checkpoint").reindex(order)["simulation_rmse_fused_deg"].to_numpy(float)
        ax.plot(fractions, values, color=LIGHT_GRAY, lw=0.6, alpha=0.46, zorder=1)
    ax.fill_between(fractions, lo, hi, color=BLUE, alpha=0.16, zorder=2)
    ax.plot(fractions, means, color=BLUE, lw=2.3, marker="o", ms=5.2, zorder=3, label="Mean participant RMSE")
    ax.scatter(fractions[0], means[0], color=RED, s=46, zorder=4, label="Least accurate checkpoint")
    for x, y in zip(fractions, means):
        ax.annotate(f"{y:.2f}°", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.3)
    ax.set_xticks(fractions, [str(int(x)) for x in fractions])
    ax.set_xlabel("Participant training data used (%)")
    ax.set_ylabel("Prediction RMSE (degrees)")
    ax.legend(loc="upper right", fontsize=8.5)
    save(fig, out, "training_rmse.png")


def simulation_outcomes(out: Path) -> None:
    data = pd.read_csv(EVIDENCE / "analysis" / "participant_primary.csv")
    values = [
        data["reference_auc"].to_numpy(float),
        data["fused_auc"].to_numpy(float),
        data["no_emg_auc"].to_numpy(float),
    ]
    labels = ["Reference", "Residual fusion", "Without sEMG"]
    colors = [GRAY, RED, BLUE]
    reduction = data["no_emg_excess_auc"].to_numpy(float) - data["fused_excess_auc"].to_numpy(float)
    t_result = stats.ttest_1samp(reduction, 0.0)
    rng = np.random.default_rng(20260826)
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.85), gridspec_kw={"width_ratios": [1.45, 0.8]})
    ax = axes[0]
    bp = ax.boxplot(values, positions=[0, 1, 2], widths=0.48, patch_artist=True, showfliers=False, medianprops={"color": INK, "lw": 1.3})
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor("white")
        box.set_edgecolor(color)
        box.set_linewidth(1.3)
    for i, (vals, color) in enumerate(zip(values, colors)):
        ax.scatter(i + rng.normal(0, 0.055, len(vals)), vals, s=18, color=color, alpha=0.62, edgecolor="white", linewidth=0.3)
    ax.set_xticks([0, 1, 2], labels)
    ax.set_ylabel("Instability AUC (score-seconds)")
    ax.set_title("A  Full-data simulation outcomes", loc="left", fontweight="bold")

    ax = axes[1]
    ax.boxplot([reduction], positions=[0], widths=0.42, patch_artist=True, showfliers=False, boxprops={"facecolor": "white", "edgecolor": RED, "lw": 1.3}, medianprops={"color": INK, "lw": 1.3})
    ax.scatter(rng.normal(0, 0.055, len(reduction)), reduction, s=18, color=RED, alpha=0.65, edgecolor="white", linewidth=0.3)
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks([0], ["Without sEMG -\nresidual fusion"])
    ax.set_ylabel("Paired excess-AUC difference")
    ax.set_title("B  Paired physical effect", loc="left", fontweight="bold")
    ax.text(0.04, 0.96, f"mean = {np.mean(reduction):.4f}\nt(64) = {t_result.statistic:.2f}\np = {t_result.pvalue:.3f}", transform=ax.transAxes, va="top", fontsize=8.8)
    save(fig, out, "simulation_outcomes.png")


def correlation_figure(out: Path) -> None:
    data = pd.read_csv(EVIDENCE / "analysis" / "participant_primary.csv")
    summary = json.loads((EVIDENCE / "analysis" / "statistical_summary.json").read_text(encoding="utf-8"))
    x = data["source_rmse_fused_deg"].to_numpy(float)
    y = data["fused_excess_auc"].to_numpy(float)
    knee = data["match_knee_rmse_deg"].to_numpy(float)
    xr = np.asarray(summary["rmse_vs_physical_outcome"]["x_residual"], dtype=float)
    yr = np.asarray(summary["rmse_vs_physical_outcome"]["y_residual"], dtype=float)
    rho_raw, p_raw = stats.spearmanr(x, y)
    rho_match, p_match = stats.spearmanr(knee, y)
    rho_partial = float(summary["rmse_vs_physical_outcome"]["rho"])
    p_partial = float(summary["rmse_vs_physical_outcome"]["permutation_p_two_sided"])
    panels = [
        (x, y, "A", "Prediction RMSE (degrees)", "Excess instability AUC", rho_raw, p_raw),
        (knee, y, "B", "Knee matching RMSE (degrees)", "Excess instability AUC", rho_match, p_match),
        (xr, yr, "C", "Residualized prediction-RMSE rank", "Residualized excess-AUC rank", rho_partial, p_partial),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.45))
    for ax, (xx, yy, label, xlabel, ylabel, rho, pvalue) in zip(axes, panels):
        ax.scatter(xx, yy, s=29, color="#6D7684", alpha=0.86, edgecolor="white", linewidth=0.45)
        beta = np.polyfit(xx, yy, 1)
        grid = np.linspace(np.min(xx), np.max(xx), 100)
        ax.plot(grid, np.polyval(beta, grid), color=RED, lw=1.7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", fontsize=14, fontweight="bold")
        ax.text(0.97, 0.97, f"rho = {rho:.3f}\np = {pvalue:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=9.0)
    save(fig, out, "correlation_analysis.png")


def accuracy_transition(out: Path) -> None:
    data = pd.read_csv(EVIDENCE / "analysis" / "checkpoint_summary.csv")
    x = data["mean_rmse_deg"].to_numpy(float)
    y = data["mean_excess_auc"].to_numpy(float)
    ylo = data["auc_ci_low"].to_numpy(float)
    yhi = data["auc_ci_high"].to_numpy(float)
    losses = data["fused_balance_losses"].to_numpy(int)
    fractions = data["fraction"].to_numpy(int)
    observed_lo = float(x[fractions == 10][0])
    observed_hi = float(x[fractions == 5][0])
    order = np.argsort(x)

    fig, axes = plt.subplots(2, 1, figsize=(7.7, 5.15), sharex=True, gridspec_kw={"height_ratios": [1.4, 0.82], "hspace": 0.08})
    ax = axes[0]
    ax.axvspan(observed_lo, observed_hi, color=RED, alpha=0.10, label="Observed transition interval")
    ax.errorbar(x[order], y[order], yerr=np.vstack([y[order] - ylo[order], yhi[order] - y[order]]), color=BLUE, marker="o", ms=5.3, lw=1.8, capsize=2.8)
    ax.set_ylabel("Mean excess instability AUC")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.text(observed_hi, y[fractions == 5][0], "5%", ha="left", va="bottom", fontsize=8.5)
    ax.text(observed_lo, y[fractions == 10][0], "10%", ha="right", va="bottom", fontsize=8.5)

    ax = axes[1]
    ax.axvspan(observed_lo, observed_hi, color=RED, alpha=0.10)
    ax.plot(x[order], losses[order], color=RED, marker="s", ms=5.0, lw=1.6)
    ax.set_xlabel("Prediction RMSE (degrees)")
    ax.set_ylabel("Balance losses\n(out of 80)")
    ax.set_ylim(-1, max(losses) + 5)
    save(fig, out, "accuracy_transition.png")


def main() -> None:
    global EVIDENCE, RECORDINGS, HIGHRES
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="Extracted Additional_file_2_reproducibility_evidence directory",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        required=True,
        help="Directory containing the eight extracted representative MP4 files",
    )
    parser.add_argument(
        "--source-images-dir",
        type=Path,
        required=True,
        help="Directory containing fig1_lw1200.png and fig2.png from Boo et al.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    EVIDENCE = args.evidence_dir.resolve()
    RECORDINGS = args.recordings_dir.resolve()
    HIGHRES = args.source_images_dir.resolve()
    for required in (
        EVIDENCE / "analysis" / "participant_primary.csv",
        RECORDINGS,
        HIGHRES / "fig1_lw1200.png",
        HIGHRES / "fig2.png",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    source_data_figure(args.output_dir)
    pipeline_figure(args.output_dir)
    architecture_figure(args.output_dir)
    ablation_figure(args.output_dir)
    simulation_sequence(args.output_dir)
    matching_figure(args.output_dir)
    representative_simulations(args.output_dir)
    training_figure(args.output_dir)
    simulation_outcomes(args.output_dir)
    correlation_figure(args.output_dir)
    accuracy_transition(args.output_dir)


if __name__ == "__main__":
    main()
