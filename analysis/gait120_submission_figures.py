"""Generate the original-draft-informed manuscript figures.

The figure sequence follows the visual logic of the author's original paper:
signal traces lead into a compact model diagram, an actual held-out trajectory,
simulation frames, the matching/stability illustrations, and direct result
plots. Schematic sEMG traces are labelled as illustrations; every quantitative
mark is generated from the archived experiment outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd
from scipy import stats


MM = 1 / 25.4

BLUE = "#4B79A1"
BLUE_PALE = "#E7F0F7"
ORANGE = "#C56343"
ORANGE_PALE = "#F6EAE4"
GREEN = "#6E8F66"
GREEN_PALE = "#EAF1E7"
WARM_PALE = "#F2EEE6"
BLACK = "#202427"
DARK = "#4E565B"
MID = "#7B8388"
LIGHT = "#B8BEC2"
GRID = "#D9DDE0"

MUSCLES = ["VL", "RF", "VM", "TA", "BF", "ST", "GM", "GL", "SM", "SL", "PL", "PB"]

DATA_ALIASES = {
    "prediction_confirmation/ablation_summary.json": "ablation_summary.json",
    "accuracy_path/checkpoints/fraction_100pct/test_predictions.npz": "test_predictions.npz",
    "physics/matching_summary.json": "matching_summary.json",
    "analysis/participant_primary.csv": "participant_primary.csv",
    "analysis/statistical_summary.json": "statistical_summary.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate eight publication figures from the extracted reproducibility "
            "archive and representative simulation recordings."
        )
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="Extracted Additional file 2 root.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        required=True,
        help="Directory containing the extracted representative MP4 recordings.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.6,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.7,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.65,
            "axes.labelcolor": BLACK,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def clean_axes(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=BLACK,
        clip_on=False,
    )


def header_note(ax: plt.Axes, text: str, x: float = 0.98) -> None:
    """Place compact statistics above the plotting field, away from data marks."""
    ax.text(
        x,
        1.015,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.1,
        linespacing=1.05,
        color=BLACK,
        clip_on=False,
    )


def top_band_note(
    ax: plt.Axes,
    text: str,
    *,
    band_fraction: float = 0.25,
    fontsize: float = 5.9,
) -> None:
    """Place a compact result note in empty space reserved above the data."""
    lower, upper = ax.get_ylim()
    span = max(upper - lower, np.finfo(float).eps)
    ax.set_ylim(lower, upper + band_fraction * span)
    ax.text(
        0.98,
        0.97,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=fontsize,
        color=DARK,
        linespacing=1.12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.96, "pad": 1.0},
        zorder=8,
    )


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.04}
    if path.suffix.lower() == ".pdf":
        kwargs["metadata"] = {"Title": filename, "Author": "Aaron Xiong"}
    else:
        kwargs["dpi"] = 400
    fig.savefig(path, **kwargs)
    plt.close(fig)
    return path


def resolve_data_path(evidence_dir: Path, relative: str) -> Path:
    direct = evidence_dir / relative
    if direct.is_file():
        return direct
    alias = DATA_ALIASES.get(relative)
    if alias:
        matches = sorted(evidence_dir.rglob(alias))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f"Could not resolve {relative} below {evidence_dir}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return mean, mean - half, mean + half


def flow_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = LIGHT,
    fontsize: float = 7.0,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.75,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=BLACK,
        zorder=3,
    )


def flow_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=DARK,
            shrinkA=4.5,
            shrinkB=5.0,
            zorder=1,
        )
    )


def schematic_emg(n: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic normalized envelopes used only as labelled illustrations."""
    x = np.linspace(-150.0, 0.0, n)
    signals: list[np.ndarray] = []
    centers = np.array([-126, -111, -96, -82, -69, -55, -42, -31, -22, -15, -9, -4])
    widths = np.array([18, 12, 20, 10, 17, 13, 15, 11, 18, 9, 14, 8])
    for index, (center, width) in enumerate(zip(centers, widths, strict=True)):
        primary = np.exp(-0.5 * ((x - center) / width) ** 2)
        secondary = 0.45 * np.exp(-0.5 * ((x - (center + 42)) / (width * 0.75)) ** 2)
        ripple = 0.035 * (1 + np.sin((index + 2) * x / 18.0 + index))
        signal = primary + secondary + ripple
        signals.append(signal / np.max(signal))
    return x, np.asarray(signals)


def load_predictions(evidence_dir: Path) -> dict[str, np.ndarray]:
    path = resolve_data_path(
        evidence_dir,
        "accuracy_path/checkpoints/fraction_100pct/test_predictions.npz",
    )
    with np.load(path, allow_pickle=False) as stored:
        return {key: np.asarray(stored[key]) for key in stored.files}


def subject_trace(
    predictions: dict[str, np.ndarray], subject: int = 67, start_frame: int = 109, length: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = np.asarray(predictions["target_frame"], dtype=np.int64)
    subjects = np.asarray(predictions["subject_number"], dtype=np.int64)
    keep = (subjects == subject) & (frames >= start_frame) & (frames < start_frame + length)
    order = np.argsort(frames[keep])
    selected_frames = frames[keep][order]
    if len(selected_frames) != length or not np.array_equal(
        selected_frames, np.arange(start_frame, start_frame + length)
    ):
        raise RuntimeError(f"Could not recover the expected S{subject:03d} trace")
    time = np.arange(length, dtype=np.float64) / 100.0
    return (
        time,
        np.asarray(predictions["target_deg"], dtype=np.float64)[keep][order],
        np.asarray(predictions["fused_prediction_deg"], dtype=np.float64)[keep][order],
        np.asarray(predictions["no_emg_prediction_deg"], dtype=np.float64)[keep][order],
    )


def history_and_target(
    predictions: dict[str, np.ndarray], subject: int = 67, first_frame: int = 70
) -> tuple[np.ndarray, np.ndarray, float]:
    frames = np.asarray(predictions["target_frame"], dtype=np.int64)
    subjects = np.asarray(predictions["subject_number"], dtype=np.int64)
    history_frames = np.arange(first_frame, first_frame + 60)
    history: list[float] = []
    for frame in history_frames:
        row = np.flatnonzero((subjects == subject) & (frames == frame))
        if len(row) != 1:
            raise RuntimeError(f"Missing S{subject:03d} frame {frame}")
        history.append(float(predictions["target_deg"][row[0]]))
    target_frame = first_frame + 69
    row = np.flatnonzero((subjects == subject) & (frames == target_frame))
    if len(row) != 1:
        raise RuntimeError(f"Missing S{subject:03d} target frame {target_frame}")
    return np.arange(-590, 10, 10), np.asarray(history), float(predictions["target_deg"][row[0]])


def find_recording(recordings_dir: Path) -> Path:
    matches = sorted(recordings_dir.rglob("panel_00_*.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one panel_00 MP4 under {recordings_dir}, found {len(matches)}"
        )
    return matches[0]


def read_video_frame(path: Path, frame_index: int) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not ok or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_index / fps


def video_panes(frame: np.ndarray) -> list[np.ndarray]:
    pane_width = frame.shape[1] // 3
    return [
        frame[32:356, column * pane_width + 3 : (column + 1) * pane_width - 3]
        for column in range(3)
    ]


def magenta_leg_mask(pane: np.ndarray) -> np.ndarray:
    """Recover the identically colored overridden right leg from a video pane."""
    hsv = cv2.cvtColor(pane, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([140, 100, 100], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8),
    )
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask > 0


def figure_experiment_overview(
    evidence_dir: Path, recordings_dir: Path, output_dir: Path
) -> Path:
    predictions = load_predictions(evidence_dir)
    time, target, fused, no_emg = subject_trace(predictions)
    emg_x, emg = schematic_emg()
    frame, _ = read_video_frame(find_recording(recordings_dir), 28)
    simulation = video_panes(frame)[1]

    fig = plt.figure(figsize=(170 * MM, 62 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.text(
        0.105,
        0.94,
        "Recorded signals",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontweight="bold",
    )
    for row, channel in enumerate((0, 2, 5, 7)):
        y0 = 0.72 - row * 0.105
        ax.add_patch(
            Rectangle(
                (0.018, y0), 0.175, 0.077, transform=ax.transAxes,
                facecolor="white", edgecolor=LIGHT, linewidth=0.55
            )
        )
        xplot = 0.025 + 0.158 * (emg_x - emg_x.min()) / np.ptp(emg_x)
        yplot = y0 + 0.012 + 0.050 * emg[channel]
        ax.plot(xplot, yplot, transform=ax.transAxes, color=BLUE, linewidth=0.75)
        ax.text(0.188, y0 + 0.039, MUSCLES[channel], transform=ax.transAxes,
                ha="right", va="center", fontsize=5.8, color=DARK)
    knee_ax = fig.add_axes([0.018, 0.125, 0.175, 0.16])
    knee_ax.plot(time, target, color=BLACK, linewidth=1.0)
    knee_ax.set_xticks([])
    knee_ax.set_yticks([])
    knee_ax.set_title("Knee angle", fontsize=6.2, pad=1.5)
    for spine in knee_ax.spines.values():
        spine.set_color(LIGHT)
        spine.set_linewidth(0.55)
    ax.text(0.105, 0.035, "sEMG + knee history", transform=ax.transAxes,
            ha="center", fontsize=6.4, color=DARK)

    flow_arrow(ax, (0.205, 0.50), (0.245, 0.50))
    flow_box(
        ax, 0.25, 0.315, 0.205, 0.37,
        "Residual-fusion predictor\n\nKinematic forecast\n+\nsEMG residual correction",
        WARM_PALE, edgecolor="#C8BEAE", fontsize=7.2,
    )
    ax.text(0.352, 0.735, "100-ms forecast", transform=ax.transAxes,
            ha="center", fontsize=6.4, color=DARK)
    flow_arrow(ax, (0.462, 0.50), (0.495, 0.50))

    prediction_ax = fig.add_axes([0.505, 0.20, 0.205, 0.57])
    prediction_ax.plot(time, target, color=BLACK, linewidth=1.05, label="Recorded")
    prediction_ax.plot(time, fused, color=BLUE, linewidth=1.0, label="Fusion")
    prediction_ax.plot(time, no_emg, color=ORANGE, linewidth=0.9, linestyle="--", label="No sEMG")
    prediction_ax.set_title("Held-out knee angle (deg)", pad=2.5, fontweight="bold")
    prediction_ax.set_xlabel("Time (s)")
    prediction_ax.legend(loc="upper left", fontsize=5.7)
    clean_axes(prediction_ax)
    flow_arrow(ax, (0.718, 0.50), (0.752, 0.50))

    flow_box(ax, 0.758, 0.57, 0.105, 0.18, "Motion\nmatching", GREEN_PALE,
             edgecolor="#AEC1AA", fontsize=6.7)
    flow_arrow(ax, (0.810, 0.56), (0.810, 0.41))
    image_ax = fig.add_axes([0.755, 0.115, 0.225, 0.29])
    image_ax.imshow(simulation)
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    image_ax.set_title("Paired MuJoCo simulation", fontsize=6.7, pad=2.0, fontweight="bold")
    for spine in image_ax.spines.values():
        spine.set_color(BLUE)
        spine.set_linewidth(0.75)

    return save_figure(fig, output_dir, "fig01_experiment_overview.pdf")


def figure_signal_model(evidence_dir: Path, output_dir: Path) -> Path:
    predictions = load_predictions(evidence_dir)
    emg_x, emg = schematic_emg()
    history_x, history, target = history_and_target(predictions)

    fig = plt.figure(figsize=(170 * MM, 98 * MM))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.28, 0.85], hspace=0.42, wspace=0.33)

    ax_emg = fig.add_subplot(grid[0, 0])
    for index, signal in enumerate(emg):
        ax_emg.plot(emg_x, signal + (len(emg) - 1 - index) * 1.15, color=BLUE, linewidth=0.75)
    ax_emg.set_yticks(np.arange(len(emg)) * 1.15)
    ax_emg.set_yticklabels(MUSCLES[::-1])
    ax_emg.set_xlim(-150, 0)
    ax_emg.set_xlabel("Time before final input frame (ms)")
    ax_emg.set_title("Illustrative normalized sEMG envelopes", pad=3, fontweight="bold")
    clean_axes(ax_emg, grid=None)
    panel_label(ax_emg, "A", x=-0.16)

    ax_knee = fig.add_subplot(grid[0, 1])
    ax_knee.plot(history_x, history, color=BLACK, linewidth=1.25)
    ax_knee.axvspan(0, 100, color=ORANGE_PALE, zorder=0)
    ax_knee.plot([100], [target], marker="o", color=ORANGE, markersize=4.2, zorder=4)
    ax_knee.axvline(0, color=MID, linewidth=0.7, linestyle="--")
    ax_knee.text(50, ax_knee.get_ylim()[1], "100-ms forecast", ha="center", va="top",
                 fontsize=6.2, color=DARK)
    ax_knee.annotate("target", (100, target), xytext=(72, target + 7), fontsize=6.4,
                     arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.7})
    ax_knee.set_xlim(-600, 120)
    ax_knee.set_xlabel("Time relative to final input frame (ms)")
    ax_knee.set_ylabel("Recorded knee angle (deg)")
    ax_knee.set_title("Kinematic history and forecast target", pad=3, fontweight="bold")
    clean_axes(ax_knee)
    panel_label(ax_knee, "B", x=-0.14)

    ax_flow = fig.add_subplot(grid[1, :])
    ax_flow.set_axis_off()
    panel_label(ax_flow, "C", x=-0.025, y=1.00)
    flow_box(ax_flow, 0.015, 0.52, 0.15, 0.32, "Knee history\n60 frames x 1", BLUE_PALE,
             edgecolor="#AFC4D5")
    flow_box(ax_flow, 0.015, 0.08, 0.15, 0.32, "sEMG history\n15 frames x 12", BLUE_PALE,
             edgecolor="#AFC4D5")
    flow_arrow(ax_flow, (0.17, 0.68), (0.23, 0.68))
    flow_arrow(ax_flow, (0.17, 0.24), (0.23, 0.24))
    flow_box(ax_flow, 0.235, 0.52, 0.17, 0.32, "Ridge\nkinematic forecast", WARM_PALE,
             edgecolor="#C8BEAE")
    flow_box(ax_flow, 0.235, 0.08, 0.17, 0.32, "Ridge\nresidual correction", GREEN_PALE,
             edgecolor="#AEC1AA")
    flow_arrow(ax_flow, (0.41, 0.68), (0.49, 0.68))
    flow_arrow(ax_flow, (0.41, 0.24), (0.49, 0.42))
    flow_box(ax_flow, 0.495, 0.40, 0.17, 0.44, "Base forecast\n+\nbounded sEMG\ncorrection",
             WARM_PALE, edgecolor="#C8BEAE", fontsize=6.6)
    flow_arrow(ax_flow, (0.67, 0.62), (0.75, 0.62))
    flow_box(ax_flow, 0.755, 0.43, 0.21, 0.38, "Predicted knee angle\n100 ms ahead",
             ORANGE_PALE, edgecolor="#D7B3A5", fontsize=7.0)
    ax_flow.text(0.58, 0.17, r"$\hat y_{fusion}=\hat y_{kin}+\gamma c_{EMG}$",
                 transform=ax_flow.transAxes, ha="center", fontsize=7.4, color=BLACK)

    return save_figure(fig, output_dir, "fig02_signal_model.pdf")


def figure_prediction_results(evidence_dir: Path, output_dir: Path) -> Path:
    predictions = load_predictions(evidence_dir)
    time, target, fused, no_emg = subject_trace(predictions)
    summary = read_json(resolve_data_path(evidence_dir, "prediction_confirmation/ablation_summary.json"))
    improvements = np.asarray(summary["improvement_deg"], dtype=np.float64)
    mean, low, high = mean_ci(improvements)

    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 67 * MM), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    ax.plot(time, target, color=BLACK, linewidth=1.25, label="Recorded")
    ax.plot(time, fused, color=BLUE, linewidth=1.05, label="Residual fusion")
    ax.plot(time, no_emg, color=ORANGE, linewidth=0.95, linestyle="--", label="Without sEMG")
    fused_rmse = float(np.sqrt(np.mean((fused - target) ** 2)))
    no_rmse = float(np.sqrt(np.mean((no_emg - target) ** 2)))
    ax.text(0.02, 0.04, f"Window RMSE: {fused_rmse:.2f} deg vs {no_rmse:.2f} deg",
            transform=ax.transAxes, fontsize=6.4, color=DARK)
    ax.set_ylim(-15, 82)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Knee angle (deg)")
    ax.set_title("Prespecified S067 simulation window", loc="left", fontweight="bold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        columnspacing=0.9,
        handlelength=1.5,
        borderaxespad=0.2,
    )
    clean_axes(ax)
    panel_label(ax, "A")

    ax = axes[1]
    ordered = np.sort(improvements)
    colors = np.where(ordered >= 0, BLUE, ORANGE)
    ax.bar(np.arange(1, len(ordered) + 1), ordered, color=colors, width=0.86, linewidth=0)
    ax.axhline(0, color=BLACK, linewidth=0.7)
    ax.axhspan(low, high, color=BLUE_PALE, zorder=0)
    ax.axhline(mean, color=BLUE, linewidth=1.15)
    ax.set_xlim(0, len(ordered) + 1)
    ax.set_xlabel("Confirmation participants (sorted)")
    ax.set_ylabel("RMSE without sEMG - fusion (deg)")
    ax.set_title("Participant-level effect", loc="left", fontweight="bold")
    top_band_note(
        ax,
        f"Mean {mean:.3f} deg\n95% CI {low:.3f} to {high:.3f}\n70/90 improved",
        band_fraction=0.33,
    )
    clean_axes(ax)
    panel_label(ax, "B")

    return save_figure(fig, output_dir, "fig03_prediction_results.pdf")


def figure_simulation_sequence(
    evidence_dir: Path, recordings_dir: Path, output_dir: Path
) -> Path:
    predictions = load_predictions(evidence_dir)
    time, _, fused, no_emg = subject_trace(predictions)
    peak_index = int(np.argmax(np.abs(fused - no_emg)))
    recording = find_recording(recordings_dir)
    capture = cv2.VideoCapture(str(recording))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Cannot recover frame rate from {recording}")
    frame_index = int(round(time[peak_index] * fps))
    frame, frame_time = read_video_frame(recording, frame_index)
    trace_index = int(round(frame_time * 100.0))
    command_difference = float(abs(fused[trace_index] - no_emg[trace_index]))
    panes = video_panes(frame)
    titles = ["Reference", "Residual fusion", "Without sEMG"]
    title_colors = [BLACK, BLUE, ORANGE]
    fig = plt.figure(figsize=(170 * MM, 99 * MM))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.08, 0.92], hspace=0.17, wspace=0.06)

    for column, (pane, title) in enumerate(zip(panes, titles, strict=True)):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(pane)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=3.0, fontsize=8.0, fontweight="bold", color=title_colors[column])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(title_colors[column])
            spine.set_linewidth(0.75)

    fusion_mask = magenta_leg_mask(panes[1])
    no_emg_mask = magenta_leg_mask(panes[2])
    union = fusion_mask | no_emg_mask
    rows, columns = np.where(union)
    if len(rows) == 0:
        raise RuntimeError("Could not recover right-leg masks from representative recording")
    x0 = max(0, int(columns.min()) - 38)
    x1 = min(panes[1].shape[1], int(columns.max()) + 39)
    y0 = max(0, int(rows.min()) - 24)
    y1 = min(panes[1].shape[0], int(rows.max()) + 25)
    crop = np.s_[y0:y1, x0:x1]

    for column, (pane, title, color) in enumerate(
        ((panes[1], "Fusion right leg", BLUE), (panes[2], "No-sEMG right leg", ORANGE))
    ):
        ax = fig.add_subplot(grid[1, column])
        ax.imshow(pane[crop])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=2.5, fontsize=7.2, fontweight="bold", color=color)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(0.8)

    ax = fig.add_subplot(grid[1, 2])
    reference_crop = cv2.cvtColor(panes[0][crop], cv2.COLOR_RGB2GRAY)
    ax.imshow(reference_crop, cmap="gray", vmin=0, vmax=255, alpha=0.58)
    fusion_crop = fusion_mask[crop].astype(float)
    no_emg_crop = no_emg_mask[crop].astype(float)
    ax.contourf(fusion_crop, levels=[0.5, 1.5], colors=[BLUE], alpha=0.22)
    ax.contourf(no_emg_crop, levels=[0.5, 1.5], colors=[ORANGE], alpha=0.22)
    ax.contour(fusion_crop, levels=[0.5], colors=[BLUE], linewidths=1.2)
    ax.contour(no_emg_crop, levels=[0.5], colors=[ORANGE], linewidths=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Aligned right-leg contours", pad=2.5, fontsize=7.2, fontweight="bold")
    ax.set_xlabel(
        f"t = {frame_time:.2f} s | commanded difference = {command_difference:.2f} deg",
        fontsize=6.2,
        labelpad=3.0,
    )
    for spine in ax.spines.values():
        spine.set_color(DARK)
        spine.set_linewidth(0.75)

    fig.subplots_adjust(left=0.015, right=0.995, top=0.93, bottom=0.075)
    return save_figure(fig, output_dir, "fig04_simulation_sequence.png")


def figure_motion_matching(evidence_dir: Path, output_dir: Path) -> Path:
    summary = read_json(resolve_data_path(evidence_dir, "physics/matching_summary.json"))
    windows = pd.DataFrame(summary["windows"])
    knee = windows["knee_rmse_deg"].to_numpy(dtype=float)
    thigh = windows["thigh_rms_deg"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 66 * MM), gridspec_kw={"wspace": 0.34})
    rng = np.random.default_rng(20260826)

    ax = axes[0]
    for k, t in zip(knee, thigh, strict=True):
        ax.plot([0, 1], [k, t], color=LIGHT, linewidth=0.45, alpha=0.55, zorder=1)
    ax.scatter(rng.normal(0, 0.025, len(knee)), knee, s=9, color=BLUE, alpha=0.72,
               edgecolor="white", linewidth=0.25, zorder=3)
    ax.scatter(rng.normal(1, 0.025, len(thigh)), thigh, s=9, color=GREEN, alpha=0.72,
               edgecolor="white", linewidth=0.25, zorder=3)
    ax.scatter([0, 1], [np.mean(knee), np.mean(thigh)], marker="D", s=26,
               color=[BLUE, GREEN], edgecolor=BLACK, linewidth=0.4, zorder=5)
    ax.set_xticks([0, 1], ["Knee RMSE", "Thigh pitch RMS"])
    ax.set_ylabel("Matching error (deg)")
    ax.set_title("Matched errors across 80 windows", loc="left", fontweight="bold")
    clean_axes(ax)
    panel_label(ax, "A")

    ax = axes[1]
    ordered = windows.sort_values("knee_rmse_deg").reset_index(drop=True)
    colors = np.where(ordered["query_id"].eq("S067_trial05_start0109"), ORANGE, BLUE)
    ax.bar(np.arange(1, len(ordered) + 1), ordered["knee_rmse_deg"],
           color=colors, width=0.86, linewidth=0)
    ax.axhline(float(np.mean(knee)), color=DARK, linestyle="--", linewidth=0.8)
    example_index = int(np.flatnonzero(ordered["query_id"].eq("S067_trial05_start0109"))[0])
    example = ordered.iloc[example_index]
    ax.annotate(
        f"S067 example\n{example.knee_rmse_deg:.2f} deg",
        (example_index + 1, example.knee_rmse_deg),
        xytext=(example_index + 8, example.knee_rmse_deg + 3.0),
        fontsize=6.3,
        arrowprops={"arrowstyle": "-", "color": ORANGE, "linewidth": 0.7},
    )
    ax.set_xlim(0, len(ordered) + 1)
    ax.set_xlabel("Simulation windows (sorted)")
    ax.set_ylabel("Knee matching RMSE (deg)")
    ax.set_title("Knee-match distribution", loc="left", fontweight="bold")
    top_band_note(
        ax,
        f"Mean {np.mean(knee):.2f} deg\n29 unique snippets",
        band_fraction=0.23,
    )
    clean_axes(ax)
    panel_label(ax, "B")

    return save_figure(fig, output_dir, "fig05_motion_matching.pdf")


def figure_xcom_concept(output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 65 * MM), gridspec_kw={"wspace": 0.22})

    ax = axes[0]
    ax.set_axis_off()
    panel_label(ax, "A", x=-0.03, y=0.98)
    ax.plot([0.08, 0.92], [0.13, 0.13], color=DARK, linewidth=1.0)
    ax.add_patch(Rectangle((0.40, 0.105), 0.18, 0.05, facecolor=WARM_PALE,
                           edgecolor=DARK, linewidth=0.7))
    ax.plot([0.49, 0.45], [0.15, 0.63], color=DARK, linewidth=1.2)
    ax.add_patch(Circle((0.45, 0.63), 0.026, facecolor=BLACK, edgecolor="none"))
    ax.text(0.42, 0.69, "CoM", ha="center", fontsize=7.0)
    ax.annotate("", xy=(0.67, 0.63), xytext=(0.47, 0.63),
                arrowprops={"arrowstyle": "-|>", "color": ORANGE, "linewidth": 1.2})
    ax.text(0.57, 0.68, "velocity", ha="center", fontsize=6.6, color=ORANGE)
    ax.plot([0.67, 0.67], [0.13, 0.63], color=BLUE, linewidth=0.8, linestyle="--")
    ax.add_patch(Circle((0.67, 0.63), 0.028, facecolor=BLUE, edgecolor="none"))
    ax.text(0.70, 0.69, "XCoM", ha="left", fontsize=7.0, color=BLUE)
    ax.text(0.50, 0.92, r"$XCoM=CoM+v_{CoM}/\sqrt{g/l}$", ha="center", fontsize=8.1)
    ax.text(0.50, 0.02, "Side view", ha="center", fontsize=6.5, color=MID)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.set_axis_off()
    panel_label(ax, "B", x=-0.03, y=0.98)
    foot_left = Rectangle((0.20, 0.26), 0.16, 0.48, angle=7, facecolor=WARM_PALE,
                          edgecolor=DARK, linewidth=0.8)
    foot_right = Rectangle((0.62, 0.26), 0.16, 0.48, angle=-7, facecolor=WARM_PALE,
                           edgecolor=DARK, linewidth=0.8)
    ax.add_patch(foot_left)
    ax.add_patch(foot_right)
    hull = Polygon([[0.19, 0.24], [0.78, 0.24], [0.80, 0.76], [0.22, 0.77]],
                   closed=True, facecolor=BLUE_PALE, edgecolor=BLUE, linewidth=0.9,
                   alpha=0.75)
    ax.add_patch(hull)
    ax.add_patch(Circle((0.47, 0.49), 0.026, facecolor=BLACK, edgecolor="none"))
    ax.text(0.47, 0.43, "CoM", ha="center", fontsize=6.5)
    ax.add_patch(Circle((0.69, 0.54), 0.029, facecolor=BLUE, edgecolor="none"))
    ax.text(0.69, 0.60, "XCoM", ha="center", fontsize=6.7, color=BLUE)
    ax.annotate("", xy=(0.79, 0.54), xytext=(0.72, 0.54),
                arrowprops={"arrowstyle": "<->", "color": ORANGE, "linewidth": 1.0})
    ax.text(0.755, 0.58, "margin", ha="center", fontsize=6.4, color=ORANGE)
    ax.text(0.50, 0.89, "Support polygon and signed XCoM margin", ha="center",
            fontsize=8.0, fontweight="bold")
    ax.text(0.50, 0.08, "Top view", ha="center", fontsize=6.5, color=MID)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return save_figure(fig, output_dir, "fig06_xcom_concept.pdf")


def figure_simulation_outcomes(evidence_dir: Path, output_dir: Path) -> Path:
    participants = pd.read_csv(resolve_data_path(evidence_dir, "analysis/participant_primary.csv"))
    difference = (
        participants["no_emg_excess_auc"].to_numpy(dtype=float)
        - participants["fused_excess_auc"].to_numpy(dtype=float)
    )
    mean, low, high = mean_ci(difference)
    rng = np.random.default_rng(20260826)

    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 67 * MM), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    values = [
        participants["reference_auc"].to_numpy(dtype=float),
        participants["fused_auc"].to_numpy(dtype=float),
        participants["no_emg_auc"].to_numpy(dtype=float),
    ]
    colors = [MID, BLUE, ORANGE]
    positions = [1, 2, 3]
    box = ax.boxplot(values, positions=positions, widths=0.48, patch_artist=True,
                     showfliers=False, medianprops={"color": BLACK, "linewidth": 1.0},
                     whiskerprops={"color": DARK, "linewidth": 0.7},
                     capprops={"color": DARK, "linewidth": 0.7})
    for patch, color in zip(box["boxes"], colors, strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    for position, value, color in zip(positions, values, colors, strict=True):
        ax.scatter(rng.normal(position, 0.055, len(value)), value, s=8.5, color=color,
                   alpha=0.62, edgecolor="none", zorder=3)
        ax.scatter([position], [np.mean(value)], marker="D", s=27, color=color,
                   edgecolor=BLACK, linewidth=0.4, zorder=5)
    ax.set_xticks(positions, ["Reference", "Residual\nfusion", "Without\nsEMG"])
    ax.set_ylabel("Instability AUC (score-s)")
    ax.set_title("Paired simulation conditions", loc="left", fontweight="bold")
    clean_axes(ax)
    panel_label(ax, "A")

    ax = axes[1]
    ordered = np.sort(difference)
    colors = np.where(ordered >= 0, BLUE, ORANGE)
    ax.bar(np.arange(1, len(ordered) + 1), ordered, color=colors, width=0.86, linewidth=0)
    ax.axhline(0, color=BLACK, linewidth=0.7)
    ax.axhspan(low, high, color=BLUE_PALE, zorder=0)
    ax.axhline(mean, color=BLUE, linewidth=1.15)
    ax.set_xlim(0, len(ordered) + 1)
    ax.set_xlabel("Participants (sorted)")
    ax.set_ylabel("No-sEMG - fusion excess AUC (score-s)")
    ax.set_title("Physical effect by participant", loc="left", fontweight="bold")
    top_band_note(
        ax,
        f"Mean {mean:.4f} score-s\n95% CI {low:.4f} to {high:.4f}\np = 0.801",
        band_fraction=0.34,
    )
    clean_axes(ax)
    panel_label(ax, "B")

    return save_figure(fig, output_dir, "fig07_simulation_outcomes.pdf")


def fit_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str = ORANGE) -> None:
    if len(x) < 2 or np.allclose(x, x[0]):
        return
    domain = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(domain, slope * domain + intercept, color=color, linewidth=1.0, zorder=4)


def figure_error_instability(evidence_dir: Path, output_dir: Path) -> Path:
    participants = pd.read_csv(resolve_data_path(evidence_dir, "analysis/participant_primary.csv"))
    summary = read_json(resolve_data_path(evidence_dir, "analysis/statistical_summary.json"))
    association = summary["rmse_vs_physical_outcome"]

    x1 = participants["source_rmse_fused_deg"].to_numpy(dtype=float)
    x2 = participants["match_knee_rmse_deg"].to_numpy(dtype=float)
    y = participants["fused_excess_auc"].to_numpy(dtype=float)
    xr = np.asarray(association["x_residual"], dtype=float)
    yr = np.asarray(association["y_residual"], dtype=float)
    rho2, p2 = stats.spearmanr(x2, y)

    fig, axes = plt.subplots(1, 3, figsize=(170 * MM, 55 * MM), gridspec_kw={"wspace": 0.38})

    panels = [
        (x1, y, "Prediction RMSE (deg)", "Excess AUC (score-s)",
         f"rho = {association['raw_spearman_rho']:.3f}\np = {association['raw_spearman_p_two_sided']:.3f}"),
        (x2, y, "Knee matching RMSE (deg)", "", f"rho = {rho2:.3f}\np = {p2:.3f}"),
        (xr, yr, "Residualized RMSE rank", "Residualized excess-AUC rank",
         f"partial rho = {association['rho']:.3f}\n95% CI {association['bootstrap_95_ci'][0]:.3f} to {association['bootstrap_95_ci'][1]:.3f}"),
    ]
    for index, (ax, (x, outcome, xlabel, ylabel, annotation)) in enumerate(zip(axes, panels, strict=True)):
        ax.scatter(x, outcome, s=13, color=BLUE, alpha=0.72, edgecolor="white",
                   linewidth=0.3, zorder=3)
        fit_line(ax, x, outcome)
        header_note(ax, annotation)
        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        clean_axes(ax, grid="both")
        panel_label(ax, chr(ord("A") + index), x=-0.16)

    return save_figure(fig, output_dir, "fig08_error_instability.pdf")


def main() -> None:
    args = parse_args()
    configure_style()
    paths = [
        figure_experiment_overview(args.evidence_dir, args.recordings_dir, args.output_dir),
        figure_signal_model(args.evidence_dir, args.output_dir),
        figure_prediction_results(args.evidence_dir, args.output_dir),
        figure_simulation_sequence(args.evidence_dir, args.recordings_dir, args.output_dir),
        figure_motion_matching(args.evidence_dir, args.output_dir),
        figure_xcom_concept(args.output_dir),
        figure_simulation_outcomes(args.evidence_dir, args.output_dir),
        figure_error_instability(args.evidence_dir, args.output_dir),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
