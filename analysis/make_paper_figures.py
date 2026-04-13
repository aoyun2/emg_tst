"""
Paper figure generator.

Run:
    python -m analysis.make_paper_figures
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image
from scipy import stats
import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import visualtorch

from emg_tst.model import CnnBiLstmLastStep

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUN = REPO_ROOT / "checkpoints" / "tst_20260405_173725_all"
SIM_RUN = REPO_ROOT / "artifacts" / "phys_eval_v2" / "runs" / "20260406_205003"
OUT_DIR = REPO_ROOT / "figures" / "paper_native"

REF_COL = "#5a5a5a"
PRED_COL = "#c44e3b"
BLUE_COL = "#3b78a8"
INK = "#202433"
MUTED = "#5b6470"
GRID = "#d9dde3"
BOX = "#f6f7f9"
WHITE = "#ffffff"
DPI = 300
BANK_PATH = REPO_ROOT / "artifacts" / "phys_eval_v2" / "reference_bank" / "mocapact_expert_snippets_right.npz"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.edgecolor": "#a7adb5",
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "savefig.dpi": DPI,
        }
    )


def _train_df() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for p in sorted(TRAIN_RUN.glob("fold_*/metrics.json")):
        m = json.loads(p.read_text())
        m["fold"] = p.parent.name
        rows.append(m)
    if not rows:
        raise RuntimeError(f"No fold metrics found under {TRAIN_RUN}")
    return pd.DataFrame(rows)


def _sim_df() -> tuple[dict[str, Any], pd.DataFrame]:
    summary = json.loads((SIM_RUN / "summary.json").read_text())
    rows: list[dict[str, Any]] = []
    for r in summary["results"]:
        rows.append(
            {
                "query_id": r["query_id"],
                "pred_rmse": r["model"]["pred_vs_gt_knee_flex_rmse_deg"],
                "match_knee_rmse": r["match"]["rmse_knee_deg"],
                "ref_auc": r["sim"]["ref"]["instability_auc"],
                "pred_auc": r["sim"]["pred"]["instability_auc"],
                "excess_auc": r["sim"]["excess"]["instability_auc_delta"],
                "compare_npz": r["artifacts"]["compare_npz"],
            }
        )
    return summary, pd.DataFrame(rows)


def _partial_df() -> pd.DataFrame:
    return pd.read_csv(SIM_RUN / "analysis" / "partial_spearman_trials.csv")


def _partial_summary() -> dict[str, Any]:
    return json.loads((SIM_RUN / "analysis" / "partial_spearman_summary.json").read_text())


def _load_npz(sim_df: pd.DataFrame, trial_idx: int) -> dict[str, Any]:
    npz = os.path.normpath(str(sim_df.iloc[trial_idx]["compare_npz"]))
    return dict(np.load(npz))


def _eval_trial_paths(sim_df: pd.DataFrame, trial_idx: int) -> tuple[Path, Path, Path]:
    npz = Path(os.path.normpath(str(sim_df.iloc[trial_idx]["compare_npz"])))
    replay_dir = npz.parent
    eval_dir = replay_dir.parent
    return eval_dir, eval_dir / "summary.json", eval_dir / "query_window.npz"


def _grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _rho_text(rho: float, p: float) -> str:
    return f"rho = {rho:+.3f}\np = {p:.3f}"


def _gt_demo_record() -> dict[str, Any]:
    p = sorted(REPO_ROOT.glob("gt_data*.npy"))[0]
    return np.load(p, allow_pickle=True).item()


def _gt_emg_thumb(ax: plt.Axes) -> None:
    rec = _gt_demo_record()
    emg = np.asarray(rec["raw_emg_channels"], dtype=float)[:, :4000]
    t = np.arange(emg.shape[1]) / 2000.0
    win = 40
    env = []
    kernel = np.ones(win, dtype=float) / float(win)
    for ch in emg:
        z = np.convolve(np.abs(ch), kernel, mode="same")
        z = (z - np.percentile(z, 5)) / max(1e-6, np.percentile(z, 95) - np.percentile(z, 5))
        env.append(np.clip(z, 0.0, 1.2))
    env = np.asarray(env)
    colors = ["#3b78a8", "#5f9fd0", "#7eb5d8", "#a7cae6"]
    offsets = np.array([2.7, 1.8, 0.9, 0.0], dtype=float)
    for i in range(env.shape[0]):
        ax.plot(t, env[i] * 0.65 + offsets[i], color=colors[i], linewidth=1.0)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(-0.15, 3.55)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.01, 0.96, "EMG envelopes", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=MUTED)
    ax.text(0.01, 0.08, "2.0 s window", transform=ax.transAxes, ha="left", va="bottom", fontsize=7.0, color=MUTED)


def _representative_xcom_thumb(ax: plt.Axes, sim_df: pd.DataFrame, trial_idx: int = 75) -> None:
    d = _load_npz(sim_df, trial_idx)
    dt = float(d["dt"])
    t = np.arange(len(d["balance_xcom_margin_ref_m"])) * dt
    ax.axhline(0.0, color="#9aa3ad", linewidth=0.8)
    ax.plot(t, d["balance_xcom_margin_ref_m"], color=REF_COL, linewidth=1.1)
    ax.plot(t, d["balance_xcom_margin_good_m"], color=PRED_COL, linewidth=1.1)
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.margins(x=0.02, y=0.18)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.01, 0.96, "XCoM margin", transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=MUTED)


def _prediction_thumb(ax: plt.Axes) -> None:
    model = _build_cnn_bilstm()
    img = visualtorch.layered_view(
        model,
        input_shape=(1, 400, 10),
        legend=False,
        draw_volume=True,
        one_dim_orientation="z",
        spacing=18,
        padding=12,
        shade_step=12,
        scale_xy=1.0,
        scale_z=0.16,
        background_fill="white",
    )
    arr = np.asarray(img.convert("RGB"))
    arr = _trim_white(arr)
    ax.imshow(arr, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.02, 0.96, "Conv1d x2  →  BiLSTM x2  →  MLP", transform=ax.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=INK)
    ax.text(0.02, 0.03, "Input: 400 timesteps × 10 channels    Output: knee angle at t + 10 ms", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color=MUTED)


def _prediction_thumb_image() -> np.ndarray:
    model = _build_cnn_bilstm()
    img = visualtorch.layered_view(
        model,
        input_shape=(1, 400, 10),
        legend=False,
        draw_volume=True,
        one_dim_orientation="z",
        spacing=14,
        padding=10,
        shade_step=12,
        scale_xy=0.92,
        scale_z=0.14,
        background_fill="white",
    )
    arr = np.asarray(img.convert("RGB"))
    return _trim_white(arr)


def _motion_match_thumb(ax: plt.Axes, sim_df: pd.DataFrame, trial_idx: int = 75) -> None:
    eval_dir, summary_path, query_path = _eval_trial_paths(sim_df, trial_idx)
    summary = json.loads(summary_path.read_text())
    match = summary["match"]
    query = np.load(query_path, allow_pickle=True)
    bank = np.load(BANK_PATH, allow_pickle=True)

    snippet_ids = list(bank["snippet_id"])
    bank_idx = snippet_ids.index(match["snippet_id"])
    start = int(match["start_step_in_snip"])
    length = int(match["L"])

    ref_thigh = np.asarray(bank["thigh_pitch_deg"][bank_idx], dtype=float)[start : start + length]
    ref_knee = np.asarray(bank["knee_deg"][bank_idx], dtype=float)[start : start + length]
    q_thigh = np.asarray(query["thigh_pitch_deg"], dtype=float)
    q_knee = np.asarray(query["knee_included_deg"], dtype=float)

    x_src = np.arange(q_thigh.size, dtype=float)
    x_dst = np.linspace(0.0, float(q_thigh.size - 1), num=length)
    q_thigh_r = np.interp(x_dst, x_src, q_thigh)
    q_knee_r = np.interp(x_dst, x_src, q_knee)
    t = np.arange(length, dtype=float) / 33.0

    ax_top = ax.inset_axes([0.04, 0.52, 0.92, 0.40])
    ax_bot = ax.inset_axes([0.04, 0.08, 0.92, 0.32])
    for a in (ax_top, ax_bot):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_color("#b9c0c9")
        a.spines["bottom"].set_color("#b9c0c9")
        a.tick_params(labelsize=4.8, length=1.8, pad=1.0)
        a.grid(True, axis="both", color=GRID, linewidth=0.45)
        a.set_axisbelow(True)
    ax_top.plot(t, ref_thigh, color=REF_COL, linewidth=0.9, label="ref")
    ax_top.plot(t, q_thigh_r, color=PRED_COL, linewidth=0.9, label="query")
    ax_top.set_ylabel("Thigh", fontsize=4.8)
    ax_top.set_xticklabels([])
    ax_top.legend(loc="upper right", fontsize=4.4, frameon=False, handlelength=1.8)
    ax_bot.plot(t, ref_knee, color=REF_COL, linewidth=0.9)
    ax_bot.plot(t, q_knee_r, color=PRED_COL, linewidth=0.9)
    ax_bot.set_ylabel("Knee", fontsize=4.8)
    ax_bot.set_xlabel("Matched time (s)", fontsize=4.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _physics_thumb(ax: plt.Axes, sim_df: pd.DataFrame, trial_idx: int = 75) -> None:
    d = _load_npz(sim_df, trial_idx)
    frames = np.asarray(d["frames"])
    frame = frames[max(0, int(frames.shape[0] // 2))]
    ax.imshow(frame, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    inset = ax.inset_axes([0.05, 0.06, 0.90, 0.24])
    dt = float(d["dt"])
    t = np.arange(len(d["balance_xcom_margin_ref_m"])) * dt
    inset.axhline(0.0, color="#9aa3ad", linewidth=0.7)
    inset.plot(t, d["balance_xcom_margin_ref_m"], color=REF_COL, linewidth=0.9)
    inset.plot(t, d["balance_xcom_margin_good_m"], color=PRED_COL, linewidth=0.9)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor((1, 1, 1, 0.86))
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.text(0.01, 0.92, "XCoM margin", transform=inset.transAxes, ha="left", va="top", fontsize=6.8, color=MUTED)


def _trim_white(arr: np.ndarray, thresh: int = 248) -> np.ndarray:
    if arr.ndim != 3:
        return arr
    mask = np.any(arr < thresh, axis=2)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return arr
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return arr[y0:y1, x0:x1, :]


def _build_cnn_bilstm() -> CnnBiLstmLastStep:
    return CnnBiLstmLastStep(
        n_emg_vars=4,
        n_imu_vars=6,
        seq_len=400,
        stem_width=32,
        hidden_size=64,
        n_layers=2,
        kernel_size=5,
        depth=2,
        dropout=0.10,
    )


def _traced_cnn_bilstm_nodes() -> list[dict[str, Any]]:
    class _Wrapped(nn.Module):
        def __init__(self, model: CnnBiLstmLastStep):
            super().__init__()
            self.conv = model.conv
            self.rnn = model.rnn
            self.head = model.head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.conv(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            z, _ = self.rnn(z)
            return self.head(z[:, -1, :])[:, 0]

    gm = symbolic_trace(_Wrapped(_build_cnn_bilstm()))
    ShapeProp(gm).propagate(torch.zeros(1, 400, 10))

    summary: dict[str, tuple[int, ...] | None] = {}
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            summary["Input"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
        elif n.op == "call_method" and n.name == "transpose":
            summary["Permute to [B,C,T]"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
        elif n.op == "call_module" and str(n.target) == "conv.0":
            summary["Conv1d 10→32\nkernel=5"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
        elif n.op == "call_module" and str(n.target) == "conv.2":
            summary["Conv1d 32→32\nkernel=5"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
        elif n.op == "call_function" and n.name == "getitem":
            summary["BiLSTM x2\nhidden=64 / dir"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
        elif n.op == "call_function" and n.name == "getitem_2":
            summary["Last time step"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
        elif n.op == "call_module" and str(n.target) == "head.0":
            summary["Linear 128→64\n+ GELU + Dropout"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
        elif n.op == "call_module" and str(n.target) == "head.3":
            summary["Linear 64→1"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
        elif n.op == "output":
            summary["Output"] = tuple(int(s) for s in n.meta["tensor_meta"].shape)
    nodes = [
        {"label": "Input", "shape": summary.get("Input"), "kind": "input"},
        {"label": "Permute to [B,C,T]", "shape": summary.get("Permute to [B,C,T]"), "kind": "op"},
        {"label": "Conv1d 10→32\nkernel=5", "shape": summary.get("Conv1d 10→32\nkernel=5"), "kind": "conv"},
        {"label": "Conv1d 32→32\nkernel=5", "shape": summary.get("Conv1d 32→32\nkernel=5"), "kind": "conv"},
        {"label": "BiLSTM x2\nhidden=64 / dir", "shape": summary.get("BiLSTM x2\nhidden=64 / dir"), "kind": "rnn"},
        {"label": "Last time step", "shape": summary.get("Last time step"), "kind": "op"},
        {"label": "Linear 128→64\n+ GELU + Dropout", "shape": summary.get("Linear 128→64\n+ GELU + Dropout"), "kind": "linear"},
        {"label": "Linear 64→1", "shape": summary.get("Linear 64→1"), "kind": "linear"},
        {"label": "Output", "shape": summary.get("Output"), "kind": "output"},
    ]
    return nodes


def fig1_pipeline() -> str:
    _, sim = _sim_df()
    eval_dir, _, _ = _eval_trial_paths(sim, 75)
    motion_plot = np.asarray(Image.open(eval_dir / "plots" / "motion_match.png").convert("RGB"))
    motion_crop = motion_plot[int(motion_plot.shape[0] * 0.09) : int(motion_plot.shape[0] * 0.58), :, :]
    d = _load_npz(sim, 75)
    dt = float(d["dt"])
    pred = np.asarray(d["knee_good_query_deg"], dtype=float)
    realized = np.asarray(d["knee_good_actual_deg"], dtype=float)
    n_trace = min(len(pred), len(realized))
    pred = pred[:n_trace]
    realized = realized[:n_trace]
    t = np.arange(n_trace) * dt
    fig = plt.figure(figsize=(15.6, 4.7))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[0.95, 1.35, 1.38, 1.05],
        left=0.035,
        right=0.985,
        top=0.90,
        bottom=0.14,
        wspace=0.35,
    )

    ax_in = fig.add_subplot(gs[0, 0])
    ax_arch = fig.add_subplot(gs[0, 1])
    gs_pred_match = gs[0, 2].subgridspec(2, 1, height_ratios=[0.44, 0.56], hspace=0.18)
    ax_out = fig.add_subplot(gs_pred_match[0, 0])
    ax_match = fig.add_subplot(gs_pred_match[1, 0])
    ax_sim = fig.add_subplot(gs[0, 3])

    rec = _gt_demo_record()
    emg = np.asarray(rec["raw_emg_channels"], dtype=float)[:, :4000]
    kernel = np.ones(40, dtype=float) / 40.0
    colors = ["#2d6ea2", "#4a8fc6", "#73addb", "#9fcaea"]
    t_raw = np.arange(emg.shape[1]) / 2000.0
    offsets = [2.85, 1.95, 1.05, 0.15]
    for idx, ch in enumerate(emg):
        z = np.convolve(np.abs(ch), kernel, mode="same")
        z = (z - np.percentile(z, 5)) / max(1e-6, np.percentile(z, 95) - np.percentile(z, 5))
        ax_in.plot(t_raw, np.clip(z, 0.0, 1.0) * 0.65 + offsets[idx], color=colors[idx], linewidth=1.0)
    ax_in.set_xlim(0, 2.0)
    ax_in.set_ylim(-0.1, 3.8)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    for spine in ax_in.spines.values():
        spine.set_color("#c9cfd8")
        spine.set_linewidth(1.0)
    ax_in.text(0.00, 1.05, "Wearable input window", transform=ax_in.transAxes, ha="left", va="bottom", fontsize=10.0, fontweight="bold", color=INK)
    ax_in.text(0.00, 0.995, "4 EMG envelopes + 6 thigh-IMU channels", transform=ax_in.transAxes, ha="left", va="top", fontsize=7.2, color=MUTED)
    ax_in.text(0.00, -0.07, "[400 × 10]", transform=ax_in.transAxes, ha="left", va="top", fontsize=7.2, color=MUTED)

    arch_img = _prediction_thumb_image()
    ax_arch.imshow(arch_img, aspect="auto")
    ax_arch.set_xticks([])
    ax_arch.set_yticks([])
    for spine in ax_arch.spines.values():
        spine.set_color("#c9cfd8")
        spine.set_linewidth(1.0)
    ax_arch.text(0.00, 1.05, "CNN-BiLSTM prediction model", transform=ax_arch.transAxes, ha="left", va="bottom", fontsize=10.0, fontweight="bold", color=INK)
    ax_arch.text(0.00, 0.995, "Conv1d → Conv1d → BiLSTM → MLP", transform=ax_arch.transAxes, ha="left", va="top", fontsize=7.2, color=MUTED)
    ax_arch.text(0.00, -0.07, "[400 × 32] → [400 × 32] → [400 × 128] → [128 → 64 → 1]", transform=ax_arch.transAxes, ha="left", va="top", fontsize=7.0, color=MUTED)

    ax_out.plot(t, pred, color=PRED_COL, linewidth=1.25, label="predicted target")
    ax_out.plot(t, realized, color=REF_COL, linewidth=1.0, linestyle="--", label="realized knee")
    ax_out.set_xlim(float(t[0]), float(t[-1]))
    ax_out.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_out.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_out.tick_params(axis="both", labelsize=7)
    ax_out.set_xlabel("Time (s)", fontsize=7.2)
    ax_out.set_ylabel("Knee angle (deg)", fontsize=7.2)
    ax_out.legend(loc="upper right", fontsize=6.2, frameon=False)
    _grid(ax_out, "both")
    ax_out.text(0.00, 1.08, "Predicted knee trajectory", transform=ax_out.transAxes, ha="left", va="bottom", fontsize=9.8, fontweight="bold", color=INK)
    ax_out.text(0.00, 1.01, "Forecast used as the query target", transform=ax_out.transAxes, ha="left", va="top", fontsize=7.1, color=MUTED)

    ax_match.imshow(motion_crop, aspect="auto")
    ax_match.set_xticks([])
    ax_match.set_yticks([])
    for spine in ax_match.spines.values():
        spine.set_color("#c9cfd8")
        spine.set_linewidth(1.0)
    ax_match.text(0.00, 1.08, "Motion matching", transform=ax_match.transAxes, ha="left", va="bottom", fontsize=9.8, fontweight="bold", color=INK)
    ax_match.text(0.00, 1.01, "Closest MoCapAct clip selected before simulation", transform=ax_match.transAxes, ha="left", va="top", fontsize=7.1, color=MUTED)

    _physics_thumb(ax_sim, sim, 75)
    ax_sim.text(0.00, 1.05, "Paired physics evaluation", transform=ax_sim.transAxes, ha="left", va="bottom", fontsize=10.0, fontweight="bold", color=INK)
    ax_sim.text(0.00, 0.995, "REF vs PRED scored with excess XCoM instability AUC", transform=ax_sim.transAxes, ha="left", va="top", fontsize=7.1, color=MUTED)

    def _fig_arrow(ax_from: plt.Axes, ax_to: plt.Axes, y_frac_from: float = 0.5, y_frac_to: float = 0.5) -> None:
        bb0 = ax_from.get_position()
        bb1 = ax_to.get_position()
        x0 = bb0.x1 + 0.010
        x1 = bb1.x0 - 0.010
        y0 = bb0.y0 + bb0.height * y_frac_from
        y1 = bb1.y0 + bb1.height * y_frac_to
        fig.add_artist(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.25,
                color="#8f99a5",
            )
        )

    _fig_arrow(ax_in, ax_arch, 0.46, 0.46)
    _fig_arrow(ax_arch, ax_out, 0.46, 0.46)
    _fig_arrow(ax_out, ax_match, 0.42, 0.50)
    _fig_arrow(ax_match, ax_sim, 0.50, 0.50)

    out = OUT_DIR / "fig1_pipeline.png"
    fig.savefig(out, dpi=300, bbox_inches=None, pad_inches=0.05)
    plt.close(fig)
    return str(out)


def fig1_pipeline_clean() -> str:
    """Clean 5-card pipeline schematic — single axis, no embedded images."""
    CARD_BG = ["#dbeafe", "#ede9fe", "#d1fae5", "#fef9c3", "#fee2e2"]
    CARD_BORDER = ["#93c5fd", "#a78bfa", "#6ee7b7", "#fcd34d", "#fca5a5"]
    TITLES = [
        "1  Wearable input",
        "2  CNN-BiLSTM",
        "3  Rolling forecast",
        "4  Motion matching",
        "5  Physics evaluation",
    ]
    CONTENT = [
        ["4 EMG envelopes", "3-axis accel + gyro", "400 samples (2.0 s)", "200 Hz sampling"],
        ["Conv1d x2 (k=5, 32 ch)", "Bi-LSTM x2 (64 hidden)", "Linear readout", "~260 k parameters"],
        ["Causal sliding window", "10 ms lookahead", "Predicted knee angle", "trajectory (deg)"],
        ["Query MoCapAct bank", "Nearest clip by knee +", "thigh orientation", "match RMSE < 25 deg"],
        ["REF: expert unmodified", "PRED: knee overridden", "Paired MuJoCo rollout", "-> Excess AUC"],
    ]
    XCS = [10.0, 30.0, 50.0, 70.0, 90.0]
    HALF_W = 8.5
    Y_BOT, Y_TOP = 3.0, 37.0
    BOX_H = Y_TOP - Y_BOT
    MID_Y = (Y_BOT + Y_TOP) / 2

    fig, ax = plt.subplots(figsize=(13.0, 3.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    for i, (xc, title, lines, bg, border) in enumerate(
        zip(XCS, TITLES, CONTENT, CARD_BG, CARD_BORDER)
    ):
        x = xc - HALF_W
        w = 2 * HALF_W

        # Card background
        ax.add_patch(FancyBboxPatch(
            (x, Y_BOT), w, BOX_H,
            boxstyle="round,pad=0.5",
            facecolor=bg, edgecolor=border,
            linewidth=1.3, zorder=2,
        ))

        # Separator under title
        sep_y = Y_TOP - 7.8
        ax.plot([x + 1.0, x + w - 1.0], [sep_y, sep_y],
                color=border, linewidth=0.9, zorder=3)

        # Title (above separator)
        ax.text(xc, Y_TOP - 1.2, title,
                ha="center", va="top",
                fontsize=8.8, fontweight="bold", color="#1e293b",
                zorder=4)

        # Content lines (below separator, evenly spaced)
        body_top = sep_y - 2.5
        n = len(lines)
        body_h = sep_y - Y_BOT - 4.0
        step = body_h / max(n, 1)
        for j, line in enumerate(lines):
            ax.text(xc, body_top - j * step, line,
                    ha="center", va="top",
                    fontsize=7.5, color="#334155",
                    zorder=4)

        # Arrow to next card
        if i < len(XCS) - 1:
            ax.annotate(
                "",
                xy=(xc + HALF_W + 2.6, MID_Y),
                xytext=(xc + HALF_W + 0.4, MID_Y),
                arrowprops=dict(
                    arrowstyle="-|>", color="#64748b",
                    lw=1.4, mutation_scale=14,
                ),
                zorder=5,
            )

    fig.suptitle(
        "Figure 1.  End-to-end evaluation pipeline",
        fontsize=10.5, fontweight="bold", y=0.98, color=INK,
    )

    out = OUT_DIR / "fig1_pipeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return str(out)


def fig2_representative_trial(sim_df: pd.DataFrame, trial_idx: int = 75) -> str:
    d = _load_npz(sim_df, trial_idx)
    dt = float(d["dt"])
    n = len(d["predicted_fall_risk_trace_ref"])
    t = np.arange(n) * dt

    xcom_ref = d["balance_xcom_margin_ref_m"]
    xcom_pred = d["balance_xcom_margin_good_m"]
    risk_ref = d["predicted_fall_risk_trace_ref"]
    risk_pred = d["predicted_fall_risk_trace_good"]
    excess = float(sim_df.iloc[trial_idx]["excess_auc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.42})

    # Panel A — XCoM margin
    ax1.axhline(0, color="#9aa3ad", linewidth=0.8, linestyle="--", zorder=1)
    ax1.fill_between(t, xcom_pred, 0, where=(xcom_pred < 0), color=PRED_COL, alpha=0.12, zorder=2)
    ax1.plot(t, xcom_ref,  color=REF_COL,  linewidth=1.5, label="REF",  zorder=3)
    ax1.plot(t, xcom_pred, color=PRED_COL, linewidth=1.5, label="PRED", zorder=4)
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("XCoM margin (m)", fontsize=9)
    ax1.legend(loc="lower left", fontsize=8, frameon=False)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _grid(ax1, "y")
    ax1.text(-0.14, 1.03, "A", transform=ax1.transAxes, fontsize=10, fontweight="bold", va="top", color=INK)

    # Panel B — per-step risk score
    ax2.fill_between(t, risk_pred, color=PRED_COL, alpha=0.12, zorder=2)
    ax2.plot(t, risk_ref,  color=REF_COL,  linewidth=1.5, label="REF",  zorder=3)
    ax2.plot(t, risk_pred, color=PRED_COL, linewidth=1.5, label="PRED", zorder=4)
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylabel("Instability score", fontsize=9)
    ax2.set_ylim(-0.02, 1.08)
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    # Mid-right: gap between PRED (≈1) and REF (≈0) after t≈0.8 s is clear
    ax2.text(0.97, 0.52, f"Excess AUC = {excess:.3f}",
             transform=ax2.transAxes, ha="right", va="center", fontsize=8, color=INK,
             bbox=dict(fc="white", ec="none", pad=1.5))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    _grid(ax2, "y")
    ax2.text(-0.14, 1.03, "B", transform=ax2.transAxes, fontsize=10, fontweight="bold", va="top", color=INK)

    out = OUT_DIR / "fig2_representative_trial.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return str(out)


def fig3_prediction(train_df: pd.DataFrame) -> str:
    df = train_df.copy().sort_values("test_rmse")
    rmse = np.asarray(df["test_rmse"].to_numpy(), dtype=float)
    x = np.arange(1, rmse.size + 1, dtype=int)
    mean_v = float(np.mean(rmse))
    med_v = float(np.median(rmse))

    bar_colors = [PRED_COL if v > 10.0 else BLUE_COL for v in rmse]

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(x, rmse, width=0.82, color=bar_colors, edgecolor=WHITE, linewidth=0.2, alpha=0.88, zorder=3)
    ax.axhline(mean_v, color="#475569", linewidth=1.0, linestyle="--", zorder=5)
    ax.axhline(med_v,  color="#475569", linewidth=1.0, linestyle=":",  zorder=5)
    # Place labels on the left where bars are short (≈4°), well below the lines
    _tb = dict(fc="white", ec="none", pad=1.5)
    ax.text(2.5, mean_v + 0.45, f"Mean {mean_v:.1f}\u00b0",
            ha="left", va="bottom", fontsize=7.5, color="#475569", bbox=_tb, zorder=6)
    ax.text(2.5, med_v - 0.45, f"Median {med_v:.1f}\u00b0",
            ha="left", va="top", fontsize=7.5, color="#475569", bbox=_tb, zorder=6)
    ax.set_xlabel("Held-out fold (sorted by RMSE)", fontsize=9)
    ax.set_ylabel("Test RMSE (deg)", fontsize=9)
    ax.set_xlim(0.2, float(rmse.size) + 0.8)
    ax.set_ylim(0, None)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 55])
    _grid(ax, "y")

    # Compact legend for bar colors
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=BLUE_COL, alpha=0.88, label="RMSE \u2264 10\u00b0"),
                 Patch(facecolor=PRED_COL, alpha=0.88, label="RMSE > 10\u00b0")],
        loc="upper left", fontsize=7.5, frameon=False,
    )

    out = OUT_DIR / "fig3_prediction_performance.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return str(out)


def fig4_simulation(sim_df: pd.DataFrame) -> str:
    ref_auc  = sim_df["ref_auc"].to_numpy()
    pred_auc = sim_df["pred_auc"].to_numpy()
    excess   = sim_df["excess_auc"].to_numpy()
    rng = np.random.default_rng(42)

    fig, (ax_cmp, ax_ex) = plt.subplots(
        1, 2, figsize=(7.0, 3.4),
        gridspec_kw={"wspace": 0.44, "width_ratios": [1.1, 1.0]},
    )

    def _box(axp: plt.Axes, data: np.ndarray, xpos: float, col: str, jitter: float = 0.20) -> None:
        axp.boxplot(
            [data], positions=[xpos], widths=0.28,
            patch_artist=True, showfliers=False,
            boxprops=dict(facecolor="none", edgecolor=INK, linewidth=1.0),
            whiskerprops=dict(color=INK, linewidth=1.0),
            capprops=dict(color=INK, linewidth=1.0),
            medianprops=dict(color=col, linewidth=1.8),
            zorder=4,
        )
        axp.scatter(
            xpos + rng.uniform(-jitter, jitter, size=data.size),
            data, s=10, color=col, edgecolors="none",
            alpha=0.70, zorder=3,
        )

    # Panel A — REF vs PRED on shared y-axis
    _box(ax_cmp, ref_auc,  1.0, REF_COL)
    _box(ax_cmp, pred_auc, 2.0, PRED_COL)
    ax_cmp.set_xticks([1.0, 2.0])
    ax_cmp.set_xticklabels(["REF", "PRED"])
    ax_cmp.set_xlim(0.45, 2.55)
    ax_cmp.set_ylabel("Instability AUC", fontsize=9)
    ax_cmp.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _grid(ax_cmp, "y")
    ax_cmp.spines["bottom"].set_visible(False)
    ax_cmp.text(0.03, 0.97, "A", transform=ax_cmp.transAxes,
                fontsize=10, fontweight="bold", va="top", color=INK,
                bbox=dict(fc="white", ec="none", pad=1.0))

    # Panel B — excess AUC
    _box(ax_ex, excess, 1.0, PRED_COL, jitter=0.18)
    ax_ex.axhline(0, color="#94a3b8", linewidth=0.9, linestyle="--", zorder=1)
    ax_ex.set_xticks([1.0])
    ax_ex.set_xticklabels(["PRED \u2212 REF"])
    ax_ex.set_xlim(0.55, 1.45)
    ax_ex.set_ylabel("Excess instability AUC", fontsize=9)
    ax_ex.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _grid(ax_ex, "y")
    ax_ex.spines["bottom"].set_visible(False)
    # Extend ylim to give annotation headroom above the top whisker
    _lo, _hi = ax_ex.get_ylim()
    ax_ex.set_ylim(_lo, _hi + (_hi - _lo) * 0.22)
    n_pos = int(np.sum(excess > 0))
    mean_ex = float(np.mean(excess))
    ax_ex.text(0.04, 0.97,
               f"{100 * n_pos / len(excess):.0f}% of trials > 0\nMean = {mean_ex:.3f}",
               transform=ax_ex.transAxes, ha="left", va="top", fontsize=7.5, color=INK,
               bbox=dict(fc="white", ec="none", pad=1.5))
    ax_ex.text(0.03, 0.03, "B", transform=ax_ex.transAxes,
               fontsize=10, fontweight="bold", va="bottom", color=INK,
               bbox=dict(fc="white", ec="none", pad=1.0))

    out = OUT_DIR / "fig4_simulation_instability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return str(out)


def fig8_model_architecture() -> str:
    nodes = _traced_cnn_bilstm_nodes()
    graph = nx.DiGraph()
    for i, node in enumerate(nodes):
        graph.add_node(i, **node)
        if i > 0:
            graph.add_edge(i - 1, i)

    pos = {i: (i, 0.0) for i in range(len(nodes))}
    kind_colors = {
        "input": "#dfe9f4",
        "conv": "#d9e7f4",
        "rnn": "#efe7d5",
        "linear": "#f4dfda",
        "op": "#eef1f4",
        "module": "#eef1f4",
        "output": "#f3f3f3",
    }

    fig, ax = plt.subplots(figsize=(12.6, 3.4))
    ax.axis("off")
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#92a0b0", width=1.5, arrows=True, arrowsize=16, arrowstyle="-|>")

    box_w = 0.86
    box_h = 0.56
    for i, data in graph.nodes(data=True):
        x, y = pos[i]
        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.015,rounding_size=0.06",
                facecolor=kind_colors.get(str(data["kind"]), "#eef1f4"),
                edgecolor="#bcc6d2",
                linewidth=1.0,
            )
        )
        shape = data.get("shape")
        shape_txt = ""
        if shape is not None:
            shape_txt = " x ".join(str(int(s)) for s in shape)
        ax.text(
            x,
            y + 0.07,
            str(data["label"]),
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold" if data["kind"] in {"input", "output", "rnn"} else None,
            color=INK,
        )
        if shape_txt:
            ax.text(x, y - 0.13, shape_txt, ha="center", va="center", fontsize=6.9, color=MUTED)

    ax.set_xlim(-0.7, len(nodes) - 0.3)
    ax.set_ylim(-0.75, 0.75)
    out = OUT_DIR / "fig8_model_architecture.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return str(out)


def fig5_correlation(trials_df: pd.DataFrame, partial_sum: dict[str, Any]) -> str:
    x_raw   = trials_df["predictor_knee_rmse_deg"].to_numpy()
    y_raw   = trials_df["outcome_value"].to_numpy()
    x_match = trials_df["control_match_knee_rmse_deg"].to_numpy()
    x_res   = trials_df["residual_predictor"].to_numpy()
    y_res   = trials_df["residual_outcome"].to_numpy()

    rho_raw,   p_raw   = stats.spearmanr(x_raw,   y_raw)
    rho_match, p_match = stats.spearmanr(x_match,  y_raw)
    rho_part = float(partial_sum["rho_partial_spearman"])
    p_part   = float(partial_sum["p_value_two_sided"])

    DOT = "#4a5568"   # same neutral slate for all three panels
    LN  = PRED_COL    # regression line in paper accent colour

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), gridspec_kw={"wspace": 0.48})

    panels = [
        (axes[0], x_raw,   y_raw, "Model RMSE (deg)",      "Excess instability AUC",    rho_raw,   p_raw,   "A"),
        (axes[1], x_match, y_raw, "Match RMSE (deg)",       "Excess instability AUC",    rho_match, p_match, "B"),
        (axes[2], x_res,   y_res, "Residualised model RMSE","Residualised excess AUC",   rho_part,  p_part,  "C"),
    ]

    for ax, xd, yd, xlab, ylab, rho, p, lbl in panels:
        ax.scatter(xd, yd, s=20, color=DOT, edgecolors="white", linewidth=0.25, alpha=0.78, zorder=3)
        m, b = np.polyfit(xd, yd, 1)
        xs = np.linspace(float(np.min(xd)), float(np.max(xd)), 200)
        ax.plot(xs, m * xs + b, color=LN, linewidth=1.2, zorder=2)
        ax.set_xlabel(xlab, fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        ax.text(0.97, 0.97, _rho_text(float(rho), float(p)),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK,
                bbox=dict(fc="white", ec="none", pad=1.5))
        _grid(ax, "both")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.text(0.03, 0.97, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top", color=INK,
                bbox=dict(fc="white", ec="none", pad=1.0))

    out = OUT_DIR / "fig5_fwl_correlation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return str(out)


def fig7_motion_matching(sim_df: pd.DataFrame, trial_idx: int = 75) -> str:
    eval_dir, summary_path, _ = _eval_trial_paths(sim_df, trial_idx)
    summ = json.loads(summary_path.read_text())
    thigh_rms = float(summ["match"]["rms_thigh_ori_err_deg"])
    knee_rmse = float(summ["match"]["rmse_knee_deg"])
    motion_plot = eval_dir / "plots" / "motion_match.png"
    img = Image.open(motion_plot).convert("RGB")
    arr = np.asarray(img)
    crop_top = int(arr.shape[0] * 0.12)
    cropped = arr[crop_top:, :, :]

    cand_rows = []
    seen: set[tuple[str, int]] = set()
    for c in summ["match"]["top_candidates"]:
        key = (str(c["snippet_id"]), int(c["start_step_in_snip"]))
        if key in seen:
            continue
        seen.add(key)
        cand_rows.append(
            {
                "label": f"{c['snippet_id']} @ {int(c['start_step_in_snip'])}",
                "score": float(c["score"]),
                "knee": float(c["rmse_knee_deg"]),
                "thigh": float(c["rms_thigh_ori_err_deg"]),
            }
        )
        if len(cand_rows) >= 4:
            break

    def _short_label(raw: str) -> str:
        name = str(raw).split(" @ ")[0].replace("CMU_", "")
        parts = name.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
        return name

    labels = [f"#{i}  {_short_label(r['label'])}" for i, r in enumerate(cand_rows, start=1)]
    scores = [r["score"] for r in cand_rows]

    fig = plt.figure(figsize=(10.8, 4.45))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.82, 1.00], left=0.040, right=0.985, top=0.965, bottom=0.105, wspace=0.12)
    ax_img = fig.add_subplot(gs[0, 0])
    gs_right = gs[0, 1].subgridspec(2, 1, height_ratios=[0.24, 0.76], hspace=0.06)
    ax_info = fig.add_subplot(gs_right[0, 0])
    ax_bar = fig.add_subplot(gs_right[1, 0])

    ax_img.imshow(cropped)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_color("#cdd3db")
        spine.set_linewidth(0.8)

    ax_info.axis("off")
    ax_info.text(0.00, 0.95, "Selected match: #1", ha="left", va="top", fontsize=8.8, fontweight="bold", color=INK)
    ax_info.text(0.00, 0.63, f"Match knee RMSE: {knee_rmse:.2f} deg", ha="left", va="top", fontsize=8.2, color=INK)
    ax_info.text(0.00, 0.37, f"Thigh orientation RMS: {thigh_rms:.2f} deg", ha="left", va="top", fontsize=8.2, color=INK)
    ax_info.text(
        0.00,
        0.08,
        "Left-bottom trace shows instantaneous thigh error; peaks can exceed the window RMS.",
        ha="left",
        va="bottom",
        fontsize=7.1,
        color=MUTED,
    )

    ypos = np.arange(len(labels))
    colors = [PRED_COL] + [BLUE_COL] * max(0, len(labels) - 1)
    bars = ax_bar.barh(ypos, scores, color=colors, edgecolor=WHITE, linewidth=0.4, alpha=0.88)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(labels, fontsize=7.0)
    ax_bar.set_xlabel("Match score (deg)")
    ax_bar.invert_yaxis()
    _grid(ax_bar, "x")
    ax_bar.set_xlim(0, max(scores) * 1.18 if scores else 1.0)
    ax_bar.tick_params(axis="y", pad=4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for bar, score in zip(bars, scores):
        ax_bar.text(
            bar.get_width() + 0.04 * ax_bar.get_xlim()[1],
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            ha="left",
            fontsize=7.0,
            color=MUTED,
        )

    out = OUT_DIR / "fig7_motion_matching.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return str(out)


def fig6_simulation_frames(sim_df: pd.DataFrame, trial_idx: int = 75) -> str:
    row = sim_df.iloc[trial_idx]
    gif_path = Path(os.path.normpath(str(row["compare_npz"]))).with_suffix(".gif")
    if not gif_path.exists():
        raise FileNotFoundError(f"Replay GIF not found: {gif_path}")

    gif = Image.open(gif_path)
    n_frames = int(getattr(gif, "n_frames", 1))
    picks = [0, max(0, n_frames // 3), max(0, 2 * n_frames // 3), max(0, n_frames - 1)]
    labels = ["Start", "Early", "Mid", "Late"]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 4.2), gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    for ax, idx, label in zip(axes.flat, picks, labels):
        gif.seek(int(idx))
        frame = np.asarray(gif.convert("RGB"))
        ax.imshow(frame, aspect="auto")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.16", facecolor=(0, 0, 0, 0.45), edgecolor="none"),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d0d4da")
            spine.set_linewidth(0.8)

    out = OUT_DIR / "fig6_simulation_frames.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def _write_captions(paths: dict[str, str], sim_df: pd.DataFrame, partial_sum: dict[str, Any]) -> str:
    mean_excess = float(sim_df["excess_auc"].mean())
    rho_raw, p_raw = stats.spearmanr(sim_df["pred_rmse"], sim_df["excess_auc"])
    text = f"""# Paper Figure Captions

## Figure 1
Integrated pipeline overview showing the wearable inputs, Conv-BiLSTM architecture, motion-matching stage, and physics evaluation stage.

## Figure 2
Representative trial showing XCoM margin traces and the resulting instability trace used for excess-AUC scoring.

## Figure 3
Sorted held-out subject-fold prediction RMSE values for the native-rate Georgia Tech benchmark.

## Figure 4
Simulation outcomes shown as REF and PRED instability AUC distributions, alongside the corresponding excess instability AUC distribution. Mean excess AUC = {mean_excess:.3f}.

## Figure 5
Raw RMSE, motion-match RMSE, and residualized RMSE associations with excess instability AUC. Raw rho = {float(rho_raw):.3f}, p = {float(p_raw):.3f}; partial rho = {float(partial_sum['rho_partial_spearman']):.3f}, p = {float(partial_sum['p_value_two_sided']):.3f}.

## Figure 6
Representative replay frames from one saved simulation artifact.

## Figure 7
Actual motion-matching retrieval for one evaluation window, showing the aligned query-versus-reference traces, the instantaneous thigh-error trace, and the top candidate scores. The annotated RMS thigh error is the window summary metric; it is lower than the peak instantaneous error.
"""
    out = OUT_DIR / "captions.md"
    out.write_text(text, encoding="utf-8")
    return str(out)


def main() -> None:
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = _train_df()
    _, sim = _sim_df()
    trials = _partial_df()
    partial = _partial_summary()

    paths = {
        "fig1": fig1_pipeline_clean(),
        "fig2": fig2_representative_trial(sim, trial_idx=75),
        "fig3": fig3_prediction(train),
        "fig4": fig4_simulation(sim),
        "fig5": fig5_correlation(trials, partial),
        "fig6": fig6_simulation_frames(sim, trial_idx=75),
        "fig7": fig7_motion_matching(sim, trial_idx=75),
    }

    (OUT_DIR / "figure_manifest.json").write_text(
        json.dumps({"figure_paths": paths, "training_run": str(TRAIN_RUN), "sim_run": str(SIM_RUN)}, indent=2),
        encoding="utf-8",
    )
    _write_captions(paths, sim, partial)
    print(f"Done. {OUT_DIR}")


if __name__ == "__main__":
    main()
