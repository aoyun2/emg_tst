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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
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
        spacing=18,
        padding=14,
        shade_step=12,
        scale_xy=1.0,
        scale_z=0.16,
        background_fill="white",
    )
    arr = np.asarray(img.convert("RGB"))
    return _trim_white(arr)


def _physics_thumb(ax: plt.Axes, sim_df: pd.DataFrame, trial_idx: int = 75) -> None:
    row = sim_df.iloc[trial_idx]
    gif_path = Path(os.path.normpath(str(row["compare_npz"]))).with_suffix(".gif")
    gif = Image.open(gif_path)
    gif.seek(max(0, int(getattr(gif, "n_frames", 1)) // 2))
    frame = np.asarray(gif.convert("RGB"))
    ax.imshow(frame, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    inset = ax.inset_axes([0.05, 0.06, 0.90, 0.24])
    d = _load_npz(sim_df, trial_idx)
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
    arch_thumb = _prediction_thumb_image()

    fig = plt.figure(figsize=(13.2, 4.45))
    gs = fig.add_gridspec(
        2,
        4,
        left=0.035,
        right=0.985,
        top=0.93,
        bottom=0.10,
        height_ratios=[0.16, 0.84],
        width_ratios=[0.92, 1.45, 0.88, 0.95],
        hspace=0.04,
        wspace=0.14,
    )
    head_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_inputs = fig.add_subplot(gs[1, 0])
    ax_model = fig.add_subplot(gs[1, 1])
    ax_match = fig.add_subplot(gs[1, 2])
    ax_sim = fig.add_subplot(gs[1, 3])

    titles = ["Wearable inputs", "Conv-BiLSTM architecture", "Motion matching", "Physics evaluation"]
    subtitles = [
        "4 EMG channels + thigh IMU",
        "400 timesteps x 10 channels to knee angle at t + 10 ms",
        "nearest reference retrieval",
        "paired REF and PRED rollout",
    ]

    for hax, title, subtitle in zip(head_axes, titles, subtitles):
        hax.axis("off")
        hax.text(0.0, 0.76, title, ha="left", va="center", fontsize=10.0, fontweight="bold", color=INK)
        hax.text(0.0, 0.08, subtitle, ha="left", va="bottom", fontsize=7.4, color=MUTED)

    for ax in (ax_inputs, ax_model, ax_match, ax_sim):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#c9cfd8")
            spine.set_linewidth(1.0)

    _gt_emg_thumb(ax_inputs)
    ax_model.imshow(arch_thumb, aspect="auto")
    ax_model.text(0.02, 0.03, "Conv1d x2  ->  BiLSTM x2  ->  MLP", transform=ax_model.transAxes, ha="left", va="bottom", fontsize=7.0, color=MUTED)
    ax_match.imshow(motion_crop, aspect="auto")
    _physics_thumb(ax_sim, sim, 75)

    panel_axes = [ax_inputs, ax_model, ax_match, ax_sim]
    for a0, a1 in zip(panel_axes[:-1], panel_axes[1:]):
        p0 = a0.get_position()
        p1 = a1.get_position()
        y_arrow = p0.y0 + p0.height * 0.50
        fig.add_artist(
            FancyArrowPatch(
                (p0.x1 + 0.008, y_arrow),
                (p1.x0 - 0.008, y_arrow),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.3,
                color="#8f99a5",
            )
        )

    out = OUT_DIR / "fig1_pipeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"wspace": 0.35})

    ax1.axhline(0, color="#9aa3ad", linewidth=1.0)
    ax1.plot(t, xcom_ref, color=REF_COL, linewidth=1.9, label="REF")
    ax1.plot(t, xcom_pred, color=PRED_COL, linewidth=1.9, label="PRED")
    ax1.fill_between(t, xcom_ref, 0, where=(xcom_ref < 0), color=REF_COL, alpha=0.10)
    ax1.fill_between(t, xcom_pred, 0, where=(xcom_pred < 0), color=PRED_COL, alpha=0.10)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("XCoM margin (m)")
    ax1.legend(loc="lower left", fontsize=8)
    _grid(ax1, "y")

    ax2.plot(t, risk_ref, color=REF_COL, linewidth=1.9, label="REF")
    ax2.plot(t, risk_pred, color=PRED_COL, linewidth=1.9, label="PRED")
    ax2.fill_between(t, risk_ref, color=REF_COL, alpha=0.10)
    ax2.fill_between(t, risk_pred, color=PRED_COL, alpha=0.10)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Instability score")
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.text(
        0.98,
        0.04,
        f"Excess AUC = {excess:.3f}",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    _grid(ax2, "y")

    out = OUT_DIR / "fig2_representative_trial.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def fig3_prediction(train_df: pd.DataFrame) -> str:
    df = train_df.copy()
    df = df.sort_values("test_rmse")
    rmse = np.asarray(df["test_rmse"].to_numpy(), dtype=float)
    x = np.arange(1, rmse.size + 1, dtype=int)

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.bar(x, rmse, width=0.78, color=BLUE_COL, edgecolor=WHITE, linewidth=0.35, alpha=0.88, zorder=3)
    ax.axhline(10.0, color=PRED_COL, linewidth=1.3, zorder=5)
    ax.set_xlabel("Held-out fold (sorted by RMSE)")
    ax.set_ylabel("Test RMSE (deg)")
    ax.set_xlim(0.5, float(rmse.size) + 0.5)
    ax.set_xticks([1, 10, 20, 30, 40, 50, 55])
    _grid(ax, "both")

    out = OUT_DIR / "fig3_prediction_performance.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def fig4_simulation(sim_df: pd.DataFrame) -> str:
    ref_auc = sim_df["ref_auc"].to_numpy()
    pred_auc = sim_df["pred_auc"].to_numpy()
    excess = sim_df["excess_auc"].to_numpy()
    rng = np.random.default_rng(42)

    fig = plt.figure(figsize=(8.4, 3.55))
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.965, bottom=0.16, top=0.96, wspace=0.34, width_ratios=[1.18, 0.72])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.boxplot(
        [ref_auc, pred_auc],
        positions=[1, 2],
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=WHITE, edgecolor=INK, linewidth=1.0),
        whiskerprops=dict(color=INK, linewidth=1.0),
        capprops=dict(color=INK, linewidth=1.0),
        medianprops=dict(color=INK, linewidth=1.2),
    )
    ax1.scatter(
        1 + rng.uniform(-0.10, 0.10, size=ref_auc.size),
        ref_auc,
        s=18,
        color=REF_COL,
        edgecolors="white",
        linewidth=0.25,
        alpha=0.75,
        zorder=3,
    )
    ax1.scatter(
        2 + rng.uniform(-0.10, 0.10, size=pred_auc.size),
        pred_auc,
        s=18,
        color=PRED_COL,
        edgecolors="white",
        linewidth=0.25,
        alpha=0.75,
        zorder=3,
    )
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["REF", "PRED"])
    ax1.set_ylabel("Instability AUC")
    ax1.set_xlim(0.55, 2.45)
    _grid(ax1, "y")

    ax2.boxplot(
        [excess],
        positions=[1],
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=WHITE, edgecolor=INK, linewidth=1.0),
        whiskerprops=dict(color=INK, linewidth=1.0),
        capprops=dict(color=INK, linewidth=1.0),
        medianprops=dict(color=INK, linewidth=1.2),
    )
    ax2.scatter(
        1 + rng.uniform(-0.035, 0.035, size=excess.size),
        excess,
        s=16,
        color=PRED_COL,
        edgecolors="white",
        linewidth=0.25,
        alpha=0.8,
        zorder=3,
    )
    ax2.axhline(0, color=INK, linewidth=1.0, zorder=1)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["PRED - REF"])
    ax2.set_ylabel("Excess instability AUC")
    ax2.set_xlim(0.86, 1.14)
    _grid(ax2, "y")

    out = OUT_DIR / "fig4_simulation_instability.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
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
    x_raw = trials_df["predictor_knee_rmse_deg"].to_numpy()
    y_raw = trials_df["outcome_value"].to_numpy()
    x_match = trials_df["control_match_knee_rmse_deg"].to_numpy()
    x_res = trials_df["residual_predictor"].to_numpy()
    y_res = trials_df["residual_outcome"].to_numpy()

    rho_raw, p_raw = stats.spearmanr(x_raw, y_raw)
    rho_match, p_match = stats.spearmanr(x_match, y_raw)
    rho_part = float(partial_sum["rho_partial_spearman"])
    p_part = float(partial_sum["p_value_two_sided"])

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), gridspec_kw={"wspace": 0.35})

    panels = [
        (axes[0], x_raw, y_raw, BLUE_COL, "Model RMSE (deg)", "Excess instability AUC", rho_raw, p_raw),
        (axes[1], x_match, y_raw, REF_COL, "Match RMSE (deg)", "Excess instability AUC", rho_match, p_match),
        (axes[2], x_res, y_res, PRED_COL, "Residualized model RMSE", "Residualized excess AUC", rho_part, p_part),
    ]

    for ax, xd, yd, clr, xlab, ylab, rho, p in panels:
        ax.scatter(xd, yd, s=24, color=clr, edgecolors="white", linewidth=0.35, alpha=0.82, zorder=3)
        m, b = np.polyfit(xd, yd, 1)
        xs = np.linspace(float(np.min(xd)), float(np.max(xd)), 200)
        ax.plot(xs, m * xs + b, color=INK, linewidth=1.4)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.text(0.97, 0.96, _rho_text(float(rho), float(p)), transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
        _grid(ax, "both")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Keep physical panel sizes identical, and reduce the visual impression of
    # unequal panels by trimming repeated long labels.
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    out = OUT_DIR / "fig5_fwl_correlation.png"
    fig.savefig(out)
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
        "fig1": fig1_pipeline(),
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
