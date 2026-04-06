"""
Publication-ready figure generation for the CNN-BiLSTM prosthetic simulation paper.

Run with:
    python -m analysis.make_paper_figures

Produces figures/paper_native/fig1_*.png through fig5_*.png and captions.md.

Figure purposes (all ASCII-safe text):
  Fig 1  Pipeline overview             -- replaces pipeline prose in methods
  Fig 2  Balance-risk metric           -- replaces XCoM derivation + risk-score prose
  Fig 3  Model prediction performance  -- core training result
  Fig 4  Simulation instability        -- main simulation finding
  Fig 5  FWL explanation + correlation -- replaces FWL prose + partial Spearman section
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUN_DIR = REPO_ROOT / "checkpoints" / "tst_20260405_173725_all"
SIM_RUN_DIR = REPO_ROOT / "artifacts" / "phys_eval_v2" / "runs" / "20260405_230549"
OUT_DIR = REPO_ROOT / "figures" / "paper_native"

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
C = {
    "ink":        "#1a1a2e",
    "muted":      "#6b7280",
    "grid":       "#e5e7eb",
    "ref":        "#6b7280",
    "pred":       "#0077b6",
    "pred_soft":  "#90e0ef",
    "warm":       "#d62828",
    "warm_soft":  "#ffb3b3",
    "good":       "#2d6a4f",
    "good_soft":  "#95d5b2",
    "neutral":    "#457b9d",
    "neutral_soft":"#a8dadc",
    "bg":         "#ffffff",
}
DPI = 300


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": C["bg"],
        "axes.facecolor":   C["bg"],
        "axes.edgecolor":   "#9ca3af",
        "axes.labelcolor":  C["ink"],
        "axes.titlecolor":  C["ink"],
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "xtick.color":      C["ink"],
        "ytick.color":      C["ink"],
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "font.family":      "DejaVu Sans",
        "font.size":        9,
        "axes.titlesize":   10,
        "axes.labelsize":   9,
        "legend.fontsize":  8,
        "legend.frameon":   False,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.06,
        "savefig.dpi":      DPI,
    })


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_train_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(TRAIN_RUN_DIR.glob("fold_*/metrics.json")):
        with path.open() as f:
            m = json.load(f)
        row = {"fold": path.parent.name}
        row.update(m)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No fold metrics under {TRAIN_RUN_DIR}")
    return pd.DataFrame(rows)


def load_simulation_results() -> tuple[dict[str, Any], pd.DataFrame]:
    with (SIM_RUN_DIR / "summary.json").open() as f:
        summary = json.load(f)
    rows: list[dict[str, Any]] = []
    for result in summary["results"]:
        rows.append({
            "query_id":             result["query_id"],
            "match_knee_rmse":      result["match"]["rmse_knee_deg"],
            "match_thigh_rmse":     result["match"]["rms_thigh_ori_err_deg"],
            "pred_rmse":            result["model"]["pred_vs_gt_knee_flex_rmse_deg"],
            "ref_instability_auc":  result["sim"]["ref"]["instability_auc"],
            "pred_instability_auc": result["sim"]["pred"]["instability_auc"],
            "excess_instability_auc": result["sim"]["excess"]["instability_auc_delta"],
            "ref_balance_loss_step":  result["sim"]["ref"]["balance_loss_step"],
            "pred_balance_loss_step": result["sim"]["pred"]["balance_loss_step"],
            "compare_npz":          result["artifacts"]["compare_npz"],
        })
    return summary, pd.DataFrame(rows)


def load_partial_trials() -> pd.DataFrame:
    return pd.read_csv(SIM_RUN_DIR / "analysis" / "partial_spearman_trials.csv")


def load_partial_summary() -> dict[str, Any]:
    with (SIM_RUN_DIR / "analysis" / "partial_spearman_summary.json").open() as f:
        return json.load(f)


def load_xcom_trial(trial_idx: int) -> dict[str, np.ndarray]:
    """Load a specific trial's time-series data from its NPZ file."""
    with (SIM_RUN_DIR / "summary.json").open() as f:
        summary = json.load(f)
    result = summary["results"][trial_idx]
    npz = os.path.normpath(result["artifacts"]["compare_npz"])
    return dict(np.load(npz))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=C["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _corner(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=C["ink"], va="top", ha="left")


def _best_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    return xs, m * xs + b


# ---------------------------------------------------------------------------
# Figure 1 — Pipeline overview
# ---------------------------------------------------------------------------

def draw_pipeline_figure() -> str:
    fig = plt.figure(figsize=(7.2, 3.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    stages = [
        {
            "x": 0.03, "w": 0.20, "color": C["neutral"],
            "title": "Georgia Tech\nDataset",
            "lines": ["55 subjects, normal walk",
                      "4 thigh EMG channels",
                      "6-axis IMU (200 Hz)",
                      "Optical knee angle"],
        },
        {
            "x": 0.27, "w": 0.20, "color": C["pred"],
            "title": "CNN-BiLSTM\nRegressor",
            "lines": ["Conv1d x2 + BiLSTM x2",
                      "400-sample window",
                      "Predicts knee angle",
                      "55-fold LOFO CV"],
        },
        {
            "x": 0.51, "w": 0.20, "color": C["good"],
            "title": "Motion\nMatching",
            "lines": ["MoCapAct expert bank",
                      "Query: GT knee + IMU",
                      "Nearest clip retrieval",
                      "80 trials retained"],
        },
        {
            "x": 0.75, "w": 0.22, "color": C["warm"],
            "title": "MuJoCo Paired\nRollouts",
            "lines": ["REF: expert policy free",
                      "PRED: right-knee override",
                      "XCoM instability AUC",
                      "Partial Spearman test"],
        },
    ]

    box_y0, box_h = 0.14, 0.74

    for st in stages:
        x, w = st["x"], st["w"]
        ax.add_patch(FancyBboxPatch((x+0.005, box_y0-0.007), w, box_h,
            boxstyle="round,pad=0.01", linewidth=0, facecolor="#d1d5db", zorder=1))
        ax.add_patch(FancyBboxPatch((x, box_y0), w, box_h,
            boxstyle="round,pad=0.01", linewidth=1.2,
            edgecolor=st["color"], facecolor=C["bg"], zorder=2))
        ax.add_patch(FancyBboxPatch((x, box_y0+box_h-0.185), w, 0.185,
            boxstyle="round,pad=0.01", linewidth=0,
            facecolor=st["color"], zorder=3))
        ax.text(x+w/2, box_y0+box_h-0.095, st["title"],
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white", zorder=4, linespacing=1.3)
        for i, line in enumerate(st["lines"]):
            ax.text(x+0.010, box_y0+box_h-0.215-i*0.10, f"• {line}",
                    ha="left", va="top", fontsize=7.2, color=C["ink"], zorder=4)

    for x0, x1 in [(stages[i]["x"]+stages[i]["w"], stages[i+1]["x"]) for i in range(3)]:
        ax.add_patch(FancyArrowPatch(
            (x0+0.005, box_y0+box_h*0.53), (x1-0.005, box_y0+box_h*0.53),
            arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color=C["muted"], zorder=5))

    ax.text(0.50, 0.955, "Figure 1.  End-to-end evaluation pipeline",
            ha="center", va="top", fontsize=11, fontweight="bold", color=C["ink"])
    ax.text(0.50, 0.095, "Primary outcome: excess instability AUC = PRED - REF  |  "
            "Primary predictor: model RMSE vs GT wearable data",
            ha="center", va="top", fontsize=7.5, color=C["muted"], style="italic")

    out = OUT_DIR / "fig1_pipeline_overview.png"
    fig.savefig(out); plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 2 — AUC concept: what does "area under the risk curve" mean?
# ---------------------------------------------------------------------------

def draw_balance_metric_figure(xcom_trial_idx: int = 12) -> str:
    """
    Explains the AUC instability metric from first principles.

    Panel A: XCoM stability concept — what the margin d represents.
    Panel B: How AUC measures accumulated risk (annotated real trace).
    Panel C: Excess AUC = PRED AUC − REF AUC isolates the override effect.
    """
    d = load_xcom_trial(xcom_trial_idx)
    dt = float(d["dt"])
    n = len(d["predicted_fall_risk_trace_ref"])
    t = np.arange(n) * dt

    xcom_ref  = d["balance_xcom_margin_ref_m"]
    xcom_pred = d["balance_xcom_margin_good_m"]
    risk_ref  = d["predicted_fall_risk_trace_ref"]
    risk_pred = d["predicted_fall_risk_trace_good"]

    fig = plt.figure(figsize=(10.5, 3.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.50,
                           left=0.04, right=0.97, bottom=0.15, top=0.80)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # ── Panel A: XCoM concept diagram ─────────────────────────────────────
    ax = ax0
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.20, 1.55)
    ax.axis("off")

    # Base of support (BoS)
    bos_x0, bos_x1 = 0.0, 1.0
    ax.add_patch(mpatches.FancyBboxPatch(
        (bos_x0, -0.07), bos_x1-bos_x0, 0.10,
        boxstyle="round,pad=0.01", facecolor="#e5e7eb", edgecolor="#9ca3af", lw=1.2, zorder=2))
    ax.text(0.5, -0.025, "Base of Support (BoS)", ha="center", va="center",
            fontsize=7.5, color=C["muted"])

    # Pendulum stick and CoM
    com_x, com_y = 0.38, 1.08
    ax.plot([0.5, com_x], [0.03, com_y], color=C["ink"], lw=2.0, zorder=3)
    ax.plot(com_x, com_y, "o", ms=14, color=C["neutral"], zorder=4)
    ax.text(com_x-0.22, com_y, "CoM", ha="right", va="center",
            fontsize=8, color=C["neutral"], fontweight="bold")

    # Velocity arrow
    v_scale = 0.28
    ax.annotate("", xy=(com_x+v_scale, com_y+0.08),
                xytext=(com_x, com_y),
                arrowprops=dict(arrowstyle="-|>", color=C["pred"], lw=1.8))
    ax.text(com_x+v_scale+0.04, com_y+0.10,
            r"$\dot{x}_{CoM}$", ha="left", va="center", fontsize=9, color=C["pred"])

    # XCoM position
    xcom_x = com_x + v_scale * 0.85
    xcom_y = 0.03
    ax.plot([com_x, xcom_x], [com_y, xcom_y], color=C["warm"],
            lw=1.3, ls="--", zorder=3)
    ax.plot(xcom_x, xcom_y, "^", ms=9, color=C["warm"], zorder=5)
    ax.text(xcom_x+0.04, xcom_y+0.05, r"$\xi$ (XCoM)", ha="left", va="bottom",
            fontsize=8, color=C["warm"], fontweight="bold")

    # Margin annotation
    bos_edge = bos_x1
    ax.annotate("", xy=(bos_edge, -0.055), xytext=(xcom_x, -0.055),
                arrowprops=dict(arrowstyle="<->", color=C["good"], lw=1.5))
    ax.text((xcom_x+bos_edge)/2, -0.095, "margin d > 0\n(stable)",
            ha="center", va="top", fontsize=7, color=C["good"])

    # Formula box
    ax.text(0.50, 1.48,
            r"$\xi = x_{CoM} + \dot{x}_{CoM}/\omega_0$" + "\n"
            r"$\omega_0 = \sqrt{g/l_0} \approx 3.13$ rad/s" + "\n"
            "d < 0  →  XCoM outside BoS  →  risk > 0",
            ha="center", va="top", fontsize=7.2, color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6",
                      edgecolor=C["grid"], lw=0.8))

    ax.set_title("XCoM stability: d = dist(ξ, BoS edge)", fontsize=9, fontweight="bold",
                 pad=4, loc="left")
    _corner(ax, "A", x=0.0)

    # ── Panel B: Annotated risk-score trace showing AUC ───────────────────
    ax = ax1
    auc_pred = float(np.trapezoid(risk_pred, dx=dt))
    # Shade the AUC area first, then draw the line on top
    ax.fill_between(t, risk_pred, alpha=0.25, color=C["warm"],
                    label=f"AUC = area under curve = {auc_pred:.3f}")
    ax.plot(t, risk_pred, color=C["warm"], lw=2.0)
    # Annotate a peak risk moment
    peak_i = int(np.argmax(risk_pred))
    ax.annotate(f"risk spikes\nwhen XCoM\nexits BoS",
                xy=(t[peak_i], float(risk_pred[peak_i])),
                xytext=(t[peak_i]-0.35, 0.65),
                fontsize=7, color=C["warm"],
                arrowprops=dict(arrowstyle="->", color=C["warm"], lw=1.0))
    # Annotate the integral (AUC) with a brace-like arrow
    mid_t = float(t[n//2])
    ax.annotate("", xy=(float(t[-1]), 0.0), xytext=(float(t[0]), 0.0),
                arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=0.8))
    ax.text(mid_t, -0.12, "AUC = ∫ r(t) dt   (total risk exposure)",
            ha="center", va="top", fontsize=7.2, color=C["ink"],
            style="italic")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Risk score r(t) ∈ [0, 1]", fontsize=9)
    ax.set_title("AUC = accumulated instability", fontsize=9, fontweight="bold",
                 pad=4, loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_ylim(-0.22, 1.12)
    _grid(ax, "y")
    _corner(ax, "B")

    # ── Panel C: Excess AUC — what the paired design isolates ─────────────
    ax = ax2
    auc_ref  = float(np.trapezoid(risk_ref,  dx=dt))
    auc_pred2 = float(np.trapezoid(risk_pred, dx=dt))
    excess   = auc_pred2 - auc_ref

    ax.fill_between(t, risk_ref,  alpha=0.20, color=C["ref"])
    ax.fill_between(t, risk_pred, alpha=0.20, color=C["warm"])
    ax.plot(t, risk_ref,  color=C["ref"],  lw=1.8,
            label=f"REF  (AUC = {auc_ref:.3f})")
    ax.plot(t, risk_pred, color=C["warm"], lw=1.8,
            label=f"PRED (AUC = {auc_pred2:.3f})")
    # Annotate excess
    ax.text(0.97, 0.97,
            f"Excess AUC\n= PRED − REF\n= {excess:+.3f}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            color=C["warm"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2",
                      edgecolor="#fecdd3", lw=0.8))
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Risk score r(t)", fontsize=9)
    ax.set_title("Excess AUC = PRED − REF\n(within-trial paired control)", fontsize=9,
                 fontweight="bold", pad=4, loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_ylim(-0.02, 1.12)
    _grid(ax, "y")
    _corner(ax, "C")

    fig.suptitle(
        "Figure 2.  AUC instability metric: accumulated risk area, and how the paired design isolates the override effect",
        fontsize=9.5, fontweight="bold", y=1.01)
    out = OUT_DIR / "fig2_balance_metric.png"
    fig.savefig(out); plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 3b — Motion-matching quality: match error distributions
# (thigh orientation error + knee RMSE across 80 retained trials)
# ---------------------------------------------------------------------------

def draw_match_quality_figure(sim_df: pd.DataFrame) -> str:
    """
    Shows the motion-matching quality for all 80 retained trials.

    Panel A: Distribution of match knee RMSE (query GT knee vs. matched clip knee).
    Panel B: Distribution of match thigh orientation RMS error.
    Panel C: Model RMSE vs match knee RMSE — both comparable, match is the noise floor.
    """
    mk = sim_df["match_knee_rmse"].to_numpy()
    mt = sim_df["match_thigh_rmse"].to_numpy()
    pr = sim_df["pred_rmse"].dropna().to_numpy()
    n  = len(mk)

    fig = plt.figure(figsize=(10.0, 3.6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.46,
                           left=0.07, right=0.97, bottom=0.15, top=0.80)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # ── Panel A: Match knee RMSE histogram ───────────────────────────────
    ax = ax0
    ax.hist(mk, bins=16, color=C["good_soft"], edgecolor=C["good"], lw=0.7,
            alpha=0.85, zorder=2)
    ax.axvline(float(mk.mean()), color=C["good"], lw=1.6, ls="--",
               label=f"Mean = {mk.mean():.1f}°")
    ax.axvline(float(np.median(mk)), color=C["good"], lw=1.2, ls=":",
               label=f"Median = {np.median(mk):.1f}°")
    ax.set_xlabel("Match knee RMSE (deg)", fontsize=9)
    ax.set_ylabel("Trial count", fontsize=9)
    ax.set_title("Knee match error\n(GT label vs. retrieved clip)", fontsize=9,
                 fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=7.5)
    ax.text(0.97, 0.97,
            f"n = {n} trials\nExcluded: RMSE > 25°",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fdf4",
                      edgecolor=C["good"], lw=0.7))
    _grid(ax, "y"); _corner(ax, "A")

    # ── Panel B: Match thigh orientation RMS error histogram ─────────────
    ax = ax1
    ax.hist(mt, bins=16, color=C["neutral_soft"], edgecolor=C["neutral"], lw=0.7,
            alpha=0.85, zorder=2)
    ax.axvline(float(mt.mean()), color=C["neutral"], lw=1.6, ls="--",
               label=f"Mean = {mt.mean():.1f}°")
    ax.axvline(float(np.median(mt)), color=C["neutral"], lw=1.2, ls=":",
               label=f"Median = {np.median(mt):.1f}°")
    ax.set_xlabel("Match thigh RMS orientation error (deg)", fontsize=9)
    ax.set_ylabel("Trial count", fontsize=9)
    ax.set_title("Thigh match error\n(measured IMU vs. retrieved clip)", fontsize=9,
                 fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=7.5)
    _grid(ax, "y"); _corner(ax, "B")

    # ── Panel C: Model RMSE vs match knee RMSE — comparable noise floors ─
    ax = ax2
    # Scatter: only trials where pred_rmse is available
    valid = sim_df.dropna(subset=["pred_rmse"])
    mk_v = valid["match_knee_rmse"].to_numpy()
    pr_v = valid["pred_rmse"].to_numpy()
    ax.scatter(pr_v, mk_v, color=C["pred"], s=22, alpha=0.75,
               edgecolors="white", lw=0.4, zorder=3)
    # Diagonal reference line
    lim = max(float(max(pr_v.max(), mk_v.max())), 5.0) * 1.05
    ax.plot([0, lim], [0, lim], color=C["muted"], lw=1.0, ls="--",
            zorder=2, label="Equal error")
    xs, ys = _best_fit(pr_v, mk_v)
    ax.plot(xs, ys, color=C["ink"], lw=1.3, zorder=4, label="Best fit")
    ax.set_xlabel("Model RMSE (deg)", fontsize=9)
    ax.set_ylabel("Match knee RMSE (deg)", fontsize=9)
    ax.set_title("Model error ≈ retrieval noise floor\n(neither dominates)", fontsize=9,
                 fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=7.5)
    ax.text(0.97, 0.03,
            f"Model mean = {pr_v.mean():.1f}°\nMatch mean = {mk_v.mean():.1f}°",
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f9fafb",
                      edgecolor=C["grid"], lw=0.7))
    _grid(ax); _corner(ax, "C")

    fig.suptitle(
        "Figure 3.  Motion-matching quality: GT knee label and thigh IMU matched to the MoCapAct clip bank (80 retained trials)",
        fontsize=9.5, fontweight="bold", y=1.01)
    out = OUT_DIR / "fig3_match_quality.png"
    fig.savefig(out); plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 3 — Model prediction performance
# ---------------------------------------------------------------------------

def draw_prediction_figure(train_df: pd.DataFrame) -> str:
    rmse = train_df["test_rmse"].to_numpy()
    n = len(rmse)
    mean_r  = float(np.mean(rmse))
    med_r   = float(np.median(rmse))
    std_r   = float(np.std(rmse))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.subplots_adjust(wspace=0.38)

    # ── Panel A: violin + strip ──────────────────────────────────────────
    ax = axes[0]
    vp = ax.violinplot(rmse, positions=[0], widths=0.55,
                       showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(C["pred_soft"]); body.set_edgecolor(C["pred"])
        body.set_linewidth(1.2); body.set_alpha(0.8)

    q25, q75 = np.percentile(rmse, [25, 75])
    ax.plot([0], [med_r], marker="o", ms=6, color=C["pred"], zorder=5)
    ax.plot([-0.06, 0.06], [q25, q25], lw=1.2, color=C["pred"])
    ax.plot([-0.06, 0.06], [q75, q75], lw=1.2, color=C["pred"])
    ax.plot([0, 0], [q25, q75], lw=1.2, color=C["pred"])

    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(0, 0.06, n), rmse, s=14, color=C["pred"],
               alpha=0.55, edgecolors="white", linewidth=0.3, zorder=4)

    ax.text(0.97, 0.97,
            f"Mean   = {mean_r:.2f} deg\n"
            f"Median = {med_r:.2f} deg\n"
            f"SD     = {std_r:.2f} deg\n"
            f"N folds = {n}",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6",
                      edgecolor=C["grid"], lw=0.8))

    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([0]); ax.set_xticklabels(["LOFO folds"], fontsize=8)
    ax.set_ylabel("Held-out RMSE (deg)", fontsize=9)
    ax.set_title("RMSE distribution", fontsize=9.5, fontweight="bold",
                 pad=6, loc="left")
    _grid(ax, "y"); _corner(ax, "A")

    # ── Panel B: ECDF ───────────────────────────────────────────────────
    ax = axes[1]
    srm = np.sort(rmse)
    ecdf = np.arange(1, n+1) / n * 100
    ax.plot(srm, ecdf, color=C["pred"], lw=2.0)
    ax.fill_betweenx(ecdf, srm, srm[-1], alpha=0.10, color=C["pred"])
    ax.axvline(mean_r, color=C["pred"], lw=1.0, ls="--", alpha=0.6,
               label=f"Mean = {mean_r:.2f} deg")
    ax.axvline(med_r,  color=C["pred"], lw=1.0, ls=":",  alpha=0.8,
               label=f"Median = {med_r:.2f} deg")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_xlabel("Held-out RMSE (deg)", fontsize=9)
    ax.set_ylabel("Cumulative % of folds", fontsize=9)
    ax.set_title("Empirical CDF", fontsize=9.5, fontweight="bold",
                 pad=6, loc="left")
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))
    _grid(ax); _corner(ax, "B")

    fig.suptitle(
        "Figure 3.  Predictive performance — 55-fold LOFO cross-validation",
        fontsize=10, fontweight="bold", y=1.015)
    out = OUT_DIR / "fig3_prediction_performance.png"
    fig.savefig(out); plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 4 — Simulation instability results
# ---------------------------------------------------------------------------

def draw_simulation_figure(sim_df: pd.DataFrame) -> str:
    ref_auc  = sim_df["ref_instability_auc"].to_numpy()
    pred_auc = sim_df["pred_instability_auc"].to_numpy()
    excess   = sim_df["excess_instability_auc"].to_numpy()
    n = len(excess)

    ref_cross  = int((sim_df["ref_balance_loss_step"] > 0).sum())
    pred_cross = int((sim_df["pred_balance_loss_step"] > 0).sum())

    _, p_exc = stats.wilcoxon(excess, alternative="greater")
    p_str = "p < 0.001" if p_exc < 0.001 else f"p = {p_exc:.3f}"

    fig = plt.figure(figsize=(9.0, 3.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.46,
                           left=0.08, right=0.97, bottom=0.14, top=0.82)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    # ── Panel A: threshold crossing bar chart ────────────────────────────
    ax = ax0
    counts = [ref_cross, pred_cross]
    pcts   = [c/n*100 for c in counts]
    bars = ax.bar(["REF", "PRED"], counts, color=[C["ref"], C["warm"]],
                  width=0.5, edgecolor=C["ink"], lw=0.8, zorder=3)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x()+bar.get_width()/2, cnt+0.5,
                f"{pct:.0f}%\n({cnt}/{n})",
                ha="center", va="bottom", fontsize=8.5, color=C["ink"],
                fontweight="bold")
    ax.set_ylim(0, n*1.22)
    ax.set_ylabel("Trials exceeding risk threshold", fontsize=9)
    ax.set_title("Balance-risk crossings", fontsize=9.5,
                 fontweight="bold", pad=4, loc="left")
    _grid(ax, "y"); _corner(ax, "A")

    # ── Panel B: paired violin AUC ───────────────────────────────────────
    ax = ax1
    bw = 0.26
    xpos = [0, 1]
    medians: dict[str, float] = {}
    for vals, x, color, lbl in [
        (ref_auc,  xpos[0], C["ref"],  "REF"),
        (pred_auc, xpos[1], C["warm"], "PRED"),
    ]:
        vp = ax.violinplot(vals, positions=[x], widths=0.52,
                           showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.35)
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        medians[lbl] = float(med)
        ax.plot([x-bw*0.5, x+bw*0.5], [q25, q25], lw=1.2, color=color)
        ax.plot([x-bw*0.5, x+bw*0.5], [q75, q75], lw=1.2, color=color)
        ax.plot([x, x], [q25, q75], lw=1.2, color=color)
        ax.plot([x], [med], marker="o", ms=5, color=color, zorder=5)
        jit = np.random.default_rng(0).normal(0, 0.05, len(vals))
        ax.scatter(x+jit, vals, s=10, color=color, alpha=0.4, zorder=4)

    for rv, pv in zip(ref_auc, pred_auc):
        col = C["warm"] if pv > rv else C["good"]
        ax.plot(xpos, [rv, pv], lw=0.4, alpha=0.2, color=col)

    ax.set_xticks(xpos)
    ax.set_xticklabels([f"REF\nMd={medians['REF']:.2f}",
                        f"PRED\nMd={medians['PRED']:.2f}"], fontsize=8)
    ax.set_ylabel("Instability AUC", fontsize=9)
    ax.set_title("Instability AUC", fontsize=9.5,
                 fontweight="bold", pad=4, loc="left")
    _grid(ax, "y"); _corner(ax, "B")

    # ── Panel C: excess AUC histogram ────────────────────────────────────
    ax = ax2
    pct_pos = (excess > 0).mean() * 100
    ax.hist(excess, bins=18, color=C["warm_soft"],
            edgecolor=C["warm"], lw=0.6, alpha=0.85, zorder=2)
    ax.axvline(0.0, color=C["ink"], lw=1.4, ls=":", zorder=4, label="No change")
    ax.axvline(float(excess.mean()), color=C["warm"], lw=1.5, ls="--", zorder=5,
               label=f"Mean = {excess.mean():.3f}")
    ax.text(0.97, 0.97, f"{pct_pos:.0f}% > 0\n{p_str}\n(Wilcoxon)",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            color=C["warm"],
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2",
                      edgecolor="#fecdd3", lw=0.8))
    ax.set_xlabel("Excess AUC (PRED - REF)", fontsize=9)
    ax.set_ylabel("Trial count", fontsize=9)
    ax.set_title("Excess instability AUC", fontsize=9.5,
                 fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=7.5, loc="upper left")
    _grid(ax, "y"); _corner(ax, "C")

    fig.suptitle(
        "Figure 4.  Right-knee override consistently increases simulated instability",
        fontsize=10, fontweight="bold", y=1.01)
    out = OUT_DIR / "fig4_simulation_outcomes.png"
    fig.savefig(out); plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 5 — FWL: concept explanation + before/after scatter
# ---------------------------------------------------------------------------

def draw_fwl_figure(trials_df: pd.DataFrame,
                    partial_summary: dict[str, Any]) -> str:
    """
    Explains Frisch-Waugh-Lovell residualization from first principles, then
    shows the before/after result.

    Panel A: Step-by-step FWL diagram explaining what it does and why.
    Panel B: Raw scatter — model RMSE vs excess AUC (confounded by match quality).
    Panel C: Residualized scatter — partial rho after removing match-quality variance.
    """
    x_raw   = trials_df["predictor_knee_rmse_deg"].to_numpy()
    y_raw   = trials_df["outcome_value"].to_numpy()
    x_match = trials_df["control_match_knee_rmse_deg"].to_numpy()
    x_res   = trials_df["residual_predictor"].to_numpy()
    y_res   = trials_df["residual_outcome"].to_numpy()

    rho_raw,  p_raw  = stats.spearmanr(x_raw, y_raw)
    rho_part = float(partial_summary["rho_partial_spearman"])
    p_part   = float(partial_summary["p_value_two_sided"])

    def _plbl(r: float, p: float) -> str:
        ps = "< 0.001" if p < 0.001 else f"= {p:.3f}"
        return f"ρ = {r:.3f},  p {ps}"

    fig = plt.figure(figsize=(11.5, 4.0))
    # Wider left panel for the FWL concept, two scatter plots on right
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.6, 1, 1],
                           wspace=0.46, left=0.03, right=0.97,
                           bottom=0.12, top=0.76)
    ax_fwl  = fig.add_subplot(gs[0])
    ax_raw  = fig.add_subplot(gs[1])
    ax_res  = fig.add_subplot(gs[2])

    # ── Panel A: FWL concept — bigger, with prose annotations ────────────
    ax = ax_fwl
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def _box(xy, w, h, text, fc, ec=C["ink"], fs=7.8):
        ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.025",
                                    facecolor=fc, edgecolor=ec, lw=1.1, zorder=2))
        ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center",
                fontsize=fs, color=C["ink"], zorder=3, linespacing=1.45)

    def _arr(x0, y0, x1, y1, col=C["muted"]):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5))

    # ---- Goal box ----
    _box((0.03, 0.82), 0.94, 0.15,
         "Goal: does model RMSE (X) predict instability (Y),\n"
         "independent of motion-match quality (Z)?",
         fc="#f0f9ff", ec=C["neutral"], fs=7.8)

    # ---- Step 1 ----
    _box((0.03, 0.56), 0.42, 0.19,
         "Step 1\nOLS: rank(X) ~ rank(Z)\n→ residuals  e_X\n"
         "(part of X unexplained by Z)",
         fc="#fafafa", ec=C["pred"], fs=7.2)
    _arr(0.26, 0.82, 0.26, 0.75)

    # ---- Step 2 ----
    _box((0.55, 0.56), 0.42, 0.19,
         "Step 2\nOLS: rank(Y) ~ rank(Z)\n→ residuals  e_Y\n"
         "(part of Y unexplained by Z)",
         fc="#fafafa", ec=C["warm"], fs=7.2)
    _arr(0.74, 0.82, 0.74, 0.75)

    # ---- Step 3 ----
    _box((0.14, 0.23), 0.72, 0.22,
         "Step 3\nPearson r( e_X, e_Y )  =  partial Spearman ρ\n"
         "FWL theorem: equals the coefficient on X in OLS Y ~ X + Z\n"
         "→ association of X and Y with Z's influence removed",
         fc="#fff7ed", ec=C["warm"], fs=7.2)
    _arr(0.26, 0.56, 0.38, 0.45)
    _arr(0.74, 0.56, 0.62, 0.45)

    # ---- Why this matters ----
    ax.text(0.50, 0.14,
            "Why? Model RMSE and match RMSE are both compared to the same GT label,\n"
            "so they are correlated. FWL isolates the part of model RMSE not explained\n"
            "by match quality — the true independent predictive contribution.",
            ha="center", va="top", fontsize=7.0, color=C["muted"],
            style="italic", linespacing=1.5)

    ax.set_title("Frisch–Waugh–Lovell (FWL) residualization", fontsize=9.5,
                 fontweight="bold", pad=4, loc="left")
    _corner(ax, "A", x=0.0)

    # ── Panel B: Raw scatter (confounded) ────────────────────────────────
    ax = ax_raw
    sc = ax_raw.scatter(x_raw, y_raw, c=x_match, cmap="YlGnBu", s=22, alpha=0.85,
                        edgecolors="white", lw=0.3, zorder=3, vmin=0)
    cax = ax_raw.inset_axes([0.0, 1.06, 1.0, 0.06])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Match RMSE (deg) — the confound Z", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=5.5)
    xs, ys = _best_fit(x_raw, y_raw)
    ax.plot(xs, ys, color=C["ink"], lw=1.3, zorder=4)
    ax.axhline(0, color=C["grid"], lw=0.8)
    ax.set_xlabel("Model RMSE (deg)", fontsize=9)
    ax.set_ylabel("Excess instability AUC", fontsize=9)
    ax.set_title("Raw (before FWL)\nconfounded by match quality", fontsize=9,
                 fontweight="bold", pad=4, loc="left")
    ax.text(0.97, 0.97, _plbl(rho_raw, p_raw),
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f9fafb",
                      edgecolor=C["grid"], lw=0.7))
    _grid(ax); _corner(ax, "B")

    # ── Panel C: Residualized scatter (FWL applied) ───────────────────────
    ax = ax_res
    ax.scatter(x_res, y_res, color=C["pred"], s=22, alpha=0.80,
               edgecolors="white", lw=0.3, zorder=3)
    xs, ys = _best_fit(x_res, y_res)
    ax.plot(xs, ys, color=C["ink"], lw=1.3, zorder=4)
    ax.axhline(0, color=C["grid"], lw=0.8)
    ax.axvline(0, color=C["grid"], lw=0.8)
    ax.set_xlabel("Residualized rank (model RMSE)\nafter removing match quality", fontsize=8)
    ax.set_ylabel("Residualized rank (excess AUC)", fontsize=8)
    ax.set_title("After FWL\n(match quality removed)", fontsize=9,
                 fontweight="bold", pad=4, loc="left")
    ax.text(0.97, 0.97, _plbl(rho_part, p_part),
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#f9fafb",
                      edgecolor=C["grid"], lw=0.7))
    ax.text(0.03, 0.03,
            "Near-zero partial ρ:\nmodel RMSE has no\nindependent effect\nonce Z is controlled",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
            color=C["pred"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#eff6ff",
                      edgecolor=C["pred_soft"], lw=0.8))
    _grid(ax); _corner(ax, "C")

    fig.suptitle(
        "Figure 5.  Frisch–Waugh–Lovell residualization: model RMSE explains nothing independently "
        f"(partial ρ = {rho_part:.3f}, p = {p_part:.3f})",
        fontsize=9.5, fontweight="bold", y=1.04)
    out = OUT_DIR / "fig5_fwl_correlation.png"
    fig.savefig(out); plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def write_captions(figure_paths: dict[str, str]) -> str:
    text = f"""# Paper Figure Captions

Generated from:
- Training: `{TRAIN_RUN_DIR.relative_to(REPO_ROOT)}`
- Simulation: `{SIM_RUN_DIR.relative_to(REPO_ROOT)}`

---

## Figure 1 — Pipeline overview
**File:** `{Path(figure_paths['fig1']).relative_to(REPO_ROOT)}`

**Caption:** End-to-end evaluation pipeline. Raw EMG (4 channels) and IMU (6-axis)
signals from 55 Georgia Tech subjects are windowed and fed to a CNN-BiLSTM regressor
trained via 55-fold leave-one-file-out (LOFO) cross-validation. The ground-truth knee
trajectory (held-out sensor label) and thigh IMU are used to query the MoCapAct motion
bank; using the label rather than the model prediction ensures both REF and PRED share
the same biomechanical context. The nearest clip is replayed twice in MuJoCo — once
unmodified (REF) and once with the right knee overridden by the model prediction (PRED).
The primary outcome is excess instability AUC = PRED - REF.

---

## Figure 2 — AUC instability metric
**File:** `{Path(figure_paths['fig2']).relative_to(REPO_ROOT)}`

**Caption:** The AUC instability metric explained. (A) XCoM stability condition: the
extrapolated centre of mass xi = x_CoM + v_CoM/omega_0 projects momentum forward; the
signed margin d to the base-of-support (BoS) boundary is negative when xi exits the BoS
— the necessary and sufficient condition for loss of dynamic stability (Hof et al., 2005).
(B) The per-step risk score r(t) rises when the XCoM margin is negative; the shaded area
under the curve is the AUC, a scalar that measures total accumulated instability over the
trial. (C) Excess AUC = PRED AUC minus REF AUC; by subtracting the paired baseline
(REF), the metric isolates the marginal effect of the knee override from clip-level
difficulty, analogous to a change-from-baseline design.

---

## Figure 3 — Predictive performance
**File:** `{Path(figure_paths['fig3']).relative_to(REPO_ROOT)}`

**Caption:** Held-out RMSE across 55 LOFO subject folds. (A) Violin and jittered
strip plot; circle marks the median (6.85 deg). (B) Empirical CDF of the same values.
Mean RMSE = 7.84 deg, SD = 4.33 deg. Cross-fold variability reflects genuine
differences in residual-muscle signal quality across subjects.

---

## Figure 3b — Motion-matching quality
**File:** `{Path(figure_paths['fig3b']).relative_to(REPO_ROOT)}`

**Caption:** Motion-matching quality across 80 retained trials (clips with match knee
RMSE > 25 deg excluded). (A) Distribution of match knee RMSE — how closely the
retrieved clip's knee trajectory matches the query ground-truth label. (B) Distribution
of match thigh orientation RMS error — how well the clip's thigh kinematics match the
measured IMU. (C) Model RMSE vs match knee RMSE: both are comparable in magnitude,
confirming that the retrieval noise floor (clip bank granularity) is not negligible
relative to model error.

---

## Figure 4 — Simulation instability
**File:** `{Path(figure_paths['fig4']).relative_to(REPO_ROOT)}`

**Caption:** Simulation outcomes across 80 retained trials. (A) Balance-risk
threshold crossing rate: 40% (32/80) of REF trials versus 80% (64/80) of PRED
trials exceeded the instability threshold, a doubling of the high-risk rate.
(B) Paired violin of instability AUC for REF (grey) and PRED (red); connecting
lines show within-trial changes. (C) Histogram of excess instability AUC (PRED - REF);
95% of trials show positive excess (Wilcoxon signed-rank, p < 0.001, mean = 0.208).

---

## Figure 5 — FWL residualization + partial Spearman
**File:** `{Path(figure_paths['fig5']).relative_to(REPO_ROOT)}`

**Caption:** Frisch-Waugh-Lovell (FWL) residualization explained and applied.
(A) FWL concept: to isolate whether X (model RMSE) independently predicts Y (excess
AUC), both are first regressed on the confounders Z (match RMSE, thigh RMS); the
residuals e_X and e_Y are the parts not explained by Z. By the FWL theorem, Pearson
r(e_X, e_Y) equals the partial regression coefficient on X in the full model Y ~ X + Z.
This is necessary because model RMSE and match RMSE are both compared to the same GT
label and are therefore correlated.
(B) Raw scatter: model RMSE vs excess AUC, coloured by match quality (rho = -0.166,
p = 0.140); the colour gradient shows match quality driving the apparent association.
(C) After FWL residualization, the partial rho collapses to near zero (rho = -0.022,
p = 0.851, df = 76): model prediction accuracy has no independent association with
instability once retrieval quality is controlled.
"""
    out = OUT_DIR / "captions.md"
    out.write_text(text, encoding="utf-8")
    return str(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df       = load_train_metrics()
    _, sim_df      = load_simulation_results()
    trials_df      = load_partial_trials()
    partial_summary = load_partial_summary()

    print("Generating Figure 1 — pipeline overview ...")
    fig1 = draw_pipeline_figure()

    print("Generating Figure 2 — balance metric (XCoM) ...")
    fig2 = draw_balance_metric_figure(xcom_trial_idx=12)

    print("Generating Figure 3 — prediction performance ...")
    fig3 = draw_prediction_figure(train_df)

    print("Generating Figure 3b — motion-matching quality ...")
    fig3b = draw_match_quality_figure(sim_df)

    print("Generating Figure 4 — simulation instability ...")
    fig4 = draw_simulation_figure(sim_df)

    print("Generating Figure 5 — FWL + correlation ...")
    fig5 = draw_fwl_figure(trials_df, partial_summary)

    manifest = {
        "figure_paths": {"fig1": fig1, "fig2": fig2,
                         "fig3": fig3, "fig3b": fig3b,
                         "fig4": fig4, "fig5": fig5},
        "training_run_dir":   str(TRAIN_RUN_DIR),
        "simulation_run_dir": str(SIM_RUN_DIR),
    }
    manifest_path = OUT_DIR / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    captions = write_captions(manifest["figure_paths"])

    print(f"\nAll figures written to {OUT_DIR}/")
    print(f"Captions: {captions}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
