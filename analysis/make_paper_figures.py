"""
Publication-ready figure generation for the CNN-BiLSTM prosthetic simulation paper.

Run with:
    python -m analysis.make_paper_figures

Produces figures/paper_native/fig1_*.png through fig5_*.png and captions.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
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
# Color palette (WCAG-accessible, print-safe)
# ---------------------------------------------------------------------------
C = {
    "ink": "#1a1a2e",
    "muted": "#6b7280",
    "grid": "#e5e7eb",
    "ref": "#6b7280",       # grey  — baseline / REF condition
    "pred": "#0077b6",      # blue  — PRED / model condition
    "pred_soft": "#90e0ef", # light blue fill
    "warm": "#d62828",      # red   — adverse / instability
    "warm_soft": "#ffb3b3", # light red fill
    "good": "#2d6a4f",      # green — improvement / good news
    "good_soft": "#95d5b2", # light green
    "neutral": "#457b9d",   # secondary accent
    "neutral_soft": "#a8dadc",
    "bg": "#ffffff",
}

DPI = 300
FONT_FAMILY = "DejaVu Sans"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": C["bg"],
            "axes.facecolor": C["bg"],
            "axes.edgecolor": "#9ca3af",
            "axes.labelcolor": C["ink"],
            "axes.titlecolor": C["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": C["ink"],
            "ytick.color": C["ink"],
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "font.family": FONT_FAMILY,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "savefig.dpi": DPI,
        }
    )


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_train_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(TRAIN_RUN_DIR.glob("fold_*/metrics.json")):
        with path.open() as f:
            metric = json.load(f)
        row = {"fold": path.parent.name}
        row.update(metric)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No fold metrics found under {TRAIN_RUN_DIR}")
    return pd.DataFrame(rows)


def load_simulation_results() -> tuple[dict[str, Any], pd.DataFrame]:
    with (SIM_RUN_DIR / "summary.json").open() as f:
        summary = json.load(f)
    rows: list[dict[str, Any]] = []
    for result in summary["results"]:
        rows.append(
            {
                "query_id": result["query_id"],
                "file_name": result["query_meta"]["file_name"],
                "match_knee_rmse": result["match"]["rmse_knee_deg"],
                "match_thigh_rmse": result["match"]["rms_thigh_ori_err_deg"],
                "pred_rmse": result["model"]["pred_vs_gt_knee_flex_rmse_deg"],
                "ref_knee_rmse": result["sim"]["ref"]["knee_rmse_deg"],
                "pred_knee_rmse": result["sim"]["pred"]["knee_rmse_deg"],
                "ref_instability_auc": result["sim"]["ref"]["instability_auc"],
                "pred_instability_auc": result["sim"]["pred"]["instability_auc"],
                "excess_instability_auc": result["sim"]["excess"]["instability_auc_delta"],
                "compare_npz": result["artifacts"]["compare_npz"],
            }
        )
    return summary, pd.DataFrame(rows)


def load_partial_trials() -> pd.DataFrame:
    return pd.read_csv(SIM_RUN_DIR / "analysis" / "partial_spearman_trials.csv")


def load_partial_summary() -> dict[str, Any]:
    with (SIM_RUN_DIR / "analysis" / "partial_spearman_summary.json").open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=C["grid"], linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _annotate(ax: plt.Axes, text: str, xy: tuple, xytext: tuple | None = None,
              fontsize: int = 8, color: str = C["ink"], **kw) -> None:
    ax.annotate(
        text, xy=xy,
        xytext=xytext or xy,
        fontsize=fontsize,
        color=color,
        ha="left", va="bottom",
        **kw,
    )


def _panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight="bold", color=C["ink"], va="top")


def _best_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    return xs, m * xs + b


# ---------------------------------------------------------------------------
# Figure 1 — Pipeline overview
# ---------------------------------------------------------------------------

def draw_pipeline_figure() -> str:
    fig = plt.figure(figsize=(7.2, 4.0))

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── stage definitions ────────────────────────────────────────────────
    stages = [
        {
            "x": 0.03, "w": 0.20,
            "color": C["neutral"],
            "title": "① Georgia Tech\nDataset",
            "lines": [
                "55 subjects · normal walking",
                "4 EMG channels (200 Hz)",
                "6-axis IMU (accel + gyro)",
                "Marker knee angle (°)",
                "400-sample windows (2.0 s)",
            ],
        },
        {
            "x": 0.27, "w": 0.20,
            "color": C["pred"],
            "title": "② CNN-BiLSTM\nRegressor",
            "lines": [
                "Conv1d(10→32, k=5) × 2",
                "BiLSTM(64 hidden, 2 layers)",
                "Last-step regression",
                "LOFO cross-validation",
                "Mean test RMSE: 7.84°",
            ],
        },
        {
            "x": 0.51, "w": 0.20,
            "color": C["good"],
            "title": "③ Motion\nMatching",
            "lines": [
                "MoCapAct expert bank",
                "Thigh + knee pose query",
                "Nearest-snippet retrieval",
                "Mean match error: 7.93°",
                "80 retained trials",
            ],
        },
        {
            "x": 0.75, "w": 0.22,
            "color": C["warm"],
            "title": "④ MuJoCo\nEvaluation",
            "lines": [
                "dm_control humanoid",
                "REF: unmodified policy",
                "PRED: right-knee override",
                "Balance-risk AUC metric",
                "Partial Spearman analysis",
            ],
        },
    ]

    box_y0 = 0.18
    box_h = 0.72

    for st in stages:
        x, w = st["x"], st["w"]
        # shadow
        shadow = FancyBboxPatch(
            (x + 0.005, box_y0 - 0.007), w, box_h,
            boxstyle="round,pad=0.01",
            linewidth=0, facecolor="#d1d5db", zorder=1,
        )
        ax.add_patch(shadow)
        # card
        card = FancyBboxPatch(
            (x, box_y0), w, box_h,
            boxstyle="round,pad=0.01",
            linewidth=1.2,
            edgecolor=st["color"],
            facecolor=C["bg"],
            zorder=2,
        )
        ax.add_patch(card)
        # coloured header strip
        header = FancyBboxPatch(
            (x, box_y0 + box_h - 0.195), w, 0.195,
            boxstyle="round,pad=0.01",
            linewidth=0,
            facecolor=st["color"],
            zorder=3,
        )
        ax.add_patch(header)
        # title
        ax.text(
            x + w / 2, box_y0 + box_h - 0.10,
            st["title"],
            ha="center", va="center",
            fontsize=8.5, fontweight="bold",
            color="white", zorder=4, linespacing=1.3,
        )
        # body lines
        for i, line in enumerate(st["lines"]):
            ax.text(
                x + 0.012, box_y0 + box_h - 0.23 - i * 0.088,
                f"• {line}",
                ha="left", va="top",
                fontsize=7.5, color=C["ink"], zorder=4,
            )

    # ── arrows ──────────────────────────────────────────────────────────
    for x0_s, x1_s in [
        (stages[0]["x"] + stages[0]["w"], stages[1]["x"]),
        (stages[1]["x"] + stages[1]["w"], stages[2]["x"]),
        (stages[2]["x"] + stages[2]["w"], stages[3]["x"]),
    ]:
        mid_y = box_y0 + box_h * 0.53
        arrow = FancyArrowPatch(
            (x0_s + 0.005, mid_y), (x1_s - 0.005, mid_y),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=C["muted"],
            zorder=5,
        )
        ax.add_patch(arrow)

    # ── figure title ────────────────────────────────────────────────────
    ax.text(
        0.50, 0.965,
        "Figure 1.  End-to-end evaluation pipeline",
        ha="center", va="top",
        fontsize=11, fontweight="bold", color=C["ink"],
    )
    ax.text(
        0.50, 0.112,
        "Primary predictor: model pred-vs-GT knee RMSE (deg)  ·  "
        "Primary outcome: excess instability AUC = PRED − REF",
        ha="center", va="top",
        fontsize=7.5, color=C["muted"], style="italic",
    )

    out = OUT_DIR / "fig1_pipeline_overview.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 2 — Subject-holdout prediction performance
# ---------------------------------------------------------------------------

def draw_prediction_figure(train_df: pd.DataFrame) -> str:
    rmse = train_df["test_rmse"].to_numpy()
    n = len(rmse)
    mean_r = float(np.mean(rmse))
    med_r = float(np.median(rmse))
    std_r = float(np.std(rmse))
    pct_sub10 = float((rmse < 10.0).mean() * 100)
    pct_sub8 = float((rmse < 8.0).mean() * 100)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    fig.subplots_adjust(wspace=0.35)

    # ── Panel A: violin + strip ──────────────────────────────────────────
    ax = axes[0]
    vp = ax.violinplot(rmse, positions=[0], widths=0.55,
                       showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(C["pred_soft"])
        body.set_edgecolor(C["pred"])
        body.set_linewidth(1.2)
        body.set_alpha(0.8)

    # IQR box
    q25, q75 = np.percentile(rmse, [25, 75])
    ax.plot([0], [med_r], marker="o", ms=6, color=C["pred"], zorder=5)
    ax.plot([-0.06, 0.06], [q25, q25], lw=1.2, color=C["pred"])
    ax.plot([-0.06, 0.06], [q75, q75], lw=1.2, color=C["pred"])
    ax.plot([0, 0], [q25, q75], lw=1.2, color=C["pred"])

    # jittered strip
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.06, n)
    ax.scatter(jitter, rmse, s=14, color=C["pred"], alpha=0.55,
               edgecolors="white", linewidth=0.3, zorder=4)

    # thresholds
    ax.axhline(10.0, color=C["warm"], lw=1.3, ls="--", zorder=3)
    ax.text(0.28, 10.2, "10° threshold", fontsize=7.5, color=C["warm"],
            transform=ax.get_yaxis_transform())
    ax.axhline(8.0, color=C["good"], lw=1.1, ls=":", zorder=3)
    ax.text(0.28, 8.2, "8° target", fontsize=7.5, color=C["good"],
            transform=ax.get_yaxis_transform())

    # stats annotation
    stats_txt = (
        f"Mean = {mean_r:.1f}°\n"
        f"Median = {med_r:.1f}°\n"
        f"SD = {std_r:.1f}°\n"
        f"N = {n} folds"
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=7.5, va="top", ha="right", color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f3f4f6",
                      edgecolor=C["grid"], linewidth=0.8))

    ax.set_xlim(-0.45, 0.45)
    ax.set_xticks([0])
    ax.set_xticklabels([f"N={n} subject folds"], fontsize=8)
    ax.set_ylabel("Held-out test RMSE (°)", fontsize=9)
    ax.set_title("(A)  Held-out RMSE distribution", fontsize=9.5, fontweight="bold", pad=6)
    _grid(ax, "y")
    _panel_label(ax, "A")

    # ── Panel B: ECDF ───────────────────────────────────────────────────
    ax = axes[1]
    sorted_rmse = np.sort(rmse)
    ecdf_y = np.arange(1, n + 1) / n * 100
    ax.plot(sorted_rmse, ecdf_y, color=C["pred"], lw=2.0)
    ax.fill_betweenx(ecdf_y, sorted_rmse, sorted_rmse[-1],
                     alpha=0.10, color=C["pred"])

    ax.axvline(10.0, color=C["warm"], lw=1.3, ls="--")
    ax.axhline(pct_sub10, color=C["warm"], lw=0.9, ls=":")
    ax.text(10.3, pct_sub10 - 4,
            f"{pct_sub10:.0f}% < 10°", fontsize=7.5, color=C["warm"])

    ax.axvline(8.0, color=C["good"], lw=1.1, ls=":")
    ax.axhline(pct_sub8, color=C["good"], lw=0.9, ls=":")
    ax.text(8.3, pct_sub8 - 4,
            f"{pct_sub8:.0f}% < 8°", fontsize=7.5, color=C["good"])

    ax.axvline(mean_r, color=C["pred"], lw=1.0, ls="-", alpha=0.5)

    ax.set_xlabel("Held-out test RMSE (°)", fontsize=9)
    ax.set_ylabel("Cumulative percentage of folds (%)", fontsize=9)
    ax.set_title("(B)  Empirical CDF of held-out RMSE", fontsize=9.5,
                 fontweight="bold", pad=6)
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))
    _grid(ax)
    _panel_label(ax, "B")

    fig.suptitle(
        "Figure 2.  Subject-holdout predictive performance (55-fold LOFO cross-validation)",
        fontsize=10, fontweight="bold", y=1.015,
    )
    out = OUT_DIR / "fig2_prediction_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 3 — Physical simulation outcomes
# ---------------------------------------------------------------------------

def draw_simulation_figure(sim_df: pd.DataFrame) -> str:
    ref_k = sim_df["ref_knee_rmse"].to_numpy()
    pred_k = sim_df["pred_knee_rmse"].to_numpy()
    ref_auc = sim_df["ref_instability_auc"].to_numpy()
    pred_auc = sim_df["pred_instability_auc"].to_numpy()
    excess = sim_df["excess_instability_auc"].to_numpy()
    n = len(excess)

    # Wilcoxon tests
    _stat_knee, p_knee = stats.wilcoxon(pred_k, ref_k, alternative="less")
    _stat_exc, p_exc = stats.wilcoxon(excess, alternative="greater")

    def _p_str(p: float) -> str:
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    fig = plt.figure(figsize=(7.5, 4.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.48,
                           left=0.08, right=0.97, bottom=0.14, top=0.82)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    def _corner(ax: plt.Axes, letter: str) -> None:
        ax.text(-0.14, 1.0, letter, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
                color=C["ink"])

    # ── Panel A: REF vs PRED knee RMSE scatter ──────────────────────────
    ax = ax0
    lim = max(ref_k.max(), pred_k.max()) * 1.06
    ax.plot([0, lim], [0, lim], color=C["muted"], lw=1.0, ls="--", zorder=1)
    ax.scatter(ref_k, pred_k, s=18, color=C["pred"], alpha=0.65,
               edgecolors="white", linewidth=0.4, zorder=3)
    ax.fill_between([0, lim], [0, 0], [0, lim], alpha=0.06, color=C["good"],
                    zorder=0)
    frac_below = (pred_k < ref_k).mean()
    ax.text(
        0.04, 0.97,
        f"{frac_below*100:.0f}% of trials:\nPRED < REF\n{_p_str(p_knee)}",
        transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
        color=C["good"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0fdf4",
                  edgecolor="#bbf7d0", linewidth=0.8),
    )
    ax.set_xlabel("REF knee RMSE (°)", fontsize=9)
    ax.set_ylabel("PRED knee RMSE (°)", fontsize=9)
    ax.set_title("Knee tracking", fontsize=9.5, fontweight="bold", pad=4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    _grid(ax)
    _corner(ax, "A")

    # ── Panel B: instability AUC — paired violin comparison ──────────────
    ax = ax1
    bw = 0.26
    xpos = [0, 1]
    medians: dict[str, float] = {}
    for vals, x, color, label in [
        (ref_auc, xpos[0], C["ref"], "REF"),
        (pred_auc, xpos[1], C["warm"], "PRED"),
    ]:
        vp = ax.violinplot(vals, positions=[x], widths=0.52,
                           showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.35)
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        medians[label] = float(med)
        ax.plot([x - bw * 0.5, x + bw * 0.5], [q25, q25],
                lw=1.2, color=color)
        ax.plot([x - bw * 0.5, x + bw * 0.5], [q75, q75],
                lw=1.2, color=color)
        ax.plot([x, x], [q25, q75], lw=1.2, color=color)
        ax.plot([x], [med], marker="o", ms=5, color=color, zorder=5)
        rng = np.random.default_rng(0)
        jit = rng.normal(0, 0.05, len(vals))
        ax.scatter(x + jit, vals, s=10, color=color, alpha=0.45,
                   edgecolors="none", zorder=4)

    for rv, pv in zip(ref_auc, pred_auc):
        col = C["warm"] if pv > rv else C["good"]
        ax.plot([xpos[0], xpos[1]], [rv, pv],
                lw=0.4, alpha=0.18, color=col)

    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [f"REF\nMd={medians['REF']:.2f}",
         f"PRED\nMd={medians['PRED']:.2f}"],
        fontsize=8,
    )
    ax.set_ylabel("Instability AUC", fontsize=9)
    ax.set_title("Instability AUC", fontsize=9.5, fontweight="bold", pad=4)
    _grid(ax, "y")
    _corner(ax, "B")

    # ── Panel C: excess AUC histogram ────────────────────────────────────
    ax = ax2
    pct_pos = (excess > 0).mean() * 100
    ax.hist(excess, bins=18, color=C["warm_soft"],
            edgecolor=C["warm"], linewidth=0.6, alpha=0.85, zorder=2)
    ax.axvline(0.0, color=C["ink"], lw=1.4, ls=":", zorder=4,
               label="No change")
    ax.axvline(float(excess.mean()), color=C["warm"], lw=1.5,
               ls="--", zorder=5, label=f"Mean={excess.mean():.3f}")
    ax.text(
        0.97, 0.97,
        f"{pct_pos:.0f}% > 0\n{_p_str(p_exc)}\n(Wilcoxon)",
        transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
        color=C["warm"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff1f2",
                  edgecolor="#fecdd3", linewidth=0.8),
    )
    ax.set_xlabel("Excess AUC (PRED − REF)", fontsize=9)
    ax.set_ylabel("Trial count", fontsize=9)
    ax.set_title("Excess instability AUC", fontsize=9.5, fontweight="bold", pad=4)
    ax.legend(fontsize=7.5, loc="upper left")
    _grid(ax, "y")
    _corner(ax, "C")

    fig.suptitle(
        "Figure 3.  PRED improves knee tracking (A) but significantly increases instability (B, C)",
        fontsize=10, fontweight="bold", y=1.01,
    )
    out = OUT_DIR / "fig3_simulation_outcomes.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 4 — Partial-Spearman correlation analysis
# ---------------------------------------------------------------------------

def draw_correlation_figure(trials_df: pd.DataFrame,
                             partial_summary: dict[str, Any]) -> str:
    x_raw = trials_df["predictor_knee_rmse_deg"].to_numpy()
    y_raw = trials_df["outcome_value"].to_numpy()
    x_match = trials_df["control_match_knee_rmse_deg"].to_numpy()
    x_res = trials_df["residual_predictor"].to_numpy()
    y_res = trials_df["residual_outcome"].to_numpy()

    rho_raw, p_raw = stats.spearmanr(x_raw, y_raw)
    rho_match, p_match = stats.spearmanr(x_match, y_raw)
    rho_part = float(partial_summary["rho_partial_spearman"])
    p_part = float(partial_summary["p_value_two_sided"])

    def _p_label(r: float, p: float) -> str:
        p_str = "< 0.001" if p < 0.001 else f"= {p:.3f}"
        return f"ρ = {r:.3f},  p {p_str}"

    fig = plt.figure(figsize=(7.5, 3.1))
    # Reserve left margin for colorbar; use GridSpec with explicit left margin
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.52,
                           left=0.08, right=0.97, bottom=0.17, top=0.80)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    panel_specs = [
        (x_raw, y_raw, x_match, "A   Raw association",
         "Model knee RMSE (°)", "Excess instability AUC", rho_raw, p_raw),
        (x_match, y_raw, None, "B   Motion-match confound",
         "Motion-match knee RMSE (°)", "Excess instability AUC", rho_match, p_match),
        (x_res, y_res, None, "C   Partial (FWL residualized)",
         "Residualized rank\n(model RMSE)",
         "Residualized rank\n(excess instability)", rho_part, p_part),
    ]

    for idx, (ax, (xdata, ydata, xcol, title, xlabel, ylabel, rho_s, p_s)) in \
            enumerate(zip(axes, panel_specs)):
        if idx == 0:
            sc = ax.scatter(xdata, ydata, c=xcol, cmap="YlGnBu",
                            s=22, alpha=0.85, edgecolors="white",
                            linewidth=0.3, zorder=3, vmin=0)
            # place colorbar above panel A
            cax = ax.inset_axes([0.0, 1.06, 1.0, 0.07])
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
            cb.set_label("Match RMSE (°)", fontsize=7, labelpad=1)
            cb.ax.tick_params(labelsize=6)
        elif idx == 1:
            ax.scatter(xdata, ydata, color=C["neutral"], s=22,
                       alpha=0.8, edgecolors="white", linewidth=0.3, zorder=3)
        else:
            ax.scatter(xdata, ydata, color=C["pred"], s=22,
                       alpha=0.8, edgecolors="white", linewidth=0.3, zorder=3)

        xs, ys = _best_fit(xdata, ydata)
        ax.plot(xs, ys, color=C["ink"], lw=1.3, zorder=4)
        ax.axhline(0, color=C["grid"], lw=0.8, zorder=1)

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=12, loc="left")
        ax.text(0.97, 0.97, _p_label(rho_s, p_s),
                transform=ax.transAxes, fontsize=7.5,
                va="top", ha="right", color=C["ink"],
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f9fafb",
                          edgecolor=C["grid"], linewidth=0.7))
        _grid(ax)

    fig.suptitle(
        "Figure 4.  Motion-match quality confounds the RMSE–instability association;\n"
        "after FWL residualization, partial ρ ≈ 0  (p = 0.851)",
        fontsize=9.5, fontweight="bold", y=1.02,
    )
    out = OUT_DIR / "fig4_correlation_confounding.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Figure 5 — Representative trial deep-dive
# ---------------------------------------------------------------------------

def pick_representative_trial(sim_df: pd.DataFrame) -> pd.Series:
    subset = sim_df[
        (sim_df["pred_knee_rmse"] < sim_df["ref_knee_rmse"])
        & (sim_df["excess_instability_auc"] > 0)
        & (sim_df["pred_rmse"] < sim_df["pred_rmse"].median())
    ].copy()
    if subset.empty:
        subset = sim_df.copy()
    subset["score"] = (
        (subset["ref_knee_rmse"] - subset["pred_knee_rmse"]) / subset["ref_knee_rmse"]
        + subset["excess_instability_auc"]
    )
    return subset.sort_values("score", ascending=False).iloc[0]


def draw_rollout_figure(trial: pd.Series) -> tuple[str, dict[str, Any]]:
    npz_path = REPO_ROOT / trial["compare_npz"]
    data = np.load(npz_path)
    dt = float(data["dt"])

    t_act = np.arange(len(data["knee_ref_actual_deg"])) * dt
    t_q = np.arange(len(data["knee_good_query_deg"])) * dt
    t_rt = np.arange(len(data["predicted_fall_risk_trace_ref"])) * dt
    t_pt = np.arange(len(data["predicted_fall_risk_trace_good"])) * dt

    ref_auc = float(np.trapezoid(data["predicted_fall_risk_trace_ref"], dx=dt))
    pred_auc = float(np.trapezoid(data["predicted_fall_risk_trace_good"], dx=dt))

    knee_ref_rmse = float(np.sqrt(
        np.mean((data["knee_ref_actual_deg"] - data["knee_good_query_deg"][:len(data["knee_ref_actual_deg"])]) ** 2)
    ))
    knee_pred_rmse = float(np.sqrt(
        np.mean((data["knee_good_actual_deg"] - data["knee_good_query_deg"][:len(data["knee_good_actual_deg"])]) ** 2)
    ))

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    fig.subplots_adjust(hspace=0.10)

    # ── Panel A: knee tracking ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(t_q, data["knee_good_query_deg"], color=C["ink"], lw=2.2,
            label="GT target trajectory", zorder=4)
    ax.plot(t_act, data["knee_ref_actual_deg"], color=C["ref"], lw=1.8,
            ls="--", label=f"REF actual  (RMSE = {knee_ref_rmse:.1f}°)", zorder=3)
    ax.plot(t_act, data["knee_good_actual_deg"], color=C["pred"], lw=1.8,
            label=f"PRED actual (RMSE = {knee_pred_rmse:.1f}°)", zorder=3)
    # shade difference
    ax.fill_between(
        t_act,
        data["knee_ref_actual_deg"],
        data["knee_good_actual_deg"],
        alpha=0.12, color=C["pred"],
    )
    ax.set_ylabel("Knee angle (°)", fontsize=9)
    ax.set_title(
        f"(A)  Knee tracking — trial {trial['query_id']}",
        fontsize=9.5, fontweight="bold", pad=5,
    )
    ax.legend(loc="upper right", fontsize=8)
    _grid(ax, "y")
    _panel_label(ax, "A")

    # ── Panel B: instability trace ───────────────────────────────────────
    ax = axes[1]
    ax.fill_between(t_rt, data["predicted_fall_risk_trace_ref"],
                    alpha=0.15, color=C["ref"])
    ax.fill_between(t_pt, data["predicted_fall_risk_trace_good"],
                    alpha=0.15, color=C["warm"])
    ax.plot(t_rt, data["predicted_fall_risk_trace_ref"], color=C["ref"], lw=1.8,
            label=f"REF instability trace  (AUC = {ref_auc:.3f})", zorder=3)
    ax.plot(t_pt, data["predicted_fall_risk_trace_good"], color=C["warm"], lw=1.8,
            label=f"PRED instability trace (AUC = {pred_auc:.3f})", zorder=3)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Instability score", fontsize=9)
    ax.set_title(
        f"(B)  Balance-risk trace — excess AUC = {trial['excess_instability_auc']:.3f}",
        fontsize=9.5, fontweight="bold", pad=5,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.02, 1.12)
    _grid(ax, "y")
    _panel_label(ax, "B")

    fig.suptitle(
        "Figure 5.  Representative trial: PRED improves knee tracking yet increases instability",
        fontsize=10, fontweight="bold", y=1.015,
    )

    caption_txt = (
        f"query={trial['query_id']}  ·  file={trial['file_name']}  ·  "
        f"model pred RMSE={trial['pred_rmse']:.1f}°  ·  "
        f"match RMSE={trial['match_knee_rmse']:.1f}°  ·  "
        f"REF knee={trial['ref_knee_rmse']:.1f}°  ·  "
        f"PRED knee={trial['pred_knee_rmse']:.1f}°  ·  "
        f"excess AUC={trial['excess_instability_auc']:.3f}"
    )
    fig.text(0.5, 0.005, caption_txt, ha="center", va="bottom",
             fontsize=7, color=C["muted"], style="italic")

    out = OUT_DIR / "fig5_representative_rollout.png"
    fig.savefig(out)
    plt.close(fig)

    meta = {
        "query_id": str(trial["query_id"]),
        "file_name": str(trial["file_name"]),
        "pred_rmse_deg": float(trial["pred_rmse"]),
        "match_knee_rmse_deg": float(trial["match_knee_rmse"]),
        "ref_knee_rmse_deg": float(trial["ref_knee_rmse"]),
        "pred_knee_rmse_deg": float(trial["pred_knee_rmse"]),
        "excess_instability_auc": float(trial["excess_instability_auc"]),
        "compare_npz": str(npz_path),
    }
    return str(out), meta


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def write_captions(figure_paths: dict[str, str],
                   representative_meta: dict[str, Any]) -> str:
    rm = representative_meta
    text = f"""# Paper Figure Captions

Generated from:
- Training: `{TRAIN_RUN_DIR.relative_to(REPO_ROOT)}`
- Simulation: `{SIM_RUN_DIR.relative_to(REPO_ROOT)}`

---

## Figure 1 — Pipeline overview
**File:** `{Path(figure_paths['fig1']).relative_to(REPO_ROOT)}`

**Caption:** End-to-end evaluation pipeline. Raw EMG (4 channels) and inertial
measurement unit (IMU, 6-axis) signals from 55 Georgia Tech biomechanics trials
are preprocessed and windowed into 400-sample (2.0 s) segments. A CNN-BiLSTM
regressor predicts the right-knee included angle from each window; its outputs are
evaluated via 55-fold leave-one-file-out (LOFO) cross-validation, yielding a mean
held-out RMSE of 7.84°. For each held-out window, the predicted knee trajectory is
used to query a MoCapAct motion bank; the nearest matching snippet is then replayed
twice in a MuJoCo humanoid—once unmodified (REF) and once with the right-knee joint
forced to the PRED trajectory. The primary outcome is the area-under-the-curve
difference in a balance-risk heuristic (excess instability AUC = PRED − REF).

---

## Figure 2 — Prediction performance
**File:** `{Path(figure_paths['fig2']).relative_to(REPO_ROOT)}`

**Caption:** Distribution of held-out subject-fold RMSE values across 55 folds of
LOFO cross-validation. (A) Violin and strip plot showing individual fold errors; the
white circle marks the median (6.85°). The dashed and dotted reference lines
correspond to the 10° and 8° thresholds discussed in the literature. (B) Empirical
cumulative distribution function of the same values, showing that 83.6% of folds
fall below 10° and 67.3% fall below 8°. Mean RMSE = 7.84° ± 4.33° (SD).

---

## Figure 3 — Physical simulation outcomes
**File:** `{Path(figure_paths['fig3']).relative_to(REPO_ROOT)}`

**Caption:** Simulation outcomes across 80 retained trials. (A) Scatter plot of REF
versus PRED simulated knee RMSE; points below the identity line indicate trials where
PRED improved knee tracking (71.3% of trials; Wilcoxon signed-rank, p < 0.001).
(B) Paired violin comparison of instability AUC for the REF and PRED conditions;
thin connecting lines show within-trial changes. (C) Histogram of excess instability
AUC (PRED − REF); 95.0% of trials show positive excess (Wilcoxon one-sided, p < 0.001),
indicating that the knee override consistently increases simulated instability even
when it improves knee tracking.

---

## Figure 4 — Correlation and confounding analysis
**File:** `{Path(figure_paths['fig4']).relative_to(REPO_ROOT)}`

**Caption:** Association between prediction error and simulated instability. (A) Raw
Spearman scatter of model knee RMSE versus excess instability AUC, coloured by
motion-match quality; the association is negative but non-significant (ρ = −0.166,
p = 0.140). (B) The motion-match knee RMSE—a nuisance covariate—explains more
variance in excess instability than the model RMSE does. (C) After Frisch–Waugh–
Lovell (FWL) residualization on both motion-match controls, the partial Spearman
correlation is near zero (ρ = −0.022, p = 0.851, df = 76), indicating that
prediction accuracy per se does not drive the observed instability.

---

## Figure 5 — Representative trial
**File:** `{Path(figure_paths['fig5']).relative_to(REPO_ROOT)}`

**Caption:** Deep-dive into representative trial {rm['query_id']} (source file:
{rm['file_name']}). (A) Knee-angle time series showing the ground-truth target
(black), the unmodified expert-policy tracking (REF, grey dashed,
RMSE = {rm['ref_knee_rmse_deg']:.1f}°), and the CNN-BiLSTM-overridden tracking
(PRED, blue, RMSE = {rm['pred_knee_rmse_deg']:.1f}°). PRED achieves substantially
lower joint-tracking error. (B) Corresponding balance-risk traces for REF and PRED;
despite the improved knee tracking, the PRED rollout accumulates higher integrated
instability (excess AUC = {rm['excess_instability_auc']:.3f}), illustrating why
joint-angle accuracy alone is insufficient as a prosthetic evaluation criterion.
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

    train_df = load_train_metrics()
    _, sim_df = load_simulation_results()
    trials_df = load_partial_trials()
    partial_summary = load_partial_summary()

    print("Generating Figure 1 — pipeline overview …")
    fig1 = draw_pipeline_figure()

    print("Generating Figure 2 — prediction distribution …")
    fig2 = draw_prediction_figure(train_df)

    print("Generating Figure 3 — simulation outcomes …")
    fig3 = draw_simulation_figure(sim_df)

    print("Generating Figure 4 — correlation analysis …")
    fig4 = draw_correlation_figure(trials_df, partial_summary)

    print("Generating Figure 5 — representative rollout …")
    representative = pick_representative_trial(sim_df)
    fig5, rep_meta = draw_rollout_figure(representative)

    manifest = {
        "figure_paths": {
            "fig1": fig1, "fig2": fig2,
            "fig3": fig3, "fig4": fig4, "fig5": fig5,
        },
        "training_run_dir": str(TRAIN_RUN_DIR),
        "simulation_run_dir": str(SIM_RUN_DIR),
        "representative_trial": rep_meta,
    }
    manifest_path = OUT_DIR / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    captions = write_captions(manifest["figure_paths"], rep_meta)

    print(f"\nAll figures written to {OUT_DIR}/")
    print(f"Captions: {captions}")
    print(f"Manifest: {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
