"""
Publication figures for the CNN-BiLSTM prosthetic knee paper.

Run:
    python -m analysis.make_paper_figures

Outputs: figures/paper_native/fig1–fig5_*.png + captions.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUN = REPO_ROOT / "checkpoints" / "tst_20260405_173725_all"
SIM_RUN   = REPO_ROOT / "artifacts"   / "phys_eval_v2" / "runs" / "20260406_205003"
OUT_DIR   = REPO_ROOT / "figures"     / "paper_native"

# ---------------------------------------------------------------------------
# Palette  (print-safe, colourblind-friendly)
# ---------------------------------------------------------------------------
REF_COL  = "#555555"   # neutral grey  — REF condition
PRED_COL = "#c0392b"   # red           — PRED / instability
MOD_COL  = "#2980b9"   # blue          — model / predictor
GOOD_COL = "#27ae60"   # green
NEUT_COL = "#7f8c8d"   # muted
INK      = "#1a1a2e"
GRID_C   = "#e0e0e0"
BG       = "#ffffff"
DPI      = 300


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   BG,
        "axes.facecolor":     BG,
        "axes.edgecolor":     "#aaaaaa",
        "axes.labelcolor":    INK,
        "axes.titlecolor":    INK,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        INK,
        "ytick.color":        INK,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
        "font.family":        "DejaVu Sans",
        "font.size":          9,
        "axes.titlesize":     10,
        "axes.labelsize":     9,
        "legend.fontsize":    8,
        "legend.frameon":     False,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,
        "savefig.dpi":        DPI,
    })


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _train_df() -> pd.DataFrame:
    rows: list[dict] = []
    for p in sorted(TRAIN_RUN.glob("fold_*/metrics.json")):
        m = json.loads(p.read_text())
        m["fold"] = p.parent.name
        rows.append(m)
    if not rows:
        raise RuntimeError(f"No fold metrics under {TRAIN_RUN}")
    return pd.DataFrame(rows)


def _sim_df() -> tuple[dict, pd.DataFrame]:
    summary = json.loads((SIM_RUN / "summary.json").read_text())
    rows: list[dict] = []
    for r in summary["results"]:
        rows.append({
            "query_id":        r["query_id"],
            "pred_rmse":       r["model"]["pred_vs_gt_knee_flex_rmse_deg"],
            "match_knee_rmse": r["match"]["rmse_knee_deg"],
            "match_thigh_rmse":r["match"]["rms_thigh_ori_err_deg"],
            "ref_auc":         r["sim"]["ref"]["instability_auc"],
            "pred_auc":        r["sim"]["pred"]["instability_auc"],
            "excess_auc":      r["sim"]["excess"]["instability_auc_delta"],
            "ref_loss_step":   r["sim"]["ref"]["balance_loss_step"],
            "pred_loss_step":  r["sim"]["pred"]["balance_loss_step"],
            "compare_npz":     r["artifacts"]["compare_npz"],
        })
    return summary, pd.DataFrame(rows)


def _partial_df() -> pd.DataFrame:
    return pd.read_csv(SIM_RUN / "analysis" / "partial_spearman_trials.csv")


def _partial_summary() -> dict:
    return json.loads(
        (SIM_RUN / "analysis" / "partial_spearman_summary.json").read_text())


def _load_npz(sim_df: pd.DataFrame, trial_idx: int) -> dict[str, Any]:
    npz = os.path.normpath(sim_df.iloc[trial_idx]["compare_npz"])
    return dict(np.load(npz))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _grid(ax: plt.Axes, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID_C, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _panel(ax: plt.Axes, letter: str,
           x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=14, fontweight="bold", color=INK,
            va="top", ha="left")


def _best_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    return xs, m * xs + b


def _rho_str(rho: float, p: float) -> str:
    ps = "< 0.001" if p < 0.001 else f"= {p:.3f}"
    return f"rho = {rho:+.3f}\np {ps}"


# ---------------------------------------------------------------------------
# Figure 1 — Pipeline schematic
# ---------------------------------------------------------------------------

def fig1_pipeline() -> str:
    fig = plt.figure(figsize=(10.2, 2.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03, 0.92,
        "Figure 1. Evaluation pipeline used in the native-rate benchmark",
        ha="left", va="top", fontsize=11, fontweight="bold", color=INK
    )
    ax.text(
        0.03, 0.85,
        "Data → preprocessing → CNN-BiLSTM prediction → motion matching → paired MuJoCo rollouts → excess-instability analysis",
        ha="left", va="top", fontsize=8.6, color=NEUT_COL
    )

    stages = [
        ("1", "Georgia Tech data", ["55 subjects", "4 EMG + 6 thigh IMU", "knee-angle label"]),
        ("2", "Preprocessing", ["EMG: 20 Hz high-pass", "rectify + 5 Hz low-pass", "native 200 Hz windows"]),
        ("3", "CNN-BiLSTM", ["400-sample window", "55-fold LOFO", "predict 10 ms ahead"]),
        ("4", "Motion matching", ["MoCapAct bank", "thigh_knee_d score", "local refine radius 30"]),
        ("5", "Simulation + stats", ["REF vs PRED rollout", "excess instability AUC", "partial Spearman"]),
    ]

    xs = np.linspace(0.08, 0.92, len(stages))
    box_w = 0.155
    box_h = 0.42
    y = 0.27

    ax.plot([xs[0], xs[-1]], [y + box_h + 0.09, y + box_h + 0.09], color=GRID_C, linewidth=2.0, zorder=1)

    for i, (x, stage) in enumerate(zip(xs, stages)):
        num, title, lines = stage

        if i < len(stages) - 1:
            x0 = x + box_w / 2 + 0.015
            x1 = xs[i + 1] - box_w / 2 - 0.015
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y + box_h + 0.09),
                    (x1, y + box_h + 0.09),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.4,
                    color=NEUT_COL,
                    zorder=2,
                )
            )

        ax.add_patch(
            Circle((x, y + box_h + 0.09), 0.028, facecolor=MOD_COL, edgecolor="white", linewidth=1.2, zorder=3)
        )
        ax.text(x, y + box_h + 0.09, num, ha="center", va="center", fontsize=8.4, color="white", fontweight="bold", zorder=4)

        ax.add_patch(
            FancyBboxPatch(
                (x - box_w / 2, y),
                box_w,
                box_h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=BG,
                edgecolor=GRID_C,
                linewidth=1.1,
                zorder=2,
            )
        )
        ax.text(x - box_w / 2 + 0.015, y + box_h - 0.07, title, ha="left", va="top", fontsize=9.2, fontweight="bold", color=INK)
        for j, line in enumerate(lines):
            ax.text(
                x - box_w / 2 + 0.018,
                y + box_h - 0.145 - j * 0.095,
                "\u2022 " + line,
                ha="left",
                va="top",
                fontsize=7.8,
                color=INK,
            )

    ax.text(
        0.03, 0.08,
        "Primary predictor: model pred-vs-GT knee RMSE. Primary outcome: excess instability AUC = PRED - REF.",
        ha="left", va="bottom", fontsize=8.0, color=NEUT_COL
    )

    out = OUT_DIR / "fig1_pipeline.png"
    fig.savefig(out)
    plt.close()
    return str(out)


# ---------------------------------------------------------------------------
# Figure 2 — Representative trial (XCoM margin + risk trace)
# ---------------------------------------------------------------------------

def fig2_representative_trial(sim_df: pd.DataFrame,
                               trial_idx: int = 75) -> str:
    d   = _load_npz(sim_df, trial_idx)
    dt  = float(d["dt"])
    n   = len(d["predicted_fall_risk_trace_ref"])
    t   = np.arange(n) * dt

    xr  = d["balance_xcom_margin_ref_m"]
    xp  = d["balance_xcom_margin_good_m"]
    rr  = d["predicted_fall_risk_trace_ref"]
    rp  = d["predicted_fall_risk_trace_good"]

    auc_ref  = float(np.trapezoid(rr, dx=dt))
    auc_pred = float(np.trapezoid(rp, dx=dt))
    row = sim_df.iloc[trial_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2),
                                    gridspec_kw={"wspace": 0.40})

    # ── Panel A: XCoM margin ──────────────────────────────────────────────
    ax1.axhline(0, color=INK, lw=1.0, ls="--", zorder=2,
                label="BoS boundary (margin = 0)")
    ax1.fill_between(t, xr, 0, where=(xr < 0), color=REF_COL,  alpha=0.12)
    ax1.fill_between(t, xp, 0, where=(xp < 0), color=PRED_COL, alpha=0.12)
    ax1.plot(t, xr, color=REF_COL,  lw=2.0, label="REF",  zorder=3)
    ax1.plot(t, xp, color=PRED_COL, lw=2.0, label="PRED", zorder=3)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("XCoM margin (m)")
    ax1.set_title("XCoM margin", fontweight="bold",
                  pad=5, loc="left")
    ax1.legend(loc="lower left", fontsize=8)
    _grid(ax1, "y")
    _panel(ax1, "A")

    # ── Panel B: risk score + AUC ─────────────────────────────────────────
    ax2.fill_between(t, rr, alpha=0.18, color=REF_COL)
    ax2.fill_between(t, rp, alpha=0.18, color=PRED_COL)
    ax2.plot(t, rr, color=REF_COL,  lw=2.0,
             label=f"REF  (AUC = {auc_ref:.2f})", zorder=3)
    ax2.plot(t, rp, color=PRED_COL, lw=2.0,
             label=f"PRED (AUC = {auc_pred:.2f})", zorder=3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Instability score r(t)")
    ax2.set_ylim(-0.03, 1.10)
    ax2.set_title("Instability trace", fontweight="bold",
                  pad=5, loc="left")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.text(0.98, 0.05,
             f"Excess AUC = {row['excess_auc']:.3f}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8.0, color=INK,
             bbox=dict(boxstyle="round,pad=0.28",
                       facecolor="#f7f7f7", edgecolor=GRID_C, lw=0.8))
    _grid(ax2, "y")
    _panel(ax2, "B")

    fig.suptitle(
        "Figure 2. Example trial used to illustrate excess-instability scoring",
        fontsize=10, fontweight="bold", y=1.02)
    fig.text(0.98, 0.99, row["query_id"], ha="right", va="top", fontsize=8, color=NEUT_COL)

    out = OUT_DIR / "fig2_representative_trial.png"
    fig.savefig(out)
    plt.close()
    return str(out)


# ---------------------------------------------------------------------------
# Figure 2b / 6 — Representative replay frames
# ---------------------------------------------------------------------------

def fig6_simulation_frames(sim_df: pd.DataFrame,
                           trial_idx: int = 75) -> str:
    row = sim_df.iloc[trial_idx]
    gif_path = Path(os.path.normpath(str(row["compare_npz"]))).with_suffix(".gif")
    if not gif_path.exists():
        raise FileNotFoundError(f"Replay GIF not found: {gif_path}")

    gif = Image.open(gif_path)
    n_frames = int(getattr(gif, "n_frames", 1))
    picks = [0, max(0, n_frames // 3), max(0, (2 * n_frames) // 3), max(0, n_frames - 1)]
    labels = ["Start", "Early", "Mid", "Late"]

    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.9), gridspec_kw={"wspace": 0.03})
    for ax, fi, lab in zip(axes, picks, labels):
        gif.seek(int(fi))
        frame = np.asarray(gif.convert("RGB"))
        ax.imshow(frame)
        ax.set_title(lab, fontsize=8.5, pad=4, loc="left", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
            spine.set_linewidth(0.8)

    fig.suptitle(
        "Figure 6. Example simulation frames from one representative replay",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.99, 0.965,
        f"{row['query_id']}  |  REF / PRED replay frames",
        ha="right", va="top", fontsize=8, color=NEUT_COL
    )
    fig.text(
        0.01, 0.01,
        "Frames sampled from the saved compare.gif artifact. Use this figure only as a qualitative visual aid; "
        "quantitative claims come from the trace and AUC figures.",
        ha="left", va="bottom", fontsize=7.4, color=NEUT_COL
    )

    out = OUT_DIR / "fig6_simulation_frames.png"
    fig.savefig(out)
    plt.close()
    return str(out)


# ---------------------------------------------------------------------------
# Figure 3 — Prediction performance (55-fold LOFO)
# ---------------------------------------------------------------------------

def fig3_prediction(train_df: pd.DataFrame) -> str:
    rmse = np.sort(train_df["test_rmse"].to_numpy())
    n    = len(rmse)
    mu   = float(rmse.mean())
    med  = float(np.median(rmse))
    sd   = float(rmse.std())

    colors = [MOD_COL if v <= 10.0 else PRED_COL for v in rmse]

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.13)

    ax.barh(np.arange(n), rmse, color=colors, height=0.8, edgecolor="none")
    ax.axvline(mu,  color=MOD_COL, lw=1.6, ls="--",
               label=f"Mean   = {mu:.2f} deg")
    ax.axvline(med, color=MOD_COL, lw=1.6, ls=":",
               label=f"Median = {med:.2f} deg")

    ax.set_xlabel("Held-out test RMSE (deg)")
    ax.set_ylabel("Subject fold (sorted by RMSE)")
    ax.set_xlim(0, rmse.max() * 1.14)
    tick_pos = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels([str(p + 1) for p in tick_pos])
    ax.legend(loc="lower right", fontsize=8)

    # colour legend patch
    from matplotlib.patches import Patch
    ax.legend(handles=[
        plt.Line2D([], [], color=MOD_COL, lw=1.6, ls="--",
                   label=f"Mean   = {mu:.2f} deg"),
        plt.Line2D([], [], color=MOD_COL, lw=1.6, ls=":",
                   label=f"Median = {med:.2f} deg"),
        Patch(facecolor=MOD_COL,  label="RMSE <= 10 deg"),
        Patch(facecolor=PRED_COL, label="RMSE > 10 deg"),
    ], loc="lower right", fontsize=8)

    ax.text(0.98, 0.55,
            f"N = {n} folds\nSD = {sd:.2f} deg",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=INK,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#f5f5f5", edgecolor=GRID_C, lw=0.8))

    _grid(ax, "x")
    ax.set_title(
        "Figure 3. Held-out prediction error across 55 subject folds",
        fontsize=10, fontweight="bold", pad=7)

    out = OUT_DIR / "fig3_prediction_performance.png"
    fig.savefig(out)
    plt.close()
    return str(out)


# ---------------------------------------------------------------------------
# Figure 4 — Simulation instability
# ---------------------------------------------------------------------------

def fig4_simulation(sim_df: pd.DataFrame) -> str:
    ref_auc  = sim_df["ref_auc"].to_numpy()
    pred_auc = sim_df["pred_auc"].to_numpy()
    excess   = sim_df["excess_auc"].to_numpy()
    n = len(excess)

    _, p_exc = stats.wilcoxon(excess, alternative="greater")
    p_str = "p < 0.001" if p_exc < 0.001 else f"p = {p_exc:.3f}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.6),
                                    gridspec_kw={"wspace": 0.42})
    fig.subplots_adjust(bottom=0.14, top=0.84)

    # ── Panel A: paired scatter REF AUC vs PRED AUC ───────────────────────
    lo = min(ref_auc.min(), pred_auc.min()) * 0.94
    hi = max(ref_auc.max(), pred_auc.max()) * 1.06
    ax1.plot([lo, hi], [lo, hi], color=NEUT_COL, lw=1.2,
             ls="--", zorder=1, label="REF = PRED")

    above  = pred_auc > ref_auc
    ax1.scatter(ref_auc[above], pred_auc[above],
                color=PRED_COL, s=20, alpha=0.75,
                edgecolors="white", linewidth=0.3, zorder=3,
                label=f"PRED > REF  ({above.mean()*100:.0f}%)")
    ax1.scatter(ref_auc[~above], pred_auc[~above],
                color=REF_COL, s=20, alpha=0.75,
                edgecolors="white", linewidth=0.3, zorder=3,
                label=f"PRED <= REF ({(~above).mean()*100:.0f}%)")

    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal")
    ax1.set_xlabel("REF instability AUC")
    ax1.set_ylabel("PRED instability AUC")
    ax1.set_title("Paired instability AUC", fontweight="bold",
                  pad=5, loc="left")
    ax1.legend(loc="upper left", fontsize=7.5)
    _grid(ax1)
    _panel(ax1, "A")

    # ── Panel B: excess AUC histogram ─────────────────────────────────────
    pct_pos = (excess > 0).mean() * 100
    ax2.hist(excess, bins=20, color=PRED_COL, alpha=0.70,
             edgecolor="white", linewidth=0.5, zorder=2)
    ax2.axvline(0, color=INK, lw=1.4, ls=":",
                zorder=4, label="No change (0)")
    ax2.axvline(excess.mean(), color=PRED_COL, lw=2.0, ls="--",
                zorder=5, label=f"Mean = {excess.mean():.3f}")

    ax2.text(0.97, 0.97,
             f"{pct_pos:.0f}% of trials > 0\nWilcoxon {p_str}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=7.8, color=INK,
             bbox=dict(boxstyle="round,pad=0.28",
                       facecolor="#f7f7f7", edgecolor=GRID_C, lw=0.8))

    ax2.set_xlabel("Excess AUC  (PRED - REF)")
    ax2.set_ylabel("Trial count")
    ax2.legend(loc="upper left", fontsize=7.5)
    ax2.set_title("Excess instability AUC", fontweight="bold",
                  pad=5, loc="left")
    _grid(ax2, "y")
    _panel(ax2, "B")

    fig.suptitle(
        f"Figure 4. Paired simulation outcomes across {n} held-out windows",
        fontsize=10, fontweight="bold", y=1.01)

    out = OUT_DIR / "fig4_simulation_instability.png"
    fig.savefig(out)
    plt.close()
    return str(out)


# ---------------------------------------------------------------------------
# Figure 5 — FWL / partial Spearman
# ---------------------------------------------------------------------------

def fig5_correlation(trials_df: pd.DataFrame,
                     partial_sum: dict) -> str:
    x_raw   = trials_df["predictor_knee_rmse_deg"].to_numpy()
    y_raw   = trials_df["outcome_value"].to_numpy()
    x_match = trials_df["control_match_knee_rmse_deg"].to_numpy()
    x_res   = trials_df["residual_predictor"].to_numpy()
    y_res   = trials_df["residual_outcome"].to_numpy()

    rho_raw,   p_raw   = stats.spearmanr(x_raw,   y_raw)
    rho_match, p_match = stats.spearmanr(x_match, y_raw)
    rho_part = float(partial_sum["rho_partial_spearman"])
    p_part   = float(partial_sum["p_value_two_sided"])

    fig = plt.figure(figsize=(9.6, 3.5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.52,
                            left=0.07, right=0.97,
                            bottom=0.17, top=0.76)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    # (ax, xdata, ydata, colour_vals, xlabel, ylabel, letter, title, rho, p)
    panels = [
        (axes[0], x_raw,   y_raw,  x_match,
         "Model RMSE (deg)",
         "Excess instability AUC",
         "A", "Raw association",
         rho_raw, p_raw),
        (axes[1], x_match, y_raw,  None,
         "Match RMSE (deg)",
         "Excess instability AUC",
         "B", "Motion-match association",
         rho_match, p_match),
        (axes[2], x_res,   y_res,  None,
         "Residualized model RMSE",
         "Residualized excess AUC",
         "C", "After FWL residualization",
         rho_part, p_part),
    ]

    for ax, xd, yd, col_vals, xlabel, ylabel, letter, title, rho, p in panels:
        if col_vals is not None:
            sc = ax.scatter(xd, yd, c=col_vals, cmap="YlOrRd_r",
                            s=22, alpha=0.85, edgecolors="white",
                            linewidth=0.3, zorder=3,
                            vmin=float(col_vals.min()),
                            vmax=float(col_vals.max()))
            cax = ax.inset_axes([0.0, 1.08, 1.0, 0.07])
            cb  = fig.colorbar(sc, cax=cax, orientation="horizontal")
            cb.set_label("Match RMSE (deg)", fontsize=7, labelpad=1)
            cb.ax.tick_params(labelsize=6)
        else:
            clr = PRED_COL if letter == "C" else NEUT_COL
            ax.scatter(xd, yd, color=clr, s=22, alpha=0.80,
                       edgecolors="white", linewidth=0.3, zorder=3)

        xs, ys = _best_fit(xd, yd)
        ax.plot(xs, ys, color=INK, lw=1.4, zorder=4)
        ax.axhline(0, color=GRID_C, lw=0.8, zorder=1)

        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_title(title, fontweight="bold", pad=4,
                     loc="left", fontsize=9.5)

        # stat annotation — below data for panels A and B (data clusters top-right)
        # above data for panel C (data centred around 0)
        va_pos = "top" if letter == "C" else "bottom"
        ay     = 0.97 if letter == "C" else 0.03
        ax.text(0.97, ay, _rho_str(rho, p),
                transform=ax.transAxes, ha="right", va=va_pos,
                fontsize=7.5, color=INK, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#f5f5f5",
                          edgecolor=GRID_C, lw=0.7))
        _grid(ax)
        _panel(ax, letter)

    fig.suptitle(
        "Figure 5. Raw and partial associations between model error and excess instability",
        fontsize=9.5, fontweight="bold", y=1.07)

    out = OUT_DIR / "fig5_fwl_correlation.png"
    fig.savefig(out)
    plt.close()
    return str(out)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

def _write_captions(paths: dict, sim_df: pd.DataFrame, partial_sum: dict) -> str:
    pct_pos = float((sim_df["excess_auc"] > 0).mean() * 100.0)
    mean_excess = float(sim_df["excess_auc"].mean())
    rho_raw, p_raw = stats.spearmanr(sim_df["pred_rmse"], sim_df["excess_auc"])
    rho_match, p_match = stats.spearmanr(sim_df["match_knee_rmse"], sim_df["excess_auc"])
    text = f"""# Paper Figure Captions

---

## Figure 1 — Pipeline
**File:** `{Path(paths['fig1']).relative_to(REPO_ROOT)}`

End-to-end evaluation pipeline. EMG (4 channels) and IMU (6-axis) signals from 55
Georgia Tech subjects are preprocessed on the native 200 Hz timebase and fed to a
CNN-BiLSTM regressor evaluated by 55-fold LOFO cross-validation. Each prediction
drives a motion-matching query into the MoCapAct expert library; the retrieved clip
is replayed twice in MuJoCo — once unmodified (REF) and once with the right knee
overridden by the CNN prediction (PRED). Primary outcome: excess instability AUC = PRED - REF.

---

## Figure 2 — Representative trial
**File:** `{Path(paths['fig2']).relative_to(REPO_ROOT)}`

One simulation trial used to illustrate how the instability outcome is computed. (A) XCoM
margin over 2.01 s; the dashed line marks the base-of-support boundary
(margin = 0 m). (B) Per-step instability score r(t) with AUC regions shaded;
integrating PRED - REF gives the excess AUC reported in the results.

---

## Figure 3 — Prediction performance
**File:** `{Path(paths['fig3']).relative_to(REPO_ROOT)}`

Held-out RMSE for each of the 55 LOFO subject folds, sorted smallest to largest.
Blue bars indicate folds at or below 10 deg; red bars exceed 10 deg. Vertical lines
mark the mean (7.84 deg, dashed) and median (6.85 deg, dotted).

---

## Figure 4 — Simulation instability
**File:** `{Path(paths['fig4']).relative_to(REPO_ROOT)}`

Simulation outcomes across 80 trials. (A) Paired scatter of REF vs PRED instability
AUC. (B) Histogram of excess AUC (PRED - REF); most values are positive and the
mean excess AUC is {mean_excess:.3f}.

---

## Figure 5 — FWL correlation analysis
**File:** `{Path(paths['fig5']).relative_to(REPO_ROOT)}`

Partial Spearman analysis via Frisch-Waugh-Lovell (FWL) residualization: regress
predictor X (model RMSE) and outcome Y (excess AUC) separately on controls Z
(match quality), then compute Spearman r of residuals. (A) Raw scatter coloured by
match RMSE: rho = {rho_raw:.3f}, p = {p_raw:.3f}. (B) Match RMSE is a stronger predictor
(rho = {rho_match:.3f}, p = {p_match:.3f}). (C) After FWL residualization the partial rho is
{float(partial_sum['rho_partial_spearman']):.3f} (p = {float(partial_sum['p_value_two_sided']):.3f}, df = {int(partial_sum['degrees_of_freedom'])}).

---

## Figure 6 — Example replay frames
**File:** `{Path(paths['fig6']).relative_to(REPO_ROOT)}`

Four frames sampled from one representative saved replay GIF. This panel is purely
qualitative and is included to show what the REF/PRED simulation artifacts look
like visually; the paper's quantitative conclusions should rely on Figures 2, 4,
and 5 rather than on single-frame visual inspection.
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

    train   = _train_df()
    _, sim  = _sim_df()
    trials  = _partial_df()
    psum    = _partial_summary()

    print("Figure 1 — pipeline ...")
    f1 = fig1_pipeline()

    print("Figure 2 — representative trial ...")
    f2 = fig2_representative_trial(sim, trial_idx=75)

    print("Figure 3 — prediction performance ...")
    f3 = fig3_prediction(train)

    print("Figure 4 — simulation instability ...")
    f4 = fig4_simulation(sim)

    print("Figure 5 — FWL correlation ...")
    f5 = fig5_correlation(trials, psum)

    print("Figure 6 — simulation frames ...")
    f6 = fig6_simulation_frames(sim, trial_idx=75)

    paths = {"fig1": f1, "fig2": f2, "fig3": f3, "fig4": f4, "fig5": f5, "fig6": f6}
    (OUT_DIR / "figure_manifest.json").write_text(
        json.dumps({"figure_paths": paths,
                    "training_run": str(TRAIN_RUN),
                    "sim_run":      str(SIM_RUN)}, indent=2),
        encoding="utf-8")

    caps = _write_captions(paths, sim, psum)
    print(f"\nDone.  {OUT_DIR}")
    print(f"Captions: {caps}")
    for k, v in paths.items():
        print(f"  {k}: {Path(v).name}")


if __name__ == "__main__":
    main()
