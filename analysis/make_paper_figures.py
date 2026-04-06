from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUN_DIR = REPO_ROOT / "checkpoints" / "tst_20260405_173725_all"
SIM_RUN_DIR = REPO_ROOT / "artifacts" / "phys_eval_v2" / "runs" / "20260405_230549"
OUT_DIR = REPO_ROOT / "figures" / "paper_native"


COLORS = {
    "ink": "#17252a",
    "muted": "#6b7c85",
    "grid": "#d8e0e3",
    "accent": "#2a6f97",
    "accent_soft": "#8ecae6",
    "warm": "#bc6c25",
    "warm_soft": "#ddb892",
    "good": "#2a9d8f",
    "bad": "#d1495b",
    "bg": "#f8fbfc",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


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
                "summary_path": str(SIM_RUN_DIR / "evals" / f"{rows.__len__():02d}_{result['query_id']}" / "summary.json"),
            }
        )
    return summary, pd.DataFrame(rows)


def load_partial_trials() -> pd.DataFrame:
    trials_csv = SIM_RUN_DIR / "analysis" / "partial_spearman_trials.csv"
    return pd.read_csv(trials_csv)


def best_fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    ys = m * xs + b
    return xs, ys


def draw_pipeline_figure() -> str:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.03, 0.22, 0.18, 0.56, "Georgia Tech Data", "4 EMG + 6 thigh IMU\n200 Hz target timebase\n400-sample windows"),
        (0.28, 0.22, 0.18, 0.56, "CNN-BiLSTM", "Conv1d(10→32)\nBiLSTM(64x2 bidir)\nlast-step knee prediction"),
        (0.53, 0.22, 0.18, 0.56, "Motion Matching", "MoCapAct bank\nthigh_knee_d matcher\nmean knee match 7.93 deg"),
        (0.78, 0.22, 0.18, 0.56, "MuJoCo Evaluation", "REF vs PRED rollout\nright-knee override\n80 retained trials"),
    ]

    for x, y, w, h, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.03",
            linewidth=1.6,
            edgecolor=COLORS["accent"],
            facecolor=COLORS["bg"],
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.10, title, fontsize=13, fontweight="bold", color=COLORS["ink"])
        ax.text(x + 0.02, y + h - 0.18, body, fontsize=10, color=COLORS["ink"], va="top", linespacing=1.4)

    for x0, x1 in [(0.21, 0.28), (0.46, 0.53), (0.71, 0.78)]:
        arrow = FancyArrowPatch(
            (x0, 0.50),
            (x1, 0.50),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.0,
            color=COLORS["warm"],
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.93,
        "Figure 1. Native-rate Georgia Tech pipeline used for the publication benchmark.",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "Primary paper outcome: excess instability AUC = PRED instability AUC - REF instability AUC.\n"
        "Primary paper predictor: model pred-vs-GT knee RMSE on the held-out query window.",
        ha="center",
        va="center",
        fontsize=10,
        color=COLORS["muted"],
    )

    out = OUT_DIR / "fig1_pipeline_overview.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def draw_prediction_figure(train_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220)

    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.04, len(train_df))
    axes[0].axhline(10.0, color=COLORS["warm"], linestyle="--", linewidth=1.4, label="10 deg target")
    axes[0].boxplot(
        train_df["test_rmse"],
        positions=[0],
        widths=0.28,
        patch_artist=True,
        boxprops=dict(facecolor=COLORS["accent_soft"], color=COLORS["accent"]),
        medianprops=dict(color=COLORS["ink"], linewidth=1.8),
        whiskerprops=dict(color=COLORS["accent"]),
        capprops=dict(color=COLORS["accent"]),
    )
    axes[0].scatter(jitter, train_df["test_rmse"], s=28, color=COLORS["accent"], alpha=0.8, edgecolor="white", linewidth=0.4)
    axes[0].set_xlim(-0.35, 0.35)
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["55 held-out folds"])
    axes[0].set_ylabel("Held-out test RMSE (deg)")
    axes[0].set_title("Subject-holdout prediction distribution")
    axes[0].grid(axis="y", color=COLORS["grid"], alpha=0.8)
    axes[0].legend(frameon=False, loc="upper right")

    sorted_rmse = np.sort(train_df["test_rmse"].to_numpy())
    ranks = np.arange(1, len(sorted_rmse) + 1)
    axes[1].plot(ranks, sorted_rmse, color=COLORS["accent"], linewidth=2.2)
    axes[1].fill_between(ranks, sorted_rmse, color=COLORS["accent_soft"], alpha=0.35)
    axes[1].axhline(float(train_df["test_rmse"].mean()), color=COLORS["good"], linestyle="--", linewidth=1.4, label="Mean")
    axes[1].axhline(float(train_df["test_rmse"].median()), color=COLORS["ink"], linestyle=":", linewidth=1.4, label="Median")
    axes[1].axhline(10.0, color=COLORS["warm"], linestyle="--", linewidth=1.4, label="10 deg target")
    axes[1].set_xlabel("Fold rank (best to worst)")
    axes[1].set_ylabel("Held-out test RMSE (deg)")
    axes[1].set_title("Most folds remain below 10 deg")
    axes[1].grid(color=COLORS["grid"], alpha=0.8)
    axes[1].legend(frameon=False, loc="upper left")

    fig.suptitle("Figure 2. Native-rate CNN-BiLSTM subject-holdout performance.", fontsize=14, fontweight="bold", y=1.02)
    out = OUT_DIR / "fig2_prediction_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def draw_simulation_figure(sim_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=220)
    order = np.argsort((sim_df["ref_knee_rmse"] - sim_df["pred_knee_rmse"]).to_numpy())
    x = np.arange(len(sim_df))

    ref_k = sim_df["ref_knee_rmse"].to_numpy()[order]
    pred_k = sim_df["pred_knee_rmse"].to_numpy()[order]
    for i in range(len(sim_df)):
        axes[0].plot([i, i], [ref_k[i], pred_k[i]], color=COLORS["grid"], linewidth=0.8, zorder=1)
    axes[0].scatter(x, ref_k, color=COLORS["muted"], s=18, label="REF", zorder=2)
    axes[0].scatter(x, pred_k, color=COLORS["good"], s=18, label="PRED", zorder=3)
    axes[0].set_title("Simulated knee tracking per trial")
    axes[0].set_xlabel("Trial index (sorted by REF-PRED improvement)")
    axes[0].set_ylabel("Simulated knee RMSE (deg)")
    axes[0].grid(color=COLORS["grid"], alpha=0.8)
    axes[0].legend(frameon=False, loc="upper right")

    ref_auc = sim_df["ref_instability_auc"].to_numpy()[order]
    pred_auc = sim_df["pred_instability_auc"].to_numpy()[order]
    for i in range(len(sim_df)):
        axes[1].plot([i, i], [ref_auc[i], pred_auc[i]], color=COLORS["grid"], linewidth=0.8, zorder=1)
    axes[1].scatter(x, ref_auc, color=COLORS["muted"], s=18, label="REF", zorder=2)
    axes[1].scatter(x, pred_auc, color=COLORS["bad"], s=18, label="PRED", zorder=3)
    axes[1].set_title("Instability AUC per trial")
    axes[1].set_xlabel("Trial index (same ordering)")
    axes[1].set_ylabel("Instability AUC")
    axes[1].grid(color=COLORS["grid"], alpha=0.8)
    axes[1].legend(frameon=False, loc="upper right")

    axes[2].hist(sim_df["excess_instability_auc"], bins=18, color=COLORS["warm_soft"], edgecolor="white")
    axes[2].axvline(0.0, color=COLORS["ink"], linestyle=":", linewidth=1.5)
    axes[2].axvline(float(sim_df["excess_instability_auc"].mean()), color=COLORS["bad"], linestyle="--", linewidth=1.5, label="Mean excess AUC")
    axes[2].set_title("Distribution of excess instability")
    axes[2].set_xlabel("Excess instability AUC (PRED - REF)")
    axes[2].set_ylabel("Trial count")
    axes[2].grid(axis="y", color=COLORS["grid"], alpha=0.8)
    axes[2].legend(frameon=False, loc="upper right")

    fig.suptitle("Figure 3. PRED improves knee tracking but not instability.", fontsize=14, fontweight="bold", y=1.02)
    out = OUT_DIR / "fig3_simulation_outcomes.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def draw_correlation_figure(trials_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=220)

    x1 = trials_df["predictor_knee_rmse_deg"].to_numpy()
    y = trials_df["outcome_value"].to_numpy()
    c = trials_df["control_match_knee_rmse_deg"].to_numpy()
    sc = axes[0].scatter(x1, y, c=c, cmap="YlGnBu", s=34, alpha=0.85, edgecolors="white", linewidth=0.35)
    xs, ys = best_fit_line(x1, y)
    r, p = spearmanr(x1, y)
    axes[0].plot(xs, ys, color=COLORS["ink"], linewidth=1.5)
    axes[0].set_title(f"Raw model error vs excess instability\nrho={r:.3f}, p={p:.3f}")
    axes[0].set_xlabel("Model knee RMSE (deg)")
    axes[0].set_ylabel("Excess instability AUC")
    axes[0].grid(color=COLORS["grid"], alpha=0.8)

    x2 = trials_df["control_match_knee_rmse_deg"].to_numpy()
    xs2, ys2 = best_fit_line(x2, y)
    r2, p2 = spearmanr(x2, y)
    axes[1].scatter(x2, y, color=COLORS["warm"], s=32, alpha=0.8, edgecolors="white", linewidth=0.35)
    axes[1].plot(xs2, ys2, color=COLORS["ink"], linewidth=1.5)
    axes[1].set_title(f"Match quality carries more signal\nrho={r2:.3f}, p={p2:.3f}")
    axes[1].set_xlabel("Motion-match knee RMSE (deg)")
    axes[1].set_ylabel("Excess instability AUC")
    axes[1].grid(color=COLORS["grid"], alpha=0.8)

    x3 = trials_df["residual_predictor"].to_numpy()
    y3 = trials_df["residual_outcome"].to_numpy()
    xs3, ys3 = best_fit_line(x3, y3)
    with (SIM_RUN_DIR / "analysis" / "partial_spearman_summary.json").open() as f:
        part = json.load(f)
    axes[2].scatter(x3, y3, color=COLORS["accent"], s=32, alpha=0.8, edgecolors="white", linewidth=0.35)
    axes[2].plot(xs3, ys3, color=COLORS["ink"], linewidth=1.5)
    axes[2].set_title(
        f"After FWL residualization\npartial rho={part['rho_partial_spearman']:.3f}, p={part['p_value_two_sided']:.3f}"
    )
    axes[2].set_xlabel("Residualized RMSE rank")
    axes[2].set_ylabel("Residualized excess instability rank")
    axes[2].grid(color=COLORS["grid"], alpha=0.8)

    cb = fig.colorbar(sc, ax=axes[0], fraction=0.048, pad=0.03)
    cb.set_label("Match knee RMSE (deg)")
    fig.suptitle("Figure 4. Motion-matching quality explains more than model RMSE.", fontsize=14, fontweight="bold", y=1.04)
    out = OUT_DIR / "fig4_correlation_confounding.png"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def pick_representative_trial(sim_df: pd.DataFrame) -> pd.Series:
    subset = sim_df[
        (sim_df["pred_knee_rmse"] < sim_df["ref_knee_rmse"])
        & (sim_df["excess_instability_auc"] > 0)
        & (sim_df["pred_rmse"] < sim_df["pred_rmse"].median())
    ].copy()
    if subset.empty:
        subset = sim_df.copy()
    subset["score"] = (subset["ref_knee_rmse"] - subset["pred_knee_rmse"]) + 0.5 * subset["excess_instability_auc"]
    subset = subset.sort_values("score", ascending=False)
    return subset.iloc[0]


def draw_rollout_figure(trial: pd.Series) -> tuple[str, dict[str, Any]]:
    npz_path = REPO_ROOT / trial["compare_npz"]
    data = np.load(npz_path)
    dt = float(data["dt"])
    t_actual = np.arange(len(data["knee_ref_actual_deg"])) * dt
    t_query = np.arange(len(data["knee_good_query_deg"])) * dt
    t_ref_trace = np.arange(len(data["predicted_fall_risk_trace_ref"])) * dt
    t_pred_trace = np.arange(len(data["predicted_fall_risk_trace_good"])) * dt

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), dpi=220, sharex=True)

    axes[0].plot(t_query, data["knee_good_query_deg"], color=COLORS["ink"], linewidth=2.2, label="GT query target")
    axes[0].plot(t_actual, data["knee_ref_actual_deg"], color=COLORS["muted"], linewidth=2.0, label="REF actual")
    axes[0].plot(t_actual, data["knee_good_actual_deg"], color=COLORS["good"], linewidth=2.0, label="PRED actual")
    axes[0].set_ylabel("Knee angle (deg)")
    axes[0].set_title("Representative trial: better knee tracking does not guarantee lower instability")
    axes[0].grid(color=COLORS["grid"], alpha=0.8)
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(t_ref_trace, data["predicted_fall_risk_trace_ref"], color=COLORS["muted"], linewidth=2.0, label="REF instability trace")
    axes[1].plot(t_pred_trace, data["predicted_fall_risk_trace_good"], color=COLORS["bad"], linewidth=2.0, label="PRED instability trace")
    axes[1].fill_between(t_ref_trace, data["predicted_fall_risk_trace_ref"], alpha=0.18, color=COLORS["muted"])
    axes[1].fill_between(t_pred_trace, data["predicted_fall_risk_trace_good"], alpha=0.14, color=COLORS["bad"])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Instability trace")
    axes[1].grid(color=COLORS["grid"], alpha=0.8)
    axes[1].legend(frameon=False, loc="upper right")

    caption = (
        f"query={trial['query_id']}  file={trial['file_name']}  "
        f"pred_rmse={trial['pred_rmse']:.2f} deg  match_rmse={trial['match_knee_rmse']:.2f} deg  "
        f"REF_knee={trial['ref_knee_rmse']:.2f} deg  PRED_knee={trial['pred_knee_rmse']:.2f} deg  "
        f"excess_auc={trial['excess_instability_auc']:.3f}"
    )
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, color=COLORS["muted"])

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


def write_captions(figure_paths: dict[str, str], representative_meta: dict[str, Any]) -> str:
    text = f"""# Paper Figure Notes

These figures were generated from the native-rate publication benchmark:
- training run: `{TRAIN_RUN_DIR.relative_to(REPO_ROOT)}`
- simulation run: `{SIM_RUN_DIR.relative_to(REPO_ROOT)}`

## Figure 1
- file: `{Path(figure_paths['fig1']).relative_to(REPO_ROOT)}`
- use: methodology overview
- caption: Native-rate Georgia Tech pipeline used in the publication benchmark, from GT preprocessing through CNN-BiLSTM prediction, motion matching, MuJoCo rollout, and excess-instability analysis.

## Figure 2
- file: `{Path(figure_paths['fig2']).relative_to(REPO_ROOT)}`
- use: predictive performance / generalization
- caption: Distribution of held-out subject-fold RMSE values across 55 folds. Most folds remain below the 10 deg threshold, with mean test RMSE 7.84 deg and median 6.85 deg.

## Figure 3
- file: `{Path(figure_paths['fig3']).relative_to(REPO_ROOT)}`
- use: simulation outcomes
- caption: Paired REF versus PRED outcomes across the 80-window simulation benchmark. PRED improves knee tracking on average, but excess instability remains positive overall.

## Figure 4
- file: `{Path(figure_paths['fig4']).relative_to(REPO_ROOT)}`
- use: correlation / confounding
- caption: Raw model RMSE is not significantly associated with excess instability, while motion-match quality carries more signal. After Frisch-Waugh-Lovell residualization on the motion-match controls, the partial association is near zero.

## Figure 5
- file: `{Path(figure_paths['fig5']).relative_to(REPO_ROOT)}`
- use: representative case study
- caption: Representative trial `{representative_meta['query_id']}` from `{representative_meta['file_name']}`. The model improves simulated knee tracking (REF {representative_meta['ref_knee_rmse_deg']:.2f} deg vs PRED {representative_meta['pred_knee_rmse_deg']:.2f} deg) even though excess instability remains positive ({representative_meta['excess_instability_auc']:.3f}), illustrating why RMSE alone is not sufficient.
"""
    out = OUT_DIR / "captions.md"
    out.write_text(text, encoding="utf-8")
    return str(out)


def main() -> None:
    _style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train_metrics()
    _, sim_df = load_simulation_results()
    trials_df = load_partial_trials()

    fig1 = draw_pipeline_figure()
    fig2 = draw_prediction_figure(train_df)
    fig3 = draw_simulation_figure(sim_df)
    fig4 = draw_correlation_figure(trials_df)
    representative_trial = pick_representative_trial(sim_df)
    fig5, representative_meta = draw_rollout_figure(representative_trial)

    manifest = {
        "figure_paths": {
            "fig1": fig1,
            "fig2": fig2,
            "fig3": fig3,
            "fig4": fig4,
            "fig5": fig5,
        },
        "training_run_dir": str(TRAIN_RUN_DIR),
        "simulation_run_dir": str(SIM_RUN_DIR),
        "representative_trial": representative_meta,
    }
    (OUT_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    captions = write_captions(manifest["figure_paths"], representative_meta)
    print(json.dumps({"manifest": str(OUT_DIR / "figure_manifest.json"), "captions": captions, **manifest}, indent=2))


if __name__ == "__main__":
    main()
