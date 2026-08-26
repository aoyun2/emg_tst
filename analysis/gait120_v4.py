from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from mocap_phys_eval.level_walking import LEVEL_WALKING_CLIP_IDS
from mocap_phys_eval.run_gait120_residual_fusion import CHECKPOINT_LABELS, PRIMARY_CHECKPOINT


VERSION = "GAIT120_LEVEL_WALKING_FINAL_ANALYSIS_V1"
SEED = 20260826
N_BOOTSTRAP = 20_000
N_PERMUTATIONS = 100_000
COLORS = {"reference": "#252525", "fused": "#087E8B", "no_emg": "#D95F02"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite required value: {value!r}")
    return result


def _mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise RuntimeError("Mean CI requires at least two finite values")
    draws = rng.choice(x, size=(N_BOOTSTRAP, x.size), replace=True).mean(axis=1)
    return float(np.mean(x)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _paired_signflip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    observed = abs(float(np.mean(x)))
    exceed = 0
    done = 0
    batch = 2_000
    while done < N_PERMUTATIONS:
        count = min(batch, N_PERMUTATIONS - done)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, x.size), replace=True)
        permuted = np.abs(np.mean(signs * x[None, :], axis=1))
        exceed += int(np.sum(permuted >= observed - 1.0e-15))
        done += count
    return float((exceed + 1) / (N_PERMUTATIONS + 1))


def _rank(values: np.ndarray) -> np.ndarray:
    return np.asarray(stats.rankdata(np.asarray(values, dtype=np.float64)), dtype=np.float64)


def _residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    c = np.asarray(controls, dtype=np.float64)
    if c.ndim == 1:
        c = c[:, None]
    design = np.column_stack([np.ones(y.size), c])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def _partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    xr = _rank(x)
    yr = _rank(y)
    cr = np.column_stack([_rank(controls[:, column]) for column in range(controls.shape[1])])
    xres = _residualize(xr, cr)
    yres = _residualize(yr, cr)
    rho = _corr(xres, yres)
    n = int(xres.size)
    df = int(n - controls.shape[1] - 2)
    t_stat = float(rho * math.sqrt(df / max(1.0e-15, 1.0 - rho * rho)))
    analytic_p = float(2.0 * stats.t.sf(abs(t_stat), df=df))

    boot = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for index in range(N_BOOTSTRAP):
        chosen = rng.integers(0, n, size=n)
        if np.unique(xres[chosen]).size < 2 or np.unique(yres[chosen]).size < 2:
            boot[index] = np.nan
        else:
            boot[index] = _corr(xres[chosen], yres[chosen])
    boot = boot[np.isfinite(boot)]

    exceed = 0
    done = 0
    batch = 2_000
    while done < N_PERMUTATIONS:
        count = min(batch, N_PERMUTATIONS - done)
        for _ in range(count):
            permuted = rng.permutation(yres)
            exceed += int(abs(_corr(xres, permuted)) >= abs(rho) - 1.0e-15)
        done += count
    permutation_p = float((exceed + 1) / (N_PERMUTATIONS + 1))
    return {
        "n_participants": n,
        "rho": rho,
        "bootstrap_95_ci": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "analytic_t": t_stat,
        "analytic_df": df,
        "analytic_p_two_sided": analytic_p,
        "permutation_p_two_sided": permutation_p,
        "x_residual": xres.tolist(),
        "y_residual": yres.tolist(),
    }


def _aggregate(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group = tuple(str(row[key]) for key in keys)
        groups.setdefault(group, []).append(row)
    numeric = (
        "source_rmse_fused_deg",
        "source_rmse_no_emg_deg",
        "simulation_rmse_fused_deg",
        "simulation_rmse_no_emg_deg",
        "match_knee_rmse_deg",
        "match_thigh_rms_deg",
        "reference_auc",
        "fused_auc",
        "no_emg_auc",
        "fused_excess_auc",
        "no_emg_excess_auc",
    )
    output: list[dict[str, Any]] = []
    for group, members in sorted(groups.items()):
        item: dict[str, Any] = {key: value for key, value in zip(keys, group)}
        item["n_windows"] = len(members)
        for name in numeric:
            values = np.asarray([float(member[name]) for member in members], dtype=np.float64)
            finite = values[np.isfinite(values)]
            item[name] = float(np.mean(finite)) if finite.size else float("nan")
        item["reference_balance_losses"] = int(sum(int(member["reference_balance_loss"]) >= 0 for member in members))
        item["fused_balance_losses"] = int(sum(int(member["fused_balance_loss"]) >= 0 for member in members))
        item["no_emg_balance_losses"] = int(sum(int(member["no_emg_balance_loss"]) >= 0 for member in members))
        output.append(item)
    return output


def _segmented_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    linear_design = np.column_stack([np.ones(x.size), x])
    linear_beta, _, _, _ = np.linalg.lstsq(linear_design, y, rcond=None)
    linear_residual = y - linear_design @ linear_beta
    linear_rss = float(np.sum(linear_residual**2))
    linear_bic = float(x.size * np.log(max(linear_rss / x.size, 1.0e-15)) + 2 * np.log(x.size))

    unique = np.unique(x)
    candidates = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.asarray(
        [knot for knot in candidates if np.sum(x <= knot) >= 2 and np.sum(x > knot) >= 2],
        dtype=np.float64,
    )
    fits: list[dict[str, Any]] = []
    for knot in candidates:
        design = np.column_stack([np.ones(x.size), x, np.maximum(0.0, x - knot)])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        rss = float(np.sum(residual**2))
        bic = float(x.size * np.log(max(rss / x.size, 1.0e-15)) + 4 * np.log(x.size))
        fits.append({"knot_rmse_deg": float(knot), "rss": rss, "bic": bic, "beta": beta.tolist()})
    if not fits:
        return {
            "identifiable": False,
            "reason": "fewer than two aggregate checkpoints on each side of every candidate knot",
            "linear_bic": linear_bic,
            "segmented_fits": [],
        }
    best = min(fits, key=lambda row: float(row["bic"]))
    improvement = float(linear_bic - float(best["bic"]))
    profile = [float(row["knot_rmse_deg"]) for row in fits if float(row["bic"]) <= float(best["bic"]) + 2.0]
    best_knot = float(best["knot_rmse_deg"])
    below = x[x <= best_knot]
    above = x[x > best_knot]
    bracket = [float(np.max(below)), float(np.min(above))]
    bracket_width = float(bracket[1] - bracket[0])
    # A preferred segmented fit does not identify a numerical cutoff when the
    # apparent bend lies inside an unsampled interval. The aggregate path has
    # only seven prospectively fixed checkpoints, so report the observed
    # transition bracket rather than inventing resolution within the gap.
    identifiable = bool(improvement >= 6.0 and bracket_width <= 2.0 and len(profile) > 0)
    return {
        "identifiable": identifiable,
        "criterion": (
            "segmented BIC at least 6 below linear BIC and the two observed "
            "checkpoint RMSEs bracketing the bend no more than 2 degrees apart"
        ),
        "linear_bic": linear_bic,
        "best_segmented_bic": float(best["bic"]),
        "bic_improvement": improvement,
        "best_segmented_midpoint_rmse_deg": best_knot,
        "observed_transition_bracket_deg": bracket,
        "transition_bracket_width_deg": bracket_width,
        "estimated_change_point_rmse_deg": best_knot if identifiable else None,
        "profile_delta_bic_2_range_deg": [min(profile), max(profile)] if identifiable else None,
        "interpretation": (
            "identifiable cutoff"
            if identifiable
            else "no precise cutoff; the change is only localized to the observed RMSE bracket"
        ),
        "linear_beta": linear_beta.tolist(),
        "best_segmented_beta": best["beta"],
        "segmented_fits": fits,
    }


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )


def _plot_primary_ablation(participants: list[dict[str, Any]], out: Path) -> None:
    no_emg = np.asarray([row["source_rmse_no_emg_deg"] for row in participants])
    fused = np.asarray([row["source_rmse_fused_deg"] for row in participants])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    for index, (left, right) in enumerate(zip(no_emg, fused)):
        ax.plot([0, 1], [left, right], color="#A7A7A7", lw=0.65, alpha=0.55, zorder=1)
    ax.scatter(np.zeros_like(no_emg), no_emg, s=17, color=COLORS["no_emg"], alpha=0.78, zorder=2)
    ax.scatter(np.ones_like(fused), fused, s=17, color=COLORS["fused"], alpha=0.78, zorder=2)
    ax.plot([0, 1], [np.mean(no_emg), np.mean(fused)], color="#111111", lw=2.2, zorder=3)
    ax.set_xticks([0, 1], ["Without sEMG", "Residual fusion"])
    ax.set_ylabel("Knee-angle RMSE (degrees)")
    ax.set_title("Held-out participant comparison", loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    gain = no_emg - fused
    ordered = np.sort(gain)
    colors = np.where(ordered > 0.0, COLORS["fused"], "#B7B7B7")
    ax.bar(np.arange(ordered.size), ordered, width=0.84, color=colors, edgecolor="none")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xlabel("Confirmation participant (ordered)")
    ax.set_ylabel("RMSE reduction with sEMG (degrees)")
    ax.set_title("Individual sEMG contribution", loc="left", fontweight="bold")
    ax.set_xticks([])
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_physics_pair(participants: list[dict[str, Any]], out: Path) -> None:
    no_emg = np.asarray([row["no_emg_excess_auc"] for row in participants])
    fused = np.asarray([row["fused_excess_auc"] for row in participants])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    for left, right in zip(no_emg, fused):
        ax.plot([0, 1], [left, right], color="#A7A7A7", lw=0.65, alpha=0.55)
    ax.scatter(np.zeros_like(no_emg), no_emg, s=17, color=COLORS["no_emg"], alpha=0.78)
    ax.scatter(np.ones_like(fused), fused, s=17, color=COLORS["fused"], alpha=0.78)
    ax.plot([0, 1], [np.mean(no_emg), np.mean(fused)], color="#111111", lw=2.2)
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks([0, 1], ["Without sEMG", "Residual fusion"])
    ax.set_ylabel("Excess instability AUC")
    ax.set_title("Participant-level physical outcome", loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    delta = no_emg - fused
    ordered = np.sort(delta)
    colors = np.where(ordered > 0.0, COLORS["fused"], "#B7B7B7")
    ax.bar(np.arange(ordered.size), ordered, width=0.84, color=colors, edgecolor="none")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xlabel("Physics participant (ordered)")
    ax.set_ylabel("AUC reduction with residual fusion")
    ax.set_title("Paired physical difference", loc="left", fontweight="bold")
    ax.set_xticks([])
    fig.tight_layout(w_pad=2.0)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_fwl(participants: list[dict[str, Any]], result: dict[str, Any], out: Path) -> None:
    x = np.asarray(result["x_residual"], dtype=np.float64)
    y = np.asarray(result["y_residual"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(4.5, 3.45))
    ax.scatter(x, y, s=28, color=COLORS["fused"], alpha=0.78, edgecolor="white", linewidth=0.35)
    if np.unique(x).size > 1:
        slope, intercept = np.polyfit(x, y, 1)
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xx, intercept + slope * xx, color="#252525", lw=1.3)
    ax.axhline(0.0, color="#777777", lw=0.65)
    ax.axvline(0.0, color="#777777", lw=0.65)
    ax.set_xlabel("Residualized rank of prediction RMSE")
    ax.set_ylabel("Residualized rank of excess instability AUC")
    ax.set_title("Prediction error and physical instability", loc="left", fontweight="bold")
    ci = result["bootstrap_95_ci"]
    ax.text(
        0.03,
        0.97,
        f"Partial Spearman $\\rho$ = {result['rho']:.2f}\n95% CI {ci[0]:.2f} to {ci[1]:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_path(checkpoints: list[dict[str, Any]], change: dict[str, Any], out: Path) -> None:
    fractions = np.asarray([float(row["fraction"]) for row in checkpoints])
    rmse = np.asarray([row["mean_rmse_deg"] for row in checkpoints])
    rmse_lo = np.asarray([row["rmse_ci_low"] for row in checkpoints])
    rmse_hi = np.asarray([row["rmse_ci_high"] for row in checkpoints])
    auc = np.asarray([row["mean_excess_auc"] for row in checkpoints])
    auc_lo = np.asarray([row["auc_ci_low"] for row in checkpoints])
    auc_hi = np.asarray([row["auc_ci_high"] for row in checkpoints])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0))
    ax = axes[0]
    ax.errorbar(
        fractions,
        rmse,
        yerr=np.vstack([rmse - rmse_lo, rmse_hi - rmse]),
        color=COLORS["fused"],
        marker="o",
        ms=4.2,
        lw=1.5,
        capsize=2.5,
    )
    ax.set_xticks(fractions, [f"{int(value)}" for value in fractions])
    ax.set_xlabel("Training data used (%)")
    ax.set_ylabel("Prediction RMSE (degrees)")
    ax.set_title("Prediction accuracy path", loc="left", fontweight="bold")

    ax = axes[1]
    ax.errorbar(
        rmse,
        auc,
        xerr=np.vstack([rmse - rmse_lo, rmse_hi - rmse]),
        yerr=np.vstack([auc - auc_lo, auc_hi - auc]),
        color=COLORS["fused"],
        marker="o",
        ms=4.2,
        lw=1.3,
        capsize=2.5,
    )
    # The 20--100% checkpoints occupy a deliberately tight RMSE range.  Label
    # that group once so the figure remains legible at journal column width.
    ax.annotate(
        "5%",
        (rmse[0], auc[0]),
        xytext=(-5, -14),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
    )
    ax.annotate(
        "10%",
        (rmse[1], auc[1]),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=7.5,
    )
    cluster_x = float(np.mean(rmse[2:]))
    cluster_y = float(np.mean(auc[2:]))
    ax.annotate(
        "20--100%",
        (cluster_x, cluster_y),
        xytext=(34, 22),
        textcoords="offset points",
        fontsize=7.5,
        ha="left",
        arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.8},
    )
    ax.axhline(0.0, color="#555555", lw=0.75)
    ax.set_xlabel("Prediction RMSE (degrees)")
    ax.set_ylabel("Mean excess instability AUC")
    ax.set_title("Physical outcome across accuracy", loc="left", fontweight="bold")
    if change["identifiable"]:
        ax.axvline(float(change["estimated_change_point_rmse_deg"]), color="#8C2D04", ls="--", lw=1.1)
    fig.tight_layout(w_pad=2.2)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _plot_simulation_sequence(primary_rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    chosen = next(row for row in primary_rows if int(row["panel_index"]) == 33)
    npz_path = Path(str(chosen["recording_npz"]))
    with np.load(npz_path, allow_pickle=False) as stored:
        frames = np.asarray(stored["frames"], dtype=np.uint8)
        dt = float(np.asarray(stored["dt"]).reshape(()))
    if frames.shape[0] < 5:
        raise RuntimeError("Representative V4 recording lacks rendered frames")
    indices = np.rint(np.linspace(0, frames.shape[0] - 1, 5)).astype(int)
    fig, axes = plt.subplots(5, 1, figsize=(7.15, 6.15))
    for axis, index in zip(axes, indices):
        axis.imshow(frames[index])
        axis.set_axis_off()
        axis.text(
            0.008,
            0.08,
            f"t = {index * dt:.2f} s",
            color="white",
            fontsize=8.0,
            transform=axis.transAxes,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.72, "edgecolor": "none"},
        )
    fig.subplots_adjust(left=0.002, right=0.998, top=0.995, bottom=0.005, hspace=0.025)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return {
        "query_id": chosen["query_id"],
        "panel_index": chosen["panel_index"],
        "recording_npz": str(npz_path),
        "frame_indices": indices.tolist(),
        "times_s": [float(index * dt) for index in indices],
    }


def _plot_simulation_metrics(primary_rows: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    chosen = next(row for row in primary_rows if int(row["panel_index"]) == 33)
    main_path = Path(str(chosen["recording_npz"]))
    no_path = Path(str(chosen["no_emg_recording_npz"])) if chosen["no_emg_recording_npz"] else None
    with np.load(main_path, allow_pickle=False) as stored:
        dt = float(np.asarray(stored["dt"]).reshape(()))
        t = np.arange(np.asarray(stored["knee_ref_actual_deg"]).size) * dt
        knee_ref = np.asarray(stored["knee_ref_actual_deg"], dtype=np.float64)
        knee_fused = np.asarray(stored["knee_good_actual_deg"], dtype=np.float64)
        x_ref = np.asarray(stored["balance_xcom_margin_ref_m"], dtype=np.float64)
        x_fused = np.asarray(stored["balance_xcom_margin_good_m"], dtype=np.float64)
        risk_ref = np.asarray(stored["predicted_fall_risk_trace_ref"], dtype=np.float64)
        risk_fused = np.asarray(stored["predicted_fall_risk_trace_good"], dtype=np.float64)
        has_bad = bool(np.asarray(stored["has_bad"]).reshape(()))
        if has_bad:
            knee_no = np.asarray(stored["knee_bad_actual_deg"], dtype=np.float64)
            x_no = np.asarray(stored["balance_xcom_margin_bad_m"], dtype=np.float64)
            risk_no = np.asarray(stored["predicted_fall_risk_trace_bad"], dtype=np.float64)
        else:
            knee_no = x_no = risk_no = np.zeros((0,), dtype=np.float64)
    if not has_bad:
        if no_path is None:
            raise RuntimeError("Representative primary recording lacks no-sEMG rollout")
        with np.load(no_path, allow_pickle=False) as stored:
            knee_no = np.asarray(stored["knee_good_actual_deg"], dtype=np.float64)
            x_no = np.asarray(stored["balance_xcom_margin_good_m"], dtype=np.float64)
            risk_no = np.asarray(stored["predicted_fall_risk_trace_good"], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(7.15, 5.25), sharex=True)
    series = (
        (knee_ref, knee_fused, knee_no, "Knee angle (degrees)"),
        (x_ref, x_fused, x_no, "XCoM support margin (m)"),
        (risk_ref, risk_fused, risk_no, "Instability index"),
    )
    for axis, (reference, fused, no_emg, ylabel) in zip(axes, series):
        axis.plot(t[: reference.size], reference, color=COLORS["reference"], lw=1.35, label="Reference")
        axis.plot(t[: fused.size], fused, color=COLORS["fused"], lw=1.35, label="Residual fusion")
        axis.plot(t[: no_emg.size], no_emg, color=COLORS["no_emg"], lw=1.25, label="Without sEMG")
        axis.set_ylabel(ylabel)
    axes[1].axhline(0.0, color="#777777", lw=0.7, ls="--")
    axes[2].set_xlabel("Simulation time (s)")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout(h_pad=0.7)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return {"query_id": chosen["query_id"], "panel_index": chosen["panel_index"], "recording_npz": str(main_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--confirmation-dir",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    confirmation_dir = args.confirmation_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "final_analysis")).resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _set_plot_style()

    status = _json(run_dir / "pipeline_status.json")
    physics = _json(run_dir / "physics_summary.json")
    protocol = _json(run_dir / "physics_protocol.level_walking_v4_1.json")
    matching = _json(run_dir / "matching_preflight" / "summary.json")
    oracle = _json(run_dir / "oracle_preflight" / "summary.json")
    if status.get("stage") != "physics_complete":
        raise RuntimeError("V4 physics is not complete")
    if not bool(matching.get("passed")) or int(matching.get("n_windows", 0)) != 80:
        raise RuntimeError("V4 matching gate did not pass 80 windows")
    if not bool(oracle.get("passed")) or int(oracle.get("passing_windows", 0)) < 8:
        raise RuntimeError("V4 oracle gate did not pass")
    if int(physics.get("n_windows_per_checkpoint", 0)) != 80:
        raise RuntimeError("V4 physics summary does not contain 80 windows per checkpoint")

    rows: list[dict[str, Any]] = []
    expected_query_ids: list[str] | None = None
    reference_fingerprints: dict[str, set[tuple[float, float, int]]] = {}
    for checkpoint_index, checkpoint in enumerate(CHECKPOINT_LABELS):
        stage_dir = run_dir / "stages" / checkpoint / "evals"
        paths = sorted(stage_dir.glob("*/summary.json"))
        if len(paths) != 80:
            raise RuntimeError(f"{checkpoint} contains {len(paths)} summaries, expected 80")
        stage_rows: list[dict[str, Any]] = []
        for path in paths:
            summary = _json(path)
            if str(summary["checkpoint"]) != checkpoint:
                raise RuntimeError(f"Checkpoint mismatch in {path}")
            match = summary["match"]
            if str(match["clip_id"]) not in LEVEL_WALKING_CLIP_IDS:
                raise RuntimeError(f"Non-walking reference in definitive V4: {match['clip_id']}")
            simulation = summary["simulation"]
            source = summary["prediction_rmse_deg"]["source_100hz"]
            exact = summary["prediction_rmse_deg"]["exact_simulation_frames"]
            row = {
                "query_id": str(summary["query_id"]),
                "panel_index": int(summary["panel_index"]),
                "subject": str(summary["subject"]),
                "checkpoint": checkpoint,
                "checkpoint_index": checkpoint_index,
                "source_rmse_fused_deg": _finite(source["fused"]),
                "source_rmse_no_emg_deg": _finite(source["no_emg"]),
                "simulation_rmse_fused_deg": _finite(exact["fused"]),
                "simulation_rmse_no_emg_deg": _finite(exact["no_emg"]),
                "match_knee_rmse_deg": _finite(match["knee_rmse_deg"]),
                "match_thigh_rms_deg": _finite(match["thigh_rms_deg"]),
                "reference_auc": _finite(simulation["reference"]["risk_auc"]),
                "fused_auc": _finite(simulation["fused"]["risk_auc"]),
                "no_emg_auc": float(simulation["no_emg"]["risk_auc"]),
                "reference_fall_risk": _finite(simulation["reference"]["fall_risk"]),
                "fused_fall_risk": _finite(simulation["fused"]["fall_risk"]),
                "no_emg_fall_risk": float(simulation["no_emg"]["fall_risk"]),
                "reference_balance_loss": int(simulation["reference"]["balance_loss_step"]),
                "fused_balance_loss": int(simulation["fused"]["balance_loss_step"]),
                "no_emg_balance_loss": int(simulation["no_emg"]["balance_loss_step"]),
                "reference_recorded_steps": int(simulation["reference"]["recorded_steps"]),
                "fused_recorded_steps": int(simulation["fused"]["recorded_steps"]),
                "no_emg_recorded_steps": int(simulation["no_emg"]["recorded_steps"]),
                "required_steps": int(match["length"]),
                "clip_id": str(match["clip_id"]),
                "snippet_id": str(match["snippet_id"]),
                "candidate_rank": int(match["candidate_rank"]),
                "recording_npz": str(summary["recording_npz"]),
                "no_emg_recording_npz": str(summary["no_emg_recording_npz"] or ""),
                "recording_gif": str(summary["recording_gif"] or ""),
                "media_rendered": bool(summary["media_rendered"]),
            }
            row["fused_excess_auc"] = row["fused_auc"] - row["reference_auc"]
            row["no_emg_excess_auc"] = (
                row["no_emg_auc"] - row["reference_auc"]
                if math.isfinite(row["no_emg_auc"])
                else float("nan")
            )
            npz_path = Path(row["recording_npz"])
            if not npz_path.exists():
                raise RuntimeError(f"Missing V4 recording: {npz_path}")
            fingerprint = (
                round(row["reference_auc"], 12),
                round(row["reference_fall_risk"], 12),
                row["reference_recorded_steps"],
            )
            reference_fingerprints.setdefault(row["query_id"], set()).add(fingerprint)
            stage_rows.append(row)
        stage_rows.sort(key=lambda item: item["panel_index"])
        query_ids = [str(row["query_id"]) for row in stage_rows]
        if expected_query_ids is None:
            expected_query_ids = query_ids
        elif query_ids != expected_query_ids:
            raise RuntimeError(f"Window panel changed at {checkpoint}")
        rows.extend(stage_rows)

    primary_rows = sorted(
        [row for row in rows if row["checkpoint"] == PRIMARY_CHECKPOINT],
        key=lambda item: item["panel_index"],
    )
    if len(primary_rows) != 80 or not all(math.isfinite(row["no_emg_auc"]) for row in primary_rows):
        raise RuntimeError("Primary V4 stage lacks paired no-sEMG outcomes")
    rendered_indices = sorted(int(row["panel_index"]) for row in primary_rows if row["media_rendered"])
    if rendered_indices != [0, 11, 22, 33, 44, 55, 66, 77]:
        raise RuntimeError(f"Unexpected V4 media indices: {rendered_indices}")
    if any(row["media_rendered"] for row in rows if row["checkpoint"] != PRIMARY_CHECKPOINT):
        raise RuntimeError("Secondary checkpoint unexpectedly contains rendered media")

    primary_by_query = {str(row["query_id"]): row for row in primary_rows}
    paired_early_terminations: list[dict[str, Any]] = []
    for row in rows:
        primary = primary_by_query[str(row["query_id"])]
        if row["reference_recorded_steps"] == primary["reference_recorded_steps"]:
            if (
                round(row["reference_auc"], 12),
                round(row["reference_fall_risk"], 12),
                row["reference_recorded_steps"],
            ) != (
                round(primary["reference_auc"], 12),
                round(primary["reference_fall_risk"], 12),
                primary["reference_recorded_steps"],
            ):
                raise RuntimeError(f"Full-length reference outcome changed: {row['query_id']}")
            continue
        if row["reference_recorded_steps"] >= primary["reference_recorded_steps"]:
            raise RuntimeError(f"Unexpected reference length change: {row['query_id']}")
        if row["fused_recorded_steps"] != row["reference_recorded_steps"]:
            raise RuntimeError(f"Early paired recording lengths differ: {row['query_id']}")
        short_path = Path(str(row["recording_npz"]))
        full_path = Path(str(primary["recording_npz"]))
        with np.load(short_path, allow_pickle=False) as short, np.load(full_path, allow_pickle=False) as full:
            n_steps = int(row["reference_recorded_steps"])
            for field in (
                "knee_ref_actual_deg",
                "root_z_ref_m",
                "upright_ref",
                "balance_xcom_margin_ref_m",
                "predicted_fall_risk_trace_ref",
            ):
                short_values = np.asarray(short[field])
                full_values = np.asarray(full[field])[:n_steps]
                if not np.array_equal(short_values, full_values):
                    raise RuntimeError(
                        f"Early reference is not an exact deterministic prefix for {row['query_id']} ({field})"
                    )
        paired_early_terminations.append(
            {
                "checkpoint": row["checkpoint"],
                "query_id": row["query_id"],
                "recorded_steps": row["reference_recorded_steps"],
                "required_steps": row["required_steps"],
                "fused_balance_loss_step": row["fused_balance_loss"],
                "handling": "retained exactly as recorded in the prespecified common-horizon AUC",
            }
        )

    _write_csv(output_dir / "window_metrics.csv", rows)
    primary_participants = _aggregate(primary_rows, ("subject",))
    checkpoint_participants = _aggregate(rows, ("checkpoint", "subject"))
    _write_csv(output_dir / "participant_primary.csv", primary_participants)
    _write_csv(output_dir / "participant_checkpoint.csv", checkpoint_participants)

    confirmation_ablation = _json(confirmation_dir / "ablation_summary.json")
    confirmation_fused = _json(confirmation_dir / "fused" / "result.json")
    confirmation_no_emg = _json(confirmation_dir / "no_emg" / "result.json")
    fused_by_subject = {
        str(row["subject"]): float(row["rmse_deg"])
        for row in confirmation_fused["test"]["participants"]
    }
    no_emg_by_subject = {
        str(row["subject"]): float(row["rmse_deg"])
        for row in confirmation_no_emg["test"]["participants"]
    }
    if set(fused_by_subject) != set(no_emg_by_subject) or len(fused_by_subject) != 90:
        raise RuntimeError("Confirmation ablation does not contain 90 paired participants")
    confirmation_participants = [
        {
            "subject": subject,
            "source_rmse_fused_deg": fused_by_subject[subject],
            "source_rmse_no_emg_deg": no_emg_by_subject[subject],
        }
        for subject in sorted(fused_by_subject)
    ]

    rng = np.random.default_rng(SEED)
    fused_excess = np.asarray([row["fused_excess_auc"] for row in primary_participants])
    no_excess = np.asarray([row["no_emg_excess_auc"] for row in primary_participants])
    physical_delta = no_excess - fused_excess
    fused_stats = _mean_ci(fused_excess, rng)
    no_stats = _mean_ci(no_excess, rng)
    delta_stats = _mean_ci(physical_delta, rng)
    physical_p = _paired_signflip_p(physical_delta, rng)

    x = np.asarray([row["source_rmse_fused_deg"] for row in primary_participants])
    y = fused_excess
    controls = np.column_stack(
        [
            [row["match_knee_rmse_deg"] for row in primary_participants],
            [row["match_thigh_rms_deg"] for row in primary_participants],
        ]
    )
    raw_spearman = stats.spearmanr(x, y)
    partial = _partial_spearman(x, y, controls, rng)

    checkpoint_fraction = {
        "fraction_005pct": 5.0,
        "fraction_010pct": 10.0,
        "fraction_020pct": 20.0,
        "fraction_040pct": 40.0,
        "fraction_060pct": 60.0,
        "fraction_080pct": 80.0,
        "fraction_100pct": 100.0,
    }
    checkpoint_summary: list[dict[str, Any]] = []
    for checkpoint in CHECKPOINT_LABELS:
        members = [row for row in checkpoint_participants if row["checkpoint"] == checkpoint]
        rmse_values = np.asarray([row["source_rmse_fused_deg"] for row in members])
        auc_values = np.asarray([row["fused_excess_auc"] for row in members])
        rmse_stats = _mean_ci(rmse_values, rng)
        auc_stats = _mean_ci(auc_values, rng)
        checkpoint_summary.append(
            {
                "checkpoint": checkpoint,
                "fraction": checkpoint_fraction[checkpoint],
                "n_participants": len(members),
                "mean_rmse_deg": rmse_stats[0],
                "rmse_ci_low": rmse_stats[1],
                "rmse_ci_high": rmse_stats[2],
                "mean_excess_auc": auc_stats[0],
                "auc_ci_low": auc_stats[1],
                "auc_ci_high": auc_stats[2],
                "fused_balance_losses": int(sum(row["fused_balance_losses"] for row in members)),
            }
        )

    change = _segmented_fit(
        np.asarray([row["mean_rmse_deg"] for row in checkpoint_summary]),
        np.asarray([row["mean_excess_auc"] for row in checkpoint_summary]),
    )
    _write_csv(output_dir / "checkpoint_summary.csv", checkpoint_summary)

    _plot_primary_ablation(confirmation_participants, figures_dir / "prediction_ablation.png")
    _plot_physics_pair(primary_participants, figures_dir / "paired_physics.png")
    _plot_fwl(primary_participants, partial, figures_dir / "fwl_partial_spearman.png")
    _plot_accuracy_path(checkpoint_summary, change, figures_dir / "accuracy_path.png")
    simulation_sequence = _plot_simulation_sequence(primary_rows, figures_dir / "simulation_sequence.png")
    simulation_metrics = _plot_simulation_metrics(primary_rows, figures_dir / "simulation_metrics.png")

    audit = {
        "version": VERSION,
        "passed": True,
        "physics_version": physics["version"],
        "physics_summary": str((run_dir / "physics_summary.json").resolve()),
        "physics_summary_sha256": _sha256(run_dir / "physics_summary.json"),
        "protocol_sha256": _sha256(run_dir / "physics_protocol.level_walking_v4_1.json"),
        "n_windows": 80,
        "n_checkpoints": 7,
        "n_simulations": len(rows),
        "n_participants": len(primary_participants),
        "rendered_primary_panel_indices": rendered_indices,
        "full_length_reference_outcomes_identical_across_checkpoints": True,
        "paired_early_terminations": paired_early_terminations,
        "n_paired_early_terminations": len(paired_early_terminations),
        "early_reference_trajectories_are_exact_prefixes": True,
        "all_selected_clips_in_fixed_level_walking_whitelist": True,
        "prediction_falls_and_truncations_retained": True,
        "primary_reference_balance_losses": int(sum(row["reference_balance_loss"] >= 0 for row in primary_rows)),
        "primary_fused_balance_losses": int(sum(row["fused_balance_loss"] >= 0 for row in primary_rows)),
        "primary_no_emg_balance_losses": int(sum(row["no_emg_balance_loss"] >= 0 for row in primary_rows)),
        "all_checkpoint_fused_balance_losses": int(sum(row["fused_balance_loss"] >= 0 for row in rows)),
        "all_checkpoint_fused_truncated": int(sum(row["fused_recorded_steps"] < row["required_steps"] for row in rows)),
        "simulation_sequence": simulation_sequence,
        "simulation_metrics": simulation_metrics,
    }
    statistics = {
        "version": VERSION,
        "participant_is_statistical_unit": True,
        "confirmed_prediction_ablation": {
            "source": str((confirmation_dir / "ablation_summary.json").resolve()),
            "source_sha256": _sha256(confirmation_dir / "ablation_summary.json"),
            "n_participants": int(confirmation_ablation["participant_count"]),
            "fused_mean_rmse_deg": float(confirmation_ablation["fused_mean_participant_rmse_deg"]),
            "no_emg_mean_rmse_deg": float(confirmation_ablation["no_emg_mean_participant_rmse_deg"]),
            "mean_improvement_deg": float(confirmation_ablation["mean_improvement_deg"]),
            "bootstrap_95_ci_deg": [float(value) for value in confirmation_ablation["bootstrap_95pct_ci_deg"]],
            "randomization_p_two_sided": float(confirmation_ablation["two_sided_randomization_p"]),
            "positive_participants": int(confirmation_ablation["positive_participants"]),
            "passed": bool(confirmation_ablation["passed"]),
        },
        "primary_physics": {
            "n_participants": len(primary_participants),
            "n_windows": len(primary_rows),
            "mean_fused_excess_auc": fused_stats[0],
            "fused_excess_auc_bootstrap_95_ci": [fused_stats[1], fused_stats[2]],
            "mean_no_emg_excess_auc": no_stats[0],
            "no_emg_excess_auc_bootstrap_95_ci": [no_stats[1], no_stats[2]],
            "mean_auc_reduction_fused_vs_no_emg": delta_stats[0],
            "auc_reduction_bootstrap_95_ci": [delta_stats[1], delta_stats[2]],
            "paired_signflip_p_two_sided": physical_p,
            "participants_with_lower_excess_auc_under_fusion": int(np.sum(physical_delta > 0.0)),
        },
        "rmse_vs_physical_outcome": {
            "raw_spearman_rho": float(raw_spearman.statistic),
            "raw_spearman_p_two_sided": float(raw_spearman.pvalue),
            "partial_spearman_controls": ["mean match knee RMSE", "mean thigh matching error"],
            **partial,
        },
        "accuracy_path": checkpoint_summary,
        "change_point": change,
        "bootstrap_resamples": N_BOOTSTRAP,
        "permutation_resamples": N_PERMUTATIONS,
        "random_seed": SEED,
    }
    _atomic_json(output_dir / "audit_summary.json", audit)
    _atomic_json(output_dir / "statistical_summary.json", statistics)
    _atomic_json(
        output_dir / "analysis_manifest.json",
        {
            "version": VERSION,
            "audit_summary": str((output_dir / "audit_summary.json").resolve()),
            "statistical_summary": str((output_dir / "statistical_summary.json").resolve()),
            "figures": sorted(str(path.resolve()) for path in figures_dir.glob("*.png")),
            "source_physics_protocol": protocol,
        },
    )
    print(json.dumps({"audit": audit, "statistics": statistics}, indent=2))


if __name__ == "__main__":
    main()
