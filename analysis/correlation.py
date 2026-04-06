from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TrialRecord:
    run_id: str
    query_id: str
    summary_path: str
    predictor_knee_rmse_deg: float
    outcome_value: float
    control_match_knee_rmse_deg: float
    control_thigh_rms_deg: float
    outcome_source: str


@dataclass(frozen=True)
class PartialSpearmanResult:
    n_total_trials: int
    n_usable_trials: int
    n_skipped_trials: int
    skipped_reasons: dict[str, int]
    predictor: str
    outcome: str
    controls: list[str]
    rho_partial_spearman: float
    t_statistic: float
    p_value_two_sided: float
    degrees_of_freedom: int
    run_ids: list[str]
    trials_csv: str
    summary_json: str
    residual_plot: str | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the paper's partial Spearman correlation from mocap_phys_eval artifacts."
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Specific run directory to include (repeatable). Defaults to the latest run under --runs-root.",
    )
    parser.add_argument(
        "--runs-root",
        default=str(Path("artifacts") / "phys_eval_v2" / "runs"),
        help="Root directory containing mocap_phys_eval run folders.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Aggregate trials across every run under --runs-root instead of just the latest one.",
    )
    parser.add_argument(
        "--outcome",
        choices=("excess_auc", "pred_auc", "ref_auc", "excess_score", "pred_score", "ref_score"),
        default="excess_auc",
        help=(
            "Outcome to analyze. Publication default is 'excess_auc' = "
            "sim.pred.instability_auc - sim.ref.instability_auc."
        ),
    )
    parser.add_argument(
        "--rollout",
        choices=("pred", "good", "bad"),
        default="pred",
        help="Which override rollout to analyze for non-excess outcomes. The paper uses 'pred'.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the CSV/JSON/plot outputs. Defaults to <run_dir>/analysis or <runs_root>/analysis_all_runs.",
    )
    return parser.parse_args()


def _latest_run_dir(runs_root: Path) -> Path:
    run_dirs = sorted(p for p in runs_root.iterdir() if p.is_dir())
    if not run_dirs:
        raise SystemExit(f"No run directories found under {runs_root}.")
    return run_dirs[-1]


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    explicit = [Path(p).expanduser().resolve() for p in args.run_dir]
    if explicit:
        return explicit

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists():
        raise SystemExit(f"Runs root does not exist: {runs_root}")
    if args.all_runs:
        run_dirs = sorted(p.resolve() for p in runs_root.iterdir() if p.is_dir())
        if not run_dirs:
            raise SystemExit(f"No run directories found under {runs_root}.")
        return run_dirs
    return [_latest_run_dir(runs_root)]


def _resolve_output_dir(args: argparse.Namespace, run_dirs: list[Path]) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    if len(run_dirs) == 1:
        return (run_dirs[0] / "analysis").resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    return (runs_root / "analysis_all_runs").resolve()


def _eval_summary_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "evals").glob("*/summary.json"))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _resolve_artifact_path(summary_path: Path, artifact_path: str | None) -> Path | None:
    if not artifact_path:
        return None
    p = Path(artifact_path)
    if p.is_absolute() and p.exists():
        return p

    candidates = [Path.cwd() / p, summary_path.parent / p]
    candidates.extend(parent / p for parent in summary_path.parents)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return None


def _risk_auc(trace: np.ndarray, dt: float) -> float:
    x = np.asarray(trace, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 1:
        return float("nan")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        return float("nan")
    if x.size == 1:
        return float(x[0] * float(dt))
    return float(np.trapezoid(x, dx=float(dt)))


def _summary_rollout_keys(rollout: str) -> tuple[str, ...]:
    key = str(rollout).strip().lower()
    if key == "pred":
        return ("pred", "good")
    if key == "good":
        return ("good", "pred")
    return (key,)


def _trace_rollout_keys(rollout: str) -> tuple[str, ...]:
    key = str(rollout).strip().lower()
    if key in {"pred", "good"}:
        return ("pred", "good")
    return (key,)


def _extract_balance_risk_auc(summary: dict[str, Any], summary_path: Path, *, rollout: str) -> tuple[float, str]:
    sim_all = summary.get("sim", {})
    for key in _summary_rollout_keys(rollout):
        sim = sim_all.get(key, {})
        auc = _safe_float(sim.get("instability_auc"))
        if math.isfinite(auc):
            return auc, f"summary.sim.{key}.instability_auc"
        auc = _safe_float(sim.get("balance_risk_auc"))
        if math.isfinite(auc):
            return auc, f"summary.sim.{key}.balance_risk_auc"

    npz_path = _resolve_artifact_path(summary_path, summary.get("artifacts", {}).get("compare_npz"))
    if npz_path is None:
        return float("nan"), "missing"

    with np.load(npz_path, allow_pickle=True) as npz:
        dt = _safe_float(np.asarray(npz["dt"]).reshape(()))
        for key in _trace_rollout_keys(rollout):
            trace_key = f"predicted_fall_risk_trace_{key}"
            if trace_key not in npz:
                continue
            auc = _risk_auc(np.asarray(npz[trace_key], dtype=np.float32), dt=dt)
            return auc, f"npz.{trace_key}"
    return float("nan"), "missing"


def _extract_instability_score(summary: dict[str, Any], summary_path: Path, *, rollout: str) -> tuple[float, str]:
    sim_all = summary.get("sim", {})
    for key in _summary_rollout_keys(rollout):
        sim = sim_all.get(key, {})
        score = _safe_float(sim.get("instability_score"))
        if math.isfinite(score):
            return score, f"summary.sim.{key}.instability_score"
        score = _safe_float(sim.get("predicted_fall_risk"))
        if math.isfinite(score):
            return score, f"summary.sim.{key}.predicted_fall_risk"

    npz_path = _resolve_artifact_path(summary_path, summary.get("artifacts", {}).get("compare_npz"))
    if npz_path is None:
        return float("nan"), "missing"

    with np.load(npz_path, allow_pickle=True) as npz:
        for key in _trace_rollout_keys(rollout):
            scalar_key = f"predicted_fall_risk_{key}"
            if scalar_key not in npz:
                continue
            score = _safe_float(np.asarray(npz[scalar_key]).reshape(()))
            if math.isfinite(score):
                return score, f"npz.{scalar_key}"
    return float("nan"), "missing"


def _extract_outcome(summary: dict[str, Any], summary_path: Path, *, rollout: str, outcome: str) -> tuple[float, str]:
    mode = str(outcome).strip().lower()
    sim_all = summary.get("sim", {})

    if mode == "excess_auc":
        exc = sim_all.get("excess", {})
        y = _safe_float(exc.get("instability_auc_delta"))
        if math.isfinite(y):
            return y, "summary.sim.excess.instability_auc_delta"
        y = _safe_float(exc.get("balance_risk_auc_delta"))
        if math.isfinite(y):
            return y, "summary.sim.excess.balance_risk_auc_delta"
        y_pred, src_pred = _extract_balance_risk_auc(summary, summary_path, rollout="pred")
        y_ref, src_ref = _extract_balance_risk_auc(summary, summary_path, rollout="ref")
        if math.isfinite(y_pred) and math.isfinite(y_ref):
            return float(y_pred - y_ref), f"derived({src_pred}-{src_ref})"
        return float("nan"), "missing"

    if mode == "excess_score":
        exc = sim_all.get("excess", {})
        y = _safe_float(exc.get("instability_score_delta"))
        if math.isfinite(y):
            return y, "summary.sim.excess.instability_score_delta"
        y_pred, src_pred = _extract_instability_score(summary, summary_path, rollout="pred")
        y_ref, src_ref = _extract_instability_score(summary, summary_path, rollout="ref")
        if math.isfinite(y_pred) and math.isfinite(y_ref):
            return float(y_pred - y_ref), f"derived({src_pred}-{src_ref})"
        return float("nan"), "missing"

    if mode == "pred_auc":
        return _extract_balance_risk_auc(summary, summary_path, rollout=rollout)
    if mode == "ref_auc":
        return _extract_balance_risk_auc(summary, summary_path, rollout="ref")
    if mode == "pred_score":
        return _extract_instability_score(summary, summary_path, rollout=rollout)
    if mode == "ref_score":
        return _extract_instability_score(summary, summary_path, rollout="ref")

    return float("nan"), "missing"


def _load_trial(summary_path: Path, *, rollout: str, outcome: str) -> tuple[TrialRecord | None, str | None]:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"load_error:{type(exc).__name__}"

    x = _safe_float(summary.get("model", {}).get("pred_vs_gt_knee_flex_rmse_deg"))
    if not math.isfinite(x):
        return None, "missing_predictor_rmse"

    y, y_source = _extract_outcome(summary, summary_path, rollout=rollout, outcome=outcome)
    if not math.isfinite(y):
        return None, f"missing_outcome:{outcome}"

    k = _safe_float(summary.get("match", {}).get("rmse_knee_deg"))
    if not math.isfinite(k):
        return None, "missing_match_rmse"

    h = _safe_float(summary.get("match", {}).get("rms_thigh_ori_err_deg"))
    if not math.isfinite(h):
        return None, "missing_thigh_rmse"

    run_id = summary_path.parents[2].name
    query_id = str(summary.get("query_id", summary_path.parent.name))
    return (
        TrialRecord(
            run_id=run_id,
            query_id=query_id,
            summary_path=str(summary_path),
            predictor_knee_rmse_deg=x,
            outcome_value=y,
            control_match_knee_rmse_deg=k,
            control_thigh_rms_deg=h,
            outcome_source=y_source,
        ),
        None,
    )


def _rank(x: np.ndarray) -> np.ndarray:
    return np.asarray(stats.rankdata(np.asarray(x, dtype=np.float64), method="average"), dtype=np.float64)


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim == 1:
        controls = controls[:, None]
    design = np.column_stack([np.ones((y.shape[0],), dtype=np.float64), controls])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _write_trials_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit("No usable trials were available after filtering.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(
    *,
    out_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    rho: float,
    p_value: float,
    n: int,
    rollout: str,
) -> str | None:
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(x, y, s=40, alpha=0.85, edgecolors="none")
    if n >= 2 and np.unique(x).size >= 2:
        m, b = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, m * xs + b, color="tab:red", lw=1.5)
    ax.set_xlabel("Residualized rank(model knee RMSE)")
    ax.set_ylabel(f"Residualized rank({rollout}) outcome")
    ax.set_title(f"Partial Spearman rho={rho:.4f}, p={p_value:.4g}, n={n}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def main() -> None:
    args = _parse_args()
    run_dirs = _resolve_run_dirs(args)
    output_dir = _resolve_output_dir(args, run_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_trials = 0
    skipped_reasons: dict[str, int] = {}
    trials: list[TrialRecord] = []

    for run_dir in run_dirs:
        for summary_path in _eval_summary_paths(run_dir):
            total_trials += 1
            trial, skip_reason = _load_trial(summary_path, rollout=args.rollout, outcome=args.outcome)
            if trial is None:
                if skip_reason is not None:
                    skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1
                continue
            trials.append(trial)

    if not trials:
        raise SystemExit(
            "No usable trials were found. The usual cause is oracle/demo runs where "
            "`model.pred_vs_gt_knee_flex_rmse_deg` is null."
        )

    x = np.asarray([t.predictor_knee_rmse_deg for t in trials], dtype=np.float64)
    y = np.asarray([t.outcome_value for t in trials], dtype=np.float64)
    k = np.asarray([t.control_match_knee_rmse_deg for t in trials], dtype=np.float64)
    h = np.asarray([t.control_thigh_rms_deg for t in trials], dtype=np.float64)

    xr = _rank(x)
    yr = _rank(y)
    kr = _rank(k)
    hr = _rank(h)

    controls = np.column_stack([kr, hr])
    x_res = _residualize(xr, controls)
    y_res = _residualize(yr, controls)

    rho = _pearson_r(x_res, y_res)
    q = int(controls.shape[1])
    n = int(len(trials))
    df = int(n - 2 - q)
    if not math.isfinite(rho) or abs(rho) >= 1.0 or df <= 0:
        t_stat = float("nan")
        p_value = float("nan")
    else:
        t_stat = float(rho * math.sqrt(df) / math.sqrt(max(1e-12, 1.0 - rho * rho)))
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df))

    trial_rows: list[dict[str, Any]] = []
    for i, trial in enumerate(trials):
        row = asdict(trial)
        row.update(
            rank_predictor_knee_rmse=float(xr[i]),
            rank_outcome_value=float(yr[i]),
            rank_control_match_knee_rmse=float(kr[i]),
            rank_control_thigh_rms=float(hr[i]),
            residual_predictor=float(x_res[i]),
            residual_outcome=float(y_res[i]),
        )
        trial_rows.append(row)

    csv_path = output_dir / "partial_spearman_trials.csv"
    _write_trials_csv(csv_path, trial_rows)

    plot_path = output_dir / "partial_spearman_scatter.png"
    residual_plot: str | None
    try:
        residual_plot = _write_plot(
            out_path=plot_path,
            x=x_res,
            y=y_res,
            rho=rho,
            p_value=p_value,
            n=n,
            rollout=args.outcome,
        )
    except Exception:
        residual_plot = None

    summary_path = output_dir / "partial_spearman_summary.json"
    result = PartialSpearmanResult(
        n_total_trials=total_trials,
        n_usable_trials=n,
        n_skipped_trials=int(total_trials - n),
        skipped_reasons=skipped_reasons,
        predictor="model.pred_vs_gt_knee_flex_rmse_deg",
        outcome=str(args.outcome),
        controls=["match.rmse_knee_deg", "match.rms_thigh_ori_err_deg"],
        rho_partial_spearman=float(rho),
        t_statistic=float(t_stat),
        p_value_two_sided=float(p_value),
        degrees_of_freedom=int(df),
        run_ids=sorted({trial.run_id for trial in trials}),
        trials_csv=str(csv_path),
        summary_json=str(summary_path),
        residual_plot=residual_plot,
    )
    summary_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    print(f"runs={len(run_dirs)} total_trials={total_trials} usable_trials={n} df={df}")
    print(f"rho_partial_spearman={rho:.6f}")
    print(f"t_statistic={t_stat:.6f}")
    print(f"p_value_two_sided={p_value:.6g}")
    print(f"summary_json={summary_path}")
    print(f"trials_csv={csv_path}")
    if residual_plot is not None:
        print(f"residual_plot={residual_plot}")


if __name__ == "__main__":
    main()
