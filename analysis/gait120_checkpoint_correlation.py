"""Match-adjusted RMSE / excess-instability association along the training path.

The confirmation experiment reports one association, measured at one accuracy
level, from a model that converges to roughly 4.7 degrees.  A null there is
ambiguous: it is consistent with prediction error genuinely not driving
simulated instability, but it is equally consistent with a converged model whose
windows are all so similar in accuracy that no association could be resolved.

Replaying the fixed physics panel at every checkpoint of the training path
separates those explanations.  Each checkpoint is a different accuracy level
evaluated on the *same* windows and the *same* matched reference motions, so the
association can be traced from a barely-trained model down to the converged one
and the accuracy at which it disappears can be located.

Three views are reported:

``per_checkpoint``
    Partial Spearman between window prediction RMSE and excess instability,
    controlling for match quality, computed across windows within one
    checkpoint.  This is the confirmation analysis repeated at each accuracy
    level, and each row carries the RMSE spread that was available to it.

``within_window``
    For each window, the Spearman correlation across checkpoints between its own
    RMSE and its own excess instability, combined over windows with a
    Fisher-z one-sample test.  Every window is its own control here, so matched
    reference motion, snippet, and initial state all cancel.

``pooled``
    All checkpoint-window pairs together, which uses the full accuracy range but
    reuses each window once per checkpoint; its interval is reported from a
    participant-level cluster bootstrap rather than assuming independence.

against mean accuracy, alongside the simpler statement of which checkpoints had
intervals excluding zero.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from analysis.correlation import _pearson_r, _rank, _residualize


VERSION = "GAIT120_CHECKPOINT_CORRELATION_V1"

BOOTSTRAP_DRAWS = 10_000
PERMUTATION_DRAWS = 20_000
SEED = 20260827

# A checkpoint needs enough windows for a partial correlation with two controls
# to mean anything; below this it is reported but not used for the drop-off fit.
MIN_USABLE_WINDOWS = 20


@dataclass(frozen=True)
class WindowRow:
    checkpoint: str
    query_id: str
    subject: str
    panel_index: int
    prediction_rmse_deg: float
    excess_instability_auc: float
    reference_auc: float
    fused_auc: float
    match_knee_rmse_deg: float
    match_thigh_rms_deg: float
    # The without-sEMG condition is only simulated at the primary checkpoint,
    # which is where the paired physics ablation is defined; elsewhere these are
    # None.
    has_no_emg: bool = False
    no_emg_auc: float = float("nan")
    no_emg_excess_auc: float = float("nan")
    no_emg_prediction_rmse_deg: float = float("nan")
    # A rollout stops as soon as any condition falls, so its instability trace is
    # shorter and its integrated area smaller. Both conditions stop together, so
    # the paired difference stays like-for-like, but the magnitude is attenuated
    # for exactly the windows that destabilised -- which are concentrated at the
    # least accurate checkpoints. Carrying duration lets that be measured instead
    # of silently shrinking the association where it should be largest.
    recorded_steps: int = 0
    expected_steps: int = 0
    dt: float = float("nan")


def _load_rows_from_csv(path: Path) -> list[WindowRow]:
    """Read the per-window table shipped with the reproducibility supplement."""
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise RuntimeError(f"No rows in {path}")
    missing = {
        "checkpoint", "query_id", "subject", "panel_index", "prediction_rmse_deg",
        "reference_auc", "fused_auc", "excess_instability_auc",
        "match_knee_rmse_deg", "match_thigh_rms_deg", "recorded_steps",
        "expected_steps",
    } - set(records[0])
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
    return [
        WindowRow(
            checkpoint=r["checkpoint"],
            query_id=r["query_id"],
            subject=r["subject"],
            panel_index=int(r["panel_index"]),
            prediction_rmse_deg=float(r["prediction_rmse_deg"]),
            excess_instability_auc=float(r["excess_instability_auc"]),
            reference_auc=float(r["reference_auc"]),
            fused_auc=float(r["fused_auc"]),
            match_knee_rmse_deg=float(r["match_knee_rmse_deg"]),
            match_thigh_rms_deg=float(r["match_thigh_rms_deg"]),
            recorded_steps=float(r["recorded_steps"]),
            expected_steps=float(r["expected_steps"]),
        )
        for r in records
    ]


def _load_rows(physics_run_dir: Path) -> list[WindowRow]:
    rows: list[WindowRow] = []
    for summary_path in sorted(physics_run_dir.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "checkpoint" not in summary or "simulation" not in summary:
            continue
        simulation = summary["simulation"]
        if "reference" not in simulation or "fused" not in simulation:
            continue
        reference_auc = float(simulation["reference"]["risk_auc"])
        fused_auc = float(simulation["fused"]["risk_auc"])
        match = summary.get("match") or {}
        has_no_emg = bool(simulation.get("has_no_emg", False))
        no_emg_auc = (
            float(simulation["no_emg"]["risk_auc"])
            if has_no_emg and "no_emg" in simulation
            else float("nan")
        )
        rows.append(
            WindowRow(
                checkpoint=str(summary["checkpoint"]),
                query_id=str(summary["query_id"]),
                subject=str(summary.get("subject", "")),
                panel_index=int(summary.get("panel_index", -1)),
                prediction_rmse_deg=float(summary["prediction_rmse_deg"]["fused"]),
                excess_instability_auc=fused_auc - reference_auc,
                reference_auc=reference_auc,
                fused_auc=fused_auc,
                match_knee_rmse_deg=float(match.get("knee_rmse_deg", np.nan)),
                match_thigh_rms_deg=float(match.get("thigh_rms_deg", np.nan)),
                has_no_emg=has_no_emg,
                no_emg_auc=no_emg_auc,
                no_emg_excess_auc=no_emg_auc - reference_auc,
                no_emg_prediction_rmse_deg=float(
                    summary["prediction_rmse_deg"].get("no_emg", np.nan)
                ),
                recorded_steps=int(simulation["fused"].get("recorded_steps", 0)),
                expected_steps=int(match.get("length", 0)),
                dt=float(simulation.get("dt", np.nan)),
            )
        )
    if not rows:
        raise RuntimeError(f"No checkpoint simulation summaries found under {physics_run_dir}")

    return rows


def partial_spearman(
    predictor: np.ndarray, outcome: np.ndarray, controls: np.ndarray
) -> float:
    """Frisch-Waugh-Lovell partial Spearman, as in the confirmation analysis."""
    ranked_controls = np.column_stack([_rank(controls[:, i]) for i in range(controls.shape[1])])
    return _pearson_r(
        _residualize(_rank(predictor), ranked_controls),
        _residualize(_rank(outcome), ranked_controls),
    )


def _permutation_p(
    predictor: np.ndarray,
    outcome: np.ndarray,
    controls: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    draws: int = PERMUTATION_DRAWS,
) -> float:
    if not np.isfinite(observed):
        return float("nan")
    exceed = 0
    for _ in range(int(draws)):
        shuffled = rng.permutation(predictor)
        value = partial_spearman(shuffled, outcome, controls)
        if np.isfinite(value) and abs(value) >= abs(observed) - 1.0e-15:
            exceed += 1
    return float((exceed + 1.0) / (int(draws) + 1.0))


def _bootstrap_ci(
    predictor: np.ndarray,
    outcome: np.ndarray,
    controls: np.ndarray,
    rng: np.random.Generator,
    draws: int = BOOTSTRAP_DRAWS,
    subjects: np.ndarray | None = None,
) -> tuple[float, float]:
    n = int(predictor.size)
    # Windows are nested within participants, so a draw takes whole
    # participants. Drawing windows would count twice those who contribute two.
    groups: list[np.ndarray] | None = None
    if subjects is not None:
        unique = np.unique(np.asarray(subjects))
        groups = [np.flatnonzero(np.asarray(subjects) == s) for s in unique.tolist()]
    values: list[float] = []
    for _ in range(int(draws)):
        if groups is None:
            index = rng.integers(0, n, size=n)
        else:
            drawn = rng.integers(0, len(groups), size=len(groups))
            index = np.concatenate([groups[i] for i in drawn.tolist()])
        value = partial_spearman(predictor[index], outcome[index], controls[index])
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < 100:
        return (float("nan"), float("nan"))
    lower, upper = np.quantile(np.asarray(values), [0.025, 0.975]).tolist()
    return (float(lower), float(upper))


def _checkpoint_result(
    label: str,
    rows: list[WindowRow],
    rng: np.random.Generator,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    permutation_draws: int = PERMUTATION_DRAWS,
) -> dict[str, Any]:
    predictor = np.asarray([r.prediction_rmse_deg for r in rows], dtype=np.float64)
    outcome = np.asarray([r.excess_instability_auc for r in rows], dtype=np.float64)
    controls = np.column_stack(
        [
            np.asarray([r.match_knee_rmse_deg for r in rows], dtype=np.float64),
            np.asarray([r.match_thigh_rms_deg for r in rows], dtype=np.float64),
        ]
    )
    finite = (
        np.isfinite(predictor)
        & np.isfinite(outcome)
        & np.all(np.isfinite(controls), axis=1)
    )
    subjects = np.asarray([r.subject for r in rows])[finite]
    predictor, outcome, controls = predictor[finite], outcome[finite], controls[finite]
    n = int(predictor.size)

    # Duration-normalised outcome: excess instability per simulated second rather
    # than integrated over however long the rollout survived. If the association
    # holds under both, truncation is not driving it.
    steps = np.asarray([r.recorded_steps for r in rows], dtype=np.float64)[finite]
    expected = np.asarray([r.expected_steps for r in rows], dtype=np.float64)[finite]
    truncated = (expected > 0) & (steps < expected)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_outcome = np.where(steps > 0, outcome / steps, np.nan)
    usable_rate = np.isfinite(rate_outcome)
    rho_rate = (
        partial_spearman(predictor[usable_rate], rate_outcome[usable_rate], controls[usable_rate])
        if int(usable_rate.sum()) >= 5
        else float("nan")
    )

    raw = float(stats.spearmanr(predictor, outcome).statistic) if n >= 3 else float("nan")
    rho = partial_spearman(predictor, outcome, controls) if n >= 5 else float("nan")
    lower, upper = (
        _bootstrap_ci(predictor, outcome, controls, rng, bootstrap_draws,
                      subjects=subjects)
        if n >= 5
        else (np.nan, np.nan)
    )
    p_value = (
        _permutation_p(predictor, outcome, controls, rho, rng, permutation_draws)
        if n >= 5
        else float("nan")
    )
    return {
        "checkpoint": label,
        "n_windows": n,
        "mean_prediction_rmse_deg": float(np.mean(predictor)) if n else float("nan"),
        # The spread available to this checkpoint's correlation.  A near-zero
        # rho where the spread is also near zero is uninformative, not evidence
        # of no association.
        "prediction_rmse_sd_deg": float(np.std(predictor, ddof=1)) if n > 1 else float("nan"),
        "prediction_rmse_iqr_deg": (
            float(np.subtract(*np.percentile(predictor, [75, 25]))) if n else float("nan")
        ),
        "mean_excess_instability_auc": float(np.mean(outcome)) if n else float("nan"),
        "excess_instability_sd": float(np.std(outcome, ddof=1)) if n > 1 else float("nan"),
        # Truncation diagnostics. A checkpoint whose rollouts end early has its
        # excess-instability magnitude attenuated, so a weak association here
        # should be read against how often that happened.
        "truncated_rollouts": int(np.sum(truncated)),
        "truncated_fraction": float(np.mean(truncated)) if n else float("nan"),
        "mean_recorded_steps": float(np.mean(steps)) if n else float("nan"),
        "mean_expected_steps": float(np.mean(expected)) if n else float("nan"),
        "partial_spearman_rho_per_step": float(rho_rate),
        "raw_spearman_rho": raw,
        "partial_spearman_rho": float(rho),
        "bootstrap_95pct_ci": [float(lower), float(upper)],
        "permutation_p_two_sided": float(p_value),
        "interval_excludes_zero": bool(
            np.isfinite(lower) and np.isfinite(upper) and (lower > 0.0 or upper < 0.0)
        ),
        "usable_for_dropoff": bool(n >= MIN_USABLE_WINDOWS),
    }


def _within_window_result(rows: list[WindowRow]) -> dict[str, Any]:
    """Correlate accuracy against instability within each window, then combine."""
    by_query: dict[str, list[WindowRow]] = {}
    for row in rows:
        by_query.setdefault(row.query_id, []).append(row)

    per_window: list[dict[str, Any]] = []
    for query_id, group in sorted(by_query.items()):
        predictor = np.asarray([r.prediction_rmse_deg for r in group], dtype=np.float64)
        outcome = np.asarray([r.excess_instability_auc for r in group], dtype=np.float64)
        if predictor.size < 4 or np.ptp(predictor) <= 0.0 or np.ptp(outcome) <= 0.0:
            continue
        rho = float(stats.spearmanr(predictor, outcome).statistic)
        if not np.isfinite(rho):
            continue
        per_window.append(
            {
                "query_id": query_id,
                "n_checkpoints": int(predictor.size),
                "spearman_rho": rho,
                "rmse_range_deg": [float(np.min(predictor)), float(np.max(predictor))],
            }
        )

    if len(per_window) < 3:
        return {"n_windows": len(per_window), "insufficient": True}

    rho = np.asarray([row["spearman_rho"] for row in per_window], dtype=np.float64)
    # Several participants contribute two windows, so the windows are averaged
    # within participant before combining. Testing all of them together would
    # count those participants twice.
    subject_of = {row.query_id: row.subject for row in rows}
    by_subject: dict[str, list[float]] = {}
    for entry in per_window:
        by_subject.setdefault(subject_of[entry["query_id"]], []).append(
            entry["spearman_rho"]
        )
    subject_rho = np.asarray(
        [float(np.mean(v)) for _, v in sorted(by_subject.items())], dtype=np.float64
    )
    # Fisher-z stabilizes the variance of a correlation before averaging.
    z = np.arctanh(np.clip(subject_rho, -0.999999, 0.999999))
    test = stats.ttest_1samp(z, 0.0)
    mean_z = float(np.mean(z))
    half_width = float(
        stats.t.ppf(0.975, df=z.size - 1) * np.std(z, ddof=1) / np.sqrt(z.size)
    )
    return {
        "n_participants": int(z.size),
        "n_windows": len(per_window),
        "mean_spearman_rho_fisher_back_transformed": float(np.tanh(mean_z)),
        "ci_95pct_rho": [
            float(np.tanh(mean_z - half_width)),
            float(np.tanh(mean_z + half_width)),
        ],
        "t_statistic": float(test.statistic),
        "p_value_two_sided": float(test.pvalue),
        "positive_windows": int(np.sum(rho > 0.0)),
        "positive_participants": int(np.sum(subject_rho > 0.0)),
        "per_window": per_window,
    }


def _pooled_result(rows: list[WindowRow], rng: np.random.Generator) -> dict[str, Any]:
    """Pool every checkpoint-window pair, clustering the bootstrap on participants."""
    predictor = np.asarray([r.prediction_rmse_deg for r in rows], dtype=np.float64)
    outcome = np.asarray([r.excess_instability_auc for r in rows], dtype=np.float64)
    controls = np.column_stack(
        [
            np.asarray([r.match_knee_rmse_deg for r in rows], dtype=np.float64),
            np.asarray([r.match_thigh_rms_deg for r in rows], dtype=np.float64),
        ]
    )
    finite = np.isfinite(predictor) & np.isfinite(outcome) & np.all(np.isfinite(controls), axis=1)
    # The bootstrap clusters on the participant; the reported window count is
    # still the number of windows, which is not the number of clusters.
    windows = np.asarray([r.query_id for r in rows])[finite]
    queries = np.asarray([r.subject for r in rows])[finite]
    predictor, outcome, controls = predictor[finite], outcome[finite], controls[finite]
    rho = partial_spearman(predictor, outcome, controls)

    unique = np.unique(queries)
    index_by_query = {q: np.flatnonzero(queries == q) for q in unique.tolist()}
    values: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS // 10):
        drawn = rng.integers(0, unique.size, size=unique.size)
        index = np.concatenate([index_by_query[unique[i]] for i in drawn.tolist()])
        value = partial_spearman(predictor[index], outcome[index], controls[index])
        if np.isfinite(value):
            values.append(float(value))
    if len(values) >= 100:
        lower, upper = np.quantile(np.asarray(values), [0.025, 0.975]).tolist()
    else:  # pragma: no cover - degenerate panels only
        lower = upper = float("nan")
    return {
        "n_pairs": int(predictor.size),
        "n_windows": int(np.unique(windows).size),
        "n_participants": int(unique.size),
        "prediction_rmse_range_deg": [float(np.min(predictor)), float(np.max(predictor))],
        "prediction_rmse_sd_deg": float(np.std(predictor, ddof=1)),
        "partial_spearman_rho": float(rho),
        "cluster_bootstrap_95pct_ci": [float(lower), float(upper)],
        "note": (
            "Each window contributes once per checkpoint; the interval is a "
            "participant-level cluster bootstrap, not an independent-sample interval."
        ),
    }



def write_participant_primary(rows: list[WindowRow], path: Path) -> int:
    """Write the per-window physics table for the paired simulation ablation.

    Only the primary checkpoint simulates the without-sEMG condition, so that is
    the checkpoint at which a paired physics comparison exists.  This table is
    what ``gait120_conventional_paired_statistics`` consumes.
    """
    paired = [row for row in rows if row.has_no_emg and np.isfinite(row.no_emg_auc)]
    paired.sort(key=lambda row: row.panel_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "query_id",
                "panel_index",
                "subject",
                "checkpoint",
                "source_rmse_fused_deg",
                "source_rmse_no_emg_deg",
                "reference_auc",
                "fused_auc",
                "no_emg_auc",
                "fused_excess_auc",
                "no_emg_excess_auc",
                "match_knee_rmse_deg",
                "thigh_rms_deg",
            ]
        )
        for row in paired:
            writer.writerow(
                [
                    row.query_id,
                    row.panel_index,
                    row.subject,
                    row.checkpoint,
                    f"{row.prediction_rmse_deg:.10g}",
                    f"{row.no_emg_prediction_rmse_deg:.10g}",
                    f"{row.reference_auc:.10g}",
                    f"{row.fused_auc:.10g}",
                    f"{row.no_emg_auc:.10g}",
                    f"{row.excess_instability_auc:.10g}",
                    f"{row.no_emg_excess_auc:.10g}",
                    f"{row.match_knee_rmse_deg:.10g}",
                    f"{row.match_thigh_rms_deg:.10g}",
                ]
            )
    return len(paired)


def accuracy_level_analysis(
    rows: list[WindowRow], rng: np.random.Generator, draws: int = 2000
) -> dict[str, Any]:
    """Relate model accuracy to simulated instability across the training path.

    This is the study's primary question: does prediction RMSE track simulated
    walking outcome, and if not everywhere, where does it stop? The unit here is
    the accuracy level, not the window. Each checkpoint contributes its mean over
    the same fixed panel, so window identity, matched motion, and reference
    stability are all held constant across the comparison and cannot drive it.

    A single monotone coefficient is the wrong summary if the relationship turns,
    so a breakpoint is fitted and the association is reported on each side of it.
    Uncertainty comes from resampling windows within every checkpoint, which is
    the level at which the data actually vary: the checkpoint means themselves are
    smooth by construction and a correlation computed on them alone would look far
    more certain than the evidence supports.
    """
    by_checkpoint: dict[str, list[WindowRow]] = {}
    for row in rows:
        by_checkpoint.setdefault(row.checkpoint, []).append(row)
    labels = sorted(by_checkpoint)

    def summarize(sample: dict[str, list[WindowRow]]) -> tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for label in labels:
            group = sample[label]
            x.append(float(np.mean([r.prediction_rmse_deg for r in group])))
            y.append(float(np.mean([r.excess_instability_auc for r in group])))
        order = np.argsort(x)
        return np.asarray(x)[order], np.asarray(y)[order]

    def breakpoint(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        """Two-segment fit; returns (breakpoint, low-side slope, high-side slope)."""
        best = None
        for split in range(3, x.size - 2):
            sse, slopes = 0.0, []
            for lo, hi in ((0, split), (split, x.size)):
                sx, sy = x[lo:hi], y[lo:hi]
                design = np.column_stack([np.ones(sx.size), sx])
                beta, *_ = np.linalg.lstsq(design, sy, rcond=None)
                sse += float(np.sum(np.square(sy - design @ beta)))
                slopes.append(float(beta[1]))
            if best is None or sse < best[0]:
                best = (sse, float((x[split - 1] + x[split]) / 2.0), slopes[0], slopes[1])
        return best[1], best[2], best[3]

    x, y = summarize(by_checkpoint)
    point, low_slope, high_slope = breakpoint(x, y)
    overall = stats.spearmanr(x, y)
    below = stats.spearmanr(x[x < point], y[x < point])
    above = stats.spearmanr(x[x >= point], y[x >= point])

    # The same windows recur at every checkpoint, so a draw has to select whole
    # participants once and carry them across all of them. Resampling inside
    # each checkpoint separately would treat one window's observations as
    # independent of one another and understate the spread of the curve.
    windows_by_subject: dict[str, list[str]] = {}
    for row in by_checkpoint[labels[0]]:
        windows_by_subject.setdefault(row.subject, []).append(row.query_id)
    subjects = sorted(windows_by_subject)
    indexed = {
        label: {row.query_id: row for row in rows}
        for label, rows in by_checkpoint.items()
    }

    points, lows, highs = [], [], []
    for _ in range(int(draws)):
        drawn = [subjects[i] for i in rng.integers(0, len(subjects), size=len(subjects))]
        query_ids = [q for subject in drawn for q in windows_by_subject[subject]]
        sample = {
            label: [indexed[label][q] for q in query_ids if q in indexed[label]]
            for label in labels
        }
        if any(len(rows) < 3 for rows in sample.values()):
            continue
        bx, by = summarize(sample)
        try:
            bp, ls, hs = breakpoint(bx, by)
        except Exception:  # pragma: no cover - degenerate resample
            continue
        points.append(bp)
        lows.append(ls)
        highs.append(hs)

    def interval(values: list[float]) -> list[float]:
        if len(values) < 100:
            return [float("nan"), float("nan")]
        return [float(v) for v in np.quantile(np.asarray(values), [0.025, 0.975])]

    return {
        "n_accuracy_levels": int(x.size),
        "mean_rmse_deg": x.tolist(),
        "mean_excess_instability": y.tolist(),
        "overall_spearman_rho": float(overall.statistic),
        "overall_p_value": float(overall.pvalue),
        "breakpoint_rmse_deg": float(point),
        "breakpoint_95pct_ci": interval(points),
        "below_breakpoint": {
            "n_levels": int(np.sum(x < point)),
            "spearman_rho": float(below.statistic),
            "p_value": float(below.pvalue),
            "slope_per_degree": low_slope,
            "slope_95pct_ci": interval(lows),
        },
        "above_breakpoint": {
            "n_levels": int(np.sum(x >= point)),
            "spearman_rho": float(above.statistic),
            "p_value": float(above.pvalue),
            "slope_per_degree": high_slope,
            "slope_95pct_ci": interval(highs),
        },
        "split_is_grid_midpoint": True,
        "note": (
            "The split is located by searching checkpoint positions and is "
            "reported as the midpoint of the two adjacent checkpoint means, so "
            "it takes one of a few discrete values rather than being a fitted "
            "changepoint. Its interval is a bootstrap distribution over that "
            "grid. Draws select whole participants and carry them across every "
            "checkpoint, because the same windows recur at all of them."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-run-dir", type=Path, required=False, default=None)
    parser.add_argument(
        "--per-window-csv", type=Path, default=None,
        help="per_window_rollouts.csv from the reproducibility supplement",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.per_window_csv is not None:
        rows = _load_rows_from_csv(args.per_window_csv.resolve())
    elif args.physics_run_dir is not None:
        rows = _load_rows(args.physics_run_dir.resolve())
    else:
        raise SystemExit("pass --physics-run-dir or --per-window-csv")
    rng = np.random.default_rng(SEED)

    by_checkpoint: dict[str, list[WindowRow]] = {}
    for row in rows:
        by_checkpoint.setdefault(row.checkpoint, []).append(row)

    per_checkpoint = [
        _checkpoint_result(label, by_checkpoint[label], rng)
        for label in sorted(by_checkpoint)
    ]
    per_checkpoint.sort(key=lambda row: row["mean_prediction_rmse_deg"])

    result = {
        "version": VERSION,
        "source": (str(args.per_window_csv.resolve())
                   if args.per_window_csv is not None
                   else str(args.physics_run_dir.resolve())),
        "predictor": "window prediction RMSE (degrees), residual fusion",
        "outcome": "excess instability AUC (fused minus paired reference)",
        "controls": ["motion-match knee RMSE", "motion-match thigh orientation RMS"],
        "n_checkpoints": len(per_checkpoint),
        "per_checkpoint": per_checkpoint,
        "within_window": _within_window_result(rows),
        "pooled": _pooled_result(rows, rng),
        # The study's primary question: does model accuracy track simulated
        # outcome, and where does that stop?
        "accuracy_level": accuracy_level_analysis(rows, rng),
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoint_correlation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    n_paired = write_participant_primary(rows, out_dir / "participant_primary.csv")
    print(f"Wrote {n_paired} paired physics rows to participant_primary.csv")

    print(f"{'checkpoint':<28} {'n':>4} {'meanRMSE':>9} {'sd':>6} {'rho':>7}  95% CI")
    for row in per_checkpoint:
        lower, upper = row["bootstrap_95pct_ci"]
        print(
            f"{row['checkpoint']:<28} {row['n_windows']:>4} "
            f"{row['mean_prediction_rmse_deg']:>9.3f} {row['prediction_rmse_sd_deg']:>6.3f} "
            f"{row['partial_spearman_rho']:>7.3f}  [{lower:.3f}, {upper:.3f}]"
        )
    within = result["within_window"]
    if not within.get("insufficient"):
        low, high = within["ci_95pct_rho"]
        print(
            f"\nwithin-window mean rho = "
            f"{within['mean_spearman_rho_fisher_back_transformed']:.3f} "
            f"[{low:.3f}, {high:.3f}], p = {within['p_value_two_sided']:.4g} "
            f"over {within['n_windows']} windows"
        )
    print(f"\nWrote {out_dir / 'checkpoint_correlation.json'}")

if __name__ == "__main__":
    main()
