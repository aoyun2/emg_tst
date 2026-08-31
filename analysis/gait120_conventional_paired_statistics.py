from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats


def one_sample_t(values: np.ndarray) -> dict[str, float | int | list[float]]:
    values = np.asarray(values, dtype=float)
    result = stats.ttest_1samp(values, 0.0, alternative="two-sided")
    interval = stats.t.interval(
        0.95,
        values.size - 1,
        loc=float(np.mean(values)),
        scale=float(stats.sem(values)),
    )
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)),
        "t": float(result.statistic),
        "df": int(result.df),
        "p_two_sided": float(result.pvalue),
        "ci_95": [float(interval[0]), float(interval[1])],
        "cohen_dz": float(np.mean(values) / np.std(values, ddof=1)),
    }


def diagnostic_and_sensitivity(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    shapiro = stats.shapiro(values)
    wilcoxon = stats.wilcoxon(
        values,
        zero_method="wilcox",
        alternative="two-sided",
        method="approx",
    )
    sign = stats.binomtest(
        int(np.sum(values > 0)),
        int(values.size),
        0.5,
        alternative="two-sided",
    )
    return {
        "shapiro_w": float(shapiro.statistic),
        "shapiro_p": float(shapiro.pvalue),
        "skewness": float(stats.skew(values, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(values, bias=False)),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_p_two_sided": float(wilcoxon.pvalue),
        "positive_count": int(np.sum(values > 0)),
        "exact_sign_p_two_sided": float(sign.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        required=True,
        help="Extracted Additional_file_2_reproducibility_evidence directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence_dir.resolve()

    ablation = json.loads(
        (evidence / "prediction_confirmation" / "ablation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    primary_difference = np.asarray(ablation["improvement_deg"], dtype=float)

    temporal = json.loads(
        (
            evidence
            / "temporal_control"
            / "temporal_control_summary.json"
        ).read_text(encoding="utf-8")
    )
    temporal_results: dict[str, object] = {}
    for key in ("aligned_common_support", "lagged_control", "aligned_vs_lagged"):
        values = np.asarray(temporal[key]["improvement_deg"], dtype=float)
        temporal_results[key] = {
            "paired_t": one_sample_t(values),
            "diagnostic_and_sensitivity": diagnostic_and_sensitivity(values),
        }

    physics = pd.read_csv(evidence / "analysis" / "participant_primary.csv")
    # Averages within participant before the test.
    _subject = next(c for c in physics.columns if "subject" in c.lower())
    _per_window = (
        physics["no_emg_excess_auc"].to_numpy(dtype=float)
        - physics["fused_excess_auc"].to_numpy(dtype=float)
    )
    physics_difference = (
        pd.DataFrame({"subject": physics[_subject], "d": _per_window})
        .groupby("subject")["d"]
        .mean()
        .to_numpy(dtype=float)
    )

    report = {
        "definition": (
            "All comparisons are participant-level paired differences. "
            "Positive prediction differences favor residual fusion."
        ),
        "prediction_ablation": {
            "paired_t": one_sample_t(primary_difference),
            "diagnostic_and_sensitivity": diagnostic_and_sensitivity(
                primary_difference
            ),
        },
        "temporal_control": temporal_results,
        "physics_comparison": {
            "difference_definition": (
                "without-sEMG excess AUC minus residual-fusion excess AUC, "
            "averaged within participant before testing"
            ),
            "paired_t": one_sample_t(physics_difference),
            "diagnostic_and_sensitivity": diagnostic_and_sensitivity(
                physics_difference
            ),
        },
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
