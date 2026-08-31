"""Sensitivity of the reported association to two design choices.

Early termination
    Refits the association with the windows whose rollouts ended early excluded,
    and with the outcome expressed as a rate per recorded second.

Controller gains
    Reports the oracle preflight, the zero-error case, at the selected gains.

    python -m analysis.gait120_sensitivity --physics-run-dir <panel> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dataclasses import replace

from analysis.gait120_checkpoint_correlation import (
    _load_rows,
    accuracy_level_analysis,
)


def _refit(rows: list, label: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result = accuracy_level_analysis(rows, rng)
    return {
        "label": label,
        "n_rollouts": len(rows),
        "split_rmse_deg": result["breakpoint_rmse_deg"],
        "split_95pct_ci": result["breakpoint_95pct_ci"],
        "slope_above": result["above_breakpoint"]["slope_per_degree"],
        "slope_above_95pct_ci": result["above_breakpoint"]["slope_95pct_ci"],
        "slope_below": result["below_breakpoint"]["slope_per_degree"],
        "slope_below_95pct_ci": result["below_breakpoint"]["slope_95pct_ci"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics-run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    rows = _load_rows(args.physics_run_dir)
    if not rows:
        print("no rollout rows found", file=sys.stderr)
        return 1

    full = max(r.recorded_steps for r in rows)
    truncated = {r.query_id for r in rows if r.recorded_steps < full}
    complete = [r for r in rows if r.query_id not in truncated]

    report: dict[str, Any] = {
        "reported": _refit(rows, "as reported", args.seed),
        "early_termination": {
            "full_length_steps": int(full),
            "truncated_rollouts": int(sum(1 for r in rows if r.recorded_steps < full)),
            "windows_ever_truncated": len(truncated),
            "excluding_those_windows": _refit(
                complete, "windows that ever truncated removed", args.seed
            ),
        },
    }

    # Equal weight per participant.
    from collections import defaultdict
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[(r.subject, r.checkpoint)].append(r)
    balanced = []
    for (_subject, _ckpt), group in by_subject.items():
        first = group[0]
        if len(group) == 1:
            balanced.append(first)
        else:
            merged = replace(
                first,
                prediction_rmse_deg=float(np.mean([g.prediction_rmse_deg for g in group])),
                excess_instability_auc=float(
                    np.mean([g.excess_instability_auc for g in group])),
            )
            balanced.append(merged)
    report["equal_participant_weight"] = _refit(
        balanced, "one value per participant per checkpoint", args.seed
    )

    # Excess instability per recorded second.
    rated = [
        replace(r, excess_instability_auc=r.excess_instability_auc / r.recorded_steps)
        for r in rows if r.recorded_steps > 0
    ]
    report["duration_normalized"] = _refit(
        rated, "excess instability per recorded step", args.seed
    )

    oracle = args.physics_run_dir / "oracle_preflight" / "summary.json"
    if oracle.exists():
        windows = json.loads(oracle.read_text(encoding="utf-8"))["windows"]
        excess = [
            float(w["metrics"]["excess_instability_auc"]["fused"]) for w in windows
        ]
        report["controller_gains"] = {
            "note": (
                "The zero-error case. The commanded trajectory is the matched "
                "clip's own recording, so any excess instability is caused by "
                "the override and its gains rather than by prediction error."
            ),
            "n_windows": len(excess),
            "mean_excess_instability_s": float(np.mean(excess)),
            "max_abs_excess_instability_s": float(np.max(np.abs(excess))),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rep, exc = report["reported"], report["early_termination"]["excluding_those_windows"]
    print(f"as reported            split {rep['split_rmse_deg']:.2f}  "
          f"above {rep['slope_above']:+.5f}  below {rep['slope_below']:+.5f}")
    print(f"truncated removed      split {exc['split_rmse_deg']:.2f}  "
          f"above {exc['slope_above']:+.5f}  below {exc['slope_below']:+.5f}   "
          f"({exc['n_rollouts']} rollouts)")
    if "controller_gains" in report:
        g = report["controller_gains"]
        print(f"zero-error case        mean {g['mean_excess_instability_s']:+.4f} s "
              f"over {g['n_windows']} windows")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
