"""Check the reproducibility records against the run and the manuscript.

The supplement carries derived records and makes no executable claim, so this
checks the things it does promise: that every record it advertises is present,
that the records agree with each other, and that the numbers in them are the
numbers the paper reports.

    python -m analysis.verify_archive
"""

from __future__ import annotations

import os
import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "manuscript" / "Additional_file_reproducibility.zip"

# every path the README names, and what it is for
ADVERTISED = [
    "results/analysis/per_window_rollouts.csv",
    "results/analysis/participant_primary.csv",
    "results/analysis/checkpoint_correlation.json",
    "results/analysis/statistical_summary.json",
    "results/prediction_confirmation/ablation_summary.json",
    "results/prediction_development/ablation_summary.json",
    "results/semg_controls/semg_controls.json",
    "results/temporal_control/temporal_control_summary.json",
    "results/kinematic_input_check/kinematic_input_check.json",
    "results/physics/panel_manifest.json",
    "results/physics/matching_summary.json",
    "results/physics/oracle_preflight_summary.json",
    "results/physics/physics_protocol.moving_target_pd_v2.json",
    "results/training_path/protocol.json",
    "results/training_path/checkpoints_manifest.json",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--runs-dir", type=Path,
                        default=Path(os.environ.get("EMG_TST_RUNS", "runs")).expanduser())
    args = parser.parse_args()

    failures: list[str] = []
    passed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        if ok:
            passed += 1
        else:
            failures.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))

    work = Path(tempfile.mkdtemp(prefix="verify_records_"))
    with zipfile.ZipFile(args.archive) as bundle:
        names = set(bundle.namelist())
        bundle.extractall(work)
    print(f"extracted {len(names)} files -> {work}\n")

    try:
        for rel in ADVERTISED:
            check(f"present: {rel}", rel in names)

        # the supplement must not ship code, since it promises none
        check("ships no code tree", not any(n.startswith("code/") for n in names))

        readme = (work / "README.txt").read_text(encoding="utf-8")
        check("README makes no executable claim",
              "python -m" not in readme and "cd code" not in readme)
        check("README points at the repository", "github.com/aoyun2" in readme)

        # the raw timing record and the derived summary must agree
        raw = json.loads(
            (work / "results/temporal_control/temporal_control_summary.json")
            .read_text(encoding="utf-8"))
        n_participants = len(raw["aligned_vs_lagged"]["improvement_deg"])
        summary = json.loads(
            (work / "results/analysis/statistical_summary.json").read_text(encoding="utf-8"))
        df = int(summary["temporal_control"]["aligned_vs_lagged"]["paired_t"]["df"])
        check("timing record agrees with its summary", n_participants == df + 1,
              f"{n_participants} participants, df={df}")

        # the records must be the ones the manuscript was built from
        live = json.loads(
            (args.runs_dir / "analysis" / "checkpoint_correlation.json")
            .read_text(encoding="utf-8"))["accuracy_level"]
        shipped = json.loads(
            (work / "results/analysis/checkpoint_correlation.json")
            .read_text(encoding="utf-8"))["accuracy_level"]
        check("split point matches the run",
              shipped["breakpoint_rmse_deg"] == live["breakpoint_rmse_deg"],
              f"{shipped['breakpoint_rmse_deg']:.4f}")
        check("split interval matches the run",
              shipped["breakpoint_95pct_ci"] == live["breakpoint_95pct_ci"],
              str([round(v, 3) for v in shipped["breakpoint_95pct_ci"]]))

        # no machine-local paths survive
        leaked = [
            n for n in names
            if n.endswith((".json", ".csv", ".txt"))
            and "aaron" in (work / n).read_text(encoding="utf-8", errors="replace").lower()
        ]
        check("no machine-local paths", not leaked, ", ".join(leaked[:3]))

        print(f"\n{passed}/{passed + len(failures)} passed")
        return 1 if failures else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
