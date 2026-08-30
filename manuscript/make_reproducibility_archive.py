"""Bundle the reproducibility supplement that accompanies the manuscript.

Carries the derived records a reviewer needs to check every reported number,
plus the code that produced them. The raw datasets and the 1,120 rollout
recordings are excluded: Gait120 and MoCapAct are public and fetched by scripts
in this bundle, and the recordings run to well over a hundred gigabytes.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CODE = [
    "emg_tst/fetch_gait120.py",
    "emg_tst/preprocess_gait120.py",
    "emg_tst/gait120_data.py",
    "emg_tst/gait120_experiment.py",
    "emg_tst/run_gait120_residual_fusion.py",
    "emg_tst/run_gait120_semg_controls.py",
    "emg_tst/run_gait120_kinematic_input_check.py",
    "emg_tst/gait120_training_path.py",
    "emg_tst/run_gait120_training_path.py",
    "emg_tst/run_gait120_temporal_control.py",
    "emg_tst/prepare_gait120_physics_panel.py",
    # The whole package: the runner imports config, run, utils, reference_bank
    # and bvh transitively, and a clean extraction cannot import it without them.
    "mocap_phys_eval/__init__.py",
    "mocap_phys_eval/run_gait120_residual_fusion.py",
    "mocap_phys_eval/recording.py",
    "mocap_phys_eval/matching.py",
    "mocap_phys_eval/sim.py",
    "mocap_phys_eval/experts.py",
    "mocap_phys_eval/config.py",
    "mocap_phys_eval/run.py",
    "mocap_phys_eval/utils.py",
    "mocap_phys_eval/reference_bank.py",
    "mocap_phys_eval/bvh.py",
    "mocap_phys_eval/plots.py",
    "mocap_phys_eval/level_walking.py",
    "analysis/gait120_checkpoint_correlation.py",
    "analysis/gait120_conventional_paired_statistics.py",
    "analysis/gait120_figures.py",
    "analysis/build_manuscript.py",
    "analysis/correlation.py",
    "docs/EXPERIMENT_PROTOCOL.md",
    "README.md",
    "requirements.txt",
    "requirements-physics.txt",
]


_LOCAL_PATH = re.compile(
    r"[A-Za-z]:(?:\\\\|/)+(?:[^\"\\/]+(?:\\\\|/)+)*?"
    r"(?P<root>emg_data|emg_tst)(?:\\\\|/)+",
)
_ANY_HOME = re.compile(r"[A-Za-z]:(?:\\\\|/)+Users(?:\\\\|/)+[^\"]*")


def _scrub(text: str) -> str:
    """Replace absolute local paths with logical roots.

    Records are written as JSON, so the replacements use forward slashes and no
    backslashes and cannot break the escaping.
    """
    text = _LOCAL_PATH.sub(lambda m: f"<{m.group('root')}>/", text)
    return _ANY_HOME.sub("<local path removed>", text)


RESULTS = [
    ("confirmation/ablation_summary.json", "prediction_confirmation/ablation_summary.json"),
    ("confirmation/protocol.json", "prediction_confirmation/protocol.json"),
    ("confirmation/data_audit.json", "prediction_confirmation/data_audit.json"),
    ("confirmation/fused/result.json", "prediction_confirmation/fused_result.json"),
    ("confirmation/no_emg/result.json", "prediction_confirmation/no_emg_result.json"),
    ("development/ablation_summary.json", "prediction_development/ablation_summary.json"),
    ("development/selection_audit.json", "prediction_development/selection_audit.json"),
    ("semg_controls/semg_controls.json", "semg_controls/semg_controls.json"),
    ("kinematic_input_check/kinematic_input_check.json",
     "kinematic_input_check/kinematic_input_check.json"),
    ("temporal_control/temporal_control_summary.json",
     "temporal_control/temporal_control_summary.json"),
    ("panel/physics_protocol.moving_target_pd_v2.json",
     "physics/physics_protocol.moving_target_pd_v2.json"),
    ("training_path/protocol.json", "training_path/protocol.json"),
    ("training_path/checkpoints/manifest.json", "training_path/checkpoints_manifest.json"),
    ("panel/panel_manifest.json", "physics/panel_manifest.json"),
    ("panel/panel_protocol.json", "physics/panel_protocol.json"),
    ("panel/matching_preflight/summary.json", "physics/matching_summary.json"),
    ("panel/oracle_preflight/summary.json", "physics/oracle_preflight_summary.json"),
    ("analysis/checkpoint_correlation.json", "analysis/checkpoint_correlation.json"),
    ("analysis/statistical_summary.json", "analysis/statistical_summary.json"),
    ("analysis/participant_primary.csv", "analysis/participant_primary.csv"),
]


def per_window_table(runs: Path) -> str:
    """One row per rollout, so the association analyses can be recomputed."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["checkpoint", "query_id", "subject", "panel_index",
                     "prediction_rmse_deg", "reference_auc", "fused_auc",
                     "excess_instability_auc", "match_knee_rmse_deg",
                     "match_thigh_rms_deg", "recorded_steps", "expected_steps"])
    paths = sorted((runs / "panel" / "stages").glob("*/evals/*/summary.json"))
    for path in paths:
        d = json.loads(path.read_text(encoding="utf-8"))
        sim, match = d["simulation"], d.get("match", {})
        writer.writerow([
            d["checkpoint"], d["query_id"], d.get("subject", ""),
            d.get("panel_index", -1),
            f'{d["prediction_rmse_deg"]["fused"]:.10g}',
            f'{sim["reference"]["risk_auc"]:.10g}',
            f'{sim["fused"]["risk_auc"]:.10g}',
            f'{sim["excess_instability_auc"]["fused"]:.10g}',
            f'{match.get("knee_rmse_deg", float("nan")):.10g}',
            f'{match.get("thigh_rms_deg", float("nan")):.10g}',
            sim["fused"].get("recorded_steps", 0), match.get("length", 0),
        ])
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=REPO / "manuscript" / "Additional_file_reproducibility.zip")
    args = parser.parse_args()
    runs = args.runs_dir.resolve()

    missing = []
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in CODE:
            source = REPO / rel
            if source.exists():
                z.write(source, f"code/{rel}")
            else:
                missing.append(rel)
        for source_rel, target_rel in RESULTS:
            source = runs / source_rel
            if source.exists():
                # Records carry absolute paths from the machine that produced
                # them, which name the account and mean nothing to a reviewer.
                z.writestr(
                    f"results/{target_rel}",
                    _scrub(source.read_text(encoding="utf-8")),
                )
            else:
                missing.append(source_rel)
        z.writestr("results/analysis/per_window_rollouts.csv", per_window_table(runs))
        z.writestr("README.txt", README)

    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, "
          f"{len(zipfile.ZipFile(args.out).namelist())} files)")
    if missing:
        # An archive that omits what its own README tells a reviewer to run is
        # not a reproducibility supplement, so this is an error and not a note.
        for m in missing:
            print("  MISSING:", m)
        raise SystemExit(
            f"{len(missing)} listed input(s) were not found; the archive is incomplete"
        )


README = """Reproducibility supplement
==========================

results/
    Every derived record the manuscript quotes. per_window_rollouts.csv holds one
    row for each simulated rollout in the analysed panel, and is sufficient to
    recompute the association analyses without rerunning any physics.

code/
    The analysis pipeline. Gait120 is fetched by code/emg_tst/fetch_gait120.py;
    the MoCapAct expert zoo is fetched on demand by code/mocap_phys_eval.
    docs/EXPERIMENT_PROTOCOL.md states the cohort split, signal windows, model
    comparison, simulation mapping, and outcome definitions.

Not included: the Gait120 recordings and the MoCapAct expert policies, both
public and downloaded by the scripts above, and the rollout recordings, which
exceed 100 GB. Rollouts are deterministic (fixed seed, deterministic
policy), so rerunning reproduces them exactly.

The physics runs that produced results/physics were made with the
moving-target PD controller, recorded in
results/physics/physics_protocol.moving_target_pd_v2.json. The runner requires
the mode to be named:

    python -m mocap_phys_eval.run_gait120_residual_fusion \\
        --moving-target-pd-v2 ...

Regenerating the reported numbers and figures from results/.
Run these from inside code/, which is where the packages live:

    cd code
    python -m analysis.gait120_checkpoint_correlation --physics-run-dir <panel> \\
        --out-dir <out>
    python -m analysis.gait120_figures --runs-dir <runs> --out-dir <figures>
    python -m analysis.build_manuscript --runs-dir <runs> --out-dir <manuscript>
"""


if __name__ == "__main__":
    main()
