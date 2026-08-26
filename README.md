# Physics-based evaluation of sEMG knee-angle regression

This repository contains the code and publication artifacts for **“Towards the
Use of Simulated Environments to Evaluate sEMG-Based Prosthetic Knee-Angle
Regression.”** The study asks whether an sEMG-related reduction in 100-ms
knee-angle prediction error also reduces instability when the paired prediction
errors are inserted into matched whole-body walking simulations.

## Final result

The participant-calibrated residual-fusion predictor was developed with 30
Gait120 participants and evaluated once in 90 separate participants.

| Result | Value |
|---|---:|
| RMSE with residual fusion | 4.748° |
| RMSE without sEMG | 5.093° |
| Mean paired improvement | 0.345° |
| 95% confidence interval | 0.252–0.438° |
| Paired test | t(89) = 7.39, p = 7.50 × 10⁻¹¹ |
| Participants improved | 70 / 90 |

All 80 fixed level-walking windows were matched to stable MoCapAct references
and evaluated at seven prediction-accuracy checkpoints (560 numerical
simulations). The sEMG prediction improvement did not produce a detectable
change in excess-instability AUC at the full-data checkpoint. A sharp physical
deterioration was observed between the measured 13.41° and 16.29° RMSE
checkpoints.

## Repository layout

| Path | Purpose |
|---|---|
| `emg_tst/gait120_data.py` | Native Gait120 MAT-file loading and causal sEMG processing |
| `emg_tst/gait120_experiment.py` | Shared example construction, participant metrics, and fail-closed output utilities |
| `emg_tst/preprocess_gait120.py` | Reproducible participant cache export |
| `emg_tst/run_gait120_residual_fusion.py` | Development and untouched confirmation ablation |
| `emg_tst/run_gait120_temporal_control.py` | 500-ms timing control |
| `emg_tst/run_gait120_accuracy_path.py` | Seven fixed training-data checkpoints |
| `emg_tst/prepare_gait120_physics_panel.py` | Fixed participant-balanced physics panel |
| `mocap_phys_eval/run_gait120_level_walking_v4.py` | Definitive level-walking motion matching and MuJoCo evaluation |
| `analysis/gait120_v4.py` | Frozen numerical physics audit and FWL analysis |
| `analysis/gait120_conventional_paired_statistics.py` | Paired tests reported in the manuscript |
| `analysis/gait120_submission_figures_original_style.py` | Publication figure generator |
| `docs/EXPERIMENT_PROTOCOL.md` | Concise definitive protocol |
| `final_submission/` | Verified PDF, editable Overleaf project, media, evidence, and submission bundle |

The older Georgia Tech acquisition and MoCapAct compatibility modules remain
where the final implementation still uses them. They are not the predictor
evaluated in the final paper.

## Data

Raw datasets and the MoCapAct expert bank are intentionally not stored in Git.

- [Gait120 dataset](https://doi.org/10.6084/m9.figshare.27677016)
- [Gait120 article](https://doi.org/10.1038/s41597-025-05391-0)
- [MoCapAct](https://microsoft.github.io/MoCapAct/)

The final analysis uses only Gait120 level walking. Predictor inputs and targets
are not interpolated or time-normalized. The matching search performs only the
documented sampling-rate conversion needed to compare the 100-Hz Gait120 query
with the 200-Hz MoCapAct reference grid.

## Environment

Create an isolated Python environment and install the project dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

MoCapAct and its pinned MuJoCo 2.2.2 backend may require a compatible Python
environment separate from the prediction and plotting environment. Exact
software and protocol records from the completed run are included in
`final_submission/Additional_file_2_reproducibility_evidence.zip`.

## Reproducing the prediction analysis

After downloading Gait120, inspect the cache-export options:

```bash
python -m emg_tst.preprocess_gait120 --help
```

Run development first, then the untouched confirmation:

```bash
python -m emg_tst.run_gait120_residual_fusion --help
```

The timing control and accuracy path are separate entry points:

```bash
python -m emg_tst.run_gait120_temporal_control --help
python -m emg_tst.run_gait120_accuracy_path --help
```

Each command writes a protocol before writing outcomes and stops on missing or
incompatible inputs.

## Reproducing the reported paired statistics

Extract `Additional_file_2_reproducibility_evidence.zip`, then run:

```bash
python -m analysis.gait120_conventional_paired_statistics \
  --evidence-dir path/to/extracted/evidence \
  --output conventional_paired_statistics.json
```

The primary ablation uses a two-sided paired t-test on the 90 participant-level
RMSE differences. Wilcoxon signed-rank and exact sign tests are emitted as
sensitivity analyses.

The complete physics audit uses explicit input directories rather than local
machine defaults:

```bash
python -m analysis.gait120_v4 \
  --run-dir path/to/completed/physics/run \
  --confirmation-dir path/to/completed/confirmation/run \
  --output-dir path/to/analysis/output
```

The figure generator likewise requires the extracted evidence, the eight
representative recordings, and the two source-data images identified in its
command-line help. No user-specific paths are embedded in the source.

## Tests

The repository’s lightweight deterministic tests do not download either
dataset:

```bash
python -m unittest discover -s tests -v
```

## Publication files

The final deliverables are under `final_submission/`:

- `main.pdf` — visually checked 34-page manuscript
- `Prosthetic_Regression_BMC_Overleaf.zip` — clean editable LaTeX project
- `Additional_file_1_simulation_recordings.zip` — eight prospectively selected
  representative MuJoCo recordings
- `Additional_file_2_reproducibility_evidence.zip` — participant results,
  complete 560-simulation summary, protocols, audits, and source snapshot
- `Prosthetic_Regression_BMC_Submission_Bundle.zip` — complete journal package
- `SHA256SUMS.txt` — integrity hashes

The manuscript currently uses “Independent researcher, United States.” Add the
preferred correspondence email and confirm the affiliation before submission.
