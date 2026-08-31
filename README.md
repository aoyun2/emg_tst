# Gait120 residual-fusion simulation study

This repository contains the analysis code for a paired study of 100-ms
knee-angle prediction and whole-body simulation. The predictor was developed
with 30 Gait120 participants and evaluated on later trials from a separate
90-participant cohort after calibration on each cohort's earlier trials.

The sEMG residual correction reduced mean participant RMSE from 5.093° to
4.748° (paired difference 0.345°, 95% confidence interval 0.254–0.436°). Three
sEMG surrogate controls test whether that gain reflects signal content rather
than the extra fitted stage.

The simulation is driven by the model: the predicted knee angle is the tracking
target of a PD override on the right knee actuator, and the paired reference
runs the unmodified expert on the same matched motion. Nothing in the rollout
reads the recorded future knee angle.

Because the model converges to a narrow error band, one association measured at
the converged fit cannot distinguish "prediction error does not drive simulated
instability" from "there was not enough error spread to tell". The physics panel
is therefore replayed at checkpoints sampled along the model's training path,
spanning the range from an untrained model down to the converged fit.

The experiment used healthy-participant data; it did not test a prosthetic
device or estimate clinical fall risk.

## Repository map

| Path | Purpose |
|---|---|
| `emg_tst/fetch_gait120.py` | Gait120 download from figshare, with checksums |
| `emg_tst/gait120_data.py` | Gait120 MAT-file loading and causal sEMG processing |
| `emg_tst/gait120_experiment.py` | Shared example construction, ridge fitting, and participant metrics |
| `emg_tst/run_gait120_residual_fusion.py` | Development and confirmation sEMG ablation |
| `emg_tst/run_gait120_temporal_control.py` | 500-ms timing control |
| `emg_tst/run_gait120_semg_controls.py` | sEMG surrogate negative controls |
| `emg_tst/gait120_training_path.py` | Gradient-descent path and checkpoint selection |
| `emg_tst/run_gait120_training_path.py` | Per-checkpoint models and test predictions |
| `emg_tst/prepare_gait120_physics_panel.py` | Fixed participant-balanced simulation panel |
| `mocap_phys_eval/run_gait120_residual_fusion.py` | Motion matching and model-driven paired MuJoCo rollouts |
| `analysis/gait120_checkpoint_correlation.py` | Association along the training path and its drop-off |
| `analysis/gait120_conventional_paired_statistics.py` | Participant-level inferential summaries |
| `docs/EXPERIMENT_PROTOCOL.md` | Human-readable analysis protocol |
| `tests/` | Deterministic unit and repository-integrity checks |

Generated runs, raw data, model banks, recordings, evidence archives, the
manuscript and the scripts that typeset and check it are excluded from Git. The
journal supplement contains the derived records needed to verify the reported
results.

## Data

The final analysis uses level walking from:

- [Gait120 dataset](https://doi.org/10.6084/m9.figshare.27677016)
- [Gait120 data descriptor](https://doi.org/10.1038/s41597-025-05391-0)
- [MoCapAct](https://microsoft.github.io/MoCapAct/)

Predictor inputs and targets are not interpolated or time-normalized. Three
rate conversions occur elsewhere: the 2000-Hz sEMG envelope contributes its
final sample in each 20-sample frame block, 100-Hz Gait120 queries are matched
against the 200-Hz MoCapAct reference grid, and 100-Hz commands are resampled
onto the 0.03-s simulation grid.

## Environments

Use separate environments for the three parts of the workflow:

```bash
# Prediction and participant-level statistics
python -m pip install -r requirements.txt

# Figure generation (OpenCV currently requires NumPy below 2.3)
python -m pip install -r requirements-figures.txt

# Legacy MoCapAct/MuJoCo simulation stack
python -m pip install -r requirements-physics.txt
```

The physics stack is version-sensitive and is best run under Python 3.10. The
completed-run supplement records the exact software versions used for the
reported simulations.

## Reproducing the analysis

Inspect each public entry point before supplying local data paths:

```bash
python -m emg_tst.fetch_gait120 --list
python -m emg_tst.preprocess_gait120 --help
python -m emg_tst.run_gait120_residual_fusion --help
python -m emg_tst.run_gait120_semg_controls --help
python -m emg_tst.run_gait120_temporal_control --help
python -m emg_tst.run_gait120_training_path --help
python -m emg_tst.prepare_gait120_physics_panel --help
python -m mocap_phys_eval.run_gait120_residual_fusion --help
```

The stages depend on each other in that order: the confirmation run fixes the
hyperparameters the training path reuses, the training path writes the
checkpoints the panel is built against, and the panel is what the simulation
replays. `emg_tst.fetch_gait120` downloads about 16.5 GB; the MoCapAct reference
bank and experts are fetched on demand by the simulation runner.

Participant-level statistics are regenerated from an extracted evidence
supplement. Both commands read the supplement directly:

```bash
python -m analysis.gait120_conventional_paired_statistics --evidence <extracted>/results --output summary.json
python -m analysis.gait120_checkpoint_correlation --per-window-csv <extracted>/results/analysis/per_window_rollouts.csv --out-dir out
```

The figures are not regenerable from the supplement: they need per-checkpoint
prediction arrays and rollout recordings that the archive does not carry.

See [the protocol](docs/EXPERIMENT_PROTOCOL.md) for the cohort split, signal
windows, model comparison, simulation mapping, and outcome definitions.

## Tests

The deterministic tests do not download either dataset:

```bash
python -m unittest discover -s tests -v
```

Older Georgia Tech acquisition utilities remain because some shared data and
coordinate helpers still depend on them. They are not the predictor evaluated
in the Gait120 study.

## License

MIT; see `LICENSE`.
