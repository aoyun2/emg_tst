# emg_tst

Real-time knee-angle prediction from thigh EMG plus thigh kinematics, evaluated inside MoCapAct physics.

Important note for writing:
- the current Word draft in [Research Paper — Aaron (3).docx](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/Research%20Paper%20%E2%80%94%20Aaron%20%283%29.docx) still describes the older custom-data + transformer pipeline
- the current repo state is different: Georgia Tech processed data + CNN-BiLSTM + GT-compatible motion matching
- use the handoff section near the end of this README, not the older transformer/custom-data description, when asking another model to draft the paper

The current main pipeline is:
- preferred data source: Georgia Tech processed biomechanics trials converted into repo-native `gt_data*.npy`
- model input: paper-style GT EMG preprocessing + right-thigh IMU only
- main model: CNN-BiLSTM last-step regressor
- alternative models available in code: residual-fusion TCN and sensor-fusion LSTM
- training: direct from converted `gt_data*.npy` recordings with lazy stride-1 windows
- simulation: rolling causal inference over held-out query windows from `samples_dataset.npy`

Paper interpretation note:
- the simulation does not produce a calibrated fall probability
- it produces a heuristic per-step instability trace from uprightness, support/contact, and XCoM margin
- the paper-default continuous outcome is therefore the excess instability AUC, `PRED - REF`, not the absolute `PRED` AUC alone

For the GT path, the repo now derives a marker-based right-thigh segment quaternion from the raw GT marker data, not just a hip-angle proxy. The publication-default motion matcher is scalar `thigh_knee_d` with `knee_weight=1.0`, `thigh_weight=0.0`, and `local_refine_radius=30`. On the current native-rate 80-window run [20260405_230549](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549), that matcher achieved mean knee match RMSE `7.93°` and median knee match RMSE `5.86°` in [summary_metrics_native.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549/summary_metrics_native.json). The marker-derived 3D thigh quaternion is still stored for diagnostics and alternative match modes.

Integration status:
- the current trainer default in [run_experiment.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/run_experiment.py) is `MODEL_ARCH = "cnn_bilstm"`
- the simulator in [run.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/mocap_phys_eval/run.py) is checkpoint-driven and can load this architecture
- the current native-rate GT subject-holdout training run is [tst_20260405_173725_all](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/checkpoints/tst_20260405_173725_all)
- that run achieved mean held-out `test_rmse = 7.84°` in [metrics_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/checkpoints/tst_20260405_173725_all/metrics_summary.json)
- the current native-rate simulation/statistics run is [20260405_230549](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549)
- the evaluator can rerun that benchmark directly via `EMG_TST_RUN_DIR=checkpoints/tst_20260405_173725_all` and `MOCAP_PHYS_EVAL_ALLOW_PARTIAL=1`

## Publication Checklist

To reproduce the current paper-ready native-rate numbers, use this order:

1. Regenerate the GT recordings if needed:

```bash
python -c "from emg_tst.gt_dataset import ensure_normal_walk_recordings; ensure_normal_walk_recordings()"
```

2. Train the model on the current native-rate path:

```bash
python -m emg_tst.run_experiment
```

3. Rebuild the held-out query pool:

```bash
python split_to_samples.py
```

4. Run the physical simulation benchmark:

```powershell
$env:EMG_TST_RUN_DIR = "checkpoints\\tst_20260405_173725_all"
$env:MOCAP_PHYS_EVAL_ALLOW_PARTIAL = "1"
$env:MOCAP_PHYS_EVAL_N_TRIALS = "80"
python -m mocap_phys_eval.run
```

5. Run the partial-correlation analysis on the new run:

```bash
python -m analysis.correlation --run-dir artifacts/phys_eval_v2/runs/<run_id>
```

The current publication run IDs referenced below were produced with this exact sequence.

---

## Methodology

### Inputs

Per native 200 Hz GT angle/IMU timestep, the GT-integrated model receives:
- `1` causal EMG envelope value from `RRF`
- `1` causal EMG envelope value from `RBF`
- `1` causal EMG envelope value from `RVL`
- `1` causal EMG envelope value from `RMGAS`
- right-thigh IMU: `RAThigh_ACCX`, `RAThigh_ACCY`, `RAThigh_ACCZ`
- right-thigh IMU: `RAThigh_GYROX`, `RAThigh_GYROY`, `RAThigh_GYROZ`

Total input width per timestep: `10`
- `4` EMG values = `4 x high-pass -> rectify -> low-pass` preprocessed channels
- `6` IMU values = `3 x accel + 3 x gyro`

The current GT path follows the paper's EMG filtering, but now keeps the actual GT angle/IMU timebase instead of an extra downsample:
- start from GT processed `raw_emg_channels`
- high-pass at `20 Hz`
- rectify with `abs(x)`
- low-pass at `5 Hz`
- keep GT processed angle + IMU at their native `200 Hz`
- align the filtered EMG envelope to that same native `200 Hz` timebase

Actual GT rates used in the repo now:
- raw EMG: `2000 Hz`
- processed angle + IMU: `200 Hz`
- model/query windows: native `200 Hz`

### Target

The supervised target is the knee included angle:
- `0 deg = fully bent`
- `180 deg = fully extended`

For the GT path, this is derived from `knee_angle_r` in the processed OpenSim angle file:
- `knee_flex_deg = clip(-knee_angle_r, 0, 180)`
- `knee_included_deg = 180 - knee_flex_deg`

### Windowing

The source window is:
- `400` samples
- `2.0 s` at 200 Hz

Training windows are generated lazily from raw recordings with `stride = 1`.
The full stride-1 pool is not consumed every epoch. Each epoch samples up to `8,192` training windows uniformly from the current fold's training pool.

### Preprocessing

For each recording:
- z-score the `4` EMG envelope channels independently
- leave the `6` thigh IMU channels unchanged at this stage

Then:
- fit a global mean/std scaler across the training recordings
- normalize the knee label by dividing by `180`

---

## Exact Architecture

The integrated main model is `CnnBiLstmLastStep` in [emg_tst/model.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/model.py).

Default architecture in [emg_tst/run_experiment.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/run_experiment.py):
- `MODEL_ARCH = "cnn_bilstm"`
- `SOURCE_WINDOW = 400`
- `CONTEXT_WINDOW = 400`
- `STEM_WIDTH = 32`
- `TCN_KERNEL_SIZE = 5`
- `CNN_DEPTH = 2`
- `LSTM_HIDDEN_SIZE = 64`
- `LSTM_LAYERS = 2`
- `DROPOUT = 0.10`

For an input tensor of shape `(B, 400, 10)`:

1. Convolutional frontend
   - input: all `10` channels together
   - `Conv1d(10 -> 32, kernel_size=5, padding=2)`
   - `GELU`
   - `Conv1d(32 -> 32, kernel_size=5, padding=2)`
   - `GELU`
   - `Dropout(0.10)`

2. Temporal recurrent encoder
   - transpose back to `(B, 400, 32)`
   - `BiLSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)`
   - use only the final timestep output, width `128`

3. Regression head
   - `Linear(128 -> 64)`
   - `GELU`
   - `Dropout(0.10)`
   - `Linear(64 -> 1)`

This is a sequence-to-one causal model. The trained output is a short-horizon included-angle forecast because `LABEL_SHIFT = 2` at the GT `200 Hz` timebase, i.e. a `10 ms` lookahead. For simulation and plotting, the repo rolls this last-step model across the full query sequence one timestep at a time and then causally aligns the forecast before applying it as the prosthetic target.

### Architecture Diagram

```text
Input window: (B, 400, 10)
  |
  +-- Conv1d(10 -> 32, k=5, pad=2)
  +-- GELU
  +-- Conv1d(32 -> 32, k=5, pad=2)
  +-- GELU
  +-- Dropout(0.10)
  |
  +-- transpose to (B, 400, 32)
  +-- BiLSTM(32 -> 64 per direction, layers=2)
  +-- take last timestep: (B, 128)
  +-- Linear(128 -> 64)
  +-- GELU
  +-- Dropout(0.10)
  +-- Linear(64 -> 1)
  |
  +--> knee included angle at t + 2 samples
```

Interpretation:
- the CNN frontend learns short-range local patterns in the fused EMG + thigh channels
- the BiLSTM models longer context across the full 2.0 s history
- the final MLP regresses the one-step-ahead knee included angle from the last contextual state

### Other Available Models

The residual-fusion TCN and sensor-fusion LSTM implementations are still available in [emg_tst/model.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/model.py), but the current default path is the CNN-BiLSTM above.

---

## Training

The main trainer is [emg_tst/run_experiment.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/run_experiment.py).

Key training settings:
- optimizer: `Adam`
- learning rate: `1e-3`
- weight decay: `1e-4`
- batch size: `128`
- train samples per epoch: `8,192`
- max epochs: `6`
- early stopping patience: `2`
- loss: `Huber` with `delta = 5 deg`
- gradient clipping: `1.0`
- train noise: disabled by default
- multimodal augmentation: disabled by default
- label target: normalized by `180`, with `LABEL_SHIFT = 2` samples (`10 ms` at the GT 200 Hz timebase)

Cross-validation:
- LOFO = Leave-One-File-Out by recording file
- one additional training file is held out for validation when possible
- best checkpoint is written immediately whenever validation RMSE improves

Reported fold metrics:
- `test_rmse`: last-step RMSE on held-out windows
- `test_seq_rmse`: rolling causal RMSE across the held-out recording

The trainer still runs three ablations automatically:
1. `ALL`
2. `THIGH-ONLY`
3. `EMG-ONLY`

The architecture is the same in all three cases. Only the active input branches change.

### Run Training

```bash
python -m emg_tst.run_experiment
```

On this AMD/DirectML setup, the CNN-BiLSTM path automatically falls back to CPU because DirectML does not support the LSTM cell used by this model cleanly.

Outputs:
- `checkpoints/<run_name>_all/fold_XX/reg_best.pt`
- `checkpoints/<run_name>_all/fold_XX/split_manifest.json`
- `checkpoints/<run_name>_all/cv_manifest.json`
- `checkpoints/<run_name>/ablation_summary.json`

---

## Data

The preferred integrated dataset path is the Georgia Tech processed dataset converted into repo-native files:
- `gt_data000.npy`
- `gt_data001.npy`
- ...

Generate the Georgia Tech normal-walk corpus with:

```bash
python -c "from emg_tst.gt_dataset import ensure_normal_walk_recordings; ensure_normal_walk_recordings()"
```

Each converted GT recording contains:
- `raw_emg_channels` from `RRF`, `RBF`, `RVL`, `RMGAS`
- `thigh_imu` from `RAThigh_{ACC,GYRO}{X,Y,Z}`
- `thigh_pitch_deg` proxy from `hip_flexion_r`
- `thigh_quat_wxyz` from raw GT right-thigh markers (`RGTR`, `RKNE`, `RMKNE`, `RTHL`, `RTHR`) after interpolation onto the processed trial timebase
- `knee_included_deg` derived from `knee_angle_r`

Legacy custom `data*.npy` recordings are still supported by the loader, but the current preferred path is GT.

### Build `samples_dataset.npy`

```bash
python split_to_samples.py
```

This file is still used for:
- physical evaluation query pools
- held-out window bookkeeping
- some visualizations

For the GT path, `samples_dataset.npy` now stores both:
- `thigh_pitch_seq`
- `thigh_quat_seq`

The simulation path uses `thigh_pitch_seq` by default for publication matching and keeps `thigh_quat_seq` available for diagnostics and alternative 3D matchers.

It is not the main training source anymore.

### Data Quality Checklist

For the current GT path:
- confirm `gt_data*.npy` was regenerated before training if the converter changed
- rebuild `samples_dataset.npy` after any data-preprocessing changes
- inspect representative GT trials with `plot_data.py`
- sanity-check a few motion matches before trusting a large MuJoCo batch

The older custom-recording rerecord notes are no longer the main path for this repo.

---

## Physical Evaluation

Run the MuJoCo evaluator on the latest compatible full-coverage `_all` checkpoints:

```bash
python -m mocap_phys_eval
```

Run the current GT subject-holdout benchmark checkpoint through the evaluator:

```powershell
$env:EMG_TST_RUN_DIR = "checkpoints\\tst_20260405_173725_all"
$env:MOCAP_PHYS_EVAL_ALLOW_PARTIAL = "1"
$env:MOCAP_PHYS_EVAL_N_TRIALS = "80"
python -m mocap_phys_eval.run
```

Replay the latest run:

```bash
python -m mocap_phys_eval.replay
```

The evaluator:
1. loads held-out query windows from `samples_dataset.npy`
2. loads the latest compatible `*_all` checkpoints
3. maps each held-out window to the correct LOFO fold via `split_manifest.json` / `cv_manifest.json`
4. rolls the last-step model causally across the full query window
5. converts predicted knee included angle to knee flexion
6. motion-matches into the MoCapAct expert bank
7. runs:
   - `REF`: matched expert policy with no override
   - `PRED`: same policy with right-knee override from the model
   - `BAD`: optional diagnostic stress trace

Paper-mode defaults:
- held-out LOFO windows only
- seed `42`
- replacement until `80` successful trials
- matcher: `thigh_knee_d`
- match weights: `knee=1.0`, `thigh=0.0`
- local refine radius: `30`

Current native-rate GT benchmark result:
- training: [metrics_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/checkpoints/tst_20260405_173725_all/metrics_summary.json)
- simulation/statistics: [summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549/summary.json)
- simulation aggregates: [summary_metrics_native.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549/summary_metrics_native.json)
- correlation output: [partial_spearman_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549/analysis/partial_spearman_summary.json)
- plot stats: [paper_plot_stats_excess.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549/analysis/paper_plot_stats_excess.json)

Current interpretation:
- mean held-out model test RMSE on the 55-fold GT subject-holdout run is `7.84°`
- median held-out model test RMSE is `6.85°`
- mean predictor RMSE across the 80 evaluated simulation windows is `8.80°`
- mean motion-match knee RMSE across those windows is `7.93°`
- using the corrected paper outcome `excess instability AUC = pred_auc - ref_auc`, raw predictor RMSE is weakly negative and non-significant (`rho = -0.166`, `p = 0.140`)
- after controlling for motion-match knee and thigh errors, the partial Spearman estimate is `rho = -0.022`, `p = 0.851`

Correlation figures:
- raw RMSE vs excess instability: [raw_rmse_vs_excess_instability.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/raw_rmse_vs_excess_instability.png)
- partial / residualized RMSE vs excess instability: [partial_rmse_vs_excess_instability.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/partial_rmse_vs_excess_instability.png)
- motion-match controls vs excess instability: [match_controls_vs_excess_instability.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/match_controls_vs_excess_instability.png)

Paper-native figure set:
- figure manifest: [figure_manifest.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/figure_manifest.json)
- figure captions: [captions.md](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/captions.md)
- Figure 1, pipeline overview: [fig1_pipeline_overview.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig1_pipeline_overview.png)
- Figure 2, fold-level prediction distribution: [fig2_prediction_distribution.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig2_prediction_distribution.png)
- Figure 3, paired simulation outcomes: [fig3_simulation_outcomes.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig3_simulation_outcomes.png)
- Figure 4, correlation and confounding: [fig4_correlation_confounding.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig4_correlation_confounding.png)
- Figure 5, representative rollout: [fig5_representative_rollout.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/paper_native/fig5_representative_rollout.png)

The main paper statistic uses:
- predictor: `model.pred_vs_gt_knee_flex_rmse_deg`
- outcome: `sim.excess.instability_auc_delta`

Instability-metric note:
- the stored scalar `predicted_fall_risk` is best interpreted as `instability_score`
- it is derived from a heuristic trace and can overcall some visually stable `REF` windows
- the paper should therefore avoid treating it as a literal fall probability
- use `excess instability AUC = pred_auc - ref_auc` as the primary continuous outcome

Current practical note:
- `mocap_phys_eval` will only use architectures that exist in saved compatible `_all` checkpoints
- so after changing the model or feature layout, always run:

```bash
python -m emg_tst.run_experiment
python split_to_samples.py
```

- before expecting the simulator to use the new model

To regenerate the full paper figure set from the finalized native-rate run:

```bash
python analysis/make_paper_figures.py
```

### Disk Setup

The MoCapAct expert zoo is large. Set:

```powershell
$env:MOCAPACT_MODELS_DIR = "D:\\mocapact_models"
```

Optional:

```powershell
$env:MOCAP_PHYS_EVAL_ARTIFACTS_DIR = "D:\\phys_eval_v2_artifacts"
```

One-time prefetch:

```bash
python -m mocap_phys_eval.prefetch
```

---

## Visualization

Sample-level prediction visualization:

```bash
python -m emg_tst.visualize
```

Dataset plots:

```bash
python plot_data.py
```

The visualizer uses the same rolling last-step inference path as the simulator.

---

## Learning Curve

Optional learning-curve sweep on the current training pipeline:

```bash
python -m emg_tst.learning_curve
```

This also trains directly from raw `data*.npy` recordings, not from `samples_dataset.npy`.

---

## Repo Structure

- [emg_tst/data.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/data.py): raw EMG snippet extraction, timestamp alignment, resampling, recording loader
- [emg_tst/model.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/model.py): TCN/LSTM model definitions and rolling last-step inference helper
- [emg_tst/run_experiment.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/run_experiment.py): main LOFO trainer and ablations
- [emg_tst/visualize.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/visualize.py): checkpoint visualization
- [split_to_samples.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/split_to_samples.py): evaluation/query-pool sample builder
- [mocap_phys_eval/run.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/mocap_phys_eval/run.py): motion matching and MuJoCo evaluation
- [analysis/correlation.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/analysis/correlation.py): partial Spearman analysis

---

## Install

```bash
pip install -r requirements_tst.txt
```

For AMD GPU support on Windows:

```bash
pip install torch-directml
```

---

## Paper Handoff

This section is the current repo-faithful summary to give another writing model.

Important:
- the code path described below is the current native-rate `200 Hz` implementation
- the exact benchmark numbers quoted below are from the current native-rate benchmark
- the canonical training run is `checkpoints/tst_20260405_173725_all`
- the canonical simulation/statistics run is `artifacts/phys_eval_v2/runs/20260405_230549`

### Current Study Framing

- Research question: whether lower knee-prediction error from a wearable-sensor regression model corresponds to lower excess simulated instability when the predicted knee trajectory is injected into a MoCapAct / MuJoCo prosthetic-override pipeline.
- Main dataset in the current repo: Georgia Tech processed biomechanics dataset converted into repo-native `gt_data*.npy` recordings.
- Main model in the current repo: CNN-BiLSTM last-step regressor, not the older custom-data transformer.
- Simulation pipeline: GT held-out windows are motion-matched into a MoCapAct reference bank, then MuJoCo runs `REF` and `PRED` rollouts while overriding only the right knee.
- Publication-default matcher:
  - `thigh_knee_d`
  - `knee_weight=1.0`
  - `thigh_weight=0.0`
  - `local_refine_radius=30`

### Data and Preprocessing

- Source signals:
  - `4` EMG channels: `RRF`, `RBF`, `RVL`, `RMGAS`
  - `6` right-thigh IMU channels: `RAThigh_ACC{X,Y,Z}`, `RAThigh_GYRO{X,Y,Z}`
  - label source: GT `knee_angle_r`
  - marker-derived 3D thigh orientation available as `thigh_quat_wxyz`
  - scalar diagnostic thigh proxy: `hip_flexion_r`
- GT preprocessing in the current repo:
  - high-pass EMG at `20 Hz`
  - rectify with absolute value
  - low-pass at `5 Hz`
  - keep GT angle + IMU at native `200 Hz`
  - resample the filtered EMG envelope onto the native `200 Hz` angle/IMU timebase
  - use `2.0 s` windows of `400` samples
  - z-score the `4` EMG channels per recording
  - fit a global scaler on training recordings
- Label convention:
  - convert GT knee flexion to included angle
  - normalize the supervised label by dividing by `180`
  - train with `LABEL_SHIFT = 2`, so the model predicts `10 ms` ahead at the native `200 Hz` timebase

### Model

- Architecture:
  - `Conv1d(10 -> 32, k=5)`
  - `Conv1d(32 -> 32, k=5)`
  - `BiLSTM(input=32, hidden=64, layers=2, bidirectional=True)`
  - `Linear(128 -> 64) -> GELU -> Dropout(0.10) -> Linear(64 -> 1)`
- Training:
  - optimizer `Adam`
  - learning rate `1e-3`
  - weight decay `1e-4`
  - batch size `128`
  - Huber loss with `delta = 5°`
  - max `6` epochs with patience `2`
  - stride-1 lazy windows with `8192` sampled training windows per epoch

### Evaluation Design

- Training benchmark:
  - subject-holdout GT run stored in [metrics_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/checkpoints/tst_20260405_173725_all/metrics_summary.json)
  - current native-rate `ALL` result: mean `test_rmse = 7.84°`, mean `test_seq_rmse = 7.88°`
- Simulation benchmark:
  - run stored in [summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549/summary.json)
  - `80` successful GT held-out windows
  - fixed seed `42`
  - model checkpoint source: `checkpoints/tst_20260405_173725_all/fold_*/reg_best.pt`
  - aggregate simulation metrics stored in [summary_metrics_native.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260405_230549/summary_metrics_native.json)
- Statistical analysis:
  - predictor: `model.pred_vs_gt_knee_flex_rmse_deg`
  - outcome: `sim.excess.instability_auc_delta`
  - nuisance controls:
    - `match.rmse_knee_deg`
    - `match.rms_thigh_ori_err_deg`
  - method:
    - rank-transform all variables
    - residualize predictor and outcome on the two motion-match controls with OLS
    - compute Pearson correlation of those residuals, which is equivalent to partial Spearman via Frisch-Waugh-Lovell
  - motivation:
    - absolute `PRED` instability inherits clip difficulty and reference-bias
    - the heuristic instability trace is not a calibrated fall probability
    - using `PRED - REF` isolates the extra instability introduced by the model-conditioned rollout

### Current Main Results

- Training:
  - mean held-out test RMSE: `7.84°`
  - median held-out test RMSE: `6.85°`
  - mean held-out MAE: `6.11°`
- 80-window simulation batch:
  - mean predictor RMSE: `8.80°`
  - mean motion-match knee RMSE: `7.93°`
  - median motion-match knee RMSE: `5.86°`
  - mean reference simulated knee RMSE: `12.09°`
  - mean predicted simulated knee RMSE: `9.60°`
  - raw predictor-error vs excess instability: `rho = -0.166`, `p = 0.140`
  - partial Spearman after motion-match controls:
    - `rho = -0.022`
    - `p = 0.851`
- Interpretation:
  - the model achieves sub-`10°` held-out prediction error on the current native-rate GT benchmark
  - however, in the current simulation study, prediction RMSE does not show a statistically significant independent association with excess instability once motion-matching quality is controlled for
  - the instability heuristic itself should not be interpreted as a literal fall probability

### Figures To Use

- raw RMSE vs excess instability: [raw_rmse_vs_excess_instability.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/raw_rmse_vs_excess_instability.png)
- residualized / partial RMSE vs excess instability: [partial_rmse_vs_excess_instability.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/partial_rmse_vs_excess_instability.png)
- motion-match controls vs excess instability: [match_controls_vs_excess_instability.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/match_controls_vs_excess_instability.png)

### Caution For Writing

- Do not describe the current repo as using the earlier custom-data transformer pipeline unless you are explicitly writing historical background.
- The current Word draft still reflects that older path and should be revised before submission.
- The public GT data used here may not be identical to the full corpus described in the external paper, so avoid claiming exact reproduction of that paper's benchmark unless you verify the dataset parity directly.
- The scalar `predicted_fall_risk` is a heuristic instability score, not a calibrated fall probability; in the paper, describe it that way and report excess instability AUC relative to `REF` as the primary outcome.
