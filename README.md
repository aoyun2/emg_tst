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

For the GT path, the repo now derives a marker-based right-thigh segment quaternion from the raw GT marker data, not just a hip-angle proxy. The default motion matcher is now `dquat_knee_d`: 3D thigh orientation plus knee dynamics, still scored with a knee-heavy weight (`knee=1.0`, `thigh=0.1`). In the current same-window GT oracle simulation check, this 3D matcher gave slightly better realized simulated knee RMSE and lower predicted risk AUC than the scalar matcher, even though the raw window-overlap knee RMSE was not better.

Integration status:
- the current trainer default in [run_experiment.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/run_experiment.py) is `MODEL_ARCH = "cnn_bilstm"`
- the simulator in [run.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/mocap_phys_eval/run.py) is checkpoint-driven and can load this architecture
- the full GT subject-holdout benchmark run currently lives under [gt_full_subject_holdout](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/gt_full_subject_holdout)
- the latest completed `ALL` fold there reached `test_rmse = 8.96 deg` in [metrics_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/gt_full_subject_holdout/all/metrics_summary.json)
- the evaluator can run that benchmark directly via `EMG_TST_RUN_DIR=artifacts/gt_full_subject_holdout/all` and `MOCAP_PHYS_EVAL_ALLOW_PARTIAL=1`

---

## Methodology

### Inputs

Per 100 Hz timestep, the GT-integrated model receives:
- `1` causal EMG envelope value from `RRF`
- `1` causal EMG envelope value from `RBF`
- `1` causal EMG envelope value from `RVL`
- `1` causal EMG envelope value from `RMGAS`
- right-thigh IMU: `RAThigh_ACCX`, `RAThigh_ACCY`, `RAThigh_ACCZ`
- right-thigh IMU: `RAThigh_GYROX`, `RAThigh_GYROY`, `RAThigh_GYROZ`

Total input width per timestep: `10`
- `4` EMG values = `4 x high-pass -> rectify -> low-pass` preprocessed channels
- `6` IMU values = `3 x accel + 3 x gyro`

The current GT path follows the paper much more closely than the older custom-data path:
- start from GT processed `raw_emg_channels`
- high-pass at `20 Hz`
- rectify with `abs(x)`
- low-pass at `5 Hz`
- resample the full GT stream to `100 Hz`

The GT-converted repo files are stored at `200 Hz`, but [data.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/data.py) resamples the GT branch to `100 Hz` before windowing so the model sees the paper-style timebase.

### Target

The supervised target is the knee included angle:
- `0 deg = fully bent`
- `180 deg = fully extended`

For the GT path, this is derived from `knee_angle_r` in the processed OpenSim angle file:
- `knee_flex_deg = clip(-knee_angle_r, 0, 180)`
- `knee_included_deg = 180 - knee_flex_deg`

### Windowing

The source window is:
- `200` samples
- `2.0 s` at 100 Hz

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
- `SOURCE_WINDOW = 200`
- `CONTEXT_WINDOW = 200`
- `STEM_WIDTH = 32`
- `TCN_KERNEL_SIZE = 5`
- `CNN_DEPTH = 2`
- `LSTM_HIDDEN_SIZE = 64`
- `LSTM_LAYERS = 2`
- `DROPOUT = 0.10`

For an input tensor of shape `(B, 200, 10)`:

1. Convolutional frontend
   - input: all `10` channels together
   - `Conv1d(10 -> 32, kernel_size=5, padding=2)`
   - `GELU`
   - `Conv1d(32 -> 32, kernel_size=5, padding=2)`
   - `GELU`
   - `Dropout(0.10)`

2. Temporal recurrent encoder
   - transpose back to `(B, 200, 32)`
   - `BiLSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)`
   - use only the final timestep output, width `128`

3. Regression head
   - `Linear(128 -> 64)`
   - `GELU`
   - `Dropout(0.10)`
   - `Linear(64 -> 1)`

This is a sequence-to-one causal model. The trained output is a one-step-ahead included-angle forecast because `LABEL_SHIFT = 1` at the GT `100 Hz` timebase. For simulation and plotting, the repo rolls this last-step model across the full query sequence one timestep at a time and then causally aligns the forecast before applying it as the prosthetic target.

### Architecture Diagram

```text
Input window: (B, 200, 10)
  |
  +-- Conv1d(10 -> 32, k=5, pad=2)
  +-- GELU
  +-- Conv1d(32 -> 32, k=5, pad=2)
  +-- GELU
  +-- Dropout(0.10)
  |
  +-- transpose to (B, 200, 32)
  +-- BiLSTM(32 -> 64 per direction, layers=2)
  +-- take last timestep: (B, 128)
  +-- Linear(128 -> 64)
  +-- GELU
  +-- Dropout(0.10)
  +-- Linear(64 -> 1)
  |
  +--> knee included angle at t + 1 sample
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
- label target: normalized by `180`, with `LABEL_SHIFT = 1` sample (`10 ms` at the GT 100 Hz timebase)

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

The simulation path uses `thigh_quat_seq` by default for `dquat_knee_d` matching.

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
$env:EMG_TST_RUN_DIR = "artifacts\\gt_full_subject_holdout\\all"
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

Current GT benchmark result:
- training: [metrics_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/gt_full_subject_holdout/all/metrics_summary.json)
- simulation/statistics: [summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260404_165818/summary.json)
- correlation output: [partial_spearman_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260404_165818/analysis/partial_spearman_summary.json)

Current interpretation:
- held-out model test RMSE on the GT subject-holdout run is `8.96°`
- mean predictor RMSE across the 80 evaluated simulation windows is `8.63°`
- raw predictor RMSE vs simulation risk is weakly positive
- after controlling for motion-match knee and thigh errors, the partial correlation is essentially null

Correlation figures:
- raw RMSE vs risk: [raw_rmse_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/raw_rmse_vs_risk.png)
- partial / residualized RMSE vs risk: [partial_rmse_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/partial_rmse_vs_risk.png)
- motion-match controls vs risk: [match_controls_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/match_controls_vs_risk.png)

The main paper statistic uses:
- predictor: `model.pred_vs_gt_knee_flex_rmse_deg`
- outcome: `sim.pred.balance_risk_auc`

Current practical note:
- `mocap_phys_eval` will only use architectures that exist in saved compatible `_all` checkpoints
- so after changing the model or feature layout, always run:

```bash
python -m emg_tst.run_experiment
python split_to_samples.py
```

- before expecting the simulator to use the new model

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

### Current Study Framing

- Research question: whether lower knee-prediction error from a wearable-sensor regression model corresponds to lower simulated balance risk when the predicted knee trajectory is injected into a MoCapAct / MuJoCo prosthetic-override pipeline.
- Main dataset in the current repo: Georgia Tech processed biomechanics dataset converted into repo-native `gt_data*.npy` recordings.
- Main model in the current repo: CNN-BiLSTM last-step regressor, not the older custom-data transformer.
- Simulation pipeline: GT held-out windows are motion-matched into a MoCapAct reference bank, then MuJoCo runs `REF` and `PRED` rollouts while overriding only the right knee.

### Data and Preprocessing

- Source signals:
  - `4` EMG channels: `RRF`, `RBF`, `RVL`, `RMGAS`
  - `6` right-thigh IMU channels: `RAThigh_ACC{X,Y,Z}`, `RAThigh_GYRO{X,Y,Z}`
  - label source: GT `knee_angle_r`
  - 3D thigh motion-match signal: marker-derived `thigh_quat_wxyz`
  - scalar diagnostic thigh proxy: `hip_flexion_r`
- GT preprocessing in the current repo:
  - high-pass EMG at `20 Hz`
  - rectify with absolute value
  - low-pass at `5 Hz`
  - resample the GT branch to `100 Hz`
  - use `2.0 s` windows of `200` samples
  - z-score the `4` EMG channels per recording
  - fit a global scaler on training recordings
- Label convention:
  - convert GT knee flexion to included angle
  - normalize the supervised label by dividing by `180`
  - train with `LABEL_SHIFT = 1`, so the model predicts one sample ahead at the `100 Hz` GT timebase

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
  - subject-holdout GT run stored in [metrics_summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/gt_full_subject_holdout/all/metrics_summary.json)
  - best current `ALL` result: `test_rmse = 8.96°`, `test_seq_rmse = 8.96°`
- Simulation benchmark:
  - run stored in [summary.json](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/artifacts/phys_eval_v2/runs/20260404_165818/summary.json)
  - `80` successful GT held-out windows
  - fixed seed `42`
  - model checkpoint source: `artifacts/gt_full_subject_holdout/all/fold_01/reg_best.pt`
- Statistical analysis:
  - predictor: `model.pred_vs_gt_knee_flex_rmse_deg`
  - outcome: `sim.pred.balance_risk_auc`
  - nuisance controls:
    - `match.rmse_knee_deg`
    - `match.rms_thigh_ori_err_deg`
  - method:
    - rank-transform all variables
    - residualize predictor and outcome on the two motion-match controls with OLS
    - compute Pearson correlation of those residuals, which is equivalent to partial Spearman via Frisch-Waugh-Lovell

### Current Main Results

- Training:
  - held-out test RMSE: `8.96°`
- 80-window simulation batch:
  - mean predictor RMSE: `8.63°`
  - raw predictor-error vs risk relationship: weak positive association
  - partial Spearman after motion-match controls:
    - `rho = -0.0105`
    - `p = 0.927`
- Interpretation:
  - the model can achieve sub-`10°` held-out prediction error on the current GT benchmark
  - however, in the current simulation study, prediction RMSE does not show a meaningful independent association with balance-risk AUC once motion-matching quality is controlled for
  - the stronger risk correlates in this run are the motion-match diagnostics, especially thigh-match error

### Figures To Use

- raw RMSE vs risk: [raw_rmse_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/raw_rmse_vs_risk.png)
- residualized / partial RMSE vs risk: [partial_rmse_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/partial_rmse_vs_risk.png)
- motion-match controls vs risk: [match_controls_vs_risk.png](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/figures/gt_correlation/match_controls_vs_risk.png)

### Caution For Writing

- Do not describe the current repo as using the earlier custom-data transformer pipeline unless you are explicitly writing historical background.
- The current Word draft still reflects that older path and should be revised before submission.
- The public GT data used here may not be identical to the full corpus described in the external paper, so avoid claiming exact reproduction of that paper's benchmark unless you verify the dataset parity directly.
