# emg_tst

Real-time EMG/IMU knee-angle prediction evaluated inside MoCapAct physics with per-snippet expert policies.

The current main pipeline is:
- model input: raw EMG snippets + residual-thigh IMU only
- model: sensor-fusion transformer
- training: direct from raw `data*.npy` recordings with lazy stride-1 windows
- simulation: rolling causal inference over held-out query windows from `samples_dataset.npy`

The calf IMU is not used as a model input. It is only part of the recording rig used to generate the training label.

---

## Methodology

### Inputs

Per 200 Hz timestep, the model receives:
- `32` raw EMG samples from sensor 1
- `32` raw EMG samples from sensor 2
- `32` raw EMG samples from sensor 3
- thigh angular velocity `(omega_x, omega_y, omega_z)`
- thigh quaternion `(w, x, y, z)`

Total input width per timestep: `103`
- `96` EMG values = `3 x 32`
- `7` IMU values = `3 omega + 4 quat`

No RMS/MAV/FFT hand-engineered EMG features are used in the main training path.

### Target

The supervised target is the recorded knee included angle:
- `0 deg = fully bent`
- `180 deg = fully extended`

The current label path is still the legacy Euler-difference recording method because it is the most stable label source on this hardware pair right now.

### Windowing

There are two distinct window lengths:
- source window: `200` samples = `1.0 s` at 200 Hz
- transformer context: last `100` samples of that source window = `0.5 s`

Training windows are generated lazily from raw recordings with `stride = 1`.

### Preprocessing

For each recording:
- z-score the first `96` EMG columns independently
- leave the thigh IMU channels unnormalized at this stage

Then:
- fit a global mean/std scaler across the training recordings
- normalize the knee label by dividing by `180`

---

## Exact Architecture

The integrated model is `SensorFusionTransformer` in [emg_tst/model.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/model.py).

For an input tensor of shape `(B, 100, 103)`:

1. EMG frontend
   - reshape the first `96` columns at each timestep into `(3 sensors, 32 lags)`
   - `Conv1d(3 -> 24, kernel_size=5, padding=2)`
   - `GELU`
   - `Conv1d(24 -> 32, kernel_size=5, padding=2)`
   - `GELU`
   - `AdaptiveAvgPool1d(1)`
   - `Linear(32 -> 48)`

2. IMU frontend
   - input is the final `7` columns `(omega_xyz + quat_wxyz)`
   - `Linear(7 -> 48)`
   - `GELU`
   - `Linear(48 -> 48)`

3. Fusion block
   - concatenate the EMG and IMU embeddings to width `96`
   - `Linear(96 -> 96)`
   - `GELU`
   - `Dropout(0.20)`

4. Temporal transformer
   - learnable positional embedding of shape `(1, 100, 96)`
   - `3` pre-norm transformer encoder layers
   - each layer uses:
     - `MultiheadAttention(embed_dim=96, num_heads=6, batch_first=True)`
     - feed-forward width `192`
     - `LayerNorm`
     - `Dropout(0.20)`
     - `GELU`

5. Output head
   - take the last token only
   - `Linear(96 -> 1)`
   - output is the normalized current knee angle for the final timestep of the context window

This is a sequence-to-one causal model. For simulation and plotting, the repo rolls this last-step model across the full query sequence one timestep at a time.

---

## Training

The main trainer is [emg_tst/run_experiment.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/run_experiment.py).

Key training settings:
- optimizer: `Adam`
- learning rate: `1e-3`
- weight decay: `1e-4`
- batch size: `256`
- max epochs: `20`
- early stopping patience: `5`
- gradient clipping: `1.0`
- train noise: Gaussian noise with `std=0.02`
- multimodal augmentation: IMU dropout probability `0.30` for the `ALL` model

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

Outputs:
- `checkpoints/<run_name>_all/fold_XX/reg_best.pt`
- `checkpoints/<run_name>_all/fold_XX/split_manifest.json`
- `checkpoints/<run_name>_all/cv_manifest.json`
- `checkpoints/<run_name>/ablation_summary.json`

---

## Data

Record `data*.npy` files with `uMyo_python_tools/rigtest_gui.py`.

Expected recording contents:
- `raw_emg_sensor1/2/3`
- `raw_emg_times1/2/3`
- `timestamps`
- `thigh_quat_wxyz`
- `knee_included_deg`

The main trainer consumes raw recordings directly. It does not train from `samples_dataset.npy`.

### Build `samples_dataset.npy`

```bash
python split_to_samples.py
```

This file is still used for:
- physical evaluation query pools
- held-out window bookkeeping
- some visualizations

It is not the main training source anymore.

---

## Physical Evaluation

Run the MuJoCo evaluator:

```bash
python -m mocap_phys_eval
```

Replay the latest run:

```bash
python -m mocap_phys_eval.replay
```

The evaluator:
1. loads held-out query windows from `samples_dataset.npy`
2. loads the latest compatible `*_all` transformer checkpoints
3. maps each held-out window to the correct LOFO fold via `split_manifest.json` / `cv_manifest.json`
4. rolls the last-step transformer causally across the full query window
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

The main paper statistic uses:
- predictor: `model.pred_vs_gt_knee_flex_rmse_deg`
- outcome: `sim.pred.balance_risk_auc`

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

The visualizer now uses the same rolling last-step transformer inference path as the simulator.

---

## Learning Curve

Optional learning-curve sweep on the same transformer pipeline:

```bash
python -m emg_tst.learning_curve
```

This also trains directly from raw `data*.npy` recordings, not from `samples_dataset.npy`.

---

## Repo Structure

- [emg_tst/data.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/data.py): raw EMG snippet extraction, timestamp alignment, resampling, recording loader
- [emg_tst/model.py](/C:/Users/aaron/OneDrive/Documents/GitHub/emg_tst/emg_tst/model.py): transformer blocks and rolling last-step inference helper
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
