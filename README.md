# emg_tst

`emg_tst` is a research codebase for knee-angle prediction from wearable signals and physics-based evaluation of those predictions. The current pipeline uses Georgia Tech lower-limb biomechanics recordings, trains a CNN-BiLSTM regressor on thigh EMG and IMU features, motion-matches held-out windows into MoCapAct, and compares paired MuJoCo rollouts.

The central question is deliberately narrow: when a model has lower knee-angle RMSE, does that translate into lower excess instability in simulation?

The current answer, for this benchmark, is no. The CNN-BiLSTM reaches sub-10 degree held-out error on most folds, but prediction RMSE has no meaningful independent association with excess instability after motion-match quality is controlled.

## Current Results

Canonical training run:

`checkpoints/tst_20260405_173725_all`

Canonical simulation run:

`artifacts/phys_eval_v2/runs/20260406_205003`

| Measure | Value |
|---|---:|
| Held-out folds | 55 |
| Mean held-out test RMSE | 7.84 deg |
| Median held-out test RMSE | 6.85 deg |
| Mean held-out MAE | 6.11 deg |
| Folds below 10 deg RMSE | 46 / 55 |
| Simulation trials | 80 |
| Mean model RMSE in simulation windows | 8.80 deg |
| Mean motion-match knee RMSE | 7.93 deg |
| Mean REF instability AUC | 0.819 |
| Mean PRED instability AUC | 1.019 |
| Mean excess instability AUC | 0.200 |
| Raw Spearman rho, RMSE vs excess AUC | -0.168, p = 0.136 |
| Partial Spearman rho after match controls | -0.019, p = 0.867 |

`excess instability AUC` is defined as `PRED - REF` for each matched clip. The instability trace is an XCoM-margin heuristic, not a calibrated fall probability.

## Reproducing the Benchmark

Install dependencies:

```bash
pip install -r requirements_tst.txt
```

Convert the Georgia Tech normal-walk recordings if the local `gt_data*.npy` files need to be rebuilt:

```bash
python -c "from emg_tst.gt_dataset import ensure_normal_walk_recordings; ensure_normal_walk_recordings()"
```

Train the CNN-BiLSTM:

```bash
python -m emg_tst.run_experiment
```

Build the held-out query pool used by the simulator:

```bash
python split_to_samples.py
```

Run the paired physical evaluation:

```powershell
$env:EMG_TST_RUN_DIR = "checkpoints\\tst_20260405_173725_all"
$env:MOCAP_PHYS_EVAL_ALLOW_PARTIAL = "1"
$env:MOCAP_PHYS_EVAL_N_TRIALS = "80"
python -m mocap_phys_eval.run
```

Run the partial Spearman analysis:

```bash
python -m analysis.correlation --run-dir artifacts/phys_eval_v2/runs/<run_id>
```

## Data and Features

The benchmark path uses converted Georgia Tech normal-walk recordings named `gt_data000.npy` through `gt_data054.npy`.

Each 200 Hz timestep contains 10 input features:

| Columns | Signal |
|---|---|
| 0-3 | EMG envelopes: `RRF`, `RBF`, `RVL`, `RMGAS` |
| 4-6 | Right anterior-thigh accelerometer: `X`, `Y`, `Z` |
| 7-9 | Right anterior-thigh gyroscope: `X`, `Y`, `Z` |

EMG preprocessing follows the benchmark implementation:

1. High-pass filter at 20 Hz.
2. Full-wave rectify.
3. Low-pass filter at 5 Hz to form the linear envelope.
4. Interpolate the envelope onto the native 200 Hz IMU and knee-angle timeline.

The target is right-knee included angle:

```text
knee_included_deg = 180 - clip(-knee_angle_r, 0, 180)
```

The convention is `180 deg = full extension`; smaller values indicate more flexion. The target is divided by 180 during training.

## Windowing

| Setting | Value |
|---|---:|
| Input window | 400 samples |
| Duration | 2.0 s at 200 Hz |
| Forecast horizon | 2 samples |
| Horizon duration | 10 ms |
| Training window stride | 1 |
| Samples per epoch | 8,192 |
| Evaluation-pool stride | 60 |

The simulator query pool is stored in `samples_dataset.npy`. It includes held-out windows, source file identity, start index, thigh pitch sequences, and marker-derived thigh quaternion sequences. The current benchmark uses scalar `thigh_knee_d` matching by default because it produced the best knee-match quality on the retained 80-window pool.

## Model

The main model is `CnnBiLstmLastStep` in `emg_tst/model.py`.

```text
Input: [B, 400, 10]
Conv1d(10 -> 32, kernel=5, padding=2) + GELU
Conv1d(32 -> 32, kernel=5, padding=2) + GELU + Dropout(0.10)
BiLSTM(hidden=64, layers=2, bidirectional=True)
Last timestep readout -> [B, 128]
Linear(128 -> 64) + GELU + Dropout(0.10)
Linear(64 -> 1)
Output: knee included angle at t + 10 ms
```

Default training settings live in `emg_tst/run_experiment.py`.

| Setting | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 128 |
| Max epochs | 6 |
| Early stopping patience | 2 |
| Loss | Huber |
| Huber delta | 5 deg |
| Gradient clipping | 1.0 |

Training uses leave-one-file-out cross-validation. Each fold holds out one converted Georgia Tech recording for testing and one additional recording for validation. The best checkpoint is selected by validation RMSE.

## Physical Evaluation

The simulator evaluates held-out query windows only.

For each retained window, the evaluator:

1. Loads the correct fold checkpoint from the LOFO manifest.
2. Runs rolling inference to produce the predicted knee trajectory.
3. Motion-matches the query into the MoCapAct reference bank.
4. Runs a `REF` rollout using the matched expert policy.
5. Runs a paired `PRED` rollout with the right knee overridden by the model prediction.
6. Computes XCoM-margin instability AUC for both rollouts.
7. Records excess instability as `AUC_PRED - AUC_REF`.

The raw columns `ref_knee_rmse` and `pred_knee_rmse` in simulation artifacts should not be used as a model-performance comparison. `PRED` is lower by construction because the PD controller directly targets the model prediction used during matching.

## MoCapAct Storage

The MoCapAct expert bank is large. An external drive is recommended:

```powershell
$env:MOCAPACT_MODELS_DIR = "D:\\mocapact_models"
```

Optional artifact redirect:

```powershell
$env:MOCAP_PHYS_EVAL_ARTIFACTS_DIR = "D:\\phys_eval_v2_artifacts"
```

One-time prefetch:

```bash
python -m mocap_phys_eval.prefetch
```

## Useful Commands

Visualize predictions from a checkpoint:

```bash
python -m emg_tst.visualize
```

Plot dataset overview figures:

```bash
python plot_data.py
```

Run the learning-curve analysis:

```bash
python -m emg_tst.learning_curve
```

Replay a completed physical-evaluation run:

```bash
python -m mocap_phys_eval.replay
```

## Repository Map

| Path | Purpose |
|---|---|
| `emg_tst/data.py` | EMG preprocessing, timestamp alignment, and recording loading |
| `emg_tst/gt_dataset.py` | Georgia Tech dataset conversion |
| `emg_tst/model.py` | CNN-BiLSTM and experimental model definitions |
| `emg_tst/run_experiment.py` | LOFO training and evaluation |
| `emg_tst/visualize.py` | Checkpoint prediction visualization |
| `split_to_samples.py` | Held-out simulator query-pool construction |
| `mocap_phys_eval/run.py` | Motion matching and paired MuJoCo evaluation |
| `mocap_phys_eval/replay.py` | Saved rollout replay |
| `analysis/correlation.py` | Partial Spearman / FWL residualization |

## Platform Notes

On Windows systems using AMD/DirectML, the CNN-BiLSTM may fall back to CPU because DirectML does not cleanly support the LSTM cell used here.

```bash
pip install torch-directml
```
