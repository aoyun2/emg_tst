# emg_tst

Research project for predicting **knee joint angle from surface EMG (electromyography) signals** using a Time Series Transformer (TST) model. The system records EMG + IMU data from wearable hardware, trains a transformer for knee angle regression, and evaluates predictions via motion-capture-driven physics simulation.

## Architecture Overview

```
Hardware (uMyo EMG + BWT901CL IMU)
    ↓ uMyo_python_tools/rigtest.py
Raw .npy recordings
    ↓ split_to_samples.py
samples_dataset.npy  (1s fixed windows, 100 Hz)
    ↓ emg_tst/run_experiment.py
Trained checkpoints  (5-fold CV)
    ↓ emg_tst/visualize.py / mocap_evaluation/run_evaluation.py
Predictions + physics simulation results
```

## Repository Structure

```
emg_tst/
├── emg_tst/                         # Core TST model package
│   ├── model.py                    # Transformer definitions (encoder, pretrainer, regressor)
│   ├── data.py                     # Data loading, feature extraction, dataset classes
│   ├── masking.py                  # Stateful Markov masking for pretraining
│   ├── run_experiment.py           # Main training script (5-fold CV, hardcoded config)
│   └── visualize.py                # Prediction visualization
├── mocap_evaluation/                # Motion capture evaluation pipeline
│   ├── bvh_parser.py               # BVH motion capture file parser
│   ├── cmu_catalog.py              # CMU mocap database index (curated subjects/trials)
│   ├── cmu_downloader.py           # CMU database batch downloader
│   ├── bandai_namco_downloader.py  # Bandai Namco dataset downloader
│   ├── lafan1_downloader.py        # Ubisoft LAFAN1 dataset downloader
│   ├── sfu_downloader.py           # SFU Motion Capture database downloader
│   ├── download_all.py             # Download all mocap datasets at once
│   ├── mocap_loader.py             # Load BVH files with standardized joint angles
│   ├── motion_matching.py          # DTW-based mocap-to-IMU signal matching
│   ├── prosthetic_sim.py           # PyBullet physics simulation
│   ├── run_evaluation.py           # End-to-end evaluation orchestrator
│   ├── visualize_match.py          # Plot matched mocap vs query curves
│   ├── sample_data.py              # Extract real walking segment curves from recordings
│   ├── external_sample_data.py     # External gait data handling
│   └── mock_data.py                # Generate synthetic knee/thigh curves for testing
├── uMyo_python_tools/               # Hardware sensor utilities (EMG device SDK)
│   ├── rigtest.py                  # Main data recording script (EMG + IMU)
│   ├── umyo_class.py               # uMyo device abstraction
│   ├── umyo_parser.py              # EMG parsing and feature extraction
│   ├── umyo_mouse.py               # Real-time EMG → mouse cursor control
│   ├── quat_math.py                # Quaternion math for IMU orientation
│   ├── display_stuff.py            # Real-time multichannel plotting
│   └── ...                         # Other hardware utility scripts
├── split_to_samples.py             # Convert recordings to fixed-length sample windows
├── plotdata.py                     # Visualize raw recordings
├── imutest.py                      # IMU streaming/visualization test
└── requirements_tst.txt            # Python dependencies
```

## Quickstart

### Dependencies

```bash
pip install -r requirements_tst.txt
```

### 1. Record Data (requires hardware)

```bash
# Record EMG + IMU data — writes data0.npy, data1.npy, ...
python uMyo_python_tools/rigtest.py

# Inspect a recording
python plotdata.py
```

> **Hardware note**: `rigtest.py` and `imutest.py` require a uMyo EMG device (serial) and a BWT901CL IMU (Bluetooth). They cannot run headlessly.

### 2. Prepare Dataset

```bash
# Slice all .npy recordings in the current directory into
# non-overlapping 1-second windows → samples_dataset.npy
python split_to_samples.py
```

Output shape: `[N_samples, seq_len=200, n_vars+1]` where the last column is the target knee angle.

### 3. Train

```bash
# 5-fold cross-validation: pretraining then fine-tuning
python -m emg_tst.run_experiment
```

Checkpoints are saved under `checkpoints/tst_YYYYMMDD_HHMMSS/fold_XX/reg_best.pt`.

All hyperparameters are hardcoded at the top of `run_experiment.py`; edit the file directly to change them.

### 4. Visualize Predictions

```bash
python -m emg_tst.visualize
```

Edit `CKPT_PATH` inside `visualize.py` to point to a specific `reg_best.pt` checkpoint.

---

## Model Details

### Architecture (`emg_tst/model.py`)

| Class | Role | Input → Output |
|-------|------|----------------|
| `TSTEncoder` | Encoder-only transformer | `[B, T, n_vars]` → `[B, T, d_model]` |
| `TSTPretrainDenoiser` | Pretraining wrapper | `[B, T, n_vars]` → `[B, T, n_vars]` (reconstruction) |
| `TSTRegressor` | Fine-tuning wrapper | `[B, T, n_vars]` → `[B, T, 1]` (knee angle) |

**Default hyperparameters** (all hardcoded in `run_experiment.py`):

| Parameter | Value |
|-----------|-------|
| `d_model` | 128 |
| `n_heads` | 8 |
| `d_ff` | 256 |
| `n_layers` | 3 |
| `dropout` | 0.1 |
| `batch_size` | 64 |
| `lr` | 3e-4 |
| Pretraining epochs | 40 |
| Fine-tuning epochs | 20 (cosine annealing LR) |

### Feature Engineering (`emg_tst/data.py`)

Per 100 Hz timestep, 13 features are extracted per EMG sensor from the 200 Hz raw waveform:

- **Time-domain (5)**: RMS, MAV, WL, ZC, SSC
- **Spectral (8)**: power in 8 frequency bands via FFT

With 3 sensors + 1 thigh angle: **40 total input features** (`n_vars=40`).

### Masking Strategy (`emg_tst/masking.py`)

Pretraining uses **stateful 2-state Markov masking**:
- ~14% of timesteps masked (`MASK_R=0.14`)
- Mean masked segment length: ~3 timesteps
- Contiguous segments for realistic reconstruction pretraining

### Training Strategy

- **5-fold cross-validation** at the sample level (no temporal leakage between 1s windows)
- Phase 1: masked reconstruction pretraining (`TSTPretrainDenoiser`)
- Phase 2: regression fine-tuning (`TSTRegressor`), encoder weights transferred
- **Full-sequence supervision**: the model predicts the knee angle at every timestep, not just the final one
- Metrics: last-step RMSE/MAE and full-sequence RMSE/MAE; best checkpoint selected by validation RMSE

### Angle Convention

**All angles use the included-angle convention**:
- `180°` = straight / neutral
- Smaller values = more flexion
- Conversion: `included = 180 - flexion`

This convention is consistent across hardware recording, BVH parsing, motion matching, and physics simulation.

---

## Mocap Evaluation

The `mocap_evaluation` package matches recorded (or synthetic) knee/thigh curves to a mocap database using DTW, then drives a physics simulation with model predictions to assess gait quality.

### Download Mocap Data

Four datasets are supported. Download any combination:

```bash
# Recommended: Bandai Namco Motiondataset-2 (~2,900 BVH files, CC BY 4.0)
python -m mocap_evaluation.bandai_namco_downloader --dest mocap_data/bandai

# CMU Graphics Lab mocap database (~300 BVH files, free)
python -m mocap_evaluation.cmu_downloader --dest mocap_data/cmu

# Ubisoft LAFAN1 (~135 BVH files, CC BY-NC-ND 4.0)
python -m mocap_evaluation.lafan1_downloader --dest mocap_data/lafan1

# SFU Motion Capture Lab (~38 BVH files, free for academic use)
python -m mocap_evaluation.sfu_downloader --dest mocap_data/sfu

# Or download everything at once
python -m mocap_evaluation.download_all --root mocap_data
```

Keep all dataset subdirectories under `mocap_data/`; the evaluator aggregates them automatically.

### Run Evaluation

**With synthetic mock data** (no real recordings needed):

```bash
python -m mocap_evaluation.run_evaluation \
  --mock-data \
  --mock-seconds 6 \
  --top-k 5 \
  --full-db \
  --out eval_mock_results.json \
  --save-mock mock_curves.npz
```

**With real walking segments from local BVH files**:

```bash
python -m mocap_evaluation.run_evaluation \
  --real-walk-data \
  --real-seconds 6 \
  --full-db \
  --top-k 5 \
  --out eval_real_walk_results.json \
  --save-real real_walk_curves.npz
```

What happens during evaluation:
1. A query (mock or real) knee+thigh curve is matched against the mocap database via DTW.
2. The top-K matches each drive a PyBullet physics simulation with the **model prediction** substituted for the right knee joint.
3. A reference simulation is also run using the **ground-truth label** for comparison.
4. Metrics reported: CoM height (fall threshold: `< 0.55m`), gait symmetry, step count, stability score.

### Visualize a Match

```bash
python -m mocap_evaluation.visualize_match \
  --aggregate-datasets \
  --mocap-dir mocap_data \
  --seconds 6 \
  --out artifacts/mock_vs_match.png
```

Generates a mock knee/thigh segment, runs motion matching, and saves a query-vs-match plot.

---

## IMU Testing

```bash
# Stream and plot IMU data in real-time over Bluetooth
python imutest.py
```

---

## Dependencies

Key packages from `requirements_tst.txt`:

| Package | Role |
|---------|------|
| `torch` | Transformer model and training |
| `numpy` | Data I/O and numerical computing |
| `scipy` | Signal resampling and spatial transforms |
| `pybullet` | Physics simulation backend |
| `matplotlib` | All plotting |
| `pillow` | GIF generation for gait visualization |
| `pywitmotion` | BWT901CL IMU Bluetooth communication |
| `pyserial` | uMyo EMG device serial communication |
