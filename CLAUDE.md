# CLAUDE.md — AI Assistant Guide for emg_tst

## Project Overview

**emg_tst** is a research project for predicting **knee joint angle from surface EMG (electromyography) signals** using a Time Series Transformer (TST) model. The system records EMG + IMU data from wearable hardware, trains a transformer for knee angle regression, and evaluates predictions via motion-capture-driven physics simulation.

### Architecture at a Glance

```
Hardware (uMyo EMG + BWT901CL IMU)
    ↓ rigtest.py
Raw .npy recordings
    ↓ split_to_samples.py
samples_dataset.npy (1s fixed windows)
    ↓ emg_tst/run_experiment.py
Trained checkpoints (5-fold CV)
    ↓ emg_tst/visualize.py / mocap_evaluation/run_evaluation.py
Predictions + physics simulation results
```

---

## Repository Structure

```
emg_tst/
├── emg_tst/                    # Core TST model package
│   ├── model.py               # Transformer definitions (encoder, pretrainer, regressor)
│   ├── data.py                # Data loading, feature extraction, dataset classes
│   ├── masking.py             # Stateful Markov masking for pretraining
│   ├── run_experiment.py      # Main training script (5-fold CV, hardcoded config)
│   └── visualize.py           # Prediction visualization
├── mocap_evaluation/           # Motion capture evaluation pipeline
│   ├── bvh_parser.py          # BVH motion capture file parser
│   ├── cmu_catalog.py         # CMU mocap database index (curated subjects/trials)
│   ├── cmu_downloader.py      # CMU database batch downloader
│   ├── bandai_namco_downloader.py  # Bandai Namco dataset downloader
│   ├── mocap_loader.py        # Load BVH files with standardized joint angles
│   ├── motion_matching.py     # DTW-based mocap-to-IMU signal matching
│   ├── prosthetic_sim.py      # MuJoCo/PyBullet physics simulation
│   ├── run_evaluation.py      # End-to-end evaluation orchestrator
│   ├── visualize_match.py     # Plot matched mocap vs query curves
│   ├── sample_data.py         # Extract real walking segment curves from recordings
│   ├── external_sample_data.py # External gait data handling
│   └── mock_data.py           # Generate synthetic knee/thigh curves for testing
├── uMyo_python_tools/          # Hardware sensor utilities (EMG device SDK)
│   ├── rigtest.py             # Main data recording script (EMG + IMU)
│   ├── umyo_class.py          # uMyo device abstraction
│   ├── umyo_parser.py         # EMG parsing and feature extraction
│   ├── umyo_mouse.py          # Real-time EMG → mouse cursor control
│   ├── quat_math.py           # Quaternion math for IMU orientation
│   ├── display_stuff.py       # Real-time multichannel plotting
│   └── ...                    # Other hardware utility scripts
├── split_to_samples.py        # Convert recordings to fixed-length sample windows
├── plotdata.py                # Visualize raw recordings
├── imutest.py                 # IMU streaming/visualization test
├── requirements_tst.txt       # Python dependencies
├── README_TST.md              # Comprehensive workflow documentation
└── README.md                  # Minimal project marker
```

---

## Development Workflows

### 1. Data Collection
```bash
# Record EMG + IMU data from hardware (saves data0.npy, data1.npy, ...)
python uMyo_python_tools/rigtest.py

# Plot a recorded file to inspect quality
python plotdata.py
```

### 2. Data Preparation
```bash
# Convert variable-length recordings into fixed 1-second windows
# Reads all .npy files in current directory, outputs samples_dataset.npy
python split_to_samples.py
```

### 3. Model Training
```bash
# Run 5-fold cross-validation training (pretraining + fine-tuning)
# All config is hardcoded in run_experiment.py
python -m emg_tst.run_experiment
```
Output: `checkpoints/tst_YYYYMMDD_HHMMSS/fold_XX/reg_best.pt`

### 4. Visualization
```bash
# Visualize predictions from the latest checkpoint
python emg_tst/visualize.py
```

### 5. Mocap Evaluation
```bash
# Evaluate with synthetic mock data (no external data needed)
python -m mocap_evaluation.run_evaluation --mock-data --full-db

# Download real mocap data first, then evaluate
python -m mocap_evaluation.bandai_namco_downloader --dest mocap_data
python -m mocap_evaluation.run_evaluation --full-db

# Download CMU mocap database
python -m mocap_evaluation.cmu_downloader
```

### 6. IMU Testing
```bash
# Stream and plot IMU data in real-time (Bluetooth)
python imutest.py
```

---

## Key Technical Details

### Model Architecture (`emg_tst/model.py`)

Three classes, each wrapping the previous:

| Class | Role | Input → Output |
|-------|------|----------------|
| `TSTEncoder` | Encoder-only transformer | `[B, T, n_vars]` → `[B, T, d_model]` |
| `TSTPretrainDenoiser` | Pretraining wrapper | `[B, T, n_vars]` → `[B, T, n_vars]` (reconstruction) |
| `TSTRegressor` | Fine-tuning wrapper | `[B, T, n_vars]` → `[B, T, 1]` (knee angle) |

**Default hyperparameters** (all hardcoded in `run_experiment.py`):
- `d_model=128`, `n_heads=8`, `d_ff=256`, `n_layers=3`
- `dropout=0.1`, `batch_size=64`, `lr=3e-4`
- Pretraining: 40 epochs; Fine-tuning: 20 epochs with cosine annealing LR

### Feature Engineering (`emg_tst/data.py`)

Per 100Hz timestep, 13 features are extracted per EMG sensor from the 200Hz raw waveform:
- **Time-domain (5)**: RMS, MAV (mean absolute value), WL (waveform length), ZC (zero crossings), SSC (slope sign changes)
- **Spectral (8)**: power in 8 frequency bands via FFT

With 3 sensors + 1 thigh angle: **40 total input features** (`n_vars=40`).

### Data Format (`.npy` files)

Recordings saved by `rigtest.py` are NumPy dictionaries with keys:
- `emg_sensor{1,2,3}`: spectrum-based EMG features at ~200Hz
- `raw_emg_sensor{1,2,3}`: raw waveforms at higher native rate
- `imu`: knee angle label (degrees, included-angle convention)
- `thigh_angle`: secondary angle input feature
- `effective_hz`: actual sampling rate

After `split_to_samples.py`: `samples_dataset.npy` array of shape `[N_samples, seq_len=200, n_vars+1]` (last column is target knee angle).

### Angle Convention

**CRITICAL**: All angles use the **included-angle** convention:
- `180°` = straight/neutral
- Smaller values = more flexion
- Conversion from flexion angle: `included = 180 - flexion`

This convention is consistent across: hardware recording (`rigtest.py`), BVH parsing (`bvh_parser.py`), motion matching (`motion_matching.py`), and simulation (`prosthetic_sim.py`).

### Masking Strategy (`emg_tst/masking.py`)

Pretraining uses **stateful 2-state Markov masking** (not i.i.d.):
- ~14% of timesteps masked (target `MASK_R=0.14`)
- Mean masked segment length: ~3 timesteps
- State machine ensures contiguous masked segments for realistic pretraining

### Training Strategy (`emg_tst/run_experiment.py`)

- **5-fold cross-validation** at the sample level (no temporal leakage between 1s windows)
- Phase 1: Pretraining with masked reconstruction (`TSTPretrainDenoiser`)
- Phase 2: Fine-tuning for regression (`TSTRegressor`), encoder weights transferred
- Metrics reported: last-step RMSE, last-step MAE, full-sequence RMSE, full-sequence MAE
- Best checkpoint selected by validation RMSE

### Motion Matching (`mocap_evaluation/motion_matching.py`)

DTW-based matching strategy:
1. Sliding window over mocap database
2. L2 pre-filter to shortlist candidates
3. DTW with Sakoe-Chiba band constraint
4. Features: joint angles + their velocities (dual-feature for robustness)

### Physics Simulation (`mocap_evaluation/prosthetic_sim.py`)

- All joints driven by mocap reference **except right knee**
- Right knee driven by **model prediction**
- Reference simulation: right knee from ground-truth label
- Metrics: CoM height (fall threshold: `< 0.55m`), gait symmetry, step count, stability score
- Backend: PyBullet (MuJoCo as optional alternative)

---

## Code Conventions

### Naming
- **Modules**: lowercase (`data.py`, `model.py`)
- **Classes**: CamelCase (`TSTEncoder`, `StandardScaler`, `BVHParser`)
- **Constants**: UPPER_SNAKE_CASE (`BATCH_SIZE`, `MASK_R`, `TARGET_FPS`)
- **Private functions**: leading underscore (`_extract_raw_features_for_sensor`)
- **Tensor shapes**: documented as `[B, T, F]` (batch, timesteps, features)

### Configuration
- **All training hyperparameters are hardcoded** in `run_experiment.py` and `split_to_samples.py`
- No CLI argument parsing in training/splitting scripts (modify source directly)
- `mocap_evaluation/run_evaluation.py` uses `argparse` for its flags

### Data Shape Convention
- Batch tensors: `[B, T, F]` — batch first, timesteps second, features last
- Sequences: `(T,)` or `(T, dim)`
- Features always in the last dimension

### Normalization
- Z-score: `(x - mean) / std`
- `StandardScaler` in `data.py` is fit on training fold only, then applied to test fold
- Prevents data leakage across CV folds

---

## Dependencies

Install from `requirements_tst.txt`:
```bash
pip install -r requirements_tst.txt
```

Key packages:
| Package | Role |
|---------|------|
| `torch` | Transformer model, training |
| `numpy` | Data I/O, numerical computing |
| `scipy` | Signal resampling, spatial transforms |
| `pybullet` | Physics simulation |
| `pillow` | GIF generation for gait visualization |
| `matplotlib` | All plotting |
| `pywitmotion` | BWT901CL IMU Bluetooth communication |
| `pyserial` | uMyo EMG device serial communication |

---

## Important Notes for AI Assistants

1. **No test suite**: This is a research project. There are no unit/integration tests. Validation happens via 5-fold CV RMSE and manual inspection of plots.

2. **Hardcoded config**: When asked to change hyperparameters, edit the constants at the top of `run_experiment.py` or `split_to_samples.py` directly — there is no config file or CLI.

3. **Angle convention is critical**: Any code touching angle values must use included-angle (180° = straight). Confusion between flexion and included-angle will break matching and simulation.

4. **Hardware dependency**: `uMyo_python_tools/rigtest.py` and `imutest.py` require physical hardware. They cannot be run in a headless/CI environment.

5. **Checkpoint paths**: Training saves checkpoints to `checkpoints/tst_YYYYMMDD_HHMMSS/fold_XX/`. Visualization scripts look for the most recent checkpoint directory.

6. **Data location**: Recordings (`.npy` files) are expected in the current working directory by default. `mocap_data/` is gitignored and holds downloaded BVH files.

7. **Full-sequence supervision**: The regressor predicts **every timestep**, not just the final one. Both last-step and full-sequence metrics are reported and matter.

8. **Physics simulation backend**: PyBullet is the default. MuJoCo support exists but requires a separate license/install. Use PyBullet for development unless specifically testing MuJoCo behavior.
