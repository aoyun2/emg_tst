# End-to-End Methodology Plan for `emg_tst`

This document is the full runbook for reproducing the paper methodology in this repo:

1. Record wearable data
2. Build non-overlapping 1.0 s windows
3. Train the TST with leave-one-file-out (LOFO) cross-validation
4. Run the MoCapAct physical evaluation on held-out LOFO windows only
5. Compute the final partial Spearman correlation using balance-risk AUC

The goal is to make the implementation match the paper as closely as possible.

---

## 1. Study Summary

The implemented methodology is:

1. Signal preprocessing
   - Resample recordings to 200 Hz
   - Extract 43 features per timestep
2. Transformer training
   - Encoder-only TST
   - Masked reconstruction pretraining
   - Regression fine-tuning
   - Outer split: leave-one-file-out (LOFO)
   - Inner validation: one additional training file held out inside each outer fold
3. Physical evaluation
   - Use only held-out LOFO test windows
   - Match each window to the MoCapAct reference bank
   - Run MuJoCo with the matched expert policy
   - Override the right knee with the fold-matched TST prediction
   - Record balance-risk AUC
4. Statistics
   - Predictor: TST knee RMSE against ground truth
   - Outcome: balance-risk AUC from the `PRED` rollout
   - Controls: motion-match knee RMSE and thigh orientation RMS geodesic error
   - Test: partial Spearman rho via rank transform + OLS residualization

If the evaluator falls back to demo BVH mode, that run is only a sanity check and does not count for the paper.

---

## 2. Hardware and Data Assumptions

Each recording session captures:

| Stream | Hardware | Placement | Rate | Used For |
|---|---|---|---|---|
| 3 x sEMG | uMyo sensors via USB base | Upper right thigh: VM, SM, BF | about 400 Hz raw | TST input features |
| 1 x IMU quaternion | BWT901CL | Right shin/calf | 200 Hz | Thigh orientation input and knee label derivation |

The knee label is computed online in `rigtest.py`:

```text
knee_included_deg = 180 - (shin_roll - thigh_yaw)
```

Angle convention:

- Label space: `0 = fully bent`, `180 = straight`
- MoCapAct knee flexion space: `0 = straight`
- Conversion used in the evaluator: `knee_flex_deg = 180 - knee_included_deg`

---

## 3. Features and Windowing

Each 200 Hz timestep contains 43 model input features:

- EMG features: 39 total
  - 13 per sensor x 3 sensors
  - 5 time-domain features: RMS, MAV, WL, ZC, SSC
  - 8 FFT band-power features
- Thigh quaternion: 4 total
  - `wxyz`

Windowing rules:

- Window length: 200 timesteps
- Effective duration: 1.0 second at 200 Hz
- Windows are consecutive and non-overlapping
- All windows from the same recording file stay together during LOFO

This is implemented by:

```bash
python split_to_samples.py
```

---

## 4. Prerequisites

### Python environment

Install dependencies:

```bash
pip install -r requirements_tst.txt
```

### MoCapAct storage

The expert zoo is large. Use a non-OneDrive drive if possible.

PowerShell example:

```powershell
$env:MOCAPACT_MODELS_DIR = "D:\mocapact_models"
```

Optional artifact location:

```powershell
$env:MOCAP_PHYS_EVAL_ARTIFACTS_DIR = "D:\phys_eval_v2_artifacts"
```

Optional Hugging Face token:

```powershell
$env:HF_TOKEN = "hf_..."
```

### One-time prefetch

Before the physical evaluation, download the expert zoo and build the reference bank:

```bash
python -m mocap_phys_eval.prefetch
```

---

## 5. Recording Session Plan

### Target dataset size

| Quantity | Target | Why |
|---|---|---|
| Separate files | 10-12 `data*.npy` recordings | Needed for valid LOFO and enough held-out windows |
| Duration per file | 3-5 minutes | Produces roughly 180-300 windows per file |
| Total usable recording time | 40-60 minutes | Enough held-out windows to support 80 successful physical trials |
| Motion variety | At least 5 activity types | Avoids a narrow motion distribution |

### Pre-recording checklist

- [ ] All 3 uMyo sensors are detected
- [ ] The BWT901CL is paired and streaming at 200 Hz
- [ ] `thigh_quat_wxyz` is present and smooth
- [ ] Knee signal changes continuously and is not stuck
- [ ] Sensor placement is documented
- [ ] `recordings_manifest.csv` has been created

Suggested manifest columns:

```csv
filename,duration_min,activity,session,notes
```

### Recording blocks

Record each block as a separate file. Stop and save between blocks.

| Block | File | Duration | Activity | Purpose |
|---|---|---|---|---|
| 0 | `data_warmup.npy` | 1-1.5 min | quiet stand, slow flex/extend, walk-in-place | sanity check |
| 1 | `data_walk_slow.npy` | 3-4 min | slow level walking | baseline gait |
| 2 | `data_walk_normal.npy` | 3-4 min | normal walking | baseline gait |
| 3 | `data_walk_fast.npy` | 3-4 min | brisk walking | speed variation |
| 4 | `data_transitions.npy` | 3-5 min | starts, stops, turns, varying step length | transition coverage |
| 5 | `data_sit_stand.npy` | 3-4 min | repeated sit-to-stand and stand-to-sit | non-walking motion |
| 6 | `data_stairs.npy` | 3-4 min | stairs or incline if available | higher difficulty motion |
| 7 | `data_mixed1.npy` | 4-5 min | free mix | broad coverage |
| 8 | `data_mixed2.npy` | 4-5 min | different free mix | broad coverage |
| 9 | `data_mixed3.npy` | 4-5 min | transition-heavy free mix | broad coverage |
| 10 | `data_reseat.npy` | 3-4 min | reseat sensors, then mixed walking | placement robustness |
| 11 | `data_extra.npy` | 3-4 min | any underrepresented motion | fill gaps |

### Between-block QC

After every 2-3 blocks, spot-check the most recent file:

```bash
python plotdata.py data_walk_slow.npy
```

Confirm:

- [ ] Effective sample rate is close to 200 Hz
- [ ] `thigh_quat_wxyz` is smooth and normalized
- [ ] Raw EMG channels are not all-zero or clipped
- [ ] Knee label is not discontinuous or saturated
- [ ] The file contains at least 3 minutes of usable data unless it is the warmup block

If a block fails QC, re-record it immediately.

---

## 6. Build the Windowed Dataset

After all recordings are collected:

```bash
python split_to_samples.py
```

Expected output:

- `samples_dataset.npy`

What this file contains:

- `X`: `(N, 200, 43)`
- `y`: scalar label per window
- `y_seq`: full knee trajectory per window
- `file_id`: recording-file identity
- `file_names`: file-name lookup
- `start`: start timestep inside the source recording

Minimum acceptance checks:

- There are many more than 80 total windows
- At least 10 recording files contributed windows
- `thigh_mode == "quat"`
- `thigh_n_features == 4`

If `samples_dataset.npy` is missing or malformed, the evaluator cannot run the real paper protocol.

---

## 7. Train the TST

Run:

```bash
python -m emg_tst.run_experiment
```

Expected training behavior:

- Outer split is LOFO when multiple files exist
- Inner validation split holds out one additional training file when possible
- The "ALL FEATURES" run is the one used by the physical evaluator

Key outputs:

- `checkpoints/tst_<timestamp>_all/fold_XX/reg_best.pt`
- `checkpoints/tst_<timestamp>_all/fold_XX/metrics.json`
- `checkpoints/tst_<timestamp>_all/fold_XX/split_manifest.json`
- `checkpoints/tst_<timestamp>_all/cv_manifest.json`
- `checkpoints/tst_<timestamp>/ablation_summary.json`

Why the split manifests matter:

- The evaluator uses them to recover the exact held-out LOFO pool
- Each held-out window is paired with the correct outer-fold checkpoint
- This is necessary to match the paper methodology exactly

Acceptance checks:

- `test_rmse` and `test_seq_rmse` are present for every fold
- Every fold directory has `reg_best.pt`
- Every fold directory has `split_manifest.json`
- The run root has `cv_manifest.json`

If your existing checkpoints were created before split manifests were added, rerun training once. The evaluator can reconstruct some cases, but the manifest-backed path is the exact one.

Optional but recommended:

```bash
python -m emg_tst.learning_curve
```

Use the learning curve to decide whether more recording time is needed to approach the RMSE target.

---

## 8. Run the Paper-Protocol Physical Evaluation

Run:

```bash
python -m mocap_phys_eval
```

When `samples_dataset.npy` and a trained `_all` LOFO run are present, the evaluator follows the paper protocol:

1. Loads the held-out LOFO test pool only
2. Uses the latest `_all` training run
3. Maps each held-out window to its matching outer-fold checkpoint
4. Samples windows uniformly without replacement with fixed seed `42`
5. Motion-matches each window to the MoCapAct bank
6. Simulates:
   - `REF`: no override
   - `PRED`: right knee forced to the TST prediction
   - `BAD`: auxiliary diagnostic only
7. If a sampled window fails motion matching or simulation, discards it
8. Continues to the next seeded held-out window until `80` successful trials are retained

Important:

- If `samples_dataset.npy` is missing, the evaluator falls back to demo BVH mode
- Demo mode is not part of the paper methodology
- If there are not enough held-out windows to produce 80 successful trials, the run is incomplete

Key outputs:

- `artifacts/phys_eval_v2/runs/<run_id>/summary.json`
- `artifacts/phys_eval_v2/runs/<run_id>/evals/<idx>_<query_id>/summary.json`
- `artifacts/phys_eval_v2/runs/<run_id>/failures/<attempt>_<query_id>.json`
- plots and replay artifacts inside each `evals/...` folder

Acceptance checks for a paper-valid run:

- `summary.json.mode == "rigtest_paper_lofo"`
- `summary.json.protocol.paper_exact == true`
- `summary.json.protocol.successful_trials == 80`
- `summary.json.protocol.failed_trials` is allowed to be nonzero
- Each retained eval summary has:
  - non-null `model.pred_vs_gt_knee_flex_rmse_deg`
  - `sim.pred.balance_risk_auc`
  - `match.rmse_knee_deg`
  - `match.rms_thigh_ori_err_deg`

If `model.pred_vs_gt_knee_flex_rmse_deg` is null, that run was effectively oracle mode and should not be used for the final paper statistic.

---

## 9. Run the Statistical Analysis

After the physical evaluation finishes, run:

```bash
python -m analysis.correlation --run-dir artifacts/phys_eval_v2/runs/<run_id>
```

This computes the paper's partial Spearman analysis with:

- `x_i`: `model.pred_vs_gt_knee_flex_rmse_deg`
- `y_i`: `sim.pred.balance_risk_auc`
- `k_i`: `match.rmse_knee_deg`
- `h_i`: `match.rms_thigh_ori_err_deg`

Method:

1. Rank-transform all four variables
2. Residualize ranked `x` and ranked `y` against ranked `[k, h]`
3. Compute Pearson correlation between residuals
4. Report the partial Spearman estimate and the corresponding `t` and `p`

Expected outputs:

- `artifacts/phys_eval_v2/runs/<run_id>/analysis/partial_spearman_summary.json`
- `artifacts/phys_eval_v2/runs/<run_id>/analysis/partial_spearman_trials.csv`
- `artifacts/phys_eval_v2/runs/<run_id>/analysis/partial_spearman_scatter.png`

For the intended paper design:

- `N = 80`
- `q = 2` controls
- `df = N - 2 - q = 76`

---

## 10. Exact End-to-End Command Order

Use this order once the sensors and recording rig are ready:

```powershell
$env:MOCAPACT_MODELS_DIR = "D:\mocapact_models"
$env:MOCAP_PHYS_EVAL_ARTIFACTS_DIR = "D:\phys_eval_v2_artifacts"
$env:HF_TOKEN = "hf_..."
```

```bash
pip install -r requirements_tst.txt
python -m mocap_phys_eval.prefetch
python split_to_samples.py
python -m emg_tst.run_experiment
python -m emg_tst.learning_curve
python -m mocap_phys_eval
python -m analysis.correlation --run-dir artifacts/phys_eval_v2/runs/<run_id>
```

If you need the latest run id:

```powershell
Get-ChildItem artifacts\phys_eval_v2\runs -Directory | Sort-Object Name | Select-Object -Last 1
```

If you redirected artifacts with `MOCAP_PHYS_EVAL_ARTIFACTS_DIR`, use that path instead.

---

## 11. Final Deliverables Checklist

You are done when all of the following exist and are valid:

- [ ] `samples_dataset.npy`
- [ ] latest `_all` training run with `reg_best.pt` for every fold
- [ ] `split_manifest.json` in every fold directory
- [ ] `cv_manifest.json` in the `_all` run root
- [ ] one physical-eval run with `mode == "rigtest_paper_lofo"`
- [ ] that run reports `successful_trials == 80`
- [ ] retained eval summaries have non-null model RMSE
- [ ] `partial_spearman_summary.json`
- [ ] `partial_spearman_trials.csv`
- [ ] `partial_spearman_scatter.png`

---

## 12. Common Failure Modes

- Only one recording file
  - LOFO is not valid
  - Fix: record multiple separate files

- Too few held-out windows
  - The evaluator cannot retain 80 successful trials
  - Fix: collect more files or longer recordings

- Missing or old training artifacts
  - No `split_manifest.json` / `cv_manifest.json`
  - Fix: rerun `python -m emg_tst.run_experiment`

- Demo fallback triggered
  - `samples_dataset.npy` was missing or unusable
  - Fix: rebuild samples and rerun

- Oracle-style eval summaries
  - `model.pred_vs_gt_knee_flex_rmse_deg` is null
  - Fix: ensure the evaluator used real trained fold checkpoints, not demo or oracle fallback

- Label quality problems
  - RMSE target may be unreachable even with more data
  - Fix: prioritize sensor placement, QC, and stable label computation

- Narrow motion coverage
  - Model performs acceptably on walking but poorly on transitions
  - Fix: record diverse motions, especially transitions

- Sensor shift during session
  - Performance degrades on later files
  - Fix: include the reseat block and document placement changes
