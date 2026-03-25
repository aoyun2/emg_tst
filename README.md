# emg_tst

Real-time EMG/IMU knee-angle prediction (TST) evaluated in **MoCapAct** physics via **per-snippet expert policies** (N=2589).

The evaluation pipeline:

- motion-matches each fixed-length TST window to the best MoCapAct snippet
- runs **real MuJoCo** simulation using the matched snippet's **expert policy**
- forces the **right knee** (prosthetic knee) to follow either:
  - `PRED`: the TST model's predicted knee angle (or oracle until your model is trained)
  - `BAD`: a smooth ~20 deg RMSE perturbation (demo stand-in)
- reports motion-matching error separately from model error

**Example output (3-panel compare: REF | PRED | BAD)**

![REF vs PRED vs BAD compare replay](docs/media/compare.gif)

## Repo Structure

- `emg_tst/`: Time Series Transformer (TST) that maps EMG + IMU features to a knee angle (your rig convention: included angle).
- `mocap_phys_eval/`: physical evaluation (motion matching + MuJoCo sim + viewer + plots).

## Install

```bash
pip install -r requirements_tst.txt
```

## Pipeline Overview

```mermaid
flowchart LR
  A["rigtest.py recording\nEMG + IMU quat + knee included"] --> B["split_to_samples.py\n1.0s windows @ 200 Hz"]
  B --> C["TST model (optional)\nreg_best.pt"]
  B --> D["mocap_phys_eval\nquery window"]
  C --> D
  D --> E["Resample to MoCapAct rate\n~33.33 Hz"]
  E --> F["Motion match over 2589 expert snippets\ndquat + knee derivative"]
  F --> G["Matched snippet expert policy\nSB3 PPO"]
  G --> H1["REF sim\nno override"]
  G --> H2["PRED sim\nright knee forced"]
  G --> H3["BAD sim\nright knee forced"]
  H1 --> I["Artifacts\nNPZ + GIF + plots + summary.json"]
  H2 --> I
  H3 --> I
```

## Disk + Download Setup (Required)

The full MoCapAct expert zoo is large:

- 8 tarballs (Hugging Face)
- ~150+ GB extracted
- plan on >=200 GB free

Set the storage location (recommended: a large non-OneDrive drive):

```powershell
$env:MOCAPACT_MODELS_DIR = "D:\\mocapact_models"
```

Permanent (new terminals only):

```powershell
setx MOCAPACT_MODELS_DIR "D:\\mocapact_models"
```

Optional: move artifacts off the repo:

```powershell
$env:MOCAP_PHYS_EVAL_ARTIFACTS_DIR = "D:\\phys_eval_v2_artifacts"
```

### Hugging Face token (avoid anonymous throttling)

```powershell
$env:HF_TOKEN = "hf_..."
```

This is never printed; it is only used in request headers.

### Resume behavior

Downloads are resumable:

- the builtin downloader writes `*.tar.gz.part`
- rerunning `python -m mocap_phys_eval.prefetch` resumes from the partial file
- extraction completion is tracked with `<MODELS_DIR>/_downloads/experts_X.extracted`

### Faster downloads (optional)

You can select a backend:

```powershell
$env:MOCAPACT_DOWNLOAD_BACKEND = "urllib"      # resumable (default if aria2c not installed)
$env:MOCAPACT_DOWNLOAD_BACKEND = "hf_transfer" # fast; not a safe resume across interruptions
```

If you want to keep tarballs after extraction:

```powershell
$env:MOCAPACT_KEEP_TARBALLS = "1"
```

## One-Time Prefetch (Recommended)

Download/extract the expert zoo and build the reference bank:

```bash
python -m mocap_phys_eval.prefetch
```

## Run The Physical Evaluation

Run the evaluator (no CLI flags):

```bash
python -m mocap_phys_eval
```

Replay the latest run:

```bash
python -m mocap_phys_eval.replay
```

## Sample Plots (From Demo Run)

Motion match quality (aligned query vs matched expert snippet):

![Motion match plot](docs/media/motion_match.png)

Quaternion alignment error (geodesic angle per step):

![Thigh quaternion match](docs/media/thigh_quat_match.png)

Simulation knee tracking (target vs actual):

![Simulation knee plot](docs/media/simulation_knee.png)

Balance signals + heuristic risk trace:

![Simulation balance plot](docs/media/simulation_balance.png)

## Using Real rigtest.py Data (The Real Pipeline)

### Data recording plan + “how much data for <3° RMSE?”

See `docs/data_recording_plan.md` for a full recording protocol (free-motion blocks, QC checklist, and how to use `python -m emg_tst.learning_curve` to estimate how many recorded hours you need to reach <3° RMSE on held-out files).

1. Record `data*.npy` using `uMyo_python_tools/rigtest.py`.
   - Required: each recording must include `thigh_quat_wxyz` (wxyz quaternion).
2. Build windowed samples:

```bash
python split_to_samples.py
```

This writes `samples_dataset.npy` with non-overlapping 1.0s windows at 200 Hz (window=200). Recordings are resampled onto an exact 200 Hz grid using rigtest timestamps to remove timing jitter (so `WINDOW=200` really means 1.0s).

Note on "window size 32" in the TST pipeline: `emg_tst/data.py` uses `RAW_WINDOW=32` as a **causal rolling raw-EMG window** (in raw samples) to compute per-timestep EMG features. Every IMU timestep gets a feature vector; at the start of a recording we left-pad the raw stream so the first few timesteps still have a full window. This does not change the TST sample length. The TST sample/window length is `WINDOW=200` timesteps (1.0s at 200 Hz).

3. Train the TST:

```bash
python -m emg_tst.run_experiment
```

This writes checkpoints under `checkpoints/**/reg_best.pt`.

Within each outer fold, the model selects the best checkpoint using an internal **train/val split** (preferably holding out an entire recording file), and the held-out test file is evaluated once at the end. This avoids picking epochs directly on the test set.

The latest `*_all/` training run also writes per-fold split manifests (`fold_XX/split_manifest.json` and `cv_manifest.json`) so the physical evaluator can recover the exact held-out LOFO pool used by the paper.

Optional (recommended while you are still collecting data): generate a learning-curve report showing how RMSE improves as you add more recorded minutes/hours:

```bash
python -m emg_tst.learning_curve
```

This writes a self-contained report under `artifacts/learning_curve/**/`:

- `learning_curve.png`
- `summary.csv` (mean test RMSE across outer folds, with 95% CI)
- `results.json` (all per-fold raw metrics)

By default it uses a subset of outer folds for runtime (`MAX_OUTER_FOLDS=8` in [emg_tst/learning_curve.py](emg_tst/learning_curve.py)).

4. Run the physical evaluator:
   - With real rigtest data + trained LOFO folds present, `python -m mocap_phys_eval` follows the paper protocol exactly.
   - With no rigtest samples present, it falls back to a small demo BVH sanity check.

5. Run the statistical analysis:

```bash
python -m analysis.correlation --run-dir artifacts/phys_eval_v2/runs/<run_id>
```

This computes the paper's partial Spearman correlation using:

- predictor: `model.pred_vs_gt_knee_flex_rmse_deg`
- outcome: `sim.pred.balance_risk_auc`
- controls: `match.rmse_knee_deg`, `match.rms_thigh_ori_err_deg`

## Angle Conventions

- Your rig label is the **absolute included knee angle**: `0 = fully bent`, `180 = straight`.
- MoCapAct's knee joint is **flexion**: `0 = straight`.
- Conversion used everywhere in `mocap_phys_eval`: `knee_flex_deg = 180 - knee_included_deg`.

## What `python -m mocap_phys_eval` Does

With real rigtest data and trained LOFO folds, the evaluator follows the paper protocol:

1. **Query window source**
   - Loads the full held-out pool from `samples_dataset.npy`.
   - Uses the latest `*_all/` training run and matches each held-out window to the correct outer-fold checkpoint.
   - Samples windows uniformly without replacement using fixed seed `42`.
   - If a sampled window fails motion matching or simulation, it is discarded and replaced by the next window in the same seeded order until `80` successful trials are retained.
   - Demo-only fallback: if `samples_dataset.npy` does not exist, the evaluator downloads a real **non-CMU** BVH and runs a small sanity-check batch.

2. **Resample**
   - Query windows are recorded at 200 Hz but MoCapAct runs at ~33.33 Hz (control timestep ~0.03s), so we resample to the simulator rate.

3. **Motion match (full expert bank)**
   - Matches against all N=2589 expert snippets.
   - Uses thigh orientation quaternion + knee flexion derivatives (offset-invariant coarse stage), then refines for the top candidates:
     - constant thigh quaternion alignment (wxyz, geodesic error)
     - constant knee sign + offset (deg)
   - Reports motion-matching error:
     - `rmse_knee_deg`
     - `rms_thigh_ori_err_deg` (RMS quaternion geodesic error)

4. **MuJoCo simulation (expert policy)**
   - `REF`: matched expert policy runs normally (no overrides).
   - `PRED`: same policy, but the **right knee actuator** is forced each step to the fold-matched TST prediction.
   - `BAD` is still recorded as an auxiliary diagnostic trace, but it is **not** part of the paper analysis.

Override rule (important):

- The RL policy controls **all other actuators** normally.
- In `PRED` and `BAD`, the RL policy **cannot directly control the right knee**, because the knee actuator command is overwritten each step.
- We also overwrite the knee actuator's internal activation state (MuJoCo filter) so the forced knee command applies on the current step (not with a 1-step lag).
- `summary.json` includes diagnostics (`ctrl_override_diag`) to confirm the applied knee control matches the forced target.

## Why Is Motion Matching Fast?

Even though we match over 2589 snippets, it's fast because:

- each query is only **1.0s** and is resampled to ~33 Hz (so ~34 frames)
- coarse matching uses derivative features (`dquat` + `d knee`) and `np.convolve`-based sliding SSE in NumPy (C-accelerated)
- only `top_k` candidates are refined; only the final best match loads an expert policy for simulation

## Prosthetic Foot / Ankle (Not Implemented Yet)

The current evaluator keeps the CMU humanoid morphology intact and only overrides the **right knee**.

This is intentional: MoCapAct's expert zoo (N=2589) is trained for the original model. If you remove/lock/change the right ankle or foot geometry/joints, the expert policies are no longer "experts" for that modified body and `REF` will often fail. Doing this rigorously requires retraining experts (or distilling a new multi-clip policy) on the modified morphology.

If/when you want to emulate a passive ankle/foot (no ankle actuation) without changing geometry, the right-foot actuators in the model are:

- `walker/rfootrx`
- `walker/rfootrz`
- `walker/rtoesrx`

5. **Stability heuristic**
   - Outputs a per-step `predicted_fall_risk_trace_*`, scalar `predicted_fall_risk_*`, and `balance_risk_auc`.
   - Uses uprightness + XCoM support margin (+ tilt-rate + smoothing), with no root-height heuristics.
   - `balance_loss_step_*` is the first timestep the heuristic considers the walker unstable.
   - Also records MoCapAct tracking-task signals in `compare.npz` (`termination_error_*`, `reward_*`, `termination_error_threshold`) so you can experiment with reward/critic-based “risk” proxies.

## Visualization

Each window records a compare replay and opens an interactive viewer with 3 panels:

- `REF` (grey walker, no override)
- `PRED` (orange walker, right leg highlighted in magenta)
- `BAD` (blue walker, right leg highlighted in magenta)

Controls:

- Mouse: RMB drag rotate, LMB drag pan, wheel zoom
- Keys: WASD or arrows translate, Q/E (or PgUp/PgDn) up/down, Shift = faster
- Keys: 1/2/3 select panel, r reset selected camera

## Outputs

Per run:

- `artifacts/phys_eval_v2/runs/<run_id>/summary.json`
- `artifacts/phys_eval_v2/runs/<run_id>/evals/<idx>_<query_id>/summary.json`
- `artifacts/phys_eval_v2/runs/<run_id>/failures/<attempt>_<query_id>.json` for discarded trial attempts
- plots under each `evals/.../plots/`
- replay under each `evals/.../replay/compare.npz` and `compare.gif`

Per analysis run:

- `artifacts/phys_eval_v2/runs/<run_id>/analysis/partial_spearman_summary.json`
- `artifacts/phys_eval_v2/runs/<run_id>/analysis/partial_spearman_trials.csv`
- `artifacts/phys_eval_v2/runs/<run_id>/analysis/partial_spearman_scatter.png`

Convenience pointers:

- `artifacts/phys_eval_v2/latest_compare.npz`
- `artifacts/phys_eval_v2/latest_compare.gif`
- `artifacts/phys_eval_v2/latest_motion_match.png`
- `artifacts/phys_eval_v2/latest_thigh_quat_match.png`
