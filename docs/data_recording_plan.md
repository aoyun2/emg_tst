# Data Recording Plan (EMG/IMU → Knee Angle) for `emg_tst`

This repo’s TST pipeline learns to predict your **IMU-derived knee included angle** (label) from EMG + IMU thigh orientation features, then evaluates those predictions in the MoCapAct physics pipeline.

This doc is a **practical recording protocol** plus a **sample-size plan** for reaching **< 3° RMSE** in *free motion* using the current `split_to_samples.py` + `emg_tst/run_experiment.py` training setup.

## What “< 3° RMSE” Means Here

- The training/eval target is **RMSE in degrees** against your saved label `knee_included_deg`.
- The label is computed online in `uMyo_python_tools/rigtest.py` from your sensors. That means:
  - Your model can only be as accurate as the label (sensor noise / drift / mis-calibration sets a floor).
  - “Sub‑3° RMSE” is **with respect to your label**, not optical MoCap ground truth.

Two metrics show up in training artifacts:

- `test_rmse`: RMSE of the **last timestep** in each 1.0s window (proxy for “real-time” prediction).
- `test_seq_rmse`: RMSE over the **entire 1.0s trajectory**.

If you care about smooth control over the whole window, track `test_seq_rmse` too.

## How the Pipeline Uses Your Recordings (Why File Structure Matters)

- `rigtest.py` writes `data*.npy` recordings with timestamps, EMG, and `thigh_quat_wxyz`.
- `split_to_samples.py`:
  - resamples each recording to **200 Hz** using timestamps
  - slices **non-overlapping 1.0s windows** (`WINDOW=200`)
  - assigns each original file a `file_id`
- `emg_tst/run_experiment.py` uses **Leave-One-File-Out (LOFO)** outer folds when multiple files exist.

Practical implication:

- A “recording file” is the unit of generalization for the default evaluation.
- You want **multiple independent files** (ideally across sessions/days) so test performance is meaningful.

## Minimum Viable Dataset (MVD)

This is the smallest dataset that tends to produce *useful* learning-curve estimates and stable LOFO metrics.

- **# files**: at least **8–10** separate `data*.npy` recordings (so LOFO has multiple folds).
- **duration per file**: **3–6 minutes** each (enough windows per fold without making files unwieldy).
- **total time**: **30–60 minutes** of usable motion.

Why these numbers:

- With non-overlapping 1.0s windows, **1 minute ≈ 60 windows**.
- 30 minutes ≈ 1800 windows; 60 minutes ≈ 3600 windows.

This is not a guarantee of <3° RMSE; it’s a pragmatic starting point so you can measure what *your* setup needs.

## Estimating “How Much Data for < 3° RMSE?” (Use the Built-in Learning Curve)

Once you have a first dataset:

1. Build windows:

   ```bash
   python split_to_samples.py
   ```

2. Run the learning curve:

   ```bash
   python -m emg_tst.learning_curve
   ```

This will train multiple times at increasing train-set sizes (in recorded minutes) and print a line like:

- `Estimated data to reach <= 3.0 deg (mean): ~X.XX hours ...`

If it prints “Target not reached”, you need more data **and/or** higher-quality labels/features.

## Recommended “Research-Grade” Dataset (Free Motion)

If your goal is robust free-motion performance (and you expect day-to-day variability), plan for:

- **2–4 sessions** on different days (electrode shift + sensor placement variability matters).
- **~30–60 minutes per session** of *usable* data.
- Total: **~1–4 hours** per subject as a reasonable planning range.

Then rely on `learning_curve.py` to turn that into an evidence-based estimate for *your* RMSE target.

## Recording Protocol (Per Session)

### 0) Before you record (setup + checks)

- Confirm all sensors are detected (`rigtest.py` requires 3 uMyo sensors).
- Confirm IMU is configured to **200 Hz** (rigtest sends the BWT901CL command).
- Confirm the thigh quaternion stream is present (the training pipeline requires `thigh_quat_wxyz`).

### 1) Warm-up / calibration block (60–90s)

Record one short block at the beginning of every session:

- 10–15s quiet stand (baseline)
- 10–15s slow knee flex/extend (large amplitude)
- 10–15s moderate knee flex/extend
- 10–15s gentle walking-in-place or short straight steps (if safe)

This gives you a consistent “sanity check” segment across sessions.

### 2) Free-motion blocks (repeat 5–10×)

Record several **short, separate files** rather than one long file.

Suggested block menu (mix as needed):

- level walking at multiple cadences (slow/normal/fast)
- starts/stops, turns, step length changes
- sit-to-stand / stand-to-sit
- stairs up/down (if available)
- ramps / inclines (if available)
- light perturbations or “unexpected” transitions (only if safe)

Aim for **coverage + transitions**, not perfect repetitions.

### 3) Rest + re-seat checks

If you expect electrode or strap shift during a session, treat it as a *feature*, not a bug:

- take a short break
- re-seat straps the way you expect in real use
- record again

Just keep notes so you can later correlate performance with placement changes.

## Data Management / Naming

`split_to_samples.py` looks for files matching `data*.npy`. You can still encode metadata in filenames as long as they start with `data`.

Example naming scheme:

- `data_s01_2026-03-06_session1_block03_freewalk.npy`

Recommended: keep a simple `recordings_manifest.csv` (subject/session/notes) alongside your files.

## Quality Control Checklist (Do This After Every Session)

1. Plot a few recordings:

   ```bash
   python plotdata.py data0.npy
   ```

2. Quick checks:
   - `effective_hz` near ~200 Hz after resampling (within a few percent)
   - knee signal isn’t “stuck”, saturating, or obviously discontinuous
   - `thigh_quat_wxyz` exists and looks normalized / smooth (no NaNs)
   - raw EMG exists (`raw_emg_sensor*`) and isn’t all zeros/clipped

3. Rebuild windows:

   ```bash
   python split_to_samples.py
   ```

If QC fails, it’s usually better to re-record that block than to “hope training will handle it”.

## Common Failure Modes (And How to Avoid Them)

- **Only one recording file** → LOFO degenerates; you’ll get optimistic leakage. Record multiple files.
- **Drift / label noise** → your RMSE target might be unattainable. Prioritize label quality.
- **Narrow motion distribution** → good RMSE in-lab, poor in free motion. Record transitions + variability.
- **Session-to-session placement shift** → model seems “great” same-day, fails next day. Train with multi-day data.
