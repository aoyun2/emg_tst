# emg_tst: EMG TST + MoCapAct Physical Evaluation

This repo has two parts:

1. `emg_tst/`: a Time Series Transformer (TST) for EMG/IMU -> knee angle inference.
2. `mocap_evaluation/`: an evaluation pipeline that takes a thigh+knee angle trajectory, motion-matches it to the CMU mocap library used by MoCapAct, and then runs a MuJoCo humanoid physics rollout while overriding the thigh+knee targets.

The goal is to evaluate a predicted knee trajectory physically (is it dynamically feasible), not just numerically.

## What The Pipeline Does

`python -m mocap_evaluation.pipeline` runs this end-to-end:

1. Gets a query trajectory:
   - If you pass `--input path/to/query.csv`, it loads that.
   - Otherwise it downloads an OpenSim sample zip from AddBiomechanics and converts the included `.trc` marker file to `artifacts/opensim_query.csv`.
2. Interprets knee angles using your rig convention and converts for MoCapAct:
   - Input `knee_angle` is treated as the included angle in degrees (`0` = fully flexed, `180` = straight).
   - The MoCapAct knee joint coordinate is knee flexion in degrees (`0` = straight).
   - The pipeline converts: `knee_flex = 180 - knee_angle`.
3. Builds/loads a motion-matching index over dm_control's fitted CMU2020 clips (the mocap source used by MoCapAct), stored at `artifacts/cmu_clip_index_left.npz`.
4. Motion-matches the query batch to a window inside the best CMU clip.
5. Runs a MoCapAct multi-clip policy in the CMU humanoid tracking task and overrides:
   - thigh actuator: `walker/lfemurrx`
   - knee actuator: `walker/ltibiarx`
6. Writes `artifacts/mocapact_eval.json` (match + physics metrics).

The first batch opens a dm_control viewer window unless you set `MOCAP_EVAL_NO_VIEWER=1`.

## Quick Start (No Trained Model Yet)

1) Extract the multi-clip policy checkpoint (small download):

```bash
tar -xvf mocapact_models/multiclip_policy.tar.gz -C mocapact_models
```

2) Run the evaluation (downloads OpenSim sample automatically):

```bash
python -m mocap_evaluation.pipeline
```

Headless (no viewer) in PowerShell:

```bash
$env:MOCAP_EVAL_NO_VIEWER=1; python -m mocap_evaluation.pipeline
```

## Using Your Own Data

Provide a CSV with headers:

- `thigh_angle` (degrees)
- `knee_angle` (degrees, included angle: `0` = fully flexed, `180` = straight)
- optional `time_s` (seconds). If present, the pipeline infers the sample rate from it. If absent, it assumes 200 Hz (matches `rigtest.py`).

Run:

```bash
python -m mocap_evaluation.pipeline --input path/to/query.csv
```

You can also pass a `rigtest.py` `.npy` dump (assumed 200 Hz). It must contain:

- `thigh_angle` (degrees)
- `knee_pred` (degrees, included angle: `0` = fully flexed, `180` = straight)

## Output

The pipeline writes a JSON summary to `artifacts/mocapact_eval.json` with:

- the query metadata (inferred sample rate, windowing)
- the best motion-match (clip id, match start/end)
- physics rollout metrics (reward, early termination, RMSE for the overridden joints)

## Install Notes

This pipeline depends on the MoCapAct stack:

- `dm_control` (MuJoCo + CMU humanoid tracking task)
- `mocapact` (env wrappers + policy code)
- `pytorch_lightning` (loads the multi-clip checkpoint)

Install base deps:

```bash
pip install -r requirements_tst.txt
```

Then install MoCapAct runtime deps:

```bash
pip install mocapact dm_control stable-baselines3
```

Notes:
- dm_control downloads the CMU2020 fitted mocap file (`cmu_2020_*.h5`) on first use.
- The repo includes runtime compatibility shims for NumPy 2.x and SB3 2.x so the multi-clip policy can run with `mocapact==0.1`.
