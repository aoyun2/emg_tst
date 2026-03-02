# emg_tst + MoCapAct Evaluation Pipeline

This repository contains:
1. A Time Series Transformer (TST) stack for EMG-to-knee-angle inference (`emg_tst/`).
2. A newly implemented **from-scratch evaluation pipeline** that physically tests predicted knee angles in a virtual simulation while motion-matching against the **MoCapAct** snippet dataset (`mocap_evaluation/`).

## What the new pipeline does

Given continuous batch data (currently from OpenSim samples, later from your real `rigtest.py` recordings):

1. **Batching**: Convert long trajectories into continuous test batches.
2. **MoCapAct retrieval**: Download and index snippet files from `microsoft/mocapact-data`.
3. **Motion matching**:
   - Build per-frame features: `[thigh, knee, d_thigh, d_knee]`.
   - Run DTW-based matching against windows from each snippet.
   - Keep top-K best matches.
4. **Angle override simulation**:
   - For each batch, run a MuJoCo two-link physical leg simulation.
   - Override target thigh angle with batch thigh angle.
   - Override knee target with model-predicted knee angle.
5. **Outputs**:
   - JSON summary per batch with best snippet match and knee tracking RMSE.

---

## New modules

- `mocap_evaluation/mocapact_dataset.py`
  - Lists and downloads snippet files from HF.
  - Validates expected snippet count (`2589`).
  - Loads thigh/knee angle trajectories from `.h5/.hdf5/.npz`.

- `mocap_evaluation/query_data.py`
  - Loads OpenSim CSV or rigtest `.npy` inputs.
  - Builds continuous overlapping/non-overlapping batches.

- `mocap_evaluation/motion_matching.py`
  - Feature builder + DTW matching with Sakoe-Chiba band.

- `mocap_evaluation/simulation.py`
  - MuJoCo physical leg model.
  - Runs control loop with thigh/knee angle override.

- `mocap_evaluation/pipeline.py`
  - End-to-end CLI: load query data → motion match → simulate → report JSON.

---

## Installation

```bash
pip install -r requirements_tst.txt
```

---

## Step 1 — Download MoCapAct snippets manually

Use `huggingface-cli` to download the full dataset snapshot to your local folder:

```bash
huggingface-cli download microsoft/mocapact-data \
  --repo-type dataset \
  --local-dir mocap_data/mocapact
```

Then validate local count (should be **2589** snippet files):

```bash
python - <<'PY'
from mocap_evaluation.mocapact_dataset import validate_snippet_count
count, ok = validate_snippet_count("mocap_data/mocapact")
print({"count": count, "ok": ok})
raise SystemExit(0 if ok else 2)
PY
```

---

## Step 2 — Prepare input trajectories

### Option A: OpenSim sample CSV (current bootstrapping)
Provide CSV with headers:
- `thigh_angle`
- `knee_angle`

### Option B: rigtest `.npy`
Provide dictionary keys:
- `thigh_angle`
- `knee_pred` (or edit key in code)

---

## Step 3 — Run end-to-end evaluation

```bash
python -m mocap_evaluation.pipeline \
  --mocapact-dir mocap_data/mocapact \
  --source opensim \
  --input data/opensim_sample.csv \
  --batch-size 400 \
  --stride 200 \
  --sample-hz 200 \
  --top-k 3 \
  --out artifacts/mocapact_eval.json
```

Example for rigtest:

```bash
python -m mocap_evaluation.pipeline \
  --mocapact-dir mocap_data/mocapact \
  --source rigtest \
  --input data/data0.npy \
  --out artifacts/mocapact_eval_rigtest.json
```

---

## Output JSON schema

`artifacts/mocapact_eval.json`:

- `expected_snippets`: expected count (`2589`)
- `loaded_snippets`: number parseable locally
- `n_batches`: number of generated query batches
- `results[]` per batch:
  - `batch_id`
  - `best_match.snippet_id`
  - `best_match.score` (lower is better)
  - `best_match.start_idx`, `best_match.end_idx`
  - `knee_sim_rmse_deg`

---

## Notes on dataset key variability

MoCapAct/HDF5 variants may store thigh/knee under different key paths. The loader includes fallback key candidates and can be extended quickly in `load_snippet_angles()`.


## Troubleshooting

- **`RuntimeError: No snippet files found in MoCapAct HF dataset.`**
  - Update to the latest code in this repo (the selector now accepts `.h5/.hdf5/.npz/.npy` files and no longer requires filenames to contain `snippet`).
  - Re-run:

```bash
huggingface-cli download microsoft/mocapact-data --repo-type dataset --local-dir mocap_data/mocapact
```

  - If this still fails, verify files actually exist under `mocap_data/mocapact` and rerun `validate_snippet_count(...)`.

---

## Existing TST training code

Your model training/data tooling remains unchanged in `emg_tst/`, `split_to_samples.py`, and `uMyo_python_tools/rigtest.py`.

