# emg_tst: EMG TST + MoCapAct Physical Evaluation

This repo has two parts:

1. `emg_tst/`: a Time Series Transformer (TST) for EMG/IMU to knee angle inference.
2. `mocap_phys_eval/`: a physical evaluation pipeline that motion-matches a single TST window against MoCapAct's **per-snippet expert set** (~2589 snippets), then runs **real MuJoCo physics** using the matched snippet's **expert policy** (not the distilled multi-clip policy).

The goal is to evaluate a predicted knee trajectory physically (dynamic feasibility and balance), not just numerically.

## Quick Start (No Trained Model Yet)

Install deps:

```bash
pip install -r requirements_tst.txt
```

Run the evaluation (one default configuration, no runtime flags):

```bash
python -m mocap_phys_eval
```

Replay the latest run anytime:

```bash
python -m mocap_phys_eval.replay
```

If you only want to download/extract the expert zoo (and build the reference bank) without running the simulation:

```bash
python -m mocap_phys_eval.prefetch
```

## Disk + Download Setup (Important)

The full MoCapAct expert zoo is large:

- ~134 GB compressed download (8 tarballs)
- ~150+ GB extracted
- Plan on >=200 GB free (more if your drive is slow or you keep tarballs)

### Where the experts are stored

By default, models are stored under `./mocapact_models/`:

- `<MODELS_DIR>/_downloads/`: tarballs + extraction markers
- `<MODELS_DIR>/experts/`: extracted expert policies (this is the big one)

To store them on another drive (recommended), set:

```powershell
$env:MOCAPACT_MODELS_DIR = "D:\\mocapact_models"
```

Tip: avoid putting `<MODELS_DIR>` under OneDrive.

To make it permanent:

```powershell
setx MOCAPACT_MODELS_DIR "D:\\mocapact_models"
# Note: `setx` does not affect the current terminal session.
```

If your system drive is very full, you can also move outputs (runs, plots, replays, downloaded query BVHs) off the repo:

```powershell
$env:MOCAP_PHYS_EVAL_ARTIFACTS_DIR = "D:\\phys_eval_v2_artifacts"
```

### Hugging Face token (avoid anonymous throttling)

If your download speed is unexpectedly low, set a Hugging Face token:

```powershell
$env:HF_TOKEN = "hf_..."
```

The downloader will use it automatically (it is never printed).

Alternative (stores token in the Hugging Face cache instead of an env var):

```powershell
hf auth login
```

### Faster downloads (optional)

Default behavior:

- Uses a builtin resumable downloader (creates `*.tar.gz.part` and resumes on rerun).
- If `aria2c` is installed and on PATH, it will be used automatically (resume + parallel).

You can force a backend:

```powershell
$env:MOCAPACT_DOWNLOAD_BACKEND = "aria2"      # requires aria2c
# or
$env:MOCAPACT_DOWNLOAD_BACKEND = "hf_transfer" # fastest single-shot; not a safe resume if interrupted
# or
$env:MOCAPACT_DOWNLOAD_BACKEND = "urllib"     # builtin resumable
```

If you want to keep the downloaded tarballs after extraction:

```powershell
$env:MOCAPACT_KEEP_TARBALLS = "1"
```

## What The Pipeline Does (Exactly)

`python -m mocap_phys_eval` runs end-to-end:

1. Downloads a **real BVH mocap clip** (non-synthetic) from a built-in demo list (it cycles between runs).
2. Extracts right-leg angles from the BVH:
   - `thigh_pitch_deg`: right thigh pitch (sagittal plane, signed)
   - `thigh_quat_wxyz`: right thigh orientation quaternion (root-relative, `wxyz`)
   - `knee_included_deg`: right knee included angle (`180 = straight`)
3. Converts to MoCapAct joint convention:
   - Your rig convention: included angle (`0 = fully bent`, `180 = straight`)
   - MoCapAct knee joint: flexion (`0 = straight`)
   - Conversion: `knee_flex_deg = 180 - knee_included_deg`
4. Resamples to the TST windowing rate and takes **one** fixed-length window:
   - `window_hz = 200 Hz`
   - `window_n = 200 samples` (1 second)
   - No aggregation across windows (matches the intended TST train/eval shape)
5. Ensures the full MoCapAct **expert** model zoo is present (downloads/extracts if missing).
6. Loads (or builds) an expert-aligned reference bank over dm_control's CMU2020 fitted motion data, indexed by **MoCapAct snippet boundaries** (so we can always map a motion match to an expert).
7. Motion-matches the one query window against the reference bank:
   - coarse stage uses quaternion-derived thigh step angles + knee derivatives (offset-invariant)
   - refine stage solves constant per-joint offsets and a constant thigh-quaternion alignment, then reports:
     - knee RMSE (deg)
     - thigh 3D orientation RMS error (deg), using quaternion geodesic distance (not a single-angle RMSE)
8. Runs **three** physics rollouts using the matched snippet's **expert policy**:
   - `REF`: no override, expert tracks the original reference
   - `GOOD`: patch the right hip pitch + right knee flexion reference to match the query window (knee is physically forced; hip is still controlled by the policy while tracking the patched reference)
   - `BAD`: same, but knee is perturbed by a smooth deterministic error (~20 deg RMSE vs `GOOD`) to demonstrate failure modes without spiky 0<->180 artifacts

Override details:

- Only two DoFs are overridden (right hip pitch and right knee flexion). The policy still controls all other joints normally.
- The override is enforced in two ways:
  1. Patch the reference targets the policy observes (and recompute reference kinematics).
  2. Force the prosthetic knee actuator each step so the policy cannot directly command it. (Hip pitch is *not* forced; it is controlled by the policy and should track the patched reference.)
- Each run writes `control_override` diagnostics into `summary.json` to confirm the knee control is being overridden (and to measure how closely the policy tracks the patched hip reference).

## Visualization

Each run records a `REF | GOOD | BAD` compare rollout and opens an interactive viewer with:

- 3 panels in one window: `REF`, `GOOD`, `BAD`
- a light-grey translucent "ghost" in each panel
- the overridden ("prosthetic") right leg highlighted in magenta in `GOOD` and `BAD`

Per-panel camera controls:

- mouse right-drag: rotate
- mouse left-drag: pan
- mouse wheel: zoom
- keys `WASD` or arrow keys: translate, `Q/E` (or `PgUp/PgDn`) up/down, `Shift` = faster
- keys `1/2/3`: select panel, `r`: reset selected camera

## Outputs

Outputs are written under:

- `artifacts/phys_eval_v2/runs/<run_id>/summary.json`
- `artifacts/phys_eval_v2/runs/<run_id>/plots/`
- `artifacts/phys_eval_v2/runs/<run_id>/replay/compare.npz` and `compare.gif`

Convenience pointers:

- `artifacts/phys_eval_v2/latest_compare.npz`
- `artifacts/phys_eval_v2/latest_compare.gif`
- `artifacts/phys_eval_v2/latest_motion_match.png`

Plots include:

- motion-match plot (query-aligned vs reference: thigh pitch + knee + thigh 3D orientation error)
- simulation joint plots (targets vs actuals, `REF/GOOD/BAD`)
- balance plots (stability metrics)
