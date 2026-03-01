#!/usr/bin/env python3
"""EMG prosthetic gait evaluation pipeline.

Pipeline
========
  1. DATA SOURCE
       Rigtest .npy recording  →  knee angle + thigh angle

  2. MODEL PREDICTION
       Real checkpoint (--checkpoint --data)  →  predicted knee angle
       OR noisy-GT baseline for demo/sanity-check

  3. MOTION MATCHING
       DTW match of (thigh_angle, knee_angle) against MoCap Act database
       → best-matching clip ID + segment  →  saved match-quality plot

  4. SIMULATION  (per scenario)
       Kinematic replay of matched clip, right knee overridden by prediction
       → interactive MuJoCo viewer (non-blocking, no freeze)
       → heuristic metrics: CoM height, fall detection, step count, gait symmetry

  5. RESULTS
       Summary printed to console + saved as sim_results.png

Scenarios
---------
  gt   — GT knee angles from the matched clip (upper-bound baseline)
  good — model prediction (or low-noise GT) — this is your model
  bad  — high-noise GT (sanity-check worst-case)

Usage
-----
  # Auto-detect: uses first data*.npy recording found in current directory
  python run_sim.py

  # Specific rigtest file:
  python run_sim.py --data-file data0.npy

  # With a real trained checkpoint:
  python run_sim.py --checkpoint checkpoints/.../reg_best.pt --data samples_dataset.npy

  # Headless (no viewer window):
  python run_sim.py --no-gui

  # Use a smaller/faster matching database:
  python run_sim.py --subset walk_tiny

Note: to validate the pipeline without hardware, use virtual_sim_test.py instead.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

# ── CLI ────────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
ap.add_argument("--data-file", default=None,
                help="Specific rigtest .npy recording")
ap.add_argument("--data", default=None,
                help="Pre-windowed samples_dataset.npy (produced by split_to_samples.py)")
ap.add_argument("--checkpoint", default=None,
                help="Trained model checkpoint (.pt). Requires --data for real inference.")
ap.add_argument("--seconds", type=float, default=5.0,
                help="Seconds of data to use for matching and simulation (default: 5)")
ap.add_argument("--subset", default="all",
                choices=["all", "locomotion_small", "walk_tiny", "run_jump_tiny"],
                help="MoCap Act database subset (default: all = ~2589 clips)")
ap.add_argument("--prefilter-k", type=int, default=100,
                help="Number of DTW candidates after L2 pre-filter (default: 100)")
ap.add_argument("--no-gui", action="store_true",
                help="Headless mode: no viewer window, saves plots only")
ap.add_argument("--scenarios", nargs="+", choices=["gt", "good", "bad"],
                default=["gt", "good", "bad"],
                help="Which scenarios to simulate (default: all three)")
ap.add_argument("--good-noise", type=float, default=5.0,
                help="Knee noise std (°) for the 'good' noisy-GT scenario")
ap.add_argument("--bad-noise", type=float, default=25.0,
                help="Knee noise std (°) for the 'bad' noisy-GT scenario")
ap.add_argument("--device", default="cpu",
                help="PyTorch device for checkpoint inference (default: cpu)")
ap.add_argument("--out", default=None,
                help="Save results as JSON file")
args = ap.parse_args()

USE_GUI    = not args.no_gui
TARGET_FPS = 200.0        # EMG pipeline rate
WINDOW     = int(TARGET_FPS)  # 1-second windows


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Data source
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("STEP 1  Data source")
print("═" * 60)

max_frames = int(args.seconds * TARGET_FPS)

# Convenience resample
def _resample(arr, src_fps, dst_fps):
    if src_fps == dst_fps:
        return arr
    n = max(2, int(round(len(arr) * dst_fps / src_fps)))
    return np.interp(
        np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr
    ).astype(np.float32)

x_for_inference = None   # feature windows if using pre-windowed dataset

# ── Auto-detect data source ───────────────────────────────────────────────────
if not args.data_file and not args.data:
    candidates = sorted(p for p in glob.glob("data*.npy") if "samples" not in p)
    if candidates:
        args.data_file = candidates[0]
        print(f"Found recording: {args.data_file}")
    else:
        sys.exit(
            "No data*.npy recordings found in the current directory.\n"
            "Record data with rigtest.py first, or use virtual_sim_test.py "
            "to validate the pipeline without hardware."
        )

# ── Load data ─────────────────────────────────────────────────────────────────
if args.data:
    # Pre-windowed dataset (samples_dataset.npy)
    ds = np.load(args.data, allow_pickle=True).item()
    X_all   = np.asarray(ds["X"])        # (N, T, n_vars)
    y_all   = np.asarray(ds["y_seq"])    # (N, T)
    n_wins  = max(1, int(np.ceil(max_frames / X_all.shape[1])))
    X_all   = X_all[:n_wins]
    y_all   = y_all[:n_wins]
    x_for_inference = X_all
    knee_query  = np.concatenate(y_all).astype(np.float32)[:max_frames]
    thigh_query = np.concatenate([X_all[i, :, -1] for i in range(len(X_all))]).astype(np.float32)[:max_frames]
    source_label = f"samples_dataset ({n_wins} windows, {args.data})"
    print(f"Source: {args.data}  →  {len(knee_query)} frames")

else:
    # Rigtest .npy recording(s)
    from emg_tst.data import load_recording
    files = (
        [args.data_file] if args.data_file
        else sorted(p for p in glob.glob("data*.npy") if "samples" not in p)
    )
    knee_chunks  = []
    thigh_chunks = []
    for fpath in files:
        X, y, meta = load_recording(fpath)
        n_complete = (len(y) // WINDOW) * WINDOW
        if n_complete == 0:
            print(f"  {fpath}: too short ({len(y)} frames), skipping")
            continue
        knee_chunks.append(y[:n_complete].astype(np.float32))
        thigh_chunks.append(X[:n_complete, -1].astype(np.float32))
        print(f"  {fpath}: {n_complete // WINDOW} windows @ {meta['effective_hz']:.0f} Hz")
    if not knee_chunks:
        sys.exit("No usable windows found.")
    knee_query  = np.concatenate(knee_chunks)[:max_frames]
    thigh_query = np.concatenate(thigh_chunks)[:max_frames]
    source_label = f"rigtest ({len(files)} file(s))"

query_secs   = len(knee_query) / TARGET_FPS
n_windows    = len(knee_query) // WINDOW
print(f"Query: {n_windows}×1 s = {len(knee_query)} frames ({query_secs:.1f} s @ {TARGET_FPS:.0f} Hz)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Model prediction
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("STEP 2  Model prediction")
print("═" * 60)

rng = np.random.default_rng(42)

if args.checkpoint and x_for_inference is not None:
    import torch
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    from emg_tst.model import TSTRegressor
    cfg  = ckpt["model_cfg"]
    task = ckpt.get("task_cfg", {})
    model = TSTRegressor(**cfg)
    model.load_state_dict(ckpt["reg_state_dict"])
    model.eval()
    scaler_mean = np.asarray(ckpt["scaler"]["mean"], dtype=np.float32)
    scaler_std  = np.asarray(ckpt["scaler"]["std"],  dtype=np.float32)

    preds = []
    with torch.no_grad():
        for i in range(len(x_for_inference)):
            x = (x_for_inference[i] - scaler_mean) / (scaler_std + 1e-8)
            t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(args.device)
            out = model(t)  # (1, T, 1) or (1, T)
            preds.append(out.squeeze().cpu().numpy())
    knee_good_pred = np.concatenate(preds).astype(np.float32)[:len(knee_query)]
    knee_good_pred = np.clip(knee_good_pred, 0.0, 180.0)
    good_rmse = float(np.sqrt(np.mean((knee_good_pred - knee_query[:len(knee_good_pred)]) ** 2)))
    pred_label = f"model ({Path(args.checkpoint).name}, RMSE={good_rmse:.1f}°)"
    print(f"Model RMSE vs GT: {good_rmse:.2f}°")
else:
    if args.checkpoint:
        print("Note: --checkpoint needs --data for inference; using noisy-GT baseline.")
    knee_good_pred = np.clip(
        knee_query + rng.normal(0.0, args.good_noise, len(knee_query)).astype(np.float32),
        0.0, 180.0,
    )
    good_rmse  = float(np.sqrt(np.mean((knee_good_pred - knee_query) ** 2)))
    pred_label = f"noisy-GT (σ={args.good_noise}°, RMSE={good_rmse:.1f}°)"
    print(f"Good prediction RMSE: {good_rmse:.2f}°  (noise σ={args.good_noise}°)")

knee_bad_pred = np.clip(
    knee_query + rng.normal(0.0, args.bad_noise, len(knee_query)).astype(np.float32),
    0.0, 180.0,
)
bad_rmse = float(np.sqrt(np.mean((knee_bad_pred - knee_query) ** 2)))
print(f"Bad  prediction RMSE: {bad_rmse:.2f}°  (noise σ={args.bad_noise}°)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Motion matching
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("STEP 3  Motion matching")
print("═" * 60)

from mocap_evaluation.db       import load_database
from mocap_evaluation.matching import find_match, plot_match_quality, clip_info_for_start

db = load_database(subset=args.subset, use_cache=True)
db_fps = float(db["fps"])
n_clips = len(db["file_boundaries"])
db_secs = len(db["knee_right"]) / db_fps
print(f"Database: {n_clips} clips  {db_secs:.0f} s  @ {db_fps:.0f} Hz  "
      f"(subset={args.subset!r})")
print(f"Query   : {query_secs:.1f} s  @ {TARGET_FPS:.0f} Hz")
print()

best_start, dtw_dist, clip_id, matched_seg = find_match(
    knee_query, thigh_query, db,
    prefilter_k=args.prefilter_k,
)
clip_info = clip_info_for_start(best_start, len(knee_query), db)

print(f"\nMatched clip : {clip_id}")
print(f"  Offset     : {clip_info['time_offset_s']:.2f} s  "
      f"duration: {clip_info['time_duration_s']:.2f} s")
print(f"  DTW dist   : {dtw_dist:.4f}")
match_rmse = float(np.sqrt(np.mean(
    (matched_seg["knee_right"] - knee_query[:len(matched_seg["knee_right"])]) ** 2
)))
print(f"  RMSE knee  : {match_rmse:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Match quality plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("STEP 4  Match quality plot")
print("═" * 60)

plot_match_quality(
    knee_query,
    thigh_query,
    matched_seg["knee_right"],
    matched_seg["hip_right"],
    dtw_dist,
    clip_id,
    fps=TARGET_FPS,
    out_path="match_quality.png",
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Physics simulation
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("STEP 5  Physics simulation")
print("═" * 60)

from mocap_evaluation.sim import run_simulation, plot_sim_results

# Signals for each scenario (at TARGET_FPS = 200 Hz; sim.py resamples internally)
SCENARIO_DEFS = {
    "gt":   ("GT knee (matched clip)",                  matched_seg["knee_right"]),
    "good": (pred_label,                                knee_good_pred),
    "bad":  (f"Bad prediction (σ={args.bad_noise}°)",   knee_bad_pred),
}

sim_results: dict = {}
n_frames_sim = len(knee_query)

for sc_key in args.scenarios:
    sc_label, knee_sig = SCENARIO_DEFS[sc_key]
    print(f"\n{'─'*55}")
    print(f"  Scenario: {sc_key.upper()} — {sc_label}")
    print(f"{'─'*55}")

    sc_gui = USE_GUI   # all scenarios get viewer; close one to proceed to next

    res = run_simulation(
        clip_id                  = clip_id,
        clip_start_frame         = clip_info["frame_offset"],
        n_frames                 = n_frames_sim,
        knee_pred_included_deg   = knee_sig,
        thigh_pred_included_deg  = thigh_query,   # IMU thigh from the query window
        use_viewer               = sc_gui,
        label                    = sc_key,
    )
    sim_results[sc_key] = res

    print(f"  fall={res['fall_detected']}  "
          f"frame={res['fall_frame']}  "
          f"steps={res['step_count']}  "
          f"stab={res['stability_score']:.3f}  "
          f"knee_RMSE={res['knee_rmse_deg']:.1f}°")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Results summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("RESULTS SUMMARY")
print("═" * 60)
print(f"Source  : {source_label}")
print(f"Match   : {clip_id}   DTW={dtw_dist:.4f}   RMSE={match_rmse:.2f}°")
print()

for sc_key in args.scenarios:
    sc_label, _ = SCENARIO_DEFS[sc_key]
    r = sim_results[sc_key]
    print(f"[{sc_key.upper()}]  {sc_label}")
    print(f"  Fall: {'YES (frame ' + str(r['fall_frame']) + ')' if r['fall_detected'] else 'no'}")
    print(f"  Steps:          {r['step_count']}  "
          f"(R={r['right_steps']} L={r['left_steps']})")
    print(f"  Gait symmetry:  {r['gait_symmetry']:.3f}  "
          f"(1.0 = perfect)")
    print(f"  Stability score: {r['stability_score']:.3f}")
    print(f"  CoM height:     {r['com_height_mean']:.3f} ± {r['com_height_std']:.3f} m  "
          f"(min={r['com_height_min']:.3f} m)")
    if not np.isnan(r["knee_rmse_deg"]):
        print(f"  Knee RMSE:      {r['knee_rmse_deg']:.1f}°   MAE={r['knee_mae_deg']:.1f}°")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Plots
# ─────────────────────────────────────────────────────────────────────────────
print("═" * 60)
print("STEP 7  Result plots")
print("═" * 60)

# Build knee signal dict for the plot
knee_plot_sigs = {}
for sc_key in args.scenarios:
    sc_label, knee_sig = SCENARIO_DEFS[sc_key]
    knee_plot_sigs[sc_key] = np.asarray(knee_sig)

scenario_labels = {k: SCENARIO_DEFS[k][0] for k in args.scenarios}

plot_sim_results(
    scenarios       = sim_results,
    scenario_labels = scenario_labels,
    knee_signals    = knee_plot_sigs,
    fps             = TARGET_FPS,
    out_path        = "sim_results.png",
)

# ─────────────────────────────────────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────────────────────────────────────
if args.out:
    def _ser(v):
        if isinstance(v, dict):
            return {k: _ser(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_ser(x) for x in v]
        if hasattr(v, "item"):
            return v.item()
        if hasattr(v, "tolist"):
            return v.tolist()
        return v

    out = {
        "source":        source_label,
        "checkpoint":    args.checkpoint,
        "subset":        args.subset,
        "matched_clip":  clip_id,
        "dtw_distance":  float(dtw_dist),
        "match_rmse":    float(match_rmse),
        "good_rmse":     float(good_rmse),
        "bad_rmse":      float(bad_rmse),
        "query_frames":  int(n_frames_sim),
        "scenarios":     _ser(sim_results),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"JSON saved → {args.out}")
