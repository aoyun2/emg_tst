#!/usr/bin/env python3
"""EMG prosthetic simulation pipeline — clean rewrite.

Architecture
============
  1. Data source      : rigtest .npy recording  OR  external OpenSim gait
  2. Model prediction : noisy-GT baseline (or real checkpoint if provided)
  3. Motion matching  : DTW on (knee, thigh) against MoCapAct reference database
  4. MoCapAct clip    : resolve matched segment → dm_control clip ID
  5. Physics sim      : dm_control CMU humanoid, knee + hip overridden
  6. Heuristics       : CoM height, foot contacts, fall detection, gait symmetry
  7. Visualisation    : mujoco viewer (GUI) + matplotlib summary plot + optional GIF

The simulation loop is implemented in ``mocapact_sim.simulate_scenario``, which
fixes the infinite-spin bug in the old ``simulate_three_scenarios_mocapact``
(that bug: when viewers failed to open ``open_viewers=[]`` → falsy → the looping
branch always triggered → infinite loop with nothing visible).

Usage
-----
  # Auto-detect data source (rigtest recording or OpenSim fallback):
    python run_quick_sim.py

  # Force external OpenSim data:
    python run_quick_sim.py --test-sample

  # Use a specific recording:
    python run_quick_sim.py --data-file data0.npy

  # Headless (no viewer window) + save a GIF:
    python run_quick_sim.py --no-gui --render-gif

  # Tune simulation length / noise:
    python run_quick_sim.py --seconds 15 --good-noise 4 --bad-noise 30
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

# ── CLI ────────────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
_ap.add_argument("--test-sample", action="store_true",
                 help="Use external OpenSim data (ignore local recordings)")
_ap.add_argument("--data-file", default=None,
                 help="Specific rigtest .npy file to use")
_ap.add_argument("--data", default=None,
                 help="Pre-windowed samples_dataset.npy (same format as run_evaluation.py)")
_ap.add_argument("--checkpoint", default=None,
                 help="Path to trained model (.pt) — runs real inference instead of dummy predictions. "
                      "Requires --data when using pre-windowed windows.")
_ap.add_argument("--n-windows", type=int, default=None,
                 help="Number of 1-second windows to use (default: all up to --seconds)")
_ap.add_argument("--no-gui", action="store_true",
                 help="Skip interactive viewer; still save plot + optional GIF")
_ap.add_argument("--render-gif", action="store_true",
                 help="Render every frame offscreen and save animated GIFs")
_ap.add_argument("--seconds", type=float, default=10.0,
                 help="Simulation duration in seconds (default: 10)")
_ap.add_argument("--good-noise", type=float, default=5.0,
                 help="Knee noise std (deg) for the 'good' prediction scenario")
_ap.add_argument("--bad-noise", type=float, default=25.0,
                 help="Knee noise std (deg) for the 'bad' prediction scenario")
_ap.add_argument("--scenarios", choices=["all", "gt", "good", "bad"], default="all",
                 help="Which scenarios to simulate (default: all three)")
_ap.add_argument("--gui-scenario", choices=["gt", "good", "bad", "all"], default="gt",
                 help="Which scenario(s) get the interactive viewer (default: gt)")
_ap.add_argument("--out", default=None,
                 help="Save results as JSON (e.g. eval_results.json).  "
                      "Replaces run_evaluation.py when combined with --checkpoint --data.")
_ap.add_argument("--device", default="cpu",
                 help="Torch device for model inference (default: cpu)")
args = _ap.parse_args()

USE_GUI      = not args.no_gui
RENDER_GIF   = args.render_gif
TOTAL_SECS   = args.seconds
GOOD_NOISE   = args.good_noise
BAD_NOISE    = args.bad_noise


def _resample(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if src_fps == dst_fps:
        return arr
    n_dst = max(2, int(round(len(arr) * dst_fps / src_fps)))
    return np.interp(
        np.linspace(0.0, 1.0, n_dst),
        np.linspace(0.0, 1.0, len(arr)),
        arr,
    ).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Data source: rigtest recording or external OpenSim gait
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 64)
print("STEP 1  Data source")
print("═" * 64)

from mocap_evaluation.mocap_loader import TARGET_FPS   # noqa: E402
WINDOW_FRAMES = int(TARGET_FPS)  # 1 second at TARGET_FPS (200)

# Auto-select data source
if not args.test_sample and not args.data_file and not args.data:
    candidates = sorted(p for p in glob.glob("data*.npy") if "samples" not in p)
    if candidates:
        args.data_file = candidates[0]
        print(f"Found recording: {args.data_file}")
    else:
        print("No data*.npy recording found — falling back to external OpenSim data.")
        print("Tip: run  uMyo_python_tools/rigtest.py  to capture your own data.")
        args.test_sample = True

max_frames = int(TOTAL_SECS * TARGET_FPS)
if args.n_windows is not None:
    max_frames = min(max_frames, args.n_windows * WINDOW_FRAMES)

# x_windows_for_inference: set when --data is provided; used by --checkpoint in STEP 2
x_windows_for_inference: "Optional[np.ndarray]" = None

if args.data:
    # Pre-windowed samples_dataset.npy — same format as run_evaluation.py
    from mocap_evaluation.paper_pipeline import _load_windows  # noqa: E402
    x_wins, y_seqs, _, _ = _load_windows(args.data, n_samples=None)
    x_windows_for_inference = x_wins  # (n_windows, seq_len, n_vars)
    knee_query  = np.concatenate(y_seqs).astype(np.float32)[:max_frames]
    thigh_query = np.concatenate([x_wins[i, :, -1] for i in range(len(x_wins))]).astype(np.float32)[:max_frames]
    source_label = f"samples_dataset ({len(x_wins)} windows, {args.data})"
    print(f"Source : {args.data}  ({len(x_wins)} windows → {len(knee_query)} frames)")
elif args.test_sample:
    from mocap_evaluation.external_sample_data import extract_external_sample_curves
    print("Downloading external OpenSim gait sample …")
    curves = extract_external_sample_curves(seconds=TOTAL_SECS)
    knee_query  = curves.knee_label_included_deg[:max_frames]
    thigh_query = curves.thigh_angle_deg[:max_frames]
    source_label = f"external OpenSim  ({curves.source_file.split('/')[-1]})"
    print(f"Source : {curves.source_file}")
else:
    from emg_tst.data import load_recording
    all_files = (
        [args.data_file] if args.data_file
        else sorted(p for p in glob.glob("data*.npy") if "samples" not in p)
    )
    knee_chunks: list[np.ndarray] = []
    thigh_chunks: list[np.ndarray] = []
    for fpath in all_files:
        X, y, meta = load_recording(fpath)
        n_complete = (len(y) // WINDOW_FRAMES) * WINDOW_FRAMES
        if n_complete == 0:
            print(f"  {fpath}: too short ({len(y)} frames), skipping")
            continue
        knee_chunks.append(y[:n_complete].astype(np.float32))
        thigh_chunks.append(X[:n_complete, -1].astype(np.float32))
        print(f"  {fpath}: {n_complete // WINDOW_FRAMES} window(s)  "
              f"@ {meta['effective_hz']:.0f} Hz")
    if not knee_chunks:
        sys.exit("No usable windows found in any recording file.")
    knee_query  = np.concatenate(knee_chunks)[:max_frames]
    thigh_query = np.concatenate(thigh_chunks)[:max_frames]
    source_label = f"rigtest ({len(all_files)} file(s))"

query_sec = len(knee_query) / TARGET_FPS
n_windows_used = len(knee_query) // WINDOW_FRAMES
print(f"Query  : {n_windows_used} × 1 s = {len(knee_query)} frames ({query_sec:.1f} s @ {TARGET_FPS} Hz)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Model prediction
#   • With --checkpoint + --data  : real model inference on pre-windowed X
#   • Otherwise                   : noisy-GT baseline (demo / sanity check)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 64)
print("STEP 2  Model prediction")
print("═" * 64)

rng = np.random.default_rng(42)

if args.checkpoint and x_windows_for_inference is not None:
    import torch  # noqa: E402
    from mocap_evaluation.paper_pipeline import _load_checkpoint, _predict as _model_predict  # noqa: E402
    device = torch.device(args.device)
    _model, _scaler = _load_checkpoint(args.checkpoint, device)
    print(f"Checkpoint : {args.checkpoint}")
    # Run inference on every window, concatenate predictions
    pred_chunks = [
        _model_predict(_model, _scaler, x_windows_for_inference[i], device)
        for i in range(len(x_windows_for_inference))
    ]
    knee_model_pred = np.concatenate(pred_chunks).astype(np.float32)
    # Align length to knee_query (windows may give more/fewer frames)
    n = min(len(knee_model_pred), len(knee_query))
    knee_good_pred = np.clip(knee_model_pred[:n], 0.0, 180.0)
    knee_query     = knee_query[:n]
    thigh_query    = thigh_query[:n]
    good_rmse = float(np.sqrt(np.mean((knee_good_pred - knee_query) ** 2)))
    print(f"Model RMSE vs GT : {good_rmse:.2f} °  ({len(pred_chunks)} windows)")
    prediction_label = f"model ({Path(args.checkpoint).name})"
else:
    if args.checkpoint:
        print("Note: --checkpoint requires --data for real inference; using noisy-GT baseline.")
    knee_good_pred = np.clip(
        knee_query + rng.normal(0.0, GOOD_NOISE, len(knee_query)).astype(np.float32),
        0.0, 180.0,
    )
    good_rmse = float(np.sqrt(np.mean((knee_good_pred - knee_query) ** 2)))
    print(f"Good prediction RMSE : {good_rmse:.2f} °  (noise std={GOOD_NOISE} °)")
    prediction_label = f"noisy-GT (σ={GOOD_NOISE}°)"

knee_bad_pred = np.clip(
    knee_query + rng.normal(0.0, BAD_NOISE, len(knee_query)).astype(np.float32),
    0.0, 180.0,
)
bad_rmse = float(np.sqrt(np.mean((knee_bad_pred - knee_query) ** 2)))
print(f"Bad  prediction RMSE : {bad_rmse:.2f} °  (noise std={BAD_NOISE} °)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Motion matching: DTW on (knee, thigh) against MoCapAct reference DB
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 64)
print("STEP 3  Motion matching")
print("═" * 64)

# load_mocapact_database uses the same dm_control HDF5 reference trajectories
# that the physics simulation uses internally.  Every clip ID it returns is
# guaranteed to exist in the HDF5 file, so create_walking_env will never see a
# missing-clip KeyError (which happened when using load_aggregated_database and
# a BVH file was converted to a clip ID not in the HDF5 — e.g. CMU_104_46).
from mocap_evaluation.mocapact_dataset import load_mocapact_database  # noqa: E402
from mocap_evaluation.motion_matching import find_best_match           # noqa: E402

db = load_mocapact_database(use_cache=True)
db_fps = float(db["fps"])
print(f"Database : {len(db['knee_right']) / db_fps:.0f} s  |  "
      f"{len(db['file_boundaries'])} clips  @  {db_fps:.0f} Hz")

print(f"Running DTW on {query_sec:.1f} s query …")
best_start, dtw_dist, matched_seg = find_best_match(knee_query, thigh_query, db)

# Identify matched file
matched_file = "?"
for s, e, f, c in db["file_boundaries"]:
    if int(s) <= best_start < int(e):
        matched_file = str(f)
        break

T_cmp = min(len(matched_seg["knee_right"]), len(knee_query))
match_rmse = float(np.sqrt(np.mean(
    (matched_seg["knee_right"][:T_cmp] - knee_query[:T_cmp]) ** 2
)))
print(f"Best match : {matched_file}  DTW={dtw_dist:.4f}  RMSE={match_rmse:.2f} °")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MoCapAct clip  (same length as matched query)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 64)
print("STEP 4  MoCapAct clip")
print("═" * 64)

from mocap_evaluation.mocapact_sim import (  # noqa: E402
    SIM_FPS,
    resolve_clip_from_match,
    load_multi_clip_policy,
    create_walking_env,
    simulate_scenario,
)

# The matched segment is the same length as the query (len(knee_query) frames at db_fps)
match_info = resolve_clip_from_match(best_start, len(knee_query), db)
print(f"Clip    : {match_info['clip_id']}")
print(f"Category: {match_info['category']}")
print(f"Offset  : {match_info['time_offset_s']:.2f} s  "
      f"Duration: {match_info['time_duration_s']:.2f} s")

# Resample matched CMU reference signal → SIM_FPS (for simulation frames)
knee_gt_sim    = _resample(matched_seg["knee_right"], db_fps, SIM_FPS)
thigh_sim      = _resample(matched_seg["hip_right"],  db_fps, SIM_FPS)
knee_good_sim  = _resample(knee_good_pred,            TARGET_FPS, SIM_FPS)
knee_bad_sim   = _resample(knee_bad_pred,             TARGET_FPS, SIM_FPS)

# Align all signals to the shortest
T_sim = min(len(knee_gt_sim), len(thigh_sim), len(knee_good_sim), len(knee_bad_sim))
knee_gt_sim   = knee_gt_sim[:T_sim]
thigh_sim     = thigh_sim[:T_sim]
knee_good_sim = knee_good_sim[:T_sim]
knee_bad_sim  = knee_bad_sim[:T_sim]
sim_seconds   = T_sim / SIM_FPS

print(f"Sim signal : {T_sim} frames  @  {SIM_FPS:.0f} Hz  ({sim_seconds:.1f} s)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Physics simulation
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 64)
print("STEP 5  Physics simulation")
print("═" * 64)

policy = load_multi_clip_policy(model_dir="mocapact_models")
print("Policy loaded.")

# Which scenarios to run
SCENARIO_DEFS = {
    "gt":   ("GT knee (CMU matched)",              knee_gt_sim,   "blue"),
    "good": (f"{prediction_label} (RMSE≈{good_rmse:.1f}°)", knee_good_sim, "orange"),
    "bad":  (f"Bad pred  (RMSE≈{bad_rmse:.1f}°)",  knee_bad_sim,  "red"),
}
run_which = list(SCENARIO_DEFS.keys()) if args.scenarios == "all" else [args.scenarios]

results: dict[str, dict] = {}

for sc_key in run_which:
    sc_label, knee_sig, color = SCENARIO_DEFS[sc_key]

    # Should this scenario get the GUI viewer?
    sc_gui = (
        USE_GUI and (args.gui_scenario == "all" or args.gui_scenario == sc_key)
    )
    sc_gif = (
        RENDER_GIF and (args.gui_scenario == "all" or args.gui_scenario == sc_key)
    )
    gif_path = f"sim_{sc_key}.gif"

    print(f"\n  [{sc_key}]  {sc_label}  {'(GUI)' if sc_gui else ''}"
          f"{'(GIF→' + gif_path + ')' if sc_gif else ''}")

    try:
        env = create_walking_env(policy=policy, clip_id=match_info["clip_id"])
    except Exception as _env_err:
        # Safety net: if a stale cache returned a clip not in the HDF5, fall
        # back to the default locomotion-small dataset so the run still works.
        print(f"    Warning: clip {match_info['clip_id']!r} failed to load "
              f"({_env_err}); falling back to default locomotion dataset.")
        env = create_walking_env(policy=policy)
    try:
        res = simulate_scenario(
            env=env,
            policy=policy,
            knee_inc_deg=knee_sig,
            thigh_inc_deg=thigh_sim,
            reference_knee_inc_deg=knee_gt_sim,
            fps=SIM_FPS,
            use_gui=sc_gui,
            match_info=match_info,
            render_offscreen=sc_gif,
            video_path=gif_path,
        )
    finally:
        env.close()

    results[sc_key] = res
    print(f"    fall={res['fall_detected']}  frame={res['fall_frame']}  "
          f"steps={res['step_count']}  stab={res['stability_score']:.3f}  "
          f"knee_RMSE={res['knee_rmse_deg']:.2f}°")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Results summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 64)
print("RESULTS")
print("═" * 64)
print(f"Source  : {source_label}")
print(f"Match   : {matched_file}  DTW={dtw_dist:.4f}  RMSE={match_rmse:.2f}°")
print(f"Query   : {n_windows_used} × 1 s ({query_sec:.1f} s)  →  "
      f"sim {T_sim} frames @ {SIM_FPS:.0f} Hz ({sim_seconds:.1f} s)\n")

for sc_key in run_which:
    sc_label, _, _ = SCENARIO_DEFS[sc_key]
    r = results[sc_key]
    print(f"{sc_label}:")
    print(f"  fall={r['fall_detected']} (frame {r['fall_frame']})  "
          f"steps={r['step_count']}  gait_sym={r['gait_symmetry']:.3f}")
    print(f"  stability={r['stability_score']:.3f}  "
          f"CoM={r['com_height_mean']:.3f}±{r['com_height_std']:.3f} m")
    print(f"  knee RMSE={r['knee_rmse_deg']:.2f}°  MAE={r['knee_mae_deg']:.2f}°")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Matplotlib comparison plot  (always saved, even in GUI mode)
# ══════════════════════════════════════════════════════════════════════════════
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

t_ax = np.arange(T_sim) / SIM_FPS
plot_path = "sim_comparison.png"

n_panels = 2 + (1 if len(run_which) > 0 else 0)
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)

# ── Panel 0: knee angle signals ───────────────────────────────────────────────
ax0 = axes[0]
ax0.plot(t_ax, knee_gt_sim,   label="GT knee (CMU matched)", lw=2, color="blue")
if "good" in results:
    ax0.plot(t_ax, knee_good_sim, lw=1.5, color="orange", alpha=0.85,
             label=f"{prediction_label} RMSE={good_rmse:.1f}°")
if "bad" in results:
    ax0.plot(t_ax, knee_bad_sim,  lw=1.5, color="red",    alpha=0.70,
             label=f"Bad pred RMSE={bad_rmse:.1f}°")
ax0.set_ylabel("Knee flexion (°)")
ax0.set_title(
    f"{source_label}  ({n_windows_used}×1 s)  →  {matched_file}  "
    f"|  DTW={dtw_dist:.4f}  MatchRMSE={match_rmse:.1f}°"
)
ax0.legend(fontsize=8)
ax0.grid(True, alpha=0.3)

# ── Panel 1: CoM height ───────────────────────────────────────────────────────
ax1 = axes[1]
color_map = {"gt": "blue", "good": "orange", "bad": "red"}
for sc_key in run_which:
    sc_label, _, color = SCENARIO_DEFS[sc_key]
    com = np.asarray(results[sc_key]["com_height_series"])
    t_com = np.arange(len(com)) / SIM_FPS
    ax1.plot(t_com, com, lw=2, color=color,
             label=f"{sc_label}  stab={results[sc_key]['stability_score']:.2f}")
ax1.axhline(0.55, color="r", ls="--", lw=1, label="Fall threshold (0.55 m)")
ax1.set_ylabel("CoM height (m)")
ax1.set_title("Centre of Mass height")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Panel 2: foot contacts ────────────────────────────────────────────────────
if n_panels > 2:
    ax2 = axes[2]
    for i, sc_key in enumerate(run_which):
        sc_label, _, color = SCENARIO_DEFS[sc_key]
        r = results[sc_key]
        y_r = 1.0 - i * 0.3
        rc = np.asarray(r["right_contact_frames"]) / SIM_FPS
        lc = np.asarray(r["left_contact_frames"])  / SIM_FPS
        ax2.scatter(rc, np.full_like(rc, y_r),        marker="|", s=100,
                    color=color, label=f"{sc_label} R")
        ax2.scatter(lc, np.full_like(lc, y_r + 0.1),  marker="|", s=100,
                    color=color, alpha=0.5, label=f"{sc_label} L")
    ax2.set_xlabel("Time (s)")
    ax2.set_title(
        "Foot contacts  |  steps: "
        + "  ".join(f"{SCENARIO_DEFS[k][0].split()[0]}={results[k]['step_count']}"
                    for k in run_which)
    )
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)
else:
    axes[-1].set_xlabel("Time (s)")

fig.tight_layout()
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Plot saved : {plot_path}")

# ══════════════════════════════════════════════════════════════════════════════
# JSON output  (--out replaces run_evaluation.py for structured results)
# ══════════════════════════════════════════════════════════════════════════════
if args.out:
    import json  # noqa: E402

    def _serialise(v):
        """Make numpy scalars / arrays JSON-serialisable."""
        if isinstance(v, dict):
            return {k: _serialise(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_serialise(x) for x in v]
        if hasattr(v, "item"):          # numpy scalar
            return v.item()
        if hasattr(v, "tolist"):        # numpy array
            return v.tolist()
        return v

    out_data = {
        "mode": "pipeline",
        "source": source_label,
        "checkpoint": args.checkpoint,
        "matched_clip": matched_file,
        "dtw_distance": float(dtw_dist),
        "match_rmse_deg": float(match_rmse),
        "good_rmse_deg": float(good_rmse),
        "bad_rmse_deg": float(bad_rmse),
        "sim_frames": int(T_sim),
        "sim_fps": float(SIM_FPS),
        "scenarios": _serialise(results),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as _f:
        json.dump(out_data, _f, indent=2)
    print(f"JSON saved : {out_path}")
