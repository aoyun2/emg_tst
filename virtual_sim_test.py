#!/usr/bin/env python3
"""Virtual simulation test: validate the motion-matching → physics pipeline.

This is a developer testing tool that uses publicly available OpenSim gait
kinematics (or synthetic signals) as a stand-in for real EMG recordings.
It is intended to be run BEFORE hardware data is collected and the TST model
is trained, to confirm that the entire pipeline works end-to-end.

Mapping to the real workflow
-----------------------------
  Real workflow:
    rigtest.py recordings  → aggregate N consecutive 1-s windows → test batch
    TST model              → predict knee angle from batch EMG features
    Motion matching        → DTW-match (thigh, knee) against MoCap Act database
    Simulation             → replay matched clip, override thigh + knee each frame

  This test script:
    OpenSim gait .mot file → split into test batches (same duration/rate)
    Identity prediction    → batch knee angle passed through unchanged (noise=0)
    Motion matching        → identical DTW pipeline
    Simulation             → identical kinematic replay + override

Pipeline (per run)
------------------
  1. LOAD   OpenSim gait data (auto-downloaded) or synthetic gait signals.
  2. BATCH  Split into N consecutive test batches (each = batch_secs seconds).
  3. DB     Load MoCap Act motion-matching database.
  4. MATCH  DTW-match each batch against the database.
  5. SIM    Kinematic replay with thigh + knee overridden by batch values.
  6. REPORT Per-batch and aggregate metrics.
  7. PLOT   Match quality + simulation results per batch + aggregate summary.

All angles use the included-angle convention (180° = full extension).

Usage examples
--------------
  python virtual_sim_test.py                       # 3 batches, try OpenSim first
  python virtual_sim_test.py --synthetic           # skip download, use synthetic
  python virtual_sim_test.py --n-batches 5 --batch-secs 3
  python virtual_sim_test.py --subset walk_tiny    # fewer clips → faster run
  python virtual_sim_test.py --no-gui              # headless, plots only
  python virtual_sim_test.py --noise 8             # simulate 8° model error
  python virtual_sim_test.py --out results.json    # save JSON summary
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── CLI ───────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
ap.add_argument(
    "--n-batches", type=int, default=3,
    help="Number of test batches to create and simulate (default: 3)",
)
ap.add_argument(
    "--batch-secs", type=float, default=5.0,
    help="Duration of each test batch in seconds (default: 5.0)",
)
ap.add_argument(
    "--subset", default="all",
    choices=["all", "locomotion_small", "walk_tiny", "run_jump_tiny"],
    help=(
        "MoCap Act database subset (default: all = ~2589 clips). "
        "Use walk_tiny or locomotion_small for a faster development run."
    ),
)
ap.add_argument(
    "--no-gui", action="store_true",
    help="Headless mode — no MuJoCo viewer window, saves plots only",
)
ap.add_argument(
    "--synthetic", action="store_true",
    help="Skip OpenSim download and use synthetic gait signals instead",
)
ap.add_argument(
    "--noise", type=float, default=0.0,
    help=(
        "Gaussian noise σ (°) added to the 'model' knee prediction "
        "(default: 0 = identity, i.e. query knee passed through unchanged). "
        "Use e.g. --noise 8 to simulate realistic model error."
    ),
)
ap.add_argument(
    "--prefilter-k", type=int, default=100,
    help="L2 pre-filter candidates passed to full DTW scoring (default: 100)",
)
ap.add_argument(
    "--out", default=None,
    help="Optional: save JSON result summary to this file path",
)

args = ap.parse_args()

TARGET_FPS   = 200.0
BATCH_FRAMES = int(args.batch_secs * TARGET_FPS)
USE_GUI      = not args.no_gui


# ── Synthetic gait signal generator ───────────────────────────────────────────

def generate_synthetic_gait(
    total_secs: float,
    fps: float = TARGET_FPS,
    stride_hz: float = 1.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (knee, thigh) signals approximating normal walking.

    Uses a Fourier series gait model based on Winter's biomechanics data.
    Outputs are in included-angle convention (180° = full extension).

    Parameters
    ----------
    total_secs : total signal duration in seconds
    fps        : output sample rate (default 200 Hz)
    stride_hz  : fundamental stride frequency (default 1.1 Hz ≈ comfortable walk)
    seed       : RNG seed for inter-stride variability noise

    Returns
    -------
    knee_inc, thigh_inc : (T,) float32 included-angle arrays
    """
    rng = np.random.default_rng(seed)
    T   = int(total_secs * fps)
    t   = np.arange(T, dtype=np.float64) / fps
    phi = 2.0 * np.pi * stride_hz * t

    # Knee: bimodal flexion pattern per stride.
    # Stance loading (~18° flexion) + swing peak (~60° flexion).
    knee_flex = (
        18.0 * np.maximum(0.0, np.sin(phi)) ** 1.4          # loading response
        + 60.0 * np.maximum(0.0, -np.sin(phi)) ** 2.0       # swing peak
        +  5.0 * np.sin(2.0 * phi + 0.4)                    # 2nd harmonic
    )
    knee_noise = rng.normal(0.0, 1.5, T).astype(np.float32)
    # included-angle = 180 - flexion; clip to plausible walking range
    knee_inc = np.clip(180.0 - knee_flex + knee_noise, 100.0, 180.0).astype(np.float32)

    # Thigh / hip: single sinusoid ± 22° around neutral, with 2nd harmonic.
    # hip included-angle = 180 - hip_flex_deg:
    #   flexion → included < 180; extension → included > 180
    hip_flex   = 22.0 * np.sin(phi + np.pi / 5.0) + 4.0 * np.sin(2.0 * phi + 0.2)
    thigh_noise = rng.normal(0.0, 1.0, T).astype(np.float32)
    thigh_inc = np.clip(180.0 - hip_flex + thigh_noise, 130.0, 210.0).astype(np.float32)

    return knee_inc, thigh_inc


# ── OpenSim data loader ────────────────────────────────────────────────────────

# Public OpenSim IK result files from opensim-core test suite.
_OPENSIM_URLS: List[str] = [
    (
        "https://raw.githubusercontent.com/opensim-org/opensim-core/main/"
        "Applications/CMC/test/subject01_walk1_ik.mot"
    ),
    (
        "https://raw.githubusercontent.com/opensim-org/opensim-core/main/"
        "Applications/Forward/test/subject01_walk1_ik.mot"
    ),
    (
        "https://raw.githubusercontent.com/opensim-org/opensim-core/main/"
        "Applications/RRA/test/subject01_walk1_ik.mot"
    ),
]


def _parse_mot(text: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Parse an OpenSim .mot file text.

    Returns (time_s, knee_flex_deg, hip_flex_deg) or None if parsing fails.

    The .mot format has a header block ending with "endheader", followed by
    a tab/space-delimited table whose first column is "time".
    Positive flexion convention for both knee and hip.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Locate the column-header line.
    hdr_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low == "endheader" and i + 1 < len(lines):
            hdr_idx = i + 1
            break
        if low.startswith("time\t") or low.startswith("time "):
            hdr_idx = i
            break
    if hdr_idx is None:
        return None

    cols    = lines[hdr_idx].replace("\t", " ").split()
    col_map = {c: idx for idx, c in enumerate(cols)}

    data_rows = [ln.replace("\t", " ") for ln in lines[hdr_idx + 1:] if ln]
    if not data_rows:
        return None

    try:
        arr = np.genfromtxt(data_rows, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    except Exception:
        return None

    if arr.shape[1] != len(cols):
        return None

    time_key = next((k for k in col_map if k.lower() == "time"), None)
    if time_key is None:
        return None
    time_s = arr[:, col_map[time_key]]

    knee_key = next(
        (k for k in ["knee_angle_r", "knee_flexion_r", "knee_angle_right"] if k in col_map),
        None,
    )
    if knee_key is None:
        return None

    hip_key = next(
        (k for k in ["hip_flexion_r", "hip_flex_r", "hip_flexion_right", "hip_angle_r"]
         if k in col_map),
        None,
    )
    if hip_key is None:
        return None

    return time_s, arr[:, col_map[knee_key]], arr[:, col_map[hip_key]]


def load_opensim_data(
    total_secs: float,
    fps: float = TARGET_FPS,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Download and parse an OpenSim gait kinematics file.

    Tries each URL in _OPENSIM_URLS in order, returning on the first success.
    If all URLs fail or the file cannot be parsed, returns None.

    Returns
    -------
    (knee_inc_deg, thigh_inc_deg) at *fps*, tiled to *total_secs* if needed.
    """
    import urllib.request

    raw_text: Optional[str] = None
    url_used: Optional[str] = None

    for url in _OPENSIM_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "emg-tst/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")
            url_used = url
            break
        except Exception as exc:
            warnings.warn(f"[opensim] {url.split('/')[-1]}: {exc}")

    if raw_text is None:
        return None

    parsed = _parse_mot(raw_text)
    if parsed is None:
        warnings.warn("[opensim] Could not parse .mot file — column names not found.")
        return None

    time_s, knee_flex, hip_flex = parsed
    print(
        f"  OpenSim file : {url_used.split('/')[-1]}  "
        f"({len(time_s)} rows, {float(time_s[-1]) - float(time_s[0]):.2f} s)"
    )

    def _interp_to_fps(ts: np.ndarray, sig: np.ndarray) -> np.ndarray:
        t0, t1 = float(ts[0]), float(ts[-1])
        n = max(2, int(round((t1 - t0) * fps)))
        return np.interp(np.linspace(t0, t1, n), ts, sig).astype(np.float32)

    knee_rs  = _interp_to_fps(time_s, knee_flex)
    thigh_rs = _interp_to_fps(time_s, hip_flex)

    # Convert to included-angle convention (180° = full extension).
    # OpenSim knee_angle_r: positive = flexion.
    # OpenSim hip_flexion_r: positive = flexion.
    knee_inc  = np.clip(180.0 - np.abs(knee_rs),  100.0, 180.0).astype(np.float32)
    thigh_inc = np.clip(180.0 - thigh_rs,           130.0, 210.0).astype(np.float32)

    # Tile to requested duration.
    n_need = int(total_secs * fps)
    while len(knee_inc) < n_need:
        knee_inc  = np.concatenate([knee_inc,  knee_inc])
        thigh_inc = np.concatenate([thigh_inc, thigh_inc])

    return knee_inc[:n_need], thigh_inc[:n_need]


# ── Batch creation ─────────────────────────────────────────────────────────────

def make_batches(
    knee:         np.ndarray,
    thigh:        np.ndarray,
    n_batches:    int,
    batch_frames: int,
) -> List[dict]:
    """Split continuous angle signals into equal-length, non-overlapping batches.

    If the signals are shorter than n_batches * batch_frames, they are tiled.

    Returns a list of dicts:
        batch_id   : int (0-indexed)
        knee       : (batch_frames,) float32 included-angle degrees
        thigh      : (batch_frames,) float32 included-angle degrees
        start_s    : float  batch start time in seconds
        duration_s : float  batch duration in seconds
    """
    n_need = n_batches * batch_frames
    if len(knee) < n_need:
        reps  = int(np.ceil(n_need / len(knee))) + 1
        knee  = np.tile(knee,  reps)[:n_need]
        thigh = np.tile(thigh, reps)[:n_need]

    batches: List[dict] = []
    for i in range(n_batches):
        s = i * batch_frames
        e = s + batch_frames
        batches.append({
            "batch_id":   i,
            "knee":       knee[s:e].copy(),
            "thigh":      thigh[s:e].copy(),
            "start_s":    s / TARGET_FPS,
            "duration_s": batch_frames / TARGET_FPS,
        })
    return batches


# ── Per-batch pipeline ─────────────────────────────────────────────────────────

def run_batch(
    batch:       dict,
    db:          dict,
    noise_std:   float,
    use_gui:     bool,
    prefilter_k: int,
    out_dir:     Path,
    n_total:     int,
) -> dict:
    """Run the full match → simulate pipeline for a single test batch.

    Two scenarios are evaluated for each batch:
      gt    — Matched-clip's own knee angle drives the simulation (upper bound).
      good  — Query knee (= model placeholder) drives the simulation.

    Both scenarios override the right thigh (hip) with the batch thigh angle.

    Parameters
    ----------
    batch       : dict from make_batches()
    db          : MoCap Act database (from load_database())
    noise_std   : σ of Gaussian noise added to the 'good' knee signal (°)
    use_gui     : open an interactive MuJoCo viewer window
    prefilter_k : L2 pre-filter candidate count
    out_dir     : directory for per-batch output plots
    n_total     : total batch count (for display only)

    Returns
    -------
    dict with match + simulation metrics.
    """
    from mocap_evaluation.matching import find_match, plot_match_quality, clip_info_for_start
    from mocap_evaluation.sim      import run_simulation, plot_sim_results

    bid   = batch["batch_id"]
    knee  = batch["knee"]
    thigh = batch["thigh"]

    print(f"\n{'═' * 60}")
    print(
        f"BATCH {bid + 1}/{n_total}  "
        f"(t={batch['start_s']:.1f}–{batch['start_s'] + batch['duration_s']:.1f} s  "
        f"{batch['duration_s']:.0f} s)"
    )
    print(f"  Knee  : {knee.min():.1f}°–{knee.max():.1f}°  mean={knee.mean():.1f}°")
    print(f"  Thigh : {thigh.min():.1f}°–{thigh.max():.1f}°  mean={thigh.mean():.1f}°")
    print("═" * 60)

    # ── Motion matching ────────────────────────────────────────────────────────
    print(f"\n[Batch {bid + 1}] Motion matching …")
    best_start, dtw_dist, clip_id, matched_seg = find_match(
        knee, thigh, db, prefilter_k=prefilter_k,
    )
    clip_info  = clip_info_for_start(best_start, len(knee), db)
    match_rmse = float(np.sqrt(np.mean(
        (matched_seg["knee_right"] - knee[:len(matched_seg["knee_right"])]) ** 2
    )))
    print(
        f"  Matched : {clip_id}  (offset {clip_info['time_offset_s']:.2f} s)\n"
        f"  DTW={dtw_dist:.4f}   knee RMSE={match_rmse:.2f}°"
    )

    # ── Match quality plot ─────────────────────────────────────────────────────
    plot_match_quality(
        knee, thigh,
        matched_seg["knee_right"],
        matched_seg["hip_right"],
        dtw_dist, clip_id,
        fps=TARGET_FPS,
        out_path=str(out_dir / f"batch{bid + 1:02d}_match.png"),
    )

    # ── Model prediction placeholder ───────────────────────────────────────────
    # Until the TST model is trained, the query knee IS the "prediction"
    # (identity mapping).  Optional noise simulates future model error.
    rng        = np.random.default_rng(bid)
    knee_model = np.clip(
        knee + rng.normal(0.0, noise_std, len(knee)).astype(np.float32),
        0.0, 180.0,
    )

    # ── Simulation ─────────────────────────────────────────────────────────────
    print(f"\n[Batch {bid + 1}] Physics simulation …")

    # scenario_defs: key → (display_label, knee_signal)
    # 'gt'   — matched clip's own knee (upper-bound: how well can it possibly go?)
    # 'good' — query/model knee (what we actually care about)
    scenario_defs: Dict[str, Tuple[str, np.ndarray]] = {
        "gt":   ("Matched-clip GT",                         matched_seg["knee_right"]),
        "good": (f"Model placeholder (σ={noise_std:.0f}°)", knee_model),
    }

    sim_results:     dict = {}
    scenario_labels: dict = {}

    for sc_key, (sc_label, knee_sig) in scenario_defs.items():
        scenario_labels[sc_key] = sc_label
        res = run_simulation(
            clip_id                 = clip_id,
            clip_start_frame        = clip_info["frame_offset"],
            n_frames                = len(knee),
            knee_pred_included_deg  = knee_sig,
            thigh_pred_included_deg = thigh,        # always override thigh from batch
            use_viewer              = use_gui,
            label                   = f"B{bid + 1}/{sc_key}",
        )
        sim_results[sc_key] = res
        print(
            f"  [{sc_key:4s}] fall={res['fall_detected']}  "
            f"steps={res['step_count']}  stab={res['stability_score']:.3f}  "
            f"CoM={res['com_height_mean']:.3f} m"
        )

    # ── Per-batch simulation results plot ──────────────────────────────────────
    plot_sim_results(
        scenarios       = sim_results,
        scenario_labels = scenario_labels,
        knee_signals    = {k: np.asarray(v) for k, (_, v) in scenario_defs.items()},
        fps             = TARGET_FPS,
        out_path        = str(out_dir / f"batch{bid + 1:02d}_sim.png"),
    )

    return {
        "batch_id":     bid,
        "dtw_dist":     float(dtw_dist),
        "match_rmse":   float(match_rmse),
        "clip_id":      clip_id,
        "frame_offset": int(clip_info["frame_offset"]),
        "sim_results":  sim_results,
    }


# ── Aggregate summary plot ─────────────────────────────────────────────────────

def plot_aggregate_summary(
    batch_results: List[dict],
    out_path:      str,
) -> None:
    """3-panel bar chart comparing DTW, match RMSE, and stability per batch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(batch_results)
    x = np.arange(1, n + 1)

    dtw_vals  = [r["dtw_dist"]   for r in batch_results]
    rmse_vals = [r["match_rmse"] for r in batch_results]
    stab_gt   = [r["sim_results"]["gt"]["stability_score"]   for r in batch_results]
    stab_md   = [r["sim_results"]["good"]["stability_score"] for r in batch_results]

    fig, axes = plt.subplots(3, 1, figsize=(max(6, n * 1.8 + 2), 10))
    fig.suptitle("Virtual simulation test — per-batch aggregate summary", fontsize=11)

    ax = axes[0]
    ax.bar(x, dtw_vals, color="steelblue", alpha=0.85)
    ax.set_ylabel("DTW distance")
    ax.set_title("Motion-matching DTW distance (lower = better match)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {i}" for i in x])
    ax.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x, dtw_vals):
        ax.text(xi, v + max(v * 0.01, 0.001), f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    ax.bar(x, rmse_vals, color="darkorange", alpha=0.85)
    ax.set_ylabel("Knee RMSE (°)")
    ax.set_title("Matched-clip knee RMSE vs query (lower = better match quality)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {i}" for i in x])
    ax.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x, rmse_vals):
        ax.text(xi, v + max(v * 0.02, 0.1), f"{v:.1f}°", ha="center", va="bottom", fontsize=8)

    ax = axes[2]
    w = 0.35
    ax.bar(x - w / 2, stab_gt, w, color="steelblue",  alpha=0.85, label="GT clip")
    ax.bar(x + w / 2, stab_md, w, color="darkorange",  alpha=0.85, label="Model pred")
    ax.set_ylabel("Stability score")
    ax.set_title("Simulation stability (1.0 = no fall + perfect gait symmetry)")
    ax.set_ylim(0.0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Batch {i}" for i in x])
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Aggregate summary plot saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path("virtual_sim_output")
    out_dir.mkdir(exist_ok=True)

    # A bit of extra data so tiling always produces enough frames.
    total_secs = args.n_batches * args.batch_secs + 2.0

    # ── STEP 1: Data source ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("STEP 1  Test data source")
    print("═" * 60)

    knee_full:  Optional[np.ndarray] = None
    thigh_full: Optional[np.ndarray] = None
    source_label = ""

    if not args.synthetic:
        print("Attempting OpenSim gait data download …")
        result = load_opensim_data(total_secs)
        if result is not None:
            knee_full, thigh_full = result
            source_label = "OpenSim gait kinematics (auto-downloaded)"
            print(f"  {len(knee_full)} frames ({len(knee_full) / TARGET_FPS:.1f} s)")
        else:
            print("  All download attempts failed — falling back to synthetic data.")

    if knee_full is None:
        print("Generating synthetic gait signals …")
        knee_full, thigh_full = generate_synthetic_gait(total_secs, TARGET_FPS)
        source_label = "synthetic gait model (Fourier series approximation)"
        print(f"  {len(knee_full)} frames ({len(knee_full) / TARGET_FPS:.1f} s)")

    print(f"\n  Source : {source_label}")
    print(f"  Knee   : {knee_full.min():.1f}°–{knee_full.max():.1f}°  mean={knee_full.mean():.1f}°")
    print(f"  Thigh  : {thigh_full.min():.1f}°–{thigh_full.max():.1f}°  mean={thigh_full.mean():.1f}°")

    # ── STEP 2: Batch creation ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("STEP 2  Batch creation")
    print("═" * 60)

    batches = make_batches(knee_full, thigh_full, args.n_batches, BATCH_FRAMES)
    print(
        f"  {len(batches)} batches × {args.batch_secs:.1f} s = "
        f"{len(batches) * BATCH_FRAMES} frames total"
    )
    for b in batches:
        bid = b["batch_id"]
        print(
            f"  Batch {bid + 1}: t={b['start_s']:.1f}–"
            f"{b['start_s'] + b['duration_s']:.1f} s  "
            f"knee={b['knee'].mean():.1f}°  thigh={b['thigh'].mean():.1f}°"
        )

    # ── STEP 3: MoCap Act database ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("STEP 3  MoCap Act database")
    print("═" * 60)

    try:
        from mocap_evaluation.db import load_database
    except ImportError as exc:
        sys.exit(
            f"Cannot import mocap_evaluation.db: {exc}\n"
            "Ensure dm_control and h5py are installed:\n"
            "  pip install dm_control h5py"
        )

    try:
        db = load_database(subset=args.subset, use_cache=True)
    except RuntimeError as exc:
        sys.exit(f"MoCap Act database error:\n  {exc}")

    db_fps  = float(db["fps"])
    n_clips = len(db["file_boundaries"])
    db_secs = len(db["knee_right"]) / db_fps
    print(
        f"  {n_clips} clips  {db_secs:.0f} s  @ {db_fps:.0f} Hz  "
        f"(subset={args.subset!r})"
    )

    if BATCH_FRAMES > len(db["knee_right"]):
        sys.exit(
            f"Batch size ({BATCH_FRAMES} frames = {args.batch_secs:.1f} s) exceeds "
            f"the entire database ({len(db['knee_right'])} frames = {db_secs:.0f} s). "
            "Use --batch-secs with a smaller value or --subset all."
        )

    # ── STEPS 4–5: Per-batch matching and simulation ───────────────────────────
    all_results: List[dict] = []
    for batch in batches:
        result = run_batch(
            batch       = batch,
            db          = db,
            noise_std   = args.noise,
            use_gui     = USE_GUI,
            prefilter_k = args.prefilter_k,
            out_dir     = out_dir,
            n_total     = args.n_batches,
        )
        all_results.append(result)

    # ── STEP 6: Aggregate results ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("AGGREGATE SUMMARY")
    print("═" * 60)
    print(f"  Source   : {source_label}")
    print(f"  Batches  : {len(all_results)} × {args.batch_secs:.1f} s")
    print(f"  DB subset: {args.subset}")
    print(f"  Noise    : {args.noise:.1f}°")
    print()

    for r in all_results:
        bid = r["batch_id"]
        sg  = r["sim_results"]["gt"]
        sm  = r["sim_results"]["good"]
        print(
            f"  Batch {bid + 1}  clip={r['clip_id']}  "
            f"DTW={r['dtw_dist']:.4f}  knee-RMSE={r['match_rmse']:.1f}°"
        )
        print(f"    GT   : fall={sg['fall_detected']}  steps={sg['step_count']}  stab={sg['stability_score']:.3f}")
        print(f"    Model: fall={sm['fall_detected']}  steps={sm['step_count']}  stab={sm['stability_score']:.3f}")
    print()

    avg_dtw  = float(np.mean([r["dtw_dist"]   for r in all_results]))
    avg_rmse = float(np.mean([r["match_rmse"] for r in all_results]))
    avg_sg   = float(np.mean([r["sim_results"]["gt"]["stability_score"]   for r in all_results]))
    avg_sm   = float(np.mean([r["sim_results"]["good"]["stability_score"] for r in all_results]))

    print(f"  Mean DTW       : {avg_dtw:.4f}")
    print(f"  Mean knee RMSE : {avg_rmse:.1f}°")
    print(f"  Mean stab (GT) : {avg_sg:.3f}")
    print(f"  Mean stab (Md) : {avg_sm:.3f}")

    # ── STEP 7: Aggregate plot ─────────────────────────────────────────────────
    print()
    plot_aggregate_summary(all_results, str(out_dir / "virtual_sim_summary.png"))

    # ── JSON output ────────────────────────────────────────────────────────────
    if args.out:
        def _serialise(v):
            if isinstance(v, dict):
                return {k: _serialise(x) for k, x in v.items()}
            if isinstance(v, list):
                return [_serialise(x) for x in v]
            if hasattr(v, "item"):
                return v.item()
            if hasattr(v, "tolist"):
                return v.tolist()
            return v

        payload = {
            "source":           source_label,
            "subset":           args.subset,
            "n_batches":        args.n_batches,
            "batch_secs":       args.batch_secs,
            "noise_std_deg":    args.noise,
            "mean_dtw":         avg_dtw,
            "mean_match_rmse":  avg_rmse,
            "mean_stab_gt":     avg_sg,
            "mean_stab_model":  avg_sm,
            "batches":          _serialise(all_results),
        }
        out_file = Path(args.out)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  JSON saved → {args.out}")

    print(f"\n  Pipeline validation complete.")
    print(f"  Output directory: {out_dir.resolve()}")
    print("  Files:")
    for p in sorted(out_dir.iterdir()):
        print(f"    {p.name}")


if __name__ == "__main__":
    main()
