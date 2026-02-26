"""
Main prosthetic evaluation pipeline.

Usage
-----
    # With a trained checkpoint and a data file:
    python -m mocap_evaluation.run_evaluation \\
        --checkpoint checkpoints/tst_20240101_120000/fold_01/reg_best.pt \\
        --data        samples_dataset.npy \\
        --mocap-dir   mocap_data/ \\
        --out         eval_results.json \\
        --n-samples   20

    # Quick smoke test (no data needed — uses synthetic everything):
    python -m mocap_evaluation.run_evaluation --smoke-test

Pipeline per test window
------------------------
1.  Load model checkpoint → TSTRegressor
2.  Load samples_dataset.npy  (or a raw .npy recording)
3.  For each test window:
      a. Run model forward pass  → predicted knee angles  (T,)
      b. Extract ground-truth imu + thigh_angle            (T,)
      c. Motion-match against mocap database               → segment (all joints)
      d. Simulate: mocap drives all joints except right knee (model prediction)
      e. Collect metrics
4.  Aggregate across windows → summary JSON
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from emg_tst.model import TSTEncoder, TSTRegressor
from emg_tst.data import StandardScaler
from mocap_evaluation.mocap_loader import (
    load_or_generate_mocap_database,
    load_full_cmu_database,
    load_aggregated_bandai_cmu_database,
)
from mocap_evaluation.motion_matching import find_best_match, find_top_k_matches
from mocap_evaluation.prosthetic_sim import simulate_prosthetic_walking
from mocap_evaluation.mock_data import generate_mock_curves, save_mock_curves
from mocap_evaluation.sample_data import extract_real_sample_curves, save_sample_curves


# ── Checkpoint loader ─────────────────────────────────────────────────────────


def load_checkpoint(path: str | Path, device: torch.device) -> tuple:
    """
    Load a reg_best.pt checkpoint saved by run_experiment.py.

    Returns
    -------
    model   : TSTRegressor (eval mode)
    scaler  : StandardScaler
    cfg     : dict  (model_cfg)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["model_cfg"]

    encoder = TSTEncoder(
        n_vars   = cfg["n_vars"],
        seq_len  = cfg["seq_len"],
        d_model  = cfg["d_model"],
        n_heads  = cfg["n_heads"],
        d_ff     = cfg["d_ff"],
        n_layers = cfg["n_layers"],
        dropout  = cfg.get("dropout", 0.1),
    )
    model = TSTRegressor(encoder, out_dim=1)
    model.load_state_dict(ckpt["reg_state_dict"])
    model.eval().to(device)

    sc = StandardScaler(
        mean_ = np.array(ckpt["scaler"]["mean"], dtype=np.float32),
        std_  = np.array(ckpt["scaler"]["std"],  dtype=np.float32),
    )
    return model, sc, cfg


# ── Dataset loader (samples_dataset.npy) ─────────────────────────────────────


def load_samples(
    samples_path: str | Path,
    max_samples: Optional[int] = None,
) -> tuple:
    """
    Load samples_dataset.npy produced by split_to_samples.py.

    Returns
    -------
    X         : (N, W, F) raw features
    y_seq     : (N, W) knee angle labels per timestep (degrees)
    thigh_col : int  column index of thigh_angle feature in X
    """
    data  = np.load(samples_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()      # allow_pickle=True returns a 0-d object array

    X     = data["X"].astype(np.float32)      # (N, W, F)
    y_seq = data["y_seq"].astype(np.float32)  # (N, W)

    if max_samples is not None:
        X     = X[:max_samples]
        y_seq = y_seq[:max_samples]

    # Thigh angle is the last feature column (from data.py load_recording)
    thigh_col = X.shape[2] - 1

    return X, y_seq, thigh_col


# ── Per-window inference ──────────────────────────────────────────────────────


@torch.no_grad()
def predict_knee_sequence(
    model: TSTRegressor,
    scaler: StandardScaler,
    x_window: np.ndarray,            # (W, F)
    feature_cols: Optional[list],
    device: torch.device,
) -> np.ndarray:
    """
    Run one 1-second window through the model.

    Returns
    -------
    pred : (W,) predicted knee angle in degrees (raw output, not normalised)
    """
    x = (x_window - scaler.mean_) / scaler.std_    # (W, F)
    if feature_cols is not None:
        x = x[:, feature_cols]
    x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)  # (1, W, n_vars)
    out = model(x_t)   # (1, W, 1)
    return out[0, :, 0].cpu().numpy()


# ── Smoke test ────────────────────────────────────────────────────────────────


def run_smoke_test(try_download: bool = True, top_k: int = 3) -> dict:
    """
    End-to-end pipeline test: Winter (2009) query -> CMU mocap matching -> PyBullet.

    Query generation
    ----------------
    Uses Winter (2009) published biomechanical norms for normal gait with
    realistic IMU sensor noise added — independent of the CMU database being
    searched.

    Matching
    --------
    Searches the **entire** database without category restriction; the query can
    match walking, running, or any other motion if that is the closest segment.
    The top-K distinct matches are returned (default K=3) and each receives its
    own simulation run and a pair of GIF files:
      smoke_test_<rank>_pred.gif  — prosthetic robot (orange right knee)
      smoke_test_<rank>_gt.gif    — ground-truth mocap robot (blue)

    Two RMSE values per match:
      model_rmse   : prediction vs true query knee (reflects model noise level)
      match_rmse   : matched segment vs query knee  (reflects DB quality)
      knee_rmse_deg: pred vs matched segment (sim metric = match + model error)
    """
    print("=" * 60)
    print("SMOKE TEST -- Winter (2009) query -> CMU mocap matching -> PyBullet")
    print("=" * 60)

    from mocap_evaluation.mocap_loader import (
        load_or_generate_mocap_database,
        _interp_gait_curve,
        _KNEE_R,
        _HIP_R,
        TARGET_FPS,
    )

    # ── Load real mocap database (no synthetic augmentation) ──────────────
    print()
    print("  Loading mocap database ...")
    db = load_or_generate_mocap_database(try_download=try_download)

    fps = TARGET_FPS   # 200 Hz
    db_dur = len(db["knee_right"]) / fps
    n_cats = len({b[3] for b in db.get("file_boundaries", [])})
    print(f"  DB: {db_dur:.1f}s @ {fps}Hz  source={db['source']}  {n_cats} categories")

    # ── Build query from Winter (2009) biomechanical norms ────────────────
    CADENCE  = 110.0
    cycle_s  = 60.0 / (CADENCE / 2.0)
    spc      = int(round(cycle_s * fps))
    N_CYCLES = 3
    T        = spc * N_CYCLES    # ≈654 frames at 200 Hz

    rng = np.random.default_rng(42)

    knee_true  = np.tile(_interp_gait_curve(_KNEE_R, spc), N_CYCLES).astype(np.float32)
    thigh_true = np.tile(_interp_gait_curve(_HIP_R,  spc), N_CYCLES).astype(np.float32)
    knee_imu   = knee_true  + rng.normal(0, 2.0, T).astype(np.float32)
    thigh_imu  = thigh_true + rng.normal(0, 1.5, T).astype(np.float32)

    # Simulated model prediction: true knee + ~1.5° RMS error (very low noise)
    predicted  = knee_true + rng.normal(0, 1.5, T).astype(np.float32)
    model_rmse = float(np.sqrt(np.mean((predicted - knee_true) ** 2)))

    print(f"  Query   : {T} frames ({T/fps:.2f}s), Winter (2009) norms + IMU noise")
    print(f"  Knee    : {knee_imu.min():.1f}° – {knee_imu.max():.1f}°")
    print(f"  Model noise (pred vs true): {model_rmse:.2f}° RMS")

    # ── Motion matching — no category restriction ─────────────────────────
    # Any segment in the database is eligible; multiple plausible matches are
    # retrieved so each can be visualised independently.
    print()
    print(f"  Finding top-{top_k} matches (no category restriction) …")
    t0      = time.time()
    matches = find_top_k_matches(
        knee_imu, thigh_imu, db,
        k=top_k,
        # No `categories` argument → search entire database
    )
    print(f"  Matching done in {time.time()-t0:.2f}s")

    all_metrics: List[dict] = []

    for rank, (start, dist, segment) in enumerate(matches, start=1):
        cat = segment.get("category", "unknown")
        match_rmse = float(np.sqrt(np.mean(
            (segment["knee_right"] - knee_imu) ** 2
        )))
        print()
        print(f"  ── Match {rank}/{top_k} ─────────────────────────────")
        print(f"     start={start}  DTW={dist:.4f}  category={cat}")
        print(f"     Segment vs query RMSE: {match_rmse:.2f}°"
              f"  ({'good' if match_rmse < 12 else 'fair — more BVH files improve this'})")

        gif_pred = f"smoke_test_{rank}_pred.gif"
        gif_gt   = f"smoke_test_{rank}_gt.gif"

        print(f"     Running PyBullet simulation …")
        t1 = time.time()
        metrics = simulate_prosthetic_walking(
            segment, predicted,
            use_physics=True, use_gui=False, fps=float(fps),
            gif_output_pred=gif_pred,
            gif_output_gt=gif_gt,
        )
        elapsed = time.time() - t1
        print(f"     Simulation: {elapsed:.2f}s  mode={metrics.get('mode')}")

        for path in (gif_pred, gif_gt):
            if Path(path).exists():
                label = "prosthetic (orange knee)" if "pred" in path else "ground-truth (blue)"
                print(f"     GIF [{label}] → {path}")

        metrics["match_rank"]  = rank
        metrics["start_idx"]   = start
        metrics["dtw_dist"]    = float(dist)
        metrics["category"]    = cat
        metrics["model_rmse"]  = model_rmse
        metrics["match_rmse"]  = match_rmse
        metrics["db_source"]   = db["source"]
        all_metrics.append(metrics)

        print(f"     knee_rmse_deg={metrics['knee_rmse_deg']:.2f}°  "
              f"fall={metrics['fall_detected']}  "
              f"stability={metrics['stability_score']:.3f}")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("── Summary ─────────────────────────────────────────────────")
    print(f"  model_rmse  : {model_rmse:.2f}°  (pred vs true query knee)")
    for m in all_metrics:
        print(
            f"  Match {m['match_rank']} [{m['category']:8s}] "
            f"match_rmse={m['match_rmse']:5.2f}°  "
            f"knee_rmse={m['knee_rmse_deg']:5.2f}°  "
            f"fall={m['fall_detected']}"
        )
    print()
    print("  knee_rmse_deg ≈ model_rmse + residual cross-subject match error.")
    print("  Download more BVH files for accuracy (recommended first):")
    print("    python -m mocap_evaluation.bandai_namco_downloader")
    print("  Optional fallback/source expansion:")
    print("    python -m mocap_evaluation.cmu_downloader")

    return all_metrics[0] if all_metrics else {}


# ── Main evaluation loop ──────────────────────────────────────────────────────


def evaluate(
    checkpoint_path: str | Path,
    samples_path: str | Path,
    mocap_dir: str | Path = "mocap_data",
    out_path: str | Path  = "eval_results.json",
    n_samples: Optional[int] = None,
    use_gui: bool = True,
    use_physics: bool = True,
    device_str: str = "cpu",
    full_database: bool = False,
    sim_backend: str = "pybullet",
    aggregate_datasets: bool = False,
    bandai_dir: Optional[str | Path] = None,
    cmu_dir: Optional[str | Path] = None,
    match_categories: Optional[List[str]] = None,
) -> dict:
    """
    Full prosthetic evaluation pipeline.

    Returns
    -------
    dict  with keys 'per_window' (list) and 'summary' (aggregated stats)
    """
    device = torch.device(device_str)
    print(f"[eval] Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[eval] Loading checkpoint: {checkpoint_path}")
    model, scaler, model_cfg = load_checkpoint(checkpoint_path, device)
    feature_cols = model_cfg.get("extra", {}).get("feature_cols", None)
    print(f"       n_vars={model_cfg['n_vars']}  seq_len={model_cfg['seq_len']}")

    # ── Load samples ────────────────────────────────────────────────────────
    print(f"[eval] Loading samples: {samples_path}")
    X, y_seq, thigh_col = load_samples(samples_path, max_samples=n_samples)
    N = len(X)
    print(f"       {N} windows  shape={X.shape}")

    # ── Load / generate mocap database ──────────────────────────────────────
    print(f"[eval] Loading mocap database from: {mocap_dir}")
    if aggregate_datasets:
        bandai_path = Path(bandai_dir) if bandai_dir else Path(mocap_dir) / "bandai"
        cmu_path = Path(cmu_dir) if cmu_dir else Path(mocap_dir) / "cmu"
        mocap_db = load_aggregated_bandai_cmu_database(
            bandai_dir=bandai_path,
            cmu_dir=cmu_path,
            try_download=True,
        )
    elif full_database:
        mocap_db = load_full_cmu_database(bvh_dir=mocap_dir)
    else:
        mocap_db = load_or_generate_mocap_database(bvh_dir=mocap_dir)
    db_dur   = len(mocap_db["knee_right"]) / mocap_db["fps"]
    n_files = len(mocap_db.get("file_boundaries", []))
    extra = f", {n_files} files" if n_files else ""
    print(f"       {db_dur:.1f} s @ {mocap_db['fps']:.0f} Hz  "
          f"(source: {mocap_db['source']}{extra})")

    # ── Per-window loop ──────────────────────────────────────────────────────
    per_window = []
    for i in range(N):
        t0 = time.time()

        x_win      = X[i]                    # (W, F)
        knee_label = y_seq[i]                # (W,) ground-truth knee angle (deg)
        thigh_sig  = x_win[:, thigh_col]     # (W,) raw thigh angle feature

        # Model prediction
        pred_knee = predict_knee_sequence(
            model, scaler, x_win, feature_cols, device
        )

        # Standardized convention: 180° = straight (rigtest enclosed angle).
        # BVH-derived database angles are normalized to the same convention.
        knee_label_inc = knee_label.astype(np.float32)
        pred_knee_inc = pred_knee.astype(np.float32)

        # Motion matching (included-angle convention)
        _, dtw_dist, segment = find_best_match(
            knee_label_inc,
            thigh_sig,
            mocap_db,
            categories=match_categories,
        )

        # Simulation consumes the same included-angle convention.
        metrics = simulate_prosthetic_walking(
            segment, pred_knee_inc,
            use_physics=use_physics,
            use_gui=use_gui,
            backend=sim_backend,
        )

        metrics["window_idx"]  = i
        metrics["dtw_dist"]    = float(dtw_dist)
        metrics["pred_rmse"]   = float(
            np.sqrt(np.mean((pred_knee - knee_label) ** 2))
        )  # RMSE is invariant to the 180-x shift — compute in original convention
        metrics["elapsed_s"]   = float(time.time() - t0)
        per_window.append(metrics)

        print(
            f"  [{i+1:3d}/{N}] "
            f"pred_rmse={metrics['pred_rmse']:.2f}°  "
            f"knee_rmse={metrics['knee_rmse_deg']:.2f}°  "
            f"fall={metrics['fall_detected']}  "
            f"stab={metrics['stability_score']:.3f}  "
            f"({metrics['elapsed_s']:.1f}s)"
        )

    # ── Aggregate summary ────────────────────────────────────────────────────
    def _agg(key):
        vals = [w[key] for w in per_window if key in w]
        if not vals:
            return {}
        arr = np.array(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max())}

    fall_rate = float(np.mean([w["fall_detected"] for w in per_window]))

    summary = {
        "n_windows":       N,
        "fall_rate":       fall_rate,
        "pred_rmse":       _agg("pred_rmse"),
        "knee_rmse_deg":   _agg("knee_rmse_deg"),
        "stability_score": _agg("stability_score"),
        "dtw_dist":        _agg("dtw_dist"),
        "com_height_std":  _agg("com_height_std"),
        "gait_symmetry":   _agg("gait_symmetry"),
        "mode":            per_window[0].get("mode", "unknown") if per_window else "unknown",
        "mocap_source":    mocap_db["source"],
    }

    result = {"per_window": per_window, "summary": summary}

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[eval] Results saved → {out_path}")
    print("[eval] Summary:")
    for k, v in summary.items():
        print(f"  {k:<22} {v}")

    return result


def evaluate_from_curves(
    knee_label_included: np.ndarray,
    thigh_angle: np.ndarray,
    predicted_knee_included: np.ndarray,
    mocap_dir: str | Path = "mocap_data",
    top_k: int = 3,
    use_gui: bool = True,
    use_physics: bool = True,
    out_path: str | Path = "eval_mock_results.json",
    full_database: bool = True,
    sim_backend: str = "pybullet",
    aggregate_datasets: bool = False,
    bandai_dir: Optional[str | Path] = None,
    cmu_dir: Optional[str | Path] = None,
    match_categories: Optional[List[str]] = None,
) -> dict:
    """Evaluate motion matching directly from label/thigh curves.

    This path is designed for early feasibility testing before recorded data is
    available from ``rigtest.py``.
    """
    knee_label_inc = knee_label_included.astype(np.float32)
    pred_knee_inc = predicted_knee_included.astype(np.float32)

    if aggregate_datasets:
        bandai_path = Path(bandai_dir) if bandai_dir else Path(mocap_dir) / "bandai"
        cmu_path = Path(cmu_dir) if cmu_dir else Path(mocap_dir) / "cmu"
        mocap_db = load_aggregated_bandai_cmu_database(
            bandai_dir=bandai_path,
            cmu_dir=cmu_path,
            try_download=True,
        )
    elif full_database:
        mocap_db = load_full_cmu_database(bvh_dir=mocap_dir)
    else:
        mocap_db = load_or_generate_mocap_database(bvh_dir=mocap_dir)

    matches = find_top_k_matches(
        imu_knee=knee_label_inc,
        imu_thigh=thigh_angle.astype(np.float32),
        mocap_db=mocap_db,
        k=top_k,
        categories=match_categories,
    )

    per_match = []
    for rank, (start, dist, segment) in enumerate(matches, start=1):
        gt_metrics = simulate_prosthetic_walking(
            segment,
            knee_label_inc,
            use_physics=use_physics,
            use_gui=use_gui,
            backend=sim_backend,
        )
        pred_metrics = simulate_prosthetic_walking(
            segment,
            pred_knee_inc,
            use_physics=use_physics,
            use_gui=use_gui,
            backend=sim_backend,
        )
        per_match.append(
            {
                "match_rank": rank,
                "start_idx": int(start),
                "dtw_dist": float(dist),
                "category": segment.get("category", "unknown"),
                "ground_truth_replaced_right_knee": gt_metrics,
                "prediction_replaced_right_knee": pred_metrics,
                "pred_vs_label_rmse_deg": float(
                    np.sqrt(np.mean((predicted_knee_included - knee_label_included) ** 2))
                ),
            }
        )

    result = {
        "mode": "curve_direct",
        "n_frames": int(len(knee_label_included)),
        "top_k": int(top_k),
        "mocap_source": mocap_db.get("source", "unknown"),
        "matches": per_match,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] Curve-direct results saved -> {out_path}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate a prosthetic knee model via motion-capture simulation"
    )
    ap.add_argument("--checkpoint",  default=None,
                    help="Path to reg_best.pt checkpoint")
    ap.add_argument("--data",        default="samples_dataset.npy",
                    help="Path to samples_dataset.npy")
    ap.add_argument("--mocap-dir",   default="mocap_data",
                    help="Directory containing BVH files (or where to download them)")
    ap.add_argument("--out",         default="eval_results.json",
                    help="Output JSON file for metrics")
    ap.add_argument("--n-samples",   type=int, default=None,
                    help="Limit number of test windows (None = all)")
    ap.add_argument("--device",      default="cpu",
                    help="torch device (cpu / cuda)")
    ap.add_argument("--no-gui",      action="store_true",
                    help="Disable simulation GUI (GUI is on by default when backend supports it)")
    ap.add_argument("--no-physics",  action="store_true",
                    help="Use kinematic evaluation only (no physics backend)")
    ap.add_argument("--smoke-test",  action="store_true",
                    help="Run quick smoke test (no checkpoint needed)")
    ap.add_argument("--full-db",     action="store_true",
                    help="Use the full local mocap database with category metadata "
                         "(auto-downloads Bandai locomotion first, CMU fallback)")
    ap.add_argument("--mock-data", action="store_true",
                    help="Run with generated mock knee/thigh curves (no checkpoint needed)")
    ap.add_argument("--mock-seconds", type=float, default=4.0,
                    help="Mock curve length in seconds")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Top-k motion matches to simulate")
    ap.add_argument("--save-mock", default=None,
                    help="Optional .npz path to save generated mock curves")
    ap.add_argument("--real-walk-data", action="store_true",
                    help="Use a real walking segment from mocap DB (thigh pitch + knee included angle only)")
    ap.add_argument("--save-real", default=None,
                    help="Optional .npz path to save extracted real walking curves")
    ap.add_argument("--real-seconds", type=float, default=4.0,
                    help="Length in seconds for extracted real walking curves")
    ap.add_argument("--match-categories", default=None,
                    help="Comma-separated category filter for motion matching (e.g. walk,run)")
    ap.add_argument("--sim-backend", default="pybullet", choices=["pybullet", "mujoco"],
                    help="Physics backend preference")
    ap.add_argument("--aggregate-datasets", action="store_true",
                    help="Aggregate Bandai + CMU datasets (separate dirs under mocap-dir by default)")
    ap.add_argument("--bandai-dir", default=None,
                    help="Bandai BVH directory (default: <mocap-dir>/bandai when aggregating)")
    ap.add_argument("--cmu-dir", default=None,
                    help="CMU BVH directory (default: <mocap-dir>/cmu when aggregating)")
    return ap.parse_args()


def main():
    args = _parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    match_categories = None
    if args.match_categories:
        match_categories = [c.strip() for c in args.match_categories.split(",") if c.strip()]

    if args.real_walk_data:
        curves = extract_real_sample_curves(
            mocap_dir=args.mocap_dir,
            seconds=args.real_seconds,
            categories=match_categories,
            full_database=args.full_db or args.aggregate_datasets,
        )
        if args.save_real:
            save_sample_curves(args.save_real, curves)
            print(f"[eval] Saved real walking curves -> {args.save_real}")
        print(f"[eval] Real sample source: {curves.source_file} [{curves.category}]")
        evaluate_from_curves(
            knee_label_included=curves.knee_label_included_deg,
            thigh_angle=curves.thigh_angle_deg,
            predicted_knee_included=curves.predicted_knee_included_deg,
            mocap_dir=args.mocap_dir,
            top_k=args.top_k,
            use_gui=not args.no_gui,
            use_physics=not args.no_physics,
            out_path=args.out,
            full_database=args.full_db,
            sim_backend=args.sim_backend,
            aggregate_datasets=args.aggregate_datasets,
            bandai_dir=args.bandai_dir,
            cmu_dir=args.cmu_dir,
            match_categories=match_categories,
        )
        return

    if args.mock_data:
        curves = generate_mock_curves(length_s=args.mock_seconds)
        if args.save_mock:
            save_mock_curves(args.save_mock, curves)
            print(f"[eval] Saved mock curves -> {args.save_mock}")
        evaluate_from_curves(
            knee_label_included=curves.knee_label_included_deg,
            thigh_angle=curves.thigh_angle_deg,
            predicted_knee_included=curves.predicted_knee_included_deg,
            mocap_dir=args.mocap_dir,
            top_k=args.top_k,
            use_gui=not args.no_gui,
            use_physics=not args.no_physics,
            out_path=args.out,
            full_database=args.full_db,
            sim_backend=args.sim_backend,
            aggregate_datasets=args.aggregate_datasets,
            bandai_dir=args.bandai_dir,
            cmu_dir=args.cmu_dir,
            match_categories=match_categories,
        )
        return

    if args.checkpoint is None:
        # Try to find the most recent checkpoint automatically
        ckpt_dir = Path("checkpoints")
        candidates = sorted(ckpt_dir.glob("*/fold_01/reg_best.pt"), reverse=True)
        if not candidates:
            print(
                "ERROR: No checkpoint found. Either:\n"
                "  1. Train a model first:  python -m emg_tst.run_experiment\n"
                "  2. Specify --checkpoint path/to/reg_best.pt\n"
                "  3. Run smoke test:       --smoke-test",
                file=sys.stderr,
            )
            sys.exit(1)
        args.checkpoint = str(candidates[0])
        print(f"[eval] Auto-selected checkpoint: {args.checkpoint}")

    if not Path(args.data).exists():
        print(
            f"ERROR: Data file not found: {args.data}\n"
            "  Run: python split_to_samples.py   (needs recorded data files)",
            file=sys.stderr,
        )
        sys.exit(1)

    evaluate(
        checkpoint_path = args.checkpoint,
        samples_path    = args.data,
        mocap_dir       = args.mocap_dir,
        out_path        = args.out,
        n_samples       = args.n_samples,
        use_gui         = not args.no_gui,
        use_physics     = not args.no_physics,
        device_str      = args.device,
        full_database   = args.full_db,
        sim_backend      = args.sim_backend,
        aggregate_datasets = args.aggregate_datasets,
        bandai_dir        = args.bandai_dir,
        cmu_dir           = args.cmu_dir,
        match_categories   = match_categories,
    )


if __name__ == "__main__":
    main()
