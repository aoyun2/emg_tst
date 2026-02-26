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
from typing import Optional

import numpy as np
import torch

from emg_tst.model import TSTEncoder, TSTRegressor
from emg_tst.data import StandardScaler
from mocap_evaluation.mocap_loader import (
    load_or_generate_mocap_database,
    load_full_cmu_database,
)
from mocap_evaluation.motion_matching import find_best_match
from mocap_evaluation.prosthetic_sim import simulate_prosthetic_walking


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


# ── Smoke test (no real data needed) ─────────────────────────────────────────


def run_smoke_test() -> dict:
    """
    Quick end-to-end test using synthetic signals throughout.
    No checkpoint or data file required.
    """
    print("=" * 60)
    print("SMOKE TEST — synthetic signals only")
    print("=" * 60)

    from mocap_evaluation.mocap_loader import generate_synthetic_gait

    fps   = 200
    T     = 400  # 2 seconds

    # Synthetic "IMU" signals: one gait cycle
    t     = np.linspace(0, 2 * math.pi, T)
    knee_imu   = (30 + 30 * np.sin(t)).astype(np.float32)    # 0–60°
    thigh_imu  = (15 * np.sin(t + 0.5)).astype(np.float32)   # ±15°

    # Synthetic "model prediction" — adds small random error to ground truth
    rng = np.random.default_rng(42)
    predicted  = knee_imu + rng.normal(0, 3, T).astype(np.float32)

    print(f"  Query length: {T/fps:.1f} s @ {fps} Hz")

    # Generate mocap database
    db = generate_synthetic_gait(n_cycles=20)
    print(f"  Mocap database: {len(db['knee_right'])/fps:.1f} s, source={db['source']}")

    # Motion matching
    t0 = time.time()
    start, dist, segment = find_best_match(knee_imu, thigh_imu, db)
    print(f"  Motion match: start={start}, DTW dist={dist:.4f}  ({time.time()-t0:.2f}s)")

    # Simulation
    print("  Running simulation …")
    t1 = time.time()
    metrics = simulate_prosthetic_walking(
        segment, predicted, use_physics=True, use_gui=False, fps=float(fps)
    )
    print(f"  Simulation done in {time.time()-t1:.2f}s")

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")

    return metrics


# ── Main evaluation loop ──────────────────────────────────────────────────────


def evaluate(
    checkpoint_path: str | Path,
    samples_path: str | Path,
    mocap_dir: str | Path = "mocap_data",
    out_path: str | Path  = "eval_results.json",
    n_samples: Optional[int] = None,
    use_gui: bool = False,
    use_physics: bool = True,
    device_str: str = "cpu",
    categories: Optional[list] = None,
    full_database: bool = False,
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
    if full_database:
        mocap_db = load_full_cmu_database(
            bvh_dir=mocap_dir,
            categories=categories,
        )
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

        # Motion matching
        _, dtw_dist, segment = find_best_match(knee_label, thigh_sig, mocap_db)

        # Simulation
        metrics = simulate_prosthetic_walking(
            segment, pred_knee,
            use_physics=use_physics,
            use_gui=use_gui,
        )

        metrics["window_idx"]  = i
        metrics["dtw_dist"]    = float(dtw_dist)
        metrics["pred_rmse"]   = float(
            np.sqrt(np.mean((pred_knee - knee_label) ** 2))
        )
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
    ap.add_argument("--gui",         action="store_true",
                    help="Show PyBullet GUI (requires display)")
    ap.add_argument("--no-physics",  action="store_true",
                    help="Use kinematic evaluation only (no PyBullet)")
    ap.add_argument("--smoke-test",  action="store_true",
                    help="Run quick smoke test with synthetic data (no files needed)")
    ap.add_argument("--full-db",     action="store_true",
                    help="Use the full CMU mocap database (auto-downloads if needed)")
    ap.add_argument("--categories",  nargs="*", default=None,
                    help="Motion categories to match against (e.g. walk run jump). "
                         "Only effective with --full-db.")
    return ap.parse_args()


def main():
    args = _parse_args()

    if args.smoke_test:
        run_smoke_test()
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
        use_gui         = args.gui,
        use_physics     = not args.no_physics,
        device_str      = args.device,
        categories      = args.categories,
        full_database   = args.full_db,
    )


if __name__ == "__main__":
    main()
