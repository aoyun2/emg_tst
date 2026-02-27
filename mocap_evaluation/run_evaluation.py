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

    # Quick test sample run (no checkpoint needed):
    python -m mocap_evaluation.run_evaluation --test-sample

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
from tqdm import tqdm

from emg_tst.model import TSTEncoder, TSTRegressor
from emg_tst.data import StandardScaler
from mocap_evaluation.mocap_loader import (
    TARGET_FPS,
    load_aggregated_database,
)
from mocap_evaluation.motion_matching import find_best_match, find_top_k_matches
from mocap_evaluation.prosthetic_sim import simulate_prosthetic_walking, FALL_HEIGHT_THRESHOLD
from mocap_evaluation.mock_data import generate_mock_curves, save_mock_curves
from mocap_evaluation.sample_data import extract_real_sample_curves, save_sample_curves
from mocap_evaluation.external_sample_data import extract_external_sample_curves


# ── Simulation visualization ─────────────────────────────────────────────────


def plot_simulation(
    metrics: dict,
    title: str,
    out_path: str | Path,
    fps: float = 200.0,
) -> None:
    """Save a 3-panel plot of simulation results (knee angles, CoM, contacts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    com = np.asarray(metrics.get("com_height_series", []))
    pred = np.asarray(metrics.get("pred_knee_series", []))
    ref = np.asarray(metrics.get("ref_knee_series", []))
    rc = metrics.get("right_contact_frames", [])
    lc = metrics.get("left_contact_frames", [])

    if pred.size == 0:
        return

    T = len(pred)
    t = np.arange(T) / fps

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Panel 1: Knee angles (flexion convention as stored)
    ax = axes[0]
    ax.plot(t, ref, label="Reference (mocap)", linewidth=1.5, alpha=0.8)
    ax.plot(t, pred, label="Predicted (model)", linewidth=1.5)
    if metrics.get("fall_detected") and metrics["fall_frame"] >= 0:
        ff = metrics["fall_frame"] / fps
        ax.axvline(ff, color="red", linestyle="--", alpha=0.6, label=f"Fall @ {ff:.2f}s")
    ax.set_ylabel("Knee flexion (deg)")
    ax.set_title(f"{title}  |  RMSE={metrics.get('knee_rmse_deg', 0):.2f}°  "
                 f"Stability={metrics.get('stability_score', 0):.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: CoM height
    ax = axes[1]
    if com.size > 0:
        ax.plot(t[:len(com)], com, linewidth=1.2, color="steelblue")
        ax.axhline(FALL_HEIGHT_THRESHOLD, color="red", linestyle="--",
                    alpha=0.5, label=f"Fall threshold ({FALL_HEIGHT_THRESHOLD}m)")
        ax.set_ylabel("CoM height (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 3: Foot contacts
    ax = axes[2]
    if rc:
        rc_t = np.asarray(rc) / fps
        ax.scatter(rc_t, np.ones(len(rc_t)), marker="|", s=30, color="tab:orange", label="Right foot")
    if lc:
        lc_t = np.asarray(lc) / fps
        ax.scatter(lc_t, np.zeros(len(lc_t)), marker="|", s=30, color="tab:blue", label="Left foot")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Left", "Right"])
    ax.set_ylabel("Foot contacts")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[viz] Simulation plot saved -> {out_path}")


# ── Window-length helper ──────────────────────────────────────────────────────


def _derive_window_seconds(
    data_path: str | Path | None,
    checkpoint_path: str | Path | None = None,
) -> float:
    """Derive sample window duration (in seconds) from the model architecture.

    When both *checkpoint_path* and *data_path* are provided the two sources
    must agree on the window length — the checkpoint's ``seq_len`` must equal
    the dataset's ``window`` field.  A mismatch indicates that the wrong
    checkpoint/dataset pair is being used and raises ``ValueError``.

    Priority (when only one source is available):
    1. Checkpoint ``model_cfg["seq_len"]`` (exact model sequence length).
    2. Dataset ``window`` field from *samples_dataset.npy*.
    3. Architecture default: 200 samples @ 200 Hz = 1.0 s.
    """
    ckpt_seq_len: int | None = None
    dataset_window: int | None = None

    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists():
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                ckpt_seq_len = int(ckpt["model_cfg"]["seq_len"])
            except Exception:
                pass

    if data_path is not None:
        p = Path(data_path)
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if isinstance(d, np.ndarray):
                d = d.item()
            dataset_window = int(d.get("window", 200))

    if ckpt_seq_len is not None and dataset_window is not None:
        if ckpt_seq_len != dataset_window:
            raise ValueError(
                f"Length mismatch: checkpoint seq_len={ckpt_seq_len} "
                f"!= dataset window={dataset_window}. "
                "Ensure the checkpoint and dataset were produced with the same window size."
            )
        return ckpt_seq_len / TARGET_FPS

    if ckpt_seq_len is not None:
        return ckpt_seq_len / TARGET_FPS

    if dataset_window is not None:
        return dataset_window / TARGET_FPS

    return 200 / TARGET_FPS


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


# ── Test sample evaluation ───────────────────────────────────────────────────


def run_test_sample(
    mocap_dir: str | Path = "mocap_data",
    out_path: str | Path = "eval_test_sample_results.json",
    top_k: int = 3,
    match_categories: Optional[List[str]] = None,
    seconds: float = 4.0,
    sample_source: str = "external",
    external_url: Optional[str] = None,
) -> dict:
    """Run the motion-matching pipeline using real test sample curves."""
    print("=" * 60)
    print("TEST SAMPLE -- real mocap query -> motion matching -> simulation")
    print("=" * 60)

    categories = tuple(match_categories) if match_categories else ("walk",)
    if sample_source == "external":
        curves = extract_external_sample_curves(
            seconds=seconds,
            source_url=external_url,
        )
    else:
        curves = extract_real_sample_curves(
            mocap_dir=mocap_dir,
            seconds=seconds,
            categories=categories,
        )
    model_rmse = float(np.sqrt(np.mean(
        (curves.predicted_knee_included_deg - curves.knee_label_included_deg) ** 2
    )))

    print(
        f"[eval] Real test sample: {len(curves.knee_label_included_deg)} frames @ {curves.fps} Hz"
    )
    print(f"[eval] Source file: {curves.source_file} [{curves.category}]")
    print(f"[eval] Pred-vs-label RMSE (model surrogate): {model_rmse:.2f}°")

    result = evaluate_from_curves(
        knee_label_included=curves.knee_label_included_deg,
        thigh_angle=curves.thigh_angle_deg,
        predicted_knee_included=curves.predicted_knee_included_deg,
        mocap_dir=mocap_dir,
        top_k=top_k,
        out_path=out_path,
        match_categories=match_categories,
    )
    result["test_sample_pred_vs_label_rmse_deg"] = model_rmse
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ── Main evaluation loop ──────────────────────────────────────────────────────


def evaluate(
    checkpoint_path: str | Path,
    samples_path: str | Path,
    mocap_dir: str | Path = "mocap_data",
    out_path: str | Path  = "eval_results.json",
    n_samples: Optional[int] = None,
    device_str: str = "cpu",
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
    mocap_db = load_aggregated_database(mocap_root=mocap_dir, try_download=True)
    db_dur   = len(mocap_db["knee_right"]) / mocap_db["fps"]
    n_files = len(mocap_db.get("file_boundaries", []))
    extra = f", {n_files} files" if n_files else ""
    print(f"       {db_dur:.1f} s @ {mocap_db['fps']:.0f} Hz  "
          f"(source: {mocap_db['source']}{extra})")

    # ── Per-window loop ──────────────────────────────────────────────────────
    per_window = []
    for i in tqdm(range(N), desc="Evaluating windows", unit="win"):
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
        # The sample's thigh angle drives the right hip actuator so that
        # both the right knee and right thigh come from the sample, not mocap.
        metrics = simulate_prosthetic_walking(
            segment, pred_knee_inc,
            use_gui=False,
            sample_thigh_right=thigh_sig,
        )

        metrics["window_idx"]  = i
        metrics["dtw_dist"]    = float(dtw_dist)
        metrics["pred_rmse"]   = float(
            np.sqrt(np.mean((pred_knee - knee_label) ** 2))
        )  # RMSE is invariant to the 180-x shift — compute in original convention
        metrics["elapsed_s"]   = float(time.time() - t0)
        per_window.append(metrics)

        # Save plots for first 5 windows
        if i < 5:
            plot_dir = Path(out_path).with_suffix("") / "plots"
            plot_simulation(metrics, f"Window {i}",
                            plot_dir / f"window_{i:04d}_sim.png")

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
    out_path: str | Path = "eval_mock_results.json",
    match_categories: Optional[List[str]] = None,
) -> dict:
    """Evaluate motion matching directly from label/thigh curves.

    This path is designed for early feasibility testing before recorded data is
    available from ``rigtest.py``.
    """
    knee_label_inc = knee_label_included.astype(np.float32)
    pred_knee_inc = predicted_knee_included.astype(np.float32)

    mocap_db = load_aggregated_database(mocap_root=mocap_dir, try_download=True)

    matches = find_top_k_matches(
        imu_knee=knee_label_inc,
        imu_thigh=thigh_angle.astype(np.float32),
        mocap_db=mocap_db,
        k=top_k,
        categories=match_categories,
    )

    per_match = []
    for rank, (start, dist, segment) in enumerate(tqdm(matches, desc="Simulating matches", unit="match"), start=1):
        # Both simulations use the sample's own thigh angle for the right hip
        # actuator so that the right leg (thigh + knee) is always driven by
        # the sample being evaluated, not by the matched mocap reference.
        gt_metrics = simulate_prosthetic_walking(
            segment,
            knee_label_inc,
            sample_thigh_right=thigh_angle,
        )
        pred_metrics = simulate_prosthetic_walking(
            segment,
            pred_knee_inc,
            sample_thigh_right=thigh_angle,
        )
        cat = segment.get("category", "unknown")

        # Save simulation plots
        plot_dir = Path(out_path).with_suffix("") / f"match_{rank:02d}_{cat}"
        plot_simulation(gt_metrics, f"Ground Truth — match #{rank} [{cat}]",
                        plot_dir / "ground_truth_sim.png", fps=mocap_db["fps"])
        plot_simulation(pred_metrics, f"Prediction — match #{rank} [{cat}]",
                        plot_dir / "prediction_sim.png", fps=mocap_db["fps"])

        per_match.append(
            {
                "match_rank": rank,
                "start_idx": int(start),
                "dtw_dist": float(dist),
                "category": cat,
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
    ap.add_argument("--test-sample",  action="store_true",
                    help="Run quick evaluation with real test sample curves (no checkpoint needed)")
    ap.add_argument("--mock-data", action="store_true",
                    help="Run with generated mock knee/thigh curves (no checkpoint needed)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Top-k motion matches to simulate")
    ap.add_argument("--save-mock", default=None,
                    help="Optional .npz path to save generated mock curves")
    ap.add_argument("--real-walk-data", action="store_true",
                    help="Use a real walking segment from mocap DB (thigh pitch + knee included angle only)")
    ap.add_argument("--save-real", default=None,
                    help="Optional .npz path to save extracted real walking curves")
    ap.add_argument("--test-sample-source", choices=["external", "mocap"], default="external",
                    help="Source for --test-sample queries (external recommended for out-of-db testing)")
    ap.add_argument("--external-sample-url", default=None,
                    help="Optional URL override for external gait sample file (.mot/.sto)")
    ap.add_argument("--match-categories", default=None,
                    help="Comma-separated category filter for motion matching (e.g. walk,run)")
    return ap.parse_args()


def main():
    args = _parse_args()

    match_categories = None
    if args.match_categories:
        match_categories = [c.strip() for c in args.match_categories.split(",") if c.strip()]

    seconds = _derive_window_seconds(args.data, args.checkpoint)
    print(f"[eval] Window duration derived from model architecture: {seconds:.3f} s")

    if args.test_sample:
        run_test_sample(
            mocap_dir=args.mocap_dir,
            out_path=args.out,
            top_k=args.top_k,
            match_categories=match_categories,
            seconds=seconds,
            sample_source=args.test_sample_source,
            external_url=args.external_sample_url,
        )
        return

    if args.real_walk_data:
        curves = extract_real_sample_curves(
            mocap_dir=args.mocap_dir,
            seconds=seconds,
            categories=match_categories,
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
            out_path=args.out,
            match_categories=match_categories,
        )
        return

    if args.mock_data:
        curves = generate_mock_curves(length_s=seconds)
        if args.save_mock:
            save_mock_curves(args.save_mock, curves)
            print(f"[eval] Saved mock curves -> {args.save_mock}")
        evaluate_from_curves(
            knee_label_included=curves.knee_label_included_deg,
            thigh_angle=curves.thigh_angle_deg,
            predicted_knee_included=curves.predicted_knee_included_deg,
            mocap_dir=args.mocap_dir,
            top_k=args.top_k,
            out_path=args.out,
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
                "  3. Run test sample:      --test-sample",
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
        device_str      = args.device,
        match_categories = match_categories,
    )


if __name__ == "__main__":
    main()
