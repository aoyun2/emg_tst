"""
Prosthetic evaluation pipeline using CMU mocap database.

Usage
-----
    # Primary: evaluate a trained model checkpoint against CMU mocap
    python -m mocap_evaluation.run_evaluation \\
        --checkpoint checkpoints/<run>/fold_01/reg_best.pt \\
        --data samples_dataset.npy

    # Fallback (no trained model): use test sample curves instead
    python -m mocap_evaluation.run_evaluation --test-sample

    # Replay a saved trajectory in MuJoCo viewer
    python -m mocap_evaluation.run_evaluation --replay path/to/traj.npz

Pipeline
--------
1.  Download CMU BVH database (cached after first parse)
2.  Load trained model + dataset  (or extract test sample curves with --test-sample)
3.  Motion-match query against CMU database (L2 pre-filter → DTW)
4.  Simulate: mocap drives all joints except right knee (model prediction)
5.  Collect metrics → JSON output
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
from tqdm import tqdm

from emg_tst.model import TSTEncoder, TSTRegressor
from emg_tst.data import StandardScaler
from mocap_evaluation.mocap_loader import (
    TARGET_FPS,
    load_aggregated_database,
)
from mocap_evaluation.motion_matching import find_best_match, find_top_k_matches
from mocap_evaluation.prosthetic_sim import (
    simulate_prosthetic_walking,
    FALL_HEIGHT_THRESHOLD,
    replay_trajectory,
    render_simulation_gif,
)
from mocap_evaluation.sample_data import extract_real_sample_curves
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
    ax.set_ylabel("Knee angle (deg, 180=straight)")
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


def plot_simulation_comparison(
    gt_metrics: dict,
    good_metrics: dict,
    bad_metrics: Optional[dict],
    title: str,
    out_path: str | Path,
    fps: float = 200.0,
) -> None:
    """Save a 3-panel comparison plot (GT vs good prediction vs bad prediction)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_knee = np.asarray(gt_metrics.get("pred_knee_series", []))
    good_knee = np.asarray(good_metrics.get("pred_knee_series", []))
    ref_knee = np.asarray(gt_metrics.get("ref_knee_series", []))
    if gt_knee.size == 0:
        return

    T = len(gt_knee)
    t = np.arange(T) / fps

    n_runs = 3 if bad_metrics is not None else 2
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Knee angles
    ax = axes[0]
    ax.plot(t, ref_knee[:T], label="Matched CMU knee (ref)", lw=1.5, color="green", alpha=0.7)
    ax.plot(t, gt_knee[:T], label="GT knee", lw=2, color="blue")
    good_rmse = good_metrics.get("knee_rmse_deg", 0)
    ax.plot(t, good_knee[:T], label=f"Good pred (RMSE={good_rmse:.1f})", lw=2, color="orange", alpha=0.8)
    if bad_metrics is not None:
        bad_knee = np.asarray(bad_metrics.get("pred_knee_series", []))
        bad_rmse = bad_metrics.get("knee_rmse_deg", 0)
        ax.plot(t[:len(bad_knee)], bad_knee[:T], label=f"Bad pred (RMSE={bad_rmse:.1f})",
                lw=2, color="red", alpha=0.7)
    ax.set_ylabel("Knee angle (deg, 180=straight)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: CoM height
    ax = axes[1]
    gt_com = np.asarray(gt_metrics.get("com_height_series", []))
    good_com = np.asarray(good_metrics.get("com_height_series", []))
    if gt_com.size > 0:
        ax.plot(t[:len(gt_com)], gt_com[:T], label="GT CoM", lw=2, color="blue")
    if good_com.size > 0:
        good_stab = good_metrics.get("stability_score", 0)
        ax.plot(t[:len(good_com)], good_com[:T],
                label=f"Good pred CoM (stab={good_stab:.2f})", lw=2, color="orange", alpha=0.8)
    if bad_metrics is not None:
        bad_com = np.asarray(bad_metrics.get("com_height_series", []))
        if bad_com.size > 0:
            bad_stab = bad_metrics.get("stability_score", 0)
            ax.plot(t[:len(bad_com)], bad_com[:T],
                    label=f"Bad pred CoM (stab={bad_stab:.2f})", lw=2, color="red", alpha=0.7)
    ax.axhline(FALL_HEIGHT_THRESHOLD, color="r", ls="--", lw=1, label="Fall threshold")
    gt_stab = gt_metrics.get("stability_score", 0)
    ax.set_ylabel("CoM height (m)")
    ax.set_title(f"CoM  |  GT stab={gt_stab:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Foot contacts
    ax = axes[2]
    runs = [("GT", gt_metrics, "blue"), ("Good", good_metrics, "orange")]
    if bad_metrics is not None:
        runs.append(("Bad", bad_metrics, "red"))
    for i, (label, m, color) in enumerate(runs):
        rc = np.asarray(m.get("right_contact_frames", [])) / fps
        lc = np.asarray(m.get("left_contact_frames", [])) / fps
        y = 1 - i * 0.25
        if len(rc) > 0:
            ax.scatter(rc, np.full_like(rc, y), marker="|", s=100, color=color, label=f"{label} R")
        if len(lc) > 0:
            ax.scatter(lc, np.full_like(lc, y + 0.08), marker="|", s=100, color=color, alpha=0.5, label=f"{label} L")
    steps_str = (f"GT={gt_metrics.get('step_count', 0)}, "
                 f"Good={good_metrics.get('step_count', 0)}")
    if bad_metrics is not None:
        steps_str += f", Bad={bad_metrics.get('step_count', 0)}"
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Foot contacts  |  Steps: {steps_str}")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[viz] Comparison plot saved -> {out_path}")


# ── Window-length helper ──────────────────────────────────────────────────────


def _derive_window_seconds(
    data_path: str | Path | None,
    checkpoint_path: str | Path | None = None,
) -> float:
    """Derive the per-sample window duration (in seconds) from the dataset or model.

    This returns the **training** window size (e.g. 1 second), not the
    evaluation segment length.  The evaluation script concatenates multiple
    training windows to form longer segments for physics simulation.

    Priority:
    1. Dataset ``window`` field from *samples_dataset.npy*.
    2. Checkpoint ``model_cfg["seq_len"]``.
    3. Architecture default: 200 samples @ 200 Hz = 1.0 s.
    """
    dataset_window: int | None = None
    ckpt_seq_len: int | None = None

    if data_path is not None:
        p = Path(data_path)
        if p.exists():
            d = np.load(p, allow_pickle=True)
            if isinstance(d, np.ndarray):
                d = d.item()
            dataset_window = int(d.get("window", 200))

    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists():
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                ckpt_seq_len = int(ckpt["model_cfg"]["seq_len"])
            except Exception:
                pass

    if dataset_window is not None:
        return dataset_window / TARGET_FPS

    if ckpt_seq_len is not None:
        return ckpt_seq_len / TARGET_FPS

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
    X             : (N, W, F) raw features
    y_seq         : (N, W) knee angle labels per timestep (degrees)
    thigh_col     : int  column index of thigh_angle feature in X
    file_ids      : (N,) int32  recording file index per window
    start_indices : (N,) int32  absolute start index of each window within its
                    recording file, or None if not available
    """
    data  = np.load(samples_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()      # allow_pickle=True returns a 0-d object array

    X     = data["X"].astype(np.float32)      # (N, W, F)
    y_seq = data["y_seq"].astype(np.float32)  # (N, W)
    file_ids = data.get("file_id", np.zeros(len(X), dtype=np.int32))
    file_ids = np.asarray(file_ids, dtype=np.int32)

    start_indices = data.get("start", None)
    if start_indices is not None:
        start_indices = np.asarray(start_indices, dtype=np.int32)

    if max_samples is not None:
        X        = X[:max_samples]
        y_seq    = y_seq[:max_samples]
        file_ids = file_ids[:max_samples]
        if start_indices is not None:
            start_indices = start_indices[:max_samples]

    # Thigh angle is the last feature column (from data.py load_recording)
    thigh_col = X.shape[2] - 1

    return X, y_seq, thigh_col, file_ids, start_indices


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
    Run a window through the model, handling length mismatches.

    If the input window is longer than the model's ``seq_len``, a sliding
    window with 50% overlap is used and overlapping predictions are averaged.
    If shorter, the input is right-padded (replicated last frame) and the
    extra predictions are discarded.

    Returns
    -------
    pred : (W,) predicted knee angle in degrees (raw output, not normalised)
    """
    x = (x_window - scaler.mean_) / scaler.std_    # (W, F)
    if feature_cols is not None:
        x = x[:, feature_cols]

    W = x.shape[0]
    model_seq_len = model.encoder.seq_len

    if W == model_seq_len:
        # Exact match — single forward pass
        x_t = torch.from_numpy(x).unsqueeze(0).float().to(device)
        out = model(x_t)   # (1, W, 1)
        return out[0, :, 0].cpu().numpy()

    if W < model_seq_len:
        # Pad to model_seq_len, run, trim back
        pad = np.tile(x[-1:], (model_seq_len - W, 1))
        x_padded = np.concatenate([x, pad], axis=0)
        x_t = torch.from_numpy(x_padded).unsqueeze(0).float().to(device)
        out = model(x_t)
        return out[0, :W, 0].cpu().numpy()

    # W > model_seq_len — sliding window with 50 % overlap
    stride = model_seq_len // 2
    pred_sum = np.zeros(W, dtype=np.float64)
    pred_cnt = np.zeros(W, dtype=np.float64)

    start = 0
    while start < W:
        end = start + model_seq_len
        if end > W:
            # Last chunk: align to the end
            start = W - model_seq_len
            end = W
        chunk = x[start:end]
        x_t = torch.from_numpy(chunk).unsqueeze(0).float().to(device)
        out = model(x_t)  # (1, model_seq_len, 1)
        preds = out[0, :, 0].cpu().numpy()
        pred_sum[start:end] += preds
        pred_cnt[start:end] += 1.0
        if end >= W:
            break
        start += stride

    return (pred_sum / np.maximum(pred_cnt, 1.0)).astype(np.float32)


# ── Test sample evaluation ───────────────────────────────────────────────────


BAD_PRED_NOISE_STD = 25.0  # degrees — intentionally poor prediction


def run_test_sample(
    mocap_dir: str | Path = "mocap_data",
    out_path: str | Path = "eval_test_sample_results.json",
    top_k: int = 3,
    seconds: float = 4.0,
    sample_source: str = "external",
    external_url: Optional[str] = None,
    render_gifs: bool = False,
    use_cache: bool = True,
) -> dict:
    """Run the motion-matching pipeline using real test sample curves.

    This is a fallback mode for when no trained model checkpoint is available.
    Runs three simulations per match: ground truth, good prediction, and
    bad prediction (high noise) to contrast stability outcomes.
    """
    print("=" * 60)
    print("TEST SAMPLE -- real mocap query -> motion matching -> simulation")
    print("=" * 60)

    if sample_source == "external":
        curves = extract_external_sample_curves(
            seconds=seconds,
            source_url=external_url,
        )
    else:
        curves = extract_real_sample_curves(
            mocap_dir=mocap_dir,
            seconds=seconds,
        )
    model_rmse = float(np.sqrt(np.mean(
        (curves.predicted_knee_included_deg - curves.knee_label_included_deg) ** 2
    )))

    # Generate a bad prediction with much higher noise
    rng = np.random.default_rng(42)
    knee_label = curves.knee_label_included_deg
    bad_pred = np.clip(
        knee_label + rng.normal(0.0, BAD_PRED_NOISE_STD, len(knee_label)).astype(np.float32),
        0.0, 180.0,
    )
    bad_rmse = float(np.sqrt(np.mean((bad_pred - knee_label) ** 2)))

    print(
        f"[eval] Real test sample: {len(knee_label)} frames @ {curves.fps} Hz"
    )
    print(f"[eval] Source file: {curves.source_file} [{curves.category}]")
    print(f"[eval] Good pred RMSE: {model_rmse:.2f}°  |  "
          f"Bad pred RMSE (noise std={BAD_PRED_NOISE_STD}): {bad_rmse:.2f}°")

    result = evaluate_from_curves(
        knee_label_included=knee_label,
        thigh_angle=curves.thigh_angle_deg,
        predicted_knee_included=curves.predicted_knee_included_deg,
        bad_predicted_knee_included=bad_pred,
        mocap_dir=mocap_dir,
        top_k=top_k,
        out_path=out_path,
        render_gifs=render_gifs,
        use_cache=use_cache,
    )
    result["test_sample_pred_vs_label_rmse_deg"] = model_rmse
    result["bad_pred_vs_label_rmse_deg"] = bad_rmse
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ── Main evaluation loop ──────────────────────────────────────────────────────


EVAL_SECONDS_DEFAULT = 5.0  # physics evaluation window (seconds)


def _find_consecutive_runs(
    start_indices: np.ndarray,
    window_size: int,
) -> list[tuple[int, int]]:
    """Find sub-runs of truly consecutive windows based on start indices.

    Each window is consecutive to the next if ``start[i+1] == start[i] + window_size``.

    Returns list of ``(offset, length)`` within the input array.
    """
    n = len(start_indices)
    if n <= 1:
        return [(0, n)]

    runs: list[tuple[int, int]] = []
    run_begin = 0
    for i in range(1, n):
        if int(start_indices[i]) != int(start_indices[i - 1]) + window_size:
            runs.append((run_begin, i - run_begin))
            run_begin = i
    runs.append((run_begin, n - run_begin))
    return runs


def _build_eval_segments(
    X: np.ndarray,
    y_seq: np.ndarray,
    file_ids: np.ndarray,
    windows_per_seg: int,
    start_indices: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate consecutive training windows into longer eval segments.

    Only windows from the **same recording file** that are **verified to be
    temporally consecutive** are grouped together.  Verification uses the
    ``start`` indices saved by ``split_to_samples.py``; if unavailable the
    old heuristic (same file-id = consecutive) is used as a fallback.

    Parameters
    ----------
    X        : (N, W, F) per-window features
    y_seq    : (N, W)    per-window knee angle labels
    file_ids : (N,)      recording file index per window
    windows_per_seg : number of consecutive windows to merge
    start_indices : (N,) absolute start index of each window in its file,
                    or None if not available (falls back to file-id grouping)

    Returns
    -------
    X_seg   : (M, W*windows_per_seg, F)
    y_seg   : (M, W*windows_per_seg)
    """
    N, W, F = X.shape
    k = windows_per_seg

    seg_X: list[np.ndarray] = []
    seg_y: list[np.ndarray] = []
    n_skipped = 0

    run_start = 0
    while run_start < N:
        fid = file_ids[run_start]
        run_end = run_start + 1
        while run_end < N and file_ids[run_end] == fid:
            run_end += 1

        # Within this same-file run, find sub-runs that are truly consecutive
        if start_indices is not None:
            sub_runs = _find_consecutive_runs(
                start_indices[run_start:run_end], W,
            )
        else:
            # No start info — assume entire file run is consecutive
            sub_runs = [(0, run_end - run_start)]

        for cr_offset, cr_len in sub_runs:
            base = run_start + cr_offset
            n_segs = cr_len // k
            for s in range(n_segs):
                idx = base + s * k
                x_parts = X[idx : idx + k]          # (k, W, F)
                y_parts = y_seq[idx : idx + k]      # (k, W)
                seg_X.append(x_parts.reshape(k * W, F))
                seg_y.append(y_parts.reshape(k * W))
            n_skipped += cr_len % k

        run_start = run_end

    if n_skipped > 0:
        print(f"[eval] Discarded {n_skipped} windows that didn't fill a segment")

    if not seg_X:
        # Not enough consecutive same-file windows — fall back to
        # concatenating all windows as a single segment.
        X_cat = X.reshape(1, N * W, F)
        y_cat = y_seq.reshape(1, N * W)
        return X_cat, y_cat

    return np.stack(seg_X), np.stack(seg_y)


def evaluate(
    checkpoint_path: str | Path,
    samples_path: str | Path,
    mocap_dir: str | Path = "mocap_data",
    out_path: str | Path  = "eval_results.json",
    n_samples: Optional[int] = None,
    device_str: str = "cpu",
    use_cache: bool = True,
    eval_seconds: float = EVAL_SECONDS_DEFAULT,
) -> dict:
    """
    Full prosthetic evaluation pipeline.

    Training windows (e.g. 1 s) are concatenated into longer evaluation
    segments (default 5 s) so the physics simulation runs long enough to
    reveal falls.  The model predicts over each segment using sliding-window
    inference.

    Returns
    -------
    dict  with keys 'per_segment' (list) and 'summary' (aggregated stats)
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
    X, y_seq, thigh_col, file_ids, start_indices = load_samples(
        samples_path, max_samples=n_samples,
    )
    N, W, F = X.shape
    train_window_sec = W / TARGET_FPS
    print(f"       {N} windows  shape={X.shape}  ({train_window_sec:.1f}s each)")

    # ── Concatenate into longer eval segments ───────────────────────────────
    windows_per_seg = max(1, int(round(eval_seconds / train_window_sec)))
    actual_eval_sec = windows_per_seg * train_window_sec
    X_seg, y_seg = _build_eval_segments(
        X, y_seq, file_ids, windows_per_seg,
        start_indices=start_indices,
    )
    M = len(X_seg)
    print(f"[eval] Eval segments: {M} × {windows_per_seg} windows "
          f"= {actual_eval_sec:.1f}s each  "
          f"(model seq_len={model_cfg['seq_len']})")

    # ── Load / generate mocap database ──────────────────────────────────────
    print(f"[eval] Loading CMU mocap database from: {mocap_dir}")
    mocap_db = load_aggregated_database(
        mocap_root=mocap_dir, try_download=True,
        datasets=["cmu"], use_cache=use_cache,
    )
    db_dur   = len(mocap_db["knee_right"]) / mocap_db["fps"]
    n_files = len(mocap_db.get("file_boundaries", []))
    extra = f", {n_files} files" if n_files else ""
    print(f"       {db_dur:.1f} s @ {mocap_db['fps']:.0f} Hz  "
          f"(source: {mocap_db['source']}{extra})")

    # ── Per-segment loop ────────────────────────────────────────────────────
    per_segment = []
    for i in tqdm(range(M), desc="Evaluating segments", unit="seg"):
        t0 = time.time()

        x_seg_i    = X_seg[i]                    # (S, F)  S = W * windows_per_seg
        knee_label = y_seg[i]                     # (S,)
        thigh_sig  = x_seg_i[:, thigh_col]        # (S,)

        # Model prediction (sliding window handles S > model seq_len)
        pred_knee = predict_knee_sequence(
            model, scaler, x_seg_i, feature_cols, device
        )

        knee_label_inc = knee_label.astype(np.float32)
        pred_knee_inc = pred_knee.astype(np.float32)

        # Motion matching on the full eval segment
        _, dtw_dist, segment = find_best_match(
            knee_label_inc,
            thigh_sig,
            mocap_db,
        )

        # Physics simulation over the full segment length
        metrics = simulate_prosthetic_walking(
            segment, pred_knee_inc,
            use_gui=False,
            sample_thigh_right=thigh_sig,
        )

        metrics["segment_idx"]  = i
        metrics["dtw_dist"]     = float(dtw_dist)
        metrics["pred_rmse"]    = float(
            np.sqrt(np.mean((pred_knee - knee_label) ** 2))
        )
        metrics["elapsed_s"]    = float(time.time() - t0)
        metrics["segment_seconds"] = actual_eval_sec
        per_segment.append(metrics)

        # Save plots for first 5 segments
        if i < 5:
            plot_dir = Path(out_path).with_suffix("") / "plots"
            plot_simulation(metrics, f"Segment {i} ({actual_eval_sec:.0f}s)",
                            plot_dir / f"segment_{i:04d}_sim.png")

    # ── Aggregate summary ────────────────────────────────────────────────────
    def _agg(key):
        vals = [w[key] for w in per_segment if key in w]
        if not vals:
            return {}
        arr = np.array(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "min": float(arr.min()), "max": float(arr.max())}

    fall_rate = float(np.mean([w["fall_detected"] for w in per_segment]))

    summary = {
        "n_segments":      M,
        "eval_seconds":    actual_eval_sec,
        "windows_per_seg": windows_per_seg,
        "fall_rate":       fall_rate,
        "pred_rmse":       _agg("pred_rmse"),
        "knee_rmse_deg":   _agg("knee_rmse_deg"),
        "stability_score": _agg("stability_score"),
        "dtw_dist":        _agg("dtw_dist"),
        "com_height_std":  _agg("com_height_std"),
        "gait_symmetry":   _agg("gait_symmetry"),
        "mode":            per_segment[0].get("mode", "unknown") if per_segment else "unknown",
        "mocap_source":    mocap_db["source"],
    }

    result = {"per_segment": per_segment, "summary": summary}

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


def play_mocap_match(
    mocap_dir: str | Path = "mocap_data",
    seconds: float = 4.0,
    sample_source: str = "external",
    external_url: Optional[str] = None,
    use_cache: bool = True,
    save_trajectory: Optional[str | Path] = None,
) -> None:
    """Find the best mocap match for sample curves and play it in the viewer.

    All joints (including right knee) are driven by the matched mocap segment
    so the user can see exactly what motion was matched.
    """
    if sample_source == "external":
        curves = extract_external_sample_curves(
            seconds=seconds,
            source_url=external_url,
        )
    else:
        curves = extract_real_sample_curves(
            mocap_dir=mocap_dir,
            seconds=seconds,
        )

    knee = curves.knee_label_included_deg.astype(np.float32)
    thigh = curves.thigh_angle_deg.astype(np.float32)

    print(f"[play-match] Sample: {len(knee)} frames @ {curves.fps} Hz "
          f"({len(knee)/curves.fps:.2f}s), source={curves.source_file}")

    mocap_db = load_aggregated_database(
        mocap_root=mocap_dir, try_download=True,
        datasets=["cmu"], use_cache=use_cache,
    )

    _, dist, segment = find_best_match(
        knee, thigh, mocap_db,
    )

    cat = segment.get("category", "unknown")
    print(f"[play-match] Best match: DTW={dist:.4f}, category={cat}")

    # ── Save a comparison plot (query vs matched) ────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        T = len(knee)
        t = np.arange(T) / curves.fps
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(t, knee, label="Query knee (sample)", linewidth=2)
        axes[0].plot(t, segment["knee_right"], label="Matched knee (mocap)",
                     linewidth=2, alpha=0.8)
        knee_rmse = float(np.sqrt(np.mean((knee - segment["knee_right"]) ** 2)))
        axes[0].set_ylabel("Knee included angle (deg)")
        axes[0].set_title(f"Query vs matched mocap  |  Knee RMSE={knee_rmse:.2f}°  "
                          f"DTW={dist:.4f}  [{cat}]")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, thigh, label="Query thigh (sample)", linewidth=2)
        axes[1].plot(t, segment["hip_right"], label="Matched hip (mocap)",
                     linewidth=2, alpha=0.8)
        thigh_rmse = float(np.sqrt(np.mean((thigh - segment["hip_right"]) ** 2)))
        axes[1].set_ylabel("Thigh included angle (deg)")
        axes[1].set_title(f"Thigh RMSE={thigh_rmse:.2f}°")
        axes[1].set_xlabel("Time (s)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = Path("play_match_comparison.png")
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        print(f"[play-match] Comparison plot saved -> {plot_path}")
    except Exception as exc:
        print(f"[play-match] Could not save comparison plot: {exc}")

    # ── Play in MuJoCo viewer ────────────────────────────────────────────
    print(f"[play-match] Playing matched mocap motion in MuJoCo viewer...")
    print(f"[play-match] All joints driven by mocap (including right knee).")

    # Drive ALL joints from mocap — use mocap knee_right as the "prediction"
    # so the viewer shows the pure matched motion.
    traj_path = save_trajectory or "play_match_traj.npz"
    simulate_prosthetic_walking(
        segment,
        segment["knee_right"],   # mocap knee → "prediction" slot
        use_gui=True,
        show_reference=False,
        save_trajectory=traj_path,
    )
    print(f"[play-match] Trajectory saved -> {traj_path}")


def evaluate_from_curves(
    knee_label_included: np.ndarray,
    thigh_angle: np.ndarray,
    predicted_knee_included: np.ndarray,
    mocap_dir: str | Path = "mocap_data",
    top_k: int = 3,
    out_path: str | Path = "eval_mock_results.json",
    render_gifs: bool = False,
    use_cache: bool = True,
    bad_predicted_knee_included: Optional[np.ndarray] = None,
) -> dict:
    """Evaluate motion matching directly from label/thigh curves.

    This path is designed for early feasibility testing before recorded data is
    available from ``rigtest.py``.

    When *bad_predicted_knee_included* is provided, a third "bad prediction"
    simulation is run per match to contrast stability outcomes.
    """
    knee_label_inc = knee_label_included.astype(np.float32)
    pred_knee_inc = predicted_knee_included.astype(np.float32)
    bad_knee_inc = (bad_predicted_knee_included.astype(np.float32)
                    if bad_predicted_knee_included is not None else None)

    mocap_db = load_aggregated_database(
        mocap_root=mocap_dir, try_download=True,
        datasets=["cmu"], use_cache=use_cache,
    )

    matches = find_top_k_matches(
        imu_knee=knee_label_inc,
        imu_thigh=thigh_angle.astype(np.float32),
        mocap_db=mocap_db,
        k=top_k,
    )

    per_match = []
    for rank, (start, dist, segment) in enumerate(tqdm(matches, desc="Simulating matches", unit="match"), start=1):
        cat = segment.get("category", "unknown")
        plot_dir = Path(out_path).with_suffix("") / f"match_{rank:02d}_{cat}"

        # Build optional GIF/trajectory paths
        gt_gif = (plot_dir / "ground_truth_sim.gif") if render_gifs else None
        pred_gif = (plot_dir / "prediction_sim.gif") if render_gifs else None
        gt_traj = plot_dir / "ground_truth_sim.traj.npz"
        pred_traj = plot_dir / "prediction_sim.traj.npz"

        # All simulations use the sample's own thigh angle for the right hip
        # actuator so that the right leg (thigh + knee) is always driven by
        # the sample being evaluated, not by the matched mocap reference.
        gt_metrics = simulate_prosthetic_walking(
            segment,
            knee_label_inc,
            sample_thigh_right=thigh_angle,
            save_trajectory=gt_traj,
            render_gif=gt_gif,
        )
        pred_metrics = simulate_prosthetic_walking(
            segment,
            pred_knee_inc,
            sample_thigh_right=thigh_angle,
            save_trajectory=pred_traj,
            render_gif=pred_gif,
        )

        # Save simulation plots
        plot_simulation(gt_metrics, f"Ground Truth -- match #{rank} [{cat}]",
                        plot_dir / "ground_truth_sim.png", fps=mocap_db["fps"])
        plot_simulation(pred_metrics, f"Good Prediction -- match #{rank} [{cat}]",
                        plot_dir / "prediction_sim.png", fps=mocap_db["fps"])

        match_entry = {
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

        # Bad prediction simulation (optional, for demo/test-sample mode)
        bad_metrics = None
        if bad_knee_inc is not None:
            bad_gif = (plot_dir / "bad_prediction_sim.gif") if render_gifs else None
            bad_traj = plot_dir / "bad_prediction_sim.traj.npz"
            bad_metrics = simulate_prosthetic_walking(
                segment,
                bad_knee_inc,
                sample_thigh_right=thigh_angle,
                save_trajectory=bad_traj,
                render_gif=bad_gif,
            )
            plot_simulation(bad_metrics, f"Bad Prediction -- match #{rank} [{cat}]",
                            plot_dir / "bad_prediction_sim.png", fps=mocap_db["fps"])
            match_entry["bad_prediction_replaced_right_knee"] = bad_metrics
            match_entry["bad_pred_vs_label_rmse_deg"] = float(
                np.sqrt(np.mean(
                    (bad_predicted_knee_included - knee_label_included) ** 2
                ))
            )

        # Save comparison plot showing all runs side by side
        plot_simulation_comparison(
            gt_metrics, pred_metrics, bad_metrics,
            title=f"Match #{rank} [{cat}]",
            out_path=plot_dir / "comparison.png",
            fps=mocap_db["fps"],
        )

        per_match.append(match_entry)

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
        description="Evaluate prosthetic knee via motion-capture simulation (CMU dataset)"
    )

    # ── Primary mode: trained model evaluation ──
    ap.add_argument("--checkpoint", default=None,
                    help="Path to reg_best.pt checkpoint (required unless --test-sample)")
    ap.add_argument("--data", default="samples_dataset.npy",
                    help="Path to samples_dataset.npy produced by split_to_samples.py")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Limit the number of evaluation windows (default: all)")
    ap.add_argument("--device", default="cpu",
                    help="Torch device (cpu or cuda)")

    # ── Fallback mode: test sample (no trained model needed) ──
    ap.add_argument("--test-sample", action="store_true",
                    help="Use test sample curves instead of a trained model "
                         "(for when no checkpoint is available)")
    ap.add_argument("--test-sample-source", choices=["external", "mocap"], default="external",
                    help="Source for test sample (external = online OpenSim, mocap = from CMU DB)")
    ap.add_argument("--external-sample-url", default=None,
                    help="Optional URL override for external gait sample file (.mot/.sto)")

    # ── Shared options ──
    ap.add_argument("--mocap-dir",   default="mocap_data",
                    help="Directory containing BVH files (or where to download them)")
    ap.add_argument("--out",         default=None,
                    help="Output JSON file for metrics (auto-selected if omitted)")
    ap.add_argument("--top-k", type=int, default=3,
                    help="Top-k motion matches to simulate (test-sample mode)")
    ap.add_argument("--render-gifs", action="store_true",
                    help="Render animated GIFs of each simulation alongside plots")
    ap.add_argument("--play-match", action="store_true",
                    help="Find the best mocap match for sample curves and play "
                         "the matched motion in the MuJoCo viewer (all joints "
                         "driven by mocap, including right knee)")
    ap.add_argument("--replay", default=None, metavar="TRAJ.npz",
                    help="Replay a previously saved trajectory file in the MuJoCo viewer")
    ap.add_argument("--replay-speed", type=float, default=1.0,
                    help="Playback speed for --replay (default: 1.0)")
    ap.add_argument("--eval-seconds", type=float, default=EVAL_SECONDS_DEFAULT,
                    help=f"Duration of each evaluation segment in seconds "
                         f"(default: {EVAL_SECONDS_DEFAULT}). Consecutive "
                         f"training windows are concatenated to reach this "
                         f"length for physics simulation.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Disable .npz caching of parsed BVH databases")
    return ap.parse_args()


def main():
    args = _parse_args()

    # ── Replay mode (standalone, no other logic needed) ──────────────
    if args.replay:
        print(f"[eval] Replaying trajectory: {args.replay}")
        replay_trajectory(args.replay, speed=args.replay_speed)
        return

    use_cache = not args.no_cache

    if args.play_match:
        # ── Play-match: visualise the matched mocap motion ───────────
        seconds = args.eval_seconds
        print(f"[eval] Play-match duration: {seconds:.1f} s")
        play_mocap_match(
            mocap_dir=args.mocap_dir,
            seconds=seconds,
            sample_source=args.test_sample_source,
            external_url=args.external_sample_url,
            use_cache=use_cache,
        )
        return

    if args.test_sample:
        # ── Fallback: test sample mode (no trained model needed) ─────
        out_path = args.out or "eval_test_sample_results.json"
        seconds = args.eval_seconds
        print(f"[eval] Test sample duration: {seconds:.1f} s")

        run_test_sample(
            mocap_dir=args.mocap_dir,
            out_path=out_path,
            top_k=args.top_k,
            seconds=seconds,
            sample_source=args.test_sample_source,
            external_url=args.external_sample_url,
            render_gifs=args.render_gifs,
            use_cache=use_cache,
        )
    else:
        # ── Primary: trained model evaluation ────────────────────────
        if args.checkpoint is None:
            print("Error: --checkpoint is required (or use --test-sample "
                  "if no trained model is available).")
            sys.exit(1)

        out_path = args.out or "eval_results.json"
        evaluate(
            checkpoint_path=args.checkpoint,
            samples_path=args.data,
            mocap_dir=args.mocap_dir,
            out_path=out_path,
            n_samples=args.n_samples,
            device_str=args.device,
            use_cache=use_cache,
            eval_seconds=args.eval_seconds,
        )


if __name__ == "__main__":
    main()
