#!/usr/bin/env python3
"""End-to-end virtual validation pipeline for EMG->knee TST + MoCapAct simulation.

This script supports two modes:
1) OpenSim smoke-test mode (no trained model required)
2) Real recording mode from rigtest.py + trained TST checkpoint

For each continuous batch it will:
  - build thigh and knee query signals
  - motion-match against MoCapAct snippets
  - simulate matched clip while overriding thigh and predicted knee
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import torch

from emg_tst.data import load_recording
from emg_tst.model import TSTEncoder, TSTRegressor
from mocap_evaluation.db import load_database
from mocap_evaluation.matching import clip_info_for_start, find_match
from mocap_evaluation.mocapact_ms import download_dataset
from mocap_evaluation.sim import run_simulation
from virtual_sim_test import TARGET_FPS, load_opensim_data


def _make_batches(knee: np.ndarray, thigh: np.ndarray, batch_frames: int, n_batches: int) -> List[dict]:
    n_need = batch_frames * n_batches
    if len(knee) < n_need:
        reps = int(np.ceil(n_need / len(knee)))
        knee = np.tile(knee, reps)
        thigh = np.tile(thigh, reps)

    out: List[dict] = []
    for i in range(n_batches):
        s = i * batch_frames
        e = s + batch_frames
        out.append({"batch_id": i, "knee": knee[s:e].copy(), "thigh": thigh[s:e].copy()})
    return out


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["model_cfg"]
    enc = TSTEncoder(
        n_vars=int(cfg["n_vars"]),
        seq_len=int(cfg["seq_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        d_ff=int(cfg["d_ff"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    )
    reg = TSTRegressor(enc, out_dim=1)
    reg.load_state_dict(ckpt["reg_state_dict"])
    reg.to(device)
    reg.eval()
    mean = np.asarray(ckpt["scaler"]["mean"], dtype=np.float32)
    std = np.asarray(ckpt["scaler"]["std"], dtype=np.float32)
    return reg, mean, std


def _predict_batch_knee(reg, mean, std, xb: np.ndarray, device: torch.device) -> np.ndarray:
    x = (xb - mean[None, :]) / std[None, :]
    with torch.no_grad():
        out = reg(torch.from_numpy(x[None, ...]).float().to(device))
    return out[0, :, 0].detach().cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the EMG/TST -> motion matching -> simulation pipeline with either "
            "OpenSim sample data or real rigtest recordings."
        )
    )
    ap.add_argument("--source", choices=["opensim", "recording"], default="opensim")
    ap.add_argument("--recording", type=Path, default=None, help="Path to rigtest .npy recording")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Trained TST checkpoint (optional)")
    ap.add_argument("--subset", default="mocapact", choices=["mocapact", "all", "locomotion_small", "walk_tiny", "run_jump_tiny"])
    ap.add_argument("--n-batches", type=int, default=3)
    ap.add_argument("--batch-secs", type=float, default=5.0)
    ap.add_argument("--prefilter-k", type=int, default=100)
    ap.add_argument("--noise", type=float, default=0.0, help="Added to predicted knee if no checkpoint")
    ap.add_argument(
        "--download-mocapact",
        action="store_true",
        help=(
            "Auto-download + extract Microsoft MoCapAct from Hugging Face and set "
            "MOCAPACT_MS_DIR for this run (subset must be 'mocapact')."
        ),
    )
    ap.add_argument(
        "--mocapact-variant",
        default="small",
        choices=["sample", "small", "large"],
        help=(
            "Which Hugging Face MoCapAct variant to download when --download-mocapact "
            "is enabled (default: small = full 2,589 snippets, 20 rollouts each)."
        ),
    )
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("virtual_sim_output/pipeline_summary.json"))
    args = ap.parse_args()

    batch_frames = int(args.batch_secs * TARGET_FPS)

    if args.source == "recording":
        if args.recording is None:
            raise SystemExit("--recording is required when --source recording")
        X, y, _meta = load_recording(args.recording)
        knee_full = y.astype(np.float32)
        thigh_full = X[:, -1].astype(np.float32)
        src = f"recording:{args.recording}"
    else:
        secs = args.n_batches * args.batch_secs + 1.0
        knee_full, thigh_full = load_opensim_data(secs)
        src = "opensim"

    batches = _make_batches(knee_full, thigh_full, batch_frames=batch_frames, n_batches=args.n_batches)

    if args.download_mocapact:
        if args.subset != "mocapact":
            raise SystemExit("--download-mocapact requires --subset mocapact")
        h5_dir = download_dataset(variant=args.mocapact_variant)
        os.environ["MOCAPACT_MS_DIR"] = str(h5_dir)
        print(f"Configured MOCAPACT_MS_DIR={h5_dir}")

    print(f"Loading database subset={args.subset} ...")
    db = load_database(subset=args.subset, use_cache=True)
    n_snippets = len(db["file_boundaries"])
    print(f"Loaded {n_snippets} snippets/clips.")
    if args.subset == "mocapact":
        if n_snippets == 2589:
            print("MoCapAct check: full dataset detected (2,589 snippets).")
        elif n_snippets < 2589:
            print(
                "MoCapAct check: partial dataset detected "
                f"({n_snippets} snippets). This is supported, but for full coverage use "
                "--download-mocapact --mocapact-variant small (or large)."
            )
        else:
            print(
                "MoCapAct check: snippet count is above 2,589 "
                f"({n_snippets}); continuing with loaded data."
            )

    reg = None
    mean = std = None
    x_recording = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.source == "recording":
        x_recording, _, _ = load_recording(args.recording)
    if args.checkpoint is not None:
        reg, mean, std = _load_model(args.checkpoint, device=device)

    results = []
    rng = np.random.default_rng(7)
    for b in batches:
        knee_gt = b["knee"]
        thigh = b["thigh"]

        if reg is not None:
            if args.source != "recording":
                raise SystemExit("Checkpoint inference requires --source recording.")
            # reconstruct feature batch from original sequence window
            s = b["batch_id"] * batch_frames
            e = s + batch_frames
            xb = x_recording[s:e].astype(np.float32)
            knee_pred = _predict_batch_knee(reg, mean, std, xb, device)
        else:
            knee_pred = knee_gt.copy()
            if args.noise > 0:
                knee_pred = knee_pred + rng.normal(0.0, args.noise, size=knee_pred.shape).astype(np.float32)

        best_start, dtw_dist, clip_id, _ = find_match(knee_pred, thigh, db, prefilter_k=args.prefilter_k)
        info = clip_info_for_start(best_start, len(knee_pred), db)

        sim = run_simulation(
            clip_id=clip_id,
            clip_start_frame=info["frame_offset"],
            n_frames=len(knee_pred),
            knee_pred_included_deg=knee_pred,
            thigh_pred_included_deg=thigh,
            use_viewer=not args.no_gui,
            label=f"batch{b['batch_id'] + 1}",
        )

        results.append({
            "batch_id": b["batch_id"],
            "clip_id": clip_id,
            "dtw_dist": float(dtw_dist),
            "sim": sim,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": src,
        "subset": args.subset,
        "n_batches": args.n_batches,
        "batch_secs": args.batch_secs,
        "mocapact_variant": args.mocapact_variant if args.subset == "mocapact" else None,
        "loaded_snippets": int(n_snippets),
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Saved summary: {args.out}")


if __name__ == "__main__":
    main()
