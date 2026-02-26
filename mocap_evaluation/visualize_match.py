"""Visualise a real test-sample query vs matched mocap segment.

Usage:
  python -m mocap_evaluation.visualize_match \
    --out test_sample_vs_match.png \
    --aggregate-datasets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mocap_evaluation.sample_data import extract_real_sample_curves
from mocap_evaluation.external_sample_data import extract_external_sample_curves
from mocap_evaluation.motion_matching import find_best_match
from mocap_evaluation.mocap_loader import (
    TARGET_FPS,
    load_or_generate_mocap_database,
    load_full_cmu_database,
    load_aggregated_bandai_cmu_database,
)


def _derive_window_seconds(data_path: str | Path) -> float:
    """Derive sample window duration (in seconds) from the dataset file.

    Reads the ``window`` field written by split_to_samples.py and divides by
    TARGET_FPS so the query length always matches the model's sequence length.
    Falls back to the architecture default (200 samples @ 200 Hz = 1 s).
    """
    p = Path(data_path)
    if p.exists():
        d = np.load(p, allow_pickle=True)
        if isinstance(d, np.ndarray):
            d = d.item()
        window = int(d.get("window", 200))
        return window / TARGET_FPS
    return 200 / TARGET_FPS


def _parse_args():
    ap = argparse.ArgumentParser(description="Test-sample-to-mocap matching visualisation")
    ap.add_argument("--mocap-dir", default="mocap_data")
    ap.add_argument("--out", default="test_sample_vs_match.png")
    ap.add_argument("--data", default="samples_dataset.npy",
                    help="samples_dataset.npy produced by split_to_samples.py "
                         "(used to derive window length)")
    ap.add_argument("--sample-source", choices=["external", "mocap"], default="external")
    ap.add_argument("--external-sample-url", default=None)
    ap.add_argument("--full-db", action="store_true")
    ap.add_argument("--aggregate-datasets", action="store_true")
    ap.add_argument("--bandai-dir", default=None)
    ap.add_argument("--cmu-dir", default=None)
    return ap.parse_args()


def main():
    args = _parse_args()

    seconds = _derive_window_seconds(args.data)
    print(f"[viz] Window duration derived from dataset: {seconds:.3f} s")

    if args.sample_source == "external":
        curves = extract_external_sample_curves(
            seconds=seconds,
            source_url=args.external_sample_url,
        )
    else:
        curves = extract_real_sample_curves(
            mocap_dir=args.mocap_dir,
            seconds=seconds,
            categories=("walk",),
            full_database=args.full_db or args.aggregate_datasets,
        )
    knee_included = curves.knee_label_included_deg.astype(np.float32)
    thigh = curves.thigh_angle_deg.astype(np.float32)

    if args.aggregate_datasets:
        bandai = Path(args.bandai_dir) if args.bandai_dir else Path(args.mocap_dir) / "bandai"
        cmu = Path(args.cmu_dir) if args.cmu_dir else Path(args.mocap_dir) / "cmu"
        db = load_aggregated_bandai_cmu_database(bandai_dir=bandai, cmu_dir=cmu, try_download=True)
    elif args.full_db:
        db = load_full_cmu_database(bvh_dir=args.mocap_dir)
    else:
        db = load_or_generate_mocap_database(bvh_dir=args.mocap_dir)

    start, dist, seg = find_best_match(knee_included, thigh, db)
    print(f"Matched start={start}, dtw={dist:.4f}, category={seg.get('category','unknown')}")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install dependencies from requirements_tst.txt"
        ) from exc

    t = np.arange(len(knee_included))
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax[0].plot(t, knee_included, label="test sample knee (included)", linewidth=2)
    ax[0].plot(t, seg["knee_right"], label="matched knee_right", linewidth=2, alpha=0.8)
    ax[0].set_ylabel("deg")
    ax[0].set_title("Test sample segment vs matched mocap segment")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(t, thigh, label="test sample thigh", linewidth=2)
    ax[1].plot(t, seg["hip_right"], label="matched hip_right", linewidth=2, alpha=0.8)
    ax[1].set_ylabel("deg")
    ax[1].set_xlabel("frame")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"Saved visualisation -> {out}")


if __name__ == "__main__":
    main()
