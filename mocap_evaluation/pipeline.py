from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .mocapact_dataset import EXPECTED_SNIPPETS, load_snippet_angles
from .motion_matching import match_batch_to_snippets
from .query_data import load_opensim_csv, load_rigtest_npy, make_contiguous_batches
from .simulation import run_leg_simulation


def _iter_local_snippets(root: Path):
    for p in sorted(list(root.rglob("*.h5")) + list(root.rglob("*.hdf5")) + list(root.rglob("*.npz"))):
        try:
            yield load_snippet_angles(p)
        except Exception:
            continue


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end evaluation pipeline: batches -> motion matching -> simulation")
    ap.add_argument("--mocapact-dir", type=str, default="mocap_data/mocapact")
    ap.add_argument("--source", choices=["opensim", "rigtest"], default="opensim")
    ap.add_argument("--input", type=str, required=True, help="OpenSim CSV or rigtest npy")
    ap.add_argument("--batch-size", type=int, default=400)
    ap.add_argument("--stride", type=int, default=200)
    ap.add_argument("--sample-hz", type=float, default=200.0)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", type=str, default="artifacts/mocapact_eval.json")
    args = ap.parse_args()

    if args.source == "opensim":
        thigh, knee_pred = load_opensim_csv(args.input)
    else:
        thigh, knee_pred = load_rigtest_npy(args.input)

    batches = make_contiguous_batches(
        thigh,
        knee_pred,
        batch_size=args.batch_size,
        stride=args.stride,
        sample_hz=args.sample_hz,
    )
    if not batches:
        raise RuntimeError("No batches generated. Reduce --batch-size or provide longer input sequence.")

    snippets = list(_iter_local_snippets(Path(args.mocapact_dir)))
    if not snippets:
        raise RuntimeError("No parseable MoCapAct snippets found. Run download first.")

    summary = {
        "expected_snippets": EXPECTED_SNIPPETS,
        "loaded_snippets": len(snippets),
        "n_batches": len(batches),
        "results": [],
    }

    for b in batches:
        matches = match_batch_to_snippets(
            b.thigh_angle_deg,
            b.knee_angle_pred_deg,
            b.sample_hz,
            snippets,
            top_k=args.top_k,
        )
        if not matches:
            continue
        best = matches[0]
        sim = run_leg_simulation(b.thigh_angle_deg, b.knee_angle_pred_deg, sample_hz=b.sample_hz)
        summary["results"].append(
            {
                "batch_id": b.batch_id,
                "best_match": {
                    "snippet_id": best.snippet_id,
                    "score": best.score,
                    "start_idx": best.start_idx,
                    "end_idx": best.end_idx,
                },
                "knee_sim_rmse_deg": sim.rmse_knee_deg,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote evaluation summary to {out}")


if __name__ == "__main__":
    main()
