"""Paper-style evaluation pipeline entrypoint.

This module replaces the older single-condition MuJoCo evaluation with a
robustness-oriented protocol:
1) motion matching against CMU mocap,
2) nominal closed-loop simulation,
3) delayed-control simulation,
4) noise-stress simulation,
5) aggregate robustness scoring.
"""

from __future__ import annotations

import argparse
import json

from mocap_evaluation.paper_pipeline import EvalConfig, RobustnessConfig, evaluate_test_sample, evaluate_with_checkpoint


def _parse_args():
    ap = argparse.ArgumentParser(description="Paper-style prosthetic evaluation")
    ap.add_argument("--checkpoint", default=None, help="Path to trained reg_best.pt")
    ap.add_argument("--data", default="samples_dataset.npy", help="Path to samples_dataset.npy")
    ap.add_argument("--test-sample", action="store_true", help="Use external/real sample mode instead of checkpoint inference")
    ap.add_argument("--test-sample-source", choices=["external", "mocap"], default="external")
    ap.add_argument("--external-sample-url", default=None)
    ap.add_argument("--mocap-dir", default="mocap_data")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--eval-seconds", type=float, default=4.0)
    ap.add_argument("--delay-ms", type=float, default=60.0)
    ap.add_argument("--noise-std-deg", type=float, default=6.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--mocapact-checkpoint", default=None,
                    help="Path to MoCapAct multi-clip policy checkpoint")
    ap.add_argument("--mocapact-model-dir", default="mocapact_models",
                    help="Directory for downloaded MoCapAct models")
    return ap.parse_args()


def main():
    args = _parse_args()
    robustness = RobustnessConfig(
        top_k_matches=args.top_k,
        delay_ms=args.delay_ms,
        noise_std_deg=args.noise_std_deg,
        eval_seconds=args.eval_seconds,
    )
    cfg = EvalConfig(
        mocap_dir=args.mocap_dir,
        out_path=args.out,
        n_samples=args.n_samples,
        device=args.device,
        use_cache=not args.no_cache,
        sample_source=args.test_sample_source,
        external_sample_url=args.external_sample_url,
        robustness=robustness,
        mocapact_checkpoint=args.mocapact_checkpoint,
        mocapact_model_dir=args.mocapact_model_dir,
    )

    if args.test_sample:
        result = evaluate_test_sample(cfg)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --test-sample is set")
        result = evaluate_with_checkpoint(args.checkpoint, args.data, cfg)

    print(json.dumps({
        "mode": result.get("mode"),
        "out": cfg.out_path,
    }, indent=2))


if __name__ == "__main__":
    main()
