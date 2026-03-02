from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mocapact_dataset import EXPECTED_SNIPPETS, validate_downloads


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Validate local MoCapAct assets (manual Hugging Face downloads): experts + optional rollouts"
    )
    ap.add_argument("--experts-root", type=str, default="", help="Extracted mocapact-models experts/ directory")
    ap.add_argument("--rollouts-root", type=str, default="", help="Extracted mocapact-data rollouts/ directory (optional)")
    ap.add_argument("--expected", type=int, default=EXPECTED_SNIPPETS)
    ap.add_argument("--out", type=str, default="", help="Optional path to write JSON report")
    args = ap.parse_args()

    rollouts_root = Path(args.rollouts_root) if args.rollouts_root else None
    experts_root = Path(args.experts_root) if args.experts_root else None
    if rollouts_root is None and experts_root is None:
        raise SystemExit("Provide at least one of --experts-root or --rollouts-root.")

    report = validate_downloads(
        rollouts_root=rollouts_root,
        experts_root=experts_root,
        expected=int(args.expected),
    )

    print(json.dumps(report, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ok = True
    if "rollout_snippets_ok" in report:
        ok = ok and bool(report["rollout_snippets_ok"])
    if "expert_policies_ok" in report:
        ok = ok and bool(report["expert_policies_ok"])
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()

