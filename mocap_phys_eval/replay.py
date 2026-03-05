from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvalConfig
from .viewer import view_compare_npz


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay the latest MoCapAct compare recording (v2).")
    ap.add_argument(
        "recording",
        nargs="?",
        default="",
        help="Optional path to a compare .npz. If omitted, opens artifacts/phys_eval_v2/latest_compare.npz.",
    )
    args = ap.parse_args()

    if str(args.recording).strip():
        path = Path(str(args.recording)).expanduser()
    else:
        cfg = EvalConfig()
        path = Path(cfg.artifacts_dir) / "latest_compare.npz"

    if not path.exists():
        raise SystemExit(f"Replay file not found: {path}")

    view_compare_npz(path)


if __name__ == "__main__":
    main()

