from __future__ import annotations

import argparse

from .mocapact_dataset import EXPECTED_SNIPPETS, download_all_snippets, validate_snippet_count


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and verify MoCapAct snippets from Hugging Face")
    ap.add_argument("--dest", type=str, default="mocap_data/mocapact")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        paths = download_all_snippets(args.dest)
        print(f"Downloaded {len(paths)} snippet files.")

    count, ok = validate_snippet_count(args.dest, expected=EXPECTED_SNIPPETS)
    print(f"Local snippet count: {count} (expected {EXPECTED_SNIPPETS})")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
