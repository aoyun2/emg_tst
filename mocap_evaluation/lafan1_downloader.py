"""
Downloader for Ubisoft La Forge Animation Dataset (LAFAN1) BVH files.

Source
------
Ubisoft La Forge — ubisoft-laforge-animation-dataset
  https://github.com/ubisoft/ubisoft-laforge-animation-dataset
License: CC BY-NC-ND 4.0 (non-commercial, no derivatives, with attribution)

Dataset overview
----------------
  • 496,672 motion frames across 77 sequences and 5 subjects
  • 15 action categories at 30 fps
  • Skeleton: standard MotionBuilder naming (RightUpLeg, RightLeg, RightFoot …)
  • Actions: aiming, dance, fallAndGetUp, fight, ground, jump, obstacles,
            push, run, walk, etc.

Files are downloaded from the raw GitHub content and stored as the original
filenames (e.g. ``aiming1_subject1.bvh``) in the destination directory.

Usage
-----
    python -m mocap_evaluation.lafan1_downloader
    python -m mocap_evaluation.lafan1_downloader --dest mocap_data/lafan1
    python -m mocap_evaluation.lafan1_downloader --list
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "ubisoft/ubisoft-laforge-animation-dataset/"
    "master/lafan1/lafan1/"
)

# Complete file list for the LAFAN1 dataset (5 subjects × multiple actions).
# Built from the GitHub repository listing.
SUBJECTS = [1, 2, 3, 4, 5]

ACTIONS: List[str] = [
    "aiming",
    "dance",
    "fallAndGetUp",
    "fight",
    "ground",
    "jumps",
    "obstacles",
    "push",
    "run",
    "sprint",
    "walk",
]

# Number of sequence variants per action (observed from the repo).
# Some actions have more variants than others.
_ACTION_MAX_VARIANTS: dict[str, int] = {
    "aiming":        3,
    "dance":         3,
    "fallAndGetUp":  2,
    "fight":         3,
    "ground":        4,
    "jumps":         2,
    "obstacles":     2,
    "push":          2,
    "run":           3,
    "sprint":        2,
    "walk":          3,
}

_RETRY_DELAYS = [2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _make_filename(action: str, variant: int, subject: int) -> str:
    return f"{action}{variant}_subject{subject}.bvh"


def _download_one(
    filename: str,
    dest_dir: Path,
    *,
    timeout: float = 30.0,
    retries: int = 4,
    verbose: bool = True,
) -> bool:
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return True

    url = _BASE_URL + filename

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 100:
                if verbose:
                    print(f"  [ok] {filename}")
                return True
            dest.unlink(missing_ok=True)
            return False
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                dest.unlink(missing_ok=True)
                return False
        except (urllib.error.URLError, OSError):
            pass

        if attempt < retries - 1:
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            time.sleep(delay)

    if verbose:
        print(f"  [FAILED] {filename}")
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_all(
    dest_dir: str | Path = "mocap_data/lafan1",
    actions: Optional[Sequence[str]] = None,
    subjects: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> List[Path]:
    """Download LAFAN1 BVH files.

    Parameters
    ----------
    dest_dir : destination directory (created if absent)
    actions  : action types to download (default: all)
    subjects : subject IDs to download (default: all five)
    verbose  : print progress
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if actions is None:
        actions = ACTIONS
    if subjects is None:
        subjects = SUBJECTS

    downloaded: List[Path] = []
    total_ok = 0

    for action in actions:
        max_var = _ACTION_MAX_VARIANTS.get(action, 3)
        for variant in range(1, max_var + 1):
            for subj in subjects:
                fname = _make_filename(action, variant, subj)
                if verbose:
                    print(f"[LAFAN1] {action}{variant} / subject{subj}")
                ok = _download_one(fname, dest_dir, verbose=verbose)
                if ok:
                    downloaded.append(dest_dir / fname)
                    total_ok += 1

    if verbose:
        print(f"\nLAFAN1: downloaded {total_ok} files to {dest_dir}/")

    return downloaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Download Ubisoft LAFAN1 BVH files.\n"
            "License: CC BY-NC-ND 4.0 — non-commercial use with attribution.\n"
            "Source:  https://github.com/ubisoft/ubisoft-laforge-animation-dataset"
        )
    )
    ap.add_argument(
        "--dest", "-d", default="mocap_data/lafan1",
        help="Destination directory (default: mocap_data/lafan1/)",
    )
    ap.add_argument(
        "--actions", "-a", nargs="*", default=None,
        help="Action types to download (default: all). Options: " + ", ".join(ACTIONS),
    )
    ap.add_argument(
        "--subjects", "-s", nargs="*", type=int, default=None,
        help="Subject IDs to download (default: 1-5)",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List available actions, then exit",
    )
    args = ap.parse_args()

    if args.list:
        print("Available actions:")
        for a in ACTIONS:
            n_var = _ACTION_MAX_VARIANTS.get(a, 3)
            print(f"  {a:<20} ({n_var} variants × {len(SUBJECTS)} subjects)")
        print(f"\nSubjects: {SUBJECTS}")
        return

    download_all(
        dest_dir=args.dest,
        actions=args.actions,
        subjects=args.subjects,
    )


if __name__ == "__main__":
    main()
