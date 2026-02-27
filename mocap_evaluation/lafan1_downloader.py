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

The BVH files are distributed as a zip archive (``lafan1.zip``) inside the
GitHub repository (stored via Git LFS).  This downloader fetches the zip,
extracts all ``.bvh`` files, and stores them in the destination directory.

Usage
-----
    python -m mocap_evaluation.lafan1_downloader
    python -m mocap_evaluation.lafan1_downloader --dest mocap_data/lafan1
    python -m mocap_evaluation.lafan1_downloader --list
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The BVH files live inside lafan1.zip in the repo (Git LFS).
_ZIP_URL = (
    "https://github.com/"
    "ubisoft/ubisoft-laforge-animation-dataset/"
    "raw/master/lafan1/lafan1.zip"
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


def _download_zip(
    dest_dir: Path,
    *,
    actions: Optional[Sequence[str]] = None,
    subjects: Optional[Sequence[int]] = None,
    retries: int = 4,
    verbose: bool = True,
) -> List[Path]:
    """Download lafan1.zip from GitHub LFS and extract BVH files."""
    if actions is None:
        actions = ACTIONS
    if subjects is None:
        subjects = SUBJECTS

    # Build set of wanted filenames for filtering
    wanted: set[str] | None = None
    if actions is not ACTIONS or subjects is not SUBJECTS:
        wanted = set()
        for action in actions:
            max_var = _ACTION_MAX_VARIANTS.get(action, 3)
            for variant in range(1, max_var + 1):
                for subj in subjects:
                    wanted.add(_make_filename(action, variant, subj))

    for attempt in range(retries):
        try:
            if verbose:
                label = f" (attempt {attempt + 1})" if attempt else ""
                print(f"[LAFAN1] Downloading lafan1.zip from GitHub LFS{label} …")
            resp = urllib.request.urlopen(_ZIP_URL, timeout=120)
            zip_bytes = resp.read()
            break
        except (urllib.error.URLError, OSError) as exc:
            if attempt < retries - 1:
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                if verbose:
                    print(f"  [retry] download failed ({exc}), waiting {delay}s …")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Failed to download lafan1.zip after {retries} attempts: {exc}"
                ) from exc

    downloaded: List[Path] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        bvh_members = [m for m in zf.namelist() if m.endswith(".bvh")]
        for member in tqdm(bvh_members, desc="Extracting LAFAN1", unit="file", disable=not verbose):
            basename = Path(member).name
            if wanted is not None and basename not in wanted:
                continue
            dest = dest_dir / basename
            if dest.exists() and dest.stat().st_size > 0:
                downloaded.append(dest)
                continue
            dest.write_bytes(zf.read(member))
            downloaded.append(dest)

    return downloaded


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

    # Skip if BVH files already present
    existing = sorted(dest_dir.glob("*.bvh"))
    if existing:
        if verbose:
            print(f"[LAFAN1] {len(existing)} BVH files already in {dest_dir}/, skipping download.")
        return existing

    downloaded = _download_zip(
        dest_dir,
        actions=actions,
        subjects=subjects,
        verbose=verbose,
    )

    if verbose:
        print(f"\nLAFAN1: extracted {len(downloaded)} BVH files to {dest_dir}/")

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
