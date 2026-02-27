"""
Downloader for SFU (Simon Fraser University) Motion Capture Database BVH files.

Source
------
SFU Motion Capture Lab
  https://mocap.cs.sfu.ca/
License: Free for academic / research use.

Dataset overview
----------------
  • ~480 BVH files across multiple subjects and motion types
  • Motions include walking, running, jumping, boxing, kicking, etc.
  • Skeleton: standard MotionBuilder naming (RightUpLeg, RightLeg, …)
  • FPS: 120 Hz

The SFU database serves individual BVH files.  This downloader fetches
a curated subset of locomotion-relevant trials.

Usage
-----
    python -m mocap_evaluation.sfu_downloader
    python -m mocap_evaluation.sfu_downloader --dest mocap_data/sfu
    python -m mocap_evaluation.sfu_downloader --list
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence, NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://mocap.cs.sfu.ca/nusmocap/"


class SFUTrial(NamedTuple):
    filename: str
    category: str
    description: str


# Curated catalog of SFU BVH files covering locomotion and common activities.
# Subject numbering and filenames follow the SFU database conventions.
CATALOG: List[SFUTrial] = [
    # Subject 1 — walking / running
    SFUTrial("0005_Walking001.bvh",    "walk",     "Normal walking"),
    SFUTrial("0005_Walking002.bvh",    "walk",     "Walking variation"),
    SFUTrial("0005_Walking003.bvh",    "walk",     "Walking slow"),
    SFUTrial("0005_Walking004.bvh",    "walk",     "Walking fast"),
    SFUTrial("0005_Walking005.bvh",    "walk",     "Walking variation"),
    SFUTrial("0005_Walking006.bvh",    "walk",     "Walking straight"),
    SFUTrial("0005_Walking007.bvh",    "walk",     "Walking with turn"),
    SFUTrial("0005_Walking008.bvh",    "walk",     "Walking brisk"),
    SFUTrial("0005_Walking009.bvh",    "walk",     "Walking circle"),
    SFUTrial("0005_Walking010.bvh",    "walk",     "Walking long"),
    SFUTrial("0005_Running001.bvh",    "run",      "Normal running"),
    SFUTrial("0005_Running002.bvh",    "run",      "Running variation"),
    SFUTrial("0005_Running003.bvh",    "run",      "Running fast"),
    SFUTrial("0005_Jogging001.bvh",    "run",      "Jogging"),
    SFUTrial("0005_Jumping001.bvh",    "jump",     "Standing jump"),
    SFUTrial("0005_Jumping002.bvh",    "jump",     "Jump variation"),
    # Subject 7 — locomotion
    SFUTrial("0007_Walking001.bvh",    "walk",     "Normal walking"),
    SFUTrial("0007_Walking002.bvh",    "walk",     "Walking variation"),
    SFUTrial("0007_Walking003.bvh",    "walk",     "Walking slow"),
    SFUTrial("0007_Walking004.bvh",    "walk",     "Walking turn"),
    SFUTrial("0007_Walking005.bvh",    "walk",     "Walking circle"),
    SFUTrial("0007_Running001.bvh",    "run",      "Normal running"),
    SFUTrial("0007_Running002.bvh",    "run",      "Running variation"),
    SFUTrial("0007_Jumping001.bvh",    "jump",     "Standing jump"),
    # Subject 12 — locomotion
    SFUTrial("0012_Walking001.bvh",    "walk",     "Normal walking"),
    SFUTrial("0012_Walking002.bvh",    "walk",     "Walking variation"),
    SFUTrial("0012_Running001.bvh",    "run",      "Normal running"),
    SFUTrial("0012_Jumping001.bvh",    "jump",     "Standing jump"),
    # Subject 15 — locomotion
    SFUTrial("0015_Walking001.bvh",    "walk",     "Normal walking"),
    SFUTrial("0015_Walking002.bvh",    "walk",     "Walking brisk"),
    SFUTrial("0015_Running001.bvh",    "run",      "Normal running"),
    # Subject 17 — locomotion
    SFUTrial("0017_Walking001.bvh",    "walk",     "Normal walking"),
    SFUTrial("0017_Walking002.bvh",    "walk",     "Walking variation"),
    SFUTrial("0017_Running001.bvh",    "run",      "Normal running"),
    SFUTrial("0017_Jumping001.bvh",    "jump",     "Standing jump"),
    # Sport / exercise
    SFUTrial("0005_Kicking001.bvh",    "sport",    "Kicking"),
    SFUTrial("0005_Boxing001.bvh",     "sport",    "Boxing"),
    SFUTrial("0007_Kicking001.bvh",    "sport",    "Kicking"),
]

_RETRY_DELAYS = [2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_one(
    trial: SFUTrial,
    dest_dir: Path,
    *,
    timeout: float = 30.0,
    retries: int = 4,
    verbose: bool = True,
) -> bool:
    dest = dest_dir / trial.filename
    if dest.exists() and dest.stat().st_size > 0:
        if verbose:
            print(f"  [skip] {trial.filename} (already exists)")
        return True

    url = _BASE_URL + trial.filename

    for attempt in range(retries):
        try:
            if verbose and attempt == 0:
                print(f"  [download] {trial.filename}")
            elif verbose:
                print(f"  [retry {attempt+1}] {trial.filename}")
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 100:
                return True
            dest.unlink(missing_ok=True)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                dest.unlink(missing_ok=True)
                if verbose:
                    print(f"  [404] {trial.filename} — not found on server")
                return False
        except (urllib.error.URLError, OSError):
            pass

        if attempt < retries - 1:
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            time.sleep(delay)

    if verbose:
        print(f"  [FAILED] {trial.filename}")
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filter_trials(
    categories: Optional[Sequence[str]] = None,
    max_per_category: Optional[int] = None,
) -> List[SFUTrial]:
    """Filter the catalog by category."""
    trials = list(CATALOG)
    if categories is not None:
        wanted = set(categories)
        trials = [t for t in trials if t.category in wanted]
    if max_per_category is not None:
        counts: dict[str, int] = {}
        filtered = []
        for t in trials:
            counts.setdefault(t.category, 0)
            if counts[t.category] < max_per_category:
                filtered.append(t)
                counts[t.category] += 1
        trials = filtered
    return trials


def download_trials(
    trials: List[SFUTrial],
    dest_dir: str | Path = "mocap_data/sfu",
    *,
    verbose: bool = True,
) -> List[Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[Path] = []
    n = len(trials)

    for i, trial in enumerate(trials, 1):
        if verbose:
            print(f"[{i}/{n}] {trial.category}: {trial.description} ({trial.filename})")
        ok = _download_one(trial, dest_dir, verbose=verbose)
        if ok:
            downloaded.append(dest_dir / trial.filename)

    if verbose:
        print(f"\nSFU: downloaded {len(downloaded)}/{n} files to {dest_dir}/")

    return downloaded


def download_all(
    dest_dir: str | Path = "mocap_data/sfu",
    categories: Optional[Sequence[str]] = None,
    max_per_category: Optional[int] = None,
    verbose: bool = True,
) -> List[Path]:
    """Download all cataloged SFU trials."""
    trials = filter_trials(categories=categories, max_per_category=max_per_category)

    if verbose:
        cat_counts: dict[str, int] = {}
        for t in trials:
            cat_counts[t.category] = cat_counts.get(t.category, 0) + 1
        print("SFU Mocap Download Plan:")
        for cat, count in sorted(cat_counts.items()):
            print(f"  {cat:<15} {count} trials")
        print(f"  {'TOTAL':<15} {len(trials)} trials")
        print()

    return download_trials(trials, dest_dir, verbose=verbose)


def summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in CATALOG:
        counts[t.category] = counts.get(t.category, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Download SFU Motion Capture Database BVH files"
    )
    ap.add_argument(
        "--categories", "-c", nargs="*", default=None,
        help="Motion categories to download (default: all). Options: " + ", ".join(summary().keys()),
    )
    ap.add_argument(
        "--dest", "-d", default="mocap_data/sfu",
        help="Destination directory (default: mocap_data/sfu/)",
    )
    ap.add_argument(
        "--max-per-category", "-m", type=int, default=None,
        help="Max trials per category (default: unlimited)",
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List available categories and trial counts, then exit",
    )
    args = ap.parse_args()

    if args.list:
        print("Available categories:")
        for cat, count in summary().items():
            print(f"  {cat:<15} {count:3d} trials")
        print(f"  {'TOTAL':<15} {len(CATALOG):3d} trials")
        return

    download_all(
        dest_dir=args.dest,
        categories=args.categories,
        max_per_category=args.max_per_category,
    )


if __name__ == "__main__":
    main()
