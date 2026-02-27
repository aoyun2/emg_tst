"""
Downloader for SFU (Simon Fraser University) Motion Capture Database BVH files.

Source
------
SFU Motion Capture Lab
  https://mocap.cs.sfu.ca/
License: Free for academic / research use.

Dataset overview
----------------
  • ~46 BVH files across 8 subjects and diverse motion types
  • Motions include walking, jogging, jumping, dance, parkour, kendo, yoga, etc.
  • Skeleton: standard MotionBuilder naming (RightUpLeg, RightLeg, …)
  • FPS: 120 Hz

The SFU database serves individual BVH files from two URL prefixes
(``nusmocap/`` for most subjects, ``nusmocapnew/`` for subject 0019).
This downloader fetches all verified trials.

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

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://mocap.cs.sfu.ca/nusmocap/"
_BASE_URL_NEW = "https://mocap.cs.sfu.ca/nusmocapnew/"


class SFUTrial(NamedTuple):
    filename: str
    category: str
    description: str
    base_url: str = _BASE_URL  # override for nusmocapnew files


# Verified catalog of SFU BVH files — every entry confirmed via HTTP 200.
# Subject numbering and filenames follow the SFU database conventions.
# File availability verified against https://mocap.cs.sfu.ca/ on 2026-02-27.
CATALOG: List[SFUTrial] = [
    # ── Subject 0005 — Male, age 18-25 ──────────────────────────────────
    SFUTrial("0005_Walking001.bvh",        "walk",       "Normal walking"),
    SFUTrial("0005_BackwardsWalk001.bvh",  "walk",       "Walking backwards"),
    SFUTrial("0005_Jogging001.bvh",        "run",        "Jogging"),
    SFUTrial("0005_SlowTrot001.bvh",       "run",        "Slow trot"),
    SFUTrial("0005_SideSkip001.bvh",       "locomotion", "Side skipping"),
    SFUTrial("0005_Stomping001.bvh",       "locomotion", "Stomping"),
    SFUTrial("0005_2FeetJump001.bvh",      "jump",       "Two-feet jump"),
    SFUTrial("0005_JumpRope001.bvh",       "jump",       "Jump rope"),

    # ── Subject 0007 — Male, age 18-25, 175cm 60kg ─────────────────────
    SFUTrial("0007_Walking001.bvh",        "walk",       "Normal walking"),
    SFUTrial("0007_Crawling001.bvh",       "locomotion", "Crawling"),
    SFUTrial("0007_Balance001.bvh",        "balance",    "Balancing"),
    SFUTrial("0007_Cartwheel001.bvh",      "acrobatics", "Cartwheel"),

    # ── Subject 0008 — Female, age 25-30 ───────────────────────────────
    SFUTrial("0008_Walking001.bvh",        "walk",       "Normal walking"),
    SFUTrial("0008_Walking002.bvh",        "walk",       "Walking variation"),
    SFUTrial("0008_Skipping001.bvh",       "locomotion", "Skipping"),
    SFUTrial("0008_ChaCha001.bvh",         "dance",      "Cha-cha dance"),
    SFUTrial("0008_Yoga001.bvh",           "sport",      "Yoga sequence"),

    # ── Subject 0012 — Male, age 18-25, 174cm 74kg ─────────────────────
    SFUTrial("0012_JumpAndRoll001.bvh",    "acrobatics", "Jump and roll"),
    SFUTrial("0012_SpeedVault001.bvh",     "acrobatics", "Speed vault"),
    SFUTrial("0012_SpeedVault002.bvh",     "acrobatics", "Speed vault variation"),

    # ── Subject 0015 — Male, age 18-25, 171cm 70kg ─────────────────────
    SFUTrial("0015_HopOverObstacle001.bvh",  "obstacle",     "Hop over obstacle"),
    SFUTrial("0015_JumpOverObstacle001.bvh", "obstacle",     "Jump over obstacle"),
    SFUTrial("0015_BasicKendo001.bvh",       "martial_arts", "Basic kendo"),
    SFUTrial("0015_Kirikaeshi001.bvh",       "martial_arts", "Kirikaeshi (kendo drill)"),
    SFUTrial("0015_KendoKata001.bvh",        "martial_arts", "Kendo kata"),

    # ── Subject 0017 — Male, age 18-25, 165cm 63kg ─────────────────────
    SFUTrial("0017_ParkourRoll001.bvh",    "acrobatics",   "Parkour roll"),
    SFUTrial("0017_JumpAndRoll001.bvh",    "acrobatics",   "Jump and roll"),
    SFUTrial("0017_JumpingOnBench001.bvh", "obstacle",     "Jumping on bench"),
    SFUTrial("0017_MonkeyVault001.bvh",    "acrobatics",   "Monkey vault"),
    SFUTrial("0017_RunningOnBench001.bvh", "obstacle",     "Running on bench"),
    SFUTrial("0017_RunningOnBench002.bvh", "obstacle",     "Running on bench variation"),
    SFUTrial("0017_SpeedVault001.bvh",     "acrobatics",   "Speed vault"),
    SFUTrial("0017_SpeedVault002.bvh",     "acrobatics",   "Speed vault variation"),
    SFUTrial("0017_WushuKicks001.bvh",     "martial_arts", "Wushu kicks"),

    # ── Subject 0018 — Female, age 18-25, 161cm 55kg ───────────────────
    SFUTrial("0018_Walking001.bvh",                   "walk",  "Normal walking"),
    SFUTrial("0018_Catwalk001.bvh",                   "walk",  "Catwalk"),
    SFUTrial("0018_TipToe001.bvh",                    "walk",  "Tiptoeing"),
    SFUTrial("0018_Moonwalk001.bvh",                  "walk",  "Moonwalk"),
    SFUTrial("0018_Bridge001.bvh",                    "sport", "Bridge pose"),
    SFUTrial("0018_DanceTurns001.bvh",                "dance", "Dance turns"),
    SFUTrial("0018_DanceTurns002.bvh",                "dance", "Dance turns variation"),
    SFUTrial("0018_TraditionalChineseDance001.bvh",   "dance", "Traditional Chinese dance"),
    SFUTrial("0018_XinJiang002.bvh",                  "dance", "Xinjiang dance variation 2"),
    SFUTrial("0018_XinJiang003.bvh",                  "dance", "Xinjiang dance variation 3"),

    # ── Subject 0019 — Male, age 18-25, 174cm 63kg (nusmocapnew path) ──
    SFUTrial("0019_BasicBollywoodDance001.bvh",   "dance", "Basic Bollywood dance",
             _BASE_URL_NEW),
    SFUTrial("0019_AdvanceBollywoodDance001.bvh", "dance", "Advanced Bollywood dance",
             _BASE_URL_NEW),
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

    url = trial.base_url + trial.filename

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

    for trial in tqdm(trials, desc="Downloading SFU", unit="file", disable=not verbose):
        ok = _download_one(trial, dest_dir, verbose=False)
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
