"""
Batch downloader for CMU Graphics Lab Motion Capture Database BVH files.

Downloads trials listed in :mod:`cmu_catalog` with retry logic,
progress reporting, and category-based filtering.

Files are saved to ``{dest_dir}/{subject:02d}_{trial:02d}.bvh``.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

from tqdm import tqdm

from mocap_evaluation.cmu_catalog import (
    CATALOG,
    TrialInfo,
    filter_trials,
    summary,
)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

_RETRY_DELAYS = [2, 4, 8, 16]  # exponential backoff seconds


def _download_one(
    trial: TrialInfo,
    dest_dir: Path,
    *,
    timeout: float = 30.0,
    retries: int = 4,
    verbose: bool = True,
) -> bool:
    """
    Download a single BVH file with retry logic.

    Returns True on success, False on failure.
    """
    dest = dest_dir / trial.filename
    if dest.exists() and dest.stat().st_size > 0:
        if verbose:
            print(f"  [skip] {trial.filename} (already exists)")
        return True

    url_https = trial.url
    url_http = url_https.replace("https://", "http://")
    urls = [url_https, url_http]

    for attempt in range(retries):
        for url in urls:
            try:
                if verbose and attempt == 0:
                    print(f"  [download] {trial.filename} ← {url}")
                elif verbose:
                    print(f"  [retry {attempt+1}] {trial.filename}")
                urllib.request.urlretrieve(url, dest)
                if dest.stat().st_size > 100:
                    return True
                # File too small — probably an error page
                dest.unlink(missing_ok=True)
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
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


def download_trials(
    trials: List[TrialInfo],
    dest_dir: str | Path = "mocap_data/cmu",
    *,
    verbose: bool = True,
) -> List[Path]:
    """
    Download a list of trials to dest_dir.

    Returns list of successfully downloaded file paths.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    downloaded: List[Path] = []
    n = len(trials)

    for trial in tqdm(trials, desc="Downloading CMU", unit="file", disable=not verbose):
        ok = _download_one(trial, dest_dir, verbose=False)
        if ok:
            downloaded.append(dest_dir / trial.filename)

    if verbose:
        print(f"\nDownloaded {len(downloaded)}/{n} files to {dest_dir}/")

    return downloaded


def download_by_category(
    categories: Optional[Sequence[str]] = None,
    dest_dir: str | Path = "mocap_data/cmu",
    max_per_category: Optional[int] = None,
    verbose: bool = True,
) -> List[Path]:
    """
    Download CMU BVH files filtered by motion category.

    Parameters
    ----------
    categories       : list of category names (None = all categories)
    dest_dir         : output directory for BVH files
    max_per_category : limit downloads per category (None = all)
    verbose          : print progress

    Returns
    -------
    List of downloaded file paths.
    """
    trials = filter_trials(
        categories=categories,
        max_per_category=max_per_category,
    )

    if verbose:
        print(f"CMU Mocap Download Plan:")
        cat_counts = {}
        for t in trials:
            cat_counts[t.category] = cat_counts.get(t.category, 0) + 1
        for cat, count in sorted(cat_counts.items()):
            print(f"  {cat:<15} {count} trials")
        print(f"  {'TOTAL':<15} {len(trials)} trials")
        print()

    return download_trials(trials, dest_dir, verbose=verbose)


def download_all(
    dest_dir: str | Path = "mocap_data/cmu",
    max_per_category: Optional[int] = None,
    verbose: bool = True,
) -> List[Path]:
    """Download all cataloged trials."""
    return download_by_category(
        categories=None,
        dest_dir=dest_dir,
        max_per_category=max_per_category,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Download CMU mocap BVH files by motion category"
    )
    ap.add_argument(
        "--categories", "-c", nargs="*", default=None,
        help="Motion categories to download (default: all). "
             "Options: " + ", ".join(summary().keys()),
    )
    ap.add_argument(
        "--dest", "-d", default="mocap_data/cmu",
        help="Destination directory (default: mocap_data/cmu/)",
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

    download_by_category(
        categories=args.categories,
        dest_dir=args.dest,
        max_per_category=args.max_per_category,
    )


if __name__ == "__main__":
    main()
