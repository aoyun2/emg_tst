"""
Downloader for Bandai Namco Research Motiondataset-2 BVH files.

Source
------
Bandai Namco Research Inc. — Bandai-Namco-Research-Motiondataset
  https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset
License: CC BY-NC 4.0 (non-commercial use with attribution)

Dataset-2 overview
------------------
  • ~2,902 BVH files at 30 fps
  • 10 motion types: walk, walk-turn-left, walk-turn-right, run,
      wave-both-hands, wave-left-hand, wave-right-hand,
      raise-up-both-hands, raise-up-left-hand, raise-up-right-hand
  • 7 styles: active, elderly, exhausted, feminine, masculine, normal, youthful
  • Skeleton: UpperLeg_R/LowerLeg_R/Foot_R … (auto-detected by mocap_loader)

This downloader targets the **locomotion** subset (walk + walk-turn variants)
and optionally run, to maximise walking-motion coverage for prosthetic gait
matching.  Files are stored as ``dataset-2_{motion}_{style}_{id:03d}.bvh``
in the destination directory.

Usage
-----
    python -m mocap_evaluation.bandai_namco_downloader
    python -m mocap_evaluation.bandai_namco_downloader --dest mocap_data/ --motions walk run
    python -m mocap_evaluation.bandai_namco_downloader --list
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
    "BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/"
    "master/dataset/Bandai-Namco-Research-Motiondataset-2/data/"
)

MOTIONS_LOCOMOTION: List[str] = [
    "walk",
    "walk-turn-left",
    "walk-turn-right",
]

MOTIONS_ALL: List[str] = MOTIONS_LOCOMOTION + [
    "run",
    "wave-both-hands",
    "wave-left-hand",
    "wave-right-hand",
    "raise-up-both-hands",
    "raise-up-left-hand",
    "raise-up-right-hand",
]

STYLES: List[str] = [
    "active",
    "elderly",
    "exhausted",
    "feminine",
    "masculine",
    "normal",
    "youthful",
]

_RETRY_DELAYS = [2, 4, 8, 16]   # exponential backoff (seconds)
_MAX_ID       = 200              # scan IDs 001…_MAX_ID per motion×style combo


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_one(
    filename: str,
    dest_dir: Path,
    *,
    timeout: float = 30.0,
    retries: int = 4,
    verbose: bool = True,
) -> bool:
    """
    Download a single BVH file.  Returns True on success, False if the file
    does not exist (404) or all retries failed.
    """
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return True   # already downloaded

    url = _BASE_URL + filename

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 100:
                if verbose:
                    print(f"  [ok] {filename}")
                return True
            dest.unlink(missing_ok=True)
            return False   # tiny file → probably 404 page, stop trying
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                dest.unlink(missing_ok=True)
                return False   # file does not exist — stop immediately
            # Other HTTP error → retry
        except (urllib.error.URLError, OSError):
            pass   # network error → retry

        if attempt < retries - 1:
            delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
            time.sleep(delay)

    if verbose:
        print(f"  [FAILED] {filename}")
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_locomotion(
    dest_dir: str | Path = "mocap_data/bandai",
    styles: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> List[Path]:
    """
    Download all walking / locomotion BVH files from Bandai Namco Dataset-2.

    Downloads ``walk``, ``walk-turn-left``, and ``walk-turn-right`` motions
    in all seven styles (active, elderly, exhausted, feminine, masculine,
    normal, youthful).

    Parameters
    ----------
    dest_dir : destination directory (created if absent)
    styles   : subset of styles to download (default: all seven)
    verbose  : print progress

    Returns
    -------
    List of successfully downloaded file paths.
    """
    return _download_motions(
        MOTIONS_LOCOMOTION,
        dest_dir=dest_dir,
        styles=styles,
        verbose=verbose,
    )


def download_all(
    dest_dir: str | Path = "mocap_data/bandai",
    motions: Optional[Sequence[str]] = None,
    styles: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> List[Path]:
    """
    Download Bandai Namco Dataset-2 BVH files.

    Parameters
    ----------
    dest_dir : destination directory
    motions  : motion types to download (default: walk + walk-turn variants)
    styles   : styles to download (default: all seven)
    verbose  : print progress
    """
    if motions is None:
        motions = MOTIONS_LOCOMOTION
    return _download_motions(motions, dest_dir=dest_dir, styles=styles, verbose=verbose)


def _download_motions(
    motions: Sequence[str],
    dest_dir: str | Path,
    styles: Optional[Sequence[str]],
    verbose: bool,
) -> List[Path]:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if styles is None:
        styles = STYLES

    downloaded: List[Path] = []
    total_ok = 0
    total_tried = 0

    for motion in motions:
        for style in styles:
            if verbose:
                print(f"\n[Bandai Namco] {motion} / {style}")
            for idx in range(1, _MAX_ID + 1):
                fname = f"dataset-2_{motion}_{style}_{idx:03d}.bvh"
                ok = _download_one(fname, dest_dir, verbose=verbose)
                total_tried += 1
                if ok:
                    downloaded.append(dest_dir / fname)
                    total_ok += 1
                else:
                    # First 404 → this style's files are exhausted
                    if verbose:
                        print(f"  (no more files for {motion}/{style} after id {idx})")
                    break

    if verbose:
        print(f"\nBandai Namco: downloaded {total_ok} files to {dest_dir}/")

    return downloaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Download Bandai Namco Research Motiondataset-2 BVH files.\n"
            "License: CC BY-NC 4.0 — non-commercial use with attribution.\n"
            "Source:  https://github.com/BandaiNamcoResearchInc/"
            "Bandai-Namco-Research-Motiondataset"
        )
    )
    ap.add_argument(
        "--dest", "-d", default="mocap_data/bandai",
        help="Destination directory (default: mocap_data/bandai/)",
    )
    ap.add_argument(
        "--motions", "-m", nargs="*", default=None,
        help=(
            "Motion types to download (default: walk + walk-turn variants). "
            "Options: " + ", ".join(MOTIONS_ALL)
        ),
    )
    ap.add_argument(
        "--styles", "-s", nargs="*", default=None,
        help=(
            "Styles to download (default: all). "
            "Options: " + ", ".join(STYLES)
        ),
    )
    ap.add_argument(
        "--list", action="store_true",
        help="List available motions and styles, then exit",
    )
    args = ap.parse_args()

    if args.list:
        print("Available motions:")
        for m in MOTIONS_ALL:
            print(f"  {m}")
        print("\nAvailable styles:")
        for s in STYLES:
            print(f"  {s}")
        return

    download_all(
        dest_dir=args.dest,
        motions=args.motions,
        styles=args.styles,
    )


if __name__ == "__main__":
    main()
