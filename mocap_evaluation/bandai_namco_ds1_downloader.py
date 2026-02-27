"""
Downloader for Bandai Namco Research Motiondataset-1 BVH files.

Source
------
Bandai Namco Research Inc. — Bandai-Namco-Research-Motiondataset
  https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset
License: CC BY-NC 4.0 (non-commercial use with attribution)

Dataset-1 overview
------------------
  • 175 BVH files at 30 fps, 36,673 total frames
  • 17 motion types: locomotion (walk, run, dash, walk-back/left/right),
      gestures (bow, bye, byebye, guide, respond, call),
      combat (punch, kick, slash), dance (dance-long, dance-short)
  • 15 styles: normal, happy, sad, angry, proud, not-confident, masculinity,
      feminine, childish, old, tired, active, musical, giant, chimpira
  • Not all actions support all styles — combat/call/respond/dance actions
      only have the "normal" style
  • Skeleton: same as Dataset-2 (UpperLeg_R/LowerLeg_R/Foot_R …)

Usage
-----
    python -m mocap_evaluation.bandai_namco_ds1_downloader
    python -m mocap_evaluation.bandai_namco_ds1_downloader --dest mocap_data/bandai_ds1
    python -m mocap_evaluation.bandai_namco_ds1_downloader --list
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Sequence

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/"
    "master/dataset/Bandai-Namco-Research-Motiondataset-1/data/"
)

STYLES: List[str] = [
    "normal", "happy", "sad", "angry", "proud",
    "not-confident", "masculinity", "feminine", "childish",
    "old", "tired", "active", "musical", "giant", "chimpira",
]

# Actions that support all 15 styles, with (action_name, max_id) tuples.
_MULTI_STYLE_ACTIONS: List[tuple[str, int]] = [
    ("walk",       2),   # 001-002 per style
    ("run",        1),
    ("dash",       1),
    ("walk-back",  1),
    ("walk-left",  1),
    ("walk-right", 1),
    ("bow",        1),
    ("bye",        1),
    ("byebye",     1),
    ("guide",      1),
]

# Actions with only the "normal" style.
_NORMAL_ONLY_ACTIONS: List[tuple[str, int]] = [
    ("respond",     1),
    ("punch",       2),   # 001-002
    ("slash",       2),   # 001-002
    ("kick",        1),
    ("call",        2),   # 001-002
    ("dance-long",  1),
    ("dance-short", 1),
]

MOTIONS_LOCOMOTION: List[str] = [
    "walk", "run", "dash", "walk-back", "walk-left", "walk-right",
]

MOTIONS_ALL: List[str] = (
    [a for a, _ in _MULTI_STYLE_ACTIONS] +
    [a for a, _ in _NORMAL_ONLY_ACTIONS]
)

_RETRY_DELAYS = [2, 4, 8, 16]


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
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return True

    url = _BASE_URL + filename

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size > 100:
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


def _iter_files(
    motions: Optional[Sequence[str]] = None,
    styles: Optional[Sequence[str]] = None,
) -> List[str]:
    """Generate the list of filenames to download."""
    if styles is None:
        styles = STYLES
    style_set = set(styles)

    multi_map = {a: n for a, n in _MULTI_STYLE_ACTIONS}
    normal_map = {a: n for a, n in _NORMAL_ONLY_ACTIONS}

    filenames: List[str] = []

    for action in (motions if motions is not None else MOTIONS_ALL):
        if action in multi_map:
            max_id = multi_map[action]
            for style in styles:
                for idx in range(1, max_id + 1):
                    filenames.append(f"dataset-1_{action}_{style}_{idx:03d}.bvh")
        elif action in normal_map:
            max_id = normal_map[action]
            if "normal" in style_set or motions is not None:
                for idx in range(1, max_id + 1):
                    filenames.append(f"dataset-1_{action}_normal_{idx:03d}.bvh")

    return filenames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_locomotion(
    dest_dir: str | Path = "mocap_data/bandai_ds1",
    styles: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> List[Path]:
    """Download locomotion subset (walk, run, dash, walk-back/left/right)."""
    return download_all(
        dest_dir=dest_dir,
        motions=MOTIONS_LOCOMOTION,
        styles=styles,
        verbose=verbose,
    )


def download_all(
    dest_dir: str | Path = "mocap_data/bandai_ds1",
    motions: Optional[Sequence[str]] = None,
    styles: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> List[Path]:
    """Download Bandai Namco Dataset-1 BVH files."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filenames = _iter_files(motions=motions, styles=styles)
    downloaded: List[Path] = []

    pbar = tqdm(filenames, desc="Downloading Bandai DS1", unit="file", disable=not verbose)
    for fname in pbar:
        ok = _download_one(fname, dest_dir, verbose=False)
        if ok:
            downloaded.append(dest_dir / fname)

    if verbose:
        print(f"\nBandai Namco DS1: downloaded {len(downloaded)}/{len(filenames)} files to {dest_dir}/")

    return downloaded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Download Bandai Namco Research Motiondataset-1 BVH files.\n"
            "License: CC BY-NC 4.0 — non-commercial use with attribution.\n"
            "Source:  https://github.com/BandaiNamcoResearchInc/"
            "Bandai-Namco-Research-Motiondataset"
        )
    )
    ap.add_argument(
        "--dest", "-d", default="mocap_data/bandai_ds1",
        help="Destination directory (default: mocap_data/bandai_ds1/)",
    )
    ap.add_argument(
        "--motions", "-m", nargs="*", default=None,
        help=(
            "Motion types to download (default: all). "
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
        print("Multi-style actions (support all 15 styles):")
        for a, n in _MULTI_STYLE_ACTIONS:
            print(f"  {a:<20} ({n} file(s) per style × {len(STYLES)} styles)")
        print("\nNormal-only actions:")
        for a, n in _NORMAL_ONLY_ACTIONS:
            print(f"  {a:<20} ({n} file(s), normal style only)")
        print(f"\nStyles ({len(STYLES)}): {', '.join(STYLES)}")
        total = len(_iter_files())
        print(f"\nTotal BVH files: {total}")
        return

    download_all(
        dest_dir=args.dest,
        motions=args.motions,
        styles=args.styles,
    )


if __name__ == "__main__":
    main()
