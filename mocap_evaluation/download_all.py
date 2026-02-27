"""
Download all mocap datasets into their respective sub-folders.

Usage
-----
    python -m mocap_evaluation.download_all
    python -m mocap_evaluation.download_all --root mocap_data

This downloads:
    mocap_data/bandai/      — Bandai Namco Motiondataset-2 (~2,900 BVH files)
    mocap_data/bandai_ds1/  — Bandai Namco Motiondataset-1 (175 BVH files)
    mocap_data/cmu/         — CMU Graphics Lab (~2,435 BVH files)
    mocap_data/lafan1/      — Ubisoft LAFAN1 (77 BVH files)
    mocap_data/sfu/         — SFU Motion Capture (46 BVH files)
"""
from __future__ import annotations

import argparse
from pathlib import Path


def download_everything(root: str | Path = "mocap_data", verbose: bool = True) -> None:
    root = Path(root)

    # ── Bandai Namco Dataset-2 ─────────────────────────────────────────
    try:
        from mocap_evaluation.bandai_namco_downloader import download_locomotion
        print("=" * 60)
        print("Downloading Bandai Namco Motiondataset-2 (locomotion)")
        print("=" * 60)
        download_locomotion(dest_dir=root / "bandai", verbose=verbose)
    except Exception as exc:
        print(f"Bandai Namco DS2 download failed: {exc}")

    # ── Bandai Namco Dataset-1 ─────────────────────────────────────────
    try:
        from mocap_evaluation.bandai_namco_ds1_downloader import download_all as ds1_download
        print("\n" + "=" * 60)
        print("Downloading Bandai Namco Motiondataset-1")
        print("=" * 60)
        ds1_download(dest_dir=root / "bandai_ds1", verbose=verbose)
    except Exception as exc:
        print(f"Bandai Namco DS1 download failed: {exc}")

    # ── CMU ───────────────────────────────────────────────────────────────
    try:
        from mocap_evaluation.cmu_downloader import download_all as cmu_download
        print("\n" + "=" * 60)
        print("Downloading CMU Motion Capture Database")
        print("=" * 60)
        cmu_download(dest_dir=root / "cmu", verbose=verbose)
    except Exception as exc:
        print(f"CMU download failed: {exc}")

    # ── LAFAN1 ────────────────────────────────────────────────────────────
    try:
        from mocap_evaluation.lafan1_downloader import download_all as lafan1_download
        print("\n" + "=" * 60)
        print("Downloading Ubisoft LAFAN1")
        print("=" * 60)
        lafan1_download(dest_dir=root / "lafan1", verbose=verbose)
    except Exception as exc:
        print(f"LAFAN1 download failed: {exc}")

    # ── SFU ───────────────────────────────────────────────────────────────
    try:
        from mocap_evaluation.sfu_downloader import download_all as sfu_download
        print("\n" + "=" * 60)
        print("Downloading SFU Motion Capture Database")
        print("=" * 60)
        sfu_download(dest_dir=root / "sfu", verbose=verbose)
    except Exception as exc:
        print(f"SFU download failed: {exc}")

    print("\n" + "=" * 60)
    print("All downloads complete.")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(description="Download all mocap datasets")
    ap.add_argument(
        "--root", "-r", default="mocap_data",
        help="Root directory for all mocap data (default: mocap_data/)",
    )
    args = ap.parse_args()
    download_everything(root=args.root)


if __name__ == "__main__":
    main()
