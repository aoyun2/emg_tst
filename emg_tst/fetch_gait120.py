"""Download the Gait120 dataset from figshare into a local data root.

The MoCapAct reference bank and its per-snippet experts are fetched on demand by
``mocap_phys_eval.experts``, but Gait120 had no fetcher, so a fresh checkout
could not rebuild the prediction cache without a manual download.  This module
closes that gap.

Files are streamed to ``<dest>/_downloads`` with resume support, verified
against the MD5 that figshare publishes for each file, and then extracted.  A
partial or corrupted download is never treated as complete.

Usage::

    python -m emg_tst.fetch_gait120 --dest data/gait120
    python -m emg_tst.fetch_gait120 --dest data/gait120 --list

After the download, build the prediction cache::

    python -m emg_tst.preprocess_gait120 \\
        --data-root data/gait120 --cache-dir data/gait120_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

# https://doi.org/10.6084/m9.figshare.27677016 -- the Gait120 data descriptor is
# https://doi.org/10.1038/s41597-025-05391-0
FIGSHARE_ARTICLE_ID = 27677016
FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}/files?page_size=1000"

CHUNK_BYTES = 1 << 20
USER_AGENT = "emg_tst-gait120-fetch/1.0"


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def list_files(article_id: int = FIGSHARE_ARTICLE_ID) -> list[dict[str, Any]]:
    """Return the file records figshare publishes for the article."""
    url = FIGSHARE_API.format(article_id=int(article_id))
    try:
        with urllib.request.urlopen(_request(url), timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"figshare returned HTTP {exc.code} for article {article_id}"
        ) from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not reach figshare: {exc.reason}") from exc
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"figshare article {article_id} lists no files")
    return payload


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(record: dict[str, Any], downloads: Path) -> Path:
    """Stream one file, resuming a partial transfer where the server allows it."""
    name = str(record["name"])
    expected_md5 = str(record.get("supplied_md5") or record.get("computed_md5") or "")
    expected_size = int(record.get("size", 0))
    final = downloads / name

    if final.exists() and (not expected_size or final.stat().st_size == expected_size):
        if not expected_md5 or _md5(final) == expected_md5:
            print(f"[gait120] have {name}")
            return final
        print(f"[gait120] {name} failed its checksum; re-downloading")
        final.unlink()

    part = final.with_suffix(final.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0

    # Part file exceeds the published size; restart it.
    if expected_size and have > expected_size:
        print(
            f"[gait120] {name}: partial file is {have - expected_size} bytes oversized, "
            "discarding and restarting"
        )
        part.unlink()
        have = 0

    # Part file is already complete; verify and rename it.
    if expected_size and have == expected_size:
        print(f"[gait120] {name}: transfer already complete, verifying")
    else:
        request = _request(str(record["download_url"]))
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            response = urllib.request.urlopen(request, timeout=300)
        except urllib.error.HTTPError as exc:
            if exc.code != 416:
                raise
            # Empty range; treat the file on disk as the whole transfer.
            print(f"[gait120] {name}: server reports no bytes remaining, verifying")
            response = None
        if response is not None:
            with response:
                resuming = response.status == 206
                if have and not resuming:
                    # Server ignored the range request; restart the transfer.
                    have = 0
                mode = "ab" if resuming and have else "wb"
                with part.open(mode) as stream:
                    written = have
                    while True:
                        block = response.read(CHUNK_BYTES)
                        if not block:
                            break
                        stream.write(block)
                        written += len(block)
                        if expected_size:
                            print(
                                f"\r[gait120] {name}: {written / expected_size:6.1%}",
                                end="",
                                flush=True,
                            )
            print()

    if expected_size and part.stat().st_size != expected_size:
        raise RuntimeError(
            f"{name} is {part.stat().st_size} bytes, expected {expected_size}"
        )
    if expected_md5:
        actual = _md5(part)
        if actual != expected_md5:
            raise RuntimeError(f"{name} checksum {actual} != published {expected_md5}")
    part.replace(final)
    return final


def _extract(archive: Path, dest: Path) -> None:
    if archive.suffix.lower() != ".zip":
        target = dest / archive.name
        if not target.exists():
            shutil.copy2(archive, target)
        return
    with zipfile.ZipFile(archive) as bundle:
        root = dest.resolve()
        for member in bundle.namelist():
            # Containment test, not a string prefix test.
            if not (dest / member).resolve().is_relative_to(root):
                raise RuntimeError(f"Refusing archive member outside dest: {member}")
        bundle.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, help="Required unless --list is given.")
    parser.add_argument("--article-id", type=int, default=FIGSHARE_ARTICLE_ID)
    parser.add_argument(
        "--list", action="store_true", help="Show the published files and exit."
    )
    args = parser.parse_args()
    if args.dest is None and not args.list:
        parser.error("--dest is required unless --list is given")

    records = list_files(int(args.article_id))
    if args.list:
        total = sum(int(r.get("size", 0)) for r in records)
        for record in records:
            print(f"{int(record.get('size', 0)) / 1e9:8.2f} GB  {record['name']}")
        print(f"{total / 1e9:8.2f} GB  total across {len(records)} files")
        return

    dest = args.dest.resolve()
    downloads = dest / "_downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    total = sum(int(r.get("size", 0)) for r in records)
    print(
        f"[gait120] {len(records)} files, {total / 1e9:.2f} GB from "
        f"figshare article {args.article_id} into {dest}"
    )
    for record in records:
        archive = _download(record, downloads)
        _extract(archive, dest)

    subjects = sorted(p.name for p in dest.glob("S[0-9][0-9][0-9]") if p.is_dir())
    print(f"[gait120] extracted {len(subjects)} subject directories into {dest}")
    if not subjects:
        print(
            "[gait120] no S### directories found; inspect the extracted layout and "
            "pass the directory that contains them as --data-root."
        )


if __name__ == "__main__":
    main()
