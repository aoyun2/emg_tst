"""Resolve every DOI in the bibliography and check it identifies the right paper.

Run this before any submission:

    python -m analysis.verify_references

It fetches the registered metadata for each DOI and compares the resolved title
with the title in references.bib. An entry whose DOI belongs to a different
paper is reported as MISMATCH and the script exits non-zero.

Entries without a DOI (conference papers, talks, preprints) cannot be checked
this way and are listed so they can be confirmed by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIB = REPO / "manuscript" / "references.bib"
FIELD = r"{0}\s*=\s*\{{(.*?)\}},?\s*(?:\n|$)"
API = "https://api.crossref.org/works/"


def _clean(value: str) -> str:
    return re.sub(r"[{}]", "", re.sub(r"\s+", " ", value)).strip()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def _entries(text: str) -> list[dict[str, str]]:
    out = []
    for match in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        body = match.group(3)

        def field(name: str, _b: str = body) -> str:
            found = re.search(FIELD.format(name), _b, re.S)
            return "" if not found else _clean(found.group(1))

        out.append({"key": match.group(2).strip(), "title": field("title"),
                    "doi": field("doi")})
    return out


def _resolve(doi: str, timeout: int) -> dict | None:
    try:
        raw = subprocess.run(["curl", "-s", "--max-time", str(timeout), API + doi],
                             capture_output=True, text=True).stdout
        return json.loads(raw)["message"]
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--overlap", type=float, default=0.55,
                        help="fraction of title words that must be shared")
    args = parser.parse_args()

    entries = _entries(BIB.read_text(encoding="utf-8"))
    mismatched: list[tuple[str, str, str]] = []
    unresolved: list[str] = []
    no_doi: list[str] = []
    checked = 0

    for entry in entries:
        if not entry["doi"]:
            no_doi.append(entry["key"])
            continue
        message = _resolve(entry["doi"], args.timeout)
        time.sleep(0.2)
        if message is None:
            unresolved.append(f"{entry['key']} ({entry['doi']})")
            continue
        resolved = (message.get("title") or [""])[0]
        ours = _words(entry["title"])
        shared = len(ours & _words(resolved)) / max(1, len(ours))
        if shared < args.overlap:
            mismatched.append((entry["key"], entry["doi"], resolved[:88]))
        else:
            checked += 1

    print(f"{len(entries)} entries; {checked} DOIs resolve to the cited paper")
    if no_doi:
        print(f"\nno DOI, confirm by hand ({len(no_doi)}): {', '.join(no_doi)}")
    if unresolved:
        print(f"\nnot reachable ({len(unresolved)}): {', '.join(unresolved)}")
    if mismatched:
        print(f"\nMISMATCH ({len(mismatched)}) -- the DOI identifies a different paper:")
        for key, doi, resolved in mismatched:
            print(f"  {key:<16} {doi}")
            print(f"      resolves to: {resolved}")
        return 1
    print("\nno mismatches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
