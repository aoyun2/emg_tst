"""Check the table and supplementary values against the run records.

verify_submission checks the headline statistics quoted in the prose. This
checks the numbers in Table 1, Table 2 and the knee-input supplementary
analysis, which are quoted nowhere else and so are not covered by it.

A check that cannot locate its data fails rather than passing quietly: an
earlier version of this script reported success on Table 2 because it looked
for the surrogate effects under the wrong key and found nothing.

    python -m analysis.verify_tables
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

REPO = Path(__file__).resolve().parents[1]


def _text(pdf: Path) -> str:
    raw = "".join(page.get_text() for page in pymupdf.open(pdf))
    lines = [ln for ln in raw.splitlines() if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]
    joined = "\n".join(lines)
    return re.sub(r"\s{2,}", " ", re.sub(r"\n", " ", re.sub(r"-\n", "", joined)))


def _find_effects(node: object, out: dict[str, float]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict) and "mean_improvement_deg" in value:
                out[key] = float(value["mean_improvement_deg"])
            _find_effects(value, out)
    elif isinstance(node, list):
        for value in node:
            _find_effects(value, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path,
                        default=Path(r"C:\Users\aaron\emg_data\runs"))
    parser.add_argument("--pdf", type=Path,
                        default=Path(r"C:\Users\aaron\Downloads\main_proof.pdf"))
    args = parser.parse_args()

    text = _text(args.pdf)
    failures: list[str] = []
    passed = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed
        if ok:
            passed += 1
        else:
            failures.append(name)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))

    correlation = json.loads(
        (args.runs_dir / "analysis" / "checkpoint_correlation.json").read_text(
            encoding="utf-8"))
    rows = correlation["per_checkpoint"]
    if not rows:
        print("FAIL  Table 1 has no checkpoint rows in the run record")
        return 1
    print(f"Table 1 -- {len(rows)} checkpoint rows")
    for row in rows:
        rmse = f"{row['mean_prediction_rmse_deg']:.2f}"
        sd = f"{row['prediction_rmse_sd_deg']:.2f}"
        check(f"RMSE {rmse}, SD {sd}", rmse in text and sd in text)

    print("\nTable 2 -- surrogate controls")
    effects: dict[str, float] = {}
    _find_effects(
        json.loads((args.runs_dir / "semg_controls" / "semg_controls.json").read_text(
            encoding="utf-8")), effects)
    if not effects:
        check("surrogate effects located in the run record", False,
              "no mean_improvement_deg anywhere")
    for name, value in sorted(effects.items()):
        shown = f"{value:+.3f}".replace("+", "")
        check(f"{name} = {value:+.4f}", shown in text, shown)

    print("\nSupplementary -- knee-angle input sensitivity")
    kin = json.loads(
        (args.runs_dir / "kinematic_input_check" / "kinematic_input_check.json")
        .read_text(encoding="utf-8"))
    penalty = kin.get("verdict", {}).get("kinematic_penalty_for_dropping_knee_deg")
    if penalty is None:
        check("penalty located in the run record", False, "key missing")
    else:
        check(f"penalty for dropping the knee = {penalty:.3f}",
              f"{penalty:.3f}" in text)

    total = passed + len(failures)
    print(f"\n{passed}/{total} matched")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
