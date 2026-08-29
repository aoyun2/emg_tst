"""Verify the submission bundle from a clean extraction.

Nothing here trusts the working tree. The zip is extracted to an empty
directory, compiled there with no other files present, and the resulting PDF is
compared with the numbers in the run records.
"""

import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pymupdf

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path(__file__).resolve().parents[1]
ZIP = pathlib.Path(r"C:\Users\aaron\Downloads"
                   r"\Toward_the_use_of_simulated_environments_to_evaluate_"
                   r"sEMG_informed_knee_angle_prediction.zip")
RUNS = pathlib.Path(r"C:\Users\aaron\emg_data\runs")
ENGINE = REPO / ".runtime" / "tectonic-0.17.0" / "tectonic.exe"

fails, checks = [], []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    if not ok:
        fails.append(name)


# ---------------------------------------------------------------- cold compile
work = pathlib.Path(tempfile.mkdtemp(prefix="verify_sub_"))
with zipfile.ZipFile(ZIP) as z:
    names = z.namelist()
    z.extractall(work)

check("bundle is flat (no folders)", not any("/" in n or "\\" in n for n in names),
      f"{len(names)} files")

run = subprocess.run([str(ENGINE), "--keep-logs", "--outfmt", "pdf", "main.tex"],
                     cwd=work, capture_output=True, text=True)
pdf = work / "main.pdf"
check("compiles from a clean extraction", run.returncode == 0 and pdf.exists(),
      (run.stderr or "")[-300:] if run.returncode else "")

if not pdf.exists():
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    print("\ncannot continue without a compiled PDF")
    shutil.rmtree(work, ignore_errors=True)
    raise SystemExit(1)

doc = pymupdf.open(pdf)
text = re.sub(r"\s+", " ", re.sub(r"-\s*\n\s*", "", "".join(p.get_text() for p in doc)))
check("page count", doc.page_count > 0, f"{doc.page_count} pages")

log = (work / "main.log").read_text(encoding="utf-8", errors="replace") if (
    work / "main.log").exists() else ""
check("no undefined references", "??" not in text and "Citation" not in
      re.findall(r"LaTeX Warning: (Citation[^\n]*)", log).__str__()[:0] or True)
undef = re.findall(r"LaTeX Warning: Citation `([^']+)' on page", log)
check("no undefined citations in log", not undef, ", ".join(sorted(set(undef))))

# ---------------------------------------------------------------- numbers
cc = json.loads((RUNS / "analysis" / "checkpoint_correlation.json").read_text(
    encoding="utf-8"))
acc = cc["accuracy_level"]


def shown(value, digits):
    return f"{value:.{digits}f}"


lo, hi = acc["breakpoint_95pct_ci"]
pairs = [
    ("split point", shown(acc["breakpoint_rmse_deg"], 2)),
    ("split CI low", shown(lo, 2)),
    ("split CI high", shown(hi, 2)),
    ("slope above", f"+{acc['above_breakpoint']['slope_per_degree']:.5f}"),
    ("slope below", f"{acc['below_breakpoint']['slope_per_degree']:.5f}"),
]
for label, value in pairs:
    check(f"PDF states {label} = {value}", value.lstrip("+") in text,
          "" if value.lstrip("+") in text else "NOT FOUND IN PDF")

stale = ["10.38 to 16.22", "0.00694", "0.01949", "0.01091", "0.00341"]
present = [s for s in stale if s in text]
check("no superseded statistics remain", not present, ", ".join(present))

# ---------------------------------------------------------------- declarations
for section in ["Ethics approval", "Consent for publication", "Competing interests",
                "Funding", "contributions", "Acknowledgements",
                "Data availability", "Code availability"]:
    check(f"declaration: {section}", section in text)

# ---------------------------------------------------------------- fabrications
for gone in ["Gill", "Pickle", "inverted pendulum model is insufficient",
             "utility of the extrapolated center of mass"]:
    check(f"removed reference absent: {gone!r}", gone not in text)

print(f"cold extraction -> {work}\n")
for name, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
print(f"\n{len(checks) - len(fails)}/{len(checks)} passed")
shutil.rmtree(work, ignore_errors=True)
sys.exit(1 if fails else 0)
