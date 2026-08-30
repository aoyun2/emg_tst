"""Run the reproducibility supplement's own instructions from a clean extraction.

The README puts the pipeline under code/ and then gives commands of the form
"python -m analysis.<module>", which only resolve with code/ on the path. This
extracts the archive, runs those commands the way its instructions read, and
checks the records it carries against the run.
"""

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
import re

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "manuscript" / "Additional_file_reproducibility.zip"
RUNS = pathlib.Path(r"C:\Users\aaron\emg_data\runs")
PY = r"C:\Users\aaron\emg_data\predenv\Scripts\python.exe"

fails, checks = [], []


def check(name, ok, detail=""):
    checks.append((name, ok, detail))
    if not ok:
        fails.append(name)


work = pathlib.Path(tempfile.mkdtemp(prefix="verify_arch_"))
with zipfile.ZipFile(ARCHIVE) as z:
    names = z.namelist()
    z.extractall(work)
print(f"extracted {len(names)} files -> {work}\n")

readme = (work / "README.txt").read_text(encoding="utf-8", errors="replace")
code = work / "code"

# the README names this file explicitly
check("docs/EXPERIMENT_PROTOCOL.md present, as the README says",
      (code / "docs" / "EXPERIMENT_PROTOCOL.md").exists()
      or (work / "docs" / "EXPERIMENT_PROTOCOL.md").exists())

# does the README tell the reader where to run from?
check("README says to run from code/",
      bool(re.search(r"cd\s+code|from\s+code/|inside\s+code/", readme)),
      "no 'cd code' anywhere in the README")

# every module the README invokes must resolve from code/, and so must every
# runner the archive ships, since a reviewer rerunning the physics imports it
modules = sorted(set(re.findall(r"python -m ([\w.]+)", readme)) | {
    "mocap_phys_eval.run_gait120_residual_fusion",
    "emg_tst.run_gait120_residual_fusion",
    "emg_tst.run_gait120_temporal_control",
    "analysis.gait120_conventional_paired_statistics",
})
for module in modules:
    run = subprocess.run([PY, "-c", f"import {module}"], cwd=code,
                         capture_output=True, text=True)
    tail = [ln for ln in (run.stderr or "").strip().splitlines() if ln.strip()]
    check(f"resolves from code/: {module}", run.returncode == 0,
          tail[-1][:88] if run.returncode and tail else "")

# actually run the first documented command end to end
out = work / "regen"
run = subprocess.run(
    [PY, "-m", "analysis.gait120_checkpoint_correlation",
     "--physics-run-dir", str(RUNS / "panel"), "--out-dir", str(out)],
    cwd=code, capture_output=True, text=True)
tail = [ln for ln in (run.stderr or "").strip().splitlines() if ln.strip()]
check("the documented regeneration command runs", run.returncode == 0,
      tail[-1][:88] if run.returncode and tail else "")

if (out / "checkpoint_correlation.json").exists():
    fresh = json.loads((out / "checkpoint_correlation.json").read_text(
        encoding="utf-8"))["accuracy_level"]
    live = json.loads((RUNS / "analysis" / "checkpoint_correlation.json").read_text(
        encoding="utf-8"))["accuracy_level"]
    check("regenerated split point matches the paper",
          abs(fresh["breakpoint_rmse_deg"] - live["breakpoint_rmse_deg"]) < 1e-9,
          f"{fresh['breakpoint_rmse_deg']:.4f}")
    check("regenerated interval matches the paper",
          max(abs(a - b) for a, b in zip(fresh["breakpoint_95pct_ci"],
                                         live["breakpoint_95pct_ci"])) < 1e-9,
          str([round(v, 3) for v in fresh["breakpoint_95pct_ci"]]))

carried = next((work / n for n in names if n.endswith("checkpoint_correlation.json")), None)
if carried:
    arch = json.loads(carried.read_text(encoding="utf-8"))["accuracy_level"]
    check("carried records are the corrected ones",
          abs(arch["breakpoint_95pct_ci"][0] - 10.3817) > 0.01,
          str([round(v, 3) for v in arch["breakpoint_95pct_ci"]]))

for name, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
print(f"\n{len(checks) - len(fails)}/{len(checks)} passed")
shutil.rmtree(work, ignore_errors=True)
sys.exit(1 if fails else 0)
