"""Compatibility shim — the active pipeline is run_quick_sim.py.

The full evaluation pipeline lives in ``run_quick_sim.py`` at the repo root.
This file delegates to it so that old invocations still work.

Equivalent commands
-------------------
Old (batch JSON, no viewer):
    python mocap_evaluation/run_evaluation.py \\
        --checkpoint reg_best.pt --data samples_dataset.npy

New (same result, add viewer by default; remove --no-gui to see the sim):
    python run_quick_sim.py \\
        --checkpoint reg_best.pt --data samples_dataset.npy \\
        --no-gui --out eval_results.json
"""
from __future__ import annotations

import subprocess
import sys

if __name__ == "__main__":
    print(
        "[run_evaluation] Forwarding to run_quick_sim.py.\n"
        "Use run_quick_sim.py directly — it is the active pipeline.\n"
        "Add --no-gui --out eval_results.json for headless JSON output.\n"
    )
    sys.exit(subprocess.call(
        [sys.executable, "run_quick_sim.py"] + sys.argv[1:],
    ))
