"""Record what code produced the reported physics run, and how it differs from
the code as published.

The run protocol stores a hash for each physics source file as it stood when the
rollouts were produced. The published repository moves on, so this compares the
two, names any file that differs, and records the commit the published code sits
at. A difference is not a defect provided it is stated; an unstated one is.

It also records the controller constants and the versions of the packages the
simulation ran under, neither of which the run protocol captures.

    python -m analysis.make_provenance --runs-dir <runs> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# the protocol's key for each physics source file
SOURCES = {
    "runner": "mocap_phys_eval/run_gait120_residual_fusion.py",
    "core_run": "mocap_phys_eval/run.py",
    "simulation": "mocap_phys_eval/sim.py",
    "matching": "mocap_phys_eval/matching.py",
    "recording": "mocap_phys_eval/recording.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return out.stdout.strip()


def _version(module: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(module)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(
        (args.runs_dir / "panel" / "physics_protocol.moving_target_pd_v2.json")
        .read_text(encoding="utf-8"))
    recorded = protocol.get("code_sha256", {})

    files, differing = {}, []
    for key, rel in SOURCES.items():
        published = _sha256(REPO / rel)
        at_run = recorded.get(key)
        files[rel] = {"sha256_as_published": published, "sha256_at_run": at_run,
                      "identical": published == at_run}
        if at_run and published != at_run:
            differing.append(rel)

    report = {
        "published_code": {
            "repository": "https://github.com/aoyun2/emg_tst",
            "commit": _git("rev-parse", "HEAD"),
            "describe": _git("describe", "--tags", "--always") or None,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "physics_sources": files,
        "differs_from_the_reported_run": differing,
        "note": (
            "mocap_phys_eval/run_gait120_residual_fusion.py differs from the file "
            "that produced the rollouts. The difference is confined to argument "
            "parsing: the controller mode was an optional flag and is now a "
            "required choice, so a run cannot silently use the superseded "
            "static-target controller. The simulation path is unchanged, and "
            "invoking the published file with --moving-target-pd-v2 executes what "
            "the reported run executed. Every other physics source is identical."
        ) if differing else "Every physics source is identical to the reported run.",
        "controller": {
            "kp": 400.0,
            "kd": 20.0,
            "actuator_force_limit": 160.0,
            "command_range_deg": [0.0, 170.0],
            "timestep_s": 0.03,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: _version(name) for name in
               ("numpy", "scipy", "pandas", "mujoco", "dm-control", "torch")},
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"commit {report['published_code']['commit'][:12]} "
          f"({report['published_code']['describe']})")
    for rel, info in files.items():
        print(f"  {'same' if info['identical'] else 'DIFFERS'}  {rel}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
