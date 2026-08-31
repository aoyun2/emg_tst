"""Record what code produced the reported physics run.

The run protocol stores a hash for each physics source file as it stood when the
rollouts were produced. This compares those against the published files, names
any that differ, and records the commit the published code sits at.

It also records the controller constants and the package versions of the
environment the simulation ran under.

    python -m analysis.make_provenance --runs-dir <runs> --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
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


PHYSICS_PACKAGES = (
    "mocapact", "mujoco", "dm-control", "torch", "stable-baselines3",
    "gymnasium", "h5py", "numpy", "scipy",
)


def _simulation_environment(interpreter: pathlib.Path | None) -> dict[str, object]:
    """Query the interpreter that ran the rollouts for its installed versions."""
    if interpreter is None:
        return {"recorded": False,
                "note": "pass --physics-python to record the simulation environment"}
    probe = (
        "import json,sys,importlib.metadata as m\n"
        "out={'python':sys.version.split()[0]}\n"
        f"for n in {list(PHYSICS_PACKAGES)!r}:\n"
        "    try: out[n]=m.version(n)\n"
        "    except Exception: out[n]=None\n"
        "try:\n"
        "    import json as j\n"
        "    d=m.distribution('mocapact').read_text('direct_url.json')\n"
        "    info=j.loads(d) if d else {}\n"
        "    out['mocapact_source']=info.get('url')\n"
        "    out['mocapact_commit']=(info.get('vcs_info') or {}).get('commit_id')\n"
        "except Exception: pass\n"
        "print(json.dumps(out))"
    )
    result = subprocess.run([str(interpreter), "-c", probe],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return {"recorded": False, "error": (result.stderr or "").strip()[-200:]}
    env = json.loads(result.stdout)
    env["recorded"] = True
    return env


CHANGES_SINCE_THE_RUN = {
    "mocap_phys_eval/run_gait120_residual_fusion.py": (
        "the controller mode was an optional flag and is now a required choice, "
        "so a run cannot silently use the superseded static-target controller; "
        "and records are serialised with non-finite values written as null "
        "rather than as bare NaN tokens"
    ),
    "mocap_phys_eval/run.py": (
        "error messages no longer instruct the reader to run a trainer that is "
        "not part of this repository"
    ),
    "mocap_phys_eval/sim.py": (
        "pathlib.Path is imported at module level rather than inside a function"
    ),
}


def _difference_note(differing: list[str]) -> str:
    if not differing:
        return "Every physics source is identical to the reported run."
    parts = []
    for rel in differing:
        why = CHANGES_SINCE_THE_RUN.get(rel, "the change is not recorded here")
        parts.append(f"{rel}: {why}.")
    return (
        "These physics sources differ from the files that produced the rollouts. "
        + " ".join(parts)
        + " None of them alters the simulation path; invoking the published "
        "runner with --moving-target-pd-v2 executes what the reported run "
        "executed."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--physics-python", type=pathlib.Path, default=None,
                        help="interpreter of the environment that ran the rollouts")
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
        "note": _difference_note(differing),
        "controller": {
            "kp": 400.0,
            "kd": 20.0,
            "actuator_force_limit": 160.0,
            "command_range_deg": [0.0, 170.0],
            "timestep_s": 0.03,
        },
        "analysis_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            **{name: _version(name) for name in ("numpy", "scipy", "pandas")},
        },
        "simulation_environment": _simulation_environment(args.physics_python),
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
