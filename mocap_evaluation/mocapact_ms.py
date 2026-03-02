"""Microsoft MoCapAct per-clip HDF5 loader (2 589 snippets).

The Microsoft MoCapAct dataset stores expert RL rollouts in per-clip HDF5
files (e.g. ``CMU_009_12.hdf5``).  Each clip file contains one or more
snippet sub-groups whose IDs follow the pattern ``CMU_009_12-165-363``
(clip_id-start_frame-end_frame).  Every snippet group contains numbered
rollout sub-groups (``0/``, ``1/``, …) with proprioceptive observations.

We extract the right-knee (rtibiarx) and right-hip (rfemurry) joint angles
from rollout-0's proprioceptive observations and use the ``observable_indices``
stored in each HDF5 to find the correct columns.

Joint ordering within ``joints_pos``
--------------------------------------
dm_control stores joint observations in **definition order** (the order they
appear in ``_CMU_MOCAP_JOINTS``):

  idx  joint
  ---  -----
   0   lfemurrz
   1   lfemurry
   2   lfemurrx
   3   ltibiarx
   4   lfootrz
   5   lfootrx
   6   ltoesrx
   7   rfemurrz
   8   rfemurry   ← right hip (rfemurry)
   9   rfemurrx
  10   rtibiarx   ← right knee (rtibiarx)
  11   rfootrz
  …

Set env variable ``MOCAPACT_MS_DIR`` to the directory containing the
per-clip ``CMU_*.hdf5`` files before running.

Dataset variants
----------------
+----------+------------------+------+-------------------+-------------------------------+
| Variant  | Rollouts/snippet | Size | Tarball names     | Path after extraction         |
+==========+==================+======+===================+===============================+
| sample   |        20        | ~1GB | sample/small.tar.gz | small/                      |
+----------+------------------+------+-------------------+-------------------------------+
| small    |        20        |~46GB | all/small/         | all/small/ (tarballs 1-3,   |
|          |                  |      | small_1..3.tar.gz  | extract to the same dir)    |
+----------+------------------+------+-------------------+-------------------------------+
| large    |       200        |~450GB| all/large/         | all/large/ (tarballs 1-43)  |
|          |                  |      | large_1..43.tar.gz |                             |
+----------+------------------+------+-------------------+-------------------------------+

Quick download
--------------
Run ``python -m mocap_evaluation.mocapact_ms --download [--variant sample|small|large]``
to download and extract automatically via huggingface_hub, OR follow the manual
steps below.

Sample only (smoke-test, ~1 GB)::

    pip install huggingface_hub
    huggingface-cli download microsoft/mocapact-data \\
        --repo-type dataset --local-dir ~/.mocapact \\
        --include "sample/small.tar.gz"
    cd ~/.mocapact && tar xf sample/small.tar.gz
    export MOCAPACT_MS_DIR=~/.mocapact/small

Full small dataset (all 2 589 snippets, ~46 GB, 3 tarballs)::

    huggingface-cli download microsoft/mocapact-data \\
        --repo-type dataset --local-dir ~/.mocapact \\
        --include "all/small/*.tar.gz"
    cd ~/.mocapact/all/small
    for f in *.tar.gz; do tar xf "$f"; done
    export MOCAPACT_MS_DIR=~/.mocapact/all/small

Full large dataset (200 rollouts/snippet, ~450 GB, 43 tarballs)::

    huggingface-cli download microsoft/mocapact-data \\
        --repo-type dataset --local-dir ~/.mocapact \\
        --include "all/large/*.tar.gz"
    cd ~/.mocapact/all/large
    for f in *.tar.gz; do tar xf "$f"; done
    export MOCAPACT_MS_DIR=~/.mocapact/all/large
"""
from __future__ import annotations

import os
import warnings
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_NATIVE_FPS = 30.0   # MoCapAct rollout frequency (matches dm_control 30 Hz)
_N_JOINTS   = 56     # CMU humanoid: 56 hinge DOFs
_ROOT_DOFS  =  7     # free-joint: pos(3) + quat(4)

# Index of rtibiarx/rfemurry within the ``joints_pos`` sub-array.
# This is the DEFINITION ORDER used by dm_control's CMU humanoid.
# Alphabetical order gives different values (rtibiarx=45, rfemurry=31);
# use MOCAPACT_JOINT_ORDER=alpha to override if your HDF5 uses that ordering.
_KNEE_JOINTS_IDX_DEF  = 10   # rtibiarx (definition order)
_HIP_JOINTS_IDX_DEF   =  8   # rfemurry (definition order)
_KNEE_JOINTS_IDX_ALPHA = 45  # rtibiarx (alphabetical order)
_HIP_JOINTS_IDX_ALPHA  = 31  # rfemurry (alphabetical order)

# Default search directories for the per-clip HDF5 files
_DEFAULT_DIRS: List[Path] = [
    Path("~/.mocapact/all/small").expanduser(),
    Path("~/.mocapact/small").expanduser(),
    Path("~/.mocapact/all/large").expanduser(),
    Path("~/.mocapact/large").expanduser(),
    Path("~/.mocapact/small_dataset").expanduser(),
    Path("~/.mocapact/large_dataset").expanduser(),
    Path("~/.mocapact").expanduser(),
    Path("./mocapact_data"),
    Path("./data/mocapact"),
]

# Paths to try (inside a snippet group) when looking for proprioceptive obs.
# Rollout-0 is tried first; RSI and start-metrics paths follow as fallbacks.
_OBS_TEMPLATES: List[str] = [
    "{snip}/0/observations/proprioceptive",
    "{snip}/0/proprioceptive",
    "{snip}/start_metrics/observations/proprioceptive",
    "{snip}/rsi_metrics/observations/proprioceptive",
]


# ── Joint-order helpers ───────────────────────────────────────────────────────

def _joint_indices() -> Tuple[int, int]:
    """Return (knee_within_joints_pos, hip_within_joints_pos) to use.

    Checks env variable ``MOCAPACT_JOINT_ORDER``:
    * ``"alpha"``      → alphabetical (45, 31)
    * ``"def"`` or ``""`` → definition order (10, 8)  [default]
    """
    order = os.environ.get("MOCAPACT_JOINT_ORDER", "def").strip().lower()
    if order == "alpha":
        return _KNEE_JOINTS_IDX_ALPHA, _HIP_JOINTS_IDX_ALPHA
    return _KNEE_JOINTS_IDX_DEF, _HIP_JOINTS_IDX_DEF


# ── Directory discovery ───────────────────────────────────────────────────────

def get_mocapact_dir() -> Optional[Path]:
    """Return the directory containing per-clip MoCapAct HDF5 files, or None."""
    env_dir = os.environ.get("MOCAPACT_MS_DIR", "").strip()
    if env_dir:
        p = Path(env_dir).expanduser()
        if p.is_dir():
            return p
        warnings.warn(f"[mocapact_ms] MOCAPACT_MS_DIR={env_dir!r} not a directory.")

    for d in _DEFAULT_DIRS:
        if d.is_dir() and any(d.glob("CMU_*.hdf5")):
            return d
    return None


def download_instructions() -> str:
    """Return a human-readable download guide string."""
    return (
        "Microsoft MoCapAct HDF5 files not found.\n\n"
        "Variants:\n"
        "  sample  ~1 GB   (a few clips, smoke-test only)\n"
        "  small  ~46 GB   (all 2 589 snippets, 20 rollouts each)  ← recommended\n"
        "  large ~450 GB   (all 2 589 snippets, 200 rollouts each)\n\n"
        "Option A — Python helper (downloads + extracts automatically):\n"
        "  pip install huggingface_hub\n"
        "  python -m mocap_evaluation.mocapact_ms --download            # sample\n"
        "  python -m mocap_evaluation.mocapact_ms --download --variant small\n"
        "  python -m mocap_evaluation.mocapact_ms --download --variant large\n\n"
        "Option B — manual CLI:\n"
        "  # sample only\n"
        "  huggingface-cli download microsoft/mocapact-data \\\n"
        "      --repo-type dataset --local-dir ~/.mocapact \\\n"
        "      --include 'sample/small.tar.gz'\n"
        "  cd ~/.mocapact && tar xf sample/small.tar.gz\n"
        "  export MOCAPACT_MS_DIR=~/.mocapact/small\n\n"
        "  # full small (3 tarballs, all 2 589 snippets)\n"
        "  huggingface-cli download microsoft/mocapact-data \\\n"
        "      --repo-type dataset --local-dir ~/.mocapact \\\n"
        "      --include 'all/small/*.tar.gz'\n"
        "  cd ~/.mocapact/all/small\n"
        "  for f in *.tar.gz; do tar xf \"$f\"; done\n"
        "  export MOCAPACT_MS_DIR=~/.mocapact/all/small\n"
    )


def download_dataset(
    target_dir: str = "~/.mocapact",
    variant: str = "sample",
) -> Path:
    """Download and extract the MoCapAct dataset via huggingface_hub.

    Parameters
    ----------
    target_dir :
        Local root directory for the download (default: ``~/.mocapact``).
    variant : ``"sample"`` | ``"small"`` | ``"large"``
        Which dataset to fetch:

        * ``"sample"`` — a handful of clips (~1 GB), good for smoke-testing.
        * ``"small"``  — all 2 589 snippets, 20 rollouts each (~46 GB).
        * ``"large"``  — all 2 589 snippets, 200 rollouts each (~450 GB).

    Returns
    -------
    Path
        Directory that contains the extracted ``CMU_*.hdf5`` files.
        Set ``MOCAPACT_MS_DIR`` to this path before running the pipeline.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed.\n"
            "Run:  pip install huggingface_hub"
        )

    import subprocess

    root = Path(target_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    _PATTERNS = {
        "sample": (["sample/small.tar.gz"], root),
        "small":  (["all/small/*.tar.gz"],  root / "all" / "small"),
        "large":  (["all/large/*.tar.gz"],  root / "all" / "large"),
    }
    if variant not in _PATTERNS:
        raise ValueError(f"variant must be 'sample', 'small', or 'large'; got {variant!r}")

    patterns, extract_dir = _PATTERNS[variant]

    print(f"[mocapact_ms] Downloading variant={variant!r} to {root} …")
    snapshot_download(
        repo_id="microsoft/mocapact-data",
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=patterns,
    )

    # Extract all tarballs that were downloaded into the target directory
    tarballs = list(extract_dir.glob("*.tar.gz"))
    if not tarballs and variant == "sample":
        # sample tarball lands at root level
        tarballs = list(root.glob("sample/small.tar.gz"))

    if not tarballs:
        raise RuntimeError(
            f"No tarballs found after download in {extract_dir}. "
            "Check the Hugging Face repo structure."
        )

    print(f"[mocapact_ms] Extracting {len(tarballs)} tarball(s) into {extract_dir} …")
    extract_dir.mkdir(parents=True, exist_ok=True)
    for tb in tarballs:
        print(f"  tar xf {tb.name}")
        subprocess.run(["tar", "xf", str(tb), "-C", str(extract_dir)], check=True)

    # After extraction the HDF5 files live directly in extract_dir
    # (or in a subdirectory named after the variant for the sample)
    h5_dir = extract_dir
    if variant == "sample":
        candidate = root / "small"
        if candidate.is_dir() and any(candidate.glob("CMU_*.hdf5")):
            h5_dir = candidate

    n_h5 = len(list(h5_dir.glob("CMU_*.hdf5")))
    print(f"[mocapact_ms] Done.  {n_h5} HDF5 clip files in {h5_dir}")
    print(f"\n  Set:  export MOCAPACT_MS_DIR={h5_dir}\n")
    return h5_dir


# ── Snippet enumeration ───────────────────────────────────────────────────────

def enumerate_snippets(mocapact_dir: Path) -> List[Tuple[str, Path]]:
    """Return ``(snippet_id, hdf5_path)`` pairs for every snippet in *mocapact_dir*.

    snippet_id format: ``CMU_009_12-165-363``  (clip-start-end)
    hdf5_path        : the per-clip file that contains this snippet.
    """
    try:
        import h5py
    except ImportError:
        warnings.warn("[mocapact_ms] h5py is not installed; cannot enumerate snippets.")
        return []

    results: List[Tuple[str, Path]] = []
    for h5_path in sorted(mocapact_dir.glob("CMU_*.hdf5")):
        try:
            with h5py.File(str(h5_path), "r") as f:
                for key in f.keys():
                    # Snippet keys: CMU_<clip>-<start>-<end>  (two trailing "-NNN")
                    parts = key.rsplit("-", 2)
                    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                        results.append((key, h5_path))
        except Exception as exc:
            warnings.warn(f"[mocapact_ms] Cannot open {h5_path.name}: {exc}")

    return results


# ── Observable-index resolution ───────────────────────────────────────────────

def _joints_pos_flat_indices(h5file) -> Optional[np.ndarray]:
    """Return the 56 flat proprioceptive indices for joints_pos, or None.

    The ``observable_indices/walker/joints_pos`` dataset stores either:
    * a 56-element int array (one flat index per joint), or
    * a 2-element [start, stop] slice.
    """
    for key in ("observable_indices/walker/joints_pos",
                "observable_indices/joints_pos"):
        if key not in h5file:
            continue
        try:
            arr = np.asarray(h5file[key], dtype=np.int64).ravel()
            if len(arr) == _N_JOINTS:
                return arr
            if len(arr) == 2:
                return np.arange(int(arr[0]), int(arr[1]), dtype=np.int64)
        except Exception:
            pass
    return None


def _resolve_knee_hip_flat_idx(h5file) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(knee_flat_idx, hip_flat_idx)`` in the flat proprio vector."""
    jp = _joints_pos_flat_indices(h5file)
    if jp is None:
        return None, None
    k_ji, h_ji = _joint_indices()
    if max(k_ji, h_ji) >= len(jp):
        return None, None
    return int(jp[k_ji]), int(jp[h_ji])


# ── Observation loading ───────────────────────────────────────────────────────

def _load_obs(h5file, snippet_id: str) -> Optional[np.ndarray]:
    """Load proprioceptive observations ``(T, N_obs)`` for *snippet_id*.

    Tries multiple path patterns.  Collapses a leading rollout dimension
    (S, T, N) → (T, N) by taking index 0 when present.
    """
    for tmpl in _OBS_TEMPLATES:
        path = tmpl.format(snip=snippet_id)
        if path not in h5file:
            continue
        try:
            obs = np.asarray(h5file[path], dtype=np.float32)
            if obs.ndim == 3:          # (S, T, N) — multiple rollouts stacked
                obs = obs[0]
            if obs.ndim == 2 and obs.shape[0] >= 2:
                return obs
        except Exception as exc:
            warnings.warn(f"[mocapact_ms] obs at {path}: {exc}")
    return None


# ── Angle-conversion helpers (same convention as db.py) ──────────────────────

def _knee_rad_to_included(rad: np.ndarray) -> np.ndarray:
    deg = np.degrees(np.abs(np.asarray(rad, dtype=np.float64)))
    return np.clip(180.0 - deg, 0.0, 180.0).astype(np.float32)


def _hip_rad_to_included(rad: np.ndarray) -> np.ndarray:
    deg = np.degrees(np.asarray(rad, dtype=np.float64))
    return (180.0 - deg).astype(np.float32)


def _resample(arr: np.ndarray, from_fps: float, to_fps: float) -> np.ndarray:
    if abs(from_fps - to_fps) < 0.5:
        return arr.astype(np.float32)
    from scipy.signal import resample_poly
    up   = int(round(to_fps))
    down = int(round(from_fps))
    g    = gcd(up, down)
    return resample_poly(arr.astype(np.float64), up // g, down // g).astype(np.float32)


# ── Per-snippet extraction ────────────────────────────────────────────────────

def extract_snippet_angles(
    snippet_id: str,
    h5_path: Path,
    target_fps: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract ``(knee_included_deg, hip_included_deg)`` at *target_fps*.

    Returns ``None`` if the snippet cannot be loaded or decoded.
    """
    try:
        import h5py
        with h5py.File(str(h5_path), "r") as f:
            if snippet_id not in f:
                return None

            knee_flat, hip_flat = _resolve_knee_hip_flat_idx(f)
            obs = _load_obs(f, snippet_id)
            if obs is None:
                return None

            if knee_flat is not None and hip_flat is not None:
                if max(knee_flat, hip_flat) < obs.shape[1]:
                    knee_rad = obs[:, knee_flat].astype(np.float64)
                    hip_rad  = obs[:, hip_flat].astype(np.float64)
                    knee = _resample(_knee_rad_to_included(knee_rad), _NATIVE_FPS, target_fps)
                    hip  = _resample(_hip_rad_to_included(hip_rad),   _NATIVE_FPS, target_fps)
                    return knee, hip

            # Fallback when observable_indices is absent: assume joints_pos
            # occupies the first 56 columns of the proprioceptive array.
            # (CMU humanoid: joints_pos is the first observable by definition.)
            k_ji, h_ji = _joint_indices()
            if obs.shape[1] > max(k_ji, h_ji):
                knee_rad = obs[:, k_ji].astype(np.float64)
                hip_rad  = obs[:, h_ji].astype(np.float64)
                knee = _resample(_knee_rad_to_included(knee_rad), _NATIVE_FPS, target_fps)
                hip  = _resample(_hip_rad_to_included(hip_rad),   _NATIVE_FPS, target_fps)
                return knee, hip

    except Exception as exc:
        warnings.warn(f"[mocapact_ms] extract {snippet_id}: {exc}")
    return None


# ── Full qpos reconstruction for sim.py kinematic replay ─────────────────────

def load_snippet_qpos(snippet_id: str) -> Optional[Tuple[np.ndarray, float]]:
    """Reconstruct full qpos ``(T, 63)`` for kinematic replay in sim.py.

    Root is fixed at (0, 0, 1.2) m with identity quaternion so the
    humanoid stands in place; the 56 joints animate from observations.

    Returns ``(qpos, fps)`` or ``None`` if the snippet cannot be loaded.
    """
    mocapact_dir = get_mocapact_dir()
    if mocapact_dir is None:
        return None

    # Derive clip file name: "CMU_009_12-165-363" → "CMU_009_12"
    parts = snippet_id.rsplit("-", 2)
    if len(parts) != 3:
        return None
    clip_id = parts[0]

    h5_path = mocapact_dir / f"{clip_id}.hdf5"
    if not h5_path.exists():
        return None

    try:
        import h5py
        with h5py.File(str(h5_path), "r") as f:
            if snippet_id not in f:
                return None

            jp_indices = _joints_pos_flat_indices(f)
            obs = _load_obs(f, snippet_id)
            if obs is None:
                return None

            T = obs.shape[0]

            # Extract all 56 joint angles
            if jp_indices is not None and len(jp_indices) == _N_JOINTS:
                max_idx = int(jp_indices.max())
                if max_idx < obs.shape[1]:
                    joints_rad = obs[:, jp_indices].astype(np.float64)
                else:
                    joints_rad = None
            else:
                joints_rad = obs[:, :_N_JOINTS].astype(np.float64) if obs.shape[1] >= _N_JOINTS else None

            if joints_rad is None or joints_rad.shape != (T, _N_JOINTS):
                return None

            # Fixed root: position (0, 0, 1.2), identity quaternion (w=1)
            pos  = np.zeros((T, 3),  dtype=np.float64)
            pos[:, 2] = 1.2
            quat = np.zeros((T, 4), dtype=np.float64)
            quat[:, 0] = 1.0

            qpos = np.concatenate([pos, quat, joints_rad], axis=1)  # (T, 63)
            return qpos, _NATIVE_FPS

    except Exception as exc:
        warnings.warn(f"[mocapact_ms] qpos for {snippet_id}: {exc}")
    return None


# ── Database builder ──────────────────────────────────────────────────────────

def load_database_ms(
    target_fps: float,
    mocapact_dir: Optional[Path] = None,
) -> dict:
    """Build a motion-matching database from Microsoft MoCapAct HDF5 files.

    Parameters
    ----------
    target_fps :
        Resample all snippets to this rate (200 Hz default).
    mocapact_dir :
        Directory with per-clip ``CMU_*.hdf5`` files.  Auto-discovered if None.

    Returns
    -------
    dict with the same keys as :func:`mocap_evaluation.db.load_database`:
    ``knee_right``, ``hip_right``, ``file_boundaries``, ``fps``, ``source``.
    """
    if mocapact_dir is None:
        mocapact_dir = get_mocapact_dir()
    if mocapact_dir is None:
        raise RuntimeError(download_instructions())

    snippets = enumerate_snippets(mocapact_dir)
    if not snippets:
        raise RuntimeError(
            f"No MoCapAct snippets found in {mocapact_dir}.\n"
            "Make sure the directory contains CMU_*.hdf5 files.\n\n"
            + download_instructions()
        )

    print(f"[mocapact_ms] Building database: {len(snippets)} snippets "
          f"from {mocapact_dir} …")

    from tqdm import tqdm

    all_knee: List[np.ndarray] = []
    all_hip:  List[np.ndarray] = []
    boundaries: List[tuple]    = []
    skipped = 0
    cursor  = 0

    for snippet_id, h5_path in tqdm(snippets, desc="Loading snippets", unit="snip"):
        result = extract_snippet_angles(snippet_id, h5_path, target_fps)
        if result is None or len(result[0]) < 2:
            skipped += 1
            continue
        knee, hip = result
        n = len(knee)
        all_knee.append(knee)
        all_hip.append(hip)
        boundaries.append((cursor, cursor + n, snippet_id, "locomotion"))
        cursor += n

    if not all_knee:
        raise RuntimeError(
            "No snippets could be loaded.  Check that the HDF5 files are "
            "accessible and that observable_indices is present."
        )

    print(f"[mocapact_ms] {len(all_knee)} snippets loaded ({skipped} skipped), "
          f"{cursor} frames @ {target_fps:.0f} Hz.")

    return {
        "knee_right":      np.concatenate(all_knee),
        "hip_right":       np.concatenate(all_hip),
        "file_boundaries": boundaries,
        "fps":             np.float32(target_fps),
        "source":          np.array("mocapact_ms"),
    }


# ── CLI entry-point: python -m mocap_evaluation.mocapact_ms ──────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Microsoft MoCapAct dataset utility.\n\n"
            "Download + extract:\n"
            "  python -m mocap_evaluation.mocapact_ms --download\n"
            "  python -m mocap_evaluation.mocapact_ms --download --variant small\n\n"
            "Inspect a local dataset directory:\n"
            "  python -m mocap_evaluation.mocapact_ms --info\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--download", action="store_true",
                    help="Download + extract the dataset from Hugging Face")
    ap.add_argument("--variant", default="sample",
                    choices=["sample", "small", "large"],
                    help=(
                        "Dataset variant to download: "
                        "'sample' (~1 GB, a few clips), "
                        "'small' (~46 GB, all 2 589 snippets, 20 rollouts), "
                        "'large' (~450 GB, all 2 589 snippets, 200 rollouts). "
                        "Default: sample"
                    ))
    ap.add_argument("--target-dir", default="~/.mocapact",
                    help="Local root directory for the download (default: ~/.mocapact)")
    ap.add_argument("--info", action="store_true",
                    help="Print info about the dataset found via MOCAPACT_MS_DIR / auto-discovery")
    args = ap.parse_args()

    if args.download:
        h5_dir = download_dataset(target_dir=args.target_dir, variant=args.variant)
        print(f"\nAdd to your shell profile:\n  export MOCAPACT_MS_DIR={h5_dir}\n")
    elif args.info:
        d = get_mocapact_dir()
        if d is None:
            print("No MoCapAct dataset found. " + download_instructions())
        else:
            snippets = enumerate_snippets(d)
            n_clips = len({s.rsplit("-", 2)[0] for s, _ in snippets})
            print(f"Dataset directory : {d}")
            print(f"Clip HDF5 files   : {n_clips}")
            print(f"Snippets          : {len(snippets)}")
    else:
        ap.print_help()
