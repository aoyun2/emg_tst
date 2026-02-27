"""Download and prepare external gait curves for out-of-database matching tests."""
from __future__ import annotations

from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
import urllib.request

import numpy as np

from mocap_evaluation.mocap_loader import TARGET_FPS
from mocap_evaluation.sample_data import SampleCurves

# Public OpenSim IK sample with hip/knee angles.
# The file moved from OpenSim/Examples/Gait2354_Simbody/ to Applications/*/test/
DEFAULT_EXTERNAL_GAIT_URL = (
    "https://raw.githubusercontent.com/opensim-org/opensim-core/main/"
    "Applications/CMC/test/subject01_walk1_ik.mot"
)

# Public fallback URLs from external online repositories/CDNs.
FALLBACK_EXTERNAL_GAIT_URLS = (
    "https://raw.githubusercontent.com/opensim-org/opensim-core/main/"
    "Applications/Forward/test/subject01_walk1_ik.mot",
    "https://raw.githubusercontent.com/opensim-org/opensim-core/main/"
    "Applications/RRA/test/subject01_walk1_ik.mot",
    "https://github.com/opensim-org/opensim-core/raw/refs/heads/main/"
    "Applications/CMC/test/subject01_walk1_ik.mot",
    "https://raw.githubusercontent.com/opensim-org/opensim-core/master/"
    "Applications/CMC/test/subject01_walk1_ik.mot",
)


def _read_text(url: str, timeout_s: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "emg-tst-eval/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _read_external_text(source_url: Optional[str], timeout_s: float = 20.0) -> tuple[str, str]:
    """Read external gait text from online sources with URL fallbacks."""
    urls = (source_url,) if source_url else (DEFAULT_EXTERNAL_GAIT_URL, *FALLBACK_EXTERNAL_GAIT_URLS)
    errors: list[str] = []

    for url in urls:
        try:
            return _read_text(url, timeout_s=timeout_s), url
        except HTTPError as exc:
            errors.append(f"{url} -> HTTP {exc.code}")
        except URLError as exc:
            errors.append(f"{url} -> URL error: {exc.reason}")

    raise RuntimeError(
        "Failed to download external gait sample from online sources. "
        f"Tried: {'; '.join(errors)}. "
        "Pass --external-sample-url with a reachable .mot/.sto file to override."
    )


def _load_opensim_table(raw_text: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        low = ln.lower()
        if low.startswith("time") and ("\t" in ln or " " in ln):
            header_idx = i
            break
        if low == "endheader" and i + 1 < len(lines):
            header_idx = i + 1
            break
    if header_idx is None:
        raise ValueError("Could not find OpenSim table header in external gait file")

    cols = [c for c in lines[header_idx].replace("\t", " ").split(" ") if c]
    arr = np.genfromtxt(lines[header_idx + 1 :], delimiter=None, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(cols):
        raise ValueError(f"Column mismatch in external gait file: {arr.shape[1]} vs {len(cols)}")

    data = {name: arr[:, i] for i, name in enumerate(cols)}
    if "time" not in data:
        raise ValueError("External gait file is missing 'time' column")
    return data["time"], data


def _pick_col(data: dict[str, np.ndarray], names: Iterable[str]) -> np.ndarray:
    for name in names:
        if name in data:
            return data[name]
    raise ValueError(f"Missing expected column. Looked for: {', '.join(names)}")


def _resample_to_target_fps(time_s: np.ndarray, signal: np.ndarray, target_fps: int) -> np.ndarray:
    t0, t1 = float(time_s[0]), float(time_s[-1])
    if t1 <= t0:
        raise ValueError("External gait timeline is invalid")
    n = max(2, int(round((t1 - t0) * float(target_fps))))
    t_new = np.linspace(t0, t1, n, endpoint=True)
    return np.interp(t_new, time_s, signal).astype(np.float32)


def extract_external_sample_curves(
    seconds: float = 4.0,
    seed: int = 13,
    pred_noise_std: float = 2.0,
    source_url: Optional[str] = None,
) -> SampleCurves:
    """Build sample curves from an external online gait file (not mocap DB).

    Uses the natural length of the external data without tiling.  If the file
    is shorter than *seconds*, all available frames are returned and motion
    matching will use a correspondingly shorter query window.
    """
    raw, url = _read_external_text(source_url)
    time_s, table = _load_opensim_table(raw)

    # OpenSim IK conventions: hip_flexion_r and knee_angle_r are flexion-like.
    hip_flex = _pick_col(table, ("hip_flexion_r", "hip_flexion_right", "hip_flex_r"))
    knee_flex = _pick_col(table, ("knee_angle_r", "knee_flexion_r", "knee_angle_right"))

    hip_inc = np.clip(180.0 - hip_flex, 0.0, 180.0)
    knee_inc = np.clip(180.0 - knee_flex, 0.0, 180.0)

    hip_rs = _resample_to_target_fps(time_s, hip_inc, TARGET_FPS)
    knee_rs = _resample_to_target_fps(time_s, knee_inc, TARGET_FPS)

    target_len = max(1, int(round(seconds * TARGET_FPS)))
    available = len(knee_rs)

    rng = np.random.default_rng(seed)

    if available >= target_len:
        # Enough data: pick a random sub-window
        start = int(rng.integers(0, available - target_len + 1))
        knee = knee_rs[start : start + target_len]
        thigh = hip_rs[start : start + target_len]
    else:
        # Use all available data (no tiling); motion matching will match
        # a mocap segment of this shorter length.
        knee = knee_rs
        thigh = hip_rs
        print(f"[external] Data shorter than {seconds:.1f}s "
              f"({available} frames = {available / TARGET_FPS:.2f}s); "
              f"using full available length for matching.")

    n = len(knee)
    pred = np.clip(
        knee + rng.normal(0.0, pred_noise_std, n).astype(np.float32),
        0.0, 180.0,
    )

    return SampleCurves(
        knee_label_included_deg=knee.astype(np.float32),
        thigh_angle_deg=thigh.astype(np.float32),
        predicted_knee_included_deg=pred.astype(np.float32),
        fps=TARGET_FPS,
        category="external_walk",
        source_file=url,
    )
