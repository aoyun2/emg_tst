from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_MODELS_DIR = Path(os.environ.get("MOCAPACT_MODELS_DIR", "mocapact_models")).expanduser()
_ARTIFACTS_DIR = Path(os.environ.get("MOCAP_PHYS_EVAL_ARTIFACTS_DIR", str(Path("artifacts") / "phys_eval_v2"))).expanduser()


@dataclass(frozen=True)
class EvalConfig:
    # Outputs.
    artifacts_dir: Path = _ARTIFACTS_DIR

    # MoCapAct multi-clip policy checkpoint (already present in this repo).
    multiclip_ckpt: Path = _MODELS_DIR / "multiclip_policy" / "full_dataset" / "model" / "model.ckpt"
    device: str = "cpu"

    # MoCapAct per-snippet experts (~2589). The pipeline will auto-download+extract if missing.
    experts_root: Path = _MODELS_DIR / "experts"
    experts_downloads_dir: Path = _MODELS_DIR / "_downloads"

    # We evaluate a "prosthetic" right leg by overriding these two DoFs.
    thigh_actuator: str = "walker/rfemurrx"  # right hip pitch
    knee_actuator: str = "walker/rtibiarx"  # right knee flexion

    # TST windowing (training/eval window length).
    window_hz: float = 200.0
    window_n: int = 200  # 1.0s at 200Hz

    # Motion matching.
    match_top_k: int = 12
    # Use quaternion-derived thigh step angles + knee derivatives for offset-invariant coarse search.
    match_feature_mode: str = "quat_knee_d"

    # Compare rollout recording.
    render_width: int = 480
    render_height: int = 360
    render_camera_id: int = 2

    # Note: when using per-snippet experts, warmup is determined by the chosen
    # window's offset within the matched snippet (we always start at snippet start).

    # Query source (BVH downloaded from the web).
    # This is a real mocap BVH (non-synthetic) and keeps the pipeline BVH-only.
    # To avoid always replaying the same motion, the pipeline cycles through this list
    # (based on how many prior runs exist under artifacts/phys_eval_v2/runs/).
    query_bvh_urls: tuple[str, ...] = (
        # CMU BVH dataset mirrored on GitHub (reliable raw URLs).
        # Source repo: https://github.com/una-dinosauria/cmu-mocap
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_01.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_02.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_03.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_04.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_05.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_06.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_07.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_08.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_09.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_10.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_11.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_12.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_13.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/001/01_14.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_01.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_02.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_03.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_04.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_05.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_06.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_07.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_08.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_09.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/002/02_10.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/003/03_01.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/003/03_02.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/003/03_03.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/003/03_04.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_01.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_02.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_03.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_04.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_05.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_06.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_07.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_08.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_09.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_10.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_11.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_12.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_13.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_14.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_15.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_16.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_17.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_18.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_19.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/005/05_20.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_01.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_02.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_03.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_04.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_05.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_06.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_07.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_08.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_09.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_10.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_11.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_12.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_13.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_14.bvh",
        "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/006/06_15.bvh",
    )

    # When picking the single TST window from the query BVH, we prefer windows with
    # high motion energy (sum(|d thigh| + |d knee|)) after skipping the first part.
    query_window_skip_s: float = 0.75
    query_window_top_k: int = 40

    # "Bad model" demo: smooth perturbation with ~20 deg RMSE (in knee flexion).
    bad_knee_rmse_deg: float = 20.0
    bad_knee_lowpass_hz: float = 3.0
