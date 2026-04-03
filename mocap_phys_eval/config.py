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

    # We evaluate a "prosthetic" right knee by overriding the knee DoF only.
    knee_actuator: str = "walker/rtibiarx"  # right knee flexion

    # TST windowing (training/eval window length).
    window_hz: float = 200.0
    window_n: int = 200  # 1.0s at 200Hz
    # Paper protocol on real rigtest data:
    # sample from the held-out LOFO pool with a fixed random seed and keep
    # replacing failed trials until this many successful evaluations are obtained.
    paper_eval_n_trials: int = 80
    paper_eval_seed: int = 42
    # Demo-only fallback window count when rigtest samples / trained folds are unavailable.
    demo_n_windows: int = 3

    # Rigtest samples dataset (produced by split_to_samples.py).
    rig_samples_path: Path = Path("samples_dataset.npy")

    # Motion matching.
    match_top_k: int = 12
    match_local_refine_radius: int = 15
    # Use thigh quaternion *relative rotation* (log) + knee derivatives for offset-invariant coarse search.
    match_feature_mode: str = "dquat_knee_d"

    # Compare rollout recording.
    render_width: int = 480
    render_height: int = 360
    render_camera_id: int = 2

    # Note: when using per-snippet experts, warmup is determined by the chosen
    # window's offset within the matched snippet (we always start at snippet start).

    # Query source (BVH downloaded from the web) used ONLY for demo when rigtest recordings
    # are not available. These clips must NOT be from CMU to avoid artificially inflating
    # motion-matching performance (bank is built from CMU2020 fitted motion).
    query_bvh_urls: tuple[str, ...] = (
        # PyMO demo BVH (not CMU). Joint names follow a CMU-like convention.
        "https://raw.githubusercontent.com/omimo/PyMO/master/demos/data/AV_8Walk_Meredith_HVHA_Rep1.bvh",
        # Bandai Namco Research Motion Dataset (not CMU). Joint names are Mixamo-like.
        "https://raw.githubusercontent.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/master/dataset/Bandai-Namco-Research-Motiondataset-1/data/dataset-1_walk-right_normal_001.bvh",
        "https://raw.githubusercontent.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset/master/dataset/Bandai-Namco-Research-Motiondataset-1/data/dataset-1_walk-left_normal_001.bvh",
    )

    # When picking the single TST window from the query BVH, we prefer windows with
    # high motion energy (sum(|d thigh| + |d knee|)) after skipping the first part.
    query_window_skip_s: float = 0.75
    query_window_top_k: int = 40

    # "Bad model" demo: smooth perturbation with ~20 deg RMSE (in knee flexion).
    bad_knee_rmse_deg: float = 20.0
    bad_knee_lowpass_hz: float = 3.0
