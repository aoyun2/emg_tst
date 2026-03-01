from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from emg_tst.data import StandardScaler
from emg_tst.model import TSTEncoder, TSTRegressor
from mocap_evaluation.mocap_loader import TARGET_FPS
from mocap_evaluation.mocapact_dataset import load_mocapact_database
from mocap_evaluation.sample_data import extract_real_sample_curves
from mocap_evaluation.external_sample_data import extract_external_sample_curves


@dataclass
class RobustnessConfig:
    top_k_matches: int = 3
    delay_ms: float = 60.0
    noise_std_deg: float = 6.0
    eval_seconds: float = 4.0


@dataclass
class EvalConfig:
    mocap_dir: str = "mocap_data"
    out_path: str = "eval_results.json"
    n_samples: Optional[int] = None
    device: str = "cpu"
    use_cache: bool = True
    sample_source: str = "external"
    external_sample_url: Optional[str] = None
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    mocapact_checkpoint: Optional[str] = None
    mocapact_model_dir: str = "mocapact_models"
    use_gui: bool = False


def _load_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_cfg"]
    encoder = TSTEncoder(
        n_vars=cfg["n_vars"],
        seq_len=cfg["seq_len"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        dropout=cfg.get("dropout", 0.1),
    )
    model = TSTRegressor(encoder, out_dim=1)
    model.load_state_dict(ckpt["reg_state_dict"])
    model.to(device).eval()
    scaler = StandardScaler(np.asarray(ckpt["scaler"]["mean"], dtype=np.float32), np.asarray(ckpt["scaler"]["std"], dtype=np.float32))
    return model, scaler


def _load_windows(samples_path: str | Path, n_samples: Optional[int]):
    data = np.load(samples_path, allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = data.item()
    x = data["X"].astype(np.float32)
    y_seq = data["y_seq"].astype(np.float32)
    file_id = np.asarray(data.get("file_id", np.zeros(len(x), dtype=np.int32)), dtype=np.int32)
    start = np.asarray(data.get("start", np.arange(len(x), dtype=np.int32)), dtype=np.int32)
    if n_samples is not None:
        x, y_seq, file_id, start = x[:n_samples], y_seq[:n_samples], file_id[:n_samples], start[:n_samples]
    return x, y_seq, file_id, start


@torch.no_grad()
def _predict(model: TSTRegressor, scaler: StandardScaler, x_window: np.ndarray, device: torch.device) -> np.ndarray:
    x = (x_window - scaler.mean_) / scaler.std_
    x_t = torch.from_numpy(x).unsqueeze(0).to(device)
    pred = model(x_t)[0, :, 0].detach().cpu().numpy()
    return pred.astype(np.float32)


def _segment_from_windows(x: np.ndarray, y: np.ndarray, file_id: np.ndarray, start: np.ndarray, target_frames: int):
    segments = []
    for fid in np.unique(file_id):
        idx = np.where(file_id == fid)[0]
        idx = idx[np.argsort(start[idx])]
        if len(idx) == 0:
            continue
        k = 0
        while k < len(idx):
            x_cat = []
            y_cat = []
            used = []
            frames = 0
            while k < len(idx) and frames < target_frames:
                wi = idx[k]
                x_cat.append(x[wi])
                y_cat.append(y[wi])
                used.append(int(wi))
                frames += len(y[wi])
                k += 1
            if frames < target_frames:
                break
            xf = np.concatenate(x_cat, axis=0)[:target_frames]
            yf = np.concatenate(y_cat, axis=0)[:target_frames]
            thigh = xf[:, -1]
            segments.append({"window_indices": used, "knee": yf, "thigh": thigh, "x": xf})
    return segments


def _apply_delay(sig: np.ndarray, delay_frames: int) -> np.ndarray:
    if delay_frames <= 0:
        return sig.copy()
    out = np.empty_like(sig)
    out[:delay_frames] = sig[0]
    out[delay_frames:] = sig[:-delay_frames]
    return out



def _scenario_metrics_mocapact(
    segment: dict,
    pred: np.ndarray,
    cfg: RobustnessConfig,
    eval_cfg: "EvalConfig",
    mocap_db: Optional[dict] = None,
) -> list:
    """Robustness evaluation using MoCapAct adaptive policy backend.

    When *mocap_db* is supplied, DTW matching is used to identify the CMU
    clip that best corresponds to the EMG recording.  The MoCapAct
    environment is then initialised with that specific clip so the policy
    walks the same motion — but with full physics, the right knee overridden
    by the model's prediction, and the right hip overridden by the recorded
    thigh angle.

    Without *mocap_db* the policy walks an arbitrary locomotion clip.
    """
    from mocap_evaluation.mocapact_sim import (
        SIM_FPS,
        simulate_scenario,
        resolve_clip_from_match,
        load_multi_clip_policy,
        create_walking_env,
    )

    delay_frames = int((cfg.delay_ms / 1000.0) * TARGET_FPS)
    rng = np.random.default_rng(0)

    # ── DTW matching (done at recording rate for best temporal resolution) ─
    best_start = None
    best_dist = float("inf")
    if mocap_db is not None and "thigh" in segment:
        from mocap_evaluation.motion_matching import find_best_match
        best_start, best_dist, _ = find_best_match(
            segment["knee"], segment["thigh"], mocap_db,
        )

    match_info = None
    if mocap_db is not None and best_start is not None:
        match_info = resolve_clip_from_match(best_start, len(pred), mocap_db)

    # ── Build the three knee signals ──────────────────────────────────
    delayed_pred = _apply_delay(pred, delay_frames)
    bad_pred = (delayed_pred
                + rng.normal(0.0, cfg.noise_std_deg, size=len(delayed_pred)).astype(np.float32))

    # ── Resample signals from recording rate (TARGET_FPS) to sim rate (SIM_FPS)
    # The simulation steps at SIM_FPS; each step consumes one frame of the
    # input signals.  Without resampling, the sim would only consume the first
    # eval_seconds*SIM_FPS / TARGET_FPS fraction of the recording (≈15% for
    # 30/200 Hz), effectively slow-motioning the gait.
    def _rs(arr: np.ndarray) -> np.ndarray:
        n_src = len(arr)
        n_dst = max(2, int(round(n_src * SIM_FPS / TARGET_FPS)))
        return np.interp(
            np.linspace(0.0, 1.0, n_dst),
            np.linspace(0.0, 1.0, n_src),
            arr,
        ).astype(np.float32)

    gt_knee_sim = _rs(segment["knee"])
    nominal_sim = _rs(pred)
    bad_sim     = _rs(bad_pred)
    thigh_sim   = _rs(segment["thigh"]) if "thigh" in segment else None

    policy = load_multi_clip_policy(
        checkpoint_path=eval_cfg.mocapact_checkpoint,
        model_dir=eval_cfg.mocapact_model_dir,
        device=eval_cfg.device,
    )

    def _run_one(knee_sig: np.ndarray) -> dict:
        """Run one scenario: create env → simulate → close env."""
        clip_id = match_info["clip_id"] if match_info else None
        try:
            env = create_walking_env(policy=policy, clip_id=clip_id)
        except Exception as exc:
            print(f"  Warning: clip {clip_id!r} unavailable ({exc}); using default.")
            env = create_walking_env(policy=policy)
        try:
            return simulate_scenario(
                env=env,
                policy=policy,
                knee_inc_deg=knee_sig,
                thigh_inc_deg=thigh_sim,
                reference_knee_inc_deg=gt_knee_sim,
                fps=SIM_FPS,
                use_gui=eval_cfg.use_gui,
                match_info=match_info,
            )
        finally:
            env.close()

    gt      = _run_one(gt_knee_sim)
    nominal = _run_one(nominal_sim)
    bad     = _run_one(bad_sim)

    robustness = float(np.mean([
        nominal.get("stability_score", 0.0),
        bad.get("stability_score", 0.0),
    ]))

    category = (match_info or {}).get("category", "mocapact_policy")
    return [{
        "match_start": best_start or 0,
        "dtw_distance": float(best_dist if np.isfinite(best_dist) else 0.0),
        "category": category,
        "ground_truth": gt,
        "nominal": nominal,
        "bad": bad,
        "robustness_score": robustness,
    }]


def evaluate_with_checkpoint(checkpoint_path: str, samples_path: str, cfg: EvalConfig) -> dict:
    device = torch.device(cfg.device)
    model, scaler = _load_checkpoint(checkpoint_path, device)
    x, y, file_id, start = _load_windows(samples_path, cfg.n_samples)
    target_frames = int(cfg.robustness.eval_seconds * TARGET_FPS)
    segments = _segment_from_windows(x, y, file_id, start, target_frames=target_frames)

    # Load MocapAct reference database for motion matching.
    # This replaces the CMU BVH database — no separate download required.
    mocap_db = load_mocapact_database(use_cache=cfg.use_cache)

    out = {"mode": "mocapact_checkpoint_eval", "config": asdict(cfg), "segments": []}
    for seg in segments:
        pred = _predict(model, scaler, seg["x"], device)
        match_metrics = _scenario_metrics_mocapact(seg, pred, cfg.robustness, cfg, mocap_db=mocap_db)
        out["segments"].append({
            "window_indices": seg["window_indices"],
            "match_metrics": match_metrics,
            "avg_robustness": float(np.mean([m["robustness_score"] for m in match_metrics])) if match_metrics else 0.0,
        })

    Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def evaluate_test_sample(cfg: EvalConfig) -> dict:
    sec = cfg.robustness.eval_seconds
    if cfg.sample_source == "external":
        curves = extract_external_sample_curves(seconds=sec, source_url=cfg.external_sample_url)
    else:
        curves = extract_real_sample_curves(mocap_dir=cfg.mocap_dir, seconds=sec)

    knee = curves.knee_label_included_deg.astype(np.float32)
    thigh = curves.thigh_angle_deg.astype(np.float32)
    # controlled degradation baseline for stress test
    pred = knee + 2.5 * np.sin(np.linspace(0, 4 * np.pi, len(knee))).astype(np.float32)
    seg = {"knee": knee, "thigh": thigh}

    mocap_db = load_mocapact_database(use_cache=cfg.use_cache)

    match_metrics = _scenario_metrics_mocapact(
        seg, pred.astype(np.float32), cfg.robustness, cfg, mocap_db=mocap_db,
    )
    mode = "mocapact_test_sample"

    out = {
        "mode": mode,
        "config": asdict(cfg),
        "sample_source": cfg.sample_source,
        "match_metrics": match_metrics,
        "avg_robustness": float(np.mean([m["robustness_score"] for m in match_metrics])) if match_metrics else 0.0,
    }
    Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out
