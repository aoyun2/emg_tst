from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from emg_tst.data import StandardScaler
from emg_tst.model import TSTEncoder, TSTRegressor
from mocap_evaluation.mocap_loader import TARGET_FPS, load_aggregated_database
from mocap_evaluation.motion_matching import find_top_k_matches
from mocap_evaluation.prosthetic_sim import simulate_prosthetic_walking
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


def _scenario_metrics(segment: dict, pred: np.ndarray, mocap_db: dict, cfg: RobustnessConfig):
    matches = find_top_k_matches(segment["knee"], segment["thigh"], mocap_db=mocap_db, k=cfg.top_k_matches)
    delay_frames = int((cfg.delay_ms / 1000.0) * TARGET_FPS)
    rng = np.random.default_rng(0)
    per_match = []
    for start, dist, match in matches:
        gt = simulate_prosthetic_walking(match, segment["knee"], sample_thigh_right=segment["thigh"])
        nominal = simulate_prosthetic_walking(match, pred, sample_thigh_right=segment["thigh"], show_reference=True, reference_knee=segment["knee"])
        delayed = simulate_prosthetic_walking(match, _apply_delay(pred, delay_frames), sample_thigh_right=segment["thigh"])
        noisy = simulate_prosthetic_walking(match, pred + rng.normal(0.0, cfg.noise_std_deg, size=len(pred)).astype(np.float32), sample_thigh_right=segment["thigh"])
        robustness = float(np.mean([
            nominal.get("stability_score", 0.0),
            delayed.get("stability_score", 0.0),
            noisy.get("stability_score", 0.0),
        ]))
        per_match.append({
            "match_start": int(start),
            "dtw_distance": float(dist),
            "category": match.get("category", "unknown"),
            "ground_truth": gt,
            "nominal": nominal,
            "delayed": delayed,
            "noisy": noisy,
            "robustness_score": robustness,
        })
    return per_match


def evaluate_with_checkpoint(checkpoint_path: str, samples_path: str, cfg: EvalConfig) -> dict:
    device = torch.device(cfg.device)
    model, scaler = _load_checkpoint(checkpoint_path, device)
    x, y, file_id, start = _load_windows(samples_path, cfg.n_samples)
    target_frames = int(cfg.robustness.eval_seconds * TARGET_FPS)
    segments = _segment_from_windows(x, y, file_id, start, target_frames=target_frames)
    mocap_db = load_aggregated_database(mocap_root=cfg.mocap_dir, try_download=True, datasets=["cmu"], use_cache=cfg.use_cache)

    out = {"mode": "paper_style_checkpoint_eval", "config": asdict(cfg), "segments": []}
    for seg in segments:
        pred = _predict(model, scaler, seg["x"], device)
        match_metrics = _scenario_metrics(seg, pred, mocap_db, cfg.robustness)
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
    mocap_db = load_aggregated_database(mocap_root=cfg.mocap_dir, try_download=True, datasets=["cmu"], use_cache=cfg.use_cache)
    knee = curves.knee_label_included_deg.astype(np.float32)
    thigh = curves.thigh_angle_deg.astype(np.float32)
    # controlled degradation baseline for stress test
    pred = knee + 2.5 * np.sin(np.linspace(0, 4 * np.pi, len(knee))).astype(np.float32)
    seg = {"knee": knee, "thigh": thigh}
    match_metrics = _scenario_metrics(seg, pred.astype(np.float32), mocap_db, cfg.robustness)
    out = {
        "mode": "paper_style_test_sample",
        "config": asdict(cfg),
        "sample_source": cfg.sample_source,
        "match_metrics": match_metrics,
        "avg_robustness": float(np.mean([m["robustness_score"] for m in match_metrics])) if match_metrics else 0.0,
    }
    Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out
