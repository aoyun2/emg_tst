from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bvh import extract_right_leg_thigh_quat_and_knee_included_deg, load_bvh
from .config import EvalConfig
from .experts import discover_expert_snippets, ensure_full_expert_zoo
from .matching import MatchCandidate, motion_match_one_window
from .plots import plot_balance_traces, plot_motion_match, plot_simulation_knee, plot_thigh_quat_match
from .recording import record_compare_rollout
from .reference_bank import ExpertSnippetBank, build_expert_snippet_bank
from .sim import OverrideSpec, load_expert_policy
from .utils import (
    dataclass_to_json_dict,
    download_to,
    ensure_dir,
    now_run_id,
    quat_conj_wxyz,
    quat_geodesic_deg_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    resample_linear,
    resample_quat_slerp_wxyz,
    set_global_determinism,
    write_json,
)


@dataclass(frozen=True)
class QueryWindow:
    query_id: str
    source: str
    sample_hz: float
    thigh_quat_wxyz: np.ndarray  # (W,4) at sample_hz
    knee_included_deg: np.ndarray  # (W,) at sample_hz, included angle (0 bent, 180 straight)
    thigh_pitch_deg: np.ndarray | None = None  # optional demo-only visualization
    X_raw: np.ndarray | None = None  # (W,F) raw features (rigtest only), for TST inference


@dataclass(frozen=True)
class LoadedTstModel:
    model: Any
    mean: np.ndarray  # (F,)
    std: np.ndarray  # (F,)
    feature_cols: np.ndarray  # (n_vars,)
    device: str
    ckpt_path: str


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = int(min(a.size, b.size))
    if n < 1:
        return float("nan")
    return float(np.sqrt(float(np.mean((a[:n] - b[:n]) ** 2))))


def _mean_abs(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size < 1:
        return float("nan")
    return float(np.mean(np.abs(a)))


def _make_bad_knee_prediction(
    knee_good_deg: np.ndarray,
    *,
    sample_hz: float,
    target_rmse_deg: float,
    lowpass_hz: float,
    seed: int = 0,
) -> np.ndarray:
    """Smooth deterministic "bad model" knee with ~target_rmse (deg) vs knee_good.

    We intentionally avoid large discontinuities (e.g., 0<->180 spikes) by generating
    band-limited noise and scaling to the desired RMSE.
    """
    knee_good = np.asarray(knee_good_deg, dtype=np.float64).reshape(-1)
    n = int(knee_good.size)
    if n < 2:
        return knee_good.astype(np.float32)

    hz = float(max(1e-6, float(sample_hz)))
    lp = float(max(0.10, float(lowpass_hz)))
    lp = min(lp, 0.45 * hz)  # keep below Nyquist

    rng = np.random.default_rng(int(seed))
    x = rng.standard_normal(n).astype(np.float64)

    # FFT low-pass (deterministic and simple).
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / hz)
    mask = freqs <= lp
    X[~mask] = 0.0
    e = np.fft.irfft(X, n=n)

    # Normalize to target RMSE.
    e = e - float(np.mean(e))
    rms = float(np.sqrt(float(np.mean(e**2))))
    if not np.isfinite(rms) or rms < 1e-8:
        return knee_good.astype(np.float32)
    e = e * (float(target_rmse_deg) / rms)

    bad = knee_good + e
    bad = np.clip(bad, 0.0, 170.0)
    return bad.astype(np.float32)


def _load_tst_model_if_available(*, device: str) -> LoadedTstModel | None:
    """Auto-discover and load a TST checkpoint if present.

    Selection policy (no CLI flags):
    - Prefer the latest training run directory that ends with `_all` (the default
      "ALL FEATURES" model produced by emg_tst/run_experiment.py).
    - Within that run, pick the fold with the lowest `metrics.json.best_rmse`.
    - If no metrics are available, fall back to the most recently modified checkpoint.
    """
    ckpts_root = Path("checkpoints")
    if not ckpts_root.exists():
        return None

    candidates = list(ckpts_root.glob("**/reg_best.pt"))
    if not candidates:
        return None

    # Group by training run dir (parent of fold dir).
    by_run: dict[Path, list[Path]] = {}
    for p in candidates:
        try:
            run_dir = p.parent.parent
        except Exception:
            run_dir = p.parent
        by_run.setdefault(run_dir, []).append(p)

    run_dirs = sorted(by_run.keys(), key=lambda p: p.name)
    preferred = [d for d in run_dirs if str(d.name).endswith("_all")]
    if preferred:
        run_dirs = preferred

    # Training run dirs are named like: tst_YYYYMMDD_HHMMSS_all
    # Lexicographic max corresponds to the latest timestamp.
    chosen_run = max(run_dirs, key=lambda p: p.name)
    run_candidates = by_run.get(chosen_run, [])
    if not run_candidates:
        return None

    import json

    best_path = None
    best_rmse = None
    for p in run_candidates:
        mpath = p.parent / "metrics.json"
        if not mpath.exists():
            continue
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            rmse = float(m.get("best_rmse"))
        except Exception:
            continue
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_path = p

    ckpt_path = best_path if best_path is not None else max(run_candidates, key=lambda p: p.stat().st_mtime)
    try:
        import torch

        from emg_tst.model import TSTEncoder, TSTRegressor
    except Exception:
        return None

    dev = str(device).strip().lower()
    torch_dev = torch.device("cpu" if dev in ("", "cpu") else dev)

    ckpt = torch.load(ckpt_path, map_location=torch_dev, weights_only=False)
    cfg = ckpt["model_cfg"]
    encoder = TSTEncoder(
        n_vars=int(cfg["n_vars"]),
        seq_len=int(cfg["seq_len"]),
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        d_ff=int(cfg["d_ff"]),
        n_layers=int(cfg["n_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(torch_dev)
    reg = TSTRegressor(encoder, out_dim=1).to(torch_dev)
    reg.load_state_dict(ckpt["reg_state_dict"], strict=True)
    reg.eval()
    for p in reg.parameters():
        p.requires_grad_(False)

    scaler = ckpt.get("scaler", {})
    mean = np.asarray(scaler.get("mean", None), dtype=np.float32)
    std = np.asarray(scaler.get("std", None), dtype=np.float32)
    if mean.ndim != 1 or std.ndim != 1 or mean.size != std.size or mean.size < 1:
        raise RuntimeError(f"Invalid scaler in checkpoint: {str(ckpt_path)!r}")

    extra = ckpt.get("extra", {})
    cols = np.asarray(extra.get("feature_cols", None), dtype=np.int64)
    if cols.ndim != 1 or cols.size != int(cfg["n_vars"]):
        raise RuntimeError(
            f"Checkpoint is missing feature_cols (needed to reproduce training-time feature selection): {str(ckpt_path)!r}"
        )

    return LoadedTstModel(
        model=reg,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        feature_cols=cols.astype(np.int64),
        device=str(torch_dev),
        ckpt_path=str(ckpt_path),
    )


def _predict_knee_included_deg_for_window(model: LoadedTstModel, X_raw: np.ndarray) -> np.ndarray:
    """Return per-timestep knee included-angle prediction in degrees for one window."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required to run the TST model.") from e

    x = np.asarray(X_raw, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"X_raw must be (W,F), got {x.shape}")
    W, F = int(x.shape[0]), int(x.shape[1])
    if model.mean.size != F or model.std.size != F:
        raise ValueError(f"Checkpoint scaler expects F={model.mean.size}, but window has F={F}")

    # Match training pipeline: normalize all raw features, then select feature_cols.
    xn = (x - model.mean[None, :]) / model.std[None, :]
    xn = xn[:, model.feature_cols]

    xb = torch.from_numpy(xn[None, :, :]).to(model.device)
    with torch.no_grad():
        out = model.model(xb)  # [1,W,1]
        pred = out[0, :, 0].detach().cpu().numpy().astype(np.float32)
    if pred.shape != (W,):
        pred = pred.reshape(-1)[:W].astype(np.float32)
    return pred


def _normalize_quat_series_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1, 4)
    if q.shape[0] < 1:
        return q.astype(np.float32)
    # Normalize and enforce sign continuity.
    qn = quat_normalize_wxyz(q)
    out = np.asarray(qn, dtype=np.float64)
    for i in range(1, int(out.shape[0])):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] *= -1.0
    return out.astype(np.float32)


def _load_rigtest_query_windows(cfg: EvalConfig, *, max_n: int, seed: int) -> list[QueryWindow]:
    samples_path = Path(cfg.rig_samples_path).expanduser()
    if not samples_path.exists():
        return []

    data = np.load(samples_path, allow_pickle=True).item()
    X = np.asarray(data["X"], dtype=np.float32)  # (N,W,F)
    y_seq = np.asarray(data.get("y_seq", None), dtype=np.float32)  # (N,W)
    if y_seq is None or y_seq.ndim != 2:
        raise RuntimeError(f"{str(samples_path)!r} is missing required key 'y_seq'. Rebuild with split_to_samples.py.")

    N, W, F = int(X.shape[0]), int(X.shape[1]), int(X.shape[2])
    if W != int(cfg.window_n):
        raise RuntimeError(
            f"samples_dataset window={W} does not match mocap_phys_eval window_n={int(cfg.window_n)}. "
            "Keep train/eval windows the same length."
        )
    if int(y_seq.shape[1]) != W:
        raise RuntimeError(f"Unexpected y_seq shape {y_seq.shape}; expected (N,{W}).")

    thigh_n = int(data.get("thigh_n_features", 0))
    thigh_mode = str(data.get("thigh_mode", "")).strip().lower()
    if thigh_n != 4 or thigh_mode != "quat":
        raise RuntimeError(
            f"samples_dataset has thigh_mode={thigh_mode!r} thigh_n_features={thigh_n}; "
            "this evaluation pipeline requires thigh_quat_wxyz (4D quaternion)."
        )

    file_names = data.get("file_names", None)
    file_id = np.asarray(data.get("file_id", np.zeros((N,), dtype=np.int32)), dtype=np.int32).reshape(-1)
    starts = np.asarray(data.get("start", np.zeros((N,), dtype=np.int32)), dtype=np.int32).reshape(-1)

    n_take = int(min(int(max_n), int(N)))
    rng = np.random.default_rng(int(seed))
    idxs = rng.choice(N, size=n_take, replace=False) if n_take < N else np.arange(N, dtype=np.int64)

    out: list[QueryWindow] = []
    for idx in idxs.tolist():
        idx = int(idx)
        xw = X[idx]
        q = _normalize_quat_series_wxyz(xw[:, F - 4 : F])
        knee_inc = np.asarray(y_seq[idx], dtype=np.float32).reshape(-1)
        if knee_inc.size != W:
            knee_inc = knee_inc[:W]

        src = f"rig:{samples_path.name}:idx={idx}"
        try:
            fid = int(file_id[idx]) if file_id.size == N else -1
            st = int(starts[idx]) if starts.size == N else -1
            if file_names is not None and fid >= 0:
                fn = str(np.asarray(file_names[fid]).reshape(()))
                src += f":file={fn}:start={st}"
        except Exception:
            pass

        out.append(
            QueryWindow(
                query_id=f"rig_{idx:06d}",
                source=src,
                sample_hz=float(cfg.window_hz),
                thigh_quat_wxyz=q.astype(np.float32),
                knee_included_deg=knee_inc.astype(np.float32),
                thigh_pitch_deg=None,
                X_raw=xw.astype(np.float32),
            )
        )
    return out


def _download_demo_bvh(cfg: EvalConfig, *, dst: Path, demo_idx: int) -> tuple[Path, str]:
    urls = tuple(getattr(cfg, "query_bvh_urls", ()))
    if not urls:
        raise RuntimeError("EvalConfig.query_bvh_urls is empty.")
    start_i = int(demo_idx) % int(len(urls))
    errs: list[str] = []
    for k in range(int(len(urls))):
        u = str(urls[(start_i + k) % int(len(urls))])
        print(f"[mocap_phys_eval] status: downloading demo BVH (try={k+1}/{len(urls)})  url={u}")
        try:
            p = download_to(u, dst, force=False, timeout_s=180.0)
            return p, u
        except Exception as e:
            errs.append(f"{u} :: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("Failed to download any demo BVH.\n  - " + "\n  - ".join(errs))


def _rank_query_window_starts(
    thigh_pitch_deg: np.ndarray,
    knee_flex_deg: np.ndarray,
    *,
    window_n: int,
    sample_hz: float,
    skip_start_s: float,
    top_k: int,
) -> list[int]:
    th = np.asarray(thigh_pitch_deg, dtype=np.float32).reshape(-1)
    kn = np.asarray(knee_flex_deg, dtype=np.float32).reshape(-1)
    n = int(min(th.size, kn.size))
    w = int(window_n)
    if n <= w:
        return [0]

    nstarts = int(n - w + 1)
    skip = int(round(float(skip_start_s) * float(sample_hz)))
    skip = int(max(0, min(skip, nstarts - 1)))

    dth = np.abs(np.diff(th[:n], prepend=th[0])).astype(np.float64)
    dkn = np.abs(np.diff(kn[:n], prepend=kn[0])).astype(np.float64)
    energy = (dth + dkn).astype(np.float64)
    win = np.ones((w,), dtype=np.float64)
    score = np.convolve(energy, win, mode="valid")
    if skip > 0:
        score[:skip] = -np.inf

    order = np.argsort(score)[::-1]
    order = order[np.isfinite(score[order])]
    if order.size < 1:
        return [int(skip)]

    k = int(max(1, min(int(top_k), int(order.size))))
    starts = [int(x) for x in order[:k].tolist()]
    if int(skip) not in starts:
        starts.append(int(skip))
    return starts


def _load_demo_query_windows(cfg: EvalConfig, *, run_dir: Path, demo_idx: int, n_windows: int) -> list[QueryWindow]:
    bvh_path, bvh_url = _download_demo_bvh(cfg, dst=run_dir / "demo_query.bvh", demo_idx=demo_idx)

    print("[mocap_phys_eval] status: parsing demo BVH and extracting right-leg kinematics")
    bvh = load_bvh(bvh_path)
    thigh_pitch_deg, thigh_quat_wxyz, knee_included_deg, bvh_hz = extract_right_leg_thigh_quat_and_knee_included_deg(bvh)

    knee_flex_deg = (180.0 - np.asarray(knee_included_deg, dtype=np.float32)).astype(np.float32)
    print(f"[mocap_phys_eval] status: resampling demo BVH to {float(cfg.window_hz):.1f}Hz")
    th200 = resample_linear(thigh_pitch_deg, src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz))
    kn200 = resample_linear(knee_flex_deg, src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz))
    thq200 = resample_quat_slerp_wxyz(
        np.asarray(thigh_quat_wxyz, dtype=np.float32), src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz)
    )

    starts = _rank_query_window_starts(
        th200,
        kn200,
        window_n=int(cfg.window_n),
        sample_hz=float(cfg.window_hz),
        skip_start_s=float(cfg.query_window_skip_s),
        top_k=max(int(cfg.query_window_top_k), int(n_windows)),
    )
    n_take = int(max(1, int(n_windows)))
    # Deterministic variety: randomly sample from the top-ranked candidates.
    # (Demo-only; rigtest windows are sampled directly from samples_dataset.npy.)
    starts = starts[: int(max(int(cfg.query_window_top_k), int(n_take)))]
    if len(starts) > n_take:
        rng = np.random.default_rng(1000 + int(demo_idx))
        starts = rng.choice(np.asarray(starts, dtype=np.int64), size=int(n_take), replace=False).tolist()
    starts = sorted(int(s) for s in starts[:n_take])

    out: list[QueryWindow] = []
    for i, s in enumerate(starts):
        s = int(s)
        w = int(cfg.window_n)
        out.append(
            QueryWindow(
                query_id=f"demo_{demo_idx:03d}_{i:02d}",
                source=f"bvh_url={bvh_url} start_s={float(s)/float(cfg.window_hz):.3f}",
                sample_hz=float(cfg.window_hz),
                thigh_quat_wxyz=_normalize_quat_series_wxyz(thq200[s : s + w]),
                knee_included_deg=(180.0 - kn200[s : s + w]).astype(np.float32),  # included for schema
                thigh_pitch_deg=th200[s : s + w].astype(np.float32),
                X_raw=None,
            )
        )
    return out


def _npz_has_key(path: Path, key: str) -> bool:
    try:
        with np.load(path, allow_pickle=True) as d:
            return bool(key in getattr(d, "files", ()))
    except Exception:
        return False


def _ensure_reference_bank(cfg: EvalConfig, *, out_root: Path) -> ExpertSnippetBank:
    ensure_full_expert_zoo(experts_root=cfg.experts_root, downloads_dir=cfg.experts_downloads_dir)
    snippets = discover_expert_snippets(cfg.experts_root)
    print(f"[mocap_phys_eval] status: experts discovered (n_snippets={len(snippets)})")
    if len(snippets) < 2500:
        raise RuntimeError(
            f"Expected ~2589 expert snippets, but found only {len(snippets)} under {str(cfg.experts_root)!r}. "
            "This usually means the expert zoo download/extraction is incomplete."
        )

    bank_path = out_root / "reference_bank" / "mocapact_expert_snippets_right.npz"
    bank_ok = False
    if bank_path.exists():
        try:
            print(f"[mocap_phys_eval] status: loading reference bank ({bank_path})")
            bank = ExpertSnippetBank.load_npz(bank_path)
            bank_ok = bool(len(bank) >= int(len(snippets))) and _npz_has_key(bank_path, "thigh_anat_quat_world_wxyz")
        except Exception:
            bank_ok = False
    if not bank_ok:
        print("[mocap_phys_eval] status: building expert snippet reference bank (this may take a while)")
        bank = build_expert_snippet_bank(experts_root=cfg.experts_root, side="right")
        bank.save_npz(bank_path)
    return bank


def _align_query_thigh_quat(*, query_thq: np.ndarray, cand: MatchCandidate) -> np.ndarray:
    qoff = quat_normalize_wxyz(np.asarray(cand.thigh_quat_offset_wxyz, dtype=np.float64).reshape(4))
    q = quat_normalize_wxyz(np.asarray(query_thq, dtype=np.float64).reshape(-1, 4))
    if bool(getattr(cand, "thigh_quat_conjugated", False)):
        q = quat_conj_wxyz(q)
    q_al = quat_mul_wxyz(qoff[None, :], q)
    return np.asarray(q_al, dtype=np.float32)


def _session_run_dir(cfg: EvalConfig, *, out_root: Path) -> tuple[Path, int, str]:
    runs_root = ensure_dir(out_root / "runs")
    try:
        demo_idx = sum(1 for p in runs_root.iterdir() if p.is_dir())
    except Exception:
        demo_idx = 0
    run_id = now_run_id()
    run_dir = ensure_dir(runs_root / run_id)
    ensure_dir(run_dir / "evals")
    write_json(run_dir / "config.json", dataclass_to_json_dict(cfg))
    return run_dir, int(demo_idx), str(run_id)


def main() -> None:
    cfg = EvalConfig()
    set_global_determinism(seed=0)

    out_root = ensure_dir(cfg.artifacts_dir)
    run_dir, demo_idx, run_id = _session_run_dir(cfg, out_root=out_root)

    models_dir_env = os.environ.get("MOCAPACT_MODELS_DIR")
    print(f"[mocap_phys_eval] status: init (run_id={run_id})")
    print(f"[mocap_phys_eval] config: MOCAPACT_MODELS_DIR={models_dir_env!r}")
    print(f"[mocap_phys_eval] config: experts_root={str(Path(cfg.experts_root).resolve())}")
    print(f"[mocap_phys_eval] config: artifacts_dir={str(Path(cfg.artifacts_dir).resolve())}")

    # 1) Query windows (rigtest first, demo BVH fallback).
    n_windows = int(max(1, int(getattr(cfg, "eval_n_windows", 1))))
    queries = _load_rigtest_query_windows(cfg, max_n=n_windows, seed=int(demo_idx) + 1)
    query_mode = "rigtest" if queries else "demo_bvh"
    if not queries:
        queries = _load_demo_query_windows(cfg, run_dir=run_dir, demo_idx=int(demo_idx), n_windows=n_windows)
    if not queries:
        raise RuntimeError("No query windows available (neither rigtest samples nor demo BVH).")

    # 2) Optional TST model.
    tst = None
    try:
        tst = _load_tst_model_if_available(device=str(cfg.device))
    except Exception as e:
        print(f"[mocap_phys_eval] warning: failed to load TST checkpoint (continuing with oracle knee): {e}")
        tst = None
    if tst is not None:
        print(f"[mocap_phys_eval] status: loaded TST checkpoint: {tst.ckpt_path}")
    else:
        print("[mocap_phys_eval] status: no TST checkpoint found; GOOD uses oracle (ground-truth) knee")

    # 3) Expert zoo + reference bank.
    print("[mocap_phys_eval] status: ensuring MoCapAct expert zoo + reference bank")
    bank = _ensure_reference_bank(cfg, out_root=out_root)
    bank_snip_to_i = {str(bank.snippet_id[i]): int(i) for i in range(len(bank))}
    match_hz = float(np.median(np.asarray(bank.sample_hz, dtype=np.float64)))
    print(f"[mocap_phys_eval] status: bank ready (n_snippets={len(bank)}; sample_hz~{match_hz:.2f})")

    # 4) Evaluate each window independently (no aggregation).
    eval_results: list[dict[str, Any]] = []
    last_compare_npz: Path | None = None

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    items = list(enumerate(queries))
    it = items if tqdm is None else tqdm(items, desc="Eval windows", unit="win")

    for qi, q in it:
        qi = int(qi)
        q_dir = ensure_dir(run_dir / "evals" / f"{qi:02d}_{q.query_id}")
        plots_dir = ensure_dir(q_dir / "plots")
        replay_dir = ensure_dir(q_dir / "replay")

        # Persist raw query window for reproducibility.
        np.savez_compressed(
            q_dir / "query_window.npz",
            query_id=np.asarray(str(q.query_id)),
            source=np.asarray(str(q.source)),
            mode=np.asarray(str(query_mode)),
            sample_hz=np.asarray(float(q.sample_hz), dtype=np.float32),
            thigh_quat_wxyz=np.asarray(q.thigh_quat_wxyz, dtype=np.float32),
            knee_included_deg=np.asarray(q.knee_included_deg, dtype=np.float32),
            thigh_pitch_deg=(
                np.asarray(q.thigh_pitch_deg, dtype=np.float32)
                if q.thigh_pitch_deg is not None
                else np.zeros((0,), dtype=np.float32)
            ),
        )

        # Resample query to bank rate for matching/simulation.
        thq = resample_quat_slerp_wxyz(q.thigh_quat_wxyz, src_hz=float(q.sample_hz), dst_hz=float(match_hz))
        kn_inc = resample_linear(q.knee_included_deg, src_hz=float(q.sample_hz), dst_hz=float(match_hz))
        kn_flex = (180.0 - np.asarray(kn_inc, dtype=np.float32)).astype(np.float32)

        L = int(min(int(thq.shape[0]), int(kn_flex.size)))
        if L < 2:
            raise RuntimeError(f"Query window too short after resampling (L={L}).")
        thq = thq[:L]
        kn_flex = kn_flex[:L]

        print(f"[mocap_phys_eval] status: motion matching ({q.query_id})  bank={len(bank)}  L={L}")
        candidates = motion_match_one_window(
            bank=bank,
            query_thigh_deg=None,
            query_thigh_quat_wxyz=thq,
            query_knee_deg=kn_flex,
            top_k=int(cfg.match_top_k),
            feature_mode=str(cfg.match_feature_mode),
        )
        if not candidates:
            raise RuntimeError("No motion-match candidates found.")
        cand = candidates[0]
        if cand.snippet_id not in bank_snip_to_i:
            raise RuntimeError(f"Matched snippet_id {cand.snippet_id!r} not found in reference bank index.")
        bi = int(bank_snip_to_i[str(cand.snippet_id)])

        # Reference series for the matched segment (snippet-local).
        ref_kn_full = np.asarray(bank.knee_deg[bi], dtype=np.float32).reshape(-1)
        ref_kn = ref_kn_full[int(cand.start_step) : int(cand.start_step) + L]
        ref_thq_full = np.asarray(bank.thigh_anat_quat_world_wxyz[bi], dtype=np.float32).reshape(-1, 4)
        ref_thq = ref_thq_full[int(cand.start_step) : int(cand.start_step) + L]

        # Align query into reference coordinates.
        query_thq_al = _align_query_thigh_quat(query_thq=thq, cand=cand)
        thigh_err_deg = quat_geodesic_deg_wxyz(quat_normalize_wxyz(ref_thq), quat_normalize_wxyz(query_thq_al)).astype(
            np.float32
        )
        kn_al = (float(cand.knee_sign) * kn_flex + float(cand.knee_offset_deg)).astype(np.float32)

        mm_plot = plot_motion_match(
            out_path=plots_dir / "motion_match.png",
            sample_hz=float(match_hz),
            ref_thigh_deg=None,
            ref_knee_deg=ref_kn,
            query_thigh_aligned_deg=None,
            query_knee_aligned_deg=kn_al,
            rmse_thigh_deg=float(cand.rmse_thigh_deg),
            rmse_knee_deg=float(cand.rmse_knee_deg),
            thigh_ori_err_deg=np.asarray(thigh_err_deg, dtype=np.float32),
            title=f"Motion Match  query={q.query_id}  snippet={cand.snippet_id}  start_in_snip={cand.start_step}  L={L}",
        )

        quat_plot = plot_thigh_quat_match(
            out_path=plots_dir / "thigh_quat_match.png",
            sample_hz=float(match_hz),
            ref_thigh_quat_wxyz=np.asarray(ref_thq, dtype=np.float32),
            query_thigh_quat_aligned_wxyz=np.asarray(query_thq_al, dtype=np.float32),
            thigh_ori_err_deg=np.asarray(thigh_err_deg, dtype=np.float32),
            title="Thigh Quaternion Match (ref vs query aligned)",
        )

        # GOOD prediction: model if available (rigtest only), else oracle ground-truth.
        good_kn_flex = kn_flex.copy()
        pred_included_200 = None
        pred_flex_200 = None
        if tst is not None and q.X_raw is not None:
            try:
                pred_included_200 = _predict_knee_included_deg_for_window(tst, q.X_raw)
                pred_flex_200 = (180.0 - np.asarray(pred_included_200, dtype=np.float32)).astype(np.float32)
                good_kn_flex = resample_linear(pred_flex_200, src_hz=float(q.sample_hz), dst_hz=float(match_hz))[:L].astype(
                    np.float32
                )
            except Exception as e:
                print(f"[mocap_phys_eval] warning: TST inference failed (using oracle knee): {e}")
                good_kn_flex = kn_flex.copy()
                pred_included_200 = None
                pred_flex_200 = None

        bad_kn_flex = _make_bad_knee_prediction(
            good_kn_flex,
            sample_hz=float(match_hz),
            target_rmse_deg=float(cfg.bad_knee_rmse_deg),
            lowpass_hz=float(cfg.bad_knee_lowpass_hz),
            seed=1234 + int(qi),
        )[:L].astype(np.float32)

        pred_bad_rmse = float(_rmse(bad_kn_flex, good_kn_flex))

        # Model error (independent of motion matching).
        pred_vs_gt_rmse = None
        if pred_flex_200 is not None:
            gt_flex_200 = (180.0 - np.asarray(q.knee_included_deg, dtype=np.float32)).astype(np.float32)
            pred_flex_200 = pred_flex_200[: int(min(pred_flex_200.size, gt_flex_200.size))]
            gt_flex_200 = gt_flex_200[: int(min(pred_flex_200.size, gt_flex_200.size))]
            pred_vs_gt_rmse = float(_rmse(pred_flex_200, gt_flex_200))

        # Align override knee series into reference joint coordinates (constant sign + offset from matching).
        good_target_kn = (float(cand.knee_sign) * good_kn_flex + float(cand.knee_offset_deg)).astype(np.float32)
        bad_target_kn = (float(cand.knee_sign) * bad_kn_flex + float(cand.knee_offset_deg)).astype(np.float32)
        good_target_kn = np.clip(good_target_kn, 0.0, 170.0).astype(np.float32)
        bad_target_kn = np.clip(bad_target_kn, 0.0, 170.0).astype(np.float32)

        # Simulation segment coordinates.
        clip_id = str(cand.clip_id)
        snip_abs_start = int(np.asarray(bank.start_step[bi]).reshape(()))
        snip_abs_end = int(np.asarray(bank.end_step[bi]).reshape(()))
        eval_start_abs = int(snip_abs_start + int(cand.start_step))
        warmup_steps = int(max(0, int(cand.start_step)))
        if (eval_start_abs + int(L) - 1) > int(snip_abs_end):
            raise RuntimeError("Matched window extends past snippet end; this should never happen.")

        # Expert policy for the matched snippet.
        expert_model_path = Path(str(np.asarray(bank.expert_model_path[bi]).reshape(())))
        policy = load_expert_policy(expert_model_path, device=str(cfg.device))

        override = OverrideSpec(knee_actuator=str(cfg.knee_actuator), knee_sign=1.0, knee_offset_deg=0.0)

        print(
            f"[mocap_phys_eval] status: sim (REF|GOOD|BAD)  clip={clip_id}  "
            f"start_abs={eval_start_abs} warmup={warmup_steps} L={L}"
        )
        rec_paths = record_compare_rollout(
            out_npz_path=replay_dir / "compare.npz",
            clip_id=str(clip_id),
            start_step=int(eval_start_abs),
            end_step=int(snip_abs_end),
            primary_steps=int(L),
            warmup_steps=int(warmup_steps),
            policy=policy,
            override=override,
            knee_good_query_deg=np.asarray(good_target_kn, dtype=np.float32),
            knee_bad_query_deg=np.asarray(bad_target_kn, dtype=np.float32),
            width=int(cfg.render_width),
            height=int(cfg.render_height),
            camera_id=int(cfg.render_camera_id),
            deterministic_policy=True,
            seed=0,
        )
        last_compare_npz = rec_paths.npz_path

        rr = np.load(rec_paths.npz_path, allow_pickle=True)
        knee_ref_actual = np.asarray(rr["knee_ref_actual_deg"], dtype=np.float32)
        knee_good_actual = np.asarray(rr["knee_good_actual_deg"], dtype=np.float32)
        knee_bad_actual = np.asarray(rr["knee_bad_actual_deg"], dtype=np.float32)

        sim_knee_plot = plot_simulation_knee(
            out_path=plots_dir / "simulation_knee.png",
            sample_hz=float(match_hz),
            ref_target_knee_deg=np.asarray(ref_kn, dtype=np.float32),
            good_target_knee_deg=np.asarray(good_target_kn[: knee_good_actual.size], dtype=np.float32),
            bad_target_knee_deg=np.asarray(bad_target_kn[: knee_bad_actual.size], dtype=np.float32),
            ref_actual_knee_deg=knee_ref_actual,
            good_actual_knee_deg=knee_good_actual,
            bad_actual_knee_deg=knee_bad_actual,
            title="Simulation: Right Knee Flexion (target vs actual)",
        )

        bal_plot = plot_balance_traces(
            out_path=plots_dir / "simulation_balance.png",
            sample_hz=float(match_hz),
            xcom_margin_ref_m=np.asarray(rr["balance_xcom_margin_ref_m"], dtype=np.float32),
            xcom_margin_good_m=np.asarray(rr["balance_xcom_margin_good_m"], dtype=np.float32),
            xcom_margin_bad_m=np.asarray(rr["balance_xcom_margin_bad_m"], dtype=np.float32),
            upright_ref=np.asarray(rr["upright_ref"], dtype=np.float32),
            upright_good=np.asarray(rr["upright_good"], dtype=np.float32),
            upright_bad=np.asarray(rr["upright_bad"], dtype=np.float32),
            risk_trace_ref=np.asarray(rr["predicted_fall_risk_trace_ref"], dtype=np.float32),
            risk_trace_good=np.asarray(rr["predicted_fall_risk_trace_good"], dtype=np.float32),
            risk_trace_bad=np.asarray(rr["predicted_fall_risk_trace_bad"], dtype=np.float32),
            balance_loss_step_ref=int(np.asarray(rr["balance_loss_step_ref"]).reshape(())),
            balance_loss_step_good=int(np.asarray(rr["balance_loss_step_good"]).reshape(())),
            balance_loss_step_bad=int(np.asarray(rr["balance_loss_step_bad"]).reshape(())),
            risk_ref=float(np.asarray(rr["predicted_fall_risk_ref"]).reshape(())),
            risk_good=float(np.asarray(rr["predicted_fall_risk_good"]).reshape(())),
            risk_bad=float(np.asarray(rr["predicted_fall_risk_bad"]).reshape(())),
            title="Simulation Balance Signals (COM margin, uprightness) + predicted fall risk",
        )

        # Diagnostics: policy output vs applied for the forced actuator.
        ctrl_pol_good = np.asarray(rr.get("ctrl_knee_good_policy", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        ctrl_app_good = np.asarray(rr.get("ctrl_knee_good_applied", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        ctrl_tgt_good = np.asarray(rr.get("ctrl_knee_good_target", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        ctrl_pol_bad = np.asarray(rr.get("ctrl_knee_bad_policy", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        ctrl_app_bad = np.asarray(rr.get("ctrl_knee_bad_applied", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        ctrl_tgt_bad = np.asarray(rr.get("ctrl_knee_bad_target", np.zeros((0,), dtype=np.float32)), dtype=np.float32)

        ctrl_diag = {
            "good": {
                "mean_abs_policy_minus_applied": float(_mean_abs(ctrl_pol_good - ctrl_app_good)),
                "mean_abs_applied_minus_target": float(_mean_abs(ctrl_app_good - ctrl_tgt_good)),
            },
            "bad": {
                "mean_abs_policy_minus_applied": float(_mean_abs(ctrl_pol_bad - ctrl_app_bad)),
                "mean_abs_applied_minus_target": float(_mean_abs(ctrl_app_bad - ctrl_tgt_bad)),
            },
        }

        sim_metrics = {
            "ref": {
                "predicted_fall_risk": float(np.asarray(rr["predicted_fall_risk_ref"]).reshape(())),
                "balance_loss_step": int(np.asarray(rr["balance_loss_step_ref"]).reshape(())),
                "knee_rmse_deg": float(_rmse(ref_kn, knee_ref_actual)),
            },
            "good": {
                "predicted_fall_risk": float(np.asarray(rr["predicted_fall_risk_good"]).reshape(())),
                "balance_loss_step": int(np.asarray(rr["balance_loss_step_good"]).reshape(())),
                "knee_rmse_deg": float(_rmse(good_target_kn, knee_good_actual)),
            },
            "bad": {
                "predicted_fall_risk": float(np.asarray(rr["predicted_fall_risk_bad"]).reshape(())),
                "balance_loss_step": int(np.asarray(rr["balance_loss_step_bad"]).reshape(())),
                "knee_rmse_deg": float(_rmse(bad_target_kn, knee_bad_actual)),
            },
        }

        top_candidates = [
            {
                "snippet_id": str(c.snippet_id),
                "start_step_in_snip": int(c.start_step),
                "score": float(c.score),
                "rmse_knee_deg": float(c.rmse_knee_deg),
                "rms_thigh_ori_err_deg": float(c.rmse_thigh_deg),
            }
            for c in candidates[: min(5, len(candidates))]
        ]

        result = {
            "query_id": str(q.query_id),
            "source": str(q.source),
            "mode": str(query_mode),
            "match": {
                "snippet_id": str(cand.snippet_id),
                "clip_id": str(cand.clip_id),
                "start_step_in_snip": int(cand.start_step),
                "start_step_abs": int(eval_start_abs),
                "L": int(L),
                "rmse_knee_deg": float(cand.rmse_knee_deg),
                "rms_thigh_ori_err_deg": float(cand.rmse_thigh_deg),
                "knee_sign": float(cand.knee_sign),
                "knee_offset_deg": float(cand.knee_offset_deg),
                "thigh_quat_offset_wxyz": [float(x) for x in cand.thigh_quat_offset_wxyz],
                "thigh_quat_conjugated": bool(getattr(cand, "thigh_quat_conjugated", False)),
                "score": float(cand.score),
                "top_candidates": top_candidates,
            },
            "model": {
                "tst_ckpt": (None if tst is None else str(tst.ckpt_path)),
                "good_is_oracle": bool(tst is None or q.X_raw is None),
                "pred_vs_gt_knee_flex_rmse_deg": pred_vs_gt_rmse,
                "bad_target_rmse_deg": float(cfg.bad_knee_rmse_deg),
                "bad_actual_rmse_deg": float(pred_bad_rmse),
            },
            "override": {
                "knee_actuator": str(cfg.knee_actuator),
                "prosthetic_knee_kp": float(np.asarray(rr.get("prosthetic_knee_kp", float("nan"))).reshape(())),
                "prosthetic_knee_kd": float(np.asarray(rr.get("prosthetic_knee_kd", float("nan"))).reshape(())),
                "prosthetic_knee_force": float(np.asarray(rr.get("prosthetic_knee_force", float("nan"))).reshape(())),
                "ctrl_override_diag": ctrl_diag,
            },
            "sim": sim_metrics,
            "artifacts": {
                "motion_match_plot": str(mm_plot),
                "thigh_quat_plot": str(quat_plot),
                "simulation_knee_plot": str(sim_knee_plot),
                "simulation_balance_plot": str(bal_plot),
                "compare_npz": str(rec_paths.npz_path),
                "compare_gif": str(rec_paths.gif_path),
            },
        }

        write_json(q_dir / "summary.json", result)
        eval_results.append(result)

        # Update top-level convenience pointers to the most recent window.
        try:
            shutil.copyfile(rec_paths.npz_path, out_root / "latest_compare.npz")
        except Exception:
            pass
        try:
            if rec_paths.gif_path.exists():
                shutil.copyfile(rec_paths.gif_path, out_root / "latest_compare.gif")
        except Exception:
            pass
        try:
            shutil.copyfile(mm_plot, out_root / "latest_motion_match.png")
        except Exception:
            pass
        try:
            shutil.copyfile(quat_plot, out_root / "latest_thigh_quat_match.png")
        except Exception:
            pass

    session = {
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "mode": str(query_mode),
        "n_windows": int(len(eval_results)),
        "bank_n_snippets": int(len(bank)),
        "bank_sample_hz": float(match_hz),
        "results": eval_results,
    }
    write_json(run_dir / "summary.json", session)

    print(f"[mocap_phys_eval] run_id={run_id}")
    print(f"[mocap_phys_eval] mode={query_mode} windows={len(eval_results)} bank_snippets={len(bank)}")
    print(f"[mocap_phys_eval] wrote: {run_dir / 'summary.json'}")
    print(f"[mocap_phys_eval] latest replay: {out_root / 'latest_compare.npz'}")

    # Launch viewer for the last recording in a separate process so it stays open.
    if last_compare_npz is not None:
        try:
            subprocess.Popen([sys.executable, "-m", "mocap_phys_eval.replay", str(last_compare_npz)])
        except Exception:
            pass


if __name__ == "__main__":
    main()
