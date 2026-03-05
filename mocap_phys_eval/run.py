from __future__ import annotations

import shutil
import subprocess
import sys
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bvh import extract_right_leg_thigh_quat_and_knee_included_deg, load_bvh
from .config import EvalConfig
from .matching import MatchCandidate, motion_match_one_window
from .plots import plot_balance_traces, plot_motion_match, plot_simulation_angles, plot_thigh_quat_match
from .recording import record_compare_rollout
from .experts import discover_expert_snippets, ensure_full_expert_zoo
from .reference_bank import ExpertSnippetBank, build_expert_snippet_bank
from .sim import OverrideSpec, load_expert_policy, run_reference_stability_check
from .utils import (
    dataclass_to_json_dict,
    download_to,
    ensure_dir,
    now_run_id,
    quat_geodesic_deg_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    resample_linear,
    resample_quat_slerp_wxyz,
    set_global_determinism,
    write_json,
)


def _make_bad_knee_prediction(
    knee_good_deg: np.ndarray, *, sample_hz: float, target_rmse_deg: float, lowpass_hz: float
) -> np.ndarray:
    """Create a smooth "bad model" knee signal with ~target_rmse_deg vs knee_good.

    Notes:
    - This is only used for the "no trained model yet" demo mode.
    - We intentionally avoid hard clipping artifacts that can produce unrealistic 0<->180 jumps.
    """
    knee_good_deg = np.asarray(knee_good_deg, dtype=np.float64).reshape(-1)
    n = int(knee_good_deg.size)
    if n < 2:
        return knee_good_deg.astype(np.float32)

    # Deterministic, smooth error profile:
    # Use a raised cosine (0..1) so the signal is always smooth and starts at 0.
    # This also avoids lower-bound clamping when the true knee flexion is near 0.
    #
    # If you want a different "bad model" later, replace this with your model output.
    import math

    hz = float(max(1e-6, float(sample_hz)))
    t = np.arange(n, dtype=np.float64) / hz
    # Pick a slow-ish frequency so the error evolves smoothly within a ~1s window.
    # Respect lowpass_hz as an upper bound.
    f = float(max(0.35, min(1.0, 0.5 * float(lowpass_hz))))
    base = 0.5 * (1.0 - np.cos(2.0 * math.pi * f * t))  # in [0, 1], starts at 0
    base_rms = float(np.sqrt(float(np.mean(base**2))))
    if base_rms < 1e-6:
        return knee_good_deg.astype(np.float32)

    amp = float(target_rmse_deg) / base_rms

    # Keep within plausible knee flexion range without introducing sharp clipping.
    # Since base is non-negative, only the upper bound matters.
    upper_room = 170.0 - knee_good_deg
    if np.any(np.isfinite(upper_room)):
        # Ignore timesteps where base is ~0 (amp limit would be infinite).
        mask = base > 1e-6
        if bool(np.any(mask)):
            amp_max = float(np.min(upper_room[mask] / base[mask]))
            if math.isfinite(amp_max):
                amp = min(amp, 0.98 * amp_max)

    err = amp * base
    bad = knee_good_deg + err
    bad = np.clip(bad, 0.0, 170.0)
    return bad.astype(np.float32)


def _pick_query_window(x: np.ndarray, *, window_n: int, prefer_start_s: float, sample_hz: float) -> int:
    # Deprecated (kept for backwards compatibility if called elsewhere).
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(x.size)
    w = int(window_n)
    if n <= w:
        return 0
    start = int(round(float(prefer_start_s) * float(sample_hz)))
    start = max(0, min(start, n - w))
    return int(start)


def _pick_demo_query_url(cfg: EvalConfig, *, demo_idx: int) -> str:
    urls = tuple(getattr(cfg, "query_bvh_urls", ()))
    if not urls:
        raise RuntimeError("EvalConfig.query_bvh_urls is empty; add at least one demo BVH URL.")
    return str(urls[int(demo_idx) % int(len(urls))])


def _pick_query_window_start(
    thigh: np.ndarray,
    knee: np.ndarray,
    *,
    window_n: int,
    sample_hz: float,
    skip_start_s: float,
    top_k: int,
    pick_idx: int,
) -> int:
    """Pick a single TST window start from a longer query by "motion energy"."""
    th = np.asarray(thigh, dtype=np.float32).reshape(-1)
    kn = np.asarray(knee, dtype=np.float32).reshape(-1)
    n = int(min(th.size, kn.size))
    w = int(window_n)
    if n <= w:
        return 0

    nstarts = int(n - w + 1)
    skip = int(round(float(skip_start_s) * float(sample_hz)))
    skip = int(max(0, min(skip, nstarts - 1)))

    # Energy over time (abs derivatives); convolve over window to get per-start score.
    dth = np.abs(np.diff(th[:n], prepend=th[0])).astype(np.float64)
    dkn = np.abs(np.diff(kn[:n], prepend=kn[0])).astype(np.float64)
    energy = (dth + dkn).astype(np.float64)
    win = np.ones((w,), dtype=np.float64)
    score = np.convolve(energy, win, mode="valid")  # (nstarts,)
    if skip > 0:
        score[:skip] = -np.inf

    order = np.argsort(score)[::-1]  # descending; -inf ends up last
    order = order[np.isfinite(score[order])]
    if order.size < 1:
        return int(skip)

    k = int(max(1, min(int(top_k), int(order.size))))
    start = int(order[int(pick_idx) % int(k)])
    return int(max(0, min(start, n - w)))


def _rank_query_window_starts(
    thigh: np.ndarray,
    knee: np.ndarray,
    *,
    window_n: int,
    sample_hz: float,
    skip_start_s: float,
    top_k: int,
) -> list[int]:
    """Return query window starts ranked by "motion energy" (descending)."""
    th = np.asarray(thigh, dtype=np.float32).reshape(-1)
    kn = np.asarray(knee, dtype=np.float32).reshape(-1)
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
    score = np.convolve(energy, win, mode="valid")  # (nstarts,)
    if skip > 0:
        score[:skip] = -np.inf

    order = np.argsort(score)[::-1]  # descending; -inf ends up last
    order = order[np.isfinite(score[order])]
    if order.size < 1:
        return [int(skip)]

    k = int(max(1, min(int(top_k), int(order.size))))
    starts = [int(x) for x in order[:k].tolist()]
    # Always include a deterministic fallback at skip (helps short/low-motion clips).
    if int(skip) not in starts:
        starts.append(int(skip))
    return starts


def main() -> None:
    cfg = EvalConfig()
    set_global_determinism(seed=0)

    out_root = ensure_dir(cfg.artifacts_dir)
    runs_root = ensure_dir(out_root / "runs")
    try:
        demo_idx = sum(1 for p in runs_root.iterdir() if p.is_dir())
    except Exception:
        demo_idx = 0
    run_id = now_run_id()
    run_dir = ensure_dir(out_root / "runs" / run_id)
    plots_dir = ensure_dir(run_dir / "plots")
    replay_dir = ensure_dir(run_dir / "replay")

    models_dir_env = os.environ.get("MOCAPACT_MODELS_DIR")
    print(f"[mocap_phys_eval] status: init (run_id={run_id})")
    print(f"[mocap_phys_eval] config: MOCAPACT_MODELS_DIR={models_dir_env!r}")
    print(f"[mocap_phys_eval] config: experts_root={str(Path(cfg.experts_root).resolve())}")
    print(f"[mocap_phys_eval] config: experts_downloads_dir={str(Path(cfg.experts_downloads_dir).resolve())}")
    write_json(run_dir / "config.json", dataclass_to_json_dict(cfg))

    # 1) Download a real BVH query (non-synthetic).
    #
    # Some public BVH hosts are flaky; we try the configured URL list in a deterministic
    # order (starting at demo_idx) and take the first that downloads successfully.
    urls = tuple(getattr(cfg, "query_bvh_urls", ()))
    if not urls:
        raise RuntimeError("EvalConfig.query_bvh_urls is empty; add at least one BVH URL.")
    start_i = int(demo_idx) % int(len(urls))
    bvh_path = None
    query_url = None
    errs: list[str] = []
    for k in range(int(len(urls))):
        u = str(urls[(start_i + k) % int(len(urls))])
        print(f"[mocap_phys_eval] status: downloading BVH (try={k+1}/{len(urls)})  url={u}")
        try:
            bvh_path = download_to(u, run_dir / "query.bvh", force=False, timeout_s=180.0)
            query_url = u
            break
        except Exception as e:
            errs.append(f"{u} :: {type(e).__name__}: {e}")
            continue
    if bvh_path is None or query_url is None:
        raise RuntimeError(
            "Failed to download any demo BVH query clip. "
            "This is usually a network/firewall/host-availability issue, not a MoCapAct issue. "
            "If you only want to download the MoCapAct expert zoo first, run: "
            "`python -m mocap_phys_eval.prefetch`. "
            "Download attempts:\n  - " + "\n  - ".join(errs)
        )

    # 2) Extract thigh pitch + thigh orientation quaternion + knee included angle from BVH,
    # then convert to knee flexion (MoCapAct joint convention).
    print("[mocap_phys_eval] status: parsing BVH and extracting right-leg angles")
    bvh = load_bvh(bvh_path)
    thigh_pitch_deg, thigh_quat_wxyz, knee_included_deg, bvh_hz = extract_right_leg_thigh_quat_and_knee_included_deg(bvh)
    knee_flex_deg = (180.0 - np.asarray(knee_included_deg, dtype=np.float32)).astype(np.float32)

    # 3) Resample to the TST windowing rate (we still evaluate *one* window, but we
    # may search over multiple candidate starts to find a stable REF match).
    print(f"[mocap_phys_eval] status: resampling to {float(cfg.window_hz):.1f}Hz")
    thigh_200 = resample_linear(thigh_pitch_deg, src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz))
    knee_200 = resample_linear(knee_flex_deg, src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz))
    thigh_quat_200 = resample_quat_slerp_wxyz(
        np.asarray(thigh_quat_wxyz, dtype=np.float32), src_hz=float(bvh_hz), dst_hz=float(cfg.window_hz)
    )

    # 4) Ensure MoCapAct expert zoo is present (download+extract if needed).
    print("[mocap_phys_eval] status: ensuring MoCapAct expert model zoo is available (~2589 snippet experts)")
    ensure_full_expert_zoo(experts_root=cfg.experts_root, downloads_dir=cfg.experts_downloads_dir)
    expected_snips = len(discover_expert_snippets(cfg.experts_root))
    if expected_snips < 2500:
        raise RuntimeError(
            f"Expected ~2589 expert snippets, but found only {expected_snips} under {str(cfg.experts_root)!r}. "
            "This usually means the expert model zoo download/extraction is incomplete. "
            f"Re-run the pipeline to resume downloads, or delete {str(cfg.experts_downloads_dir)!r}/*.extracted to force re-extract."
        )

    # 5) Load/build reference bank aligned to the expert snippet boundaries.
    bank_path = out_root / "reference_bank" / "mocapact_expert_snippets_right.npz"
    bank_ok = False
    if bank_path.exists():
        try:
            print(f"[mocap_phys_eval] status: loading reference bank ({bank_path})")
            has_anat = False
            try:
                with np.load(bank_path, allow_pickle=True) as d0:
                    has_anat = bool("thigh_anat_quat_wxyz" in getattr(d0, "files", ()))
            except Exception:
                has_anat = False
            bank = ExpertSnippetBank.load_npz(bank_path)
            bank_ok = bool(len(bank) >= int(expected_snips)) and bool(has_anat)
        except Exception:
            bank_ok = False
    if not bank_ok:
        print("[mocap_phys_eval] status: building expert snippet reference bank from dm_control CMU2020 HDF5")
        bank = build_expert_snippet_bank(experts_root=cfg.experts_root, side="right")
        bank.save_npz(bank_path)

    match_hz = float(np.median(bank.sample_hz.astype(np.float64)))
    print(f"[mocap_phys_eval] status: expert snippet bank ready (n_snippets={len(bank)})")

    # 6) Motion-match exactly one TST window (no aggregation; no stability gating/cherry-picking).
    #
    # We *search over a handful of candidate window starts* (ranked by motion energy)
    # and pick the one with the best motion-match score. This improves match RMSE
    # without aggregating windows.
    snip_to_i = {str(bank.snippet_id[i]): int(i) for i in range(len(bank))}

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    starts = _rank_query_window_starts(
        thigh_200,
        knee_200,
        window_n=int(cfg.window_n),
        sample_hz=float(cfg.window_hz),
        skip_start_s=float(cfg.query_window_skip_s),
        top_k=int(cfg.query_window_top_k),
    )

    print(f"[mocap_phys_eval] status: motion matching (windows={len(starts)}; bank_clips={len(bank)})")
    w = int(cfg.window_n)
    best_score = float("inf")
    best_payload: (
        tuple[
            int,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
            list[MatchCandidate],
            MatchCandidate,
        ]
        | None
    ) = None

    starts_it = starts
    if tqdm is not None:
        starts_it = tqdm(starts_it, desc="Query windows", unit="win", leave=False)

    for s in starts_it:
        s = int(s)
        if s < 0 or (s + w) > int(min(thigh_200.size, knee_200.size)):
            continue
        th200 = thigh_200[s : s + w]
        thq200 = thigh_quat_200[s : s + w]
        kn200 = knee_200[s : s + w]
        if th200.size != w or kn200.size != w or int(thq200.shape[0]) != w:
            continue

        th = resample_linear(th200, src_hz=float(cfg.window_hz), dst_hz=float(match_hz))
        kn = resample_linear(kn200, src_hz=float(cfg.window_hz), dst_hz=float(match_hz))
        thq = resample_quat_slerp_wxyz(thq200, src_hz=float(cfg.window_hz), dst_hz=float(match_hz))
        L = int(min(th.size, kn.size, int(thq.shape[0])))
        th = th[:L]
        kn = kn[:L]
        thq = thq[:L]
        if L < 2:
            continue

        candidates = motion_match_one_window(
            bank=bank,
            query_thigh_deg=th,
            query_thigh_quat_wxyz=thq,
            query_knee_deg=kn,
            top_k=int(cfg.match_top_k),
            feature_mode=str(cfg.match_feature_mode),
        )
        if not candidates:
            continue
        cand0 = candidates[0]
        if float(cand0.score) < float(best_score):
            best_score = float(cand0.score)
            best_payload = (
                s,
                th200,
                thq200,
                kn200,
                th.astype(np.float32),
                thq.astype(np.float32),
                kn.astype(np.float32),
                int(L),
                candidates,
                cand0,
            )
            # Early exit once the match is already strong.
            if float(best_score) <= 8.0:
                break

    if best_payload is None:
        raise RuntimeError("No motion-match candidates found for any query window start.")

    (
        start,
        thigh_win_200,
        thigh_quat_win_200,
        knee_win_200,
        thigh_win,
        thigh_quat_win,
        knee_win,
        L,
        selected_candidates,
        selected,
    ) = best_payload

    bi = snip_to_i.get(str(selected.snippet_id))
    if bi is None:
        raise RuntimeError("Selected snippet is missing from expert bank (unexpected).")
    snippet_start_abs = int(np.asarray(bank.start_step[int(bi)]).reshape(()))
    snippet_end_abs = int(np.asarray(bank.end_step[int(bi)]).reshape(()))
    clip_id = str(np.asarray(bank.clip_id[int(bi)]).reshape(()))
    expert_model_path = str(np.asarray(bank.expert_model_path[int(bi)]).reshape(()))

    # Load the matched snippet's expert policy (SB3 PPO).
    print("[mocap_phys_eval] status: loading matched snippet expert policy")
    policy = load_expert_policy(expert_model_path, device=str(cfg.device))

    # Evaluation window is inside the snippet; we always start the env at snippet start.
    warmup_steps = int(max(0, int(selected.start_step)))
    eval_start_abs = int(snippet_start_abs + int(selected.start_step))
    selected_ref = run_reference_stability_check(
        policy=policy,
        clip_id=str(clip_id),
        start_step=int(eval_start_abs),
        end_step=int(snippet_end_abs),
        primary_steps=int(L),
        warmup_steps=int(warmup_steps),
        deterministic_policy=True,
        seed=0,
    )
    ref_checks = [
        {
            "snippet_id": selected.snippet_id,
            "clip_id": clip_id,
            "snippet_start_step": int(snippet_start_abs),
            "snippet_end_step": int(snippet_end_abs),
            "window_start_step_in_snippet": int(selected.start_step),
            "window_start_step_abs": int(eval_start_abs),
            "rmse_thigh_deg": float(selected.rmse_thigh_deg),
            "rmse_knee_deg": float(selected.rmse_knee_deg),
            "score": float(selected.score),
            "ref_fall_step": int(selected_ref.fall_step),
            "ref_risk": float(selected_ref.predicted_fall_risk),
            "expert_model_path": expert_model_path,
        }
    ]

    # Persist the *chosen* query window (no aggregation; exactly one TST window).
    np.savez_compressed(
        run_dir / "query_window.npz",
        thigh_pitch_deg=np.asarray(thigh_win_200, dtype=np.float32),
        thigh_quat_wxyz=np.asarray(thigh_quat_win_200, dtype=np.float32),
        knee_flex_deg=np.asarray(knee_win_200, dtype=np.float32),
        sample_hz=np.asarray(float(cfg.window_hz), dtype=np.float32),
        source_bvh=str(bvh_path),
        source_bvh_url=str(query_url),
        demo_idx=np.asarray(int(demo_idx), dtype=np.int64),
        start_idx=np.asarray(int(start), dtype=np.int64),
        start_s=np.asarray(float(start) / float(cfg.window_hz), dtype=np.float32),
    )

    # 8) Targets: GOOD uses ground-truth knee as placeholder prediction; BAD is a smooth perturbed knee.
    knee_good = knee_win.astype(np.float32)
    knee_bad = _make_bad_knee_prediction(
        knee_good,
        sample_hz=float(match_hz),
        target_rmse_deg=float(cfg.bad_knee_rmse_deg),
        lowpass_hz=float(cfg.bad_knee_lowpass_hz),
    )
    demo_bad_pred_rmse = float(np.sqrt(float(np.mean((knee_bad.astype(np.float64) - knee_good.astype(np.float64)) ** 2))))

    # Fetch reference series for the matched segment (snippet-local series).
    clip_end_step = int(snippet_end_abs)

    ref_hip = (
        np.asarray(bank.hip_deg[int(bi)], dtype=np.float32).reshape(-1)[selected.start_step : selected.start_step + L]
    )
    ref_th_pitch = (
        np.asarray(bank.thigh_pitch_deg[int(bi)], dtype=np.float32)
        .reshape(-1)[selected.start_step : selected.start_step + L]
    )
    ref_thq = (
        np.asarray(getattr(bank, "thigh_anat_quat_wxyz", bank.thigh_quat_wxyz)[int(bi)], dtype=np.float32)
        .reshape(-1, 4)[selected.start_step : selected.start_step + L]
    )
    ref_kn = (
        np.asarray(bank.knee_deg[int(bi)], dtype=np.float32).reshape(-1)[selected.start_step : selected.start_step + L]
    )

    # Alignment: map query signals into the reference coordinate system (constant sign + offset).
    th_pitch_al = (float(selected.thigh_sign) * thigh_win + float(selected.thigh_offset_deg)).astype(np.float32)
    kn_good_al = (float(selected.knee_sign) * knee_good + float(selected.knee_offset_deg)).astype(np.float32)
    kn_bad_al = (float(selected.knee_sign) * knee_bad + float(selected.knee_offset_deg)).astype(np.float32)
    thq_aligned = None
    try:
        qoff = quat_normalize_wxyz(np.asarray(selected.thigh_quat_offset_wxyz, dtype=np.float64).reshape(4))
        thq_al = quat_mul_wxyz(qoff[None, :], quat_normalize_wxyz(thigh_quat_win))
        thq_aligned = np.asarray(thq_al, dtype=np.float32)
        thigh_ori_err_deg = quat_geodesic_deg_wxyz(quat_normalize_wxyz(ref_thq), thq_al).astype(np.float32)
    except Exception:
        thq_aligned = None
        thigh_ori_err_deg = None

    # Override targets in joint space:
    # - Hip: adjust reference hip pitch by the delta required to match thigh *segment* pitch.
    # - Knee: use aligned knee flexion directly (joint coordinate).
    hip_target_deg = (ref_hip + (th_pitch_al - ref_th_pitch)).astype(np.float32)
    knee_target_good_deg = kn_good_al.astype(np.float32)
    knee_target_bad_deg = kn_bad_al.astype(np.float32)

    # Keep targets inside the CMU humanoid joint limits to avoid invalid reference features.
    hip_target_deg = np.clip(hip_target_deg, -160.0, 20.0).astype(np.float32)
    knee_target_good_deg = np.clip(knee_target_good_deg, 0.0, 170.0).astype(np.float32)
    knee_target_bad_deg = np.clip(knee_target_bad_deg, 0.0, 170.0).astype(np.float32)

    # Simulation override is applied in joint space (targets are already rfemurrx/rtibiarx degrees),
    # so no additional sign/offset is applied inside the simulator.
    override = OverrideSpec(
        thigh_actuator=str(cfg.thigh_actuator),
        knee_actuator=str(cfg.knee_actuator),
        thigh_sign=1.0,
        knee_sign=1.0,
        thigh_offset_deg=0.0,
        knee_offset_deg=0.0,
    )

    # 9) Record compare rollout (REF | GOOD | BAD).
    print("[mocap_phys_eval] status: running MuJoCo simulation (REF | GOOD | BAD) + recording replay")
    replay_npz = replay_dir / "compare.npz"
    rec_paths = record_compare_rollout(
        out_npz_path=replay_npz,
        clip_id=str(clip_id),
        start_step=int(eval_start_abs),
        end_step=int(clip_end_step),
        primary_steps=int(L),
        warmup_steps=int(warmup_steps),
        policy=policy,
        override=override,
        thigh_query_deg=hip_target_deg,
        knee_good_query_deg=knee_target_good_deg,
        knee_bad_query_deg=knee_target_bad_deg,
        width=int(cfg.render_width),
        height=int(cfg.render_height),
        camera_id=int(cfg.render_camera_id),
        deterministic_policy=True,
        seed=0,
    )

    # Update "latest" pointers.
    latest_npz = out_root / "latest_compare.npz"
    latest_gif = out_root / "latest_compare.gif"
    try:
        shutil.copyfile(rec_paths.npz_path, latest_npz)
    except Exception:
        pass
    try:
        if rec_paths.gif_path.exists():
            shutil.copyfile(rec_paths.gif_path, latest_gif)
    except Exception:
        pass

    # 10) Plots.
    print("[mocap_phys_eval] status: writing plots + summary")
    # ref_hip / ref_th_pitch / ref_kn and aligned query (th_pitch_al / kn_*_al) were computed earlier.

    mm_plot = plot_motion_match(
        out_path=plots_dir / "motion_match.png",
        sample_hz=float(match_hz),
        ref_thigh_deg=ref_th_pitch,
        ref_knee_deg=ref_kn,
        query_thigh_aligned_deg=th_pitch_al,
        query_knee_aligned_deg=kn_good_al,
        rmse_thigh_deg=float(selected.rmse_thigh_deg),
        rmse_knee_deg=float(selected.rmse_knee_deg),
        thigh_ori_err_deg=(None if thigh_ori_err_deg is None else np.asarray(thigh_ori_err_deg, dtype=np.float32)),
        title=f"Motion Match  snippet={selected.snippet_id}  clip={clip_id}  start_in_snip={selected.start_step}  L={L}",
    )
    latest_mm = out_root / "latest_motion_match.png"
    try:
        shutil.copyfile(mm_plot, latest_mm)
    except Exception:
        pass

    if thigh_ori_err_deg is not None and thq_aligned is not None:
        quat_plot = plot_thigh_quat_match(
            out_path=plots_dir / "thigh_quat_match.png",
            sample_hz=float(match_hz),
            ref_thigh_quat_wxyz=np.asarray(ref_thq, dtype=np.float32),
            query_thigh_quat_aligned_wxyz=np.asarray(thq_aligned, dtype=np.float32),
            thigh_ori_err_deg=np.asarray(thigh_ori_err_deg, dtype=np.float32),
            title="Thigh Quaternion Match (ref vs query aligned)",
        )
        latest_q = out_root / "latest_thigh_quat_match.png"
        try:
            shutil.copyfile(quat_plot, latest_q)
        except Exception:
            pass

    # Load sim traces from the recording for plotting.
    rr = np.load(rec_paths.npz_path, allow_pickle=True)
    # Simulation outcomes (from the recorded compare rollout).
    fall_step_ref = int(np.asarray(rr["fall_step_ref"]).reshape(()))
    fall_step_good = int(np.asarray(rr["fall_step_good"]).reshape(()))
    fall_step_bad = int(np.asarray(rr["fall_step_bad"]).reshape(()))
    risk_ref = float(np.asarray(rr["predicted_fall_risk_ref"]).reshape(()))
    risk_good = float(np.asarray(rr["predicted_fall_risk_good"]).reshape(()))
    risk_bad = float(np.asarray(rr["predicted_fall_risk_bad"]).reshape(()))

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        n = int(min(a.size, b.size))
        if n < 1:
            return float("nan")
        return float(np.sqrt(float(np.mean((a[:n] - b[:n]) ** 2))))

    def _mean_abs(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        if a.size < 1:
            return float("nan")
        a = a[np.isfinite(a)]
        if a.size < 1:
            return float("nan")
        return float(np.mean(np.abs(a)))

    def _mean_abs_delta(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        n = int(min(a.size, b.size))
        if n < 1:
            return float("nan")
        d = a[:n] - b[:n]
        d = d[np.isfinite(d)]
        if d.size < 1:
            return float("nan")
        return float(np.mean(np.abs(d)))

    # Control-override diagnostics (to verify RL cannot command the prosthetic joints).
    ctrl_override = None
    try:
        ctrl_override = {
            "good": {
                "mean_abs_policy_minus_applied_thigh_ctrl": float(
                    _mean_abs_delta(rr["ctrl_thigh_good_policy"], rr["ctrl_thigh_good_applied"])
                ),
                "mean_abs_policy_minus_applied_knee_ctrl": float(
                    _mean_abs_delta(rr["ctrl_knee_good_policy"], rr["ctrl_knee_good_applied"])
                ),
                "mean_abs_applied_minus_target_thigh_ctrl": float(
                    _mean_abs_delta(rr["ctrl_thigh_good_applied"], rr["ctrl_thigh_target"])
                ),
                "mean_abs_applied_minus_target_knee_ctrl": float(
                    _mean_abs_delta(rr["ctrl_knee_good_applied"], rr["ctrl_knee_good_target"])
                ),
            },
            "bad": {
                "mean_abs_policy_minus_applied_thigh_ctrl": float(
                    _mean_abs_delta(rr["ctrl_thigh_bad_policy"], rr["ctrl_thigh_bad_applied"])
                ),
                "mean_abs_policy_minus_applied_knee_ctrl": float(
                    _mean_abs_delta(rr["ctrl_knee_bad_policy"], rr["ctrl_knee_bad_applied"])
                ),
                "mean_abs_applied_minus_target_thigh_ctrl": float(
                    _mean_abs_delta(rr["ctrl_thigh_bad_applied"], rr["ctrl_thigh_target"])
                ),
                "mean_abs_applied_minus_target_knee_ctrl": float(
                    _mean_abs_delta(rr["ctrl_knee_bad_applied"], rr["ctrl_knee_bad_target"])
                ),
            },
        }
    except Exception:
        ctrl_override = None

    ref_actual_th = np.asarray(rr["thigh_ref_actual_deg"], dtype=np.float32)
    ref_actual_kn = np.asarray(rr["knee_ref_actual_deg"], dtype=np.float32)
    good_actual_th = np.asarray(rr["thigh_good_actual_deg"], dtype=np.float32)
    good_actual_kn = np.asarray(rr["knee_good_actual_deg"], dtype=np.float32)
    bad_actual_th = np.asarray(rr["thigh_bad_actual_deg"], dtype=np.float32)
    bad_actual_kn = np.asarray(rr["knee_bad_actual_deg"], dtype=np.float32)

    sim_metrics = {
        "ref": {
            "fall_step": int(fall_step_ref),
            "predicted_fall_risk": float(risk_ref),
            "rmse_thigh_deg": float(_rmse(ref_hip, ref_actual_th)),
            "rmse_knee_deg": float(_rmse(ref_kn, ref_actual_kn)),
        },
        "good": {
            "fall_step": int(fall_step_good),
            "predicted_fall_risk": float(risk_good),
            "rmse_thigh_deg": float(_rmse(hip_target_deg, good_actual_th)),
            "rmse_knee_deg": float(_rmse(knee_target_good_deg, good_actual_kn)),
        },
        "bad": {
            "fall_step": int(fall_step_bad),
            "predicted_fall_risk": float(risk_bad),
            "rmse_thigh_deg": float(_rmse(hip_target_deg, bad_actual_th)),
            "rmse_knee_deg": float(_rmse(knee_target_bad_deg, bad_actual_kn)),
        },
    }
    plot_simulation_angles(
        out_path=plots_dir / "simulation_angles.png",
        sample_hz=float(match_hz),
        ref_target_thigh_deg=ref_hip,
        ref_target_knee_deg=ref_kn,
        good_target_thigh_deg=hip_target_deg,
        good_target_knee_deg=knee_target_good_deg,
        bad_target_thigh_deg=hip_target_deg,
        bad_target_knee_deg=knee_target_bad_deg,
        ref_actual_thigh_deg=np.asarray(rr["thigh_ref_actual_deg"], dtype=np.float32),
        ref_actual_knee_deg=np.asarray(rr["knee_ref_actual_deg"], dtype=np.float32),
        good_actual_thigh_deg=np.asarray(rr["thigh_good_actual_deg"], dtype=np.float32),
        good_actual_knee_deg=np.asarray(rr["knee_good_actual_deg"], dtype=np.float32),
        bad_actual_thigh_deg=np.asarray(rr["thigh_bad_actual_deg"], dtype=np.float32),
        bad_actual_knee_deg=np.asarray(rr["knee_bad_actual_deg"], dtype=np.float32),
        title="Simulation Joint Angles (targets vs actuals)",
    )

    plot_balance_traces(
        out_path=plots_dir / "simulation_balance.png",
        sample_hz=float(match_hz),
        com_margin_ref_m=np.asarray(rr["balance_margin_ref_m"], dtype=np.float32),
        com_margin_good_m=np.asarray(rr["balance_margin_good_m"], dtype=np.float32),
        com_margin_bad_m=np.asarray(rr["balance_margin_bad_m"], dtype=np.float32),
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

    # 11) Summary JSON.
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "query": {
            "bvh_url": str(query_url),
            "bvh_path": str(bvh_path),
            "bvh_sample_hz": float(bvh_hz),
            "tst_window_hz": float(cfg.window_hz),
            "tst_window_n": int(cfg.window_n),
            "demo_idx": int(demo_idx),
            "window_start_idx": int(start),
            "window_start_s": float(start) / float(cfg.window_hz),
        },
        "reference_bank": {
            "kind": "mocapact_expert_snippets",
            "bank_path": str(bank_path),
            "n_clips": int(len(bank)),
            "match_hz": float(match_hz),
        },
        "match": {
            "feature_mode": str(cfg.match_feature_mode),
            "top_k": int(cfg.match_top_k),
            "candidates": [c.__dict__ for c in selected_candidates],
            "ref_stability_checks": ref_checks,
            "selected": selected.__dict__,
        },
        "override": {
            "thigh_actuator": override.thigh_actuator,
            "knee_actuator": override.knee_actuator,
            "thigh_sign": override.thigh_sign,
            "knee_sign": override.knee_sign,
            "thigh_offset_deg": override.thigh_offset_deg,
            "knee_offset_deg": override.knee_offset_deg,
        },
        "reference_stability_selected": (selected_ref.__dict__ if selected_ref is not None else None),
        "control_override": ctrl_override,
        "demo": {
            "bad_prediction_rmse_deg": float(demo_bad_pred_rmse),
            "bad_prediction_target_rmse_deg": float(cfg.bad_knee_rmse_deg),
        },
        "simulation": sim_metrics,
        "artifacts": {
            "compare_npz": str(rec_paths.npz_path),
            "compare_gif": str(rec_paths.gif_path),
            "latest_compare_npz": str(latest_npz),
            "latest_compare_gif": str(latest_gif),
            "plots_dir": str(plots_dir),
        },
    }
    write_json(run_dir / "summary.json", summary)

    # Minimal console summary (so it's obvious what happened when run from a terminal).
    print(f"[mocap_phys_eval] run_id={run_id}")
    print(
        f"[mocap_phys_eval] query: demo_idx={int(demo_idx)}  start_s={float(start) / float(cfg.window_hz):.2f}  bvh={query_url}"
    )
    print(
        f"[mocap_phys_eval] match: snippet={selected.snippet_id}  clip={clip_id}  "
        f"start_in_snip={int(selected.start_step)} (abs={int(eval_start_abs)})  L={L}  "
        f"rms_thigh_ori={selected.rmse_thigh_deg:.2f}deg rmse_knee={selected.rmse_knee_deg:.2f}deg"
    )
    if selected_ref is not None:
        try:
            print(
                f"[mocap_phys_eval] REF stability: fall_step={selected_ref.fall_step} risk={selected_ref.predicted_fall_risk:.2f}"
            )
        except Exception:
            pass
    print(f"[mocap_phys_eval] wrote: {run_dir / 'summary.json'}")
    try:
        print(
            f"[mocap_phys_eval] sim: REF fall={sim_metrics['ref']['fall_step']} risk={sim_metrics['ref']['predicted_fall_risk']:.2f}  "
            f"GOOD fall={sim_metrics['good']['fall_step']} risk={sim_metrics['good']['predicted_fall_risk']:.2f}  "
            f"BAD fall={sim_metrics['bad']['fall_step']} risk={sim_metrics['bad']['predicted_fall_risk']:.2f}"
        )
    except Exception:
        pass
    print(f"[mocap_phys_eval] replay: {latest_npz}")
    print(f"[mocap_phys_eval] gif:    {latest_gif}")
    print(f"[mocap_phys_eval] match:  {out_root / 'latest_motion_match.png'}")

    # 12) Launch viewer in a separate process so the window survives pipeline exit.
    try:
        subprocess.Popen([sys.executable, "-m", "mocap_phys_eval.replay", str(rec_paths.npz_path)])
    except Exception:
        pass


if __name__ == "__main__":
    main()
