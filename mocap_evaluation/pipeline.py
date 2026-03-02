from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

from .cmu_mocap import resolve_default_cmu_h5_path
from .motion_matching import match_batch_to_snippets
from .query_data import load_opensim_csv, load_rigtest_npy, make_contiguous_batches
from .simulation import (
    OverrideConfig,
    load_distillation_policy,
    run_mocapact_multiclip_simulation,
    visualize_mocapact_multiclip,
)
from .snippet_index import build_cmu_clip_index, write_index_manifest, SnippetIndex

DEFAULT_QUERY_URL = "https://addbiomechanics.org/assets/examples/Rajagopal2015SampleData.zip"


def _resample_linear(x: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) < 2 or abs(float(src_hz) - float(dst_hz)) < 1e-6:
        return x
    t_src = np.arange(len(x), dtype=np.float64) / float(src_hz)
    n_dst = max(2, int(round(t_src[-1] * float(dst_hz))) + 1)
    t_dst = np.arange(n_dst, dtype=np.float64) / float(dst_hz)
    return np.interp(t_dst, t_src, x).astype(np.float32)


def _apply_affine_deg(x: np.ndarray, *, sign: float, offset_deg: float) -> np.ndarray:
    """Apply the same (offset, sign) convention used by the sim override, but in degrees."""
    return (np.asarray(x, dtype=np.float32) + float(offset_deg)) * float(sign)


def _csv_has_time_column(path: Path) -> bool:
    try:
        header = path.read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return False
    cols = [c.strip().lower() for c in header.split(",") if c.strip()]
    return any(c in {"time_s", "time", "t"} for c in cols)


def _download_csv(url: str, dest: Path, *, force: bool = False) -> Path:
    dest = dest.expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix == ".zip":
        # If we already converted this zip to a CSV that includes time, reuse it.
        if dest.exists() and not force and _csv_has_time_column(dest):
            return dest
        zip_path = dest.with_suffix(dest.suffix + ".zip")
        _download_file(url, zip_path, force=force)
        csv_path, _sample_hz = _convert_zip_to_csv(zip_path, dest)
        return csv_path

    if dest.exists() and not force:
        return dest
    _download_file(url, dest, force=force)
    return dest


def _download_file(url: str, dest: Path, *, force: bool) -> Path:
    dest = dest.expanduser()
    if dest.exists() and not force:
        return dest
    try:
        req = Request(url, headers={"User-Agent": "MoCapAct/Downloader"})
        with urlopen(req) as resp:
            dest.write_bytes(resp.read())
    except URLError as exc:
        raise RuntimeError(f"Failed to download query data from {url}") from exc
    return dest


def _convert_zip_to_csv(zip_path: Path, dest: Path) -> tuple[Path, float]:
    with zipfile.ZipFile(zip_path) as z:
        trc_candidates = [name for name in z.namelist() if name.lower().endswith(".trc")]
        if not trc_candidates:
            raise RuntimeError("No TRC file found inside the downloaded zip.")
        with z.open(trc_candidates[0]) as trc_fp:
            time_s, thigh, knee, sample_hz = _trc_to_angles(trc_fp)
    data = np.stack([time_s, thigh, knee], axis=1)
    np.savetxt(dest, data, delimiter=",", header="time_s,thigh_angle,knee_angle", comments="", fmt="%.6f")
    return dest, float(sample_hz)


def _trc_to_angles(trc_stream: "zipfile.ZipExtFile") -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    reader = io.TextIOWrapper(trc_stream, encoding="utf-8", errors="ignore")
    lines = [ln.rstrip("\r\n") for ln in reader if ln.strip()]

    # Try to parse the TRC DataRate. This is a de-facto standard header in OpenSim TRC files.
    sample_hz: float | None = None
    for i, ln in enumerate(lines[:20]):
        parts = ln.split()
        if parts and parts[0].lower() == "datarate" and i + 1 < len(lines):
            try:
                sample_hz = float(lines[i + 1].split()[0])
            except Exception:
                sample_hz = None
            break

    try:
        header_idx = next(i for i, ln in enumerate(lines) if ln.lower().startswith("frame#"))
    except StopIteration as exc:
        raise RuntimeError("TRC column header not found (missing 'Frame#').") from exc

    names_line = lines[header_idx]
    marker_names = names_line.split()[2:]
    if not marker_names:
        raise RuntimeError("No marker names found in TRC header.")
    n_markers = len(marker_names)
    expected_cols = 2 + 3 * n_markers

    data_lines = lines[header_idx + 2 :]
    if not data_lines:
        raise RuntimeError("TRC file contained no data rows.")

    data_rows: list[list[float]] = []
    for ln in data_lines:
        parts = ln.split()
        if len(parts) < expected_cols:
            # Skip malformed / truncated rows.
            continue
        try:
            row = [float(x) for x in parts[:expected_cols]]
        except ValueError:
            continue
        data_rows.append(row)

    if not data_rows:
        raise RuntimeError("No numeric rows parsed from TRC file.")

    arr = np.asarray(data_rows, dtype=np.float64)
    time_s = np.asarray(arr[:, 1], dtype=np.float64).reshape(-1)

    def _marker(name: str) -> np.ndarray:
        if name not in marker_names:
            raise RuntimeError(f"Required marker '{name}' not found in TRC.")
        idx = marker_names.index(name)
        start = 2 + idx * 3
        return arr[:, start : start + 3]

    hip_positions = 0.5 * (_marker("L.ASIS") + _marker("L.PSIS"))
    knee_positions = _marker("L.Knee")
    ankle_positions = _marker("L.Ankle")
    thigh_vec = hip_positions - knee_positions
    shank_vec = ankle_positions - knee_positions
    throat = np.linalg.norm(thigh_vec, axis=1, keepdims=True)
    shin = np.linalg.norm(shank_vec, axis=1, keepdims=True)
    thigh_unit = thigh_vec / np.clip(throat, 1e-6, np.inf)
    shank_unit = shank_vec / np.clip(shin, 1e-6, np.inf)
    knee_dot = np.sum(thigh_unit * shank_unit, axis=1)
    knee_angle = np.degrees(np.arccos(np.clip(knee_dot, -1.0, 1.0)))
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # Y is up in the sample TRC
    thigh_dot = np.sum(thigh_unit * vertical, axis=1)
    thigh_angle = np.degrees(np.arccos(np.clip(thigh_dot, -1.0, 1.0)))

    if sample_hz is None:
        # Fall back to the median delta between successive Time values if present.
        try:
            t = arr[:, 1].astype(np.float64)
            dt = np.diff(t)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size:
                sample_hz = float(1.0 / float(np.median(dt)))
        except Exception:
            sample_hz = None

    return (
        time_s.astype(np.float32),
        thigh_angle.astype(np.float32),
        knee_angle.astype(np.float32),
        float(sample_hz or 200.0),
    )

def main() -> None:
    ap = argparse.ArgumentParser(
        description="MoCapAct physical evaluation pipeline: motion-match -> run MoCapAct sim w/ thigh+knee overrides"
    )

    ap.add_argument(
        "--input",
        type=str,
        default="",
        help="Optional query CSV with columns thigh_angle,knee_angle (and optional time_s). If omitted, downloads the default OpenSim TRC sample and converts it.",
    )
    ap.add_argument("--out", type=str, default="artifacts/mocapact_eval.json", help="Where to write the JSON summary.")
    args = ap.parse_args()

    # Single "right" default configuration (multi-clip policy + CMU2020 reference trajectories).
    multiclip_ckpt = Path("mocapact_models/multiclip_policy/full_dataset/model/model.ckpt")
    device = "cpu"
    side = "left"

    match_features = "thigh"
    top_k = 3
    dtw_band = 40

    batch_size_default = 400
    stride_default = 200

    override = OverrideConfig(
        thigh_actuator=("walker/lfemurrx" if side == "left" else "walker/rfemurrx"),
        knee_actuator=("walker/ltibiarx" if side == "left" else "walker/rtibiarx"),
        thigh_sign=1.0,
        knee_sign=1.0,
        thigh_offset_deg=0.0,
        knee_offset_deg=0.0,
    )

    no_viewer = os.environ.get("MOCAP_EVAL_NO_VIEWER", "").strip().lower() in {"1", "true", "yes"}

    out_path = Path(args.out)
    index_path = Path("artifacts") / f"cmu_clip_index_{side}.npz"

    cmu_h5 = resolve_default_cmu_h5_path()

    input_path = Path(str(args.input).strip()) if str(args.input).strip() else None
    query_url_used: str | None = None
    if input_path is None:
        query_url_used = DEFAULT_QUERY_URL
        input_path = _download_csv(
            query_url_used,
            Path("artifacts/opensim_query.csv"),
            force=False,
        )

    if str(input_path).lower().endswith(".npy"):
        thigh, knee_included_deg = load_rigtest_npy(input_path)
        query_hz = 200.0
    else:
        thigh, knee_included_deg, query_hz = load_opensim_csv(input_path)
    # Your rigtest convention: included angle (0 = fully flexed, 180 = straight).
    # MoCapAct joint coordinate convention for *tibia rx* is knee flexion (0 = straight, ~170 = max flex).
    knee_pred = (180.0 - np.asarray(knee_included_deg, dtype=np.float32)).astype(np.float32)
    resolved_input = input_path.resolve()

    n_query = int(min(len(thigh), len(knee_pred)))
    if n_query < 2:
        raise RuntimeError(f"Query trajectory too short ({n_query} samples).")

    batch_size = int(min(int(batch_size_default), n_query))
    stride = int(min(int(stride_default), batch_size))

    batches = make_contiguous_batches(
        thigh,
        knee_pred,
        batch_size=batch_size,
        stride=stride,
        sample_hz=float(query_hz),
    )
    if not batches:
        raise RuntimeError("No batches generated. Query trajectory is too short for the configured windowing.")

    # Index: build or load (always CMU reference trajectories as used by MoCapAct).
    if not index_path.exists():
        idx = build_cmu_clip_index(
            cmu_h5_path=cmu_h5,
            side=str(side),
            limit=None,
        )
        idx.save_npz(index_path)
        write_index_manifest(index_path, index=idx, index_kind="cmu_clips", expert_map=None)
    else:
        idx = SnippetIndex.load_npz(index_path)

    snippets = list(idx.iter_snippets())
    if not snippets:
        raise RuntimeError("Snippet index is empty.")

    # Use snippet reference rate as the sim/control rate.
    sim_hz = float(np.median(idx.sample_hz.astype(np.float64)))

    multiclip_policy = load_distillation_policy(str(multiclip_ckpt), device=str(device))

    summary: dict = {
        "cmu_h5": str(cmu_h5),
        "policy": "multiclip",
        "reference_bank": "mocapact_cmu2020",
        "multiclip_ckpt": str(multiclip_ckpt),
        "device": str(device),
        "index": str(index_path),
        "query": {
            "input": str(resolved_input),
            "query_url": query_url_used,
            "query_hz": float(query_hz),
            "batch_size_default": int(batch_size_default),
            "stride_default": int(stride_default),
            "batch_size_used": int(batch_size),
            "stride_used": int(stride),
            "knee_input_convention": "included_deg_0flexed_180straight",
            "knee_sim_convention": "flexion_deg_0straight",
            "knee_converted_to_flexion": True,
        },
        "match": {"features": match_features, "top_k": int(top_k), "dtw_band": int(dtw_band)},
        "override": {
            "side": side,
            "thigh_actuator": override.thigh_actuator,
            "knee_actuator": override.knee_actuator,
            "thigh_sign": override.thigh_sign,
            "knee_sign": override.knee_sign,
            "thigh_offset_deg": override.thigh_offset_deg,
            "knee_offset_deg": override.knee_offset_deg,
        },
        "n_snippets_indexed": len(snippets),
        "sim_hz": sim_hz,
        "n_batches": len(batches),
        "results": [],
    }

    visualized = False
    for b in batches:
        # Apply the same angle convention used by the override so motion-matching and sim "agree"
        # on what a thigh/knee angle means.
        thigh_for_match = _apply_affine_deg(
            b.thigh_angle_deg, sign=float(override.thigh_sign), offset_deg=float(override.thigh_offset_deg)
        )
        knee_for_match = _apply_affine_deg(
            b.knee_angle_pred_deg, sign=float(override.knee_sign), offset_deg=float(override.knee_offset_deg)
        )

        matches = match_batch_to_snippets(
            thigh_for_match,
            b.sample_hz,
            snippets,
            query_knee_pred=(knee_for_match if match_features == "thigh_knee" else None),
            feature_mode=str(match_features),
            top_k=int(top_k),
            dtw_band=int(dtw_band),
        )
        if not matches:
            summary["results"].append({"batch_id": b.batch_id, "error": "no_match"})
            continue

        best = matches[0]
        from .mocapact_dataset import parse_snippet_id

        clip_id, start_step, end_step = parse_snippet_id(best.snippet_id)

        # Resample overrides to control rate, and clamp to remaining clip length from the match start.
        thigh_sim = _resample_linear(b.thigh_angle_deg, b.sample_hz, sim_hz)
        knee_sim = _resample_linear(b.knee_angle_pred_deg, b.sample_hz, sim_hz)
        snippet_len = int(end_step - start_step + 1)
        match_start_step = int(start_step + best.start_idx)
        remaining = max(0, snippet_len - int(best.start_idx))
        max_steps = int(min(len(thigh_sim), len(knee_sim), remaining))
        if max_steps < 1:
            summary["results"].append({"batch_id": b.batch_id, "error": "match_too_close_to_clip_end"})
            continue

        thigh_sim = thigh_sim[:max_steps]
        knee_sim = knee_sim[:max_steps]
        sim_start_step = match_start_step
        sim_end_step = int(sim_start_step + max_steps - 1)

        if not no_viewer and not visualized:
            try:
                visualize_mocapact_multiclip(
                    clip_id=clip_id,
                    start_step=sim_start_step,
                    end_step=sim_end_step,
                    policy=multiclip_policy,
                    thigh_angle_deg=thigh_sim,
                    knee_angle_deg=knee_sim,
                    override=override,
                    deterministic_policy=True,
                    warmup_steps=0,
                )
            except Exception as e:
                print(f"Visualization failed (continuing headless): {e}")
            visualized = True

        sim_res = run_mocapact_multiclip_simulation(
            snippet_id=best.snippet_id,
            clip_id=clip_id,
            start_step=int(sim_start_step),
            end_step=int(sim_end_step),
            policy=multiclip_policy,
            thigh_angle_deg=thigh_sim,
            knee_angle_deg=knee_sim,
            override=override,
            deterministic_policy=True,
            warmup_steps=0,
            max_steps=max_steps,
        )

        summary["results"].append(
            {
                "batch_id": b.batch_id,
                "best_match": {
                    "snippet_id": best.snippet_id,
                    "clip_id": best.clip_id,
                    "start_step": int(start_step),
                    "end_step": int(end_step),
                    "score": best.score,
                    "start_idx": best.start_idx,
                    "end_idx": best.end_idx,
                    "sim_start_step": int(sim_start_step),
                    "sim_end_step": int(sim_end_step),
                },
                "simulation": {
                    "n_steps_total": sim_res.n_steps_total,
                    "n_steps_overridden": sim_res.n_steps_overridden,
                    "terminated_early": sim_res.terminated_early,
                    "total_reward": sim_res.total_reward,
                    "thigh_rmse_deg": sim_res.thigh_rmse_deg,
                    "knee_rmse_deg": sim_res.knee_rmse_deg,
                },
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
