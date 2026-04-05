from __future__ import annotations

import functools
import re
import struct
import urllib.request
import zlib
from pathlib import Path

import numpy as np

from mocap_phys_eval.utils import rotmat_to_quat_wxyz


GT_PROCESSED_URL = (
    "https://repository.gatech.edu/server/api/core/bitstreams/"
    "03f9679f-28ce-4d8b-b195-4b3b1aa4adc9/content"
)
GT_RAW_URL = (
    "https://repository.gatech.edu/server/api/core/bitstreams/"
    "0c050d51-54f0-4ccf-9bcd-65521c0e20c3/content"
)
GT_CENTRAL_DIR_OFFSET = 13376804602
GT_CENTRAL_DIR_SIZE = 1481632

GT_DEFAULT_EMG_COLS = ("RRF", "RBF", "RVL", "RMGAS")
GT_DEFAULT_THIGH_IMU_COLS = (
    "RAThigh_ACCX",
    "RAThigh_ACCY",
    "RAThigh_ACCZ",
    "RAThigh_GYROX",
    "RAThigh_GYROY",
    "RAThigh_GYROZ",
)

GT_SMOKE_TRIALS: tuple[tuple[str, str], ...] = (
    ("gt_data0.npy", "AB01/normal_walk_1_1-2/AB01_normal_walk_1_1-2"),
    ("gt_data1.npy", "AB02/normal_walk_1_1-2/AB02_normal_walk_1_1-2"),
    ("gt_data2.npy", "AB01/stairs_1_1_up/AB01_stairs_1_1_up"),
    ("gt_data3.npy", "AB01/stairs_1_2_down/AB01_stairs_1_2_down"),
    ("gt_data4.npy", "AB02/stairs_1_1_up/AB02_stairs_1_1_up"),
    ("gt_data5.npy", "AB02/stairs_1_2_down/AB02_stairs_1_2_down"),
)

# Empirically best simple hip-angle proxy found so far for GT -> MoCapAct matching.
# Interpret OpenSim hip angles as intrinsic rotations about:
#   x = flexion/extension
#   y = adduction/abduction
#   z = internal/external rotation
# then compose in zyx order with a sign convention that best matches the bank's
# world-frame anatomical thigh orientation.
GT_THIGH_QUAT_ORDER = "zyx"
GT_THIGH_QUAT_SIGNS = (-1.0, 1.0, -1.0)


def list_processed_trial_prefixes() -> list[str]:
    entries = _central_directory_entries()
    prefixes: set[str] = set()
    for name in entries:
        if not name.endswith("_angle.csv"):
            continue
        p = Path(name)
        stem = p.stem
        if stem.endswith("_angle"):
            prefixes.add(str((p.parent / stem[: -len("_angle")]).as_posix()))
    return sorted(prefixes)


def list_gt_normal_walk_prefixes(*, numeric_only: bool = True) -> list[str]:
    out: list[str] = []
    for prefix in list_processed_trial_prefixes():
        task = Path(prefix).parent.name
        if not task.startswith("normal_walk_1_"):
            continue
        if numeric_only and (task.endswith("shuffle") or task.endswith("skip")):
            continue
        out.append(prefix)
    return sorted(out)


def _range_get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={int(start)}-{int(end)}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def _range_get_suffix(url: str, suffix_bytes: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes=-{int(suffix_bytes)}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def _central_directory_entries(url: str = GT_PROCESSED_URL) -> dict[str, tuple[int, int, int]]:
    cd = _range_get(url, GT_CENTRAL_DIR_OFFSET, GT_CENTRAL_DIR_OFFSET + GT_CENTRAL_DIR_SIZE - 1)
    out: dict[str, tuple[int, int, int]] = {}
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        method = int(struct.unpack_from("<H", cd, pos + 10)[0])
        comp_size = int(struct.unpack_from("<I", cd, pos + 20)[0])
        uncomp_size = int(struct.unpack_from("<I", cd, pos + 24)[0])
        name_len = int(struct.unpack_from("<H", cd, pos + 28)[0])
        extra_len = int(struct.unpack_from("<H", cd, pos + 30)[0])
        comment_len = int(struct.unpack_from("<H", cd, pos + 32)[0])
        lho = int(struct.unpack_from("<I", cd, pos + 42)[0])
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8")
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        csize = int(comp_size)
        usize = int(uncomp_size)
        off = int(lho)
        if 0xFFFFFFFF in (comp_size, uncomp_size, lho):
            epos = 0
            while epos + 4 <= len(extra):
                hid, dlen = struct.unpack_from("<HH", extra, epos)
                payload = extra[epos + 4 : epos + 4 + dlen]
                if hid == 0x0001:
                    p = 0
                    if uncomp_size == 0xFFFFFFFF:
                        usize = int(struct.unpack_from("<Q", payload, p)[0])
                        p += 8
                    if comp_size == 0xFFFFFFFF:
                        csize = int(struct.unpack_from("<Q", payload, p)[0])
                        p += 8
                    if lho == 0xFFFFFFFF:
                        off = int(struct.unpack_from("<Q", payload, p)[0])
                        p += 8
                    break
                epos += 4 + dlen
        out[name] = (off, csize, method)
        pos += 46 + name_len + extra_len + comment_len
    return out


@functools.lru_cache(maxsize=4)
def _zip64_central_directory_entries_dynamic(url: str) -> dict[str, tuple[int, int, int]]:
    tail = _range_get_suffix(url, 131072)
    idx64 = tail.rfind(b"PK\x06\x07")
    if idx64 < 0:
        raise RuntimeError(f"ZIP64 locator not found for {url}")
    _, _, eocd64_off, _ = struct.unpack("<IIQI", tail[idx64 : idx64 + 20])
    eocd64 = _range_get(url, int(eocd64_off), int(eocd64_off) + 200)
    if eocd64[:4] != b"PK\x06\x06":
        raise RuntimeError(f"ZIP64 EOCD not found for {url}")
    vals = struct.unpack("<IQHHIIQQQQ", eocd64[:56])
    cd_size = int(vals[-2])
    cd_off = int(vals[-1])
    cd = _range_get(url, cd_off, cd_off + cd_size - 1)
    out: dict[str, tuple[int, int, int]] = {}
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        method = int(struct.unpack_from("<H", cd, pos + 10)[0])
        comp_size = int(struct.unpack_from("<I", cd, pos + 20)[0])
        uncomp_size = int(struct.unpack_from("<I", cd, pos + 24)[0])
        name_len = int(struct.unpack_from("<H", cd, pos + 28)[0])
        extra_len = int(struct.unpack_from("<H", cd, pos + 30)[0])
        comment_len = int(struct.unpack_from("<H", cd, pos + 32)[0])
        lho = int(struct.unpack_from("<I", cd, pos + 42)[0])
        name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8")
        extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        csize = int(comp_size)
        off = int(lho)
        if 0xFFFFFFFF in (comp_size, uncomp_size, lho):
            epos = 0
            while epos + 4 <= len(extra):
                hid, dlen = struct.unpack_from("<HH", extra, epos)
                payload = extra[epos + 4 : epos + 4 + dlen]
                if hid == 0x0001:
                    p = 0
                    if uncomp_size == 0xFFFFFFFF:
                        p += 8
                    if comp_size == 0xFFFFFFFF:
                        csize = int(struct.unpack_from("<Q", payload, p)[0])
                        p += 8
                    if lho == 0xFFFFFFFF:
                        off = int(struct.unpack_from("<Q", payload, p)[0])
                        p += 8
                    break
                epos += 4 + dlen
        out[name] = (off, csize, method)
        pos += 46 + name_len + extra_len + comment_len
    return out


def _extract_member(member: str, *, out_path: Path, url: str = GT_PROCESSED_URL) -> Path:
    entries = _central_directory_entries(url)
    return _extract_member_from_entries(member, out_path=out_path, url=url, entries=entries)


def _extract_member_from_entries(
    member: str,
    *,
    out_path: Path,
    url: str,
    entries: dict[str, tuple[int, int, int]],
) -> Path:
    if member not in entries:
        raise FileNotFoundError(f"ZIP archive is missing member {member!r}")
    off, csize, method = entries[member]
    lfh = _range_get(url, off, off + 29)
    if lfh[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local zip header for {member!r}")
    name_len = int(struct.unpack_from("<H", lfh, 26)[0])
    extra_len = int(struct.unpack_from("<H", lfh, 28)[0])
    data_start = off + 30 + name_len + extra_len
    comp = _range_get(url, data_start, data_start + csize - 1)
    if method == 0:
        raw = comp
    elif method == 8:
        raw = zlib.decompress(comp, -15)
    else:
        raise RuntimeError(f"Unsupported zip method {method} for {member!r}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    return out_path


def ensure_processed_csv(prefix: str, *, out_root: Path) -> tuple[Path, Path, Path]:
    out_root = Path(out_root)
    parts = (
        ("angle", out_root / f"{Path(prefix).name}_angle.csv"),
        ("emg", out_root / f"{Path(prefix).name}_emg.csv"),
        ("imu_real", out_root / f"{Path(prefix).name}_imu_real.csv"),
    )
    out_paths: dict[str, Path] = {}
    for suffix, path in parts:
        if not path.exists():
            member = f"{prefix}_{suffix}.csv"
            member = member.replace("\\", "/")
            _extract_member(member, out_path=path)
        out_paths[suffix] = path
    return out_paths["angle"], out_paths["emg"], out_paths["imu_real"]


def _processed_angle_csv_to_subject_and_raw_task(angle_csv: Path) -> tuple[str, str]:
    stem = Path(angle_csv).stem
    if stem.endswith("_angle"):
        stem = stem[: -len("_angle")]
    parts = stem.split("_", 1)
    if len(parts) != 2:
        raise RuntimeError(f"Could not parse GT processed angle stem {stem!r}")
    subject, task = parts
    raw_task = str(task)
    raw_task = re.sub(r"_(up|down)$", "", raw_task)
    raw_task = re.sub(r"_\d+-\d+$", "", raw_task)
    return subject, raw_task


def ensure_raw_trial_files_for_angle_csv(
    angle_csv: Path,
    *,
    out_root: Path = Path("gt_dataset") / "raw_full",
) -> tuple[Path, Path]:
    subject, raw_task = _processed_angle_csv_to_subject_and_raw_task(Path(angle_csv))
    out_root = Path(out_root)
    joint_csv = out_root / f"{subject}_{raw_task}_Joint_Angle.csv"
    trc_path = out_root / f"{subject}_{raw_task}.trc"
    entries = _zip64_central_directory_entries_dynamic(GT_RAW_URL)
    if not joint_csv.exists():
        member = f"{subject}/CSV_Data/{raw_task}/Joint_Angle.csv"
        _extract_member_from_entries(member, out_path=joint_csv, url=GT_RAW_URL, entries=entries)
    if not trc_path.exists():
        member = f"{subject}/MarkerData/{raw_task}.trc"
        _extract_member_from_entries(member, out_path=trc_path, url=GT_RAW_URL, entries=entries)
    return joint_csv, trc_path


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(Path(path), delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(header):
        raise RuntimeError(f"CSV shape/header mismatch for {str(path)!r}: {data.shape[1]} vs {len(header)}")
    return header, data


def _col_index(headers: list[str], name: str) -> int:
    try:
        return headers.index(name)
    except ValueError as e:
        raise KeyError(f"Required column {name!r} not found. Available={headers[:8]}...") from e


def _normalize_rows(v: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(vv, axis=1, keepdims=True)
    return vv / np.clip(n, float(eps), None)


def _interp_matrix_by_time(t_src: np.ndarray, x_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    ts = np.asarray(t_src, dtype=np.float64).reshape(-1)
    xs = np.asarray(x_src, dtype=np.float64)
    td = np.asarray(t_dst, dtype=np.float64).reshape(-1)
    out = np.empty((int(td.size), int(xs.shape[1])), dtype=np.float64)
    for j in range(int(xs.shape[1])):
        col = xs[:, j]
        good = np.isfinite(col)
        if np.count_nonzero(good) < 2:
            out[:, j] = 0.0
        else:
            out[:, j] = np.interp(td, ts[good], col[good])
    return out


def _read_trc_markers(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 6:
        raise RuntimeError(f"TRC file too short: {str(path)!r}")
    marker_names = [n for n in lines[3].split("\t")[2:] if str(n).strip()]
    rows: list[list[float]] = []
    for line in lines[5:]:
        if not str(line).strip():
            continue
        parts = str(line).split("\t")
        vals: list[float] = []
        for p in parts:
            p = str(p).strip()
            vals.append(float(p) if p else float("nan"))
        rows.append(vals)
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Bad TRC numeric payload for {str(path)!r}")
    t = arr[:, 1].astype(np.float64)
    coords = arr[:, 2:].astype(np.float64)
    markers: dict[str, np.ndarray] = {}
    for i, name in enumerate(marker_names):
        markers[str(name)] = coords[:, 3 * i : 3 * i + 3].astype(np.float64)
    return t, markers


def thigh_quat_from_markers(
    *,
    marker_time: np.ndarray,
    markers: dict[str, np.ndarray],
    query_time: np.ndarray,
) -> np.ndarray:
    required = ("RGTR", "RKNE", "RMKNE", "RTHL", "RTHR")
    for name in required:
        if name not in markers:
            raise KeyError(f"TRC is missing required marker {name!r}")

    hip = _interp_matrix_by_time(marker_time, markers["RGTR"], query_time)
    knee_lat = _interp_matrix_by_time(marker_time, markers["RKNE"], query_time)
    knee_med = _interp_matrix_by_time(marker_time, markers["RMKNE"], query_time)
    thigh_l = _interp_matrix_by_time(marker_time, markers["RTHL"], query_time)
    thigh_r = _interp_matrix_by_time(marker_time, markers["RTHR"], query_time)

    knee_mid = 0.5 * (knee_lat + knee_med)
    y_axis = _normalize_rows(hip - knee_mid)
    x_hint = _normalize_rows(thigh_r - thigh_l)
    z_axis = _normalize_rows(np.cross(x_hint, y_axis))
    x_axis = _normalize_rows(np.cross(y_axis, z_axis))
    R = np.stack([x_axis, y_axis, z_axis], axis=-1)
    q = np.asarray(rotmat_to_quat_wxyz(R), dtype=np.float64).reshape(-1, 4)
    return _quat_fix_sign_continuity_wxyz(q)


def load_marker_based_thigh_orientation(
    angle_csv: Path,
    *,
    raw_cache_dir: Path = Path("gt_dataset") / "raw_full",
) -> dict[str, np.ndarray]:
    angle_signals = load_processed_angle_signals(angle_csv)
    t_out = np.asarray(angle_signals["time"], dtype=np.float64).reshape(-1)
    _, trc_path = ensure_raw_trial_files_for_angle_csv(angle_csv, out_root=raw_cache_dir)
    t_raw, markers = _read_trc_markers(trc_path)
    q = thigh_quat_from_markers(marker_time=t_raw, markers=markers, query_time=t_out)
    return {
        "time": t_out.astype(np.float64),
        "thigh_quat_wxyz": q.astype(np.float32),
    }


def _quat_normalize_wxyz(q: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(qq, axis=-1, keepdims=True)
    n = np.clip(n, float(eps), None)
    return qq / n


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = aa[..., 0], aa[..., 1], aa[..., 2], aa[..., 3]
    bw, bx, by, bz = bb[..., 0], bb[..., 1], bb[..., 2], bb[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_fix_sign_continuity_wxyz(q: np.ndarray) -> np.ndarray:
    qq = _quat_normalize_wxyz(np.asarray(q, dtype=np.float64).reshape(-1, 4))
    for i in range(1, int(qq.shape[0])):
        if float(np.dot(qq[i - 1], qq[i])) < 0.0:
            qq[i] *= -1.0
    return qq.astype(np.float32)


def _axis_angle_quat_wxyz(axis: str, angle_rad: np.ndarray) -> np.ndarray:
    ang = np.asarray(angle_rad, dtype=np.float64).reshape(-1)
    half = 0.5 * ang
    c = np.cos(half)
    s = np.sin(half)
    z = np.zeros_like(c)
    if axis == "x":
        return np.stack([c, s, z, z], axis=1)
    if axis == "y":
        return np.stack([c, z, s, z], axis=1)
    if axis == "z":
        return np.stack([c, z, z, s], axis=1)
    raise ValueError(f"Unsupported axis {axis!r}")


def hip_angles_to_quat_wxyz(
    hip_flexion_deg: np.ndarray,
    hip_adduction_deg: np.ndarray,
    hip_rotation_deg: np.ndarray,
    *,
    order: str = GT_THIGH_QUAT_ORDER,
    signs: tuple[float, float, float] = GT_THIGH_QUAT_SIGNS,
) -> np.ndarray:
    flex = np.asarray(hip_flexion_deg, dtype=np.float64).reshape(-1) * float(signs[0])
    add = np.asarray(hip_adduction_deg, dtype=np.float64).reshape(-1) * float(signs[1])
    rot = np.asarray(hip_rotation_deg, dtype=np.float64).reshape(-1) * float(signs[2])
    n = int(min(flex.size, add.size, rot.size))
    if n < 1:
        return np.zeros((0, 4), dtype=np.float32)
    angle_map = {
        "x": np.deg2rad(flex[:n]),
        "y": np.deg2rad(add[:n]),
        "z": np.deg2rad(rot[:n]),
    }
    q = np.repeat(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), n, axis=0)
    for axis in str(order).lower():
        q_axis = _axis_angle_quat_wxyz(axis, angle_map[axis])
        q = _quat_mul_wxyz(q, q_axis)
    return _quat_fix_sign_continuity_wxyz(q)


def load_processed_angle_signals(angle_csv: Path) -> dict[str, np.ndarray]:
    headers, angle = _read_csv(angle_csv)
    t200 = angle[:, _col_index(headers, "time")].astype(np.float64)
    knee_angle_r = angle[:, _col_index(headers, "knee_angle_r")].astype(np.float32)
    hip_flexion_r = angle[:, _col_index(headers, "hip_flexion_r")].astype(np.float32)
    hip_adduction_r = angle[:, _col_index(headers, "hip_adduction_r")].astype(np.float32)
    hip_rotation_r = angle[:, _col_index(headers, "hip_rotation_r")].astype(np.float32)

    knee_flex_deg = np.clip(-knee_angle_r, 0.0, 180.0).astype(np.float32)
    knee_included_deg = (180.0 - knee_flex_deg).astype(np.float32)
    thigh_pitch_deg = hip_flexion_r.astype(np.float32)
    thigh_quat_wxyz = hip_angles_to_quat_wxyz(
        hip_flexion_r,
        hip_adduction_r,
        hip_rotation_r,
    )
    return {
        "time": t200.astype(np.float64),
        "knee_included_deg": knee_included_deg.astype(np.float32),
        "hip_flexion_deg": hip_flexion_r.astype(np.float32),
        "hip_adduction_deg": hip_adduction_r.astype(np.float32),
        "hip_rotation_deg": hip_rotation_r.astype(np.float32),
        "thigh_pitch_deg": thigh_pitch_deg.astype(np.float32),
        "thigh_quat_wxyz": thigh_quat_wxyz.astype(np.float32),
    }


def convert_processed_trial(
    *,
    angle_csv: Path,
    emg_csv: Path,
    imu_csv: Path,
    out_path: Path,
    emg_cols: tuple[str, ...] = GT_DEFAULT_EMG_COLS,
    thigh_imu_cols: tuple[str, ...] = GT_DEFAULT_THIGH_IMU_COLS,
) -> Path:
    angle_signals = load_processed_angle_signals(angle_csv)
    emg_headers, emg = _read_csv(emg_csv)
    imu_headers, imu = _read_csv(imu_csv)

    t200 = np.asarray(angle_signals["time"], dtype=np.float64)
    knee_included_deg = np.asarray(angle_signals["knee_included_deg"], dtype=np.float32)
    hip_flexion_r = np.asarray(angle_signals["hip_flexion_deg"], dtype=np.float32)
    hip_adduction_r = np.asarray(angle_signals["hip_adduction_deg"], dtype=np.float32)
    hip_rotation_r = np.asarray(angle_signals["hip_rotation_deg"], dtype=np.float32)
    thigh_pitch_deg = np.asarray(angle_signals["thigh_pitch_deg"], dtype=np.float32)
    thigh_quat_wxyz = np.asarray(angle_signals["thigh_quat_wxyz"], dtype=np.float32)

    t_raw = emg[:, _col_index(emg_headers, "time")].astype(np.float64)
    emg_idx = [_col_index(emg_headers, c) for c in emg_cols]
    raw_emg = emg[:, emg_idx].astype(np.float32).T

    imu_t = imu[:, _col_index(imu_headers, "time")].astype(np.float64)
    imu_idx = [_col_index(imu_headers, c) for c in thigh_imu_cols]
    thigh_imu = imu[:, imu_idx].astype(np.float32)

    rec = {
        "source_dataset": "georgia_tech_processed",
        "source_angle_csv": str(angle_csv),
        "source_emg_csv": str(emg_csv),
        "source_imu_csv": str(imu_csv),
        "timestamps": t200.astype(np.float64),
        "effective_hz": np.float32(200.0),
        "knee_included_deg": knee_included_deg.astype(np.float32),
        "thigh_pitch_deg": thigh_pitch_deg.astype(np.float32),
        "hip_flexion_deg": hip_flexion_r.astype(np.float32),
        "hip_adduction_deg": hip_adduction_r.astype(np.float32),
        "hip_rotation_deg": hip_rotation_r.astype(np.float32),
        "thigh_quat_wxyz": thigh_quat_wxyz.astype(np.float32),
        "raw_emg_channels": raw_emg.astype(np.float32),
        "raw_emg_times": t_raw.astype(np.float64),
        "emg_channel_names": np.asarray(list(emg_cols)),
        "thigh_imu": thigh_imu.astype(np.float32),
        "thigh_imu_times": imu_t.astype(np.float64),
        "thigh_imu_names": np.asarray(list(thigh_imu_cols)),
    }
    out_path = Path(out_path)
    np.save(out_path, rec, allow_pickle=True)
    return out_path


def ensure_smoke_recordings(
    *,
    out_dir: Path = Path("."),
    raw_cache_dir: Path = Path("gt_dataset") / "processed_sample",
) -> list[Path]:
    out_dir = Path(out_dir)
    raw_cache_dir = Path(raw_cache_dir)
    out_paths: list[Path] = []
    for file_name, prefix in GT_SMOKE_TRIALS:
        out_path = out_dir / file_name
        if out_path.exists():
            out_paths.append(out_path)
            continue
        angle_csv, emg_csv, imu_csv = ensure_processed_csv(prefix, out_root=raw_cache_dir)
        convert_processed_trial(
            angle_csv=angle_csv,
            emg_csv=emg_csv,
            imu_csv=imu_csv,
            out_path=out_path,
        )
        out_paths.append(out_path)
    return out_paths


def ensure_normal_walk_recordings(
    *,
    out_dir: Path = Path("."),
    raw_cache_dir: Path = Path("gt_dataset") / "processed_full",
    numeric_only: bool = True,
) -> list[Path]:
    out_dir = Path(out_dir)
    raw_cache_dir = Path(raw_cache_dir)
    prefixes = list_gt_normal_walk_prefixes(numeric_only=bool(numeric_only))
    out_paths: list[Path] = []
    for i, prefix in enumerate(prefixes):
        out_path = out_dir / f"gt_data{i:03d}.npy"
        if not out_path.exists():
            angle_csv, emg_csv, imu_csv = ensure_processed_csv(prefix, out_root=raw_cache_dir)
            convert_processed_trial(
                angle_csv=angle_csv,
                emg_csv=emg_csv,
                imu_csv=imu_csv,
                out_path=out_path,
            )
        out_paths.append(out_path)
    return out_paths
