from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def now_run_id() -> str:
    # Local time is fine for artifact naming.
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    def _default(o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        try:
            import numpy as np

            if isinstance(o, (np.generic,)):
                return o.item()
        except Exception:
            pass
        return str(o)

    p.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    return p


def dataclass_to_json_dict(obj: Any) -> dict[str, Any]:
    try:
        return asdict(obj)
    except Exception:
        return dict(obj.__dict__)


def download_to(url: str, dst: str | Path, *, force: bool = False, timeout_s: float = 60.0) -> Path:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return dst

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "emg_tst/mocap_phys_eval_v2 (python urllib)",
        },
    )
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        data = resp.read()
    dst.write_bytes(data)
    return dst


def resample_linear(x: np.ndarray, *, src_hz: float, dst_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < 2:
        return x.copy()
    src_hz = float(src_hz)
    dst_hz = float(dst_hz)
    if abs(src_hz - dst_hz) < 1e-6:
        return x.copy()

    t_src = np.arange(x.size, dtype=np.float64) / src_hz
    t_end = float(t_src[-1])
    n_dst = int(round(t_end * dst_hz)) + 1
    t_dst = np.arange(n_dst, dtype=np.float64) / dst_hz
    y = np.interp(t_dst, t_src, x).astype(np.float32)
    return y


def quat_normalize_wxyz(q: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Normalize quaternion(s) in wxyz order."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must have last dim 4 (wxyz), got shape={q.shape}")
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, float(eps), None)
    return (q / n).astype(np.float64)


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must have last dim 4 (wxyz), got shape={q.shape}")
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion multiply in wxyz order. Broadcasts over leading dims."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError(f"quat mul expects last dim 4 (wxyz), got {a.shape} and {b.shape}")
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack([w, x, y, z], axis=-1).astype(np.float64)


def quat_geodesic_deg_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Geodesic angle between quaternions in degrees, in [0, 180]."""
    a = quat_normalize_wxyz(a)
    b = quat_normalize_wxyz(b)
    # q and -q represent the same rotation; use abs(dot) for shortest distance.
    d = np.sum(a * b, axis=-1)
    d = np.clip(np.abs(d), 0.0, 1.0)
    ang = 2.0 * np.arccos(d)
    return np.degrees(ang).astype(np.float64)


def quat_average_wxyz(q: np.ndarray, *, weights: np.ndarray | None = None) -> np.ndarray:
    """Average quaternions using the Markley method.

    This is sign-invariant and works well for small dispersion, which is the case
    for "constant alignment offset" estimation over a short window.
    """
    q = quat_normalize_wxyz(q)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quat_average expects shape (N,4), got {q.shape}")
    n = int(q.shape[0])
    if n < 1:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if weights is None:
        w = np.ones((n,), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != n:
            raise ValueError(f"weights must have shape ({n},), got {w.shape}")
        if not np.any(np.isfinite(w)):
            w = np.ones((n,), dtype=np.float64)
        w = np.where(np.isfinite(w), w, 0.0)

    # A = sum_i w_i * q_i q_i^T
    A = np.zeros((4, 4), dtype=np.float64)
    for i in range(n):
        qi = q[i].reshape(4, 1)
        A += float(w[i]) * (qi @ qi.T)
    # Largest-eigenvalue eigenvector.
    vals, vecs = np.linalg.eigh(A)
    q_avg = vecs[:, int(np.argmax(vals))].reshape(4)
    # Deterministic sign: enforce w >= 0.
    if q_avg[0] < 0.0:
        q_avg = -q_avg
    return quat_normalize_wxyz(q_avg)


def quat_align_constant_offset_wxyz(
    ref_q: np.ndarray, query_q: np.ndarray, *, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_offset, query_aligned) where q_offset * query ~ ref."""
    ref_q = quat_normalize_wxyz(ref_q, eps=eps)
    query_q = quat_normalize_wxyz(query_q, eps=eps)
    n = int(min(ref_q.shape[0], query_q.shape[0]))
    if n < 1:
        q_off = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return q_off, query_q.copy()
    ref_q = ref_q[:n]
    query_q = query_q[:n]
    q_err = quat_mul_wxyz(ref_q, quat_conj_wxyz(query_q))
    q_off = quat_average_wxyz(q_err)
    q_al = quat_mul_wxyz(q_off[None, :], query_q)
    return q_off.astype(np.float64), quat_normalize_wxyz(q_al, eps=eps)


def _quat_fix_sign_continuity_wxyz(q: np.ndarray) -> np.ndarray:
    """Flip signs to keep consecutive dot(q[i], q[i-1]) >= 0."""
    q = quat_normalize_wxyz(q)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected (T,4), got {q.shape}")
    out = q.copy()
    for i in range(1, int(out.shape[0])):
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] *= -1.0
    return out


def resample_quat_slerp_wxyz(q: np.ndarray, *, src_hz: float, dst_hz: float) -> np.ndarray:
    """Resample quaternion timeseries with SLERP (wxyz)."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"q must have shape (T,4), got {q.shape}")
    if q.shape[0] < 2:
        return quat_normalize_wxyz(q).astype(np.float32)
    src_hz = float(src_hz)
    dst_hz = float(dst_hz)
    if abs(src_hz - dst_hz) < 1e-6:
        return _quat_fix_sign_continuity_wxyz(q).astype(np.float32)

    q = _quat_fix_sign_continuity_wxyz(q)
    t_src = np.arange(q.shape[0], dtype=np.float64) / src_hz
    t_end = float(t_src[-1])
    n_dst = int(round(t_end * dst_hz)) + 1
    t_dst = np.arange(n_dst, dtype=np.float64) / dst_hz

    idx_f = np.clip(np.searchsorted(t_src, t_dst, side="right") - 1, 0, q.shape[0] - 2).astype(np.int64)
    idx_n = idx_f + 1
    t0 = t_src[idx_f]
    t1 = t_src[idx_n]
    # Avoid divide-by-zero if timestamps collide (shouldn't happen).
    denom = np.where((t1 - t0) > 1e-12, (t1 - t0), 1.0)
    u = (t_dst - t0) / denom

    q0 = q[idx_f]
    q1 = q[idx_n]
    # Ensure shortest path.
    dot = np.sum(q0 * q1, axis=1)
    flip = dot < 0.0
    q1 = q1.copy()
    q1[flip] *= -1.0
    dot = np.abs(dot)
    dot = np.clip(dot, 0.0, 1.0)

    # If very close, lerp + renorm.
    close = dot > 0.9995
    out = np.empty((n_dst, 4), dtype=np.float64)
    if np.any(close):
        qc = (1.0 - u[close, None]) * q0[close] + u[close, None] * q1[close]
        out[close] = quat_normalize_wxyz(qc)
    if np.any(~close):
        theta = np.arccos(dot[~close])
        sin_theta = np.sin(theta)
        a = np.sin((1.0 - u[~close]) * theta) / sin_theta
        b = np.sin(u[~close] * theta) / sin_theta
        out[~close] = a[:, None] * q0[~close] + b[:, None] * q1[~close]
        out[~close] = quat_normalize_wxyz(out[~close])

    return out.astype(np.float32)


def rotmat_to_quat_wxyz(R: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Convert rotation matrix/matrices to quaternion(s) in wxyz order.

    Args:
      R: shape (3,3) or (...,3,3)
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (...,3,3), got {R.shape}")
    orig_shape = R.shape[:-2]
    Rs = R.reshape((-1, 3, 3))
    out = np.empty((Rs.shape[0], 4), dtype=np.float64)

    for i in range(int(Rs.shape[0])):
        m = Rs[i]
        tr = float(m[0, 0] + m[1, 1] + m[2, 2])
        if tr > 0.0:
            S = float(np.sqrt(tr + 1.0)) * 2.0
            w = 0.25 * S
            x = float(m[2, 1] - m[1, 2]) / S
            y = float(m[0, 2] - m[2, 0]) / S
            z = float(m[1, 0] - m[0, 1]) / S
        elif float(m[0, 0]) > float(m[1, 1]) and float(m[0, 0]) > float(m[2, 2]):
            S = float(np.sqrt(1.0 + float(m[0, 0]) - float(m[1, 1]) - float(m[2, 2]))) * 2.0
            w = float(m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = float(m[0, 1] + m[1, 0]) / S
            z = float(m[0, 2] + m[2, 0]) / S
        elif float(m[1, 1]) > float(m[2, 2]):
            S = float(np.sqrt(1.0 + float(m[1, 1]) - float(m[0, 0]) - float(m[2, 2]))) * 2.0
            w = float(m[0, 2] - m[2, 0]) / S
            x = float(m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = float(m[1, 2] + m[2, 1]) / S
        else:
            S = float(np.sqrt(1.0 + float(m[2, 2]) - float(m[0, 0]) - float(m[1, 1]))) * 2.0
            w = float(m[1, 0] - m[0, 1]) / S
            x = float(m[0, 2] + float(m[2, 0])) / S
            y = float(m[1, 2] + float(m[2, 1])) / S
            z = 0.25 * S
        out[i, :] = np.asarray([w, x, y, z], dtype=np.float64)

    out = quat_normalize_wxyz(out, eps=eps)
    if not orig_shape:
        return out.reshape((4,))
    return out.reshape(orig_shape + (4,))


def set_global_determinism(seed: int = 0) -> None:
    # Keep runs deterministic between restarts. This matters for "REF should not fall".
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
