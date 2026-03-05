from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_HF_BASE = "https://huggingface.co/microsoft/mocapact-models/resolve/main"
_HF_REPO_ID = "microsoft/mocapact-models"
_HF_REVISION = "main"

# Full MoCapAct expert model zoo is split across multiple tarballs.
_FULL_EXPERT_TARBALL_URLS: tuple[str, ...] = tuple(
    f"{_HF_BASE}/all/experts/experts_{i}.tar.gz" for i in range(1, 9)
)
_FULL_EXPERT_TARBALL_FILENAMES: tuple[str, ...] = tuple(f"all/experts/experts_{i}.tar.gz" for i in range(1, 9))


def _is_no_space_error(e: BaseException) -> bool:
    return isinstance(e, OSError) and int(getattr(e, "errno", -1)) in (28, 112)


def _get_hf_token() -> str | None:
    """Return a Hugging Face token if available (env vars only; we never log it)."""
    try:
        # Prefer the official resolution order (env var first, then token file
        # created by `hf auth login`).
        from huggingface_hub import get_token  # type: ignore

        tok = get_token()
        if tok:
            tok = str(tok).strip()
            if tok:
                return tok
    except Exception:
        pass

    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            v = str(v).strip()
            if v:
                return v
    return None


@dataclass(frozen=True)
class ExpertSnippet:
    """One MoCapAct clip-snippet expert policy."""

    snippet_id: str  # e.g. "CMU_083_33-0-194"
    clip_id: str  # e.g. "CMU_083_33"
    start_step: int
    end_step: int
    model_dir: Path  # .../eval_rsi/model


def _tqdm() -> object | None:
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:  # pragma: no cover
        return None


def _env_flag(name: str, *, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _auth_headers() -> dict[str, str]:
    token = _get_hf_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _download_backend() -> str:
    """Pick download backend.

    Values:
    - "auto" (default): aria2c if installed, else urllib (resumable).
    - "aria2": requires `aria2c` in PATH.
    - "urllib": builtin resumable downloader.
    - "hf_transfer": fast chunked downloader (no safe resume across interruptions).
    """
    v = str(os.environ.get("MOCAPACT_DOWNLOAD_BACKEND", "auto")).strip().lower()
    if v in ("", "auto"):
        if shutil.which("aria2c"):
            return "aria2"
        return "urllib"
    if v in ("aria2", "urllib", "hf_transfer"):
        return v
    return "auto"


def _download_with_aria2(url: str, dst: Path, *, force: bool, extra_headers: dict[str, str] | None) -> Path:
    aria2c = shutil.which("aria2c")
    if not aria2c:
        raise RuntimeError("aria2c not found in PATH (install aria2 or use MOCAPACT_DOWNLOAD_BACKEND=urllib).")

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return dst

    # If we previously used the builtin downloader, it left a ".part" file; aria2
    # resumes based on the final filename, so rename it in-place.
    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists() and not dst.exists():
        try:
            part.replace(dst)
        except Exception:
            pass

    # Conservative defaults; override with env vars if needed.
    conn = str(int(os.environ.get("MOCAPACT_ARIA2_CONNECTIONS", "16")))
    split = str(int(os.environ.get("MOCAPACT_ARIA2_SPLIT", conn)))
    min_split = str(os.environ.get("MOCAPACT_ARIA2_MIN_SPLIT_SIZE", "1M"))

    cmd: list[str] = [
        aria2c,
        "--continue=true",
        f"--max-connection-per-server={conn}",
        f"--split={split}",
        f"--min-split-size={min_split}",
        "--file-allocation=none",
        "--allow-overwrite=true" if force else "--allow-overwrite=false",
        "-d",
        str(dst.parent),
        "-o",
        str(dst.name),
        url,
    ]
    if extra_headers:
        for k, v in extra_headers.items():
            if v:
                cmd.insert(-1, f"--header={str(k)}: {str(v)}")

    try:
        subprocess.run(cmd, check=True)
    except BaseException as e:
        if _is_no_space_error(e):
            raise OSError(
                28,
                "No space left on device while downloading MoCapAct experts. "
                "Set MOCAPACT_MODELS_DIR to a drive with >=200GB free.",
            ) from e
        raise
    return dst


def _download_with_hf_transfer(url: str, dst: Path, *, force: bool, extra_headers: dict[str, str] | None) -> Path:
    """Fast chunked downloader using `hf_transfer`.

    Note: This backend does not provide a safe byte-accurate resume if interrupted,
    because the file may be written out-of-order. Prefer aria2/urllib if you need resume.
    """
    try:
        import hf_transfer  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("hf_transfer is not installed; pip install hf-transfer or use urllib.") from e

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return dst

    # Always download into a temp file and replace at the end so we never "half-finish" dst.
    tmp = dst.with_suffix(dst.suffix + ".hf.part")
    try:
        tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
    except Exception:
        pass

    total = None
    try:
        req = urllib.request.Request(str(url), method="HEAD", headers=(extra_headers or {}))
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            clen = resp.headers.get("Content-Length")
            if clen is not None:
                total = int(clen)
    except Exception:
        total = None

    max_files = int(os.environ.get("MOCAPACT_HF_TRANSFER_MAX_FILES", "16"))
    chunk_mb = int(os.environ.get("MOCAPACT_HF_TRANSFER_CHUNK_MB", "16"))
    chunk_size = int(max(1, chunk_mb) * 1024 * 1024)

    tqdm = _tqdm()
    pbar = None
    if tqdm is not None and total is not None and total > 0:
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dst.name, leave=False)

    downloaded = 0

    def _cb(delta: int) -> None:
        nonlocal downloaded
        try:
            d = int(delta)
        except Exception:
            return
        if d <= 0:
            return
        if pbar is not None:
            if total is not None:
                inc = min(d, max(0, int(total) - int(downloaded)))
            else:
                inc = d
            if inc:
                pbar.update(int(inc))
        downloaded += d

    try:
        hf_transfer.download(
            str(url),
            str(tmp),
            max_files=max_files,
            chunk_size=chunk_size,
            parallel_failures=0,
            max_retries=int(os.environ.get("MOCAPACT_HF_TRANSFER_MAX_RETRIES", "8")),
            headers=(extra_headers or None),
            callback=_cb,
        )
    except BaseException as e:
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        if _is_no_space_error(e):
            raise OSError(
                28,
                "No space left on device while downloading MoCapAct experts. "
                "Set MOCAPACT_MODELS_DIR to a drive with >=200GB free.",
            ) from e
        raise
    finally:
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass

    tmp.replace(dst)
    return dst


def _download_expert_tarball(url: str, dst: Path, *, force: bool, timeout_s: float) -> Path:
    extra_headers = _auth_headers()

    # The builtin urllib downloader supports true resume via ".part".
    # If such a file exists, always resume with urllib regardless of backend.
    if dst.with_suffix(dst.suffix + ".part").exists():
        return _stream_download(url, dst, force=force, timeout_s=timeout_s, extra_headers=extra_headers)

    backend = _download_backend()
    if backend == "aria2":
        return _download_with_aria2(url, dst, force=force, extra_headers=extra_headers)
    if backend == "hf_transfer":
        return _download_with_hf_transfer(url, dst, force=force, extra_headers=extra_headers)
    return _stream_download(url, dst, force=force, timeout_s=timeout_s, extra_headers=extra_headers)


def _stream_download(
    url: str,
    dst: str | Path,
    *,
    force: bool = False,
    timeout_s: float = 120.0,
    extra_headers: dict[str, str] | None = None,
) -> Path:
    """Stream download (supports resume if a .part file exists)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return dst

    tmp = dst.with_suffix(dst.suffix + ".part")
    start = int(tmp.stat().st_size) if tmp.exists() else 0

    headers: dict[str, str] = {}
    if extra_headers:
        headers.update({str(k): str(v) for k, v in extra_headers.items() if v})
    if start > 0:
        headers["Range"] = f"bytes={start}-"

    req = urllib.request.Request(str(url), headers=headers)
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        code = int(getattr(resp, "status", resp.getcode()))
        if start > 0 and code != 206:
            # Server didn't honor Range; restart.
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            start = 0

        total = None
        try:
            clen = resp.headers.get("Content-Length")
            if clen is not None:
                total = int(clen) + int(start)
        except Exception:
            total = None

        tqdm = _tqdm()
        pbar = None
        if tqdm is not None and total is not None and total > 0:
            pbar = tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=dst.name, leave=False)

        mode = "ab" if start > 0 else "wb"
        try:
            with tmp.open(mode) as f:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    if pbar is not None:
                        pbar.update(len(chunk))
        except BaseException as e:
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
            if _is_no_space_error(e):
                raise OSError(
                    28,
                    "No space left on device while downloading MoCapAct experts. "
                    "Set MOCAPACT_MODELS_DIR to a drive with >=200GB free, and consider deleting "
                    "already-extracted tarballs under <MODELS_DIR>/_downloads/.",
                ) from e
            raise
        if pbar is not None:
            pbar.close()

    tmp.replace(dst)
    return dst


def _is_within_dir(base: Path, target: Path) -> bool:
    try:
        base_r = base.resolve()
        targ_r = target.resolve()
        return str(targ_r).startswith(str(base_r))
    except Exception:
        return False


def _extract_tarball(tar_path: Path, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tqdm = _tqdm()
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getmembers()
        it: Iterable[tarfile.TarInfo] = members
        if tqdm is not None:
            it = tqdm(members, desc=f"Extract {tar_path.name}", unit="file", leave=False)
        for m in it:
            # Basic path traversal protection.
            dest = out_dir / str(m.name)
            if not _is_within_dir(out_dir, dest):
                continue
            try:
                tf.extract(m, path=out_dir)
            except BaseException as e:
                if _is_no_space_error(e):
                    raise OSError(
                        28,
                        "No space left on device while extracting MoCapAct experts. "
                        "Set MOCAPACT_MODELS_DIR to a drive with >=200GB free, and consider deleting "
                        "already-extracted tarballs under <MODELS_DIR>/_downloads/.",
                    ) from e
                # Ignore rare extraction errors (e.g., path issues); we'll validate by scanning for models after.
                continue


_SNIP_RE = re.compile(r"^(CMU_[0-9]{3}_[0-9]{2})-([0-9]+)-([0-9]+)$")


def discover_expert_snippets(experts_root: str | Path) -> list[ExpertSnippet]:
    """Find all extracted expert models under `experts_root`."""
    root = Path(experts_root)
    if not root.exists():
        return []

    by_snip: dict[str, tuple[int, ExpertSnippet]] = {}

    # Find all model dirs by locating best_model.zip; policy root is its parent dir.
    for zip_path in root.rglob("best_model.zip"):
        model_dir = zip_path.parent
        if not (model_dir / "vecnormalize.pkl").exists():
            continue

        # Expected layout: <clip_id>/<snippet_id>/<eval_*>/model/best_model.zip
        try:
            eval_dir = model_dir.parent
            snippet_dir = eval_dir.parent
            clip_dir = snippet_dir.parent
            eval_name = str(eval_dir.name)
            snippet_id = str(snippet_dir.name)
            clip_id = str(clip_dir.name)
        except Exception:
            continue

        m = _SNIP_RE.match(snippet_id)
        if m is None:
            continue
        clip_from_snip, s0, s1 = m.group(1), m.group(2), m.group(3)
        if clip_from_snip != clip_id:
            # Some archives may include extra nesting; trust the snippet_id.
            clip_id = clip_from_snip

        start_step = int(s0)
        end_step = int(s1)
        snip = ExpertSnippet(
            snippet_id=snippet_id,
            clip_id=clip_id,
            start_step=int(start_step),
            end_step=int(end_step),
            model_dir=model_dir,
        )

        # Prefer eval_rsi over eval_start over anything else.
        pri = 2 if eval_name == "eval_rsi" else 1 if eval_name == "eval_start" else 0
        prev = by_snip.get(snippet_id)
        if prev is None or pri > prev[0]:
            by_snip[snippet_id] = (pri, snip)

    out = [v[1] for v in by_snip.values()]
    out.sort(key=lambda x: (x.clip_id, x.start_step, x.end_step))
    return out


def ensure_full_expert_zoo(*, experts_root: str | Path, downloads_dir: str | Path) -> Path:
    """Ensure the full expert model zoo tarballs are downloaded+extracted."""
    experts_root = Path(experts_root)
    downloads_dir = Path(downloads_dir)
    experts_root.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    keep_tars = _env_flag("MOCAPACT_KEEP_TARBALLS", default=False)
    backend = _download_backend()
    have_token = bool(_get_hf_token())
    print(
        f"[mocap_phys_eval] status: expert zoo download backend={backend} "
        f"token={'yes' if have_token else 'no'} keep_tars={'yes' if keep_tars else 'no'}"
    )

    # We intentionally do NOT early-exit based on "found >= 2500" here.
    #
    # MoCapAct's paper reports 2589 snippets, and the Hugging Face model zoo is
    # split across multiple tarballs. Stopping early can leave the zoo incomplete,
    # which directly harms motion matching (smaller reference bank).
    #
    # Instead, we rely on per-tarball ".extracted" markers to make this function
    # resumable and idempotent: rerunning will skip already-extracted tarballs.
    _ = discover_expert_snippets(experts_root)

    for i, url in enumerate(_FULL_EXPERT_TARBALL_URLS, start=1):
        tar_path = downloads_dir / f"experts_{i}.tar.gz"
        marker = downloads_dir / f"experts_{i}.extracted"
        if marker.exists():
            if not keep_tars and tar_path.exists():
                try:
                    tar_path.unlink()
                except Exception:
                    pass
            continue

        before = len(discover_expert_snippets(experts_root))
        _download_expert_tarball(url, tar_path, force=False, timeout_s=600.0)
        _extract_tarball(tar_path, out_dir=experts_root)
        after = len(discover_expert_snippets(experts_root))

        # Only stamp completion if extraction appears to have produced new snippet policies.
        if after > before:
            try:
                marker.write_text("ok\n", encoding="utf-8")
            except Exception:
                pass
            if not keep_tars:
                try:
                    tar_path.unlink()
                except Exception:
                    pass
    return experts_root
