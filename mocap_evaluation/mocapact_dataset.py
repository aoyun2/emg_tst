from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

EXPECTED_SNIPPETS = 2589

# Canonical MoCapAct snippet id format: "<clip_id>-<start_step>-<end_step>"
_SNIPPET_ID_RE = re.compile(r"^(?P<clip_id>[A-Za-z0-9_]+)-(?P<start>\d+)-(?P<end>\d+)$")


def is_snippet_id(name: str) -> bool:
    return _SNIPPET_ID_RE.match(name) is not None


def parse_snippet_id(snippet_id: str) -> tuple[str, int, int]:
    """Parse a MoCapAct snippet id like 'CMU_083_33-0-194'."""
    m = _SNIPPET_ID_RE.match(snippet_id)
    if not m:
        raise ValueError(f"Not a valid MoCapAct snippet id: {snippet_id!r}")
    return m.group("clip_id"), int(m.group("start")), int(m.group("end"))


@dataclass(frozen=True)
class ExpertPolicy:
    """Filesystem pointers + metadata for a per-snippet MoCapAct expert policy."""

    snippet_id: str
    clip_id: str
    start_step: int
    end_step: int

    expert_dir: Path
    clip_info_path: Path

    # Typically ".../eval_rsi/model"
    model_dir: Path
    best_model_zip: Path
    vecnormalize_pkl: Path | None


def iter_expert_policies(experts_root: str | Path) -> Iterator[ExpertPolicy]:
    """Yield expert policies found under an extracted `microsoft/mocapact-models` experts directory.

    Handles both:
      - flat layout:  <root>/<snippet_id>/clip_info.json
      - nested layout: <root>/<clip_id>/<snippet_id>/clip_info.json
    """
    root = Path(experts_root)
    for clip_info in sorted(root.rglob("clip_info.json")):
        try:
            info = json.loads(clip_info.read_text(encoding="utf-8"))
        except Exception:
            continue

        clip_id = str(info.get("clip_id", "")).strip()
        start_step = int(info.get("start_step", -1))
        end_step = int(info.get("end_step", -1))
        if not clip_id or start_step < 0 or end_step < 0:
            continue

        snippet_id = f"{clip_id}-{start_step}-{end_step}"
        expert_dir = clip_info.parent

        # Prefer eval_rsi (as in the paper). Fall back to any best_model.zip we can find.
        preferred = expert_dir / "eval_rsi" / "model" / "best_model.zip"
        if preferred.exists():
            model_dir = preferred.parent
            best_model_zip = preferred
        else:
            zips = sorted(expert_dir.rglob("best_model.zip"))
            if not zips:
                continue
            best_model_zip = zips[0]
            model_dir = best_model_zip.parent

        vecnormalize = model_dir / "vecnormalize.pkl"
        vecnormalize_pkl = vecnormalize if vecnormalize.exists() else None

        yield ExpertPolicy(
            snippet_id=snippet_id,
            clip_id=clip_id,
            start_step=start_step,
            end_step=end_step,
            expert_dir=expert_dir,
            clip_info_path=clip_info,
            model_dir=model_dir,
            best_model_zip=best_model_zip,
            vecnormalize_pkl=vecnormalize_pkl,
        )


def load_expert_policy_map(experts_root: str | Path) -> dict[str, ExpertPolicy]:
    out: dict[str, ExpertPolicy] = {}
    for pol in iter_expert_policies(experts_root):
        out[pol.snippet_id] = pol
    return out


def iter_rollout_hdf5_files(rollouts_root: str | Path) -> Iterator[Path]:
    root = Path(rollouts_root)
    exts = {".h5", ".hdf5"}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def iter_rollout_snippets(rollouts_root: str | Path) -> Iterator[tuple[str, Path]]:
    """Yield (snippet_id, hdf5_path) pairs by scanning extracted MoCapAct rollout HDF5 files."""
    import h5py  # local import: optional dependency until rollouts are used

    for h5_path in iter_rollout_hdf5_files(rollouts_root):
        try:
            with h5py.File(h5_path, "r") as h5:
                for name in h5.keys():
                    if is_snippet_id(name):
                        yield str(name), h5_path
        except Exception:
            continue


def load_rollout_snippet_map(rollouts_root: str | Path) -> dict[str, Path]:
    """Return a mapping snippet_id -> one HDF5 file that contains that snippet group."""
    out: dict[str, Path] = {}
    for snippet_id, h5_path in iter_rollout_snippets(rollouts_root):
        out.setdefault(snippet_id, h5_path)
    return out


def validate_expected_count(actual: int, *, expected: int = EXPECTED_SNIPPETS) -> tuple[int, bool]:
    return actual, actual == expected


def validate_downloads(
    *,
    rollouts_root: str | Path | None = None,
    experts_root: str | Path | None = None,
    expected: int = EXPECTED_SNIPPETS,
) -> dict:
    """Best-effort validation for local MoCapAct assets.

    Returns a dict suitable for JSON logging.
    """
    out: dict = {"expected_snippets": int(expected)}

    if rollouts_root is not None:
        rollouts_map = load_rollout_snippet_map(rollouts_root)
        count, ok = validate_expected_count(len(rollouts_map), expected=expected)
        out["rollouts_root"] = str(Path(rollouts_root))
        out["rollout_snippets_found"] = int(count)
        out["rollout_snippets_ok"] = bool(ok)

    if experts_root is not None:
        experts_map = load_expert_policy_map(experts_root)
        count, ok = validate_expected_count(len(experts_map), expected=expected)
        out["experts_root"] = str(Path(experts_root))
        out["expert_policies_found"] = int(count)
        out["expert_policies_ok"] = bool(ok)

    return out
