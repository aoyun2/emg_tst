"""MoCapAct-based prosthetic evaluation.

Uses Microsoft's MoCapAct pre-trained locomotion policies (multi-clip or
clip-snippet experts) running on dm_control's CMU Humanoid to evaluate
prosthetic knee predictions.

Instead of driving all joints from mocap reference and only overriding the
right knee (as ``prosthetic_sim.py`` does), this module:

1. Loads a pre-trained MoCapAct multi-clip policy that has *learned* to walk.
2. At each simulation step, the policy proposes actions for **all** 56 joints.
3. The right-knee actuator is **overridden** with the EMG model's predicted
   angle, while all other joints follow the policy output.
4. Because the policy is *adaptive* (it reacts to proprioceptive feedback),
   the rest of the body tries to compensate for the imposed knee angle —
   producing a much more realistic evaluation than kinematic replay.

The same ``EvalMetrics`` / ``stability_score`` / ``robustness_score`` contract
used by ``prosthetic_sim.py`` is preserved, so the evaluation pipeline
(``paper_pipeline.py``) can swap between backends transparently.

Public APIs accept included-angle convention (degrees):
- 180 = straight / neutral
- smaller values = increased flexion magnitude

Requirements
------------
pip install mocapact dm_control stable-baselines3
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ── Optional imports ──────────────────────────────────────────────────────────

_MOCAPACT_AVAILABLE = False
_DM_CONTROL_AVAILABLE = False

try:
    import dm_control  # noqa: F401
    from dm_control.locomotion.tasks.reference_pose import cmu_subsets
    _DM_CONTROL_AVAILABLE = True
except Exception:
    pass

# ── NumPy 2.0 compatibility shims ────────────────────────────────────────────
# Many attributes were removed in NumPy 2.0 that gym/sb3/dm_control still use.
import numpy as _np
_NP2_ALIASES = {
    # scalar type aliases removed in 2.0
    "bool": _np.bool_,
    "int": _np.int_,
    "float": _np.float64,
    "complex": _np.complex128,
    "object": _np.object_,
    "str": _np.str_,
    # scalar constant aliases removed in 2.0
    "infty": _np.inf,
    "Inf": _np.inf,
    "Infinity": _np.inf,
    "NaN": float("nan"),
    "PINF": _np.inf,
    "NINF": -_np.inf,
    # reduction aliases removed in 2.0
    "product": _np.prod,
    "cumproduct": _np.cumprod,
    "sometrue": _np.any,
    "alltrue": _np.all,
    "row_stack": _np.vstack,
    # type alias removed in 2.0
    "VisibleDeprecationWarning": getattr(_np, "exceptions", _np).VisibleDeprecationWarning
    if hasattr(getattr(_np, "exceptions", None) or object(), "VisibleDeprecationWarning")
    else DeprecationWarning,
}
import warnings as _warnings
with _warnings.catch_warnings():
    # Some names (e.g. np.object, np.str) raise FutureWarning when accessed on
    # NumPy 1.x even via hasattr().  Suppress all warnings during the check.
    _warnings.simplefilter("ignore")
    for _name, _val in _NP2_ALIASES.items():
        if not hasattr(_np, _name):
            setattr(_np, _name, _val)

# ── gym.spaces compatibility shims ───────────────────────────────────────────
# Patch missing attributes introduced in different gym versions so that
# checkpoints saved with one gym version load correctly on another.
try:
    import gym.spaces.box as _gsb

    # low_repr / high_repr: added in gym 0.26 as plain instance attrs.
    # Use __getattr__ (fallback-only lookup) so the attribute is synthesised
    # whenever it's missing — regardless of whether the Box was created fresh,
    # unpickled from an old checkpoint, or restored via __new__.
    if not hasattr(_gsb.Box, "low_repr"):
        _orig_box_getattr = _gsb.Box.__getattr__ if hasattr(_gsb.Box, "__getattr__") else None
        def _box_getattr(self, name: str):
            if name == "low_repr":
                return str(self.low)
            if name == "high_repr":
                return str(self.high)
            if _orig_box_getattr is not None:
                return _orig_box_getattr(self, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        _gsb.Box.__getattr__ = _box_getattr  # type: ignore[attr-defined]

    # is_bounded: added in gym 0.26; older Box had no such method
    if not hasattr(_gsb.Box, "is_bounded"):
        def _is_bounded(self, manner: str = "both") -> bool:
            below = bool(_np.all(_np.isfinite(self.low)))
            above = bool(_np.all(_np.isfinite(self.high)))
            if manner == "both":
                return below and above
            if manner == "below":
                return below
            if manner == "above":
                return above
            raise ValueError(f"manner must be 'both', 'below', or 'above', got {manner!r}")
        _gsb.Box.is_bounded = _is_bounded  # type: ignore[attr-defined]
except Exception:
    pass

try:
    import gym.spaces.space as _gss
    # np_random property: gym 0.26 renamed _np_random → np_random
    if not hasattr(_gss.Space, "np_random"):
        @property  # type: ignore[misc]
        def _np_random_prop(self):
            if not hasattr(self, "_np_random"):
                self._np_random = _np.random.RandomState()
            return self._np_random
        @_np_random_prop.setter
        def _np_random_prop(self, rng):
            self._np_random = rng
        _gss.Space.np_random = _np_random_prop  # type: ignore[attr-defined]
except Exception:
    pass

# ── SB3 gym↔gymnasium observation-space compatibility shim ───────────────────
# SB3 2.x uses `gymnasium`; MoCapAct checkpoints were saved with `gym` 0.21.
# When the checkpoint is unpickled the observation_space is a gym.spaces.Dict,
# but SB3's obs_to_tensor asserts isinstance(obs_space, gymnasium.spaces.Dict)
# → fails even though the object looks and acts like a Dict.
#
# Fix: patch BasePolicy.obs_to_tensor to coerce the stored space into whatever
# `spaces` class SB3 actually imported, right before the isinstance check.
try:
    import importlib as _il
    _sb3pol = _il.import_module("stable_baselines3.common.policies")
    _sb3_sp = _sb3pol.__dict__.get("spaces")  # the `spaces` module SB3 uses
    if _sb3_sp is not None and hasattr(_sb3pol, "BasePolicy"):

        def _coerce_to_sb3_space(sp, _s=_sb3_sp):
            """Rebuild *sp* using SB3's own spaces module, preserving structure."""
            if isinstance(sp, _s.Space):
                return sp
            name = type(sp).__name__
            if name == "Dict" or (
                hasattr(sp, "spaces") and isinstance(getattr(sp, "spaces", None), dict)
            ):
                return _s.Dict({k: _coerce_to_sb3_space(v, _s) for k, v in sp.spaces.items()})
            if name == "Box" or (
                hasattr(sp, "low") and hasattr(sp, "high") and hasattr(sp, "shape")
            ):
                return _s.Box(low=sp.low, high=sp.high, shape=sp.shape, dtype=sp.dtype)
            if name == "Discrete" or (
                hasattr(sp, "n") and not hasattr(sp, "low") and not hasattr(sp, "spaces")
            ):
                kw = {"start": sp.start} if hasattr(sp, "start") else {}
                return _s.Discrete(sp.n, **kw)
            if name == "MultiDiscrete" or hasattr(sp, "nvec"):
                return _s.MultiDiscrete(sp.nvec, dtype=getattr(sp, "dtype", _np.int64))
            if name == "MultiBinary":
                return _s.MultiBinary(sp.n)
            return sp  # unknown — pass through unchanged

        _orig_o2t = _sb3pol.BasePolicy.obs_to_tensor

        def _patched_o2t(self, observation):
            # If obs is a dict but obs_space isn't the right Dict class, coerce once.
            if isinstance(observation, dict) and not isinstance(
                self.observation_space, _sb3_sp.Dict
            ):
                sp = self.observation_space
                if hasattr(sp, "spaces") and isinstance(getattr(sp, "spaces", None), dict):
                    self.observation_space = _coerce_to_sb3_space(sp)
            return _orig_o2t(self, observation)

        _sb3pol.BasePolicy.obs_to_tensor = _patched_o2t  # type: ignore[method-assign]
except Exception:
    pass

# ── SB3 1.x → 2.x extract_features signature shim ───────────────────────────
# SB3 2.x changed BaseModel.extract_features(obs) to
# BaseModel.extract_features(obs, features_extractor).
# MoCapAct was built against SB3 1.x and calls the old single-arg form.
try:
    import importlib as _il
    import inspect as _inspect
    _sb3base_mod = _il.import_module("stable_baselines3.common.policies")
    if hasattr(_sb3base_mod, "BaseModel"):
        _ef = _sb3base_mod.BaseModel.extract_features
        _ef_params = list(_inspect.signature(_ef).parameters.keys())
        # Only patch if features_extractor is a required param (SB3 2.x behaviour)
        if "features_extractor" in _ef_params and (
            _inspect.signature(_ef).parameters["features_extractor"].default
            is _inspect.Parameter.empty
        ):
            def _patched_ef(self, obs, features_extractor=None):
                if features_extractor is None:
                    features_extractor = self.features_extractor
                return _ef(self, obs, features_extractor)
            _sb3base_mod.BaseModel.extract_features = _patched_ef  # type: ignore[method-assign]
except Exception:
    pass

# ── pkg_resources compatibility shim ─────────────────────────────────────────
# On Python 3.12+ Windows venvs, setuptools may be installed but
# pkg_resources (which lives inside it) is occasionally not importable.
# mocapact / stable-baselines3 import pkg_resources for version lookups.
# Provide a minimal shim so the import succeeds; real version queries fall
# back to importlib.metadata which is always available in Python 3.8+.
try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    import types as _types
    import importlib.metadata as _ilm

    _pr = _types.ModuleType("pkg_resources")

    class _Dist:  # minimal Distribution stand-in
        def __init__(self, name: str) -> None:
            try:
                self.version = _ilm.version(name)
            except _ilm.PackageNotFoundError:
                self.version = "0.0.0"
        def __str__(self) -> str:
            return self.version

    class _DistributionNotFound(Exception):
        pass

    def _get_distribution(name: str) -> _Dist:
        try:
            return _Dist(name)
        except Exception:
            raise _DistributionNotFound(name)

    def _require(requirements) -> list:
        return []

    def _parse_version(v: str):
        from importlib.metadata import version as _v
        return v

    _pr.get_distribution = _get_distribution  # type: ignore[attr-defined]
    _pr.require = _require  # type: ignore[attr-defined]
    _pr.parse_version = _parse_version  # type: ignore[attr-defined]
    _pr.DistributionNotFound = _DistributionNotFound  # type: ignore[attr-defined]
    _pr.VersionConflict = Exception  # type: ignore[attr-defined]
    _sys.modules["pkg_resources"] = _pr

_MOCAPACT_IMPORT_ERROR: str = ""
try:
    from mocapact.distillation import model as npmp_model
    from mocapact.envs import tracking as mocapact_tracking
    from mocapact.sb3 import utils as mocapact_utils
    from mocapact import observables as mocapact_obs
    _MOCAPACT_AVAILABLE = True
except BaseException as _e:
    # Use BaseException (not just Exception) to catch import-time errors
    # like SystemExit or errors from C extensions on Windows.
    _MOCAPACT_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"
    import warnings as _warnings
    _warnings.warn(
        f"[mocapact_sim] mocapact could not be imported — "
        f"MoCapAct-based simulation will be unavailable.\n"
        f"  Error: {_MOCAPACT_IMPORT_ERROR}\n"
        f"  Run: from mocap_evaluation.mocapact_sim import check_requirements; "
        f"print(check_requirements())",
        stacklevel=1,
    )

# MuJoCo (for metric collection from dm_control's physics)
try:
    import mujoco
    _MUJOCO_AVAILABLE = True
except Exception:
    _MUJOCO_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────────────

SIM_FPS = 30.0          # MoCapAct default control freq (dm_control default)
FALL_HEIGHT_THRESHOLD = 0.55
DEFAULT_EVAL_SECONDS = 4.0

# dm_control CMU humanoid right knee actuator name and known alphabetical index.
# The 56 actuators are sorted alphabetically; rtibiarx lands at index 44.
_RIGHT_KNEE_ACTUATOR = "rtibiarx"
_RIGHT_KNEE_IDX_FALLBACK = 44

# Right knee joint range in the CMU humanoid V2020 (radians, flexion only).
# 0.01 rad ≈ fully extended, 2.967 rad ≈ 170° flexion.
_KNEE_RANGE_RAD = (0.01, 2.96706)

# dm_control CMU humanoid right hip (sagittal flexion/extension) actuator.
# rfemurry controls the thigh swing in the sagittal plane (Y-axis rotation).
# Alphabetically it sits a few positions before rtibiarx; index 37 is the
# known fallback — runtime discovery is always attempted first.
_RIGHT_HIP_ACTUATOR = "rfemurry"
_RIGHT_HIP_IDX_FALLBACK = 37

# Right hip joint range (radians): −60° (extension) to +75° (flexion).
# Positive = thigh swings forward, negative = thigh swings back.
_HIP_RANGE_RAD = (-1.047, 1.309)

# Default multi-clip policy checkpoint search order (relative to model dir).
# The HF repo bundles a tarball with two variants:
#   multiclip_policy/full_dataset/model/model.ckpt       (trained on ALL 1,144 clips)
#   multiclip_policy/locomotion_dataset/model/model.ckpt  (locomotion subset only)
# full_dataset is preferred so the policy can follow any CMU clip.
_DEFAULT_MULTI_CLIP_CKPT = "multiclip_policy/full_dataset/model/model.ckpt"
_DEFAULT_MULTI_CLIP_CKPT_ALT = "multiclip_policy/locomotion_dataset/model/model.ckpt"

# HuggingFace model repo
_HF_REPO = "microsoft/mocapact-models"
_HF_TARBALL = "multiclip_policy.tar.gz"

# Frame rate of the aggregated mocap database (from mocap_loader.TARGET_FPS).
_DB_FPS = 200


# ── Clip resolution (BVH match → dm_control clip) ───────────────────────────

def _bvh_to_clip_id(bvh_filename: str) -> str:
    """Convert a BVH filename to a dm_control CMU clip ID.

    ``"09_12.bvh"`` → ``"CMU_009_12"``
    """
    stem = bvh_filename.replace(".bvh", "")
    parts = stem.split("_")
    subject = int(parts[0])
    trial = int(parts[1])
    return f"CMU_{subject:03d}_{trial:02d}"


def resolve_clip_from_match(
    best_start: int,
    window_length: int,
    mocap_db: dict,
    db_fps: float = _DB_FPS,
) -> dict:
    """Resolve a DTW match index to a dm_control clip specification.

    After ``motion_matching.find_best_match`` returns ``best_start``, call
    this to identify which CMU clip it came from and where in that clip the
    window sits.  The result can be passed directly to
    ``simulate_mocapact_prosthetic(match_info=...)``.

    Parameters
    ----------
    best_start :
        Start index in the concatenated mocap database (at *db_fps*).
    window_length :
        Number of frames in the matched window (at *db_fps*).
    mocap_db :
        Aggregated mocap database dict (must contain ``file_boundaries``).
    db_fps :
        Frame rate of the mocap database (default 200 Hz).

    Returns
    -------
    dict
        ``clip_id``          – dm_control clip name, e.g. ``"CMU_009_12"``
        ``bvh_filename``     – original BVH filename
        ``category``         – motion category from the CMU catalog
        ``frame_in_file``    – frame offset within the BVH file (at db_fps)
        ``time_offset_s``    – time offset in seconds from clip start
        ``time_duration_s``  – matched window duration in seconds
    """
    boundaries = mocap_db.get("file_boundaries", [])
    for start, end, fname, cat in boundaries:
        start, end = int(start), int(end)
        if start <= best_start < end:
            frame_in_file = best_start - start
            # fname may be a MocapAct clip ID (e.g. "CMU_009_12") stored
            # directly by mocapact_dataset, or a BVH filename (e.g. "09_12.bvh")
            # from the legacy CMU loader.
            fname = str(fname)
            if fname.startswith("CMU_"):
                clip_id = fname
            else:
                clip_id = _bvh_to_clip_id(fname)
            return {
                "clip_id": clip_id,
                "bvh_filename": fname,
                "category": cat,
                "frame_in_file": frame_in_file,
                "time_offset_s": frame_in_file / db_fps,
                "time_duration_s": window_length / db_fps,
            }
    raise ValueError(
        f"best_start={best_start} not found in file_boundaries "
        f"({len(boundaries)} entries)"
    )


def _make_clip_collection(clip_id: str):
    """Create a dm_control ``ClipCollection`` for a single CMU clip."""
    from dm_control.locomotion.tasks.reference_pose import types
    return types.ClipCollection(ids=[clip_id])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _included_to_flexion(angles_deg: np.ndarray) -> np.ndarray:
    """Convert included-angle convention (180=straight) to flexion degrees."""
    return 180.0 - np.asarray(angles_deg, dtype=np.float64)


def _contact_events(frames: List[int], min_gap: int = 5) -> int:
    """Count distinct foot-contact events from a sorted list of frame indices."""
    if not frames:
        return 0
    out, prev = 1, frames[0]
    for fr in frames[1:]:
        if fr - prev > min_gap:
            out += 1
        prev = fr
    return out


def _gait_symmetry(right_frames: List[int], left_frames: List[int]) -> float:
    """Compute gait asymmetry ratio (0 = perfect symmetry)."""
    def intervals(frames: List[int]) -> np.ndarray:
        if len(frames) < 2:
            return np.array([], dtype=np.float64)
        u = sorted(set(frames))
        vals = []
        prev = u[0]
        for f in u[1:]:
            if f - prev > 3:
                vals.append(f - prev)
            prev = f
        return np.asarray(vals, dtype=np.float64)

    r = intervals(right_frames)
    l_int = intervals(left_frames)
    if r.size == 0 or l_int.size == 0:
        return 0.5
    mr, ml = float(np.mean(r)), float(np.mean(l_int))
    den = mr + ml
    return 0.0 if den < 1e-9 else float(abs(mr - ml) / den)


# ── EvalMetrics (same contract as prosthetic_sim.py) ──────────────────────────

@dataclass
class MoCapActEvalMetrics:
    """Per-frame evaluation metrics from a MoCapAct-based simulation."""
    com_height: List[float]
    pred_knee: List[float]
    ref_knee: List[float]
    right_contact_frames: List[int]
    left_contact_frames: List[int]
    fall_detected: bool
    fall_frame: int

    @classmethod
    def empty(cls) -> "MoCapActEvalMetrics":
        return cls([], [], [], [], [], False, -1)

    def to_dict(self) -> dict:
        pred = np.asarray(self.pred_knee, dtype=np.float64)
        ref = np.asarray(self.ref_knee, dtype=np.float64)
        err = pred - ref
        com = np.asarray(self.com_height, dtype=np.float64)

        rmse = float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0
        mae = float(np.mean(np.abs(err))) if err.size else 0.0
        com_mean = float(np.mean(com)) if com.size else 0.0
        com_std = float(np.std(com)) if com.size else 0.0

        sr = _contact_events(self.right_contact_frames)
        sl = _contact_events(self.left_contact_frames)
        sym = _gait_symmetry(self.right_contact_frames, self.left_contact_frames)

        stability = 1.0 - min(com_std / 0.25, 1.0) * 0.55 - sym * 0.25 - (0.35 if self.fall_detected else 0.0)
        stability = max(0.0, min(1.0, stability))
        return {
            "com_height_mean": com_mean,
            "com_height_std": com_std,
            "fall_detected": bool(self.fall_detected),
            "fall_frame": int(self.fall_frame),
            "knee_rmse_deg": rmse,
            "knee_mae_deg": mae,
            "step_count": int(sr + sl),
            "gait_symmetry": float(sym),
            "stability_score": float(stability),
            "com_height_series": com.tolist(),
            "pred_knee_series": (180.0 - pred).tolist(),
            "ref_knee_series": (180.0 - ref).tolist(),
            "right_contact_frames": list(self.right_contact_frames),
            "left_contact_frames": list(self.left_contact_frames),
        }


# ── Knee-actuator index discovery ────────────────────────────────────────────

def _get_physics(env):
    """Extract the dm_control physics handle from a Gym-wrapped environment."""
    if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_env"):
        return env.unwrapped._env.physics
    if hasattr(env, "physics"):
        return env.physics
    raise AttributeError("Cannot extract dm_control physics from env")


def _find_hip_actuator_index(env) -> int:
    """Find the index of the right hip (rfemurry) actuator in dm_control's action array."""
    try:
        physics = _get_physics(env)
        for i in range(physics.model.nu):
            name = physics.model.id2name(i, "actuator")
            if name == _RIGHT_HIP_ACTUATOR:
                return i
    except Exception:
        pass
    return _RIGHT_HIP_IDX_FALLBACK


def _hip_angle_to_ctrl(angle_flexion_rad: float, env=None) -> float:
    """Convert a hip flexion angle (radians) to the normalized [-1, 1] ctrl value.

    Positive = thigh forward (flexion), negative = thigh back (extension).
    """
    low, high = _HIP_RANGE_RAD

    if env is not None:
        try:
            physics = _get_physics(env)
            hip_idx = _find_hip_actuator_index(env)
            jnt_id = physics.model.actuator_trnid[hip_idx, 0]
            jnt_range = physics.model.jnt_range[jnt_id]
            low, high = float(jnt_range[0]), float(jnt_range[1])
        except Exception:
            pass

    angle_clamped = max(low, min(high, angle_flexion_rad))
    if abs(high - low) < 1e-10:
        return 0.0
    ctrl = 2.0 * (angle_clamped - low) / (high - low) - 1.0
    return float(np.clip(ctrl, -1.0, 1.0))


def _find_knee_actuator_index(env) -> int:
    """Find the index of the right knee actuator in dm_control's action array.

    The CMU humanoid has 56 actuators sorted alphabetically.  The right knee
    actuator is named ``rtibiarx`` and sits at index 44.  We verify by
    inspecting the physics model and fall back to the known index.
    """
    try:
        physics = _get_physics(env)
        for i in range(physics.model.nu):
            name = physics.model.id2name(i, "actuator")
            if name == _RIGHT_KNEE_ACTUATOR:
                return i
    except Exception:
        pass
    # Known alphabetical index for rtibiarx in the CMU humanoid V2020
    return _RIGHT_KNEE_IDX_FALLBACK


def _knee_angle_to_ctrl(angle_flexion_rad: float, env=None) -> float:
    """Convert a knee flexion angle (radians) to the normalized [-1, 1] ctrl value.

    The CMU humanoid V2020 uses position actuators with normalised controls
    in [-1, 1].  The joint range for ``rtibiarx`` is [0.01, 2.967] rad.
    We read the range from the physics model when available, otherwise use
    the known constant.
    """
    low, high = _KNEE_RANGE_RAD

    # Try to read the actual range from the physics model
    if env is not None:
        try:
            physics = _get_physics(env)
            knee_idx = _find_knee_actuator_index(env)
            jnt_id = physics.model.actuator_trnid[knee_idx, 0]
            jnt_range = physics.model.jnt_range[jnt_id]
            low, high = float(jnt_range[0]), float(jnt_range[1])
        except Exception:
            pass

    angle_clamped = max(low, min(high, angle_flexion_rad))
    if abs(high - low) < 1e-10:
        return 0.0
    ctrl = 2.0 * (angle_clamped - low) / (high - low) - 1.0
    return float(np.clip(ctrl, -1.0, 1.0))


# ── Policy loading ───────────────────────────────────────────────────────────

def _download_model(model_dir: str | Path) -> Path:
    """Download the default multi-clip policy from HuggingFace if needed.

    The HF repo ``microsoft/mocapact-models`` hosts a tarball
    ``multiclip_policy.tar.gz`` that extracts to::

        multiclip_policy/
            full_dataset/model/model.ckpt         (all clips — preferred)
            locomotion_dataset/model/model.ckpt   (walking-focused — fallback)
    """
    model_dir = Path(model_dir)

    # Check if either variant already exists
    for rel in (_DEFAULT_MULTI_CLIP_CKPT, _DEFAULT_MULTI_CLIP_CKPT_ALT):
        p = model_dir / rel
        if p.exists():
            return p

    print(f"[MoCapAct] Downloading multi-clip policy to {model_dir} ...")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Try huggingface_hub first
    try:
        from huggingface_hub import hf_hub_download
        tarball = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_HF_TARBALL,
            cache_dir=str(model_dir / ".cache"),
            local_dir=str(model_dir),
        )
        import tarfile
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(path=str(model_dir))
        for rel in (_DEFAULT_MULTI_CLIP_CKPT, _DEFAULT_MULTI_CLIP_CKPT_ALT):
            p = model_dir / rel
            if p.exists():
                return p
    except ImportError:
        pass
    except Exception as exc:
        print(f"[MoCapAct] huggingface_hub download failed: {exc}")

    # Fallback: huggingface-cli
    try:
        subprocess.check_call(
            ["huggingface-cli", "download", _HF_REPO, _HF_TARBALL,
             "--local-dir", str(model_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        tarball_path = model_dir / _HF_TARBALL
        if tarball_path.exists():
            import tarfile
            with tarfile.open(str(tarball_path), "r:gz") as tf:
                tf.extractall(path=str(model_dir))
        for rel in (_DEFAULT_MULTI_CLIP_CKPT, _DEFAULT_MULTI_CLIP_CKPT_ALT):
            p = model_dir / rel
            if p.exists():
                return p
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not download MoCapAct model.  Please download manually:\n"
        f"  pip install huggingface_hub\n"
        f"  huggingface-cli download {_HF_REPO} {_HF_TARBALL} --local-dir {model_dir}\n"
        f"  cd {model_dir} && tar xzf {_HF_TARBALL}"
    )


def load_multi_clip_policy(
    checkpoint_path: Optional[str | Path] = None,
    model_dir: str | Path = "mocapact_models",
    device: str = "cpu",
):
    """Load a MoCapAct multi-clip (NPMP) policy.

    Parameters
    ----------
    checkpoint_path :
        Direct path to a ``best_model.ckpt``.  If None, downloads from HF.
    model_dir :
        Directory for downloaded models (only used if checkpoint_path is None).
    device :
        PyTorch device string.

    Returns
    -------
    policy
        A ``NpmpPolicy`` instance with ``.predict(obs, state, deterministic)``
        and ``.initial_state(deterministic)`` methods.
    """
    if not _MOCAPACT_AVAILABLE:
        raise RuntimeError(
            "MoCapAct is required.  Install with:\n"
            "  pip install mocapact dm_control stable-baselines3\n"
            f"Import error was: {_MOCAPACT_IMPORT_ERROR}"
        )

    if checkpoint_path is None:
        checkpoint_path = _download_model(model_dir)
    else:
        checkpoint_path = Path(checkpoint_path)

    print(f"[MoCapAct] Loading multi-clip policy from {checkpoint_path}")
    # PyTorch 2.6+ made weights_only=True the default. Gym/dm_control objects
    # embedded in MoCapAct checkpoints are not in the safe globals list, so
    # loading fails. Register all gym.spaces classes as safe globals (the
    # officially supported PyTorch 2.6+ approach), then also patch
    # torch.serialization.load as a belt-and-suspenders fallback in case
    # PyTorch Lightning resolves the reference before our patch.
    import torch as _torch
    import torch.serialization as _ts

    # Register gym.spaces types as safe
    try:
        import gym.spaces as _gs
        _safe = [
            getattr(_gs, _n)
            for _n in dir(_gs)
            if isinstance(getattr(_gs, _n), type)
        ]
        if hasattr(_ts, "add_safe_globals"):
            _ts.add_safe_globals(_safe)
    except Exception:
        pass

    # Belt-and-suspenders: patch serialization.load so weights_only=False
    # is the default regardless of how PL calls it.
    _orig_sl = _ts.load
    def _permissive_sl(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_sl(*args, **kwargs)
    _ts.load = _permissive_sl  # type: ignore[assignment]
    _orig_load = _torch.load
    def _permissive_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    _torch.load = _permissive_load  # type: ignore[assignment]
    try:
        policy = npmp_model.NpmpPolicy.load_from_checkpoint(
            str(checkpoint_path), map_location=device,
        )
    finally:
        _ts.load = _orig_sl  # type: ignore[assignment]
        _torch.load = _orig_load  # type: ignore[assignment]
    policy.eval()
    return policy


def load_clip_expert(
    expert_path: str | Path,
    device: str = "cpu",
):
    """Load a MoCapAct clip-snippet expert (SB3 PPO).

    Parameters
    ----------
    expert_path :
        Path to the expert model directory (contains ``best_model.zip`` and
        ``vecnormalize.pkl``).
    device :
        PyTorch device string.

    Returns
    -------
    expert
        A Stable-Baselines3 model with ``.predict(obs, deterministic)`` method.
    """
    if not _MOCAPACT_AVAILABLE:
        raise RuntimeError(
            "MoCapAct is required.\n"
            + (f"Import error was: {_MOCAPACT_IMPORT_ERROR}" if _MOCAPACT_IMPORT_ERROR else "")
        )

    return mocapact_utils.load_policy(
        str(expert_path),
        mocapact_obs.TIME_INDEX_OBSERVABLES,
        device=device,
    )


# ── Environment creation ─────────────────────────────────────────────────────

def create_walking_env(
    dataset=None,
    ref_steps: Optional[Tuple[int, ...]] = None,
    policy=None,
    clip_id: Optional[str] = None,
):
    """Create a MoCapAct tracking Gym environment for walking evaluation.

    Parameters
    ----------
    dataset :
        A ``ClipCollection`` or subset constant.  Defaults to
        ``cmu_subsets.LOCOMOTION_SMALL`` (~40 min of walking clips).
        Ignored when *clip_id* is set.
    ref_steps :
        Lookahead steps for the reference pose.  If None and *policy*
        is provided, uses ``policy.ref_steps``.  Otherwise ``(1,2,3,4,5)``.
    policy :
        If provided, reads ``ref_steps`` from the policy to ensure
        compatibility with its observation expectations.
    clip_id :
        If provided, creates a single-clip ``ClipCollection`` for this
        CMU clip (e.g. ``"CMU_009_12"``).  The policy will track this
        specific motion instead of a random clip from the default dataset.

    Returns
    -------
    env
        A ``MocapTrackingGymEnv`` instance (Gym-compatible).
    """
    if not _MOCAPACT_AVAILABLE or not _DM_CONTROL_AVAILABLE:
        raise RuntimeError(
            "MoCapAct + dm_control are required.  Install with:\n"
            "  pip install mocapact dm_control stable-baselines3"
        )

    if clip_id is not None:
        dataset = _make_clip_collection(clip_id)
    elif dataset is None:
        dataset = cmu_subsets.LOCOMOTION_SMALL

    if ref_steps is None:
        if policy is not None and hasattr(policy, "ref_steps"):
            ref_steps = tuple(policy.ref_steps)
        else:
            ref_steps = (1, 2, 3, 4, 5)

    env = mocapact_tracking.MocapTrackingGymEnv(
        dataset=dataset,
        ref_steps=ref_steps,
    )
    return env


# ── Shared step helpers ──────────────────────────────────────────────────────

def _collect_step_metrics(
    metrics: "MoCapActEvalMetrics",
    physics,
    pred_deg: float,
    ref_flex_deg,
    policy_knee_ctrl: float,
    t: int,
) -> None:
    """Append per-step data to *metrics* (CoM, knee, contacts, fall detection)."""
    try:
        com_h = float(physics.data.subtree_com[0, 2])
    except Exception:
        com_h = float(physics.center_of_mass()[2])
    metrics.com_height.append(com_h)
    metrics.pred_knee.append(pred_deg)

    if ref_flex_deg is not None:
        metrics.ref_knee.append(float(ref_flex_deg[min(t, len(ref_flex_deg) - 1)]))
    else:
        low, high = _KNEE_RANGE_RAD
        natural_rad = low + (policy_knee_ctrl + 1.0) * (high - low) / 2.0
        metrics.ref_knee.append(float(np.degrees(natural_rad)))

    try:
        ncon = physics.data.ncon
        r_contact = l_contact = False
        for c in range(ncon):
            geom1 = int(physics.data.contact[c].geom1)
            geom2 = int(physics.data.contact[c].geom2)
            try:
                n1 = physics.model.id2name(geom1, "geom")
                n2 = physics.model.id2name(geom2, "geom")
            except Exception:
                continue
            pair = n1 + " " + n2
            if not r_contact and ("rfoot" in pair or "rtoes" in pair):
                r_contact = True
            if not l_contact and ("lfoot" in pair or "ltoes" in pair):
                l_contact = True
            if r_contact and l_contact:
                break
        if r_contact:
            metrics.right_contact_frames.append(t)
        if l_contact:
            metrics.left_contact_frames.append(t)
    except Exception:
        pass

    if (not metrics.fall_detected) and com_h < FALL_HEIGHT_THRESHOLD:
        metrics.fall_detected = True
        metrics.fall_frame = t


def _build_sim_result(
    metrics: "MoCapActEvalMetrics",
    match_info,
    fps: float,
    mode: str = "mocapact_policy",
) -> dict:
    """Convert *metrics* into the result dict used by the evaluation pipeline."""
    out = metrics.to_dict()
    out["mode"] = mode
    if match_info is not None:
        out["matched_clip"] = match_info["clip_id"]
        out["matched_category"] = match_info.get("category", "unknown")
    out["fall_prediction"] = {
        "predicted_fall": metrics.fall_detected,
        "time_to_fall_s": float(metrics.fall_frame / fps) if metrics.fall_detected else float("inf"),
        "confidence": 1.0 if metrics.fall_detected else 0.0,
        "com_velocity": 0.0,
        "com_acceleration": 0.0,
        "min_com": float(min(metrics.com_height)) if metrics.com_height else 1.0,
        "trend_slope": 0.0,
    }
    return out


# ── Core simulation loop ─────────────────────────────────────────────────────

def simulate_mocapact_prosthetic(
    predicted_knee: np.ndarray,
    predicted_thigh: Optional[np.ndarray] = None,
    policy=None,
    policy_checkpoint: Optional[str | Path] = None,
    model_dir: str | Path = "mocapact_models",
    eval_seconds: float = DEFAULT_EVAL_SECONDS,
    fps: float = SIM_FPS,
    device: str = "cpu",
    reference_knee: Optional[np.ndarray] = None,
    use_gui: bool = False,
    match_info: Optional[dict] = None,
) -> dict:
    """Run a MoCapAct-based prosthetic simulation.

    The policy controls all joints except the right knee and (optionally) the
    right hip, which are overridden with ``predicted_knee`` and
    ``predicted_thigh`` respectively.

    Parameters
    ----------
    predicted_knee :
        Model-predicted right knee angle (included-angle, degrees).
    predicted_thigh :
        Recorded right thigh angle (included-angle, degrees).  When provided,
        overrides the right hip (rfemurry) actuator so the thigh swing comes
        entirely from your IMU data rather than the MoCapAct policy.
    policy :
        Pre-loaded MoCapAct policy.  If None, loads from checkpoint.
    policy_checkpoint :
        Path to policy checkpoint (used if ``policy`` is None).
    model_dir :
        Directory for downloaded models.
    eval_seconds :
        Duration of the simulation in seconds.
    fps :
        Control frequency (Hz).  MoCapAct default is ~30 Hz.
    device :
        PyTorch device.
    reference_knee :
        Ground-truth knee angle (included-angle, degrees) for error metrics.
        If None, the policy's natural knee angle is used as reference.
    use_gui :
        Launch interactive viewer (requires display + mujoco viewer).
    match_info :
        Dict from ``resolve_clip_from_match``.  When provided the environment
        is initialised with the **specific CMU clip** that the DTW match came
        from, so the policy walks the same motion that was recorded during the
        EMG session.  When ``None`` a random clip from ``LOCOMOTION_SMALL``
        is used.

    Returns
    -------
    dict
        Evaluation metrics dict compatible with ``prosthetic_sim.py`` output.
    """
    if not _MOCAPACT_AVAILABLE or not _DM_CONTROL_AVAILABLE:
        raise RuntimeError(
            "MoCapAct + dm_control required.  Install with:\n"
            "  pip install mocapact dm_control stable-baselines3\n"
            + (f"Import error was: {_MOCAPACT_IMPORT_ERROR}" if _MOCAPACT_IMPORT_ERROR else "")
        )

    # ── Load policy ───────────────────────────────────────────────────
    if policy is None:
        policy = load_multi_clip_policy(
            checkpoint_path=policy_checkpoint,
            model_dir=model_dir,
            device=device,
        )

    # ── Create environment ────────────────────────────────────────────
    clip_id = match_info["clip_id"] if match_info else None
    env = create_walking_env(policy=policy, clip_id=clip_id)
    knee_idx = _find_knee_actuator_index(env)
    hip_idx = _find_hip_actuator_index(env)

    # ── Prepare predicted knee signal ─────────────────────────────────
    pred_flex_deg = _included_to_flexion(predicted_knee)
    pred_flex_rad = np.radians(pred_flex_deg)

    if reference_knee is not None:
        ref_flex_deg = _included_to_flexion(reference_knee)
    else:
        ref_flex_deg = None

    # ── Prepare predicted thigh/hip signal ────────────────────────────
    if predicted_thigh is not None:
        pred_hip_rad = np.radians(_included_to_flexion(np.asarray(predicted_thigh, dtype=np.float64)))
    else:
        pred_hip_rad = None

    # ── Determine frame count ─────────────────────────────────────────
    max_steps = int(eval_seconds * fps)
    T = min(max_steps, len(pred_flex_deg))

    # ── Simulation loop ───────────────────────────────────────────────
    metrics = MoCapActEvalMetrics.empty()
    obs = env.reset()

    # Get the physics handle for metric collection
    physics = _get_physics(env)

    # ── Open interactive MuJoCo viewer (use_gui=True) ─────────────────
    _viewer = None
    if use_gui:
        try:
            import mujoco.viewer as _mv
            _viewer = _mv.launch_passive(physics.model, physics.data)
            print("[MoCapAct] Viewer opened — close the window to stop early.")
        except Exception as _ve:
            print(f"[MoCapAct] Could not open mujoco viewer: {_ve}")
            # Fall back to dm_control's own viewer (blocking — runs in a thread)
            try:
                from dm_control import viewer as _dmv
                import threading as _threading
                _inner_env = env.unwrapped._env if hasattr(env, "unwrapped") else env
                _dmv_thread = _threading.Thread(
                    target=_dmv.launch, args=(_inner_env,), daemon=True
                )
                _dmv_thread.start()
            except Exception as _ve2:
                print(f"[MoCapAct] dm_control viewer also unavailable: {_ve2}")

    # Initialise the policy's recurrent state (embedding for NPMP)
    is_npmp = hasattr(policy, "initial_state")
    if is_npmp:
        embed = policy.initial_state(deterministic=False)
    else:
        embed = None

    for t in tqdm(range(T), desc="MoCapAct sim", unit="step", leave=False):
        # Stop if the viewer window was closed
        if _viewer is not None and not _viewer.is_running():
            break

        # ── Policy proposes action for all 56 joints ──────────────────
        if is_npmp:
            action, embed = policy.predict(obs, state=embed, deterministic=False)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        # Remember the policy's intended knee action before we override it
        policy_knee_ctrl = float(action[knee_idx])

        # ── Override right knee with prosthetic prediction ────────────
        knee_ctrl = _knee_angle_to_ctrl(pred_flex_rad[t], env)
        action[knee_idx] = knee_ctrl

        # ── Override right hip with recorded thigh angle ──────────────
        if pred_hip_rad is not None:
            hip_ctrl = _hip_angle_to_ctrl(pred_hip_rad[min(t, len(pred_hip_rad) - 1)], env)
            action[hip_idx] = hip_ctrl

        # ── Step environment ──────────────────────────────────────────
        obs, reward, done, info = env.step(action)

        # ── Sync viewer ───────────────────────────────────────────────
        if _viewer is not None and _viewer.is_running():
            _viewer.sync()

        _collect_step_metrics(metrics, physics, float(pred_flex_deg[t]), ref_flex_deg, policy_knee_ctrl, t)

        if done:
            break

    if _viewer is not None:
        try:
            _viewer.close()
        except Exception:
            pass

    env.close()
    return _build_sim_result(metrics, match_info, fps)


# ── Pipeline-compatible wrapper ───────────────────────────────────────────────

# Module-level policy cache to avoid reloading for every call
_cached_policy = None
_cached_policy_path = None


def simulate_prosthetic_walking_mocapact(
    predicted_knee: np.ndarray,
    sample_thigh_right: Optional[np.ndarray] = None,
    policy_checkpoint: Optional[str | Path] = None,
    model_dir: str | Path = "mocapact_models",
    eval_seconds: float = DEFAULT_EVAL_SECONDS,
    device: str = "cpu",
    reference_knee: Optional[np.ndarray] = None,
    show_reference: bool = False,
    use_gui: bool = False,
    mocap_db: Optional[dict] = None,
    best_start: Optional[int] = None,
    **kwargs,
) -> dict:
    """Drop-in replacement for ``prosthetic_sim.simulate_prosthetic_walking``.

    When *mocap_db* and *best_start* are supplied the environment is
    initialised with the **exact CMU clip** that DTW matched against the
    EMG recording.  This way the policy walks the same motion that was
    recorded during the session, but with full physics — and the right knee
    is overridden with the EMG model's prediction.

    Parameters
    ----------
    predicted_knee :
        Right knee angle prediction (included-angle, degrees).
    sample_thigh_right :
        Recorded right thigh angle (included-angle, degrees).  Overrides the
        right hip (rfemurry) actuator so the thigh swing is driven by your IMU
        data rather than the MoCapAct policy.
    policy_checkpoint :
        Path to MoCapAct multi-clip checkpoint.
    model_dir :
        Directory for downloaded models.
    eval_seconds :
        Simulation duration in seconds.
    device :
        PyTorch device.
    reference_knee :
        Ground-truth knee angle (included-angle, degrees).
    show_reference :
        Ignored (no dual-humanoid support in MoCapAct mode).
    use_gui :
        Launch interactive viewer.
    mocap_db :
        Aggregated mocap database (from ``load_aggregated_database``).
        Must contain ``file_boundaries``.  When provided together with
        *best_start*, the matched clip is resolved and the MoCapAct
        environment tracks that specific CMU motion.
    best_start :
        Start index returned by ``find_best_match`` (at 200 Hz database
        rate).  Used together with *mocap_db*.
    **kwargs :
        Absorbs extra kwargs for compatibility (fps, save_trajectory, etc.).

    Returns
    -------
    dict
        Evaluation metrics matching ``prosthetic_sim`` output contract.
    """
    global _cached_policy, _cached_policy_path

    # Reuse cached policy if checkpoint hasn't changed
    ckpt_key = str(policy_checkpoint or model_dir)
    if _cached_policy is not None and _cached_policy_path == ckpt_key:
        policy = _cached_policy
    else:
        policy = load_multi_clip_policy(
            checkpoint_path=policy_checkpoint,
            model_dir=model_dir,
            device=device,
        )
        _cached_policy = policy
        _cached_policy_path = ckpt_key

    # Resolve the DTW match → specific CMU clip
    match_info = None
    if mocap_db is not None and best_start is not None:
        match_info = resolve_clip_from_match(
            best_start, len(predicted_knee), mocap_db,
        )

    return simulate_mocapact_prosthetic(
        predicted_knee=predicted_knee,
        predicted_thigh=sample_thigh_right,
        policy=policy,
        eval_seconds=eval_seconds,
        device=device,
        reference_knee=reference_knee,
        use_gui=use_gui,
        match_info=match_info,
    )


# ── Three-scenario comparison ────────────────────────────────────────────────

def simulate_three_scenarios_mocapact(
    gt_knee: np.ndarray,
    nominal_knee: np.ndarray,
    bad_knee: np.ndarray,
    sample_thigh_right: Optional[np.ndarray] = None,
    policy=None,
    policy_checkpoint: Optional[str | Path] = None,
    model_dir: str | Path = "mocapact_models",
    eval_seconds: float = DEFAULT_EVAL_SECONDS,
    device: str = "cpu",
    reference_knee: Optional[np.ndarray] = None,
    use_gui: bool = False,
    match_info: Optional[dict] = None,
) -> Tuple[dict, dict, dict]:
    """Run ground-truth, nominal, and bad scenarios in lock-step.

    When *use_gui* is True all three viewer windows are opened **before** any
    stepping begins, so they share the same GLFW session and avoid the WGL
    "context in use" crash that occurs when viewers are opened sequentially.

    Parameters
    ----------
    gt_knee, nominal_knee, bad_knee :
        Right knee signals for each scenario (included-angle, degrees).
    reference_knee :
        Ground-truth knee for error metrics (usually the same as *gt_knee*).

    Returns
    -------
    Tuple of (gt_result, nominal_result, bad_result) dicts.
    """
    if not _MOCAPACT_AVAILABLE or not _DM_CONTROL_AVAILABLE:
        raise RuntimeError(
            "MoCapAct + dm_control required.  Install with:\n"
            "  pip install mocapact dm_control stable-baselines3\n"
            + (f"Import error was: {_MOCAPACT_IMPORT_ERROR}" if _MOCAPACT_IMPORT_ERROR else "")
        )

    if policy is None:
        policy = load_multi_clip_policy(
            checkpoint_path=policy_checkpoint,
            model_dir=model_dir,
            device=device,
        )

    clip_id = match_info["clip_id"] if match_info else None
    envs = [create_walking_env(policy=policy, clip_id=clip_id) for _ in range(3)]

    knee_idx = _find_knee_actuator_index(envs[0])
    hip_idx = _find_hip_actuator_index(envs[0])

    def _prep(arr):
        deg = _included_to_flexion(np.asarray(arr, dtype=np.float64)).astype(np.float32)
        return deg, np.radians(deg)

    deg_arrs = [_prep(k) for k in (gt_knee, nominal_knee, bad_knee)]
    # deg_arrs[i] = (deg, rad)

    ref_flex_deg = (
        _included_to_flexion(np.asarray(reference_knee, dtype=np.float64))
        if reference_knee is not None else None
    )
    pred_hip_rad = (
        np.radians(_included_to_flexion(np.asarray(sample_thigh_right, dtype=np.float64)))
        if sample_thigh_right is not None else None
    )

    T = min(int(eval_seconds * SIM_FPS), *(len(d[0]) for d in deg_arrs))
    metrics_list = [MoCapActEvalMetrics.empty() for _ in range(3)]
    physics_list = [_get_physics(e) for e in envs]
    obss = [e.reset() for e in envs]

    # Open all viewers before stepping to keep contexts alive simultaneously
    viewers: List[Optional[object]] = [None, None, None]
    if use_gui:
        labels = ["Ground Truth", "Nominal", "Bad (delay+noise)"]
        for i, (phys, label) in enumerate(zip(physics_list, labels)):
            try:
                import mujoco.viewer as _mv
                viewers[i] = _mv.launch_passive(phys.model, phys.data)
                print(f"[MoCapAct] {label} viewer opened.")
            except Exception as exc:
                print(f"[MoCapAct] Could not open {label} viewer: {exc}")

    is_npmp = hasattr(policy, "initial_state")
    embeds = [policy.initial_state(deterministic=False) if is_npmp else None for _ in range(3)]
    dones = [False, False, False]

    for t in tqdm(range(T), desc="3-scenario sim", unit="step", leave=False):
        open_viewers = [v for v in viewers if v is not None]
        if open_viewers and all(not v.is_running() for v in open_viewers):
            break

        for i in range(3):
            if dones[i]:
                continue
            if is_npmp:
                action, embeds[i] = policy.predict(obss[i], state=embeds[i], deterministic=False)
            else:
                action, _ = policy.predict(obss[i], deterministic=True)
            policy_knee_ctrl = float(action[knee_idx])
            action[knee_idx] = _knee_angle_to_ctrl(deg_arrs[i][1][t], envs[i])
            if pred_hip_rad is not None:
                action[hip_idx] = _hip_angle_to_ctrl(
                    pred_hip_rad[min(t, len(pred_hip_rad) - 1)], envs[i]
                )
            obss[i], _, dones[i], _ = envs[i].step(action)
            _collect_step_metrics(
                metrics_list[i], physics_list[i],
                float(deg_arrs[i][0][t]), ref_flex_deg, policy_knee_ctrl, t,
            )

        for v in viewers:
            if v is not None and v.is_running():
                v.sync()

    for v in viewers:
        if v is not None:
            try:
                v.close()
            except Exception:
                pass
    for e in envs:
        e.close()

    labels_mode = ["mocapact_gt", "mocapact_nominal", "mocapact_bad"]
    results = tuple(
        _build_sim_result(m, match_info, SIM_FPS, mode)
        for m, mode in zip(metrics_list, labels_mode)
    )
    return results  # type: ignore[return-value]


# ── Availability check ───────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if MoCapAct + dm_control are importable."""
    return _MOCAPACT_AVAILABLE and _DM_CONTROL_AVAILABLE


def check_requirements() -> str:
    """Return a human-readable status of MoCapAct requirements.

    Sub-import diagnosis runs each import in a **fresh subprocess** so that
    cached sys.modules state from a previous failed import cannot mask the
    real failure.
    """
    import sys

    lines = []
    lines.append(f"dm_control:  {'OK' if _DM_CONTROL_AVAILABLE else 'MISSING (pip install dm_control)'}")
    mocapact_status = "OK" if _MOCAPACT_AVAILABLE else f"FAILED ({_MOCAPACT_IMPORT_ERROR or 'unknown error'})"
    lines.append(f"mocapact:    {mocapact_status}")
    lines.append(f"mujoco:      {'OK' if _MUJOCO_AVAILABLE else 'MISSING (pip install mujoco)'}")

    if not _MOCAPACT_AVAILABLE:
        lines.append("\nDiagnosing mocapact sub-imports (fresh subprocess per import):")
        for stmt in [
            "from mocapact.distillation import model",
            "from mocapact.envs import tracking",
            "from mocapact.sb3 import utils",
            "from mocapact import observables",
        ]:
            result = subprocess.run(
                [sys.executable, "-c", stmt],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                lines.append(f"  OK      {stmt}")
            else:
                err = (result.stderr or result.stdout).strip()
                lines.append(f"  FAILED  {stmt}\n          {err}")

    return "\n".join(lines)
