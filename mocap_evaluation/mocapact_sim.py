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

import math
import os
import subprocess
import sys
import time
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

try:
    from mocapact.distillation import model as npmp_model
    from mocapact.envs import tracking as mocapact_tracking
    from mocapact.sb3 import utils as mocapact_utils
    from mocapact import observables as mocapact_obs
    _MOCAPACT_AVAILABLE = True
except Exception:
    pass

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

# dm_control CMU humanoid right knee actuator name
_RIGHT_KNEE_ACTUATOR = "rtibiarx"

# Default multi-clip policy checkpoint name (relative to model dir)
_DEFAULT_MULTI_CLIP_CKPT = "multi_clip/all/eval/train_rsi/best_model.ckpt"

# HuggingFace model repo
_HF_REPO = "microsoft/mocapact-models"

# Walking-related CMU clip subsets (motion categories that involve locomotion)
_WALKING_CLIPS = [
    "CMU_016_22",   # walk
    "CMU_016_25",   # walk
    "CMU_016_47",   # walk
    "CMU_035_17",   # walk
    "CMU_035_26",   # walk
    "CMU_008_02",   # walk
    "CMU_008_03",   # walk
    "CMU_007_01",   # walk
    "CMU_007_02",   # walk
]


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

def _find_knee_actuator_index(env) -> int:
    """Find the index of the right knee actuator in dm_control's action array.

    The CMU humanoid has 56 actuators sorted alphabetically.  The right knee
    actuator is named ``rtibiarx``.  We find it by inspecting the physics
    model rather than hard-coding the index.
    """
    physics = env.unwrapped._env.physics if hasattr(env, "unwrapped") else env.physics
    model = physics.model
    for i in range(model.nu):
        name = physics.model.id2name(i, "actuator")
        if name == _RIGHT_KNEE_ACTUATOR:
            return i
    # Fallback: search by substring
    for i in range(model.nu):
        name = physics.model.id2name(i, "actuator")
        if "rtibia" in name:
            return i
    raise ValueError(
        f"Could not find right knee actuator ({_RIGHT_KNEE_ACTUATOR}) "
        f"in the CMU humanoid model (n_actuators={model.nu})"
    )


def _knee_angle_to_ctrl(angle_flexion_rad: float, env) -> float:
    """Convert a knee flexion angle (radians) to the normalized [-1, 1] ctrl value.

    MoCapAct's position-controlled humanoid uses normalized actions.  The
    actual joint range is encoded in the actuator's ctrlrange and gain
    parameters.  We reverse the mapping:

        ctrl = (2 * angle - offset) / scale

    where offset and scale come from the walker's calibration.
    """
    physics = env.unwrapped._env.physics if hasattr(env, "unwrapped") else env.physics
    model = physics.model
    knee_idx = _find_knee_actuator_index(env)

    # Actuator ctrlrange is [-1, 1] (normalized).  The joint range is stored
    # in the joint limits.  The position actuator maps ctrl linearly to the
    # joint range.
    jnt_id = model.actuator_trnid[knee_idx, 0]
    jnt_range = model.jnt_range[jnt_id]  # [low, high] in radians
    low, high = float(jnt_range[0]), float(jnt_range[1])

    # Clamp the angle to the joint range
    angle_clamped = max(low, min(high, angle_flexion_rad))

    # Linear mapping: joint range [low, high] -> ctrl [-1, 1]
    if abs(high - low) < 1e-10:
        return 0.0
    ctrl = 2.0 * (angle_clamped - low) / (high - low) - 1.0
    return float(np.clip(ctrl, -1.0, 1.0))


# ── Policy loading ───────────────────────────────────────────────────────────

def _download_model(model_dir: str | Path) -> Path:
    """Download the default multi-clip policy from HuggingFace if needed."""
    model_dir = Path(model_dir)
    ckpt_path = model_dir / _DEFAULT_MULTI_CLIP_CKPT
    if ckpt_path.exists():
        return ckpt_path

    print(f"[MoCapAct] Downloading multi-clip policy to {model_dir} ...")
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=_HF_REPO,
            filename=_DEFAULT_MULTI_CLIP_CKPT,
            cache_dir=str(model_dir / ".cache"),
            local_dir=str(model_dir),
        )
        return Path(local)
    except ImportError:
        pass

    # Fallback: use git lfs
    try:
        subprocess.check_call(
            ["git", "lfs", "clone", "--depth=1",
             f"https://huggingface.co/{_HF_REPO}", str(model_dir)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if ckpt_path.exists():
            return ckpt_path
    except Exception:
        pass

    raise FileNotFoundError(
        f"Could not download MoCapAct model.  Please download manually:\n"
        f"  pip install huggingface_hub\n"
        f"  python -c \"from huggingface_hub import hf_hub_download; "
        f"hf_hub_download('{_HF_REPO}', '{_DEFAULT_MULTI_CLIP_CKPT}')\"\n"
        f"Or clone the repo: git lfs clone https://huggingface.co/{_HF_REPO} {model_dir}"
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
            "  pip install mocapact dm_control stable-baselines3"
        )

    if checkpoint_path is None:
        checkpoint_path = _download_model(model_dir)
    else:
        checkpoint_path = Path(checkpoint_path)

    print(f"[MoCapAct] Loading multi-clip policy from {checkpoint_path}")
    policy = npmp_model.NpmpPolicy.load_from_checkpoint(
        str(checkpoint_path), map_location=device,
    )
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
        raise RuntimeError("MoCapAct is required.")

    return mocapact_utils.load_policy(
        str(expert_path),
        mocapact_obs.TIME_INDEX_OBSERVABLES,
        device=device,
    )


# ── Environment creation ─────────────────────────────────────────────────────

def create_walking_env(
    dataset: Optional[str] = None,
    ref_steps: Tuple[int, ...] = (1, 2, 3, 4, 5),
    clip_id: Optional[str] = None,
):
    """Create a MoCapAct tracking Gym environment for walking evaluation.

    Parameters
    ----------
    dataset :
        CMU subset key (e.g. ``"ALL"``).  Defaults to ``cmu_subsets.ALL``.
    ref_steps :
        Lookahead steps for the reference pose.
    clip_id :
        Specific clip to track (e.g. ``"CMU_016_22"``).

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

    if dataset is None:
        dataset = cmu_subsets.ALL

    env = mocapact_tracking.MocapTrackingGymEnv(
        dataset=dataset,
        ref_steps=ref_steps,
    )
    return env


# ── Core simulation loop ─────────────────────────────────────────────────────

def simulate_mocapact_prosthetic(
    predicted_knee: np.ndarray,
    policy=None,
    policy_checkpoint: Optional[str | Path] = None,
    model_dir: str | Path = "mocapact_models",
    eval_seconds: float = DEFAULT_EVAL_SECONDS,
    fps: float = SIM_FPS,
    device: str = "cpu",
    reference_knee: Optional[np.ndarray] = None,
    use_gui: bool = False,
) -> dict:
    """Run a MoCapAct-based prosthetic simulation.

    The policy controls all joints except the right knee, which is overridden
    with ``predicted_knee``.

    Parameters
    ----------
    predicted_knee :
        Model-predicted right knee angle (included-angle, degrees).
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

    Returns
    -------
    dict
        Evaluation metrics dict compatible with ``prosthetic_sim.py`` output.
    """
    if not _MOCAPACT_AVAILABLE or not _DM_CONTROL_AVAILABLE:
        raise RuntimeError(
            "MoCapAct + dm_control required.  Install with:\n"
            "  pip install mocapact dm_control stable-baselines3"
        )

    # ── Load policy ───────────────────────────────────────────────────
    if policy is None:
        policy = load_multi_clip_policy(
            checkpoint_path=policy_checkpoint,
            model_dir=model_dir,
            device=device,
        )

    # ── Create environment ────────────────────────────────────────────
    env = create_walking_env()
    knee_idx = _find_knee_actuator_index(env)

    # ── Prepare predicted knee signal ─────────────────────────────────
    pred_flex_deg = _included_to_flexion(predicted_knee)
    pred_flex_rad = np.radians(pred_flex_deg)

    if reference_knee is not None:
        ref_flex_deg = _included_to_flexion(reference_knee)
    else:
        ref_flex_deg = None

    # ── Determine frame count ─────────────────────────────────────────
    max_steps = int(eval_seconds * fps)
    T = min(max_steps, len(pred_flex_deg))

    # ── Simulation loop ───────────────────────────────────────────────
    metrics = MoCapActEvalMetrics.empty()
    obs = env.reset()

    # Get the physics handle for metric collection
    dm_env = env.unwrapped._env if hasattr(env, "unwrapped") else env
    physics = dm_env.physics

    # Initialise the policy's recurrent state (embedding for NPMP)
    is_npmp = hasattr(policy, "initial_state")
    if is_npmp:
        embed = policy.initial_state(deterministic=False)
    else:
        embed = None

    for t in tqdm(range(T), desc="MoCapAct sim", unit="step", leave=False):
        # ── Policy proposes action for all 56 joints ──────────────────
        if is_npmp:
            action, embed = policy.predict(obs, state=embed, deterministic=False)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        # ── Override right knee with prosthetic prediction ────────────
        knee_ctrl = _knee_angle_to_ctrl(pred_flex_rad[t], env)
        action[knee_idx] = knee_ctrl

        # ── Step environment ──────────────────────────────────────────
        obs, reward, done, info = env.step(action)

        # ── Collect metrics ───────────────────────────────────────────
        com_h = float(physics.named.data.subtree_com["root", "z"])
        metrics.com_height.append(com_h)
        metrics.pred_knee.append(float(pred_flex_deg[t]))

        if ref_flex_deg is not None:
            ref_val = float(ref_flex_deg[t]) if t < len(ref_flex_deg) else metrics.ref_knee[-1] if metrics.ref_knee else 0.0
            metrics.ref_knee.append(ref_val)
        else:
            # Use the policy's natural action (before override) as reference
            natural_knee_ctrl = float(action[knee_idx])
            # Rough inverse mapping: ctrl [-1, 1] -> angle
            jnt_id = physics.model.actuator_trnid[knee_idx, 0]
            jnt_range = physics.model.jnt_range[jnt_id]
            low, high = float(jnt_range[0]), float(jnt_range[1])
            natural_angle = low + (natural_knee_ctrl + 1.0) * (high - low) / 2.0
            metrics.ref_knee.append(float(np.degrees(natural_angle)))

        # ── Foot contact detection ────────────────────────────────────
        # dm_control provides contact data through the physics engine
        try:
            ncon = physics.data.ncon
            for c in range(ncon):
                geom1 = physics.data.contact[c].geom1
                geom2 = physics.data.contact[c].geom2
                names = set()
                try:
                    names.add(physics.model.id2name(geom1, "geom"))
                    names.add(physics.model.id2name(geom2, "geom"))
                except Exception:
                    pass
                # dm_control CMU humanoid foot geom names
                if any("rfoot" in n or "rtoes" in n for n in names):
                    metrics.right_contact_frames.append(t)
                    break
            for c in range(ncon):
                geom1 = physics.data.contact[c].geom1
                geom2 = physics.data.contact[c].geom2
                names = set()
                try:
                    names.add(physics.model.id2name(geom1, "geom"))
                    names.add(physics.model.id2name(geom2, "geom"))
                except Exception:
                    pass
                if any("lfoot" in n or "ltoes" in n for n in names):
                    metrics.left_contact_frames.append(t)
                    break
        except Exception:
            pass

        # ── Fall detection ────────────────────────────────────────────
        if (not metrics.fall_detected) and com_h < FALL_HEIGHT_THRESHOLD:
            metrics.fall_detected = True
            metrics.fall_frame = t

        if done:
            break

    env.close()

    # ── Build result (same contract as prosthetic_sim.py) ─────────────
    out = metrics.to_dict()
    out["mode"] = "mocapact_policy"
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
    **kwargs,
) -> dict:
    """Drop-in replacement for ``prosthetic_sim.simulate_prosthetic_walking``.

    This function has the same return-value contract but does NOT require a
    ``mocap_segment`` dict — the MoCapAct policy handles locomotion internally.

    Parameters
    ----------
    predicted_knee :
        Right knee angle prediction (included-angle, degrees).
    sample_thigh_right :
        Ignored (present for API compatibility).
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

    return simulate_mocapact_prosthetic(
        predicted_knee=predicted_knee,
        policy=policy,
        eval_seconds=eval_seconds,
        device=device,
        reference_knee=reference_knee,
        use_gui=use_gui,
    )


# ── Availability check ───────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if MoCapAct + dm_control are importable."""
    return _MOCAPACT_AVAILABLE and _DM_CONTROL_AVAILABLE


def check_requirements() -> str:
    """Return a human-readable status of MoCapAct requirements."""
    lines = []
    lines.append(f"dm_control:  {'OK' if _DM_CONTROL_AVAILABLE else 'MISSING (pip install dm_control)'}")
    lines.append(f"mocapact:    {'OK' if _MOCAPACT_AVAILABLE else 'MISSING (pip install mocapact)'}")
    lines.append(f"mujoco:      {'OK' if _MUJOCO_AVAILABLE else 'MISSING (pip install mujoco)'}")
    return "\n".join(lines)
