from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .reference_bank import CMU_MOCAP_JOINTS

# NumPy 2.x removed `np.infty`; mocapact 0.1 still references it.
if not hasattr(np, "infty"):
    np.infty = np.inf  # type: ignore[attr-defined]

# Fall thresholds (meters). These are tuned for the CMU humanoid walker model.
FALL_Z_THRESHOLD_M = 0.65
HARD_FALL_Z_THRESHOLD_M = 0.45
FALL_UPRIGHT_THRESHOLD = 0.55


@dataclass(frozen=True)
class OverrideSpec:
    knee_actuator: str
    knee_sign: float
    knee_offset_deg: float


@dataclass(frozen=True)
class RefStability:
    fall_step: int
    min_root_z_m: float
    min_upright: float
    min_balance_margin_m: float
    outside_support_frac: float
    max_outward_com_speed_mps: float
    predicted_fall_risk: float
    predicted_likely_fall: bool


def load_multiclip_policy(ckpt_path: str, *, device: str = "cpu") -> Any:
    """Load MoCapAct's multi-clip (distillation) policy from a Lightning checkpoint."""
    try:
        from mocapact import utils as mocap_utils  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing mocapact. Install `mocapact` to load the multi-clip policy.") from e

    import os.path as osp

    ckpt_path = str(ckpt_path)

    is_in_eval_dir = osp.exists(osp.join(osp.dirname(osp.dirname(osp.dirname(ckpt_path))), "model_constructor.txt"))
    model_constructor_path = (
        osp.join(osp.dirname(osp.dirname(osp.dirname(ckpt_path))), "model_constructor.txt")
        if is_in_eval_dir
        else osp.join(osp.dirname(osp.dirname(ckpt_path)), "model_constructor.txt")
    )

    try:
        with open(model_constructor_path, "r", encoding="utf-8") as f:
            model_cls = mocap_utils.str_to_callable(f.readline().strip())
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find model_constructor.txt next to checkpoint: {model_constructor_path}") from e

    import torch

    orig_torch_load = torch.load

    def _torch_load_compat(*args: Any, **kwargs: Any) -> Any:
        # PyTorch 2.6 defaults weights_only=True, which breaks older Lightning checkpoints.
        if "weights_only" not in kwargs or kwargs["weights_only"] is None:
            kwargs["weights_only"] = False
        return orig_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat  # type: ignore[assignment]
    try:
        policy = model_cls.load_from_checkpoint(ckpt_path, map_location=str(device))
    finally:
        torch.load = orig_torch_load  # type: ignore[assignment]

    def _to_gymnasium_space(space: Any) -> Any:
        # SB3>=2 uses gymnasium; MoCapAct checkpoints often store gym spaces.
        try:
            import gym  # type: ignore
            import gymnasium  # type: ignore
        except Exception:
            return space

        if isinstance(space, gym.spaces.Dict):
            return gymnasium.spaces.Dict({k: _to_gymnasium_space(v) for k, v in space.spaces.items()})
        if isinstance(space, gym.spaces.Box):
            return gymnasium.spaces.Box(
                low=np.asarray(space.low),
                high=np.asarray(space.high),
                shape=space.shape,
                dtype=space.dtype,
            )
        if isinstance(space, gym.spaces.Discrete):
            return gymnasium.spaces.Discrete(int(space.n))
        return space

    try:
        policy.eval()
    except Exception:
        pass
    try:
        for p in policy.parameters():
            p.requires_grad_(False)
    except Exception:
        pass

    # Rebuild spaces to avoid issues with pickled gym spaces (and to match SB3>=2's gymnasium expectation).
    try:
        policy.observation_space = _to_gymnasium_space(getattr(policy, "observation_space", None))
        policy.action_space = _to_gymnasium_space(getattr(policy, "action_space", None))
    except Exception:
        pass

    # MoCapAct (0.1) expects an SB3<2 extract_features signature; patch for SB3>=2 if present.
    try:
        import types as py_types
        from stable_baselines3.common.policies import BaseModel as Sb3BaseModel

        def _extract_features_compat(self: Any, obs: Any) -> Any:
            try:
                return Sb3BaseModel.extract_features(self, obs, self.features_extractor)
            except TypeError:
                return Sb3BaseModel.extract_features(self, obs)

        policy.extract_features = py_types.MethodType(_extract_features_compat, policy)
    except Exception:
        pass

    return policy


def load_expert_policy(model_dir: str | Path, *, device: str = "cpu") -> Any:
    """Load a per-snippet MoCapAct expert (SB3 PPO) from an extracted model zoo directory."""
    from pathlib import Path

    model_dir = Path(model_dir)
    if not model_dir.is_absolute():
        model_dir = (Path.cwd() / model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    try:
        from mocapact import observables  # type: ignore
        from mocapact.sb3 import utils as sb3_utils  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing mocapact runtime deps required to load expert policies.") from e

    # Experts use TIME_INDEX_OBSERVABLES (proprio + time_in_clip), and do not depend on ref_steps.
    return sb3_utils.load_policy(str(model_dir), observables.TIME_INDEX_OBSERVABLES, device=str(device), seed=0)


def _reference_joint_index(joint_name: str) -> int:
    key = str(joint_name).strip()
    try:
        return int(CMU_MOCAP_JOINTS.index(key))
    except ValueError as e:
        raise KeyError(f"Unknown CMU joint name {joint_name!r}") from e


def _joint_name_from_actuator(name: str) -> str:
    return str(name).split("/")[-1]


def _to_rad(deg: np.ndarray, *, sign: float, offset_deg: float) -> np.ndarray:
    x = np.asarray(deg, dtype=np.float64).reshape(-1)
    return np.deg2rad(sign * x + float(offset_deg)).astype(np.float64)


def _patch_reference_joint_series(env: Any, *, joint_name: str, targets_rad: np.ndarray, warmup_steps: int, max_ref_step: int) -> None:
    """Patch dm_control's reference trajectory for one joint (in task._clip_reference_features['joints'])."""
    task = env._env.task  # noqa: SLF001
    joints = np.asarray(task._clip_reference_features["joints"]).copy()  # noqa: SLF001
    if joints.ndim != 2:
        raise RuntimeError(f"Unexpected joints feature shape: {joints.shape}")
    j_idx = _reference_joint_index(joint_name)
    warmup = max(0, int(warmup_steps))

    tgt = np.asarray(targets_rad, dtype=np.float64).reshape(-1)
    if tgt.size < 1:
        return
    T = int(joints.shape[0])
    n = int(tgt.shape[0])

    # Reference indexing: task._time_step starts at 0 after reset and increments by 1 per step.
    # The reference at index `t` is what the policy observes as "ref_steps=0" at time_step=t.
    for i in range(n):
        t = warmup + i
        if 0 <= t < T:
            joints[t, j_idx] = float(tgt[i])

    # Extend last value into the ref-step lookahead window.
    last = float(tgt[-1])
    for k in range(1, max(0, int(max_ref_step)) + 1):
        t = warmup + (n - 1) + k
        if 0 <= t < T:
            joints[t, j_idx] = last

    task._clip_reference_features["joints"] = joints  # noqa: SLF001


def _recompute_reference_kinematics(env: Any) -> None:
    """Recompute kinematic reference features from patched joints (keeps policy obs consistent)."""
    try:
        from dm_control.locomotion.tasks.reference_pose import utils  # type: ignore
        import mujoco  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("dm_control + mujoco are required for reference kinematics recomputation.") from e

    task = env._env.task  # noqa: SLF001
    physics = env._env.physics  # noqa: SLF001
    walker = task._walker  # noqa: SLF001

    ref = task._clip_reference_features  # noqa: SLF001
    joints = np.asarray(ref["joints"], dtype=np.float64)
    pos = np.asarray(ref["position"], dtype=np.float64)
    quat = np.asarray(ref["quaternion"], dtype=np.float64)
    vel = np.asarray(ref["velocity"], dtype=np.float64)
    angvel = np.asarray(ref["angular_velocity"], dtype=np.float64)

    T = int(joints.shape[0])
    if T < 2:
        return

    dt = float(getattr(task, "_control_timestep", 0.03))  # noqa: SLF001
    dt = float(dt if dt > 0 else 0.03)

    joints_vel = np.empty_like(joints, dtype=np.float64)
    joints_vel[0] = (joints[1] - joints[0]) / dt
    joints_vel[1:] = (joints[1:] - joints[:-1]) / dt

    # Allocate outputs using existing shapes as canonical.
    com_out = np.asarray(ref["center_of_mass"], dtype=np.float64).copy()
    ee_out = np.asarray(ref["end_effectors"], dtype=np.float64).copy()
    app_out = np.asarray(ref["appendages"], dtype=np.float64).copy()
    body_pos_out = np.asarray(ref["body_positions"], dtype=np.float64).copy()
    body_quat_out = np.asarray(ref["body_quaternions"], dtype=np.float64).copy()

    # Ensure correct shapes (T, ...).
    if com_out.shape[0] != T:
        com_out = np.empty((T, 3), dtype=np.float64)
    if ee_out.shape[0] != T:
        ee_out = np.empty((T,) + ee_out.shape[1:], dtype=np.float64)
    if app_out.shape[0] != T:
        app_out = np.empty((T,) + app_out.shape[1:], dtype=np.float64)
    if body_pos_out.shape[0] != T:
        body_pos_out = np.empty((T,) + body_pos_out.shape[1:], dtype=np.float64)
    if body_quat_out.shape[0] != T:
        body_quat_out = np.empty((T,) + body_quat_out.shape[1:], dtype=np.float64)

    props = getattr(task, "_props", None)  # noqa: SLF001
    for t in range(T):
        features_t = {
            "position": pos[t],
            "quaternion": quat[t],
            "joints": joints[t],
            "velocity": vel[t],
            "angular_velocity": angvel[t],
            "joints_velocity": joints_vel[t],
        }
        utils.set_walker_from_features(physics, walker, features_t)
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
        feats = utils.get_features(physics, walker, props=props)

        com_out[t] = np.asarray(feats["center_of_mass"], dtype=np.float64)
        ee_out[t] = np.asarray(feats["end_effectors"], dtype=np.float64)
        app_out[t] = np.asarray(feats["appendages"], dtype=np.float64)
        body_pos_out[t] = np.asarray(feats["body_positions"], dtype=np.float64)
        body_quat_out[t] = np.asarray(feats["body_quaternions"], dtype=np.float64)

    # Commit recomputed arrays back into the task.
    ref["joints_velocity"] = joints_vel.astype(np.float64)
    ref["center_of_mass"] = com_out.astype(np.float64)
    ref["end_effectors"] = ee_out.astype(np.float64)
    ref["appendages"] = app_out.astype(np.float64)
    ref["body_positions"] = body_pos_out.astype(np.float64)
    ref["body_quaternions"] = body_quat_out.astype(np.float64)


def _reinitialize_walker_from_reference(env: Any) -> None:
    try:
        from dm_control.locomotion.tasks.reference_pose import utils  # type: ignore
        import tree  # type: ignore
        import mujoco  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("dm_control + dm-tree + mujoco are required to reinitialize the walker.") from e

    task = env._env.task  # noqa: SLF001
    physics = env._env.physics  # noqa: SLF001
    t = int(getattr(task, "_time_step", 0))  # noqa: SLF001
    target = tree.map_structure(lambda x: x[t], task._clip_reference_features)  # noqa: SLF001
    utils.set_walker_from_features(physics, task._walker, target)  # noqa: SLF001
    try:
        task._update_ghost(physics)  # noqa: SLF001
    except Exception:
        pass
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)


def _refresh_gym_observation(env: Any) -> Any | None:
    """Best-effort get an observation after patching reference features."""
    try:
        from dm_control.locomotion.tasks.reference_pose import tracking  # type: ignore
    except Exception:
        tracking = None

    try:
        updater = getattr(env, "_observation_updater", None)
        if updater is None:
            return None
        try:
            updater.update()
        except Exception:
            pass
        dm_obs = updater.get_observation()
        spaces = getattr(getattr(env, "observation_space", None), "spaces", None)
        if not isinstance(dm_obs, dict) or not isinstance(spaces, dict):
            return None
        obs: dict[str, np.ndarray] = {}
        for k, sp in spaces.items():
            try:
                v = dm_obs[k]
            except Exception:
                continue
            if getattr(sp, "dtype", None) == np.uint8:
                obs[str(k)] = np.asarray(v).squeeze()
            else:
                obs[str(k)] = np.asarray(v).ravel().astype(sp.dtype)
        return obs
    except Exception:
        return None


def make_tracking_env(
    *,
    clip_id: str,
    start_step: int,
    end_step: int | None,
    ref_steps: tuple[int, ...],
    seed: int = 0,
) -> Any:
    try:
        from dm_control.locomotion.tasks.reference_pose import types  # type: ignore
        from mocapact.envs import tracking  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing mocapact + dm_control runtime deps.") from e

    if end_step is None:
        dataset = types.ClipCollection(ids=[str(clip_id)], start_steps=[int(start_step)])
    else:
        dataset = types.ClipCollection(ids=[str(clip_id)], start_steps=[int(start_step)], end_steps=[int(end_step)])
    task_kwargs = {
        "reward_type": "comic",
        "min_steps": 9,
        "ghost_offset": np.array([0.8, 0.0, 0.0], dtype=np.float32),
        "always_init_at_clip_start": True,
        "termination_error_threshold": 0.3,
    }
    env = tracking.MocapTrackingGymEnv(dataset=dataset, ref_steps=tuple(int(x) for x in ref_steps), act_noise=0.0, task_kwargs=task_kwargs)
    # MoCapAct's DmControlWrapper.seed() historically used `if seed:` which ignores seed=0.
    # We still call env.seed(...) for compatibility, then force the composer env RNG.
    seed = int(seed)
    try:
        env.seed(int(seed))
    except Exception:
        pass
    try:
        rs = np.random.RandomState(int(seed))
        env._env.random_state.set_state(rs.get_state())  # noqa: SLF001
    except Exception:
        pass
    try:
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(int(seed))
    except Exception:
        pass
    try:
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(int(seed))
    except Exception:
        pass
    _patch_tracking_no_error_truncation(env)
    return env


def patch_env_reference_for_override(
    env: Any,
    *,
    override: OverrideSpec,
    knee_deg: np.ndarray,
    warmup_steps: int = 0,
) -> Any:
    """Patch env's reference joint trajectory for the overridden knee.

    IMPORTANT (determinism + experts):
    - Per-snippet MoCapAct experts (TIME_INDEX_OBSERVABLES) do NOT condition on
      reference observables, so we only patch the reference so the *ghost* can
      visualize the overridden motion and for logging/debugging.
    - Do NOT recompute reference kinematics or reinitialize the walker here.
      Doing so perturbs MuJoCo internal state enough that two otherwise-identical
      rollouts can diverge after a single step (and can cause immediate falls).
    """
    task = env._env.task  # noqa: SLF001
    ref_steps = np.asarray(getattr(task, "_ref_steps", (0,)), dtype=np.int64)  # noqa: SLF001
    max_ref_step = int(np.max(ref_steps)) if ref_steps.size else 0

    q_kn = _to_rad(knee_deg, sign=override.knee_sign, offset_deg=override.knee_offset_deg)
    n = int(q_kn.size)
    if n < 2:
        raise RuntimeError("Override trajectory too short.")

    _patch_reference_joint_series(
        env,
        joint_name=_joint_name_from_actuator(override.knee_actuator),
        targets_rad=q_kn[:n],
        warmup_steps=int(warmup_steps),
        max_ref_step=int(max_ref_step),
    )

    # Best-effort refresh of cached observations (reference observables may use
    # the patched joint series), but we intentionally do not touch physics state.
    return _refresh_gym_observation(env)


def _patch_tracking_no_error_truncation(env: Any) -> None:
    """Disable early truncation due to tracking error spikes (keep rollout running)."""
    try:
        task = env._env.task  # noqa: SLF001
    except Exception:
        return
    try:
        # IMPORTANT:
        # - We do NOT mutate `_termination_error_threshold` because that also changes reward channels
        #   (and the PPO value function calibration).
        # - Instead we override the termination hook to ignore `_should_truncate` and only terminate
        #   at end-of-mocap.
        if bool(getattr(task, "_codex_no_truncation_patched", False)):
            return
        task._codex_no_truncation_patched = True  # noqa: SLF001

        orig_should_terminate = getattr(task, "should_terminate_episode", None)
        orig_get_discount = getattr(task, "get_discount", None)
        if callable(orig_should_terminate):
            task._codex_orig_should_terminate_episode = orig_should_terminate  # noqa: SLF001
        if callable(orig_get_discount):
            task._codex_orig_get_discount = orig_get_discount  # noqa: SLF001

        def _should_terminate_episode(_physics: Any) -> bool:
            try:
                return bool(getattr(task, "_end_mocap", False))
            except Exception:
                return False

        def _get_discount(_physics: Any) -> float:
            # Keep discount stable even if tracking error exceeds threshold.
            try:
                if bool(getattr(task, "_end_mocap", False)) and callable(orig_get_discount):
                    return float(orig_get_discount(_physics))
            except Exception:
                pass
            return 1.0

        task.should_terminate_episode = _should_terminate_episode  # type: ignore[assignment]
        task.get_discount = _get_discount  # type: ignore[assignment]
    except Exception:
        return


def _uprightness_cos_tilt(physics: Any, *, body_id: int | None) -> float:
    if body_id is None:
        return float("nan")
    try:
        xmat = np.asarray(physics.data.xmat[int(body_id)], dtype=np.float64).reshape(9)
        return float(xmat[7])  # local +Y axis dot world +Z? In dm_control CMU, this works as uprightness.
    except Exception:
        return float("nan")


def _root_z(physics: Any, *, body_id: int | None) -> float:
    if body_id is None:
        return float("nan")
    try:
        return float(np.asarray(physics.data.xpos[int(body_id), 2]).reshape(()))
    except Exception:
        return float("nan")


def _is_fall(root_z_m: float, upright: float) -> bool:
    z = float(root_z_m)
    if not math.isfinite(z):
        return False
    if z < float(HARD_FALL_Z_THRESHOLD_M):
        return True
    if z >= float(FALL_Z_THRESHOLD_M):
        return False
    u = float(upright)
    if not math.isfinite(u):
        return True
    return u < float(FALL_UPRIGHT_THRESHOLD)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _moving_average(x: np.ndarray, *, window: int) -> np.ndarray:
    """Simple moving average with edge padding (returns same length as x)."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(v.size)
    w = int(max(1, int(window)))
    if n < 1:
        return v.astype(np.float64)
    if w <= 1 or n == 1:
        return v.copy()
    kernel = np.ones((w,), dtype=np.float64) / float(w)
    # Pad with edge values so we don't artificially depress the ends.
    pad = w // 2
    left = np.full((pad,), float(v[0]), dtype=np.float64)
    right = np.full((w - 1 - pad,), float(v[-1]), dtype=np.float64)
    vp = np.concatenate([left, v, right], axis=0)
    y = np.convolve(vp, kernel, mode="valid")
    return y.astype(np.float64)


def _risk_trace_from_upright_and_margin(
    *,
    upright: np.ndarray,
    margin_m: np.ndarray,
    had_contact: np.ndarray | None,
    dt: float,
    tail_n: int = 10,
) -> np.ndarray:
    """Heuristic per-step instability score in [0, 1].

    Inputs:
    - upright: cosine of tilt (1=upright). In the CMU humanoid, the root's +Y axis is "up",
      so this is effectively dot(root_y_axis, world_z_axis).
    - margin_m: signed distance to the support polygon boundary (m). Positive means inside.
      This is typically the XCoM margin (more predictive), but COM margin also works.
    - had_contact: optional boolean trace indicating whether either foot contacted the ground.

    Design goals:
    - Avoid root-height based heuristics (crouching/kneeling should not be flagged).
    - Reduce false positives on dynamic walking (brief negative margins can occur).
    - Catch "about to fall" cases earlier using margin *trend* + tilt-rate.
    """
    u = np.asarray(upright, dtype=np.float64).reshape(-1)
    m = np.asarray(margin_m, dtype=np.float64).reshape(-1)
    n = int(min(u.size, m.size))
    if n < 1:
        return np.zeros((0,), dtype=np.float32)

    if had_contact is not None:
        hc0 = np.asarray(had_contact, dtype=np.bool_).reshape(-1)
        if int(hc0.size) < int(n):
            # Missing or truncated contact trace; treat as "unknown" (assume contact).
            hc = None
        else:
            hc = hc0[:n]
    else:
        hc = None

    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        dt = 0.03

    W = int(max(3, min(int(tail_n), n)))
    out = np.zeros((n,), dtype=np.float64)

    prev_ui = float(u[0]) if math.isfinite(float(u[0])) else 0.0
    for i in range(n):
        ui = float(u[i])
        if not math.isfinite(ui):
            ui = prev_ui

        # Tilt risk:
        # - Absolute tilt is permissive (we avoid flagging crouching/leaning as a fall).
        # - Tilt *rate* catches tipping early.
        tilt_abs_r = _clamp01((0.75 - ui) / 0.35)  # 0 at 0.75, 1 at 0.40
        if i == 0:
            du = 0.0
        else:
            du = (ui - prev_ui) / float(dt)
        prev_ui = ui
        tilt_rate_r = _clamp01((-du - 0.40) / 1.20)  # 0 at -0.40/s, 1 at -1.60/s
        tilt_r = 1.0 - (1.0 - float(tilt_abs_r)) * (1.0 - float(tilt_rate_r))

        # Support risk from margin magnitude + trend.
        j0 = int(max(0, i - W + 1))
        mw = m[j0 : i + 1]
        mw = mw[np.isfinite(mw)]
        if mw.size < 1:
            # Can't compute a support margin (e.g., aerial phase or contact detection glitch).
            # Keep this as *mild* risk rather than immediately declaring a likely fall.
            support_r = 0.25
        else:
            mmin = float(np.min(mw))
            mend = float(mw[-1])
            if mw.size >= 2:
                slope = float((float(mw[-1]) - float(mw[0])) / (float(mw.size - 1) * float(dt)))
            else:
                slope = 0.0

            # Thresholds tuned empirically on REF walking/running so stable motion stays low-risk,
            # while large negative (and/or rapidly decreasing) margins drive risk to 1.
            r_mmin = _clamp01((-mmin - 0.42) / 0.40)  # 0 at -42cm, 1 at -82cm
            r_mend = _clamp01((-mend - 0.30) / 0.35)  # 0 at -30cm, 1 at -65cm
            r_slope = _clamp01((-slope - 0.15) / 0.45)  # 0 at -0.15 m/s, 1 at -0.60 m/s

            support_r = float(_clamp01(0.55 * r_mmin + 0.25 * r_mend + 0.20 * r_slope))

        # Combine such that either signal can dominate.
        out[i] = 1.0 - (1.0 - float(tilt_r)) * (1.0 - float(support_r))

    return np.clip(out.astype(np.float32), 0.0, 1.0)


def predict_fall_risk_from_traces(
    *,
    fell: bool,
    upright: np.ndarray,
    com_margin_m: np.ndarray,
    xcom_margin_m: np.ndarray | None = None,
    had_contact: np.ndarray | None = None,
    dt: float,
    tail_n: int = 10,
) -> tuple[float, bool, np.ndarray]:
    """Return (scalar_risk, likely_fall, risk_trace)."""
    # Prefer XCoM margin when available: it incorporates COM velocity and is more predictive.
    m = np.asarray(xcom_margin_m if xcom_margin_m is not None else com_margin_m, dtype=np.float32)
    risk_trace = _risk_trace_from_upright_and_margin(
        upright=np.asarray(upright, dtype=np.float32),
        margin_m=m,
        had_contact=(None if had_contact is None else np.asarray(had_contact, dtype=np.bool_)),
        dt=float(dt),
        tail_n=int(tail_n),
    )
    if risk_trace.size < 1:
        return (1.0 if bool(fell) else float("nan")), bool(fell), risk_trace

    # Smooth before summarizing to reduce single-frame spikes.
    risk_smooth = _moving_average(risk_trace, window=5)
    W = int(max(3, min(int(tail_n), int(risk_smooth.size))))
    tail = np.asarray(risk_smooth[-W:], dtype=np.float64)
    # Use a high percentile of the tail for robustness, but allow sustained peaks to show.
    r = float(np.nanpercentile(tail, 90))
    if bool(fell):
        r = 1.0
    r = _clamp01(r) if math.isfinite(r) else 1.0
    return r, bool(r >= 0.70), np.asarray(risk_trace, dtype=np.float32)


def detect_balance_loss_step(*, risk_trace: np.ndarray, upright: np.ndarray) -> int:
    """Return first step considered a 'balance loss' (not necessarily an actual fall)."""
    r = np.asarray(risk_trace, dtype=np.float64).reshape(-1)
    u = np.asarray(upright, dtype=np.float64).reshape(-1)
    n = int(min(r.size, u.size))
    if n < 1:
        return -1
    # Smooth risk to avoid triggering on single-frame spikes.
    r_s = _moving_average(r, window=5)
    consec = 0
    for i in range(n):
        ri = float(r_s[i])
        ui = float(u[i])
        if math.isfinite(ui) and ui < 0.40:
            return int(i)
        if math.isfinite(ri) and ri >= 0.85:
            consec += 1
            if consec >= 4:
                return int(i - consec + 1)
        else:
            consec = 0
    return -1


def _predict_fall_risk(
    *,
    fell: bool,
    min_balance_margin_m: float,
    balance_margin_end_m: float,
    outside_support_frac: float,
    balance_margin_slope_end_mps: float,
    max_outward_com_speed_mps: float,
) -> tuple[float, bool]:
    if bool(fell):
        return 1.0, True

    mmin = float(min_balance_margin_m)
    mend = float(balance_margin_end_m)
    out = float(outside_support_frac)
    dm = float(balance_margin_slope_end_mps)
    vout = float(max_outward_com_speed_mps)

    m_use = mend if math.isfinite(mend) else mmin
    if math.isfinite(m_use):
        risk_m = _clamp01((0.02 - float(m_use)) / 0.12)
    else:
        risk_m = 1.0

    if math.isfinite(out):
        risk_out = _clamp01(out / 0.20)
    else:
        risk_out = 1.0

    if math.isfinite(dm):
        risk_dm = _clamp01((-dm - 0.05) / 0.35)
    else:
        risk_dm = 0.0

    if math.isfinite(vout):
        risk_vout = _clamp01((vout - 0.10) / 0.50)
    else:
        risk_vout = 0.0

    risk = float(0.55 * risk_m + 0.15 * risk_out + 0.25 * risk_dm + 0.05 * risk_vout)
    return risk, bool(risk >= 0.70)


# Balance metrics (COM / XCoM vs support polygon).
_FOOT_GEOMS_LEFT = ("lfoot", "lfoot_ch", "ltoes0", "ltoes1", "ltoes2")
_FOOT_GEOMS_RIGHT = ("rfoot", "rfoot_ch", "rtoes0", "rtoes1", "rtoes2")
_FOOT_BODIES_LEFT = ("walker/lfoot", "walker/ltoes")
_FOOT_BODIES_RIGHT = ("walker/rfoot", "walker/rtoes")


def _ground_geom_ids(physics: Any) -> set[int]:
    m = physics.model
    ids: set[int] = set()
    for gid in range(int(getattr(m, "ngeom", 0))):
        try:
            name = m.id2name(int(gid), "geom")
        except Exception:
            continue
        if not name:
            continue
        s = str(name)
        if s.startswith("walker/") or s.startswith("ghost/"):
            continue
        ids.add(int(gid))
    return ids


def _geom_ids(physics: Any, names: tuple[str, ...]) -> set[int]:
    m = physics.model
    out: set[int] = set()
    for n in names:
        for cand in (f"walker/{n}", str(n)):
            try:
                out.add(int(m.name2id(str(cand), "geom")))
            except Exception:
                continue
    return out


def _body_ids(physics: Any, names: tuple[str, ...]) -> list[int]:
    m = physics.model
    out: list[int] = []
    for n in names:
        try:
            out.append(int(m.name2id(str(n), "body")))
        except Exception:
            continue
    return out


def _walker_com_xy(physics: Any, *, root_body_id: int | None) -> np.ndarray:
    if root_body_id is None:
        return np.asarray([float("nan"), float("nan")], dtype=np.float64)
    try:
        com = np.asarray(physics.data.subtree_com[int(root_body_id)], dtype=np.float64).reshape(3)
        # IMPORTANT: copy; MuJoCo's data arrays are views into internal buffers that
        # change in-place each mj_forward. Keeping a view would corrupt velocity
        # estimates (prev_com would silently update).
        return np.array(com[:2], dtype=np.float64, copy=True)
    except Exception:
        return np.asarray([float("nan"), float("nan")], dtype=np.float64)


def _support_contact_points_xy(physics: Any, *, foot_geoms: set[int], ground_geoms: set[int]) -> np.ndarray:
    pts: list[list[float]] = []
    try:
        ncon = int(getattr(physics.data, "ncon", 0))
    except Exception:
        ncon = 0
    for i in range(max(0, ncon)):
        try:
            c = physics.data.contact[int(i)]
            g1 = int(getattr(c, "geom1"))
            g2 = int(getattr(c, "geom2"))
        except Exception:
            continue
        if (g1 in foot_geoms and g2 in ground_geoms) or (g2 in foot_geoms and g1 in ground_geoms):
            try:
                pos = np.asarray(getattr(c, "pos"), dtype=np.float64).reshape(3)
            except Exception:
                continue
            pts.append([float(pos[0]), float(pos[1])])
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return pts.copy()

    pts_list = sorted({(float(x), float(y)) for x, y in pts})
    if len(pts_list) < 3:
        return np.asarray(pts_list, dtype=np.float64).reshape(-1, 2)

    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts_list:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts_list):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64).reshape(-1, 2)


def _dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(2)
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = float(max(0.0, min(1.0, t)))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _signed_margin_to_support(p_xy: np.ndarray, support_xy: np.ndarray) -> float:
    p = np.asarray(p_xy, dtype=np.float64).reshape(2)
    pts = np.asarray(support_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 1:
        return float("nan")
    if pts.shape[0] == 1:
        return -float(np.linalg.norm(p - pts[0]))
    if pts.shape[0] == 2:
        return -_dist_point_to_segment(p, pts[0], pts[1])

    hull = _convex_hull_2d(pts)
    if hull.shape[0] < 3:
        return _signed_margin_to_support(p, hull)

    inside = True
    min_dist = float("inf")
    H = int(hull.shape[0])
    for i in range(H):
        a = hull[i]
        b = hull[(i + 1) % H]
        ab = b - a
        ap = p - a
        cross = float(ab[0] * ap[1] - ab[1] * ap[0])
        if cross < -1e-9:
            inside = False
        min_dist = min(min_dist, _dist_point_to_segment(p, a, b))
    if not math.isfinite(min_dist):
        return float("nan")
    return float(min_dist) if inside else -float(min_dist)


def _balance_metrics_step(
    *,
    physics: Any,
    root_body_id: int | None,
    left_foot_geoms: set[int],
    right_foot_geoms: set[int],
    ground_geoms: set[int],
    left_foot_bodies: list[int],
    right_foot_bodies: list[int],
    dt: float,
    prev_com_xy: np.ndarray | None,
    foot_height_threshold_m: float = 0.12,
    foot_rect_half_width_m: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, bool, np.ndarray]:
    com_xy = _walker_com_xy(physics, root_body_id=root_body_id)
    if prev_com_xy is None or not np.all(np.isfinite(prev_com_xy)) or not np.all(np.isfinite(com_xy)) or dt <= 0:
        com_vel_xy = np.asarray([0.0, 0.0], dtype=np.float64)
    else:
        com_vel_xy = (com_xy - np.asarray(prev_com_xy, dtype=np.float64).reshape(2)) / float(dt)

    w0 = 3.13
    xcom_xy = com_xy + com_vel_xy / float(w0)

    left_pts = _support_contact_points_xy(physics, foot_geoms=left_foot_geoms, ground_geoms=ground_geoms)
    right_pts = _support_contact_points_xy(physics, foot_geoms=right_foot_geoms, ground_geoms=ground_geoms)
    had_left = bool(left_pts.shape[0] > 0)
    had_right = bool(right_pts.shape[0] > 0)
    had_contact = bool(had_left or had_right)
    support_pts = (
        np.concatenate([left_pts, right_pts], axis=0) if had_contact else np.empty((0, 2), dtype=np.float64)
    )

    def _foot_rect_points(foot_toes_ids: list[int]) -> list[list[float]]:
        if len(foot_toes_ids) < 2:
            return []
        try:
            p0 = np.asarray(physics.data.xpos[int(foot_toes_ids[0])], dtype=np.float64).reshape(3)
            p1 = np.asarray(physics.data.xpos[int(foot_toes_ids[1])], dtype=np.float64).reshape(3)
        except Exception:
            return []
        if not (math.isfinite(float(p0[2])) and math.isfinite(float(p1[2]))):
            return []
        a = np.asarray([float(p0[0]), float(p0[1])], dtype=np.float64)
        b = np.asarray([float(p1[0]), float(p1[1])], dtype=np.float64)
        d = b - a
        nd = float(np.linalg.norm(d))
        if nd < 1e-6:
            return [[float(a[0]), float(a[1])]]
        u = d / nd
        perp = np.asarray([-u[1], u[0]], dtype=np.float64)
        w = float(max(0.0, foot_rect_half_width_m))
        pts = [(a + w * perp).tolist(), (a - w * perp).tolist(), (b + w * perp).tolist(), (b - w * perp).tolist()]
        return [[float(x), float(y)] for x, y in pts]

    # Only augment support using body geometry when that foot is *actually in contact*.
    # This prevents a swing foot (or airborne phase) from artificially enlarging the
    # support polygon and corrupting the stability margin.
    aug: list[list[float]] = []
    if had_left:
        aug.extend(_foot_rect_points(list(left_foot_bodies)))
    if had_right:
        aug.extend(_foot_rect_points(list(right_foot_bodies)))
    if aug:
        support_pts = np.concatenate([support_pts, np.asarray(aug, dtype=np.float64).reshape(-1, 2)], axis=0)

    if support_pts.shape[0] < 1 or not np.all(np.isfinite(support_pts)):
        return (
            com_xy,
            com_vel_xy,
            xcom_xy,
            float("nan"),
            float("nan"),
            had_contact,
            np.asarray([float("nan"), float("nan")], dtype=np.float64),
        )

    centroid = np.mean(support_pts, axis=0).astype(np.float64).reshape(2)
    # We track both:
    # - COM margin: "is the center of mass over the base of support?"
    # - XCoM margin: capture-point style margin (more predictive, but can be outside during stepping).
    com_margin = _signed_margin_to_support(com_xy, support_pts)
    xcom_margin = _signed_margin_to_support(xcom_xy, support_pts)
    return com_xy, com_vel_xy, xcom_xy, float(com_margin), float(xcom_margin), had_contact, centroid


def run_reference_stability_check(
    *,
    policy: Any,
    clip_id: str,
    start_step: int,
    end_step: int | None,
    primary_steps: int,
    warmup_steps: int = 0,
    deterministic_policy: bool = True,
    seed: int = 0,
) -> RefStability:
    """Run REF-only simulation on the unmodified reference segment and compute stability metrics."""
    ref_steps = getattr(policy, "ref_steps", (0,))
    try:
        ref_steps_arr = np.asarray(ref_steps, dtype=np.int64).reshape(-1)
        max_ref_step = int(np.max(ref_steps_arr)) if ref_steps_arr.size else 0
        ref_steps_tuple = tuple(int(x) for x in ref_steps_arr.tolist())
    except Exception:
        try:
            max_ref_step = int(max(ref_steps)) if ref_steps is not None else 0
        except Exception:
            max_ref_step = 0
        try:
            ref_steps_tuple = tuple(int(x) for x in (ref_steps or (0,)))
        except Exception:
            ref_steps_tuple = (0,)
    eval_start_step = int(start_step)
    warmup_steps = int(max(0, int(warmup_steps)))
    env_start_step = int(max(0, int(eval_start_step) - int(warmup_steps)))
    warmup = int(eval_start_step - env_start_step)

    env = make_tracking_env(
        clip_id=str(clip_id),
        start_step=int(env_start_step),
        end_step=(int(end_step) if end_step is not None else None),
        ref_steps=ref_steps_tuple,
        seed=int(seed),
    )

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) == 2 else reset_out

    physics = env._env.physics  # noqa: SLF001
    try:
        root_id = int(physics.model.name2id("walker/root", "body"))
    except Exception:
        root_id = None

    # Balance helpers (only tracked during the evaluation window, after warmup).
    ground_geoms = _ground_geom_ids(physics)
    left_geoms = _geom_ids(physics, _FOOT_GEOMS_LEFT)
    right_geoms = _geom_ids(physics, _FOOT_GEOMS_RIGHT)
    left_bodies = _body_ids(physics, _FOOT_BODIES_LEFT)
    right_bodies = _body_ids(physics, _FOOT_BODIES_RIGHT)

    try:
        dt = float(getattr(env._env.task, "_control_timestep", 0.03))  # noqa: SLF001
        if not math.isfinite(dt) or dt <= 0:
            dt = 0.03
    except Exception:
        dt = 0.03

    # Torch no-grad helps performance.
    try:
        import torch

        policy.eval()
        no_grad = torch.no_grad()
    except Exception:
        no_grad = None

    def _predict(obs_in: Any, state_in: Any) -> tuple[np.ndarray, Any]:
        if no_grad is not None:
            with no_grad:  # type: ignore[attr-defined]
                act, next_state = policy.predict(obs_in, state=state_in, deterministic=deterministic_policy)
        else:
            act, next_state = policy.predict(obs_in, state=state_in, deterministic=deterministic_policy)
        return np.asarray(act, dtype=np.float32), next_state

    state: Any = None
    fall_step = -1

    # 1) Warmup: start earlier so we don't reset mid-motion.
    for _t in range(int(warmup)):
        act, state = _predict(obs, state)
        step_out = env.step(act)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, _reward, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            obs, _reward, done, _info = step_out
        u = _uprightness_cos_tilt(physics, body_id=root_id)
        if done or (math.isfinite(u) and float(u) < 0.40):
            # Fell (or terminated) before reaching the eval window.
            return RefStability(
                fall_step=0,
                min_root_z_m=float(_root_z(physics, body_id=root_id)),
                min_upright=float(u if math.isfinite(u) else float("nan")),
                min_balance_margin_m=float("nan"),
                outside_support_frac=float("nan"),
                max_outward_com_speed_mps=float("nan"),
                predicted_fall_risk=1.0,
                predicted_likely_fall=True,
            )

    # 2) Evaluate over the primary window only.
    prev_com_xy: np.ndarray | None = None
    outside = 0
    count = 0
    min_margin = float("inf")
    tail_margin: list[float] = []
    max_outward = 0.0
    margin_end = float("nan")
    min_root_z = float("inf")
    min_upright = float("inf")
    upright_trace: list[float] = []
    xcom_margin_trace: list[float] = []
    had_contact_trace: list[bool] = []
    terminated_early = False

    for t in range(int(primary_steps)):
        act, state = _predict(obs, state)
        step_out = env.step(act)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, _reward, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            obs, _reward, done, _info = step_out

        z = _root_z(physics, body_id=root_id)
        u = _uprightness_cos_tilt(physics, body_id=root_id)
        if math.isfinite(z):
            min_root_z = min(min_root_z, z)
        if math.isfinite(u):
            min_upright = min(min_upright, u)
        upright_trace.append(float(u))

        # Balance (XCoM margin).
        com_xy, com_vel_xy, xcom_xy, _com_margin, xcom_margin, had_contact, centroid = _balance_metrics_step(
            physics=physics,
            root_body_id=root_id,
            left_foot_geoms=left_geoms,
            right_foot_geoms=right_geoms,
            ground_geoms=ground_geoms,
            left_foot_bodies=left_bodies,
            right_foot_bodies=right_bodies,
            dt=float(dt),
            prev_com_xy=prev_com_xy,
        )
        prev_com_xy = com_xy
        xcom_margin_trace.append(float(xcom_margin))
        had_contact_trace.append(bool(had_contact))
        if math.isfinite(float(xcom_margin)):
            min_margin = min(min_margin, float(xcom_margin))
            margin_end = float(xcom_margin)
            if float(xcom_margin) < 0.0:
                outside += 1
            count += 1
            tail_margin.append(float(xcom_margin))
            if len(tail_margin) > 10:
                tail_margin = tail_margin[-10:]

            # Outward COM speed along XCoM direction from support centroid.
            if np.all(np.isfinite(centroid)) and np.all(np.isfinite(xcom_xy)) and np.all(np.isfinite(com_vel_xy)):
                v = xcom_xy - centroid
                nv = float(np.linalg.norm(v))
                if nv > 1e-6:
                    outward = float(np.dot(com_vel_xy, v / nv))
                    max_outward = max(float(max_outward), float(outward))

        if done:
            # If the env terminates early, we treat that as unstable.
            terminated_early = True
            break

    outside_frac = float(outside / max(1, count))
    risk, likely, risk_trace = predict_fall_risk_from_traces(
        fell=bool(terminated_early),
        upright=np.asarray(upright_trace, dtype=np.float32),
        com_margin_m=np.asarray(np.zeros((len(xcom_margin_trace),), dtype=np.float32)),
        xcom_margin_m=np.asarray(xcom_margin_trace, dtype=np.float32),
        had_contact=np.asarray(had_contact_trace, dtype=np.bool_),
        dt=float(dt),
        tail_n=10,
    )
    fall_step = detect_balance_loss_step(risk_trace=risk_trace, upright=np.asarray(upright_trace, dtype=np.float32))

    return RefStability(
        fall_step=int(fall_step),
        min_root_z_m=float(min_root_z if math.isfinite(min_root_z) else float("nan")),
        min_upright=float(min_upright if math.isfinite(min_upright) else float("nan")),
        min_balance_margin_m=float(min_margin if math.isfinite(min_margin) else float("nan")),
        outside_support_frac=float(outside_frac),
        max_outward_com_speed_mps=float(max_outward if math.isfinite(max_outward) else float("nan")),
        predicted_fall_risk=float(risk),
        predicted_likely_fall=bool(likely),
    )


def tint_geoms_by_prefix(physics: Any, *, prefix: str, rgba: tuple[float, float, float, float]) -> None:
    m = physics.model
    col = np.asarray(rgba, dtype=np.float32).reshape(4)
    for gid in range(int(getattr(m, "ngeom", 0))):
        try:
            name = m.id2name(int(gid), "geom")
        except Exception:
            continue
        if not name:
            continue
        if not str(name).startswith(str(prefix)):
            continue
        m.geom_rgba[int(gid)] = col


def highlight_overridden_leg(physics: Any, *, override: OverrideSpec) -> None:
    # Override is right leg in v2, but keep generic.
    name = str(override.knee_actuator).split("/")[-1]
    side = "left" if name.startswith("l") else "right" if name.startswith("r") else "unknown"
    if side == "left":
        names = (
            "lfemur_upper",
            "lfemur",
            "ltibia",
            "lfoot",
            "lfoot_ch",
            "ltoes0",
            "ltoes1",
            "ltoes2",
        )
    elif side == "right":
        names = (
            "rfemur_upper",
            "rfemur",
            "rtibia",
            "rfoot",
            "rfoot_ch",
            "rtoes0",
            "rtoes1",
            "rtoes2",
        )
    else:
        return

    m = physics.model
    rgba = np.array([0.95, 0.15, 0.85, 1.0], dtype=np.float32)
    for n in names:
        for cand in (f"walker/{n}", n):
            try:
                gid = int(m.name2id(str(cand), "geom"))
            except Exception:
                continue
            m.geom_rgba[gid] = rgba
