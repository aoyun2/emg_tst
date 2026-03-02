from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .mocapact_dataset import ExpertPolicy

# NumPy 2.x removed `np.infty`; some MoCapAct versions still reference it.
if not hasattr(np, "infty"):
    np.infty = np.inf  # type: ignore[attr-defined]


@dataclass(frozen=True)
class OverrideConfig:
    """How we override the expert's actions for thigh/knee joints."""

    thigh_actuator: str
    knee_actuator: str

    thigh_sign: float = 1.0
    knee_sign: float = 1.0
    thigh_offset_deg: float = 0.0
    knee_offset_deg: float = 0.0


@dataclass(frozen=True)
class SimulationResult:
    snippet_id: str
    n_steps_total: int
    n_steps_overridden: int
    terminated_early: bool
    total_reward: float

    thigh_rmse_deg: float
    knee_rmse_deg: float


def _to_rad(deg: np.ndarray, *, sign: float, offset_deg: float) -> np.ndarray:
    x = (np.asarray(deg, dtype=np.float32) + float(offset_deg)) * float(sign)
    return np.deg2rad(x).astype(np.float32)


def _ctrl_from_qpos(target_qpos: float, *, lower: float, upper: float) -> float:
    """Inverse of dm_control's CMU humanoid position-actuator scaling.

    See dm_control.locomotion.walkers.cmu_humanoid.CMUHumanoidPositionControlled.cmu_pose_to_actuation.
    """
    if upper <= lower:
        return 0.0
    return float((2.0 * target_qpos - (upper + lower)) / (upper - lower))


def _find_actuator_index(physics: Any, name: str) -> int:
    # dm_control's MjModel exposes name lookup via name2id/id2name.
    m = physics.model
    candidates = [name]
    if not name.startswith("walker/"):
        candidates.append(f"walker/{name}")

    for cand in candidates:
        try:
            return int(m.name2id(str(cand), "actuator"))
        except Exception:
            continue

    try:
        avail = [m.id2name(i, "actuator") for i in range(int(m.nu))]
    except Exception:
        avail = []
    raise KeyError(f"Actuator {name!r} not found. Available examples: {avail[:10]}...")


def _joint_id_for_actuator(physics: Any, actuator_idx: int) -> int:
    trnid = np.asarray(physics.model.actuator_trnid[actuator_idx], dtype=np.int32)
    return int(trnid[0])


def _joint_range_for_actuator(physics: Any, actuator_idx: int) -> tuple[float, float]:
    # Map actuator -> joint (via trnid) -> joint range.
    jnt_id = _joint_id_for_actuator(physics, actuator_idx)
    rng = np.asarray(physics.model.jnt_range[jnt_id], dtype=np.float64).reshape(2)
    return float(rng[0]), float(rng[1])


def _qpos_addr_for_actuator(physics: Any, actuator_idx: int) -> int:
    jnt_id = _joint_id_for_actuator(physics, actuator_idx)
    return int(np.asarray(physics.model.jnt_qposadr[jnt_id]).reshape(()))


def _get_qpos_by_addr(physics: Any, qpos_addr: int) -> float:
    return float(np.asarray(physics.data.qpos[int(qpos_addr)]).reshape(()))


def run_mocapact_expert_simulation(
    *,
    expert: ExpertPolicy,
    thigh_angle_deg: np.ndarray,
    knee_angle_deg: np.ndarray,
    override: OverrideConfig,
    deterministic_policy: bool = True,
    warmup_steps: int = 0,
    max_steps: int | None = None,
) -> SimulationResult:
    """Run a MoCapAct clip-expert policy, overriding thigh/knee targets each step.

    This requires the user to have installed:
      - mocapact
      - stable-baselines3
      - dm_control
      - mujoco (via dm_control)
    """
    try:
        from mocapact import observables  # type: ignore
        from mocapact.clip_expert import utils as clip_expert_utils  # type: ignore
        from mocapact.envs import tracking  # type: ignore
        from mocapact.sb3 import utils as sb3_utils  # type: ignore
    except Exception as e:  # pragma: no cover - depends on external install
        raise RuntimeError(
            "Missing MoCapAct runtime dependencies. Install `mocapact`, `stable-baselines3`, and `dm_control`."
        ) from e

    env_kwargs = clip_expert_utils.make_env_kwargs(
        str(expert.clip_id),
        mocap_path=None,
        start_step=int(expert.start_step),
        end_step=int(expert.end_step),
        min_steps=10,
        ghost_offset=0.0,
        always_init_at_clip_start=True,
        termination_error_threshold=0.3,
        act_noise=0.0,
    )
    env = tracking.MocapTrackingGymEnv(**env_kwargs)
    policy = sb3_utils.load_policy(str(expert.model_dir), observables.TIME_INDEX_OBSERVABLES)

    warmup = max(0, int(warmup_steps))

    # Resample / alignment should be handled by caller; we just clamp to min length.
    thigh_qpos = _to_rad(thigh_angle_deg, sign=override.thigh_sign, offset_deg=override.thigh_offset_deg)
    knee_qpos = _to_rad(knee_angle_deg, sign=override.knee_sign, offset_deg=override.knee_offset_deg)
    n = int(min(len(thigh_qpos), len(knee_qpos)))
    if max_steps is not None:
        n = min(n, int(max_steps))

    # Gym version compatibility: reset() may return obs or (obs, info)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) == 2 else reset_out

    # Resolve actuator indices after reset (some wrappers recreate physics/data on reset).
    physics = env._env.physics  # noqa: SLF001 - external library convention
    idx_thigh = _find_actuator_index(physics, override.thigh_actuator)
    idx_knee = _find_actuator_index(physics, override.knee_actuator)
    thigh_lo, thigh_hi = _joint_range_for_actuator(physics, idx_thigh)
    knee_lo, knee_hi = _joint_range_for_actuator(physics, idx_knee)
    qpos_thigh_adr = _qpos_addr_for_actuator(physics, idx_thigh)
    qpos_knee_adr = _qpos_addr_for_actuator(physics, idx_knee)

    thigh_err: list[float] = []
    knee_err: list[float] = []
    total_reward = 0.0
    terminated_early = False

    total_target_steps = warmup + n
    for t in range(total_target_steps):
        action, _state = policy.predict(obs, deterministic=deterministic_policy)
        action = np.asarray(action, dtype=np.float32).copy()

        # Override: only after warmup. Index into the provided override arrays.
        if t >= warmup:
            i = t - warmup
            action[idx_thigh] = np.clip(
                _ctrl_from_qpos(float(thigh_qpos[i]), lower=thigh_lo, upper=thigh_hi), -1.0, 1.0
            )
            action[idx_knee] = np.clip(
                _ctrl_from_qpos(float(knee_qpos[i]), lower=knee_lo, upper=knee_hi), -1.0, 1.0
            )

        step_out = env.step(action)
        # Gym API variants:
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _info = step_out

        total_reward += float(reward)

        if t >= warmup:
            i = t - warmup
            # Measure achieved joint angles in qpos space (radians) and compute instantaneous error.
            thigh_now = _get_qpos_by_addr(physics, qpos_thigh_adr)
            knee_now = _get_qpos_by_addr(physics, qpos_knee_adr)
            thigh_err.append(float(thigh_now - float(thigh_qpos[i])))
            knee_err.append(float(knee_now - float(knee_qpos[i])))

        if done:
            terminated_early = t < (total_target_steps - 1)
            break

    if not thigh_err:
        raise RuntimeError("Simulation produced zero steps. Check environment reset/termination.")

    thigh_rmse = float(np.sqrt(np.mean(np.square(np.asarray(thigh_err, dtype=np.float64)))))
    knee_rmse = float(np.sqrt(np.mean(np.square(np.asarray(knee_err, dtype=np.float64)))))

    return SimulationResult(
        snippet_id=expert.snippet_id,
        n_steps_total=(warmup + len(thigh_err)),
        n_steps_overridden=len(thigh_err),
        terminated_early=terminated_early,
        total_reward=float(total_reward),
        thigh_rmse_deg=float(np.rad2deg(thigh_rmse)),
        knee_rmse_deg=float(np.rad2deg(knee_rmse)),
    )


def load_distillation_policy(policy_ckpt: str, *, device: str = "cpu") -> Any:
    """Load a MoCapAct distillation policy (e.g., multi-clip NpmpPolicy) from a Lightning checkpoint."""
    try:
        from mocapact import utils as mocap_utils  # type: ignore
    except Exception as e:  # pragma: no cover - external install
        raise RuntimeError("Missing mocapact. Install `mocapact` to load multi-clip policies.") from e

    import os.path as osp

    ckpt_path = str(policy_ckpt)

    # Mirror mocapact.distillation.evaluate's model_constructor resolution.
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
        raise FileNotFoundError(
            f"Could not find model_constructor.txt next to checkpoint. Looked at: {model_constructor_path}"
        ) from e

    # PyTorch 2.6+ defaults `torch.load(weights_only=True)` which breaks older Lightning
    # checkpoints that store non-tensor objects (e.g., gym spaces) in the checkpoint dict.
    # MoCapAct checkpoints are trusted artifacts, so we opt into full unpickling here.
    import torch

    orig_torch_load = torch.load

    def _torch_load_compat(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs or kwargs["weights_only"] is None:
            kwargs["weights_only"] = False
        return orig_torch_load(*args, **kwargs)

    def _to_gymnasium_space(space: Any) -> Any:
        # SB3>=2 uses gymnasium; MoCapAct checkpoints often store gym spaces.
        try:
            import gym
            import gymnasium
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

    torch.load = _torch_load_compat  # type: ignore[assignment]
    try:
        policy = model_cls.load_from_checkpoint(ckpt_path, map_location=device)
    finally:
        torch.load = orig_torch_load  # type: ignore[assignment]

    try:
        policy.observation_space = _to_gymnasium_space(getattr(policy, "observation_space", None))
        policy.action_space = _to_gymnasium_space(getattr(policy, "action_space", None))
    except Exception:
        pass

    # MoCapAct 0.1 calls `self.extract_features(obs)` assuming the SB3<=1.x signature.
    # SB3>=2.x changed it to require an explicit features_extractor argument.
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


def run_mocapact_multiclip_simulation(
    *,
    snippet_id: str,
    clip_id: str,
    start_step: int,
    end_step: int,
    policy: Any,
    thigh_angle_deg: np.ndarray,
    knee_angle_deg: np.ndarray,
    override: OverrideConfig,
    deterministic_policy: bool = True,
    warmup_steps: int = 0,
    max_steps: int | None = None,
) -> SimulationResult:
    """Run a MoCapAct multi-clip policy on a clip snippet, overriding thigh/knee targets each step."""
    try:
        from dm_control.locomotion.tasks.reference_pose import types  # type: ignore
        from mocapact.envs import tracking  # type: ignore
    except Exception as e:  # pragma: no cover - depends on external install
        raise RuntimeError("Missing MoCapAct runtime dependencies. Install `mocapact` and `dm_control`.") from e

    dataset = types.ClipCollection(
        ids=[str(clip_id)],
        start_steps=[int(start_step)],
        end_steps=[int(end_step)],
    )

    ref_steps = getattr(policy, "ref_steps", (0,))
    task_kwargs = {
        "reward_type": "comic",
        "min_steps": 9,  # matches mocapact defaults (min_steps=10 -> task min_steps=9)
        "ghost_offset": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "always_init_at_clip_start": True,
        "termination_error_threshold": 0.3,
    }
    env = tracking.MocapTrackingGymEnv(
        dataset=dataset,
        ref_steps=tuple(int(x) for x in ref_steps),
        act_noise=0.0,
        task_kwargs=task_kwargs,
    )

    warmup = max(0, int(warmup_steps))

    thigh_qpos = _to_rad(thigh_angle_deg, sign=override.thigh_sign, offset_deg=override.thigh_offset_deg)
    knee_qpos = _to_rad(knee_angle_deg, sign=override.knee_sign, offset_deg=override.knee_offset_deg)
    n = int(min(len(thigh_qpos), len(knee_qpos)))
    if max_steps is not None:
        n = min(n, int(max_steps))

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) == 2 else reset_out

    physics = env._env.physics  # noqa: SLF001 - external library convention
    idx_thigh = _find_actuator_index(physics, override.thigh_actuator)
    idx_knee = _find_actuator_index(physics, override.knee_actuator)
    thigh_lo, thigh_hi = _joint_range_for_actuator(physics, idx_thigh)
    knee_lo, knee_hi = _joint_range_for_actuator(physics, idx_knee)
    qpos_thigh_adr = _qpos_addr_for_actuator(physics, idx_thigh)
    qpos_knee_adr = _qpos_addr_for_actuator(physics, idx_knee)

    thigh_err: list[float] = []
    knee_err: list[float] = []
    total_reward = 0.0
    terminated_early = False

    state = None

    total_target_steps = warmup + n
    for t in range(total_target_steps):
        action, state = policy.predict(obs, state=state, deterministic=deterministic_policy)
        action = np.asarray(action, dtype=np.float32).copy()

        if t >= warmup:
            i = t - warmup
            action[idx_thigh] = np.clip(
                _ctrl_from_qpos(float(thigh_qpos[i]), lower=thigh_lo, upper=thigh_hi), -1.0, 1.0
            )
            action[idx_knee] = np.clip(
                _ctrl_from_qpos(float(knee_qpos[i]), lower=knee_lo, upper=knee_hi), -1.0, 1.0
            )

        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, reward, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _info = step_out

        total_reward += float(reward)

        if t >= warmup:
            i = t - warmup
            thigh_now = _get_qpos_by_addr(physics, qpos_thigh_adr)
            knee_now = _get_qpos_by_addr(physics, qpos_knee_adr)
            thigh_err.append(float(thigh_now - float(thigh_qpos[i])))
            knee_err.append(float(knee_now - float(knee_qpos[i])))

        if done:
            terminated_early = t < (total_target_steps - 1)
            break

    if not thigh_err:
        raise RuntimeError("Simulation produced zero steps. Check environment reset/termination.")

    thigh_rmse = float(np.sqrt(np.mean(np.square(np.asarray(thigh_err, dtype=np.float64)))))
    knee_rmse = float(np.sqrt(np.mean(np.square(np.asarray(knee_err, dtype=np.float64)))))

    return SimulationResult(
        snippet_id=str(snippet_id),
        n_steps_total=(warmup + len(thigh_err)),
        n_steps_overridden=len(thigh_err),
        terminated_early=terminated_early,
        total_reward=float(total_reward),
        thigh_rmse_deg=float(np.rad2deg(thigh_rmse)),
        knee_rmse_deg=float(np.rad2deg(knee_rmse)),
    )


# Backwards-compatible alias for older code.
run_mocapact_simulation = run_mocapact_expert_simulation


def _make_sb3_predictor(policy: Any, *, deterministic: bool) -> Callable[[Any, Any], tuple[np.ndarray, Any]]:
    def predict(obs: Any, state: Any) -> tuple[np.ndarray, Any]:
        action, _state = policy.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32), state

    return predict


def _make_multiclip_predictor(policy: Any, *, deterministic: bool) -> Callable[[Any, Any], tuple[np.ndarray, Any]]:
    def predict(obs: Any, state: Any) -> tuple[np.ndarray, Any]:
        action, next_state = policy.predict(obs, state=state, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32), next_state

    return predict


def _launch_visualization(
    env: Any,
    policy_predict: Callable[[Any, Any], tuple[np.ndarray, Any]],
    override: OverrideConfig,
    thigh_deg: np.ndarray,
    knee_deg: np.ndarray,
    warmup_steps: int,
) -> None:
    try:
        from dm_control.viewer import application
    except Exception as e:
        raise RuntimeError("dm_control.viewer is required for visualization.") from e

    physics = env._env.physics
    idx_thigh = _find_actuator_index(physics, override.thigh_actuator)
    idx_knee = _find_actuator_index(physics, override.knee_actuator)
    thigh_lo, thigh_hi = _joint_range_for_actuator(physics, idx_thigh)
    knee_lo, knee_hi = _joint_range_for_actuator(physics, idx_knee)
    qpos_thigh = _to_rad(thigh_deg, sign=override.thigh_sign, offset_deg=override.thigh_offset_deg)
    qpos_knee = _to_rad(knee_deg, sign=override.knee_sign, offset_deg=override.knee_offset_deg)
    max_override = min(len(qpos_thigh), len(qpos_knee))

    step = 0
    state = None

    def policy_fn(time_step: Any) -> np.ndarray:
        nonlocal step, state
        obs = env.get_observation(time_step)
        action, state = policy_predict(obs, state)

        if warmup_steps <= step < warmup_steps + max_override:
            idx = int(step - warmup_steps)
            action[idx_thigh] = np.clip(
                _ctrl_from_qpos(float(qpos_thigh[idx]), lower=thigh_lo, upper=thigh_hi), -1.0, 1.0
            )
            action[idx_knee] = np.clip(
                _ctrl_from_qpos(float(qpos_knee[idx]), lower=knee_lo, upper=knee_hi), -1.0, 1.0
            )

        step += 1
        return action

    viewer_app = application.Application(title="MoCapAct Override", width=1024, height=768)
    viewer_app.launch(environment_loader=env.dm_env, policy=policy_fn)


def visualize_mocapact_expert(
    *,
    expert: ExpertPolicy,
    thigh_angle_deg: np.ndarray,
    knee_angle_deg: np.ndarray,
    override: OverrideConfig,
    deterministic_policy: bool = True,
    warmup_steps: int = 0,
) -> None:
    try:
        from mocapact.clip_expert import utils as clip_expert_utils  # type: ignore
        from mocapact.envs import tracking  # type: ignore
        from mocapact.sb3 import utils as sb3_utils  # type: ignore
        from mocapact import observables  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing mocapact dependencies for visualization.") from e

    env_kwargs = clip_expert_utils.make_env_kwargs(
        str(expert.clip_id),
        mocap_path=None,
        start_step=int(expert.start_step),
        end_step=int(expert.end_step),
        min_steps=10,
        ghost_offset=0.0,
        always_init_at_clip_start=True,
        termination_error_threshold=0.3,
        act_noise=0.0,
    )
    env = tracking.MocapTrackingGymEnv(**env_kwargs)
    policy = sb3_utils.load_policy(str(expert.model_dir), observables.TIME_INDEX_OBSERVABLES)
    predictor = _make_sb3_predictor(policy, deterministic=deterministic_policy)

    # Touch physics once before viewer to avoid lazy errors.
    _ = env._env.physics
    _launch_visualization(env, predictor, override, thigh_angle_deg, knee_angle_deg, int(warmup_steps))


def visualize_mocapact_multiclip(
    *,
    clip_id: str,
    start_step: int,
    end_step: int,
    policy: Any,
    thigh_angle_deg: np.ndarray,
    knee_angle_deg: np.ndarray,
    override: OverrideConfig,
    deterministic_policy: bool = True,
    warmup_steps: int = 0,
) -> None:
    try:
        from dm_control.locomotion.tasks.reference_pose import types  # type: ignore
        from mocapact.envs import tracking  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing mocapact dependencies for visualization.") from e

    dataset = types.ClipCollection(
        ids=[str(clip_id)],
        start_steps=[int(start_step)],
        end_steps=[int(end_step)],
    )

    ref_steps = getattr(policy, "ref_steps", (0,))
    task_kwargs = {
        "reward_type": "comic",
        "min_steps": 9,
        "ghost_offset": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "always_init_at_clip_start": True,
        "termination_error_threshold": 0.3,
    }
    env = tracking.MocapTrackingGymEnv(
        dataset=dataset,
        ref_steps=tuple(int(x) for x in ref_steps),
        act_noise=0.0,
        task_kwargs=task_kwargs,
    )

    predictor = _make_multiclip_predictor(policy, deterministic=deterministic_policy)
    _launch_visualization(env, predictor, override, thigh_angle_deg, knee_angle_deg, int(warmup_steps))
