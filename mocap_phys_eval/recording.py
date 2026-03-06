from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sim import (
    FALL_Z_THRESHOLD_M,
    OverrideSpec,
    _is_fall,
    _balance_metrics_step,
    _body_ids,
    _geom_ids,
    _ground_geom_ids,
    _root_z,
    _uprightness_cos_tilt,
    detect_balance_loss_step,
    highlight_overridden_leg,
    make_tracking_env,
    predict_fall_risk_from_traces,
    tint_geoms_by_prefix,
)

# Prosthetic knee tracking gains (GOOD/BAD only).
# The default CMU humanoid "position-controlled" actuators are intentionally compliant.
# For evaluation, we want the overridden knee angle to be physically realized (not just
# changing the policy action), so we increase knee servo gains for GOOD/BAD.
_PROSTHETIC_KNEE_KP = 800.0
_PROSTHETIC_KNEE_KD = 40.0
_PROSTHETIC_KNEE_FORCE = 800.0


@dataclass(frozen=True)
class CompareRecordingPaths:
    npz_path: Path
    gif_path: Path


def _qpos_addr_for_actuator(physics: Any, actuator_name: str) -> int:
    m = physics.model
    aid = int(m.name2id(str(actuator_name), "actuator"))
    try:
        jid = int(m.actuator_trnid[aid, 0])
    except Exception as e:
        raise RuntimeError(f"Could not resolve actuator->joint mapping for {actuator_name!r}") from e
    try:
        return int(m.jnt_qposadr[jid])
    except Exception as e:
        raise RuntimeError(f"Could not resolve joint->qpos adr for {actuator_name!r}") from e


def _joint_range_for_actuator(physics: Any, actuator_name: str) -> tuple[float, float]:
    m = physics.model
    aid = int(m.name2id(str(actuator_name), "actuator"))
    try:
        jid = int(m.actuator_trnid[aid, 0])
    except Exception as e:
        raise RuntimeError(f"Could not resolve actuator->joint mapping for {actuator_name!r}") from e
    try:
        rng = np.asarray(m.jnt_range[jid], dtype=np.float64).reshape(2)
        return float(rng[0]), float(rng[1])
    except Exception as e:
        raise RuntimeError(f"Could not read joint range for actuator {actuator_name!r}") from e


def _ctrl_from_qpos(target_qpos: float, *, lower: float, upper: float) -> float:
    """Inverse of dm_control's CMU humanoid position-actuator scaling."""
    lo = float(lower)
    hi = float(upper)
    if hi <= lo:
        return 0.0
    return float((2.0 * float(target_qpos) - (hi + lo)) / (hi - lo))


def _targets_qpos_rad(target_deg: np.ndarray, *, sign: float, offset_deg: float) -> np.ndarray:
    deg = np.asarray(target_deg, dtype=np.float64).reshape(-1)
    return np.deg2rad(float(sign) * deg + float(offset_deg)).astype(np.float64)


def _retune_general_position_actuator_pd(
    physics: Any,
    *,
    actuator_name: str,
    kp: float,
    kd: float,
    force: float,
) -> None:
    """Retune a CMU humanoid `<general>` actuator to behave like a PD position servo.

    This keeps the existing action semantics: `ctrl` in [-1,1] maps linearly to the
    joint range, but increases tracking strength/damping.
    """
    m = physics.model
    aid = int(m.name2id(str(actuator_name), "actuator"))
    jid = int(m.actuator_trnid[aid, 0])
    lo, hi = np.asarray(m.jnt_range[jid], dtype=np.float64).reshape(2).tolist()
    lo = float(lo)
    hi = float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return

    a = 0.5 * (hi - lo)  # q_des = a*ctrl + b
    b = 0.5 * (hi + lo)
    kp = float(kp)
    kd = float(kd)
    f = float(abs(force))

    # General actuator (joint transmission, affine bias):
    #   tau = (kp*a)*ctrl + (kp*b) + (-kp)*q + (-kd)*qvel
    try:
        m.actuator_gainprm[aid, 0] = float(kp * a)
        m.actuator_biasprm[aid, 0] = float(kp * b)
        m.actuator_biasprm[aid, 1] = float(-kp)
        m.actuator_biasprm[aid, 2] = float(-kd)
    except Exception:
        return
    try:
        m.actuator_forcerange[aid, 0] = float(-f)
        m.actuator_forcerange[aid, 1] = float(f)
    except Exception:
        pass


def record_compare_rollout(
    *,
    out_npz_path: str | Path,
    clip_id: str,
    start_step: int,
    end_step: int | None,
    primary_steps: int,
    warmup_steps: int = 0,
    policy: Any,
    override: OverrideSpec,
    knee_good_query_deg: np.ndarray,
    knee_bad_query_deg: np.ndarray,
    width: int,
    height: int,
    camera_id: int,
    deterministic_policy: bool = True,
    seed: int = 0,
) -> CompareRecordingPaths:
    """Record a (REF | GOOD | BAD) compare rollout to a replayable NPZ and a quick GIF."""
    out_npz_path = Path(out_npz_path).expanduser()
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import mujoco  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mujoco python package is required for recording.") from e

    # Optional overlay text (nice-to-have).
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:  # pragma: no cover
        Image = None  # type: ignore[assignment]
        ImageDraw = None  # type: ignore[assignment]
        ImageFont = None  # type: ignore[assignment]

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
    # Use the caller-provided end_step (typically the full clip end step) so
    # MoCapAct's time-normalized observables behave as intended. Only fall back
    # to a minimal slice if end_step is omitted.
    eval_start_step = int(start_step)
    warmup_steps = int(max(0, int(warmup_steps)))
    env_start_step = int(max(0, int(eval_start_step) - int(warmup_steps)))
    warmup = int(eval_start_step - env_start_step)

    # NOTE: ClipCollection end_steps are treated as inclusive indices in the
    # rest of this codebase (and in MoCapAct snippet naming: "<clip>-<start>-<end>").
    #
    # We need reference frames up to:
    #   eval_start_step + (primary_steps - 1) + max_ref_step
    min_needed_end = int(eval_start_step + max(0, int(primary_steps) - 1) + int(max_ref_step))
    use_end_step = None if end_step is None else int(max(int(end_step), int(min_needed_end)))

    env_ref = make_tracking_env(
        clip_id=str(clip_id),
        start_step=int(env_start_step),
        end_step=use_end_step,
        ref_steps=ref_steps_tuple,
        seed=int(seed),
    )
    env_good = make_tracking_env(
        clip_id=str(clip_id),
        start_step=int(env_start_step),
        end_step=use_end_step,
        ref_steps=ref_steps_tuple,
        seed=int(seed),
    )
    env_bad = make_tracking_env(
        clip_id=str(clip_id),
        start_step=int(env_start_step),
        end_step=use_end_step,
        ref_steps=ref_steps_tuple,
        seed=int(seed),
    )

    reset_ref = env_ref.reset()
    reset_good = env_good.reset()
    reset_bad = env_bad.reset()
    obs_ref = reset_ref[0] if isinstance(reset_ref, tuple) and len(reset_ref) == 2 else reset_ref
    obs_good = reset_good[0] if isinstance(reset_good, tuple) and len(reset_good) == 2 else reset_good
    obs_bad = reset_bad[0] if isinstance(reset_bad, tuple) and len(reset_bad) == 2 else reset_bad

    # IMPORTANT: do NOT patch the reference for GOOD/BAD.
    #
    # - The user wants the ghost to always display the original (matched) reference motion.
    # - MoCapAct snippet experts (TIME_INDEX_OBSERVABLES) don't condition on reference features.
    #   Physical overrides should only affect the *walker*, not the ghost.

    # Tint walkers/ghosts so REF/GOOD/BAD are visually distinct.
    for env in (env_ref, env_good, env_bad):
        tint_geoms_by_prefix(env._env.physics, prefix="ghost/", rgba=(0.85, 0.85, 0.85, 0.25))  # noqa: SLF001
    tint_geoms_by_prefix(env_ref._env.physics, prefix="walker/", rgba=(0.35, 0.35, 0.35, 1.0))  # noqa: SLF001
    tint_geoms_by_prefix(env_good._env.physics, prefix="walker/", rgba=(1.00, 0.55, 0.15, 1.0))  # noqa: SLF001
    tint_geoms_by_prefix(env_bad._env.physics, prefix="walker/", rgba=(0.15, 0.65, 1.00, 1.0))  # noqa: SLF001

    highlight_overridden_leg(env_good._env.physics, override=override)  # noqa: SLF001
    highlight_overridden_leg(env_bad._env.physics, override=override)  # noqa: SLF001

    phys_ref = env_ref._env.physics  # noqa: SLF001
    phys_good = env_good._env.physics  # noqa: SLF001
    phys_bad = env_bad._env.physics  # noqa: SLF001

    # Physical override (prosthetic knee):
    # We force the knee flexion actuator in GOOD/BAD each step so the RL policy
    # cannot directly control the knee trajectory.
    knee_act_id = int(phys_good.model.name2id(str(override.knee_actuator), "actuator"))
    knee_lo, knee_hi = _joint_range_for_actuator(phys_good, override.knee_actuator)
    knee_good_qpos = _targets_qpos_rad(knee_good_query_deg, sign=override.knee_sign, offset_deg=override.knee_offset_deg)
    knee_bad_qpos = _targets_qpos_rad(knee_bad_query_deg, sign=override.knee_sign, offset_deg=override.knee_offset_deg)

    # Many CMU humanoid actuators use an internal first-order filter (actuator_dyntype=filter)
    # with tau ~ control_timestep. Overriding ctrl alone can introduce a full-step lag, which
    # inflates the realized knee-angle RMSE even with high gains.
    #
    # To make the prosthetic knee track the requested target at the current step (research intent),
    # we also overwrite the actuator activation state for the overridden actuator.
    try:
        knee_actadr_good = int(phys_good.model.actuator_actadr[knee_act_id])
        knee_actnum_good = int(phys_good.model.actuator_actnum[knee_act_id])
    except Exception:
        knee_actadr_good = -1
        knee_actnum_good = 0
    try:
        knee_actadr_bad = int(phys_bad.model.actuator_actadr[knee_act_id])
        knee_actnum_bad = int(phys_bad.model.actuator_actnum[knee_act_id])
    except Exception:
        knee_actadr_bad = -1
        knee_actnum_bad = 0

    # Renderers (one per panel so they can have independent cameras in interactive replay).
    rend_ref = mujoco.Renderer(phys_ref.model.ptr, int(height), int(width))
    rend_good = mujoco.Renderer(phys_good.model.ptr, int(height), int(width))
    rend_bad = mujoco.Renderer(phys_bad.model.ptr, int(height), int(width))

    # Use MuJoCo's fixed camera if it exists; else default.
    def _cam_for(physics: Any) -> Any:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        try:
            if int(camera_id) >= 0:
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # type: ignore[attr-defined]
                cam.fixedcamid = int(camera_id)
        except Exception:
            pass
        return cam

    cam_ref = _cam_for(phys_ref)
    cam_good = _cam_for(phys_good)
    cam_bad = _cam_for(phys_bad)

    try:
        dt = float(getattr(env_ref._env.task, "_control_timestep", 0.03))  # noqa: SLF001
        if not math.isfinite(dt) or dt <= 0:
            dt = 0.03
    except Exception:
        dt = 0.03

    try:
        root_id_ref = int(phys_ref.model.name2id("walker/root", "body"))
    except Exception:
        root_id_ref = None
    try:
        root_id_good = int(phys_good.model.name2id("walker/root", "body"))
    except Exception:
        root_id_good = None
    try:
        root_id_bad = int(phys_bad.model.name2id("walker/root", "body"))
    except Exception:
        root_id_bad = None

    # Balance metrics (COM vs support).
    ground_ref = _ground_geom_ids(phys_ref)
    ground_good = _ground_geom_ids(phys_good)
    ground_bad = _ground_geom_ids(phys_bad)
    l_geoms_ref = _geom_ids(phys_ref, ("lfoot", "lfoot_ch", "ltoes0", "ltoes1", "ltoes2"))
    r_geoms_ref = _geom_ids(phys_ref, ("rfoot", "rfoot_ch", "rtoes0", "rtoes1", "rtoes2"))
    l_geoms_good = _geom_ids(phys_good, ("lfoot", "lfoot_ch", "ltoes0", "ltoes1", "ltoes2"))
    r_geoms_good = _geom_ids(phys_good, ("rfoot", "rfoot_ch", "rtoes0", "rtoes1", "rtoes2"))
    l_geoms_bad = _geom_ids(phys_bad, ("lfoot", "lfoot_ch", "ltoes0", "ltoes1", "ltoes2"))
    r_geoms_bad = _geom_ids(phys_bad, ("rfoot", "rfoot_ch", "rtoes0", "rtoes1", "rtoes2"))
    l_bodies_ref = _body_ids(phys_ref, ("walker/lfoot", "walker/ltoes"))
    r_bodies_ref = _body_ids(phys_ref, ("walker/rfoot", "walker/rtoes"))
    l_bodies_good = _body_ids(phys_good, ("walker/lfoot", "walker/ltoes"))
    r_bodies_good = _body_ids(phys_good, ("walker/rfoot", "walker/rtoes"))
    l_bodies_bad = _body_ids(phys_bad, ("walker/lfoot", "walker/ltoes"))
    r_bodies_bad = _body_ids(phys_bad, ("walker/rfoot", "walker/rtoes"))

    def _bal_state() -> dict[str, Any]:
        return {
            "prev_com": None,
            "min_margin": float("inf"),
            "margin_end": float("nan"),
            "outside": 0,
            "count": 0,
            "tail_margin": [],
            "max_outward": 0.0,
            # Full trace (per recorded step). This is used for debugging + plots,
            # and makes it much easier to diagnose bad risk estimates.
            "margin_trace": [],
            "xcom_margin_trace": [],
        }

    bal_ref = _bal_state()
    bal_good = _bal_state()
    bal_bad = _bal_state()

    # Joint qpos addr for actual angles.
    knee_adr_ref = _qpos_addr_for_actuator(phys_ref, override.knee_actuator)
    knee_adr_good = _qpos_addr_for_actuator(phys_good, override.knee_actuator)
    knee_adr_bad = _qpos_addr_for_actuator(phys_bad, override.knee_actuator)

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

    def _step(env: Any, action: np.ndarray) -> tuple[Any, bool]:
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs2, _reward, terminated, truncated, _info = step_out
            done = bool(terminated or truncated)
        else:
            obs2, _reward, done, _info = step_out
        return obs2, bool(done)

    def _update_balance(
        *,
        physics: Any,
        root_id: int | None,
        ground: set[int],
        l_geoms: set[int],
        r_geoms: set[int],
        l_bodies: list[int],
        r_bodies: list[int],
        state: dict[str, Any],
    ) -> None:
        prev = state.get("prev_com", None)
        com_xy, com_vel_xy, xcom_xy, com_margin, xcom_margin, had_contact, centroid = _balance_metrics_step(
            physics=physics,
            root_body_id=root_id,
            left_foot_geoms=l_geoms,
            right_foot_geoms=r_geoms,
            ground_geoms=ground,
            left_foot_bodies=l_bodies,
            right_foot_bodies=r_bodies,
            dt=float(dt),
            prev_com_xy=prev if isinstance(prev, np.ndarray) else None,
        )
        state["prev_com"] = com_xy
        try:
            mt = state.get("margin_trace", None)
            if isinstance(mt, list):
                mt.append(float(com_margin))
            else:
                state["margin_trace"] = [float(com_margin)]
            xmt = state.get("xcom_margin_trace", None)
            if isinstance(xmt, list):
                xmt.append(float(xcom_margin))
            else:
                state["xcom_margin_trace"] = [float(xcom_margin)]
            hct = state.get("had_contact_trace", None)
            if isinstance(hct, list):
                hct.append(bool(had_contact))
            else:
                state["had_contact_trace"] = [bool(had_contact)]
        except Exception:
            pass
        if not math.isfinite(float(com_margin)):
            return
        state["min_margin"] = min(float(state.get("min_margin", float("inf"))), float(com_margin))
        state["margin_end"] = float(com_margin)
        state["count"] = int(state.get("count", 0)) + 1
        if float(com_margin) < 0.0:
            state["outside"] = int(state.get("outside", 0)) + 1
        tm = state.get("tail_margin", [])
        if isinstance(tm, list):
            tm.append(float(com_margin))
            if len(tm) > 10:
                tm[:] = tm[-10:]
        state["tail_margin"] = tm
        if np.all(np.isfinite(centroid)) and np.all(np.isfinite(xcom_xy)) and np.all(np.isfinite(com_vel_xy)):
            v = xcom_xy - centroid
            nv = float(np.linalg.norm(v))
            if nv > 1e-6:
                outward = float(np.dot(com_vel_xy, v / nv))
                state["max_outward"] = max(float(state.get("max_outward", 0.0)), float(outward))

    font = None
    if ImageFont is not None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    # Per-step traces (for plots + overlay).
    steps = int(primary_steps)
    frames: list[np.ndarray] = []
    states_ref: list[np.ndarray] = []
    states_good: list[np.ndarray] = []
    states_bad: list[np.ndarray] = []
    root_z_ref: list[float] = []
    root_z_good: list[float] = []
    root_z_bad: list[float] = []
    upright_ref: list[float] = []
    upright_good: list[float] = []
    upright_bad: list[float] = []
    knee_ref_actual: list[float] = []
    knee_good_actual: list[float] = []
    knee_bad_actual: list[float] = []

    # Control override diagnostics: policy output vs applied action for the forced actuator.
    # This is the quickest way to prove the RL policy is NOT controlling the prosthetic knee.
    knee_ctrl_ref_policy: list[float] = []
    knee_ctrl_good_policy: list[float] = []
    knee_ctrl_bad_policy: list[float] = []

    knee_ctrl_good_applied: list[float] = []
    knee_ctrl_bad_applied: list[float] = []

    knee_ctrl_good_target: list[float] = []
    knee_ctrl_bad_target: list[float] = []

    state_ref: Any = None
    state_good: Any = None
    state_bad: Any = None

    done_any = False
    fell_ref_hard = False
    fell_good_hard = False
    fell_bad_hard = False

    # Warmup: start earlier so we don't reset mid-motion, then roll forward to the
    # evaluation start. We do NOT apply prosthetic forcing during warmup.
    if int(warmup) > 0:
        for _t in range(int(warmup)):
            act_ref, state_ref = _predict(obs_ref, state_ref)
            act_good_pol, state_good = _predict(obs_good, state_good)
            act_bad_pol, state_bad = _predict(obs_bad, state_bad)

            obs_ref, done_ref = _step(env_ref, np.asarray(act_ref, dtype=np.float32).reshape(-1))
            obs_good, done_good = _step(env_good, np.asarray(act_good_pol, dtype=np.float32).reshape(-1))
            obs_bad, done_bad = _step(env_bad, np.asarray(act_bad_pol, dtype=np.float32).reshape(-1))
            done_any = bool(done_any or done_ref or done_good or done_bad)
            if done_any:
                break

    # Prosthetic knee: retune knee actuator gains for GOOD/BAD so the override is
    # realized in joint space (otherwise it's often too compliant to matter).
    #
    # IMPORTANT: do this *after* warmup so REF/GOOD/BAD start the evaluation
    # window from comparable physical states.
    _retune_general_position_actuator_pd(
        phys_good,
        actuator_name=str(override.knee_actuator),
        kp=float(_PROSTHETIC_KNEE_KP),
        kd=float(_PROSTHETIC_KNEE_KD),
        force=float(_PROSTHETIC_KNEE_FORCE),
    )
    _retune_general_position_actuator_pd(
        phys_bad,
        actuator_name=str(override.knee_actuator),
        kp=float(_PROSTHETIC_KNEE_KP),
        kd=float(_PROSTHETIC_KNEE_KD),
        force=float(_PROSTHETIC_KNEE_FORCE),
    )

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        tqdm = None  # type: ignore[assignment]

    step_it = range(steps)
    if tqdm is not None:
        step_it = tqdm(step_it, desc="Simulating (REF|GOOD|BAD)", unit="step", leave=False)

    for t in step_it:
        # REF/GOOD/BAD each run the policy normally, but GOOD/BAD have their
        # prosthetic knee actuator physically forced each step so the RL
        # controller cannot directly command it.
        act_ref, state_ref = _predict(obs_ref, state_ref)
        act_good_pol, state_good = _predict(obs_good, state_good)
        act_bad_pol, state_bad = _predict(obs_bad, state_bad)

        try:
            ar = np.asarray(act_ref, dtype=np.float32).reshape(-1)
        except Exception:
            ar = np.zeros((0,), dtype=np.float32)
        try:
            ag = np.asarray(act_good_pol, dtype=np.float32).reshape(-1)
        except Exception:
            ag = np.zeros((0,), dtype=np.float32)
        try:
            ab = np.asarray(act_bad_pol, dtype=np.float32).reshape(-1)
        except Exception:
            ab = np.zeros((0,), dtype=np.float32)

        # Record "policy outputs" (pre-override).
        try:
            knee_ctrl_ref_policy.append(float(ar[knee_act_id]))
        except Exception:
            knee_ctrl_ref_policy.append(float("nan"))
        try:
            knee_ctrl_good_policy.append(float(ag[knee_act_id]))
        except Exception:
            knee_ctrl_good_policy.append(float("nan"))
        try:
            knee_ctrl_bad_policy.append(float(ab[knee_act_id]))
        except Exception:
            knee_ctrl_bad_policy.append(float("nan"))

        # Start from each env's policy action, then force the overridden knee actuator.
        act_good = np.asarray(ag, dtype=np.float32).reshape(-1).copy()
        act_bad = np.asarray(ab, dtype=np.float32).reshape(-1).copy()

        kn_good_des = float(knee_good_qpos[min(t, int(knee_good_qpos.size) - 1)])
        kn_bad_des = float(knee_bad_qpos[min(t, int(knee_bad_qpos.size) - 1)])

        kn_good_ctrl = float(np.clip(_ctrl_from_qpos(kn_good_des, lower=knee_lo, upper=knee_hi), -1.0, 1.0))
        kn_bad_ctrl = float(np.clip(_ctrl_from_qpos(kn_bad_des, lower=knee_lo, upper=knee_hi), -1.0, 1.0))
        act_good[knee_act_id] = kn_good_ctrl
        act_bad[knee_act_id] = kn_bad_ctrl

        # Record applied controls (after overriding) + targets.
        knee_ctrl_good_applied.append(float(act_good[knee_act_id]))
        knee_ctrl_bad_applied.append(float(act_bad[knee_act_id]))
        knee_ctrl_good_target.append(float(kn_good_ctrl))
        knee_ctrl_bad_target.append(float(kn_bad_ctrl))

        # Also overwrite the actuator's activation state to avoid a 1-step lag from
        # internal filtering dynamics (common in the CMU humanoid position actuators).
        if int(knee_actnum_good) > 0 and int(knee_actadr_good) >= 0:
            try:
                phys_good.data.act[int(knee_actadr_good) : int(knee_actadr_good) + int(knee_actnum_good)] = float(
                    act_good[knee_act_id]
                )
            except Exception:
                pass
        if int(knee_actnum_bad) > 0 and int(knee_actadr_bad) >= 0:
            try:
                phys_bad.data.act[int(knee_actadr_bad) : int(knee_actadr_bad) + int(knee_actnum_bad)] = float(
                    act_bad[knee_act_id]
                )
            except Exception:
                pass

        obs_ref, done_ref = _step(env_ref, np.asarray(ar, dtype=np.float32).reshape(-1))
        obs_good, done_good = _step(env_good, act_good)
        obs_bad, done_bad = _step(env_bad, act_bad)
        done_any = bool(done_any or done_ref or done_good or done_bad)

        # Record states for interactive replay.
        try:
            states_ref.append(np.asarray(phys_ref.get_state(), dtype=np.float32))
            states_good.append(np.asarray(phys_good.get_state(), dtype=np.float32))
            states_bad.append(np.asarray(phys_bad.get_state(), dtype=np.float32))
        except Exception:
            # States are optional; the NPZ still contains frames.
            pass

        # Root + upright.
        zr = _root_z(phys_ref, body_id=root_id_ref)
        zg = _root_z(phys_good, body_id=root_id_good)
        zb = _root_z(phys_bad, body_id=root_id_bad)
        ur = _uprightness_cos_tilt(phys_ref, body_id=root_id_ref)
        ug = _uprightness_cos_tilt(phys_good, body_id=root_id_good)
        ub = _uprightness_cos_tilt(phys_bad, body_id=root_id_bad)
        root_z_ref.append(float(zr))
        root_z_good.append(float(zg))
        root_z_bad.append(float(zb))
        upright_ref.append(float(ur))
        upright_good.append(float(ug))
        upright_bad.append(float(ub))

        # Hard-fall detection (separate from env termination). This avoids
        # misclassifying "end of clip" termination as a fall.
        try:
            fell_ref_hard = bool(fell_ref_hard or _is_fall(float(zr), float(ur)))
            fell_good_hard = bool(fell_good_hard or _is_fall(float(zg), float(ug)))
            fell_bad_hard = bool(fell_bad_hard or _is_fall(float(zb), float(ub)))
        except Exception:
            pass

        # Actual overridden joints.
        knee_ref_actual.append(float(np.rad2deg(float(np.asarray(phys_ref.data.qpos[knee_adr_ref]).reshape(())))))
        knee_good_actual.append(float(np.rad2deg(float(np.asarray(phys_good.data.qpos[knee_adr_good]).reshape(())))))
        knee_bad_actual.append(float(np.rad2deg(float(np.asarray(phys_bad.data.qpos[knee_adr_bad]).reshape(())))))

        _update_balance(physics=phys_ref, root_id=root_id_ref, ground=ground_ref, l_geoms=l_geoms_ref, r_geoms=r_geoms_ref, l_bodies=l_bodies_ref, r_bodies=r_bodies_ref, state=bal_ref)
        _update_balance(physics=phys_good, root_id=root_id_good, ground=ground_good, l_geoms=l_geoms_good, r_geoms=r_geoms_good, l_bodies=l_bodies_good, r_bodies=r_bodies_good, state=bal_good)
        _update_balance(physics=phys_bad, root_id=root_id_bad, ground=ground_bad, l_geoms=l_geoms_bad, r_geoms=r_geoms_bad, l_bodies=l_bodies_bad, r_bodies=r_bodies_bad, state=bal_bad)

        # Render panels.
        rend_ref.update_scene(phys_ref.data.ptr, camera=cam_ref)
        rend_good.update_scene(phys_good.data.ptr, camera=cam_good)
        rend_bad.update_scene(phys_bad.data.ptr, camera=cam_bad)
        img_ref = np.asarray(rend_ref.render(), dtype=np.uint8)
        img_good = np.asarray(rend_good.render(), dtype=np.uint8)
        img_bad = np.asarray(rend_bad.render(), dtype=np.uint8)
        frame = np.concatenate([img_ref, img_good, img_bad], axis=1)

        if Image is not None and ImageDraw is not None:
            pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil)
            pad = 8
            w = int(width)
            draw.text((pad + 0 * w, pad), "REF (no override)", fill=(255, 255, 255), font=font)
            draw.text((pad + 1 * w, pad), "GOOD (override)", fill=(255, 255, 255), font=font)
            draw.text((pad + 2 * w, pad), "BAD (override)", fill=(255, 255, 255), font=font)
            frame = np.asarray(pil, dtype=np.uint8)

        frames.append(frame)

        # Hard stop if someone already fell badly; keeps GIF readable.
        if done_any or fell_ref_hard or fell_good_hard or fell_bad_hard:
            break

    # Predict stability ("fall risk") from balance traces.
    #
    # We avoid root-height based heuristics (crouching/kneeling should be OK) and use:
    # - uprightness (cos tilt)
    # - XCoM support margin (signed distance to support polygon)
    com_margin_ref = np.asarray(bal_ref.get("margin_trace", [])[: len(upright_ref)], dtype=np.float32)
    com_margin_good = np.asarray(bal_good.get("margin_trace", [])[: len(upright_good)], dtype=np.float32)
    com_margin_bad = np.asarray(bal_bad.get("margin_trace", [])[: len(upright_bad)], dtype=np.float32)
    xcom_margin_ref = np.asarray(bal_ref.get("xcom_margin_trace", [])[: len(upright_ref)], dtype=np.float32)
    xcom_margin_good = np.asarray(bal_good.get("xcom_margin_trace", [])[: len(upright_good)], dtype=np.float32)
    xcom_margin_bad = np.asarray(bal_bad.get("xcom_margin_trace", [])[: len(upright_bad)], dtype=np.float32)
    had_contact_ref = np.asarray(bal_ref.get("had_contact_trace", [])[: len(upright_ref)], dtype=np.bool_)
    had_contact_good = np.asarray(bal_good.get("had_contact_trace", [])[: len(upright_good)], dtype=np.bool_)
    had_contact_bad = np.asarray(bal_bad.get("had_contact_trace", [])[: len(upright_bad)], dtype=np.bool_)

    fell_ref = bool(fell_ref_hard)
    fell_good = bool(fell_good_hard)
    fell_bad = bool(fell_bad_hard)

    risk_ref, likely_ref, risk_trace_ref = predict_fall_risk_from_traces(
        fell=fell_ref,
        upright=np.asarray(upright_ref, dtype=np.float32),
        com_margin_m=com_margin_ref,
        xcom_margin_m=xcom_margin_ref,
        had_contact=had_contact_ref,
        dt=float(dt),
    )
    risk_good, likely_good, risk_trace_good = predict_fall_risk_from_traces(
        fell=fell_good,
        upright=np.asarray(upright_good, dtype=np.float32),
        com_margin_m=com_margin_good,
        xcom_margin_m=xcom_margin_good,
        had_contact=had_contact_good,
        dt=float(dt),
    )
    risk_bad, likely_bad, risk_trace_bad = predict_fall_risk_from_traces(
        fell=fell_bad,
        upright=np.asarray(upright_bad, dtype=np.float32),
        com_margin_m=com_margin_bad,
        xcom_margin_m=xcom_margin_bad,
        had_contact=had_contact_bad,
        dt=float(dt),
    )

    # "Balance loss" step is the first time we consider the walker unstable, even if it
    # hasn't fully fallen within the short evaluation window.
    balance_loss_step_ref = detect_balance_loss_step(risk_trace=risk_trace_ref, upright=np.asarray(upright_ref, dtype=np.float32))
    balance_loss_step_good = detect_balance_loss_step(risk_trace=risk_trace_good, upright=np.asarray(upright_good, dtype=np.float32))
    balance_loss_step_bad = detect_balance_loss_step(risk_trace=risk_trace_bad, upright=np.asarray(upright_bad, dtype=np.float32))

    # Stack.
    arr = np.stack(frames, axis=0).astype(np.uint8)
    states_ref_arr = np.stack(states_ref, axis=0).astype(np.float32) if states_ref else np.zeros((0, 0), dtype=np.float32)
    states_good_arr = np.stack(states_good, axis=0).astype(np.float32) if states_good else np.zeros((0, 0), dtype=np.float32)
    states_bad_arr = np.stack(states_bad, axis=0).astype(np.float32) if states_bad else np.zeros((0, 0), dtype=np.float32)

    np.savez_compressed(
        out_npz_path,
        frames=arr,
        dt=np.asarray(float(dt), dtype=np.float32),
        width=np.asarray(int(width), dtype=np.int32),
        height=np.asarray(int(height), dtype=np.int32),
        camera_id=np.asarray(int(camera_id), dtype=np.int32),
        clip_id=np.asarray(str(clip_id)),
        env_start_step=np.asarray(int(env_start_step), dtype=np.int64),
        warmup_steps=np.asarray(int(warmup), dtype=np.int64),
        start_step=np.asarray(int(start_step), dtype=np.int64),
        end_step=np.asarray(int(use_end_step if use_end_step is not None else min_needed_end), dtype=np.int64),
        primary_steps=np.asarray(int(steps), dtype=np.int64),
        override_knee_actuator=np.asarray(str(override.knee_actuator)),
        override_knee_sign=np.asarray(float(override.knee_sign), dtype=np.float32),
        override_knee_offset_deg=np.asarray(float(override.knee_offset_deg), dtype=np.float32),
        prosthetic_knee_kp=np.asarray(float(_PROSTHETIC_KNEE_KP), dtype=np.float32),
        prosthetic_knee_kd=np.asarray(float(_PROSTHETIC_KNEE_KD), dtype=np.float32),
        prosthetic_knee_force=np.asarray(float(_PROSTHETIC_KNEE_FORCE), dtype=np.float32),
        states_ref=states_ref_arr,
        states_good=states_good_arr,
        states_bad=states_bad_arr,
        fall_z_threshold_m=np.asarray(float(FALL_Z_THRESHOLD_M), dtype=np.float32),
        # Backwards-compat: fall_step_* now means "balance loss step" (not root-height based).
        fall_step_ref=np.asarray(int(balance_loss_step_ref), dtype=np.int64),
        fall_step_good=np.asarray(int(balance_loss_step_good), dtype=np.int64),
        fall_step_bad=np.asarray(int(balance_loss_step_bad), dtype=np.int64),
        balance_loss_step_ref=np.asarray(int(balance_loss_step_ref), dtype=np.int64),
        balance_loss_step_good=np.asarray(int(balance_loss_step_good), dtype=np.int64),
        balance_loss_step_bad=np.asarray(int(balance_loss_step_bad), dtype=np.int64),
        predicted_fall_risk_ref=np.asarray(float(risk_ref), dtype=np.float32),
        predicted_fall_risk_good=np.asarray(float(risk_good), dtype=np.float32),
        predicted_fall_risk_bad=np.asarray(float(risk_bad), dtype=np.float32),
        predicted_fall_risk_trace_ref=np.asarray(risk_trace_ref, dtype=np.float32),
        predicted_fall_risk_trace_good=np.asarray(risk_trace_good, dtype=np.float32),
        predicted_fall_risk_trace_bad=np.asarray(risk_trace_bad, dtype=np.float32),
        predicted_likely_fall_ref=np.asarray(bool(likely_ref), dtype=np.bool_),
        predicted_likely_fall_good=np.asarray(bool(likely_good), dtype=np.bool_),
        predicted_likely_fall_bad=np.asarray(bool(likely_bad), dtype=np.bool_),
        balance_margin_ref_m=np.asarray(bal_ref.get("margin_trace", [])[: len(root_z_ref)], dtype=np.float32),
        balance_margin_good_m=np.asarray(bal_good.get("margin_trace", [])[: len(root_z_good)], dtype=np.float32),
        balance_margin_bad_m=np.asarray(bal_bad.get("margin_trace", [])[: len(root_z_bad)], dtype=np.float32),
        balance_xcom_margin_ref_m=np.asarray(bal_ref.get("xcom_margin_trace", [])[: len(root_z_ref)], dtype=np.float32),
        balance_xcom_margin_good_m=np.asarray(
            bal_good.get("xcom_margin_trace", [])[: len(root_z_good)], dtype=np.float32
        ),
        balance_xcom_margin_bad_m=np.asarray(bal_bad.get("xcom_margin_trace", [])[: len(root_z_bad)], dtype=np.float32),
        root_z_ref_m=np.asarray(root_z_ref, dtype=np.float32),
        root_z_good_m=np.asarray(root_z_good, dtype=np.float32),
        root_z_bad_m=np.asarray(root_z_bad, dtype=np.float32),
        upright_ref=np.asarray(upright_ref, dtype=np.float32),
        upright_good=np.asarray(upright_good, dtype=np.float32),
        upright_bad=np.asarray(upright_bad, dtype=np.float32),
        knee_ref_actual_deg=np.asarray(knee_ref_actual, dtype=np.float32),
        knee_good_actual_deg=np.asarray(knee_good_actual, dtype=np.float32),
        knee_bad_actual_deg=np.asarray(knee_bad_actual, dtype=np.float32),
        knee_good_query_deg=np.asarray(knee_good_query_deg[:steps], dtype=np.float32),
        knee_bad_query_deg=np.asarray(knee_bad_query_deg[:steps], dtype=np.float32),
        ctrl_knee_ref_policy=np.asarray(knee_ctrl_ref_policy[:steps], dtype=np.float32),
        ctrl_knee_good_policy=np.asarray(knee_ctrl_good_policy[:steps], dtype=np.float32),
        ctrl_knee_bad_policy=np.asarray(knee_ctrl_bad_policy[:steps], dtype=np.float32),
        ctrl_knee_good_applied=np.asarray(knee_ctrl_good_applied[:steps], dtype=np.float32),
        ctrl_knee_bad_applied=np.asarray(knee_ctrl_bad_applied[:steps], dtype=np.float32),
        ctrl_knee_good_target=np.asarray(knee_ctrl_good_target[:steps], dtype=np.float32),
        ctrl_knee_bad_target=np.asarray(knee_ctrl_bad_target[:steps], dtype=np.float32),
    )

    gif_path = out_npz_path.with_suffix(".gif")
    if Image is not None:
        try:
            duration_ms = int(max(1, round(float(dt) * 1000.0)))
            base = Image.fromarray(arr[0]).convert("P", palette=Image.ADAPTIVE, colors=256)
            imgs = [base]
            for i in range(1, int(arr.shape[0])):
                imgs.append(Image.fromarray(arr[i]).quantize(palette=base))
            imgs[0].save(
                gif_path,
                save_all=True,
                append_images=imgs[1:],
                duration=duration_ms,
                loop=0,
                disposal=2,
            )
        except Exception:
            pass

    return CompareRecordingPaths(npz_path=out_npz_path, gif_path=gif_path)
