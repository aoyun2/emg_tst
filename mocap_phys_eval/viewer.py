from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .sim import OverrideSpec, highlight_overridden_leg, make_tracking_env, tint_geoms_by_prefix


def view_compare_npz(path: str | Path) -> None:
    """Interactive compare replay (REF | GOOD | BAD) with per-panel camera controls."""
    npz_path = Path(path).expanduser()
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    try:
        import tkinter as tk
    except Exception as e:  # pragma: no cover
        raise RuntimeError("tkinter is required for the interactive viewer on Windows.") from e

    try:
        from PIL import Image, ImageDraw, ImageFont, ImageTk  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Pillow (PIL) is required for the interactive viewer.") from e

    try:
        import mujoco  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("mujoco python package is required for the interactive viewer.") from e

    raw = np.load(npz_path, allow_pickle=True)
    frames = np.asarray(raw.get("frames", np.zeros((0, 0, 0, 3), dtype=np.uint8)), dtype=np.uint8)
    dt = float(np.asarray(raw.get("dt", 0.03)).reshape(()))
    width = int(np.asarray(raw.get("width", 480)).reshape(()))
    height = int(np.asarray(raw.get("height", 360)).reshape(()))

    clip_id = str(np.asarray(raw.get("clip_id", "")).reshape(()))
    start_step = int(np.asarray(raw.get("start_step", 0)).reshape(()))
    end_step = int(np.asarray(raw.get("end_step", start_step + 100)).reshape(()))

    states_ref = np.asarray(raw.get("states_ref", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    states_good = np.asarray(raw.get("states_good", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
    states_bad = np.asarray(raw.get("states_bad", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)

    has_states = bool(states_ref.ndim == 2 and states_ref.shape[0] >= 2 and states_ref.shape[1] >= 1)
    if not has_states:
        raise RuntimeError("Recording does not contain physics states (states_ref/...).")

    override = OverrideSpec(
        thigh_actuator=str(np.asarray(raw.get("override_thigh_actuator", "walker/rfemurrx")).reshape(())),
        knee_actuator=str(np.asarray(raw.get("override_knee_actuator", "walker/rtibiarx")).reshape(())),
        thigh_sign=float(np.asarray(raw.get("override_thigh_sign", 1.0)).reshape(())),
        knee_sign=float(np.asarray(raw.get("override_knee_sign", 1.0)).reshape(())),
        thigh_offset_deg=float(np.asarray(raw.get("override_thigh_offset_deg", 0.0)).reshape(())),
        knee_offset_deg=float(np.asarray(raw.get("override_knee_offset_deg", 0.0)).reshape(())),
    )

    # Scalars / traces for status text.
    # Backwards-compat: fall_step_* was repurposed to "balance loss step". Prefer the explicit key.
    fall_step_ref = int(np.asarray(raw.get("balance_loss_step_ref", raw.get("fall_step_ref", -1))).reshape(()))
    fall_step_good = int(np.asarray(raw.get("balance_loss_step_good", raw.get("fall_step_good", -1))).reshape(()))
    fall_step_bad = int(np.asarray(raw.get("balance_loss_step_bad", raw.get("fall_step_bad", -1))).reshape(()))
    risk_ref = float(np.asarray(raw.get("predicted_fall_risk_ref", float("nan"))).reshape(()))
    risk_good = float(np.asarray(raw.get("predicted_fall_risk_good", float("nan"))).reshape(()))
    risk_bad = float(np.asarray(raw.get("predicted_fall_risk_bad", float("nan"))).reshape(()))
    risk_trace_ref = np.asarray(raw.get("predicted_fall_risk_trace_ref", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    risk_trace_good = np.asarray(raw.get("predicted_fall_risk_trace_good", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    risk_trace_bad = np.asarray(raw.get("predicted_fall_risk_trace_bad", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    likely_ref = bool(np.asarray(raw.get("predicted_likely_fall_ref", False)).reshape(()))
    likely_good = bool(np.asarray(raw.get("predicted_likely_fall_good", False)).reshape(()))
    likely_bad = bool(np.asarray(raw.get("predicted_likely_fall_bad", False)).reshape(()))
    root_z_ref = np.asarray(raw.get("root_z_ref_m", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    root_z_good = np.asarray(raw.get("root_z_good_m", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    root_z_bad = np.asarray(raw.get("root_z_bad_m", np.zeros((0,), dtype=np.float32)), dtype=np.float32).reshape(-1)

    # Create fresh envs only to obtain matching MuJoCo models for rendering.
    # The physics states drive everything during replay.
    # Policy ref_steps are irrelevant here; use a minimal default.
    env_ref = make_tracking_env(clip_id=clip_id, start_step=start_step, end_step=end_step, ref_steps=(0,), seed=0)
    env_good = make_tracking_env(clip_id=clip_id, start_step=start_step, end_step=end_step, ref_steps=(0,), seed=0)
    env_bad = make_tracking_env(clip_id=clip_id, start_step=start_step, end_step=end_step, ref_steps=(0,), seed=0)
    env_ref.reset()
    env_good.reset()
    env_bad.reset()

    # Tint + highlight.
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

    rend_ref = mujoco.Renderer(phys_ref.model.ptr, int(height), int(width))
    rend_good = mujoco.Renderer(phys_good.model.ptr, int(height), int(width))
    rend_bad = mujoco.Renderer(phys_bad.model.ptr, int(height), int(width))

    cam_ref = mujoco.MjvCamera()
    cam_good = mujoco.MjvCamera()
    cam_bad = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam_ref)
    mujoco.mjv_defaultCamera(cam_good)
    mujoco.mjv_defaultCamera(cam_bad)

    panels = [
        {"name": "REF", "phys": phys_ref, "rend": rend_ref, "cam": cam_ref},
        {"name": "GOOD", "phys": phys_good, "rend": rend_good, "cam": cam_good},
        {"name": "BAD", "phys": phys_bad, "rend": rend_bad, "cam": cam_bad},
    ]

    root = tk.Tk()
    root.title(f"MoCapAct Compare Replay (REF | GOOD | BAD)  clip={clip_id}  start={start_step}  [{npz_path.name}]")

    img_lbl = tk.Label(root)
    img_lbl.pack()

    info = tk.Frame(root)
    info.pack(fill="x", padx=8, pady=(4, 0))
    ref_info = tk.Label(info, text="", anchor="w", justify="left", fg="#444444")
    good_info = tk.Label(info, text="", anchor="w", justify="left", fg="#d55e00")
    bad_info = tk.Label(info, text="", anchor="w", justify="left", fg="#0072b2")
    ref_info.pack(side="top", fill="x")
    good_info.pack(side="top", fill="x")
    bad_info.pack(side="top", fill="x")

    controls = tk.Frame(root)
    controls.pack(fill="x", padx=8, pady=(6, 8))
    cur_idx = tk.IntVar(value=0)
    scale = tk.Scale(
        controls,
        from_=0,
        to=max(0, int(states_ref.shape[0]) - 1),
        orient="horizontal",
        variable=cur_idx,
        showvalue=True,
        length=int(width) * 3,
    )
    scale.pack(side="top", fill="x")

    btn_row = tk.Frame(controls)
    btn_row.pack(side="top", fill="x", pady=(6, 0))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def _render_pil() -> Any:
        ims: list[np.ndarray] = []
        for p in panels:
            rr = p["rend"]
            pp = p["phys"]
            cc = p["cam"]
            rr.update_scene(pp.data.ptr, camera=cc)
            ims.append(np.asarray(rr.render(), dtype=np.uint8))
        frame = np.concatenate(ims, axis=1)
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        pad = 8
        w = int(width)
        draw.text((pad + 0 * w, pad), "REF (no override)", fill=(255, 255, 255), font=font)
        draw.text((pad + 1 * w, pad), "GOOD (override)", fill=(255, 255, 255), font=font)
        draw.text((pad + 2 * w, pad), "BAD (override)", fill=(255, 255, 255), font=font)
        draw.text(
            (pad, int(height) - 18),
            "Mouse: RMB-drag rotate, LMB-drag pan, wheel zoom. Keys: WASD/Arrows move, Q/E (PgUp/PgDn) up/down, Shift=fast, 1/2/3 select panel, r reset.",
            fill=(255, 255, 255),
            font=font,
        )
        return pil

    def _show_pil(pil_img: Any) -> None:
        tk_img = ImageTk.PhotoImage(pil_img)
        img_lbl.configure(image=tk_img)
        img_lbl.image = tk_img  # keep alive

    def _panel_from_x(x: int) -> int:
        w = int(width)
        if w <= 0:
            return 0
        return int(max(0, min(2, int(x) // w)))

    active_panel = {"idx": 0}
    drag = {"active": False, "btn": 0, "x": 0, "y": 0}

    def _on_mouse_down(ev: Any) -> None:
        drag["active"] = True
        drag["btn"] = int(getattr(ev, "num", 0))
        drag["x"] = int(getattr(ev, "x", 0))
        drag["y"] = int(getattr(ev, "y", 0))
        active_panel["idx"] = _panel_from_x(int(getattr(ev, "x", 0)))

    def _on_mouse_up(_ev: Any) -> None:
        drag["active"] = False

    def _on_mouse_move(ev: Any) -> None:
        if not drag["active"]:
            return
        x = int(getattr(ev, "x", 0))
        y = int(getattr(ev, "y", 0))
        dx = float(x - int(drag["x"]))
        dy = float(y - int(drag["y"]))
        drag["x"] = x
        drag["y"] = y

        p = panels[int(active_panel["idx"])]
        cam = p["cam"]
        btn = int(drag["btn"])
        # RMB rotate, LMB pan.
        if btn == 3:
            cam.azimuth = float(cam.azimuth) - 0.2 * dx
            cam.elevation = float(cam.elevation) - 0.2 * dy
        else:
            # Pan in the ground plane (camera-relative), so it feels consistent
            # regardless of the current azimuth.
            az = float(cam.azimuth) * (math.pi / 180.0)
            fwd = np.asarray([math.cos(az), math.sin(az), 0.0], dtype=np.float64)
            right = np.asarray([-math.sin(az), math.cos(az), 0.0], dtype=np.float64)
            pan = (-0.002 * dx) * right + (0.002 * dy) * fwd
            cam.lookat[0] = float(cam.lookat[0]) + float(pan[0])
            cam.lookat[1] = float(cam.lookat[1]) + float(pan[1])
        _render_at(int(cur_idx.get()))

    def _on_wheel(ev: Any) -> None:
        delta = float(getattr(ev, "delta", 0.0))
        if delta == 0.0:
            return
        p = panels[int(active_panel["idx"])]
        cam = p["cam"]
        # Windows delta is 120 per notch.
        cam.distance = float(cam.distance) * (0.90 if delta > 0 else 1.10)
        _render_at(int(cur_idx.get()))

    def _on_key(ev: Any) -> None:
        ks = str(getattr(ev, "keysym", "")).strip().lower()
        ch = str(getattr(ev, "char", "")).strip().lower()

        if ch in {"1", "2", "3"}:
            active_panel["idx"] = int(ch) - 1
            _render_at(int(cur_idx.get()))
            return

        cam = panels[int(active_panel["idx"])]["cam"]

        if ch == "r":
            mujoco.mjv_defaultCamera(cam)
            _render_at(int(cur_idx.get()))
            return

        # Translate lookat with WASD/arrow keys (camera-relative).
        shift = bool(int(getattr(ev, "state", 0)) & 0x0001)
        step = float(cam.distance) * (0.10 if shift else 0.04)
        az = float(cam.azimuth) * (math.pi / 180.0)
        fwd = np.asarray([math.cos(az), math.sin(az), 0.0], dtype=np.float64)
        right = np.asarray([-math.sin(az), math.cos(az), 0.0], dtype=np.float64)

        moved = False
        if ks in {"w", "up"}:
            cam.lookat[:] = cam.lookat + step * fwd
            moved = True
        if ks in {"s", "down"}:
            cam.lookat[:] = cam.lookat - step * fwd
            moved = True
        if ks in {"d", "right"}:
            cam.lookat[:] = cam.lookat + step * right
            moved = True
        if ks in {"a", "left"}:
            cam.lookat[:] = cam.lookat - step * right
            moved = True
        if ks in {"q", "prior"}:  # PageUp
            cam.lookat[2] = float(cam.lookat[2]) + step
            moved = True
        if ks in {"e", "next"}:  # PageDown
            cam.lookat[2] = float(cam.lookat[2]) - step
            moved = True

        if moved:
            _render_at(int(cur_idx.get()))

    img_lbl.bind("<ButtonPress-1>", _on_mouse_down)
    img_lbl.bind("<ButtonPress-3>", _on_mouse_down)
    img_lbl.bind("<ButtonRelease-1>", _on_mouse_up)
    img_lbl.bind("<ButtonRelease-3>", _on_mouse_up)
    img_lbl.bind("<B1-Motion>", _on_mouse_move)
    img_lbl.bind("<B3-Motion>", _on_mouse_move)
    img_lbl.bind("<MouseWheel>", _on_wheel)
    root.bind("<Key>", _on_key)

    stopped = {"v": False}
    playing = {"v": False}
    after_id: dict[str, str | None] = {"id": None}
    scrub_guard = {"v": False}

    def _on_close() -> None:
        stopped["v"] = True
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol("WM_DELETE_WINDOW", _on_close)

    def _fmt_line(
        tag: str,
        *,
        i: int,
        fall_step: int,
        risk: float,
        risk_trace: np.ndarray,
        likely: bool,
        z_trace: np.ndarray,
    ) -> str:
        parts = [f"{tag}  t={float(i)*float(dt):.2f}s"]
        if fall_step >= 0:
            parts.append(f"BAL_LOSS@{float(fall_step)*float(dt):.2f}s")
            if i >= fall_step:
                parts.append("(unstable)")
        else:
            parts.append("no_balance_loss_in_recording")

        # Per-frame risk (if present).
        try:
            if risk_trace.size:
                parts.append(f"risk_now={float(risk_trace[min(i, int(risk_trace.size)-1)]):.2f}")
        except Exception:
            pass

        # Scalar risk summary.
        if math.isfinite(float(risk)):
            parts.append(f"risk={float(risk):.2f}")
            if likely:
                parts.append("LIKELY_FALL")
        try:
            if z_trace.size:
                parts.append(f"z={float(z_trace[min(i, int(z_trace.size)-1)]):.2f}m")
        except Exception:
            pass
        return "  ".join(parts)

    def _render_at(i: int) -> None:
        i = int(max(0, min(int(i), int(states_ref.shape[0]) - 1)))
        phys_ref.set_state(np.asarray(states_ref[i], dtype=np.float64))
        phys_good.set_state(np.asarray(states_good[i], dtype=np.float64))
        phys_bad.set_state(np.asarray(states_bad[i], dtype=np.float64))
        mujoco.mj_forward(phys_ref.model.ptr, phys_ref.data.ptr)
        mujoco.mj_forward(phys_good.model.ptr, phys_good.data.ptr)
        mujoco.mj_forward(phys_bad.model.ptr, phys_bad.data.ptr)
        _show_pil(_render_pil())

        ref_info.configure(
            text=_fmt_line(
                "REF",
                i=i,
                fall_step=fall_step_ref,
                risk=risk_ref,
                risk_trace=risk_trace_ref,
                likely=likely_ref,
                z_trace=root_z_ref,
            )
        )
        good_info.configure(
            text=_fmt_line(
                "GOOD",
                i=i,
                fall_step=fall_step_good,
                risk=risk_good,
                risk_trace=risk_trace_good,
                likely=likely_good,
                z_trace=root_z_good,
            )
        )
        bad_info.configure(
            text=_fmt_line(
                "BAD",
                i=i,
                fall_step=fall_step_bad,
                risk=risk_bad,
                risk_trace=risk_trace_bad,
                likely=likely_bad,
                z_trace=root_z_bad,
            )
        )

    def _seek(i: int) -> None:
        scrub_guard["v"] = True
        try:
            cur_idx.set(int(i))
        finally:
            scrub_guard["v"] = False
        _render_at(int(i))

    def _pause() -> None:
        playing["v"] = False
        if after_id["id"] is not None:
            try:
                root.after_cancel(after_id["id"])
            except Exception:
                pass
            after_id["id"] = None

    def _tick() -> None:
        if stopped["v"] or not playing["v"]:
            return
        nxt = int(cur_idx.get()) + 1
        if nxt >= int(states_ref.shape[0]):
            playing["v"] = False
            after_id["id"] = None
            return
        _seek(nxt)
        after_id["id"] = root.after(int(round(float(dt) * 1000.0)), _tick)

    def _play() -> None:
        if int(states_ref.shape[0]) < 1:
            return
        if int(cur_idx.get()) >= (int(states_ref.shape[0]) - 1):
            _seek(0)
        playing["v"] = True
        _tick()

    def _replay() -> None:
        _pause()
        _seek(0)
        _play()

    def _on_scrub(_val: str) -> None:
        if scrub_guard["v"]:
            return
        _pause()
        _render_at(int(cur_idx.get()))

    scale.configure(command=_on_scrub)

    tk.Button(btn_row, text="Replay", command=_replay, width=10).pack(side="left")
    tk.Button(btn_row, text="Play", command=_play, width=10).pack(side="left", padx=(6, 0))
    tk.Button(btn_row, text="Pause", command=_pause, width=10).pack(side="left", padx=(6, 0))
    tk.Button(btn_row, text="Close", command=_on_close, width=10).pack(side="right")

    # Show last frame initially (often most informative for stability).
    _seek(int(states_ref.shape[0]) - 1)
    root.mainloop()
