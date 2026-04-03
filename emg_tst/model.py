from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TSTEncoderLayer(nn.Module):
    """Pre-norm transformer encoder block for time-series inputs."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.causal = bool(causal)
        self.attn = nn.MultiheadAttention(
            embed_dim=int(d_model),
            num_heads=int(n_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.drop1 = nn.Dropout(float(dropout))
        self.ln1 = nn.LayerNorm(int(d_model))

        self.ff = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), int(d_model)),
        )
        self.drop2 = nn.Dropout(float(dropout))
        self.ln2 = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_mask = None
        if self.causal:
            t = int(x.shape[1])
            attn_mask = torch.ones((t, t), device=x.device, dtype=torch.bool).triu(1)
        z = self.ln1(x)
        x = x + self.drop1(
            self.attn(
                z,
                z,
                z,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        )
        x = x + self.drop2(self.ff(self.ln2(x)))
        return x


class SensorFusionTransformer(nn.Module):
    """Last-step transformer for raw EMG snippets + thigh IMU.

    Input layout per timestep:
    - first `n_emg_vars`: EMG snippet block (typically 3 sensors x 32 lags = 96)
    - final `n_imu_vars`: thigh omega_xyz + thigh_quat_wxyz (typically 7)

    The model first compresses the intra-timestep EMG snippet locally, projects the
    thigh IMU branch separately, fuses them, then applies temporal self-attention
    across timesteps. The output is a single normalized knee-angle prediction for
    the final timestep of the context window.
    """

    def __init__(
        self,
        *,
        n_emg_vars: int,
        n_imu_vars: int,
        seq_len: int,
        d_model: int = 96,
        n_heads: int = 6,
        d_ff: int = 192,
        n_layers: int = 3,
        dropout: float = 0.2,
        causal: bool = False,
    ):
        super().__init__()
        self.n_emg_vars = int(max(0, n_emg_vars))
        self.n_imu_vars = int(max(0, n_imu_vars))
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)
        self.causal = bool(causal)

        self.n_emg_sensors = 3 if (self.n_emg_vars > 0 and self.n_emg_vars % 3 == 0) else 0
        self.emg_vars_per_sensor = (self.n_emg_vars // 3) if self.n_emg_sensors == 3 else 0
        self.use_emg_conv = bool(self.n_emg_sensors == 3 and self.emg_vars_per_sensor >= 8)

        half = int(self.d_model // 2)
        if self.n_emg_vars > 0:
            if self.use_emg_conv:
                self.emg_frontend = nn.Sequential(
                    nn.Conv1d(self.n_emg_sensors, 24, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(24, 32, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.emg_proj = nn.Linear(32, half)
            else:
                self.emg_frontend = None
                self.emg_proj = nn.Linear(self.n_emg_vars, half)
        else:
            self.emg_frontend = None
            self.emg_proj = None

        if self.n_imu_vars > 0:
            self.imu_proj = nn.Sequential(
                nn.Linear(self.n_imu_vars, half),
                nn.GELU(),
                nn.Linear(half, half),
            )
        else:
            self.imu_proj = None

        if self.n_emg_vars > 0 and self.n_imu_vars > 0:
            fuse_in = self.d_model
        elif self.n_emg_vars > 0 or self.n_imu_vars > 0:
            fuse_in = half
        else:
            raise ValueError("SensorFusionTransformer requires at least one modality.")

        self.fuse = nn.Sequential(
            nn.Linear(int(fuse_in), self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.drop = nn.Dropout(float(dropout))
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout=float(dropout),
                    causal=bool(causal),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.head = nn.Linear(self.d_model, 1)

    def _embed_emg(self, x_emg: torch.Tensor) -> torch.Tensor:
        if self.emg_proj is None:
            raise RuntimeError("EMG branch is disabled.")
        if self.use_emg_conv:
            b, t, _ = x_emg.shape
            x_bt = x_emg.reshape(b * t, self.n_emg_sensors, self.emg_vars_per_sensor)
            z = self.emg_frontend(x_bt).squeeze(-1).reshape(b, t, 32)
            return self.emg_proj(z)
        return self.emg_proj(x_emg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F] input, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != int(self.seq_len):
            raise ValueError(f"Expected seq_len={self.seq_len}, got T={int(x.shape[1])}")

        parts: list[torch.Tensor] = []
        if self.n_emg_vars > 0:
            parts.append(self._embed_emg(x[:, :, : self.n_emg_vars]))
        if self.n_imu_vars > 0:
            parts.append(self.imu_proj(x[:, :, self.n_emg_vars : self.n_emg_vars + self.n_imu_vars]))
        if not parts:
            raise RuntimeError("No active input branches.")

        z = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        z = self.fuse(z)
        z = self.drop(z + self.pos_emb)
        for layer in self.layers:
            z = layer(z)
        return self.head(z[:, -1, :])[:, 0]


@torch.no_grad()
def rolling_last_step_predict(
    model: SensorFusionTransformer,
    x: torch.Tensor | np.ndarray,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    """Run a last-step model causally across a whole sequence.

    For timestep `t`, the model sees the most recent `seq_len` samples ending at `t`.
    When `t < seq_len - 1`, the prefix is left-padded by repeating the first sample.
    Returns a tensor of shape `(T,)` on CPU.
    """

    if isinstance(x, np.ndarray):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    else:
        xt = x.detach().to(dtype=torch.float32)
    if xt.ndim != 2:
        raise ValueError(f"Expected [T,F] input, got shape={tuple(xt.shape)}")

    device = next(model.parameters()).device
    seq_len = int(model.seq_len)
    total_t = int(xt.shape[0])
    if total_t < 1:
        return torch.zeros((0,), dtype=torch.float32)

    xt = xt.to(device)
    pad = xt[:1].repeat(seq_len - 1, 1)
    padded = torch.cat([pad, xt], dim=0)
    preds: list[torch.Tensor] = []
    bs = int(max(1, batch_size))

    for start in range(0, total_t, bs):
        stop = min(start + bs, total_t)
        chunk = padded[start : stop + seq_len - 1]
        windows = chunk.unfold(0, seq_len, 1).permute(0, 2, 1).contiguous()
        pred = model(windows).detach().cpu()
        preds.append(pred)

    return torch.cat(preds, dim=0)
