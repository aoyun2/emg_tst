from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _normalized_model_type(model_type: str) -> str:
    key = str(model_type).strip().lower()
    aliases = {
        "tcn": "sensor_fusion_last_step_tcn",
        "sensor_fusion_last_step_tcn": "sensor_fusion_last_step_tcn",
        "direct_tcn": "direct_last_step_tcn",
        "direct_last_step_tcn": "direct_last_step_tcn",
        "residual_tcn": "residual_fusion_tcn",
        "residual_fusion_tcn": "residual_fusion_tcn",
        "lstm": "sensor_fusion_last_step_lstm",
        "sensor_fusion_last_step_lstm": "sensor_fusion_last_step_lstm",
        "cnn_bilstm": "cnn_bilstm_last_step",
        "cnn-bilstm": "cnn_bilstm_last_step",
        "cnn_bilstm_last_step": "cnn_bilstm_last_step",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported model_type={model_type!r}")
    return aliases[key]


class SensorFusionStem(nn.Module):
    """Encode per-timestep raw EMG snippets and thigh IMU into one latent vector."""

    def __init__(
        self,
        *,
        n_emg_vars: int,
        n_imu_vars: int,
        stem_width: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_emg_vars = int(max(0, n_emg_vars))
        self.n_imu_vars = int(max(0, n_imu_vars))
        self.stem_width = int(stem_width)
        self.dropout_p = float(dropout)

        self.n_emg_sensors = 3 if (self.n_emg_vars > 0 and self.n_emg_vars % 3 == 0) else 0
        self.emg_vars_per_sensor = (self.n_emg_vars // 3) if self.n_emg_sensors == 3 else 0
        self.use_emg_conv = bool(self.n_emg_sensors == 3 and self.emg_vars_per_sensor >= 8)

        if self.n_emg_vars > 0 and self.n_imu_vars > 0:
            emg_width = int(max(16, round(self.stem_width * 0.625)))
            imu_width = int(self.stem_width - emg_width)
        else:
            emg_width = int(self.stem_width if self.n_emg_vars > 0 else 0)
            imu_width = int(self.stem_width if self.n_imu_vars > 0 else 0)
        self.emg_width = int(emg_width)
        self.imu_width = int(imu_width)

        if self.n_emg_vars > 0:
            if self.use_emg_conv:
                self.emg_frontend = nn.Sequential(
                    nn.Conv1d(self.n_emg_sensors, 16, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.Conv1d(16, 24, kernel_size=5, padding=2),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.emg_proj = nn.Linear(24, self.emg_width)
            else:
                self.emg_frontend = None
                self.emg_proj = nn.Sequential(
                    nn.Linear(self.n_emg_vars, self.emg_width),
                    nn.GELU(),
                )
        else:
            self.emg_frontend = None
            self.emg_proj = None

        if self.n_imu_vars > 0:
            self.imu_proj = nn.Sequential(
                nn.Linear(self.n_imu_vars, self.imu_width),
                nn.GELU(),
                nn.Linear(self.imu_width, self.imu_width),
            )
        else:
            self.imu_proj = None

        fuse_in = int(self.emg_width + self.imu_width)
        if fuse_in < 1:
            raise ValueError("SensorFusionStem requires at least one active modality.")
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, self.stem_width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def _embed_emg(self, x_emg: torch.Tensor) -> torch.Tensor:
        if self.emg_proj is None:
            raise RuntimeError("EMG branch is disabled.")
        if self.use_emg_conv:
            b, t, _ = x_emg.shape
            x_bt = x_emg.reshape(b * t, self.n_emg_sensors, self.emg_vars_per_sensor)
            z = self.emg_frontend(x_bt).squeeze(-1).reshape(b, t, 24)
            return self.emg_proj(z)
        return self.emg_proj(x_emg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F] input, got shape={tuple(x.shape)}")

        parts: list[torch.Tensor] = []
        if self.n_emg_vars > 0:
            parts.append(self._embed_emg(x[:, :, : self.n_emg_vars]))
        if self.n_imu_vars > 0:
            imu = x[:, :, self.n_emg_vars : self.n_emg_vars + self.n_imu_vars]
            parts.append(self.imu_proj(imu))
        if not parts:
            raise RuntimeError("No active input branches.")

        z = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return self.fuse(z)


class ResidualTemporalBlock(nn.Module):
    """Simple residual TCN block with same-length dilated convolutions."""

    def __init__(self, *, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = int(((int(kernel_size) - 1) * int(dilation)) // 2)
        self.norm1 = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad, dilation=int(dilation))
        self.norm2 = nn.GroupNorm(1, int(channels))
        self.conv2 = nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad, dilation=int(dilation))
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv1(self.norm1(x))
        z = self.drop(self.act(z))
        z = self.conv2(self.norm2(z))
        z = self.drop(self.act(z))
        return x + z


class SensorFusionTCN(nn.Module):
    """Last-step TCN for raw EMG snippets plus thigh IMU."""

    def __init__(
        self,
        *,
        n_emg_vars: int,
        n_imu_vars: int,
        seq_len: int,
        stem_width: int = 64,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.model_type = "sensor_fusion_last_step_tcn"
        self.n_emg_vars = int(max(0, n_emg_vars))
        self.n_imu_vars = int(max(0, n_imu_vars))
        self.seq_len = int(seq_len)
        self.stem_width = int(stem_width)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.dropout = float(dropout)

        self.stem = SensorFusionStem(
            n_emg_vars=int(self.n_emg_vars),
            n_imu_vars=int(self.n_imu_vars),
            stem_width=int(self.stem_width),
            dropout=float(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    channels=int(self.stem_width),
                    kernel_size=int(self.kernel_size),
                    dilation=int(d),
                    dropout=float(dropout),
                )
                for d in self.dilations
            ]
        )
        self.head = nn.Linear(int(self.stem_width), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F] input, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != int(self.seq_len):
            raise ValueError(f"Expected seq_len={int(self.seq_len)}, got T={int(x.shape[1])}")
        z = self.stem(x).transpose(1, 2).contiguous()
        for block in self.blocks:
            z = block(z)
        return self.head(z[:, :, -1])[:, 0]


class SensorFusionLSTM(nn.Module):
    """Plain LSTM baseline using the same per-timestep sensor-fusion stem."""

    def __init__(
        self,
        *,
        n_emg_vars: int,
        n_imu_vars: int,
        seq_len: int,
        stem_width: int = 64,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.model_type = "sensor_fusion_last_step_lstm"
        self.n_emg_vars = int(max(0, n_emg_vars))
        self.n_imu_vars = int(max(0, n_imu_vars))
        self.seq_len = int(seq_len)
        self.stem_width = int(stem_width)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)

        self.stem = SensorFusionStem(
            n_emg_vars=int(self.n_emg_vars),
            n_imu_vars=int(self.n_imu_vars),
            stem_width=int(self.stem_width),
            dropout=float(dropout),
        )
        self.rnn = nn.LSTM(
            input_size=int(self.stem_width),
            hidden_size=int(self.hidden_size),
            num_layers=int(self.n_layers),
            batch_first=True,
            dropout=(float(dropout) if int(self.n_layers) > 1 else 0.0),
        )
        self.head = nn.Linear(int(self.hidden_size), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F] input, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != int(self.seq_len):
            raise ValueError(f"Expected seq_len={int(self.seq_len)}, got T={int(x.shape[1])}")
        z = self.stem(x)
        z, _ = self.rnn(z)
        return self.head(z[:, -1, :])[:, 0]


class CnnBiLstmLastStep(nn.Module):
    """Direct temporal CNN frontend followed by a bidirectional LSTM head."""

    def __init__(
        self,
        *,
        n_emg_vars: int,
        n_imu_vars: int,
        seq_len: int,
        stem_width: int = 32,
        hidden_size: int = 64,
        n_layers: int = 2,
        kernel_size: int = 5,
        depth: int = 2,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.model_type = "cnn_bilstm_last_step"
        self.n_emg_vars = int(max(0, n_emg_vars))
        self.n_imu_vars = int(max(0, n_imu_vars))
        self.n_vars = int(self.n_emg_vars + self.n_imu_vars)
        self.seq_len = int(seq_len)
        self.stem_width = int(stem_width)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.kernel_size = int(kernel_size)
        self.depth = int(depth)
        self.dropout = float(dropout)

        pad = int((int(kernel_size) - 1) // 2)
        convs: list[nn.Module] = [
            nn.Conv1d(int(self.n_vars), int(self.stem_width), int(self.kernel_size), padding=pad),
            nn.GELU(),
        ]
        for _ in range(max(0, int(self.depth) - 1)):
            convs.extend(
                [
                    nn.Conv1d(int(self.stem_width), int(self.stem_width), int(self.kernel_size), padding=pad),
                    nn.GELU(),
                    nn.Dropout(float(self.dropout)),
                ]
            )
        self.conv = nn.Sequential(*convs)
        self.rnn = nn.LSTM(
            input_size=int(self.stem_width),
            hidden_size=int(self.hidden_size),
            num_layers=int(self.n_layers),
            batch_first=True,
            bidirectional=True,
            dropout=(float(self.dropout) if int(self.n_layers) > 1 else 0.0),
        )
        self.head = nn.Sequential(
            nn.Linear(int(2 * self.hidden_size), int(self.hidden_size)),
            nn.GELU(),
            nn.Dropout(float(self.dropout)),
            nn.Linear(int(self.hidden_size), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F] input, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != int(self.seq_len):
            raise ValueError(f"Expected seq_len={int(self.seq_len)}, got T={int(x.shape[1])}")
        if int(x.shape[2]) != int(self.n_vars):
            raise ValueError(f"Expected n_vars={int(self.n_vars)}, got F={int(x.shape[2])}")
        z = self.conv(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        z, _ = self.rnn(z)
        return self.head(z[:, -1, :])[:, 0]


class DirectResidualTemporalBlock(nn.Module):
    """Plain residual TCN block without modality-specific normalization or stems."""

    def __init__(self, *, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = int((int(kernel_size) - 1) // 2)
        self.conv1 = nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad)
        self.conv2 = nn.Conv1d(int(channels), int(channels), int(kernel_size), padding=pad)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.drop(self.act(self.conv1(x)))
        z = self.conv2(z)
        return x + z


class DirectLastStepTCN(nn.Module):
    """Plain TCN over the full input feature vector at each timestep."""

    def __init__(
        self,
        *,
        n_emg_vars: int,
        n_imu_vars: int,
        seq_len: int,
        stem_width: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.10,
        depth: int = 4,
    ):
        super().__init__()
        self.model_type = "direct_last_step_tcn"
        self.n_emg_vars = int(max(0, n_emg_vars))
        self.n_imu_vars = int(max(0, n_imu_vars))
        self.n_vars = int(self.n_emg_vars + self.n_imu_vars)
        self.seq_len = int(seq_len)
        self.stem_width = int(stem_width)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.depth = int(depth)

        self.inp = nn.Conv1d(int(self.n_vars), int(self.stem_width), int(self.kernel_size), padding=int((int(self.kernel_size) - 1) // 2))
        self.blocks = nn.ModuleList(
            [
                DirectResidualTemporalBlock(
                    channels=int(self.stem_width),
                    kernel_size=int(self.kernel_size),
                    dropout=float(self.dropout),
                )
                for _ in range(int(self.depth))
            ]
        )
        self.head = nn.Linear(int(self.stem_width), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F] input, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != int(self.seq_len):
            raise ValueError(f"Expected seq_len={int(self.seq_len)}, got T={int(x.shape[1])}")
        if int(x.shape[2]) != int(self.n_vars):
            raise ValueError(f"Expected n_vars={int(self.n_vars)}, got F={int(x.shape[2])}")
        z = self.inp(x.transpose(1, 2).contiguous())
        for block in self.blocks:
            z = block(z)
        return self.head(z[:, :, -1])[:, 0]


class _BranchTCN(nn.Module):
    """Small direct TCN branch that returns a scalar last-step output."""

    def __init__(self, *, n_vars: int, width: int, kernel_size: int, dropout: float, depth: int):
        super().__init__()
        self.n_vars = int(n_vars)
        self.width = int(width)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.depth = int(depth)
        pad = int((int(kernel_size) - 1) // 2)
        self.inp = nn.Conv1d(int(n_vars), int(width), int(kernel_size), padding=pad)
        self.blocks = nn.ModuleList(
            [
                DirectResidualTemporalBlock(
                    channels=int(width),
                    kernel_size=int(kernel_size),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.head = nn.Linear(int(width), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inp(x.transpose(1, 2).contiguous())
        for block in self.blocks:
            z = block(z)
        return self.head(z[:, :, -1])[:, 0]


class ResidualFusionTCN(nn.Module):
    """Predict a thigh-based baseline plus an EMG-driven residual correction."""

    def __init__(
        self,
        *,
        n_emg_vars: int,
        n_imu_vars: int,
        seq_len: int,
        stem_width: int = 24,
        kernel_size: int = 5,
        dropout: float = 0.10,
        depth: int = 3,
    ):
        super().__init__()
        self.model_type = "residual_fusion_tcn"
        self.n_emg_vars = int(max(0, n_emg_vars))
        self.n_imu_vars = int(max(0, n_imu_vars))
        self.n_vars = int(self.n_emg_vars + self.n_imu_vars)
        self.seq_len = int(seq_len)
        self.stem_width = int(stem_width)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)
        self.depth = int(depth)

        if self.n_emg_vars < 1 or self.n_imu_vars < 1:
            raise ValueError("ResidualFusionTCN requires both EMG and IMU inputs.")

        self.thigh_branch = _BranchTCN(
            n_vars=int(self.n_imu_vars),
            width=int(self.stem_width),
            kernel_size=int(self.kernel_size),
            dropout=float(self.dropout),
            depth=int(self.depth),
        )
        self.emg_branch = _BranchTCN(
            n_vars=int(self.n_emg_vars),
            width=int(self.stem_width),
            kernel_size=int(self.kernel_size),
            dropout=float(self.dropout),
            depth=int(self.depth),
        )
        self.gate = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,T,F] input, got shape={tuple(x.shape)}")
        if int(x.shape[1]) != int(self.seq_len):
            raise ValueError(f"Expected seq_len={int(self.seq_len)}, got T={int(x.shape[1])}")
        if int(x.shape[2]) != int(self.n_vars):
            raise ValueError(f"Expected n_vars={int(self.n_vars)}, got F={int(x.shape[2])}")

        x_emg = x[:, :, : self.n_emg_vars]
        x_imu = x[:, :, self.n_emg_vars : self.n_emg_vars + self.n_imu_vars]
        base = self.thigh_branch(x_imu)
        corr = self.emg_branch(x_emg)
        gate = self.gate(torch.stack([base.detach(), corr], dim=1)).squeeze(1)
        return base + self.res_scale * gate * corr


def build_last_step_model(**cfg) -> nn.Module:
    model_type = _normalized_model_type(str(cfg.get("model_type", "tcn")))
    common = {
        "n_emg_vars": int(cfg["n_emg_vars"]),
        "n_imu_vars": int(cfg["n_imu_vars"]),
        "seq_len": int(cfg["seq_len"]),
        "stem_width": int(cfg.get("stem_width", 64)),
        "dropout": float(cfg.get("dropout", 0.15)),
    }
    if model_type == "sensor_fusion_last_step_tcn":
        dilations = cfg.get("dilations", (1, 2, 4, 8))
        if isinstance(dilations, np.ndarray):
            dilations = tuple(int(x) for x in dilations.reshape(-1).tolist())
        return SensorFusionTCN(
            **common,
            kernel_size=int(cfg.get("kernel_size", 5)),
            dilations=tuple(int(x) for x in dilations),
        )
    if model_type == "direct_last_step_tcn":
        return DirectLastStepTCN(
            **common,
            kernel_size=int(cfg.get("kernel_size", 5)),
            depth=int(cfg.get("depth", 4)),
        )
    if model_type == "residual_fusion_tcn":
        return ResidualFusionTCN(
            **common,
            kernel_size=int(cfg.get("kernel_size", 5)),
            depth=int(cfg.get("depth", 3)),
        )
    if model_type == "sensor_fusion_last_step_lstm":
        return SensorFusionLSTM(
            **common,
            hidden_size=int(cfg.get("hidden_size", common["stem_width"])),
            n_layers=int(cfg.get("n_layers", 2)),
        )
    if model_type == "cnn_bilstm_last_step":
        return CnnBiLstmLastStep(
            **common,
            hidden_size=int(cfg.get("hidden_size", common["stem_width"] * 2)),
            n_layers=int(cfg.get("n_layers", 2)),
            kernel_size=int(cfg.get("kernel_size", 5)),
            depth=int(cfg.get("depth", 2)),
        )
    raise ValueError(f"Unsupported model_type={model_type!r}")


@torch.no_grad()
def rolling_last_step_predict(
    model: nn.Module,
    x: torch.Tensor | np.ndarray,
    *,
    batch_size: int = 512,
) -> torch.Tensor:
    """Run any last-step model causally across a full sequence."""

    if isinstance(x, np.ndarray):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    else:
        xt = x.detach().to(dtype=torch.float32)
    if xt.ndim != 2:
        raise ValueError(f"Expected [T,F] input, got shape={tuple(xt.shape)}")

    seq_len = int(getattr(model, "seq_len", 0))
    if seq_len < 1:
        raise ValueError("rolling_last_step_predict requires model.seq_len >= 1")

    device = next(model.parameters()).device
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
        preds.append(model(windows).detach().cpu())

    return torch.cat(preds, dim=0)
