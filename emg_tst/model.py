from __future__ import annotations
import torch
from torch import nn

class TSTEncoderLayerBN(nn.Module):
    """
    Transformer encoder layer with BatchNorm (time-series variant).
    x shape: [B,T,D]
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)

        ff_out = self.ff(x)
        x = x + self.drop2(ff_out)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        return x

class TSTEncoder(nn.Module):
    """
    Encoder-only Time Series Transformer.
    Input x: [B,T,M] where M = number of variables/features.
    """
    def __init__(
        self,
        n_vars: int,
        seq_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.seq_len = seq_len
        self.d_model = d_model

        self.input_proj = nn.Linear(n_vars, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TSTEncoderLayerBN(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.shape[1]}")
        z = self.input_proj(x) + self.pos_emb
        z = self.drop(z)
        for layer in self.layers:
            z = layer(z, key_padding_mask=key_padding_mask)
        return z  # [B,T,D]

class TSTPretrainDenoiser(nn.Module):
    """Pretrain head: reconstruct masked inputs."""
    def __init__(self, encoder: TSTEncoder):
        super().__init__()
        self.encoder = encoder
        self.recon_head = nn.Linear(encoder.d_model, encoder.n_vars)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.recon_head(z)  # [B,T,M]

class TSTRegressor(nn.Module):
    """Fine-tune head: per-timestep regression."""
    def __init__(self, encoder: TSTEncoder, out_dim: int = 1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)  # [B,T,out_dim]
