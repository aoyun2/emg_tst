from __future__ import annotations
import torch

@torch.no_grad()
def stateful_variable_mask(
    batch_size: int,
    seq_len: int,
    n_vars: int,
    *,
    r: float = 0.15,     # masked ratio
    lm: int = 3,         # mean masked segment length
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Stateful per-variable masking (2-state Markov chain).
    Returns mask in {0,1}, shape [B,T,M], where 1=unmasked and 0=masked.

    masked->unmasked prob p_m = 1/lm
    unmasked->masked prob p_u = p_m * r/(1-r)

    This creates masked "runs" instead of iid Bernoulli points.
    """
    if not (0.0 < r < 1.0):
        raise ValueError(f"r must be in (0,1); got {r}")
    if lm <= 0:
        raise ValueError(f"lm must be positive; got {lm}")

    dev = torch.device(device)
    p_m = 1.0 / float(lm)
    p_u = p_m * (r / (1.0 - r))

    # state: True=unmasked, False=masked
    state = torch.rand((batch_size, n_vars), device=dev) > r
    mask = torch.empty((batch_size, seq_len, n_vars), device=dev, dtype=torch.bool)

    for t in range(seq_len):
        mask[:, t, :] = state
        u = torch.rand((batch_size, n_vars), device=dev)
        to_masked = state & (u < p_u)
        to_unmasked = (~state) & (u < p_m)
        if to_masked.any() or to_unmasked.any():
            state = state.clone()
            state[to_masked] = False
            state[to_unmasked] = True

    return mask.to(dtype=torch.float32)
