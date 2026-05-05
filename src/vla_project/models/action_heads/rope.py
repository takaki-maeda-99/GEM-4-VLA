import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotate_half, matching the VLA-Adapter reference
    (action_heads.py:197-202). The legacy half-and-half implementation
    `cat([-x[..., D/2:], x[..., :D/2]], dim=-1)` is consistent with a
    cos/sin layout `[c0, c1, ..., c0, c1, ...]` (LLaMA-style); the reference
    uses an interleaved swap with the same `cat([freqs, freqs], dim=-1)`
    cos/sin. To exactly match the reference's positional encoding so that
    a fresh-init action head sees the same per-position rotation pattern
    as the published Bridge baseline (61% LIBERO success), we mirror the
    interleaved convention even though it is internally inconsistent with
    the cos/sin layout.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot
