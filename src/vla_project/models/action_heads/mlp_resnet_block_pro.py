import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vla_project.models.action_heads.rope import RotaryEmbedding, apply_rope


class MLPResNetBlock_Pro(nn.Module):
    """Pre-LN transformer block with 4× FFN and dual residual connections.

    Three attention branches merged via concat-then-softmax:

      - self(x):         RoPE on q/k_self
      - adapter(h_a, p): RoPE on k_adapter
      - task(h_t):       RoPE on k_task, scaled by ratio_g = tanh(gating_factor)

    Mirrors the reference's ``use_proper_ffn=True`` mode
    (``vla-gemma-4/VLA-Adapter/prismatic/models/action_heads.py:349``). The
    legacy ``use_proper_ffn=False`` path of that reference — a Sequential
    ``LayerNorm + Linear(D→D) + ReLU`` applied to ``out + x`` with no
    post-FFN residual — was verified at runtime to collapse t-axis
    variation through a 35-block stack at inference, producing identical
    action-chunk anchors regardless of input. We do not carry that mode.
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.ffn_up = nn.Linear(dim, dim * mlp_ratio)
        self.ffn_down = nn.Linear(dim * mlp_ratio, dim)

        self.gating_factor = nn.Parameter(torch.zeros(1))
        self.rope = RotaryEmbedding(dim=self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        ratio_g = torch.tanh(self.gating_factor)

        h_adapter = torch.cat([h_a, p], dim=1)
        h_task = h_t

        B, T, _ = x.shape
        K_a = h_adapter.shape[1]
        K_t = h_task.shape[1]

        x_attn_in = self.norm1(x)

        def _heads(t: torch.Tensor, L: int) -> torch.Tensor:
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = _heads(self.q_proj(x_attn_in), T)
        k_s = _heads(self.k_self(x_attn_in), T)
        v_s = _heads(self.v_self(x_attn_in), T)
        k_a = _heads(self.k_adapter(h_adapter), K_a)
        v_a = _heads(self.v_adapter(h_adapter), K_a)
        k_t = _heads(self.k_task(h_task), K_t)
        v_t = _heads(self.v_task(h_task), K_t)

        cos, sin = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q, k_s = apply_rope(q, k_s, cos, sin)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_a = apply_rope(k_a, k_a, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_t = apply_rope(k_t, k_t, cos_t, sin_t)

        scores = torch.cat(
            [
                torch.matmul(q, k_s.transpose(-2, -1)),
                torch.matmul(q, k_a.transpose(-2, -1)),
                torch.matmul(q, k_t.transpose(-2, -1)) * ratio_g,
            ],
            dim=-1,
        ) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)

        v = torch.cat([v_s, v_a, v_t], dim=2)
        attn_out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, self.dim)
        attn_out = self.o_proj(attn_out)

        x = x + attn_out
        x = x + self.ffn_down(F.gelu(self.ffn_up(self.norm2(x))))
        return x
