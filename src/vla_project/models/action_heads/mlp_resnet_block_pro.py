import math

import torch
import torch.nn as nn

from vla_project.models.action_heads.rope import RotaryEmbedding, apply_rope


class MLPResNetBlock_Pro(nn.Module):
    """Reference's ``use_proper_ffn=False`` legacy block (the configuration
    that hit 73% LIBERO at 10k steps in the reference's own ablation table).

    Three attention branches merged via concat-then-softmax:
      - self(x):         RoPE on q/k_self
      - adapter(h_a, p): RoPE on k_adapter
      - task(h_t):       RoPE on k_task, scaled by ratio_g = tanh(gating_factor)

    The post-attention path is a Sequential(LayerNorm, Linear(D→D), ReLU)
    applied to ``out + x`` (no separate post-FFN residual). Earlier we
    used the ``use_proper_ffn=True`` (pre-LN + 4× FFN + dual residual) variant
    on the assumption that it would fix a t-axis collapse we observed at
    35 stacked blocks. The reference's ablation shows proper_ffn HURT
    performance (73% → 50%); the t-axis collapse goes away when
    ``num_blocks`` is reduced to the reference default (24, vs Gemma4's
    35-layer all-tap). Use this legacy variant with ``num_blocks=24``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_wrist_bridge: bool = False,
        gating_init: float = 0.0,
        gating_init_wrist: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_wrist_bridge = use_wrist_bridge

        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_adapter = nn.Linear(dim, dim)
        self.v_adapter = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

        # gating_factor controls task cross-attn (h_t) ratio via tanh(g).
        # Default 0 init matches the reference; gating_init>0 bootstraps the
        # task stream so the action head sees scene-derived h_t from step 1
        # instead of waiting for tanh(0) ramp.
        # 2026-05-03 finding: with bs=8 (vs reference's effective bs=32 from
        # 2-GPU DDP), the gating ramp is ~4x slower. Diagnostic at v9
        # step_2500 showed gating still at ~0.001 after 2500 steps, with
        # the action head outputting constant predictions across obs/time.
        # Warm-starting the gating gives immediate cross-attn signal so
        # the head can learn to depend on h_t / h_w_bridge from the start.
        self.gating_factor = nn.Parameter(torch.tensor([float(gating_init)]))
        self.rope = RotaryEmbedding(dim=self.head_dim)

        # Wrist bridge cross-attn (4th attn branch, vla-gemma-4 #015 option B).
        # Only allocated when ``use_wrist_bridge=True`` so legacy ckpts (no
        # k_wrist / v_wrist params) can still load by setting the flag False.
        if use_wrist_bridge:
            self.k_wrist = nn.Linear(dim, dim)
            self.v_wrist = nn.Linear(dim, dim)
            self.gating_factor_wrist = nn.Parameter(torch.tensor([float(gating_init_wrist)]))

    def forward(
        self,
        x: torch.Tensor,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        p: torch.Tensor,
        h_w_l: torch.Tensor = None,  # noqa: RUF013 — keep Optional via None default
    ) -> torch.Tensor:
        ratio_g = torch.tanh(self.gating_factor)

        h_adapter = torch.cat([h_a, p], dim=1)
        h_task = h_t

        B, T, _ = x.shape
        K_a = h_adapter.shape[1]
        K_t = h_task.shape[1]
        use_wrist = (h_w_l is not None) and self.use_wrist_bridge
        K_w = h_w_l.shape[1] if use_wrist else 0

        def _heads(t: torch.Tensor, L: int) -> torch.Tensor:
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = _heads(self.q_proj(x), T)
        k_s = _heads(self.k_self(x), T)
        v_s = _heads(self.v_self(x), T)
        k_a = _heads(self.k_adapter(h_adapter), K_a)
        v_a = _heads(self.v_adapter(h_adapter), K_a)
        k_t = _heads(self.k_task(h_task), K_t)
        v_t = _heads(self.v_task(h_task), K_t)
        if use_wrist:
            k_w = _heads(self.k_wrist(h_w_l), K_w)
            v_w = _heads(self.v_wrist(h_w_l), K_w)

        cos, sin = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q, k_s = apply_rope(q, k_s, cos, sin)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_a = apply_rope(k_a, k_a, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_t = apply_rope(k_t, k_t, cos_t, sin_t)
        if use_wrist:
            cos_w, sin_w = self.rope(seq_len=K_w, device=x.device, dtype=x.dtype)
            _, k_w = apply_rope(k_w, k_w, cos_w, sin_w)

        scores_list = [
            torch.matmul(q, k_s.transpose(-2, -1)),
            torch.matmul(q, k_a.transpose(-2, -1)),
            torch.matmul(q, k_t.transpose(-2, -1)) * ratio_g,
        ]
        if use_wrist:
            ratio_g_wrist = torch.tanh(self.gating_factor_wrist)
            scores_list.append(torch.matmul(q, k_w.transpose(-2, -1)) * ratio_g_wrist)
        scores = torch.cat(scores_list, dim=-1) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)

        v_list = [v_s, v_a, v_t]
        if use_wrist:
            v_list.append(v_w)
        v = torch.cat(v_list, dim=2)
        attn_out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, self.dim)
        attn_out = self.o_proj(attn_out)

        return self.ffn(attn_out + x)
