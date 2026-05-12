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
        ungated_streams: bool = False,
        use_proper_residual: bool = False,
        proper_ffn_mode: str = "legacy",
        layer_scale_init: float = 0.0,
        mlp_ratio: float = 1.0,
        use_soft_prompt_cross_attn: bool = False,
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

        # FFN construction. Three modes:
        #   - "legacy" (default): ``LN -> Linear -> ReLU``. Matches VLA-Adapter
        #     upstream (action_heads.py:296-300). One-sided non-negative output,
        #     unsafe under residual accumulation.
        #   - "linear_only": ``LN -> Linear``. Restores signed residual updates
        #     by dropping the trailing ReLU. Minimal change to legacy.
        #   - "proper_mlp": ``LN -> Linear(dim, mid) -> GELU -> Linear(mid, dim)``.
        #     Standard Pre-LN MLP (activation sandwiched, final Linear can emit
        #     signed values). Matches X-VLA upstream Mlp at transformer.py:263.
        #     ``mlp_ratio`` controls mid dim (= dim * mlp_ratio, default 1.0
        #     keeps param count low; 4.0 matches X-VLA default but +8x ffn params).
        self.proper_ffn_mode = proper_ffn_mode
        if proper_ffn_mode == "legacy":
            self.ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
            )
        elif proper_ffn_mode == "linear_only":
            self.ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
            )
        elif proper_ffn_mode == "proper_mlp":
            mid = int(dim * mlp_ratio)
            self.ffn = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mid),
                nn.GELU(),
                nn.Linear(mid, dim),
            )
        else:
            raise ValueError(
                "proper_ffn_mode must be 'legacy' / 'linear_only' / 'proper_mlp'; "
                "got {!r}".format(proper_ffn_mode)
            )

        # LayerScale (CaiT/timm convention): per-channel learnable γ on the
        # residual branch, init small so the branch starts as ~identity. Only
        # allocated when ``layer_scale_init > 0``. Combined with proper_mlp this
        # gives the v44 "clean Pre-LN" path while keeping legacy ckpts loadable.
        self.layer_scale_init = float(layer_scale_init)
        if layer_scale_init > 0.0:
            self.layer_scale = nn.Parameter(
                torch.full((dim,), float(layer_scale_init))
            )
        else:
            self.layer_scale = None

        # gating_factor controls task cross-attn (h_t) ratio via tanh(g).
        # Default 0 init matches the reference; gating_init>0 bootstraps the
        # task stream so the action head sees scene-derived h_t from step 1
        # instead of waiting for tanh(0) ramp.
        # 2026-05-03 finding #1: with bs=8 (vs reference's effective bs=32
        # from 2-GPU DDP), the gating ramp is ~4x slower. v9 step_5000 had
        # max gating still ≈0.024.
        # 2026-05-03 finding #2: bf16 precision kills warm-init updates. At
        # gating=0.5 the ULP is 0.5 * 2^-7 ≈ 0.0039, larger than typical
        # AdamW updates (lr=2e-4 × grad). v10 step_2500 shows EXACTLY 0.5
        # for all blocks — no movement at all because each update rounds
        # to zero.
        # Workaround: ``ungated_streams=True`` removes both gating factors
        # and uses a fixed scale of 1.0 for task/wrist cross-attn streams.
        # The model can still learn to suppress unhelpful streams via the
        # k/v projection weights (which can shrink toward zero).
        self.ungated_streams = ungated_streams
        # v42 (2026-05-11): when True, replace the non-residual
        # ``ffn(attn_out + x)`` with proper Pre-LN style ``x + ffn(attn_out + x)``
        # so block contributions accumulate in a residual stream instead of
        # each block fully overwriting the previous one. v33/v37/v39/v41 all
        # showed concentrated "only deepest block trains" patterns rooted in
        # the legacy non-residual structure (matches VLA-Adapter upstream
        # /VLA-Adapter/prismatic/models/action_heads.py:409). X-VLA upstream
        # /X-VLA/models/transformer.py:279-280 uses proper residual.
        self.use_proper_residual = use_proper_residual
        if not ungated_streams:
            self.gating_factor = nn.Parameter(torch.tensor([float(gating_init)]))
        self.rope = RotaryEmbedding(dim=self.head_dim)

        # Wrist bridge cross-attn (4th attn branch, vla-gemma-4 #015 option B).
        # Only allocated when ``use_wrist_bridge=True`` so legacy ckpts (no
        # k_wrist / v_wrist params) can still load by setting the flag False.
        if use_wrist_bridge:
            self.k_wrist = nn.Linear(dim, dim)
            self.v_wrist = nn.Linear(dim, dim)
            if not ungated_streams:
                self.gating_factor_wrist = nn.Parameter(torch.tensor([float(gating_init_wrist)]))

        # arch v3: independent soft_prompt cross-attn stream (AQ pattern).
        # soft_prompt is scattered into the LLM input embedding at the
        # soft_prompt slot, Gemma processes it through self-attention, then we
        # slice the per-layer hidden at the soft_prompt position and feed it to
        # this block's k_soft_prompt/v_soft_prompt. Mirrors how action_queries
        # flow into h_a -> adapter cross-attn. Only allocated when
        # ``use_soft_prompt_cross_attn=True`` so legacy ckpts (no
        # k_soft_prompt/v_soft_prompt params) stay loadable with this flag off.
        self.use_soft_prompt_cross_attn = use_soft_prompt_cross_attn
        if use_soft_prompt_cross_attn:
            self.k_soft_prompt = nn.Linear(dim, dim)
            self.v_soft_prompt = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        h_a: torch.Tensor,
        h_t: torch.Tensor,
        p: torch.Tensor = None,  # noqa: RUF013 — None when proprio_in_llm=True
        h_w_l: torch.Tensor = None,  # noqa: RUF013 — keep Optional via None default
        h_sp_l: torch.Tensor = None,  # noqa: RUF013 — arch v3 soft_prompt per-layer hidden
        h_t_mask: torch.Tensor = None,  # noqa: RUF013 — (B, K_t) bool mask for prompt pad
    ) -> torch.Tensor:
        if self.ungated_streams:
            ratio_g = 1.0
        else:
            ratio_g = torch.tanh(self.gating_factor)

        # ``p`` is the proprio token (B, 1, D). When ``proprio_in_llm=True`` at
        # the policy level, proprio is scattered into LLM input embeddings
        # instead and ``p`` is None — the adapter bank is then just ``h_a``
        # (LLM hidden states at action positions, which now also encode
        # proprio). This forces the LLM to learn proprio-aware representations
        # for cross-embodiment training. v33-style configs keep p concatenated.
        h_adapter = h_a if p is None else torch.cat([h_a, p], dim=1)
        h_task = h_t

        B, T, _ = x.shape
        K_a = h_adapter.shape[1]
        K_t = h_task.shape[1]
        use_wrist = (h_w_l is not None) and self.use_wrist_bridge
        K_w = h_w_l.shape[1] if use_wrist else 0
        use_soft_prompt_ca = (h_sp_l is not None) and self.use_soft_prompt_cross_attn
        K_sp = h_sp_l.shape[1] if use_soft_prompt_ca else 0

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
        if use_soft_prompt_ca:
            k_sp = _heads(self.k_soft_prompt(h_sp_l), K_sp)
            v_sp = _heads(self.v_soft_prompt(h_sp_l), K_sp)

        cos, sin = self.rope(seq_len=T, device=x.device, dtype=x.dtype)
        q, k_s = apply_rope(q, k_s, cos, sin)
        cos_a, sin_a = self.rope(seq_len=K_a, device=x.device, dtype=x.dtype)
        _, k_a = apply_rope(k_a, k_a, cos_a, sin_a)
        cos_t, sin_t = self.rope(seq_len=K_t, device=x.device, dtype=x.dtype)
        _, k_t = apply_rope(k_t, k_t, cos_t, sin_t)
        if use_wrist:
            cos_w, sin_w = self.rope(seq_len=K_w, device=x.device, dtype=x.dtype)
            _, k_w = apply_rope(k_w, k_w, cos_w, sin_w)
        if use_soft_prompt_ca:
            cos_sp, sin_sp = self.rope(seq_len=K_sp, device=x.device, dtype=x.dtype)
            _, k_sp = apply_rope(k_sp, k_sp, cos_sp, sin_sp)

        scores_list = [
            torch.matmul(q, k_s.transpose(-2, -1)),
            torch.matmul(q, k_a.transpose(-2, -1)),
            torch.matmul(q, k_t.transpose(-2, -1)) * ratio_g,
        ]
        if use_wrist:
            ratio_g_wrist = 1.0 if self.ungated_streams else torch.tanh(self.gating_factor_wrist)
            scores_list.append(torch.matmul(q, k_w.transpose(-2, -1)) * ratio_g_wrist)
        if use_soft_prompt_ca:
            # arch v3: soft_prompt cross-attn is ungated (AQ pattern).
            scores_list.append(torch.matmul(q, k_sp.transpose(-2, -1)))
        scores = torch.cat(scores_list, dim=-1) / math.sqrt(self.head_dim)

        # arch v3: mask pad positions in the task slice. Codex round 3 warned
        # gating is applied to the task term BEFORE this concat (ratio_g
        # multiplied above), so masked_fill here with -inf is safe — the slice
        # at [T+K_a : T+K_a+K_t] already includes the gating factor.
        if h_t_mask is not None:
            K_total = scores.shape[-1]
            full_mask = scores.new_ones(B, 1, 1, K_total, dtype=torch.bool)
            full_mask[:, :, :, T + K_a : T + K_a + K_t] = h_t_mask[:, None, None, :]
            scores = scores.masked_fill(~full_mask, -1e4)

        weights = torch.softmax(scores, dim=-1)

        v_list = [v_s, v_a, v_t]
        if use_wrist:
            v_list.append(v_w)
        if use_soft_prompt_ca:
            v_list.append(v_sp)
        v = torch.cat(v_list, dim=2)
        attn_out = torch.matmul(weights, v).transpose(1, 2).reshape(B, T, self.dim)
        attn_out = self.o_proj(attn_out)

        # v42/v44 proper-residual path: x carries through additively, so block
        # contributions accumulate. Legacy path (default) replaces x with
        # ``ffn(attn_out + x)`` — the non-residual collapse that forced
        # only the deepest block to learn in v33/v37/v41.
        # v44 (2026-05-11): also apply LayerScale γ (per-channel, init small)
        # so the residual branch starts ~identity and v42's positive-drift
        # instability (grad_max 221k) is dampened. Codex round 8 verdict:
        # proper_mlp + LayerScale = clean Pre-LN; legacy LN→Linear→ReLU
        # accumulation was a one-sided drift bug.
        if self.use_proper_residual:
            ffn_out = self.ffn(attn_out + x)
            if self.layer_scale is not None:
                ffn_out = ffn_out * self.layer_scale
            return x + ffn_out
        return self.ffn(attn_out + x)
