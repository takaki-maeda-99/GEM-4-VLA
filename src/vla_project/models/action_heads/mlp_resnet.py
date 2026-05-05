from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from vla_project.models.action_heads.mlp_resnet_block_pro import MLPResNetBlock_Pro


class MLPResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        action_dim: int,
        use_grad_checkpoint: bool = False,
        use_wrist_bridge: bool = False,
        gating_init: float = 0.0,
        gating_init_wrist: float = 0.0,
        ungated_streams: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.use_grad_checkpoint = use_grad_checkpoint
        self.use_wrist_bridge = use_wrist_bridge
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList(
            [
                MLPResNetBlock_Pro(
                    dim=hidden_dim,
                    use_wrist_bridge=use_wrist_bridge,
                    gating_init=gating_init,
                    gating_init_wrist=gating_init_wrist,
                    ungated_streams=ungated_streams,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def _run_block(self, blk: nn.Module, x: torch.Tensor,
                   h_a_i: torch.Tensor, h_t_i: torch.Tensor, p: torch.Tensor,
                   h_w_l: Optional[torch.Tensor] = None) -> torch.Tensor:
        return blk(x, h_a=h_a_i, h_t=h_t_i, p=p, h_w_l=h_w_l)

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, input_dim]
        h_a: torch.Tensor,                     # [B, num_layers+1, K_a, D]
        h_t: torch.Tensor,                     # [B, num_layers+1, K_t, D]
        p: torch.Tensor,                       # [B, 1, D]
        h_w: Optional[torch.Tensor] = None,    # [B, K_w, D]  final-layer wrist (Bridge self-attn pool)
        h_sp: Optional[torch.Tensor] = None,   # [B, K_sp, D] final-layer soft prompt (Bridge self-attn pool)
        h_w_bridge: Optional[torch.Tensor] = None,  # [B, num_blocks+1, K_w_b, D] per-layer wrist (Bridge cross-attn)
    ) -> torch.Tensor:
        """Bridge form (``action_heads.py:133-176`` ``use_pro_version=True``,
        ``use_proper_ffn=True``, ``use_xvla_style=False``):

          x = LN1(x); fc1; ReLU
          x = cat(x, h_w, h_sp)                  # self-attn pool extended
          for block: x = block(x, h_a[i+1], h_t[i+1], p, h_w_l=h_w_bridge[:, i+1])
          x = x[:, :action_len, :]               # trim back to action positions
          x = fc2(LN2(x))

        When ``h_w_bridge`` is supplied, the per-layer wrist features feed
        the block's 4th cross-attn branch (``k_wrist`` / ``v_wrist``).
        Mirrors vla-gemma-4 wristb_b16_v2 (73% LIBERO baseline). vla-gemma-4
        also drops ``h_w`` from the self-attn pool concat when wrist_bridge
        is active to remove redundancy
        (``action_heads.py:166-169``); we follow the same convention.
        """
        x = self.relu(self.fc1(self.layer_norm1(x)))
        action_len = x.shape[1]
        wrist_bridge_active = h_w_bridge is not None and self.use_wrist_bridge
        if h_w is not None and not wrist_bridge_active:
            x = torch.cat([x, h_w], dim=1)
        if h_sp is not None:
            x = torch.cat([x, h_sp], dim=1)
        for i, blk in enumerate(self.blocks):
            h_a_i = h_a[:, i + 1]
            h_t_i = h_t[:, i + 1]
            h_w_l = h_w_bridge[:, i + 1] if wrist_bridge_active else None
            if self.use_grad_checkpoint and self.training:
                x = ckpt.checkpoint(
                    self._run_block, blk, x, h_a_i, h_t_i, p, h_w_l, use_reentrant=False
                )
            else:
                x = self._run_block(blk, x, h_a_i, h_t_i, p, h_w_l)
        x = x[:, :action_len, :]
        x = self.fc2(self.layer_norm2(x))
        return x
