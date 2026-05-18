import torch
import torch.nn as nn

from vla_project.models.action_heads.mlp_resnet import MLPResNet


class L1RegressionActionHead(nn.Module):
    """Reduced from VLA-Adapter L1RegressionActionHead.

    The caller owns action-query initialization and loss computation. By
    default this module returns hidden states for VLAPolicy's external action
    decoder; ``output_action_dim=True`` switches to the vla-gemma-4 baseline
    shape where the internal MLP directly emits actions.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_action_chunks: int,
        num_blocks: int,
        num_task_tokens: int,
        use_grad_checkpoint: bool = False,
        use_wrist_bridge: bool = False,
        gating_init: float = 0.0,
        gating_init_wrist: float = 0.0,
        ungated_streams: bool = False,
        use_proper_residual: bool = False,
        proper_ffn_mode: str = "legacy",
        layer_scale_init: float = 0.0,
        mlp_ratio: float = 1.0,
        output_action_dim: bool = False,
        use_soft_prompt_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.num_task_tokens = num_task_tokens
        self.use_wrist_bridge = use_wrist_bridge
        self.output_action_dim = output_action_dim

        # The MLPResNet was originally fed [B, T, action_dim*hidden_dim]; we
        # keep the same input dim so the FC1 size matches the reference. The
        # caller reshapes `x` to [B, T, action_dim*hidden_dim] before passing.
        # ``output_action_dim=True`` matches the vla-gemma-4 baseline where
        # ``action_head.model.fc2`` directly outputs ``action_dim`` and there
        # is no external action_decoder. Default False keeps the legacy
        # GEM-4-VLA shape (head outputs hidden_dim, an external
        # ``action_decoder`` projects to action_dim).
        out_dim = action_dim if output_action_dim else hidden_dim
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=action_dim * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=out_dim,
            action_dim=action_dim,
            use_grad_checkpoint=use_grad_checkpoint,
            use_wrist_bridge=use_wrist_bridge,
            gating_init=gating_init,
            gating_init_wrist=gating_init_wrist,
            ungated_streams=ungated_streams,
            use_proper_residual=use_proper_residual,
            proper_ffn_mode=proper_ffn_mode,
            layer_scale_init=layer_scale_init,
            mlp_ratio=mlp_ratio,
            use_soft_prompt_cross_attn=use_soft_prompt_cross_attn,
        )

    def forward(
        self,
        x: torch.Tensor,                  # [B, T, A*D]
        h_a: torch.Tensor,                # [B, L+1, Q, D]
        h_t: torch.Tensor,                # [B, L+1, K_t, D]
        p=None,                           # [B, 1, D] or None when proprio_in_llm=True
        h_w=None,                         # [B, K_w, D] (Bridge self-attn pool)
        h_sp=None,                        # [B, K_sp, D] (Bridge self-attn pool)
        h_w_bridge=None,                  # [B, num_blocks+1, K_w_b, D] (per-layer wrist cross-attn)
        h_sp_per_layer=None,              # [B, num_blocks+1, K_sp, D] arch v3 soft_prompt per-layer hidden
        h_t_mask=None,                    # [B, K_t] arch v3 task-stream pad mask (prompt only)
    ) -> torch.Tensor:
        return self.model(
            x, h_a=h_a, h_t=h_t, p=p, h_w=h_w, h_sp=h_sp, h_w_bridge=h_w_bridge,
            h_sp_per_layer=h_sp_per_layer, h_t_mask=h_t_mask,
        )
