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
        output_action_dim: bool = False,
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
        # X-VLA-Adapter shape (head outputs hidden_dim, an external
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
        )

    def forward(
        self,
        x: torch.Tensor,                  # [B, T, A*D]
        h_a: torch.Tensor,                # [B, L+1, Q, D]
        h_t: torch.Tensor,                # [B, L+1, K_t, D]
        p: torch.Tensor,                  # [B, 1, D]
        h_w=None,                         # [B, K_w, D] (Bridge self-attn pool)
        h_sp=None,                        # [B, K_sp, D] (Bridge self-attn pool)
        h_w_bridge=None,                  # [B, num_blocks+1, K_w_b, D] (per-layer wrist cross-attn)
    ) -> torch.Tensor:
        return self.model(
            x, h_a=h_a, h_t=h_t, p=p, h_w=h_w, h_sp=h_sp, h_w_bridge=h_w_bridge,
        )
