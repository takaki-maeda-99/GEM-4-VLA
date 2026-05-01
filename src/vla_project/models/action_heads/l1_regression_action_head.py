import torch
import torch.nn as nn

from vla_project.models.action_heads.mlp_resnet import MLPResNet


class L1RegressionActionHead(nn.Module):
    """Reduced from VLA-Adapter L1RegressionActionHead.

    Differences from VLA-Adapter original:
      - `x` is provided by caller (LastAction-projected sequence) — no zero init.
      - Output dim is the LLM hidden dim D, not action_dim. Final A-dim
        projection is done by the per-domain `action_decoder` in VLAPolicy.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        num_action_chunks: int,
        num_blocks: int,
        num_task_tokens: int,
        use_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.num_task_tokens = num_task_tokens

        # The MLPResNet was originally fed [B, T, action_dim*hidden_dim]; we
        # keep the same input dim so the FC1 size matches the reference. The
        # caller reshapes `x` to [B, T, action_dim*hidden_dim] before passing.
        self.model = MLPResNet(
            num_blocks=num_blocks,
            input_dim=action_dim * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            action_dim=action_dim,
            use_grad_checkpoint=use_grad_checkpoint,
        )

    def forward(
        self,
        x: torch.Tensor,                  # [B, T, A*D]
        h_a: torch.Tensor,                # [B, L+1, Q, D]
        h_t: torch.Tensor,                # [B, L+1, K_t, D]
        p: torch.Tensor,                  # [B, 1, D]
        h_w=None,                         # [B, K_w, D] (Bridge self-attn pool)
        h_sp=None,                        # [B, K_sp, D] (Bridge self-attn pool)
    ) -> torch.Tensor:
        return self.model(x, h_a=h_a, h_t=h_t, p=p, h_w=h_w, h_sp=h_sp)
