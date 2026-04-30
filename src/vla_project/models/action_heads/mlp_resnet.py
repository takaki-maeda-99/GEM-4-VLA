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
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.use_grad_checkpoint = use_grad_checkpoint
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList(
            [MLPResNetBlock_Pro(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def _run_block(self, blk: nn.Module, x: torch.Tensor,
                   h_a_i: torch.Tensor, h_t_i: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return blk(x, h_a=h_a_i, h_t=h_t_i, p=p)

    def forward(
        self,
        x: torch.Tensor,            # [B, T, input_dim]
        h_a: torch.Tensor,          # [B, num_layers+1, K_a, D]
        h_t: torch.Tensor,          # [B, num_layers+1, K_t, D]
        p: torch.Tensor,            # [B, 1, D]
    ) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for i, blk in enumerate(self.blocks):
            h_a_i = h_a[:, i + 1]
            h_t_i = h_t[:, i + 1]
            if self.use_grad_checkpoint and self.training:
                x = ckpt.checkpoint(
                    self._run_block, blk, x, h_a_i, h_t_i, p, use_reentrant=False
                )
            else:
                x = self._run_block(blk, x, h_a_i, h_t_i, p)
        x = self.fc2(self.layer_norm2(x))
        return x
