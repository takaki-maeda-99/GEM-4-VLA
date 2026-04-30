import torch
import torch.nn as nn

from vla_project.models.action_heads.mlp_resnet_block_pro import MLPResNetBlock_Pro


class MLPResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.ModuleList(
            [MLPResNetBlock_Pro(dim=hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,            # [B, T, input_dim]
        h_a: torch.Tensor,          # [B, num_layers+1, K_a, D]
        h_t: torch.Tensor,          # [B, num_layers+1, K_t, D]
        p: torch.Tensor,            # [B, 1, D]
    ) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for i, blk in enumerate(self.blocks):
            x = blk(x, h_a=h_a[:, i + 1], h_t=h_t[:, i + 1], p=p)
        x = self.fc2(self.layer_norm2(x))
        return x
