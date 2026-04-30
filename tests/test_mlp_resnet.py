import torch
from vla_project.models.action_heads.mlp_resnet import MLPResNet


def test_stack_forward_shape():
    B, T, D = 2, 8, 32
    L = 4   # 4-block stack
    K_t, K_a = 16, 8
    model = MLPResNet(num_blocks=L, hidden_dim=D, action_dim=7,
                      input_dim=D * 7, output_dim=7)
    x = torch.randn(B, T, D * 7)
    h_t = torch.randn(B, L + 1, K_t, D)
    h_a = torch.randn(B, L + 1, K_a, D)
    p = torch.randn(B, 1, D)
    y = model(x, h_a=h_a, h_t=h_t, p=p)
    assert y.shape == (B, T, 7)
