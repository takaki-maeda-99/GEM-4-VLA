import torch
from vla_project.models.action_heads.l1_regression_action_head import L1RegressionActionHead


def test_predict_action_shape():
    B, T, D, A = 2, 8, 16, 7
    L = 3
    K_t = 12
    head = L1RegressionActionHead(
        hidden_dim=D, action_dim=A, num_action_chunks=T,
        num_blocks=L, num_task_tokens=K_t,
    )
    x_init = torch.randn(B, T, A * D)   # MLPResNet.fc1 expects A*D
    h_a = torch.randn(B, L + 1, 64, D)
    h_t = torch.randn(B, L + 1, K_t, D)
    p = torch.randn(B, 1, D)
    out = head(x_init, h_a=h_a, h_t=h_t, p=p)
    assert out.shape == (B, T, D)
