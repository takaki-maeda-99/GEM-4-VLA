import math
import torch
from vla_project.models.action_heads.mlp_resnet_block_pro import MLPResNetBlock_Pro


def test_forward_shape():
    B, T, D = 2, 8, 64
    Ka = 65   # h_a (64 action queries) + 1 (proprio) — caller concatenates
    Kt = 256  # task tokens
    blk = MLPResNetBlock_Pro(dim=D)
    x = torch.randn(B, T, D)
    h_a = torch.randn(B, 64, D)
    p = torch.randn(B, 1, D)
    h_t = torch.randn(B, Kt, D)
    out = blk(x, h_a=h_a, h_t=h_t, p=p)
    assert out.shape == (B, T, D)


def test_gating_init_is_zero():
    blk = MLPResNetBlock_Pro(dim=64)
    assert torch.equal(blk.gating_factor, torch.zeros(1))
