import torch
from vla_project.models.action_heads.rope import RotaryEmbedding, apply_rope


def test_rope_shapes_and_does_not_change_q_norm():
    B, H, L, Dh = 2, 4, 6, 8
    q = torch.randn(B, H, L, Dh)
    k = torch.randn(B, H, L, Dh)
    rope = RotaryEmbedding(dim=Dh)
    cos, sin = rope(seq_len=L, device=q.device, dtype=q.dtype)
    qr, kr = apply_rope(q, k, cos, sin)
    assert qr.shape == q.shape
    assert kr.shape == k.shape
    # RoPE is norm-preserving along last dim
    torch.testing.assert_close(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4, rtol=1e-4)
