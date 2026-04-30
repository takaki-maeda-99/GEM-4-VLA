import torch

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_forward_shape_and_loss_finite():
    cfg = VLAPolicyConfig(
        num_domains=2, hidden_dim=32, action_dim=7, action_chunk_len=8, proprio_dim=8,
        prompt_max_len=10, num_blocks=4,
    )
    policy = VLAPolicy(cfg, vision_encoder=_StubSig(), gemma=_StubGemma())
    B = 2
    batch = dict(
        domain_id=torch.zeros(B, dtype=torch.long),
        scene_image=torch.randn(B, 3, 224, 224),
        wrist_image=torch.randn(B, 3, 224, 224),
        prompt_input_ids=torch.zeros(B, 10, dtype=torch.long),
        prompt_attention_mask=torch.ones(B, 10, dtype=torch.long),
        proprio=torch.randn(B, 8),
        last_action_chunk=torch.randn(B, 8, 7),
        target_action=torch.randn(B, 8, 7),
        action_mask=torch.ones(B, 8, dtype=torch.bool),
    )
    pred, loss = policy(batch)
    assert pred.shape == (B, 8, 7)
    assert torch.isfinite(loss)
