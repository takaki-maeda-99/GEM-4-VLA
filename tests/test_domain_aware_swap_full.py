import torch

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_pred_action_differs_per_domain():
    cfg = VLAPolicyConfig(num_domains=2, hidden_dim=32, num_blocks=4, prompt_max_len=10)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    # ensure per-domain weights are actually distinct
    for proj in [policy.scene_proj, policy.wrist_proj, policy.proprio_proj,
                 policy.action_decoder]:
        torch.nn.init.normal_(proj.fc.weight, std=0.5)

    B = 1
    common = dict(
        scene_image=torch.randn(B, 3, 224, 224),
        wrist_image=torch.randn(B, 3, 224, 224),
        prompt_input_ids=torch.zeros(B, 10, dtype=torch.long),
        prompt_attention_mask=torch.ones(B, 10, dtype=torch.long),
        proprio=torch.randn(B, 8),
        last_action_chunk=torch.randn(B, 8, 7),
        target_action=torch.randn(B, 8, 7),
        action_mask=torch.ones(B, 8, dtype=torch.bool),
    )
    pred0, _ = policy({**common, "domain_id": torch.zeros(B, dtype=torch.long)})
    pred1, _ = policy({**common, "domain_id": torch.ones(B, dtype=torch.long)})
    assert not torch.allclose(pred0, pred1)
