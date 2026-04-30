import torch.nn as nn
from vla_project.training.optim import build_optimizer

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_param_groups_present_and_no_frozen_group():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    # simulate Stage 1 freeze policy
    for p in policy.vision_encoder.parameters():
        p.requires_grad = False
    for p in policy.gemma.parameters():
        p.requires_grad = False
    optim = build_optimizer(policy, lr=1e-4, soft_lr_coef=2.0, weight_decay=0.01)
    names = {g["name"] for g in optim.param_groups}
    assert {"soft_prompts", "action_queries", "domain_projs", "action_head"} <= names
    assert "vlm_frozen" not in names
