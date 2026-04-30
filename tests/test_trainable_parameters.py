import torch.nn as nn

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def _freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def test_stage1_freeze_policy_only_adapters_trainable():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, vision_encoder=_StubSig(), gemma=_StubGemma())
    _freeze_module(policy.vision_encoder)
    _freeze_module(policy.gemma)

    trainable_names = {n for n, p in policy.named_parameters() if p.requires_grad}
    expected_prefixes = (
        "scene_proj", "wrist_proj", "proprio_proj", "last_action_proj",
        "action_decoder", "soft_prompt_hub", "action_query_hub", "action_head",
    )
    for n in trainable_names:
        assert n.startswith(expected_prefixes), f"unexpected trainable: {n}"
    for prefix in expected_prefixes:
        assert any(n.startswith(prefix) for n in trainable_names), f"missing trainable: {prefix}"
