import torch
import torch.nn as nn
from vla_project.training.optim import build_optimizer

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_param_groups_present_and_no_frozen_group():
    """Stage 1: SigLIP + Gemma frozen; the rest is partitioned into named
    groups with the documented per-group LR coefficients."""
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    for p in policy.vision_encoder.parameters():
        p.requires_grad = False
    for p in policy.gemma.parameters():
        p.requires_grad = False
    optim = build_optimizer(policy, lr=1e-4, weight_decay=0.01)
    names = {g["name"] for g in optim.param_groups}
    # New name set: gemma_lora / siglip / soft_prompts / action_queries / projections / action_head.
    # Frozen modules (siglip + gemma in Stage 1) contribute zero trainable
    # params and so are dropped (empty group filter).
    assert {"soft_prompts", "action_queries", "projections", "action_head"} <= names
    assert "siglip" not in names
    assert "gemma_lora" not in names


def test_default_per_group_lr_coefficients():
    """SigLIP/Gemma 0.1, SoftPrompt 0.1, ActionQuery/Projections/ActionHead 1.0
    relative to base lr — matches GEM-4-VLA convention."""
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    base = 1e-4
    optim = build_optimizer(policy, lr=base, weight_decay=0.01)
    by_name = {g["name"]: g["lr"] for g in optim.param_groups}
    assert by_name["action_head"]    == base
    assert by_name["action_queries"] == base
    assert by_name["projections"]    == base
    assert abs(by_name["soft_prompts"] - 0.1 * base) < 1e-12


def test_lr_coefs_overrides_defaults():
    """User-supplied lr_coefs dict wins over defaults, partial override OK."""
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    base = 1e-4
    optim = build_optimizer(
        policy, lr=base, weight_decay=0.0,
        lr_coefs={"action_head": 0.5},
    )
    by_name = {g["name"]: g["lr"] for g in optim.param_groups}
    assert by_name["action_head"] == 0.5 * base
    assert abs(by_name["soft_prompts"] - 0.1 * base) < 1e-12


def test_soft_lr_coef_deprecated_alias_still_works():
    """soft_lr_coef kwarg still works for back-compat with old configs."""
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    optim = build_optimizer(policy, lr=1e-4, soft_lr_coef=2.0, weight_decay=0.0)
    by_name = {g["name"]: g["lr"] for g in optim.param_groups}
    assert by_name["soft_prompts"] == 2e-4


def test_gemma_lora_picked_up_when_trainable():
    """Bug fix: Gemma LoRA params (any requires_grad=True under model.gemma)
    must land in the 'gemma_lora' group at 0.1 × base lr."""
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    for p in policy.vision_encoder.parameters():
        p.requires_grad = False
    for p in policy.gemma.parameters():
        p.requires_grad = False
    # Mimic LoRA injection by registering a fresh trainable parameter.
    policy.gemma.fake_lora = nn.Parameter(torch.zeros(2, 2))
    base = 1e-4
    optim = build_optimizer(policy, lr=base, weight_decay=0.0)
    by_name = {g["name"]: g["lr"] for g in optim.param_groups}
    assert "gemma_lora" in by_name
    assert abs(by_name["gemma_lora"] - 0.1 * base) < 1e-12
