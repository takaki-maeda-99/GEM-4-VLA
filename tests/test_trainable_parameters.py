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
        "scene_proj", "wrist_proj", "proprio_proj",
        "action_decoder", "soft_prompt_hub", "action_query_hub", "action_head",
    )
    for n in trainable_names:
        assert n.startswith(expected_prefixes), f"unexpected trainable: {n}"
    for prefix in expected_prefixes:
        assert any(n.startswith(prefix) for n in trainable_names), f"missing trainable: {prefix}"


def test_lora_makes_only_lora_params_trainable() -> None:
    """Stage 2: with LoRA enabled, LoRA params are trainable; base Gemma stays frozen.

    We use a stub Gemma replacement and inject LoRA into a synthetic linear-
    bearing module so we don't need real Gemma weights. The relevant
    boundary check is: trainable params under the gemma sub-module are all
    LoRA-named.
    """
    import torch.nn as nn

    from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper, _apply_lora

    # Build an empty wrapper, plant a synthetic text_model with q_proj/v_proj,
    # freeze it, then apply LoRA via the helper directly.
    wrapper = Gemma4Wrapper(_skip_load=True)
    inner = nn.Sequential(
        nn.Linear(4, 4),  # q_proj-by-name? not exactly — we'll alias below.
    )

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 4)
            self.v_proj = nn.Linear(4, 4)
            self.k_proj = nn.Linear(4, 4)

    class _Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Block(), _Block()])

    stub = _Stub()
    for p in stub.parameters():
        p.requires_grad = False
    wrapper.text_model = stub
    _apply_lora(wrapper.text_model, {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]})

    trainable_names = [n for n, p in wrapper.named_parameters() if p.requires_grad]
    assert trainable_names, "no trainable params after LoRA injection"
    for n in trainable_names:
        assert "lora_" in n, f"non-lora param trainable: {n}"
    # k_proj must NOT have lora and must remain frozen.
    for n, p in wrapper.named_parameters():
        if "k_proj" in n:
            assert not p.requires_grad, f"k_proj should be frozen: {n}"
