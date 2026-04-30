import torch

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def test_h_a_comes_from_action_positions():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4, prompt_max_len=10)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())

    B = 1
    batch = {
        "domain_id": torch.zeros(B, dtype=torch.long),
        "scene_image": torch.randn(B, 3, 224, 224),
        "wrist_image": torch.randn(B, 3, 224, 224),
        "prompt_input_ids": torch.zeros(B, 10, dtype=torch.long),
        "prompt_attention_mask": torch.ones(B, 10, dtype=torch.long),
        "proprio": torch.randn(B, 8),
        "last_action_chunk": torch.randn(B, 8, 7),
        "target_action": torch.randn(B, 8, 7),
        "action_mask": torch.ones(B, 8, dtype=torch.bool),
    }
    # Re-run the front of forward to reach hidden_states + indices.
    packed = policy.input_packer(batch["prompt_input_ids"], batch["prompt_attention_mask"])
    raw = policy.gemma.embed_tokens(packed.input_ids)
    out = policy.gemma(packed.input_ids, packed.attention_mask, inputs_embeds=raw)
    # In _StubGemma: hs[i] = raw + i; so hs at action_idx for layer i
    # equals raw[..., action_idx, :] + i.
    expected_layer1 = raw[:, packed.idx["action"][0]] + 1
    actual_layer1 = out.hidden_states[:, 1, packed.idx["action"][0]]
    assert torch.allclose(actual_layer1, expected_layer1)
