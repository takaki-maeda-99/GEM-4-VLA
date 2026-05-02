"""Pin that InputPacker honors config-driven token counts (not just constants).

V6 layout: [BOS][prompt][scene][PROPRIO 1][action][EOS]. Soft prompts and
wrist tokens no longer enter the LLM input — they are consumed by the action
head's self-attn pool via h_w / h_sp directly.
"""
import torch

from vla_project.data import constants as C
from vla_project.data.packing.input_packer import InputPacker


def test_default_constructor_matches_constants() -> None:
    pk = InputPacker(bos_id=2, eos_id=1, prompt_max_len=10)
    out = pk(
        torch.zeros(1, 10, dtype=torch.long),
        torch.ones(1, 10, dtype=torch.long),
    )
    expected_len = (
        1
        + 10
        + C.NUM_SCENE_TOKENS
        + 1                              # PROPRIO placeholder
        + C.NUM_ACTION_TOKENS
        + 1
    )
    assert out.input_ids.shape == (1, expected_len)
    assert out.idx["prompt"].shape == (1, 10)
    assert out.idx["scene"].shape == (1, C.NUM_SCENE_TOKENS)
    assert out.idx["proprio"].shape == (1, 1)
    assert out.idx["action"].shape == (1, C.NUM_ACTION_TOKENS)
    assert "soft" not in out.idx
    assert "wrist" not in out.idx


def test_custom_token_counts() -> None:
    pk = InputPacker(
        bos_id=2, eos_id=1, prompt_max_len=10,
        num_scene_tokens=16,
        num_action_queries=12,
    )
    out = pk(
        torch.zeros(2, 10, dtype=torch.long),
        torch.ones(2, 10, dtype=torch.long),
    )
    expected_len = 1 + 10 + 16 + 1 + 12 + 1
    assert out.input_ids.shape == (2, expected_len)
    assert out.idx["scene"].shape == (2, 16)
    assert out.idx["action"].shape == (2, 12)


def test_rejects_non_positive_counts() -> None:
    import pytest
    with pytest.raises(ValueError):
        InputPacker(bos_id=2, eos_id=1, prompt_max_len=10, num_scene_tokens=0)
    with pytest.raises(ValueError):
        InputPacker(bos_id=2, eos_id=1, prompt_max_len=10, num_action_queries=-1)
