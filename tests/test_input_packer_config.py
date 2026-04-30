"""Pin that InputPacker honors config-driven token counts (not just constants)."""
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
        + C.NUM_SOFT_PROMPT_TOKENS
        + C.NUM_SCENE_TOKENS
        + 10
        + C.NUM_WRIST_TOKENS
        + C.NUM_ACTION_TOKENS
        + 1
    )
    assert out.input_ids.shape == (1, expected_len)
    assert out.idx["soft"].shape == (1, C.NUM_SOFT_PROMPT_TOKENS)
    assert out.idx["wrist"].shape == (1, C.NUM_WRIST_TOKENS)
    assert out.idx["action"].shape == (1, C.NUM_ACTION_TOKENS)


def test_custom_token_counts() -> None:
    pk = InputPacker(
        bos_id=2, eos_id=1, prompt_max_len=10,
        num_soft_prompt_tokens=8,
        num_scene_tokens=16,
        num_wrist_tokens=49,
        num_action_queries=12,
    )
    out = pk(
        torch.zeros(2, 10, dtype=torch.long),
        torch.ones(2, 10, dtype=torch.long),
    )
    expected_len = 1 + 8 + 16 + 10 + 49 + 12 + 1
    assert out.input_ids.shape == (2, expected_len)
    assert out.idx["soft"].shape  == (2, 8)
    assert out.idx["scene"].shape == (2, 16)
    assert out.idx["wrist"].shape == (2, 49)
    assert out.idx["action"].shape == (2, 12)


def test_wrist_pool_value_within_constants() -> None:
    pk = InputPacker(
        bos_id=2, eos_id=1, prompt_max_len=10,
        num_wrist_tokens=49,
    )
    out = pk(
        torch.zeros(1, 10, dtype=torch.long),
        torch.ones(1, 10, dtype=torch.long),
    )
    assert out.idx["wrist"].shape == (1, 49)


def test_rejects_non_positive_counts() -> None:
    import pytest
    with pytest.raises(ValueError):
        InputPacker(bos_id=2, eos_id=1, prompt_max_len=10, num_wrist_tokens=0)
    with pytest.raises(ValueError):
        InputPacker(bos_id=2, eos_id=1, prompt_max_len=10, num_action_queries=-1)
