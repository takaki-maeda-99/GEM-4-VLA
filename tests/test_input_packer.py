import torch

from vla_project.data import constants as C
from vla_project.data.packing.input_packer import InputPacker, PackedIDs


def test_layout_and_indices():
    """V6 layout: [BOS][prompt][scene][PROPRIO][action][EOS]."""
    packer = InputPacker(
        bos_id=2, eos_id=1,
        prompt_max_len=10,
    )
    prompt_ids = torch.tensor([[42, 43, 44, 0, 0, 0, 0, 0, 0, 0]])
    prompt_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
    packed: PackedIDs = packer(prompt_ids, prompt_mask)
    L_expected = (
        1                              # BOS
        + 10                            # prompt
        + C.NUM_SCENE_TOKENS
        + 1                              # PROPRIO placeholder
        + C.NUM_ACTION_TOKENS
        + 1                              # EOS
    )
    assert packed.input_ids.shape == (1, L_expected)
    assert packed.input_ids[0, 0].item() == 2
    assert packed.input_ids[0, -1].item() == 1
    # New layout has no soft / wrist in the LLM input.
    assert "soft" not in packed.idx
    assert "wrist" not in packed.idx
    prompt_idx = packed.idx["prompt"][0]
    scene_idx = packed.idx["scene"][0]
    proprio_idx = packed.idx["proprio"][0]
    action_idx = packed.idx["action"][0]
    assert prompt_idx.numel() == 10
    assert scene_idx.numel() == C.NUM_SCENE_TOKENS
    assert proprio_idx.numel() == 1
    assert action_idx.numel() == C.NUM_ACTION_TOKENS

    # Block ordering: prompt < scene < proprio < action.
    assert prompt_idx[-1].item() < scene_idx[0].item()
    assert scene_idx[-1].item() < proprio_idx[0].item()
    assert proprio_idx[-1].item() < action_idx[0].item()

    assert (packed.input_ids[0, scene_idx] == C.IMAGE_SOFT_TOKEN_ID).all()
    assert (packed.input_ids[0, proprio_idx] == C.PROPRIO_PLACEHOLDER_IDX).all()
    assert (packed.input_ids[0, action_idx] >= C.ACTION_TOKEN_BEGIN_IDX).all()


def test_attention_mask_respects_prompt_padding():
    packer = InputPacker(bos_id=2, eos_id=1, prompt_max_len=4)
    prompt_ids = torch.tensor([[10, 11, 0, 0]])
    prompt_mask = torch.tensor([[1, 1, 0, 0]])
    packed = packer(prompt_ids, prompt_mask)
    # The two padded prompt positions must have attention_mask = 0
    prompt_start = packed.idx["prompt"][0][0].item()
    am = packed.attention_mask[0]
    assert am[prompt_start + 0].item() == 1
    assert am[prompt_start + 1].item() == 1
    assert am[prompt_start + 2].item() == 0
    assert am[prompt_start + 3].item() == 0
