import torch

from vla_project.data import constants as C
from vla_project.data.packing.input_packer import InputPacker, PackedIDs


def test_layout_and_indices():
    packer = InputPacker(
        bos_id=2, eos_id=1,
        prompt_max_len=10,
    )
    prompt_ids = torch.tensor([[42, 43, 44, 0, 0, 0, 0, 0, 0, 0]])
    prompt_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
    packed: PackedIDs = packer(prompt_ids, prompt_mask)
    L_expected = (
        1                              # BOS
        + C.NUM_SOFT_PROMPT_TOKENS
        + C.NUM_SCENE_TOKENS
        + 10                            # prompt
        + C.NUM_WRIST_TOKENS
        + C.NUM_ACTION_TOKENS
        + 1                              # EOS
    )
    assert packed.input_ids.shape == (1, L_expected)
    assert packed.input_ids[0, 0].item() == 2
    assert packed.input_ids[0, -1].item() == 1
    soft_idx = packed.idx["soft"][0]
    scene_idx = packed.idx["scene"][0]
    wrist_idx = packed.idx["wrist"][0]
    action_idx = packed.idx["action"][0]
    assert soft_idx.numel() == C.NUM_SOFT_PROMPT_TOKENS
    assert scene_idx.numel() == C.NUM_SCENE_TOKENS
    assert wrist_idx.numel() == C.NUM_WRIST_TOKENS
    assert action_idx.numel() == C.NUM_ACTION_TOKENS

    assert (packed.input_ids[0, soft_idx] >= C.SOFT_PROMPT_BEGIN_IDX).all()
    assert (packed.input_ids[0, scene_idx] == C.IMAGE_SOFT_TOKEN_ID).all()
    assert (packed.input_ids[0, wrist_idx] >= C.WRIST_PLACEHOLDER_BEGIN_IDX).all()
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
