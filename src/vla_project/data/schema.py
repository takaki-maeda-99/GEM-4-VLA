from typing import TypedDict

import torch

from vla_project.data import constants as C


class Batch(TypedDict):
    domain_id: torch.Tensor          # [B] long
    scene_image: torch.Tensor        # [B, 3, H, W] float
    wrist_image: torch.Tensor        # [B, 3, H, W] float
    prompt_input_ids: torch.Tensor   # [B, Lt] long
    prompt_attention_mask: torch.Tensor  # [B, Lt] long
    proprio: torch.Tensor            # [B, D_prop] float
    last_action_chunk: torch.Tensor  # [B, T, A] float
    target_action: torch.Tensor      # [B, T, A] float
    action_mask: torch.Tensor        # [B, T] bool


def validate_batch(batch: Batch) -> None:
    B = batch["domain_id"].shape[0]
    assert batch["domain_id"].dtype == torch.long
    assert batch["domain_id"].shape == (B,)

    assert batch["scene_image"].shape[:2] == (B, 3)
    assert batch["scene_image"].shape[-2:] == (
        C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE,
    )
    assert batch["wrist_image"].shape == batch["scene_image"].shape

    assert batch["prompt_input_ids"].dtype == torch.long
    assert batch["prompt_attention_mask"].shape == batch["prompt_input_ids"].shape

    assert batch["proprio"].shape == (B, C.PROPRIO_DIM)

    T, A = C.ACTION_CHUNK_LEN, C.ACTION_DIM
    assert batch["last_action_chunk"].shape == (B, T, A)
    assert batch["target_action"].shape == (B, T, A)
    assert batch["action_mask"].shape == (B, T)
    assert batch["action_mask"].dtype == torch.bool
