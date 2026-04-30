import pytest
import torch

from vla_project.data.schema import Batch, validate_batch


def _make_batch(B=2):
    return Batch(
        domain_id=torch.zeros(B, dtype=torch.long),
        scene_image=torch.randn(B, 3, 224, 224),
        wrist_image=torch.randn(B, 3, 224, 224),
        prompt_input_ids=torch.zeros(B, 50, dtype=torch.long),
        prompt_attention_mask=torch.ones(B, 50, dtype=torch.long),
        proprio=torch.randn(B, 8),
        last_action_chunk=torch.randn(B, 8, 7),
        target_action=torch.randn(B, 8, 7),
        action_mask=torch.ones(B, 8, dtype=torch.bool),
    )


def test_validate_batch_accepts_valid():
    batch = _make_batch()
    validate_batch(batch)


def test_validate_batch_rejects_wrong_action_dim():
    batch = _make_batch()
    batch["target_action"] = torch.randn(2, 8, 5)
    with pytest.raises(AssertionError):
        validate_batch(batch)
