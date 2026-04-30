"""Forward pass works when use_wrist_pool=True (49 tokens instead of 256)."""
import torch

from vla_project.data import constants as C
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from tests._stubs import _StubGemma, _StubSig


def _make_batch(B: int = 1) -> dict:
    return {
        "domain_id": torch.zeros(B, dtype=torch.long),
        "scene_image": torch.randn(B, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE),
        "wrist_image": torch.randn(B, 3, C.SIGLIP_IMAGE_SIZE, C.SIGLIP_IMAGE_SIZE),
        "prompt_input_ids": torch.zeros(B, C.DEFAULT_PROMPT_MAX_LEN, dtype=torch.long),
        "prompt_attention_mask": torch.ones(B, C.DEFAULT_PROMPT_MAX_LEN, dtype=torch.long),
        "proprio": torch.randn(B, C.PROPRIO_DIM),
        "last_action_chunk": torch.randn(B, C.ACTION_CHUNK_LEN, C.ACTION_DIM),
        "target_action": torch.randn(B, C.ACTION_CHUNK_LEN, C.ACTION_DIM),
        "action_mask": torch.ones(B, C.ACTION_CHUNK_LEN, dtype=torch.bool),
    }


def test_forward_with_wrist_pool() -> None:
    cfg = VLAPolicyConfig(
        num_domains=1,
        hidden_dim=32,
        num_blocks=4,
        use_wrist_pool=True,
        wrist_pool_tokens=49,
    )
    model = VLAPolicy(cfg, _StubSig(), _StubGemma())
    pred, loss = model(_make_batch(B=1))
    assert pred.shape == (1, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)


def test_forward_without_wrist_pool_unchanged_default() -> None:
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    model = VLAPolicy(cfg, _StubSig(), _StubGemma())
    pred, _loss = model(_make_batch(B=1))
    assert pred.shape == (1, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
