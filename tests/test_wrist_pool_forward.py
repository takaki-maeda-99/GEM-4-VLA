"""Forward pass with use_wrist_pool=True; integer-factor fast path equivalence."""
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


def test_forward_with_wrist_pool_integer_factor() -> None:
    """Default 8x8 (=64) pool from 16x16 raw — integer factor 2."""
    cfg = VLAPolicyConfig(
        num_domains=1, hidden_dim=32, num_blocks=4,
        use_wrist_pool=True, wrist_pool_tokens=64,
    )
    model = VLAPolicy(cfg, _StubSig(), _StubGemma())
    pred, loss = model(_make_batch(B=1))
    assert pred.shape == (1, C.ACTION_CHUNK_LEN, C.ACTION_DIM)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(loss)


def test_forward_with_wrist_pool_non_integer_factor() -> None:
    """7x7 (=49) pool from 16x16 — non-integer factor, falls through to
    adaptive_avg_pool2d. Same forward-shape contract."""
    cfg = VLAPolicyConfig(
        num_domains=1, hidden_dim=32, num_blocks=4,
        use_wrist_pool=True, wrist_pool_tokens=49,
    )
    model = VLAPolicy(cfg, _StubSig(), _StubGemma())
    pred, _loss = model(_make_batch(B=1))
    assert pred.shape == (1, C.ACTION_CHUNK_LEN, C.ACTION_DIM)


def test_forward_without_wrist_pool_unchanged_default() -> None:
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4)
    model = VLAPolicy(cfg, _StubSig(), _StubGemma())
    pred, _loss = model(_make_batch(B=1))
    assert pred.shape == (1, C.ACTION_CHUNK_LEN, C.ACTION_DIM)


def test_pool_wrist_integer_factor_matches_adaptive() -> None:
    """The integer-factor fast path produces numerically identical output to
    adaptive_avg_pool2d when factor is integer (e.g. 16 -> 8)."""
    cfg = VLAPolicyConfig(
        num_domains=1, hidden_dim=32, num_blocks=4,
        use_wrist_pool=True, wrist_pool_tokens=64,
    )
    model = VLAPolicy(cfg, _StubSig(), _StubGemma())
    x = torch.randn(2, 256, 1152)  # match _StubSig output dim
    fast = model._pool_wrist(x)
    # Reference path: adaptive pool channel-first.
    side = 16
    pooled_side = 8
    grid = x.transpose(1, 2).reshape(2, 1152, side, side)
    ref = (
        torch.nn.functional.adaptive_avg_pool2d(grid, (pooled_side, pooled_side))
        .reshape(2, 1152, pooled_side * pooled_side)
        .transpose(1, 2)
    )
    assert fast.shape == ref.shape == (2, 64, 1152)
    assert torch.allclose(fast, ref, atol=1e-5)
