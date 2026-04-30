"""Tests for denormalize_action_q99 (inverse of normalize_action_q99)."""
import torch

from vla_project.data.normalization import (
    Q99Stats,
    denormalize_action_q99,
    normalize_action_q99,
)


def _stats(q01_val: float = -2.0, q99_val: float = 2.0, gripper_passthrough: bool = True) -> Q99Stats:
    A = 7
    return Q99Stats(
        q01=torch.tensor([q01_val] * A, dtype=torch.float32),
        q99=torch.tensor([q99_val] * A, dtype=torch.float32),
        mask=torch.tensor([True] * (A - 1) + [not gripper_passthrough], dtype=torch.bool),
    )


def test_denormalize_round_trip_clipped() -> None:
    stats = _stats(-2.0, 2.0, gripper_passthrough=True)
    raw = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        [1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 1.0],
        [-2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=torch.float32)
    normed = normalize_action_q99(raw, stats)
    denormed = denormalize_action_q99(normed, stats)
    assert torch.allclose(denormed[:, :6], raw[:, :6], atol=1e-5)
    assert torch.allclose(denormed[:, 6], raw[:, 6])


def test_denormalize_outside_range_does_not_inflate() -> None:
    stats = _stats(-2.0, 2.0, gripper_passthrough=True)
    normed = torch.tensor([[ 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.5]], dtype=torch.float32)
    out = denormalize_action_q99(normed, stats)
    assert out[0, 0].item() == 2.0
    assert out[0, 1].item() == -2.0


def test_denormalize_preserves_dtype_and_shape() -> None:
    stats = _stats()
    normed = torch.zeros(4, 5, 7, dtype=torch.float32)
    out = denormalize_action_q99(normed, stats)
    assert out.shape == normed.shape
    assert out.dtype == normed.dtype


def test_denormalize_rejects_wrong_last_dim() -> None:
    import pytest
    stats = _stats()
    with pytest.raises(ValueError):
        denormalize_action_q99(torch.zeros(2, 5), stats)


def test_denormalize_mask_false_passthrough() -> None:
    stats = _stats()
    # Note: 0.42 isn't exactly representable in float32, so the value the
    # tensor stores is 0.41999998... — a literal `== 0.42` comparison
    # against `.item()` would fail purely on float-precision grounds, even
    # though the value is preserved bit-for-bit by passthrough. Compare to
    # what we actually put in instead.
    normed = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.42]], dtype=torch.float32)
    out = denormalize_action_q99(normed, stats)
    assert out[0, 6].item() == normed[0, 6].item()
