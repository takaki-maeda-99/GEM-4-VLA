"""Tests for ee6d_loss / ee6d_loss_components."""
import torch

from vla_project.training.losses_ee6d import ee6d_loss, ee6d_loss_components


def test_pos_only_error_isolated():
    """target[..., 0] = 1, all else 0 → pos loss = 1/3 (mean over 3 slots),
    rot and grip = 0."""
    B, T = 2, 4
    pred = torch.zeros(B, T, 20)
    target = torch.zeros(B, T, 20)
    target[..., 0] = 1.0
    mask = torch.ones(B, T, dtype=torch.bool)

    c = ee6d_loss_components(pred, target, mask)
    assert abs(c["pos"].item() - 1.0 / 3.0) < 1e-6
    assert c["rot"].item() == 0.0
    assert c["grip"].item() == 0.0


def test_padding_dims_ignored():
    """Errors in [10:20] (bimanual padding) must NOT contribute to any loss."""
    B, T = 2, 4
    pred = torch.zeros(B, T, 20)
    target = torch.zeros(B, T, 20)
    target[..., 15] = 100.0  # pad slot blown out
    mask = torch.ones(B, T, dtype=torch.bool)

    c = ee6d_loss_components(pred, target, mask)
    assert c["pos"].item() == 0.0
    assert c["rot"].item() == 0.0
    assert c["grip"].item() == 0.0
    # And the total loss is also zero.
    assert ee6d_loss(pred, target, mask).item() == 0.0


def test_mask_excludes_timesteps():
    B, T = 2, 4
    pred = torch.zeros(B, T, 20)
    target = torch.zeros(B, T, 20)
    target[:, 0, 0] = 1.0  # error only at t=0
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 1:] = True

    c = ee6d_loss_components(pred, target, mask)
    assert c["pos"].item() == 0.0


def test_total_loss_obeys_per_channel_weights():
    """total = w_pos*pos + w_rot*rot + w_grip*grip with channel L1 means."""
    pred = torch.zeros(2, 4, 20)
    target = torch.zeros(2, 4, 20)
    target[..., 0:3] = 1.0   # pos slot mean abs = 1.0
    target[..., 3:9] = 2.0   # rot slot mean abs = 2.0
    target[..., 9:10] = 3.0  # grip mean abs = 3.0
    mask = torch.ones(2, 4, dtype=torch.bool)

    total = ee6d_loss(pred, target, mask, w_pos=1.0, w_rot=0.5, w_grip=2.0)
    expected = 1.0 * 1.0 + 0.5 * 2.0 + 2.0 * 3.0  # = 8.0
    assert abs(total.item() - expected) < 1e-5


def test_components_are_grad_safe_under_bf16():
    """bf16 inputs → fp32 internal accumulation → fp32 scalar loss."""
    pred = torch.randn(2, 8, 20, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn(2, 8, 20, dtype=torch.bfloat16)
    mask = torch.ones(2, 8, dtype=torch.bool)

    loss = ee6d_loss(pred, target, mask)
    assert loss.dtype == torch.float32
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_rejects_wrong_last_dim():
    import pytest
    pred = torch.zeros(2, 4, 7)  # native 7-dim, not ee6d
    target = torch.zeros(2, 4, 7)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with pytest.raises(ValueError):
        ee6d_loss(pred, target, mask)
