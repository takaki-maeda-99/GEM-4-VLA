"""Tests for the pure `compute_q99_stats` helper added to data/normalization.py."""
from typing import List

import numpy as np
import pytest
import torch

from vla_project.data.normalization import (
    Q99Stats,
    compute_q99_stats,
    normalize_action_q99,
)


def _ramp_actions(n: int = 1000, a: int = 7) -> np.ndarray:
    """Linear ramp from -10 to 10 on each dim, shifted per dim, with a binary final dim."""
    base = np.linspace(-10.0, 10.0, n, dtype=np.float32)  # [N]
    arr = np.stack([base + i for i in range(a - 1)], axis=1)  # [N, A-1]
    gripper = (np.arange(n) % 2).astype(np.float32).reshape(-1, 1)  # binary 0/1
    return np.concatenate([arr, gripper], axis=1)  # [N, A]


def test_returns_q99_stats_with_correct_shapes() -> None:
    arr = _ramp_actions(500, 7)
    mask = [True, True, True, True, True, True, False]
    stats = compute_q99_stats(arr, mask=mask)
    assert isinstance(stats, Q99Stats)
    assert stats.q01.shape == (7,)
    assert stats.q99.shape == (7,)
    assert stats.mask.shape == (7,)
    assert stats.mask.dtype == torch.bool
    assert stats.mask.tolist() == mask


def test_default_mask_is_all_true() -> None:
    arr = _ramp_actions(100, 7)
    stats = compute_q99_stats(arr)
    assert stats.mask.tolist() == [True] * 7


def test_q01_q99_match_numpy_quantiles() -> None:
    arr = _ramp_actions(10000, 7)
    stats = compute_q99_stats(arr)
    expected_q01 = np.quantile(arr, 0.01, axis=0)
    expected_q99 = np.quantile(arr, 0.99, axis=0)
    assert np.allclose(stats.q01.numpy(), expected_q01, atol=1e-4)
    assert np.allclose(stats.q99.numpy(), expected_q99, atol=1e-4)


def test_round_trip_into_normalize() -> None:
    """Computed stats applied via normalize_action_q99 produce values in [-1, 1] on mask=True dims."""
    arr = _ramp_actions(1000, 7)
    mask = [True] * 6 + [False]
    stats = compute_q99_stats(arr, mask=mask)
    normed = normalize_action_q99(torch.from_numpy(arr), stats)
    # mask=True dims fall in [-1, 1] (within numerical slack)
    assert normed[:, :6].abs().max().item() <= 1.0 + 1e-6
    # mask=False dim passes through (still 0 or 1)
    assert torch.all((normed[:, 6] == 0.0) | (normed[:, 6] == 1.0))


def test_accepts_torch_input() -> None:
    arr = torch.from_numpy(_ramp_actions(200, 7))
    stats = compute_q99_stats(arr)
    assert stats.q01.shape == (7,)


def test_rejects_wrong_rank() -> None:
    arr = np.zeros((10, 8, 7), dtype=np.float32)  # 3-D input
    with pytest.raises(ValueError):
        compute_q99_stats(arr)


def test_rejects_mask_length_mismatch() -> None:
    arr = _ramp_actions(50, 7)
    with pytest.raises(ValueError):
        compute_q99_stats(arr, mask=[True, True, True])  # only 3 entries
