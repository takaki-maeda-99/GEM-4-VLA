"""q99 denorm with mask: dims where mask=True are q99-inverse;
dims where mask=False pass through unchanged.

Mirrors data/normalization.py:denormalize_action_q99 (the source of
truth) — this test exists because the deployment side now owns its own
copy of the logic, decoupled from the dataset module.
"""
from __future__ import annotations

import numpy as np
import pytest

from vla_project.deployment.wire_io import q99_denorm_with_mask


def _stats(action_dim: int = 4) -> dict:
    """Synthetic stats: q01 = -1, q99 = +1 → span=2; any mean/std/min/max ok."""
    return {
        "q01":  [-1.0] * action_dim,
        "q99":  [+1.0] * action_dim,
        "mask": [True, True, True, False],
        "mean": [0.0] * action_dim,
        "std":  [1.0] * action_dim,
        "min":  [-1.0] * action_dim,
        "max":  [+1.0] * action_dim,
    }


def test_denorm_masked_dims_are_q99_inverse():
    stats = _stats()
    # action_norm in [-1, +1] → expected in [-1, +1] (since q01=-1, q99=+1)
    action_norm = np.array([[0.0, 0.5, -0.5, 0.0]], dtype=np.float32)
    out = q99_denorm_with_mask(action_norm, stats)
    np.testing.assert_allclose(out[0, :3], [0.0, 0.5, -0.5], rtol=1e-6)


def test_denorm_unmasked_dim_passes_through():
    stats = _stats()
    action_norm = np.array([[0.0, 0.0, 0.0, 3.14]], dtype=np.float32)
    out = q99_denorm_with_mask(action_norm, stats)
    assert out[0, 3] == pytest.approx(3.14)


def test_denorm_dim_mismatch_raises():
    stats = _stats(action_dim=4)
    action_norm = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # 3 dims, stats want 4
    with pytest.raises(ValueError, match="action_dim"):
        q99_denorm_with_mask(action_norm, stats)
