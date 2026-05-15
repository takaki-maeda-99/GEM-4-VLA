"""F3 proprio sanity: isfinite + |normed| thresholds after q99 normalize.

Catches: NaN/Inf in proprio (F3a), and degenerate cases like deg/rad
swap which manifest as |normed| → 30+ after q99 (F3b).
"""
from __future__ import annotations

import numpy as np
import pytest

from vla_project.deployment.wire_io import (
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
    normalize_proprio_q99,
)


def _stats(dim: int = 4) -> dict:
    return {
        "q01":  [-1.0] * dim,
        "q99":  [+1.0] * dim,
        "mask": [True, True, True, False],
        "mean": [0.0] * dim,
        "std":  [1.0] * dim,
        "min":  [-1.0] * dim,
        "max":  [+1.0] * dim,
    }


def test_normalize_within_range_no_clip():
    raw = np.array([0.0, 0.5, -0.5, 0.0], dtype=np.float32)
    out, warned = normalize_proprio_q99(raw, _stats())
    assert warned is False
    np.testing.assert_allclose(out[:3], [0.0, 0.5, -0.5], atol=1e-6)


def test_normalize_passthrough_on_unmasked_dim():
    raw = np.array([0.0, 0.0, 0.0, 99.0], dtype=np.float32)
    out, _ = normalize_proprio_q99(raw, _stats())
    assert out[3] == pytest.approx(99.0)


def test_non_finite_raises():
    raw = np.array([0.0, np.nan, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        normalize_proprio_q99(raw, _stats())


def test_hard_threshold_raises():
    # masked dim values that produce |normed| > PROPRIO_OOD_HARD_ABS after q99.
    raw = np.array([100.0, 0.0, 0.0, 0.0], dtype=np.float32)  # → |normed| = 99 > HARD
    with pytest.raises(ValueError, match="hard"):
        normalize_proprio_q99(raw, _stats())


def test_warn_threshold_clips_and_flags():
    # masked dim values that produce |normed| in (WARN, HARD]
    raw = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)  # → |normed| = 2 > WARN, < HARD
    out, warned = normalize_proprio_q99(raw, _stats())
    assert warned is True
    assert -PROPRIO_OOD_WARN_ABS <= out[0] <= PROPRIO_OOD_WARN_ABS  # clipped
