"""Tests for HoldPositionChunkPredictor — emit zero ee_delta + native midpoint
gripper. See spec §Section 5 lines 433-460 for the design rationale (zero
across all columns would silently command CLOSED for normalized_0_1 native)."""
import numpy as np
import pytest

from vla_project.deployment.predictors.hold_position import HoldPositionChunkPredictor


def test_chunk_len_and_action_dim_are_constructor_args():
    p = HoldPositionChunkPredictor(chunk_len=8, action_dim=7)
    assert p.chunk_len == 8
    assert p.action_dim == 7


def test_predict_returns_zeros_for_ee_delta_columns():
    p = HoldPositionChunkPredictor(chunk_len=8, action_dim=7)
    out = p.predict(obs={})
    assert out.shape == (8, 7)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[:, :6], np.zeros((8, 6), dtype=np.float32))


def test_predict_gripper_column_is_default_midpoint():
    p = HoldPositionChunkPredictor(chunk_len=8, action_dim=7)
    out = p.predict(obs={})
    np.testing.assert_array_equal(out[:, 6], np.full(8, 0.5, dtype=np.float32))


def test_predict_gripper_column_uses_configured_midpoint():
    p = HoldPositionChunkPredictor(chunk_len=8, action_dim=7, gripper_native_midpoint=0.0)
    out = p.predict(obs={})
    np.testing.assert_array_equal(out[:, 6], np.zeros(8, dtype=np.float32))


def test_predict_obs_is_unused_does_not_raise():
    """HoldPosition does not read obs at all."""
    p = HoldPositionChunkPredictor(chunk_len=4, action_dim=7)
    out_a = p.predict({})
    out_b = p.predict({"scene_image": "garbage"})
    np.testing.assert_array_equal(out_a, out_b)
