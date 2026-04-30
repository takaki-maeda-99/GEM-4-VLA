import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vla_project.data.normalization import (
    Q99Stats,
    load_q99_stats,
    normalize_action_q99,
)


def _write_stats(tmp_path: Path) -> Path:
    payload = {
        "libero_spatial_no_noops": {
            "action": {
                "q01": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0],
                "q99": [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
                "mask": [True, True, True, True, True, True, False],
            }
        }
    }
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_q99_stats_round_trip(tmp_path: Path) -> None:
    p = _write_stats(tmp_path)
    stats = load_q99_stats(p, unnorm_key="libero_spatial_no_noops")
    assert isinstance(stats, Q99Stats)
    assert stats.q01.shape == (7,)
    assert stats.q99.shape == (7,)
    assert stats.mask.shape == (7,)
    assert stats.mask.dtype == torch.bool
    assert stats.mask[-1].item() is False  # gripper dim unchanged


def test_normalize_action_q99_clips_to_unit(tmp_path: Path) -> None:
    p = _write_stats(tmp_path)
    stats = load_q99_stats(p, unnorm_key="libero_spatial_no_noops")
    raw = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],   # midpoint of q01..q99 -> 0
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0],   # above q99 -> clipped to 1
        [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 0.0],  # below q01 -> -1
    ], dtype=torch.float32)
    out = normalize_action_q99(raw, stats)
    assert out.shape == raw.shape
    assert out.dtype == torch.float32
    # First 6 dims (mask=True) clipped into [-1, 1]
    assert torch.allclose(out[0, :6], torch.zeros(6), atol=1e-6)
    assert torch.allclose(out[1, :6], torch.ones(6), atol=1e-6)
    assert torch.allclose(out[2, :6], -torch.ones(6), atol=1e-6)
    # Last dim (mask=False, gripper) untouched
    assert out[0, 6].item() == pytest.approx(0.5)
    assert out[1, 6].item() == pytest.approx(1.0)
