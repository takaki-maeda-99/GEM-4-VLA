"""Shared test helpers for the deployment tests."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def pytest_configure(config):
    torch.manual_seed(0)


def write_synthetic_ckpt(tmp_path: Path, *, use_wrist_bridge: bool = False) -> Path:
    """Phase 0 ckpt with the minimal meta.json keys validate_startup_xvla demands.

    Pattern: identical to the inline fixture in test_serve_smoke.py:
    test_predict_hard_required_wrist_missing_at_request_returns_422.
    Future cleanup: dedupe that one too.
    """
    ckpt_dir = tmp_path / "fake_v36"
    ckpt_dir.mkdir()
    meta = {
        "step": 0,
        "cfg": {
            "model": {
                "num_domains": 1,
                "use_wrist_bridge": use_wrist_bridge,
                "wrist_in_llm": False,
                "wrist_view_dropout_p": 0.0,
            },
            "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8},
        },
        "norm_stats": {
            "libero_spatial_no_noops": {
                "action": {"mean": [0.0]*7, "std": [1.0]*7, "q01": [-1.0]*7,
                           "q99": [1.0]*7, "mask": [True]*6 + [False]},
                "proprio": {"mean": [0.0]*8, "std": [1.0]*8, "q01": [-1.0]*8,
                            "q99": [1.0]*8, "mask": [True]*8},
            }
        },
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))
    return ckpt_dir
