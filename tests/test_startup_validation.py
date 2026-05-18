"""Startup validation: non-contract checks per spec §8.

Verifies:
  1. domain_id ∈ [0, cfg.model.num_domains)
  2. unnorm_key ∈ meta.norm_stats
  3. action_chunk_len: cfg.data ↔ cfg.model agree (where both exist)
  4. norm_stats[key].action dim ↔ cfg.model action dim
  5. norm_stats[key].proprio dim ↔ cfg.model.proprio_dim
  6. q01/q99/mask/std/min/max shapes agree
  7. wrist hard-required derivation (returns the bool, does not assert)
  8. native_action missing → warn but do not raise

This is logic-only; it does not load a model.
"""
from __future__ import annotations

import logging

import pytest

from vla_project.deployment.startup_validation import (
    HardFailAssertion,
    derive_wrist_hard_required,
    resolve_unnorm_key,
    validate_runtime,
)


def _good_meta(unnorm_key: str = "k", action_dim: int = 7, proprio_dim: int = 8) -> dict:
    return {
        "cfg": {
            "data": {"unnorm_key": unnorm_key, "action_chunk_len": 8, "domain_id": 0},
            "model": {"num_domains": 16, "proprio_dim": proprio_dim, "action_chunk_len": 8},
            "language": {"model_name": "google/gemma-4-E2B"},
        },
        "norm_stats": {
            unnorm_key: {
                "action":  {k: [0.0] * action_dim for k in ("q01", "q99", "mean", "std", "min", "max")}
                          | {"mask": [True] * action_dim},
                "proprio": {k: [0.0] * proprio_dim for k in ("q01", "q99", "mean", "std", "min", "max")}
                          | {"mask": [True] * proprio_dim},
            }
        },
        "step": 1000,
        "git_commit": "abc123",
    }


def test_resolve_unnorm_key_single_auto():
    meta = _good_meta()
    assert resolve_unnorm_key(meta, override=None) == "k"


def test_resolve_unnorm_key_multiple_requires_override():
    meta = _good_meta()
    meta["norm_stats"]["other"] = meta["norm_stats"]["k"]
    with pytest.raises(HardFailAssertion, match="--unnorm-key"):
        resolve_unnorm_key(meta, override=None)


def test_resolve_unnorm_key_override_valid():
    meta = _good_meta()
    assert resolve_unnorm_key(meta, override="k") == "k"


def test_resolve_unnorm_key_override_missing():
    meta = _good_meta()
    with pytest.raises(HardFailAssertion, match="not in"):
        resolve_unnorm_key(meta, override="ghost")


def test_validate_runtime_passes_on_good_meta():
    validate_runtime(_good_meta(), unnorm_key="k", domain_id=0, model_action_dim=7)


def test_validate_runtime_domain_id_out_of_range():
    with pytest.raises(HardFailAssertion, match="domain_id"):
        validate_runtime(_good_meta(), unnorm_key="k", domain_id=100, model_action_dim=7)


def test_validate_runtime_action_dim_mismatch():
    with pytest.raises(HardFailAssertion, match="action_dim"):
        validate_runtime(_good_meta(), unnorm_key="k", domain_id=0, model_action_dim=99)


def test_validate_runtime_q99_shape_mismatch():
    meta = _good_meta()
    meta["norm_stats"]["k"]["action"]["q99"] = [0.0] * 5  # wrong len
    with pytest.raises(HardFailAssertion, match="action_dim"):
        validate_runtime(meta, unnorm_key="k", domain_id=0, model_action_dim=7)


def test_validate_runtime_missing_native_action_warns(caplog):
    with caplog.at_level("WARNING"):
        validate_runtime(_good_meta(), unnorm_key="k", domain_id=0, model_action_dim=7)
    assert any("native_action" in rec.message for rec in caplog.records)


def test_derive_wrist_hard_required_bridge_true():
    meta = _good_meta()
    meta["cfg"]["model"]["use_wrist_bridge"] = True
    assert derive_wrist_hard_required(meta) is True


def test_derive_wrist_hard_required_dropout_zero_in_llm():
    meta = _good_meta()
    meta["cfg"]["model"]["wrist_in_llm"] = True
    meta["cfg"]["model"]["wrist_view_dropout_p"] = 0.0
    assert derive_wrist_hard_required(meta) is True


def test_derive_wrist_hard_required_dropout_nonzero():
    meta = _good_meta()
    meta["cfg"]["model"]["wrist_in_llm"] = True
    meta["cfg"]["model"]["wrist_view_dropout_p"] = 0.5
    assert derive_wrist_hard_required(meta) is False
