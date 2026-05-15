"""ModelRuntime: loads meta.json + raises MetaJsonError on bad input.

Tests here exercise only the meta-loading path (no model weights needed).
from_export stops early on missing/malformed meta.json before touching the
neural network, so these tests run without GPU or real checkpoints.
"""
import json

import pytest

from vla_project.deployment.runtime import ModelRuntime, MetaJsonError


def _write_meta(tmp_path, payload):
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(payload))
    return p


def test_from_export_missing_meta_json_raises(tmp_path):
    with pytest.raises(MetaJsonError, match="meta.json"):
        ModelRuntime.from_export(tmp_path)


def test_from_export_missing_step_key_raises(tmp_path):
    _write_meta(tmp_path, {
        "cfg": {"model": {"num_domains": 1}},
        "norm_stats": {},
    })
    with pytest.raises(MetaJsonError, match="step"):
        ModelRuntime.from_export(tmp_path)


def test_from_export_missing_norm_stats_key_raises(tmp_path):
    _write_meta(tmp_path, {
        "step": 0,
        "cfg": {"model": {"num_domains": 1}},
    })
    with pytest.raises(MetaJsonError, match="norm_stats"):
        ModelRuntime.from_export(tmp_path)
