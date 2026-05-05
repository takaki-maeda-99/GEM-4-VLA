"""ModelRuntime stub: loads meta.json + provides startup validation hooks.
Full forward path is Phase 1.

Also covers the build_app() startup assertion errors that are wired through
domain_adapter.validate_startup_xvla; this file focuses on the runtime side
(meta loading + paths)."""
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


def test_from_export_loads_step_and_cfg_norm_stats(tmp_path):
    _write_meta(tmp_path, {
        "step": 15000,
        "cfg": {"model": {"num_domains": 1}, "data": {"unnorm_key": "k"}},
        "norm_stats": {"k": {"action": {}, "proprio": {}}},
    })
    rt = ModelRuntime.from_export(tmp_path)
    assert rt.step == 15000
    assert rt.cfg["model"]["num_domains"] == 1
    assert "k" in rt.norm_stats


def test_call_raises_not_implemented_in_phase_0(tmp_path):
    _write_meta(tmp_path, {
        "step": 0,
        "cfg": {"model": {"num_domains": 1}, "data": {"unnorm_key": "k"}},
        "norm_stats": {"k": {}},
    })
    rt = ModelRuntime.from_export(tmp_path)
    with pytest.raises(NotImplementedError, match="Phase 1"):
        rt({})
