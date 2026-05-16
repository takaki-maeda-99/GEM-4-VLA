"""/metadata response builder — spec §4 schema."""
from __future__ import annotations

from vla_project.deployment.metadata import build_metadata_response


def _meta(with_native: bool = False) -> dict:
    m = {
        "step": 30000,
        "git_commit": "abc123",
        "cfg": {
            "data": {"unnorm_key": "bottle_pick", "action_chunk_len": 8, "domain_id": 13},
            "model": {"num_domains": 16, "proprio_dim": 8},
            "language": {"model_name": "google/gemma-4-E2B"},
        },
        "norm_stats": {
            "bottle_pick": {
                "action":  {"q99": [0.0] * 7,  "mean": [0.0] * 7,  "mask": [True] * 7},
                "proprio": {"q99": [0.0] * 8,  "mean": [0.0] * 8,  "mask": [True] * 8},
            }
        },
    }
    if with_native:
        m["native_action"] = {
            "units": "meter_axisangle_rad",
            "frame": "world",
            "gripper": {"kind": "absolute", "units": "normalized_0_1",
                        "sign": {"closed": 0, "open": 1}},
        }
    return m


def test_metadata_minimum_fields():
    resp = build_metadata_response(
        _meta(), unnorm_key="bottle_pick", domain_id=13,
        has_post_process=False, post_process_path=None,
    )
    assert resp["step"] == 30000
    assert resp["model_name"] == "google/gemma-4-E2B"
    assert resp["git_commit"] == "abc123"
    assert resp["action_dim"] == 7
    assert resp["proprio_dim"] == 8
    assert resp["action_chunk_len"] == 8
    assert resp["domain_id"] == 13
    assert resp["num_domains"] == 16
    assert resp["unnorm_key"] == "bottle_pick"
    assert resp["native_action"] is None
    assert resp["has_post_process"] is False
    assert resp["post_process_module"] is None


def test_metadata_with_native_action():
    resp = build_metadata_response(
        _meta(with_native=True), unnorm_key="bottle_pick", domain_id=13,
        has_post_process=False, post_process_path=None,
    )
    assert resp["native_action"]["frame"] == "world"
    assert resp["native_action"]["gripper"]["kind"] == "absolute"


def test_metadata_post_process_path():
    resp = build_metadata_response(
        _meta(), unnorm_key="bottle_pick", domain_id=13,
        has_post_process=True, post_process_path="/cache/foo/post_process.py",
    )
    assert resp["has_post_process"] is True
    assert resp["post_process_module"] == "/cache/foo/post_process.py"
