"""Integration: build_app accepts checkpoint only (no deploy yaml).

Uses a tiny local fake ckpt dir to avoid HF round-trips. The bottle
HF ckpt end-to-end check is a manual smoke (Task 14).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vla_project.deployment.inference_server import build_app


def _fake_local_ckpt(tmp_path: Path) -> Path:
    """Skeleton ckpt with meta.json only — model.pt absent.

    build_app with predictor_kind='hold_position' should NOT need the model.
    """
    meta = {
        "step": 1,
        "git_commit": "test",
        "cfg": {
            "data": {"unnorm_key": "k", "action_chunk_len": 8, "domain_id": 0},
            "model": {"num_domains": 4, "proprio_dim": 8, "action_chunk_len": 8},
            "language": {"model_name": "google/gemma-4-E2B"},
        },
        "norm_stats": {
            "k": {
                "action":  {fld: [0.0] * 7 for fld in ("q01", "q99", "mean", "std", "min", "max")}
                          | {"mask": [True] * 7},
                "proprio": {fld: [0.0] * 8 for fld in ("q01", "q99", "mean", "std", "min", "max")}
                          | {"mask": [True] * 8},
            }
        },
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def test_build_app_hold_position_metadata_endpoint(tmp_path):
    ckpt = _fake_local_ckpt(tmp_path)
    app = build_app(
        checkpoint=str(ckpt),
        predictor_kind="hold_position",
        domain_id=0,
        unnorm_key=None,
        trust_checkpoint_code=False,
    )
    client = TestClient(app)
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == 1
    assert body["action_dim"] == 7
    assert body["proprio_dim"] == 8
    assert body["has_post_process"] is False
    assert body["native_action"] is None


def test_build_app_healthz(tmp_path):
    ckpt = _fake_local_ckpt(tmp_path)
    app = build_app(
        checkpoint=str(ckpt), predictor_kind="hold_position", domain_id=0,
        unnorm_key=None, trust_checkpoint_code=False,
    )
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_build_app_rejects_unknown_unnorm_key(tmp_path):
    ckpt = _fake_local_ckpt(tmp_path)
    with pytest.raises(Exception, match="not in"):
        build_app(
            checkpoint=str(ckpt), predictor_kind="hold_position", domain_id=0,
            unnorm_key="ghost", trust_checkpoint_code=False,
        )


def test_build_app_rejects_out_of_range_domain_id(tmp_path):
    ckpt = _fake_local_ckpt(tmp_path)
    with pytest.raises(Exception, match="domain_id"):
        build_app(
            checkpoint=str(ckpt), predictor_kind="hold_position", domain_id=99,
            unnorm_key=None, trust_checkpoint_code=False,
        )
