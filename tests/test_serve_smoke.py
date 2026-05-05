"""End-to-end smoke for the FastAPI app via TestClient.

Spec §Section 6 Phase 0 acceptance gate items:
  (a) valid request → 200 with shape [8, 7], cols 0..5 zero, col 6 ≈ 0.5
  (b) missing scene_image → 422
  (c) proprio length wrong → 422
  (d) missing wrist_image when soft-required (zero-fill) → 200
      (the hard-required case requires xvla_adapter mode → Phase 1)
  (e) injected sleep → 200 + latency_budget_exceeded log
  (f) /healthz → ok
"""
import base64
import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vla_project.deployment.inference_server import build_app

DEPLOY_YAML = "configs/deploy/v36_libero_spatial.yaml"


def _b64_jpeg(size=224):
    img = Image.new("RGB", (size, size), color=(127, 127, 127))
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def client():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
        inject_sleep_s=0.0,
    )
    return TestClient(app)


@pytest.fixture
def slow_client():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
        inject_sleep_s=0.4,
    )
    return TestClient(app)


def _valid_request_body():
    return {
        "image_primary": _b64_jpeg(),
        "image_wrist": _b64_jpeg(),
        "proprio": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.5],
        "instruction": "pick up the bottle",
    }


def test_healthz_returns_ok(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["predictor"] == "HoldPositionChunkPredictor"
    assert "ready_at_ns" in body


def test_predict_holdposition_shape_and_content(client):
    r = client.post("/predict", json=_valid_request_body())
    assert r.status_code == 200
    body = r.json()
    actions = body["actions"]
    assert len(actions) == 8
    assert all(len(row) == 7 for row in actions)
    for row in actions:
        for v in row[:6]:
            assert v == pytest.approx(0.0)
        assert row[6] == pytest.approx(0.5)


def test_predict_missing_scene_image_is_422(client):
    body = _valid_request_body()
    del body["image_primary"]
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_predict_proprio_wrong_length_is_422(client):
    body = _valid_request_body()
    body["proprio"] = [0.0] * 5  # too short
    r = client.post("/predict", json=body)
    assert r.status_code == 422


def test_predict_wrist_omitted_zero_fills_and_returns_200(client):
    """v36 has wrist_view_dropout_p=0.3 → soft-required. Without ckpt loaded
    (HoldPosition mode), validate_startup_hold_position skips the wrist
    requirement check entirely. The runtime path zero-fills missing wrist."""
    body = _valid_request_body()
    body.pop("image_wrist")
    r = client.post("/predict", json=body)
    assert r.status_code == 200


def test_predict_hard_required_wrist_missing_at_request_returns_422(tmp_path):
    """When the deploy yaml's request_fields.wrist_image is set AND the ckpt
    cfg flags hard-required wrist, runtime should reject a request that
    omits wrist with 422. We can't load a real ckpt in Phase 0, so we
    simulate by forcing the `inference_server.build_app` to set a server-
    side flag wrist_hard_required=True via an xvla_adapter mode startup
    with a synthetic meta.json fixture that has use_wrist_bridge=True."""
    # Build a synthetic ckpt dir that satisfies validate_startup_xvla AND
    # has use_wrist_bridge=True so wrist is hard-required.
    ckpt_dir = tmp_path / "fake_v33"
    ckpt_dir.mkdir()
    meta = {
        "step": 0,
        "cfg": {
            "model": {
                "num_domains": 1,
                "use_wrist_bridge": True,
                "wrist_in_llm": False,
                "wrist_view_dropout_p": 0.0,
            },
            "data": {"unnorm_key": "libero_spatial_no_noops", "action_chunk_len": 8},
        },
        "norm_stats": {
            "libero_spatial_no_noops": {
                "action": {"mean": [0.0]*7, "std": [1.0]*7, "q01": [-1.0]*7, "q99": [1.0]*7,
                           "mask": [True]*6 + [False]},
                "proprio": {"mean": [0.0]*8, "std": [1.0]*8, "q01": [-1.0]*8, "q99": [1.0]*8,
                            "mask": [True]*8},
            }
        },
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta))
    app = build_app(
        predictor_kind="xvla_adapter",
        checkpoint=ckpt_dir,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
        inject_sleep_s=0.0,
    )
    c = TestClient(app)
    body = _valid_request_body()
    body.pop("image_wrist")
    r = c.post("/predict", json=body)
    # Phase 0: predict() raises NotImplementedError → 500; before it does,
    # preprocess + wrist_hard_required check should reject hard-required
    # wrist absent with 422.
    assert r.status_code == 422
    assert "wrist" in r.json()["detail"].lower()


def test_inject_sleep_emits_latency_budget_exceeded(slow_client, caplog):
    """Server still returns 200 within MimicRec's 5s timeout, but the log
    line carries latency_budget_exceeded=true. Phase 0 acceptance gate item 4."""
    caplog.set_level(logging.INFO, logger="vla_project.deployment")
    r = slow_client.post("/predict", json=_valid_request_body())
    assert r.status_code == 200
    # Find the per-request log line emitted by inference_server._log_request.
    matched = [rec for rec in caplog.records if rec.name == "vla_project.deployment"]
    assert matched, "expected at least one log record from vla_project.deployment"
    payload = json.loads(matched[-1].getMessage())
    assert payload.get("latency_budget_exceeded") is True
    assert payload["elapsed_ms"] > 266.0
