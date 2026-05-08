"""Minimal TDD test for build_app — full smoke is in test_serve_smoke.py
(Task 11). This file enforces TDD discipline for Task 8 itself: the
build_app entry must fail import before the implementation lands."""
from fastapi.testclient import TestClient

from vla_project.deployment.inference_server import build_app


def test_build_app_returns_fastapi_with_healthz():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path="configs/deploy/v36_libero_spatial.yaml",
        domain_id=0,
        inject_sleep_s=0.0,
    )
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_admin_schema_route_returns_200():
    """F5 regression: /admin/schema must respond alongside /healthz on a
    minimal hold_position-mode build_app."""
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path="configs/deploy/v36_libero_spatial.yaml",
        domain_id=0,
        inject_sleep_s=0.0,
    )
    client = TestClient(app)
    resp = client.get("/admin/schema")
    assert resp.status_code == 200
    assert "predictor" in resp.json()
