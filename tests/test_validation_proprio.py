"""F3: proprio non-finite reject + OOD warn / hard-reject.

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F3.
Three behaviors:
  - F3a: NaN/inf → 422 unconditionally.
  - F3b warn: |normed| > PROPRIO_OOD_WARN_ABS (1.0) → control flow continues
    (clip absorbs), structured WARNING log with event=proprio_ood emitted.
  - F3b hard: |normed| > PROPRIO_OOD_HARD_ABS (10.0) → 422 with msg containing
    'unit mismatch'. Hard reject runs first; no proprio_ood warn is emitted
    when raising.
"""
import base64
import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vla_project.deployment.domain_adapter import (
    DomainAdapter,
    PROPRIO_OOD_HARD_ABS,
    PROPRIO_OOD_WARN_ABS,
    load_deploy_config,
)
from vla_project.deployment.inference_server import build_app

DEPLOY_YAML = "configs/deploy/v36_libero_spatial.yaml"


def _b64_jpeg() -> str:
    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def client():
    """hold_position mode — does not normalize proprio (no norm_stats), so OOD
    tests need xvla_adapter mode. We use hold_position only for the NaN/inf path
    which fires before normalization."""
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


# ----- F3a: non-finite proprio (any predictor mode) -----

def test_proprio_nan_returns_422(client):
    """v36_libero_spatial.yaml has proprio.source.total_dim == 7.

    Python's json.dumps refuses NaN by default, so we serialise with
    allow_nan=True and POST as raw bytes with explicit Content-Type.
    """
    proprio = [0.0] * 7
    proprio[3] = float("nan")
    body = {"image_primary": _b64_jpeg(), "proprio": proprio, "instruction": "x"}
    raw = json.dumps(body, allow_nan=True).encode()
    resp = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    # FastAPI / pydantic may accept NaN in list[float] via JSON parser; the
    # check happens server-side in DomainAdapter.preprocess. Either way the
    # final HTTP code must be 422.
    assert resp.status_code == 422
    detail = str(resp.json()["detail"]).lower()
    assert "non-finite" in detail or "nan" in detail or "infinite" in detail


def test_proprio_inf_returns_422(client):
    proprio = [0.0] * 7
    proprio[6] = float("inf")
    body = {"image_primary": _b64_jpeg(), "proprio": proprio, "instruction": "x"}
    raw = json.dumps(body, allow_nan=True).encode()
    resp = client.post("/predict", content=raw, headers={"Content-Type": "application/json"})
    assert resp.status_code == 422


# ----- F3b: OOD warn + hard reject (xvla_adapter mode required for norm_stats) -----
#
# We exercise _normalize_proprio directly with synthetic norm_stats rather than
# spinning up xvla_adapter mode (which requires a ckpt export dir). This keeps
# the test independent of test fixtures for the broader xvla_adapter setup.


@pytest.fixture
def adapter_with_norm():
    """Build a DomainAdapter with synthetic q01/q99 spanning [-1, +1] so that
    raw proprio values map directly to `normed` of the same magnitude.

    With q01 = [-1]*8, q99 = [+1]*8, span = 2, the normalize formula
    `2 * (x - q01) / span - 1 = x` so raw == normed. This makes the test
    threshold values readable: feeding x=1.5 produces normed=1.5.
    """
    cfg = load_deploy_config(DEPLOY_YAML)
    norm_stats = {
        "proprio": {
            "q01": [-1.0] * cfg.proprio.adapt.output_dim,
            "q99": [+1.0] * cfg.proprio.adapt.output_dim,
            "mean": [0.0] * cfg.proprio.adapt.output_dim,
            "mask": [True] * cfg.proprio.adapt.output_dim,
        },
        "action": {
            "q01": [-1.0] * cfg.ckpt.expected_action_dim,
            "q99": [+1.0] * cfg.ckpt.expected_action_dim,
            "mean": [0.0] * cfg.ckpt.expected_action_dim,
            "mask": [True] * cfg.ckpt.expected_action_dim,
        },
    }
    return DomainAdapter(cfg=cfg, norm_stats=norm_stats, domain_id=0)


def test_normalize_in_range_no_warn(adapter_with_norm, caplog):
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        out = adapter_with_norm._normalize_proprio(x)
    assert out.shape == x.shape
    assert not any("proprio_ood" in r.message for r in caplog.records)


def test_normalize_excess_1p5_emits_warn_and_clips(adapter_with_norm, caplog):
    """|normed|=1.5: above WARN threshold (1.0), below HARD threshold (10.0).
    Expected: WARNING log with event=proprio_ood, output clipped to ±1."""
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    x[2] = 1.5  # any single dim out
    with caplog.at_level(logging.WARNING):
        out = adapter_with_norm._normalize_proprio(x)
    # Clipped to ±1
    assert abs(out[2]) <= 1.0
    # Warn fired
    warns = [r for r in caplog.records if "proprio_ood" in r.message]
    assert len(warns) == 1
    payload = json.loads(warns[0].message)
    assert payload["event"] == "proprio_ood"
    assert 2 in payload["ood_dims"]
    assert payload["ood_dim_count"] >= 1


def test_normalize_excess_10_boundary_warn_only(adapter_with_norm, caplog):
    """|normed|=10.0 exactly: at HARD boundary; spec says > 10 is hard, so
    exactly 10 is warn-only (not raised)."""
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    x[0] = PROPRIO_OOD_HARD_ABS  # 10.0
    with caplog.at_level(logging.WARNING):
        out = adapter_with_norm._normalize_proprio(x)
    assert abs(out[0]) <= 1.0
    assert any("proprio_ood" in r.message for r in caplog.records)


def test_normalize_excess_above_hard_raises(adapter_with_norm, caplog):
    """|normed|=11.0: above HARD threshold (10.0). Expected: ValueError with
    'unit mismatch' message; NO proprio_ood warn (per F3 ordering)."""
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    x[5] = 11.0
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError) as exc:
            adapter_with_norm._normalize_proprio(x)
    assert "unit mismatch" in str(exc.value).lower()
    assert "5" in str(exc.value)  # dim 5 named
    # F3 ordering: hard reject suppresses the proprio_ood warn line
    assert not any("proprio_ood" in r.message for r in caplog.records)


def test_normalize_excess_just_above_hard_raises(adapter_with_norm):
    x = np.zeros(adapter_with_norm.cfg.proprio.adapt.output_dim, dtype=np.float32)
    x[0] = PROPRIO_OOD_HARD_ABS + 0.01  # 10.01
    with pytest.raises(ValueError):
        adapter_with_norm._normalize_proprio(x)
