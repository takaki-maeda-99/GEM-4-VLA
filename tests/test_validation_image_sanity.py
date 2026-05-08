"""F1: image resolution sanity bound, header-parse-first to avoid pixel-decode
allocation on absurdly-large images.

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F1.
Bound: 64 ≤ side ≤ 4096. Outside → ValueError (→ 422 at HTTP layer).

Heavy boundary cases (4096×4096) live as unit tests on _decode_jpeg_b64
directly to avoid full FastAPI round-trips for large images.
"""
import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vla_project.deployment.domain_adapter import (
    DomainAdapter,
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
)
from vla_project.deployment.inference_server import build_app

DEPLOY_YAML = "configs/deploy/v36_libero_spatial.yaml"


def _b64_jpeg_size(w: int, h: int) -> str:
    img = Image.new("RGB", (w, h), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----- Unit-level: _decode_jpeg_b64 direct call (heavy boundaries here) -----

def test_decode_32x32_rejected():
    with pytest.raises(ValueError) as exc:
        DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(32, 32))
    assert "out of sanity bound" in str(exc.value).lower()


def test_decode_5000x5000_rejected():
    with pytest.raises(ValueError) as exc:
        DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(5000, 5000))
    assert "out of sanity bound" in str(exc.value).lower()


def test_decode_64x64_boundary_accepted():
    img = DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(IMAGE_MIN_SIDE, IMAGE_MIN_SIDE))
    assert img.shape == (IMAGE_MIN_SIDE, IMAGE_MIN_SIDE, 3)


def test_decode_4096x4096_boundary_accepted():
    img = DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    assert img.shape == (IMAGE_MAX_SIDE, IMAGE_MAX_SIDE, 3)


def test_decode_4097x4097_rejected():
    with pytest.raises(ValueError) as exc:
        DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(IMAGE_MAX_SIDE + 1, IMAGE_MAX_SIDE + 1))
    assert "out of sanity bound" in str(exc.value).lower()


def test_decode_anisotropic_one_dim_too_small_rejected():
    """480×32 — width OK, height < min."""
    with pytest.raises(ValueError):
        DomainAdapter._decode_jpeg_b64(_b64_jpeg_size(480, 32))


# ----- Integration-level: full FastAPI request path -----

@pytest.fixture
def client():
    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path=DEPLOY_YAML,
        domain_id=0,
    )
    return TestClient(app)


def test_request_with_32x32_image_returns_422(client):
    body = {
        "image_primary": _b64_jpeg_size(32, 32),
        "proprio": [0.0] * 7,  # v36_libero_spatial.yaml proprio.source.total_dim
        "instruction": "x",
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 422
    assert "out of sanity bound" in str(resp.json()["detail"]).lower()


def test_request_with_224x224_image_returns_200(client):
    body = {
        "image_primary": _b64_jpeg_size(224, 224),
        "proprio": [0.0] * 7,  # v36_libero_spatial.yaml proprio.source.total_dim
        "instruction": "x",
    }
    resp = client.post("/predict", json=body)
    assert resp.status_code == 200
