"""F2: instruction byte-length sanity (Phase 0 — pydantic-only).

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F2.
Empty allowed (matches Phase 0 spec L153 'may be empty in pre-start states').
Upper bound 10 000 UTF-8 bytes.
"""
import base64

import pytest
from pydantic import ValidationError

from vla_project.deployment.schemas import PredictRequest


def _b64_jpeg(n_bytes: int = 64) -> str:
    return base64.b64encode(b"\xff\xd8\xff" + b"\x00" * n_bytes).decode("ascii")


def _kwargs(instruction: str) -> dict:
    return dict(
        image_primary=_b64_jpeg(),
        proprio=[0.0] * 7,
        instruction=instruction,
    )


def test_instruction_empty_string_is_valid():
    """Phase 0 spec L153: instruction may be empty in pre-start states."""
    req = PredictRequest(**_kwargs(""))
    assert req.instruction == ""


def test_instruction_short_ascii_is_valid():
    req = PredictRequest(**_kwargs("pick up the bottle"))
    assert req.instruction == "pick up the bottle"


def test_instruction_at_byte_limit_is_valid():
    """10 000 ASCII bytes is the boundary — must accept."""
    s = "a" * 10_000
    req = PredictRequest(**_kwargs(s))
    assert len(req.instruction.encode("utf-8")) == 10_000


def test_instruction_over_byte_limit_is_rejected():
    s = "a" * 10_001
    with pytest.raises(ValidationError) as exc:
        PredictRequest(**_kwargs(s))
    assert "byte length" in str(exc.value).lower()


def test_instruction_multibyte_utf8_counted_in_bytes_not_chars():
    """Japanese 'あ' = 3 bytes in UTF-8. 5000 chars × 3 = 15000 bytes > 10000."""
    s = "あ" * 5000
    with pytest.raises(ValidationError) as exc:
        PredictRequest(**_kwargs(s))
    assert "byte length" in str(exc.value).lower()


def test_instruction_just_under_byte_limit_is_valid():
    """9 999 ASCII bytes — boundary minus one."""
    s = "a" * 9_999
    req = PredictRequest(**_kwargs(s))
    assert len(req.instruction.encode("utf-8")) == 9_999


def test_instruction_over_byte_limit_returns_422_via_http():
    """End-to-end: a >10 000-byte instruction must return HTTP 422 (NOT 500
    from a JSON-serialization crash in the RequestValidationError handler).

    This test exists specifically because pydantic v2's `exc.errors()` includes
    a `ctx: {'error': ValueError(...)}` field that Starlette's default JSON
    encoder cannot serialize. The fix in inference_server.py uses
    `fastapi.encoders.jsonable_encoder` to convert the exception to a string.
    """
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from vla_project.deployment.inference_server import build_app

    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    app = build_app(
        predictor_kind="hold_position",
        checkpoint=None,
        deploy_config_path="configs/deploy/v36_libero_spatial.yaml",
        domain_id=0,
    )
    c = TestClient(app)
    body = {
        "image_primary": b64,
        "proprio": [0.0] * 7,
        "instruction": "a" * 10_001,
    }
    resp = c.post("/predict", json=body)
    assert resp.status_code == 422
    detail_str = str(resp.json()["detail"]).lower()
    assert "byte length" in detail_str
