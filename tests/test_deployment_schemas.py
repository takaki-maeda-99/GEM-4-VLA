"""Tests for deployment.schemas — pydantic v2 PredictRequest / PredictResponse.

Spec §Section 3 "Per-request data flow" defines the wire shape MimicRec sends.
"""
import base64
import pytest
from pydantic import ValidationError

from vla_project.deployment.schemas import PredictRequest, PredictResponse


def _b64_jpeg(n_bytes: int = 64) -> str:
    return base64.b64encode(b"\xff\xd8\xff" + b"\x00" * n_bytes).decode("ascii")


def test_predict_request_minimal_round_trip():
    """Required fields only: image_primary, proprio, instruction."""
    req = PredictRequest(
        image_primary=_b64_jpeg(),
        proprio=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5],
        instruction="pick up the bottle",
    )
    assert req.image_primary.startswith("/9j/")  # base64 JPEG marker
    assert req.image_wrist is None
    assert req.proprio == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.5]
    assert req.instruction == "pick up the bottle"
    assert req.model_version is None
    assert req.t_mono_ns is None


def test_predict_request_full_with_aliased_underscore_field():
    """The wire field is `_t_mono_ns` (underscore-prefixed) — pydantic v2 needs
    populate_by_name + alias to expose it as `t_mono_ns` on the model."""
    req = PredictRequest.model_validate({
        "image_primary": _b64_jpeg(),
        "image_wrist": _b64_jpeg(),
        "proprio": [0.0] * 7,
        "instruction": "stir the pot",
        "model_version": "x_vla_v36",
        "_t_mono_ns": {"state": 1, "image:front": 2},
    })
    assert req.image_wrist is not None
    assert req.model_version == "x_vla_v36"
    assert req.t_mono_ns == {"state": 1, "image:front": 2}


def test_predict_request_missing_image_primary_raises():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({
            "proprio": [0.0] * 7,
            "instruction": "x",
        })


def test_predict_request_proprio_must_be_list_of_numbers():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({
            "image_primary": _b64_jpeg(),
            "proprio": "not a list",
            "instruction": "x",
        })


def test_predict_request_instruction_can_be_empty_string():
    """Spec §Section 3 says empty instruction is valid in pre-start states."""
    req = PredictRequest(
        image_primary=_b64_jpeg(),
        proprio=[0.0] * 7,
        instruction="",
    )
    assert req.instruction == ""


def test_predict_response_round_trip():
    resp = PredictResponse(actions=[[0.0] * 7 for _ in range(8)])
    dumped = resp.model_dump()
    assert dumped == {"actions": [[0.0] * 7 for _ in range(8)]}


def test_predict_response_actions_must_be_list_of_lists():
    with pytest.raises(ValidationError):
        PredictResponse.model_validate({"actions": "not a list"})
