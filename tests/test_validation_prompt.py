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
