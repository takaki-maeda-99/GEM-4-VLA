"""Pydantic v2 wire schemas for the inference HTTP server.

PredictRequest mirrors what MimicRec sends per its contract YAML; PredictResponse
is what the server returns. Field naming follows the MimicRec spec excerpts in
docs/superpowers/specs/2026-05-06-vla-inference-server-design.md §Section 3.

The wire field `_t_mono_ns` is exposed as `t_mono_ns` on the model because
pydantic v2 reserves leading-underscore names as private attributes; we use
`populate_by_name=True` + `Field(alias="_t_mono_ns")`.

Validation features (per docs/superpowers/specs/2026-05-08-server-request-
validation-design.md):
  - F2: instruction must be ≤ INSTRUCTION_MAX_BYTES UTF-8 bytes (empty allowed).
  - F4: typo guard added in a later step (model_validator).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# F2: instruction byte-length sanity (UTF-8 byte count, NOT char count — pydantic's
# native max_length constrains chars, but multibyte UTF-8 means 10 000 chars can
# be 30 000+ bytes for Japanese / emoji).
INSTRUCTION_MAX_BYTES: int = 10_000


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_primary: str
    image_wrist: str | None = None
    proprio: list[float]
    instruction: str
    model_version: str | None = None
    t_mono_ns: dict[str, Any] | None = Field(default=None, alias="_t_mono_ns")

    @field_validator("instruction")
    @classmethod
    def _instruction_byte_length(cls, v: str) -> str:
        n = len(v.encode("utf-8"))
        if n > INSTRUCTION_MAX_BYTES:
            raise ValueError(
                f"instruction byte length {n} > {INSTRUCTION_MAX_BYTES} "
                f"(UTF-8 byte count, not char count)"
            )
        return v


class PredictResponse(BaseModel):
    actions: list[list[float]]
