"""Pydantic v2 wire schemas for the inference HTTP server.

PredictRequest mirrors what MimicRec sends per its contract YAML; PredictResponse
is what the server returns. Field naming follows the MimicRec spec excerpts in
docs/superpowers/specs/2026-05-06-vla-inference-server-design.md §Section 3.

The wire field `_t_mono_ns` is exposed as `t_mono_ns` on the model because
pydantic v2 reserves leading-underscore names as private attributes; we use
`populate_by_name=True` + `Field(alias="_t_mono_ns")`.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_primary: str
    image_wrist: str | None = None
    proprio: list[float]
    instruction: str
    model_version: str | None = None
    t_mono_ns: dict[str, Any] | None = Field(default=None, alias="_t_mono_ns")


class PredictResponse(BaseModel):
    actions: list[list[float]]
