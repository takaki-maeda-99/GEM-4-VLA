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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# F2: instruction byte-length sanity (UTF-8 byte count, NOT char count — pydantic's
# native max_length constrains chars, but multibyte UTF-8 means 10 000 chars can
# be 30 000+ bytes for Japanese / emoji).
INSTRUCTION_MAX_BYTES: int = 10_000


# F4: typo guard — fields the wire schema models. Used by the model_validator
# below to compute Damerau-Levenshtein distance for near-miss detection.
_MODELED_FIELDS: frozenset[str] = frozenset({
    "image_primary", "image_wrist",
    "proprio", "instruction",
    "model_version",
    # Both forms are accepted under populate_by_name=True: the wire alias
    # "_t_mono_ns" (per MimicRec contract) and the Python attribute name
    # "t_mono_ns". Both must be in the set so the typo guard does not
    # reject the in-Python construction path.
    "_t_mono_ns", "t_mono_ns",
})


def _damerau_levenshtein_within_one(a: str, b: str) -> bool:
    """Return True iff Damerau-Levenshtein distance(a, b) ≤ 1.

    Bounded check (we only care about distance ≤ 1), so this avoids the
    full DP matrix. Catches:
      - 0 edits (a == b)
      - 1 substitution
      - 1 insertion
      - 1 deletion
      - 1 transposition of adjacent characters (this is the Damerau extension
        — typical typos like 'pirmary' vs 'primary' are transpositions, which
        plain Levenshtein counts as distance 2.)
    """
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    # Substitution / transposition (la == lb)
    if la == lb:
        diffs = [i for i in range(la) if a[i] != b[i]]
        if len(diffs) == 1:
            return True  # single substitution
        if len(diffs) == 2 and diffs[0] + 1 == diffs[1]:
            i, j = diffs
            if a[i] == b[j] and a[j] == b[i]:
                return True  # adjacent transposition
        return False
    # Insertion / deletion: pin the longer string as `lo` (long), shorter as `sh`.
    lo, sh = (a, b) if la > lb else (b, a)
    # Try to find a single skip in `lo` that makes them equal.
    i = j = 0
    skipped = False
    while i < len(lo) and j < len(sh):
        if lo[i] == sh[j]:
            i += 1
            j += 1
        elif not skipped:
            i += 1
            skipped = True
        else:
            return False
    return True


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    image_primary: str
    image_wrist: str | None = None
    proprio: list[float]
    instruction: str
    model_version: str | None = None
    t_mono_ns: dict[str, Any] | None = Field(default=None, alias="_t_mono_ns")

    @model_validator(mode="before")
    @classmethod
    def _typo_guard(cls, data: Any) -> Any:
        """F4: catch wire-level typos before extra='ignore' silently drops them.

        Two ordered rules; first match wins:
          1. Near-miss: any unknown key within Damerau-Levenshtein 1 of a
             modeled field → ValueError('did you mean X?').
          2. Image-prefix: any unknown key starting with 'image_' that didn't
             trip rule 1 → ValueError('unknown image field; known: ...').
        Other unknowns pass through and are dropped by extra='ignore'.
        """
        if not isinstance(data, dict):
            return data  # let pydantic handle non-dict inputs naturally
        for key in list(data.keys()):
            if key in _MODELED_FIELDS:
                continue
            # Rule 1: near-miss
            for modeled in _MODELED_FIELDS:
                if _damerau_levenshtein_within_one(key, modeled):
                    raise ValueError(
                        f"unknown field {key!r}; did you mean {modeled!r}?"
                    )
            # Rule 2: image_* prefix fallback
            if key.startswith("image_"):
                raise ValueError(
                    f"unknown image field {key!r}; "
                    f"known: image_primary, image_wrist"
                )
        return data

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
