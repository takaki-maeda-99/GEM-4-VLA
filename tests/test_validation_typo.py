"""F4: wire-typo guard preserving extra='ignore' forward-compat.

Spec: docs/superpowers/specs/2026-05-08-server-request-validation-design.md §F4.
Two-rule guard:
  1. Damerau-Levenshtein distance ≤ 1 from a modeled field name → 'did you mean X?'
  2. image_* prefix that didn't trip rule 1 → 'unknown image field; known: ...'
  Otherwise: silently ignored (preserves extra='ignore' for future _trace etc.).
"""
import base64

import pytest
from pydantic import ValidationError

from vla_project.deployment.schemas import PredictRequest


def _b64_jpeg(n_bytes: int = 64) -> str:
    return base64.b64encode(b"\xff\xd8\xff" + b"\x00" * n_bytes).decode("ascii")


def _base_kwargs() -> dict:
    return dict(
        image_primary=_b64_jpeg(),
        proprio=[0.0] * 7,
        instruction="test",
    )


# ----- Rule 1: near-miss (Damerau-Levenshtein ≤ 1) -----

def test_typo_image_pirmary_caught_as_near_miss():
    """Transposition: image_pirmary (swap r-i) → distance 1 from image_primary."""
    raw = _base_kwargs()
    raw["image_pirmary"] = _b64_jpeg()
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "image_pirmary" in msg
    assert "image_primary" in msg
    assert "did you mean" in msg.lower()


def test_typo_image_wirst_caught_as_near_miss():
    """Transposition: image_wirst (swap r-i) → distance 1 from image_wrist."""
    raw = _base_kwargs()
    raw["image_wirst"] = _b64_jpeg()
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "image_wirst" in msg
    assert "image_wrist" in msg


def test_typo_propio_caught_as_near_miss():
    """Deletion: propio (drop r) → distance 1 from proprio."""
    raw = _base_kwargs()
    raw["propio"] = [0.0] * 7
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "propio" in msg
    assert "proprio" in msg


def test_typo_t_mono_n_caught_as_near_miss():
    """Deletion: _t_mono_n (drop trailing s) → distance 1 from _t_mono_ns."""
    raw = _base_kwargs()
    raw["_t_mono_n"] = {"state": 1}
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "_t_mono_n" in msg
    assert "_t_mono_ns" in msg


def test_typo_model_versionn_caught_as_near_miss():
    """Insertion: model_versionn (extra n) → distance 1 from model_version."""
    raw = _base_kwargs()
    raw["model_versionn"] = "x_vla_v36"
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "model_versionn" in msg
    assert "model_version" in msg


# ----- Rule 2: image_* prefix without near miss -----

def test_image_camera_left_caught_by_image_prefix_rule():
    """image_camera_left is not within distance 1 of any modeled field; falls
    through to the image_* prefix rule."""
    raw = _base_kwargs()
    raw["image_camera_left"] = _b64_jpeg()
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "image_camera_left" in msg
    assert "unknown image field" in msg.lower()
    assert "image_primary" in msg
    assert "image_wrist" in msg


# ----- Forward-compat: silently-ignored unknown fields -----

def test_underscore_request_id_silently_ignored():
    """Forward-compat: future MimicRec observability fields (_request_id, _trace,
    _session_token) must pass through. Distance from any modeled field > 1."""
    raw = _base_kwargs()
    raw["_request_id"] = "abc-123"
    req = PredictRequest.model_validate(raw)
    assert req.image_primary  # accepted


def test_underscore_trace_silently_ignored():
    raw = _base_kwargs()
    raw["_trace"] = {"span": "x"}
    req = PredictRequest.model_validate(raw)
    assert req.image_primary


def test_unrelated_unknown_field_silently_ignored():
    """A field with no edit-distance and no image_ prefix passes silently."""
    raw = _base_kwargs()
    raw["unrelated_metadata_field"] = 42
    req = PredictRequest.model_validate(raw)
    assert req.image_primary


# ----- Regression: populate_by_name=True + typo guard contract -----

def test_underscored_t_mono_ns_alias_accepted_via_wire():
    """Regression: `_t_mono_ns` (the wire alias) is accepted as-is."""
    raw = _base_kwargs()
    raw["_t_mono_ns"] = {"state": 1}
    req = PredictRequest.model_validate(raw)
    assert req.t_mono_ns == {"state": 1}


def test_t_mono_ns_python_name_accepted_via_populate_by_name():
    """Regression: `populate_by_name=True` + the typo guard must agree —
    the Python attribute name `t_mono_ns` (no underscore) must be accepted
    too. This was Critical reviewer feedback after the typo guard initially
    rejected this case as a near-miss of `_t_mono_ns`."""
    raw = _base_kwargs()
    raw["t_mono_ns"] = {"state": 1}
    req = PredictRequest.model_validate(raw)
    assert req.t_mono_ns == {"state": 1}


# ----- Rule 1: substitution edit type -----

def test_typo_substitution_caught_as_near_miss():
    """Substitution: instrxction (x for u) → distance 1 from instruction."""
    raw = _base_kwargs()
    raw["instrxction"] = "test"
    raw.pop("instruction")  # avoid duplicate
    with pytest.raises(ValidationError) as exc:
        PredictRequest.model_validate(raw)
    msg = str(exc.value)
    assert "instrxction" in msg
    assert "instruction" in msg


# ----- Forward-compat: _session_token -----

def test_underscore_session_token_silently_ignored():
    """Forward-compat: `_session_token` is one of the three explicitly named
    MimicRec observability fields the spec promises to ignore."""
    raw = _base_kwargs()
    raw["_session_token"] = "tok-abc"
    req = PredictRequest.model_validate(raw)
    assert req.image_primary
