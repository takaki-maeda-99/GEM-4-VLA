"""JPEG decode + F1 image-side sanity (min=64, max=4096)."""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from vla_project.deployment.wire_io import (
    IMAGE_MAX_SIDE,
    IMAGE_MIN_SIDE,
    decode_jpeg_b64,
)


def _b64_jpeg(h: int, w: int) -> str:
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_decode_returns_uint8_hwc_rgb():
    arr = decode_jpeg_b64(_b64_jpeg(IMAGE_MIN_SIDE, IMAGE_MIN_SIDE))
    assert arr.dtype == np.uint8
    assert arr.shape == (IMAGE_MIN_SIDE, IMAGE_MIN_SIDE, 3)


def test_decode_min_side_passes():
    decode_jpeg_b64(_b64_jpeg(IMAGE_MIN_SIDE, IMAGE_MIN_SIDE))


def test_decode_max_side_passes():
    decode_jpeg_b64(_b64_jpeg(IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))


def test_decode_below_min_side_rejects():
    with pytest.raises(ValueError, match="below"):
        decode_jpeg_b64(_b64_jpeg(32, 32))


def test_decode_above_max_side_rejects():
    with pytest.raises(ValueError, match="above"):
        decode_jpeg_b64(_b64_jpeg(IMAGE_MAX_SIDE + 1, IMAGE_MAX_SIDE + 1))


def test_decode_lopsided_rejects():
    with pytest.raises(ValueError):
        decode_jpeg_b64(_b64_jpeg(480, 32))  # 32 < min_side
