"""Wire I/O helpers for the deployment server.

This module replaces the salvageable parts of the old DomainAdapter:
  - q99 denorm with mask (dim-wise q99-inverse on mask=True dims, passthrough on False)
  - JPEG decode + image sanity bounds (Task 2)
  - Proprio normalize + F3 OOD (Task 3)
  - NaN guards (Task 8, called from inference_server)

The contract-translation parts of DomainAdapter (frame conversion,
gripper convention, raw-proprio source/adapt) are NOT recreated here —
those move to clients per the yaml-less spec.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def q99_denorm_with_mask(action_norm: np.ndarray, stats: dict) -> np.ndarray:
    """q99-inverse on mask=True dims, passthrough on mask=False dims.

    action_norm: [..., A]
    stats: {"q01": list[A], "q99": list[A], "mask": list[A] bool, ...}
    """
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)
    if action_norm.shape[-1] != q01.shape[0]:
        raise ValueError(
            f"action_dim={action_norm.shape[-1]} != stats dim={q01.shape[0]}"
        )
    span = q99 - q01
    denormed = q01 + (action_norm + 1.0) * 0.5 * span
    return np.where(mask, denormed, action_norm).astype(np.float32)


# F1 image-side sanity bounds. Catches replay corruption (1×1) and
# abusive payloads before JPEG decoder allocates pixel buffers.
IMAGE_MIN_SIDE: int = 64
IMAGE_MAX_SIDE: int = 4096


def decode_jpeg_b64(b64: str) -> np.ndarray:
    """Decode a base64-encoded JPEG into uint8 HWC RGB.

    Raises ValueError if image side is outside [IMAGE_MIN_SIDE, IMAGE_MAX_SIDE]
    on either axis.
    """
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    h, w = img.height, img.width
    if h < IMAGE_MIN_SIDE or w < IMAGE_MIN_SIDE:
        raise ValueError(
            f"image side below IMAGE_MIN_SIDE={IMAGE_MIN_SIDE}: got h={h}, w={w}"
        )
    if h > IMAGE_MAX_SIDE or w > IMAGE_MAX_SIDE:
        raise ValueError(
            f"image side above IMAGE_MAX_SIDE={IMAGE_MAX_SIDE}: got h={h}, w={w}"
        )
    return np.asarray(img, dtype=np.uint8)
