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
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("vla_project.deployment.wire_io")


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
    # F1: header-parse-first. Image.open() reads only the JPEG header
    # (no pixel decode); .size returns (W, H) from the header. We bound
    # the dimensions before convert("RGB") forces full pixel decode,
    # so an attacker / corrupt payload can't allocate gigabytes via
    # an oversized header.
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    if h < IMAGE_MIN_SIDE or w < IMAGE_MIN_SIDE:
        raise ValueError(
            f"image side below IMAGE_MIN_SIDE={IMAGE_MIN_SIDE}: got h={h}, w={w}"
        )
    if h > IMAGE_MAX_SIDE or w > IMAGE_MAX_SIDE:
        raise ValueError(
            f"image side above IMAGE_MAX_SIDE={IMAGE_MAX_SIDE}: got h={h}, w={w}"
        )
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# F3 proprio OOD thresholds. Computed against normalized values
# (after q01/q99 mapping). Values that exceed PROPRIO_OOD_WARN_ABS but
# stay under HARD are clipped + warned. Values above HARD raise.
# Calibration: 10x the q-range catches deg/rad swap; 1x is the soft
# OOD warning for legitimate startup poses.
PROPRIO_OOD_WARN_ABS: float = 1.0
PROPRIO_OOD_HARD_ABS: float = 10.0


def normalize_proprio_q99(
    proprio_raw: np.ndarray, stats: dict
) -> tuple[np.ndarray, bool]:
    """Normalize raw proprio via q99 to [~ -1, +1] with passthrough on mask=False.

    Returns (normalized, warned). `warned` is True if any masked dim's
    |normed| exceeded PROPRIO_OOD_WARN_ABS (and was clipped). Raises
    ValueError if any value is non-finite or any masked dim's |normed|
    exceeds PROPRIO_OOD_HARD_ABS.
    """
    if not np.isfinite(proprio_raw).all():
        bad_dims = np.where(~np.isfinite(proprio_raw))[0].tolist()
        raise ValueError(f"proprio contains non-finite values at dims {bad_dims}")
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    mask = np.asarray(stats["mask"], dtype=bool)
    span = q99 - q01
    # F3-aware normalize: protect from div-by-zero on degenerate masked dims.
    safe_span = np.where(span > 0, span, 1.0)
    normed = 2.0 * (proprio_raw - q01) / safe_span - 1.0
    # Passthrough on mask=False dims (gripper passthrough at training time).
    normed = np.where(mask, normed, proprio_raw)
    abs_normed = np.abs(normed)
    hard_viol = (abs_normed > PROPRIO_OOD_HARD_ABS) & mask
    if hard_viol.any():
        bad = np.where(hard_viol)[0].tolist()
        raise ValueError(
            f"proprio normalized |x|>{PROPRIO_OOD_HARD_ABS} (hard) at dims {bad}; "
            f"likely deg/rad swap or wrong proprio dim"
        )
    warn_viol = (abs_normed > PROPRIO_OOD_WARN_ABS) & mask
    warned = bool(warn_viol.any())
    if warned:
        bad = np.where(warn_viol)[0].tolist()
        logger.warning(
            f"proprio normalized |x|>{PROPRIO_OOD_WARN_ABS} (warn) at dims {bad}; clipping"
        )
        normed = np.where(
            mask,
            np.clip(normed, -PROPRIO_OOD_WARN_ABS, PROPRIO_OOD_WARN_ABS),
            normed,
        )
    return normed.astype(np.float32), warned
