"""Post-process gripper normalization for ReBotArm-style continuous-gripper FT
models (bottle / cookie / jar etc).

The training datasets normalize action dims 0-5 (EE-delta) to [-1, 1] via q01
/ q99 (mask=True) but leave action dim 6 (gripper) as raw passthrough
(mask=False, divided by ``_GRIPPER_DIVISOR=100`` in the dataset class). At
deploy time we want a [0, 1] gripper signal where ``raw_min → 0`` (closed)
and ``raw_max → 1`` (open); this module provides that post-process.

Two equivalent shapes:

- Functions ``normalize_gripper`` / ``denormalize_gripper``: stateless, numpy
  or torch tensor in, same type out.
- ``GripperNormalizer`` nn.Module: same logic, drop-in for pipelines that
  expect a Module.

Default range ``[-6.0, 0.0]`` matches the bottle / cookie observed gripper
raw distribution (the dataset's gripper_pos divided by 100 is roughly in
``[-0.06, 0.00]``, i.e. raw gripper_pos in ``[-6, 0]``). Override via
constructor / function args per-FT.

Usage:

    from vla_project.deployment.gripper_normalizer import (
        normalize_gripper, denormalize_gripper, GripperNormalizer,
    )

    # pred[..., 6] from policy: passthrough = gripper_pos / 100, e.g. -0.043.
    # We multiply ×100 inside to map back to raw gripper_pos units before
    # normalizing.
    grip_norm = normalize_gripper(pred[..., 6], raw_min=-6.0, raw_max=0.0)
    # → in [0, 1] (clipped, 0=closed, 1=open)

    # Inverse, e.g. to convert a user-supplied normalized target back to raw:
    grip_raw_div100 = denormalize_gripper(grip_norm, raw_min=-6.0, raw_max=0.0)
    # × 100 to get gripper_pos for the robot.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import torch
import torch.nn as nn

_GRIPPER_DIVISOR = 100.0   # matches LeRobotSO101Dataset._GRIPPER_DIVISOR
_DEFAULT_RAW_MIN = -6.0    # raw gripper_pos lower bound  → norm 0 (closed)
_DEFAULT_RAW_MAX = 0.0     # raw gripper_pos upper bound  → norm 1 (open)
_EPS = 1e-8

ArrayLike = Union[np.ndarray, torch.Tensor]


def _clamp(x: ArrayLike, lo: float, hi: float) -> ArrayLike:
    if isinstance(x, torch.Tensor):
        return x.clamp(lo, hi)
    return np.clip(x, lo, hi)


def normalize_gripper(
    pred_grip: ArrayLike,
    raw_min: float = _DEFAULT_RAW_MIN,
    raw_max: float = _DEFAULT_RAW_MAX,
) -> ArrayLike:
    """Map a model-output gripper value (in /100 scale) to [0, 1] using
    ``[raw_min, raw_max]`` as the raw-unit bounds.

    Args:
        pred_grip: tensor or ndarray of any shape. Expected to be the raw
            gripper output from the policy (i.e. before any ×100 conversion;
            value is in ``gripper_pos / _GRIPPER_DIVISOR`` scale).
        raw_min, raw_max: raw gripper_pos bounds (NOT divided by 100). The
            function applies ``×100`` internally before normalizing.
            ``raw_min`` maps to 0 (closed), ``raw_max`` maps to 1 (open).

    Returns:
        Tensor / ndarray of the same shape with values clipped to [0, 1].
    """
    if raw_max <= raw_min:
        raise ValueError(f"raw_max ({raw_max}) must be > raw_min ({raw_min})")
    width = raw_max - raw_min
    raw = pred_grip * _GRIPPER_DIVISOR
    norm = (raw - raw_min) / max(width, _EPS)
    return _clamp(norm, 0.0, 1.0)


def denormalize_gripper(
    norm_grip: ArrayLike,
    raw_min: float = _DEFAULT_RAW_MIN,
    raw_max: float = _DEFAULT_RAW_MAX,
) -> ArrayLike:
    """Inverse of :func:`normalize_gripper`. Maps a [0, 1] normalized value
    back to the policy-output scale (gripper_pos / 100), NOT raw gripper_pos.
    Multiply by 100 to get the raw gripper_pos command for the robot.
    """
    if raw_max <= raw_min:
        raise ValueError(f"raw_max ({raw_max}) must be > raw_min ({raw_min})")
    width = raw_max - raw_min
    raw = norm_grip * width + raw_min
    return raw / _GRIPPER_DIVISOR


class GripperNormalizer(nn.Module):
    """nn.Module wrapper around :func:`normalize_gripper`.

    Stores ``raw_min`` / ``raw_max`` as buffers so the bounds travel with
    the module via ``state_dict`` / ``torch.save``.

    Output is in ``[0, 1]`` with ``raw_min → 0`` (closed) and
    ``raw_max → 1`` (open).

    Example:
        norm = GripperNormalizer(raw_min=-6.0, raw_max=0.0)
        pred_action = policy(batch)  # (B, T, 7) — gripper at index 6
        pred_action[..., 6] = norm(pred_action[..., 6])
    """

    def __init__(
        self,
        raw_min: float = _DEFAULT_RAW_MIN,
        raw_max: float = _DEFAULT_RAW_MAX,
    ) -> None:
        super().__init__()
        if raw_max <= raw_min:
            raise ValueError(f"raw_max ({raw_max}) must be > raw_min ({raw_min})")
        self.register_buffer("raw_min", torch.tensor(float(raw_min)))
        self.register_buffer("raw_max", torch.tensor(float(raw_max)))

    def forward(self, pred_grip: torch.Tensor) -> torch.Tensor:
        width = (self.raw_max - self.raw_min).clamp_min(_EPS)
        raw = pred_grip * _GRIPPER_DIVISOR
        norm = (raw - self.raw_min) / width
        return norm.clamp(0.0, 1.0)

    def inverse(self, norm_grip: torch.Tensor) -> torch.Tensor:
        """[0, 1] → policy-scale gripper (/100). ×100 for robot command."""
        width = self.raw_max - self.raw_min
        raw = norm_grip * width + self.raw_min
        return raw / _GRIPPER_DIVISOR

    def extra_repr(self) -> str:
        return f"raw_min={float(self.raw_min):.3f}, raw_max={float(self.raw_max):.3f}"
