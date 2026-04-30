"""EE6D loss split into per-channel L1 contributions.

The X-VLA 20-dim layout is ``[xyz(3), rot6d(6), gripper(1), pad(10)]``.
Each channel group has different units (xyz in meters, rot6d in [-1, 1],
gripper qpos in [0, 0.04]m), so summing them with equal L1 weight is
already a reasonable per-channel-mean balance — but we keep them separate
in :func:`ee6d_loss_components` so they can be logged individually.

Padding slots [10:20] reserve the bimanual right-arm slots; for single-arm
LIBERO the target is always zero and we exclude them from the loss to
avoid an irrelevant "predict zero" gradient.

Internal accumulation is fp32 even for bf16 inputs: bf16 sums underflow
on small magnitudes (matches the convention in :mod:`losses`).
"""
from __future__ import annotations

from typing import Dict

import torch


_POS = slice(0, 3)
_ROT = slice(3, 9)
_GRIP = slice(9, 10)


def _masked_l1(
    pred: torch.Tensor, target: torch.Tensor, mask_f: torch.Tensor, slc: slice
) -> torch.Tensor:
    diff = (pred[..., slc] - target[..., slc]).abs() * mask_f
    n_chan = slc.stop - slc.start
    denom = mask_f.sum() * n_chan
    return diff.sum() / denom.clamp_min(1.0)


def ee6d_loss_components(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Split L1 loss by channel group. Padding [10:20] is dropped.

    Args:
        pred:   ``(B, T, A)`` model output. ``A >= 10``.
        target: ``(B, T, A)`` ground truth.
        mask:   ``(B, T)`` bool — True means the timestep contributes.

    Returns:
        Dict with keys ``'pos'``, ``'rot'``, ``'grip'``. Each value is a
        scalar fp32 tensor (mean abs error within the group, masked
        timesteps only). Pad slots are not included anywhere.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target {tuple(target.shape)}")
    if pred.shape[-1] < 10:
        raise ValueError(
            f"pred last dim must be >= 10 (EE6D layout); got {pred.shape[-1]}"
        )

    pred_f = pred.to(torch.float32)
    target_f = target.to(torch.float32)
    mask_f = mask.to(torch.float32).unsqueeze(-1)  # (B, T, 1)
    return {
        "pos":  _masked_l1(pred_f, target_f, mask_f, _POS),
        "rot":  _masked_l1(pred_f, target_f, mask_f, _ROT),
        "grip": _masked_l1(pred_f, target_f, mask_f, _GRIP),
    }


def ee6d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    w_pos: float = 1.0,
    w_rot: float = 1.0,
    w_grip: float = 1.0,
) -> torch.Tensor:
    """Total EE6D loss = ``w_pos*pos + w_rot*rot + w_grip*grip``.

    Each per-channel term is the masked L1 mean within its slot range.
    Padding slots are ignored. See :func:`ee6d_loss_components` for the
    per-channel breakdown.
    """
    c = ee6d_loss_components(pred, target, mask)
    return w_pos * c["pos"] + w_rot * c["rot"] + w_grip * c["grip"]
