"""X-VLA action alignment math: quat/rot6d conversions + 20-dim EE6D encoding.

The 20-dim X-VLA action layout is::

    [xyz (3)] + [rot6d (6)] + [gripper (1)] + [zero padding (10)] = 20

- xyz: absolute end-effector position in robot base frame
- rot6d: first two columns of rotation matrix (Zhou et al. 2019)
- gripper: scalar in {0, 1}
- padding: reserved for the bimanual case (right-arm xyz+rot6d).
  Single-arm fills with zeros.

Quaternion convention here: ``(qx, qy, qz, qw)`` (scalar-last), matching
LIBERO / robosuite ``robot0_eef_quat``.
"""
from __future__ import annotations

from typing import Tuple

import torch


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Scalar-last quaternion (..., 4) → rotation matrix (..., 3, 3)."""
    if quat.shape[-1] != 4:
        raise ValueError(f"quat must have last dim 4 (xyzw); got {tuple(quat.shape)}")
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(dim=-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz),  2 * (xy - wz),      2 * (xz + wy),
        2 * (xy + wz),      1 - 2 * (xx + zz),  2 * (yz - wx),
        2 * (xz - wy),      2 * (yz + wx),      1 - 2 * (xx + yy),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def quat_to_rot6d(quat: torch.Tensor) -> torch.Tensor:
    """Scalar-last quaternion (..., 4) → rot6d (..., 6).

    rot6d = first two columns of the rotation matrix, concatenated:
    ``[c1[0], c1[1], c1[2], c2[0], c2[1], c2[2]]`` (column-major flatten).
    Matching :func:`rot6d_to_matrix`'s ``a1 = rot6d[0:3]`` /
    ``a2 = rot6d[3:6]`` interpretation. Naive ``R[:, :2].reshape(6)`` is
    row-major and interleaves the two columns — do NOT use that.
    """
    R = quat_to_matrix(quat)  # (..., 3, 3)
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """rot6d (..., 6) → orthonormalized rotation matrix (..., 3, 3).

    Procedure (Zhou et al. 2019):
      a1 = rot6d[..., 0:3];  b1 = a1 / ||a1||
      a2 = rot6d[..., 3:6];  b2 = a2 - (a2·b1) b1; b2 /= ||b2||
      b3 = b1 × b2
      R = [b1 | b2 | b3]
    Robust to small noise: `b1` and `b2` are renormalized.
    """
    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must have last dim 6; got {tuple(rot6d.shape)}")
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a2_proj = (a2 * b1).sum(dim=-1, keepdim=True) * b1
    b2_raw = a2 - a2_proj
    b2 = b2_raw / b2_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3) with columns b1/b2/b3


def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) → scalar-last quaternion (..., 4).

    Uses the Shepperd / sign-stable variant: pick the largest among
    {1+R00+R11+R22, 1+R00-R11-R22, 1-R00+R11-R22, 1-R00-R11+R22}.
    """
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R must end in (3, 3); got {tuple(R.shape)}")
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = R00 + R11 + R22

    # 4 candidate "magnitude squared" terms; take the max for numerical stability.
    t0 = 1.0 + trace
    t1 = 1.0 + R00 - R11 - R22
    t2 = 1.0 - R00 + R11 - R22
    t3 = 1.0 - R00 - R11 + R22
    cands = torch.stack([t0, t1, t2, t3], dim=-1)  # (..., 4)
    idx = cands.argmax(dim=-1)                     # (...,)

    sqrt_t = (cands.gather(-1, idx.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)) ** 0.5  # (...,)
    inv = 0.5 / sqrt_t  # (...,)

    # Build quaternion components per case.
    qw0 = 0.5 * sqrt_t
    qx0 = (R21 - R12) * inv
    qy0 = (R02 - R20) * inv
    qz0 = (R10 - R01) * inv

    qx1 = 0.5 * sqrt_t
    qw1 = (R21 - R12) * inv
    qy1 = (R10 + R01) * inv
    qz1 = (R02 + R20) * inv

    qy2 = 0.5 * sqrt_t
    qw2 = (R02 - R20) * inv
    qx2 = (R10 + R01) * inv
    qz2 = (R21 + R12) * inv

    qz3 = 0.5 * sqrt_t
    qw3 = (R10 - R01) * inv
    qx3 = (R02 + R20) * inv
    qy3 = (R21 + R12) * inv

    qw = torch.where(idx == 0, qw0, torch.where(idx == 1, qw1, torch.where(idx == 2, qw2, qw3)))
    qx = torch.where(idx == 0, qx0, torch.where(idx == 1, qx1, torch.where(idx == 2, qx2, qx3)))
    qy = torch.where(idx == 0, qy0, torch.where(idx == 1, qy1, torch.where(idx == 2, qy2, qy3)))
    qz = torch.where(idx == 0, qz0, torch.where(idx == 1, qz1, torch.where(idx == 2, qz2, qz3)))

    q = torch.stack([qx, qy, qz, qw], dim=-1)  # scalar-last
    # Canonicalize sign so qw >= 0 (q and -q represent the same rotation).
    q = torch.where(q[..., 3:4] < 0, -q, q)
    return q


def rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
    """rot6d (..., 6) → scalar-last quaternion (..., 4) via matrix."""
    return matrix_to_quat(rot6d_to_matrix(rot6d))


def ee_pose_to_action20(
    pos: torch.Tensor,
    quat: torch.Tensor,
    gripper: torch.Tensor,
) -> torch.Tensor:
    """Pack (xyz, quat, gripper) into the X-VLA 20-dim layout.

    Args:
        pos: (..., 3) end-effector position in base frame.
        quat: (..., 4) scalar-last quaternion of the end-effector rotation.
        gripper: (..., 1) or (...,) scalar in [0, 1].

    Returns:
        (..., 20) tensor with layout
        ``[xyz | rot6d | gripper | 10 zero-pad]``. dtype follows ``pos``.
    """
    if pos.shape[-1] != 3:
        raise ValueError(f"pos last dim must be 3; got {tuple(pos.shape)}")
    rot6d = quat_to_rot6d(quat).to(pos.dtype)
    if gripper.dim() == pos.dim() - 1:
        gripper = gripper.unsqueeze(-1)
    if gripper.shape[-1] != 1:
        raise ValueError(f"gripper last dim must be 1; got {tuple(gripper.shape)}")
    pad_shape = list(pos.shape[:-1]) + [10]
    pad = torch.zeros(pad_shape, dtype=pos.dtype, device=pos.device)
    return torch.cat([pos, rot6d, gripper.to(pos.dtype), pad], dim=-1)


def action20_to_ee_pose(action20: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`ee_pose_to_action20`. Padding is dropped.

    Returns:
        (pos (..., 3), quat (..., 4) scalar-last, gripper (..., 1))
    """
    if action20.shape[-1] != 20:
        raise ValueError(f"action20 last dim must be 20; got {tuple(action20.shape)}")
    pos = action20[..., 0:3]
    rot6d = action20[..., 3:9]
    gripper = action20[..., 9:10]
    quat = rot6d_to_quat(rot6d)
    return pos, quat, gripper


def anchor_offsets(window_seconds: float, num_anchors: int, fps: int) -> list:
    """Evenly-spaced anchor offsets in seconds, [0, window_seconds].

    Used as ``delta_timestamps`` for LeRobot to fetch proprio at anchor
    times. Spacing = window_seconds / (num_anchors - 1). For
    LIBERO-style fps=10, window=4, num=30 the spacing is non-integer
    (≈0.138 s ≈ 1.38 frames); LeRobot picks the nearest frame.

    fps is currently informational (not used to round). Included so
    callers can compute integer-frame variants externally if needed.
    """
    if num_anchors < 2:
        raise ValueError(f"num_anchors must be >= 2; got {num_anchors}")
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0; got {window_seconds}")
    if fps <= 0:
        raise ValueError(f"fps must be > 0; got {fps}")
    step = window_seconds / (num_anchors - 1)
    return [k * step for k in range(num_anchors)]
