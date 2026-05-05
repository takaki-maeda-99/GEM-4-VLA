"""Math primitives: quat/rot6d/matrix round-trips + EE6D pack/unpack."""
import math

import pytest
import torch

from vla_project.data.transforms.action_alignment import (
    action20_to_ee_pose,
    anchor_offsets,
    ee_pose_to_action20,
    matrix_to_quat,
    quat_to_matrix,
    quat_to_rot6d,
    rot6d_to_matrix,
    rot6d_to_quat,
)


# ---------- quaternion ↔ matrix ----------

def _identity_quat() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 1.0])  # scalar-last


def test_identity_quat_to_matrix() -> None:
    R = quat_to_matrix(_identity_quat())
    assert torch.allclose(R, torch.eye(3), atol=1e-6)


def test_quat_round_trip_via_matrix_random() -> None:
    torch.manual_seed(0)
    q = torch.randn(50, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    # Canonicalize input sign so equality is well-defined.
    q = torch.where(q[:, 3:4] < 0, -q, q)
    R = quat_to_matrix(q)
    q_round = matrix_to_quat(R)
    assert torch.allclose(q, q_round, atol=1e-5)


def test_matrix_round_trip_random() -> None:
    torch.manual_seed(1)
    q = torch.randn(20, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_matrix(q)
    R_round = quat_to_matrix(matrix_to_quat(R))
    assert torch.allclose(R, R_round, atol=1e-5)


# ---------- rot6d ↔ matrix ----------

def test_rot6d_round_trip_random() -> None:
    """matrix → rot6d → matrix is identity (rot6d preserves the rotation)."""
    torch.manual_seed(2)
    q = torch.randn(20, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_matrix(q)
    rot6d = quat_to_rot6d(q)  # use the canonical column-major impl
    R_round = rot6d_to_matrix(rot6d)
    assert torch.allclose(R, R_round, atol=1e-5)


def test_rot6d_to_matrix_orthonormalizes_perturbed_input() -> None:
    """Even with small Gaussian noise added to a valid rot6d, the recovered
    matrix is orthonormal (det=+1, columns unit-norm and orthogonal)."""
    torch.manual_seed(3)
    q = torch.randn(10, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    rot6d = quat_to_rot6d(q) + 0.01 * torch.randn(10, 6)
    R = rot6d_to_matrix(rot6d)
    # Orthonormal: R^T @ R == I
    eye = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
    assert torch.allclose(R.transpose(-1, -2) @ R, eye, atol=1e-5)
    # Right-handed: det == +1
    det = torch.linalg.det(R)
    assert torch.allclose(det, torch.ones(10), atol=1e-5)


def test_rot6d_round_trip_via_quat() -> None:
    torch.manual_seed(4)
    q = torch.randn(10, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[:, 3:4] < 0, -q, q)
    rot6d = quat_to_rot6d(q)
    q_round = rot6d_to_quat(rot6d)
    assert torch.allclose(q, q_round, atol=1e-5)


# ---------- EE6D pack/unpack ----------

def test_ee_pose_to_action20_layout() -> None:
    pos = torch.tensor([[0.1, 0.2, 0.3]])
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # identity
    gripper = torch.tensor([[0.7]])
    a = ee_pose_to_action20(pos, quat, gripper)
    assert a.shape == (1, 20)
    # xyz preserved
    assert torch.allclose(a[..., 0:3], pos)
    # rot6d for identity = [c1=[1,0,0], c2=[0,1,0]] concatenated.
    expected = torch.tensor([[1., 0., 0., 0., 1., 0.]])
    assert torch.allclose(a[..., 3:9], expected, atol=1e-6)
    # gripper preserved
    assert torch.allclose(a[..., 9:10], gripper)
    # padding zeros
    assert torch.all(a[..., 10:20] == 0.0)


def test_ee_pose_to_action20_round_trip() -> None:
    torch.manual_seed(5)
    pos = torch.randn(8, 3)
    q = torch.randn(8, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    q = torch.where(q[:, 3:4] < 0, -q, q)
    gripper = torch.rand(8, 1)
    a = ee_pose_to_action20(pos, q, gripper)
    pos_back, q_back, g_back = action20_to_ee_pose(a)
    assert torch.allclose(pos, pos_back, atol=1e-6)
    assert torch.allclose(q, q_back, atol=1e-5)
    assert torch.allclose(gripper, g_back, atol=1e-6)


def test_ee_pose_to_action20_squeezes_scalar_gripper() -> None:
    """gripper passed as (...,) instead of (..., 1) is auto-unsqueezed."""
    pos = torch.zeros(4, 3)
    q = torch.tensor([[0., 0., 0., 1.]] * 4)
    g = torch.tensor([0.0, 0.5, 1.0, 0.5])  # shape (4,)
    a = ee_pose_to_action20(pos, q, g)
    assert a.shape == (4, 20)
    assert torch.allclose(a[..., 9], g)


def test_ee_pose_rejects_wrong_shapes() -> None:
    pos = torch.zeros(2, 3)
    q4 = torch.tensor([[0., 0., 0., 1.], [0., 0., 0., 1.]])
    g = torch.zeros(2, 1)
    with pytest.raises(ValueError):
        ee_pose_to_action20(torch.zeros(2, 4), q4, g)  # bad pos
    with pytest.raises(ValueError):
        ee_pose_to_action20(pos, torch.zeros(2, 3), g)  # bad quat
    with pytest.raises(ValueError):
        action20_to_ee_pose(torch.zeros(2, 19))


# ---------- anchor offsets ----------

def test_anchor_offsets_libero_default() -> None:
    """LIBERO: 4-second window, 30 anchors, fps=10. Spacing = 4/29 ≈ 0.138 s."""
    offs = anchor_offsets(window_seconds=4.0, num_anchors=30, fps=10)
    assert len(offs) == 30
    assert offs[0] == 0.0
    assert abs(offs[-1] - 4.0) < 1e-9
    # Uniform spacing
    diffs = [offs[i + 1] - offs[i] for i in range(29)]
    expected = 4.0 / 29
    assert all(abs(d - expected) < 1e-9 for d in diffs)


def test_anchor_offsets_validations() -> None:
    with pytest.raises(ValueError):
        anchor_offsets(window_seconds=4.0, num_anchors=1, fps=10)
    with pytest.raises(ValueError):
        anchor_offsets(window_seconds=0.0, num_anchors=10, fps=10)
    with pytest.raises(ValueError):
        anchor_offsets(window_seconds=4.0, num_anchors=10, fps=0)


# ---------- matrix_to_axis_angle ----------

from vla_project.data.transforms.action_alignment import (
    action20_to_ee_delta,
    matrix_to_axis_angle,
)


def test_axis_angle_identity_is_zero() -> None:
    R = torch.eye(3).expand(5, 3, 3)
    aa = matrix_to_axis_angle(R)
    assert torch.allclose(aa, torch.zeros(5, 3), atol=1e-6)


def test_axis_angle_pure_x_rotation() -> None:
    """Rx(π/3) → axis-angle = (π/3, 0, 0)."""
    theta = math.pi / 3
    R = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(theta), -math.sin(theta)],
        [0.0, math.sin(theta),  math.cos(theta)],
    ])
    aa = matrix_to_axis_angle(R)
    expected = torch.tensor([theta, 0.0, 0.0])
    assert torch.allclose(aa, expected, atol=1e-5)


def test_axis_angle_pure_y_rotation_negative() -> None:
    """Ry(-π/4) → axis-angle = (0, -π/4, 0)."""
    theta = -math.pi / 4
    R = torch.tensor([
        [ math.cos(theta), 0.0, math.sin(theta)],
        [ 0.0,             1.0, 0.0],
        [-math.sin(theta), 0.0, math.cos(theta)],
    ])
    aa = matrix_to_axis_angle(R)
    expected = torch.tensor([0.0, theta, 0.0])
    assert torch.allclose(aa, expected, atol=1e-5)


def test_axis_angle_round_trip_via_aa_to_matrix() -> None:
    """aa → R (via Rodrigues from quat) → aa recovers the input."""
    torch.manual_seed(7)
    # Random axis-angles with magnitudes well below π to avoid the singularity.
    aa_in = torch.randn(20, 3) * 0.3   # magnitudes ~ 0.3 rad
    # Build R via quaternion: q = (sin(θ/2)*n, cos(θ/2)).
    theta = aa_in.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    n = aa_in / theta
    q_xyz = torch.sin(theta / 2) * n
    q_w = torch.cos(theta / 2)
    q = torch.cat([q_xyz, q_w], dim=-1)  # scalar-last
    R = quat_to_matrix(q)
    aa_out = matrix_to_axis_angle(R)
    assert torch.allclose(aa_in, aa_out, atol=1e-5)


# ---------- action20_to_ee_delta ----------

def test_ee_delta_identity_when_pred_equals_current() -> None:
    """pred = current pose → delta xyz=0, aa=0, gripper passthrough."""
    pos = torch.tensor([0.5, 0.1, 0.3])
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    gripper = torch.tensor([0.02])
    proprio = torch.cat([pos, quat, gripper])
    pad = torch.zeros(10)
    pred = torch.cat([
        pos,
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # identity rot6d
        torch.tensor([0.04]),                          # different gripper
        pad,
    ])
    delta = action20_to_ee_delta(pred, proprio)
    assert delta.shape == (7,)
    assert torch.allclose(delta[0:3], torch.zeros(3), atol=1e-6)
    assert torch.allclose(delta[3:6], torch.zeros(3), atol=1e-6)
    assert abs(float(delta[6]) - 0.04) < 1e-6  # gripper passthrough


def test_ee_delta_pure_translation() -> None:
    """pred xyz != current → delta xyz exact; rot delta zero."""
    proprio = torch.tensor([0.5, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02])
    pad = torch.zeros(10)
    pred = torch.cat([
        torch.tensor([0.7, 0.2, 0.4]),                 # pos shift
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # identity rot6d
        torch.tensor([0.02]),
        pad,
    ])
    delta = action20_to_ee_delta(pred, proprio)
    assert torch.allclose(delta[0:3], torch.tensor([0.2, 0.1, 0.1]), atol=1e-6)
    assert torch.allclose(delta[3:6], torch.zeros(3), atol=1e-5)


def test_ee_delta_batched_anchors() -> None:
    """Vectorize over a chunk of T anchors with a single current proprio."""
    T = 30
    proprio = torch.tensor([0.5, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0, 0.02])
    pred = torch.zeros(T, 20)
    pred[:, 0:3] = proprio[0:3]                # all anchors at current pos
    pred[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # identity rot6d
    pred[:, 9] = 0.02
    delta = action20_to_ee_delta(pred, proprio.unsqueeze(0).expand(T, -1))
    assert delta.shape == (T, 7)
    assert torch.allclose(delta[:, 0:6], torch.zeros(T, 6), atol=1e-6)


def test_ee_delta_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError):
        action20_to_ee_delta(torch.zeros(19), torch.zeros(8))
    with pytest.raises(ValueError):
        action20_to_ee_delta(torch.zeros(20), torch.zeros(7))
