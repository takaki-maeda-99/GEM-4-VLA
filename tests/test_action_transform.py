import torch
from vla_project.data.transforms.action import action_slice


def test_action_slice_delta_indices():
    abs_traj = torch.tensor([[1.0, 2.0], [4.0, 6.0], [5.0, 7.0]])  # H=2, D=2
    out = action_slice(abs_traj, idx_for_delta=[0])
    assert torch.equal(out["proprio"], torch.tensor([1.0, 2.0]))
    assert out["action"].shape == (2, 2)
    assert out["action"][0, 0].item() == 4.0 - 1.0
