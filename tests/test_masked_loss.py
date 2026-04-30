import torch
from vla_project.training.losses import masked_l1, masked_huber


def test_masked_l1_ignores_padded():
    pred = torch.tensor([[[1.0, 0.0]], [[0.0, 0.0]]])      # [2,1,2]
    targ = torch.tensor([[[0.0, 0.0]], [[10.0, 10.0]]])
    mask = torch.tensor([[True], [False]])
    loss = masked_l1(pred, targ, mask)
    # only first sample contributes; |1-0| + |0-0| over 2 elements -> 0.5
    torch.testing.assert_close(loss, torch.tensor(0.5))


def test_masked_huber_finite():
    pred = torch.randn(2, 8, 7)
    targ = torch.randn(2, 8, 7)
    mask = torch.ones(2, 8, dtype=torch.bool)
    loss = masked_huber(pred, targ, mask, beta=0.1)
    assert torch.isfinite(loss)
