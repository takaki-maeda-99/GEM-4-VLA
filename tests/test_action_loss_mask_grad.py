import torch

from vla_project.training.losses import masked_l1


def test_masked_positions_have_zero_grad():
    pred = torch.randn(2, 4, 3, requires_grad=True)
    target = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False],
                         [True, False, False, False]])
    loss = masked_l1(pred, target, mask)
    loss.backward()
    # Masked rows must have grad == 0
    assert torch.equal(pred.grad[0, 2:], torch.zeros(2, 3))
    assert torch.equal(pred.grad[1, 1:], torch.zeros(3, 3))
    # Unmasked rows must have non-zero grad somewhere
    assert pred.grad[0, :2].abs().sum() > 0
