import torch

from vla_project.models.projectors.action_queries import ActionQueryHub


def test_shape():
    hub = ActionQueryHub(num_queries=64, hidden_dim=16)
    a = hub(2)
    assert a.shape == (2, 64, 16)


def test_broadcast_same_across_batch():
    """Shared queries: every batch entry sees the same [Q, D] tensor."""
    hub = ActionQueryHub(num_queries=4, hidden_dim=8)
    a = hub(3)
    assert torch.equal(a[0], a[1])
    assert torch.equal(a[0], a[2])
