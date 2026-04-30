import torch
from vla_project.models.projectors.action_queries import ActionQueryHub


def test_shape_and_per_domain_distinct():
    hub = ActionQueryHub(num_domains=2, num_queries=64, hidden_dim=16)
    a = hub(torch.tensor([0, 1]))
    assert a.shape == (2, 64, 16)
    assert not torch.allclose(a[0], a[1])
