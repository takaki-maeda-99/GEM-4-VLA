import torch

from vla_project.models.projectors.domain_aware_linear import DomainAwareLinear


def test_2d_input_shape():
    layer = DomainAwareLinear(input_size=8, output_size=16, num_domains=4)
    x = torch.randn(3, 8)
    domain_id = torch.tensor([0, 1, 3], dtype=torch.long)
    y = layer(x, domain_id)
    assert y.shape == (3, 16)


def test_3d_input_shape():
    layer = DomainAwareLinear(input_size=8, output_size=16, num_domains=4)
    x = torch.randn(3, 5, 8)
    domain_id = torch.tensor([0, 1, 3], dtype=torch.long)
    y = layer(x, domain_id)
    assert y.shape == (3, 5, 16)


def test_different_domains_yield_different_outputs():
    layer = DomainAwareLinear(input_size=4, output_size=4, num_domains=3)
    torch.nn.init.normal_(layer.fc.weight, std=0.5)
    torch.nn.init.normal_(layer.bias.weight, std=0.5)
    x = torch.randn(1, 4)
    y0 = layer(x, torch.tensor([0]))
    y1 = layer(x, torch.tensor([1]))
    assert not torch.allclose(y0, y1)
