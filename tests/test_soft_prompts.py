import torch

from vla_project.models.projectors.soft_prompts import SoftPromptHub


def test_shape_and_per_domain_distinct():
    hub = SoftPromptHub(num_domains=3, num_tokens=4, hidden_dim=8)
    out0 = hub(torch.tensor([0, 0]))
    out1 = hub(torch.tensor([1, 2]))
    assert out0.shape == (2, 4, 8)
    assert out1.shape == (2, 4, 8)
    assert not torch.allclose(out0[0], out1[0])
