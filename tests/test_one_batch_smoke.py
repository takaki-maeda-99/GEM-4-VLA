import torch
from torch.utils.data import DataLoader

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset


def test_full_forward_backward():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4, prompt_max_len=10)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    for p in policy.vision_encoder.parameters():
        p.requires_grad = False
    for p in policy.gemma.parameters():
        p.requires_grad = False

    ds = SyntheticLIBEROBatchDataset(length=2, prompt_max_len=10)
    dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    batch = next(iter(dl))

    pred, loss = policy(batch)
    assert pred.shape == (2, 8, 7)
    assert torch.isfinite(loss)

    loss.backward()
    for n, p in policy.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"bad grad: {n}"
        else:
            assert p.grad is None, f"frozen got grad: {n}"
