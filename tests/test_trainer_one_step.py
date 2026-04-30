import torch
from torch.utils.data import DataLoader

from tests._stubs import _StubSig, _StubGemma
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset
from vla_project.training.optim import build_optimizer
from vla_project.training.trainer import Trainer, TrainerConfig


def test_one_training_step_decreases_or_holds_loss():
    cfg = VLAPolicyConfig(num_domains=1, hidden_dim=32, num_blocks=4, prompt_max_len=10)
    policy = VLAPolicy(cfg, _StubSig(), _StubGemma())
    ds = SyntheticLIBEROBatchDataset(length=4, prompt_max_len=10)
    dl = DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    optim = build_optimizer(policy, lr=1e-4, soft_lr_coef=1.0, weight_decay=0.0)
    trainer = Trainer(policy, optim, TrainerConfig(max_steps=2))
    losses = trainer.fit(dl)
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
    assert len(losses) == 2
