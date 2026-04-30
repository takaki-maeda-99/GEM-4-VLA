"""Thin training entrypoint. Heavy lifting lives in vla_project.training.trainer."""
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.siglip import SigLIPEncoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.training.optim import build_optimizer
from vla_project.training.trainer import Trainer, TrainerConfig
from vla_project.utils.seed import set_seed


def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[train] device={device} dtype={dtype}")

    policy_cfg = VLAPolicyConfig(**cfg.model)
    vision = SigLIPEncoder(model_name=cfg.vision.model_name)
    gemma = Gemma4Wrapper(model_name=cfg.language.model_name, freeze=True)
    policy = VLAPolicy(policy_cfg, vision, gemma).to(device).to(dtype)

    ds = SyntheticLIBEROBatchDataset(length=cfg.data.length, prompt_max_len=policy_cfg.prompt_max_len)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=ds.collate_fn)

    optim = build_optimizer(
        policy, lr=cfg.train.lr,
        soft_lr_coef=cfg.train.soft_lr_coef, weight_decay=cfg.train.weight_decay,
    )
    trainer = Trainer(policy, optim, TrainerConfig(max_steps=cfg.train.max_steps))
    losses = trainer.fit(dl)
    print(f"[train] losses={losses}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
