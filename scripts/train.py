"""Thin training entrypoint. Heavy lifting lives in vla_project.training.trainer."""
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from vla_project.data.datasets.libero_dataset import SyntheticLIBEROBatchDataset
from vla_project.data.datasets.lerobot_libero_dataset import LeRobotLiberoDataset
from vla_project.data.datasets.weighted_multi_dataset import WeightedMultiDataset
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.siglip import SigLIPEncoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.training.optim import build_optimizer
from vla_project.training.trainer import Trainer, TrainerConfig
from vla_project.utils.seed import set_seed


def _build_dataloader(cfg: DictConfig, prompt_max_len: int, language_model_name: str):
    data_type = cfg.data.get("type", "libero_synthetic")
    if data_type == "libero_synthetic":
        ds = SyntheticLIBEROBatchDataset(
            length=cfg.data.length, prompt_max_len=prompt_max_len,
        )
        return DataLoader(ds, batch_size=cfg.train.batch_size, collate_fn=ds.collate_fn)
    if data_type == "libero_lerobot_real":
        tok = GemmaPromptTokenizer(model_name=language_model_name, max_len=prompt_max_len)
        ds = LeRobotLiberoDataset(
            repo_id=cfg.data.repo_id,
            stats_path=cfg.data.stats_path,
            unnorm_key=cfg.data.unnorm_key,
            fps=cfg.data.fps,
            tokenizer=tok,
            episodes=list(cfg.data.episodes) if cfg.data.get("episodes") else None,
            action_chunk_len=int(cfg.data.get("action_chunk_len", 8)),
            download_videos=bool(cfg.data.get("download_videos", False)),
            domain_id=int(cfg.data.get("domain_id", 0)),
            max_samples=cfg.data.get("max_samples", None),
            last_action_chunk_mode=str(cfg.data.get("last_action_chunk_mode", "zero")),
            action_format=str(cfg.data.get("action_format", "native")),
            anchor_window_s=float(cfg.data.get("anchor_window_s", 0.0)),
            task_index_filter=cfg.data.get("task_index_filter", None),
        )
        return DataLoader(
            ds, batch_size=cfg.train.batch_size,
            collate_fn=LeRobotLiberoDataset.collate_fn,
        )
    if data_type == "libero_lerobot_multidomain":
        tok = GemmaPromptTokenizer(model_name=language_model_name, max_len=prompt_max_len)
        children: list = []
        weights: list = []
        for src in cfg.data.sources:
            children.append(LeRobotLiberoDataset(
                repo_id=src.repo_id,
                stats_path=src.stats_path,
                unnorm_key=src.unnorm_key,
                fps=src.fps,
                tokenizer=tok,
                episodes=list(src.episodes) if src.get("episodes") else None,
                action_chunk_len=int(src.get("action_chunk_len", 8)),
                download_videos=bool(cfg.data.get("download_videos", False)),
                domain_id=int(src.domain_id),
                max_samples=src.get("max_samples", None),
                last_action_chunk_mode=str(src.get("last_action_chunk_mode", "zero")),
                action_format=str(src.get("action_format", "native")),
                anchor_window_s=float(src.get("anchor_window_s", 0.0)),
            ))
            weights.append(float(src.weight))
        ds = WeightedMultiDataset(children, weights, seed=int(cfg.data.get("seed", 0)))
        return DataLoader(
            ds, batch_size=cfg.train.batch_size,
            collate_fn=LeRobotLiberoDataset.collate_fn,
        )
    raise ValueError(f"unknown cfg.data.type: {data_type!r}")


def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)

    # Construct Accelerator early so we can read the correct per-rank device.
    # In single-process mode this is a no-op; under accelerate launch it
    # reads LOCAL_RANK and resolves to cuda:LOCAL_RANK so FSDP / DDP both
    # see the model on the expected device.
    from accelerate import Accelerator
    # wandb is ENABLED by default. Set `wandb.enabled: false` in the config,
    # or export `WANDB_MODE=disabled` (no run, no files) / `WANDB_MODE=offline`
    # (local cache only, no server) to opt out for one-off smoke runs.
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", True):
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(
            project_name=wandb_cfg.get("project", "vla-project"),
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={
                "wandb": {
                    "name": wandb_cfg.get("name"),
                    "tags": list(wandb_cfg.get("tags", [])) or None,
                }
            },
        )
        print(f"[train] wandb tracking enabled: project={wandb_cfg.get('project', 'vla-project')!r}")
    else:
        accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[train] device={device} dtype={dtype}")

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    lora_cfg = model_dict.pop("lora", None)
    policy_cfg = VLAPolicyConfig(**model_dict)
    vision = SigLIPEncoder(model_name=cfg.vision.model_name)
    gemma = Gemma4Wrapper(
        model_name=cfg.language.model_name,
        freeze=True,
        lora=lora_cfg,
    )
    policy = VLAPolicy(policy_cfg, vision, gemma).to(device).to(dtype)
    compile_mode = str(cfg.train.get("compile_mode", "off"))
    if compile_mode != "off":
        # `mode in {"default", "reduce-overhead", "max-autotune"}` per torch
        # docs. fullgraph=False allows graph breaks (Gemma's HF code path has
        # Python control flow that can't always be traced into a single graph).
        #
        # Empirical 2026-05-01 on dl40 A100 (bs=1, 35 blocks, bf16):
        #   compile_mode=off            -> 400.4 ms / step
        #   compile_mode='default'      -> 173.3 ms / step  (2.3x faster)
        #
        # !! Known limitation: combining compile_mode with model.use_grad_
        # checkpoint=true triggers an InductorError (KeyError: 'op39') in the
        # backward compilation. Set use_grad_checkpoint=false when compiling.
        print(f"[train] applying torch.compile(mode={compile_mode!r}, fullgraph=False)")
        policy = torch.compile(policy, mode=compile_mode, fullgraph=False)

    dl = _build_dataloader(
        cfg, prompt_max_len=policy_cfg.prompt_max_len,
        language_model_name=cfg.language.model_name,
    )

    lr_coefs = cfg.train.get("lr_coefs", None)
    if lr_coefs is not None:
        lr_coefs = OmegaConf.to_container(lr_coefs, resolve=True)
    optim = build_optimizer(
        policy, lr=cfg.train.lr,
        soft_lr_coef=cfg.train.get("soft_lr_coef"),
        weight_decay=cfg.train.weight_decay,
        lr_coefs=lr_coefs,
    )
    trainer_cfg = TrainerConfig(
        max_steps=cfg.train.max_steps,
        save_every=cfg.train.get("save_every"),
        save_dir=cfg.train.get("save_dir"),
        warmup_steps=int(cfg.train.get("warmup_steps", 0)),
        min_lr_ratio=float(cfg.train.get("min_lr_ratio", 1.0)),
        freeze_steps=int(cfg.train.get("freeze_steps", 0)),
        grad_clip_norm=float(cfg.train.get("grad_clip_norm", 1.0)),
    )
    trainer = Trainer(policy, optim, trainer_cfg, accelerator=accelerator)
    losses = trainer.fit(dl)
    print(f"[train] losses={losses}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
