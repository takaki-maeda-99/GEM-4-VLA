"""Closed-loop LIBERO evaluation entrypoint.

Loads a fresh (or checkpointed) VLAPolicy, builds an XVLAAdapterPolicy,
runs evaluate_libero over the configured task list, and prints aggregated
metrics. Checkpoint loading is optional via cfg.checkpoint.path.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict

import torch
from omegaconf import OmegaConf

from vla_project.data import constants as C
from vla_project.data.normalization import (
    load_norm_stats_payload,
    load_q99_proprio_stats,
    load_q99_stats,
    q99_stats_from_block,
)
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.evaluation.libero_eval import evaluate_libero
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.factory import build_vision_encoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.policies.xvla_adapter_policy import XVLAAdapterPolicy
from vla_project.robots.sim_robot import LIBEROSimRobot
from vla_project.training.checkpoint import load_checkpoint
from vla_project.utils.seed import set_seed


def _assert_checkpoint_stats_match(
    checkpoint_meta: Dict[str, Any],
    *,
    stats_path: str,
    unnorm_key: str,
) -> None:
    """Fail fast when a checkpoint's embedded stats disagree with eval config.

    Older checkpoints may not have ``norm_stats``; those keep the legacy
    behavior and rely on ``cfg.data.stats_path``.
    """
    ckpt_stats = checkpoint_meta.get("norm_stats")
    if not ckpt_stats:
        return
    if unnorm_key not in ckpt_stats:
        raise KeyError(
            f"checkpoint norm_stats has no {unnorm_key!r}; "
            f"available: {list(ckpt_stats.keys())}"
        )
    cfg_stats = load_norm_stats_payload(stats_path, unnorm_key)[unnorm_key]
    ckpt_ds = ckpt_stats[unnorm_key]
    for block_name in ("action", "proprio"):
        if block_name not in ckpt_ds or block_name not in cfg_stats:
            if block_name in ckpt_ds or block_name in cfg_stats:
                raise ValueError(
                    f"checkpoint/config stats mismatch: block {block_name!r} "
                    f"present in checkpoint={block_name in ckpt_ds}, "
                    f"config={block_name in cfg_stats}"
                )
            continue
        ckpt_block = q99_stats_from_block(ckpt_ds[block_name])
        cfg_block = q99_stats_from_block(cfg_stats[block_name])
        for name in ("q01", "q99"):
            a = getattr(ckpt_block, name)
            b = getattr(cfg_block, name)
            if a.shape != b.shape or not torch.allclose(a, b, atol=1e-6, rtol=0.0):
                max_diff = float((a - b).abs().max().item()) if a.shape == b.shape else float("inf")
                raise ValueError(
                    f"checkpoint/config stats mismatch for {block_name}.{name}: "
                    f"shape checkpoint={tuple(a.shape)} config={tuple(b.shape)} "
                    f"max_abs_diff={max_diff}"
                )
        if not torch.equal(ckpt_block.mask, cfg_block.mask):
            raise ValueError(f"checkpoint/config stats mismatch for {block_name}.mask")


def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[eval] device={device} dtype={dtype}")

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    lora_cfg = model_dict.pop("lora", None)
    policy_cfg = VLAPolicyConfig(**model_dict)

    vision = build_vision_encoder(
        vision_type=str(cfg.vision.get("type", "hf")),
        model_name=cfg.vision.model_name,
    )
    gemma = Gemma4Wrapper(
        model_name=cfg.language.model_name, freeze=True, lora=lora_cfg
    )
    model = VLAPolicy(policy_cfg, vision, gemma).to(device).to(dtype)
    model.eval()

    checkpoint_meta: Dict[str, Any] = {}
    if cfg.get("checkpoint", {}).get("path"):
        # ``strict=False`` allows loading the converted baseline ckpt (which
        # is missing wrist_proj weights — ResNet-18 path is dead when
        # use_wrist_bridge=True) and any forward-compat ckpts that omit a
        # subset of params.
        strict = bool(cfg.checkpoint.get("strict", True))
        checkpoint_meta = load_checkpoint(cfg.checkpoint.path, model, strict=strict)
        print(f"[eval] loaded checkpoint step={checkpoint_meta.get('step')!r} strict={strict}")

    _assert_checkpoint_stats_match(
        checkpoint_meta,
        stats_path=cfg.data.stats_path,
        unnorm_key=cfg.data.unnorm_key,
    )
    stats = load_q99_stats(cfg.data.stats_path, cfg.data.unnorm_key)
    proprio_stats = load_q99_proprio_stats(cfg.data.stats_path, cfg.data.unnorm_key)
    tok = GemmaPromptTokenizer(
        model_name=cfg.language.model_name, max_len=policy_cfg.prompt_max_len
    )
    image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)

    # ``cfg.eval.domain_id`` selects which DA row the policy uses at inference.
    # Single-domain LIBERO eval keeps 0 (matches how every LIBERO ckpt was
    # trained). For an OXE-pretrained ckpt, set this to the LIBERO domain_id
    # used during finetune (or to the OXE domain_id closest to the deploy
    # embodiment when probing zero-shot transfer). v37 ckpts also embed a
    # per-domain norm-stats manifest in meta.json; use that to pick the
    # right unnorm_key when finetuning, instead of relying on cfg.data.stats_path.
    eval_domain_id = int(cfg.eval.get("domain_id", 0))
    policy = XVLAAdapterPolicy(
        model=model, tokenizer=tok, image_transform=image_tx,
        norm_stats=stats, action_chunk_len=policy_cfg.action_chunk_len,
        domain_id=eval_domain_id,
        compile_mode=str(cfg.eval.get("compile_mode", "off")),
        action_format=str(cfg.data.get("action_format", "native")),
        proprio_stats=proprio_stats,
    )

    def _make_robot(task_idx: int) -> LIBEROSimRobot:
        return LIBEROSimRobot(
            bddl_path_root=cfg.robot.bddl_path_root,
            task_suite=cfg.robot.task_suite,
            task_idx=task_idx,
            image_size=cfg.robot.image_size,
            seed=cfg.seed,
            libero_path=cfg.robot.libero_path,
        )

    metrics = evaluate_libero(
        policy=policy,
        robot_factory=_make_robot,
        task_idxs=list(cfg.eval.task_idxs),
        num_episodes_per_task=int(cfg.eval.num_episodes_per_task),
        max_steps=int(cfg.eval.max_steps),
        num_steps_wait=int(cfg.eval.num_steps_wait),
        video_dir=cfg.eval.get("video_dir", None),
        video_fps=int(cfg.eval.get("video_fps", 10)),
        video_ext=str(cfg.eval.get("video_ext", "gif")),
    )
    summary = {"overall": metrics["overall"], "per_task": metrics["per_task"]}
    print(f"[eval] metrics={json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main(sys.argv[1])
