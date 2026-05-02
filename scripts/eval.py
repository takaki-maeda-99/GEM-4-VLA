"""Closed-loop LIBERO evaluation entrypoint.

Loads a fresh (or checkpointed) VLAPolicy, builds an XVLAAdapterPolicy,
runs evaluate_libero over the configured task list, and prints aggregated
metrics. Checkpoint loading is optional via cfg.checkpoint.path.
"""
from __future__ import annotations

import json
import sys

import torch
from omegaconf import OmegaConf

from vla_project.data import constants as C
from vla_project.data.normalization import load_q99_proprio_stats, load_q99_stats
from vla_project.data.transforms.image import SiglipImageTransform
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.evaluation.libero_eval import evaluate_libero
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.siglip import SigLIPEncoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig
from vla_project.policies.xvla_adapter_policy import XVLAAdapterPolicy
from vla_project.robots.sim_robot import LIBEROSimRobot
from vla_project.training.checkpoint import load_checkpoint
from vla_project.utils.seed import set_seed


def main(cfg_path: str) -> None:
    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[eval] device={device} dtype={dtype}")

    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    lora_cfg = model_dict.pop("lora", None)
    policy_cfg = VLAPolicyConfig(**model_dict)

    vision = SigLIPEncoder(model_name=cfg.vision.model_name)
    gemma = Gemma4Wrapper(
        model_name=cfg.language.model_name, freeze=True, lora=lora_cfg
    )
    model = VLAPolicy(policy_cfg, vision, gemma).to(device).to(dtype)
    model.eval()

    if cfg.get("checkpoint", {}).get("path"):
        meta = load_checkpoint(cfg.checkpoint.path, model)
        print(f"[eval] loaded checkpoint step={meta.get('step')!r}")

    stats = load_q99_stats(cfg.data.stats_path, cfg.data.unnorm_key)
    proprio_stats = load_q99_proprio_stats(cfg.data.stats_path, cfg.data.unnorm_key)
    tok = GemmaPromptTokenizer(
        model_name=cfg.language.model_name, max_len=policy_cfg.prompt_max_len
    )
    image_tx = SiglipImageTransform(size=C.SIGLIP_IMAGE_SIZE, training=False)

    policy = XVLAAdapterPolicy(
        model=model, tokenizer=tok, image_transform=image_tx,
        norm_stats=stats, action_chunk_len=policy_cfg.action_chunk_len,
        domain_id=0,
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
    )
    summary = {"overall": metrics["overall"], "per_task": metrics["per_task"]}
    print(f"[eval] metrics={json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main(sys.argv[1])
