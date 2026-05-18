"""arch v3 smoke test: build policy, run a synthetic forward, verify no shape errors."""
import torch
from omegaconf import OmegaConf

import sys
sys.path.insert(0, "/misc/dl00/takaki/GEM-4-VLA/src")

from vla_project.data.constants import (
    ACTION_DIM, NUM_ACTION_TOKENS, NUM_SOFT_PROMPT_TOKENS, PROPRIO_DIM,
)
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.factory import build_vision_encoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


def main():
    cfg = OmegaConf.load("/misc/dl00/takaki/GEM-4-VLA/configs/train/oxe_pretrain_v47_arch_v3_libero_dl50_bs8.yaml")
    md = OmegaConf.to_container(cfg.model, resolve=True)
    lora_cfg = md.pop("lora", None)

    # Construct policy (will trigger __post_init__ validation)
    print("[smoke] constructing VLAPolicyConfig (validates flags)...")
    policy_cfg = VLAPolicyConfig(**md)
    print(f"[smoke] config OK: prompt_in_task_stream={policy_cfg.prompt_in_task_stream} "
          f"proprio_in_task_stream={policy_cfg.proprio_in_task_stream} "
          f"soft_prompt_as_cross_attn_stream={policy_cfg.soft_prompt_as_cross_attn_stream} "
          f"legacy_external_in_self_pool={policy_cfg.legacy_external_in_self_pool}")

    print("[smoke] building vision/gemma...")
    vision = build_vision_encoder(
        vision_type=str(cfg.vision.get("type", "hf")),
        model_name=cfg.vision.model_name,
    )
    gemma = Gemma4Wrapper(
        model_name=cfg.language.model_name, freeze=True, lora=lora_cfg,
    )
    device = "cuda"
    dtype = torch.bfloat16
    policy = VLAPolicy(policy_cfg, vision, gemma).to(device).to(dtype)
    print(f"[smoke] policy built, params: {sum(p.numel() for p in policy.parameters()):,}")

    # check k_soft_prompt / v_soft_prompt exist in each block
    blk = policy.action_head.model.blocks[0]
    print(f"[smoke] block 0 has k_soft_prompt: {hasattr(blk, 'k_soft_prompt')}")
    print(f"[smoke] block 0 has v_soft_prompt: {hasattr(blk, 'v_soft_prompt')}")
    print(f"[smoke] block 0 use_soft_prompt_cross_attn: {blk.use_soft_prompt_cross_attn}")

    # Synthetic batch (bs=2)
    B = 2
    batch = {
        "scene_image": torch.randn(B, 3, 224, 224, device=device, dtype=dtype),
        "wrist_image": torch.randn(B, 3, 224, 224, device=device, dtype=dtype),
        "proprio": torch.randn(B, PROPRIO_DIM, device=device, dtype=dtype),
        "prompt_input_ids": torch.randint(0, 1000, (B, policy_cfg.prompt_max_len), device=device),
        "prompt_attention_mask": torch.cat([
            torch.ones(B, 5, dtype=torch.long, device=device),
            torch.zeros(B, policy_cfg.prompt_max_len - 5, dtype=torch.long, device=device),
        ], dim=1),
        "domain_id": torch.tensor([0, 12], device=device, dtype=torch.long),
        "wrist_mask": torch.ones(B, dtype=torch.bool, device=device),
        "target_action": torch.randn(B, policy_cfg.action_chunk_len, ACTION_DIM, device=device, dtype=dtype),
        "action_mask": torch.ones(B, policy_cfg.action_chunk_len, dtype=torch.bool, device=device),
    }
    print("[smoke] running forward...")
    policy.eval()
    with torch.no_grad():
        pred, loss = policy(batch)
    print(f"[smoke] pred.shape={tuple(pred.shape)} loss={float(loss):.4f}")
    print("[smoke] OK")


if __name__ == "__main__":
    main()
