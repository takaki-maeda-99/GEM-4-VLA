"""Diagnose v45 step_35000: weight L2 norms (A) + input occlusion sensitivity (B).

A. Iterate over policy.named_parameters(); group by module prefix and report
   total L2 norm per group. For DA per-domain projectors (scene_proj,
   wrist_proj, proprio_proj, action_decoder.heads), also report per-domain
   row norms — tells you which domains "learned more" during pretrain.

B. Occlusion sensitivity on a small LIBERO Spatial batch (domain_id=12):
   baseline forward loss, then per-modality zero-out (scene/wrist/proprio/
   language tokens). Reports Δloss per modality.

Usage:
    .venv/bin/python tools/v45_diagnose.py [ckpt_dir]

Defaults to outputs/oxe_pretrain_v45_nb18even_proper_mlp_alllinear_libero_dl50_bs8/checkpoints/step_35000.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from vla_project.data.datasets.rlds_libero_dataset import RLDSLiberoDataset
from vla_project.data.transforms.language import GemmaPromptTokenizer
from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.factory import build_vision_encoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig


DEFAULT_CKPT = (
    "outputs/oxe_pretrain_v45_nb18even_proper_mlp_alllinear_libero_dl50_bs8"
    "/checkpoints/step_35000"
)


def load_policy(ckpt_dir: str, device: str = "cuda"):
    meta = json.loads((Path(ckpt_dir) / "meta.json").read_text())
    cfg = OmegaConf.create(meta["cfg"])
    md = OmegaConf.to_container(cfg.model, resolve=True)
    lora_cfg = md.pop("lora", None)
    policy_cfg = VLAPolicyConfig(**md)
    vision = build_vision_encoder(
        vision_type=str(cfg.vision.get("type", "hf")),
        model_name=cfg.vision.model_name,
    )
    gemma = Gemma4Wrapper(
        model_name=cfg.language.model_name, freeze=True, lora=lora_cfg,
    )
    policy = VLAPolicy(policy_cfg, vision, gemma).to(device).to(torch.bfloat16)
    sd = torch.load(Path(ckpt_dir) / "model.pt", map_location="cpu", weights_only=False)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing[:5]:
        print(f"  first missing: {missing[:5]}")
    if unexpected[:5]:
        print(f"  first unexpected: {unexpected[:5]}")
    policy.eval()
    return policy, cfg


def _group_for(name: str) -> str:
    lname = name.lower()
    if "lora_" in lname:
        return "gemma_lora"
    if "siglip" in lname or "vision_encoder" in lname or name.startswith("vision."):
        return "vision_frozen"
    if "language_model" in lname or "gemma" in lname or name.startswith("language."):
        return "gemma_base"
    if "scene_proj" in lname:
        return "scene_proj"
    if "wrist_proj" in lname or "wrist_bridge" in lname:
        return "wrist_proj"
    if "proprio_proj" in lname or "proprio_projector" in lname:
        return "proprio_proj"
    if "action_queries" in lname or "action_query" in lname:
        return "action_queries"
    if "action_decoder" in lname:
        return "action_decoder"
    if "action_head" in lname or "head_blocks" in lname or "mlp_resnet" in lname:
        return "action_head"
    if "soft_prompt" in lname:
        return "soft_prompts"
    if "head_proj" in lname:
        return "head_projections"
    if "input_packer" in lname:
        return "input_packer"
    return "other"


def part_a_weight_norms(policy):
    print("\n=== A. Weight L2 norms (v45 step_35000) ===")
    group_sq = defaultdict(float)
    group_count = defaultdict(int)
    # For DA per-domain rows: aggregate per (group, domain)
    domain_sq = defaultdict(lambda: defaultdict(float))
    num_domains = None

    for name, p in policy.named_parameters():
        grp = _group_for(name)
        sq = p.detach().float().pow(2).sum().item()
        group_sq[grp] += sq
        group_count[grp] += p.numel()
        # DA per-domain: first dim = num_domains (== 13 here)
        if p.dim() >= 2 and p.shape[0] == 13 and grp in (
            "scene_proj", "wrist_proj", "proprio_proj", "action_decoder", "head_projections",
        ):
            num_domains = 13
            for d in range(13):
                domain_sq[grp][d] += p[d].detach().float().pow(2).sum().item()

    print(f"\n  {'group':<20s} {'L2_norm':>12s} {'params':>14s}")
    print(f"  {'-'*20} {'-'*12} {'-'*14}")
    for g in sorted(group_sq):
        l2 = group_sq[g] ** 0.5
        print(f"  {g:<20s} {l2:>12.3f} {group_count[g]:>14,d}")
    total = sum(group_sq.values()) ** 0.5
    print(f"  {'TOTAL':<20s} {total:>12.3f} {sum(group_count.values()):>14,d}")

    if num_domains:
        print(f"\n  Per-domain L2 (rows of DA projectors, num_domains={num_domains}):")
        print(f"    domain_id ->", " ".join(f"d{d:2d}" for d in range(13)))
        for g in sorted(domain_sq):
            vals = [domain_sq[g][d] ** 0.5 for d in range(13)]
            row = " ".join(f"{v:6.2f}" for v in vals)
            print(f"    {g:<14s}: {row}")
        # Highlight which domains have anomalously low/high norm
        print("\n  Domain row z-scores (per projector, z = (x - mean) / std):")
        for g in sorted(domain_sq):
            vals = [domain_sq[g][d] ** 0.5 for d in range(13)]
            m = sum(vals) / len(vals)
            v = sum((x - m) ** 2 for x in vals) / len(vals)
            s = v ** 0.5 if v > 0 else 1.0
            zs = [(x - m) / s for x in vals]
            row = " ".join(f"{z:+6.2f}" for z in zs)
            print(f"    {g:<14s}: {row}")


def part_b_occlusion(policy, cfg, n_batches: int = 3, batch_size: int = 4):
    print("\n=== B. Occlusion sensitivity (LIBERO Spatial, domain_id=12) ===")
    tok = GemmaPromptTokenizer(model_name=cfg.language.model_name, max_len=20)
    ds = RLDSLiberoDataset(
        data_dir="/misc/dl00/takaki/vla-gemma-4/data/modified_libero_rlds",
        dataset_name="libero_spatial_no_noops",
        tokenizer=tok,
        action_chunk_len=8,
        shuffle_buffer_size=512,
        train=True,
        domain_id=12,
        seed=42,
    )
    dl = DataLoader(ds, batch_size=batch_size, collate_fn=RLDSLiberoDataset.collate_fn)
    device = next(policy.parameters()).device
    dtype = next(policy.parameters()).dtype

    def _to_device(b):
        out = {}
        for k, v in b.items():
            if torch.is_tensor(v):
                if v.dtype.is_floating_point:
                    out[k] = v.to(device).to(dtype)
                else:
                    out[k] = v.to(device)
            else:
                out[k] = v
        return out

    def _loss(b):
        _, loss = policy(b)
        return float(loss.detach())

    losses = defaultdict(list)
    n = 0
    with torch.no_grad():
        for batch in dl:
            if n >= n_batches:
                break
            n += 1
            b = _to_device(batch)
            losses["baseline"].append(_loss(b))
            # no_scene
            losses["no_scene"].append(_loss({**b, "scene_image": torch.zeros_like(b["scene_image"])}))
            # no_wrist
            if "wrist_image" in b:
                losses["no_wrist"].append(_loss({**b, "wrist_image": torch.zeros_like(b["wrist_image"])}))
            # no_proprio
            if "proprio" in b:
                losses["no_proprio"].append(_loss({**b, "proprio": torch.zeros_like(b["proprio"])}))
            # no_lang: blank the prompt (attention_mask 0 will make Gemma ignore)
            if "prompt_input_ids" in b:
                losses["no_lang"].append(_loss({
                    **b,
                    "prompt_input_ids": torch.zeros_like(b["prompt_input_ids"]),
                    "prompt_attention_mask": torch.zeros_like(b["prompt_attention_mask"]),
                }))

    base = sum(losses["baseline"]) / len(losses["baseline"])
    print(f"\n  baseline loss (mean over {len(losses['baseline'])} batches × bs={batch_size}): {base:.4f}")
    print(f"  {'occlusion':<14s} {'mean_loss':>10s} {'Δ vs base':>10s} {'rel %':>10s}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
    for k in ("no_scene", "no_wrist", "no_proprio", "no_lang"):
        vs = losses.get(k, [])
        if not vs:
            continue
        m = sum(vs) / len(vs)
        delta = m - base
        rel = (delta / base) * 100 if base > 0 else 0.0
        print(f"  {k:<14s} {m:>10.4f} {delta:>+10.4f} {rel:>+9.1f}%")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT
    print(f"[diagnose] ckpt = {ckpt}")
    policy, cfg = load_policy(ckpt)
    part_a_weight_norms(policy)
    part_b_occlusion(policy, cfg)


if __name__ == "__main__":
    main()
