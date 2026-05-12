"""Dump per-block weight L2 norms inside policy.action_head.blocks (18 blocks)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper
from vla_project.models.vision.factory import build_vision_encoder
from vla_project.models.vla_policy import VLAPolicy, VLAPolicyConfig

CKPT = "outputs/oxe_pretrain_v45_nb18even_proper_mlp_alllinear_libero_dl50_bs8/checkpoints/step_35000"

meta = json.loads((Path(CKPT) / "meta.json").read_text())
cfg = OmegaConf.create(meta["cfg"])
md = OmegaConf.to_container(cfg.model, resolve=True)
lora_cfg = md.pop("lora", None)
policy = VLAPolicy(
    VLAPolicyConfig(**md),
    build_vision_encoder(vision_type="timm", model_name=cfg.vision.model_name),
    Gemma4Wrapper(model_name=cfg.language.model_name, freeze=True, lora=lora_cfg),
).to("cuda").to(torch.bfloat16)
sd = torch.load(Path(CKPT) / "model.pt", map_location="cpu", weights_only=False)
sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()} if any(k.startswith("_orig_mod.") for k in sd) else sd
policy.load_state_dict(sd, strict=False)
policy.eval()

# action_head.blocks[i].<sub>.<weight/bias>
per_block = defaultdict(lambda: defaultdict(float))
fc1_sq = 0.0
fc2_sq = 0.0
ln1_sq = 0.0
ln2_sq = 0.0
prefix = "action_head.model."
for name, p in policy.named_parameters():
    if not name.startswith(prefix):
        continue
    sq = p.detach().float().pow(2).sum().item()
    rest = name[len(prefix):]
    # action_head.layer_norm1 / fc1 / layer_norm2 / fc2 / blocks.<i>.<sub>
    if rest.startswith("blocks."):
        parts = rest.split(".")
        blk = int(parts[1])
        sub = parts[2]
        per_block[blk][sub] += sq
        per_block[blk]["_total"] += sq
    elif rest.startswith("layer_norm1"):
        ln1_sq += sq
    elif rest.startswith("layer_norm2"):
        ln2_sq += sq
    elif rest.startswith("fc1"):
        fc1_sq += sq
    elif rest.startswith("fc2"):
        fc2_sq += sq

print("\n=== action_head per-block L2 norms (v45 step_35000) ===\n")
print(f"  layer_norm1       L2={ln1_sq**0.5:8.3f}  (input LN)")
print(f"  fc1               L2={fc1_sq**0.5:8.3f}  (input proj)")
print(f"  layer_norm2       L2={ln2_sq**0.5:8.3f}  (output LN)")
print(f"  fc2               L2={fc2_sq**0.5:8.3f}  (output proj)")
print()

# Collect distinct sub-module names
subs = set()
for b in per_block.values():
    subs.update(b)
subs = sorted(s for s in subs if s != "_total")

# Print per-block totals
print(f"  {'block':>5s}  " + " ".join(f"{s:>12s}" for s in subs) + f"  {'TOTAL':>10s}")
totals = []
for blk in sorted(per_block):
    row = [f"{per_block[blk][s]**0.5:12.3f}" for s in subs]
    tot = per_block[blk]["_total"] ** 0.5
    totals.append(tot)
    print(f"  {blk:>5d}  " + " ".join(row) + f"  {tot:>10.3f}")

# Aggregate stats
print(f"\n  blocks total mean={sum(totals)/len(totals):.3f} min={min(totals):.3f} max={max(totals):.3f} (across {len(totals)} blocks)")
