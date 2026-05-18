"""Convert vla-gemma-4 baseline trainable_state_dict to our model.pt format.

Reads ``trainable_state_dict`` from a baseline ``latest_checkpoint.pt`` (e.g.
``libero_b_siglip_10k_wristb_b16_v2``) and rewrites it into the layout that
our :class:`vla_project.training.checkpoint.load_checkpoint` expects when
the model is built with ``use_baseline_projectors=True``.

Mapping (baseline → ours):
  ``vision_projector.fc{1,2,3}.*`` → ``scene_proj.fc{1,2,3}.*``
  ``proprio_projector.fc{1,2}.*``  → ``proprio_proj.fc{1,2}.*``
  ``wrist_projector_bridge.*``     → unchanged
  ``action_queries.weight``        → ``action_query_hub.queries``
  ``action_head.model.layer_norm{1,2}.*`` / ``fc{1,2}.*`` → unchanged
  ``action_head.model.mlp_resnet_blocks.{i}.*`` → ``action_head.model.blocks.{i}.*``
  ``wrist_encoder.*``              → DROPPED (dead when use_wrist_bridge=True)

Run via the gemma4 venv (the baseline ckpt was saved with transformers 5.x):

    PYTHONPATH=/misc/dl00/takaki/GEM-4-VLA/src \\
    /misc/dl00/takaki/vla-gemma-4/.venv-gemma4/bin/python \\
        tools/convert_baseline_ckpt.py <baseline.pt> <out_dir>

The output directory will contain ``model.pt`` + ``meta.json`` matching
``training/checkpoint.save_checkpoint``'s schema, suitable for loading via
``training/checkpoint.load_checkpoint``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import torch


def _convert(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    dropped: list = []
    for k, v in state.items():
        if k.startswith("wrist_encoder."):
            dropped.append(k)
            continue
        if k.startswith("vision_projector."):
            new = k.replace("vision_projector.", "scene_proj.", 1)
        elif k.startswith("proprio_projector."):
            new = k.replace("proprio_projector.", "proprio_proj.", 1)
        elif k == "action_queries.weight":
            new = "action_query_hub.queries"
        elif k.startswith("action_head.model.mlp_resnet_blocks."):
            new = k.replace(
                "action_head.model.mlp_resnet_blocks.",
                "action_head.model.blocks.",
                1,
            )
        elif k.startswith(
            ("action_head.model.layer_norm", "action_head.model.fc",
             "wrist_projector_bridge.")
        ):
            new = k
        else:
            print(f"[convert] WARN unmapped key, dropping: {k}")
            dropped.append(k)
            continue
        out[new] = v
    print(f"[convert] kept {len(out)} keys, dropped {len(dropped)} (wrist_encoder + unmapped)")
    return out


def main(src_path: str, dst_dir: str) -> None:
    src = Path(src_path)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[convert] loading {src}")
    payload = torch.load(str(src), map_location="cpu", weights_only=False)
    state = payload["trainable_state_dict"]
    cfg = payload.get("cfg", {})
    step = payload.get("gradient_step_idx", -1)

    converted = _convert(state)

    # Save as our format (model.pt + meta.json).
    model_pt = dst / "model.pt"
    meta_json = dst / "meta.json"
    torch.save(converted, model_pt)
    meta = {
        "step": int(step) if isinstance(step, int) else step,
        "source_ckpt": str(src.resolve()),
        "source_cfg_subset": {
            k: cfg.get(k)
            for k in [
                "vision_backbone_type",
                "use_wrist_bridge",
                "use_xvla_style",
                "use_proper_ffn",
                "training_mode",
                "lora_r",
                "num_action_head_blocks",
                "proprio_dim",
                "action_dim",
                "num_action_chunks",
                "data_format",
                "dataset_name",
                "siglip_use_tensor_transform",
            ]
        },
        "note": (
            "converted from vla-gemma-4 trainable_state_dict via "
            "tools/convert_baseline_ckpt.py. Build VLAPolicy with "
            "use_baseline_projectors=True to load."
        ),
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    print(f"[convert] wrote {model_pt} ({model_pt.stat().st_size/1e9:.2f} GB)")
    print(f"[convert] wrote {meta_json}")
    print(f"[convert] meta={json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <baseline.pt> <out_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
