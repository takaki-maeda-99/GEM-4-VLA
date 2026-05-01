from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from vla_project.models.vla_policy import VLAPolicy


def _trainable(params):
    return [p for p in params if p.requires_grad]


# Default per-group LR multipliers (X-VLA-Adapter convention).
#   - SigLIP / Gemma backbone: 0.1 × base_lr  (when unfrozen / LoRA)
#   - SoftPrompt:              0.1 × base_lr
#   - Bridge / ActionQuery:    1.0 × base_lr
#   - ActionHead:              1.0 × base_lr
DEFAULT_LR_COEFS: Dict[str, float] = {
    "gemma_lora":     0.1,
    "siglip":         0.1,
    "soft_prompts":   0.1,
    "action_queries": 1.0,
    "projections":    1.0,
    "action_head":    1.0,
}


def build_optimizer(
    model: VLAPolicy,
    lr: float,
    soft_lr_coef: Optional[float] = None,  # deprecated: use lr_coefs['soft_prompts']
    weight_decay: float = 0.0,
    *,
    lr_coefs: Optional[Dict[str, float]] = None,
):
    """Build AdamW with per-group LR multipliers.

    Frozen params are excluded entirely (no momentum allocated). Trainable
    params are partitioned into the following groups, each with its own LR
    coefficient ``lr_coefs[group] * lr``:

      - ``gemma_lora``: any requires_grad=True param under model.gemma
        (typically the LoRA adapters injected by Plan 5 / Stage 2)
      - ``siglip``:     any requires_grad=True param under
        model.vision_encoder (Stage 1+2 keep this frozen by default)
      - ``soft_prompts``:   model.soft_prompt_hub
      - ``action_queries``: model.action_query_hub
      - ``projections``:    DA Linears (scene/wrist/proprio/last_action/action_decoder)
      - ``action_head``:    model.action_head

    ``soft_lr_coef`` is a deprecated alias that overrides
    ``lr_coefs['soft_prompts']`` if provided. Prefer the ``lr_coefs`` dict.
    """
    coefs = dict(DEFAULT_LR_COEFS)
    if lr_coefs is not None:
        coefs.update(lr_coefs)
    if soft_lr_coef is not None:
        coefs["soft_prompts"] = soft_lr_coef

    gemma_lora    = _trainable(model.gemma.parameters())
    siglip        = _trainable(model.vision_encoder.parameters())
    soft          = _trainable(model.soft_prompt_hub.parameters())
    aq            = _trainable(model.action_query_hub.parameters())
    head          = _trainable(model.action_head.parameters())
    projs         = _trainable(
        list(model.scene_proj.parameters())
        + list(model.wrist_proj.parameters())
        + list(model.proprio_proj.parameters())
        + list(model.action_decoder.parameters())
    )

    groups = [
        {"name": "gemma_lora",     "params": gemma_lora, "lr": lr * coefs["gemma_lora"],     "weight_decay": weight_decay},
        {"name": "siglip",         "params": siglip,     "lr": lr * coefs["siglip"],         "weight_decay": weight_decay},
        {"name": "soft_prompts",   "params": soft,       "lr": lr * coefs["soft_prompts"],   "weight_decay": weight_decay},
        {"name": "action_queries", "params": aq,         "lr": lr * coefs["action_queries"], "weight_decay": weight_decay},
        {"name": "projections",    "params": projs,      "lr": lr * coefs["projections"],    "weight_decay": weight_decay},
        {"name": "action_head",    "params": head,       "lr": lr * coefs["action_head"],    "weight_decay": weight_decay},
    ]
    # Drop empty groups (frozen modules contribute nothing).
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95))
