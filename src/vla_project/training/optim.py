import torch
import torch.nn as nn

from vla_project.models.vla_policy import VLAPolicy


def _trainable(params):
    return [p for p in params if p.requires_grad]


def build_optimizer(model: VLAPolicy, lr: float, soft_lr_coef: float, weight_decay: float):
    """Build AdamW with per-group LRs. Frozen params (SigLIP, Gemma in Stage 1)
    are *excluded* — not added with lr=0 — so AdamW does not allocate momentum
    state for them.
    """
    soft = _trainable(model.soft_prompt_hub.parameters())
    aq = _trainable(model.action_query_hub.parameters())
    head = _trainable(model.action_head.parameters())
    domain_projs = _trainable(
        list(model.scene_proj.parameters())
        + list(model.wrist_proj.parameters())
        + list(model.proprio_proj.parameters())
        + list(model.last_action_proj.parameters())
        + list(model.action_decoder.parameters())
    )

    groups = [
        {"name": "soft_prompts", "params": soft, "lr": lr * soft_lr_coef, "weight_decay": weight_decay},
        {"name": "action_queries", "params": aq, "lr": lr, "weight_decay": weight_decay},
        {"name": "domain_projs", "params": domain_projs, "lr": lr, "weight_decay": weight_decay},
        {"name": "action_head", "params": head, "lr": lr, "weight_decay": weight_decay},
    ]
    # filter out empty groups (defensive, e.g. if VLAPolicy has no soft prompts)
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, betas=(0.9, 0.95))
