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
    "gemma_lora":         0.1,
    "siglip":             0.1,
    # ``wrist_siglip``: LoRA adapters injected into the wrist-only SigLIP
    # encoder copy (when ``cfg.wrist_siglip_lora`` is set). Empty otherwise.
    "wrist_siglip":       0.1,
    "soft_prompts":       0.1,
    "action_queries":     1.0,
    # ``projections`` (legacy single group) split 2026-05-04 into:
    #   - ``scene_projection`` : scene_proj (= modules whose output is
    #     scattered into the LLM input embeddings, "above the LLM")
    #   - ``head_projections`` : proprio_proj / wrist_proj / wrist_projector_bridge
    #     / action_decoder (modules feeding action_head directly, "below the LLM")
    # Configs may still set ``projections``: when present and the new keys
    # are NOT, both new groups inherit that value (back-compat for v25-v27).
    "projections":        1.0,
    "scene_projection":   1.0,
    "head_projections":   1.0,
    "action_head":        1.0,
}


def build_optimizer(
    model: VLAPolicy,
    lr: float,
    soft_lr_coef: Optional[float] = None,  # deprecated: use lr_coefs['soft_prompts']
    weight_decay: float = 0.0,
    *,
    lr_coefs: Optional[Dict[str, float]] = None,
    optimizer_kind: str = "adamw",
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
    # Back-compat: when only the legacy ``projections`` key was supplied and
    # the user didn't explicitly override the new split, propagate it to both
    # new groups so v25-v27 configs keep working unchanged.
    if lr_coefs is not None and "projections" in lr_coefs:
        if "scene_projection" not in lr_coefs:
            coefs["scene_projection"] = lr_coefs["projections"]
        if "head_projections" not in lr_coefs:
            coefs["head_projections"] = lr_coefs["projections"]
    if soft_lr_coef is not None:
        coefs["soft_prompts"] = soft_lr_coef

    gemma_lora    = _trainable(model.gemma.parameters())
    siglip        = _trainable(model.vision_encoder.parameters())
    # ``wrist_siglip``: LoRA params from the wrist-only SigLIP copy (when
    # ``cfg.wrist_siglip_lora`` is set). Returns an empty list otherwise so the
    # group is dropped by the empty-group filter below.
    wrist_siglip_params = (
        list(model.wrist_vision_encoder.parameters())
        if getattr(model, "wrist_vision_encoder", None) is not None
        else []
    )
    wrist_siglip  = _trainable(wrist_siglip_params)
    soft          = _trainable(
        model.soft_prompt_hub.parameters() if model.soft_prompt_hub is not None else []
    )
    aq            = _trainable(model.action_query_hub.parameters())
    head          = _trainable(model.action_head.parameters())
    # ``scene_projection``: modules feeding LLM input embeddings (above LLM).
    scene_proj_params = list(model.scene_proj.parameters())
    if getattr(model, "scene_wrist_dinov2_llm_proj", None) is not None:
        scene_proj_params += list(model.scene_wrist_dinov2_llm_proj.parameters())
    scene_projection = _trainable(scene_proj_params)
    # ``head_projections``: modules feeding action_head directly (below LLM).
    head_proj_params = (
        list(model.wrist_proj.parameters())
        + list(model.proprio_proj.parameters())
        + list(model.action_decoder.parameters())
    )
    if getattr(model, "wrist_projector_bridge", None) is not None:
        head_proj_params += list(model.wrist_projector_bridge.parameters())
    if getattr(model, "wrist_dinov2_projector", None) is not None:
        head_proj_params += list(model.wrist_dinov2_projector.parameters())
    if getattr(model, "wrist_dinov2_gate", None) is not None:
        head_proj_params += [model.wrist_dinov2_gate]
    head_projections = _trainable(head_proj_params)

    groups = [
        {"name": "gemma_lora",       "params": gemma_lora,       "lr": lr * coefs["gemma_lora"],       "weight_decay": weight_decay},
        {"name": "siglip",           "params": siglip,           "lr": lr * coefs["siglip"],           "weight_decay": weight_decay},
        {"name": "wrist_siglip",     "params": wrist_siglip,     "lr": lr * coefs["wrist_siglip"],     "weight_decay": weight_decay},
        {"name": "soft_prompts",     "params": soft,             "lr": lr * coefs["soft_prompts"],     "weight_decay": weight_decay},
        {"name": "action_queries",   "params": aq,               "lr": lr * coefs["action_queries"],   "weight_decay": weight_decay},
        {"name": "scene_projection", "params": scene_projection, "lr": lr * coefs["scene_projection"], "weight_decay": weight_decay},
        {"name": "head_projections", "params": head_projections, "lr": lr * coefs["head_projections"], "weight_decay": weight_decay},
        {"name": "action_head",      "params": head,             "lr": lr * coefs["action_head"],      "weight_decay": weight_decay},
    ]
    # Drop empty groups (frozen modules contribute nothing).
    groups = [g for g in groups if g["params"]]
    # AdamW betas: vla-gemma-4 73% baseline uses PyTorch default (0.9, 0.999)
    # for non-pretrain (Stage 1-2) finetune runs. The X-VLA pretrain convention
    # (0.9, 0.95) is reserved for the multi-domain Stage 3 pretrain only.
    # Verified at vla-gemma-4 ``finetune_gemma4.py:1031`` (no betas arg →
    # defaults). Mismatch was a likely contributor to our train loss not
    # tracking the baseline.
    if optimizer_kind == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if optimizer_kind == "adamw_8bit":
        # bitsandbytes 8-bit AdamW: stores momentum buffers in 8-bit (block-quantized
        # to fp32 master) → ~75% optimizer-state memory reduction vs vanilla AdamW.
        # Used in v37 OXE pretrain to fit per-GPU bs=8 across DA-2-MLP × num_domains=9
        # weights on A100-40GB. Requires bitsandbytes ≥0.41 (verified 0.49.2 in env).
        # Numerical behavior is well-characterized in the LLM finetuning community
        # (LoRA/QLoRA workflows) and matches AdamW within a few percent on convergence.
        try:
            from bitsandbytes.optim import AdamW8bit
        except ImportError as e:
            raise ImportError(
                "optimizer_kind='adamw_8bit' requires bitsandbytes; install or "
                "fall back to 'adamw' (memory cost ≈ +600 MB per GPU at num_domains=9)."
            ) from e
        return AdamW8bit(groups, betas=(0.9, 0.999))
    raise ValueError(
        f"unknown optimizer_kind={optimizer_kind!r} (expected 'adamw' | 'adamw_8bit')"
    )
