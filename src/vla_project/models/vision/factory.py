"""Vision encoder construction from config values."""
from __future__ import annotations

import torch.nn as nn


def build_vision_encoder(vision_type: str = "hf", model_name: str | None = None) -> nn.Module:
    """Build a SigLIP encoder backend.

    ``hf`` is the project default. ``timm`` is the vla-gemma-4 baseline-compatible
    backend used by v25 configs.
    """
    if vision_type == "hf":
        from vla_project.models.vision.siglip import SigLIPEncoder

        return SigLIPEncoder(model_name=model_name)
    if vision_type == "timm":
        from vla_project.models.vision.siglip_timm import SigLIPTimmEncoder

        return SigLIPTimmEncoder(model_name=model_name)
    raise ValueError(f"vision_type must be 'hf' or 'timm'; got {vision_type!r}")
