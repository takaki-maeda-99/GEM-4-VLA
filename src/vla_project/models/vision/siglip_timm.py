"""timm-backed SigLIP encoder, baseline-equivalent.

Wraps ``prismatic.models.backbones.vision.siglip_vit.SigLIPViTBackbone`` (the
exact backbone vla-gemma-4's 73% baseline used) so it satisfies the same
``forward`` / ``forward_all_layers`` contract as our HF-based
:class:`vla_project.models.vision.siglip.SigLIPEncoder`.

Why duplicate the HF encoder: empirically
(``/home/takaki/.claude/projects/.../memory/hf_vs_timm_siglip.md``) HF
``SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-224")`` and
timm ``vit_so400m_patch14_siglip_224.v2_webli`` produce significantly
different features on the same input (abs diff mean ≈ 1.7). Baseline trained
its action_head + wrist_projector_bridge against timm features; matching
that requires the timm path here, since random projectors + frozen LLM + Mode
B don't have any way to absorb the backbone-distribution mismatch.

Usage: build via ``cfg.vision.type='timm'`` in train/eval configs (default
remains 'hf' for backward compat with v15-v24 ckpts).

Input contract: pixel_values is (B, 3, 224, 224) float in [-1, 1] (i.e.
already passed through ``SiglipImageTransform`` / equivalent normalization).
The timm backbone does NOT do its own normalization here; we feed the same
tensor that the HF path uses.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SigLIPTimmEncoder(nn.Module):
    """timm SigLIP-So400m wrapper, frozen, matching SigLIPEncoder API."""

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        nn.Module.__init__(obj)
        return obj

    def __init__(
        self,
        model_name: Optional[str] = "google/siglip-so400m-patch14-224",  # accepted but unused
        hidden_dim: int = 1152,
        num_tokens: int = 256,
        image_size: int = 224,
        _skip_load: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.image_size = image_size
        self.backbone: Optional[nn.Module] = None
        if not _skip_load:
            from prismatic.models.backbones.vision.siglip_vit import SigLIPViTBackbone
            self.backbone = SigLIPViTBackbone(
                vision_backbone_id="siglip-vit-so400m",
                image_resize_strategy="resize-naive",
                default_image_size=image_size,
            )
            assert self.backbone.embed_dim == hidden_dim, (
                f"timm SigLIP embed_dim={self.backbone.embed_dim} != expected {hidden_dim}"
            )
            assert self.backbone.num_patches == num_tokens, (
                f"timm SigLIP num_patches={self.backbone.num_patches} != expected {num_tokens}"
            )
            self.freeze()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Final-layer block output, (B, 256, 1152).

        Matches baseline ``encode_scene → vision_backbone(x)``: backbone
        forward returns the patch features at SigLIP's last block. We pass
        the already-normalized [-1, 1] tensor directly.
        """
        assert self.backbone is not None
        return self.backbone(pixel_values)

    def forward_all_layers(self, pixel_values: torch.Tensor, num_layers: int) -> torch.Tensor:
        """Per-layer block outputs at SigLIP indices 0..num_layers-1.

        Matches baseline wrist_projector_bridge feed:
        ``featurizer.get_intermediate_layers(x, n=list(range(num_layers)))``.
        Returns (B, num_layers, 256, 1152) stacked along dim=1.
        """
        assert self.backbone is not None
        feats = self.backbone.featurizer.get_intermediate_layers(
            pixel_values, n=list(range(num_layers))
        )
        return torch.stack(feats, dim=1)
