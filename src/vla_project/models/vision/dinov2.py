"""Frozen DINOv2 vision encoder for auxiliary dense wrist features."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    """Wrap ``transformers.AutoModel`` DINOv2 and expose patch tokens only.

    Input contract: ``pixel_values`` is (B, 3, 224, 224) normalized with the
    DINO/ImageNet mean and std. Forward returns patch tokens (B, 256, D) for
    ViT/14 models; the CLS token is dropped.
    """

    def __init__(
        self,
        model_name: Optional[str] = "facebook/dinov2-base",
        hidden_dim: int = 768,
        num_tokens: int = 256,
        _skip_load: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.model: Optional[nn.Module] = None
        if not _skip_load:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name)
            self.freeze()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.model is not None, "DINOv2Encoder.forward called before model loaded"
        out = self.model(pixel_values=pixel_values).last_hidden_state
        patch = out[:, 1:, :]
        assert patch.shape[1:] == (self.num_tokens, self.hidden_dim), (
            f"expected (B, {self.num_tokens}, {self.hidden_dim}), got {tuple(patch.shape)}"
        )
        return patch
