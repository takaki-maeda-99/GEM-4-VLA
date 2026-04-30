from typing import Optional

import torch
import torch.nn as nn


class SigLIPEncoder(nn.Module):
    """Wraps `transformers.SiglipVisionModel`. Always frozen.

    Forward returns the **last hidden state** (`[B, N, D_vis]`).
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        nn.Module.__init__(obj)
        return obj

    def __init__(
        self,
        model_name: Optional[str] = "google/siglip-so400m-patch14-224",
        hidden_dim: int = 1152,
        num_tokens: int = 256,
        _skip_load: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.model: Optional[nn.Module] = None
        if not _skip_load:
            from transformers import SiglipVisionModel
            self.model = SiglipVisionModel.from_pretrained(model_name)
            self.freeze()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        assert self.model is not None, (
            "SigLIPEncoder.forward called before model loaded "
            "(used _skip_load=True without overriding self.model?)"
        )
        out = self.model(pixel_values=pixel_values).last_hidden_state
        assert out.shape[1:] == (self.num_tokens, self.hidden_dim), (
            f"expected (B, {self.num_tokens}, {self.hidden_dim}), "
            f"got {tuple(out.shape)}"
        )
        return out
