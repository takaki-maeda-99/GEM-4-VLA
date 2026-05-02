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

    def forward_all_layers(self, pixel_values: torch.Tensor, num_layers: int) -> torch.Tensor:
        """Return per-layer features for the wrist_bridge stream.

        Output shape: ``(B, num_layers, num_tokens, hidden_dim)``.

        ``num_layers`` is **inclusive of the embedding** (idx 0); indices
        1..num_layers-1 are the first (num_layers-1) transformer block
        outputs. For ``num_blocks=24`` action-head blocks, pass
        ``num_layers=25`` so block i in the head can index
        ``h_w_bridge[:, i+1]`` (1..24 covering block_outputs 0..23).

        Mirrors vla-gemma-4's
        ``modeling_prismatic_gemma4.py:557-578`` per_layer mode (with the
        timm-specific ``get_intermediate_layers`` replaced by HF's
        ``output_hidden_states=True``). The HF SigLIPVision encoder for
        ``so400m-patch14-224`` returns 28 hidden states (1 embedding + 27
        blocks); we slice the first ``num_layers``.
        """
        assert self.model is not None
        out = self.model(pixel_values=pixel_values, output_hidden_states=True)
        all_hs = out.hidden_states  # tuple length 28 of (B, 256, D)
        if num_layers > len(all_hs):
            raise ValueError(
                f"num_layers={num_layers} > available SigLIP hidden_states={len(all_hs)}"
            )
        stacked = torch.stack(all_hs[:num_layers], dim=1)  # (B, num_layers, 256, D)
        assert stacked.shape[2:] == (self.num_tokens, self.hidden_dim)
        return stacked
