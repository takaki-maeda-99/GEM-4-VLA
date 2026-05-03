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
        """Return per-layer **block outputs** for the wrist_bridge stream.

        Output shape: ``(B, num_layers, num_tokens, hidden_dim)`` where
        index ``k`` is the output of SigLIP transformer block ``k`` (NOT
        the embedding layer, which is hidden_states[0] in HF — we skip it).

        Matches vla-gemma-4 timm semantics:
        ``featurizer.get_intermediate_layers(x, n=list(range(num_layers)))``
        returns block outputs at indices 0..num_layers-1, NOT the embedding.

        Earlier we returned hidden_states[:num_layers] which started with
        the embedding output and dropped the deepest block — an off-by-one
        that fed the head's block i with SigLIP block (i-1) output instead
        of SigLIP block (i+1) as in the 73% baseline.

        For ``num_blocks=24`` head blocks, pass ``num_layers=num_blocks+1=25``;
        head block i indexes ``h_w_bridge[:, i+1]`` (idx 1..24) which now
        correctly maps to SigLIP block 1..24 outputs.
        """
        assert self.model is not None
        out = self.model(pixel_values=pixel_values, output_hidden_states=True)
        # all_hs[0] is the embedding output; all_hs[1..N] are block outputs
        # (for so400m-patch14-224 N=27 → tuple length 28).
        all_hs = out.hidden_states
        # We want block outputs 0..num_layers-1 = all_hs[1..num_layers].
        if num_layers + 1 > len(all_hs):
            raise ValueError(
                f"num_layers={num_layers} requires {num_layers+1} hidden_states "
                f"(embedding + {num_layers} blocks); SigLIP returned only {len(all_hs)}"
            )
        stacked = torch.stack(all_hs[1 : num_layers + 1], dim=1)
        assert stacked.shape[2:] == (self.num_tokens, self.hidden_dim)
        return stacked
