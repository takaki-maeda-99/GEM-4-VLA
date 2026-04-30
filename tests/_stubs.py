"""Shared lightweight stubs for VLAPolicy unit tests (no network, no real weights)."""
import torch
import torch.nn as nn


class _StubSig(nn.Module):
    """Mimics SigLIPEncoder output shape without loading weights."""

    def __init__(self):
        super().__init__()
        self.hidden_dim = 1152
        self.num_tokens = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.num_tokens, self.hidden_dim)

    def freeze(self):
        pass


class _StubGemma(nn.Module):
    """Mimics Gemma4Wrapper API: embed_tokens / precompute_ple / forward.

    Hidden states are deterministic functions of input_ids so tests can verify
    that downstream code reads the expected positions.
    """

    num_layers = 4
    hidden_dim = 32

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(300_000, self.hidden_dim)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def precompute_ple(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        return torch.zeros(B, L, self.num_layers, 4)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds=None,
        per_layer_inputs=None,
    ):
        from vla_project.models.language.gemma4_wrapper import Gemma4Out
        if per_layer_inputs is None:
            per_layer_inputs = self.precompute_ple(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # Each layer returns inputs_embeds + i so layer index is recoverable.
        hs = torch.stack([inputs_embeds + i for i in range(self.num_layers + 1)], dim=1)
        return Gemma4Out(hidden_states=hs, per_layer_inputs=per_layer_inputs)
