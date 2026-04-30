from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class Gemma4Out:
    hidden_states: torch.Tensor   # [B, num_layers+1, L, D]
    per_layer_inputs: torch.Tensor  # [B, L, num_layers, ple_dim]


class Gemma4Wrapper(nn.Module):
    """Loads Gemma4-E2B `text_model` and runs forward with PLE precompute.

    Sequence of operations follows
    `vla-gemma-4/.../modeling_prismatic_gemma4.py:563-617`:

      1. `per_layer_inputs = get_per_layer_inputs(input_ids)` under `no_grad`
      2. caller computes `inputs_embeds` (clone + scatter_into_embeds)
      3. `text_model(inputs_embeds, per_layer_inputs, ...)` returns hidden_states tuple

    The wrapper exposes:
      - `embed_tokens(input_ids)` for the caller to obtain raw embeddings
      - `forward(inputs_embeds, per_layer_inputs, attention_mask)` returns Gemma4Out
    """

    def __init__(
        self,
        model_name: Optional[str] = "google/gemma-3n-E2B",  # placeholder; verify before training
        freeze: bool = True,
        _skip_load: bool = False,
    ) -> None:
        super().__init__()
        self.text_model: Optional[nn.Module] = None
        self.num_layers: int = 0
        if _skip_load:
            return
        if model_name is None:
            raise ValueError(
                "Gemma4Wrapper requires a model_name unless _skip_load=True is set"
            )
        from transformers import AutoModelForCausalLM
        full = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.text_model = getattr(full, "model", full)
        self.num_layers = self.text_model.config.num_hidden_layers
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.text_model.eval()

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.text_model is not None, "Gemma4Wrapper not loaded"
        return self.text_model.embed_tokens(input_ids)

    def precompute_ple(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.text_model is not None, "Gemma4Wrapper not loaded"
        with torch.no_grad():
            # Gemma4 / Gemma3n signature: get_per_layer_inputs(input_ids, **kwargs).
            # We pass only input_ids; second positional was previously `inputs_embeds`
            # in some HF revisions and is keyword-only in current ones.
            return self.text_model.get_per_layer_inputs(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> Gemma4Out:
        if per_layer_inputs is None:
            per_layer_inputs = self.precompute_ple(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        B, L = input_ids.shape
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        out = self.text_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        hs = torch.stack(out.hidden_states, dim=1)  # [B, layers+1, L, D]
        return Gemma4Out(hidden_states=hs, per_layer_inputs=per_layer_inputs)
