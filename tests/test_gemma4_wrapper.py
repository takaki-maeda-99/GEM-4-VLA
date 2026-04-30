import torch
import torch.nn as nn

from vla_project.models.language.gemma4_wrapper import Gemma4Wrapper


class _StubText(nn.Module):
    def __init__(self, hidden=8, layers=4):
        super().__init__()
        self.layers = layers
        self.hidden = hidden
        self.embed = nn.Embedding(50, hidden)

    def get_per_layer_inputs(self, input_ids, **kwargs):
        B, L = input_ids.shape
        return torch.zeros(B, L, self.layers, 4)

    def embed_tokens(self, input_ids):
        return self.embed(input_ids)

    def forward(
        self,
        inputs_embeds=None,
        per_layer_inputs=None,
        attention_mask=None,
        position_ids=None,
        use_cache=False,
        output_hidden_states=True,
    ):
        B, L, D = inputs_embeds.shape
        hs = tuple(inputs_embeds + i for i in range(self.layers + 1))
        return type("Out", (), {"hidden_states": hs})()


def test_forward_returns_stacked_hidden_states():
    text = _StubText()
    wrapper = Gemma4Wrapper(model_name=None, _skip_load=True)
    wrapper.text_model = text
    wrapper.num_layers = text.layers

    B, L = 2, 5
    input_ids = torch.zeros(B, L, dtype=torch.long)
    am = torch.ones(B, L, dtype=torch.long)
    out = wrapper(input_ids, am)
    assert out.hidden_states.shape == (B, text.layers + 1, L, text.hidden)
    assert out.per_layer_inputs.shape == (B, L, text.layers, 4)
