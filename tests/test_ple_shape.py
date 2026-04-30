import torch

from tests._stubs import _StubGemma


def test_per_layer_inputs_shape():
    gemma = _StubGemma()
    B, L = 2, 7
    input_ids = torch.zeros(B, L, dtype=torch.long)
    am = torch.ones(B, L, dtype=torch.long)
    out = gemma(input_ids, am)
    assert out.per_layer_inputs.shape[:2] == (B, L)
    assert out.per_layer_inputs.shape[2] == gemma.num_layers
