import torch

from vla_project.models.vision.siglip import SigLIPEncoder


def test_output_shape_with_stub(monkeypatch):
    enc = SigLIPEncoder.__new__(SigLIPEncoder)
    enc.hidden_dim = 1152
    enc.num_tokens = 256
    enc._stub = True

    def fake_forward(self, pixel_values):
        B = pixel_values.shape[0]
        return torch.zeros(B, self.num_tokens, self.hidden_dim)

    monkeypatch.setattr(SigLIPEncoder, "forward", fake_forward)

    out = enc(torch.randn(4, 3, 224, 224))
    assert out.shape == (4, 256, 1152)


def test_frozen_by_default():
    # avoid hitting the network: stub the inner model
    enc = SigLIPEncoder(model_name=None, _skip_load=True)
    enc.model = torch.nn.Linear(3, 3)  # dummy module to check freezing
    enc.freeze()
    for p in enc.model.parameters():
        assert p.requires_grad is False
