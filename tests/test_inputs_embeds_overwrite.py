import torch
from vla_project.models.language.embed_overwrite import scatter_into_embeds


def test_overwrite_replaces_at_indices_only():
    B, L, D = 2, 7, 4
    base = torch.zeros(B, L, D)
    new = torch.ones(B, 3, D)
    idx = torch.tensor([[1, 3, 5], [0, 2, 6]])
    out = scatter_into_embeds(base, idx, new)
    for b in range(B):
        for k, pos in enumerate(idx[b].tolist()):
            assert torch.equal(out[b, pos], new[b, k])
        zeros = torch.tensor([p for p in range(L) if p not in idx[b].tolist()])
        for pos in zeros.tolist():
            assert torch.equal(out[b, pos], torch.zeros(D))
