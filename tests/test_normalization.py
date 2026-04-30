import torch

from vla_project.data.normalization import NormalizationStats, normalize, denormalize


def test_normalize_denormalize_roundtrip():
    stats = NormalizationStats(
        mean=torch.tensor([0.0, 1.0]),
        std=torch.tensor([2.0, 0.5]),
    )
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = normalize(x, stats)
    z = denormalize(y, stats)
    torch.testing.assert_close(x, z)


def test_zero_std_clamped():
    stats = NormalizationStats(
        mean=torch.tensor([0.0]),
        std=torch.tensor([0.0]),
    )
    x = torch.tensor([1.0])
    # must not div-by-zero
    y = normalize(x, stats)
    assert torch.isfinite(y).all()
