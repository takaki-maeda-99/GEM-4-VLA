from dataclasses import dataclass

import torch


@dataclass
class NormalizationStats:
    mean: torch.Tensor   # [D]
    std: torch.Tensor    # [D]


def _safe_std(std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return std.clamp_min(eps)


def normalize(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return (x - stats.mean) / _safe_std(stats.std)


def denormalize(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return x * _safe_std(stats.std) + stats.mean
